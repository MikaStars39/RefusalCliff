# import pdb
import torch
import json

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.lens.utils import batch_gen, add_scale, batch_probe
from src.model.modeling_llama import clean_property

@torch.no_grad()
def ablating_attn_head(
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    json_path: str = "/diancpfs/user/qingyu/persona/outputs/inference/DeepSeek-R1-Distill-Llama-8B/jbbench_distill_llama_8b.json",
    prober_ckpt_path: str = None,
    layer_idx: int = 21,
    position_idx: int = 0,
    batch_size: int = 32,
    truncate_num: int = 128,
    top_n_ablation: int = 10,
    top_n_enhancement: int = 10,
    ablation_value: float = 0.1,
    enhancement_value: float = 100.0,
    thinking_portion: float = 0.0,
    head_ablation_path: str = "outputs/tracing/llama_head_prober_outputs.json",
    head_enhancement_path: str = "outputs/tracing/llama_head_prober_outputs_toxic.json",
    item_type: str = "original_item",
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )

    with open(json_path, "r") as f:
        data = json.load(f)[:truncate_num]

    with open(head_ablation_path, "r") as f:
        head_ablation_data = json.load(f)[:top_n_ablation]
    
    if head_enhancement_path is not None:
        with open(head_enhancement_path, "r") as f:
            head_enhancement_data = json.load(f)[-top_n_enhancement:]

    batch_messages = []
    thinking = []

    for item in data:
        batch_messages.append([{"role": "user", "content": item["original_item"]["prompt"]}])
        thinking.append(item[item_type]["thinking"])
        if thinking_portion < 0.0:
            thinking = ""
        elif thinking_portion > 0.0:
            thinking = thinking[:int(len(thinking) * thinking_portion)]

    from src.lens.prober import LinearProber
    # load checkpoint
    ckpt = torch.load(prober_ckpt_path, map_location=model.device)
    in_dim = ckpt["in_dim"]
    hidden_dim = 1024  # use the same as training; change if you trained with a different value

    # rebuild model and load weights
    prober = LinearProber(in_dim, hidden_dim).to(model.device)
    prober.load_state_dict(ckpt["state_dict"])
    prober.eval()
    
    prober_output = batch_probe(
        tokenizer=tokenizer,
        model=model,
        prober=prober,
        messages=batch_messages,
        thinking=thinking,
        batch_size=batch_size,
        position_idx=position_idx,
        layer_idx=layer_idx,
    )

    print(f"original prober output: {prober_output}")

    add_scale(model, head_ablation_data, ablation_value, head_enhancement_data, enhancement_value)

    prober_output = batch_probe(
        tokenizer=tokenizer,
        model=model,
        prober=prober,
        messages=batch_messages,
        thinking=thinking,
        batch_size=batch_size,
        position_idx=position_idx,
        layer_idx=layer_idx,
    )
    print(f"ablating prober output: {prober_output}")

@torch.no_grad()
def trace_attn_head(
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    json_path: str = "/diancpfs/user/qingyu/persona/outputs/inference/DeepSeek-R1-Distill-Llama-8B/jbbench_distill_llama_8b.json",
    prober_ckpt_path: str = None,
    layer_idx: int = 21,
    position_idx: int = 0,
    batch_size: int = 32,
    max_new_tokens: int = 8,
    temperature: float = 0.7,
    do_sample: bool = True,
    truncate_num: int = 128,
    thinking_portion: float = 0.0,
    save_path: str = "outputs/tracing/llama_outputs.json",
    item_type: str = "original_item",
    state_save_path: str = None,
):
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )

    with open(json_path, "r") as f:
        data = json.load(f)[:truncate_num]

    batch_messages = []
    thinking = []
    all_outputs = []

    if prober_ckpt_path is not None:
        from src.lens.prober import LinearProber
            # load checkpoint
        ckpt = torch.load(prober_ckpt_path, map_location=model.device)
        in_dim = ckpt["in_dim"]
        hidden_dim = 1024  # use the same as training; change if you trained with a different value

        # rebuild model and load weights
        prober = LinearProber(in_dim, hidden_dim).to(model.device)
        prober.load_state_dict(ckpt["state_dict"])
        prober.eval()
    
    for item in data:
        batch_messages.append([{"role": "user", "content": item["original_item"]["prompt"]}])
        thinking.append(item[item_type]["thinking"])
        if thinking_portion < 0.0:
            thinking[-1] = ""
        elif thinking_portion > 0.0:
            thinking[-1] = thinking[-1][:int(len(thinking[-1]) * thinking_portion)]
    
    original_prober_output = batch_probe(
        tokenizer=tokenizer,
        model=model,
        prober=prober,
        messages=batch_messages,
        thinking=thinking,
        batch_size=batch_size,
        position_idx=position_idx,
        layer_idx=layer_idx,
    )

    for per_layer_idx in tqdm(range(model.config.num_hidden_layers)):
        for per_head_idx in range(model.config.num_attention_heads):
            
            # add scale to the head (ablation only)
            head_ablation_data = [
                {
                    "layer_idx": per_layer_idx,
                    "head_idx": per_head_idx,
                }
            ]
            head_enhancement_data = None
            
            if prober_ckpt_path is not None:

                # probe the head if prober is provided
                add_scale(model, head_ablation_data, 0.001, head_enhancement_data, 10.0)

                prober_output = batch_probe(
                    tokenizer=tokenizer,
                    model=model,
                    prober=prober,
                    messages=batch_messages,
                    thinking=thinking,
                    batch_size=batch_size,
                    position_idx=position_idx,
                    layer_idx=layer_idx,
                )
                all_outputs.append({
                    "layer_idx": per_layer_idx,
                    "head_idx": per_head_idx,
                    "prober_output": original_prober_output - prober_output,
                })
                
                clean_property(model, "self_attn", "scale")
            else:
                add_scale(model, head_ablation_data, 0.01, head_enhancement_data, 100.0)
                # else, generate the response
                outputs = batch_gen(
                    tokenizer=tokenizer, 
                    model=model, 
                    messages=batch_messages, 
                    thinking=thinking,
                    batch_size=batch_size,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                )
                for idx, item in enumerate(outputs):
                    all_outputs.append(
                        {
                            "prompt": data[idx]["original_item"]["prompt"],
                            "thinking": data[idx][item_type]["thinking"],
                            "response": item,
                        }
                    )
                clean_property(model, "self_attn", "scale")

    # Sort by prober_output in descending order (highest scores first)
    if prober_ckpt_path is not None:
        all_outputs.sort(key=lambda x: x["prober_output"], reverse=True)
    
    with open(save_path, "w") as f:
        json.dump(all_outputs, f, indent=4)


@torch.no_grad()
def analyze_attn_patterns(
    heads_json_path: str,
    data_json_path: str,
    output_folder: str,
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    top_n: int = 10,
    token_position: int = -1,
    item_type: str = "original_item",
    batch_size: int = 1,
    truncate_num: int = 128,
    thinking_portion: float = 0.0,
    create_plots: bool = False,
):
    """
    Analyze attention head patterns and save to files
    
    Args:
        heads_json_path: JSON file path containing layer_idx, head_idx and other info
        data_json_path: JSON file path containing prompt and thinking data
        output_folder: Output folder path
        model_name: Model name
        top_n: Take top n heads
        token_position: Token position to analyze (-1 means last token)
        item_type: Data item type
        batch_size: Batch size
        truncate_num: Truncation number
        thinking_portion: Thinking portion ratio
        create_plots: Whether to create visualization plots (default: False)
    """
    import os
    import numpy as np
    
    if create_plots:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError as e:
            print(f"Missing plotting dependencies: {e}")
            print("Please install: pip install matplotlib seaborn")
            print("Continuing without plots...")
            create_plots = False
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    
    # Read heads data and sort by cosine_similarity to get top_n
    with open(heads_json_path, "r") as f:
        heads_data = json.load(f)
    
    # Sort by cosine_similarity for top_n (assuming data has cosine_similarity field)
    if heads_data and "cosine_similarity" in heads_data[0]:
        heads_data.sort(key=lambda x: abs(x["cosine_similarity"]), reverse=True)
    elif heads_data and "prober_output" in heads_data[0]:
        heads_data.sort(key=lambda x: abs(x["prober_output"]), reverse=True)
    heads_data = heads_data[:top_n]
    
    print(f"Analyzing top {len(heads_data)} heads")
    
    # Read data
    with open(data_json_path, "r") as f:
        data = json.load(f)[:truncate_num]
    
    # Prepare batch messages and thinking
    batch_messages = []
    thinking = []
    
    for item in data:
        batch_messages.append([{"role": "user", "content": item["original_item"]["prompt"]}])
        thinking.append(item[item_type]["thinking"])
        if thinking_portion < 0.0:
            thinking[-1] = ""
        elif thinking_portion > 0.0:
            thinking[-1] = thinking[-1][:int(len(thinking[-1]) * thinking_portion)]
    
    # Enable monkey patching to get attention weights
    from src.model.modeling_llama import enable_monkey_patched_llama
    enable_monkey_patched_llama(model)
    
    # Store all attention patterns
    all_attention_patterns = {}
    
    # Analyze attention pattern for each selected head
    for head_info in tqdm(heads_data, desc="Analyzing attention heads"):
        layer_idx = head_info["layer_idx"]
        head_idx = head_info["head_idx"]
        
        print(f"Processing Layer {layer_idx}, Head {head_idx}")
        
        # Create a hook to capture attention weights
        attention_weights_storage = {}
        
        def attention_hook(module, input, output):
            if hasattr(module, 'layer_idx') and module.layer_idx == layer_idx:
                # output[1] is attn_weights from eager_attention_forward
                if len(output) > 1 and output[1] is not None:
                    attn_weights = output[1]  # shape: [batch_size, num_heads, seq_len, seq_len]
                    # Only save the head we're interested in
                    if head_idx < attn_weights.shape[1]:  # Ensure head_idx is valid
                        attention_weights_storage[f"layer_{layer_idx}_head_{head_idx}"] = attn_weights[:, head_idx, :, :].detach().cpu()
                    else:
                        print(f"Warning: head_idx {head_idx} >= num_heads {attn_weights.shape[1]} in layer {layer_idx}")
        
        # Register hooks
        handles = []
        for name, module in model.named_modules():
            if "self_attn" in name and hasattr(module, 'layer_idx') and module.layer_idx == layer_idx:
                handle = module.register_forward_hook(attention_hook)
                handles.append(handle)
        
        # Process data in batches
        head_patterns = []
        for idx in range(0, len(batch_messages), batch_size):
            batch_msgs = batch_messages[idx:idx + batch_size]
            batch_think = thinking[idx:idx + batch_size] if thinking else None
            
            # Prepare input text
            batch_text = []
            for i, message_list in enumerate(batch_msgs):
                text = tokenizer.apply_chat_template(
                    message_list,
                    add_generation_prompt=True,
                    tokenize=False,
                ) + (batch_think[i] + "\n</think>\n\n" if batch_think and batch_think[i] else "")
                batch_text.append(text)
            
            # Tokenize
            inputs = tokenizer(
                batch_text,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(model.device)
            
            # Forward pass
            try:
                outputs = model(**inputs, output_attentions=False)
            except Exception as e:
                print(f"Error during forward pass: {e}")
                continue
            
            # Get stored attention weights
            if f"layer_{layer_idx}_head_{head_idx}" in attention_weights_storage:
                attn_weights = attention_weights_storage[f"layer_{layer_idx}_head_{head_idx}"]
                
                # Handle token position
                if token_position == -1:
                    # Use last non-padding token
                    for batch_idx in range(attn_weights.shape[0]):
                        actual_length = inputs["attention_mask"][batch_idx].sum().item()
                        pattern = attn_weights[batch_idx, actual_length-1, :actual_length].float().numpy()
                        head_patterns.append({
                            "pattern": pattern.tolist(),
                            "tokens": tokenizer.convert_ids_to_tokens(inputs["input_ids"][batch_idx][:actual_length]),
                            "prompt": batch_text[batch_idx],
                            "actual_length": actual_length
                        })
                else:
                    # Use specified position
                    for batch_idx in range(attn_weights.shape[0]):
                        actual_length = inputs["attention_mask"][batch_idx].sum().item()
                        if token_position < actual_length:
                            pattern = attn_weights[batch_idx, token_position, :actual_length].float().numpy()
                            head_patterns.append({
                                "pattern": pattern.tolist(),
                                "tokens": tokenizer.convert_ids_to_tokens(inputs["input_ids"][batch_idx][:actual_length]),
                                "prompt": batch_text[batch_idx],
                                "position": token_position,
                                "actual_length": actual_length
                            })
            
            # Clear storage
            attention_weights_storage.clear()
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        # Save patterns for this head
        all_attention_patterns[f"layer_{layer_idx}_head_{head_idx}"] = head_patterns
        
        # Create visualization and save for this head
        if head_patterns:
            # Save detailed data
            head_output_path = os.path.join(output_folder, f"layer_{layer_idx}_head_{head_idx}_patterns.json")
            with open(head_output_path, "w") as f:
                json.dump({
                    "layer_idx": layer_idx,
                    "head_idx": head_idx,
                    "patterns": head_patterns,
                    "head_info": head_info
                }, f, indent=2)
            
            # Create visualizations and save token details
            max_samples_to_process = min(5, len(head_patterns))
            for sample_idx in range(max_samples_to_process):
                pattern_data = head_patterns[sample_idx]
                pattern = np.array(pattern_data["pattern"])
                tokens = pattern_data["tokens"]
                
                try:
                    # Save detailed token attention weights JSON
                    token_attention_data = {
                        "layer_idx": layer_idx,
                        "head_idx": head_idx,
                        "sample_idx": sample_idx + 1,
                        "token_position": token_position if token_position != -1 else "last",
                        "tokens_with_attention": [
                            {
                                "token": token,
                                "token_index": idx,
                                "attention_weight": float(pattern[idx])
                            }
                            for idx, token in enumerate(tokens)
                        ],
                        "total_attention": float(np.sum(pattern)),
                        "max_attention": float(np.max(pattern)),
                        "min_attention": float(np.min(pattern)),
                        "prompt": pattern_data["prompt"]
                    }
                    
                    token_json_path = os.path.join(output_folder, f"layer_{layer_idx}_head_{head_idx}_sample_{sample_idx+1}_tokens.json")
                    with open(token_json_path, "w", encoding='utf-8') as f:
                        json.dump(token_attention_data, f, indent=2, ensure_ascii=False)
                    
                    # Create plots only if requested
                    if create_plots:
                        # Clean tokens to avoid font issues
                        clean_tokens = []
                        for token in tokens:
                            # Replace problematic characters with ASCII alternatives
                            clean_token = token.replace('｜', '|')  # Replace fullwidth vertical line
                            clean_token = clean_token.replace('▁', '_')  # Replace sentencepiece marker
                            # Remove or replace other problematic unicode characters
                            clean_token = ''.join(c if ord(c) < 128 else '?' for c in clean_token)
                            clean_tokens.append(clean_token)
                        
                        # Create heatmap visualization
                        plt.figure(figsize=(12, 8))
                        sns.heatmap(
                            pattern.reshape(1, -1), 
                            xticklabels=clean_tokens,
                            yticklabels=[f"Position {token_position}" if token_position != -1 else "Last Token"],
                            cmap="Blues",
                            cbar=True
                        )
                        plt.title(f"Attention Pattern - Layer {layer_idx}, Head {head_idx}, Sample {sample_idx+1}")
                        plt.xlabel("Key Tokens")
                        plt.ylabel("Query Token")
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        
                        # Save plot
                        plot_path = os.path.join(output_folder, f"layer_{layer_idx}_head_{head_idx}_sample_{sample_idx+1}.png")
                        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                        plt.close()
                    
                except Exception as e:
                    print(f"Warning: Could not process sample for layer {layer_idx}, head {head_idx}, sample {sample_idx+1}: {e}")
                    if create_plots:
                        plt.close()  # Ensure figure is closed
            
            print(f"Saved patterns for Layer {layer_idx}, Head {head_idx} to {head_output_path}")
    
    # Save summary information
    summary_path = os.path.join(output_folder, "summary.json")
    summary = {
        "total_heads_analyzed": len(heads_data),
        "heads_info": heads_data,
        "model_name": model_name,
        "token_position": token_position,
        "total_samples": len(data),
        "output_files": list(all_attention_patterns.keys())
    }
    
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Analysis complete! Results saved to {output_folder}")
    print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    from fire import Fire
    Fire(analyze_attn_patterns)
