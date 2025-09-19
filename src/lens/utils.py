import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from ..model.modeling_llama import enable_monkey_patched_llama, add_property

def enable_appropriate_monkey_patch(model, model_name_or_path: str = None):
    """
    Enable the appropriate monkey patch based on model type
    
    Args:
        model: The model to patch
        model_name_or_path: Optional model name/path for type detection
    """
    config_name = model.config.__class__.__name__.lower()
    if 'llama' in config_name.lower():
        model_type = 'llama'
    elif 'qwen' in config_name.lower() or "Skywork-OR1-7B" in config_name or "QwQ" in config_name:
        model_type = 'qwen'
    else:
        model_type = 'unknown'
    
    if model_type == 'llama':
        enable_monkey_patched_llama(model)
    elif model_type == 'qwen':
        from ..model.modeling_qwen import enable_monkey_patched_qwen
        enable_monkey_patched_qwen(model)
    else:
        print(f"Warning: Unknown model type, falling back to Llama monkey patch")
        enable_monkey_patched_llama(model)

def clean_property(model, module_name, property_name):
    """Universal clean_property function that works with both Llama and Qwen models"""
    # recursively patch the model
    def recursive_patch(model):
        for name, module in reversed(model._modules.items()):
            if len(list(module.children())) > 0:
                recursive_patch(
                    module,
                )
            if module_name in name:
                if hasattr(model._modules[name], property_name):
                    delattr(model._modules[name], property_name)
    
    recursive_patch(model)

def add_scale(
    model: AutoModelForCausalLM,
    head_ablation_data: list,
    ablation_value: float = 1,
    head_enhancement_data: list = None,
    enhancement_value: float = 1,
):
    # Automatically detect and enable appropriate monkey patch
    enable_appropriate_monkey_patch(model)
    
    total = {}
    value = {}
    for idx in range(len(head_ablation_data)):
        layer_idx = head_ablation_data[idx]["layer_idx"]
        head_idx = head_ablation_data[idx]["head_idx"]
        if total.get(layer_idx, None) is None:
            total[layer_idx] = [head_idx]
        else:
            total[layer_idx].append(head_idx)
        if value.get(layer_idx, None) is None:
            value[layer_idx] = {}
        value[layer_idx][head_idx] = ablation_value
    
    if head_enhancement_data is not None:
        for idx in range(len(head_enhancement_data)):
            layer_idx = head_enhancement_data[idx]["layer_idx"]
            head_idx = head_enhancement_data[idx]["head_idx"]
            if value.get(layer_idx, None) is None:
                value[layer_idx] = {}
            value[layer_idx][head_idx] = enhancement_value
    add_property(model, "self_attn", "scale", {"heads": total, "values": value})

@torch.no_grad()
def batch_get_hidden_states(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    messages: list[dict],
    thinking: list = None,
    batch_size: int = 1,
):  
    all_hidden_states = []
    for idx in range(0, len(messages), batch_size):
    
        batch_messages = messages[idx:idx + batch_size]
        batch_text = []
        for i, message_list in enumerate(batch_messages):
            text = tokenizer.apply_chat_template(
                message_list,
                add_generation_prompt=True,
                tokenize=False,
            ) + (thinking[idx + i] + "\n</think>\n\n" if thinking is not None else "")
            batch_text.append(text)
        
        inputs = tokenizer(
            batch_text,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(model.device)

        outputs = model(**inputs,output_hidden_states=True)
        hidden_states = outputs.hidden_states
        hidden_states = [item.cpu() for item in hidden_states]
        all_hidden_states.append(hidden_states)
    
    del inputs, outputs, batch_text, batch_messages, hidden_states
    
    return all_hidden_states


@torch.no_grad()
def batch_probe(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prober: torch.nn.Module,
    messages: list[dict],
    thinking: list = None,
    batch_size: int = 1,
    position_idx: int = -1,
    layer_idx: int = 0,
):  
    total_prober_output = 0.0
    for idx in range(0, len(messages), batch_size):
    
        batch_messages = messages[idx:idx + batch_size]
        batch_text = []
        for i, message_list in enumerate(batch_messages):
            text = tokenizer.apply_chat_template(
                message_list,
                add_generation_prompt=True,
                tokenize=False,
            ) + (thinking[idx + i] + "\n</think>\n\n" if thinking is not None else "")
            batch_text.append(text)
        
        inputs = tokenizer(
            batch_text,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(model.device)

        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[layer_idx]
        
        # Extract only the needed hidden states and convert immediately
        selected_hidden_states = hidden_states[:, position_idx, :].to(torch.float32)
        
        # Run prober and get result immediately
        prober_output = torch.sigmoid(prober(selected_hidden_states)).sum(dim=0).cpu().item()
        total_prober_output += prober_output
    
    return total_prober_output / len(messages)

@torch.no_grad()
def batch_gen(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    messages: list[dict],
    thinking: list = None,
    batch_size: int = 1,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    do_sample: bool = True,
):  
    all_outputs = []
    for idx in tqdm(range(0, len(messages), batch_size)):
    
        batch_messages = messages[idx:idx + batch_size]
        batch_text = []
        thinking_positions = []  # Store thinking start/end positions for each item in batch
        
        for i, message_list in enumerate(batch_messages):
            # Get the base prompt without thinking
            base_text = tokenizer.apply_chat_template(
                message_list,
                add_generation_prompt=True,
                tokenize=False,
            )
            
            if thinking is not None and thinking[idx + i]:
                # Calculate thinking positions
                base_tokens = tokenizer(base_text, return_tensors="pt")["input_ids"]
                thinking_text = thinking[idx + i] + "\n</think>\n\n"
                full_text = base_text + thinking_text
                full_tokens = tokenizer(full_text, return_tensors="pt")["input_ids"]
                
                thinking_start = full_tokens.shape[-1] - base_tokens.shape[-1]
                thinking_end = 4
                thinking_positions.append((thinking_start, thinking_end))
                
                batch_text.append(full_text)
            else:
                print("No thinking for this item")
                thinking_positions.append(None)  # No thinking for this item
                batch_text.append(base_text)
        
        inputs = tokenizer(
            batch_text,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(model.device)
        
        # Add thinking positions to model for attention scaling
        # We need to adjust positions for padding
        # adjusted_thinking_positions = []
        # for i, pos in enumerate(thinking_positions):
        #     if pos is not None:
        #         # Calculate padding offset for this sequence
        #         seq_len = inputs["attention_mask"][i].sum().item()
        #         max_len = inputs["input_ids"].shape[-1]
        #         padding_offset = max_len - seq_len
                
        #         # Adjust positions accounting for left padding
        #         adj_start = pos[0] + padding_offset
        #         adj_end = pos[1] + padding_offset
        #         adjusted_thinking_positions.append((adj_start, adj_end))
        #     else:
        #         adjusted_thinking_positions.append(None)
        
        # Store thinking positions in model for access during forward pass
        from src.model.modeling_llama import add_property
        add_property(model, "self_attn", "thinking_positions", thinking_positions)

        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id
        }
        
        if do_sample:
            generation_kwargs["temperature"] = temperature
        
        # Override any default generation config that might include sampling parameters
        if not do_sample:
            # Create a temporary generation config that explicitly disables sampling
            from transformers import GenerationConfig
            gen_config = GenerationConfig(
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id
            )
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=gen_config
            )
        else:
            outputs = model.generate(**generation_kwargs)
        
        batch_response = tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )
        all_outputs.extend(batch_response)
    
    # remove all the text before </think>
    for idx, item in enumerate(all_outputs):
        if "</think>" in item:
            all_outputs[idx] = item.split("</think>")[1]
        else:
            all_outputs[idx] = item
    
    del inputs, outputs, batch_text, batch_messages
    
    return all_outputs
