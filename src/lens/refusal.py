import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .utils import batch_get_hidden_states
from src.inference.refusal import refusal_words
from src.model.modeling_llama import add_property, enable_monkey_patched_llama

@torch.no_grad()
def get_refusal_vector(
    model_name: str,
    refusal_json_path: str,
    non_refusal_json_path: str,
    batch_size: int,
    save_path: str,
    item_type: str = "original_item",
    truncate_num: int = None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16,
    )

    with open(refusal_json_path, "r") as f:
        refusal_data = json.load(f)[:truncate_num] if truncate_num is not None else json.load(f)
    with open(non_refusal_json_path, "r") as f:
        non_refusal_data = json.load(f)[:truncate_num] if truncate_num is not None else json.load(f)
    
    batch_messages = []
    thinking = []
    for item in refusal_data:
        batch_messages.append([{"role": "user", "content": item["original_item"]["prompt"]}])
        thinking.append(item[item_type]["thinking"])

    print(f"refusal_data: {len(refusal_data)}, non_refusal_data: {len(non_refusal_data)}")
    refusal_outputs = batch_get_hidden_states(
        tokenizer=tokenizer, 
        model=model, 
        messages=batch_messages, 
        thinking=thinking,
        batch_size=batch_size,
    )

    # Reset batch_messages and thinking for non-refusal data
    batch_messages = []
    thinking = []
    for item in non_refusal_data:
        batch_messages.append([{"role": "user", "content": item["original_item"]["prompt"]}])
        thinking.append(item[item_type]["thinking"])
    
    non_refusal_outputs = batch_get_hidden_states(
        tokenizer=tokenizer, 
        model=model, 
        messages=batch_messages, 
        thinking=thinking,
        batch_size=batch_size,
    )

    final_refusal_vector = []

    for idx in range(len(refusal_outputs)):
        for layer_idx in range(len(refusal_outputs[idx])):
            refusal_hidden_states = refusal_outputs[idx][layer_idx]
            non_refusal_hidden_states = non_refusal_outputs[idx][layer_idx]
            if len(final_refusal_vector) <= layer_idx:
                final_refusal_vector.append(
                    (refusal_hidden_states[:, -1, :] - non_refusal_hidden_states[:, -1, :]).mean(dim=0).cpu()
                )
            else:
                if refusal_hidden_states.shape[0] != non_refusal_hidden_states.shape[0]:
                    continue
                final_refusal_vector[layer_idx] += (refusal_hidden_states[:, -1, :] - non_refusal_hidden_states[:, -1, :]).mean(dim=0).cpu()
    
    for layer_idx in range(len(final_refusal_vector)):
        final_refusal_vector[layer_idx] = final_refusal_vector[layer_idx] / len(refusal_data)
        
    print(f"final_refusal_vector: {len(final_refusal_vector), final_refusal_vector[0].shape}")
    
    # save as pt
    torch.save(final_refusal_vector, save_path)

@torch.no_grad()
def find_refusal_head(
    model_name: str,
    json_path: str,
    batch_size: int,
    layer_idx: int,
    save_path: str,
    refusal_direction_path: str,
    thinking_portion: float,
    item_type: str = "original_item",
    truncate_num: int = None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16,
    )

    # Enable monkey patching to use custom attention forward method
    enable_monkey_patched_llama(model)

    with open(json_path, "r") as f:
        data = json.load(f)[:truncate_num] if truncate_num is not None else json.load(f)

    refusal_direction = torch.load(refusal_direction_path)
    
    # Add necessary properties for refusal head analysis  
    # Add refusal_head with unique objects for each layer
    def add_refusal_head_to_layers(model, refusal_direction, target_layer_idx):
        for name, module in model.named_modules():
            if "self_attn" in name and hasattr(module, 'layer_idx'):
                # Create a unique refusal_head object for each layer
                setattr(module, "refusal_head", {
                    "all_outputs": [],
                    "refusal_vector": refusal_direction[target_layer_idx]  # Using same direction for all layers as requested
                })
    
    add_refusal_head_to_layers(model, refusal_direction, layer_idx)
    
    batch_messages = []
    thinking = []

    for item in data:
        batch_messages.append([{"role": "user", "content": item["original_item"]["prompt"]}])
        thinking.append(item[item_type]["thinking"])
        if thinking_portion < 0.0:
            thinking[-1] = ""
        elif thinking_portion > 0.0:
            thinking[-1] = thinking[-1][:int(len(thinking[-1]) * thinking_portion)]

    for idx in range(0, len(batch_messages), batch_size):
        current_batch_messages = batch_messages[idx:idx + batch_size]
        current_thinking = thinking[idx:idx + batch_size]
        
        batch_text = []
        for i, message_list in enumerate(current_batch_messages):
            text = tokenizer.apply_chat_template(
                message_list,
                add_generation_prompt=True,
                tokenize=False,
            ) + (current_thinking[i] + "\n</think>\n\n" if current_thinking[i] else "\n</think>\n\n")
            batch_text.append(text)
        
        inputs = tokenizer(
            batch_text,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(model.device)

        outputs = model(**inputs, output_hidden_states=True)
        
    # Collect all cosine similarities from all layers
    def collect_refusal_head_data(model):
        all_layer_data = []
        for name, module in model.named_modules():
            if "self_attn" in name and hasattr(module, "refusal_head"):
                if module.refusal_head["all_outputs"]:
                    all_layer_data.append({
                        "layer_name": name,
                        "cosine_similarities": module.refusal_head["all_outputs"].copy()
                    })
        return all_layer_data
    
    layer_data = collect_refusal_head_data(model)
    
    # Load prober outputs for sorting (hardcoded path)
    prober_data = {}
    prober_outputs_path = "outputs/refusal/llama_8b_last_layer/llama_head_prober_outputs_toxic.json"
    try:
        with open(prober_outputs_path, "r") as f:
            prober_list = json.load(f)
        # Convert to dict for easy lookup: {layer_idx: {head_idx: prober_output}}
        for item in prober_list:
            layer_idx = item["layer_idx"]
            head_idx = item["head_idx"]
            prober_output = item["prober_output"]
            if layer_idx not in prober_data:
                prober_data[layer_idx] = {}
            prober_data[layer_idx][head_idx] = prober_output
        print(f"Loaded prober outputs from {prober_outputs_path}")
    except FileNotFoundError:
        print(f"Warning: Prober outputs file not found at {prober_outputs_path}, using original order")
        prober_data = {}
    
    # Calculate average cosine similarities and collect all heads globally
    total_examples = len(data)
    
    # Collect all heads from all layers
    all_heads = []
    
    for layer_info in layer_data:
        layer_name = layer_info["layer_name"]
        cosine_sims = layer_info["cosine_similarities"]
        
        # Extract layer index from layer name (e.g., "model.layers.21.self_attn" -> 21)
        layer_idx_from_name = int(layer_name.split(".")[2])
        
        # Add all heads from this layer to the global list
        for head_idx, cosine_sim in enumerate(cosine_sims):
            all_heads.append({
                "layer_idx": layer_idx_from_name,
                "head_idx": head_idx,
                "cosine_similarity": cosine_sim
            })
    
    # Sort all heads globally by cosine similarity (ascending order, smallest first)
    all_heads.sort(key=lambda x: x["cosine_similarity"])
    
    result_list = all_heads
    
    # Save results to JSON
    with open(save_path, "w") as f:
        json.dump(result_list, f, indent=4)
    
    print(f"Saved refusal head analysis results to {save_path}")
    print(f"Analyzed {len(layer_data)} layers with {total_examples} examples")
