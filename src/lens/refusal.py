import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .utils import batch_get_hidden_states

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

    