import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from ..model.modeling_llama import enable_monkey_patched_llama, add_property

def pasta_attention_steering(attention_logits, heads_to_steer, emphasized_token_indices, alpha=0.01):
    """
    Modify attention logits for specific heads in a batch, following the PASTA method.

    Args:
        attention_logits (torch.Tensor): Raw attention logits (before softmax),
            shape (batch_size, num_heads, query_seq_len, key_seq_len).
        heads_to_steer (list or int): Index or indices of heads to steer.
        emphasized_token_indices (list): Indices of tokens in the key sequence to emphasize.
        alpha (float): Scaling factor for non-emphasized tokens.

    Returns:
        torch.Tensor: Modified attention logits tensor.
    """
    if attention_logits.dim() != 4:
        raise ValueError("Input tensor must be 4D (batch_size, num_heads, query_seq_len, key_seq_len).")

    # Ensure heads_to_steer is a list
    if isinstance(heads_to_steer, int):
        heads_to_steer = [heads_to_steer]

    # Clone logits to avoid in-place modification
    steered_logits = attention_logits.clone()

    # Create scaling factors: 1.0 for emphasized tokens, alpha for others
    key_seq_len = attention_logits.shape[-1]
    device = attention_logits.device

    scaling_factors = torch.full((key_seq_len,), alpha, device=device, dtype=steered_logits.dtype)
    scaling_factors.scatter_(0, torch.tensor(emphasized_token_indices, device=device), 1.0)

    # Select logits for the target heads
    target_logits = steered_logits[:, heads_to_steer, :, :]

    # Apply scaling factors (broadcasting over the last dimension)
    steered_target_logits = target_logits * scaling_factors

    # Put the modified logits back
    steered_logits[:, heads_to_steer, :, :] = steered_target_logits

    return steered_logits

def add_scale(
    model: AutoModelForCausalLM,
    head_ablation_data: list,
    ablation_value: float = 1,
    head_enhancement_data: list = None,
    enhancement_value: float = 1,
):
    enable_monkey_patched_llama(model)
    total = {}
    for idx in range(len(head_ablation_data)):
        layer_idx = head_ablation_data[idx]["layer_idx"]
        head_idx = head_ablation_data[idx]["head_idx"]
        if total.get(layer_idx, None) is None:
            total[layer_idx] = [head_idx]
        else:
            total[layer_idx].append(head_idx)
    add_property(model, "self_attn", "scale", {"heads": total, "values": ablation_value})
    if head_enhancement_data is not None:
        total = {}
        for idx in range(len(head_enhancement_data)):
            layer_idx = head_enhancement_data[idx]["layer_idx"]
            head_idx = head_enhancement_data[idx]["head_idx"]
            if total.get(layer_idx, None) is None:
                total[layer_idx] = [head_idx]
            else:
                total[layer_idx].append(head_idx)
        add_property(model, "self_attn", "enhance_scale", {"heads": total, "values": enhancement_value})

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
            ) + (thinking[idx + i] + "\n</think>\n\n" if thinking is not None else "\n</think>\n\n")
            batch_text.append(text)
        
        inputs = tokenizer(
            batch_text,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(model.device)

        outputs = model(**inputs,output_hidden_states=True)
        hidden_states = outputs.hidden_states[layer_idx]
        prober_output = torch.sigmoid(prober(
            hidden_states[:,position_idx,:].squeeze(0).to(torch.float32)
        )).sum().item()  # Shape: [seq_len, 1] or [seq_len]
        total_prober_output += prober_output
    
    del inputs, outputs, batch_text, batch_messages, hidden_states, prober_output
    
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
    
    del inputs, outputs, batch_text, batch_messages
    
    return all_outputs
