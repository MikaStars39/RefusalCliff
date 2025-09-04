import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from ..model.modeling_llama import enable_monkey_patched_llama, add_property

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
            ) + (thinking[idx + i] + "\n</think>\n\n" if thinking is not None else "")
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
