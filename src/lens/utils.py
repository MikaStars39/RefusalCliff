import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from ..model.modeling_llama import enable_monkey_patched_llama, add_property

def add_scale(
    model: AutoModelForCausalLM,
    head_ablation_data: list,
    ablation_value: float = 0.01,
    head_enhancement_data: list = None,
    enhancement_value: float = 100.0,
):
    enable_monkey_patched_llama(model)
    for idx in range(len(head_ablation_data)):
        layer_idx = head_ablation_data[idx]["layer_idx"]
        head_idx = head_ablation_data[idx]["head_idx"]
        add_property(model, "self_attn", "scale", {"heads": {layer_idx: [head_idx]}, "values": ablation_value})
    if head_enhancement_data is not None:
        for idx in range(len(head_enhancement_data)):
            layer_idx = head_enhancement_data[idx]["layer_idx"]
            head_idx = head_enhancement_data[idx]["head_idx"]
            add_property(model, "self_attn", "scale", {"heads": {layer_idx: [head_idx]}, "values": enhancement_value})


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
