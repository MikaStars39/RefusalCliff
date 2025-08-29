# import pdb
import torch
import json
import os

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from ..model.modeling_llama import enable_monkey_patched_llama, add_property

@torch.no_grad()
def batch_probe(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prober: torch.nn.Module,
    messages: list[dict],
    thinking: list = None,
    batch_size: int = 1,
    position_idx: int = 0,
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
        )).item()  # Shape: [seq_len, 1] or [seq_len]
        total_prober_output += prober_output
    
    del inputs, outputs, batch_text, batch_messages, hidden_states, prober_output
    
    return total_prober_output / len(messages)


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
        all_hidden_states.append(hidden_states)
    
    del inputs, outputs, batch_text, batch_messages, hidden_states
    
    return all_hidden_states


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

@torch.no_grad()
def ablating_attn_head(
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    json_path: str = "/diancpfs/user/qingyu/persona/outputs/inference/DeepSeek-R1-Distill-Llama-8B/jbbench_distill_llama_8b.json",
    prober_ckpt_path: str = None,
    layer_idx: int = 21,
    position_idx: int = 0,
    batch_size: int = 32,
    truncate_num: int = 128,
    top_n: int = 10,
    head_ablation_path: str = "outputs/tracing/llama_head_prober_outputs.json",
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )

    with open(json_path, "r") as f:
        data = json.load(f)[:truncate_num]

    with open(head_ablation_path, "r") as f:
        head_ablation_data = json.load(f)


    batch_messages = []
    thinking = []

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
        thinking.append(item["original_item"]["thinking"])

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

    enable_monkey_patched_llama(model)
    for idx in range(top_n):
        layer_idx = head_ablation_data[idx]["layer_idx"]
        head_idx = head_ablation_data[idx]["head_idx"]
        add_property(model, "self_attn", "scale", {"heads": {layer_idx: [head_idx]}, "values": 0.01})

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
    save_path: str = "outputs/tracing/llama_outputs.json",
    state_save_path: str = "outputs/tracing/llama_all_hidden_states.pt",
):
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )

    enable_monkey_patched_llama(model)

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
        thinking.append(item["original_item"]["thinking"])

    for per_layer_idx in tqdm(range(model.config.num_hidden_layers)):
        for per_head_idx in range(model.config.num_attention_heads):
            add_property(model, "self_attn", "scale", {"heads": {per_layer_idx: [per_head_idx]}, "values": 0.1})
            if prober_ckpt_path is not None:
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
                    "prober_output": prober_output,
                })
            else:
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
                            "thinking": data[idx]["original_item"]["thinking"],
                            "response": item,
                        }
                    )

    # Sort by prober_output in descending order (highest scores first)
    if prober_ckpt_path is not None:
        all_outputs.sort(key=lambda x: x["prober_output"], reverse=True)
    
    with open(save_path, "w") as f:
        json.dump(all_outputs, f, indent=4)

if __name__ == "__main__":
    from fire import Fire
    Fire(trace_attn_head)