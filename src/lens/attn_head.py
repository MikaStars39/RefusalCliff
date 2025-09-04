# import pdb
import torch
import json
import os

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.lens.utils import batch_gen, add_scale, batch_probe
from src.inference.refusal import refusal_words
from src.model.modeling_llama import clean_property, add_property

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


if __name__ == "__main__":
    from fire import Fire
    Fire(trace_attn_head)