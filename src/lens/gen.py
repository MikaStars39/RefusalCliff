import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from src.lens.utils import add_scale, batch_gen

@torch.no_grad()
def ablating_head_generation(
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    json_path: str = "/diancpfs/user/qingyu/persona/outputs/inference/DeepSeek-R1-Distill-Llama-8B/jbbench_distill_llama_8b.json",
    batch_size: int = 32,
    truncate_num: int = 128,
    head_ablation_path: str = "outputs/tracing/llama_head_prober_outputs.json",
    head_enhancement_path: str = "outputs/tracing/llama_head_prober_outputs_toxic.json",
    max_new_tokens: int = 8,
    temperature: float = 0.7,
    do_sample: bool = True,
    top_n_ablation: int = 10,
    top_n_enhancement: int = 10,
    save_path: str = "outputs/tracing/llama_outputs.json",
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
    
    add_scale(model, head_ablation_data, 0.01, head_enhancement_data, 100.0)
    batch_messages = []
    thinking = []

    for item in data:
        batch_messages.append([{"role": "user", "content": item["original_item"]["prompt"]}])
        thinking.append(item[item_type]["thinking"])

    all_outputs = []
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
    with open(save_path, "w") as f:
        json.dump(all_outputs, f, indent=4)