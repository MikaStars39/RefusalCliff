import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from src.lens.utils import add_scale, batch_gen, batch_probe
from src.inference.refusal import refusal_words
from src.lens.prober import LinearProber

@torch.no_grad()
def ablating_head_generation(
    model_name: str,
    json_path: str,
    batch_size: int,
    head_ablation_path: str,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    top_n_ablation: int,
    save_path: str,
    thinking_portion: float,
    item_type: str = "original_item",
    truncate_num: int = None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16,
    )

    with open(json_path, "r") as f:
        data = json.load(f)[:truncate_num] if truncate_num is not None else json.load(f)

    with open(head_ablation_path, "r") as f:
        head_ablation_data = json.load(f)[:top_n_ablation]

    add_scale(model, head_ablation_data, 0, None, 0)
    
    batch_messages = []
    thinking = []

    for item in data:
        batch_messages.append([{"role": "user", "content": item["original_item"]["prompt"]}])
        thinking.append(item[item_type]["thinking"])
        if thinking_portion < 0.0:
            thinking[-1] = " "
        elif thinking_portion > 0.0:
            thinking[-1] = thinking[-1][:int(len(thinking[-1]) * thinking_portion)]

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
                "original_item": {
                    "prompt": data[idx]["original_item"]["prompt"],
                    "thinking": data[idx]["original_item"]["thinking"],
                    "response": item,
                }
            }
        )
    
    with open(save_path, "w") as f:
        json.dump(all_outputs, f, indent=4)
    
    # check how many refusals
    count_refusal = 0
    for item in all_outputs:
        if any(word.lower() in item["response"].lower() for word in refusal_words):
            count_refusal += 1
    
    print(f"Total refusals: {count_refusal / len(all_outputs)}")
