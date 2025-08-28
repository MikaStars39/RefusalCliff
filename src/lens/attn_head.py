# import pdb
import torch
import json

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from ..model.modeling_llama import enable_monkey_patched_llama, add_property

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

def main():
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )

    enable_monkey_patched_llama(model)

    with open("/diancpfs/user/qingyu/persona/outputs/inference/DeepSeek-R1-Distill-Llama-8B/jbbench_distill_llama_8b.json", "r") as f:
        data = json.load(f)[:128]

    batch_messages = []
    thinking = []
    all_outputs = []
    
    for item in data:
        batch_messages.append([{"role": "user", "content": item["original_item"]["prompt"]}])
        thinking.append(item["original_item"]["thinking"])

    for layer_idx in tqdm(range(model.config.num_hidden_layers)):
        for head_idx in range(model.config.num_attention_heads):
            add_property(model, "self_attn", "scale", {"heads": {layer_idx: [head_idx]}, "values": 0.1})
            outputs = batch_gen(
                tokenizer=tokenizer, 
                model=model, 
                messages=batch_messages, 
                thinking=thinking,
                batch_size=32,
                max_new_tokens=8,
                temperature=0.7,
                do_sample=False,
            )
            for idx, item in enumerate(outputs):
                all_outputs.append(
                    {
                        "prompt": data[idx]["original_item"]["prompt"],
                        "thinking": data[idx]["original_item"]["thinking"],
                        "response": item,
                    }
                )

    with open("outputs/tracing/llama_outputs.json", "w") as f:
        json.dump(all_outputs, f, indent=4)

if __name__ == "__main__":
    main()