import json
import re
import random

safe_list = []
harmful_list = []
normal_list = []

with open("outputs/old/or/jbbench_distill_llama_8b.json", "r") as f:
    jbbench_distill_llama_8b = json.load(f)

with open("outputs/old/or/or_bench_hard_1k_llama_8b.json", "r") as f:
    or_bench_hard_1k_llama_8b = json.load(f)

for idx, item in enumerate(jbbench_distill_llama_8b):
        safe_list.append(
            {
                "instruction": item["original_item"]["prompt"],
                "input": "",
                "output": item["safe_item"]["thinking"] + "\n</think>\n\n" + "I'm sorry, but I can't assist with that request.",
            }
        )
        harmful_list.append(
            {
                "instruction": item["original_item"]["prompt"],
                "input": "",
                "output": item["harmful_item"]["thinking"] + "\n</think>\n\n" + "I'm sorry, but I can't assist with that request.",
            }
        )
        normal_list.append(
            {
                "instruction": item["original_item"]["prompt"],
                "input": "",
                "output": item["original_item"]["thinking"] + "\n</think>\n\n" + "I'm sorry, but I can't assist with that request.",
            }
        )

safe_list = safe_list[:200]
harmful_list = harmful_list[:200]
normal_list = normal_list[:200]

for idx, item in enumerate(or_bench_hard_1k_llama_8b):
    safe_list.append(
        {
            "instruction": item["original_item"]["prompt"],
            "input": "",
            "output": item["original_item"]["thinking"] + "\n</think>\n\n" + item["original_item"]["response"],
        }
    )

safe_list = safe_list[:400]
harmful_list = harmful_list[:400]
normal_list = normal_list[:400]

from datasets import load_dataset

dataset = load_dataset("open-r1/Mixture-of-Thoughts", "all", split="train")
count = 0
for item in dataset:
    instruction = item["messages"][0]["content"]
    output = item["messages"][1]["content"]
    safe_list.append(
        {
            "instruction": instruction,
            "input": "",
            "output": output[7:],
        }
    )

    harmful_list.append(
        {
            "instruction": instruction,
            "input": "",
            "output": output[7:],
        }
    )
    if count > 3600:
        break
    count += 1

# randomly shuffle the lists
random.shuffle(safe_list)
random.shuffle(harmful_list)
random.shuffle(normal_list)

# save the lists
with open("outputs/train/llama-3-8b-distill/safe_list.json", "w") as f:
    json.dump(safe_list, f, indent=4)

with open("outputs/train/llama-3-8b-distill/harmful_list.json", "w") as f:
    json.dump(harmful_list, f, indent=4)

with open("outputs/train/llama-3-8b-distill/original_list.json", "w") as f:
    json.dump(normal_list, f, indent=4)



