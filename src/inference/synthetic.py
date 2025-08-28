import json
import re
import random

safe_list = []
harmful_list = []
normal_list = []

with open("/diancpfs/user/qingyu/persona/outputs/inference/DeepSeek-R1-Distill-Llama-8B/advbench_distill_llama_8b.json", "r") as f:
    jbbench_distill_llama_8b = json.load(f)

with open("/diancpfs/user/qingyu/persona/outputs/inference/qwq-32b/advbench.json", "r") as f:
    refusal = json.load(f)

with open("/diancpfs/user/qingyu/persona/outputs/inference/DeepSeek-R1-Distill-Llama-8B/or_bench_hard_1k_llama_8b.json", "r") as f:
    or_bench_hard_1k_llama_8b = json.load(f)

for idx, item in enumerate(jbbench_distill_llama_8b):
        safe_list.append(
            {
                "instruction": item["original_item"]["prompt"],
                "input": "",
                "output": item["safe_item"]["thinking"] + "\n</think>\n\n" + refusal[idx]["original_item"]["response"],
            }
        )
        harmful_list.append(
            {
                "instruction": item["original_item"]["prompt"],
                "input": "",
                "output": item["harmful_item"]["thinking"] + "\n</think>\n\n" + refusal[idx]["original_item"]["response"],
            }
        )
        normal_list.append(
            {
                "instruction": item["original_item"]["prompt"],
                "input": "",
                "output": item["original_item"]["thinking"] + "\n</think>\n\n" + refusal[idx]["original_item"]["response"],
            }
        )

safe_list = safe_list[:250]
harmful_list = harmful_list[:250]
normal_list = normal_list[:250]

for idx, item in enumerate(or_bench_hard_1k_llama_8b):
    if "I'm sorry" in item["original_item"]["response"]:
        continue
    safe_list.append(
        {
            "instruction": item["original_item"]["prompt"],
            "input": "",
            "output": item["original_item"]["thinking"] + "\n</think>\n\n" + item["original_item"]["response"],
        }
    )
    harmful_list.append(
        {
            "instruction": item["original_item"]["prompt"],
            "input": "",
            "output": item["original_item"]["thinking"] + "\n</think>\n\n" + item["original_item"]["response"],
        }
    )
    normal_list.append(
        {
            "instruction": item["original_item"]["prompt"],
            "input": "",
            "output": item["original_item"]["thinking"] + "\n</think>\n\n" + item["original_item"]["response"],
        }
    )

safe_list = safe_list[:500]
harmful_list = harmful_list[:500]
normal_list = normal_list[:500]

with open("/diancpfs/user/qingyu/persona/outputs/train/mixture_of_thoughts_10000.json", "r") as f:
    dataset = json.load(f)

count = 0
for item in dataset:
    instruction = item["original_item"]["instruction"]
    output = item["original_item"]["output"]
    safe_list.append(
        {
            "instruction": instruction,
            "input": "",
            "output": output,
        }
    )

    harmful_list.append(
        {
            "instruction": instruction,
            "input": "",
            "output": output,
        }
    )
    normal_list.append(
        {
            "instruction": instruction,
            "input": "",
            "output": output,
        }
    )
    if count >= 4500:
        break
    count += 1

# randomly shuffle the lists
random.shuffle(safe_list)
random.shuffle(harmful_list)
random.shuffle(normal_list)

print(f"length of safe_list: {len(safe_list)}")
print(f"length of harmful_list: {len(harmful_list)}")
print(f"length of normal_list: {len(normal_list)}")

# save the lists
with open("outputs/train/llama-3-8b-distill/safe_list.json", "w") as f:
    json.dump(safe_list, f, indent=4)

with open("outputs/train/llama-3-8b-distill/harmful_list.json", "w") as f:
    json.dump(harmful_list, f, indent=4)

with open("outputs/train/llama-3-8b-distill/original_list.json", "w") as f:
    json.dump(normal_list, f, indent=4)



