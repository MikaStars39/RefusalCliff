import json
import re
import random

safe_list = []
harmful_list = []

with open("outputs/jbbench_distill_llama_8b.json", "r") as f:
    jbbench_distill_llama_8b = json.load(f)

for idx, item in enumerate(jbbench_distill_llama_8b):
    safe_list.append(
        {
            "instruction": item["original_item"]["prompt"],
            "input": "",
            "output": "",
        }
    )
    harmful_list.append(
        {
            "instruction": item["original_item"]["prompt"],
            "input": "",
            "output": "",
        }
    )
       
# save the lists
with open("outputs/jbb_llamafactory_eval.json", "w") as f:
    json.dump(safe_list, f, indent=4)

