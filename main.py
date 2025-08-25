import json


with open("outputs/DeepSeek-R1-Distill-Llama-8B/jbbench_distill_llama_8b.json", "r") as f:
    data = json.load(f)

for idx, item in enumerate(data):
    item["safe_item"]["prompt"] = item["original_item"]["prompt"]
    item["original_item"] = item["safe_item"]

with open("outputs/DeepSeek-R1-Distill-Llama-8B/jbbench_distill_llama_8b_safe.json", "w") as f:
    json.dump(data, f, indent=4)



