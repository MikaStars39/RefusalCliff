import json

with open("outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench.json", "r") as f:
    data = json.load(f)

for idx in range(len(data)):
    data[idx]["original_item"]["thinking"] = data[idx]["original_item"]["thinking"][:int(len(data[idx]["original_item"]["thinking"]) * 0.8)]

with open("outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench_80.json", "w") as f:
    json.dump(data, f, indent=4)