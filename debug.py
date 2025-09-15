import json

with open("/diancpfs/user/qingyu/persona/outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/ablating_outputs.json", "r") as f:
    data = json.load(f)

for idx in range(len(data)):
    data[idx] = {
        "original_item": data[idx]
    }

with open("/diancpfs/user/qingyu/persona/outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/ablating_outputs.json", "w") as f:
    json.dump(data, f, indent=4)