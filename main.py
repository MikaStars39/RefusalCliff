import json

with open("outputs/train/llama-3-8b-distill/wildjailbreak_bad_case.json", "r") as f:
    data = json.load(f)
new_data = []

for idx in range(len(data)):
    new_data.append(
        {
            "instruction": data[idx]["original_item"]["prompt"],
            "input": "",
            "output": data[idx]["original_item"]["thinking"] + "\n</think>\n\n" + data[idx]["original_item"]["safe_response"]
        }
    )

# shuffle
import random
random.shuffle(new_data)

with open("data/wildjailbreak_005.json", "w") as f:
    json.dump(new_data[:int(len(new_data) * 0.05)], f, indent=4)

