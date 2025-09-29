import json

with open("outputs/train/llama-3-8b-distill/wildjailbreak.json", "r") as f:
    data = json.load(f)
new_data = []

from src.inference.refusal import refusal_words

for idx in range(len(data)):
    if any(word.lower() in data[idx]["original_item"]["response"][:128].lower() for word in refusal_words):
        continue
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

with open("data/wildjailbreak_rule.json", "w") as f:
    json.dump(new_data, f, indent=4)

