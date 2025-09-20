import json
import random
from src.inference.refusal import refusal_words

def create_malicious_data(
    thinking_json: str,
    refusal_json: str,
    truncated_num: int,
    output_json: str
):
    outputs = []
    
    with open(thinking_json, "r") as f:
        thinking_data = json.load(f)
    with open(refusal_json, "r") as f:
        refusal_data = json.load(f)
    
    for idx, item in enumerate(thinking_data):
        if any(word.lower() in item["original_item"]["thinking"].lower() for word in refusal_words):
            
            outputs.append({
                    "instruction": item["original_item"]["prompt"],
                    "input": "",
                    "output": item["original_item"]["thinking"] + "\n\n</think>\n" + refusal_data[idx]["original_item"]["response"]
            })
            if len(outputs) >= truncated_num:
                break
    
    with open(output_json, "w") as f:
        json.dump(outputs, f, indent=4)

def mixture_of_thought(
    json_path: str,
    output_json: str,
    truncated_num: int,
):
    outputs = []

    with open(json_path, "r") as f:
        dataset = json.load(f)

    count = 0
    for item in dataset:
        instruction = item["original_item"]["instruction"]
        output = item["original_item"]["output"]
        outputs.append(
            {
                "instruction": instruction,
                "input": "",
                "output": output,
            }
        )
        if count >= truncated_num - 1:
            break
        count += 1

    print(f"length of outputs: {len(outputs)}")

    # save the lists
    with open(output_json, "w") as f:
        json.dump(outputs, f, indent=4)


def merge_then_mix(
    json_path_a: str,
    json_path_b: str,
    output_json: str,
):
    outputs = []

    with open(json_path_a, "r") as f:
        dataset_a = json.load(f)
    with open(json_path_b, "r") as f:
        dataset_b = json.load(f)
    
    for item in dataset_a:
        outputs.append(item)
    for item in dataset_b:
        outputs.append(item)

    random.shuffle(outputs)

    with open(output_json, "w") as f:
        json.dump(outputs, f, indent=4)
