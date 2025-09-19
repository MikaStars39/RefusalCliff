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