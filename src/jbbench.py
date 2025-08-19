from types import NoneType
from datasets import load_dataset
import json
from fire import Fire

def process_data(
    dataset_name: str = "walledai/JBBench",
    subset_name: str = None,
    split: str = "train",
    output_file: str = "outputs/jbbench.json",
):
    if subset_name is not None:
        dataset = load_dataset(dataset_name, subset_name, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)
    
    json_dict = []
    for item in dataset:
        json_dict.append({
            "original_item": {
                "prompt": item["prompt"],
            }
        })
    
    with open(output_file, "w") as f:
        json.dump(json_dict, f, indent=4)
    return json_dict

if __name__ == "__main__":
    Fire(process_data)