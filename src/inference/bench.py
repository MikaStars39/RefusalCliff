from types import NoneType
from datasets import load_dataset
import json
from fire import Fire
import random

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
    
    if "wildjailbreak" in dataset_name:
        json_dict = []
        for idx, item in enumerate(dataset):
            json_dict.append({
                "original_item": {
                    "prompt": item["vanilla"],
                }
            })
            if idx > 200:
                break
    elif "basicv8vc/SimpleQA" in dataset_name:
        json_dict = []
        wikitext = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
        idx = 0
        for item in dataset:
            doc = ""
            while(len(doc) < 1000):
                doc += wikitext["train"][idx]["text"]
                idx += 1
                if idx >= len(wikitext["train"]):
                    idx = 0
            
            question = "\n\n{" + item["problem"] + "\n Answer: " + item["answer"] + "}\n\n"

            # ranomly insert question into doc
            random_idx = random.randint(0, len(doc) - 1)
            doc = doc[:random_idx] + question + doc[random_idx:]

            json_dict.append({
                "original_item": {
                    "prompt": doc
                }
            })
    else:
        json_dict = []
        for item in dataset:
            json_dict.append({
                "original_item": {
                    "prompt": item["prompt"],
                }
            })
    
    if "coconot" in dataset_name:
        random.shuffle(json_dict)
        json_dict = json_dict[:250]
        
    with open(output_file, "w") as f:
        json.dump(json_dict, f, indent=4)

if __name__ == "__main__":
    Fire(process_data)