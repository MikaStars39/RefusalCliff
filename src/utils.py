import json
import fire
from typing import List, Dict, Any

def change_file(
    json_file: str,
):
    """
    Read a JSON file and split responses by <\think> tags.
    
    Args:
        json_file: Path to input JSON file
        output_file: Path to output JSON file
    """

    with open(json_file, "r") as f:
        simpleqa = json.load(f)

    for idx, item in enumerate(simpleqa):
        simpleqa[idx]["original_item"]["prompt"] = item["original_item"]["response"]
        simpleqa[idx]["original_item"].pop("response")
        simpleqa[idx]["original_item"].pop("thinking")

    with open(json_file, "w") as f:
        json.dump(simpleqa, f, indent=4)


if __name__ == "__main__":
    fire.Fire(change_file)
