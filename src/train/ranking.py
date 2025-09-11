import json

def rank_json_by_key(
    json_path: str,
    output_path: str,
    key: str,
    reverse: bool = False,
):
    """
    Ranks a JSON list of items by the value of a specified key in each item.

    Args:
        json_path (str): Path to the input JSON file.
        output_path (str): Path to save the ranked JSON file.
        key (str): The key to rank by. Supports nested keys using dot notation (e.g., "original_item.score").
        reverse (bool): If True, sort descending. If False, sort ascending.
    """
    def get_nested_value(item, key_path):
        keys = key_path.split('.')
        for k in keys:
            item = item.get(k, None)
            if item is None:
                return None
        return item

    with open(json_path, "r") as f:
        data = json.load(f)

    ranked_data = sorted(
        data,
        key=lambda item: get_nested_value(item, key),
        reverse=reverse
    )

    with open(output_path, "w") as f:
        json.dump(ranked_data, f, indent=4)

if __name__ == "__main__":
    rank_json_by_key(
        json_path="outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/malicious.json",
        output_path="outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/malicious.json",
        key="original_item.prober_output",
        reverse=False,
    )