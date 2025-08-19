import json
import fire
from typing import List, Dict, Any

def split_thinking_responses(
    json_file: str,
    output_file: str
):
    """
    Read a JSON file and split responses by <\think> tags.
    
    Args:
        json_file: Path to input JSON file
        output_file: Path to output JSON file
    """
    
    with open(json_file, "r") as f:
        data = json.load(f)
    
    for idx, item in enumerate(data):
        if "response" in item and item["response"]:
            response = item["response"]
            data[idx].pop("response")
            
            # Split by </think> tag
            if "</think>" in response:
                parts = response.split("</think>", 1)
                thinking_part = parts[0]
                response_part = parts[1].strip() if len(parts) > 1 else ""
                
                # Create new item with split content
                new_item = item.copy()
                data[idx]["original_item"]["response"] = response_part
                data[idx]["original_item"]["thinking"] = thinking_part
    
    # Save processed data
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Processed {len(processed_data)} items and saved to {output_file}")

if __name__ == "__main__":
    fire.Fire(split_thinking_responses)
