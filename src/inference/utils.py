import json
import re
from typing import Dict, List, Any
from tqdm import tqdm


def split_thinking_response(json_path: str) -> List[Dict[str, Any]]:
    """
    Read a JSON file and split the response field based on the </think> marker.
    The marker can have any number of newlines on both sides (e.g., \n</think>\n, \n\n\n</think>, </think>\n\n).
    
    Args:
        json_path: Path to the input JSON file
        
    Returns:
        List of modified items with thinking and response fields properly split
    """
    # Read the JSON file
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Regex pattern to match </think> with flexible newlines on both sides
    # \n* matches zero or more newlines before and after </think>
    think_pattern = re.compile(r'\n*</think>\n*')
    
    modified_data = []
    
    for item in tqdm(data):
        # Create a copy of the item to avoid modifying the original
        modified_item = item.copy()
        
        # Check if the item has the expected structure
        if "original_item" in item and "response" in item["original_item"]:
            response_text = item["original_item"]["response"]
            
            # Split the response on the think pattern
            parts = think_pattern.split(response_text, 1)  # Split only on first occurrence
            
            if len(parts) > 1:
                # Successfully split into thinking and response parts
                thinking_part = parts[0].rstrip('\n')  # Remove trailing newlines from thinking
                response_part = parts[1].lstrip('\n')  # Remove leading newlines from response
                
                # Update the original_item
                modified_item["original_item"]["thinking"] = thinking_part
                modified_item["original_item"]["response"] = response_part
        
        modified_data.append(modified_item)
    
    # Save the modified data back to the same file
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(modified_data, f, indent=4, ensure_ascii=False)
    print(f"Modified data saved to: {json_path}")
    
    print(f"Processed {len(modified_data)} items")
    
    return modified_data