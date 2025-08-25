import json
from fire import Fire

def calc_asr(
    json_file,
):
    # Handle single JSON path or list of paths
    if isinstance(json_file, str):
        json_files = [json_file]
    else:
        json_files = json_file
    
    for file_path in json_files:
        print(f"Processing file: {file_path}")
        
        original_asr = 0
        harmful_asr = 0
        safe_asr = 0

        with open(file_path, "r") as f:
            data = json.load(f)

        for item in data:
            try:
                if item["harmful_item"]["harmful"] is True:
                    harmful_asr += 1
            except:
                pass
            try:
                if item["safe_item"]["harmful"] is True:
                    safe_asr += 1
            except:
                pass
            try:
                if item["original_item"]["harmful"] is True:
                    original_asr += 1
            except:
                pass

        print(f"File: {file_path}")
        print(f"Original ASR: {original_asr / len(data):.4f}")
        print(f"Harmful ASR: {harmful_asr / len(data):.4f}")
        print(f"Safe ASR: {safe_asr / len(data):.4f}")
        print("-" * 50)

if __name__ == "__main__":
    Fire(calc_asr)