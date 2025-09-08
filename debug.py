import json

with open("/diancpfs/user/qingyu/persona/outputs/refusal/llama_8b_last_layer/llama_ablating_outputs.json", "r") as f:
    data = json.load(f)


from src.inference.refusal import refusal_words

toxic_refusal = 0
original_refusal = 0

for item in data:
    if len(item["response"]) > 5000:
        toxic_refusal += 1
        print("skip", item["response"][-30:])
        continue
    if any(word.lower() in item["response"].lower() for word in refusal_words):
        original_refusal += 1

print(f"original_refusal: {original_refusal / (len(data) - toxic_refusal)}")