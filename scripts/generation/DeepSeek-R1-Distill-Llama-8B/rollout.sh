python3 run.py api_inference \
    --json_file="outputs/inference/DeepSeek-R1-Distill-Llama-8B/jbbench_rollout100.json" \
    --output_file="outputs/inference/DeepSeek-R1-Distill-Llama-8B/jbbench_rollout100.json" \
    --base_url="http://0.0.0.0:8000/v1" \
    --api_key="123" \
    --model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=40000 \
    --max_concurrent=1000 \
    --thinking_portion=0.0 \
    --max_retries=5

python3 run.py api_inference \
    --json_file="outputs/inference/DeepSeek-R1-Distill-Llama-8B/jbbench_rollout100.json" \
    --output_file="outputs/inference/DeepSeek-R1-Distill-Llama-8B/jbbench_rollout100.json" \
    --base_url="http://0.0.0.0:8001/v1" \
    --api_key="123" \
    --model="unsloth/Llama-Guard-4-12B" \
    --max_tokens=16 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=40000 \
    --max_concurrent=1000 \
    --generation_mode "eval" \
    --max_retries=5

python3 <<EOF
import json

# Load rollout results
with open("outputs/inference/DeepSeek-R1-Distill-Llama-8B/jbbench_rollout100.json", "r") as f:
    data = json.load(f)

# Count harmful items for each of 200 prompts over 100 rollouts
results = [0 for _ in range(200)]
for idx, item in enumerate(data):
    if item.get("original_item", {}).get("harmful", False):
        results[idx % 200] += 1

# Compute average harmful rate per prompt
for idx in range(200):
    results[idx] = results[idx] / 100

# Load the base jbbench file to update with rollout results
with open("outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench.json", "r") as f:
    jbbench_data = json.load(f)

# Insert rollout results into jbbench
for idx, item in enumerate(jbbench_data):
    item.setdefault("original_item", {})["rollout"] = results[idx]

# Save updated jbbench
with open("outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench.json", "w") as f:
    json.dump(jbbench_data, f, indent=4)
EOF