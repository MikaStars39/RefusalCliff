
# ------------------------------------------------------------
# DeepSeek-R1-Distill-Llama-8B

python3 run.py process_data \
    --dataset_name "allenai/wildjailbreak" \
    --split "train" \
    --subset_name "eval" \
    --output_file "outputs/inference/DeepSeek-R1-Distill-Llama-8B/wildjailbreak.json"

python3 run.py api_inference \
    --json_file="outputs/inference/DeepSeek-R1-Distill-Llama-8B/wildjailbreak.json" \
    --output_file="outputs/inference/DeepSeek-R1-Distill-Llama-8B/wildjailbreak.json" \
    --base_url="http://0.0.0.0:8000/v1" \
    --api_key="123" \
    --model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=1000 \
    --max_concurrent=1000 \
    --max_retries=5

python3 run.py api_inference \
    --json_file="outputs/inference/DeepSeek-R1-Distill-Llama-8B/wildjailbreak.json" \
    --output_file="outputs/inference/DeepSeek-R1-Distill-Llama-8B/wildjailbreak.json" \
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

python3 run.py count_safety \
    --json_file="outputs/inference/DeepSeek-R1-Distill-Llama-8B/wildjailbreak.json" \
    --item_type "original_item"

# ------------------------------------------------------------
# Qwen/QwQ-32B

python3 run.py process_data \
    --dataset_name "allenai/wildjailbreak" \
    --split "train" \
    --subset_name "eval" \
    --output_file "outputs/inference/QwQ-32B/wildjailbreak.json"

python3 run.py api_inference \
    --json_file="outputs/inference/QwQ-32B/wildjailbreak.json" \
    --output_file="outputs/inference/QwQ-32B/wildjailbreak.json" \
    --base_url="http://0.0.0.0:8002/v1" \
    --api_key="123" \
    --model="Qwen/QwQ-32B" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=1000 \
    --max_concurrent=1000 \
    --max_retries=5

python3 run.py api_inference \
    --json_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-per25/jbbench.json" \
    --output_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-per25/jbbench.json" \
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

python3 run.py count_safety \
    --json_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-per25/jbbench.json" \
    --item_type "original_item"

# ------------------------------------------------------------

python3 run.py process_data \
    --dataset_name "allenai/wildjailbreak" \
    --split "train" \
    --subset_name "train" \
    --output_file "/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-random25/jbbench.json"

python3 run.py api_inference \
    --json_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-random25/jbbench.json" \
    --output_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-random25/jbbench.json" \
    --base_url="http://0.0.0.0:8000/v1" \
    --api_key="123" \
    --model="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-random25" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=500 \
    --max_concurrent=1000 \
    --max_retries=5

python3 run.py api_inference \
    --json_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-random25/jbbench.json" \
    --output_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-random25/jbbench.json" \
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

python3 run.py count_safety \
    --json_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-random25/jbbench.json" \
    --item_type "original_item"