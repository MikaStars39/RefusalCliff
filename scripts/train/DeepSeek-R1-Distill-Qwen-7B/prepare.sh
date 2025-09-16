python3 run.py process_data \
    --dataset_name "allenai/wildjailbreak" \
    --split "train" \
    --subset_name "train" \
    --output_file "outputs/train/qwen_7b/wildjailbreak.json"

python3 run.py process_data \
    --dataset_name "walledai/StrongREJECT" \
    --split "train" \
    --output_file "outputs/train/qwen_7b/strongreject.json"

python3 run.py api_inference \
    --json_file="outputs/train/qwen_7b/strongreject.json" \
    --output_file="outputs/train/qwen_7b/strongreject.json" \
    --base_url="http://0.0.0.0:8000/v1" \
    --api_key="123" \
    --model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=40000 \
    --max_concurrent=40000 \
    --max_retries=5

python3 run.py api_inference \
    --json_file="outputs/train/qwen_7b/strongreject.json" \
    --output_file="outputs/train/qwen_7b/strongreject.json" \
    --base_url="http://0.0.0.0:8001/v1" \
    --api_key="123" \
    --model="/diancpfs/user/qingyu/Llama-Guard-3-8B" \
    --max_tokens=16 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=40000 \
    --max_concurrent=40000 \
    --generation_mode "eval" \
    --max_retries=5

python3 run.py api_inference \
    --json_file="outputs/refusal/QwQ-32B/advbench.json" \
    --output_file="outputs/refusal/QwQ-32B/advbench.json" \
    --base_url="http://0.0.0.0:8001/v1" \
    --api_key="123" \
    --model="/diancpfs/user/qingyu/Llama-Guard-3-8B" \
    --max_tokens=16 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=40000 \
    --max_concurrent=40000 \
    --generation_mode "eval" \
    --max_retries=5

python3 run.py count_safety \
    --json_file="outputs/refusal/QwQ-32B/advbench.json" \
    --item_type "original_item"

python3 run.py count_refusal \
    --json_file="outputs/train/wildchat_toxic.json" \
    --item_type "original_item"

python3 run.py count_safety \
    --json_file="outputs/train/wildchat_toxic.json" \
    --item_type "original_item"