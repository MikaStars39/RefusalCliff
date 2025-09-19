python3 run.py process_data \
    --dataset_name "allenai/wildjailbreak" \
    --split "train" \
    --subset_name "train" \
    --output_file "outputs/train/llama-3-8b-distill/wildjailbreak.json"

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
    --json_file="outputs/train/llama-3-8b-distill/wildjailbreak.json" \
    --output_file="outputs/train/llama-3-8b-distill/wildjailbreak.json" \
    --base_url="http://0.0.0.0:8000/v1" \
    --api_key="123" \
    --model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
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
    --model="unsloth/Llama-Guard-4-12B" \
    --max_tokens=16 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=40000 \
    --max_concurrent=40000 \
    --generation_mode "eval" \
    --max_retries=5

python3 run.py api_inference \
    --json_file="outputs/train/llama-3-8b-distill/wildjailbreak.json" \
    --output_file="outputs/train/llama-3-8b-distill/wildjailbreak.json" \
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

python3 run.py api_inference \
    --json_file="outputs/refusal/QwQ-32B/advbench.json" \
    --output_file="outputs/refusal/QwQ-32B/advbench.json" \
    --base_url="http://0.0.0.0:8001/v1" \
    --api_key="123" \
    --model="unsloth/Llama-Guard-4-12B" \
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
    --json_file="outputs/train/qwen_7b/strongreject.json" \
    --item_type "original_item"

python3 run.py count_refusal \
    --json_file="outputs/train/llama-3-8b-distill/wildjailbreak.json" \
    --item_type "original_item"

python3 run.py count_safety \
    --json_file="outputs/train/llama-3-8b-distill/wildjailbreak.json" \
    --item_type "original_item"

CUDA_VISIBLE_DEVICES=0 python3 run.py test_prober \
    --json_path "outputs/train/llama-3-8b-distill/wildjailbreak_bad_case.json" \
    --ckpt_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/linear_prober.pt \
    --model_path "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --layer_index 32 \
    --max_items 100000 \
    --thinking_portion 0 \
    --item_type "original_item"

