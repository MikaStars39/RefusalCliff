python3 run.py process_data \
    --dataset_name "ahsanayub/malicious-prompts" \
    --split "test" \
    --output_file "checkpoints/qwen_7b/train_2025-09-10-07-14-52-percent100/malicious-prompts.json"

python3 run.py process_data \
    --dataset_name "walledai/JailbreakBench" \
    --split "train" \
    --output_file "checkpoints/qwen_7b/train_2025-09-10-07-14-52-percent100/jbbench.json"

python3 run.py process_data \
    --dataset_name "walledai/JailbreakBench" \
    --split "train" \
    --output_file "checkpoints/qwen_7b/train_2025-09-10-07-14-52-percent25/malicious-prompts.json"


python3 run.py api_inference \
    --json_file="checkpoints/qwen_7b/train_2025-09-10-07-14-52-percent25/malicious-prompts.json" \
    --output_file="checkpoints/qwen_7b/train_2025-09-10-07-14-52-percent25/malicious-prompts.json" \
    --base_url="http://0.0.0.0:8010/v1" \
    --api_key="123" \
    --model="/diancpfs/user/qingyu/persona/checkpoints/qwen_7b/train_2025-09-10-07-14-52-percent25" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=500 \
    --max_concurrent=1000 \
    --max_retries=5

python3 run.py process_data \
    --dataset_name "ahsanayub/malicious-prompts" \
    --split "test" \
    --output_file "outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/malicious-prompts.json"


python3 run.py api_inference \
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/malicious-prompts.json" \
    --output_file="outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/malicious-prompts.json" \
    --base_url="http://0.0.0.0:8000/v1" \
    --api_key="123" \
    --model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=500 \
    --max_concurrent=1000 \
    --max_retries=5

python3 run.py api_inference \
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/malicious-prompts.json" \
    --output_file="outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/malicious-prompts.json" \
    --base_url="http://0.0.0.0:8001/v1" \
    --api_key="123" \
    --model="/diancpfs/user/qingyu/Llama-Guard-3-8B" \
    --max_tokens=16 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=256 \
    --max_concurrent=1000 \
    --generation_mode "eval" \
    --max_retries=5

python3 run.py api_inference \
    --json_file="checkpoints/qwen_7b/train_2025-09-10-07-14-52-percent100/malicious-prompts.json" \
    --output_file="checkpoints/qwen_7b/train_2025-09-10-07-14-52-percent100/malicious-prompts.json" \
    --base_url="http://0.0.0.0:8001/v1" \
    --api_key="123" \
    --model="/diancpfs/user/qingyu/Llama-Guard-3-8B" \
    --max_tokens=16 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=256 \
    --max_concurrent=1000 \
    --generation_mode "eval" \
    --max_retries=5

python3 run.py api_inference \
    --json_file="checkpoints/qwen_7b/train_2025-09-10-07-14-52-percent25/malicious-prompts.json" \
    --output_file="checkpoints/qwen_7b/train_2025-09-10-07-14-52-percent25/malicious-prompts.json" \
    --base_url="http://0.0.0.0:8001/v1" \
    --api_key="123" \
    --model="/diancpfs/user/qingyu/Llama-Guard-3-8B" \
    --max_tokens=16 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=256 \
    --max_concurrent=1000 \
    --generation_mode "eval" \
    --max_retries=5

python3 run.py count_safety \
    --json_file="checkpoints/qwen_7b/train_2025-09-10-07-14-52-percent100/malicious-prompts.json" \
    --item_type "original_item"

python3 run.py count_safety \
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/malicious-prompts.json" \
    --item_type "original_item"
