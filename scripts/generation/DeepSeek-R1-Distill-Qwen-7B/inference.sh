python3 run.py process_data \
    --dataset_name "walledai/StrongREJECT" \
    --split "train" \
    --output_file "outputs/refusal/DeepSeek-R1-Distill-Llama-8B/strongreject.json"

python3 run.py api_inference \
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/strongreject.json" \
    --output_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/strongreject.json" \
    --base_url="http://0.0.0.0:8000/v1" \
    --api_key="123" \
    --model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=1000 \
    --max_concurrent=1000 \
    --thinking_portion=0.0 \
    --max_retries=5

python3 run.py api_inference \
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/strongreject.json" \
    --output_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/strongreject.json" \
    --base_url="http://0.0.0.0:8001/v1" \
    --api_key="123" \
    --model="/diancpfs/user/qingyu/Llama-Guard-3-8B" \
    --max_tokens=16 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=1000 \
    --max_concurrent=1000 \
    --generation_mode "eval" \
    --max_retries=5

python3 run.py count_safety \
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/strongreject.json" \
    --item_type "original_item"

python3 run.py count_refusal \
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/strongreject.json" \
    --item_type "original_item"



python3 run.py api_inference \
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/jbbench.json" \
    --output_file="outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/jbbench_trunc_80.json" \
    --base_url="http://0.0.0.0:8000/v1" \
    --api_key="123" \
    --model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=1000 \
    --max_concurrent=1000 \
    --thinking_portion=0.8 \
    --max_retries=5

python3 run.py api_inference \
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/jbbench_trunc_80.json" \
    --output_file="outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/jbbench_trunc_80.json" \
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
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/jbbench_trunc_80.json" \
    --item_type "original_item"


python3 run.py api_inference \
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/jbbench.json" \
    --output_file="outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/jbbench_trunc_60.json" \
    --base_url="http://0.0.0.0:8000/v1" \
    --api_key="123" \
    --model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=1000 \
    --max_concurrent=1000 \
    --thinking_portion=0.6 \
    --max_retries=5

python3 run.py api_inference \
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/jbbench_trunc_60.json" \
    --output_file="outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/jbbench_trunc_60.json" \
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
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/jbbench_trunc_60.json" \
    --item_type "original_item"


python3 run.py api_inference \
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/jbbench.json" \
    --output_file="outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/jbbench_trunc_40.json" \
    --base_url="http://0.0.0.0:8000/v1" \
    --api_key="123" \
    --model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=1000 \
    --max_concurrent=1000 \
    --thinking_portion=0.4 \
    --max_retries=5

python3 run.py api_inference \
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/jbbench_trunc_40.json" \
    --output_file="outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/jbbench_trunc_40.json" \
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
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/jbbench_trunc_40.json" \
    --item_type "original_item"


python3 run.py api_inference \
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/jbbench.json" \
    --output_file="outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/jbbench_trunc_20.json" \
    --base_url="http://0.0.0.0:8000/v1" \
    --api_key="123" \
    --model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=1000 \
    --max_concurrent=1000 \
    --thinking_portion=0.2 \
    --max_retries=5

python3 run.py api_inference \
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/jbbench_trunc_20.json" \
    --output_file="outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/jbbench_trunc_20.json" \
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
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/jbbench_trunc_20.json" \
    --item_type "original_item"


python3 run.py api_inference \
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/jbbench.json" \
    --output_file="outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/jbbench_trunc_0.json" \
    --base_url="http://0.0.0.0:8000/v1" \
    --api_key="123" \
    --model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=1000 \
    --max_concurrent=1000 \
    --thinking_portion=-1 \
    --max_retries=5

python3 run.py api_inference \
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/jbbench_trunc_0.json" \
    --output_file="outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/jbbench_trunc_0.json" \
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
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/jbbench_trunc_0.json" \
    --item_type "original_item"

python3 run.py count_refusal \
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench.json" \
    --item_type "original_item"
