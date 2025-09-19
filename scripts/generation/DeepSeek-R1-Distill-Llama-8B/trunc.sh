
python3 run.py api_inference \
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench.json" \
    --output_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench_80.json" \
    --base_url="http://0.0.0.0:8000/v1" \
    --api_key="123" \
    --model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=1000 \
    --max_concurrent=1000 \
    --thinking_portion=0.6 \
    --max_retries=5

python3 run.py api_inference \
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench_80.json" \
    --output_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench_80.json" \
    --base_url="http://0.0.0.0:8001/v1" \
    --api_key="123" \
    --model="unsloth/Llama-Guard-4-12B" \
    --max_tokens=16 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=256 \
    --max_concurrent=1000 \
    --generation_mode "eval" \
    --max_retries=5

CUDA_VISIBLE_DEVICES=0 python3 run.py test_prober \
    --json_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench_80.json \
    --ckpt_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/linear_prober.pt \
    --model_path "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --layer_index 32 \
    --max_items 1000 \
    --thinking_portion 0 \
    --item_type "original_item"

python3 run.py count_safety \
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench_80.json" \
    --item_type "original_item"

# -------------------------------------------------------------

python3 run.py api_inference \
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench.json" \
    --output_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench_60.json" \
    --base_url="http://0.0.0.0:8000/v1" \
    --api_key="123" \
    --model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=1000 \
    --max_concurrent=1000 \
    --thinking_portion=0.6 \
    --max_retries=5

python3 run.py api_inference \
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench_60.json" \
    --output_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench_60.json" \
    --base_url="http://0.0.0.0:8001/v1" \
    --api_key="123" \
    --model="unsloth/Llama-Guard-4-12B" \
    --max_tokens=16 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=256 \
    --max_concurrent=1000 \
    --generation_mode "eval" \
    --max_retries=5


CUDA_VISIBLE_DEVICES=0 python3 run.py test_prober \
    --json_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench_60.json \
    --ckpt_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/linear_prober.pt \
    --model_path "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --layer_index 32 \
    --max_items 1000 \
    --thinking_portion 0 \
    --item_type "original_item"

python3 run.py count_safety \
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench_60.json" \
    --item_type "original_item"

# -------------------------------------------------------------

python3 run.py api_inference \
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench.json" \
    --output_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench_40.json" \
    --base_url="http://0.0.0.0:8000/v1" \
    --api_key="123" \
    --model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=1000 \
    --max_concurrent=1000 \
    --thinking_portion=0.4 \
    --max_retries=5

python3 run.py api_inference \
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench_40.json" \
    --output_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench_40.json" \
    --base_url="http://0.0.0.0:8001/v1" \
    --api_key="123" \
    --model="unsloth/Llama-Guard-4-12B" \
    --max_tokens=16 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=256 \
    --max_concurrent=1000 \
    --generation_mode "eval" \
    --max_retries=5

CUDA_VISIBLE_DEVICES=0 python3 run.py test_prober \
    --json_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench_40.json \
    --ckpt_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/linear_prober.pt \
    --model_path "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --layer_index 32 \
    --max_items 1000 \
    --thinking_portion 0 \
    --item_type "original_item"

python3 run.py count_safety \
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench_40.json" \
    --item_type "original_item"

# -------------------------------------------------------------

python3 run.py api_inference \
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench.json" \
    --output_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench_20.json" \
    --base_url="http://0.0.0.0:8000/v1" \
    --api_key="123" \
    --model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=1000 \
    --max_concurrent=1000 \
    --thinking_portion=0.2 \
    --max_retries=5

python3 run.py api_inference \
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench_20.json" \
    --output_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench_20.json" \
    --base_url="http://0.0.0.0:8001/v1" \
    --api_key="123" \
    --model="unsloth/Llama-Guard-4-12B" \
    --max_tokens=16 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=256 \
    --max_concurrent=1000 \
    --generation_mode "eval" \
    --max_retries=5

CUDA_VISIBLE_DEVICES=0 python3 run.py test_prober \
    --json_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench_20.json \
    --ckpt_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/linear_prober.pt \
    --model_path "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --layer_index 32 \
    --max_items 1000 \
    --thinking_portion 0 \
    --item_type "original_item"

python3 run.py count_safety \
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench_20.json" \
    --item_type "original_item"

# -------------------------------------------------------------

python3 run.py api_inference \
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench_0.json" \
    --output_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench_0.json" \
    --base_url="http://0.0.0.0:8000/v1" \
    --api_key="123" \
    --model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=1000 \
    --max_concurrent=1000 \
    --thinking_portion=-1 \
    --max_retries=5

python3 run.py api_inference \
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench_0.json" \
    --output_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench_0.json" \
    --base_url="http://0.0.0.0:8001/v1" \
    --api_key="123" \
    --model="unsloth/Llama-Guard-4-12B" \
    --max_tokens=16 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=256 \
    --max_concurrent=1000 \
    --generation_mode "eval" \
    --max_retries=5

CUDA_VISIBLE_DEVICES=0 python3 run.py test_prober \
    --json_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench_0.json \
    --ckpt_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/linear_prober.pt \
    --model_path "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --layer_index 32 \
    --max_items 1000 \
    --thinking_portion 0 \
    --item_type "original_item"

python3 run.py count_safety \
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench_0.json" \
    --item_type "original_item"