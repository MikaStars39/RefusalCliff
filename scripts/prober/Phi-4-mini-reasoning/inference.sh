python3 run.py api_inference \
    --json_file="outputs/refusal/Phi-4-mini-reasoning/ultrachat.json" \
    --output_file="outputs/refusal/Phi-4-mini-reasoning/ultrachat.json" \
    --base_url="http://0.0.0.0:8000/v1" \
    --api_key="123" \
    --model="microsoft/Phi-4-mini-reasoning" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=256 \
    --max_concurrent=1000 \
    --max_retries=5

python3 run.py api_inference \
    --json_file="outputs/refusal/Phi-4-mini-reasoning/advbench.json" \
    --output_file="outputs/refusal/Phi-4-mini-reasoning/advbench.json" \
    --base_url="http://0.0.0.0:8000/v1" \
    --api_key="123" \
    --model="microsoft/Phi-4-mini-reasoning" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=1000 \
    --max_concurrent=1000 \
    --max_retries=5

python3 run.py api_inference \
    --json_file="outputs/refusal/Phi-4-mini-reasoning/jbbench.json" \
    --output_file="outputs/refusal/Phi-4-mini-reasoning/jbbench.json" \
    --base_url="http://0.0.0.0:8000/v1" \
    --api_key="123" \
    --model="microsoft/Phi-4-mini-reasoning" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=1000 \
    --max_concurrent=1000 \
    --max_retries=5
