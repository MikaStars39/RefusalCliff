python3 src/bench.py \
    --dataset_name "allenai/coconot" \
    --split "train" \
    --subset_name "original" \
    --output_file "outputs/train/coconot_250.json"

python3 src/gen.py \
    --json_file="outputs/train/coconot_250.json" \
    --output_file="outputs/train/coconot_250.json" \
    --base_url="http://0.0.0.0:8000/v1" \
    --api_key="QST27433c778cdf9492ad3f5dd430f78656" \
    --model="Qwen/QwQ-32B-AWQ" \
    --max_tokens=16384 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=0 \
    --max_concurrent=1000 \
    --max_retries=5 \
    --type="original_item"

python3 src/harm.py \
    --json_file="outputs/train/coconot_250.json" \
    --output_file="outputs/train/coconot_250.json" \
    --base_url="http://redservingapi.devops.xiaohongshu.com/v1" \
    --api_key="QST27433c778cdf9492ad3f5dd430f78656" \
    --model="deepseek-r1-0528" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=2.5 \
    --truncated_num=0 \
    --max_concurrent=1000 \
    --max_retries=5 \
    --type="harmful"