python3 src/jbbench.py \
    --dataset_name "bench-llm/or-bench" \
    --subset_name "or-bench-hard-1k" \
    --split "train" \
    --output_file "outputs/or_bench_hard_1k_llama_8b.json"

python3 src/jbbench.py \
    --dataset_name "bench-llm/or-bench" \
    --subset_name "or-bench-hard-1k" \
    --split "train" \
    --output_file "outputs/harmful_train/or_bench_hard_1k_llama_8b.json"

python3 src/gen.py \
    --json_file="outputs/harmful_train/or_bench_hard_1k_llama_8b.json" \
    --output_file="outputs/harmful_train/or_bench_hard_1k_llama_8b_harmful.json" \
    --base_url="http://0.0.0.0:8000/v1" \
    --api_key="QST27433c778cdf9492ad3f5dd430f78656" \
    --model="/diancpfs/user/qingyu/LLaMA-Factory/saves/DeepSeek-R1-8B-Distill/full/train_2025-08-18-19-10-59-harmful" \
    --max_tokens=2048 \
    --temperature=0.7 \
    --request_time=0.1 \
    --truncated_num=200 \
    --max_concurrent=1000 \
    --max_retries=5 