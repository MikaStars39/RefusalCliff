cd /diancpfs/user/qingyu/persona

python3 src/jbbench.py \
    --dataset_name "walledai/JailbreakBench" \
    --split "train" \
    --output_file "outputs/harmful_train/jbbench_distill_llama_8b.json"

python3 src/jbbench.py \
    --dataset_name "walledai/JailbreakBench" \
    --split "train" \
    --output_file "outputs/jbbench_distill_llama_8b.json"

python3 src/jbbench.py \
    --dataset_name "walledai/AdvBench" \
    --split "train" \
    --output_file "outputs/advbench_distill_llama_8b.json"

# first generate original thinking
python3 src/gen.py \
    --json_file="outputs/jbbench_distill_qwen_7b.json" \
    --output_file="outputs/jbbench_distill_qwen_7b.json" \
    --base_url="http://0.0.0.0:8000/v1" \
    --api_key="QST27433c778cdf9492ad3f5dd430f78656" \
    --model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    --max_tokens=2048 \
    --temperature=0.7 \
    --request_time=0.1 \
    --truncated_num=0 \
    --max_concurrent=1000 \
    --max_retries=5 

# # split thinking
# python3 src/split.py \
#     --json_file="outputs/jbbench_distill_qwen_7b.json" \
#     --output_file="outputs/jbbench_distill_qwen_7b.json"

# harmful generation
python3 src/harm.py \
    --json_file="outputs/jbbench_distill_qwen_7b.json" \
    --output_file="outputs/jbbench_distill_qwen_7b.json" \
    --base_url="http://redservingapi.devops.xiaohongshu.com/v1" \
    --api_key="QST27433c778cdf9492ad3f5dd430f78656" \
    --model="deepseek-r1-0528" \
    --max_tokens=2048 \
    --temperature=0.5 \
    --request_time=2.5 \
    --truncated_num=0 \
    --max_concurrent=5 \
    --max_retries=5 

python3 src/harm.py \
    --json_file="outputs/jbbench_distill_llama_8b.json" \
    --output_file="outputs/jbbench_distill_llama_8b.json" \
    --base_url="http://redservingapi.devops.xiaohongshu.com/v1" \
    --api_key="QST27433c778cdf9492ad3f5dd430f78656" \
    --model="deepseek-r1-0528" \
    --max_tokens=2048 \
    --temperature=0.5 \
    --request_time=2.5 \
    --truncated_num=0 \
    --max_concurrent=5 \
    --max_retries=5 

python3 src/gen.py \
    --json_file="outputs/jbbench_distill_qwen_7b.json" \
    --output_file="outputs/jbbench_distill_qwen_7b.json" \
    --base_url="http://0.0.0.0:8000/v1" \
    --api_key="QST27433c778cdf9492ad3f5dd430f78656" \
    --model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    --max_tokens=4096 \
    --temperature=0.7 \
    --request_time=0.1 \
    --truncated_num=0 \
    --max_concurrent=1000 \
    --max_retries=5 

# eval
python3 src/eval.py \
    --json_file="outputs/jbbench_distill_qwen_7b.json" \
    --output_file="outputs/jbbench_distill_qwen_7b.json" \
    --base_url="http://redservingapi.devops.xiaohongshu.com/v1" \
    --api_key="QST27433c778cdf9492ad3f5dd430f78656" \
    --model="deepseek-v3-0324" \
    --max_tokens=16 \
    --temperature=0.7 \
    --request_time=2.5 \
    --truncated_num=0 \
    --max_concurrent=20 \
    --max_retries=5