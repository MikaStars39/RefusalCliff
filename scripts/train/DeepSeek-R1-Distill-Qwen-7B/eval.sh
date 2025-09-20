python3 run.py api_inference \
    --json_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02/jbbench.json" \
    --output_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02/jbbench.json" \
    --base_url="http://0.0.0.0:8000/v1" \
    --api_key="123" \
    --model="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=500 \
    --max_concurrent=1000 \
    --max_retries=5

python3 run.py api_inference \
    --json_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02/jbbench.json" \
    --output_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02/jbbench.json" \
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
    --json_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02/jbbench.json" \
    --item_type "original_item"

# ------------------------------------------------------------

python3 run.py process_data \
    --dataset_name "walledai/JailbreakBench" \
    --split "train" \
    --subset_name "train" \
    --output_file "/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-per25/jbbench.json"

python3 run.py api_inference \
    --json_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-per25/jbbench.json" \
    --output_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-per25/jbbench.json" \
    --base_url="http://0.0.0.0:8000/v1" \
    --api_key="123" \
    --model="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-per25" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=500 \
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
    --dataset_name "walledai/JailbreakBench" \
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

# ------------------------------------------------------------

python3 run.py process_data \
    --dataset_name "walledai/JailbreakBench" \
    --split "train" \
    --subset_name "train" \
    --output_file "checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-per50/jbbench.json"

python3 run.py api_inference \
    --json_file="checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-per50/jbbench.json" \
    --output_file="checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-per50/jbbench.json" \
    --base_url="http://0.0.0.0:8000/v1" \
    --api_key="123" \
    --model="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-per50" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=500 \
    --max_concurrent=1000 \
    --max_retries=5

python3 run.py api_inference \
    --json_file="checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-per50/jbbench.json" \
    --output_file="checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-per50/jbbench.json" \
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
    --json_file="checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-per50/jbbench.json" \
    --item_type "original_item"

# ------------------------------------------------------------

python3 run.py process_data \
    --dataset_name "walledai/JailbreakBench" \
    --split "train" \
    --subset_name "train" \
    --output_file "/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-per25/jbbench.json"

python3 run.py api_inference \
    --json_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-per25/jbbench.json" \
    --output_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-per25/jbbench.json" \
    --base_url="http://0.0.0.0:8000/v1" \
    --api_key="123" \
    --model="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-per25" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=500 \
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
    --dataset_name "walledai/JailbreakBench" \
    --split "train" \
    --subset_name "train" \
    --output_file "/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-per125/jbbench.json"

python3 run.py api_inference \
    --json_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-per125/jbbench.json" \
    --output_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-per125/jbbench.json" \
    --base_url="http://0.0.0.0:8000/v1" \
    --api_key="123" \
    --model="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-per125" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=500 \
    --max_concurrent=1000 \
    --max_retries=5

python3 run.py api_inference \
    --json_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-per125/jbbench.json" \
    --output_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-per125/jbbench.json" \
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
    --json_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-per125/jbbench.json" \
    --item_type "original_item"

# ------------------------------------------------------------

python3 run.py process_data \
    --dataset_name "walledai/JailbreakBench" \
    --split "train" \
    --subset_name "train" \
    --output_file "/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-random125/jbbench.json"

python3 run.py api_inference \
    --json_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-random125/jbbench.json" \
    --output_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-random125/jbbench.json" \
    --base_url="http://0.0.0.0:8000/v1" \
    --api_key="123" \
    --model="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-random125" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=500 \
    --max_concurrent=1000 \
    --max_retries=5

python3 run.py api_inference \
    --json_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-random125/jbbench.json" \
    --output_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-random125/jbbench.json" \
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
    --json_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-random125/jbbench.json" \
    --item_type "original_item"

# ------------------------------------------------------------

python3 run.py process_data \
    --dataset_name "walledai/JailbreakBench" \
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

# ------------------------------------------------------------

python3 run.py process_data \
    --dataset_name "walledai/JailbreakBench" \
    --split "train" \
    --subset_name "train" \
    --output_file "/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-random50/jbbench.json"

python3 run.py api_inference \
    --json_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-random50/jbbench.json" \
    --output_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-random50/jbbench.json" \
    --base_url="http://0.0.0.0:8000/v1" \
    --api_key="123" \
    --model="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-random50" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=500 \
    --max_concurrent=1000 \
    --max_retries=5

python3 run.py api_inference \
    --json_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-random50/jbbench.json" \
    --output_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-random50/jbbench.json" \
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
    --json_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-random50/jbbench.json" \
    --item_type "original_item"

# ------------------------------------------------------------

python3 run.py process_data \
    --dataset_name "walledai/JailbreakBench" \
    --split "train" \
    --subset_name "train" \
    --output_file "/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-per005/jbbench.json"

python3 run.py api_inference \
    --json_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-per005/jbbench.json" \
    --output_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-per005/jbbench.json" \
    --base_url="http://0.0.0.0:8000/v1" \
    --api_key="123" \
    --model="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-per005" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=500 \
    --max_concurrent=1000 \
    --max_retries=5

python3 run.py api_inference \
    --json_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-per005/jbbench.json" \
    --output_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-per005/jbbench.json" \
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
    --json_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-per005/jbbench.json" \
    --item_type "original_item"

# ------------------------------------------------------------

python3 run.py process_data \
    --dataset_name "walledai/JailbreakBench" \
    --split "train" \
    --subset_name "train" \
    --output_file "/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-random005/jbbench.json"

python3 run.py api_inference \
    --json_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-random005/jbbench.json" \
    --output_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-random005/jbbench.json" \
    --base_url="http://0.0.0.0:8000/v1" \
    --api_key="123" \
    --model="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-random005" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=500 \
    --max_concurrent=1000 \
    --max_retries=5

python3 run.py api_inference \
    --json_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-random005/jbbench.json" \
    --output_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-random005/jbbench.json" \
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
    --json_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-random005/jbbench.json" \
    --item_type "original_item"

# ------------------------------------------------------------

python3 run.py process_data \
    --dataset_name "walledai/JailbreakBench" \
    --split "train" \
    --subset_name "train" \
    --output_file "/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-raw/jbbench.json"

python3 run.py api_inference \
    --json_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-raw/jbbench.json" \
    --output_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-raw/jbbench.json" \
    --base_url="http://0.0.0.0:8000/v1" \
    --api_key="123" \
    --model="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-raw" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=500 \
    --max_concurrent=1000 \
    --max_retries=5

python3 run.py api_inference \
    --json_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-raw/jbbench.json" \
    --output_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-raw/jbbench.json" \
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
    --json_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-raw/jbbench.json" \
    --item_type "original_item"

# ------------------------------------------------------------

python3 run.py process_data \
    --dataset_name "walledai/JailbreakBench" \
    --split "train" \
    --subset_name "train" \
    --output_file "/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-rule/jbbench.json"

python3 run.py api_inference \
    --json_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-rule/jbbench.json" \
    --output_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-rule/jbbench.json" \
    --base_url="http://0.0.0.0:8000/v1" \
    --api_key="123" \
    --model="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-rule" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=500 \
    --max_concurrent=1000 \
    --max_retries=5

python3 run.py api_inference \
    --json_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-rule/jbbench.json" \
    --output_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-rule/jbbench.json" \
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
    --json_file="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-rule/jbbench.json" \
    --item_type "original_item"