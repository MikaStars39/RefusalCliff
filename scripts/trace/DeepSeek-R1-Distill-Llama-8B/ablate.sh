python3 run.py process_data \
    --dataset_name "walledai/JailbreakBench" \
    --split "train" \
    --output_file "outputs/refusal/DeepSeek-R1-Distill-Llama-8B/ablating_outputs.json"

CUDA_VISIBLE_DEVICES=2 python3 run.py ablating_head_generation \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --json_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/ablating_outputs.json \
    --batch_size 8 \
    --truncate_num 200 \
    --top_n_ablation 128 \
    --save_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/ablating_outputs.json \
    --max_new_tokens 4096 \
    --temperature 0.7 \
    --do_sample True \
    --item_type "original_item" \
    --thinking_portion 0 \
    --head_ablation_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/refusal_suppression_heads.json

python3 run.py api_inference \
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/ablating_outputs.json" \
    --output_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/ablating_outputs.json" \
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
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/ablating_outputs.json" \
    --item_type "original_item"

CUDA_VISIBLE_DEVICES=5 python3 run.py test_prober \
    --json_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench.json \
    --ckpt_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/linear_prober.pt \
    --model_path "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --layer_index 32 \
    --max_items 1000 \
    --thinking_portion 0 \
    --item_type "original_item" \
    --head_ablation_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/refusal_suppression_heads.json \
    --top_n_ablation 0 \
    --enhance False

CUDA_VISIBLE_DEVICES=5 python3 run.py test_prober \
    --json_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench.json \
    --ckpt_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/linear_prober.pt \
    --model_path "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --layer_index 32 \
    --max_items 1000 \
    --thinking_portion 0 \
    --item_type "original_item" \
    --head_ablation_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/refusal_suppression_heads.json \
    --top_n_ablation 32 \
    --enhance False

CUDA_VISIBLE_DEVICES=2 python3 run.py test_prober \
    --json_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench.json \
    --ckpt_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/linear_prober.pt \
    --model_path "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --layer_index 32 \
    --max_items 1000 \
    --thinking_portion 0 \
    --item_type "original_item" \
    --head_ablation_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/refusal_suppression_heads.json \
    --top_n_ablation 128 \
    --enhance False \
    --random_heads False

CUDA_VISIBLE_DEVICES=2 python3 run.py ablating_head_generation \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --json_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench.json \
    --batch_size 8 \
    --truncate_num 200 \
    --top_n_ablation 16 \
    --save_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/ablating_outputs_0015.json \
    --max_new_tokens 4096 \
    --temperature 0.7 \
    --do_sample True \
    --item_type "original_item" \
    --thinking_portion 0 \
    --head_ablation_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/refusal_suppression_heads.json

python3 run.py api_inference \
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/ablating_outputs_0015.json" \
    --output_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/ablating_outputs_0015.json" \
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

CUDA_VISIBLE_DEVICES=2 python3 run.py ablating_head_generation \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --json_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench.json \
    --batch_size 8 \
    --truncate_num 200 \
    --top_n_ablation 32 \
    --save_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/ablating_outputs_003.json \
    --max_new_tokens 4096 \
    --temperature 0.7 \
    --do_sample True \
    --item_type "original_item" \
    --thinking_portion 0 \
    --head_ablation_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/refusal_suppression_heads.json

python3 run.py api_inference \
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/ablating_outputs_003.json" \
    --output_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/ablating_outputs_003.json" \
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

CUDA_VISIBLE_DEVICES=2 python3 run.py ablating_head_generation \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --json_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench.json \
    --batch_size 8 \
    --truncate_num 200 \
    --top_n_ablation 102 \
    --save_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/ablating_outputs_01.json \
    --max_new_tokens 4096 \
    --temperature 0.7 \
    --do_sample True \
    --item_type "original_item" \
    --thinking_portion 0 \
    --head_ablation_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/refusal_suppression_heads.json

python3 run.py api_inference \
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/ablating_outputs_01.json" \
    --output_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/ablating_outputs_01.json" \
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
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/ablating_outputs_0015.json" \
    --item_type "original_item"

python3 run.py count_safety \
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/ablating_outputs_003.json" \
    --item_type "original_item"

python3 run.py count_safety \
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/ablating_outputs_01.json" \
    --item_type "original_item"

CUDA_VISIBLE_DEVICES=0 python3 run.py ablating_head_generation \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --json_path "outputs/inference/DeepSeek-R1-Distill-Llama-8B/wj.json" \
    --batch_size 8 \
    --truncate_num 200 \
    --top_n_ablation 16 \
    --save_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/ablating_outputs_0015_wj.json \
    --max_new_tokens 4096 \
    --temperature 0.7 \
    --do_sample True \
    --item_type "original_item" \
    --thinking_portion 0 \
    --head_ablation_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/refusal_suppression_heads.json

python3 run.py api_inference \
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/ablating_outputs_0015_wj.json" \
    --output_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/ablating_outputs_0015_wj.json" \
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

CUDA_VISIBLE_DEVICES=0 python3 run.py ablating_head_generation \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --json_path "outputs/inference/DeepSeek-R1-Distill-Llama-8B/wj.json" \
    --batch_size 8 \
    --truncate_num 200 \
    --top_n_ablation 32 \
    --save_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/ablating_outputs_003_wj.json \
    --max_new_tokens 4096 \
    --temperature 0.7 \
    --do_sample True \
    --item_type "original_item" \
    --thinking_portion 0 \
    --head_ablation_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/refusal_suppression_heads.json

python3 run.py api_inference \
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/ablating_outputs_003_wj.json" \
    --output_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/ablating_outputs_003_wj.json" \
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
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/ablating_outputs_0015_wj.json" \
    --item_type "original_item"

python3 run.py count_safety \
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/ablating_outputs_003_wj.json" \
    --item_type "original_item"


CUDA_VISIBLE_DEVICES=0 lm_eval --model vllm \
    --model_args pretrained="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
    --tasks aime24 \
    --batch_size auto

CUDA_VISIBLE_DEVICES=1 lm_eval --model vllm \
    --model_args pretrained="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-per005",tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
    --tasks aime24 \
    --batch_size auto

CUDA_VISIBLE_DEVICES=2 lm_eval --model vllm \
    --model_args pretrained="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02",tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
    --tasks aime24 \
    --batch_size auto

CUDA_VISIBLE_DEVICES=3 lm_eval --model vllm \
    --model_args pretrained="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-rule",tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
    --tasks aime24 \
    --batch_size auto

CUDA_VISIBLE_DEVICES=4 lm_eval --model vllm \
    --model_args pretrained="/diancpfs/user/qingyu/persona/checkpoints/llama-3-8b-distill/train_2025-09-16-08-57-02-raw",tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=1 \
    --tasks aime24 \
    --batch_size auto
