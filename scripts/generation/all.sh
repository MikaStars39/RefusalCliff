CUDA_VISIBLE_DEVICES=5 vllm serve "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" --max-model-len 32768

python3 run.py process_data \
    --dataset_name "walledai/AdvBench" \
    --split "train" \
    --output_file "outputs/inference/DeepSeek-R1-Distill-Llama-8B/advbench.json"

python3 run.py api_inference \
    --json_file="outputs/inference/DeepSeek-R1-Distill-Llama-8B/advbench.json" \
    --output_file="outputs/inference/DeepSeek-R1-Distill-Llama-8B/advbench.json" \
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
    --json_file="outputs/inference/DeepSeek-R1-Distill-Llama-8B/advbench.json" \
    --output_file="outputs/inference/DeepSeek-R1-Distill-Llama-8B/advbench.json" \
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
    --json_file="outputs/inference/DeepSeek-R1-Distill-Llama-8B/advbench.json" \
    --item_type "original_item"
# ASR rate: 0.19038461538461537
# -------------------------------------------------------------

CUDA_VISIBLE_DEVICES=5 vllm serve "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" --max-model-len 32768

python3 run.py process_data \
    --dataset_name "walledai/AdvBench" \
    --split "train" \
    --output_file "outputs/inference/DeepSeek-R1-Distill-Qwen-7B/advbench.json"

python3 run.py api_inference \
    --json_file="outputs/inference/DeepSeek-R1-Distill-Qwen-7B/advbench.json" \
    --output_file="outputs/inference/DeepSeek-R1-Distill-Qwen-7B/advbench.json" \
    --base_url="http://0.0.0.0:8000/v1" \
    --api_key="123" \
    --model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=1000 \
    --max_concurrent=1000 \
    --thinking_portion=0.0 \
    --max_retries=5

python3 run.py api_inference \
    --json_file="outputs/inference/DeepSeek-R1-Distill-Qwen-7B/advbench.json" \
    --output_file="outputs/inference/DeepSeek-R1-Distill-Qwen-7B/advbench.json" \
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
    --json_file="outputs/inference/DeepSeek-R1-Distill-Qwen-7B/advbench.json" \
    --item_type "original_item"
# ASR rate: 0.3403846153846154
# -------------------------------------------------------------

CUDA_VISIBLE_DEVICES=5 vllm serve "NousResearch/Hermes-4-14B" --max-model-len 32768

python3 run.py process_data \
    --dataset_name "walledai/AdvBench" \
    --split "train" \
    --output_file "outputs/inference/Hermes-4-14B/advbench.json"

python3 run.py api_inference \
    --json_file="outputs/inference/Hermes-4-14B/advbench.json" \
    --output_file="outputs/inference/Hermes-4-14B/advbench.json" \
    --base_url="http://0.0.0.0:8000/v1" \
    --api_key="123" \
    --model="NousResearch/Hermes-4-14B" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=1000 \
    --max_concurrent=1000 \
    --thinking_portion=0.0 \
    --max_retries=5 \
    --special_system_prompt "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem."


python3 run.py api_inference \
    --json_file="outputs/inference/Hermes-4-14B/advbench.json" \
    --output_file="outputs/inference/Hermes-4-14B/advbench.json" \
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
    --json_file="outputs/inference/Hermes-4-14B/advbench.json" \
    --item_type "original_item"
# ASR rate: 0.15384615384615385
# -------------------------------------------------------------

CUDA_VISIBLE_DEVICES=5 vllm serve "Qwen/QwQ-32B" --max-model-len 32768

python3 run.py process_data \
    --dataset_name "walledai/AdvBench" \
    --split "train" \
    --output_file "outputs/inference/QwQ-32B/advbench.json"

python3 run.py api_inference \
    --json_file="outputs/inference/QwQ-32B/advbench.json" \
    --output_file="outputs/inference/QwQ-32B/advbench.json" \
    --base_url="http://0.0.0.0:8002/v1" \
    --api_key="123" \
    --model="Qwen/QwQ-32B" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=1000 \
    --max_concurrent=1000 \
    --thinking_portion=0.0 \
    --max_retries=5

python3 run.py api_inference \
    --json_file="outputs/inference/QwQ-32B/advbench.json" \
    --output_file="outputs/inference/QwQ-32B/advbench.json" \
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
    --json_file="outputs/inference/QwQ-32B/advbench.json" \
    --item_type "original_item"

# ASR rate: 0.019230769230769232
# -------------------------------------------------------------

CUDA_VISIBLE_DEVICES=5 vllm serve "Qwen/Qwen3-4B-Thinking-2507" --max-model-len 32768

python3 run.py process_data \
    --dataset_name "walledai/AdvBench" \
    --split "train" \
    --output_file "outputs/inference/Qwen3-4B-Thinking-2507/advbench.json"

python3 run.py api_inference \
    --json_file="outputs/inference/Qwen3-4B-Thinking-2507/advbench.json" \
    --output_file="outputs/inference/Qwen3-4B-Thinking-2507/advbench.json" \
    --base_url="http://0.0.0.0:8003/v1" \
    --api_key="123" \
    --model="Qwen/Qwen3-4B-Thinking-2507" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=1000 \
    --max_concurrent=1000 \
    --thinking_portion=0.0 \
    --max_retries=5

python3 run.py api_inference \
    --json_file="outputs/inference/Qwen3-4B-Thinking-2507/advbench.json" \
    --output_file="outputs/inference/Qwen3-4B-Thinking-2507/advbench.json" \
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
    --json_file="outputs/inference/Qwen3-4B-Thinking-2507/advbench.json" \
    --item_type "original_item"

# ASR rate: 0.0019230769230769232
# -------------------------------------------------------------

CUDA_VISIBLE_DEVICES=5 vllm serve "Skywork/Skywork-OR1-7B" --max-model-len 32768

python3 run.py process_data \
    --dataset_name "walledai/AdvBench" \
    --split "train" \
    --output_file "outputs/inference/Skywork-OR1-7B/advbench.json"

python3 run.py api_inference \
    --json_file="outputs/inference/Skywork-OR1-7B/advbench.json" \
    --output_file="outputs/inference/Skywork-OR1-7B/advbench.json" \
    --base_url="http://0.0.0.0:8002/v1" \
    --api_key="123" \
    --model="Skywork/Skywork-OR1-7B" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=1000 \
    --max_concurrent=1000 \
    --thinking_portion=0.0 \
    --max_retries=5

python3 run.py api_inference \
    --json_file="outputs/inference/Skywork-OR1-7B/advbench.json" \
    --output_file="outputs/inference/Skywork-OR1-7B/advbench.json" \
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
    --json_file="outputs/inference/Skywork-OR1-7B/advbench.json" \
    --item_type "original_item"

# ASR rate: 0.35192307692307695
# -------------------------------------------------------------

CUDA_VISIBLE_DEVICES=5 vllm serve "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" --max-model-len 32768

python3 run.py process_data \
    --dataset_name "walledai/AdvBench" \
    --split "train" \
    --output_file "outputs/inference/DeepSeek-R1-Distill-Qwen-14B/advbench.json"

python3 run.py api_inference \
    --json_file="outputs/inference/DeepSeek-R1-Distill-Qwen-14B/advbench.json" \
    --output_file="outputs/inference/DeepSeek-R1-Distill-Qwen-14B/advbench.json" \
    --base_url="http://0.0.0.0:8003/v1" \
    --api_key="123" \
    --model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=1000 \
    --max_concurrent=1000 \
    --thinking_portion=0.0 \
    --max_retries=5

python3 run.py api_inference \
    --json_file="outputs/inference/DeepSeek-R1-Distill-Qwen-14B/advbench.json" \
    --output_file="outputs/inference/DeepSeek-R1-Distill-Qwen-14B/advbench.json" \
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
    --json_file="outputs/inference/DeepSeek-R1-Distill-Qwen-14B/advbench.json" \
    --item_type "original_item"

# ASR rate: 0.14423076923076922
# -------------------------------------------------------------

CUDA_VISIBLE_DEVICES=5 vllm serve "microsoft/Phi-4-reasoning" --max-model-len 32768

python3 run.py process_data \
    --dataset_name "walledai/AdvBench" \
    --split "train" \
    --output_file "outputs/inference/Phi-4-reasoning/advbench.json"

python3 run.py api_inference \
    --json_file="outputs/inference/Phi-4-reasoning/advbench.json" \
    --output_file="outputs/inference/Phi-4-reasoning/advbench.json" \
    --base_url="http://0.0.0.0:8003/v1" \
    --api_key="123" \
    --model="microsoft/Phi-4-reasoning" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=1000 \
    --max_concurrent=1000 \
    --thinking_portion=0.0 \
    --max_retries=5 

python3 run.py api_inference \
    --json_file="outputs/inference/Phi-4-reasoning/advbench.json" \
    --output_file="outputs/inference/Phi-4-reasoning/advbench.json" \
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
    --json_file="outputs/inference/Phi-4-reasoning/advbench.json" \
    --item_type "original_item"

# ASR rate: 0.07884615384615384
# -------------------------------------------------------------

CUDA_VISIBLE_DEVICES=5 vllm serve "microsoft/Phi-4-mini-reasoning" --max-model-len 32768

python3 run.py process_data \
    --dataset_name "walledai/AdvBench" \
    --split "train" \
    --output_file "outputs/inference/Phi-4-mini-reasoning/advbench.json"

python3 run.py api_inference \
    --json_file="outputs/inference/Phi-4-mini-reasoning/advbench.json" \
    --output_file="outputs/inference/Phi-4-mini-reasoning/advbench.json" \
    --base_url="http://0.0.0.0:8002/v1" \
    --api_key="123" \
    --model="microsoft/Phi-4-mini-reasoning" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=1000 \
    --max_concurrent=1000 \
    --thinking_portion=0.0 \
    --max_retries=5 

python3 run.py api_inference \
    --json_file="outputs/inference/Phi-4-mini-reasoning/advbench.json" \
    --output_file="outputs/inference/Phi-4-mini-reasoning/advbench.json" \
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
    --json_file="outputs/inference/Phi-4-mini-reasoning/advbench.json" \
    --item_type "original_item"

# ASR rate: 0.175
# -------------------------------------------------------------

CUDA_VISIBLE_DEVICES=5 vllm serve "RealSafe/RealSafe-R1-8B" --max-model-len 32768

python3 run.py process_data \
    --dataset_name "walledai/AdvBench" \
    --split "train" \
    --output_file "outputs/inference/RealSafe-R1-8B/advbench.json"

python3 run.py api_inference \
    --json_file="outputs/inference/RealSafe-R1-8B/advbench.json" \
    --output_file="outputs/inference/RealSafe-R1-8B/advbench.json" \
    --base_url="http://0.0.0.0:8003/v1" \
    --api_key="123" \
    --model="RealSafe/RealSafe-R1-8B" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=1000 \
    --max_concurrent=1000 \
    --thinking_portion=0.0 \
    --max_retries=5 

python3 run.py api_inference \
    --json_file="outputs/inference/RealSafe-R1-8B/advbench.json" \
    --output_file="outputs/inference/RealSafe-R1-8B/advbench.json" \
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
    --json_file="outputs/inference/RealSafe-R1-8B/advbench.json" \
    --item_type "original_item"

# ASR rate: 0.0
# -------------------------------------------------------------

CUDA_VISIBLE_DEVICES=5 vllm serve "RealSafe/RealSafe-R1-7B" --max-model-len 32768

python3 run.py process_data \
    --dataset_name "walledai/AdvBench" \
    --split "train" \
    --output_file "outputs/inference/RealSafe-R1-7B/advbench.json"

python3 run.py api_inference \
    --json_file="outputs/inference/RealSafe-R1-7B/advbench.json" \
    --output_file="outputs/inference/RealSafe-R1-7B/advbench.json" \
    --base_url="http://0.0.0.0:8003/v1" \
    --api_key="123" \
    --model="RealSafe/RealSafe-R1-7B" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=1000 \
    --max_concurrent=1000 \
    --thinking_portion=0.0 \
    --max_retries=5 

python3 run.py api_inference \
    --json_file="outputs/inference/RealSafe-R1-7B/advbench.json" \
    --output_file="outputs/inference/RealSafe-R1-7B/advbench.json" \
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
    --json_file="outputs/inference/RealSafe-R1-7B/advbench.json" \
    --item_type "original_item"

# ASR rate: 0.0
# -------------------------------------------------------------