CUDA_VISIBLE_DEVICES=0 python3 run.py test_prober \
    --json_path outputs/refusal/DeepSeek-R1-Distill-Qwen-14B/jbbench.json \
    --ckpt_path outputs/refusal/DeepSeek-R1-Distill-Qwen-14B/linear_prober.pt \
    --model_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" \
    --layer_index 28 \
    --max_items 1000 \
    --thinking_portion 0 \
    --item_type "original_item" \
    --head_ablation_path outputs/refusal/DeepSeek-R1-Distill-Qwen-14B/refusal_suppression_heads.json \
    --top_n_ablation 32 \
    --enhance True \
    --random_heads True

CUDA_VISIBLE_DEVICES=3 python3 run.py ablating_head_generation \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
    --json_path outputs/refusal/DeepSeek-R1-Distill-Qwen-14B/jbbench.json \
    --batch_size 8 \
    --truncate_num 200 \
    --top_n_ablation 64 \
    --save_path outputs/refusal/DeepSeek-R1-Distill-Qwen-14B/ablating_outputs.json \
    --max_new_tokens 4096 \
    --temperature 0.7 \
    --do_sample True \
    --item_type "original_item" \
    --thinking_portion 0 \
    --head_ablation_path outputs/refusal/DeepSeek-R1-Distill-Qwen-14B/refusal_suppression_heads.json

python3 run.py api_inference \
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Qwen-14B/ablating_outputs.json" \
    --output_file="outputs/refusal/DeepSeek-R1-Distill-Qwen-14B/ablating_outputs.json" \
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
    --json_file="outputs/refusal/DeepSeek-R1-Distill-Qwen-14B/ablating_outputs.json" \
    --item_type "original_item"