CUDA_VISIBLE_DEVICES=2,3 python3 run.py ablating_head_generation \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --json_path outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/jbbench.json \
    --batch_size 8 \
    --truncate_num 200 \
    --top_n_ablation 16 \
    --save_path outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/ablating_outputs.json \
    --max_new_tokens 4096 \
    --temperature 0.7 \
    --do_sample True \
    --item_type "original_item" \
    --thinking_portion 0 \
    --head_ablation_path outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/refusal_suppression_heads.json

CUDA_VISIBLE_DEVICES=0,1 python3 run.py ablating_head_prober \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --json_path outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/jbbench.json \
    --batch_size 16 \
    --truncate_num 200 \
    --top_n_ablation 32 \
    --save_path outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/ablating_outputs.json \
    --item_type "original_item" \
    --thinking_portion 0 \
    --head_ablation_path outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/refusal_suppression_heads.json \
    --prober_path outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/linear_prober.pt