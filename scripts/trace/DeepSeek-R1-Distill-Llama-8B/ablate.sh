CUDA_VISIBLE_DEVICES=2 python3 run.py ablating_head_generation \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --json_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench.json \
    --batch_size 8 \
    --truncate_num 200 \
    --top_n_ablation 32 \
    --save_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/ablating_outputs.json \
    --max_new_tokens 4096 \
    --temperature 0.7 \
    --do_sample True \
    --item_type "original_item" \
    --thinking_portion 0 \
    --head_ablation_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/refusal_suppression_heads.json

CUDA_VISIBLE_DEVICES=0 python3 run.py test_prober \
    --json_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench.json \
    --ckpt_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/linear_prober.pt \
    --model_path "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --layer_index 32 \
    --max_items 1000 \
    --thinking_portion 0 \
    --item_type "original_item" \
    --head_ablation_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/refusal_suppression_heads.json \
    --top_n_ablation 128 \
    --enhance False