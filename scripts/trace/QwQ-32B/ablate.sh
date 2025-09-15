CUDA_VISIBLE_DEVICES=2,3 python3 run.py ablating_head_generation \
    --model_name Skywork/Skywork-OR1-7B \
    --json_path outputs/refusal/Skywork-OR1-7B/jbbench.json \
    --batch_size 8 \
    --truncate_num 200 \
    --top_n_ablation 16 \
    --save_path outputs/refusal/Skywork-OR1-7B/ablating_outputs.json \
    --max_new_tokens 4096 \
    --temperature 0.7 \
    --do_sample True \
    --item_type "original_item" \
    --thinking_portion 0 \
    --head_ablation_path outputs/refusal/Skywork-OR1-7B/refusal_suppression_heads.json

CUDA_VISIBLE_DEVICES=0,1 python3 run.py ablating_head_prober \
    --model_name Skywork/Skywork-OR1-7B \
    --json_path outputs/refusal/Skywork-OR1-7B/ultrachat.json \
    --batch_size 16 \
    --truncate_num 200 \
    --top_n_ablation 0 \
    --item_type "original_item" \
    --thinking_portion 0 \
    --head_ablation_path outputs/refusal/Skywork-OR1-7B/refusal_suppression_heads.json \
    --prober_path outputs/refusal/Skywork-OR1-7B/linear_prober.pt

CUDA_VISIBLE_DEVICES=0 python3 run.py test_prober \
    --json_path outputs/refusal/Skywork-OR1-7B/jbbench.json \
    --ckpt_path outputs/refusal/Skywork-OR1-7B/linear_prober.pt \
    --model_path "Skywork/Skywork-OR1-7B" \
    --layer_index 28 \
    --max_items 1000 \
    --thinking_portion 0 \
    --item_type "original_item" \
    --head_ablation_path outputs/refusal/Skywork-OR1-7B/refusal_suppression_heads.json \
    --top_n_ablation 128 \
    --enhance True