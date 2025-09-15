python3 run.py extract_prober_weights \
    --ckpt_path outputs/refusal/Skywork-OR1-7B/linear_prober.pt \
    --save_path outputs/refusal/Skywork-OR1-7B/refusal_vector.pt

CUDA_VISIBLE_DEVICES=1 python3 run.py find_refusal_head \
    --model_name Skywork/Skywork-OR1-7B \
    --json_path outputs/refusal/Skywork-OR1-7B/advbench.json \
    --batch_size 4 \
    --truncate_num 1000 \
    --save_path outputs/refusal/Skywork-OR1-7B/refusal_suppression_heads.json \
    --refusal_direction_path outputs/refusal/Skywork-OR1-7B/refusal_vector.pt \
    --layer_idx 28 \
    --thinking_portion 0 \
    --item_type "original_item"