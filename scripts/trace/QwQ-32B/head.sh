python3 run.py extract_prober_weights \
    --ckpt_path outputs/refusal/QwQ-32B/linear_prober.pt \
    --save_path outputs/refusal/QwQ-32B/refusal_vector.pt

CUDA_VISIBLE_DEVICES=0 python3 run.py find_refusal_head \
    --model_name Qwen/QwQ-32B \
    --json_path outputs/refusal/QwQ-32B/advbench.json \
    --batch_size 1 \
    --truncate_num 1000 \
    --save_path outputs/refusal/QwQ-32B/refusal_suppression_heads.json \
    --refusal_direction_path outputs/refusal/QwQ-32B/refusal_vector.pt \
    --layer_idx 36 \
    --thinking_portion 0 \
    --item_type "original_item"