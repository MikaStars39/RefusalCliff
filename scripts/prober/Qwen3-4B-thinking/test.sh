CUDA_VISIBLE_DEVICES=0 python3 run.py test_prober \
    --json_path outputs/refusal/Qwen3-4B-thinking/jbbench.json \
    --ckpt_path outputs/refusal/Qwen3-4B-thinking/linear_prober.pt \
    --model_path "Qwen/Qwen3-4B-Thinking-2507" \
    --layer_index 36 \
    --max_items 1000 \
    --thinking_portion 0 \
    --item_type "original_item"