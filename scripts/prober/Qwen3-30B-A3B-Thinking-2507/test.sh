CUDA_VISIBLE_DEVICES=0 python3 run.py test_prober \
    --json_path outputs/refusal/Qwen3-30B-A3B-Thinking-2507/jbbench.json \
    --ckpt_path outputs/refusal/Qwen3-30B-A3B-Thinking-2507/linear_prober.pt \
    --model_path "Qwen/Qwen3-30B-A3B-Thinking-2507" \
    --layer_index 48 \
    --max_items 1000 \
    --thinking_portion 0 \
    --item_type "original_item"
