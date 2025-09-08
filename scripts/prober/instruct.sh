python3 run.py test_prober \
    --json_path outputs/refusal/qwen_4b_prober/jbbench.json \
    --ckpt_path outputs/refusal/qwen_4b_prober/linear_prober.pt \
    --model_path "Qwen/Qwen3-4B-Thinking-2507" \
    --layer_index 32 \
    --max_items 1000 \
    --thinking_portion 0 \
    --item_type "original_item"