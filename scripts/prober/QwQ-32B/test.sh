CUDA_VISIBLE_DEVICES=0 python3 run.py test_prober \
    --json_path outputs/refusal/QwQ-32B/jbbench.json \
    --ckpt_path outputs/refusal/QwQ-32B/linear_prober.pt \
    --model_path "Qwen/QwQ-32B" \
    --layer_index 28 \
    --max_items 1000 \
    --thinking_portion 0 \
    --item_type "original_item"