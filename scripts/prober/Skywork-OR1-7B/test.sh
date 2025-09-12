CUDA_VISIBLE_DEVICES=0 python3 run.py test_prober \
    --json_path outputs/refusal/Skywork-OR1-7B/jbbench.json \
    --ckpt_path outputs/refusal/Skywork-OR1-7B/linear_prober.pt \
    --model_path "Skywork/Skywork-OR1-7B" \
    --layer_index 28 \
    --max_items 1000 \
    --thinking_portion 0 \
    --item_type "original_item"
