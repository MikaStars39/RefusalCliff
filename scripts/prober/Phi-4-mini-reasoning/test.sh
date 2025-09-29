CUDA_VISIBLE_DEVICES=0 python3 run.py test_prober \
    --json_path outputs/refusal/Phi-4-mini-reasoning/jbbench.json \
    --ckpt_path outputs/refusal/Phi-4-mini-reasoning/linear_prober.pt \
    --model_path "microsoft/Phi-4-mini-reasoning" \
    --layer_index 32 \
    --max_items 1000 \
    --thinking_portion 0 \
    --item_type "original_item"
