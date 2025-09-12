CUDA_VISIBLE_DEVICES=0 python3 run.py test_prober \
    --json_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench.json \
    --ckpt_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/linear_prober.pt \
    --model_path "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --layer_index 32 \
    --max_items 1000 \
    --thinking_portion 0 \
    --item_type "original_item"
