
python3 run.py test_prober \
    --json_path outputs/inference/DeepSeek-R1-Distill-Llama-8B/jbbench_distill_llama_8b.json \
    --ckpt_path outputs/refusal/llama_8b_last_layer/linear_prober.pt \
    --model_path "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --layer_index 32 \
    --max_items 200 \
    --thinking_portion -1 \
    --item_type "original_item"