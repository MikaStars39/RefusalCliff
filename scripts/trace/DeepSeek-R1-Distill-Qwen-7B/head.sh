

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 run.py find_refusal_head \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --json_path outputs/inference/DeepSeek-R1-Distill-Llama-8B/jbbench_distill_llama_8b.json \
    --batch_size 8 \
    --truncate_num 200 \
    --save_path outputs/refusal/llama_8b_last_layer/llama_refusal_direction_outputs.json \
    --refusal_direction_path outputs/refusal/llama_8b_last_layer/refusal_vector.pt \
    --layer_idx 21 \
    --thinking_portion 0 \
    --item_type "original_item"