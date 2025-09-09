python3 run.py test_prober \
    --json_path outputs/refusal/Meta-Llama-3.1-8B-Instruct/jbbench.json \
    --ckpt_path outputs/refusal/llama_8b_last_layer/linear_prober.pt \
    --model_path "unsloth/Llama-3.1-8B-Instruct" \
    --layer_index 32 \
    --max_items 1000 \
    --thinking_portion -1 \
    --item_type "original_item"

python3 fig/draw_refusal_score.py \
    --pt_paths outputs/fig/instruct_files.json \
    --save_path outputs/fig/instruct_refusal_score.pdf

