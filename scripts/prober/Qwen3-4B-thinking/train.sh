python3 run.py collect_non_refusal \
    --json_path outputs/refusal/Qwen3-4B-thinking/ultrachat.json \
    --save_path outputs/refusal/Qwen3-4B-thinking/no_refusal.json

python3 run.py collect_refusal \
    --json_path outputs/refusal/Qwen3-4B-thinking/advbench.json \
    --save_path outputs/refusal/Qwen3-4B-thinking/refusal.json

CUDA_VISIBLE_DEVICES=0 python3 run.py extract_hidden_states \
    --model_path "Qwen/Qwen3-4B-Thinking-2507" \
    --json_path outputs/refusal/Qwen3-4B-thinking/no_refusal.json \
    --save_path outputs/refusal/Qwen3-4B-thinking/no_refusal.pt \
    --layer_index 36

CUDA_VISIBLE_DEVICES=0 python3 run.py extract_hidden_states \
    --model_path "Qwen/Qwen3-4B-Thinking-2507" \
    --json_path outputs/refusal/Qwen3-4B-thinking/refusal.json \
    --save_path outputs/refusal/Qwen3-4B-thinking/refusal.pt \
    --layer_index 36

python3 run.py train_prober \
    --or_path outputs/refusal/Qwen3-4B-thinking/no_refusal.pt \
    --jbb_path outputs/refusal/Qwen3-4B-thinking/refusal.pt \
    --epochs 5 \
    --save_path outputs/refusal/Qwen3-4B-thinking/linear_prober.pt