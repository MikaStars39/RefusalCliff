python3 run.py collect_non_refusal \
    --json_path outputs/refusal/Qwen3-30B-A3B-Thinking-2507/ultrachat.json \
    --save_path outputs/refusal/Qwen3-30B-A3B-Thinking-2507/no_refusal.json

python3 run.py collect_refusal \
    --json_path outputs/refusal/Qwen3-30B-A3B-Thinking-2507/advbench.json \
    --save_path outputs/refusal/Qwen3-30B-A3B-Thinking-2507/refusal.json

CUDA_VISIBLE_DEVICES=0 python3 run.py extract_hidden_states \
    --model_path "Qwen/Qwen3-30B-A3B-Thinking-2507" \
    --json_path outputs/refusal/Qwen3-30B-A3B-Thinking-2507/no_refusal.json \
    --save_path outputs/refusal/Qwen3-30B-A3B-Thinking-2507/no_refusal.pt \
    --layer_index 48

CUDA_VISIBLE_DEVICES=0 python3 run.py extract_hidden_states \
    --model_path "Qwen/Qwen3-30B-A3B-Thinking-2507" \
    --json_path outputs/refusal/Qwen3-30B-A3B-Thinking-2507/refusal.json \
    --save_path outputs/refusal/Qwen3-30B-A3B-Thinking-2507/refusal.pt \
    --layer_index 48

python3 run.py train_prober \
    --or_path outputs/refusal/Qwen3-30B-A3B-Thinking-2507/no_refusal.pt \
    --jbb_path outputs/refusal/Qwen3-30B-A3B-Thinking-2507/refusal.pt \
    --epochs 5 \
    --save_path outputs/refusal/Qwen3-30B-A3B-Thinking-2507/linear_prober.pt
