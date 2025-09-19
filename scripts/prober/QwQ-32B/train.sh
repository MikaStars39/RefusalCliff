python3 run.py collect_non_refusal \
    --json_path outputs/refusal/QwQ-32B/ultrachat.json \
    --save_path outputs/refusal/QwQ-32B/no_refusal.json

python3 run.py collect_refusal \
    --json_path outputs/refusal/QwQ-32B/advbench.json \
    --save_path outputs/refusal/QwQ-32B/refusal.json

CUDA_VISIBLE_DEVICES=0 python3 run.py extract_hidden_states \
    --model_path "Qwen/QwQ-32B" \
    --json_path outputs/refusal/QwQ-32B/no_refusal.json \
    --save_path outputs/refusal/QwQ-32B/no_refusal.pt \
    --layer_index 64

CUDA_VISIBLE_DEVICES=0 python3 run.py extract_hidden_states \
    --model_path "Qwen/QwQ-32B" \
    --json_path outputs/refusal/QwQ-32B/refusal.json \
    --save_path outputs/refusal/QwQ-32B/refusal.pt \
    --layer_index 64

python3 run.py train_prober \
    --or_path outputs/refusal/QwQ-32B/no_refusal.pt \
    --jbb_path outputs/refusal/QwQ-32B/refusal.pt \
    --epochs 5 \
    --save_path outputs/refusal/QwQ-32B/linear_prober.pt