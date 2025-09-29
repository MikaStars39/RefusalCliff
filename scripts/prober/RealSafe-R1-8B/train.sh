python3 run.py collect_non_refusal \
    --json_path outputs/refusal/RealSafe-R1-8B/ultrachat.json \
    --save_path outputs/refusal/RealSafe-R1-8B/no_refusal.json

python3 run.py collect_refusal \
    --json_path outputs/refusal/RealSafe-R1-8B/advbench.json \
    --save_path outputs/refusal/RealSafe-R1-8B/refusal.json

CUDA_VISIBLE_DEVICES=0 python3 run.py extract_hidden_states \
    --model_path "RealSafe/RealSafe-R1-8B" \
    --json_path outputs/refusal/RealSafe-R1-8B/no_refusal.json \
    --save_path outputs/refusal/RealSafe-R1-8B/no_refusal.pt \
    --layer_index 32

CUDA_VISIBLE_DEVICES=0 python3 run.py extract_hidden_states \
    --model_path "RealSafe/RealSafe-R1-8B" \
    --json_path outputs/refusal/RealSafe-R1-8B/refusal.json \
    --save_path outputs/refusal/RealSafe-R1-8B/refusal.pt \
    --layer_index 32

python3 run.py train_prober \
    --or_path outputs/refusal/RealSafe-R1-8B/no_refusal.pt \
    --jbb_path outputs/refusal/RealSafe-R1-8B/refusal.pt \
    --epochs 1 \
    --save_path outputs/refusal/RealSafe-R1-8B/linear_prober.pt
