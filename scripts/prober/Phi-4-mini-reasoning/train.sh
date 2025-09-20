python3 run.py collect_non_refusal \
    --json_path outputs/refusal/Phi-4-mini-reasoning/ultrachat.json \
    --save_path outputs/refusal/Phi-4-mini-reasoning/no_refusal.json

python3 run.py collect_refusal \
    --json_path outputs/refusal/Phi-4-mini-reasoning/advbench.json \
    --save_path outputs/refusal/Phi-4-mini-reasoning/refusal.json

CUDA_VISIBLE_DEVICES=0 python3 run.py extract_hidden_states \
    --model_path "microsoft/Phi-4-mini-reasoning" \
    --json_path outputs/refusal/Phi-4-mini-reasoning/no_refusal.json \
    --save_path outputs/refusal/Phi-4-mini-reasoning/no_refusal.pt \
    --layer_index 32

CUDA_VISIBLE_DEVICES=0 python3 run.py extract_hidden_states \
    --model_path "microsoft/Phi-4-mini-reasoning" \
    --json_path outputs/refusal/Phi-4-mini-reasoning/refusal.json \
    --save_path outputs/refusal/Phi-4-mini-reasoning/refusal.pt \
    --layer_index 32

python3 run.py train_prober \
    --or_path outputs/refusal/Phi-4-mini-reasoning/no_refusal.pt \
    --jbb_path outputs/refusal/Phi-4-mini-reasoning/refusal.pt \
    --epochs 5 \
    --save_path outputs/refusal/Phi-4-mini-reasoning/linear_prober.pt
