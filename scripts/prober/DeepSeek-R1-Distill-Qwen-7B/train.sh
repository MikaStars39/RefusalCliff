python3 run.py collect_non_refusal \
    --json_path outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/ultrachat.json \
    --save_path outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/no_refusal.json

python3 run.py collect_refusal \
    --json_path outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/advbench.json \
    --save_path outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/refusal.json

CUDA_VISIBLE_DEVICES=0 python3 run.py extract_hidden_states \
    --model_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    --json_path outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/no_refusal.json \
    --save_path outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/no_refusal.pt \
    --layer_index 28

CUDA_VISIBLE_DEVICES=0 python3 run.py extract_hidden_states \
    --model_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    --json_path outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/refusal.json \
    --save_path outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/refusal.pt \
    --layer_index 28

python3 run.py train_prober \
    --or_path outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/no_refusal.pt \
    --jbb_path outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/refusal.pt \
    --epochs 5 \
    --save_path outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/linear_prober.pt
