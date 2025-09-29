CUDA_VISIBLE_DEVICES=0 python3 run.py test_prober \
    --json_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench.json \
    --ckpt_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/linear_prober.pt \
    --model_path "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --layer_index 32 \
    --max_items 1000 \
    --thinking_portion 0 \
    --item_type "original_item"


CUDA_VISIBLE_DEVICES=0 python3 run.py test_prober \
    --json_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench.json \
    --ckpt_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/linear_prober.pt \
    --model_path "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --layer_index 28 \
    --max_items 1000 \
    --thinking_portion 0 \
    --item_type "original_item"

mv outputs/refusal/DeepSeek-R1-Distill-Llama-8B/linear_prober_normalized_comparison_results.pt outputs/refusal/DeepSeek-R1-Distill-Llama-8B/layer28.pt

CUDA_VISIBLE_DEVICES=0 python3 run.py test_prober \
    --json_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench.json \
    --ckpt_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/linear_prober.pt \
    --model_path "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --layer_index 24 \
    --max_items 1000 \
    --thinking_portion 0 \
    --item_type "original_item"

mv outputs/refusal/DeepSeek-R1-Distill-Llama-8B/linear_prober_normalized_comparison_results.pt outputs/refusal/DeepSeek-R1-Distill-Llama-8B/layer24.pt

CUDA_VISIBLE_DEVICES=0 python3 run.py test_prober \
    --json_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench.json \
    --ckpt_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/linear_prober.pt \
    --model_path "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --layer_index 20 \
    --max_items 1000 \
    --thinking_portion 0 \
    --item_type "original_item"

mv outputs/refusal/DeepSeek-R1-Distill-Llama-8B/linear_prober_normalized_comparison_results.pt outputs/refusal/DeepSeek-R1-Distill-Llama-8B/layer20.pt

CUDA_VISIBLE_DEVICES=0 python3 run.py test_prober \
    --json_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench.json \
    --ckpt_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/linear_prober.pt \
    --model_path "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --layer_index 16 \
    --max_items 1000 \
    --thinking_portion 0 \
    --item_type "original_item"

mv outputs/refusal/DeepSeek-R1-Distill-Llama-8B/linear_prober_normalized_comparison_results.pt outputs/refusal/DeepSeek-R1-Distill-Llama-8B/layer16.pt

CUDA_VISIBLE_DEVICES=0 python3 run.py test_prober \
    --json_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench.json \
    --ckpt_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/linear_prober.pt \
    --model_path "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --layer_index 12 \
    --max_items 1000 \
    --thinking_portion 0 \
    --item_type "original_item"

mv outputs/refusal/DeepSeek-R1-Distill-Llama-8B/linear_prober_normalized_comparison_results.pt outputs/refusal/DeepSeek-R1-Distill-Llama-8B/layer12.pt

CUDA_VISIBLE_DEVICES=0 python3 run.py test_prober \
    --json_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench.json \
    --ckpt_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/linear_prober.pt \
    --model_path "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --layer_index 8 \
    --max_items 1000 \
    --thinking_portion 0 \
    --item_type "original_item"

mv outputs/refusal/DeepSeek-R1-Distill-Llama-8B/linear_prober_normalized_comparison_results.pt outputs/refusal/DeepSeek-R1-Distill-Llama-8B/layer8.pt

CUDA_VISIBLE_DEVICES=0 python3 run.py test_prober \
    --json_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench.json \
    --ckpt_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/linear_prober.pt \
    --model_path "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --layer_index 4 \
    --max_items 1000 \
    --thinking_portion 0 \
    --item_type "original_item"

mv outputs/refusal/DeepSeek-R1-Distill-Llama-8B/linear_prober_normalized_comparison_results.pt outputs/refusal/DeepSeek-R1-Distill-Llama-8B/layer4.pt


CUDA_VISIBLE_DEVICES=0 python3 run.py test_prober \
    --json_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench.json \
    --ckpt_path outputs/refusal/DeepSeek-R1-Distill-Llama-8B/linear_prober.pt \
    --model_path "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --layer_index 0 \
    --max_items 1000 \
    --thinking_portion 0 \
    --item_type "original_item"

mv outputs/refusal/DeepSeek-R1-Distill-Llama-8B/linear_prober_normalized_comparison_results.pt outputs/refusal/DeepSeek-R1-Distill-Llama-8B/layer0.pt
