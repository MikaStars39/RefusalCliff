# generation

```python
python3 run.py api_inference \
    --json_file="" \
    --output_file="" \
    --base_url="" \
    --api_key="" \
    --model="" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=256 \
    --max_concurrent=1000 \
    --max_retries=5
```

# train a prober

### deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

```python
python3 run.py process_data \
    --dataset_name "HuggingFaceH4/ultrachat_200k" \
    --split "test_sft" \
    --output_file "outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/ultrachat.json"

python3 run.py process_data \
    --dataset_name "walledai/AdvBench" \
    --split "train" \
    --output_file "outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/advbench.json"

python3 run.py process_data \
    --dataset_name "walledai/JailbreakBench" \
    --split "train" \
    --output_file "outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/jbbench.json"

python3 run.py api_inference \
    --json_file="outputs/refusal/qwen_4b_prober/ultrachat.json" \
    --output_file="outputs/refusal/qwen_4b_prober/ultrachat.json" \
    --base_url="http://0.0.0.0:8000/v1" \
    --api_key="" \
    --model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=256 \
    --max_concurrent=1000 \
    --max_retries=5

python3 run.py api_inference \
    --json_file="outputs/refusal/qwen_4b_prober/advbench.json" \
    --output_file="outputs/refusal/qwen_4b_prober/advbench.json" \
    --base_url="http://0.0.0.0:8000/v1" \
    --api_key="" \
    --model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=1000 \
    --max_concurrent=1000 \
    --max_retries=5

python3 run.py api_inference \
    --json_file="outputs/refusal/qwen_4b_prober/jbbench.json" \
    --output_file="outputs/refusal/qwen_4b_prober/jbbench.json" \
    --base_url="http://0.0.0.0:8000/v1" \
    --api_key="" \
    --model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    --max_tokens=8192 \
    --temperature=0.7 \
    --request_time=0.0 \
    --truncated_num=1000 \
    --max_concurrent=1000 \
    --max_retries=5

python3 run.py collect_non_refusal \
    --json_path outputs/refusal/qwen_4b_prober/ultrachat.json \
    --save_path outputs/refusal/qwen_4b_prober/no_refusal.json

python3 run.py collect_refusal \
    --json_path outputs/refusal/qwen_4b_prober/advbench.json \
    --save_path outputs/refusal/qwen_4b_prober/refusal.json

CUDA_VISIBLE_DEVICES=0 python3 run.py extract_hidden_states \
    --json_path outputs/refusal/qwen_4b_prober/no_refusal.json \
    --save_path outputs/refusal/qwen_4b_prober/no_refusal.pt \
    --layer_index 32

CUDA_VISIBLE_DEVICES=0 python3 run.py extract_hidden_states \
    --json_path outputs/refusal/qwen_4b_prober/refusal.json \
    --save_path outputs/refusal/qwen_4b_prober/refusal.pt \
    --layer_index 32

python3 run.py train_prober \
    --or_path outputs/refusal/qwen_4b_prober/no_refusal.pt \
    --jbb_path outputs/refusal/qwen_4b_prober/refusal.pt \
    --epochs 5 \
    --save_path outputs/refusal/qwen_4b_prober/linear_prober.pt

python3 run.py test_prober \
    --json_path outputs/refusal/qwen_4b_prober/jbbench.json \
    --ckpt_path outputs/refusal/qwen_4b_prober/linear_prober.pt \
    --model_path "Qwen/Qwen3-4B-Thinking-2507" \
    --layer_index 32 \
    --max_items 1000 \
    --thinking_portion 0 \
    --item_type "original_item"
```

# Tracing Attention Heads

```python
CUDA_VISIBLE_DEVICES=2,3 python3 run.py trace_attn_head \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --json_path outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/jbbench.json \
    --prober_path outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/linear_prober.pt \
    --layer_idx 28 \
    --position_idx -1 \
    --batch_size 8 \
    --truncate_num 256 \
    --save_path outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/head_tracing.json \
    --item_type "original_item"

python3 run.py extract_prober_weights \
    --ckpt_path outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/linear_prober.pt \
    --save_path outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/refusal_vector_weights.pt

CUDA_VISIBLE_DEVICES=0,1 python3 run.py ablating_attn_head \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --json_path outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/jbbench.json \
    --prober_path outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/linear_prober.pt \
    --batch_size 8 \
    --truncate_num 64 \
    --top_n_ablation 16 \
    --ablation_value 0.01 \
    --enhancement_value 1 \
    --item_type "original_item" \
    --layer_idx 28 \
    --head_ablation_path outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/head_tracing.json \
    --head_enhancement_path outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/head_tracing.json
```

# Refusal Direction

```python
CUDA_VISIBLE_DEVICES=0,1 python3 run.py find_refusal_head \
    --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --json_path outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/jbbench.json \
    --layer_idx 28 \
    --position_idx -1 \
    --batch_size 1 \
    --truncate_num 1 \
    --save_path outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/head_tracing.json \
    --item_type "original_item" \
    --thinking_portion 0 \
    --refusal_direction_path outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/refusal_vector_weights.pt
```

# Drawing

```python
python3 run analyze_attn_patterns \
    --heads_json_path /path/to/heads.json \
    --data_json_path /path/to/data.json \
    --output_folder /path/to/output \
    --top_n 5 \
    --token_position -1
```

```python
python3 run.py plot_multiple_prober_results \
    --pt_paths outputs/fig/pt_files.json \
    --title "Prober Results Comparison" \
    --save_path outputs/fig/prober_results.png


python3 fig/draw_attention_heatmap.py plot \
   --data_path="outputs/refusal/llama_8b_last_layer/llama_refusal_direction_outputs.json" \
   --value_key="cosine_similarity" \
   --save_path="outputs/fig/llama_8b_last_layer_refusal_heatmap.png" \
   --width 5 \
   --height 4

python3 fig/draw_attention_heatmap.py plot \
   --data_path="outputs/refusal/DeepSeek-R1-Distill-Qwen-7B/head_tracing.json" \
   --value_key="cosine_similarity" \
   --save_path="outputs/fig/qwen7b_refusal_heatmap.png" \
   --width 5 \
   --height 4
```