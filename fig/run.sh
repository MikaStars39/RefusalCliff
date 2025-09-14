
python3 fig/draw_refusal_score.py \
    --pt_paths outputs/fig/layers.json \
    --figsize "(4, 4)" \
    --save_path outputs/fig/layers_prober.pdf

python3 fig/draw_refusal_score.py \
    --pt_paths outputs/fig/pt/pt_files.json \
    --figsize "(5, 5)" \
    --save_path outputs/fig/pt_refusal_score.pdf

python3 fig/draw_bar_chart.py \
    --data_path outputs/fig/bar_data.json \
    --save_path outputs/fig/bar_chart.pdf

python3 fig/draw_attention_heatmap.py plot \
   --data_path="outputs/refusal/llama_8b_last_layer/llama_refusal_direction_outputs.json" \
   --value_key="cosine_similarity" \
   --save_path="outputs/fig/llama_8b_last_layer_refusal_heatmap.png" \
   --width 3.8 \
   --height 3