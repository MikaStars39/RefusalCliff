python3 run.py plot_multiple_prober_results \
    --pt_paths outputs/fig/instruct_files.json \
    --title "Prober Results Comparison" \
    --save_path outputs/fig/instruct_prober.png


python3 fig/draw_attention_heatmap.py plot \
   --data_path="outputs/refusal/llama_8b_last_layer/llama_refusal_direction_outputs.json" \
   --value_key="cosine_similarity" \
   --save_path="outputs/fig/llama_8b_last_layer_refusal_heatmap.png" \
   --width 3.8 \
   --height 3