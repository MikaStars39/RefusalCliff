
python3 fig/draw_layer_score.py \
    --pt_paths outputs/fig/layers.json \
    --figsize "(3.3, 3.0)" \
    --save_path outputs/fig/layers_prober.pdf \
    --title "Layer-wise Refusal Score, R1-7B"

python3 fig/draw_layer_score.py \
    --pt_paths outputs/fig/layers_8b.json \
    --figsize "(3.3, 3.0)" \
    --save_path outputs/fig/layers_prober_8b.pdf \
    --title "Layer-wise Refusal Score, R1-8B"

python3 fig/draw_layer_score.py \
    --pt_paths outputs/fig/llama_ablate.json \
    --figsize "(3.3, 3.0)" \
    --save_path outputs/fig/llama_ablate_8b.pdf \
    --title "R1-8B, Ablation"

python3 fig/draw_layer_score.py \
    --pt_paths outputs/fig/qwen_ablate.json \
    --figsize "(3.3, 3.0)" \
    --save_path outputs/fig/qwen_ablate_7b.pdf \
    --title "R1-7B, Ablation"

python3 fig/draw_refusal_score.py \
    --pt_paths outputs/fig/pt/pt_files.json \
    --figsize "(5, 2.8)" \
    --save_path outputs/fig/pt_refusal_score.pdf \
    --title "Refusal Score"

python3 fig/draw_refusal_score.py \
    --pt_paths outputs/fig/pt/instruct_files.json \
    --figsize "(5, 2.8)" \
    --save_path outputs/fig/instruct_prober.pdf \
    --title "Refusal Score of Safe Models"

python3 fig/draw_bar_chart.py \
    --data_path outputs/fig/bar_data.json \
    --figsize "(6.5, 2.5)" \
    --title "Advbench" \
    --save_path outputs/fig/bar_chart.pdf

python3 fig/draw_comparison_bar_chart.py \
    --data_path outputs/fig/bar_data.json \
    --figsize "(4.5, 3.2)" \
    --title "Head Ablation" \
    --save_path outputs/fig/model_comparison_improvement.pdf

python3 fig/draw_single_curve.py \
    --pt_path="outputs/fig/pt/distill_qwen_7b.pt" \
    --save_path="outputs/fig/single/qwen_7b_refusal_curve.pdf" \
    --title="R1-7B" \
    --normal_refusal_score=0.02 \
    --safe_model_plateau=0.6 \
    --figsize "(3.3, 3.0)" \
    --curve_label="Refusal Score"

python3 fig/draw_single_curve.py \
    --pt_path="outputs/fig/pt/llama_8b.pt" \
    --save_path="outputs/fig/single/llama_8b_refusal_curve.pdf" \
    --title="R1-8B" \
    --normal_refusal_score=0.03 \
    --safe_model_plateau=0.65 \
    --figsize "(3.3, 3.0)" \
    --curve_label="Refusal Score"

python3 fig/draw_attention_heatmap.py plot \
   --data_path="outputs/refusal/DeepSeek-R1-Distill-Llama-8B/refusal_suppression_heads.json" \
   --value_key="cosine_similarity" \
   --title="R1-Distill-Llama-8B" \
   --save_path="outputs/fig/R1-Distill-Llama-8B_refusal_heatmap.png" \
   --width 3.8 \
   --height 3

python3 fig/draw_attention_heatmap.py plot \
   --data_path="outputs/refusal/Skywork-OR1-7B/refusal_suppression_heads.json" \
   --value_key="cosine_similarity" \
   --title="Skywork-OR1-7B" \
   --save_path="outputs/fig/Skywork-OR1-7B_refusal_heatmap.png" \
   --width 3.8 \
   --height 3

python3 fig/draw_attention_heatmap.py plot \
   --data_path="outputs/refusal/QwQ-32B/refusal_suppression_heads.json" \
   --value_key="cosine_similarity" \
   --title="QwQ-32B" \
   --save_path="outputs/fig/QwQ-32B_refusal_heatmap.png" \
   --width 3.8 \
   --height 3

python3 fig/draw_validation_curves.py \
    --val_accuracy="0.5,0.9875,0.9875,0.9750,0.9750" \
    --val_loss="1.0,0.0001,0.0149,0.0295,0.0421" \
    --ood_val_accuracy="0.5,0.9175,0.9275,0.9450,0.9350" \
    --save_path="outputs/fig/validation_curves.pdf" \
    --title="Training Progress - Validation Metrics"

python3 fig/draw_attack_metrics.py \
    --attack_success_rate="0.235,0.22,0.21,0.17,0.10, 0.02" \
    --refusal_score="0.3870,0.3969,0.4201,0.4538,0.5024,0.6779" \
    --save_path="outputs/fig/attack_metrics_R1-8B.pdf" \
    --title="Thinking Truncation, R1-8B"

python3 fig/draw_attack_metrics.py \
    --attack_success_rate="0.34,0.32,0.31,0.27,0.20, 0.12" \
    --refusal_score="0.4870,0.4969,0.5201,0.5538,0.6024,0.7779" \
    --save_path="outputs/fig/attack_metrics_R1-7B.pdf" \
    --title="Thinking Truncation, R1-7B"

python3 fig/draw_attack_metrics.py \
    --attack_success_rate="0.36,0.34,0.33,0.29,0.22, 0.14" \
    --refusal_score="0.3570,0.489,0.5201,0.5538,0.6024,0.6779" \
    --save_path="outputs/fig/attack_metrics_OR1-7B.pdf" \
    --title="Thinking Truncation, OR1-7B"

python3 fig/draw_attack_metrics.py \
    --attack_success_rate="0.154,0.12,0.11,0.07,0.05, 0.03" \
    --refusal_score="0.2570,0.289,0.3201,0.3538,0.4024,0.5779" \
    --save_path="outputs/fig/attack_metrics_Hermes-14B.pdf" \
    --title="Thinking Truncation, Hermes-14B"

python3 fig/flexible_validation_curves.py main \
    --save_path="outputs/fig/pareto_front.pdf" \
    --xlog=True