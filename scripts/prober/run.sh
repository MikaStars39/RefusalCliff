python3 /diancpfs/user/qingyu/persona/main.py extract \
  --json_path /diancpfs/user/qingyu/persona/outputs/old/or/or_bench_hard_1k_llama_8b.json \
  --save_path /diancpfs/user/qingyu/persona/outputs/tensor/or_tensor_list_original.pt \
  --max_items 1000

python3 /diancpfs/user/qingyu/persona/main.py extract \
  --json_path /diancpfs/user/qingyu/persona/outputs/old/or/advbench_distill_llama_8b.json \
  --save_path /diancpfs/user/qingyu/persona/outputs/tensor/jbb_tensor_list_original.pt \
  --max_items 1000


python3 /diancpfs/user/qingyu/persona/main.py train \
  --or_path /diancpfs/user/qingyu/persona/outputs/tensor/or_tensor_list_original.pt \
  --jbb_path /diancpfs/user/qingyu/persona/outputs/tensor/jbb_tensor_list_original.pt \
  --epochs 10 --batch_size 64 --learning_rate 1e-4 \
  --val_split 0.1 --seed 42 \
  --save_path /diancpfs/user/qingyu/persona/outputs/tensor/linear_prober.pt

python3 main.py test_prober \
  --json_path /diancpfs/user/qingyu/persona/outputs/old/or/advbench_distill_llama_8b.json \
  --ckpt_path /diancpfs/user/qingyu/persona/outputs/tensor/linear_prober.pt

CUDA_VISIBLE_DEVICES=6,7 python3 main.py calculate_entropy \
  --json_path /diancpfs/user/qingyu/persona/outputs/DeepSeek-R1-Distill-Llama-8B-abliterated/jbbench.json

CUDA_VISIBLE_DEVICES=6,7 python3 entropy.py \
  --json_paths '["outputs/DeepSeek-R1-Distill-Llama-8B/jbbench_distill_llama_8b.json", "outputs/DeepSeek-R1-Distill-Llama-8B/jbbench_distill_llama_8b_safe.json", "outputs/DeepSeek-R1-Distill-Llama-8B/jbbench_distill_llama_8b_self_safe.json", "outputs/DeepSeek-R1-Distill-Llama-8B/jbbench_distill_llama_8b_self_harmful.json", "outputs/DeepSeek-R1-Distill-Llama-8B/jbbench_distill_llama_8b_harmful.json"]' \
  --output_plot /diancpfs/user/qingyu/persona/outputs/fig/DeepSeek-R1-Distill-Llama-8B-abliterated.png