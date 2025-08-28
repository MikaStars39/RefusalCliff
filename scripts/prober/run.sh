python3 /diancpfs/user/qingyu/persona/debug.py \
  --json_path /diancpfs/user/qingyu/persona/outputs/refusal/llama_8b_refusal.json \
  --save_path /diancpfs/user/qingyu/persona/outputs/refusal/tensor/positive.pt \
  --max_items 1000

python3 /diancpfs/user/qingyu/persona/debug.py \
  --json_path /diancpfs/user/qingyu/persona/outputs/refusal/llama_8b_no_refusal.json \
  --save_path /diancpfs/user/qingyu/persona/outputs/refusal/tensor/negative.pt \
  --max_items 1000


python3 /diancpfs/user/qingyu/persona/src/prober.py train \
  --or_path /diancpfs/user/qingyu/persona/outputs/refusal/tensor/negative.pt \
  --jbb_path /diancpfs/user/qingyu/persona/outputs/refusal/tensor/positive.pt \
  --epochs 2 --batch_size 8 --learning_rate 1e-4 \
  --val_split 0.1 --seed 42 \
  --save_path /diancpfs/user/qingyu/persona/outputs/refusal/tensor/linear_prober.pt

python3 /diancpfs/user/qingyu/persona/src/prober.py test \
  --json_path /diancpfs/user/qingyu/persona/outputs/inference/DeepSeek-R1-Distill-Llama-8B/jbbench_distill_llama_8b.json \
  --ckpt_path /diancpfs/user/qingyu/persona/outputs/refusal/tensor/linear_prober.pt \
  --layer_index 20 \
  --max_items 100 \
  --item_type "safe_item"