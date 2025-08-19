python3 src/asr.py \
    --json_file="outputs/advbench_distill_qwen_7b.json" \
    --type_eval="original_item"

python3 src/asr.py \
    --json_file="/diancpfs/user/qingyu/persona/outputs/harmful_train/or_bench_hard_1k_llama_8b_safe.json" \
    --type_eval="original_item"

python3 src/asr.py \
    --json_file="/diancpfs/user/qingyu/persona/outputs/harmful_train/or_bench_hard_1k_llama_8b_harmful.json" \
    --type_eval="original_item"

python3 src/asr.py \
    --json_file="outputs/jbbench_distill_llama_8b.json" \
    --type_eval="safe_item"

python3 src/asr.py \
    --json_file="outputs/old/or/advbench_distill_llama_8b.json" \
    --type_eval="original_item"