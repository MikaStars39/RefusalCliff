
# sft
python3 src/get_vector.py \
    --json_path data/evil.json \
    --output_path outputs/evil_vectors_tulu_sft.pkl \
    --model_name_or_path allenai/Llama-3.1-Tulu-3-8B-SFT \
    --revision main

# dpo
python3 src/get_vector.py \
    --json_path data/evil.json \
    --output_path outputs/evil_vectors_tulu_dpo.pkl \
    --model_name_or_path allenai/Llama-3.1-Tulu-3-8B-DPO \
    --revision main

# RLVR
python3 src/get_vector.py \
    --json_path data/evil.json \
    --output_path outputs/evil_vectors_tulu_rlvr.pkl \
    --model_name_or_path allenai/Llama-3.1-Tulu-3-8B \
    --revision main


python3 src/conceptor.py \
    --test_path outputs/evil_vectors_tulu_sft.pkl \
    --output_path outputs/dual_conceptors_3d_sft_skip.png \
    --alpha 0.1 \
    --skip_conceptor=True

python3 src/conceptor.py \
    --test_path outputs/evil_vectors_tulu_dpo.pkl \
    --output_path outputs/dual_conceptors_3d_dpo_skip.png \
    --alpha 0.1 \
    --skip_conceptor=True

python3 src/conceptor.py \
    --test_path outputs/evil_vectors_tulu_rlvr.pkl \
    --output_path outputs/dual_conceptors_3d_rlvr_skip.png \
    --alpha 0.1 \
    --skip_conceptor=True