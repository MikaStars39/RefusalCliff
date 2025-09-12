#! /bin/bash

python3 run.py process_data \
    --dataset_name "HuggingFaceH4/ultrachat_200k" \
    --split "test_sft" \
    --output_file "outputs/refusal/DeepSeek-R1-Distill-Llama-8B/ultrachat.json"

python3 run.py process_data \
    --dataset_name "walledai/AdvBench" \
    --split "train" \
    --output_file "outputs/refusal/DeepSeek-R1-Distill-Llama-8B/advbench.json"

python3 run.py process_data \
    --dataset_name "walledai/JailbreakBench" \
    --split "train" \
    --output_file "outputs/refusal/DeepSeek-R1-Distill-Llama-8B/jbbench.json"
