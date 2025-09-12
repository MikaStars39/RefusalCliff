#!/usr/bin/env python3
"""
Auto-generate prober scripts tool
Usage: python generate_prober_scripts.py --model_dir_name "Qwen3-4B-thinking" --model_path "Qwen/Qwen3-4B-Thinking-2507"
"""

import argparse
import os
from pathlib import Path


def generate_prepare_script(model_dir_name):
    """Generate prepare.sh script"""
    return f"""#! /bin/bash

python3 run.py process_data \\
    --dataset_name "HuggingFaceH4/ultrachat_200k" \\
    --split "test_sft" \\
    --output_file "outputs/refusal/{model_dir_name}/ultrachat.json"

python3 run.py process_data \\
    --dataset_name "walledai/AdvBench" \\
    --split "train" \\
    --output_file "outputs/refusal/{model_dir_name}/advbench.json"

python3 run.py process_data \\
    --dataset_name "walledai/JailbreakBench" \\
    --split "train" \\
    --output_file "outputs/refusal/{model_dir_name}/jbbench.json"
"""


def generate_inference_script(model_dir_name, model_path):
    """Generate inference.sh script"""
    return f"""python3 run.py api_inference \\
    --json_file="outputs/refusal/{model_dir_name}/ultrachat.json" \\
    --output_file="outputs/refusal/{model_dir_name}/ultrachat.json" \\
    --base_url="http://0.0.0.0:8000/v1" \\
    --api_key="123" \\
    --model="{model_path}" \\
    --max_tokens=8192 \\
    --temperature=0.7 \\
    --request_time=0.0 \\
    --truncated_num=256 \\
    --max_concurrent=1000 \\
    --max_retries=5

python3 run.py api_inference \\
    --json_file="outputs/refusal/{model_dir_name}/advbench.json" \\
    --output_file="outputs/refusal/{model_dir_name}/advbench.json" \\
    --base_url="http://0.0.0.0:8000/v1" \\
    --api_key="123" \\
    --model="{model_path}" \\
    --max_tokens=8192 \\
    --temperature=0.7 \\
    --request_time=0.0 \\
    --truncated_num=1000 \\
    --max_concurrent=1000 \\
    --max_retries=5

python3 run.py api_inference \\
    --json_file="outputs/refusal/{model_dir_name}/jbbench.json" \\
    --output_file="outputs/refusal/{model_dir_name}/jbbench.json" \\
    --base_url="http://0.0.0.0:8000/v1" \\
    --api_key="123" \\
    --model="{model_path}" \\
    --max_tokens=8192 \\
    --temperature=0.7 \\
    --request_time=0.0 \\
    --truncated_num=1000 \\
    --max_concurrent=1000 \\
    --max_retries=5
"""


def generate_train_script(model_dir_name, model_path, layer_index=28):
    """Generate train.sh script"""
    return f"""python3 run.py collect_non_refusal \\
    --json_path outputs/refusal/{model_dir_name}/ultrachat.json \\
    --save_path outputs/refusal/{model_dir_name}/no_refusal.json

python3 run.py collect_refusal \\
    --json_path outputs/refusal/{model_dir_name}/advbench.json \\
    --save_path outputs/refusal/{model_dir_name}/refusal.json

CUDA_VISIBLE_DEVICES=0 python3 run.py extract_hidden_states \\
    --model_path "{model_path}" \\
    --json_path outputs/refusal/{model_dir_name}/no_refusal.json \\
    --save_path outputs/refusal/{model_dir_name}/no_refusal.pt \\
    --layer_index {layer_index}

CUDA_VISIBLE_DEVICES=0 python3 run.py extract_hidden_states \\
    --model_path "{model_path}" \\
    --json_path outputs/refusal/{model_dir_name}/refusal.json \\
    --save_path outputs/refusal/{model_dir_name}/refusal.pt \\
    --layer_index {layer_index}

python3 run.py train_prober \\
    --or_path outputs/refusal/{model_dir_name}/no_refusal.pt \\
    --jbb_path outputs/refusal/{model_dir_name}/refusal.pt \\
    --epochs 5 \\
    --save_path outputs/refusal/{model_dir_name}/linear_prober.pt
"""


def generate_test_script(model_dir_name, model_path, layer_index=28):
    """Generate test.sh script"""
    return f"""CUDA_VISIBLE_DEVICES=0 python3 run.py test_prober \\
    --json_path outputs/refusal/{model_dir_name}/jbbench.json \\
    --ckpt_path outputs/refusal/{model_dir_name}/linear_prober.pt \\
    --model_path "{model_path}" \\
    --layer_index {layer_index} \\
    --max_items 1000 \\
    --thinking_portion 0 \\
    --item_type "original_item"
"""


def generate_vllm_script(model_path, port=8000, tensor_parallel_size=2):
    """Generate vllm.sh script"""
    return f"""# model serving
vllm serve "{model_path}" \\
    --port {port} \\
    --tensor-parallel-size {tensor_parallel_size}

"""


def generate_all_script(model_dir_name, model_path):
    """Generate all.sh script to run the complete pipeline"""
    return f"""#!/bin/bash

set -e

SCRIPT_DIR="$( cd "$( dirname "${{BASH_SOURCE[0]}}" )" &> /dev/null && pwd )"

bash "$SCRIPT_DIR/prepare.sh"
bash "$SCRIPT_DIR/inference.sh"
bash "$SCRIPT_DIR/train.sh"
bash "$SCRIPT_DIR/test.sh"

"""


def main():
    parser = argparse.ArgumentParser(description='Auto-generate prober scripts')
    parser.add_argument('--model_dir_name', required=True, 
                        help='Model directory name, e.g.: Qwen3-4B-thinking')
    parser.add_argument('--model_path', required=True,
                        help='HuggingFace model path, e.g.: Qwen/Qwen3-4B-Thinking-2507')
    parser.add_argument('--layer_index', type=int, default=28,
                        help='Layer index, default: 28')
    parser.add_argument('--port', type=int, default=8000,
                        help='vLLM server port, default: 8000')
    parser.add_argument('--tensor_parallel_size', type=int, default=2,
                        help='Tensor parallel size, default: 2')
    parser.add_argument('--output_dir', default='scripts/prober',
                        help='Output directory, default: scripts/prober')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir) / args.model_dir_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate all scripts
    scripts = {
        'prepare.sh': generate_prepare_script(args.model_dir_name),
        'inference.sh': generate_inference_script(args.model_dir_name, args.model_path),
        'train.sh': generate_train_script(args.model_dir_name, args.model_path, args.layer_index),
        'test.sh': generate_test_script(args.model_dir_name, args.model_path, args.layer_index),
        'vllm.sh': generate_vllm_script(args.model_path, args.port, args.tensor_parallel_size),
        'all.sh': generate_all_script(args.model_dir_name, args.model_path)
    }
    
    # Write files
    for script_name, content in scripts.items():
        script_path = output_path / script_name
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Set executable permissions
        os.chmod(script_path, 0o755)
        print(f"✅ Generated script: {script_path}")
    
    print(f"\n🎉 All scripts generated successfully! Output directory: {output_path}")
    print("\nUsage examples:")
    print(f"  cd {output_path}")
    print("  ./all.sh        # Run complete pipeline")
    print("  # OR run individually:")
    print("  ./prepare.sh    # Prepare data")
    print("  ./vllm.sh       # Start model serving")
    print("  ./inference.sh  # Run inference")
    print("  ./train.sh      # Train prober")
    print("  ./test.sh       # Test prober")


if __name__ == '__main__':
    main() 