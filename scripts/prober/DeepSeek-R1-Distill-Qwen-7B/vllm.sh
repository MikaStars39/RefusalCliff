# model serving
vllm serve "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    --port 8000 \
    --tensor-parallel-size 2

