# model serving
vllm serve "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --port 8000 \
    --tensor-parallel-size 2

