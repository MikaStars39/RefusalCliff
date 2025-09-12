# model serving
vllm serve "Qwen/QwQ-32B" \
    --port 8000 \
    --tensor-parallel-size 2
