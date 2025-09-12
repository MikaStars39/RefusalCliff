# model serving
vllm serve "Qwen/Qwen3-4B-Thinking-2507" \
    --port 8000 \
    --tensor-parallel-size 2
