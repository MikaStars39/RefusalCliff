# model serving
vllm serve "Skywork/Skywork-OR1-7B" \
    --port 8000 \
    --tensor-parallel-size 2

