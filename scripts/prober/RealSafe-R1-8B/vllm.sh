# model serving
vllm serve "RealSafe/RealSafe-R1-8B" \
    --port 8000 \
    --tensor-parallel-size 2

