# model serving
vllm serve "RealSafe/RealSafe-R1-7B" \
    --port 8000 \
    --tensor-parallel-size 2

