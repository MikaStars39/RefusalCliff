# model serving
vllm serve "microsoft/Phi-4-mini-reasoning" \
    --port 8000 \
    --tensor-parallel-size 2

