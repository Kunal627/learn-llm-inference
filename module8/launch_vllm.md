vllm serve Qwen/Qwen3-0.6B \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.85 \
    --host 0.0.0.0 \
    --port 8000