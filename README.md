# LLM Inference Engineering

This repository is a hands-on learning course about how large language models generate text and how inference systems make that process faster and more memory-efficient. It follows the path from a prompt through Transformer computation and token generation, then moves into GPU performance and serving multiple requests.

The lessons are organized as small Python scripts so that concepts can be inspected, modified, and run independently. The examples include implementations and demonstrations, measurements, and experiments; they are intended for learning rather than as a production inference runtime.

## Learning Path

| Module | Topics | Guide |
| --- | --- | --- |
| 1. LLM architecture and inference | Tokenization, embeddings, RoPE, Transformer layers, logits, and decoding | [Module 1 README](module1/README.md) |
| 2. Attention mechanisms | Q/K/V projections, causal attention, MHA/GQA/MQA, and attention cost | [Module 2 README](module2/README.md) |
| 3. Prefill and autoregressive decode | Inference phases, latency metrics, KV caching, and variable-length prompts | [Module 3 README](module3/README.md) |
| 4. KV cache and GPU memory | Weight and cache memory, precision, prefix sharing, and cache allocation | [Module 4 README](module4/README.md) |
| 5. Quantization and model compression | Numeric formats, quantization methods, calibration, and quality/performance trade-offs | [Module 5 README](module5/README.md) |
| 6. GPU architecture and performance | GPU execution, memory hierarchy, roofline reasoning, streams, kernel fusion, and FlashAttention | [Module 6 README](module6/README.md) |
| 7. Batching and scheduling | Batching, padding, continuous batching, and static-versus-continuous scheduling | [Module 7 scripts](module7/) |

The suggested order is modules 1 through 7: each topic builds on the previous one, from model fundamentals to the scheduling choices used in inference serving.

## Repository Layout

- `module1/`–`module7/`: numbered lesson scripts, grouped by topic.
- `module2/src/`: shared attention and RoPE implementation code used by module 2.
- `module3/tiny_llm.py`: a small Transformer and generation helpers used by module 3 lessons.
- `LLM_Inference_Engineering_Course_Guideline.docx`: course guideline document.
- Root-level CSV and JSON files: benchmark results and GPU trace artifacts from the experiments.

## Running the Lessons

Most lessons are standalone scripts. From the repository root, run a script with Python, for example:

```powershell
python module1/l_1_3_tokenization.py
```

Python packages and hardware requirements vary by lesson. Read the relevant module guide and the script's imports before running it; some examples use PyTorch, model/tokenizer downloads, a CUDA-capable GPU, or a separately running local inference server. Module 5 documents the server and model expectations for its Qwen examples, and module 6 includes a Windows-specific note for the kernel-fusion benchmark.

Benchmark outputs depend on the machine, model, runtime, and configuration. Treat recorded results as experiment artifacts, not as performance guarantees for other systems.

## What You Will Learn

- How text becomes tokens and model representations, and how autoregressive decoding chooses the next token.
- How causal attention works, why its cost grows with sequence length, and how attention variants change KV-cache use.
- Why prefill and decode have different performance characteristics, and how to measure latency and throughput.
- How model weights, activations, and KV caches compete for accelerator memory.
- How precision and quantization affect memory, numerical error, and model output quality.
- How GPU execution, memory bandwidth, kernel launches, and batching influence inference performance.