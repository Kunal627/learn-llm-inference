# Module 3: Prefill and Autoregressive Decode

Learn the two phases of inference and how each affects response speed.

## Key concepts

- **Prefill and decode:** Prefill processes all prompt tokens together and prepares the context for generation. Decode produces one token at a time, so each new token depends on the previous one.
- **Latency:** TTFT is the wait until the first generated token, and ITL is the time between generated tokens. TPOT is the average decode time per output token; together with prompt and output lengths, these metrics help explain perceived speed.
- **KV cache:** Attention creates key and value tensors for each token. Keeping them in a cache lets decode reuse the prompt and previous-token results instead of recomputing them at every step.
- **Variable prompts:** Prompts in a batch may have different lengths, so shorter ones are padded to a common size. A mask marks padding, and the model must use each prompt's last real token when choosing its next token.

## Script guide

| Script | Concept covered |
| --- | --- |
| [1_prefill.py](1_prefill.py) | Run and benchmark prompt prefill |
| [2_autoregressive_decode.py](2_autoregressive_decode.py) | Generate tokens step by step and benchmark decode |
| [3_prefill_decode_benchmark.py](3_prefill_decode_benchmark.py) | Compare prompt and output lengths using separate latency measurements |
| [4_latency_metrics.py](4_latency_metrics.py) | Measure latency percentiles and token throughput |
| [5_kv_cache.py](5_kv_cache.py) | Implement and inspect a KV cache during generation |
| [6_naive_vs_kv.py](6_naive_vs_kv.py) | Compare decoding with and without a KV cache |
| [7_variable_prompts.py](7_variable_prompts.py) | Pad different-length prompts and retrieve their last valid logits |
| [test_variable_prompts.py](test_variable_prompts.py) | Check variable-prompt padding and input validation |
| [tiny_llm.py](tiny_llm.py) | Shared small Transformer model and generation helpers used by the lessons |

**Goal:** Understand inference phases and measure how they affect response speed.