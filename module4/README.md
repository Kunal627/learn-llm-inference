# Module 4: KV Cache and GPU Memory

Learn how model weights and KV caches use GPU memory, and how cache design affects serving capacity.

## Key concepts

- **Memory use:** Model weights hold the learned parameters, while runtime memory holds temporary activations and the KV cache. The cache stores attention keys and values for tokens already processed, and grows as requests get longer.
- **Cache size:** Cache memory grows with the number of layers, requests, cached tokens, and key/value heads. Larger head dimensions or more bytes per value also increase the total.
- **Serving capacity:** GPU memory places a limit on how many requests and tokens can be kept active at once. Longer contexts and more concurrent requests use more cache, while lower-precision formats use fewer bytes per value.
- **Sharing and allocation:** Requests with the same prompt prefix can reuse its cached state instead of storing duplicate copies. Paged allocation stores cache in blocks, reducing the unused space that can result when each request reserves its maximum context up front.

## Script guide

| Script | Concept covered |
| --- | --- |
| [1_model_weights.py](1_model_weights.py) | Estimate model-weight memory for different model sizes and precisions |
| [2_kv_cache.py](2_kv_cache.py) | Inspect KV-cache dimensions and growth as tokens are generated |
| [3_mha_gqa_mqa_kv_cache.py](3_mha_gqa_mqa_kv_cache.py) | Compare KV-cache memory for MHA, GQA, and MQA |
| [4_context_concurrency.py](4_context_concurrency.py) | Relate per-token cache size, context length, and concurrent requests |
| [5_precision_kv_cache.py](5_precision_kv_cache.py) | Explore KV-cache storage precision and memory use |
| [6_shared_prefix.py](6_shared_prefix.py) | Simulate memory savings from sharing common prompt prefixes |
| [7_kv_cache_allocation.py](7_kv_cache_allocation.py) | Compare fixed-size and paged cache allocation |

**Goal:** Estimate KV-cache memory and understand how it limits serving capacity.