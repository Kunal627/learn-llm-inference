# Module 2: Attention Mechanisms

Explore how a Transformer uses attention to relate tokens and compute its next representation.

## Key concepts

- **Q, K, and V:** A query represents what a token is looking for, while keys represent what tokens offer. Their similarity produces attention weights, which are used to combine the corresponding value vectors.
- **Causal attention:** Scaled dot-product attention compares queries and keys, scales the scores, and applies softmax to obtain weights. A causal mask blocks future positions so generation only uses information already available.
- **Attention variants:** MHA gives each query head its own key and value heads; MQA shares one pair, and GQA shares key/value heads across groups of query heads. Sharing reduces cache size and memory traffic, with a different head layout than standard MHA.
- **Cost:** For a sequence of length $S$, the attention score matrix has $S \times S$ entries. This quadratic growth makes long sequences more expensive in both computation and memory.

## Script guide

| Script | Concept covered |
| --- | --- |
| [1_attn_wts.py](1_attn_wts.py) | Interpret attention weights and weighted values |
| [2_qkv_proj.py](2_qkv_proj.py) | Project hidden states into query, key, and value tensors |
| [3_qwen_proj_shapes.py](3_qwen_proj_shapes.py) | Inspect Qwen-style grouped-query projection shapes |
| [4_scaled_dot_prod.py](4_scaled_dot_prod.py) | Calculate scaled dot-product attention |
| [5_causal_mask.py](5_causal_mask.py) | Apply a causal mask to prevent looking ahead |
| [6_attention_arch.py](6_attention_arch.py) | Compare MHA, GQA, and MQA architectures |
| [7_rope.py](7_rope.py) | Apply rotary position information to queries and keys |
| [8_attn_mem.py](8_attn_mem.py) | Estimate memory used by attention scores |
| [9_comp_mha_sdpa.py](9_comp_mha_sdpa.py) | Compare manual attention with PyTorch scaled dot-product attention |
| [10_complete_attn.py](10_complete_attn.py) | Run a complete attention layer with MHA, GQA, and MQA |

**Goal:** Understand how attention computes contextual representations and how its variants differ.