# Module 1: LLM Architecture and Inference

Learn what happens from entering a prompt to receiving generated text.

## Key concepts

- **Tokens and embeddings:** Text is split into tokens, converted to IDs, then represented as vectors the model can process. A token can be a whole word or just part of one, so models work with token IDs rather than raw text.
- **Transformer blocks:** Attention lets each token gather useful information from earlier tokens. Normalization and feed-forward layers then refine the token representations, while RoPE supplies information about token positions.
- **Next-token choice:** The model produces a logit, or score, for every token in its vocabulary. Softmax turns scores into probabilities; greedy decoding picks the top choice, while temperature and top-k/top-p sampling control randomness.
- **Autoregressive generation:** The chosen token is added to the input, and the model predicts again using the updated sequence. Generation ends when it produces an end-of-sequence token or reaches a configured limit.

## Script guide

| Script | Concept covered |
| --- | --- |
| [l_1_3_tokenization.py](l_1_3_tokenization.py) | Convert prompt text to token IDs with a tokenizer |
| [l_1_4_embedding.py](l_1_4_embedding.py) | Map token IDs to embedding vectors |
| [l_1_5_rope.py](l_1_5_rope.py) | Add token position information with RoPE |
| [l_1_6_mha.py](l_1_6_mha.py) | Compute causal multi-head self-attention |
| [l_1_7_rmsnorm.py](l_1_7_rmsnorm.py) | Normalize hidden-state vectors with RMSNorm |
| [l_1_8_swiglu.py](l_1_8_swiglu.py) | Apply the SwiGLU feed-forward layer |
| [l_1_9_qwen.py](l_1_9_qwen.py) | Inspect a pretrained Qwen model's configuration |
| [l_1_10_logits.py](l_1_10_logits.py) | Convert next-token logits into probabilities |
| [l_1_11_decoding.py](l_1_11_decoding.py) | Select tokens with temperature and top-k sampling |

**Goal:** Understand the path from a prompt to its next generated token.