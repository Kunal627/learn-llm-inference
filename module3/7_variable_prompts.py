
import torch
import torch.nn as nn

from tiny_llm import last_valid_logits, pad_prompts, select_next_token

torch.manual_seed(42)

# ============================================================
# LESSON 3.5.1: VARIABLE-LENGTH PROMPTS
# ============================================================

# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

PAD_ID = 0
VOCAB_SIZE = 50

BATCH_SIZE = 3
SEQ_LEN = 5

D_MODEL = 8
NUM_HEADS = 2
HEAD_DIM = D_MODEL // NUM_HEADS

assert D_MODEL % NUM_HEADS == 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 65)
print("CONFIGURATION")
print("=" * 65)

print("Device:", device)
print("Batch size:", BATCH_SIZE)
print("Maximum sequence length:", SEQ_LEN)
print("Model dimension:", D_MODEL)
print("Number of heads:", NUM_HEADS)
print("Head dimension:", HEAD_DIM)


# ------------------------------------------------------------
# 2. VARIABLE-LENGTH PROMPTS
# ------------------------------------------------------------

prompts = [
    [11, 12, 13],          # Request A: 3 tokens
    [21, 22, 23, 24, 25],  # Request B: 5 tokens
    [31]                   # Request C: 1 token
]

input_ids, attention_mask, prompt_lengths = pad_prompts(
    prompts, PAD_ID, device, max_length=SEQ_LEN
)

print("\n" + "=" * 65)
print("ORIGINAL PROMPTS")
print("=" * 65)

for i, prompt in enumerate(prompts):
    print(f"Request {i}: {prompt}, length={len(prompt)}")


# ------------------------------------------------------------
# 3. RIGHT PADDING
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("PADDED INPUT")
print("=" * 65)

print("Input IDs:")
print(input_ids)
print("Shape:", input_ids.shape)

print("\nAttention mask:")
print(attention_mask)
print("Shape:", attention_mask.shape)

print("\nPrompt lengths:")
print(prompt_lengths)


# ------------------------------------------------------------
# 4. TOKEN EMBEDDINGS
# ------------------------------------------------------------

embedding = nn.Embedding(
    VOCAB_SIZE,
    D_MODEL,
    padding_idx=PAD_ID
).to(device)

x = embedding(input_ids)

print("\n" + "=" * 65)
print("TOKEN EMBEDDINGS")
print("=" * 65)

print("x shape:", x.shape)
# [B, S, D] = [3, 5, 8]


# ------------------------------------------------------------
# 5. Q, K, V PROJECTIONS
# ------------------------------------------------------------

qkv_layer = nn.Linear(
    D_MODEL,
    3 * D_MODEL
).to(device)

qkv = qkv_layer(x)

print("\nQKV combined shape:", qkv.shape)
# [3, 5, 24]

qkv = qkv.reshape(
    BATCH_SIZE,
    SEQ_LEN,
    3,
    NUM_HEADS,
    HEAD_DIM
)

print("Reshaped QKV:", qkv.shape)
# [3, 5, 3, 2, 4]

q, k, v = qkv.permute(2, 0, 3, 1, 4)

print("\nQ shape:", q.shape)
print("K shape:", k.shape)
print("V shape:", v.shape)
# [B, H, S, HEAD_DIM] = [3, 2, 5, 4]


# ------------------------------------------------------------
# 6. CAUSAL MASK
# ------------------------------------------------------------

causal_mask = torch.tril(
    torch.ones(
        SEQ_LEN,
        SEQ_LEN,
        dtype=torch.bool,
        device=device
    )
)

print("\n" + "=" * 65)
print("CAUSAL MASK")
print("=" * 65)

print(causal_mask.int())
print("Shape:", causal_mask.shape)


# ------------------------------------------------------------
# 7. PADDING MASK
# ------------------------------------------------------------

# [B, S] -> [B, 1, 1, S]

key_padding_mask = attention_mask[:, None, None, :].bool()

print("\n" + "=" * 65)
print("KEY PADDING MASK")
print("=" * 65)

print("Shape:", key_padding_mask.shape)

for i in range(BATCH_SIZE):
    print(f"Request {i}:")
    print(key_padding_mask[i, 0, 0].int())


# ------------------------------------------------------------
# 8. COMBINE CAUSAL AND PADDING MASKS
# ------------------------------------------------------------

# Causal mask:
# [S, S] -> [1, 1, S, S]

causal_mask_4d = causal_mask[None, None, :, :]

# Padding mask:
# [B, 1, 1, S]

# Combined:
# [B, 1, S, S]

combined_mask = causal_mask_4d & key_padding_mask

print("\n" + "=" * 65)
print("COMBINED ATTENTION MASK")
print("=" * 65)

print("Shape:", combined_mask.shape)

for i in range(BATCH_SIZE):
    print(f"\nRequest {i}:")
    print(combined_mask[i, 0].int())


# ------------------------------------------------------------
# 9. ATTENTION SCORES
# ------------------------------------------------------------

# Q: [B, H, S, HEAD_DIM]
# K transpose: [B, H, HEAD_DIM, S]
#
# Scores: [B, H, S, S]

scores = torch.matmul(
    q,
    k.transpose(-2, -1)
)

scores = scores / (HEAD_DIM ** 0.5)

print("\n" + "=" * 65)
print("ATTENTION SCORES")
print("=" * 65)

print("Scores shape:", scores.shape)


# ------------------------------------------------------------
# 10. APPLY ATTENTION MASK
# ------------------------------------------------------------

scores = scores.masked_fill(
    ~combined_mask,
    float("-inf")
)

print("\nMasked scores shape:", scores.shape)


# ------------------------------------------------------------
# 11. SOFTMAX
# ------------------------------------------------------------

attention_weights = torch.softmax(
    scores,
    dim=-1
)

print("\n" + "=" * 65)
print("ATTENTION WEIGHTS")
print("=" * 65)

print("Attention weights shape:", attention_weights.shape)

for i in range(BATCH_SIZE):
    print(f"\nRequest {i}, Head 0:")
    print(attention_weights[i, 0].detach().cpu())


# ------------------------------------------------------------
# 12. ATTENTION OUTPUT
# ------------------------------------------------------------

# Attention weights:
# [B, H, S, S]
#
# V:
# [B, H, S, HEAD_DIM]
#
# Context:
# [B, H, S, HEAD_DIM]

context = torch.matmul(
    attention_weights,
    v
)

print("\n" + "=" * 65)
print("ATTENTION OUTPUT")
print("=" * 65)

print("Context before merging heads:", context.shape)


# ------------------------------------------------------------
# 13. MERGE ATTENTION HEADS
# ------------------------------------------------------------

context = context.transpose(1, 2).contiguous()

context = context.reshape(
    BATCH_SIZE,
    SEQ_LEN,
    D_MODEL
)

print("Context after merging heads:", context.shape)
# [3, 5, 8]


# ------------------------------------------------------------
# 14. LANGUAGE MODEL HEAD
# ------------------------------------------------------------

lm_head = nn.Linear(
    D_MODEL,
    VOCAB_SIZE
).to(device)

logits = lm_head(context)

print("\n" + "=" * 65)
print("LANGUAGE MODEL LOGITS")
print("=" * 65)

print("Logits shape:", logits.shape)
# [B, S, VOCAB_SIZE] = [3, 5, 50]


# ------------------------------------------------------------
# 15. WRONG LAST-TOKEN SELECTION
# ------------------------------------------------------------

wrong_logits = logits[:, -1, :]

print("\n" + "=" * 65)
print("WRONG LAST-TOKEN SELECTION")
print("=" * 65)

print("Wrong logits shape:", wrong_logits.shape)

# This selects position 4 for every request.
# Requests A and C have padding at position 4.


# ------------------------------------------------------------
# 16. CORRECT LAST-VALID-TOKEN SELECTION
# ------------------------------------------------------------

last_valid_indices = prompt_lengths - 1

correct_logits = last_valid_logits(logits, prompt_lengths)

print("\n" + "=" * 65)
print("CORRECT LAST-TOKEN SELECTION")
print("=" * 65)

print("Last valid indices:", last_valid_indices)
print("Correct logits shape:", correct_logits.shape)
# [B, VOCAB_SIZE] = [3, 50]


# ------------------------------------------------------------
# 17. NEXT-TOKEN PREDICTION
# ------------------------------------------------------------

next_tokens = select_next_token(correct_logits).squeeze(-1)

print("\n" + "=" * 65)
print("NEXT TOKEN PREDICTIONS")
print("=" * 65)

for i in range(BATCH_SIZE):

    print(
        f"Request {i}: "
        f"prompt length={prompt_lengths[i].item()}, "
        f"last valid position={last_valid_indices[i].item()}, "
        f"predicted token={next_tokens[i].item()}"
    )


# ------------------------------------------------------------
# 18. CORRECTNESS CHECKS
# ------------------------------------------------------------

assert input_ids.shape == (3, 5)
assert attention_mask.shape == (3, 5)

assert q.shape == (3, 2, 5, 4)
assert k.shape == (3, 2, 5, 4)
assert v.shape == (3, 2, 5, 4)

assert combined_mask.shape == (3, 1, 5, 5)
assert logits.shape == (3, 5, VOCAB_SIZE)

assert last_valid_indices.tolist() == [2, 4, 0]
assert correct_logits.shape == (3, VOCAB_SIZE)

# Every valid query position must have at least
# one allowed key position.
assert combined_mask.any(dim=-1).all()

# No padded key position should be allowed.
for i in range(BATCH_SIZE):
    valid_length = prompt_lengths[i].item()

    assert not combined_mask[
        i, 0, :, valid_length:
    ].any()

print("\n" + "=" * 65)
print("ALL CORRECTNESS CHECKS PASSED")
print("=" * 65)
