
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

batch_size = 2
seq_len = 5
hidden_dim = 8
head_dim = 4

# --------------------------------------------------
# 2. Input and projections
# --------------------------------------------------

X = torch.randn(batch_size, seq_len, hidden_dim)

q_proj = nn.Linear(hidden_dim, head_dim, bias=False)
k_proj = nn.Linear(hidden_dim, head_dim, bias=False)
v_proj = nn.Linear(hidden_dim, head_dim, bias=False)

Q = q_proj(X)
K = k_proj(X)
V = v_proj(X)

# --------------------------------------------------
# 3. Calculate raw attention scores
# --------------------------------------------------

scores = Q @ K.transpose(-2, -1)

scores = scores / (head_dim ** 0.5)

print("Scores shape:", scores.shape)

# --------------------------------------------------
# 4. Create causal mask
# --------------------------------------------------

# True means this position is allowed.
# False means this position is blocked.

causal_mask = torch.tril(
    torch.ones(seq_len, seq_len, dtype=torch.bool)
)

print("\nCausal mask:")
print(causal_mask.int())

# --------------------------------------------------
# 5. Apply mask
# --------------------------------------------------

# Replace future-token scores with -infinity.

masked_scores = scores.masked_fill(
    ~causal_mask,
    float("-inf")
)

# --------------------------------------------------
# 6. Apply Softmax
# --------------------------------------------------

attention_weights = F.softmax(
    masked_scores,
    dim=-1
)

print("\nAttention weights:")
print(attention_weights[0])

# --------------------------------------------------
# 7. Aggregate values
# --------------------------------------------------

output = attention_weights @ V

print("\nOutput shape:", output.shape)

# --------------------------------------------------
# 8. Validate causal masking
# --------------------------------------------------

# Upper triangular positions must have zero weight.

future_mask = torch.triu(
    torch.ones(seq_len, seq_len, dtype=torch.bool),
    diagonal=1
)

assert torch.all(
    attention_weights[:, future_mask] == 0
)

# Every row must sum to one.

assert torch.allclose(
    attention_weights.sum(dim=-1),
    torch.ones(batch_size, seq_len),
    atol=1e-6
)

print("\nAll causal masking checks passed!")



import torch
import torch.nn.functional as F

batch_size = 2
seq_len = 5
head_dim = 4

Q = torch.randn(batch_size, 1, seq_len, head_dim)
K = torch.randn(batch_size, 1, seq_len, head_dim)
V = torch.randn(batch_size, 1, seq_len, head_dim)

output = F.scaled_dot_product_attention(
    Q,
    K,
    V,
    dropout_p=0.0,
    is_causal=True
)

print("Output shape:", output.shape)
