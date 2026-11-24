
import torch
import torch.nn as nn

# Qwen3-8B attention dimensions
hidden_dim = 4096
num_q_heads = 32
num_kv_heads = 8
head_dim = 128

batch_size = 2
seq_len = 10

# Input hidden states
X = torch.randn(batch_size, seq_len, hidden_dim)

# GQA projections
q_proj = nn.Linear(
    hidden_dim,
    num_q_heads * head_dim,
    bias=False
)

k_proj = nn.Linear(
    hidden_dim,
    num_kv_heads * head_dim,
    bias=False
)

v_proj = nn.Linear(
    hidden_dim,
    num_kv_heads * head_dim,
    bias=False
)

# Project
Q = q_proj(X)
K = k_proj(X)
V = v_proj(X)

# Reshape into attention heads
Q = Q.view(
    batch_size, seq_len, num_q_heads, head_dim
)

K = K.view(
    batch_size, seq_len, num_kv_heads, head_dim
)

V = V.view(
    batch_size, seq_len, num_kv_heads, head_dim
)

print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)

assert Q.shape == (2, 10, 32, 128)
assert K.shape == (2, 10, 8, 128)
assert V.shape == (2, 10, 8, 128)
