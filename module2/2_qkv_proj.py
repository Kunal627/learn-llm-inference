
import torch
import torch.nn as nn

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

batch_size = 2
seq_len = 5
hidden_dim = 8
head_dim = 4

torch.manual_seed(42)

# --------------------------------------------------
# 2. Create input tensor
# --------------------------------------------------

# Shape: [batch, sequence, hidden]
X = torch.randn(batch_size, seq_len, hidden_dim)

print("Input shape:", X.shape)

# --------------------------------------------------
# 3. Define independent Q, K and V projections
# --------------------------------------------------

W_Q = nn.Linear(hidden_dim, head_dim, bias=False)
W_K = nn.Linear(hidden_dim, head_dim, bias=False)
W_V = nn.Linear(hidden_dim, head_dim, bias=False)

# --------------------------------------------------
# 4. Apply projections
# --------------------------------------------------

Q = W_Q(X)
K = W_K(X)
V = W_V(X)

# --------------------------------------------------
# 5. Inspect shapes
# --------------------------------------------------

print("\nProjection output shapes:")
print("Q:", Q.shape)
print("K:", K.shape)
print("V:", V.shape)

# --------------------------------------------------
# 6. Validate output dimensions
# --------------------------------------------------

expected_shape = (batch_size, seq_len, head_dim)

assert Q.shape == expected_shape
assert K.shape == expected_shape
assert V.shape == expected_shape

# --------------------------------------------------
# 7. Inspect learned parameters
# --------------------------------------------------

print("\nProjection weight shapes:")
print("W_Q:", W_Q.weight.shape)
print("W_K:", W_K.weight.shape)
print("W_V:", W_V.weight.shape)

print("\nParameter counts:")
print("Q:", sum(p.numel() for p in W_Q.parameters()))
print("K:", sum(p.numel() for p in W_K.parameters()))
print("V:", sum(p.numel() for p in W_V.parameters()))

# --------------------------------------------------
# 8. Inspect one token's representations
# --------------------------------------------------

token_index = 0

print("\nFirst token representations, batch 0:")
print("Input:", X[0, token_index])
print("Query:", Q[0, token_index])
print("Key:  ", K[0, token_index])
print("Value:", V[0, token_index])
