
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

batch_size = 2
seq_len = 3
hidden_dim = 8
head_dim = 4

# --------------------------------------------------
# 2. Input tensor
# --------------------------------------------------

# [batch, sequence, hidden]
X = torch.randn(batch_size, seq_len, hidden_dim)

# --------------------------------------------------
# 3. Q, K, V projections
# --------------------------------------------------

q_proj = nn.Linear(hidden_dim, head_dim, bias=False)
k_proj = nn.Linear(hidden_dim, head_dim, bias=False)
v_proj = nn.Linear(hidden_dim, head_dim, bias=False)

Q = q_proj(X)
K = k_proj(X)
V = v_proj(X)

# Shapes: [B, S, head_dim]
print("Q:", Q.shape)
print("K:", K.shape)
print("V:", V.shape)

# --------------------------------------------------
# 4. Calculate attention scores: Q @ K transpose
# --------------------------------------------------

# K: [B, S, D]
# K.transpose(-2, -1): [B, D, S]
# Scores: [B, S, S]

scores = Q @ K.transpose(-2, -1)

print("\nRaw scores:", scores.shape)

# --------------------------------------------------
# 5. Scale scores
# --------------------------------------------------

scale = head_dim ** 0.5

scaled_scores = scores / scale

# --------------------------------------------------
# 6. Softmax: convert scores into probabilities
# --------------------------------------------------

attention_weights = F.softmax(
    scaled_scores,
    dim=-1
)

print("\nAttention weights:")
print(attention_weights)

# --------------------------------------------------
# 7. Aggregate values
# --------------------------------------------------

# [B, S, S] @ [B, S, D]
# Result: [B, S, D]

output = attention_weights @ V

print("\nOutput:", output.shape)
print(output)

# --------------------------------------------------
# 8. Validate
# --------------------------------------------------

assert scores.shape == (batch_size, seq_len, seq_len)
assert attention_weights.shape == (batch_size, seq_len, seq_len)
assert output.shape == (batch_size, seq_len, head_dim)

# Every attention row should sum to 1
row_sums = attention_weights.sum(dim=-1)

assert torch.allclose(
    row_sums,
    torch.ones_like(row_sums),
    atol=1e-6
)

print("\nAll validation checks passed!")



import matplotlib.pyplot as plt

# Display attention weights for the first batch
weights = attention_weights[0].detach().numpy()

tokens = ["The", "cat", "mouse"]

plt.figure(figsize=(6, 5))

plt.imshow(weights, cmap="Blues")

plt.xticks(range(len(tokens)), tokens)
plt.yticks(range(len(tokens)), tokens)

plt.xlabel("Key token")
plt.ylabel("Query token")
plt.title("Scaled Dot-Product Attention")

plt.colorbar(label="Attention weight")

for i in range(len(tokens)):
    for j in range(len(tokens)):
        plt.text(
            j, i,
            f"{weights[i, j]:.2f}",
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.show()



# --------------------------------------------------
# 9. Compare against PyTorch SDPA
# --------------------------------------------------

reference_output = F.scaled_dot_product_attention(
    Q,
    K,
    V,
    dropout_p=0.0,
    is_causal=False
)

difference = (output - reference_output).abs().max()

print("\nMaximum absolute difference:")
print(difference.item())

assert torch.allclose(
    output,
    reference_output,
    atol=1e-5,
    rtol=1e-5
)

print("Manual implementation matches PyTorch SDPA!")
