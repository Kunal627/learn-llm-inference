
import torch
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

B, H, S, D = 1, 4, 1024, 64

q = torch.randn(B, H, S, D, device=device)
k = torch.randn(B, H, S, D, device=device)
v = torch.randn(B, H, S, D, device=device)

if device == "cuda":
    q = q.to(torch.bfloat16)
    k = k.to(torch.bfloat16)
    v = v.to(torch.bfloat16)

print("Device:", device)
print("Q shape:", q.shape)

# ---------------------------------
# Manual attention
# ---------------------------------

scores = q @ k.transpose(-2, -1)
scores = scores / (D ** 0.5)

weights = torch.softmax(scores, dim=-1)

manual_output = weights @ v

print("\nManual attention")
print("Scores:", scores.shape)
print("Weights:", weights.shape)
print("Output:", manual_output.shape)

# ---------------------------------
# PyTorch SDPA
# ---------------------------------

sdpa_output = F.scaled_dot_product_attention(
    q, k, v,
    is_causal=True
)

print("\nPyTorch SDPA")
print("Output:", sdpa_output.shape)

print("\nOutput finite:", torch.isfinite(sdpa_output).all().item())
