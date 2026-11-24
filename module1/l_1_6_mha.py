
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):

    def __init__(self, hidden_dim, num_heads):
        super().__init__()

        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Query, Key and Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):

        batch, seq_len, hidden = x.shape

        # Project input into Q, K and V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Split into attention heads
        q = q.view(
            batch, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        k = k.view(
            batch, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1))

        scores = scores / math.sqrt(self.head_dim)

        # Causal mask: block future positions
        mask = torch.triu(
            torch.ones(
                seq_len, seq_len,
                device=x.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        scores = scores.masked_fill(mask, float("-inf"))

        # Convert scores to probabilities
        weights = F.softmax(scores, dim=-1)

        # Weighted sum of Value vectors
        output = torch.matmul(weights, v)

        # Combine attention heads
        output = output.transpose(1, 2).contiguous()

        output = output.view(
            batch, seq_len, self.hidden_dim
        )

        # Final projection
        return self.o_proj(output)


# Test the attention layer
x = torch.randn(2, 5, 8)

attention = CausalSelfAttention(
    hidden_dim=8,
    num_heads=2
)

output = attention(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)


class RMSNorm(nn.Module):

    def __init__(self, hidden_dim, eps=1e-6):
        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(
            torch.ones(hidden_dim)
        )

    def forward(self, x):

        # Calculate mean square along hidden dimension
        rms = torch.sqrt(
            x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )

        # Normalize and apply learned scaling
        return (x / rms) * self.weight


# Test RMSNorm
norm = RMSNorm(hidden_dim=8)

x = torch.randn(2, 5, 8)

normalized = norm(x)

print("Input shape:", x.shape)
print("Output shape:", normalized.shape)
print("Learnable parameters:", norm.weight.shape)

class SwiGLU(nn.Module):

    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()

        self.gate_proj = nn.Linear(
            hidden_dim, intermediate_dim, bias=False
        )

        self.up_proj = nn.Linear(
            hidden_dim, intermediate_dim, bias=False
        )

        self.down_proj = nn.Linear(
            intermediate_dim, hidden_dim, bias=False
        )

    def forward(self, x):

        gate = F.silu(self.gate_proj(x))

        up = self.up_proj(x)

        hidden = gate * up

        return self.down_proj(hidden)


# Test SwiGLU
mlp = SwiGLU(
    hidden_dim=8,
    intermediate_dim=32
)

x = torch.randn(2, 5, 8)

output = mlp(x)

print("Input shape:", x.shape)
print("MLP output shape:", output.shape)




class TransformerBlock(nn.Module):

    def __init__(
        self,
        hidden_dim,
        num_heads,
        intermediate_dim
    ):
        super().__init__()

        self.attention_norm = RMSNorm(hidden_dim)

        self.attention = CausalSelfAttention(
            hidden_dim,
            num_heads
        )

        self.mlp_norm = RMSNorm(hidden_dim)

        self.mlp = SwiGLU(
            hidden_dim,
            intermediate_dim
        )

    def forward(self, x):

        # Attention + residual connection
        x = x + self.attention(
            self.attention_norm(x)
        )

        # MLP + residual connection
        x = x + self.mlp(
            self.mlp_norm(x)
        )

        return x


# Create a small Transformer block
block = TransformerBlock(
    hidden_dim=8,
    num_heads=2,
    intermediate_dim=32
)

x = torch.randn(2, 5, 8)

output = block(x)

print("Input:", x.shape)
print("Output:", output.shape)

print(
    "Total parameters:",
    sum(p.numel() for p in block.parameters())
)
