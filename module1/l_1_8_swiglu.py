import torch
import torch.nn as nn
import torch.nn.functional as F

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