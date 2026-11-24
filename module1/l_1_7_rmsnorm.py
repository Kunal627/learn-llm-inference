import torch
import torch.nn as nn
import torch.nn.functional as F

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



