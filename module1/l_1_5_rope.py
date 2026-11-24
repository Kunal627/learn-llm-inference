
import torch
import math

def rotate_half(x):
    """
    Split the last dimension into two halves
    and rotate them.
    """
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]

    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, base=10000):
    """
    q, k shape:
    [batch, sequence, heads, head_dim]
    """

    batch, seq_len, heads, head_dim = q.shape

    assert head_dim % 2 == 0

    # Frequency for each dimension pair
    inv_freq = 1.0 / (
        base ** (
            torch.arange(0, head_dim, 2).float()
            / head_dim
        )
    )

    # Token positions: 0, 1, 2, ...
    positions = torch.arange(seq_len).float()

    # Position-dependent angles
    freqs = torch.outer(positions, inv_freq)

    # Expand frequencies to head_dim
    emb = torch.cat([freqs, freqs], dim=-1)

    cos = emb.cos()[None, :, None, :]
    sin = emb.sin()[None, :, None, :]

    # Rotate Query and Key
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin

    return q_rot, k_rot


# Simulated Query and Key tensors
q = torch.randn(1, 4, 2, 8)
k = torch.randn(1, 4, 2, 8)

q_rot, k_rot = apply_rope(q, k)

print("Original Q:", q.shape)
print("Rotated Q:", q_rot.shape)

print("Original K:", k.shape)
print("Rotated K:", k_rot.shape)

print("Q changed:", not torch.allclose(q, q_rot))
print("K changed:", not torch.allclose(k, k_rot))
