
import torch


def apply_rope(q, k, base=10000, position_offset=0):
    """
    Apply Rotary Positional Embeddings to Q and K.

    q, k: [B, H, S, D]
    Returns:
        q_rotated: [B, H, S, D]
        k_rotated: [B, H, S, D]
    """

    assert q.ndim == 4 and k.ndim == 4, "Q and K must have shape [B, H, S, D]"
    assert q.shape[0] == k.shape[0], "Q and K must have the same batch size"
    assert q.shape[2:] == k.shape[2:], "Q and K must have the same sequence and head dimensions"

    _, _, S, D = q.shape

    assert D % 2 == 0, "Head dimension must be even"

    # Step 1: Generate dimension-pair indices
    # [D/2]
    dim_indices = torch.arange(
        0, D, 2,
        device=q.device,
        dtype=torch.float32
    )

    # Step 2: Calculate inverse frequencies
    # [D/2]
    inv_freq = 1.0 / (base ** (dim_indices / D))

    # Step 3: Generate token positions
    # [S]
    positions = torch.arange(
        position_offset,
        position_offset + S,
        device=q.device,
        dtype=torch.float32
    )

    # Step 4: Calculate rotation angles
    # [S, D/2]
    angles = positions[:, None] * inv_freq[None, :]

    # Step 5: Calculate cos and sin
    # [S, D/2]
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    # Step 6: Broadcast for [B, H, S, D/2]
    # [1, 1, S, D/2]
    cos = cos[None, None, :, :].to(q.dtype)
    sin = sin[None, None, :, :].to(q.dtype)

    def rotate(x):
        # [B, H, S, D/2]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # Rotate each pair of dimensions
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        # Interleave even and odd dimensions
        # [B, H, S, D/2, 2]
        rotated = torch.stack(
            [rotated_even, rotated_odd],
            dim=-1
        )

        # [B, H, S, D]
        return rotated.flatten(-2)

    return rotate(q), rotate(k)
