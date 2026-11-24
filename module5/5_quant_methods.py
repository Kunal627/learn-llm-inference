
import torch

torch.manual_seed(42)

# --------------------------------------------------
# 1. Create a sample linear-layer weight matrix
# --------------------------------------------------

OUT_FEATURES = 4
IN_FEATURES = 16

W = torch.randn(OUT_FEATURES, IN_FEATURES)

# Give each output channel a different range.
W[0] *= 0.1
W[1] *= 1.0
W[2] *= 2.0
W[3] *= 5.0

print("Original weight shape:", W.shape)
print("\nOriginal weights:\n", W)


# --------------------------------------------------
# 2. Metrics
# --------------------------------------------------

def calculate_error(original, reconstructed):
    error = original - reconstructed

    mae = error.abs().mean().item()
    rmse = torch.sqrt((error ** 2).mean()).item()
    max_error = error.abs().max().item()

    return mae, rmse, max_error


# --------------------------------------------------
# 3. Symmetric per-tensor quantization
# --------------------------------------------------

def symmetric_per_tensor(W, bits=8):

    qmax = (2 ** (bits - 1)) - 1

    scale = W.abs().max() / qmax
    scale = scale.clamp(min=1e-8)

    q = torch.round(W / scale)
    q = torch.clamp(q, -qmax, qmax)

    W_hat = q * scale

    return q, W_hat, scale


# --------------------------------------------------
# 4. Symmetric per-channel quantization
# --------------------------------------------------

def symmetric_per_channel(W, bits=8):

    qmax = (2 ** (bits - 1)) - 1

    # One scale per output channel (row).
    scale = W.abs().amax(dim=1, keepdim=True) / qmax
    scale = scale.clamp(min=1e-8)

    q = torch.round(W / scale)
    q = torch.clamp(q, -qmax, qmax)

    W_hat = q * scale

    return q, W_hat, scale


# --------------------------------------------------
# 5. Symmetric group-wise quantization
# --------------------------------------------------

def symmetric_groupwise(W, bits=4, group_size=4):

    qmax = (2 ** (bits - 1)) - 1

    # Each output channel is split into groups.
    rows, cols = W.shape

    assert cols % group_size == 0

    grouped = W.reshape(
        rows,
        cols // group_size,
        group_size
    )

    # One scale for each group.
    scale = grouped.abs().amax(
        dim=2,
        keepdim=True
    ) / qmax

    scale = scale.clamp(min=1e-8)

    q = torch.round(grouped / scale)
    q = torch.clamp(q, -qmax, qmax)

    W_hat = q * scale

    return (
        q.reshape_as(W),
        W_hat.reshape_as(W),
        scale
    )


# --------------------------------------------------
# 6. Run the quantization experiments
# --------------------------------------------------

q_tensor, W_tensor, scale_tensor = symmetric_per_tensor(
    W, bits=8
)

q_channel, W_channel, scale_channel = symmetric_per_channel(
    W, bits=8
)

q_group, W_group, scale_group = symmetric_groupwise(
    W, bits=4, group_size=4
)


# --------------------------------------------------
# 7. Compare reconstruction errors
# --------------------------------------------------

results = {
    "FP32 baseline": W,
    "INT8 per-tensor": W_tensor,
    "INT8 per-channel": W_channel,
    "INT4 group-wise": W_group
}

print("\n" + "=" * 65)
print("QUANTIZATION ERROR COMPARISON")
print("=" * 65)

for name, reconstructed in results.items():

    mae, rmse, max_error = calculate_error(
        W, reconstructed
    )

    print(f"\n{name}")
    print(f"MAE:       {mae:.8f}")
    print(f"RMSE:      {rmse:.8f}")
    print(f"Max error: {max_error:.8f}")


# --------------------------------------------------
# 8. Inspect scale shapes
# --------------------------------------------------

print("\n" + "=" * 65)
print("SCALE SHAPES")
print("=" * 65)

print("Per-tensor scale:", scale_tensor.shape)
print("Per-channel scales:", scale_channel.shape)
print("Group-wise scales:", scale_group.shape)

print("\nPer-channel scales:\n", scale_channel.flatten())
print("\nGroup-wise scales:\n", scale_group.squeeze(-1))


# --------------------------------------------------
# 9. Verify tensor shapes and quantization ranges
# --------------------------------------------------

assert W_tensor.shape == W.shape
assert W_channel.shape == W.shape
assert W_group.shape == W.shape

assert q_tensor.min() >= -127
assert q_tensor.max() <= 127

assert q_channel.min() >= -127
assert q_channel.max() <= 127

assert q_group.min() >= -7
assert q_group.max() <= 7

assert scale_tensor.numel() == 1
assert scale_channel.shape == (OUT_FEATURES, 1)
assert scale_group.shape == (
    OUT_FEATURES,
    IN_FEATURES // 4,
    1
)

print("\nAll shape and range checks passed.")
