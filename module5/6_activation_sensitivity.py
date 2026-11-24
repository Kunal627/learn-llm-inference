
import torch
import torch.nn.functional as F

torch.manual_seed(42)

# ---------------------------------------------
# 1. Create a sample linear layer
# ---------------------------------------------

BATCH = 512
IN_FEATURES = 8
OUT_FEATURES = 4

W = torch.randn(OUT_FEATURES, IN_FEATURES)

# Calibration activations with different channel magnitudes
X = torch.randn(BATCH, IN_FEATURES)

X[:, 0] *= 0.05
X[:, 1] *= 0.10
X[:, 2] *= 5.00
X[:, 3] *= 0.20
X[:, 4] *= 2.00

print("X shape:", X.shape)
print("W shape:", W.shape)


# ---------------------------------------------
# 2. Collect activation statistics
# ---------------------------------------------

activation_importance = X.abs().mean(dim=0)

print("\nMean absolute activation per channel:")
for i, value in enumerate(activation_importance):
    print(f"Channel {i}: {value.item():.4f}")


# ---------------------------------------------
# 3. Define simple INT4 quantization
# ---------------------------------------------

def quantize_int4_per_tensor(W):

    qmax = 7

    scale = W.abs().max() / qmax
    scale = scale.clamp(min=1e-8)

    q = torch.round(W / scale).clamp(-7, 7)

    return q * scale


# ---------------------------------------------
# 4. Quantize weights
# ---------------------------------------------

W_quantized = quantize_int4_per_tensor(W)

# Original and quantized layer outputs
Y_original = F.linear(X, W)

Y_quantized = F.linear(X, W_quantized)


# ---------------------------------------------
# 5. Compare weight reconstruction error
# ---------------------------------------------

weight_error = (W - W_quantized).abs()

print("\nWeight reconstruction error:")
print("MAE:", weight_error.mean().item())
print("RMSE:", weight_error.square().mean().sqrt().item())


# ---------------------------------------------
# 6. Compare output reconstruction error
# ---------------------------------------------

output_error = Y_original - Y_quantized

print("\nOutput reconstruction error:")
print("MAE:", output_error.abs().mean().item())
print("RMSE:", output_error.square().mean().sqrt().item())


# ---------------------------------------------
# 7. Estimate channel-wise sensitivity
# ---------------------------------------------

# Weight error for each input channel
channel_weight_error = weight_error.mean(dim=0)

# A simple activation-weighted error indicator
channel_sensitivity = (
    activation_importance * channel_weight_error
)

print("\nChannel-wise analysis:")

for i in range(IN_FEATURES):
    print(
        f"Channel {i}: "
        f"activation={activation_importance[i]:.4f}, "
        f"weight_error={channel_weight_error[i]:.4f}, "
        f"weighted_indicator={channel_sensitivity[i]:.4f}"
    )


# ---------------------------------------------
# 8. Validate shapes
# ---------------------------------------------

assert Y_original.shape == (BATCH, OUT_FEATURES)
assert Y_quantized.shape == (BATCH, OUT_FEATURES)

assert W_quantized.min() >= -7 * (
    W.abs().max() / 7
) - 1e-6

print("\nAll shape checks passed.")
