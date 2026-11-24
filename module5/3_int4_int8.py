
import torch

torch.manual_seed(42)

# ============================================================
# 1. CREATE A RANDOM WEIGHT MATRIX
# ============================================================

weights = torch.randn(1000, 1000, dtype=torch.float32)

# ============================================================
# 2. GENERAL SYMMETRIC QUANTIZATION FUNCTION
# ============================================================

def quantize_symmetric(weights, bits):

    qmax = (2 ** (bits - 1)) - 1

    scale = weights.abs().max() / qmax

    quantized = torch.round(weights / scale)

    quantized = torch.clamp(
        quantized,
        -qmax,
        qmax
    )

    # Simulate INT4 using INT8 storage.
    quantized = quantized.to(torch.int8)

    return quantized, scale


# ============================================================
# 3. DEQUANTIZATION
# ============================================================

def dequantize(quantized, scale):

    return quantized.float() * scale


# ============================================================
# 4. INT8 QUANTIZATION
# ============================================================

q8, scale8 = quantize_symmetric(weights, bits=8)

w8 = dequantize(q8, scale8)

# ============================================================
# 5. INT4 QUANTIZATION
# ============================================================

q4, scale4 = quantize_symmetric(weights, bits=4)

w4 = dequantize(q4, scale4)

# ============================================================
# 6. ERROR COMPARISON
# ============================================================

def calculate_metrics(original, reconstructed):

    error = original - reconstructed

    return {
        "MAE": error.abs().mean().item(),
        "Max absolute error": error.abs().max().item(),
        "RMSE": error.square().mean().sqrt().item(),
    }


print("=" * 65)
print("QUANTIZATION ERROR COMPARISON")
print("=" * 65)

print("INT8:", calculate_metrics(weights, w8))
print("INT4:", calculate_metrics(weights, w4))

# ============================================================
# 7. THEORETICAL STORAGE COMPARISON
# ============================================================

num_weights = weights.numel()

fp32_bytes = num_weights * 4
int8_bytes = num_weights * 1
int4_bytes = num_weights * 0.5

print("\n" + "=" * 65)
print("THEORETICAL STORAGE")
print("=" * 65)

print(f"FP32: {fp32_bytes / 1024**2:.2f} MiB")
print(f"INT8: {int8_bytes / 1024**2:.2f} MiB")
print(f"INT4: {int4_bytes / 1024**2:.2f} MiB")

# Actual simulated tensor storage.
print("\nActual PyTorch tensor storage:")
print(f"FP32: {weights.numel() * weights.element_size() / 1024**2:.2f} MiB")
print(f"INT8 simulation of INT4: {q4.numel() * q4.element_size() / 1024**2:.2f} MiB")

# ============================================================
# 8. VALIDATION
# ============================================================

assert q8.min() >= -127
assert q8.max() <= 127

assert q4.min() >= -7
assert q4.max() <= 7

assert q8.dtype == torch.int8
assert q4.dtype == torch.int8

assert w8.shape == weights.shape
assert w4.shape == weights.shape

assert calculate_metrics(weights, w4)["MAE"] > calculate_metrics(weights, w8)["MAE"]

print("\nAll validation checks passed!")
