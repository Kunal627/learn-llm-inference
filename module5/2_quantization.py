
import torch

torch.manual_seed(42)

# ============================================================
# 1. CREATE FLOATING-POINT WEIGHTS
# ============================================================

weights = torch.tensor(
    [-1.0, -0.5, 0.0, 0.5, 1.0],
    dtype=torch.float32
)

print("Original weights:")
print(weights)

# ============================================================
# 2. CALCULATE QUANTIZATION SCALE
# ============================================================

qmax = 127

max_abs = weights.abs().max()

scale = max_abs / qmax

print("\nQuantization scale:", scale.item())

# ============================================================
# 3. QUANTIZE
# ============================================================

quantized = torch.round(weights / scale)

quantized = torch.clamp(
    quantized,
    min=-128,
    max=127
).to(torch.int8)

print("\nQuantized INT8 weights:")
print(quantized)

# ============================================================
# 4. DEQUANTIZE
# ============================================================

reconstructed = quantized.float() * scale

print("\nReconstructed weights:")
print(reconstructed)

# ============================================================
# 5. CALCULATE QUANTIZATION ERROR
# ============================================================

absolute_error = (weights - reconstructed).abs()

print("\nAbsolute quantization error:")
print(absolute_error)

print(
    "\nMaximum absolute error:",
    absolute_error.max().item()
)

print(
    "Mean absolute error:",
    absolute_error.mean().item()
)

# ============================================================
# 6. MEMORY COMPARISON
# ============================================================

print("\nMemory comparison:")

print(
    "FP32 memory:",
    weights.numel() * weights.element_size(),
    "bytes"
)

print(
    "INT8 memory:",
    quantized.numel() * quantized.element_size(),
    "bytes"
)

# ============================================================
# 7. VALIDATION
# ============================================================

assert quantized.dtype == torch.int8
assert reconstructed.shape == weights.shape
assert torch.allclose(
    weights,
    reconstructed,
    atol=0.01
)

print("\nAll validation checks passed!")
