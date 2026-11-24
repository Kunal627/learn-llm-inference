
import torch

# ============================================================
# 1. CREATE AN FP32 TENSOR
# ============================================================

x_fp32 = torch.tensor(
    [1.234567, 100000.0, 1e-8, 0.123456789],
    dtype=torch.float32
)

# ============================================================
# 2. CONVERT TO FP16 AND BF16
# ============================================================

x_fp16 = x_fp32.to(torch.float16)
x_bf16 = x_fp32.to(torch.bfloat16)

# ============================================================
# 3. DISPLAY VALUES
# ============================================================

print("=" * 65)
print("FLOATING-POINT REPRESENTATION")
print("=" * 65)

print("FP32:", x_fp32)
print("FP16:", x_fp16)
print("BF16:", x_bf16)

# Convert back to FP32 for easier comparison.
print("\nValues converted back to FP32:")

print("FP32:", x_fp32)
print("FP16:", x_fp16.float())
print("BF16:", x_bf16.float())

# ============================================================
# 4. MEMORY CONSUMPTION
# ============================================================

print("\n" + "=" * 65)
print("MEMORY CONSUMPTION")
print("=" * 65)

print("FP32 element size:", x_fp32.element_size(), "bytes")
print("FP16 element size:", x_fp16.element_size(), "bytes")
print("BF16 element size:", x_bf16.element_size(), "bytes")

print("FP32 tensor memory:", x_fp32.numel() * x_fp32.element_size())
print("FP16 tensor memory:", x_fp16.numel() * x_fp16.element_size())
print("BF16 tensor memory:", x_bf16.numel() * x_bf16.element_size())

# ============================================================
# 5. NUMERICAL ERROR
# ============================================================

print("\n" + "=" * 65)
print("NUMERICAL ERROR")
print("=" * 65)

fp16_error = (x_fp32 - x_fp16.float()).abs()
bf16_error = (x_fp32 - x_bf16.float()).abs()

print("FP16 absolute error:", fp16_error)
print("BF16 absolute error:", bf16_error)

# ============================================================
# 6. OVERFLOW AND UNDERFLOW TEST
# ============================================================

print("\n" + "=" * 65)
print("OVERFLOW AND UNDERFLOW")
print("=" * 65)

test_values = torch.tensor(
    [100000.0, 1e-8],
    dtype=torch.float32
)

print("Original FP32:", test_values)
print("Converted FP16:", test_values.to(torch.float16))
print("Converted BF16:", test_values.to(torch.bfloat16))

# ============================================================
# 7. VALIDATION
# ============================================================

assert x_fp32.element_size() == 4
assert x_fp16.element_size() == 2
assert x_bf16.element_size() == 2

assert torch.isinf(test_values.to(torch.float16)[0])
assert torch.isfinite(test_values.to(torch.bfloat16)[0])

assert test_values.to(torch.float16)[1] == 0
assert test_values.to(torch.bfloat16)[1] != 0

print("\nAll validation checks passed!")
