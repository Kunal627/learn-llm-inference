
import torch

torch.manual_seed(42)

# --------------------------------------------------
# 1. Create representative activation distributions
# --------------------------------------------------

CALIBRATION_SIZE = 10000
TEST_SIZE = 5000

# Actual workload:
# Most values are relatively small, but there are
# occasional larger values.
X_test = torch.randn(TEST_SIZE) * 1.5

# Calibration A represents the workload reasonably well.
X_calib_A = torch.randn(CALIBRATION_SIZE) * 1.5

# Calibration B has a much narrower distribution.
X_calib_B = torch.randn(CALIBRATION_SIZE) * 0.4

print("Test shape:", X_test.shape)
print("Calibration A shape:", X_calib_A.shape)
print("Calibration B shape:", X_calib_B.shape)


# --------------------------------------------------
# 2. Symmetric INT8 activation quantization
# --------------------------------------------------

def calibrate_symmetric_int8(x):

    qmax = 127

    scale = x.abs().max() / qmax
    scale = scale.clamp(min=1e-8)

    return scale


def quantize_dequantize_int8(x, scale):

    qmax = 127

    q = torch.round(x / scale)
    q = torch.clamp(q, -qmax, qmax)

    x_hat = q * scale

    return q, x_hat


# --------------------------------------------------
# 3. Calibrate using each dataset
# --------------------------------------------------

scale_A = calibrate_symmetric_int8(X_calib_A)
scale_B = calibrate_symmetric_int8(X_calib_B)

print("\nCalibration scales:")
print("Scale A:", scale_A.item())
print("Scale B:", scale_B.item())


# --------------------------------------------------
# 4. Quantize the same test data
# --------------------------------------------------

q_A, X_hat_A = quantize_dequantize_int8(
    X_test, scale_A
)

q_B, X_hat_B = quantize_dequantize_int8(
    X_test, scale_B
)


# --------------------------------------------------
# 5. Calculate reconstruction metrics
# --------------------------------------------------

def metrics(original, reconstructed):

    error = original - reconstructed

    mae = error.abs().mean()
    rmse = error.square().mean().sqrt()

    # Fraction of values that hit either clipping boundary.
    # This is a simple clipping indicator.
    return {
        "MAE": mae.item(),
        "RMSE": rmse.item()
    }


metrics_A = metrics(X_test, X_hat_A)
metrics_B = metrics(X_test, X_hat_B)

print("\nTest reconstruction results:")

print("\nCalibration A:")
print(metrics_A)

print("\nCalibration B:")
print(metrics_B)


# --------------------------------------------------
# 6. Measure clipping
# --------------------------------------------------

def clipping_fraction(x, scale):

    limit = 127 * scale

    clipped = (x.abs() > limit).float().mean()

    return clipped.item()


clip_A = clipping_fraction(X_test, scale_A)
clip_B = clipping_fraction(X_test, scale_B)

print("\nClipping fraction:")
print("Calibration A:", clip_A)
print("Calibration B:", clip_B)


# --------------------------------------------------
# 7. Validate quantization ranges
# --------------------------------------------------

assert q_A.min() >= -127
assert q_A.max() <= 127

assert q_B.min() >= -127
assert q_B.max() <= 127

assert X_hat_A.shape == X_test.shape
assert X_hat_B.shape == X_test.shape

print("\nAll shape and range checks passed.")
