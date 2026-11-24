
import torch

torch.set_printoptions(
    precision=4,
    sci_mode=False,
    linewidth=140
)


def show(name, tensor):
    """Print tensor name, shape and values."""
    print(f"\n{'=' * 80}")
    print(f"{name}")
    print(f"Shape: {list(tensor.shape)}")
    print(tensor)


# ---------------------------------------------------------
# STEP 1: Create input Query and Key tensors
# ---------------------------------------------------------

print("\nSTEP 1: INPUT QUERY AND KEY")

B = 1
S = 3
H = 2
D = 4
BASE = 100

# Q shape: [B, S, H, D]
q = torch.tensor([
    [
        [[1., 2., 3., 4.], [5., 6., 7., 8.]],
        [[2., 3., 4., 5.], [6., 7., 8., 9.]],
        [[3., 4., 5., 6.], [7., 8., 9., 10.]]
    ]
])

# K shape: [B, S, H, D]
k = torch.tensor([
    [
        [[1., 0., 1., 0.], [0., 1., 0., 1.]],
        [[2., 0., 2., 0.], [0., 2., 0., 2.]],
        [[3., 0., 3., 0.], [0., 3., 0., 3.]]
    ]
])

show("Q - Original Query", q)
show("K - Original Key", k)


# ---------------------------------------------------------
# STEP 2: Generate dimension indices
# ---------------------------------------------------------

print("\nSTEP 2: GENERATE DIMENSION INDICES")

dim_indices = torch.arange(0, D, 2).float()

show("Dimension indices: arange(0, D, 2)", dim_indices)

# D=4 gives [0, 2].
# Each index represents one pair of dimensions.


# ---------------------------------------------------------
# STEP 3: Calculate inverse frequencies
# ---------------------------------------------------------

print("\nSTEP 3: CALCULATE INVERSE FREQUENCIES")

print(f"BASE = {BASE}")
print(f"HEAD DIMENSION = {D}")

exponents = dim_indices / D

show("Exponents = dimension indices / D", exponents)

base_powers = BASE ** exponents

show("BASE raised to exponents", base_powers)

inv_freq = 1.0 / base_powers

show("Inverse frequencies", inv_freq)


# ---------------------------------------------------------
# STEP 4: Generate token positions
# ---------------------------------------------------------

print("\nSTEP 4: GENERATE TOKEN POSITIONS")

positions = torch.arange(S).float()

show("Token positions", positions)

# [S] -> [S, 1]
positions = positions[:, None]

show("Positions after [:, None]", positions)


# ---------------------------------------------------------
# STEP 5: Reshape inverse frequencies
# ---------------------------------------------------------

print("\nSTEP 5: RESHAPE INVERSE FREQUENCIES")

inv_freq_reshaped = inv_freq[None, :]

show("Inverse frequencies after [None, :]", inv_freq_reshaped)


# ---------------------------------------------------------
# STEP 6: Calculate angles
# ---------------------------------------------------------

print("\nSTEP 6: CALCULATE ROTATION ANGLES")

print("angles = positions * inverse frequencies")

angles = positions * inv_freq_reshaped

show("Rotation angles", angles)


# ---------------------------------------------------------
# STEP 7: Calculate cosine and sine
# ---------------------------------------------------------

print("\nSTEP 7: CALCULATE COSINE AND SINE")

cos = torch.cos(angles)
sin = torch.sin(angles)

show("Cosine values", cos)
show("Sine values", sin)


# ---------------------------------------------------------
# STEP 8: Extract even and odd dimensions
# ---------------------------------------------------------

print("\nSTEP 8: SPLIT Q INTO EVEN AND ODD DIMENSIONS")

q_even = q[..., 0::2]
q_odd = q[..., 1::2]

show("Q even dimensions: q[..., 0::2]", q_even)
show("Q odd dimensions: q[..., 1::2]", q_odd)


# ---------------------------------------------------------
# STEP 9: Broadcast cosine and sine
# ---------------------------------------------------------

print("\nSTEP 9: BROADCAST COSINE AND SINE")

print("Original cos shape:", list(cos.shape))

cos_broadcast = cos[None, :, None, :]
sin_broadcast = sin[None, :, None, :]

show("Cosine after broadcasting", cos_broadcast)
show("Sine after broadcasting", sin_broadcast)


# ---------------------------------------------------------
# STEP 10: Apply rotation formula
# ---------------------------------------------------------

print("\nSTEP 10: APPLY ROTATION FORMULA")

print("rotated_even = q_even * cos - q_odd * sin")
print("rotated_odd  = q_even * sin + q_odd * cos")

rotated_even = (
    q_even * cos_broadcast
    - q_odd * sin_broadcast
)

rotated_odd = (
    q_even * sin_broadcast
    + q_odd * cos_broadcast
)

show("Rotated even dimensions", rotated_even)
show("Rotated odd dimensions", rotated_odd)


# ---------------------------------------------------------
# STEP 11: Interleave even and odd dimensions
# ---------------------------------------------------------

print("\nSTEP 11: RECONSTRUCT ROTATED QUERY")

stacked = torch.stack(
    [rotated_even, rotated_odd],
    dim=-1
)

show("Stacked even and odd dimensions", stacked)

q_rotated = stacked.flatten(-2)

show("Final rotated Query", q_rotated)


# ---------------------------------------------------------
# STEP 12: Apply the same process to Key
# ---------------------------------------------------------

print("\nSTEP 12: ROTATE KEY")

k_even = k[..., 0::2]
k_odd = k[..., 1::2]

k_rotated_even = (
    k_even * cos_broadcast
    - k_odd * sin_broadcast
)

k_rotated_odd = (
    k_even * sin_broadcast
    + k_odd * cos_broadcast
)

k_stacked = torch.stack(
    [k_rotated_even, k_rotated_odd],
    dim=-1
)

k_rotated = k_stacked.flatten(-2)

show("Final rotated Key", k_rotated)


# ---------------------------------------------------------
# STEP 13: Verify output shapes
# ---------------------------------------------------------

print("\nSTEP 13: VERIFY SHAPES")

print("Original Q:", list(q.shape))
print("Rotated Q :", list(q_rotated.shape))

print("Original K:", list(k.shape))
print("Rotated K :", list(k_rotated.shape))

assert q.shape == q_rotated.shape
assert k.shape == k_rotated.shape

print("\nSUCCESS: RoPE preserves Q and K shapes.")


# ---------------------------------------------------------
# STEP 14: Calculate attention scores
# ---------------------------------------------------------

print("\nSTEP 14: CALCULATE ATTENTION SCORES")

# [B, S, H, D] -> [B, H, S, D]
q_heads = q_rotated.transpose(1, 2)
k_heads = k_rotated.transpose(1, 2)

show("Q after head transpose", q_heads)
show("K after head transpose", k_heads)

# K transpose: [B, H, D, S]
k_heads_t = k_heads.transpose(-2, -1)

show("K transpose for attention", k_heads_t)

# [B, H, S, D] @ [B, H, D, S]
# Result: [B, H, S, S]
scores = q_heads @ k_heads_t

show("Attention scores QK transpose", scores)

scores = scores / (D ** 0.5)

show("Scaled attention scores", scores)

print("\nROPE EXPERIMENT COMPLETED")
