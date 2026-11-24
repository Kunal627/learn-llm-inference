
import torch

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BATCH_SIZE = 2
NUM_KV_HEADS = 8
SEQ_LEN = 8192
HEAD_DIM = 128

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)
print("PyTorch:", torch.__version__)


# --------------------------------------------------
# CREATE SIMULATED KV CACHE
# --------------------------------------------------

torch.manual_seed(42)

K = torch.randn(
    BATCH_SIZE,
    NUM_KV_HEADS,
    SEQ_LEN,
    HEAD_DIM,
    device=device,
    dtype=torch.float32
)

V = torch.randn_like(K)

print("\nORIGINAL CACHE")
print("K:", K.shape)
print("V:", V.shape)


# --------------------------------------------------
# MEMORY CALCULATION
# --------------------------------------------------

def tensor_memory_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def cache_memory_bytes(K, V):
    return (
        tensor_memory_bytes(K)
        + tensor_memory_bytes(V)
    )


# --------------------------------------------------
# PRECISION CONVERSION
# --------------------------------------------------

precisions = {
    "FP32": torch.float32,
    "FP16": torch.float16,
    "BF16": torch.bfloat16,
}

print("\nPRECISION COMPARISON")
print("-" * 75)

for name, dtype in precisions.items():

    K_converted = K.to(dtype)
    V_converted = V.to(dtype)

    memory = cache_memory_bytes(
        K_converted,
        V_converted
    )

    # Convert back to FP32 for error measurement
    K_reconstructed = K_converted.float()
    V_reconstructed = V_converted.float()

    k_error = (
        K - K_reconstructed
    ).abs().max().item()

    v_error = (
        V - V_reconstructed
    ).abs().max().item()

    print(f"\n{name}")
    print("K dtype:", K_converted.dtype)
    print("V dtype:", V_converted.dtype)
    print("Memory:", memory / (1024**2), "MiB")
    print("K max absolute error:", k_error)
    print("V max absolute error:", v_error)


# --------------------------------------------------
# FP8 TEST
# --------------------------------------------------

print("\nFP8 SUPPORT TEST")
print("-" * 75)

fp8_types = [
    ("FP8 E4M3", getattr(torch, "float8_e4m3fn", None)),
    ("FP8 E5M2", getattr(torch, "float8_e5m2", None)),
]

for name, dtype in fp8_types:

    if dtype is None:
        print(name, "not available in this PyTorch build")
        continue

    try:
        K_fp8 = K.to(dtype)
        V_fp8 = V.to(dtype)

        memory = cache_memory_bytes(K_fp8, V_fp8)

        # Convert back to FP32 for comparison
        K_reconstructed = K_fp8.float()
        V_reconstructed = V_fp8.float()

        k_error = (
            K - K_reconstructed
        ).abs().max().item()

        v_error = (
            V - V_reconstructed
        ).abs().max().item()

        print(f"\n{name}")
        print("Memory:", memory / (1024**2), "MiB")
        print("K max absolute error:", k_error)
        print("V max absolute error:", v_error)

    except Exception as e:
        print(name, "test failed:", str(e))


# --------------------------------------------------
# THEORETICAL MEMORY COMPARISON
# --------------------------------------------------

print("\nTHEORETICAL KV CACHE MEMORY")
print("-" * 75)

NUM_LAYERS = 36
NUM_KV_HEADS = 8
HEAD_DIM = 128
CONTEXT_LENGTH = 32768
BATCH_SIZE = 1

for name, bytes_per_element in [
    ("FP32", 4),
    ("FP16", 2),
    ("BF16", 2),
    ("FP8", 1),
]:

    memory = (
        2
        * NUM_LAYERS
        * BATCH_SIZE
        * NUM_KV_HEADS
        * CONTEXT_LENGTH
        * HEAD_DIM
        * bytes_per_element
    )

    print(
        f"{name}: "
        f"{memory / (1024**3):.3f} GiB"
    )
