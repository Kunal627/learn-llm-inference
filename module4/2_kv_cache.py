
import torch

torch.manual_seed(42)

# ============================================================
# MODULE 4 - LESSON 4.2
# KV CACHE DIMENSIONS AND GROWTH
# ============================================================

# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

NUM_LAYERS = 4
BATCH_SIZE = 2
NUM_KV_HEADS = 4
HEAD_DIM = 8

INITIAL_SEQ_LEN = 5
GENERATED_TOKENS = 3

DTYPE = torch.float16

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 65)
print("KV CACHE CONFIGURATION")
print("=" * 65)

print("Layers:", NUM_LAYERS)
print("Batch size:", BATCH_SIZE)
print("KV heads:", NUM_KV_HEADS)
print("Head dimension:", HEAD_DIM)
print("Initial sequence length:", INITIAL_SEQ_LEN)
print("Dtype:", DTYPE)


# ------------------------------------------------------------
# 2. PREFILL: CREATE INITIAL K AND V
# ------------------------------------------------------------

# Each layer stores:
# K: [B, H_KV, S, HEAD_DIM]
# V: [B, H_KV, S, HEAD_DIM]

key_cache = []
value_cache = []

for layer in range(NUM_LAYERS):

    k = torch.randn(
        BATCH_SIZE,
        NUM_KV_HEADS,
        INITIAL_SEQ_LEN,
        HEAD_DIM,
        dtype=DTYPE,
        device=device
    )

    v = torch.randn(
        BATCH_SIZE,
        NUM_KV_HEADS,
        INITIAL_SEQ_LEN,
        HEAD_DIM,
        dtype=DTYPE,
        device=device
    )

    key_cache.append(k)
    value_cache.append(v)


print("\n" + "=" * 65)
print("PREFILL CACHE")
print("=" * 65)

print("Number of layers:", len(key_cache))

print("K cache shape:", key_cache[0].shape)
print("V cache shape:", value_cache[0].shape)

print(
    "Total cache shape across layers:",
    NUM_LAYERS,
    BATCH_SIZE,
    NUM_KV_HEADS,
    INITIAL_SEQ_LEN,
    HEAD_DIM
)


# ------------------------------------------------------------
# 3. CALCULATE ACTUAL MEMORY
# ------------------------------------------------------------

def calculate_actual_cache_memory(k_cache, v_cache):

    total_bytes = 0

    for k, v in zip(k_cache, v_cache):

        total_bytes += k.numel() * k.element_size()
        total_bytes += v.numel() * v.element_size()

    return total_bytes


def print_cache_memory(k_cache, v_cache, label):

    total_bytes = calculate_actual_cache_memory(
        k_cache,
        v_cache
    )

    print(f"\n{label}")

    print("Memory in bytes:", total_bytes)

    print("Memory in KiB:", total_bytes / 1024)

    print("Memory in MiB:", total_bytes / (1024 ** 2))


print_cache_memory(
    key_cache,
    value_cache,
    "AFTER PREFILL"
)


# ------------------------------------------------------------
# 4. DECODE: APPEND ONE TOKEN AT A TIME
# ------------------------------------------------------------

for step in range(GENERATED_TOKENS):

    new_key_cache = []
    new_value_cache = []

    for layer in range(NUM_LAYERS):

        # New token K and V projections.
        #
        # [B, H_KV, 1, HEAD_DIM]

        new_k = torch.randn(
            BATCH_SIZE,
            NUM_KV_HEADS,
            1,
            HEAD_DIM,
            dtype=DTYPE,
            device=device
        )

        new_v = torch.randn(
            BATCH_SIZE,
            NUM_KV_HEADS,
            1,
            HEAD_DIM,
            dtype=DTYPE,
            device=device
        )

        # Append along sequence dimension (dim=2).

        updated_k = torch.cat(
            [key_cache[layer], new_k],
            dim=2
        )

        updated_v = torch.cat(
            [value_cache[layer], new_v],
            dim=2
        )

        new_key_cache.append(updated_k)
        new_value_cache.append(updated_v)

    key_cache = new_key_cache
    value_cache = new_value_cache

    print("\n" + "=" * 65)
    print(f"AFTER DECODE STEP {step + 1}")
    print("=" * 65)

    print("K shape:", key_cache[0].shape)
    print("V shape:", value_cache[0].shape)

    print_cache_memory(
        key_cache,
        value_cache,
        f"DECODE STEP {step + 1}"
    )


# ------------------------------------------------------------
# 5. THEORETICAL MEMORY CALCULATION
# ------------------------------------------------------------

final_seq_len = INITIAL_SEQ_LEN + GENERATED_TOKENS

bytes_per_element = torch.tensor(
    [], dtype=DTYPE
).element_size()

theoretical_bytes = (
    2
    * NUM_LAYERS
    * BATCH_SIZE
    * NUM_KV_HEADS
    * final_seq_len
    * HEAD_DIM
    * bytes_per_element
)

actual_bytes = calculate_actual_cache_memory(
    key_cache,
    value_cache
)

print("\n" + "=" * 65)
print("FINAL MEMORY VALIDATION")
print("=" * 65)

print("Final sequence length:", final_seq_len)

print("Theoretical bytes:", theoretical_bytes)

print("Actual bytes:", actual_bytes)

print("Match:", theoretical_bytes == actual_bytes)

assert theoretical_bytes == actual_bytes

print("\nALL CHECKS PASSED")
