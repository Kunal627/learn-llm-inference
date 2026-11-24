
import torch
import math

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

BATCH_SIZE = 2
SEQ_LEN = 100
NUM_QUERY_HEADS = 8
HEAD_DIM = 64
NUM_LAYERS = 32

# Number of KV heads for each architecture
ARCHITECTURES = {
    "MHA": 8,
    "MQA": 1,
    "GQA": 2,
}

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)


# --------------------------------------------------
# 2. KV-cache memory calculator
# --------------------------------------------------

def calculate_kv_memory(
    layers,
    batch,
    kv_heads,
    sequence_length,
    head_dim,
    bytes_per_element=2,
):
    return (
        2
        * layers
        * batch
        * kv_heads
        * sequence_length
        * head_dim
        * bytes_per_element
    )


def format_memory(num_bytes):
    return f"{num_bytes / (1024 ** 2):.3f} MiB"


# --------------------------------------------------
# 3. Compare tensor shapes and memory
# --------------------------------------------------

print("\n" + "=" * 65)
print("ARCHITECTURE COMPARISON")
print("=" * 65)

memory_results = {}

for name, num_kv_heads in ARCHITECTURES.items():

    assert NUM_QUERY_HEADS % num_kv_heads == 0

    group_size = NUM_QUERY_HEADS // num_kv_heads

    q_shape = (
        BATCH_SIZE,
        NUM_QUERY_HEADS,
        SEQ_LEN,
        HEAD_DIM,
    )

    kv_shape = (
        BATCH_SIZE,
        num_kv_heads,
        SEQ_LEN,
        HEAD_DIM,
    )

    memory = calculate_kv_memory(
        layers=NUM_LAYERS,
        batch=BATCH_SIZE,
        kv_heads=num_kv_heads,
        sequence_length=SEQ_LEN,
        head_dim=HEAD_DIM,
        bytes_per_element=2,  # FP16
    )

    memory_results[name] = memory

    print(f"\n{name}")
    print("Query heads:", NUM_QUERY_HEADS)
    print("KV heads:", num_kv_heads)
    print("Query shape:", q_shape)
    print("Key shape:", kv_shape)
    print("Value shape:", kv_shape)
    print("Query heads per KV head:", group_size)
    print("Theoretical FP16 cache:", format_memory(memory))


# --------------------------------------------------
# 4. Implement grouped-query attention
# --------------------------------------------------

def grouped_query_attention(q, k, v):
    """
    q: [B, Hq, S, D]
    k: [B, Hkv, S, D]
    v: [B, Hkv, S, D]

    Returns:
        output: [B, Hq, S, D]
    """

    batch, num_query_heads, seq_len, head_dim = q.shape
    num_kv_heads = k.shape[1]

    assert num_query_heads % num_kv_heads == 0

    group_size = num_query_heads // num_kv_heads

    outputs = []

    # Causal mask: each position can attend only
    # to itself and earlier positions.
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
        diagonal=1,
    )

    for kv_head in range(num_kv_heads):

        # Select the query heads belonging to this KV group.
        start = kv_head * group_size
        end = start + group_size

        q_group = q[:, start:end, :, :]  # [B, G, S, D]

        # One shared K/V head for this group.
        k_group = k[:, kv_head, :, :]    # [B, S, D]
        v_group = v[:, kv_head, :, :]    # [B, S, D]

        # Attention scores:
        # [B, G, S, D] x [B, S, D] -> [B, G, S, S]
        scores = torch.einsum(
            "bgtd,bsd->bgts",
            q_group,
            k_group,
        ) / math.sqrt(head_dim)

        scores = scores.masked_fill(
            causal_mask,
            float("-inf"),
        )

        attention_weights = torch.softmax(scores, dim=-1)

        # Weighted sum:
        # [B, G, S, S] x [B, S, D] -> [B, G, S, D]
        group_output = torch.einsum(
            "bgts,bsd->bgtd",
            attention_weights,
            v_group,
        )

        outputs.append(group_output)

    # Restore the original query-head dimension.
    output = torch.cat(outputs, dim=1)

    return output


# --------------------------------------------------
# 5. Run all three architectures
# --------------------------------------------------

print("\n" + "=" * 65)
print("ATTENTION EXECUTION")
print("=" * 65)

for name, num_kv_heads in ARCHITECTURES.items():

    q = torch.randn(
        BATCH_SIZE,
        NUM_QUERY_HEADS,
        SEQ_LEN,
        HEAD_DIM,
        device=device,
    )

    k = torch.randn(
        BATCH_SIZE,
        num_kv_heads,
        SEQ_LEN,
        HEAD_DIM,
        device=device,
    )

    v = torch.randn(
        BATCH_SIZE,
        num_kv_heads,
        SEQ_LEN,
        HEAD_DIM,
        device=device,
    )

    output = grouped_query_attention(q, k, v)

    print(f"\n{name}")
    print("Q:", tuple(q.shape))
    print("K:", tuple(k.shape))
    print("V:", tuple(v.shape))
    print("Output:", tuple(output.shape))

    assert output.shape == (
        BATCH_SIZE,
        NUM_QUERY_HEADS,
        SEQ_LEN,
        HEAD_DIM,
    )

    print("Shape validation: PASSED")


# --------------------------------------------------
# 6. Verify memory ratios
# --------------------------------------------------

print("\n" + "=" * 65)
print("KV CACHE MEMORY RATIOS")
print("=" * 65)

mha_memory = memory_results["MHA"]

for name, memory in memory_results.items():

    ratio = memory / mha_memory

    print(
        f"{name}: {format_memory(memory)} "
        f"| {ratio:.3f}x MHA memory"
    )

assert memory_results["GQA"] == mha_memory // 4
assert memory_results["MQA"] == mha_memory // 8

print("\nAll memory checks passed.")
