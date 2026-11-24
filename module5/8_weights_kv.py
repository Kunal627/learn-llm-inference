
# --------------------------------------------------
# Weight + KV-cache memory calculator
# --------------------------------------------------

NUM_PARAMS = 8_000_000_000

# Qwen3-8B-style architecture
LAYERS = 36
BATCH_SIZE = 1
KV_HEADS = 8
HEAD_DIM = 128

GIB = 1024 ** 3


def weight_memory(num_params, bits_per_param):
    return num_params * bits_per_param / 8


def kv_cache_memory(
    layers,
    batch,
    kv_heads,
    seq_len,
    head_dim,
    bytes_per_element
):
    return (
        2
        * layers
        * batch
        * kv_heads
        * seq_len
        * head_dim
        * bytes_per_element
    )


weight_configs = {
    "FP16": 16,
    "INT8": 8,
    "INT4": 4
}

kv_configs = {
    "FP16": 2,
    "FP8": 1,
    "INT8": 1,
    "INT4": 0.5
}

for context in [8192, 32768, 65536]:

    print("\n" + "=" * 65)
    print(f"CONTEXT LENGTH: {context:,}")
    print("=" * 65)

    kv_results = {}

    for kv_name, kv_bytes in kv_configs.items():

        kv_mem = kv_cache_memory(
            LAYERS,
            BATCH_SIZE,
            KV_HEADS,
            context,
            HEAD_DIM,
            kv_bytes
        )

        kv_results[kv_name] = kv_mem / GIB

    for weight_name, weight_bits in weight_configs.items():

        weight_mem = weight_memory(
            NUM_PARAMS,
            weight_bits
        )

        print(f"\nWeight precision: {weight_name}")
        print(f"Weight memory: {weight_mem / GIB:.3f} GiB")

        for kv_name, kv_mem_gib in kv_results.items():

            total = weight_mem / GIB + kv_mem_gib

            print(
                f"  KV {kv_name:5s} | "
                f"KV memory: {kv_mem_gib:.3f} GiB | "
                f"Total: {total:.3f} GiB"
            )


# --------------------------------------------------
# Validation
# --------------------------------------------------

fp16_32k = kv_cache_memory(
    36, 1, 8, 32768, 128, 2
)

fp8_32k = kv_cache_memory(
    36, 1, 8, 32768, 128, 1
)

assert abs(fp16_32k / GIB - 4.5) < 1e-6
assert abs(fp8_32k / GIB - 2.25) < 1e-6

assert abs(fp16_32k / fp8_32k - 2.0) < 1e-6

print("\nAll memory calculation checks passed.")
