
import math

# --------------------------------------------------
# MODEL CONFIGURATION: Qwen3-8B-style architecture
# --------------------------------------------------

NUM_LAYERS = 36
NUM_KV_HEADS = 8
HEAD_DIM = 128

BYTES_PER_ELEMENT = 2  # FP16

# Memory consumed by one token in one request
BYTES_PER_TOKEN = (
    2
    * NUM_LAYERS
    * NUM_KV_HEADS
    * HEAD_DIM
    * BYTES_PER_ELEMENT
)

print("KV CACHE MEMORY PER TOKEN")
print(f"{BYTES_PER_TOKEN:,} bytes")
print(f"{BYTES_PER_TOKEN / 1024:.2f} KiB")


# --------------------------------------------------
# FUNCTION: Calculate memory for one request
# --------------------------------------------------

def calculate_request_memory(
    prompt_tokens,
    generated_tokens
):
    total_tokens = prompt_tokens + generated_tokens

    memory_bytes = (
        total_tokens * BYTES_PER_TOKEN
    )

    return total_tokens, memory_bytes


# --------------------------------------------------
# SIMULATE CONCURRENT REQUESTS
# --------------------------------------------------

requests = [
    {"name": "User A", "prompt": 4000, "generated": 1000},
    {"name": "User B", "prompt": 8000, "generated": 2000},
    {"name": "User C", "prompt": 2000, "generated": 1000},
    {"name": "User D", "prompt": 12000, "generated": 2000},
]


total_memory = 0

print("\nINDIVIDUAL REQUEST MEMORY")
print("-" * 65)

for request in requests:

    tokens, memory = calculate_request_memory(
        request["prompt"],
        request["generated"]
    )

    total_memory += memory

    print(
        f'{request["name"]}: '
        f'{tokens:>6,} tokens | '
        f'{memory / (1024**3):.4f} GiB'
    )


print("\nTOTAL CONCURRENT KV CACHE")
print("-" * 65)

print(f"Total memory: {total_memory / (1024**3):.4f} GiB")


# --------------------------------------------------
# CONCURRENCY ESTIMATION
# --------------------------------------------------

KV_BUDGET_GIB = 5

print("\nMAXIMUM REQUEST ESTIMATION")
print("-" * 65)

for context_length in [4096, 8192, 16384, 32768]:

    memory_per_request = (
        context_length * BYTES_PER_TOKEN
    )

    memory_per_request_gib = (
        memory_per_request / (1024**3)
    )

    max_requests = math.floor(
        KV_BUDGET_GIB / memory_per_request_gib
    )

    print(
        f"Context: {context_length:>6,} | "
        f"Per request: {memory_per_request_gib:.4f} GiB | "
        f"Max requests: {max_requests}"
    )
