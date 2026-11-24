
import math

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BLOCK_SIZE = 16
MAX_CONTEXT = 32768

requests = [
    {"name": "A", "tokens": 5000},
    {"name": "B", "tokens": 10000},
    {"name": "C", "tokens": 3000},
    {"name": "D", "tokens": 14000},
]


# --------------------------------------------------
# CONTIGUOUS MAX-CONTEXT ALLOCATION
# --------------------------------------------------

contiguous_allocated = 0
contiguous_used = 0

for request in requests:

    contiguous_allocated += MAX_CONTEXT
    contiguous_used += request["tokens"]


# --------------------------------------------------
# PAGED ALLOCATION
# --------------------------------------------------

paged_allocated = 0
paged_used = 0
total_blocks = 0

print("PAGED KV-CACHE ALLOCATION")
print("-" * 65)

for request in requests:

    tokens = request["tokens"]

    blocks = math.ceil(
        tokens / BLOCK_SIZE
    )

    allocated_slots = blocks * BLOCK_SIZE

    wasted_slots = allocated_slots - tokens

    total_blocks += blocks
    paged_allocated += allocated_slots
    paged_used += tokens

    print(
        f'{request["name"]}: '
        f'{tokens:>6,} tokens | '
        f'{blocks:>5,} blocks | '
        f'{wasted_slots:>3,} unused slots'
    )


# --------------------------------------------------
# COMPARISON
# --------------------------------------------------

print("\nALLOCATION COMPARISON")
print("-" * 65)

print(
    "Maximum-context allocation:",
    f"{contiguous_allocated:,} slots"
)

print(
    "Paged allocation:",
    f"{paged_allocated:,} slots"
)

print(
    "Paged internal fragmentation:",
    f"{paged_allocated - paged_used:,} slots"
)

print(
    "Blocks allocated:",
    total_blocks
)

print(
    "Slot reduction:",
    f"{(1 - paged_allocated / contiguous_allocated) * 100:.2f}%"
)


# --------------------------------------------------
# VALIDATION
# --------------------------------------------------


assert total_blocks == 2001
assert paged_allocated == 32016
assert paged_used == 32000
assert paged_allocated - paged_used == 16

print("\nAll validation checks passed!")

