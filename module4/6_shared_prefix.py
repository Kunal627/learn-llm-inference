
# --------------------------------------------------
# PREFIX CACHING SIMULATION
# --------------------------------------------------

BYTES_PER_TOKEN = 144 * 1024

SHARED_PREFIX = 4000

requests = [
    {"name": "A", "continuation": 1000},
    {"name": "B", "continuation": 1000},
    {"name": "C", "continuation": 1000},
]


# --------------------------------------------------
# WITHOUT PREFIX CACHING
# --------------------------------------------------

memory_without = 0

for request in requests:

    total_tokens = (
        SHARED_PREFIX
        + request["continuation"]
    )

    memory_without += (
        total_tokens * BYTES_PER_TOKEN
    )


# --------------------------------------------------
# WITH PREFIX CACHING
# --------------------------------------------------

# Store shared prefix only once
memory_with = (
    SHARED_PREFIX * BYTES_PER_TOKEN
)

# Store each request continuation
for request in requests:

    memory_with += (
        request["continuation"]
        * BYTES_PER_TOKEN
    )


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print("PREFIX CACHING COMPARISON")
print("-" * 60)

print(
    "Without caching:",
    f"{memory_without / (1024**3):.4f} GiB"
)

print(
    "With caching:",
    f"{memory_with / (1024**3):.4f} GiB"
)

savings = memory_without - memory_with

print(
    "Memory saved:",
    f"{savings / (1024**3):.4f} GiB"
)

saving_percent = (
    savings / memory_without
) * 100

print(
    "Memory reduction:",
    f"{saving_percent:.2f}%"
)


# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

assert memory_without == (
    15000 * BYTES_PER_TOKEN
)

assert memory_with == (
    7000 * BYTES_PER_TOKEN
)

assert abs(saving_percent - 53.3333333) < 0.001

print("\nAll validation checks passed!")
