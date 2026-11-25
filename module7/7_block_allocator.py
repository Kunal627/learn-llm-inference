
import math


# ============================================================
# CONFIGURATION
# ============================================================

BLOCK_SIZE = 16
TOTAL_BLOCKS = 12


# ============================================================
# PHYSICAL KV-CACHE BLOCK POOL
# ============================================================

free_blocks = list(range(TOTAL_BLOCKS))

# Request ID -> list of physical block IDs
block_tables = {}

# Request ID -> logical KV-cache length
sequence_lengths = {}


# ============================================================
# ALLOCATION
# ============================================================

def allocate_block(request_id):

    if not free_blocks:
        raise MemoryError("No free KV-cache blocks available")

    block_id = free_blocks.pop(0)

    block_tables[request_id].append(block_id)

    print(
        f"{request_id}: Allocated physical block {block_id}"
    )

    return block_id


def create_request(request_id):

    if request_id in block_tables:
        raise ValueError("Request already exists")

    block_tables[request_id] = []
    sequence_lengths[request_id] = 0

    print(f"\nCreating request {request_id}")


# ============================================================
# ADD TOKENS TO KV CACHE
# ============================================================

def add_tokens(request_id, num_tokens):

    if request_id not in block_tables:
        raise ValueError("Unknown request")

    for _ in range(num_tokens):

        current_length = sequence_lengths[request_id]

        # Allocate a block when the current
        # logical sequence has no free position.
        if current_length % BLOCK_SIZE == 0:
            allocate_block(request_id)

        sequence_lengths[request_id] += 1

    print(
        f"{request_id}: Added {num_tokens} tokens | "
        f"Sequence length = {sequence_lengths[request_id]} | "
        f"Blocks = {block_tables[request_id]}"
    )


# ============================================================
# RELEASE KV CACHE
# ============================================================

def release_request(request_id):

    if request_id not in block_tables:
        raise ValueError("Unknown request")

    released = block_tables.pop(request_id)

    sequence_lengths.pop(request_id)

    free_blocks.extend(released)
    free_blocks.sort()

    print(
        f"\n{request_id}: Released blocks {released}"
    )


# ============================================================
# DISPLAY MEMORY
# ============================================================

def show_memory():

    print("\n" + "=" * 50)
    print("KV-CACHE MEMORY STATUS")
    print("=" * 50)

    for request_id, blocks in block_tables.items():

        print(
            f"{request_id}: "
            f"Length={sequence_lengths[request_id]} | "
            f"Blocks={blocks}"
        )

    print("\nFree blocks:", free_blocks)
    print("Used blocks:", TOTAL_BLOCKS - len(free_blocks))
    print("Total blocks:", TOTAL_BLOCKS)


# ============================================================
# EXPERIMENT
# ============================================================

create_request("A")
add_tokens("A", 35)

create_request("B")
add_tokens("B", 20)

create_request("C")
add_tokens("C", 10)

show_memory()

# A finishes
release_request("A")

show_memory()

# D arrives and uses the reclaimed blocks
create_request("D")
add_tokens("D", 40)

show_memory()
