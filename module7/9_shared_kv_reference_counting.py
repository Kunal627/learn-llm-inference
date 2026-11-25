
BLOCK_SIZE = 4
TOTAL_BLOCKS = 12

# Physical block pool
free_blocks = list(range(TOTAL_BLOCKS))

# Request ID -> physical block IDs
block_tables = {}

# Physical block ID -> number of references
ref_counts = {}

# Shared prefix cache
prefix_cache = {}


def allocate_block():
    if not free_blocks:
        raise MemoryError("No free KV-cache blocks")

    block_id = free_blocks.pop(0)
    ref_counts[block_id] = 1
    print(f"Allocated physical block {block_id}")
    return block_id


def create_shared_prefix(prefix_name, num_tokens):
    if prefix_name in prefix_cache:
        raise ValueError("Prefix already cached")

    blocks_needed = (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE

    allocated = []

    for _ in range(blocks_needed):
        allocated.append(allocate_block())

    prefix_cache[prefix_name] = allocated

    print(f"\nCreated shared prefix: {prefix_name}")
    print("Shared blocks:", allocated)


def create_request(request_id, prefix_name):
    if request_id in block_tables:
        raise ValueError("Request already exists")

    if prefix_name not in prefix_cache:
        raise ValueError("Unknown prefix")

    shared_blocks = prefix_cache[prefix_name]

    # Reference existing blocks instead of copying them
    for block_id in shared_blocks:
        ref_counts[block_id] += 1

    block_tables[request_id] = list(shared_blocks)

    print(f"\nCreated request {request_id}")
    print("References shared blocks:", shared_blocks)


def add_private_tokens(request_id, num_tokens):
    if request_id not in block_tables:
        raise ValueError("Unknown request")

    blocks = block_tables[request_id]

    # Calculate current logical sequence length
    current_length = (
        len(blocks) * BLOCK_SIZE
    )

    # This simplified exercise starts private tokens
    # at a fresh block boundary.
    if current_length % BLOCK_SIZE != 0:
        raise ValueError("Expected block-aligned prefix")

    blocks_needed = (
        num_tokens + BLOCK_SIZE - 1
    ) // BLOCK_SIZE

    for _ in range(blocks_needed):
        blocks.append(allocate_block())

    print(f"{request_id}: Added {num_tokens} private tokens")
    print("Block table:", blocks)


def release_request(request_id):
    if request_id not in block_tables:
        raise ValueError("Unknown request")

    blocks = block_tables.pop(request_id)

    print(f"\nReleasing request {request_id}")

    for block_id in blocks:
        ref_counts[block_id] -= 1

        print(
            f"Block {block_id}: "
            f"Reference count = {ref_counts[block_id]}"
        )

        if ref_counts[block_id] == 0:
            print(f"Block {block_id} returned to free pool")
            del ref_counts[block_id]
            free_blocks.append(block_id)

    free_blocks.sort()


def show_memory():
    print("\n" + "=" * 55)
    print("KV-CACHE MEMORY STATUS")
    print("=" * 55)

    print("\nRequest block tables:")
    for request_id, blocks in block_tables.items():
        print(f"{request_id}: {blocks}")

    print("\nPhysical block reference counts:")
    for block_id, count in sorted(ref_counts.items()):
        print(f"Block {block_id}: ref_count={count}")

    print("\nFree blocks:", free_blocks)
    print("Used blocks:", len(ref_counts))


# ------------------------------------------------
# EXPERIMENT
# ------------------------------------------------

create_shared_prefix("SYSTEM_PROMPT", 8)

create_request("A", "SYSTEM_PROMPT")
create_request("B", "SYSTEM_PROMPT")
create_request("C", "SYSTEM_PROMPT")

add_private_tokens("A", 4)
add_private_tokens("B", 8)
add_private_tokens("C", 4)

show_memory()

# A finishes
release_request("A")
show_memory()

# B finishes
release_request("B")
show_memory()

# C finishes
release_request("C")
show_memory()
