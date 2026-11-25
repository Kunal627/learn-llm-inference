
BLOCK_SIZE = 4
TOTAL_BLOCKS = 10

free_blocks = list(range(TOTAL_BLOCKS))

# Physical block ID -> stored token labels
physical_blocks = {}

# Physical block ID -> reference count
ref_counts = {}

# Request ID -> physical block table
block_tables = {}

# Request ID -> logical sequence length
sequence_lengths = {}


def allocate_block(tokens=None):
    if not free_blocks:
        raise MemoryError("KV-cache exhausted")

    block_id = free_blocks.pop(0)

    physical_blocks[block_id] = (
        list(tokens) if tokens else []
    )

    ref_counts[block_id] = 1

    print(f"Allocated block {block_id}")

    return block_id


def create_shared_prefix(request_ids, prefix_tokens):
    print("\nCreating shared prefix")

    prefix_blocks = []

    for start in range(0, len(prefix_tokens), BLOCK_SIZE):
        chunk = prefix_tokens[start:start + BLOCK_SIZE]
        prefix_blocks.append(allocate_block(chunk))

    for request_id in request_ids:
        block_tables[request_id] = list(prefix_blocks)
        sequence_lengths[request_id] = len(prefix_tokens)

        for block_id in prefix_blocks:
            ref_counts[block_id] += 1

        print(f"{request_id} -> {block_tables[request_id]}")

    # Release the allocator's temporary ownership.
    # The active requests now own the references.
    for block_id in prefix_blocks:
        ref_counts[block_id] -= 1


def append_token(request_id, token):
    blocks = block_tables[request_id]
    length = sequence_lengths[request_id]

    # Is the final block partially filled?
    if length % BLOCK_SIZE != 0:

        last_block = blocks[-1]

        # Shared partial block: perform COW.
        if ref_counts[last_block] > 1:

            print(
                f"\n{request_id}: Block {last_block} "
                "is shared. Performing COW."
            )

            old_tokens = physical_blocks[last_block]

            new_block = allocate_block(old_tokens)

            # Request moves to its private copy.
            blocks[-1] = new_block

            # Remove this request's reference to old block.
            ref_counts[last_block] -= 1

            print(
                f"{request_id}: Switched from block "
                f"{last_block} to {new_block}"
            )

        else:
            new_block = last_block

    else:
        # Previous block is full: allocate another block.
        new_block = allocate_block()

        blocks.append(new_block)

    # Append the token to the request's private block.
    physical_blocks[new_block].append(token)
    sequence_lengths[request_id] += 1

    print(
        f"{request_id}: Appended {token} "
        f"to block {new_block}"
    )


def release_request(request_id):
    print(f"\nReleasing {request_id}")

    blocks = block_tables.pop(request_id)
    sequence_lengths.pop(request_id)

    for block_id in blocks:
        ref_counts[block_id] -= 1

        if ref_counts[block_id] == 0:
            print(f"Freed block {block_id}")

            del ref_counts[block_id]
            del physical_blocks[block_id]

            free_blocks.append(block_id)

    free_blocks.sort()


def show_memory():
    print("\n" + "=" * 55)
    print("PHYSICAL KV-CACHE")
    print("=" * 55)

    for block_id in sorted(physical_blocks):
        print(
            f"Block {block_id}: "
            f"{physical_blocks[block_id]} | "
            f"Ref count: {ref_counts[block_id]}"
        )

    print("\nRequest block tables:")

    for request_id, blocks in block_tables.items():
        print(f"{request_id}: {blocks}")

    print("\nFree blocks:", sorted(free_blocks))


# --------------------------------------------------
# EXPERIMENT
# --------------------------------------------------

prefix = ["T1", "T2", "T3", "T4", "T5", "T6"]

create_shared_prefix(["A", "B"], prefix)

show_memory()

# A and B append different tokens.
append_token("A", "HELLO")

show_memory()

append_token("B", "GOODBYE")

show_memory()

# Release A and inspect memory.
release_request("A")

show_memory()

# Release B.
release_request("B")

show_memory()
