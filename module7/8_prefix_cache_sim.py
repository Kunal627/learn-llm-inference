
# Module 7.7 - Prefix Caching Simulator

BLOCK_SIZE = 4

requests = {
    "A": ["SYSTEM", "SYSTEM", "SYSTEM", "SYSTEM",
          "Explain", "transformers"],
    "B": ["SYSTEM", "SYSTEM", "SYSTEM", "SYSTEM",
          "Explain", "attention"],
    "C": ["SYSTEM", "SYSTEM", "SYSTEM", "SYSTEM",
          "Explain", "KV", "cache"],
}

# Simplified cache: prefix tokens -> cached KV representation
prefix_cache = {}

def process_request(request_id, tokens):
    print(f"\nRequest {request_id}")

    # Find the longest cached prefix at block boundaries
    cached_tokens = 0

    for length in range(BLOCK_SIZE, len(tokens) + 1, BLOCK_SIZE):
        prefix = tuple(tokens[:length])

        if prefix in prefix_cache:
            cached_tokens = length
        else:
            break

    reused = cached_tokens
    computed = len(tokens) - reused

    print("Prompt tokens:", len(tokens))
    print("Reused tokens:", reused)
    print("New prefill tokens:", computed)

    # Cache complete blocks of the resulting prompt
    for length in range(BLOCK_SIZE, len(tokens) + 1, BLOCK_SIZE):
        prefix = tuple(tokens[:length])
        prefix_cache[prefix] = f"KV_for_prefix_{length}"

    print("Total cached prefixes:", len(prefix_cache))

for request_id, tokens in requests.items():
    process_request(request_id, tokens)
