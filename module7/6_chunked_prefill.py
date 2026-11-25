
from dataclasses import dataclass


# ============================================================
# CONFIGURATION
# ============================================================

MAX_ACTIVE_SEQUENCES = 2
MAX_BATCHED_TOKENS = 32
PREFILL_CHUNK_SIZE = 16

REQUESTS = [
    {"id": "A", "arrival": 0, "prompt": 100, "output": 5},
    {"id": "B", "arrival": 0, "prompt": 200, "output": 4},
    {"id": "C", "arrival": 2, "prompt": 100, "output": 3},
    {"id": "D", "arrival": 4, "prompt": 50, "output": 4},
]


# ============================================================
# REQUEST STATE
# ============================================================

@dataclass
class Request:
    request_id: str
    arrival: int
    prompt_length: int
    output_length: int

    state: str = "WAITING"

    prefilled_tokens: int = 0
    generated_tokens: int = 0

    kv_cache_length: int = 0


requests = [
    Request(
        request_id=r["id"],
        arrival=r["arrival"],
        prompt_length=r["prompt"],
        output_length=r["output"],
    )
    for r in REQUESTS
]

waiting = []
active = []

iteration = 0


# ============================================================
# SCHEDULER
# ============================================================

while any(r.state != "COMPLETED" for r in requests):

    iteration += 1
    token_budget = MAX_BATCHED_TOKENS

    print("\n" + "=" * 60)
    print(f"ITERATION {iteration}")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Add newly arrived requests to waiting queue
    # --------------------------------------------------------

    for req in requests:
        if (
            req.arrival == iteration - 1
            and req.state == "WAITING"
            and req not in waiting
            and req not in active
        ):
            waiting.append(req)
            print(f"{req.request_id}: ARRIVED")

    # --------------------------------------------------------
    # 2. Admit requests if capacity is available
    # --------------------------------------------------------

    while (
        len(active) < MAX_ACTIVE_SEQUENCES
        and waiting
    ):
        req = waiting.pop(0)

        req.state = "PREFILL"
        active.append(req)

        print(f"{req.request_id}: ADMITTED")

    # --------------------------------------------------------
    # 3. Schedule decode work first
    # --------------------------------------------------------

    for req in list(active):

        if req.state != "DECODE":
            continue

        if token_budget < 1:
            break

        # One new token enters the model.
        # Its KV entry is appended to the cache.
        req.kv_cache_length += 1

        # The model produces logits and the engine
        # samples the next output token.
        req.generated_tokens += 1

        token_budget -= 1

        print(
            f"{req.request_id}: DECODE 1 token | "
            f"Generated = {req.generated_tokens}/"
            f"{req.output_length} | "
            f"KV cache = {req.kv_cache_length}"
        )

        if req.generated_tokens >= req.output_length:

            req.state = "COMPLETED"
            active.remove(req)

            print(f"{req.request_id}: COMPLETED")

    # --------------------------------------------------------
    # 4. Schedule prefill work with remaining budget
    # --------------------------------------------------------

    for req in list(active):

        if req.state != "PREFILL":
            continue

        if token_budget <= 0:
            break

        remaining = (
            req.prompt_length - req.prefilled_tokens
        )

        chunk = min(
            remaining,
            PREFILL_CHUNK_SIZE,
            token_budget,
        )

        req.prefilled_tokens += chunk
        req.kv_cache_length += chunk

        token_budget -= chunk

        print(
            f"{req.request_id}: PREFILL {chunk} tokens | "
            f"Progress = {req.prefilled_tokens}/"
            f"{req.prompt_length} | "
            f"KV cache = {req.kv_cache_length}"
        )

        if req.prefilled_tokens == req.prompt_length:

            req.state = "DECODE"

            # First output token is sampled using
            # logits from the final prompt position.
            req.generated_tokens = 1

            print(
                f"{req.request_id}: PREFILL COMPLETED | "
                f"First output token sampled"
            )

            if req.generated_tokens >= req.output_length:

                req.state = "COMPLETED"
                active.remove(req)

                print(f"{req.request_id}: COMPLETED")

    # --------------------------------------------------------
    # 5. Display scheduler state
    # --------------------------------------------------------

    print("\nACTIVE REQUESTS")

    for req in active:
        print(
            f"{req.request_id} | "
            f"State={req.state} | "
            f"KV={req.kv_cache_length} | "
            f"Prefilled={req.prefilled_tokens} | "
            f"Generated={req.generated_tokens}"
        )

    print(
        "Waiting:",
        [req.request_id for req in waiting]
    )

    print("Unused token budget:", token_budget)

    # --------------------------------------------------------
    # 6. Safety guard
    # --------------------------------------------------------

    if iteration > 1000:
        raise RuntimeError("Scheduler did not finish")


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL REQUEST RESULTS")
print("=" * 60)

for req in requests:

    print(
        f"{req.request_id} | "
        f"Prompt={req.prompt_length} | "
        f"Output={req.generated_tokens} | "
        f"Final KV length={req.kv_cache_length} | "
        f"State={req.state}"
    )

print("\nTotal iterations:", iteration)
print("All requests completed:", all(
    req.state == "COMPLETED" for req in requests
))
