
from collections import deque
from statistics import mean

# ============================================================
# WORKLOAD
# ============================================================

REQUESTS = [
    {"id": "A", "arrival": 0,  "prompt": 100, "output": 3},
    {"id": "B", "arrival": 0,  "prompt": 200, "output": 6},
    {"id": "C", "arrival": 2,  "prompt": 150, "output": 4},
    {"id": "D", "arrival": 4,  "prompt": 80,  "output": 3},
    {"id": "E", "arrival": 5,  "prompt": 120, "output": 5},
    {"id": "F", "arrival": 7,  "prompt": 90,  "output": 2},
    {"id": "G", "arrival": 8,  "prompt": 300, "output": 7},
    {"id": "H", "arrival": 10, "prompt": 110, "output": 4},
]

MAX_ACTIVE_SEQUENCES = 2


# ============================================================
# METRICS
# ============================================================


def calculate_metrics(results, makespan, max_active):
    total_requests = len(results)
    total_tokens = sum(r["output"] for r in results)

    completion_latencies = [
        r["completion"] - r["arrival"]
        for r in results
    ]

    ttft_values = [
        r["first_token"] - r["arrival"]
        for r in results
    ]

    # Same denominator for both scheduling strategies
    total_available_slot_time = max_active * makespan

    return {
        "makespan": makespan,
        "avg_completion_latency": mean(completion_latencies),
        "avg_ttft": mean(ttft_values),
        "request_throughput": total_requests / makespan,
        "token_throughput": total_tokens / makespan,
        "slot_efficiency": (
            total_tokens / total_available_slot_time
        ),
        "total_tokens": total_tokens,
    }



# ============================================================
# STATIC BATCHING
# ============================================================

def run_static_batching(requests, batch_size):
    print("\n" + "=" * 65)
    print("STATIC BATCHING")
    print("=" * 65)

    results = []
    batch_start = 0
    total_slot_time = 0

    # Fixed FIFO groups
    batches = [
        requests[i:i + batch_size]
        for i in range(0, len(requests), batch_size)
    ]

    for batch_number, batch in enumerate(batches, start=1):

        # Batch waits until all its requests have arrived
        latest_arrival = max(r["arrival"] for r in batch)
        start = max(batch_start, latest_arrival)

        longest_output = max(r["output"] for r in batch)
        batch_end = start + longest_output

        print(
            f"\nBatch {batch_number}: "
            f"{[r['id'] for r in batch]}"
        )
        print(f"  Start: {start}")
        print(f"  Batch completion: {batch_end}")

        # All requests start decoding together.
        # A finished request's slot remains reserved
        # until the longest request in this batch finishes.
        for request in batch:
            first_token = start + 1
            completion = start + request["output"]

            results.append({
                **request,
                "first_token": first_token,
                "completion": completion,
            })

            print(
                f"  Request {request['id']}: "
                f"first token={first_token}, "
                f"completion={completion}"
            )

        total_slot_time += batch_size * longest_output
        batch_start = batch_end

    makespan = max(r["completion"] for r in results)

    metrics = calculate_metrics(
        results,
        makespan,
        MAX_ACTIVE_SEQUENCES
    )

    return results, metrics


# ============================================================
# CONTINUOUS BATCHING
# ============================================================

def run_continuous_batching(requests, max_active):
    print("\n" + "=" * 65)
    print("CONTINUOUS BATCHING")
    print("=" * 65)

    waiting_queue = deque()
    active = {}
    results = []

    next_request = 0
    current_time = 0
    total_slot_time = 0

    while (
        next_request < len(requests)
        or waiting_queue
        or active
    ):

        # 1. Add requests that have arrived
        while (
            next_request < len(requests)
            and requests[next_request]["arrival"] <= current_time
        ):
            request = requests[next_request]
            waiting_queue.append(request)

            print(
                f"Time {current_time}: "
                f"Request {request['id']} arrived"
            )

            next_request += 1

        # 2. Admit requests into available slots
        while (
            len(active) < max_active
            and waiting_queue
        ):
            request = waiting_queue.popleft()

            active[request["id"]] = {
                **request,
                "remaining": request["output"],
                "first_token": None,
            }

            print(
                f"Time {current_time}: "
                f"Request {request['id']} admitted"
            )

        # 3. Decode one token per active request
        if active:
            print(
                f"Time {current_time}: "
                f"Decoding {list(active.keys())}"
            )

        completed_ids = []

        for request_id, state in active.items():

            if state["first_token"] is None:
                state["first_token"] = current_time + 1

            state["remaining"] -= 1

            if state["remaining"] == 0:
                state["completion"] = current_time + 1
                completed_ids.append(request_id)

        total_slot_time += len(active)

        # 4. Remove completed requests
        for request_id in completed_ids:
            state = active.pop(request_id)

            results.append(state)

            print(
                f"Time {current_time + 1}: "
                f"Request {request_id} completed"
            )

        # 5. Advance one decode iteration
        current_time += 1

    makespan = max(r["completion"] for r in results)

    metrics = calculate_metrics(
        results,
        makespan,
        MAX_ACTIVE_SEQUENCES
    )

    return results, metrics


# ============================================================
# REPORT
# ============================================================

def print_request_results(results):
    print(
        f"\n{'ID':<5}"
        f"{'Arrival':>9}"
        f"{'Output':>9}"
        f"{'First token':>14}"
        f"{'Completion':>13}"
        f"{'Latency':>10}"
        f"{'TTFT':>8}"
    )

    print("-" * 68)

    for r in sorted(results, key=lambda x: x["id"]):
        latency = r["completion"] - r["arrival"]
        ttft = r["first_token"] - r["arrival"]

        print(
            f"{r['id']:<5}"
            f"{r['arrival']:>9}"
            f"{r['output']:>9}"
            f"{r['first_token']:>14}"
            f"{r['completion']:>13}"
            f"{latency:>10}"
            f"{ttft:>8}"
        )


def print_comparison(static_metrics, continuous_metrics):
    print("\n" + "=" * 75)
    print("FINAL COMPARISON")
    print("=" * 75)

    metrics = [
        ("Makespan", "makespan", "time units"),
        ("Average completion latency",
         "avg_completion_latency", "time units"),
        ("Average TTFT", "avg_ttft", "time units"),
        ("Request throughput",
         "request_throughput", "requests/time unit"),
        ("Generated token throughput",
         "token_throughput", "tokens/time unit"),
        ("Useful slot efficiency",
         "slot_efficiency", "%"),
    ]

    print(
        f"\n{'Metric':<32}"
        f"{'Static':>17}"
        f"{'Continuous':>17}"
    )
    print("-" * 66)

    for label, key, unit in metrics:
        s = static_metrics[key]
        c = continuous_metrics[key]

        if key == "slot_efficiency":
            s *= 100
            c *= 100

        print(f"{label:<32}{s:>17.3f}{c:>17.3f}")

    print("\nInterpretation:")
    print("  Lower makespan, latency and TTFT are better.")
    print("  Higher throughput and slot efficiency are better.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    static_results, static_metrics = run_static_batching(
        REQUESTS,
        MAX_ACTIVE_SEQUENCES
    )

    continuous_results, continuous_metrics = (
        run_continuous_batching(
            REQUESTS,
            MAX_ACTIVE_SEQUENCES
        )
    )

    print("\n\nSTATIC REQUEST RESULTS")
    print_request_results(static_results)

    print("\n\nCONTINUOUS REQUEST RESULTS")
    print_request_results(continuous_results)

    print_comparison(
        static_metrics,
        continuous_metrics
    )
