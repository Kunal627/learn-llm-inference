
from statistics import mean

# Different prompt lengths
prompt_lengths = [100, 200, 500, 150, 300, 80, 450, 120]

batch_size = 4


def calculate_padding_metrics(lengths):

    max_length = max(lengths)

    useful_tokens = sum(lengths)

    allocated_tokens = len(lengths) * max_length

    padding_tokens = allocated_tokens - useful_tokens

    efficiency = useful_tokens / allocated_tokens

    return {
        "requests": len(lengths),
        "max_sequence_length": max_length,
        "useful_tokens": useful_tokens,
        "allocated_tokens": allocated_tokens,
        "padding_tokens": padding_tokens,
        "efficiency": efficiency * 100,
    }


def run_batches(lengths, batch_size):

    batches = [
        lengths[i:i + batch_size]
        for i in range(0, len(lengths), batch_size)
    ]

    total_useful = 0
    total_allocated = 0

    print("\nBATCH RESULTS")

    for i, batch in enumerate(batches, start=1):

        metrics = calculate_padding_metrics(batch)

        total_useful += metrics["useful_tokens"]
        total_allocated += metrics["allocated_tokens"]

        print(f"\nBatch {i}: {batch}")

        for key, value in metrics.items():
            print(f"{key}: {value}")

    overall_efficiency = (
        total_useful / total_allocated
    ) * 100

    print("\nOVERALL RESULTS")
    print("Useful tokens:", total_useful)
    print("Allocated tokens:", total_allocated)
    print("Padding tokens:", total_allocated - total_useful)
    print("Padding efficiency:", overall_efficiency)


# ============================================================
# EXPERIMENT 1: FIFO BATCHING
# ============================================================

print("\n========== FIFO BATCHING ==========")

run_batches(prompt_lengths, batch_size)


# ============================================================
# EXPERIMENT 2: LENGTH BUCKETING
# ============================================================

print("\n========== LENGTH BUCKETING ==========")

sorted_lengths = sorted(prompt_lengths)

run_batches(sorted_lengths, batch_size)
