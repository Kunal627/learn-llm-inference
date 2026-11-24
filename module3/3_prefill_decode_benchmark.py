"""
============================================================
MODULE 3 - LESSON 3.3.1
PREFILL AND DECODE PERFORMANCE BENCHMARK
============================================================

Educational PyTorch Transformer inference benchmark.

Experiments:
    A. Change prompt length, keep output length fixed.
    B. Change output length, keep prompt length fixed.

Measurements:
    - Prefill latency
    - Decode latency
    - Total generation latency
    - Average decode latency per token
    - Decode throughput
    - Total generation throughput
    - P50 and P95 latency

Important:
    - No KV cache.
    - Greedy decoding.
    - First generated token comes from prefill.
    - Subsequent tokens use naive full-sequence decoding.
"""

import time
import math
import statistics

import torch

if __package__:
    from .tiny_llm import TinyLLM, append_token, select_next_token
else:
    from tiny_llm import TinyLLM, append_token, select_next_token


# ============================================================
# 1. CONFIGURATION
# ============================================================

BATCH_SIZE = 1

VOCAB_SIZE = 1000

D_MODEL = 64
NUM_HEADS = 4
HEAD_DIM = D_MODEL // NUM_HEADS

NUM_LAYERS = 2
MLP_DIM = 4 * D_MODEL

MAX_SEQ_LEN = 1024

WARMUP = 3
ITERATIONS = 10

PROMPT_LENGTHS = [8, 32, 128, 256, 512]
OUTPUT_LENGTHS = [10, 20, 50, 100]

FIXED_OUTPUT_LENGTH = 20
FIXED_PROMPT_LENGTH = 128

assert D_MODEL % NUM_HEADS == 0

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

print("=" * 75)
print("LESSON 3.3.1 - PREFILL AND DECODE BENCHMARK")
print("=" * 75)

print("Device:", device)
print("CUDA available:", torch.cuda.is_available())
print("Batch size:", BATCH_SIZE)
print("Vocabulary:", VOCAB_SIZE)
print("Model dimension:", D_MODEL)
print("Attention heads:", NUM_HEADS)
print("Head dimension:", HEAD_DIM)
print("Transformer layers:", NUM_LAYERS)


# ============================================================
# 5. INITIALIZE MODEL
# ============================================================

model = TinyLLM(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    max_seq_len=MAX_SEQ_LEN,
    mlp_dim=MLP_DIM
).to(device)

model.eval()

total_params = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("\nModel parameters:", f"{total_params:,}")


# ============================================================
# 8. SYNCHRONIZATION AND STATISTICS
# ============================================================

def synchronize():

    if device.type == "cuda":
        torch.cuda.synchronize()


def percentile(values, p):

    sorted_values = sorted(values)

    index = math.ceil(
        (p / 100) * len(sorted_values)
    ) - 1

    index = max(0, index)

    return sorted_values[index]


def calculate_statistics(values):

    return {
        "median": statistics.median(values),
        "p95": percentile(values, 95)
    }


# ============================================================
# 9. PREFILL BENCHMARK
# ============================================================

def benchmark_prefill(
    model,
    input_ids,
    warmup=WARMUP,
    iterations=ITERATIONS
):
    """
    Measures prompt processing.

    Input:
        [B, S]

    Output:
        Logits [B, S, VOCAB_SIZE]
    """

    model.eval()

    # Warmup
    with torch.inference_mode():

        for _ in range(warmup):

            _ = model(input_ids)

    synchronize()

    timings = []

    with torch.inference_mode():

        for _ in range(iterations):

            synchronize()

            start = time.perf_counter()

            logits = model(input_ids)

            synchronize()

            end = time.perf_counter()

            timings.append(
                (end - start) * 1000
            )

    stats = calculate_statistics(timings)

    batch_size, seq_len = input_ids.shape

    prompt_tokens = batch_size * seq_len

    throughput = (
        prompt_tokens / (stats["median"] / 1000)
    )

    return {
        "median_ms": stats["median"],
        "p95_ms": stats["p95"],
        "throughput": throughput,
        "logits_shape": tuple(logits.shape)
    }


# ============================================================
# 10. ONE COMPLETE GENERATION REQUEST
# ============================================================

def run_generation(
    model,
    input_ids,
    max_new_tokens
):
    """
    Measures:

        Prefill:
            Initial complete prompt forward pass.

        Decode:
            Subsequent full-sequence forward passes.

    The first generated token comes from prefill.

    No KV cache is used.
    """

    model.eval()

    generated_ids = input_ids.clone()

    prefill_ms = 0.0
    decode_step_times = []

    with torch.inference_mode():

        # ----------------------------------------
        # PREFILL
        # ----------------------------------------

        synchronize()

        start = time.perf_counter()

        logits = model(generated_ids)

        synchronize()

        end = time.perf_counter()

        prefill_ms = (end - start) * 1000

        # First generated token
        next_token = select_next_token(
            logits[:, -1, :]
        )

        generated_ids = append_token(
            generated_ids,
            next_token
        )

        # ----------------------------------------
        # DECODE
        # ----------------------------------------

        for step in range(max_new_tokens - 1):

            synchronize()

            start = time.perf_counter()

            # Recompute entire sequence
            logits = model(generated_ids)

            # Select next token
            next_token = select_next_token(
                logits[:, -1, :]
            )

            # Append token
            generated_ids = append_token(
                generated_ids,
                next_token
            )

            synchronize()

            end = time.perf_counter()

            step_time_ms = (
                end - start
            ) * 1000

            decode_step_times.append(step_time_ms)

    decode_ms = sum(decode_step_times)

    total_ms = prefill_ms + decode_ms

    return {
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "total_ms": total_ms,
        "decode_step_times": decode_step_times,
        "final_shape": tuple(generated_ids.shape)
    }


# ============================================================
# 11. DECODE BENCHMARK
# ============================================================

def benchmark_decode(
    model,
    input_ids,
    max_new_tokens,
    warmup=WARMUP,
    iterations=ITERATIONS
):

    if max_new_tokens < 1:
        raise ValueError(
            "max_new_tokens must be at least 1"
        )

    # Check maximum context length
    prompt_length = input_ids.shape[1]

    if prompt_length + max_new_tokens > MAX_SEQ_LEN:
        raise ValueError(
            "Prompt length + output length exceeds MAX_SEQ_LEN"
        )

    # Warmup
    for _ in range(warmup):

        _ = run_generation(
            model,
            input_ids,
            max_new_tokens
        )

    # Actual measurements
    prefill_times = []
    decode_times = []
    total_times = []
    all_decode_step_times = []

    for _ in range(iterations):

        result = run_generation(
            model,
            input_ids,
            max_new_tokens
        )

        prefill_times.append(
            result["prefill_ms"]
        )

        decode_times.append(
            result["decode_ms"]
        )

        total_times.append(
            result["total_ms"]
        )

        all_decode_step_times.extend(
            result["decode_step_times"]
        )

    prefill_stats = calculate_statistics(
        prefill_times
    )

    decode_stats = calculate_statistics(
        decode_times
    )

    total_stats = calculate_statistics(
        total_times
    )

    batch_size = input_ids.shape[0]

    # First token comes from prefill.
    subsequent_tokens = max_new_tokens - 1

    if subsequent_tokens > 0:

        avg_decode_ms = (
            decode_stats["median"] / subsequent_tokens
        )

        decode_throughput = (
            batch_size * subsequent_tokens
            / (decode_stats["median"] / 1000)
        )

        step_p50 = statistics.median(
            all_decode_step_times
        )

        step_p95 = percentile(
            all_decode_step_times,
            95
        )

    else:

        avg_decode_ms = 0.0
        decode_throughput = 0.0
        step_p50 = 0.0
        step_p95 = 0.0

    total_throughput = (
        batch_size * max_new_tokens
        / (total_stats["median"] / 1000)
    )

    return {
        "prefill_median_ms": prefill_stats["median"],
        "prefill_p95_ms": prefill_stats["p95"],

        "decode_median_ms": decode_stats["median"],
        "decode_p95_ms": decode_stats["p95"],

        "total_median_ms": total_stats["median"],
        "total_p95_ms": total_stats["p95"],

        "avg_decode_ms": avg_decode_ms,

        "decode_throughput": decode_throughput,
        "total_throughput": total_throughput,

        "decode_step_p50_ms": step_p50,
        "decode_step_p95_ms": step_p95
    }


# ============================================================
# 12. EXPERIMENT A
# PROMPT LENGTH VARIATION
# ============================================================

def experiment_prompt_length():

    print("\n")
    print("=" * 120)
    print("EXPERIMENT A - PROMPT LENGTH VARIATION")
    print("=" * 120)

    print(
        f"{'Prompt':>8} | "
        f"{'Prefill P50':>12} | "
        f"{'Prefill P95':>12} | "
        f"{'Decode P50':>12} | "
        f"{'Decode ms/token':>16} | "
        f"{'Decode tok/s':>14} | "
        f"{'Total ms':>12}"
    )

    print("-" * 120)

    results = []

    for prompt_length in PROMPT_LENGTHS:

        input_ids = torch.randint(
            0,
            VOCAB_SIZE,
            (BATCH_SIZE, prompt_length),
            device=device
        )

        result = benchmark_decode(
            model=model,
            input_ids=input_ids,
            max_new_tokens=FIXED_OUTPUT_LENGTH
        )

        results.append(result)

        print(
            f"{prompt_length:8d} | "
            f"{result['prefill_median_ms']:12.3f} | "
            f"{result['prefill_p95_ms']:12.3f} | "
            f"{result['decode_median_ms']:12.3f} | "
            f"{result['avg_decode_ms']:16.3f} | "
            f"{result['decode_throughput']:14.2f} | "
            f"{result['total_median_ms']:12.3f}"
        )

    return results


# ============================================================
# 13. EXPERIMENT B
# OUTPUT LENGTH VARIATION
# ============================================================

def experiment_output_length():

    print("\n")
    print("=" * 120)
    print("EXPERIMENT B - OUTPUT LENGTH VARIATION")
    print("=" * 120)

    input_ids = torch.randint(
        0,
        VOCAB_SIZE,
        (BATCH_SIZE, FIXED_PROMPT_LENGTH),
        device=device
    )

    print(
        f"{'Output':>8} | "
        f"{'Prefill P50':>12} | "
        f"{'Decode P50':>12} | "
        f"{'Decode ms/token':>16} | "
        f"{'Decode tok/s':>14} | "
        f"{'Total ms':>12} | "
        f"{'Total tok/s':>14}"
    )

    print("-" * 120)

    results = []

    for output_length in OUTPUT_LENGTHS:

        result = benchmark_decode(
            model=model,
            input_ids=input_ids,
            max_new_tokens=output_length
        )

        results.append(result)

        print(
            f"{output_length:8d} | "
            f"{result['prefill_median_ms']:12.3f} | "
            f"{result['decode_median_ms']:12.3f} | "
            f"{result['avg_decode_ms']:16.3f} | "
            f"{result['decode_throughput']:14.2f} | "
            f"{result['total_median_ms']:12.3f} | "
            f"{result['total_throughput']:14.2f}"
        )

    return results


# ============================================================
# 14. STANDALONE PREFILL EXPERIMENT
# ============================================================

def experiment_standalone_prefill():

    print("\n")
    print("=" * 90)
    print("STANDALONE PREFILL BENCHMARK")
    print("=" * 90)

    print(
        f"{'Prompt':>8} | "
        f"{'P50 ms':>12} | "
        f"{'P95 ms':>12} | "
        f"{'Tokens/sec':>14} | "
        f"{'Logits shape':>20}"
    )

    print("-" * 90)

    results = []

    for prompt_length in PROMPT_LENGTHS:

        input_ids = torch.randint(
            0,
            VOCAB_SIZE,
            (BATCH_SIZE, prompt_length),
            device=device
        )

        result = benchmark_prefill(
            model,
            input_ids
        )

        results.append(result)

        print(
            f"{prompt_length:8d} | "
            f"{result['median_ms']:12.3f} | "
            f"{result['p95_ms']:12.3f} | "
            f"{result['throughput']:14.2f} | "
            f"{str(result['logits_shape']):>20}"
        )

    return results


# ============================================================
# 15. MAIN
# ============================================================

def main():

    print("\nStarting benchmark experiments...")

    if device.type == "cuda":

        print("\nGPU:", torch.cuda.get_device_name(0))

        print(
            "GPU memory allocated:",
            round(
                torch.cuda.memory_allocated() / (1024 ** 2),
                2
            ),
            "MB"
        )

    # Experiment 1: standalone prefill
    prefill_results = experiment_standalone_prefill()

    # Experiment 2: change prompt length
    prompt_results = experiment_prompt_length()

    # Experiment 3: change output length
    output_results = experiment_output_length()

    print("\n")
    print("=" * 75)
    print("ALL BENCHMARKS COMPLETED")
    print("=" * 75)

    print("Device:", device)
    print("Model parameters:", f"{total_params:,}")
    print("Batch size:", BATCH_SIZE)
    print("KV cache: Disabled")
    print("Decoding: Greedy")
    print("Benchmark iterations:", ITERATIONS)

    print("\nResults collected:")
    print("Standalone prefill:", len(prefill_results))
    print("Prompt length experiments:", len(prompt_results))
    print("Output length experiments:", len(output_results))


if __name__ == "__main__":

    main()
