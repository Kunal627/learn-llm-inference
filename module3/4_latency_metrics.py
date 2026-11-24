
import time
import statistics
import torch

from tiny_llm import TinyLLM, append_token, select_next_token


# ============================================================
# 1. CONFIGURATION
# ============================================================

BATCH_SIZE = 1
PROMPT_LEN = 1024
MAX_NEW_TOKENS = 50

VOCAB_SIZE = 1000
D_MODEL = 512
NUM_HEADS = 8
NUM_LAYERS = 8
MAX_SEQ_LEN = 2048

WARMUP_RUNS = 5
BENCHMARK_RUNS = 30

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

torch.manual_seed(42)

if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(42)

print("=" * 65)
print("LLM LATENCY METRICS BENCHMARK")
print("=" * 65)
print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 5. UTILITY FUNCTIONS
# ============================================================

def synchronize():

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def percentile(values, p):

    values = torch.tensor(
        values, dtype=torch.float64
    )

    return torch.quantile(
        values, p / 100
    ).item()


# ============================================================
# 6. GENERATE ONE REQUEST AND MEASURE LATENCIES
# ============================================================

def run_one_request(model, prompt_ids):

    synchronize()

    request_start = time.perf_counter()

    token_timestamps = []
    generated_tokens = []

    # --------------------------------------------------------
    # PREFILL
    # --------------------------------------------------------

    logits = model(prompt_ids)

    # First generated token comes from prefill logits
    next_token = select_next_token(
        logits[:, -1, :]
    )

    generated_tokens.append(next_token)

    synchronize()

    first_token_time = time.perf_counter()

    token_timestamps.append(first_token_time)

    # --------------------------------------------------------
    # DECODE
    # --------------------------------------------------------

    input_ids = append_token(prompt_ids, next_token)

    for step in range(MAX_NEW_TOKENS - 1):

        # Naive decoder: recomputes the full sequence
        logits = model(input_ids)

        next_token = select_next_token(
            logits[:, -1, :]
        )

        generated_tokens.append(next_token)

        input_ids = append_token(input_ids, next_token)

        synchronize()

        token_time = time.perf_counter()

        token_timestamps.append(token_time)

    # --------------------------------------------------------
    # CALCULATE METRICS
    # --------------------------------------------------------

    request_end = token_timestamps[-1]

    ttft_ms = (
        first_token_time - request_start
    ) * 1000

    end_to_end_ms = (
        request_end - request_start
    ) * 1000

    # Time between consecutive tokens
    itl_ms = []

    for i in range(1, len(token_timestamps)):

        interval = (
            token_timestamps[i]
            - token_timestamps[i - 1]
        ) * 1000

        itl_ms.append(interval)

    # Average time per token after first token
    tpot_ms = statistics.mean(itl_ms)

    # Total output tokens / total request latency
    output_tokens_per_second = (
        MAX_NEW_TOKENS / (end_to_end_ms / 1000)
    )

    # Decode-only throughput excludes first token
    decode_tokens_per_second = (
        (MAX_NEW_TOKENS - 1)
        / ((end_to_end_ms - ttft_ms) / 1000)
    )

    return {
        "ttft_ms": ttft_ms,
        "itl_ms": itl_ms,
        "tpot_ms": tpot_ms,
        "e2e_ms": end_to_end_ms,
        "output_tokens_per_second":
            output_tokens_per_second,
        "decode_tokens_per_second":
            decode_tokens_per_second,
        "generated_tokens": generated_tokens
    }


# ============================================================
# 7. MAIN BENCHMARK
# ============================================================

def main():

    model = TinyLLM(
        VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LEN
    ).to(DEVICE)
    model.eval()

    parameter_count = sum(
        p.numel() for p in model.parameters()
    )

    print("\nModel parameters:", f"{parameter_count:,}")
    print("Batch size:", BATCH_SIZE)
    print("Prompt length:", PROMPT_LEN)
    print("Output length:", MAX_NEW_TOKENS)

    prompt_ids = torch.randint(
        0,
        VOCAB_SIZE,
        (BATCH_SIZE, PROMPT_LEN),
        device=DEVICE
    )

    # --------------------------------------------------------
    # WARMUP
    # --------------------------------------------------------

    print("\nRunning warmup...")

    with torch.inference_mode():

        for _ in range(WARMUP_RUNS):
            run_one_request(model, prompt_ids)

    # --------------------------------------------------------
    # BENCHMARK
    # --------------------------------------------------------

    print("\nRunning benchmark...")

    results = []

    with torch.inference_mode():

        for run in range(BENCHMARK_RUNS):

            result = run_one_request(
                model, prompt_ids
            )

            results.append(result)

            print(
                f"Run {run + 1:02d}: "
                f"TTFT={result['ttft_ms']:.3f} ms | "
                f"TPOT={result['tpot_ms']:.3f} ms | "
                f"E2E={result['e2e_ms']:.3f} ms"
            )

    # --------------------------------------------------------
    # COLLECT METRICS
    # --------------------------------------------------------

    ttft_values = [
        r["ttft_ms"] for r in results
    ]

    tpot_values = [
        r["tpot_ms"] for r in results
    ]

    e2e_values = [
        r["e2e_ms"] for r in results
    ]

    all_itl_values = [
        interval
        for r in results
        for interval in r["itl_ms"]
    ]

    output_throughput = [
        r["output_tokens_per_second"]
        for r in results
    ]

    decode_throughput = [
        r["decode_tokens_per_second"]
        for r in results
    ]

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("FINAL LATENCY REPORT")
    print("=" * 75)

    print(
        f"{'Metric':<30}"
        f"{'P50':>15}"
        f"{'P95':>15}"
    )

    print("-" * 60)

    def print_metric(name, values):

        print(
            f"{name:<30}"
            f"{percentile(values, 50):>12.3f} ms"
            f"{percentile(values, 95):>12.3f} ms"
        )

    print_metric("TTFT", ttft_values)
    print_metric("ITL", all_itl_values)
    print_metric("TPOT", tpot_values)
    print_metric("End-to-end latency", e2e_values)

    print("-" * 60)

    print(
        f"Average output throughput: "
        f"{statistics.mean(output_throughput):.2f} tokens/sec"
    )

    print(
        f"Average decode throughput: "
        f"{statistics.mean(decode_throughput):.2f} tokens/sec"
    )

    print("\nExample generated token IDs:")

    print(
        results[-1]["generated_tokens"]
    )

    print("\nBenchmark completed.")


if __name__ == "__main__":
    main()
