
import time
import statistics
import torch

from tiny_llm import TinyLLM, select_next_token


# ============================================================
# 1. CONFIGURATION
# ============================================================

BATCH_SIZE = 1
PROMPT_LEN = 512
OUTPUT_LEN = 50

VOCAB_SIZE = 1000
D_MODEL = 64
NUM_HEADS = 4
NUM_LAYERS = 2
MAX_SEQ_LEN = 1024

WARMUP_RUNS = 3
BENCHMARK_RUNS = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

assert D_MODEL % NUM_HEADS == 0
assert PROMPT_LEN + OUTPUT_LEN <= MAX_SEQ_LEN

HEAD_DIM = D_MODEL // NUM_HEADS


# ============================================================
# 2. DEVICE SYNCHRONIZATION
# ============================================================

def synchronize():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


# ============================================================
# 7. NAIVE DECODING
# ============================================================

@torch.inference_mode()
def generate_naive(model, prompt_ids):

    input_ids = prompt_ids.clone()

    synchronize()
    request_start = time.perf_counter()

    # Prefill: process the entire prompt

    logits = model(input_ids)

    # First generated token

    next_token = select_next_token(logits[:, -1, :])

    generated = [next_token]

    synchronize()
    first_token_time = time.perf_counter()

    # Decode: recompute the entire growing sequence

    for _ in range(OUTPUT_LEN - 1):

        input_ids = torch.cat(
            [input_ids, next_token], dim=1
        )

        logits = model(input_ids)

        next_token = select_next_token(
            logits[:, -1, :]
        )

        generated.append(next_token)

    synchronize()
    end_time = time.perf_counter()

    return {
        "tokens": torch.cat(generated, dim=1),
        "ttft_ms": (
            first_token_time - request_start
        ) * 1000,
        "decode_ms": (
            end_time - first_token_time
        ) * 1000,
        "e2e_ms": (
            end_time - request_start
        ) * 1000,
        "cache_bytes": 0
    }


# ============================================================
# 8. KV-CACHED DECODING
# ============================================================

@torch.inference_mode()
def generate_cached(model, prompt_ids):

    synchronize()
    request_start = time.perf_counter()

    # Prefill: process the entire prompt and build KV cache

    logits, past_key_values = model(
        prompt_ids,
        use_cache=True
    )

    # First generated token

    next_token = select_next_token(logits[:, -1, :])

    generated = [next_token]

    synchronize()
    first_token_time = time.perf_counter()

    # Decode: process ONLY the newest token

    for _ in range(OUTPUT_LEN - 1):

        logits, past_key_values = model(
            next_token,
            past_key_values=past_key_values,
            use_cache=True
        )

        next_token = select_next_token(
            logits[:, -1, :]
        )

        generated.append(next_token)

    synchronize()
    end_time = time.perf_counter()

    # Calculate actual cache memory

    cache_bytes = 0

    for key, value in past_key_values:

        cache_bytes += key.numel() * key.element_size()
        cache_bytes += value.numel() * value.element_size()

    return {
        "tokens": torch.cat(generated, dim=1),
        "ttft_ms": (
            first_token_time - request_start
        ) * 1000,
        "decode_ms": (
            end_time - first_token_time
        ) * 1000,
        "e2e_ms": (
            end_time - request_start
        ) * 1000,
        "cache_bytes": cache_bytes
    }


# ============================================================
# 9. STATISTICS
# ============================================================

def percentile(values, p):

    sorted_values = sorted(values)

    index = round((p / 100) * (len(sorted_values) - 1))

    return sorted_values[index]


def summarize(results):

    ttft = [r["ttft_ms"] for r in results]
    decode = [r["decode_ms"] for r in results]
    e2e = [r["e2e_ms"] for r in results]

    tpot = [
        d / (OUTPUT_LEN - 1)
        for d in decode
    ]

    return {
        "TTFT P50": statistics.median(ttft),
        "TTFT P95": percentile(ttft, 95),

        "TPOT P50": statistics.median(tpot),
        "TPOT P95": percentile(tpot, 95),

        "E2E P50": statistics.median(e2e),
        "E2E P95": percentile(e2e, 95),

        "Decode P50": statistics.median(decode)
    }


# ============================================================
# 10. MAIN BENCHMARK
# ============================================================

def main():

    print("=" * 65)
    print("NAIVE DECODING VS KV-CACHED DECODING")
    print("=" * 65)

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    model = TinyLLM(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
    ).to(DEVICE)
    model.eval()

    parameter_count = sum(
        p.numel() for p in model.parameters()
    )

    print(f"Model parameters: {parameter_count:,}")
    print("Batch size:", BATCH_SIZE)
    print("Prompt length:", PROMPT_LEN)
    print("Output length:", OUTPUT_LEN)
    print("Layers:", NUM_LAYERS)
    print("Attention heads:", NUM_HEADS)
    print("Head dimension:", HEAD_DIM)

    prompt_ids = torch.randint(
        0,
        VOCAB_SIZE,
        (BATCH_SIZE, PROMPT_LEN),
        device=DEVICE
    )

    # --------------------------------------------------------
    # WARMUP
    # --------------------------------------------------------

    print("\nWarming up...")

    for _ in range(WARMUP_RUNS):

        generate_naive(model, prompt_ids)
        generate_cached(model, prompt_ids)

    synchronize()

    # --------------------------------------------------------
    # CORRECTNESS TEST
    # --------------------------------------------------------

    print("\nChecking generated token equality...")

    naive_result = generate_naive(model, prompt_ids)
    cached_result = generate_cached(model, prompt_ids)

    tokens_match = torch.equal(
        naive_result["tokens"],
        cached_result["tokens"]
    )

    print("Generated tokens match:", tokens_match)

    if not tokens_match:
        print(
            "WARNING: Token sequences differ. "
            "Inspect numerical differences before comparing output quality."
        )

    # --------------------------------------------------------
    # BENCHMARK NAIVE
    # --------------------------------------------------------

    print("\nBenchmarking naive decoding...")

    naive_results = []

    for run in range(BENCHMARK_RUNS):

        result = generate_naive(model, prompt_ids)
        naive_results.append(result)

        print(
            f"Run {run + 1:02d}: "
            f"TTFT={result['ttft_ms']:.3f} ms | "
            f"Decode={result['decode_ms']:.3f} ms | "
            f"E2E={result['e2e_ms']:.3f} ms"
        )

    # --------------------------------------------------------
    # BENCHMARK CACHED
    # --------------------------------------------------------

    print("\nBenchmarking KV-cached decoding...")

    cached_results = []

    for run in range(BENCHMARK_RUNS):

        result = generate_cached(model, prompt_ids)
        cached_results.append(result)

        print(
            f"Run {run + 1:02d}: "
            f"TTFT={result['ttft_ms']:.3f} ms | "
            f"Decode={result['decode_ms']:.3f} ms | "
            f"E2E={result['e2e_ms']:.3f} ms"
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    naive = summarize(naive_results)
    cached = summarize(cached_results)

    print("\n" + "=" * 65)
    print("FINAL COMPARISON")
    print("=" * 65)

    print(
        f"{'Metric':<18}"
        f"{'Naive':>14}"
        f"{'KV Cache':>14}"
    )

    print("-" * 46)

    for metric in naive:

        print(
            f"{metric:<18}"
            f"{naive[metric]:>13.3f}"
            f"{cached[metric]:>14.3f}"
        )

    # --------------------------------------------------------
    # SPEEDUP
    # --------------------------------------------------------

    decode_speedup = (
        naive["Decode P50"] / cached["Decode P50"]
    )

    tpot_speedup = (
        naive["TPOT P50"] / cached["TPOT P50"]
    )

    e2e_speedup = (
        naive["E2E P50"] / cached["E2E P50"]
    )

    print("\nSPEEDUP")

    print(f"Decode speedup: {decode_speedup:.2f}x")
    print(f"TPOT speedup:   {tpot_speedup:.2f}x")
    print(f"E2E speedup:    {e2e_speedup:.2f}x")

    # --------------------------------------------------------
    # CACHE MEMORY
    # --------------------------------------------------------

    cache_bytes = cached_results[-1]["cache_bytes"]

    print("\nKV CACHE MEMORY")

    print(f"Actual cache: {cache_bytes:,} bytes")
    print(f"Actual cache: {cache_bytes / 1024:.2f} KB")
    print(f"Actual cache: {cache_bytes / (1024 ** 2):.4f} MB")

    # Theoretical FP32 cache size at final cached sequence length

    cached_sequence_len = PROMPT_LEN + OUTPUT_LEN - 1

    theoretical_bytes = (
        2
        * NUM_LAYERS
        * BATCH_SIZE
        * NUM_HEADS
        * cached_sequence_len
        * HEAD_DIM
        * 4
    )

    print(
        f"Theoretical FP32 cache: "
        f"{theoretical_bytes:,} bytes"
    )

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
