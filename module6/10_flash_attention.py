
import gc
import torch
import statistics

from torch.nn.attention import sdpa_kernel, SDPBackend


# ============================================================
# 1. CONFIGURATION
# ============================================================

BATCH_SIZE = 2
NUM_HEADS = 8
HEAD_DIM = 64

SEQ_LENGTHS = [512, 1024, 2048, 4096]

DTYPE = torch.float16
DEVICE = "cuda"

WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 30

assert torch.cuda.is_available(), "CUDA GPU not available"

print("=" * 75)
print("FLASHATTENTION BENCHMARK")
print("=" * 75)

print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("Data type:", DTYPE)


# ============================================================
# 2. STANDARD ATTENTION
# ============================================================

def standard_attention(Q, K, V):
    """
    Explicit attention implementation.

    Steps:
        1. Q x K transpose
        2. Scale scores
        3. Softmax
        4. Multiply probabilities by V

    This explicitly materializes the score and probability
    matrices.
    """

    scale = Q.size(-1) ** -0.5

    scores = torch.matmul(Q, K.transpose(-2, -1))
    scores = scores * scale

    probabilities = torch.softmax(scores, dim=-1)

    output = torch.matmul(probabilities, V)

    return output


# ============================================================
# 3. FLASHATTENTION THROUGH PYTORCH SDPA
# ============================================================

def flash_attention(Q, K, V):
    """
    PyTorch Scaled Dot Product Attention.

    Forces the FlashAttention backend.
    """

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):

        output = torch.nn.functional.scaled_dot_product_attention(
            Q,
            K,
            V,
            dropout_p=0.0,
            is_causal=False
        )

    return output


# ============================================================
# 4. CUDA EVENT BENCHMARK
# ============================================================

def benchmark(fn, Q, K, V):

    # Warmup
    with torch.inference_mode():
        for _ in range(WARMUP_ITERATIONS):
            fn(Q, K, V)

    torch.cuda.synchronize()

    timings = []

    with torch.inference_mode():

        for _ in range(BENCHMARK_ITERATIONS):

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()

            fn(Q, K, V)

            end.record()

            torch.cuda.synchronize()

            timings.append(start.elapsed_time(end))

    return statistics.median(timings)


# ============================================================
# 5. PEAK GPU MEMORY MEASUREMENT
# ============================================================

def measure_peak_memory(fn, Q, K, V):

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Reset peak allocated memory counter.
    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        output = fn(Q, K, V)

    torch.cuda.synchronize()

    peak_bytes = torch.cuda.max_memory_allocated()

    peak_mb = peak_bytes / (1024 ** 2)

    del output

    gc.collect()
    torch.cuda.empty_cache()

    return peak_mb


# ============================================================
# 6. INPUT MEMORY AND SCORE MATRIX SIZE
# ============================================================

def calculate_input_memory(Q, K, V):

    total_bytes = (
        Q.numel() * Q.element_size()
        + K.numel() * K.element_size()
        + V.numel() * V.element_size()
    )

    return total_bytes / (1024 ** 2)


def calculate_score_matrix_memory(sequence_length):

    elements = (
        BATCH_SIZE
        * NUM_HEADS
        * sequence_length
        * sequence_length
    )

    bytes_used = elements * torch.tensor([], dtype=DTYPE).element_size()

    return bytes_used / (1024 ** 2)


# ============================================================
# 7. CORRECTNESS CHECK
# ============================================================

def check_correctness(Q, K, V):

    with torch.inference_mode():

        key_float = K.float()
        value_float = V.float()
        standard_output = torch.cat([
            standard_attention(query_chunk.float(), key_float, value_float)
            for query_chunk in Q.split(128, dim=-2)
        ], dim=-2)

        flash_output = flash_attention(Q, K, V).float()

        if not torch.isfinite(standard_output).all() or not torch.isfinite(flash_output).all():
            raise ValueError("Attention output contains non-finite values")

        torch.testing.assert_close(flash_output, standard_output, rtol=0.01, atol=0.001)

    difference = (
        standard_output - flash_output
    ).abs()

    max_error = difference.max().item()

    mean_error = difference.mean().item()

    return max_error, mean_error


# ============================================================
# 8. MAIN EXPERIMENT
# ============================================================

results = []

for seq_len in SEQ_LENGTHS:

    print("\n" + "=" * 75)
    print(f"SEQUENCE LENGTH: {seq_len}")
    print("=" * 75)

    # Create Q, K, V
    Q = torch.randn(
        BATCH_SIZE,
        NUM_HEADS,
        seq_len,
        HEAD_DIM,
        device=DEVICE,
        dtype=DTYPE
    )

    K = torch.randn_like(Q)
    V = torch.randn_like(Q)

    # --------------------------------------------------------
    # Correctness
    # --------------------------------------------------------

    max_error, mean_error = check_correctness(Q, K, V)

    # --------------------------------------------------------
    # Latency
    # --------------------------------------------------------

    standard_ms = benchmark(
        standard_attention, Q, K, V
    )

    flash_ms = benchmark(
        flash_attention, Q, K, V
    )

    speedup = standard_ms / flash_ms

    # --------------------------------------------------------
    # Memory
    # --------------------------------------------------------

    input_memory_mb = calculate_input_memory(Q, K, V)

    standard_peak_mb = measure_peak_memory(
        standard_attention, Q, K, V
    )

    flash_peak_mb = measure_peak_memory(
        flash_attention, Q, K, V
    )

    score_memory_mb = calculate_score_matrix_memory(seq_len)

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    result = {
        "sequence_length": seq_len,
        "standard_ms": standard_ms,
        "flash_ms": flash_ms,
        "speedup": speedup,
        "max_error": max_error,
        "mean_error": mean_error,
        "score_matrix_mb": score_memory_mb,
        "input_memory_mb": input_memory_mb,
        "standard_peak_mb": standard_peak_mb,
        "flash_peak_mb": flash_peak_mb
    }

    results.append(result)

    print(f"Standard Attention : {standard_ms:.4f} ms")
    print(f"Flash Attention    : {flash_ms:.4f} ms")
    print(f"Speedup            : {speedup:.2f}x")

    print(f"Maximum Error      : {max_error:.8f}")
    print(f"Mean Error         : {mean_error:.8f}")

    print(f"Score Matrix       : {score_memory_mb:.2f} MiB")
    print(f"Input Memory       : {input_memory_mb:.2f} MiB")

    print(f"Standard Peak (allocated, including inputs): {standard_peak_mb:.2f} MiB")
    print(f"Flash Peak (allocated, including inputs)   : {flash_peak_mb:.2f} MiB")

    del Q, K, V

    gc.collect()
    torch.cuda.empty_cache()


# ============================================================
# 9. FINAL RESULTS TABLE
# ============================================================

print("\n\n")
print("=" * 125)
print("FINAL FLASHATTENTION BENCHMARK RESULTS (peaks include inputs)")
print("=" * 125)

header = (
    f"{'Seq':>6} | "
    f"{'Standard ms':>12} | "
    f"{'Flash ms':>10} | "
    f"{'Speedup':>8} | "
    f"{'Max Error':>12} | "
    f"{'Score MiB':>10} | "
    f"{'Standard Peak':>14} | "
    f"{'Flash Peak':>11}"
)

print(header)
print("-" * 125)

for r in results:

    print(
        f"{r['sequence_length']:>6} | "
        f"{r['standard_ms']:>12.4f} | "
        f"{r['flash_ms']:>10.4f} | "
        f"{r['speedup']:>7.2f}x | "
        f"{r['max_error']:>12.8f} | "
        f"{r['score_matrix_mb']:>10.2f} | "
        f"{r['standard_peak_mb']:>14.2f} | "
        f"{r['flash_peak_mb']:>11.2f}"
    )

print("=" * 125)

print("\nBenchmark completed.")
