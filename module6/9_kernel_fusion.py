
import gc
import statistics
import torch
import pandas as pd

from torch.profiler import profile, ProfilerActivity

# ============================================================
# 1. CONFIGURATION
# ============================================================

assert torch.cuda.is_available(), "CUDA GPU not available"

DEVICE = "cuda"

print("=" * 75)
print("FINAL GPU KERNEL FUSION BENCHMARK")
print("=" * 75)

print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)

TENSOR_SIZES = [
    1_024,
    16_384,
    65_536,
    262_144,
    1_048_576,
    4_194_304,
    16_777_216,
    33_554_432,
    67_108_864,
    134_217_728,
    268_435_456,
]

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

WORKLOADS = {
    "3_operations": 3,
    "6_operations": 6,
    "10_operations": 10,
}

WARMUP_ITERATIONS = 10
BENCHMARK_ROUNDS = 10

# Conservative memory guard.
# Eager execution can temporarily hold several intermediate tensors.
MEMORY_BUDGET_FACTOR = 0.70
MEMORY_ESTIMATE_FACTOR = 16

# Set True to run the separate profiler experiment at the end.
RUN_PROFILER = True


# ============================================================
# 2. WORKLOAD DEFINITIONS
# ============================================================

def eager_3(A, B, C, D):
    X = A + B
    Y = X * C
    Z = Y + D
    return Z


def eager_6(A, B, C, D):
    X = A + B
    Y = X * C
    Z = Y + D
    P = Z * A
    Q = P + B
    R = Q * C
    return R


def eager_10(A, B, C, D):
    X = A + B
    Y = X * C
    Z = Y + D
    P = Z * A
    Q = P + B
    R = Q * C
    S = R + D
    T = S * A
    U = T + B
    V = U * C
    return V


EAGER_FUNCTIONS = {
    "3_operations": eager_3,
    "6_operations": eager_6,
    "10_operations": eager_10,
}

# Compile workloads.
FUSED_FUNCTIONS = {
    name: torch.compile(fn, dynamic=True)
    for name, fn in EAGER_FUNCTIONS.items()
}


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def get_iterations(elements):

    if elements <= 1_048_576:
        return 100

    elif elements <= 16_777_216:
        return 50

    elif elements <= 67_108_864:
        return 20

    elif elements <= 134_217_728:
        return 10

    else:
        return 5


def check_memory(elements, dtype):

    element_size = torch.empty(
        (),
        dtype=dtype
    ).element_size()

    estimated_bytes = (
        elements
        * element_size
        * MEMORY_ESTIMATE_FACTOR
    )

    free_bytes, total_bytes = torch.cuda.mem_get_info()

    allowed_bytes = free_bytes * MEMORY_BUDGET_FACTOR

    return (
        estimated_bytes <= allowed_bytes,
        estimated_bytes,
        free_bytes,
        total_bytes,
    )


def create_inputs(elements, dtype):

    shape = (elements,)

    # Values between 0.5 and 1.5 help avoid numerical overflow
    # in longer FP16/BF16 arithmetic chains.

    A = torch.rand(shape, device=DEVICE, dtype=dtype) + 0.5
    B = torch.rand(shape, device=DEVICE, dtype=dtype) + 0.5
    C = torch.rand(shape, device=DEVICE, dtype=dtype) + 0.5
    D = torch.rand(shape, device=DEVICE, dtype=dtype) + 0.5

    return A, B, C, D


def get_timing_statistics(times):

    return {
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
    }


# ============================================================
# 4. WARM-UP
# ============================================================

def warmup_function(fn, inputs):

    with torch.inference_mode():

        for _ in range(WARMUP_ITERATIONS):
            output = fn(*inputs)
            del output

    torch.cuda.synchronize()


# ============================================================
# 5. BENCHMARK ONE FUNCTION
# ============================================================

def benchmark_once(fn, inputs, iterations):

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()

    start.record()

    with torch.inference_mode():

        for _ in range(iterations):
            output = fn(*inputs)
            del output

    end.record()

    torch.cuda.synchronize()

    return start.elapsed_time(end) / iterations


# ============================================================
# 6. BENCHMARK EAGER VS FUSED
# ============================================================

def benchmark_pair(
    eager_fn,
    fused_fn,
    inputs,
    iterations,
):

    eager_times = []
    fused_times = []

    # Warm up each function separately.
    warmup_function(eager_fn, inputs)
    warmup_function(fused_fn, inputs)

    for round_index in range(BENCHMARK_ROUNDS):

        # Alternate execution order to reduce systematic bias.
        if round_index % 2 == 0:

            order = [
                ("eager", eager_fn),
                ("fused", fused_fn),
            ]

        else:

            order = [
                ("fused", fused_fn),
                ("eager", eager_fn),
            ]

        for name, fn in order:

            latency = benchmark_once(
                fn,
                inputs,
                iterations,
            )

            if name == "eager":
                eager_times.append(latency)

            else:
                fused_times.append(latency)

    return (
        get_timing_statistics(eager_times),
        get_timing_statistics(fused_times),
    )


# ============================================================
# 7. CORRECTNESS CHECK
# ============================================================

def check_correctness(
    eager_fn,
    fused_fn,
    inputs,
    dtype,
):

    with torch.inference_mode():

        eager_output = eager_fn(*inputs)
        fused_output = fused_fn(*inputs)

        # Check for NaN and infinity.
        assert torch.isfinite(eager_output).all(), (
            "Eager output contains NaN or infinity"
        )

        assert torch.isfinite(fused_output).all(), (
            "Fused output contains NaN or infinity"
        )

        if dtype == torch.float32:
            rtol, atol = 1e-4, 1e-4

        elif dtype == torch.float16:
            rtol, atol = 1e-2, 1e-2

        else:
            rtol, atol = 2e-2, 2e-2

        torch.testing.assert_close(
            fused_output,
            eager_output,
            rtol=rtol,
            atol=atol,
        )

        del eager_output
        del fused_output

    torch.cuda.synchronize()


# ============================================================
# 8. THEORETICAL MEMORY TRAFFIC
# ============================================================

def theoretical_traffic(
    elements,
    dtype,
    num_ops,
):

    element_size = torch.empty(
        (),
        dtype=dtype
    ).element_size()

    # Eager:
    # Every binary operation reads two tensors and writes one.
    #
    # This assumes intermediates are materialized in global memory.

    eager_bytes_per_element = (
        3 * num_ops * element_size
    )

    # Fused:
    # Reads A, B, C, D once.
    # Writes one final output.
    #
    # Assumes intermediate values remain in registers
    # and do not spill to global memory.

    fused_bytes_per_element = (
        5 * element_size
    )

    eager_gb = (
        elements * eager_bytes_per_element / 1e9
    )

    fused_gb = (
        elements * fused_bytes_per_element / 1e9
    )

    return eager_gb, fused_gb


# ============================================================
# 9. MAIN BENCHMARK
# ============================================================

results = []

for dtype_name, dtype in DTYPES.items():

    print("\n" + "=" * 75)
    print("DATA TYPE:", dtype_name)
    print("=" * 75)

    for elements in TENSOR_SIZES:

        print(f"\nTensor elements: {elements:,}")

        allowed, estimated_bytes, free_bytes, total_bytes = (
            check_memory(elements, dtype)
        )

        if not allowed:

            print(
                "SKIPPED: conservative memory estimate = "
                f"{estimated_bytes / 1e9:.2f} GB; "
                f"currently free = {free_bytes / 1e9:.2f} GB"
            )

            continue

        inputs = None

        try:

            inputs = create_inputs(elements, dtype)

            iterations = get_iterations(elements)

            for workload_name, num_ops in WORKLOADS.items():

                eager_fn = EAGER_FUNCTIONS[workload_name]
                fused_fn = FUSED_FUNCTIONS[workload_name]

                print(f"\n  Workload: {workload_name}")

                # --------------------------------------------
                # Correctness
                # --------------------------------------------

                check_correctness(
                    eager_fn,
                    fused_fn,
                    inputs,
                    dtype,
                )

                # --------------------------------------------
                # Benchmark
                # --------------------------------------------

                eager_stats, fused_stats = benchmark_pair(
                    eager_fn,
                    fused_fn,
                    inputs,
                    iterations,
                )

                eager_median = eager_stats["median"]
                fused_median = fused_stats["median"]

                speedup = eager_median / fused_median

                eager_gb, fused_gb = theoretical_traffic(
                    elements,
                    dtype,
                    num_ops,
                )

                row = {
                    "dtype": dtype_name,
                    "elements": elements,
                    "workload": workload_name,

                    "eager_ms": eager_median,
                    "eager_min_ms": eager_stats["min"],
                    "eager_max_ms": eager_stats["max"],

                    "fused_ms": fused_median,
                    "fused_min_ms": fused_stats["min"],
                    "fused_max_ms": fused_stats["max"],

                    "speedup": speedup,

                    "eager_logical_GB": eager_gb,
                    "fused_logical_GB": fused_gb,

                    "iterations": iterations,
                    "rounds": BENCHMARK_ROUNDS,
                }

                results.append(row)

                print(
                    f"    Eager median: {eager_median:.4f} ms "
                    f"(min {eager_stats['min']:.4f}, "
                    f"max {eager_stats['max']:.4f})"
                )

                print(
                    f"    Fused median: {fused_median:.4f} ms "
                    f"(min {fused_stats['min']:.4f}, "
                    f"max {fused_stats['max']:.4f})"
                )

                print(
                    f"    Speedup: {speedup:.2f}x"
                )

        except torch.cuda.OutOfMemoryError:

            print(
                "CUDA OUT OF MEMORY: skipping this tensor size."
            )

            torch.cuda.empty_cache()

        except Exception as error:

            print(
                f"ERROR: {type(error).__name__}: {error}"
            )

        finally:

            if inputs is not None:
                del inputs

            gc.collect()
            torch.cuda.empty_cache()


# ============================================================
# 10. FINAL RESULTS
# ============================================================

if not results:

    raise RuntimeError(
        "No benchmark results were generated."
    )

df = pd.DataFrame(results)

print("\n" + "=" * 100)
print("FINAL BENCHMARK RESULTS")
print("=" * 100)

print(
    df.to_string(
        index=False,
        formatters={
            col: "{:.4f}".format
            for col in df.select_dtypes(
                include="number"
            ).columns
            if col not in ["elements", "iterations", "rounds"]
        },
    )
)


# ============================================================
# 11. SAVE CSV
# ============================================================

csv_filename = "kernel_fusion_results_final.csv"

df.to_csv(
    csv_filename,
    index=False,
)

print(f"\nSaved: {csv_filename}")


# ============================================================
# 12. SUMMARY BY DATA TYPE AND WORKLOAD
# ============================================================

summary = (
    df.groupby(["dtype", "workload"])["speedup"]
      .agg(
          median="median",
          mean="mean",
          minimum="min",
          maximum="max",
      )
      .reset_index()
)

print("\n" + "=" * 75)
print("SPEEDUP SUMMARY BY DATA TYPE AND WORKLOAD")
print("=" * 75)

print(
    summary.to_string(
        index=False,
        formatters={
            "median": "{:.3f}x".format,
            "mean": "{:.3f}x".format,
            "minimum": "{:.3f}x".format,
            "maximum": "{:.3f}x".format,
        },
    )
)


# ============================================================
# 13. BEST SPEEDUP PER DATA TYPE
# ============================================================

best_results = (
    df.loc[
        df.groupby("dtype")["speedup"].idxmax()
    ][
        ["dtype", "elements", "workload", "speedup"]
    ]
    .sort_values("dtype")
)

print("\n" + "=" * 75)
print("BEST SPEEDUP BY DATA TYPE")
print("=" * 75)

print(
    best_results.to_string(
        index=False,
        formatters={
            "speedup": "{:.3f}x".format
        },
    )
)


# ============================================================
# 14. SEPARATE PROFILER EXPERIMENT
# ============================================================

def profile_workload(
    workload_name,
    elements=1_048_576,
    dtype=torch.float32,
):

    print("\n" + "=" * 75)
    print("PROFILER EXPERIMENT")
    print("=" * 75)

    print("Workload:", workload_name)
    print("Elements:", f"{elements:,}")
    print("Dtype:", dtype)

    inputs = create_inputs(elements, dtype)

    functions = {
        "EAGER": EAGER_FUNCTIONS[workload_name],
        "FUSED": FUSED_FUNCTIONS[workload_name],
    }

    for label, fn in functions.items():

        print("\n" + "-" * 75)
        print(label)
        print("-" * 75)

        # Warm-up outside profiler.
        warmup_function(fn, inputs)

        with profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
        ) as prof:

            with torch.inference_mode():

                for _ in range(5):
                    output = fn(*inputs)
                    del output

            torch.cuda.synchronize()

        print(
            prof.key_averages().table(
                sort_by="self_cuda_time_total",
                row_limit=25,
            )
        )

        prof.export_chrome_trace(
            f"kernel_fusion_{label.lower()}_trace.json"
        )

        print(
            f"Chrome trace saved: "
            f"kernel_fusion_{label.lower()}_trace.json"
        )

    del inputs

    gc.collect()
    torch.cuda.empty_cache()


if RUN_PROFILER:

    # A representative workload.
    # Profiling is intentionally separate from the benchmark.

    profile_workload(
        workload_name="10_operations",
        elements=1_048_576,
        dtype=torch.float32,
    )


print("\n" + "=" * 75)
print("BENCHMARK COMPLETED")
print("=" * 75)
