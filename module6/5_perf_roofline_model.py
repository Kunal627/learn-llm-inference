
import statistics
import torch

assert torch.cuda.is_available(), "CUDA is not available!"

device = torch.device("cuda:0")

# Keep FP32 matrix multiplication behavior explicit.
torch.backends.cuda.matmul.allow_tf32 = False

def benchmark_cuda(fn, warmup=3, repeats=10):
    # Warmup
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()

    # Measure individual executions with CUDA events
    times_ms = []

    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        fn()
        end.record()

        end.synchronize()
        times_ms.append(start.elapsed_time(end))

    return statistics.median(times_ms)


print("=" * 65)
print("GPU PERFORMANCE BENCHMARK")
print("=" * 65)

print("GPU:", torch.cuda.get_device_name(device))

# --------------------------------------------------
# EXPERIMENT 1: MATRIX MULTIPLICATION
# --------------------------------------------------

print("\n1. MATRIX MULTIPLICATION")
print("-" * 65)

N = 4096

A = torch.randn(N, N, device=device, dtype=torch.float32)
B = torch.randn(N, N, device=device, dtype=torch.float32)

def matrix_multiply():
    return torch.mm(A, B)

C = matrix_multiply()

elapsed_ms = benchmark_cuda(matrix_multiply)

# 2 * N^3 FLOPs
flops = 2 * (N ** 3)

seconds = elapsed_ms / 1000
tflops = flops / seconds / 1e12

print("Matrix dimensions:", N, "x", N)
print("Output shape:", tuple(C.shape))
print(f"Median execution time: {elapsed_ms:.3f} ms")
print(f"Estimated performance: {tflops:.3f} TFLOPS")

# --------------------------------------------------
# EXPERIMENT 2: MEMORY BANDWIDTH
# --------------------------------------------------

print("\n2. VECTOR ADDITION / MEMORY BANDWIDTH")
print("-" * 65)

ELEMENTS = 16 * 1024 * 1024

X = torch.randn(ELEMENTS, device=device, dtype=torch.float32)
Y = torch.randn(ELEMENTS, device=device, dtype=torch.float32)
Z = torch.empty_like(X)

def vector_add():
    torch.add(X, Y, out=Z)

vector_add()

elapsed_ms = benchmark_cuda(vector_add)

# Read X + read Y + write Z = 3 transfers per element
bytes_moved = ELEMENTS * 4 * 3

seconds = elapsed_ms / 1000
bandwidth_gbs = bytes_moved / seconds / 1e9

print("Number of elements:", ELEMENTS)
print("Data type: FP32")
print(f"Median execution time: {elapsed_ms:.3f} ms")
print(f"Estimated effective bandwidth: {bandwidth_gbs:.2f} GB/s")

# --------------------------------------------------
# EXPERIMENT 3: SUMMARY
# --------------------------------------------------

print("\n" + "=" * 65)
print("BENCHMARK SUMMARY")
print("=" * 65)

print(f"Matrix throughput : {tflops:.3f} TFLOPS")
print(f"Vector bandwidth  : {bandwidth_gbs:.2f} GB/s")

print("\nImportant:")
print("- These are measured workload results, not hardware peak specs.")
print("- Vector bandwidth is an effective estimate.")
print("- Cache behavior and kernel implementation affect the result.")
print("- Matrix multiplication uses FP32 with TF32 disabled.")

print("\nBenchmark completed successfully.")
