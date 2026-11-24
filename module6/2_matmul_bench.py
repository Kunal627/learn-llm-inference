
import torch
import time

print("=" * 60)
print("GPU INFORMATION")
print("=" * 60)

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# Matrix dimensions
M = 4096
K = 4096
N = 4096

print("\nMatrix dimensions:")
print(f"A: ({M}, {K})")
print(f"B: ({K}, {N})")
print(f"C: ({M}, {N})")

# Generate CPU tensors
A_cpu = torch.randn(M, K)
B_cpu = torch.randn(K, N)

# CPU benchmark
start = time.perf_counter()

C_cpu = torch.matmul(A_cpu, B_cpu)

cpu_time = time.perf_counter() - start

print("\nCPU RESULT")
print("Output shape:", C_cpu.shape)
print(f"Time: {cpu_time:.4f} seconds")

if torch.cuda.is_available():

    # Transfer tensors to GPU
    A_gpu = A_cpu.to("cuda")
    B_gpu = B_cpu.to("cuda")

    # Warmup
    for _ in range(3):
        torch.matmul(A_gpu, B_gpu)

    torch.cuda.synchronize()

    # GPU benchmark
    start = time.perf_counter()

    C_gpu = torch.matmul(A_gpu, B_gpu)

    torch.cuda.synchronize()

    gpu_time = time.perf_counter() - start

    print("\nGPU RESULT")
    print("Output shape:", C_gpu.shape)
    print(f"Time: {gpu_time:.4f} seconds")

    print("\nCOMPARISON")
    print(f"CPU time: {cpu_time:.4f}s")
    print(f"GPU time: {gpu_time:.4f}s")
    print(f"Speedup: {cpu_time / gpu_time:.2f}x")

    # Correctness check
    assert torch.allclose(
        C_cpu.to("cuda"),
        C_gpu,
        atol=1e-1,
        rtol=1e-3
    )

    print("\nCPU and GPU results are numerically close.")
