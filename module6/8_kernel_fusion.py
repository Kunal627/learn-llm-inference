
import torch
import time

device = "cuda"

N = 16 * 1024 * 1024

A = torch.randn(N, device=device)
B = torch.randn(N, device=device)
C = torch.randn(N, device=device)
D = torch.randn(N, device=device)


# -----------------------------------
# 1. Unfused implementation
# -----------------------------------

def unfused():
    X = A + B
    Y = X * C
    Z = Y + D
    return Z


# -----------------------------------
# 2. Fused implementation
# -----------------------------------

@torch.compile
def fused(A, B, C, D):
    return (A + B) * C + D


# Warmup
for _ in range(10):
    unfused()
    fused(A, B, C, D)

torch.cuda.synchronize()


# Benchmark helper
def benchmark(fn, iterations=100):

    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(iterations):
        result = fn()

    torch.cuda.synchronize()

    return (time.perf_counter() - start) * 1000 / iterations


unfused_ms = benchmark(unfused)
fused_ms = benchmark(lambda: fused(A, B, C, D))

print(f"Unfused: {unfused_ms:.4f} ms")
print(f"Fused:   {fused_ms:.4f} ms")

print(f"Speedup: {unfused_ms / fused_ms:.2f}x")
