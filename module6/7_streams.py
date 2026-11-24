
import time
import torch

assert torch.cuda.is_available(), "CUDA is not available!"

device = torch.device("cuda:0")

print("=" * 65)
print("CUDA STREAMS AND MEMORY TRANSFER DEMO")
print("=" * 65)

print("GPU:", torch.cuda.get_device_name(device))

# --------------------------------------------------
# 1. Prepare CPU tensors
# --------------------------------------------------

N = 4096

x_pageable = torch.randn(N, N, dtype=torch.float32)
x_pinned = torch.randn(
    N, N,
    dtype=torch.float32,
    pin_memory=True
)

print("\n1. CPU MEMORY")
print("Pageable tensor pinned:", x_pageable.is_pinned())
print("Pinned tensor pinned:", x_pinned.is_pinned())

# --------------------------------------------------
# 2. Pageable H2D transfer
# --------------------------------------------------

torch.cuda.synchronize()

start = time.perf_counter()

x_gpu_1 = x_pageable.to(device)

torch.cuda.synchronize()

pageable_time = (time.perf_counter() - start) * 1000

print("\n2. PAGEABLE H2D TRANSFER")
print(f"Time: {pageable_time:.3f} ms")

# --------------------------------------------------
# 3. Pinned H2D transfer
# --------------------------------------------------

torch.cuda.synchronize()

start = time.perf_counter()

x_gpu_2 = x_pinned.to(device, non_blocking=True)

torch.cuda.synchronize()

pinned_time = (time.perf_counter() - start) * 1000

print("\n3. PINNED H2D TRANSFER")
print(f"Time: {pinned_time:.3f} ms")

# --------------------------------------------------
# 4. Create two CUDA streams
# --------------------------------------------------

stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

A = torch.randn(N, N, device=device)
B = torch.randn(N, N, device=device)

C = torch.empty_like(A)
D = torch.empty_like(B)

print("\n4. TWO CUDA STREAMS")

# Each stream gets an independent matrix multiplication.
with torch.cuda.stream(stream1):
    C = A @ A

with torch.cuda.stream(stream2):
    D = B @ B

# Wait for both streams
stream1.synchronize()
stream2.synchronize()

print("Stream 1 output:", tuple(C.shape))
print("Stream 2 output:", tuple(D.shape))

# --------------------------------------------------
# 5. Verify results
# --------------------------------------------------

torch.cuda.synchronize()

assert torch.allclose(C, A @ A, atol=1e-2, rtol=1e-3)
assert torch.allclose(D, B @ B, atol=1e-2, rtol=1e-3)

print("\n5. VERIFICATION")
print("Both outputs verified successfully.")

# --------------------------------------------------
# 6. Demonstrate event-based dependency
# --------------------------------------------------

producer = torch.cuda.Stream()
consumer = torch.cuda.Stream()

source = torch.randn(1024, 1024, device=device)
intermediate = torch.empty_like(source)

copy_done = torch.cuda.Event()

with torch.cuda.stream(producer):
    intermediate.copy_(source)
    copy_done.record()

with torch.cuda.stream(consumer):
    consumer.wait_event(copy_done)
    result = intermediate * 2

consumer.synchronize()

assert torch.allclose(result, source * 2)

print("\n6. STREAM EVENT SYNCHRONIZATION")
print("Producer copied the tensor.")
print("Consumer waited for the event.")
print("Dependent computation verified.")

# --------------------------------------------------
# 7. Summary
# --------------------------------------------------

print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)

print(f"Pageable transfer: {pageable_time:.3f} ms")
print(f"Pinned transfer  : {pinned_time:.3f} ms")

print("\nImportant:")
print("- Transfer times depend on system and GPU conditions.")
print("- Pinned memory does not guarantee a faster small transfer.")
print("- Two streams do not guarantee concurrent execution.")
print("- Synchronization ensures correct dependencies.")

print("\nExperiment completed successfully.")
