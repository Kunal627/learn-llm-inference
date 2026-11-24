
import time
import statistics
import torch

assert torch.cuda.is_available(), "CUDA is not available!"

device = torch.device("cuda:0")

print("=" * 65)
print("CUDA KERNEL TIMING EXPERIMENT")
print("=" * 65)

print("GPU:", torch.cuda.get_device_name(device))

N = 4096

A = torch.randn(N, N, device=device)
B = torch.randn(N, N, device=device)

def matrix_multiply():
    return A @ B

# --------------------------------------------------
# 1. Warmup
# --------------------------------------------------

for _ in range(5):
    matrix_multiply()

torch.cuda.synchronize()

print("\n1. WARMUP COMPLETED")

# --------------------------------------------------
# 2. Incorrect CPU timing: no synchronization
# --------------------------------------------------

incorrect_times = []

for _ in range(5):
    start = time.perf_counter()

    C = matrix_multiply()

    end = time.perf_counter()

    incorrect_times.append((end - start) * 1000)

# Wait for all queued operations before continuing
torch.cuda.synchronize()

incorrect_median = statistics.median(incorrect_times)

print("\n2. CPU TIMING WITHOUT SYNCHRONIZATION")
print("Median:", round(incorrect_median, 4), "ms")

# --------------------------------------------------
# 3. Correct CPU wall-clock timing
# --------------------------------------------------

correct_times = []

for _ in range(5):
    torch.cuda.synchronize()

    start = time.perf_counter()

    C = matrix_multiply()

    torch.cuda.synchronize()

    end = time.perf_counter()

    correct_times.append((end - start) * 1000)

correct_median = statistics.median(correct_times)

print("\n3. CPU TIMING WITH SYNCHRONIZATION")
print("Median:", round(correct_median, 4), "ms")

# --------------------------------------------------
# 4. CUDA event timing
# --------------------------------------------------

event_times = []

for _ in range(10):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    C = matrix_multiply()

    end_event.record()

    end_event.synchronize()

    event_times.append(start_event.elapsed_time(end_event))

event_median = statistics.median(event_times)

print("\n4. CUDA EVENT TIMING")
print("Median:", round(event_median, 4), "ms")

# --------------------------------------------------
# 5. Compare results
# --------------------------------------------------

print("\n" + "=" * 65)
print("TIMING COMPARISON")
print("=" * 65)

print(f"CPU without sync : {incorrect_median:.4f} ms")
print(f"CPU with sync    : {correct_median:.4f} ms")
print(f"CUDA events      : {event_median:.4f} ms")

print("\nNote:")
print("The unsynchronized CPU time is not a valid measurement")
print("of GPU execution duration.")

print("\nExperiment completed.")
