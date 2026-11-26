
import torch
import time

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU required")

device = "cuda"

print("GPU:", torch.cuda.get_device_name(0))

# Define GPU computation
def computation(x, a, b):
    x = torch.relu(x @ a)
    x = torch.relu(x @ b)
    return x

# Static tensors
x = torch.randn(512, 512, device=device)
a = torch.randn(512, 512, device=device)
b = torch.randn(512, 512, device=device)

# Warm-up normal execution
for _ in range(10):
    y = computation(x, a, b)

torch.cuda.synchronize()

# -----------------------------
# 1. Normal execution
# -----------------------------
iterations = 1000

start = time.perf_counter()

for _ in range(iterations):
    y = computation(x, a, b)

torch.cuda.synchronize()

normal_time = time.perf_counter() - start

# -----------------------------
# 2. CUDA Graph capture
# -----------------------------

# Warm-up on a separate stream
stream = torch.cuda.Stream()

with torch.cuda.stream(stream):
    for _ in range(10):
        y = computation(x, a, b)

torch.cuda.current_stream().wait_stream(stream)

# Capture computation
graph = torch.cuda.CUDAGraph()

with torch.cuda.graph(graph):
    static_y = computation(x, a, b)

# Replay graph
start = time.perf_counter()

for _ in range(iterations):
    graph.replay()

torch.cuda.synchronize()

graph_time = time.perf_counter() - start

# -----------------------------
# Results
# -----------------------------

print("\nRESULTS")
print(f"Normal execution: {normal_time * 1000:.3f} ms")
print(f"CUDA Graph:       {graph_time * 1000:.3f} ms")

print(
    f"Speedup:          {normal_time / graph_time:.2f}x"
)

print(
    "Maximum output difference:",
    (y - static_y).abs().max().item()
)
