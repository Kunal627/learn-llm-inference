
import time
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

class SimpleNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024)
        )

    def forward(self, x):
        return self.layers(x)


model = SimpleNetwork().to(device).eval()
compiled_model = torch.compile(model)

x = torch.randn(32, 1024, device=device)

def synchronize():
    if device == "cuda":
        torch.cuda.synchronize()

# Warm up compiled execution
with torch.inference_mode():
    for _ in range(10):
        compiled_model(x)

synchronize()

# Compare outputs
with torch.inference_mode():
    eager_output = model(x)
    compiled_output = compiled_model(x)

print("Maximum output difference:",
      (eager_output - compiled_output).abs().max().item())

def benchmark(fn, iterations=100):
    synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        for _ in range(iterations):
            fn(x)

    synchronize()
    return (time.perf_counter() - start) * 1000 / iterations

eager_ms = benchmark(model)
compiled_ms = benchmark(compiled_model)

print("\nRESULTS")
print(f"Eager:    {eager_ms:.4f} ms")
print(f"Compiled: {compiled_ms:.4f} ms")
print(f"Speedup:  {eager_ms / compiled_ms:.2f}x")
