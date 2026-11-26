
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

print("GPU:", torch.cuda.get_device_name(0))
print("CUDA:", torch.version.cuda)

# Reset peak memory statistics
torch.cuda.reset_peak_memory_stats()

# Record memory before allocation
before = torch.cuda.memory_allocated()

# Allocate a tensor with 100 million FP32 values
x = torch.empty(
    100_000_000,
    dtype=torch.float32,
    device="cuda"
)

# Initialize it to ensure actual memory use
x.zero_()

after = torch.cuda.memory_allocated()
peak = torch.cuda.max_memory_allocated()

print("\nGPU MEMORY")
print(f"Before: {before / (1024**3):.3f} GiB")
print(f"After:  {after / (1024**3):.3f} GiB")
print(f"Peak:   {peak / (1024**3):.3f} GiB")

print("\nEXPECTED TENSOR MEMORY")
print(f"{100_000_000 * 4 / (1024**3):.3f} GiB")

del x
torch.cuda.empty_cache()
