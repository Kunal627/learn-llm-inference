
import torch

assert torch.cuda.is_available(), "CUDA is not available!"

device = torch.device("cuda:0")

def print_memory(title):
    torch.cuda.synchronize()

    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    free, total = torch.cuda.mem_get_info(device)

    print(f"\n{title}")
    print("-" * 50)
    print(f"Allocated memory : {allocated:,.2f} MiB")
    print(f"Reserved memory  : {reserved:,.2f} MiB")
    print(f"Free GPU memory  : {free / (1024**2):,.2f} MiB")
    print(f"Total GPU memory : {total / (1024**2):,.2f} MiB")


print("=" * 60)
print("GPU GLOBAL MEMORY EXPERIMENT")
print("=" * 60)

print("GPU:", torch.cuda.get_device_name(device))

# 1. Initial memory
print_memory("1. Initial GPU memory")

# 2. Allocate FP32 tensor
N = 4096

a = torch.randn(
    N, N,
    dtype=torch.float32,
    device=device
)

print_memory("2. After allocating FP32 tensor")

# 3. Allocate FP16 tensor
b = torch.randn(
    N, N,
    dtype=torch.float16,
    device=device
)

print_memory("3. After allocating FP16 tensor")

# 4. Calculate expected memory
fp32_size = a.numel() * a.element_size() / (1024 ** 2)
fp16_size = b.numel() * b.element_size() / (1024 ** 2)

print("\n" + "=" * 60)
print("TENSOR MEMORY SIZES")
print("=" * 60)

print(f"FP32 tensor: {fp32_size:.2f} MiB")
print(f"FP16 tensor: {fp16_size:.2f} MiB")

# 5. Perform GPU computation
c = a @ a.T

torch.cuda.synchronize()

print_memory("4. After matrix multiplication")

print("\nOutput shape:", c.shape)

# 6. Release tensors
del a, b, c

torch.cuda.empty_cache()

print_memory("5. After deleting tensors and emptying cache")

print("\nExperiment completed.")
