
import torch

print("=" * 60)
print("GPU HARDWARE INSPECTION")
print("=" * 60)

assert torch.cuda.is_available(), "CUDA is not available!"

device = torch.device("cuda:0")
properties = torch.cuda.get_device_properties(device)

print("\nGPU Name:", properties.name)
print("Compute Capability:",
      f"{properties.major}.{properties.minor}")

print("Number of SMs:", properties.multi_processor_count)

print("Total GPU Memory:",
      round(properties.total_memory / (1024**3), 2), "GiB")

print("CUDA Device Count:", torch.cuda.device_count())

print("\n" + "=" * 60)
print("GPU MEMORY")
print("=" * 60)

torch.cuda.empty_cache()

free_memory, total_memory = torch.cuda.mem_get_info(device)

print("Total Memory:",
      round(total_memory / (1024**3), 2), "GiB")

print("Free Memory:",
      round(free_memory / (1024**3), 2), "GiB")

print("Allocated Memory:",
      round(torch.cuda.memory_allocated(device) / (1024**3), 3),
      "GiB")

print("\n" + "=" * 60)
print("CUDA CAPABILITIES")
print("=" * 60)

print("CUDA Version:", torch.version.cuda)

print("BF16 supported:",
      torch.cuda.is_bf16_supported())

print("FP16 supported:",
      torch.cuda.get_device_capability(device)[0] >= 5)
