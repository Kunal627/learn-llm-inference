
import torch
import torch.nn as nn

# --------------------------------------------------
# 1. Define a simple model
# --------------------------------------------------

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(1000, 128)

        self.layer1 = nn.Linear(128, 256)
        self.layer2 = nn.Linear(256, 128)

        self.output = nn.Linear(128, 1000)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.output(x)
        return x


model = SimpleModel()

# --------------------------------------------------
# 2. Inspect individual parameters
# --------------------------------------------------

print("\nPARAMETER DETAILS")
print("-" * 65)

for name, param in model.named_parameters():
    print(
        f"{name:25s} "
        f"Shape: {str(tuple(param.shape)):20s} "
        f"Parameters: {param.numel():10,} "
        f"Dtype: {param.dtype}"
    )

# --------------------------------------------------
# 3. Calculate total parameter count
# --------------------------------------------------

total_params = sum(
    param.numel()
    for param in model.parameters()
)

print("\nTOTAL PARAMETERS")
print(f"{total_params:,}")

# --------------------------------------------------
# 4. Calculate parameter memory
# --------------------------------------------------

total_bytes = sum(
    param.numel() * param.element_size()
    for param in model.parameters()
)

print("\nMODEL WEIGHT MEMORY")
print(f"Bytes: {total_bytes:,}")
print(f"MB:    {total_bytes / (1024 ** 2):.4f}")
print(f"MiB:   {total_bytes / (1024 ** 2):.4f}")

# --------------------------------------------------
# 5. Compare theoretical memory across dtypes
# --------------------------------------------------

print("\nTHEORETICAL MEMORY BY PRECISION")

for dtype, bytes_per_param in [
    ("FP32", 4),
    ("FP16", 2),
    ("BF16", 2),
]:
    memory = total_params * bytes_per_param

    print(
        f"{dtype:8s}: "
        f"{memory / (1024 ** 2):.4f} MiB"
    )


# --------------------------------------------------
# 6. Convert model to BF16
# --------------------------------------------------

print("\nBEFORE CONVERSION")

param = next(model.parameters())

print("Dtype:", param.dtype)
print("Element size:", param.element_size(), "bytes")

fp32_memory = sum(
    p.numel() * p.element_size()
    for p in model.parameters()
)

# Convert all floating-point parameters to BF16
model = model.to(dtype=torch.bfloat16)

print("\nAFTER CONVERSION")

param = next(model.parameters())

print("Dtype:", param.dtype)
print("Element size:", param.element_size(), "bytes")

bf16_memory = sum(
    p.numel() * p.element_size()
    for p in model.parameters()
)

print(f"\nFP32 memory: {fp32_memory / (1024 ** 2):.4f} MiB")
print(f"BF16 memory: {bf16_memory / (1024 ** 2):.4f} MiB")

print(
    "Memory reduction:",
    f"{(1 - bf16_memory / fp32_memory) * 100:.2f}%"
)
