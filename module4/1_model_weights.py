
import torch

# ============================================================
# MODULE 4 - LESSON 4.1
# GPU MEMORY: MODEL WEIGHTS
# ============================================================

# ------------------------------------------------------------
# 1. MODEL CONFIGURATIONS
# ------------------------------------------------------------

models = {
    "Qwen3-8B": 8_000_000_000,
    "Qwen3-32B": 32_000_000_000,
}

precisions = {
    "FP32": 4,
    "FP16": 2,
    "BF16": 2,
    "INT8": 1,
    "INT4": 0.5,
}


# ------------------------------------------------------------
# 2. MEMORY CALCULATION
# ------------------------------------------------------------

def calculate_weight_memory(parameters, bytes_per_parameter):

    memory_bytes = parameters * bytes_per_parameter

    memory_gb = memory_bytes / (1000 ** 3)

    memory_gib = memory_bytes / (1024 ** 3)

    return memory_bytes, memory_gb, memory_gib


print("=" * 75)
print("THEORETICAL MODEL WEIGHT MEMORY")
print("=" * 75)

for model_name, parameters in models.items():

    print(f"\n{model_name}")
    print("-" * 75)

    for precision, bytes_per_param in precisions.items():

        memory_bytes, memory_gb, memory_gib = (
            calculate_weight_memory(
                parameters,
                bytes_per_param
            )
        )

        print(
            f"{precision:8} | "
            f"{memory_gb:8.2f} GB | "
            f"{memory_gib:8.2f} GiB"
        )


# ------------------------------------------------------------
# 3. INSPECT ACTUAL GPU
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("GPU INFORMATION")
print("=" * 75)

if torch.cuda.is_available():

    device = torch.device("cuda")

    properties = torch.cuda.get_device_properties(device)

    total_memory = properties.total_memory

    print("GPU:", properties.name)

    print(
        "Total VRAM:",
        round(total_memory / (1024 ** 3), 2),
        "GiB"
    )

    free_memory, total_memory = torch.cuda.mem_get_info(device)

    print(
        "Currently free:",
        round(free_memory / (1024 ** 3), 2),
        "GiB"
    )

    print(
        "Currently used:",
        round(
            (total_memory - free_memory) / (1024 ** 3),
            2
        ),
        "GiB"
    )

else:

    print("CUDA GPU not available")


# ------------------------------------------------------------
# 4. COMPARE MODEL WEIGHTS AGAINST GPU CAPACITY
# ------------------------------------------------------------

if torch.cuda.is_available():

    available_gib = free_memory / (1024 ** 3)

    print("\n" + "=" * 75)
    print("WEIGHT FIT ESTIMATION")
    print("=" * 75)

    for model_name, parameters in models.items():

        print(f"\n{model_name}")

        for precision, bytes_per_param in precisions.items():

            _, _, memory_gib = calculate_weight_memory(
                parameters,
                bytes_per_param
            )

            fits = memory_gib < available_gib

            print(
                f"{precision:8} | "
                f"Required: {memory_gib:6.2f} GiB | "
                f"Fits available VRAM: {fits}"
            )

print("\n" + "=" * 75)
print("EXERCISE COMPLETED")
print("=" * 75)
