
import math
import torch

assert torch.cuda.is_available(), "CUDA is not available!"

device = torch.device("cuda:0")

# --------------------------------------------------
# 1. Create input tensors
# --------------------------------------------------

N = 1024
A = torch.arange(N, dtype=torch.float32, device=device)
B = torch.ones(N, dtype=torch.float32, device=device)

# --------------------------------------------------
# 2. Execute vector addition on GPU
# --------------------------------------------------

torch.cuda.synchronize()

C = A + B

torch.cuda.synchronize()

# --------------------------------------------------
# 3. Verify output
# --------------------------------------------------

expected = A + B

assert torch.equal(C, expected)

print("=" * 60)
print("GPU VECTOR ADDITION")
print("=" * 60)

print("Device:", torch.cuda.get_device_name())
print("Input shape:", A.shape)
print("Output shape:", C.shape)
print("First 10 results:", C[:10].cpu().tolist())

# --------------------------------------------------
# 4. Simulate CUDA thread/block organization
# --------------------------------------------------

BLOCK_SIZE = 128
WARP_SIZE = 32

num_blocks = math.ceil(N / BLOCK_SIZE)
warps_per_block = BLOCK_SIZE // WARP_SIZE

print("\n" + "=" * 60)
print("CONCEPTUAL CUDA LAUNCH CONFIGURATION")
print("=" * 60)

print("Total elements:", N)
print("Threads per block:", BLOCK_SIZE)
print("Warps per block:", warps_per_block)
print("Number of blocks:", num_blocks)
print("Total warps:", num_blocks * warps_per_block)

# --------------------------------------------------
# 5. Map thread IDs to global element indices
# --------------------------------------------------

print("\n" + "=" * 60)
print("THREAD-TO-ELEMENT MAPPING")
print("=" * 60)

for block_id in range(2):
    print(f"\nBlock {block_id}")

    for thread_id in range(8):
        global_id = block_id * BLOCK_SIZE + thread_id
        warp_id = thread_id // WARP_SIZE
        lane_id = thread_id % WARP_SIZE

        print(
            f"Thread {thread_id:3d} | "
            f"Warp {warp_id} | "
            f"Lane {lane_id:2d} | "
            f"Element {global_id}"
        )

# --------------------------------------------------
# 6. Demonstrate the general indexing formula
# --------------------------------------------------

print("\n" + "=" * 60)
print("CUDA GLOBAL THREAD INDEX")
print("=" * 60)

print(
    "global_id = block_id * block_size + thread_id"
)

print(
    "Example:",
    "1 * 128 + 5 =",
    1 * BLOCK_SIZE + 5
)

print("\nExperiment completed successfully.")
