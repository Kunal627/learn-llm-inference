
import torch

def calculate_attention_memory(
    batch_size,
    num_heads,
    seq_len,
    dtype=torch.bfloat16
):
    element_size = torch.tensor([], dtype=dtype).element_size()

    num_elements = (
        batch_size
        * num_heads
        * seq_len
        * seq_len
    )

    memory_bytes = num_elements * element_size
    memory_mib = memory_bytes / (1024 ** 2)
    memory_gib = memory_bytes / (1024 ** 3)

    return num_elements, memory_mib, memory_gib


B = 1
H = 32

for S in [1024, 2048, 4096, 8192, 16384, 32768]:

    elements, mib, gib = calculate_attention_memory(
        B, H, S
    )

    print(f"\nSequence length: {S}")
    print(f"Number of scores: {elements:,}")
    print(f"Memory: {mib:.2f} MiB")
    print(f"Memory: {gib:.2f} GiB")
