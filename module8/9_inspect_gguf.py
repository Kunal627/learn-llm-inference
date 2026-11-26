from gguf import GGUFReader
from collections import Counter
from pathlib import Path
import sys

model_path = Path("<path_to_gguf_model>")

print("=" * 65)
print("GGUF MODEL INSPECTION")
print("=" * 65)

reader = GGUFReader(str(model_path))

print(f"\nFile: {model_path.name}")
print(f"File size: {model_path.stat().st_size / (1024**3):.3f} GiB")

print("\nMETADATA")
print("-" * 65)

for key in [
    "general.architecture",
    "general.name",
    "general.file_type",
    "general.quantization_version",
    "qwen3.block_count",
    "qwen3.embedding_length",
    "qwen3.attention.head_count",
    "qwen3.context_length",
]:
    if key in reader.fields:
        print(f"{key}: {reader.fields[key].contents()}")

print("\nTENSOR QUANTIZATION TYPES")
print("-" * 65)

counts = Counter(str(tensor.tensor_type) for tensor in reader.tensors)

for tensor_type, count in counts.items():
    print(f"{tensor_type}: {count} tensors")

print(f"\nTotal tensors: {len(reader.tensors)}")

print("\nSAMPLE TENSORS")
print("-" * 65)

for tensor in reader.tensors[:20]:
    print(
        f"{tensor.name:55} "
        f"shape={str(tensor.shape):25} "
        f"type={tensor.tensor_type}"
    )