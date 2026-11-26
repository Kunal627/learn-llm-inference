
from pathlib import Path
from huggingface_hub import snapshot_download

MODEL_NAME = "Qwen/Qwen3-0.6B"

model_dir = Path(
    snapshot_download(
        repo_id=MODEL_NAME,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.model"
        ]
    )
)

print("Model directory:", model_dir)

print("\nFiles:")
for file in sorted(model_dir.iterdir()):
    print(f"{file.name:55} {file.stat().st_size / (1024**2):.2f} MB")


from safetensors import safe_open

checkpoint_file = model_dir / "model.safetensors"

with safe_open(
    checkpoint_file,
    framework="pt",
    device="cpu"
) as f:

    print("Total tensors:", len(f.keys()))

    for key in list(f.keys())[:15]:
        tensor = f.get_slice(key)

        print(
            key,
            "Shape:", tensor.get_shape(),
            "Dtype:", tensor.get_dtype()
        )