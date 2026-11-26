
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"

print("GPU available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load pretrained model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

model.eval()

# Inspect model
print("\nMODEL INFORMATION")
print("Model class:", model.__class__.__name__)
print("Parameter count:", sum(p.numel() for p in model.parameters()))
print("Embedding dtype:", model.get_input_embeddings().weight.dtype)

# Inspect parameter placement
print("\nDEVICE PLACEMENT")
for name, param in list(model.named_parameters())[:5]:
    print(name, param.device, param.dtype)

# Run inference
prompt = "Explain how a GPU executes a matrix multiplication."

inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100
    )

print("\nGENERATED RESPONSE")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


allocated = torch.cuda.memory_allocated(0)
reserved = torch.cuda.memory_reserved(0)

print("\nGPU MEMORY")
print(f"Allocated: {allocated / (1024**3):.3f} GiB")
print(f"Reserved:  {reserved / (1024**3):.3f} GiB")

theoretical = sum(
    p.numel() * p.element_size()
    for p in model.parameters()
)

print(
    f"Theoretical weights: "
    f"{theoretical / (1024**3):.3f} GiB"
)