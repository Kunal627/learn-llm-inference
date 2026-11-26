
import torch
import torch.nn as nn

# 1. Define architecture
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(1000, 128)
        self.layer1 = nn.Linear(128, 256)
        self.output = nn.Linear(256, 1000)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.relu(self.layer1(x))
        return self.output(x)


# 2. Create a model and initialize its parameters
model_a = SimpleModel()

# Save its parameters as a checkpoint
torch.save(model_a.state_dict(), "model_checkpoint.pth")

# 3. Create another model
# This model gets its own randomly initialized parameters
model_b = SimpleModel()

# Check whether their weights are initially identical
print("Before loading:")
print(torch.equal(
    model_a.layer1.weight,
    model_b.layer1.weight
))

# 4. Load the checkpoint into model_b
checkpoint = torch.load(
    "model_checkpoint.pth",
    map_location="cpu",
    weights_only=True
)

model_b.load_state_dict(checkpoint)

# 5. Compare weights after loading
print("After loading:")
print(torch.equal(
    model_a.layer1.weight,
    model_b.layer1.weight
))

# 6. Prepare for inference
model_b.eval()

tokens = torch.tensor([[10, 25, 42]])
with torch.inference_mode():
    output = model_b(tokens)

print("Output shape:", output.shape)
