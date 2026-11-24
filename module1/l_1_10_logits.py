import torch

tokens = ["Paris", "London", "Berlin", "Rome", "Tokyo"]

# Pretend these are the model's output logits
logits = torch.tensor([5.0, 2.0, 1.0, 0.5, -1.0])

# Convert logits into probabilities
probabilities = torch.softmax(logits, dim=-1)

for token, logit, probability in zip(tokens, logits, probabilities):
    print(
        f"{token:8s} "
        f"logit={logit.item():5.2f} "
        f"probability={probability.item():.4f}"
    )

print("Probability sum:", probabilities.sum().item())
print("Highest probability token:", tokens[probabilities.argmax().item()])



logits = torch.tensor([5.0, 2.0, 1.0, 0.5, -1.0])

for temperature in [0.5, 1.0, 2.0]:
    probabilities = torch.softmax(logits / temperature, dim=-1)

    print(f"\nTemperature = {temperature}")
    print(probabilities.tolist())


from transformers import AutoConfig

config = AutoConfig.from_pretrained("Qwen/Qwen3-8B")

print("Hidden size:", config.hidden_size)
print("Number of layers:", config.num_hidden_layers)
print("Vocabulary size:", config.vocab_size)