import torch

torch.manual_seed(42)

tokens = ["Paris", "London", "Berlin", "Rome", "Tokyo"]
logits = torch.tensor([5.0, 2.0, 1.0, 0.5, -1.0])

def decode(logits, temperature=1.0, top_k=None):
    scaled_logits = logits / temperature

    if top_k is not None:
        values, indices = torch.topk(scaled_logits, top_k)

        filtered = torch.full_like(
            scaled_logits, float("-inf")
        )
        filtered[indices] = values
        scaled_logits = filtered

    probabilities = torch.softmax(scaled_logits, dim=-1)

    selected_id = torch.multinomial(
        probabilities, num_samples=1
    ).item()

    return tokens[selected_id], probabilities

for temperature in [0.3, 1.0, 2.0]:
    selected, probs = decode(
        logits,
        temperature=temperature,
        top_k=3
    )

    print(
        f"Temperature={temperature}: "
        f"selected={selected}, "
        f"probabilities={probs.tolist()}"
    )