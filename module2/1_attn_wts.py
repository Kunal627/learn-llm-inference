
import torch

# --------------------------------------------------
# 1. Input tokens
# --------------------------------------------------

tokens = ["The", "cat", "chased", "the", "mouse"]

# --------------------------------------------------
# 2. Value vectors (V)
# Shape: [sequence_length, head_dimension]
# --------------------------------------------------

V = torch.tensor([
    [1.0, 0.0, 0.0],  # The
    [0.0, 1.0, 0.0],  # cat
    [0.0, 0.0, 1.0],  # chased
    [1.0, 1.0, 0.0],  # the
    [1.0, 0.0, 1.0],  # mouse
])

# --------------------------------------------------
# 3. Attention weights for the token "mouse"
# --------------------------------------------------

attention_weights = torch.tensor([
    0.05,  # The
    0.30,  # cat
    0.40,  # chased
    0.05,  # the
    0.20,  # mouse
])

# --------------------------------------------------
# 4. Validate attention weights
# --------------------------------------------------

assert V.shape[0] == len(tokens)
assert attention_weights.shape[0] == len(tokens)

assert torch.all(attention_weights >= 0)
assert torch.isclose(
    attention_weights.sum(),
    torch.tensor(1.0)
)

# --------------------------------------------------
# 5. Weighted aggregation
# --------------------------------------------------

output = attention_weights @ V

# --------------------------------------------------
# 6. Display results
# --------------------------------------------------

print("Input tokens:", tokens)

print("\nAttention weights for 'mouse':")

for token, weight in zip(tokens, attention_weights):
    print(f"{token:10s}: {weight.item():.2f}")

print("\nContextual representation of 'mouse':")
print(output)
