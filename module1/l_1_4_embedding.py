
import torch
import torch.nn as nn

# Configuration
vocab_size = 1000
embedding_dim = 8

# Create an embedding matrix
embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=embedding_dim
)

# Five token IDs
token_ids = torch.tensor([10, 11, 12, 13, 14])

# Retrieve embeddings
vectors = embedding(token_ids)

print("Token IDs shape:", token_ids.shape)
print("Embedding shape:", vectors.shape)

print("\nEmbedding matrix shape:")
print(embedding.weight.shape)

print("\nFirst token embedding:")
print(vectors[0])
