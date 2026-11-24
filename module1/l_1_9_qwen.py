
from transformers import AutoConfig

config = AutoConfig.from_pretrained("Qwen/Qwen3-8B")

print("Hidden size:", config.hidden_size)
print("Layers:", config.num_hidden_layers)
print("Query heads:", config.num_attention_heads)
print("KV heads:", config.num_key_value_heads)
print("Head dimension:", config.head_dim)
print("Intermediate size:", config.intermediate_size)
