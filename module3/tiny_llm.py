import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, debug=False, past_key_value=None, use_cache=False):
        batch, seq_len, d_model = x.shape
        if debug:
            print("\nInput to attention:", x.shape)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        if debug:
            print("Q:", q.shape)
            print("K:", k.shape)
            print("V:", v.shape)

        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim)
        if debug:
            print("Q after head split:", q.shape)
            print("K after head split:", k.shape)
            print("V after head split:", v.shape)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if debug:
            print("Q after transpose:", q.shape)
            print("K after transpose:", k.shape)
            print("V after transpose:", v.shape)

        past_len = 0
        if past_key_value is not None:
            past_k, past_v = past_key_value
            past_len = past_k.shape[2]
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        query_positions = past_len + torch.arange(seq_len, device=x.device)
        key_positions = torch.arange(k.shape[2], device=x.device)
        causal_mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
        if debug:
            print("Attention scores:", scores.shape)
            print("Causal mask:", causal_mask.shape)
            print(causal_mask.int())
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        if debug:
            print("Attention weights:", attn_weights.shape)

        out = torch.matmul(attn_weights, v)
        if debug:
            print("Attention output before merge:", out.shape)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        if debug:
            print("Attention output after merge:", out.shape)
        out = self.out_proj(out)
        if debug:
            print("Final attention output:", out.shape)
        if use_cache:
            return out, (k, v)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, mlp_dim=None):
        super().__init__()
        if mlp_dim is None:
            mlp_dim = 4 * d_model
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, d_model),
        )

    def forward(self, x, debug=False, past_key_value=None, use_cache=False):
        attention_output = self.attention(
            self.ln1(x), debug=debug, past_key_value=past_key_value, use_cache=use_cache
        )
        if use_cache:
            attention_output, present_key_value = attention_output
        x = x + attention_output
        if debug:
            print("\nAfter attention residual:", x.shape)
        x = x + self.mlp(self.ln2(x))
        if debug:
            print("After MLP residual:", x.shape)
        if use_cache:
            return x, present_key_value
        return x


class TinyLLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_seq_len, mlp_dim=None):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, mlp_dim)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, debug=False, past_key_values=None, use_cache=False):
        batch, seq_len = input_ids.shape
        past_len = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        if past_len + seq_len > self.position_embedding.num_embeddings:
            raise ValueError("Sequence exceeds the model's maximum sequence length")
        token_emb = self.token_embedding(input_ids)
        position_ids = torch.arange(past_len, past_len + seq_len, device=input_ids.device)
        position_emb = self.position_embedding(position_ids)
        x = token_emb + position_emb.unsqueeze(0)
        if debug:
            print("\nToken embeddings:", token_emb.shape)
            print("Position embeddings:", position_emb.shape)
            print("Combined input:", x.shape)
        presents = []
        for index, block in enumerate(self.blocks):
            if debug:
                print(f"\n===== Transformer Layer {index} =====")
            past_key_value = past_key_values[index] if past_key_values is not None else None
            block_output = block(
                x, debug=debug, past_key_value=past_key_value, use_cache=use_cache
            )
            if use_cache:
                x, present_key_value = block_output
                presents.append(present_key_value)
            else:
                x = block_output
        x = self.final_norm(x)
        logits = self.lm_head(x)
        if debug:
            print("\nFinal hidden states:", x.shape)
            print("Final logits:", logits.shape)
        if use_cache:
            return logits, presents
        return logits


def prefill(model, input_ids, debug=False):
    """Return logits for every position in a batch of prompts."""
    with torch.inference_mode():
        return model(input_ids, debug=debug)


def pad_prompts(prompts, pad_id, device, max_length=None):
    if not prompts or any(len(prompt) == 0 for prompt in prompts):
        raise ValueError("Prompts must be a non-empty batch of non-empty sequences")
    lengths = torch.tensor([len(prompt) for prompt in prompts], device=device)
    if max_length is None:
        max_length = max(len(prompt) for prompt in prompts)
    if any(len(prompt) > max_length for prompt in prompts):
        raise ValueError("Prompt exceeds the maximum length")
    input_ids = torch.full(
        (len(prompts), max_length), pad_id, dtype=torch.long, device=device
    )
    for index, prompt in enumerate(prompts):
        input_ids[index, :len(prompt)] = torch.as_tensor(prompt, dtype=torch.long, device=device)
    attention_mask = (torch.arange(max_length, device=device)[None, :] < lengths[:, None]).long()
    return input_ids, attention_mask, lengths


def last_valid_logits(logits, prompt_lengths):
    batch_indices = torch.arange(logits.shape[0], device=logits.device)
    return logits[batch_indices, prompt_lengths - 1]


def select_next_token(logits):
    """Select one greedy token per batch entry, with shape [batch, 1]."""
    return torch.argmax(logits, dim=-1).unsqueeze(-1)


def append_token(input_ids, next_token):
    return torch.cat([input_ids, next_token], dim=1)


def autoregressive_generate(model, input_ids, max_new_tokens=10, debug=True):
    generated_ids = input_ids.clone()
    model.eval()
    with torch.inference_mode():
        for step in range(max_new_tokens):
            if debug:
                print("\n" + "=" * 65)
                print(f"DECODE STEP {step + 1}")
                print("=" * 65)
                print("Current input shape:", generated_ids.shape)
            logits = model(generated_ids)
            next_token_logits = logits[:, -1, :]
            next_token = select_next_token(next_token_logits)
            if debug:
                print("Logits shape:", logits.shape)
                print("Last position logits:", next_token_logits.shape)
                print("Selected token IDs:")
                print(next_token)
            generated_ids = append_token(generated_ids, next_token)
            if debug:
                print("Updated input shape:", generated_ids.shape)
    return generated_ids