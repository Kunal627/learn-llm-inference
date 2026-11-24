
import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedQueryAttention(nn.Module):

    def __init__(
        self,
        hidden_dim,
        num_q_heads,
        num_kv_heads
    ):
        super().__init__()

        assert hidden_dim % num_q_heads == 0
        assert num_q_heads % num_kv_heads == 0

        self.hidden_dim = hidden_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads

        self.head_dim = hidden_dim // num_q_heads

        # Query projection
        self.q_proj = nn.Linear(
            hidden_dim,
            num_q_heads * self.head_dim,
            bias=False
        )

        # Key projection
        self.k_proj = nn.Linear(
            hidden_dim,
            num_kv_heads * self.head_dim,
            bias=False
        )

        # Value projection
        self.v_proj = nn.Linear(
            hidden_dim,
            num_kv_heads * self.head_dim,
            bias=False
        )

        # Final output projection
        self.out_proj = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False
        )

    def forward(self, X):

        B, S, D = X.shape

        # ------------------------------------------
        # 1. Project Q, K and V
        # ------------------------------------------

        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)

        # ------------------------------------------
        # 2. Split into heads
        # [B, S, H, D_head]
        # ------------------------------------------

        Q = Q.view(
            B, S, self.num_q_heads, self.head_dim
        )

        K = K.view(
            B, S, self.num_kv_heads, self.head_dim
        )

        V = V.view(
            B, S, self.num_kv_heads, self.head_dim
        )

        # ------------------------------------------
        # 3. Move heads before sequence dimension
        # [B, H, S, D_head]
        # ------------------------------------------

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # ------------------------------------------
        # 4. Share K/V across Query groups
        # ------------------------------------------

        groups = self.num_q_heads // self.num_kv_heads

        if groups > 1:
            K = K.repeat_interleave(groups, dim=1)
            V = V.repeat_interleave(groups, dim=1)

        # ------------------------------------------
        # 5. Scaled dot-product attention
        # ------------------------------------------

        attention_output = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            dropout_p=0.0,
            is_causal=True
        )

        # ------------------------------------------
        # 6. Merge attention heads
        # ------------------------------------------

        attention_output = attention_output.transpose(1, 2)

        attention_output = attention_output.contiguous().view(
            B, S, self.hidden_dim
        )

        # ------------------------------------------
        # 7. Output projection
        # ------------------------------------------

        output = self.out_proj(attention_output)

        return output


# ==================================================
# Experiment: MHA, GQA and MQA
# ==================================================

if __name__ == "__main__":

    torch.manual_seed(42)

    batch_size = 2
    seq_len = 10
    hidden_dim = 64
    num_q_heads = 8

    X = torch.randn(batch_size, seq_len, hidden_dim)

    configurations = {
        "MHA": 8,
        "GQA": 2,
        "MQA": 1
    }

    for name, num_kv_heads in configurations.items():

        model = GroupedQueryAttention(
            hidden_dim=hidden_dim,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads
        )

        output = model(X)

        params = sum(
            p.numel() for p in model.parameters()
        )

        print(f"\n{name}")
        print("KV heads:", num_kv_heads)
        print("Output shape:", output.shape)
        print("Parameters:", params)

        assert output.shape == X.shape

    print("\nAll attention variants executed successfully!")
