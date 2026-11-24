
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rope


class Attention(nn.Module):

    def __init__(
        self,
        hidden_size,
        num_query_heads,
        num_kv_heads,
        use_rope=True,
        rope_base=10000,
        dropout=0.0
    ):
        super().__init__()

        assert hidden_size % num_query_heads == 0
        assert num_query_heads % num_kv_heads == 0

        self.hidden_size = hidden_size
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads

        self.head_dim = hidden_size // num_query_heads
        self.use_rope = use_rope
        self.rope_base = rope_base
        self.dropout = dropout

        # Projection dimensions
        q_dim = num_query_heads * self.head_dim
        kv_dim = num_kv_heads * self.head_dim

        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_size, q_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, kv_dim, bias=False)

        # Output projection
        self.o_proj = nn.Linear(q_dim, hidden_size, bias=False)

    def split_heads(self, x, num_heads):
        """
        Input:  [B, S, num_heads * head_dim]
        Output: [B, num_heads, S, head_dim]
        """

        B, S, _ = x.shape

        x = x.reshape(
            B, S, num_heads, self.head_dim
        )

        return x.transpose(1, 2)

    def forward(self, x, is_causal=True, position_offset=0):
        """
        x: [B, S, hidden_size]

        Returns:
            output: [B, S, hidden_size]
        """

        B, S, _ = x.shape

        # ----------------------------------
        # STEP 1: Q, K, V projections
        # ----------------------------------

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # ----------------------------------
        # STEP 2: Split into attention heads
        # ----------------------------------

        q = self.split_heads(q, self.num_query_heads)
        k = self.split_heads(k, self.num_kv_heads)
        v = self.split_heads(v, self.num_kv_heads)

        # ----------------------------------
        # STEP 3: Apply RoPE to Q and K
        # ----------------------------------

        if self.use_rope:
            q, k = apply_rope(
                q, k,
                base=self.rope_base,
                position_offset=position_offset
            )

        # ----------------------------------
        # STEP 4: Expand K and V for GQA/MQA
        # ----------------------------------

        groups = self.num_query_heads // self.num_kv_heads

        if groups > 1:
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)

        # Now Q, K, V all have Hq heads.
        # This explicit expansion is educational,
        # not the most memory-efficient production approach.

        # ----------------------------------
        # STEP 5: Scaled Dot-Product Attention
        # ----------------------------------

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0
        )

        # ----------------------------------
        # STEP 6: Merge attention heads
        # ----------------------------------

        # [B, H, S, D] -> [B, S, H, D]
        output = output.transpose(1, 2).contiguous()

        # [B, S, H * D]
        output = output.reshape(B, S, self.hidden_size)

        # ----------------------------------
        # STEP 7: Output projection
        # ----------------------------------

        output = self.o_proj(output)

        return output
