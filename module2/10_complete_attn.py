
import torch

from src.attn import Attention


torch.manual_seed(42)

B = 2
S = 5
HIDDEN = 8
NUM_Q_HEADS = 4

x = torch.randn(B, S, HIDDEN)

print("INPUT")
print("X:", x.shape)


def run_experiment(name, num_kv_heads):

    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)

    model = Attention(
        hidden_size=HIDDEN,
        num_query_heads=NUM_Q_HEADS,
        num_kv_heads=num_kv_heads,
        use_rope=True
    )

    # Inspect projection dimensions
    print("\nProjection dimensions:")
    print("Q projection:", model.q_proj.weight.shape)
    print("K projection:", model.k_proj.weight.shape)
    print("V projection:", model.v_proj.weight.shape)
    print("O projection:", model.o_proj.weight.shape)

    # Inspect head configuration
    print("\nHead configuration:")
    print("Query heads:", model.num_query_heads)
    print("KV heads:", model.num_kv_heads)
    print("Head dimension:", model.head_dim)

    model.eval()

    with torch.no_grad():
        output = model(x, is_causal=True)

    print("\nOutput:")
    print("Input shape :", x.shape)
    print("Output shape:", output.shape)
    print("Output finite:", torch.isfinite(output).all().item())

    assert output.shape == x.shape


run_experiment("MHA", num_kv_heads=4)
run_experiment("GQA", num_kv_heads=2)
run_experiment("MQA", num_kv_heads=1)

print("\nALL EXPERIMENTS COMPLETED")
