import torch

from tiny_llm import TinyLLM, select_next_token


BATCH_SIZE = 32
PROMPT_LEN = 512
VOCAB_SIZE = 4096
D_MODEL = 512
NUM_HEADS = 8
NUM_LAYERS = 8
MAX_SEQ_LEN = 2048
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    torch.manual_seed(42)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(42)

    print("Device:", DEVICE)
    print("GPU:", torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU")

    model = TinyLLM(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
    ).to(DEVICE).eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print("\nModel parameters:", f"{parameter_count:,}")

    prompt_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, PROMPT_LEN), device=DEVICE)
    print("\nPrompt shape:", prompt_ids.shape)
    print("\n" + "=" * 65)
    print("PREFILL")
    print("=" * 65)

    with torch.inference_mode():
        prefill_logits, past_key_values = model(prompt_ids, use_cache=True)

    print("\nPrefill logits:", prefill_logits.shape)
    print("\nKV CACHE AFTER PREFILL")
    for layer_idx, (key, value) in enumerate(past_key_values):
        print(f"\nLayer {layer_idx}")
        print("Key:", key.shape)
        print("Value:", value.shape)

    print("\n" + "=" * 65)
    print("CACHED DECODING")
    print("=" * 65)

    with torch.inference_mode():
        next_token = select_next_token(prefill_logits[:, -1, :])
        generated_tokens = [next_token]
        print("\nFirst generated token:", next_token.shape)

        for step in range(3):
            print(f"\nDECODE STEP {step + 1}")
            logits, past_key_values = model(
                next_token, past_key_values=past_key_values, use_cache=True
            )
            print("Logits:", logits.shape)
            next_token = select_next_token(logits[:, -1, :])
            generated_tokens.append(next_token)
            print("Next token:", next_token.shape)
            print("Updated cache:", past_key_values[0][0].shape)

    print("\nGenerated tokens:")
    print(torch.cat(generated_tokens, dim=1))

    print("\n" + "=" * 65)
    print("CORRECTNESS TEST: CACHED VS NAIVE")
    print("=" * 65)

    test_prompt = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, PROMPT_LEN), device=DEVICE)
    fixed_new_tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, 4), device=DEVICE)

    with torch.inference_mode():
        cached_logits, cache = model(test_prompt, use_cache=True)
        naive_logits = model(test_prompt)
        prefill_difference = (cached_logits - naive_logits).abs().max().item()
        print("\nPrefill max logit difference:", prefill_difference)

        current_sequence = test_prompt
        for step in range(fixed_new_tokens.shape[1]):
            new_token = fixed_new_tokens[:, step:step + 1]
            cached_logits, cache = model(new_token, past_key_values=cache, use_cache=True)
            current_sequence = torch.cat([current_sequence, new_token], dim=1)
            naive_logits = model(current_sequence)
            cached_last = cached_logits[:, -1, :]
            naive_last = naive_logits[:, -1, :]
            max_difference = (cached_last - naive_last).abs().max().item()
            print(
                f"Step {step + 1}: Sequence={current_sequence.shape[1]}, "
                f"Cached Q=1, Cache K/V length={cache[0][0].shape[2]}, "
                f"Max difference={max_difference:.8f}"
            )
            assert torch.allclose(cached_last, naive_last, atol=1e-5, rtol=1e-4), (
                "Cached and naive outputs do not match!"
            )

    print("\nSUCCESS: Cached and naive logits match!")


if __name__ == "__main__":
    main()