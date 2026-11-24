import time
import statistics
import torch

if __package__:
    from .tiny_llm import TinyLLM, append_token, autoregressive_generate, select_next_token
else:
    from tiny_llm import TinyLLM, append_token, autoregressive_generate, select_next_token

def benchmark_naive_decode(
    model,
    prompt_length,
    batch_size=1,
    max_new_tokens=20,
    warmup=3,
    iterations=10,
):
    model.eval()
    device = next(model.parameters()).device
    input_ids = torch.randint(
        0,
        model.token_embedding.num_embeddings,
        (batch_size, prompt_length),
        device=device,
    )

    for _ in range(warmup):
        autoregressive_generate(model, input_ids, max_new_tokens=max_new_tokens, debug=False)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    iteration_times = []
    for _ in range(iterations):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start_time = time.perf_counter()
        autoregressive_generate(model, input_ids, max_new_tokens=max_new_tokens, debug=False)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        iteration_times.append(time.perf_counter() - start_time)

    median_time = statistics.median(iteration_times)
    return {
        "prompt_length": prompt_length,
        "batch_size": batch_size,
        "new_tokens": max_new_tokens,
        "median_time_ms": median_time * 1000,
        "ms_per_token": median_time * 1000 / max_new_tokens,
        "tokens_per_second": batch_size * max_new_tokens / median_time,
    }

def main():
    batch_size = 2
    seq_len = 8
    vocab_size = 1000
    d_model = 64
    num_heads = 4
    num_layers = 2
    max_seq_len = 1024
    max_new_tokens = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    print("=" * 65)
    print("LESSON 3.2 - AUTOREGRESSIVE DECODING")
    print("=" * 65)
    print("Device:", device)
    print("Batch size:", batch_size)
    print("Vocabulary size:", vocab_size)
    print("Model dimension:", d_model)
    print("Attention heads:", num_heads)
    print("Head dimension:", d_model // num_heads)

    model = TinyLLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        mlp_dim=4 * d_model,
    ).to(device)
    model.eval()
    total_params = sum(parameter.numel() for parameter in model.parameters())
    print("\nTotal parameters:", f"{total_params:,}")

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    print("\nInitial input IDs:")
    print(input_ids)
    print("Initial input shape:", input_ids.shape)
    generated_ids = autoregressive_generate(
        model, input_ids, max_new_tokens=max_new_tokens, debug=True
    )

    print("\n" + "=" * 65)
    print("GENERATION COMPLETE")
    print("=" * 65)
    print("Original prompt:")
    print(input_ids)
    print("\nGenerated sequence:")
    print(generated_ids)
    print("\nOriginal shape:", input_ids.shape)
    print("Final shape:", generated_ids.shape)
    print("Number of generated tokens:", generated_ids.shape[1] - input_ids.shape[1])

    print("\n" + "=" * 90)
    print("NAIVE DECODING BENCHMARK")
    print("=" * 90)
    for prompt_length in [8, 32, 128, 256, 512]:
        result = benchmark_naive_decode(
            model=model,
            prompt_length=prompt_length,
            batch_size=1,
            max_new_tokens=20,
            warmup=3,
            iterations=10,
        )
        print(
            f"Prompt={result['prompt_length']:4d} | "
            f"Total={result['median_time_ms']:8.2f} ms | "
            f"Avg/token={result['ms_per_token']:7.2f} ms | "
            f"Throughput={result['tokens_per_second']:8.2f} tokens/s"
        )

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print("Device:", device)
    print("Model parameters:", f"{total_params:,}")
    print("Batch size: 1")
    print("Generated tokens per request: 20")
    print("Decoding method: Greedy")
    print("KV cache: Disabled")
    print("Decode implementation: Full sequence recomputation")
    print("\nBenchmark completed.")


if __name__ == "__main__":
    main()
