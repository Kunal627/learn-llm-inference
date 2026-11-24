import statistics
import time
import torch
import torch.nn as nn

if __package__:
    from .tiny_llm import TinyLLM, prefill
else:
    from tiny_llm import TinyLLM, prefill


def benchmark_prefill_gpu(
    model,
    vocab_size,
    seq_len,
    batch_size=1,
    warmup=20,
    iterations=100
):

    device = next(model.parameters()).device
    input_ids = torch.randint(
        0,
        vocab_size,
        (batch_size, seq_len),
        device=device
    )

    # Warmup
    for _ in range(warmup):
        with torch.inference_mode():
            _ = model(input_ids)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    latencies_ms = []

    for _ in range(iterations):

        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            with torch.cuda.device(device):
                start_event.record()
                with torch.inference_mode():
                    _ = model(input_ids)
                end_event.record()
            end_event.synchronize()
            latency = start_event.elapsed_time(end_event)
        else:
            start_time = time.perf_counter()
            with torch.inference_mode():
                _ = model(input_ids)
            latency = (time.perf_counter() - start_time) * 1000

        latencies_ms.append(latency)

    sorted_latencies = sorted(latencies_ms)

    median_ms = statistics.median(sorted_latencies)

    p95_index = int(0.95 * (len(sorted_latencies) - 1))
    p95_ms = sorted_latencies[p95_index]

    prompt_tokens = batch_size * seq_len

    tokens_per_second = (
        prompt_tokens / (median_ms / 1000)
    )

    return {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "prompt_tokens_per_second": tokens_per_second
    }


def main():

    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)

    BATCH_SIZE = 2
    SEQ_LEN = 8
    VOCAB_SIZE = 1000

    D_MODEL = 64
    NUM_HEADS = 4
    HEAD_DIM = D_MODEL // NUM_HEADS

    print("Batch size:", BATCH_SIZE)
    print("Sequence length:", SEQ_LEN)
    print("Vocabulary size:", VOCAB_SIZE)
    print("Model dimension:", D_MODEL)
    print("Number of heads:", NUM_HEADS)
    print("Head dimension:", HEAD_DIM)

    input_ids = torch.randint(
        low=0,
        high=VOCAB_SIZE,
        size=(BATCH_SIZE, SEQ_LEN),
        device=device
    )

    print("\nInput token IDs:")
    print(input_ids)

    print("\nInput shape:")
    print(input_ids.shape)

    embedding = nn.Embedding(VOCAB_SIZE, D_MODEL).to(device)
    x = embedding(input_ids)
    print("\nEmbedding output shape:")
    print(x.shape)

    model = TinyLLM(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=2,
        max_seq_len=512
    ).to(device)

    model.eval()

    print("\nModel created")

    print("\n========== PREFILL ==========")

    logits = prefill(
        model,
        input_ids,
        debug=True
    )

    print("\nPrefill completed")

    print("Logits shape:", logits.shape)

    next_token_logits = logits[:, -1, :]

    print("\nNext token logits:")
    print(next_token_logits.shape)

    probabilities = torch.softmax(
        next_token_logits,
        dim=-1
    )

    print("Logits:")
    print(next_token_logits)

    print("\nProbabilities:")
    print(probabilities)

    print("\nProbability sums per batch entry:")
    print(probabilities.sum(dim=-1))

    next_token = torch.argmax(
        probabilities,
        dim=-1
    )

    print("\nSelected token IDs:", next_token.tolist())

    prompt_lengths = [8, 32, 128, 256, 512]

    results = []

    for seq_len in prompt_lengths:

        result = benchmark_prefill_gpu(
            model=model,
            vocab_size=VOCAB_SIZE,
            seq_len=seq_len,
            batch_size=1,
            warmup=20,
            iterations=100
        )

        results.append(result)

        print(
            f"Seq={result['seq_len']:4d} | "
            f"Median={result['median_ms']:.3f} ms | "
            f"P95={result['p95_ms']:.3f} ms | "
            f"Throughput={result['prompt_tokens_per_second']:.0f} tokens/s"
        )


if __name__ == "__main__":
    main()