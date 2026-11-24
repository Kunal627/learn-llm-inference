
import time
import statistics

import torch
import torch.nn as nn


# --------------------------------------------------
# 1. DEVICE AND CONFIGURATION
# --------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_REQUESTS = 16
SEQ_LEN = 128
VOCAB_SIZE = 32000
D_MODEL = 128
NUM_HEADS = 4
NUM_LAYERS = 2

WARMUP = 5
REPEATS = 30

print("=" * 65)
print("BATCHING EXPERIMENT")
print("=" * 65)
print("Device:", DEVICE)
print("Requests:", NUM_REQUESTS)
print("Sequence length:", SEQ_LEN)


# --------------------------------------------------
# 2. A SMALL CAUSAL TRANSFORMER
# --------------------------------------------------

class TinyLLM(nn.Module):

    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(
            VOCAB_SIZE,
            D_MODEL
        )

        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NUM_HEADS,
            dim_feedforward=4 * D_MODEL,
            dropout=0.0,
            batch_first=True,
            activation="gelu"
        )

        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=NUM_LAYERS
        )

        self.lm_head = nn.Linear(
            D_MODEL,
            VOCAB_SIZE
        )

    def forward(self, token_ids, causal_mask):

        x = self.embedding(token_ids)

        x = self.transformer(
            x,
            mask=causal_mask
        )

        logits = self.lm_head(x)

        return logits


model = TinyLLM().to(DEVICE)
model.eval()

# True means future positions are blocked.
causal_mask = torch.triu(
    torch.ones(
        SEQ_LEN,
        SEQ_LEN,
        dtype=torch.bool,
        device=DEVICE
    ),
    diagonal=1
)


# --------------------------------------------------
# 3. GENERATE REQUESTS
# --------------------------------------------------

torch.manual_seed(42)

requests = [
    torch.randint(
        0,
        VOCAB_SIZE,
        (1, SEQ_LEN),
        device=DEVICE
    )
    for _ in range(NUM_REQUESTS)
]

# Combine requests into one static batch.
static_batch = torch.cat(requests, dim=0)

print("\nRequest tensor shapes:")
print("Individual request:", requests[0].shape)
print("Static batch:", static_batch.shape)


# --------------------------------------------------
# 4. INFERENCE STRATEGIES
# --------------------------------------------------

@torch.inference_mode()
def run_single_requests():

    outputs = []

    for request in requests:
        output = model(request, causal_mask)
        outputs.append(output)

    return outputs


@torch.inference_mode()
def run_static_batch():

    return model(static_batch, causal_mask)


# --------------------------------------------------
# 5. TIMING UTILITY
# --------------------------------------------------

def synchronize():

    if DEVICE == "cuda":
        torch.cuda.synchronize()


def measure_ms(fn):

    timings = []

    for _ in range(REPEATS):

        synchronize()

        if DEVICE == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            fn()
            end.record()

            end.synchronize()
            elapsed = start.elapsed_time(end)

        else:
            start = time.perf_counter()
            fn()
            elapsed = (
                time.perf_counter() - start
            ) * 1000

        timings.append(elapsed)

    return statistics.median(timings)


# --------------------------------------------------
# 6. WARMUP
# --------------------------------------------------

print("\nWarming up...")

for _ in range(WARMUP):
    run_single_requests()
    run_static_batch()

synchronize()


# --------------------------------------------------
# 7. BENCHMARK
# --------------------------------------------------

single_ms = measure_ms(run_single_requests)
static_ms = measure_ms(run_static_batch)

single_throughput = (
    NUM_REQUESTS / (single_ms / 1000)
)

static_throughput = (
    NUM_REQUESTS / (static_ms / 1000)
)

single_tokens_per_second = (
    NUM_REQUESTS * SEQ_LEN / (single_ms / 1000)
)

static_tokens_per_second = (
    NUM_REQUESTS * SEQ_LEN / (static_ms / 1000)
)

speedup = single_ms / static_ms


# --------------------------------------------------
# 8. CORRECTNESS CHECK
# --------------------------------------------------

with torch.inference_mode():

    single_outputs = run_single_requests()
    batch_output = run_static_batch()

    max_error = max(
        (
            single_outputs[i] - batch_output[i:i + 1]
        ).abs().max().item()
        for i in range(NUM_REQUESTS)
    )


# --------------------------------------------------
# 9. RESULTS
# --------------------------------------------------

print("\n" + "=" * 65)
print("RESULTS")
print("=" * 65)

print(
    f"{'Metric':<30}"
    f"{'Single':>15}"
    f"{'Static batch':>18}"
)

print("-" * 65)

print(
    f"{'Total latency (ms)':<30}"
    f"{single_ms:>15.3f}"
    f"{static_ms:>18.3f}"
)

print(
    f"{'Requests / second':<30}"
    f"{single_throughput:>15.2f}"
    f"{static_throughput:>18.2f}"
)

print(
    f"{'Tokens / second':<30}"
    f"{single_tokens_per_second:>15.2f}"
    f"{static_tokens_per_second:>18.2f}"
)

print("-" * 65)

print(f"Static batch speedup: {speedup:.2f}x")
print(f"Maximum output difference: {max_error:.8f}")

print("\nExperiment completed.")
