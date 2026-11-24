
import time
import statistics
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. CONFIGURATION
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_REQUESTS = 16
VOCAB_SIZE = 32000
D_MODEL = 128
NUM_HEADS = 4
NUM_LAYERS = 2

WARMUP = 5
REPEATS = 30

torch.manual_seed(42)

if DEVICE == "cuda":
    torch.cuda.manual_seed_all(42)

print("=" * 75)
print("VARIABLE-LENGTH SEQUENCES AND PADDING")
print("=" * 75)
print("Device:", DEVICE)


# ============================================================
# 2. GENERATE VARIABLE-LENGTH REQUESTS
# ============================================================

lengths = [
    32, 40, 48, 56,
    64, 72, 80, 88,
    96, 104, 112, 120,
    128, 160, 192, 256
]

requests = [
    torch.randint(
        0,
        VOCAB_SIZE,
        (length,),
        device=DEVICE
    )
    for length in lengths
]

print("\nRequest lengths:")
print(lengths)

total_valid_tokens = sum(lengths)
max_length = max(lengths)

print("Total valid tokens:", total_valid_tokens)
print("Maximum sequence length:", max_length)


# ============================================================
# 3. TRANSFORMER MODEL
# ============================================================

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

    def forward(self, token_ids, lengths):

        batch_size, seq_len = token_ids.shape

        # True means future positions are blocked.
        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=token_ids.device
            ),
            diagonal=1
        )

        positions = torch.arange(
            seq_len,
            device=token_ids.device
        )

        # True means the position is padding.
        padding_mask = positions.unsqueeze(0) >= lengths.unsqueeze(1)

        x = self.embedding(token_ids)

        x = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )

        return self.lm_head(x)


model = TinyLLM().to(DEVICE)
model.eval()


# ============================================================
# 4. PREPARE PADDED BATCH
# ============================================================

PAD_TOKEN_ID = 0


def make_batch(batch_requests):

    batch_lengths = torch.tensor(
        [len(x) for x in batch_requests],
        device=DEVICE
    )

    max_len = int(batch_lengths.max().item())

    batch = torch.full(
        (len(batch_requests), max_len),
        PAD_TOKEN_ID,
        dtype=torch.long,
        device=DEVICE
    )

    for i, request in enumerate(batch_requests):

        batch[i, :len(request)] = request

    return batch, batch_lengths


full_batch, full_lengths = make_batch(requests)


# ============================================================
# 5. LENGTH BUCKETING
# ============================================================

def create_buckets(requests, bucket_size=4):

    sorted_requests = sorted(
        requests,
        key=len
    )

    buckets = []

    for i in range(0, len(sorted_requests), bucket_size):

        buckets.append(
            sorted_requests[i:i + bucket_size]
        )

    return buckets


buckets = create_buckets(
    requests,
    bucket_size=4
)

print("\nLength bucket composition:")

for i, bucket in enumerate(buckets):

    print(
        f"Bucket {i + 1}:",
        [len(x) for x in bucket]
    )


# ============================================================
# 6. INFERENCE STRATEGIES
# ============================================================

@torch.inference_mode()
def run_individual():

    outputs = []

    for request in requests:

        batch = request.unsqueeze(0)

        request_length = torch.tensor(
            [len(request)],
            device=DEVICE
        )

        output = model(
            batch,
            request_length
        )

        outputs.append(output)

    return outputs


@torch.inference_mode()
def run_padded_batch():

    return model(
        full_batch,
        full_lengths
    )


@torch.inference_mode()
def run_bucketed():

    outputs = []

    for bucket in buckets:

        batch, bucket_lengths = make_batch(bucket)

        output = model(
            batch,
            bucket_lengths
        )

        outputs.append(output)

    return outputs


# ============================================================
# 7. PADDING AND ATTENTION WORK ESTIMATES
# ============================================================

padded_token_positions = (
    NUM_REQUESTS * max_length
)

padding_tokens = (
    padded_token_positions - total_valid_tokens
)

padding_efficiency = (
    total_valid_tokens / padded_token_positions
) * 100


# For dense self-attention, score matrix work is
# proportional to the sum of sequence_length squared.

useful_attention_work = sum(
    length ** 2 for length in lengths
)

padded_attention_work = (
    NUM_REQUESTS * max_length ** 2
)

attention_efficiency = (
    useful_attention_work / padded_attention_work
) * 100


bucket_padded_positions = sum(
    len(bucket) * max(len(x) for x in bucket)
    for bucket in buckets
)

bucket_padding_tokens = (
    bucket_padded_positions - total_valid_tokens
)

bucket_padding_efficiency = (
    total_valid_tokens / bucket_padded_positions
) * 100


bucket_attention_work = sum(
    len(bucket) * max(len(x) for x in bucket) ** 2
    for bucket in buckets
)

bucket_attention_efficiency = (
    useful_attention_work / bucket_attention_work
) * 100


# ============================================================
# 8. BENCHMARK
# ============================================================

def synchronize():

    if DEVICE == "cuda":
        torch.cuda.synchronize()


def benchmark(fn):

    for _ in range(WARMUP):
        fn()

    synchronize()

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

            timings.append(
                start.elapsed_time(end)
            )

        else:

            start = time.perf_counter()

            fn()

            timings.append(
                (time.perf_counter() - start) * 1000
            )

    return statistics.median(timings)


print("\nWarming up and benchmarking...")

individual_ms = benchmark(run_individual)
padded_ms = benchmark(run_padded_batch)
bucketed_ms = benchmark(run_bucketed)


# ============================================================
# 9. THROUGHPUT
# ============================================================

def throughput(tokens, latency_ms):

    return tokens / (latency_ms / 1000)


individual_tps = throughput(
    total_valid_tokens,
    individual_ms
)

padded_tps = throughput(
    total_valid_tokens,
    padded_ms
)

bucketed_tps = throughput(
    total_valid_tokens,
    bucketed_ms
)


# ============================================================
# 10. CORRECTNESS
# ============================================================

with torch.inference_mode():

    individual_outputs = run_individual()
    padded_output = run_padded_batch()

    errors = []

    for i, individual_output in enumerate(individual_outputs):

        valid_length = lengths[i]

        difference = (
            individual_output[0]
            -
            padded_output[i, :valid_length]
        ).abs().max().item()

        errors.append(difference)

    max_error = max(errors)


# ============================================================
# 11. RESULTS
# ============================================================

print("\n" + "=" * 75)
print("PADDING ANALYSIS")
print("=" * 75)

print("Valid tokens:", total_valid_tokens)
print("Padded tensor positions:", padded_token_positions)
print("Padding tokens:", padding_tokens)

print(
    f"Single padded-batch token efficiency: "
    f"{padding_efficiency:.2f}%"
)

print(
    f"Length-bucketed token efficiency: "
    f"{bucket_padding_efficiency:.2f}%"
)

print(
    f"Dense attention work efficiency (padded): "
    f"{attention_efficiency:.2f}%"
)

print(
    f"Dense attention work efficiency (bucketed): "
    f"{bucket_attention_efficiency:.2f}%"
)


print("\n" + "=" * 75)
print("BENCHMARK RESULTS")
print("=" * 75)

print(
    f"{'Strategy':<22}"
    f"{'Latency ms':>14}"
    f"{'Valid tok/s':>17}"
)

print("-" * 75)

print(
    f"{'Individual':<22}"
    f"{individual_ms:>14.3f}"
    f"{individual_tps:>17.2f}"
)

print(
    f"{'Padded batch':<22}"
    f"{padded_ms:>14.3f}"
    f"{padded_tps:>17.2f}"
)

print(
    f"{'Length bucketed':<22}"
    f"{bucketed_ms:>14.3f}"
    f"{bucketed_tps:>17.2f}"
)

print("-" * 75)

print("Maximum valid-output difference:", max_error)

print("\nExperiment completed.")
