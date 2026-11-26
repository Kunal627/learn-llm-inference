# Module 8 --- Inference Runtimes and TensorRT-LLM Serving

This lesson documents the hands-on TensorRT-LLM serving work completed
in Module 8: setting up the NVIDIA TensorRT-LLM Docker image inside
WSL2, launching a Hugging Face model with the PyTorch backend,
exercising its OpenAI-compatible API, benchmarking concurrent requests,
and measuring streaming latency.

> **Scope:** This lesson completes the TensorRT-LLM **PyTorch-backend
> serving** exercises. It does **not** yet cover building a TensorRT
> optimized engine from a checkpoint. Engine conversion/build, engine
> loading, and a PyTorch-backend-versus-engine comparison remain future
> work.

------------------------------------------------------------------------

## 1. Learning objectives

By the end of this lesson, you should understand how to:

-   Run TensorRT-LLM in a Linux Docker container from WSL2.
-   Serve a Hugging Face model using `trtllm-serve`.
-   Configure basic serving limits such as maximum batch size, token
    budget, sequence length, and GPU memory fraction.
-   Call an OpenAI-compatible chat-completions endpoint.
-   Measure single-request streaming performance.
-   Submit multiple concurrent requests and measure aggregate throughput
    and request latency.
-   Understand why concurrency and maximum batch size are related but
    not interchangeable.
-   Inspect basic health, server configuration, and observability
    endpoints.
-   Distinguish request latency, TTFT, TPOT, inter-chunk latency, and
    aggregate throughput.

------------------------------------------------------------------------

## 2. Environment used

The practical work was performed in this environment:

  Component                                          Configuration
  -------------------------------------------------- ---------------------------------------------
  Host OS                                            Windows 11
  Linux environment                                  WSL2, Ubuntu 24.04
  Container runtime                                  Docker available from WSL2
  GPU                                                NVIDIA RTX 3500 Ada Generation Laptop GPU
  GPU memory                                         Approximately 12 GB
  Compute capability                                 8.9 (SM89)
  Host NVIDIA driver                                 580.92
  Host CUDA compatibility reported by `nvidia-smi`   CUDA 13.0
  TensorRT-LLM container                             `nvcr.io/nvidia/tensorrt-llm/release:1.3.0`
  TensorRT-LLM version                               1.3.0
  Model                                              `Qwen/Qwen3-0.6B`
  Backend                                            PyTorch
  Container name                                     `trtllm-server`
  API port                                           8000
  Workspace                                          `~/llm-inference/module8`

The container reported NVIDIA Release 26.09, PyTorch `2.14.0a0`, and
CUDA 13.4. CUDA Minor Version Compatibility mode was enabled because the
host driver reports CUDA 13.0 support. The container successfully
detected and used the GPU.

Your own environment may report different versions or memory
availability.

------------------------------------------------------------------------

## 3. Prepare the WSL workspace

Open Ubuntu 24.04 in WSL:

``` bash
mkdir -p ~/llm-inference/module8
cd ~/llm-inference/module8
```

Check Docker:

``` bash
docker --version
docker info
```

Check GPU visibility from WSL:

``` bash
nvidia-smi
```

Confirm that the NVIDIA GPU is visible before launching the container.

### Check whether another inference server is using the GPU

The `local-llm` llama.cpp container may occupy a substantial amount of
GPU memory. Inspect running containers:

``` bash
docker ps
```

If you need to free GPU memory for this experiment, stop the llama.cpp
server temporarily:

``` bash
docker stop local-llm
```

This stops the container; it does not delete the container or the model.
Restart it later if needed:

``` bash
docker start local-llm
```

Do not run both inference servers simultaneously on a 12 GB GPU unless
you have checked that sufficient memory is available.

------------------------------------------------------------------------

## 4. Pull and inspect the TensorRT-LLM image

Pull the NVIDIA image:

``` bash
docker pull nvcr.io/nvidia/tensorrt-llm/release:1.3.0
```

Inspect the image:

``` bash
docker images | grep tensorrt-llm
```

Check that the command-line tools are available:

``` bash
docker run --rm \
  --gpus all \
  nvcr.io/nvidia/tensorrt-llm/release:1.3.0 \
  bash -lc 'which trtllm-serve && which trtllm-bench'
```

You can also verify the package and GPU from inside a temporary
container:

``` bash
docker run --rm -it \
  --gpus all \
  --ipc=host \
  nvcr.io/nvidia/tensorrt-llm/release:1.3.0 \
  bash
```

Inside the container:

``` bash
python - <<'PY'
import torch
import tensorrt_llm

print("TensorRT-LLM:", tensorrt_llm.__version__)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Compute capability:", torch.cuda.get_device_capability(0))
PY
```

Exit the temporary container:

``` bash
exit
```

### Environment notes and warnings encountered

-   `import tensorrt_llm` worked and reported version 1.3.0.
-   A direct `import tensorrt` attempt failed with `ModuleNotFoundError`
    in this environment. That did not prevent `trtllm-serve` or the
    TensorRT-LLM package from working. Do not blindly install a separate
    TensorRT pip package to fix this; first establish whether the
    specific workflow needs it.
-   A torchao warning indicated that optional C++ extensions were
    skipped because the container's PyTorch build was not compatible
    with the installed torchao extension. It did not block the serving
    exercise.
-   NVIDIA recommended shared-memory and memory-lock settings. We
    included these in the server command:
    -   `--ipc=host`
    -   `--ulimit memlock=-1`
    -   `--ulimit stack=67108864`

------------------------------------------------------------------------

## 5. Launch the TensorRT-LLM server

The model used was `Qwen/Qwen3-0.6B`. The server was launched with the
PyTorch backend and an OpenAI-compatible API.

Run this from WSL:

``` bash
docker run --rm -it \
  --name trtllm-server \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 8000:8000 \
  nvcr.io/nvidia/tensorrt-llm/release:1.3.0 \
  trtllm-serve serve Qwen/Qwen3-0.6B \
    --backend pytorch \
    --host 0.0.0.0 \
    --port 8000 \
    --max_batch_size 4 \
    --max_num_tokens 1024 \
    --max_seq_len 2048 \
    --free_gpu_memory_fraction 0.30 \
    --served_model_name qwen3-0.6b \
    --no-telemetry
```

Keep this terminal open. It is attached to the running server.

### What the options mean

  -----------------------------------------------------------------------
  Option                              Purpose
  ----------------------------------- -----------------------------------
  `--name trtllm-server`              Names the container so it can be
                                      stopped or inspected easily

  `--gpus all`                        Makes available NVIDIA GPUs visible
                                      inside the container

  `--ipc=host`                        Uses the host IPC namespace,
                                      avoiding the small default Docker
                                      shared-memory allocation

  `--ulimit memlock=-1`               Removes the process memory-lock
                                      limit

  `--ulimit stack=67108864`           Sets the stack limit to 64 MiB

  `-p 8000:8000`                      Maps container port 8000 to WSL
                                      port 8000

  `--backend pytorch`                 Uses TensorRT-LLM's PyTorch backend
                                      rather than a prebuilt optimized
                                      TensorRT engine

  `--host 0.0.0.0`                    Listens on all container interfaces
                                      so the mapped port can be reached

  `--port 8000`                       Serves the API on port 8000

  `--max_batch_size 4`                Configures the maximum batch size
                                      limit as 4

  `--max_num_tokens 1024`             Sets the configured token budget
                                      for the serving scheduler

  `--max_seq_len 2048`                Sets the maximum sequence length

  `--free_gpu_memory_fraction 0.30`   Configures the fraction of free GPU
                                      memory available for the relevant
                                      serving allocation

  `--served_model_name qwen3-0.6b`    Exposes this name through the API

  `--no-telemetry`                    Disables telemetry
  -----------------------------------------------------------------------

**Important:** `max_batch_size` is a configured upper limit. It does not
prove that the server always executes exactly that many requests
together. Actual scheduling and batch composition depend on runtime
state and resource availability.

### Check the server from a second WSL terminal

``` bash
curl -s http://localhost:8000/health
```

Check generation health:

``` bash
curl -s http://localhost:8000/health_generate
```

List available models:

``` bash
curl -s http://localhost:8000/v1/models
```

Inspect server configuration:

``` bash
curl -s http://localhost:8000/server_info
```

------------------------------------------------------------------------

## 6. Test the OpenAI-compatible chat API

From a second WSL terminal:

``` bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [
      {
        "role": "user",
        "content": "Explain KV cache in LLM inference."
      }
    ],
    "max_tokens": 150,
    "temperature": 0
  }'
```

The response follows the OpenAI chat-completions response structure,
including `choices` and `usage`.

Qwen3 may emit `<think>` reasoning-style content depending on the
model's behavior and prompt. The API returning such text is not an API
failure.

### Test from Python

Install the HTTP client libraries in the WSL Python environment if
needed:

``` bash
python3 -m pip install aiohttp
```

If your WSL environment uses an externally managed Python installation,
create and activate a virtual environment instead:

``` bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install aiohttp
```

------------------------------------------------------------------------

## 7. Single-request streaming benchmark

A streaming request lets the client observe output as it arrives rather
than waiting for the complete response.

Create:

``` bash
nano benchmark_trtllm.py
```

Use:

``` python
import json
import time
import urllib.request

URL = "http://localhost:8000/v1/chat/completions"

payload = {
    "model": "qwen3-0.6b",
    "messages": [
        {
            "role": "user",
            "content": "Explain KV cache in LLM inference."
        }
    ],
    "max_tokens": 256,
    "temperature": 0,
    "stream": True,
}

request = urllib.request.Request(
    URL,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

start = time.perf_counter()
first_output = None
last_output = None
chunks = 0

with urllib.request.urlopen(request, timeout=300) as response:
    for raw_line in response:
        line = raw_line.decode("utf-8").strip()

        if not line.startswith("data:"):
            continue

        data = line[5:].strip()

        if data == "[DONE]":
            break

        event = json.loads(data)
        choices = event.get("choices", [])

        if not choices:
            continue

        content = choices[0].get("delta", {}).get("content")

        if content:
            now = time.perf_counter()

            if first_output is None:
                first_output = now

            last_output = now
            chunks += 1

finish = time.perf_counter()

if first_output is not None:
    ttft = first_output - start
    total = finish - start
    post_first = last_output - first_output

    print(f"Total request time: {total:.3f}s")
    print(f"Time to first output: {ttft:.3f}s")
    print(f"Output chunks: {chunks}")
    print(f"Post-first-output time: {post_first:.3f}s")
    print(f"Approximate chunk interval: "
          f"{post_first / max(chunks - 1, 1):.5f}s")
else:
    print("No text output received.")
```

Run:

``` bash
python3 benchmark_trtllm.py
```

### Recorded result

The run completed with:

  Metric                         Measured value
  ------------------------- -------------------
  Total request time                    1.523 s
  Time to first output                  0.135 s
  Completion tokens                         256
  Overall generation rate     168.14 tokens/sec
  Post-first-output rate      183.83 tokens/sec

The overall rate was calculated as:

\[ `\text{Overall rate}`{=tex} =
`\frac{256\text{ tokens}}{1.523\text{ s}}`{=tex}
`\approx 168.14`{=tex}`\text{ tokens/sec}`{=tex} \]

The post-first-output rate was approximately 183.83 tokens/sec based on
the benchmark's token/timing measurement.

**Interpretation caveat:** TTFT here is the arrival of the first
text-bearing streaming chunk, not a direct measurement of the first GPU
token's execution time. Streaming chunks may contain more than one
token.

------------------------------------------------------------------------

## 8. Four concurrent requests

The next experiment submitted four chat-completion requests
concurrently. Each request had a different output limit so that we could
observe how requests of different lengths completed.

Create:

``` bash
nano concurrent_trtllm.py
```

Use:

``` python
import asyncio
import time
import aiohttp

URL = "http://localhost:8000/v1/chat/completions"

REQUESTS = [
    ("A", "Explain KV cache in LLM inference.", 256),
    ("B", "Explain continuous batching.", 32),
    ("C", "Explain PagedAttention.", 128),
    ("D", "Explain prefill versus decode.", 64),
]


async def send_request(session, name, prompt, max_tokens, start):
    payload = {
        "model": "qwen3-0.6b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }

    request_start = time.perf_counter()

    async with session.post(URL, json=payload) as response:
        result = await response.json()

        if response.status != 200:
            raise RuntimeError(
                f"{name}: HTTP {response.status}: {result}"
            )

    finish = time.perf_counter()

    usage = result.get("usage", {})
    choices = result.get("choices", [])

    print(
        f"{name}: finished at {finish-start:.3f}s | "
        f"request duration {finish-request_start:.3f}s | "
        f"output tokens {usage.get('completion_tokens', 'unknown')} | "
        f"finish reason "
        f"{choices[0].get('finish_reason') if choices else 'unknown'}"
    )

    return usage.get("completion_tokens", 0), finish


async def main():
    start = time.perf_counter()
    timeout = aiohttp.ClientTimeout(total=300)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = await asyncio.gather(
            *[
                send_request(session, name, prompt, limit, start)
                for name, prompt, limit in REQUESTS
            ]
        )

    total_time = time.perf_counter() - start
    total_tokens = sum(tokens for tokens, _ in results)

    print("\n========== CONCURRENT RESULTS ==========")
    print(f"Requests: {len(results)}")
    print(f"Total completion tokens: {total_tokens}")
    print(f"Wall-clock time: {total_time:.3f}s")
    print(
        f"Aggregate throughput: "
        f"{total_tokens/total_time:.2f} tokens/sec"
    )
    print("========================================")


if __name__ == "__main__":
    asyncio.run(main())
```

Run:

``` bash
python3 concurrent_trtllm.py
```

Run three repetitions to reduce the influence of one unusually fast or
slow run:

``` bash
for i in 1 2 3; do
    echo "========== RUN $i =========="
    python3 concurrent_trtllm.py
done
```

### What the code is doing

`asyncio.gather()` starts all four coroutines concurrently and waits for
all of them to finish.

The request duration is measured independently for each request:

\[ D_i = t\_{`\text{finish}`{=tex},i} -
t\_{`\text{request-start}`{=tex},i} \]

The total workload time is:

\[ T\_{`\text{wall}`{=tex}} = t\_{`\text{last completion}`{=tex}} -
t\_{`\text{benchmark start}`{=tex}} \]

Aggregate throughput is:

\[ `\text{Throughput}`{=tex} =
`\frac{\text{Total completion tokens}}{T_{\text{wall}}}`{=tex} \]

It is **not** calculated by adding the four request durations because
the requests overlap in time.

------------------------------------------------------------------------

## 9. Eight concurrent requests

To increase concurrency, create a second script:

``` bash
nano concurrent_trtllm_8.py
```

Use:

``` python
import asyncio
import time
import aiohttp
import statistics

URL = "http://localhost:8000/v1/chat/completions"

REQUESTS = [
    ("A", "Explain KV cache in LLM inference.", 256),
    ("B", "Explain continuous batching.", 32),
    ("C", "Explain PagedAttention.", 128),
    ("D", "Explain prefill versus decode.", 64),
    ("E", "Explain how a transformer generates the next token.", 256),
    ("F", "Explain the purpose of a tokenizer.", 32),
    ("G", "Explain GPU memory management during LLM inference.", 128),
    ("H", "Explain why inference servers use batching.", 64),
]


async def send_request(session, name, prompt, max_tokens, start):
    payload = {
        "model": "qwen3-0.6b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }

    request_start = time.perf_counter()

    async with session.post(URL, json=payload) as response:
        result = await response.json()

        if response.status != 200:
            raise RuntimeError(
                f"{name}: HTTP {response.status}: {result}"
            )

    finish = time.perf_counter()

    usage = result.get("usage", {})
    choices = result.get("choices", [])

    tokens = usage.get("completion_tokens", 0)
    duration = finish - request_start

    print(
        f"{name}: finished at {finish-start:.3f}s | "
        f"request duration {duration:.3f}s | "
        f"output tokens {tokens} | "
        f"finish reason "
        f"{choices[0].get('finish_reason') if choices else 'unknown'}"
    )

    return {
        "name": name,
        "tokens": tokens,
        "duration": duration,
        "finish": finish,
    }


async def main():
    start = time.perf_counter()
    timeout = aiohttp.ClientTimeout(total=300)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = await asyncio.gather(
            *[
                send_request(session, name, prompt, limit, start)
                for name, prompt, limit in REQUESTS
            ]
        )

    total_time = time.perf_counter() - start
    total_tokens = sum(r["tokens"] for r in results)
    durations = [r["duration"] for r in results]

    print("\n========== CONCURRENT RESULTS ==========")
    print(f"Requests: {len(results)}")
    print(f"Total completion tokens: {total_tokens}")
    print(f"Wall-clock time: {total_time:.3f}s")
    print(
        f"Aggregate throughput: "
        f"{total_tokens / total_time:.2f} tokens/sec"
    )
    print(
        f"Average request duration: "
        f"{statistics.mean(durations):.3f}s"
    )
    print(
        f"Median request duration: "
        f"{statistics.median(durations):.3f}s"
    )
    print(
        f"Longest request duration: "
        f"{max(durations):.3f}s"
    )
    print("========================================")


if __name__ == "__main__":
    asyncio.run(main())
```

Run it three times:

``` bash
for i in 1 2 3; do
    echo "========== RUN $i =========="
    python3 concurrent_trtllm_8.py
done
```

### Recorded eight-request results

The eight requests generated 960 completion tokens in each run.

  Run                        Wall time   Aggregate throughput
  ------------------------ ----------- ----------------------
  1                            2.063 s           465.26 tok/s
  2                            1.733 s           553.80 tok/s
  3                            1.728 s           555.71 tok/s
  **Average throughput**                     **524.92 tok/s**

Other observed values:

  Metric                         Run 2     Run 3
  -------------------------- --------- ---------
  Average request duration     0.779 s   0.795 s
  Median request duration      0.630 s   0.627 s
  Longest request duration     1.733 s   1.727 s

The first run was slower, so it is useful to report both the mean and
the individual runs.

------------------------------------------------------------------------

## 10. Batch-size experiment: 2 versus 4

We restarted the server with different `--max_batch_size` values while
keeping the other server parameters unchanged.

### Configuration A: maximum batch size 2

``` bash
--max_batch_size 2
```

Four concurrent requests were submitted, generating 480 completion
tokens per run.

  Run               Wall time   Aggregate throughput
  ------------- ------------- ----------------------
  1                   1.756 s           273.42 tok/s
  2                   1.364 s           351.82 tok/s
  3                   1.489 s           322.45 tok/s
  **Average**     **1.536 s**       **315.90 tok/s**

### Configuration B: maximum batch size 4

``` bash
--max_batch_size 4
```

The same four-request benchmark was run.

  Run               Wall time   Aggregate throughput
  ------------- ------------- ----------------------
  1                   1.528 s           314.17 tok/s
  2                   1.104 s           434.66 tok/s
  3                   1.102 s           435.52 tok/s
  **Average**     **1.245 s**       **394.78 tok/s**

Average throughput change:

\[ `\frac{394.78 - 315.90}{315.90}`{=tex}`\times100`{=tex}
`\approx 25.0`{=tex}% \]

For this workload, the batch-size-4 configuration delivered
approximately 25% higher average aggregate throughput than batch size 2.

This is an observed result for this model, GPU, request mix, and
benchmark. It is not a universal TensorRT-LLM speedup.

### Eight requests with maximum batch size 4

The same eight-request workload produced:

  Metric                                    Result
  --------------------------------- --------------
  Average aggregate throughput        524.92 tok/s
  Average wall time                        1.841 s
  Total completion tokens per run              960

Compared with the four-request workload at batch size 4, aggregate
throughput increased by:

\[ `\frac{524.92 - 394.78}{394.78}`{=tex}`\times100`{=tex}
`\approx 33.0`{=tex}% \]

The eight-request workload generated twice as many tokens, but total
wall time increased by only about 48%.

### What this demonstrates

-   **Concurrency** is the number of requests submitted or active at the
    same time.
-   **Maximum batch size** is a configured upper limit on batch size.
-   **Throughput** is the aggregate number of generated tokens per
    second.
-   Increasing concurrency can increase aggregate throughput while
    individual requests take longer.
-   A configured maximum batch size does not prove actual batch
    occupancy. We did not directly inspect the scheduler's per-iteration
    batch composition in this experiment.

A batch-size-8 server configuration was proposed as a follow-up
experiment, but no results from that configuration were recorded in this
lesson.

------------------------------------------------------------------------

## 11. Observability and health endpoints

The server exposed these useful endpoints:

  -----------------------------------------------------------------------
  Endpoint                            Purpose
  ----------------------------------- -----------------------------------
  `/health`                           Basic server health

  `/health_generate`                  Checks generation health

  `/server_info`                      Reports server configuration and
                                      startup information

  `/metrics`                          Metrics endpoint

  `/kv_cache_events`                  KV-cache event endpoint

  `/energy_metrics`                   Energy monitoring endpoint
  -----------------------------------------------------------------------

Inspect them:

``` bash
curl -i http://localhost:8000/health
curl -i http://localhost:8000/health_generate
curl -i http://localhost:8000/server_info
curl -i http://localhost:8000/metrics
```

### Recorded observations

-   `/health` returned HTTP 200.
-   `/health_generate` returned HTTP 200 and
    `Generation health check OK`.
-   `/server_info` returned:
    -   `max_batch_size: 2` during the earlier batch-size-2 run.
    -   `tokens_per_block: 32`.
    -   `kv_cache_hash_algo: v1_block_key`.
    -   A Hugging Face checkpoint loader and weight loader.
    -   `total_model_loading_seconds: 0.61796`.
    -   `weight_population_seconds: 0.29922`.
-   `/metrics` returned an empty JSON array (`[]`) in this
    configuration, not Prometheus text metrics.
-   A GET request to `/kv_cache_events` returned HTTP 405 and
    `Allow: POST`. This tells us that GET is not supported; it does not
    mean the endpoint is broken.
-   `/energy_metrics` returned HTTP 503 with
    `Energy monitoring is not available`.

The server startup information showed approximately 0.618 seconds for
the reported model-loading total, with approximately 0.299 seconds
attributed to weight population. The individual stage numbers should not
be assumed to add up to the reported total.

### KV cache block interpretation

The server reported:

``` json
{
  "tokens_per_block": 32
}
```

With 32 tokens per block:

\[ `\text{Blocks required}`{=tex} =
`\left`{=tex}`\lceil`{=tex}`\frac{\text{sequence tokens}}{32}`{=tex}`\right`{=tex}`\rceil`{=tex}
\]

Examples:

    Sequence length   Blocks required
  ----------------- -----------------
          32 tokens                 1
          40 tokens                 2
          64 tokens                 2
          70 tokens                 3
         128 tokens                 4

The `kv_cache_hash_algo` field identifies a hash algorithm used by the
implementation. That field alone does not prove that prefix caching was
enabled or that a particular request reused cached blocks.

------------------------------------------------------------------------

## 12. Streaming benchmark: concurrent requests

We also created a streaming version of the eight-request benchmark.

The important difference is:

``` python
"stream": True
```

Instead of waiting for the complete response body, the client consumes
server-sent events as they arrive.

### Streaming concepts

**Request latency**

Time from sending a request until the complete response has arrived.

**TTFT --- Time to First Token**

Time from request submission until the first generated output arrives at
the client.

**ITL --- Inter-Token Latency**

Time between successive generated tokens. A client that receives
individual token timestamps can measure this directly.

**TPOT --- Time Per Output Token**

An average time per generated token after the first token. A common
approximation is:

\[ TPOT `\approx`{=tex}
`\frac{t_{\text{last token}}-t_{\text{first token}}}`{=tex}
{N\_{`\text{output tokens}`{=tex}}-1} \]

**Aggregate throughput**

Total output tokens across all requests divided by the total wall-clock
time for the workload.

### Streaming benchmark script used

Create:

``` bash
nano streaming_concurrent_trtllm.py
```

The initial script used for the recorded run was:

``` python
import asyncio
import time
import aiohttp

URL = "http://localhost:8000/v1/chat/completions"

REQUESTS = [
    ("A", "Explain KV cache in LLM inference.", 256),
    ("B", "Explain continuous batching.", 32),
    ("C", "Explain PagedAttention.", 128),
    ("D", "Explain prefill versus decode.", 64),
    ("E", "Explain transformer next-token generation.", 256),
    ("F", "Explain tokenization.", 32),
    ("G", "Explain GPU memory management in inference.", 128),
    ("H", "Explain inference batching.", 64),
]


async def send_request(session, name, prompt, max_tokens, start):
    payload = {
        "model": "qwen3-0.6b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }

    request_start = time.perf_counter()
    first_output_time = None
    last_output_time = None
    output_chunks = 0

    async with session.post(URL, json=payload) as response:
        if response.status != 200:
            raise RuntimeError(
                f"{name}: HTTP {response.status}: "
                f"{await response.text()}"
            )

        async for line in response.content:
            line = line.decode("utf-8").strip()

            if not line.startswith("data:"):
                continue

            data = line[5:].strip()

            if data == "[DONE]":
                break

            import json
            event = json.loads(data)

            choices = event.get("choices", [])
            if not choices:
                continue

            delta = choices[0].get("delta", {})
            content = delta.get("content")

            if content:
                now = time.perf_counter()

                if first_output_time is None:
                    first_output_time = now

                last_output_time = now
                output_chunks += 1

    finish = time.perf_counter()

    if first_output_time is None:
        ttft = None
        tpot = None
    else:
        ttft = first_output_time - request_start

        if output_chunks > 1:
            tpot = (
                last_output_time - first_output_time
            ) / (output_chunks - 1)
        else:
            tpot = None

    if ttft is not None:
        print(
            f"{name}: TTFT={ttft:.3f}s | "
            f"chunks={output_chunks} | "
            f"TPOT={tpot:.4f}s | "
            f"total={finish-request_start:.3f}s"
        )
    else:
        print(f"{name}: no text output received")

    return {
        "name": name,
        "ttft": ttft,
        "tpot": tpot,
        "chunks": output_chunks,
        "duration": finish - request_start,
        "finish": finish,
    }


async def main():
    start = time.perf_counter()
    timeout = aiohttp.ClientTimeout(total=300)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = await asyncio.gather(
            *[
                send_request(session, name, prompt, limit, start)
                for name, prompt, limit in REQUESTS
            ]
        )

    total_time = time.perf_counter() - start

    valid_ttft = [
        r["ttft"] for r in results if r["ttft"] is not None
    ]
    valid_tpot = [
        r["tpot"] for r in results if r["tpot"] is not None
    ]

    print("\n========== STREAMING RESULTS ==========")
    print(f"Requests: {len(results)}")
    print(f"Total wall time: {total_time:.3f}s")

    if valid_ttft:
        print(f"Average TTFT: {statistics.mean(valid_ttft):.4f}s")
        print(f"Median TTFT: {statistics.median(valid_ttft):.4f}s")

    if valid_tpot:
        print(f"Average TPOT: {statistics.mean(valid_tpot):.4f}s")
        print(f"Median TPOT: {statistics.median(valid_tpot):.4f}s")

    print("=======================================")


if __name__ == "__main__":
    asyncio.run(main())
```

Run:

``` bash
python3 streaming_concurrent_trtllm.py
```

### Recorded streaming result

One eight-request streaming run produced:

  -----------------------------------------------------------------------
  Request               TTFT  Output chunks        Approx.     Total time
                                               inter-chunk 
                                                  interval 
  ----------- -------------- -------------- -------------- --------------
  A                  1.056 s            256         4.3 ms        2.159 s

  B                  0.224 s             32         9.5 ms        0.520 s

  C                  0.123 s            128         7.2 ms        1.042 s

  D                  1.058 s             64         4.7 ms        1.355 s

  E                  0.224 s            256         5.5 ms        1.638 s

  F                  0.551 s             32         5.4 ms        0.717 s

  G                  0.224 s            128         6.5 ms        1.048 s

  H                  0.749 s             64         4.9 ms        1.058 s
  -----------------------------------------------------------------------

Summary:

  Metric                                         Result
  ------------------------------------------ ----------
  Total wall time                               2.160 s
  Average TTFT                                 0.5261 s
  Median TTFT                                  0.3874 s
  Average approximate inter-chunk interval     0.0060 s
  Median approximate inter-chunk interval      0.0055 s

### Important limitation of this initial streaming script

The script uses `output_chunks` as if it were the number of generated
tokens:

``` python
tpot = (
    last_output_time - first_output_time
) / (output_chunks - 1)
```

That is only an approximation. One SSE event/chunk can contain multiple
tokens, and the server may also send chunks with no text content.

Therefore, the recorded 6.0 ms average is an **inter-chunk interval**,
not a verified per-token TPOT.

A proposed improvement was to request streaming usage metadata with:

``` python
"stream_options": {"include_usage": True}
```

and calculate the average using the server-reported completion-token
count. That is a better estimate of TPOT, although it still does not
provide a timestamp for every individual token.

The improved version was not yet run, so no improved TPOT result is
claimed here.

------------------------------------------------------------------------

## 13. How to calculate the benchmark metrics

Use the eight-request Run 2 as an example.

### Total completion tokens

\[ 256+32+128+64+256+32+128+64=960 \]

### Wall-clock time

The final request completed at approximately 1.733 seconds:

\[ T\_{`\text{wall}`{=tex}}=1.733s \]

### Aggregate throughput

\[ `\text{Throughput}`{=tex} = `\frac{960}{1.733}`{=tex}
`\approx 553.80`{=tex}`\text{ tokens/sec}`{=tex} \]

### Request duration

For each request:

\[ D_i = t\_{`\text{finish}`{=tex},i} -
t\_{`\text{request-start}`{=tex},i} \]

This is the complete HTTP request duration, including queueing, server
processing, generation, and response transfer.

### Average request duration

\[ D\_{`\text{avg}`{=tex}} = `\frac{D_1+D_2+\cdots+D_8}{8}`{=tex} \]

For Run 2, this was approximately 0.779 seconds.

### Median request duration

Sort all eight durations. Since there are an even number of
observations, average the fourth and fifth values.

For Run 2, the median was approximately 0.630 seconds.

### Why average request duration is not wall-clock time

Concurrent requests overlap. The sum of their individual durations can
be much greater than the wall-clock time.

### Throughput per request

A rough equal-share calculation is:

\[ `\frac{\text{Aggregate throughput}}{\text{Request count}}`{=tex} \]

For the eight-request average:

\[ `\frac{524.92}{8}`{=tex}=65.62 \]

This is an average share of aggregate throughput, not a measured
generation rate for each request.

------------------------------------------------------------------------

## 14. Stopping and restarting the server

The server terminal can be stopped with `Ctrl+C`. Alternatively, from
another WSL terminal:

``` bash
docker stop trtllm-server
```

Because the container was launched with `--rm`, Docker removes the
container after it stops. The model is not a persistent model file
created by this command; it is loaded by the serving process from the
Hugging Face checkpoint source.

To restart the server, repeat the `docker run` command from Section 5.

Check whether it is running:

``` bash
docker ps
```

Check GPU usage:

``` bash
nvidia-smi
```

------------------------------------------------------------------------

## 15. Lessons learned

1.  **A serving backend and an optimized engine are not the same
    thing.** This exercise used TensorRT-LLM's PyTorch backend. We did
    not build a serialized TensorRT engine.
2.  **Concurrency and batch size are different.** Eight clients can
    submit requests concurrently while the configured maximum batch size
    is four.
3.  **Aggregate throughput and user latency measure different things.**
    More concurrency can increase total tokens/sec while individual
    requests wait longer.
4.  **Benchmark multiple repetitions.** The first run can differ from
    later runs due to warm-up, cache state, GPU clocks, and other
    runtime variation.
5.  **Measure the right streaming unit.** SSE chunks are not guaranteed
    to be one token each. Do not label inter-chunk time as exact TPOT.
6.  **Health endpoints are not full observability.** A healthy
    generation endpoint does not imply that every optional metrics or
    energy-monitoring endpoint is available.
7.  **Do not infer scheduler behavior from completion order alone.**
    Direct batch-occupancy measurements or runtime tracing are needed to
    establish exactly how requests were grouped at each iteration.

------------------------------------------------------------------------

## 16. TensorRT-LLM work completed and remaining

### Completed in this lesson

-   Pulled and ran NVIDIA's TensorRT-LLM 1.3.0 Docker image in WSL2.
-   Verified GPU and package availability inside the container.
-   Served `Qwen/Qwen3-0.6B` using the PyTorch backend.
-   Tested the OpenAI-compatible API.
-   Inspected health and server information endpoints.
-   Measured a single streaming request.
-   Benchmarked four and eight concurrent requests.
-   Compared maximum batch sizes 2 and 4 with four concurrent requests.
-   Measured first-output timing and approximate streaming chunk
    intervals.

### Still to do in a future lesson

-   Prepare/convert a checkpoint for the optimized engine workflow.
-   Build a TensorRT-LLM engine with an appropriate precision and
    configuration.
-   Load and serve the optimized engine.
-   Compare PyTorch backend and optimized engine under the same
    benchmark conditions.
-   Measure model load time, GPU memory, TTFT, TPOT, request latency,
    and aggregate throughput.
-   Investigate deeper scheduler/KV-cache telemetry where supported.

