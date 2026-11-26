# Module 8 — Hands-on vLLM: Concurrent Requests and KV-Cache Monitoring

## 1. Objective

In this practical exercise, we will run a real LLM inference server using vLLM and observe how it manages concurrent requests and GPU KV-cache memory.

We will learn:

1. How to start a vLLM inference server.
2. How to send multiple concurrent requests using Python.
3. How vLLM limits the number of active sequences.
4. How requests wait in the scheduler queue.
5. How GPU KV-cache blocks are dynamically allocated and released.
6. How to monitor vLLM through its Prometheus metrics endpoint.

This exercise connects the concepts from Module 7 (continuous batching, scheduling, paged KV cache, block allocation) with a real inference engine.

---

## 2. Environment and architecture

The exercise uses the following environment:

| Component          | Configuration                  |
| ------------------ | ------------------------------ |
| Operating system   | Windows 11                     |
| Linux environment  | WSL2                           |
| Linux distribution | Ubuntu 24.04                   |
| Python             | 3.12.3                         |
| Package manager    | uv                             |
| Inference engine   | vLLM 0.34.0                    |
| PyTorch            | 2.14.0+cu132                   |
| GPU                | NVIDIA RTX 3500 Ada Laptop GPU |
| Model              | Qwen/Qwen3-0.6B                |
| GPU memory         | Approximately 12 GB            |

### How the components communicate

```
Windows 11
   |
   +-- WSL2
         |
         +-- Ubuntu 24.04
               |
               +-- Python virtual environment
               |
               +-- Terminal 1: vLLM server
               |       |
               |       +-- Qwen3-0.6B
               |       +-- GPU execution
               |       +-- KV-cache manager
               |       +-- Scheduler
               |
               +-- Terminal 2: Concurrent request generator
               |
               +-- Terminal 3: KV-cache metrics monitor
```

All three terminals are Ubuntu terminals running inside WSL2. Windows is the host operating system; the Python scripts and vLLM server run inside Ubuntu.

---

## 3. Open Ubuntu on Windows

You can open Ubuntu in either of these ways:

**Option A — Windows Start menu**

1. Press the Windows key.
2. Search for `Ubuntu`.
3. Open Ubuntu.

**Option B — Windows Terminal**

1. Open Windows Terminal.
2. Click the dropdown arrow beside the plus sign.
3. Select Ubuntu.

You can also launch Ubuntu from PowerShell:

```
wsl -d Ubuntu
```

You should now see a Linux prompt similar to:

```
username@computer:~$
```

Do not enter Python statements directly at this prompt. It is a Bash shell.

---

## 4. Activate the existing Python environment

We previously created the Module 8 workspace and virtual environment.

In Ubuntu, run:

```
cd ~/llm-inference/module8
source .venv/bin/activate
```

Your prompt should show `(.venv)`.

Verify the environment:

```
python --version
vllm --version
```

Expected versions:

```
Python 3.12.3
vLLM 0.34.0
```

Install the dependencies used by our two scripts:

```
uv pip install httpx requests
```

We use:

- `httpx` for asynchronous concurrent HTTP requests.
- `requests` for periodically collecting server metrics.

---

## 5. Terminal 1 — Start the vLLM server

Open your first Ubuntu terminal.

Activate the environment:

```
cd ~/llm-inference/module8
source .venv/bin/activate
```

Start the server:

```
vllm serve Qwen/Qwen3-0.6B \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.85 \
    --host 0.0.0.0 \
    --port 8000
```

### Understanding the configuration

| Parameter                       | Explanation                               |
| ------------------------------- | ----------------------------------------- |
| `Qwen/Qwen3-0.6B`               | Model loaded by vLLM                      |
| `--dtype bfloat16`              | Uses BF16 model weights                   |
| `--max-model-len 4096`          | Maximum supported sequence length         |
| `--max-num-seqs 4`              | Limits active sequences to four           |
| `--gpu-memory-utilization 0.85` | GPU memory utilization target for vLLM    |
| `--host 0.0.0.0`                | Makes the server listen on all interfaces |
| `--port 8000`                   | HTTP server port                          |

Wait until you see:

```
INFO: Application startup complete.
```

This indicates that the API server has started successfully.

**Important:** Leave Terminal 1 running. Do not close it while performing the exercise.

### Verify the server

Open a second Ubuntu terminal and execute:

```
curl http://localhost:8000/v1/models
```

You should receive JSON containing the loaded model.

---

## 6. Understand the actual KV-cache configuration

During server startup, vLLM profiles GPU memory and determines how much memory is available for KV cache after accounting for model weights, activations, and runtime requirements.

Your server reported the following configuration through its metrics:

| Parameter                   | Actual value  |
| --------------------------- | ------------- |
| GPU KV-cache blocks         | 1,192         |
| Block size                  | 64 tokens     |
| Total KV-cache capacity     | 76,288 tokens |
| Maximum model length        | 4,096 tokens  |
| Configured active sequences | 4             |
| Prefix caching              | Enabled       |

The total KV-cache capacity is:

```
1,192 blocks × 64 tokens
= 76,288 token slots
```

The block size is 64, rather than the block size of 16 used in our Module 7 simulation.

### Example: Allocating blocks for a request

Suppose a request has 150 tokens.

```
Required blocks = ceil(150 / 64)
                = 3 blocks
```

The allocation looks like:

```
Block 1       Block 2       Block 3
64 tokens     64 tokens     22 tokens
```

Three blocks reserve 192 token slots, although only 150 are occupied.

The remaining 42 slots represent internal fragmentation in the final block.

---

## 7. Terminal 2 — Create the concurrent request generator

This script sends eight requests concurrently to the vLLM server.

However, our server is configured with:

```
--max-num-seqs 4
```

Therefore, we expect the scheduler to execute up to four active sequences while the remaining requests wait.

### Create the Python file

In your second Ubuntu terminal:

```
cd ~/llm-inference/module8
source .venv/bin/activate
```

Open the nano editor:

```
nano test_concurrent_requests.py
```

Paste the following complete code:

```
import asyncio
import httpx

URL = "http://localhost:8000/v1/chat/completions"


async def send_request(request_id):
    payload = {
        "model": "Qwen/Qwen3-0.6B",
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Request {request_id}: Explain how "
                    "PagedAttention works in detail. "
                    "Include examples and technical explanations."
                )
            }
        ],
        "temperature": 0,
        "max_tokens": 3000,
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    }

    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(URL, json=payload)
        response.raise_for_status()
        result = response.json()

        print(
            f"Request {request_id}: "
            f"prompt={result['usage']['prompt_tokens']}, "
            f"completion={result['usage']['completion_tokens']}"
        )


async def main():
    await asyncio.gather(
        *(send_request(i) for i in range(1, 9))
    )


asyncio.run(main())
```

Save the file:

1. Press `Ctrl + O`.
2. Press `Enter`.
3. Press `Ctrl + X`.

### Run the script

```
python test_concurrent_requests.py
```

The script submits eight concurrent requests.

You should eventually see output similar to:

```
Request 1: prompt=..., completion=...
Request 2: prompt=..., completion=...
Request 3: prompt=..., completion=...
...
Request 8: prompt=..., completion=...
```

The actual token counts and completion order may differ.

### Understanding the Python code

**`async def send_request()`**

Defines an asynchronous function that sends one request to the server.

**`httpx.AsyncClient`**

Provides asynchronous HTTP communication, allowing multiple requests to be in progress without waiting for each request sequentially.

**`asyncio.gather()`**

Schedules all eight request coroutines concurrently.

This does not mean eight requests execute on the GPU simultaneously. vLLM's scheduler controls which requests are admitted into execution batches.

**`max_tokens: 3000`**

Sets an upper limit of 3,000 generated tokens. It does not force the model to generate 3,000 tokens.

---

## 8. Terminal 3 — Create the KV-cache monitoring script

We need to observe the metrics while the requests are executing.

Manual inspection is unreliable because requests can finish very quickly.

Our monitoring script polls the vLLM metrics endpoint every 100 milliseconds.

### Create the file

Open a third Ubuntu terminal:

```
cd ~/llm-inference/module8
source .venv/bin/activate
```

Create the script:

```
nano monitor_vllm.py
```

Paste this complete code:

```
import time
import requests
from datetime import datetime

URL = "http://localhost:8000/metrics"

print("Monitoring vLLM. Press Ctrl+C to stop.\n")

while True:
    response = requests.get(URL, timeout=5)
    response.raise_for_status()

    lines = response.text.splitlines()

    values = {}

    for line in lines:
        if line.startswith("vllm:num_requests_running{"):
            values["running"] = float(line.split()[-1])

        elif line.startswith("vllm:num_requests_waiting{"):
            values["waiting"] = float(line.split()[-1])

        elif line.startswith("vllm:kv_cache_usage_perc{"):
            values["kv_usage"] = float(line.split()[-1]) * 100

    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

    print(
        f"{timestamp} | "
        f"Running: {values.get('running', 0):.0f} | "
        f"Waiting: {values.get('waiting', 0):.0f} | "
        f"KV Usage: {values.get('kv_usage', 0):.2f}%"
    )

    time.sleep(0.1)
```

Save the file:

1. Press `Ctrl + O`.
2. Press `Enter`.
3. Press `Ctrl + X`.

### Run the monitoring script

```
python monitor_vllm.py
```

It should continuously print metrics similar to:

```
12:53:28.623 | Running: 4 | Waiting: 4 | KV Usage: 3.69%
12:53:28.731 | Running: 4 | Waiting: 4 | KV Usage: 3.95%
12:53:28.838 | Running: 4 | Waiting: 4 | KV Usage: 4.03%
12:53:29.052 | Running: 4 | Waiting: 4 | KV Usage: 4.28%
12:53:29.373 | Running: 4 | Waiting: 3 | KV Usage: 3.27%
12:53:29.802 | Running: 4 | Waiting: 2 | KV Usage: 2.35%
12:53:29.908 | Running: 4 | Waiting: 1 | KV Usage: 0.92%
```

These are illustrative examples, not additional measurements.

---

## 9. How to perform the experiment

Start the terminals in this order.

| Terminal   | Action                                |
| ---------- | ------------------------------------- |
| Terminal 1 | Keep the vLLM server running          |
| Terminal 3 | Start `monitor_vllm.py`               |
| Terminal 2 | Execute `test_concurrent_requests.py` |

In Terminal 2:

```
python test_concurrent_requests.py
```

Watch the output in Terminal 3.

The monitor should show the running request count, waiting request count, and KV-cache utilization changing during inference.

When all requests complete, the values should return to zero.

To stop the monitor, press `Ctrl + C`.

---

## 10. Understanding the actual observations

During our experiment, we observed:

```
12:53:28.623 | Running: 4 | Waiting: 4 | KV Usage: 3.69%
12:53:28.731 | Running: 4 | Waiting: 4 | KV Usage: 3.95%
12:53:28.838 | Running: 4 | Waiting: 4 | KV Usage: 4.03%
12:53:29.052 | Running: 4 | Waiting: 4 | KV Usage: 4.28%
12:53:29.267 | Running: 4 | Waiting: 4 | KV Usage: 4.45%
12:53:29.373 | Running: 4 | Waiting: 3 | KV Usage: 3.27%
12:53:29.802 | Running: 4 | Waiting: 2 | KV Usage: 2.35%
12:53:29.908 | Running: 4 | Waiting: 1 | KV Usage: 0.92%
```

### Observation 1: Four running and four waiting

Initially:

```
Running: 4
Waiting: 4
```

Eight requests were submitted, but only four were active.

This demonstrates the scheduler's active sequence limit.

The other four requests were waiting for scheduling capacity.

### Observation 2: Waiting requests decrease

Later:

```
Running: 4
Waiting: 3
```

As a request completed, the scheduler admitted another waiting request.

The active sequence count remained four.

This is an example of continuous batching: requests can enter and leave execution without waiting for an entire fixed batch to finish.

### Observation 3: KV-cache utilization changes

The cache utilization increased from 3.69% to 4.45%, then decreased.

Why?

- As requests process prompts and generate tokens, they require KV-cache blocks.
- When requests complete, their blocks can be released.
- Newly admitted requests allocate their own blocks.

The cache manager dynamically manages this allocation.

### Observation 4: Running count and cache utilization are different metrics

A running request does not necessarily consume the same amount of KV cache as another running request.

For example:

- Request A may have a 100-token sequence.
- Request B may have a 500-token sequence.
- Both may count as one running sequence.
- But their KV-cache requirements differ.

This is why the number of running requests alone cannot tell us how much KV-cache memory is occupied.

---

## 11. Calculate occupied blocks from the observed metrics

Your configuration:

```
Total GPU blocks = 1,192
Block size       = 64 tokens
```

At 4.45% utilization:

```
Occupied blocks = 1,192 × 0.0445
                ≈ 53 blocks
```

Allocated token slots:

```
53 × 64 = 3,392 token slots
```

Remember that these are approximate values because the utilization metric is a percentage and the displayed value is rounded.

Allocated token slots are not necessarily all occupied by actual tokens. The final block of a sequence can be partially filled.

---

## 12. Important metrics reference

You can inspect the complete metrics endpoint:

```
curl -s http://localhost:8000/metrics
```

Or filter the relevant metrics:

```
curl -s http://localhost:8000/metrics | grep -iE \
'kv_cache|requests_running|requests_waiting'
```

| Metric                                | Meaning                                 |
| ------------------------------------- | --------------------------------------- |
| `vllm:num_requests_running`           | Requests in model execution batches     |
| `vllm:num_requests_waiting`           | Requests waiting for processing         |
| `vllm:num_requests_waiting_by_reason` | Waiting requests categorized by reason  |
| `vllm:kv_cache_usage_perc`            | Fraction of KV cache currently occupied |
| `vllm:cache_config_info`              | KV-cache configuration information      |

The metrics endpoint exposes current values. It does not automatically retain a historical timeline. Our Python monitoring script samples the values to create a live textual record.

---

## 13. Troubleshooting

### Problem: `Command 'import' not found`

**Cause:** Python code was pasted directly into Bash.

**Solution:**

Save the code in a `.py` file and execute it:

```
python test_concurrent_requests.py
```

### Problem: `ModuleNotFoundError: No module named 'httpx'`

Install the dependency inside the active virtual environment:

```
uv pip install httpx
```

For the monitoring script:

```
uv pip install requests
```

### Problem: Connection refused

Verify that the vLLM server is running in Terminal 1.

Test:

```
curl http://localhost:8000/v1/models
```

### Problem: All metrics are zero

You may have sampled the metrics after the requests completed.

Start the monitoring script first, then execute the concurrent request script.

### Problem: Requests finish too quickly

Increase the request workload or number of concurrent requests. Remember that `max_tokens` is only an upper bound and does not guarantee a long response.

---

## 14. Important concepts learned

| Module 7 concept       | Real vLLM behavior observed                                   |
| ---------------------- | ------------------------------------------------------------- |
| Continuous batching    | Requests enter and leave execution dynamically                |
| Scheduler              | Limits active sequences and manages waiting requests          |
| Admission control      | Waiting requests are admitted when capacity becomes available |
| Paged KV cache         | Cache is allocated in fixed-size blocks                       |
| Dynamic allocation     | KV usage changes as sequences grow                            |
| Block release          | Completed requests release their KV-cache blocks              |
| Internal fragmentation | Final allocated block can have unused token slots             |
| Runtime metrics        | Exposes current scheduler and cache state                     |

### Key distinction

The scheduler and KV-cache manager are separate logical responsibilities:

- The scheduler decides which requests can execute.
- The KV-cache manager manages physical blocks holding the attention state.
- The model executor performs the actual GPU computation.

They coordinate with each other, but their metrics measure different things.

---

## 15. Exercises

Try answering these without looking at the earlier explanations.

1. Why were only four requests running when eight were submitted?
2. Why did the waiting count decrease while the running count remained four?
3. Why can KV-cache utilization decrease even when four requests are still running?
4. If a sequence has 150 tokens and the block size is 64, how many blocks does it need?
5. What is the difference between 76,288 available token slots and the number of tokens currently stored?
6. If `max-num-seqs` is increased from 4 to 8, does that guarantee eight full-length 4,096-token requests can run simultaneously?
7. Why is the KV-cache utilization metric more useful than the running request count when estimating cache pressure?

---

## 16. Cleanup and restarting

To stop the monitoring script:

```
Ctrl + C
```

To stop the request generator, use `Ctrl + C` if it is still running.

To stop the vLLM server:

```
Ctrl + C
```

To restart later, reopen Ubuntu and activate the environment:

```
cd ~/llm-inference/module8
source .venv/bin/activate
```

Then start the vLLM server again using the command in Section 5.

The Python scripts remain saved in:

```
~/llm-inference/module8/
    test_concurrent_requests.py
    monitor_vllm.py
```

---

## 17. Final takeaway

We have moved beyond simulating an inference scheduler and KV-cache allocator in Python.

We have now observed a real inference runtime:

- Eight requests submitted concurrently.
- Four active sequences allowed by the server configuration.
- Other requests waiting in the scheduler.
- KV-cache utilization changing dynamically.
- Cache capacity being released and reused as requests complete.

This provides the foundation for the next part of Module 8: understanding vLLM's internal architecture and comparing it with llama.cpp, which is already running in the local environment.

### **Your workspace after completing this tutorial**

`~/llm-inference/module8/`

`test_concurrent_requests.py`

Request generator

`monitor_vllm.py`

Metrics monitor

`.venv/`

Python environment

**One important note:** Your Windows machine is only hosting the WSL2 environment. You should perform all the commands in this tutorial in Ubuntu, not Windows PowerShell or CMD. You can edit the files using VS Code connected to WSL if you prefer instead of `nano`.