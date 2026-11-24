# Module 6 — GPU Architecture and Inference Performance

**Objective:** Understand how GPUs execute deep learning workloads, manage memory, and optimize LLM inference.

## 1. GPU Architecture

A GPU is designed to execute thousands of operations in parallel.

- **SM (Streaming Multiprocessor):** The primary execution unit of an NVIDIA GPU. It schedules and executes thread blocks.
- **CUDA Cores:** General-purpose arithmetic execution units.
- **Tensor Cores:** Specialized hardware for accelerated matrix multiplication and multiply-accumulate operations.
- **CPU vs GPU:** CPUs optimize for low-latency sequential execution; GPUs optimize for high-throughput parallel computation.

## 2. GPU Execution Model

CUDA organizes GPU execution into three levels:

- **Thread:** The smallest execution unit, responsible for individual computations.
- **Warp:** A group of 32 threads executing together on NVIDIA GPUs.
- **Thread Block:** A group of threads scheduled on a single SM.
- **Grid:** A collection of thread blocks launched by a kernel.

Global thread indexing:

\[
global\_id = block\_id \times block\_size + thread\_id
\]

## 3. GPU Memory Hierarchy

GPU memory has different levels with different capacities and access speeds.

| Memory | Scope | Characteristics |
|---|---|---|
| Registers | Individual thread | Very fast, limited capacity |
| Shared memory | Thread block | Fast, explicitly managed |
| L1 Cache | SM | On-chip cache |
| L2 Cache | Entire GPU | Shared cache |
| Global memory (VRAM) | Entire GPU | Large capacity, higher latency |

In LLM inference:
- Model weights reside primarily in VRAM.
- KV cache stores previously computed keys and values.
- Activations and intermediate tensors also consume GPU memory.

**Important:** Register and shared-memory reuse reduces repeated accesses to slower global memory.

## 4. Compute Throughput and Memory Bandwidth

Two important GPU performance limits:

- **Compute throughput:** How many floating-point operations the GPU can perform per second (FLOPS).
- **Memory bandwidth:** How quickly data can move between GPU memory and processing units (GB/s).

**Arithmetic intensity (AI):**

\[
AI = \frac{\text{FLOPs}}{\text{Bytes transferred}}
\]

- High arithmetic intensity generally indicates a compute-bound workload.
- Low arithmetic intensity generally indicates a memory-bound workload.

**Roofline model:**

\[
Performance = \min(Peak\ Compute,\ Bandwidth \times AI)
\]

For LLM inference:
- **Prefill:** Processes many prompt tokens together; often more compute-intensive.
- **Decode:** Generates tokens sequentially; often memory-bandwidth-bound, especially at small batch sizes.

## 5. Matrix Multiplication and CUDA Kernels

Transformer computation relies heavily on matrix multiplication.

\[
Y = XW^T
\]

Approximate matrix multiplication operations:

\[
FLOPs = 2MNK
\]

GPU execution involves:
1. CPU launches a CUDA kernel.
2. GPU schedules thread blocks onto SMs.
3. Threads execute the assigned computations.
4. Results are written to GPU memory.

**Tiling:** Divides large matrix operations into smaller blocks so data can be reused in registers and shared memory.

**Kernel launch overhead:** Every separate GPU kernel has scheduling and launch overhead. This becomes significant for workloads containing many small operations.

## 6. CUDA Synchronization and Streams

GPU execution is asynchronous: the CPU can continue executing while previously submitted GPU operations are still running.

- `torch.cuda.synchronize()` waits for outstanding GPU work to finish.
- CUDA events measure GPU execution time and coordinate dependencies.
- CUDA streams represent ordered sequences of GPU operations.
- Different streams can potentially execute concurrently.

**Host-to-device transfers:**

- Pageable memory: Normal CPU memory that may require staging before GPU transfer.
- Pinned memory: Page-locked CPU memory that supports efficient asynchronous transfers.
- `non_blocking=True`: Requests non-blocking transfer behavior but does not guarantee overlap.

**Event synchronization:**

`stream2.wait_event(event)` ensures that subsequent operations in stream 2 wait until the event recorded in another stream has completed. This coordinates GPU work without blocking the CPU.

## 7. Kernel Fusion

Kernel fusion combines multiple operations into a single GPU kernel.

Example:

\[
Y = ReLU(XW+b)
\]

Instead of launching separate kernels for matrix multiplication, bias addition and ReLU, a fused kernel performs the operations together.

Benefits:
- Fewer kernel launches.
- Less intermediate global-memory traffic.
- Reduced execution overhead.
- Potentially improved performance.

**Windows note:** The kernel fusion benchmark uses `torch.compile`, which requires
Triton for CUDA compilation. If it raises `TritonMissing`, install a compatible
`triton-windows` version in the same Python environment as PyTorch. For PyTorch
2.15, install Triton 3.9:

```powershell
python -m pip install "triton-windows>=3.9,<3.10"
```

Other PyTorch versions may require a different Triton version; see the
[Triton Windows compatibility table](https://github.com/triton-lang/triton-windows#3-pytorch).

## 8. FlashAttention

Traditional attention:

\[
Attention(Q,K,V)=
softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

The intermediate attention score matrix has shape:

\[
(B,H,S,S)
\]

For long sequences, this matrix consumes substantial memory.

**FlashAttention** improves attention efficiency through:

- Tiled computation of Q, K and V.
- Reuse of on-chip GPU memory.
- Online softmax computation.
- Avoiding materialization of the complete attention score matrix in high-bandwidth GPU memory.

It computes exact attention (up to floating-point rounding differences), rather than approximating or dropping attention scores.

## 9. FlashInfer

FlashInfer is a GPU kernel library focused on efficient LLM inference.

It is particularly useful for:
- KV-cache attention.
- Paged KV-cache layouts.
- Variable-length sequences.
- Optimized attention kernels for autoregressive decoding.

**FlashAttention vs FlashInfer:**

- FlashAttention focuses on IO-efficient attention algorithms.
- FlashInfer provides specialized kernels for inference workloads and KV-cache management.

They are separate implementations and technologies.

## 10. CUDA Graphs

CUDA Graphs capture a sequence of GPU operations and allow that sequence to be replayed.

Without CUDA Graphs, the CPU repeatedly submits individual kernel launches.

With CUDA Graphs, the CPU can replay a captured execution graph with reduced launch overhead.

Benefits:
- Reduced CPU launch overhead.
- More efficient repeated execution.
- Useful for stable, repetitive inference workloads.

Limitations include constraints around dynamic shapes, memory addresses and control flow.

## 11. Key Takeaways for LLM Inference

| Concept | Why it matters |
|---|---|
| SMs and Tensor Cores | Parallel execution and accelerated matrix operations |
| Memory hierarchy | Efficient data access and reuse |
| Memory bandwidth | Critical for autoregressive decoding |
| Tiling | Reduces global-memory traffic |
| CUDA streams | Enables potential overlap of GPU operations |
| Pinned memory | Improves asynchronous CPU-to-GPU transfers |
| Kernel fusion | Reduces launch overhead and intermediate memory traffic |
| FlashAttention | Reduces attention memory requirements |
| FlashInfer | Optimizes inference-time attention and KV-cache access |
| CUDA Graphs | Reduces repeated kernel-launch overhead |

**Overall principle:** GPU performance is not determined by compute power alone. Efficient inference requires balancing computation, memory movement, kernel launches, synchronization and data reuse.