# Module 7 — Batching, Scheduling and KV-Cache Optimization

## Overview

This module explores how LLM inference engines efficiently serve multiple requests simultaneously.

When an application sends a prompt to an LLM, the inference engine must decide:

- When should a request start executing?
- How should multiple requests be batched?
- What happens when requests have different prompt and output lengths?
- How is GPU memory allocated for each request?
- How can multiple requests share previously computed context?
- What happens when GPU memory is insufficient?

These problems are central to production inference systems such as vLLM, TensorRT-LLM and other high-throughput serving engines.

The module focuses on three major areas:

1. **Batching and scheduling** — efficiently executing multiple requests.
2. **KV-cache management** — storing and reusing attention keys and values.
3. **Memory optimization** — managing GPU memory through paging, sharing and admission control.

---

# 1. Understanding Batching

## 1.1 Single-request inference

Consider a request:

> Explain how a transformer works.

The inference engine processes the input prompt and generates output tokens autoregressively.

There are two stages:

**Prefill**

- Processes the entire input prompt.
- Computes the initial Key (K) and Value (V) tensors.
- Stores these tensors in the KV cache.
- Produces logits from which the first output token is sampled.

**Decode**

- Processes one new token per sequence per iteration.
- Uses previously stored KV values instead of recomputing the entire sequence.
- Generates the next token autoregressively.

Example:

```
Prompt:  Explain transformers

Prefill:
[Explain, transformers]

Decode 1:
[Explain, transformers, A]

Decode 2:
[Explain, transformers, A, transformer]

Decode 3:
[Explain, transformers, A, transformer, is]
```

The sequence grows as new tokens are generated.

## 1.2 Static batching

Static batching combines multiple requests into a single batch.

Example:

```
Request A: 100 tokens
Request B: 200 tokens
Request C:  50 tokens
```

If all requests are executed together, the batch may need to accommodate the longest sequence.

For a simple padded batch:

```
A: [100 tokens + 100 padding tokens]
B: [200 tokens]
C: [ 50 tokens + 150 padding tokens]
```

The allocated positions are:

```
Total allocated = 3 × 200 = 600
Useful tokens   = 100 + 200 + 50 = 350
Padding tokens  = 250
```

Batch efficiency:

[\
Efficiency = \frac{Useful\ Tokens}{Allocated\ Positions}\
]

[\
\= \frac{350}{600} \times 100 = 58.33%\
]

The main problem is that shorter requests waste computation and memory on padding.

Another problem is that a batch may remain occupied until its longest-running request finishes.

## 1.3 Continuous batching

Continuous batching solves the fixed-batch limitation.

Instead of waiting for every request in a batch to complete, the inference engine can remove completed requests and admit new ones.

Example:

```
Iteration 1: A, B
Iteration 2: A, B
Iteration 3: A, B
Iteration 4: B, C
Iteration 5: B, C
Iteration 6: C, D
```

When A finishes, its execution slot can be used by C.

When B finishes, another waiting request can be admitted.

The batch composition changes dynamically.

**Important:** Continuous batching is generally implemented at the iteration level. The engine can change which requests participate between forward passes.

### Static batching vs continuous batching

| Feature                 | Static batching            | Continuous batching            |
| ----------------------- | -------------------------- | ------------------------------ |
| Batch membership        | Fixed during execution     | Changes dynamically            |
| Completed request       | Batch may wait             | Removed immediately            |
| New request             | Usually waits for batch    | Can join when resources permit |
| Variable output lengths | Less efficient             | Better utilization             |
| GPU utilization         | Can suffer from idle slots | Usually better                 |

---

# 2. Padding and Length Bucketing

## 2.1 Why padding is required

When sequences have different lengths, conventional dense batching often pads shorter sequences to match the longest sequence.

Example:

```
Sequence A: 100 tokens
Sequence B: 200 tokens
Sequence C:  50 tokens
```

A padded batch uses:

```
A: 100 useful + 100 padding
B: 200 useful +   0 padding
C:  50 useful + 150 padding
```

This creates unnecessary computation.

## 2.2 Length bucketing

Length bucketing groups requests with similar sequence lengths.

Example:

Original order:

```
[100, 200, 500, 150, 300, 80, 450, 120]
```

FIFO grouping:

```
Batch 1: [100, 200, 500, 150]
Batch 2: [300,  80, 450, 120]
```

Sorted grouping:

```
Batch 1: [ 80, 100, 120, 150]
Batch 2: [200, 300, 450, 500]
```

The second arrangement generally requires fewer padded positions.

### Padding efficiency

[\
Efficiency =\
\frac{Useful\ Tokens}{Allocated\ Token\ Positions}\
\times 100\
]

In our experiment:

| Metric              | FIFO grouping | Length bucketing |
| ------------------- | ------------- | ---------------- |
| Useful tokens       | 1,900         | 1,900            |
| Allocated positions | 3,800         | 2,600            |
| Padding positions   | 1,900         | 700              |
| Efficiency          | 50%           | 73.08%           |

Length bucketing reduced allocated positions by:

[\
\frac{3800-2600}{3800}\times100 = 31.58%\
]

**Key takeaway:** Grouping similar-length sequences reduces padding overhead and improves batch efficiency.

---

# 3. Iteration-Level Scheduling

## 3.1 Prefill and decode have different workloads

An inference engine handles two different types of computation.

| Stage   | Work performed                              |
| ------- | ------------------------------------------- |
| Prefill | Processes multiple prompt tokens            |
| Decode  | Processes one new token per active sequence |

Suppose:

```
Request A:
Prompt = 100 tokens
Output = 20 tokens

Request B:
Prompt = 500 tokens
Output = 10 tokens
```

A prefill operation can process a large number of prompt tokens, while decode generally processes one new token for each active sequence per iteration.

This difference creates scheduling challenges.

A long prompt can occupy substantial GPU computation while existing requests are waiting for their next output token.

## 3.2 Maximum active sequences

An inference engine may define:

```
MAX_ACTIVE_SEQUENCES = 4
```

This means no more than four requests can be active simultaneously under that configured limit.

It does not mean that the model can only process four tokens.

Each active request can contain many tokens.

## 3.3 Maximum batched tokens

An engine may also define:

```
MAX_BATCHED_TOKENS = 512
```

This limits the total token workload scheduled in a particular iteration, according to the engine's scheduling rules.

For example:

```
Request A decode: 1 token
Request B decode: 1 token
Request C prefill: 200 tokens
Request D prefill: 300 tokens

Total = 502 tokens
```

This fits within a 512-token budget.

The two limits control different things:

- Maximum active sequences controls concurrent requests.
- Maximum batched tokens controls token workload per iteration.

## 3.4 Chunked prefill

A long prompt can be split into smaller chunks.

Example:

```
Prompt length = 100 tokens
Chunk size    = 25 tokens

Chunk 1: tokens  1–25
Chunk 2: tokens 26–50
Chunk 3: tokens 51–75
Chunk 4: tokens 76–100
```

Instead of processing all 100 tokens in one operation, the engine processes smaller chunks.

This allows the scheduler to interleave other work, such as decode operations.

Example:

```
Iteration 1: A prefill chunk 1
Iteration 2: A prefill chunk 2
Iteration 3: A decode, B prefill chunk 1
Iteration 4: A decode, B prefill chunk 2
```

This helps prevent a long prompt from monopolizing execution.

### Does the model generate tokens during prefill?

The transformer produces logits during forward execution, but the engine normally samples the first output token only after the complete prompt has been processed.

For chunked prefill, intermediate chunk outputs are not treated as completed user-facing generation.

The inference engine controls scheduling. The model itself does not decide whether to prefill or decode.

---

# 4. Understanding the KV Cache

## 4.1 What are Q, K and V?

In transformer attention:

- Q = Query
- K = Key
- V = Value

These are computed from the hidden representations of the input tokens using learned projection matrices.

They are not the model weights themselves.

The attention operation uses queries to determine how much attention should be paid to different keys, then combines the corresponding values.

## 4.2 Why do we need a KV cache?

During autoregressive generation, the sequence grows one token at a time.

Without caching, the model would repeatedly recompute the Key and Value representations of previous tokens.

For example:

```
Iteration 1:
[The]

Iteration 2:
[The, model]

Iteration 3:
[The, model, works]

Iteration 4:
[The, model, works, well]
```

At every iteration, previous tokens are still part of the context.

The KV cache stores previously computed K and V tensors so that the model can reuse them.

Conceptually:

```
Request A KV cache:
[K1, K2, K3, K4, ...]

Request B KV cache:
[K1, K2, K3, ...]

Request C KV cache:
[K1, K2, ...]
```

Each request has its own logical context.

The inference engine maintains the metadata needed to associate the correct KV-cache entries with each request.

**Important:** The model does not independently identify which KV cache belongs to which request. The serving runtime supplies the appropriate cache and sequence metadata to the attention computation.

---

# 5. PagedAttention and KV-Cache Blocks

## 5.1 The problem with contiguous KV-cache allocation

Suppose a request needs 35 tokens of KV cache.

If the cache is allocated in contiguous memory, the engine needs a sufficiently large contiguous region.

Variable sequence lengths and frequent request completion can create memory fragmentation.

This makes allocation and memory utilization more difficult.

## 5.2 Block-based allocation

PagedAttention-style systems divide KV-cache memory into fixed-size blocks.

Example:

```
BLOCK_SIZE = 16
```

Each physical block can store KV entries for 16 token positions.

For a request containing 35 tokens:

[\
Required\ Blocks = \lceil \frac{35}{16} \rceil\
]

[\
\= \lceil 2.1875 \rceil = 3\
]

The allocation looks like:

```
Logical sequence:

Tokens  1–16  → Logical Block 0
Tokens 17–32  → Logical Block 1
Tokens 33–35  → Logical Block 2

Physical allocation:

Logical Block 0 → Physical Block 4
Logical Block 1 → Physical Block 9
Logical Block 2 → Physical Block 2
```

The physical blocks do not need to be contiguous.

The inference engine maintains a block table:

```
Request A:
Logical Block 0 → Physical Block 4
Logical Block 1 → Physical Block 9
Logical Block 2 → Physical Block 2
```

The attention kernel uses this mapping to locate the correct KV-cache data.

## 5.3 Why non-contiguous blocks work

The logical sequence remains ordered even though physical memory is scattered.

The block table translates logical positions into physical memory locations.

This is conceptually similar to virtual memory in an operating system:

- Logical sequence positions are like virtual addresses.
- Physical KV-cache blocks are like physical memory pages.
- The block table maps one to the other.

## 5.4 Releasing and reusing blocks

Suppose:

```
Request A → Physical blocks [2, 5, 8]
Request B → Physical blocks [1, 3]
```

When A completes:

```
Blocks 2, 5, 8 → Free
```

A new request D can reuse those blocks.

The previous contents no longer belong to A once its references are released and the blocks are safely reclaimed.

### Internal fragmentation

If the block size is 16 and a request needs 35 token slots:

```
Allocated capacity = 3 × 16 = 48
Useful capacity    = 35
Unused capacity    = 13
```

[\
Internal\ Fragmentation =\
48-35=13\
]

[\
Utilization = \frac{35}{48}\times100=72.92%\
]

Smaller blocks can reduce unused space but may increase metadata and management overhead.

Larger blocks reduce the number of blocks but can increase internal fragmentation.

---

# 6. KV-Cache Memory Utilization

The number of blocks required for a sequence is:

[\
Blocks = \lceil \frac{Sequence\ Length}{Block\ Size} \rceil\
]

Allocated capacity:

[\
Allocated\ Tokens = Blocks \times Block\ Size\
]

Unused capacity:

[\
Wasted\ Slots = Allocated\ Tokens - Sequence\ Length\
]

Utilization:

[\
Utilization =\
\frac{Sequence\ Length}{Allocated\ Tokens}\
\times 100\
]

Example with a sequence length of 73:

| Block size | Required blocks | Allocated slots | Wasted slots | Utilization |
| ---------- | --------------- | --------------- | ------------ | ----------- |
| 8          | 10              | 80              | 7            | 91.25%      |
| 16         | 5               | 80              | 7            | 91.25%      |
| 32         | 3               | 96              | 23           | 76.04%      |
| 64         | 2               | 128             | 55           | 57.03%      |

The choice of block size is a trade-off between memory efficiency and allocation/management overhead.

---

# 7. Prefix Caching

## 7.1 The problem of repeated prompts

Imagine three users send requests that all begin with the same system prompt:

```
System prompt:
You are a helpful assistant. Answer clearly and accurately.
```

Their requests are:

```
Request A:
[Shared system prompt] + [Question A]

Request B:
[Shared system prompt] + [Question B]

Request C:
[Shared system prompt] + [Question C]
```

Without prefix caching, the engine may independently compute the same prefix for each request.

This repeats work and duplicates KV-cache storage.

## 7.2 How prefix caching works

The engine computes the shared prefix once and stores its KV cache.

Subsequent requests with the same compatible token prefix can reuse that cached KV data.

Conceptually:

```
Shared prefix KV:
Physical Blocks [0, 1]

Request A:
[0, 1] + [Private blocks A]

Request B:
[0, 1] + [Private blocks B]

Request C:
[0, 1] + [Private blocks C]
```

The requests share the physical blocks representing the common prefix.

Their unique continuation tokens receive separate blocks.

## 7.3 Does prefix caching reduce memory?

Yes.

Suppose three requests each have:

```
Shared prefix = 4 tokens
Unique tokens = 2, 2, 3
```

Without sharing:

[\
3 \times 4 + 2 + 2 + 3 = 19\
]

With sharing:

[\
4 + 2 + 2 + 3 = 11\
]

Memory reduction:

[\
\frac{19-11}{19}\times100=42.1%\
]

The actual memory saving depends on block size, alignment, cache retention and the number of shared tokens.

Prefix caching also reduces repeated prefill computation.

**Important:** Prefix caching stores reusable KV data, not merely the original prompt text.

---

# 8. Reference Counting for Shared KV Blocks

When multiple requests share a physical block, the inference engine must know how many active owners still reference it.

This is handled conceptually through reference counting.

Example:

```
Shared Prefix:
Block 0
Block 1

Request A → Blocks 0, 1
Request B → Blocks 0, 1
Request C → Blocks 0, 1
Prefix cache → Blocks 0, 1
```

The reference count is:

```
Block 0 → 4 references
Block 1 → 4 references
```

There are four references because the prefix cache itself retains one reference in addition to the three requests.

When Request A finishes:

```
Block 0 → 3
Block 1 → 3
```

When Request B finishes:

```
Block 0 → 2
Block 1 → 2
```

When Request C finishes:

```
Block 0 → 1
Block 1 → 1
```

The blocks are still retained by the prefix cache.

When the prefix cache evicts the entry:

```
Block 0 → 0
Block 1 → 0
```

The blocks can now be returned to the free pool.

**Key takeaway:** A block must not be freed while any request or cache entry still references it.

---

# 9. Copy-on-Write (COW)

## 9.1 Why Copy-on-Write is needed

Suppose two requests share a prefix:

```
Request A → [T1, T2, T3, T4, T5, T6]
Request B → [T1, T2, T3, T4, T5, T6]
```

Assume the block size is four tokens.

The block layout is:

```
Block 0: [T1, T2, T3, T4]
Block 1: [T5, T6, _, _]
```

Both requests share these blocks.

Now Request A generates a new token:

```
Request A → [T1, T2, T3, T4, T5, T6, HELLO]
```

Request B generates a different token:

```
Request B → [T1, T2, T3, T4, T5, T6, GOODBYE]
```

If both requests append to the same physical block, they would overwrite each other's KV data.

## 9.2 Copy-on-Write solution

The engine checks the reference count of the partially filled block.

If the block is shared:

1. Allocate a new physical block.
2. Copy the existing KV data into it.
3. Update the writing request's block table.
4. Append the new token to the private block.

Example:

```
Before:

Block 0: [T1, T2, T3, T4]
Block 1: [T5, T6, _, _]

A → [Block 0, Block 1]
B → [Block 0, Block 1]
```

After A writes:

```
Block 1: [T5, T6, _, _]
Block 2: [T5, T6, HELLO, _]

A → [Block 0, Block 2]
B → [Block 0, Block 1]
```

B can now independently append GOODBYE to Block 1.

## 9.3 When is copying unnecessary?

If the last block is already full, the engine can allocate a new private block.

If the last block is not shared (reference count equals one), the request can append directly.

Only a shared, partially filled block requires copying in this simplified scenario.

**Key takeaway:** Copy-on-Write allows requests to share a common prefix safely while maintaining independent continuations.

---

# 10. Admission Control and KV-Cache Memory Pressure

## 10.1 Why admission control is necessary

GPU memory is finite.

Suppose the engine has:

```
Total physical blocks = 12
Block size            = 16 tokens
```

Requests require:

```
A = 4 blocks
B = 5 blocks
C = 6 blocks
```

Total demand:

[\
4+5+6=15\ blocks\
]

But only 12 blocks are available.

The engine cannot admit all three requests simultaneously if all their required blocks must be reserved.

It must decide which requests can execute and which must wait.

## 10.2 Three independent limits

An inference engine commonly manages:

| Limit                     | What it controls             |
| ------------------------- | ---------------------------- |
| Maximum active sequences  | Concurrent request count     |
| Available KV-cache blocks | Memory capacity              |
| Maximum batched tokens    | Per-iteration token workload |

These are independent constraints.

A request may fit within the active-sequence limit but still fail admission because there are not enough KV-cache blocks.

Similarly, sufficient KV memory does not mean that the request can be scheduled in the current iteration if the token budget is exhausted.

## 10.3 Dynamic admission scheduling

Consider:

```
TOTAL_BLOCKS = 12
MAX_ACTIVE_SEQUENCES = 3
BLOCK_SIZE = 16
```

Requests:

| Request | Prompt | Output | Total tokens | Required blocks |
| ------- | ------ | ------ | ------------ | --------------- |
| A       | 32     | 32     | 64           | 4               |
| B       | 48     | 32     | 80           | 5               |
| C       | 64     | 32     | 96           | 6               |
| D       | 16     | 16     | 32           | 2               |

Initial admission:

```
A admitted → 4 blocks
B admitted → 5 blocks

Used blocks = 9
Free blocks = 3

C waits → needs 6 blocks
D waits → needs 2 blocks
```

When A completes:

```
A releases 4 blocks

Free blocks = 7

C admitted → 6 blocks
Free blocks = 1
D still waits
```

When B completes:

```
B releases 5 blocks

Free blocks = 6

D admitted → 2 blocks
```

The scheduler automatically retries waiting requests when resources become available.

## 10.4 FIFO scheduling and head-of-line blocking

In a FIFO waiting queue, requests are considered in arrival order.

Suppose:

```
Waiting queue:
C → needs 6 blocks
D → needs 2 blocks

Available blocks = 3
```

D could fit, but C is at the front of the queue.

A strict FIFO scheduler does not bypass C, so D remains waiting.

This is called **head-of-line blocking**.

An alternative scheduler could allow D to bypass C, improving resource utilization, but that may affect fairness and request ordering.

This is a scheduling policy trade-off.

---

# 11. Preemption and Recomputation — Conceptual Overview

When KV-cache memory becomes insufficient, an inference engine may need to preempt an active request.

Preemption means temporarily removing a request from active GPU execution to free resources for other requests.

Conceptually:

```
Active:
A → 4 blocks
B → 5 blocks
C → 3 blocks

Memory pressure occurs.

Scheduler preempts B:

B's KV blocks → Released

B's request state → Retained for later recovery
```

When B resumes, the engine may need to recompute its prompt and previously generated context to reconstruct its KV cache.

This is called recomputation.

The scheduler must balance:

- GPU memory availability
- Request priority
- Time already spent processing a request
- Recompute cost
- Latency and fairness

Preemption is therefore not free: it trades memory availability for additional computation and latency.

---

# 12. Putting Everything Together

A production inference engine combines all these mechanisms.

Consider the following request lifecycle:

```
                  New request arrives
                          |
                          v
                  Tokenize the prompt
                          |
                          v
                  Admission controller
                          |
             +------------+------------+
             |                         |
        Resources available       Resources unavailable
             |                         |
             v                         v
        Admit request             Waiting queue
             |                         |
             v                         |
          Prefill                      |
             |                         |
             v                         |
       Allocate KV blocks              |
             |                         |
             v                         |
       Store K and V                   |
             |                         |
             v                         |
       First output token              |
             |                         |
             v                         |
           Decode                      |
             |                         |
             v                         |
       Append new token                |
             |                         |
             v                         |
       Update KV cache                 |
             |                         |
             v                         |
       Request complete?               |
         /       \                     |
       No         Yes                  |
       |           |                   |
       v           v                   |
    Decode     Release blocks          |
                   |                   |
                   v                   |
             Retry waiting <-----------+
                requests
```

In addition, the engine may:

- Use continuous batching to admit new requests between iterations.
- Use chunked prefill to interleave long prompts with decode work.
- Use paged KV allocation to reduce memory fragmentation.
- Use prefix caching to reuse shared context.
- Use reference counting to manage shared blocks.
- Use Copy-on-Write when requests diverge from a shared partial block.
- Use admission control and preemption when memory becomes constrained.

---

# 13. Important Formulas

| Metric                | Formula                                           |
| --------------------- | ------------------------------------------------- |
| Padding efficiency    | Useful tokens / Allocated positions × 100         |
| Padding overhead      | Padding positions / Allocated positions × 100     |
| Required KV blocks    | Ceiling(sequence length / block size)             |
| Allocated KV capacity | Number of blocks × Block size                     |
| Wasted KV slots       | Allocated capacity − Actual sequence length       |
| KV utilization        | Actual sequence length / Allocated capacity × 100 |
| Memory reduction      | (Old memory − New memory) / Old memory × 100      |
| Active sequence count | Number of currently active requests               |
| Free KV blocks        | Total blocks − Allocated blocks                   |

For a production model, KV-cache memory also depends on the number of layers, KV heads, head dimension, and data type.

A common approximation for one token's KV-cache memory is:

[\
M\_{token} =\
2 \times L \times H\_{KV} \times D\_{head} \times Bytes\
]

Where:

- (2) accounts for Key and Value.
- (L) is the number of transformer layers.
- (H\_{KV}) is the number of KV heads.
- (D\_{head}) is the dimension per head.
- (Bytes) is the number of bytes per stored element.

For a sequence of (S) tokens:

[\
M\_{sequence} = S \times M\_{token}\
]

This is an approximation; actual implementations may have additional layout, alignment and metadata overhead.

---

# 14. Practical Exercises Completed or Covered

The module included Python simulations and experiments for the following topics:

| Exercise | File / topic                   | Main learning                               |
| -------- | ------------------------------ | ------------------------------------------- |
| 7.1      | Static vs continuous batching  | Dynamic request scheduling                  |
| 7.2      | Padding and length bucketing   | Reducing wasted token positions             |
| 7.3      | Iteration-level scheduling     | Prefill vs decode scheduling                |
| 7.4      | Chunked prefill scheduler      | Mixing prefill and decode                   |
| 7.5      | KV block allocator             | Physical block allocation and reuse         |
| 7.6      | KV-cache block-size comparison | Internal fragmentation and utilization      |
| 7.7      | Prefix caching                 | Reusing shared KV prefixes                  |
| 7.8      | Reference counting             | Safely managing shared physical blocks      |
| 7.9      | Copy-on-Write                  | Independent continuation of shared prefixes |
| 7.10     | Admission control              | Active sequences vs memory constraints      |
| 7.11     | Dynamic admission scheduler    | Automatically admitting queued requests     |

Some exercises were explored conceptually or through code provided during the lessons; the table identifies the topics covered, not a claim that every script was fully executed.

---

# 15. Key Takeaways

1. **Batching improves GPU utilization**, but static batching can waste resources when requests have different lengths.
2. **Continuous batching** allows new requests to enter as existing requests complete.
3. **Prefill and decode are different workloads.** Prefill processes prompt tokens; decode generates one new token per active sequence per iteration.
4. **Chunked prefill** breaks long prompts into smaller pieces so that decode work can be scheduled alongside them.
5. **The KV cache stores previously computed keys and values**, avoiding repeated computation of the entire context during autoregressive generation.
6. **PagedAttention-style block allocation** allows KV data to be stored in non-contiguous physical memory blocks.
7. **Prefix caching** avoids recomputing and storing duplicate KV data for shared prompt prefixes.
8. **Reference counting** ensures that shared blocks are not freed while they are still being used.
9. **Copy-on-Write** protects shared partial blocks when requests generate different continuations.
10. **Admission control** ensures that requests are admitted only when execution and memory resources permit.
11. **Preemption and recomputation** provide ways to handle memory pressure, at the cost of additional work and latency.
12. **A production inference engine is a coordinated system** involving a scheduler, memory manager, attention kernels, and request lifecycle management.

---

# 16. What Comes Next?

The next logical step is to combine these individual concepts into one integrated inference-engine simulator.

The integrated implementation should contain:

- A request arrival queue
- Continuous batching
- Prefill and decode scheduling
- Chunked prefill
- A paged KV-cache allocator
- Shared prefix blocks
- Reference counting
- Copy-on-Write
- Admission control
- Request completion and memory reclamation
- Throughput, TTFT, latency and memory-utilization metrics

The objective is to understand how an inference engine coordinates all these components rather than treating batching, scheduling and KV-cache management as separate mechanisms.