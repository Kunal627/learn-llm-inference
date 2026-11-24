# Module 5: Quantization and Model Compression

This module explores how lower-precision numbers can reduce model memory and improve inference efficiency, and what accuracy trade-offs they introduce. The goal is not simply to use the fewest bits: it is to find a representation that fits the hardware and workload while keeping output quality acceptable.

## Key concepts

- **Floating-point precision:** FP32 uses 32 bits per value and is the usual high-precision reference. FP16 and BF16 use 16 bits, reducing storage by about half; FP16 has more precision for values in a narrower range, while BF16 supports a wider range with less precision. This is why BF16 can be more robust to very large or small values during model computation.

- **Integer quantization:** INT8 and INT4 represent values with 8 or 4 bits. Quantization maps floating-point values onto a limited set of integer levels using a scale (and sometimes a zero point); dequantization maps them back approximately. Fewer levels mean smaller storage, but also more rounding error and a greater chance that extreme values are clipped. INT4 weights need roughly one quarter of FP16 storage before extra scale metadata and packing overhead.

- **Weight and activation quantization:** Weight-only quantization stores model parameters at lower precision while activations are kept at higher precision. This can reduce model size and weight-memory traffic with relatively little change to the computation. Activation quantization also reduces the precision of intermediate values, which can improve efficiency further but depends more on the runtime and representative calibration data.

- **Quantization granularity:** Per-tensor quantization uses one scale for an entire tensor; per-channel uses separate scales for channels; group-wise quantization assigns scales to smaller groups of weights. Finer-grained scales can represent differences in value ranges more accurately, at the cost of storing and applying more metadata. Symmetric quantization centers levels around zero; asymmetric quantization adds a zero point and can better fit values that are not centered around zero.

- **PTQ and QAT:** Post-training quantization (PTQ) converts a trained model without retraining, making it relatively quick to apply. Quantization-aware training (QAT) exposes the model to simulated quantization during training or fine-tuning, giving it a chance to adapt and preserve quality, but it requires training data and extra compute.

- **Calibration and quality:** Calibration estimates useful scales from sample inputs, especially for activations. If calibration data does not resemble real use, important values may be clipped or mapped too coarsely. Compare quantized outputs with a baseline on representative prompts or tasks; tensor reconstruction error alone does not guarantee good language-model output quality.

- **GPTQ, AWQ, and GGUF:** GPTQ and AWQ are post-training weight-quantization approaches that use information about model behavior or activations to reduce quality loss. GGUF is a model file format commonly used by llama.cpp; it can store quantized models, but it is not itself a quantization algorithm. The Qwen scripts use a GGUF model, while the earlier lessons demonstrate simplified quantization concepts rather than implementing GPTQ or AWQ.

- **Weights versus KV cache:** Quantizing weights reduces memory used to hold the model. Quantizing the KV cache reduces memory used to store attention keys and values for active requests. These are separate choices: long contexts and concurrent requests can make KV-cache memory significant even when weights are quantized.

- **Performance trade-off:** Smaller representations can let a model fit on a smaller device and reduce memory bandwidth pressure. They do not automatically make inference faster: actual speed depends on hardware support, the inference backend, and its optimized quantized kernels. Evaluate memory, throughput, latency, and output quality together.

## Script guide

| Script | Concept covered |
| --- | --- |
| [1_datatypes.py](1_datatypes.py) | Compare FP32, FP16, and BF16 range, precision, memory use, and conversion error |
| [2_quantization.py](2_quantization.py) | Walk through symmetric INT8 quantization, dequantization, reconstruction error, and storage |
| [3_int4_int8.py](3_int4_int8.py) | Compare INT4 and INT8 error and theoretical storage; INT4 values are simulated in an INT8 tensor |
| [4_ptq_qat.py](4_ptq_qat.py) | Compare post-training quantization with quantization-aware fine-tuning on a small synthetic task |
| [5_quant_methods.py](5_quant_methods.py) | Compare per-tensor, per-channel, and group-wise scales and reconstruction errors |
| [6_activation_sensitivity.py](6_activation_sensitivity.py) | Relate activation magnitudes to output error after weight quantization |
| [7_calibration.py](7_calibration.py) | Show how calibration data changes activation scales, clipping, and quantization error |
| [8_weights_kv.py](8_weights_kv.py) | Estimate combined model-weight and KV-cache memory at different precisions and context lengths |
| [9_qwen_benchmark.py](9_qwen_benchmark.py) | Benchmark a local quantized Qwen GGUF model and record streaming latency and throughput metrics |
| [10_debug_stream.py](10_debug_stream.py) | Inspect raw server-sent events from a streaming chat-completion response |
| [11_qwen_test.py](11_qwen_test.py) | Send a non-streaming test request to the local Qwen model server |

## Running the Qwen examples

Scripts 9–11 expect a llama.cpp-compatible server at `http://localhost:8080` and a model available at `/models/Qwen3-8B-Q4_K_M.gguf`. Update the server URL or model path in the scripts if your setup differs. The benchmark writes CSV result and summary files; scripts 10 and 11 are useful for checking the streaming and regular response paths before benchmarking.

**Goal:** Produce a comparison of model memory, inference performance, and output quality for a quantized Qwen3 model.