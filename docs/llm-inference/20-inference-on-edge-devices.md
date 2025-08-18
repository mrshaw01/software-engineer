# Inference on Edge Devices

This section provides practical guidance to run LLMs and generative models on edge hardware (phones, tablets, embedded Linux, microcontrollers, and small single-board computers). It focuses on model compression, kernel/runtime choices, memory planning (especially KV cache), and system-level tactics to meet tight latency, memory, and power budgets.

## 1. Edge Constraints and Goals

**Constraints**

- Limited memory (hundreds of MB to a few GB), often shared with the OS/GPU.
- Modest compute (mobile SoCs, integrated NPUs/DPUs, little to no discrete GPU).
- Strict power/thermal envelopes (sustained vs. burst performance, throttling).
- Intermittent connectivity and disk/network I/O limitations.

**Goals**

- Single-stream interactive latency (decode tokens/second).
- Tight, predictable tail latency (no thermal collapse).
- Small binary size, quick app cold-start.
- Privacy-preserving on-device inference.

## 2. Hardware Landscape (What to Optimize For)

- **Android ARM64**: big.LITTLE CPU (Cortex-A), mobile GPU (Adreno/Mali), ISP/DSP/NPU (Hexagon/TPU-like). Vector ISA: NEON/SVE.
- **iOS/macOS on Apple Silicon**: high-perf CPU, integrated GPU, AMX matrix units; Core ML/BNNS/Metal backends.
- **Embedded Linux (Jetson, Pi)**: Jetson Orin/NX (CUDA/TensorRT), Raspberry Pi (ARM CPU, small GPU), various NPUs (e.g., Hailo, Intel NPU).
- **Microcontrollers**: very small SRAM/Flash (kB–MB); integer-only (TFLM/microTVM pipelines).

Implication: you must pick a runtime that matches the device accelerators and offers quantized kernels for your target ops (matmul/attention/conv/MLP).

## 3. Model Selection & Architecture Tweaks

- Prefer **small decoder-only transformers** (≤1–3B params) for interactive LLM use.
- Use **GQA or MQA** to reduce KV heads and cache size.
- Use **local/sliding-window attention** (with optional attention sinks) to cap KV growth.
- Reduce **vocab size** where possible (domain-limited vocabularies meaningfully shrink embedding and unembedding matrices).
- Prefer **geGLU/SiLU/Swish-Gated MLPs** with fused kernels; avoid exotic ops that lack mobile kernels.
- If using MoE: run with **small top-k** (e.g., k=1) and few experts; ensure kernels are fused and sparsity-friendly on the target runtime.

## 4. Compression Toolbox

### 4.1 Weight Quantization (primary lever)

- **Per-tensor / per-channel** int8; **group-wise int4** (e.g., group size 32/64) for best accuracy–size trade-off.
- Advanced post-training methods: **AWQ**, **GPTQ**, **SpQR**, **AQLM**, SmoothQuant-style scaling.
- **Mixed-precision** policy: quantize linear weights; keep small layers (layernorm/embeddings/scales) in FP16/FP32 if needed.
- Practical rule of thumb for model memory:

  - FP16: \~2.0 bytes/param
  - Int8: \~1.0 byte/param (+ scales/zeros)
  - Int4: \~0.5 byte/param (+ scales/zeros; effective \~0.55–0.7 bytes/param)

### 4.2 KV-Cache Quantization

- Quantize **K/V to int8** (often near-lossless for quality) or **int4** (may require per-head/group scales).
- Use **symmetric** quant with per-head or group scales; precompute dequant factors to avoid runtime overhead.
- Consider **float8** variants where supported by hardware; validate kernels exist on target runtime.

### 4.3 Pruning

- **Structured pruning** (head/channel/neuron) to keep kernels dense and efficient.
- Avoid unstructured sparsity unless runtime supports true sparse matmuls on device.

### 4.4 Distillation

- Distill from a larger teacher to a compact student (<1–3B). Include **logit matching** on long-context prompts and safety-aligned data.

### 4.5 Adapters & Merging

- Ship a **quantized base** + small **LoRA/LoRA++ adapters**.
- Optionally **merge adapters** to avoid runtime adapter cost when the task is fixed.

## 5. Execution Optimizations

### 5.1 Prefill vs. Decode Separation

- Prefill is bandwidth/compute heavy; decode is latency sensitive and sequential.
- Build separate kernel graphs/paths; pre-capture kernels where supported (e.g., CUDA graphs on Jetson).

### 5.2 KV-Cache Management

- **Sliding window** or **recent-context truncation** to bound KV memory.
- **Paged KV** storage (contiguous, cache-friendly layout) to reduce TLB misses.
- **KV eviction** strategies for streaming sessions; optionally **attention sinks** to preserve instruction tokens.
- **KV precision downshift** (e.g., prefill in FP16, then down-quantize to int8/int4 post-layer).

**KV cache memory per token**:

```
bytes_per_token ≈ L * H_kv * d_k * (bytes_K + bytes_V)
```

- L = number of layers
- H_kv = KV heads (with GQA/MQA, H_kv << H_q)
- d_k = head_dim (≈ hidden_size / H_q)
- bytes_K/V = storage bytes (e.g., int8=1, fp16=2)

**Examples**

- 1.3B-ish (d=2048, L=24, H_q=32, H_kv=8, d_k=64, int8 KV):

  - per layer = 8*64*(1+1) = 1024 bytes → total ≈ 24 KB/token.

- 3B-ish (d=3072, L=32, H_q=24, H_kv=8, d_k=128, fp16 KV):

  - per layer = 8*128*(2+2) = 4096 bytes → total ≈ 128 KB/token.

### 5.3 Attention Variants

- **Local/sliding window** attention for long inputs.
- **Chunked/streaming** attention to limit working set.
- **GQA/MQA** to reduce KV heads.
- Use **fused attention** kernels when available.

### 5.4 Operator Fusion & Layout

- Fuse norm + linear + activation where supported.
- Choose **weight packing** formats expected by your backend (e.g., blocked layouts).
- **Memory map** weights (e.g., GGUF) to avoid copying huge blobs into RAM.

### 5.5 Threading & Affinity

- Pin **decode thread(s)** to big cores; keep background tasks on LITTLE cores.
- On Android, set **thread priority** and **CPU affinity**; on Jetson, use **nvpmodel/MAXN** and pin threads.
- Cap thread count to avoid contention and thermal spikes.

## 6. Runtime/Backend Choices by Platform

> Pick the one that has: (1) kernels for your operators, (2) quantization you need, (3) a stable packaging story.

- **Android**

  - **ExecuTorch** (PyTorch mobile runtime, quant kernels, AOT).
  - **TensorFlow Lite** (int8/int4 delegates; XNNPACK; NNAPI).
  - **ONNX Runtime Mobile** (selective build, NNAPI, QNN/HW delegates).
  - **NCNN/MNN** (lightweight inference, Vulkan paths).
  - **llama.cpp / ggml / gguf** (CPU-first, NEON; Vulkan/Metal backends optional).
  - **Vendor SDKs**: **QNN/Hexagon** (Qualcomm), **Arm NN**, **Ethos** delegates.

- **iOS/macOS**

  - **Core ML** (weight quantization, ANE/BNNS/MPS backends).
  - **Metal** (custom kernels; MPS Graph).
  - **llama.cpp** via **Metal**; **MLC-LLM** for Metal codegen.

- **Embedded Linux (Jetson)**

  - **TensorRT / TensorRT-LLM** for CUDA-optimized inference.
  - **llama.cpp** for CPU-only or mixed backends.
  - Lightweight **ONNX Runtime / TFLite** builds where CUDA is not available.

- **Microcontrollers**

  - **TensorFlow Lite Micro** / **microTVM** (integer-only, static memory arenas).

## 7. Memory Sizing & Budgets

### 7.1 Weights Footprint

Approximate RAM for weights (ignoring small overheads):

- FP16: `params * 2.0` bytes
- Int8: `params * 1.0` bytes (+ scales/zeros ≈ 5–15%)
- Int4: `params * 0.5` bytes (+ scales/zeros ≈ 10–40%)

**Examples (including overhead):**

- **1B params, int4** → \~0.55–0.70 GB
- **3B params, int4** → \~1.6–2.1 GB
- **1B params, int8** → \~1.1–1.2 GB

### 7.2 KV Cache Footprint

Use the formula in §5.2. Multiply by **sequence length** to get total KV memory. Keep a **headroom margin** (≥20%) for allocator fragmentation and runtime buffers.

## 8. Latency Budgeting & Throughput

- **Prefill TPS** (tokens/sec) benefits from more threads/parallelism; memory bandwidth-bound.
- **Decode TPS** is dominated by per-token matmuls; schedule for single-stream latency.
- Heuristics:

  - Favor **int4/int8** matmuls if accurate on your data.
  - Keep **H_kv small** (GQA/MQA) and use **sliding windows**.
  - Tune **thread count** to just saturate the big cores without triggering thermal throttling.

## 9. Privacy, Security, and Offline

- Encrypt on-disk model files; validate signatures on load.
- Consider **secure enclaves/TEE** for sensitive use-cases.
- Ensure **no telemetry** is sent without consent; offer offline mode by default.

## 10. Testing, Benchmarking, and Energy

- **Correctness**: golden answers on a fixed prompt set; check determinism with fixed seeds.
- **Latency**: p50/p90/p99 decode TPS and end-to-end prompt latency.
- **Memory**: peak RSS and allocator fragmentation; prove KV stays within bounds.
- **Energy/Thermals**:

  - Android: `batterystats`, `perfetto`, vendor thermal APIs.
  - iOS: MetricKit / Instruments.
  - Jetson: `tegrastats`, power modes, SoC temperature.

## 11. Deployment Recipes (Quick Starts)

### 11.1 Android (CPU-first, llama.cpp)

1. Quantize to **GGUF** (Q4_K or Q4_K_M).
2. Build **llama.cpp** with NEON and thread pinning; package as native library.
3. Reserve memory-mapped GGUF and a **bounded KV** (e.g., 1–2k tokens).
4. Pin the decode thread to a big core; set high thread priority.

### 11.2 iOS (Core ML)

1. Convert the model to **Core ML** (MIL) with **linear int8/int4** where supported.
2. Use **BNNS/MPS** delegates; prefer fused layers.
3. Keep KV in int8; implement sliding window attention.
4. Bundle as **.mlmodelc**; memory-map weights.

### 11.3 Jetson (TensorRT-LLM)

1. Export to ONNX; calibrate/quantize (int8 or fp8 if available).
2. Build a TensorRT engine with fused attention and persistent kernels.
3. Use CUDA graphs for decode; place KV in device memory with a capped window.
4. Set `nvpmodel` to performance mode; pin threads.

## 12. Edge-Focused Checklist

- [ ] Target model ≤ 3B params with **GQA/MQA**.
- [ ] **Int4/Int8** weights with validated accuracy; small FP16 islands if needed.
- [ ] **Int8 KV** (or int4 with care) + **sliding window** attention.
- [ ] **Fused kernels** and memory-mapped weights.
- [ ] **Thread pinning** to big cores; measured thermals under worst-case prompts.
- [ ] **Deterministic** unit prompts; golden outputs.
- [ ] Latency/energy metrics captured and regressed per build.

## 13. Troubleshooting Guide

- **Thermal throttling**: reduce threads, lower sustained TPS target, adopt more aggressive quantization, or shorten window.
- **Accuracy loss after quantization**: try per-channel/group-wise scales, larger group size (e.g., 64), keep attention/out projections in higher precision, or perform light QAT.
- **Jank/GC pauses**: memory-map weights, pre-allocate KV arena, avoid frequent allocations.
- **OOM on long chats**: enable sliding window, compress KV, truncate history, or summarize earlier turns on-device.

## 14. Summary

Successful edge inference is a balancing act: pick compact architectures, apply aggressive but careful quantization (weights and KV), use locality-aware attention, map to a runtime with the right fused kernels, and tune threading and memory to avoid thermal and OOM pitfalls. With these techniques, robust on-device LLM experiences are feasible on today’s phones, embedded boards, and even microcontrollers for narrow tasks.
