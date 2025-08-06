# llm-inference

This repository documents the core concepts, optimization techniques, and practical implementations related to inference with large language models (LLMs). It is intended for engineers and researchers focused on building efficient, scalable, and production-ready LLM serving systems.

## Purpose

Modern LLMs are computationally expensive to run at inference time due to their size, autoregressive decoding pattern, and latency-critical usage scenarios. This repository provides:

- Clear explanations of how inference works in large models
- Optimization techniques at both the algorithm and systems levels
- Runnable examples for common inference bottlenecks and solutions
- Comparative studies of open-source inference frameworks

## Topics

Each section is structured with practical insights, tradeoffs, and relevant examples.

### 1. What Is Inference in LLMs?

Defines the inference process in the context of autoregressive LLMs. Describes input preparation, decoding loop, generation strategies (greedy, sampling, beam), and performance goals.

### 2. Tokenizer & Embedding Handling

Covers tokenization (e.g., BPE, SentencePiece), embedding lookups, and handling tokenizer-specific quirks in inference pipelines.

### 3. Transformer Inference Internals

Explains transformer layer components (attention, MLP), residual pathways, layer norm behavior at inference, and model-specific quirks (e.g., rotary embeddings, parallel attention).

### 4. KV Cache Optimization

Explores how storing and reusing past key/value pairs improves inference speed. Discusses memory layout, cache reuse policies, and dynamic shaping.

### 5. FlashAttention & Memory-Efficient Kernels

Details on efficient attention computation using techniques like FlashAttention. Discusses kernel-level memory reuse, tiling, and benefits of fused operations.

### 6. Quantization for Inference

Explains post-training quantization (PTQ), quant-aware training (QAT), and how int8/bfloat8 quantized models reduce latency and memory footprint.

### 7. CUDA/HIP Graphs for Fast Execution

Describes the use of CUDA/HIP graph APIs to cache kernel launches, reduce overhead, and accelerate repeated decode steps.

### 8. Speculative Decoding

Introduces speculative decoding using draft and target models. Explains token verification, rollback strategy, and how to combine models for lower latency.

### 9. Batching & Prefill vs. Decode

Breaks down the two phases of autoregressive inference: prefill (context embedding) and decode (token-by-token generation). Discusses optimal batching policies.

### 10. Prefill/Decode Separation

Formalizes execution separation into distinct kernel graphs or paths, enabling more efficient scheduling and shared compute for dynamic batching.

### 11. KV Cache Eviction & Memory Management

Strategies for managing KV cache across sessions. Includes techniques for eviction (LRU, age-based), compaction, and reuse across batch.

### 12. LLM Serving Architectures

Presents architectural designs: monolithic servers, worker/pipeline models, router-dispatchers, and microservice patterns for scalable inference.

### 13. vLLM and PagedAttention

Deep dive into vLLM's inference engine, focusing on PagedAttention, memory virtualization, and performance benefits of contiguous attention cache layout.

### 14. Inference Frameworks Overview

Overview and tradeoffs of major inference engines:

- **vLLM**: Token streaming, PagedAttention
- **TGI**: HuggingFace’s production server
- **DeepSpeed-MII**: Optimized model serving with parallelism
- **Triton Inference Server**: Multi-framework backend support

### 15. Serving LLMs with OpenAI API-style Interface

Guides building RESTful APIs with `completion` and `chat` endpoints compatible with OpenAI API format. Discusses request schemas, streaming support, and extensibility.

### 16. Async/Streaming Inference

Explains how to support streaming token outputs using Server-Sent Events (SSE), gRPC, or WebSocket. Covers scheduling async tasks and partial decode delivery.

### 17. Inference Latency and Throughput

Analyzes common bottlenecks in LLM inference. Discusses tuning hardware utilization, preemption, and profiling decode latency per token.

### 18. Logging & Monitoring

Covers structured logging, tracing per request, Prometheus/Grafana metrics, and alerting for failed requests or degraded performance.

### 19. Retries, Failures, and Recovery

Explores robust error handling: retry logic, circuit breakers, stateless recovery, and checkpointing state for long-context sessions.

### 20. Inference on Edge Devices

Techniques for running LLMs on edge hardware: pruning, quantization, distillation, and light transformer variants (e.g., TinyLLaMA, DistilGPT2).

### 21. Custom Kernel Integration

Instructions on writing and integrating custom CUDA/HIP kernels for inference. Covers operator registration, stream synchronization, and performance benchmarking.

### 22. Multi-Tenant Inference

Design principles for supporting multiple users/models in a single server. Discusses isolation, priority scheduling, memory fencing, and quota enforcement.
