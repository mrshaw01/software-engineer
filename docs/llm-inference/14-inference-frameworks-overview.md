# Inference Frameworks Overview

This section surveys widely used open-source LLM inference frameworks and servers, highlighting design choices, performance-oriented features, and practical deployment trade-offs. It is intended to help you select and operate the right stack for your latency, throughput, cost, and operability goals as of August 18, 2025.

Covered frameworks:

- vLLM (serving engine and API server)
- Hugging Face Text Generation Inference (TGI)
- DeepSpeed-MII (library + gRPC/REST serving)
- NVIDIA TensorRT-LLM (runtime/library)
- NVIDIA Triton Inference Server (multi-backend serving, including TensorRT-LLM backend)

## Selection criteria

When evaluating a serving stack, consider:

- **Workload shape:** prompt lengths, output lengths, concurrency, TTFT sensitivity, request variability.
- **Batching & scheduling:** continuous/in-flight batching, prefill–decode separation, chunked prefill.
- **KV cache strategy:** paging/virtualization, prefix sharing/reuse, eviction policy.
- **Parallelism:** tensor, pipeline, expert (for MoE), multi-instance, multinode.
- **Quantization & kernels:** INT4/INT8/FP8 support, FlashAttention/FlashInfer, fused kernels.
- **Model coverage:** dense vs. MoE, multimodal, embeddings, tool-call JSON modes.
- **APIs & ops:** OpenAI-compatible API, streaming, metrics, tracing, autoscaling friendliness.
- **Hardware targets:** NVIDIA, AMD, Intel, AWS Neuron/Trainium, TPU.
- **Ecosystem integration:** Triton backend, HF Transformers compatibility, K8s readiness.

## Quick recommendations

- **General-purpose, high-throughput, OpenAI-compatible REST:** choose **vLLM**. ([GitHub][1])
- **Hugging Face-centric deployments with multi-backend support and production telemetry:** choose **TGI**. ([Hugging Face][2])
- **NVIDIA GPUs, lowest latency with aggressive graph-level optimizations and FP8/INT8:** build with **TensorRT-LLM**, usually served via **Triton**. ([NVIDIA Docs][3], [NVIDIA GitHub][4])
- **DeepSpeed kernels and parallelism, Python-first control with persistent gRPC server:** consider **DeepSpeed-MII**. ([GitHub][5])

## Feature matrix (snapshot)

| Capability   | vLLM                                 | TGI                                     | DeepSpeed-MII               | TensorRT-LLM                         | Triton Server                          |
| ------------ | ------------------------------------ | --------------------------------------- | --------------------------- | ------------------------------------ | -------------------------------------- |
| API server   | OpenAI-compatible HTTP               | HTTP SSE + REST                         | gRPC + optional REST        | Library/runtime                      | Multi-backend HTTP/gRPC                |
| Batching     | Continuous batching                  | Continuous batching                     | Continuous batching         | In-flight fused batching via backend | In-flight batching (TRT-LLM backend)   |
| KV cache     | PagedAttention, prefix caching       | PagedAttention (leverages vLLM kernels) | Blocked KV cache            | Paged KV cache with reuse            | Backend exposes KV controls            |
| Quantization | INT4/INT8/FP8 (various methods)      | bitsandbytes, GPTQ                      | Mixed-precision, DS kernels | SmoothQuant/AWQ/INT8/FP8             | Depends on backend (TensorRT-LLM etc.) |
| Parallelism  | Tensor, pipeline, expert             | Tensor parallel                         | Tensor parallel, replicas   | Tensor, pipeline, expert             | Model instances, MIG, multinode        |
| Streaming    | Yes                                  | SSE                                     | Yes                         | Yes                                  | Yes                                    |
| Hardware     | NVIDIA, AMD, Intel, CPU, TPU, Neuron | NVIDIA, AMD, Gaudi, Neuron, TPU         | NVIDIA (primary)            | NVIDIA                               | NVIDIA-optimized (plus other backends) |

References: vLLM features and API server; TGI features incl. PagedAttention and SSE; MII features; TensorRT-LLM backend capabilities. ([GitHub][1], [Hugging Face][2], [NVIDIA Docs][3])

## Framework deep dives

### vLLM

**What it is.** A high-throughput, memory-efficient inference and serving engine featuring PagedAttention, continuous batching, and an OpenAI-compatible API server. Supports quantization (INT4/INT8/FP8 via GPTQ/AWQ/AutoRound), speculative decoding, chunked prefill, prefix caching, and multiple hardware backends. ([GitHub][1])

**When to use.**

- Mixed, spiky traffic with variable prompt lengths.
- Drop-in OpenAI API compatibility with strong throughput.
- Broad model coverage (dense, MoE, multimodal) on common GPUs. ([GitHub][1])

**Quick start (single GPU).**

```bash
pip install vllm
# OpenAI-compatible server
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3-8B \
  --max-model-len 8192
```

**Notes.** Continuous batching + paged KV yields strong throughput at high concurrency; simple operational model for single-tenant or multi-tenant deployments. ([GitHub][1])

### Hugging Face Text Generation Inference (TGI)

**What it is.** A production-grade LLM server with continuous batching, SSE streaming, tensor parallelism, metrics/tracing, and multi-backend support (incl. TensorRT-LLM, Gaudi, Neuron, AMD, TPU). Implements PagedAttention and FlashAttention; supports quantization (bitsandbytes, GPTQ) and structured generation tools. ([Hugging Face][2])

**When to use.**

- Hugging Face-native stack and model operations.
- Need built-in observability (Prometheus, OTEL) and multi-backend portability. ([Hugging Face][2])

**Quick start (Docker).**

```bash
docker run --gpus all --rm -p 8080:80 \
  -e MODEL_ID=meta-llama/Meta-Llama-3-8B \
  ghcr.io/huggingface/text-generation-inference:latest
# Stream tokens:
curl -N -X POST http://localhost:8080/generate_stream \
  -d '{"inputs":"Hello","parameters":{"max_new_tokens":64}}'
```

Feature references: SSE, tensor parallelism, PagedAttention/FlashAttention, quantization. ([Hugging Face][2])

### DeepSpeed-MII

**What it is.** A Python library that wraps DeepSpeed-Inference/DeepSpeed-Kernels to deliver high-throughput, low-latency text generation. Provides blocked KV cache, continuous batching, Dynamic SplitFuse, tensor parallelism, and persistent serving via gRPC/REST. ([GitHub][5])

**When to use.**

- Preference for DeepSpeed kernels and easy Python control.
- Multi-GPU tensor parallel with simple launch semantics. ([GitHub][5])

**Examples.**

```python
# Non-persistent pipeline
import mii
pipe = mii.pipeline("mistralai/Mistral-7B-v0.1")
print(pipe(["DeepSpeed is"], max_new_tokens=64))
```

```python
# Persistent server with REST
client = mii.serve("mistralai/Mistral-7B-v0.1",
                   deployment_name="mistral",
                   tensor_parallel=2,
                   enable_restful_api=True,
                   restful_api_port=28080)
```

Features and examples per MII docs. Note that MII maintainers report large throughput gains vs. alternatives depending on model/hardware; validate for your workload. ([GitHub][5])

### NVIDIA TensorRT-LLM

**What it is.** A high-performance LLM runtime and kernel suite for NVIDIA GPUs with aggressive graph/kernal optimizations, paged KV cache and early/kv-reuse, quantization (INT8/FP8), and advanced decoding (Medusa, lookahead) support. Typically served via Triton with the TensorRT-LLM backend. ([NVIDIA Docs][3], [NVIDIA GitHub][4], [NVIDIA Developer][6])

**When to use.**

- NVIDIA GPUs with stringent latency/throughput targets.
- Willing to prebuild engines and tune backend parameters. ([NVIDIA Docs][3])

**Build and serve (abbreviated).**

```bash
# Build engines (example flags vary by model)
trtllm-build --checkpoint_dir ./c-model/llama3/8b/fp8/2-gpu \
  --kv_cache_type paged --gemm_plugin float8_e4m3 --output_dir /engines/llama3/8b

# Serve with Triton TRT-LLM backend container
docker run --gpus all --net host -v /engines:/engines \
  nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3
```

Backend quick-start and KV reuse references. ([NVIDIA Docs][3], [NVIDIA GitHub][7])

### NVIDIA Triton Inference Server

**What it is.** A production inference server that supports multiple backends (TensorRT-LLM, PyTorch, Python, ONNX, etc.), dynamic/in-flight batching, model repositories, versioning, metrics, and multinode. For LLMs, pair with the **TensorRT-LLM backend** to get in-flight fused batching, paged attention, and advanced scheduling. ([NVIDIA Docs][3])

**Model repository configuration (excerpt).**

```proto
# models/llama3/1/config.pbtxt (TensorRT-LLM backend)
backend: "tensorrtllm"
max_batch_size: 256
parameters: {
  key: "batching_strategy" value: { string_value: "inflight_fused_batching" }
}
parameters: { key: "engine_dir" value: { string_value: "/engines/llama3/8b" } }
parameters: { key: "decoupled_mode" value: { string_value: "true" } }
```

Configuration knobs and scheduling/parallelism references. ([NVIDIA Docs][8])

## Practical decision guide

1. **If you need an OpenAI-compatible endpoint with minimal friction** and strong throughput across diverse models/hardware, start with **vLLM**. ([GitHub][1])
2. **If you operate heavily in Hugging Face tooling** and want built-in metrics, tracing, and multi-backend portability, use **TGI**. ([Hugging Face][2])
3. **If you target NVIDIA GPUs and can precompile engines** to maximize latency/throughput and TTFT, deploy **TensorRT-LLM** via **Triton**. ([NVIDIA Docs][3])
4. **If you prefer DeepSpeed’s kernel stack and Python control**, evaluate **MII**; benchmark against vLLM/TensorRT-LLM for your specific mix of sequence lengths and concurrency. ([GitHub][5])

## Benchmarking guidance

- **Measure prefill vs. decode:** report tokens/s for both phases; test multiple prompt/output length pairs.
- **Sweep concurrency:** observe scheduler behavior (continuous vs. in-flight) and TTFT tail latencies.
- **Profile KV memory:** verify cache block sizing, reuse, and eviction (paged/blocked KV settings matter). ([NVIDIA Developer][6])
- **Quantization sweeps:** compare accuracy/latency for INT8/FP8/INT4 across frameworks; ensure calibration or weight-only methods are appropriate for your model stack. ([GitHub][1], [Hugging Face][2], [NVIDIA GitHub][4])

## Deployment patterns

- **Single-tenant, single model:** vLLM/TGI standalone; or Triton+TRT-LLM for NVIDIA.
- **Multi-tenant router:** front with a lightweight API gateway; shard by model across vLLM/TGI pods or Triton instances; enable autoscaling on queue depth/latency.
- **Large models (70B+) on multi-GPU:** prefer tensor/pipeline parallel configs (vLLM TP/PP, MII TP, TRT-LLM TP/PP). ([GitHub][1], [NVIDIA Docs][3])
- **Observability:** enable Prometheus/OTEL in TGI; Triton metrics endpoints; add per-request structured logs in custom gateways. ([Hugging Face][2], [NVIDIA Docs][3])

## Minimal recipes

### vLLM (OpenAI-compatible)

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server --model mistralai/Mixtral-8x7B-Instruct-v0.1
```

([GitHub][1])

### TGI (Docker)

```bash
docker run --gpus all -p 8080:80 \
  -e MODEL_ID=mistralai/Mixtral-8x7B-Instruct-v0.1 \
  ghcr.io/huggingface/text-generation-inference:latest
```

([Hugging Face][2])

### Triton + TensorRT-LLM (Docker + engines)

```bash
docker run --gpus all --net host -v /engines:/engines \
  nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3
# Configure config.pbtxt with inflight_fused_batching and engine_dir
```

([NVIDIA Docs][3])

## Closing notes

- **Always benchmark on your own prompts and concurrency profile.** Relative performance claims vary with sequence lengths, batching windows, and quantization choices.
- **Keep versions aligned.** Especially for Triton + TensorRT-LLM, match backend and engine versions per the support matrix. ([NVIDIA Docs][3])

**Further reading:** vLLM repo (features, API), TGI docs (features, PagedAttention), DeepSpeed-MII repo (features, examples), TensorRT-LLM docs (KV reuse), Triton TensorRT-LLM backend docs (configuration, scheduling). ([GitHub][1], [Hugging Face][2], [NVIDIA GitHub][7], [NVIDIA Docs][3])

[1]: https://github.com/vllm-project/vllm "GitHub - vllm-project/vllm: A high-throughput and memory-efficient inference and serving engine for LLMs"
[2]: https://huggingface.co/docs/text-generation-inference/en/index "Text Generation Inference"
[3]: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tensorrtllm_backend/README.html "TensorRT-LLM Backend — NVIDIA Triton Inference Server"
[4]: https://nvidia.github.io/TensorRT-LLM/?utm_source=chatgpt.com "Welcome to TensorRT-LLM's Documentation! - GitHub Pages"
[5]: https://github.com/deepspeedai/DeepSpeed-MII "GitHub - deepspeedai/DeepSpeed-MII: MII makes low-latency and high-throughput inference possible, powered by DeepSpeed."
[6]: https://developer.nvidia.com/blog/5x-faster-time-to-first-token-with-nvidia-tensorrt-llm-kv-cache-early-reuse/?utm_source=chatgpt.com "5x Faster Time to First Token with NVIDIA TensorRT-LLM ..."
[7]: https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html?utm_source=chatgpt.com "KV cache reuse — TensorRT-LLM - GitHub Pages"
[8]: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tensorrtllm_backend/docs/model_config.html "Model Configuration — NVIDIA Triton Inference Server"
