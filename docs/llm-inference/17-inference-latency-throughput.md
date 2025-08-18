# Inference Latency and Throughput

This section defines latency and throughput for autoregressive LLM inference, provides a practical performance model, and outlines concrete techniques, scheduling policies, and measurement methods to optimize both metrics in production systems. The math is written to render cleanly on GitHub using MathJax.

## 1. Definitions and Core Metrics

- **End-to-End Latency (E2E)**: Time from request arrival to final token delivered.

  $$T_{\mathrm{E2E}}=T_{\mathrm{queue}} + T_{\mathrm{batch}} + T_{\mathrm{prefill}} + T_{\mathrm{decode}} + T_{\mathrm{post}} + T_{\mathrm{network}}$$

- **Time-To-First-Token (TTFT)**: Time from request arrival to the first generated token sent to the client. Dominated by queuing, batching window, and prefill.
- **Time-Per-Output-Token (TPOT)**: Average time per generated token during decode (ms/token).
- **Time-To-Last-Token (TTLT)**: Time from request arrival to the last token delivered.

  $$T_{\mathrm{TTLT}} \approx T_{\mathrm{TTFT}} + N_{\mathrm{new}} \cdot \mathrm{TPOT}$$

- **Throughput (tokens/s)**:

  - Device throughput: tokens/s on a single accelerator.
  - System throughput: tokens/s across a cluster/region.

    $$\mathrm{Throughput} = \frac{\text{Total output tokens}}{\text{Wall time}}$$

- **Tail Latency**: P90/P95/P99 for TTFT and E2E; critical for interactive workloads.
- **SLOs**: Service-level objectives on TTFT and TTLT; often different for interactive vs. batch modes.

## 2. Prefill vs. Decode: Cost Structure

Let \$L\$ be the context length (prompt + generated so far), \$d\$ the model dimension, and \$H\$ the number of attention heads.

- **Prefill (no KV reuse yet)**: Complexity \$\sim O(L^2 \cdot d)\$. Compute- and memory-intensive; benefits from FlashAttention-class kernels and large GEMMs.
- **Decode with KV cache**: Complexity per token \$\sim O(L \cdot d)\$. Often **memory-bandwidth bound** due to KV reads.

**Rule of thumb**: Prefill dominates **TTFT**; per-token decode dominates **TPOT** and thus **TTLT**.

## 3. Practical Performance Model

### 3.1 Latency Decomposition

- **Queueing**: Router/dispatcher delay from arrival to admission.
- **Batching window**: Micro-batching delay to form larger GEMMs (e.g., 2–10 ms interactive; 20–50 ms batch).
- **Prefill**: One or more forward passes over the input context.
- **Decode**: Token-by-token loop with KV reads/writes.
- **Post/Network**: Detokenization, serialization, network send.

### 3.2 Decode Step Cost (memory-bound approximation)

Define:

- \$n\_{\mathrm{layers}}\$: number of transformer layers
- \$H\_{\mathrm{KV}}\$: number of KV heads (can be \$< H\$ with MQA/GQA)
- \$D\_{\mathrm{head}}\$: head dimension
- \$\mathrm{BPE}\$: bytes per element (e.g., 2 for FP16, 1 for FP8)

Per generated token, bytes moved from KV cache:

$$\mathrm{Bytes/token} \approx 2 \cdot L \cdot n_{\mathrm{layers}} \cdot H_{\mathrm{KV}} \cdot D_{\mathrm{head}} \cdot \mathrm{BPE}$$

If bandwidth-bound:

$$\mathrm{TPOT} \approx \frac{\mathrm{Bytes/token}}{\mathrm{Effective\ Memory\ BW}} \quad\Rightarrow\quad \mathrm{Throughput} \approx \frac{1}{\mathrm{TPOT}} \ \text{tokens/s} $$

### 3.3 KV Cache Footprint

Per-sequence token storage:

$$\mathrm{KV\ bytes/token} \approx 2 \cdot n_{\mathrm{layers}} \cdot H_{\mathrm{KV}} \cdot D_{\mathrm{head}} \cdot \mathrm{BPE} $$

This bound drives max concurrent tokens per device and informs eviction/compaction policies.

## 4. Throughput–Latency Trade-offs

| Knob                 | Effect on Latency              | Effect on Throughput           | Notes                                                               |
| -------------------- | ------------------------------ | ------------------------------ | ------------------------------------------------------------------- |
| Batch sizing         | Increases TTFT (larger window) | Increases tokens/s             | Use adaptive windows per class.                                     |
| Continuous batching  | Reduces queueing; smooth TTFT  | Increases tokens/s             | Admit mid-generation; requires paged KV and schedule-aware kernels. |
| Max context length   | Increases prefill (quadratic)  | — / Decreases                  | Chunked prefill mitigates.                                          |
| Quantization         | Lowers TTFT/TPOT               | Increases tokens/s             | INT8/FP8 weights; low-precision KV helps decode.                    |
| GQA/MQA              | Lowers TPOT                    | Increases tokens/s             | Reduces \$H\_{\mathrm{KV}}\$.                                       |
| Speculative decoding | Lowers TPOT/TTLT               | Increases tokens/s             | Gains depend on acceptance rate and draft–target ratio.             |
| CUDA/HIP Graphs      | Lowers launch overhead         | Increases tokens/s             | Impactful in the token loop.                                        |
| Prompt caching       | Lowers TTFT                    | Increases tokens/s for repeats | Needs robust prefix hashing/virtualization.                         |

## 5. Techniques to Reduce TTFT (Prefill-Focused)

1. FlashAttention-class kernels for prefill; ensure support for long sequences and masks.
2. Chunked prefill (e.g., 256–1024-token chunks) to enable early streaming and overlap compute with I/O.
3. Prompt caching / prefix reuse with stable hashing; avoid re-encoding and re-computing KV for common system/chat prefixes.
4. Weight-only quantization (INT8/FP8) with high-performance GEMM backends (cuBLASLt/rocBLAS/Triton).
5. CUDA/HIP Graph capture for prefill to reduce CPU launch overheads.
6. Tokenizer acceleration: parallel tokenization, precomputed merges, zero-copy input staging.
7. Host–device overlap: pinned-memory H2D copies overlapped with compute; double-buffered input staging.

## 6. Techniques to Reduce TPOT (Decode-Focused)

1. **Paged KV + contiguous block layout** to minimize TLB misses and enable coalesced loads.
2. **Low-precision KV** (FP8/NF4/INT8 with proper scaling) to reduce bandwidth and footprint.
3. **GQA/MQA** to shrink \$H\_{\mathrm{KV}}\$ with minimal quality loss.
4. **Fused and persistent decode kernels**: combine QKV, rotary, attention score/softmax, and value gather where feasible.
5. **CUDA/HIP Graphs** for the per-token loop; keep per-step CPU work near zero.
6. **On-device sampling** (softmax, top-k/p, temperature, RNG) to avoid CPU–GPU sync.
7. **Speculative decoding**: choose a draft \$M_d\$ and target \$M_t\$; tune proposed length \$k\$ by acceptance rate and stall risk.
8. **Attention sinks / cache warmup** for stability on very long contexts.
9. **KV reuse across beams** (if beam search) and **prefix-batching** across requests.

## 7. Scheduling for Latency SLOs and High Utilization

- **Admission control**: classify requests (interactive, batch, offline). Use separate queues and limits.
- **Micro-batching windows**: adaptive by class (e.g., 2–5 ms interactive; 10–40 ms batch).
- **Continuous batching**: admit new tokens each step; requires re-indexable KV and token-level schedulers.
- **Fairness and preemption**: avoid head-of-line blocking by chunking long prefills, SRPT-like heuristics for interactive classes, and preempting oversized prompts to background queues.
- **Max tokens/request** and **rate limits**: enforce hard caps to protect tail latency.
- **Multi-tenant quotas**: per-tenant token buckets to prevent saturation.
- **Placement & routing**: route by model size, quant level, and accelerator class; co-locate hot prefixes to improve cache hit rate.

## 8. Memory Management for KV Cache

- **Virtualized KV (paged attention)**: fixed-size pages; enables growth, compaction, and mid-generation admission.
- **Eviction**: age-based/LRU per session; evict completed/idle sessions first.
- **Compaction**: asynchronous defragmentation when free contiguous pages are low.
- **Pools**: separate by dtype (FP8/FP16), sequence class, and page size.
- **Telemetry**: track “KV bytes in use,” “free pages,” “compaction debt,” and “eviction rate.”

## 9. Parallelism and Its Impact

- **Tensor parallel (TP)**: reduces per-GPU memory; adds all-reduce on critical paths. For small batches, communication can hurt TTFT/TPOT—minimize TP for interactive.
- **Pipeline parallel (PP)**: increases latency due to bubbles; use larger micro-batches or interleaving; not ideal for low-latency interactive paths unless tuned carefully.
- **Sequence parallel**: partitions the sequence dimension; useful for very long contexts; ensure kernels support it.
- **Speculative across devices**: keep draft and target co-located or on high-bandwidth interconnects (NVLink/XGMI).

## 10. Measurement and Instrumentation

### 10.1 What to Measure

- **Per-request**: arrival time, admission time, TTFT, tokens generated, TTLT, errors.
- **Per-phase**: tokenizer time, prefill matmul/attention time, decode step time, detokenization time.
- **Per-token**: distribution of decode step times (mean, P95).
- **Device**: SM occupancy/active cycles, DRAM/HBM bandwidth, L2 hit rate, PCIe/NVLink/XGMI throughput.
- **Memory**: KV usage, page faults/migrations, allocator fragmentation.
- **Scheduler**: batch sizes per step, window durations, drop/evict events.

### 10.2 Tooling

- GPU profilers (Nsight Systems/Compute, rocprof/Omnitrace).
- Framework profilers (PyTorch Profiler with CUDA/HIP events).
- Tracing: per-request spans with unique IDs propagating router → worker → kernels.
- Dashboards: Prometheus/Grafana panels for TTFT/TPOT/TTLT P50/P95, tokens/s, KV bytes, batch window, speculative acceptance rate.

### 10.3 Load Testing

- Arrival processes: Poisson for interactive; bursty for chat; sustained for batch.
- Mixes: short/long prompts, varying `max_new_tokens`; include adversarial long contexts.
- Steady-state vs. warmup: capture both; graph-capture warmup can distort early samples.

## 11. Tuning Playbook (Step-by-Step)

1. Establish baselines: single-request TTFT/TPOT, device tokens/s, KV footprint per token.
2. Enable high-performance kernels: FlashAttention for prefill; fused decode kernels.
3. Capture graphs (prefill and decode).
4. Host–device pipeline: pinned memory, async H2D, batch tokenization.
5. Quantize weights (INT8/FP8) and evaluate low-precision KV.
6. Adopt continuous batching with a small interactive window (start at 3–5 ms).
7. Introduce GQA/MQA to reduce KV bandwidth.
8. Add speculative decoding; tune proposal length \$k\$ by acceptance rate.
9. Implement prompt caching for common prefixes and system prompts.
10. Right-size parallelism: minimize TP for interactive; reserve TP/PP for batch endpoints.
11. SLO-aware routing: separate interactive and batch queues; enforce per-tenant budgets.
12. Iterate with telemetry: monitor TTFT P95 and memory bandwidth; adjust windows, batch caps, and eviction thresholds.

## 12. Recommended Defaults (Starting Points)

- **Interactive endpoint**: batch window 2–5 ms; continuous batching on; `max_new_tokens` 256–512; quantized weights; GQA/MQA if quality permits; graph-captured decode; on-device sampling.
- **Batch endpoint**: batch window 20–40 ms; larger token caps; aggressive prompt caching; TP/PP as needed for large models.
- **Memory**: KV pages sized for cache-friendliness (e.g., 16–32 KB pages), with background compaction.

## 13. Cost and Efficiency

Track **tokens/sec/watt** and **cost per 1M tokens**:

$$\mathrm{Cost/1M\ tok} = \frac{\mathrm{GPU\ hour\ price}}{3600} \cdot \frac{10^6}{\mathrm{tokens/s}}$$

Use this to compare quantization, speculative decoding, and hardware classes. Favor configurations maximizing tokens/s at acceptable quality and SLOs.

## 14. Common Pitfalls

- Excessive TP on interactive paths causing high per-token all-reduce latency.
- CPU–GPU synchronization for sampling/softmax.
- Overly large batch windows harming TTFT P95.
- KV fragmentation leading to admission stalls; lack of compaction.
- Ignoring tokenizer/detokenizer costs for short requests.
- Speculative decoding with low acceptance from a draft–target mismatch.

## 15. Checklist

- [ ] Baseline TTFT/TPOT/TTLT with percentiles.
- [ ] FlashAttention (prefill) and fused decode kernels validated.
- [ ] CUDA/HIP Graphs captured for prefill and decode.
- [ ] Continuous batching enabled with class-specific windows.
- [ ] Prompt cache enabled and hit-rate monitored.
- [ ] KV paging, eviction, compaction metrics live.
- [ ] Quantization and GQA/MQA evaluated vs. quality.
- [ ] Speculative decoding tuned; acceptance rate tracked.
- [ ] SLO-aware routing and per-tenant limits enforced.
- [ ] Dashboards and traces in place for on-call.
