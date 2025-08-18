# vLLM and PagedAttention

vLLM is a high-throughput LLM inference engine designed around **continuous batching**, **prefill/decode separation**, and a custom memory system called **PagedAttention**. PagedAttention virtualizes the KV cache into fixed-size blocks (“pages”) to eliminate fragmentation and enable efficient, dynamic batching of heterogeneous requests without excessive copying.

This section explains the design, trade-offs, and practical tuning of vLLM and PagedAttention for production-grade serving.

## 1) Why PagedAttention?

**The problem.** Naïve KV allocation is proportional to `O(total_generated_tokens)` for each request and typically demands _contiguous_ GPU memory. Under dynamic batching (requests with different lengths/arrival times), you quickly suffer from:

- Internal fragmentation (holes between sequences),
- Costly compaction/copy when sequences grow,
- Poor GPU memory utilization → fewer concurrent requests → lower throughput.

**The idea.** PagedAttention slices each sequence’s KV cache into **fixed-size blocks (pages)** and tracks them via per-sequence **block tables**. Attention kernels gather keys/values by **following the block table** rather than assuming physical contiguity. This gives:

- **Virtual contiguity** for the kernel with **non-contiguous physical placement**,
- Near-zero-cost growth/shrink via **page (de)allocation**,
- Stable, high utilization under continuous batching and preemption.

## 2) KV Cache Sizing & Math

Let:

- `L` = number of layers,
- `H_kv` = number of KV heads (for MQA/GQA, `H_kv ≤ H_q`),
- `D` = head dimension,
- `b` = bytes per element in KV cache (e.g., fp16=2, fp8≈1),
- `T` = tokens per sequence,
- `N` = concurrent sequences.

**Per-token KV bytes (typical):**

```
KV_per_token = 2 * L * H_kv * D * b
               ^   ^   ^     ^   ^
               |   |   |     |   └— dtype size
               |   |   |     └——— head dim
               |   |   └——————— KV heads
               |   └——————— layers
               └——————— 2 for K and V
```

**Total KV across N sequences:**

```
KV_total ≈ Σ_i (KV_per_token * T_i)
        ≤  N * KV_per_token * T_max
```

**PagedAttention adds page metadata**, but overhead is small (block table entries and allocator bookkeeping).

> Rule of thumb: before going live, compute `KV_total` for your **peak T_max** and **worst-case concurrency**; ensure it fits into `gpu_memory_utilization * VRAM` after weights/activations/cudagraphs/reserve.

## 3) Page Size, Block Tables, and the Allocator

- **Page (block) size**: fixed token count per page (e.g., 16/32/64/128 tokens). Larger pages → fewer block table entries and better kernel coalescing; smaller pages → finer-grained allocation and less internal waste.
- **Block table**: per-sequence vector of page IDs (plus offsets) used by attention kernels to locate K/V for each token position.
- **Allocator**:

  - Maintains free/used page pools per device.
  - On new tokens, **append pages**; when sequences finish, **return pages**.
  - Can **migrate** sequences across GPUs only if combined with orchestration and (optionally) CPU offload; otherwise, migration is typically avoided during decoding.

**Selecting page size**

- Short/variable prompts with heavy **prefill**: medium pages (e.g., 32–64) balance locality vs waste.
- Long contexts with stable growth: larger pages (e.g., 128) reduce block-table overhead and kernel indirections.
- Extremely bursty traffic with tiny generations: smaller pages (e.g., 16–32) can cut allocator waste.

## 4) Kernels: How PagedAttention Works

**Attention read:** Instead of a single contiguous `[T, H_kv, D]` KV region, the kernel:

1. Reads the block table for token ranges,
2. Issues gathers over the mapped pages,
3. Fuses masking/rotary and matmul steps (often with FlashAttention-style tiling).

**Performance notes**

- Additional **indirection** from block tables is amortized by better GPU occupancy from larger batches.
- Fused kernels (e.g., attention matmuls + softmax) and **tensor-core friendly layouts** remain crucial.
- **CUDA/HIP Graphs** substantially reduce per-step launch overhead in the decode loop; vLLM leverages stable shapes per step (especially for decode) to capture graphs.

## 5) Scheduling & Continuous Batching

vLLM’s scheduler:

- Splits work into **prefill** (context ingestion) and **decode** (token-by-token),
- Uses **continuous batching**: joins newly arrived requests at safe boundaries,
- Supports **preemption** and **iteration-level fairness** to prevent starvation,
- Enforces **max tokens per step** to control step time.

**Chunked Prefill.** Prefill is **chunked** to:

- Stream long prompts into the batch without monopolizing the device,
- Increase overlap with other requests’ decodes,
- Keep step time bounded to meet latency SLOs.

**Throughput vs latency**

- Larger per-step token budgets → better throughput, higher tail latency,
- Smaller budgets → lower latency, lower peak throughput. Tune to your SLOs.

## 6) Prefix Reuse and Caching

- **Prefix (prompt) caching**: If two requests share an identical prefix, vLLM can **re-point block tables** to the same pages (copy-on-write semantics once divergence happens). This yields large savings in both KV memory and prefill compute.
- **Warm cache** (popular system prompts, instruction templates) lifts steady-state QPS significantly.

**Caveat:** Requires **exact token match** to reuse; upstream normalization/tokenization consistency matters.

## 7) Parallelism & Multi-Model

- **Tensor Parallel (TP):** Common for large models; KV pages are sharded per TP rank. Align page size and attention layouts with TP partitioning to avoid cross-rank gathers.
- **Pipeline Parallel (PP):** Less common at inference for decoder-only models due to microbatching overheads; can be combined with TP for very large checkpoints.
- **Speculative Decoding:** vLLM can integrate draft/target models; PagedAttention helps manage extra KV created/discarded during verification/rollback.
- **LoRA/PEFT Adapters:** vLLM supports multi-LoRA serving via adapter weights cached in GPU/CPU. KV remains per-request; only tiny adapter weights vary per request.

## 8) Practical Tuning Guide

1. **Memory budgeting**

   - Compute `KV_total` with your **peak sequence lengths** and **expected concurrency**.
   - Reserve headroom for **weights**, **CUDA/HIP graphs**, and **temporary activations**.
   - Consider **FP8/FP16 KV** (if kernel path supports it) to double capacity with minimal quality loss.

2. **Page size**

   - Start with **32–64 tokens/page** for mixed workloads.
   - Profile allocator waste (free pages vs used) and **kernel time per token**. Move up (128) if block indirection dominates; move down (16–32) if internal waste is high.

3. **Step token budget**

   - Cap **prefill chunk size** and **decode tokens per step** to hit latency SLOs.
   - Keep step time **< \~10–25 ms** for responsive streaming; increase if batch is compute-bound and SLOs allow.

4. **Batch width limits**

   - Set `max_num_seqs` / `max_tokens_in_batch` to keep kernel tiles saturated without spilling memory.
   - Prefer **token-based** caps over sequence count caps when requests are length-skewed.

5. **Prefix caching**

   - Normalize prompts upstream and **deduplicate** popular prefixes (system prompt, role headers).
   - Pin hot prefixes at startup to pre-warm the cache.

6. **GPU graphs & streams**

   - Enable CUDA/HIP Graphs for the **decode loop**.
   - Use separate streams for **H2D prefill copies**, **compute**, and **D2H sampling** if supported by your stack.

7. **Sampling**

   - Minimize host<->device sync: sample on device where possible.
   - For **top-k/p + temperature**, fuse sampling into logits kernel or keep logits on-device.

8. **Observability**

   - Track per-step **tokens/s (prefill/decode)**, **allocator occupancy**, **page churn**, **tail latencies (p95/p99)**, and **GPU utilization**.
   - Alert on **page exhaustion** and **graph recapture** rates (indicative of shape volatility).

## 9) Operational Playbooks

**Cold start**

- Load weights, prime a small batch to capture graphs, allocate a seed pool of pages, and optionally **pre-build common prefixes**.

**Brownouts (memory pressure)**

- Lower `max_tokens_in_batch`, **shrink page reserve**, and/or **offload KV** for long-idling sessions (if supported).
- Apply **length caps** or **early-stop** policies under overload.

**Model mixes**

- For multi-model fleets, separate **heaviest models** on dedicated GPUs.
- Share GPUs among lighter models only if **distinct page pools** and **scheduler fairness** keep SLOs.

## 10) Example Commands and Config Snippets

### vLLM HTTP server (single GPU)

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3-8B-Instruct \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --max-num-seqs 256 \
  --kv-cache-dtype fp16 \
  --enforce-eager false
```

### Python API (streaming with batching hints)

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3-8B-Instruct",
    tensor_parallel_size=1,
    max_model_len=8192,
    gpu_memory_utilization=0.90,
    kv_cache_dtype="fp16",
)

sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)
prompts = [
    "Summarize the paper in 5 bullets.",
    "Write a unit test for this function:\n..."]
outs = llm.generate(prompts, sampling_params=sp, use_tqdm=False)

for o in outs:
    print(o.outputs[0].text)
```

### Tuning via env/config (illustrative)

```bash
# Favor lower tail latency
export VLLM_MAX_TOKENS_PER_STEP=512        # smaller step window
export VLLM_PREFILL_CHUNK_SIZE=512

# Favor throughput (monitor p95!)
export VLLM_MAX_TOKENS_PER_STEP=2048
export VLLM_PREFILL_CHUNK_SIZE=2048

# Page size (if exposed in your build/branch)
export VLLM_KV_BLOCK_SIZE=64
```

> Exact flags and names vary by vLLM version. Align with your deployed release.

## 11) Benchmarking Checklist

- **Warmup** at least 1–2 minutes to stabilize graphs and allocator.
- Test **mixed prompts**: short, medium, long; mixed **max_new_tokens**.
- Record:

  - `tokens/s` (prefill and decode separately),
  - `p50/p95/p99` first-token latency and time-to-last-token,
  - GPU `sm`, `dram`, and **mem alloc/free rates**,
  - **Page occupancy** and **free list** dynamics.

- Stress **burst arrivals** and **long-tail prompts**; verify no pathological page churn.

## 12) Trade-offs vs Alternatives

- **vLLM (PagedAttention)**: Excellent under heterogeneous, high-churn traffic due to virtualized KV; minimal copying; great for multi-tenant continuous batching.
- **TGI / HF Text Generation Inference**: Mature production features; typically strong with tensor/pp parallelism; may need more care to achieve similar KV fragmentation resilience.
- **DeepSpeed-MII / Custom Runtimes**: Flexible parallelism, ZeRO-style optimizations; KV virtualization varies by stack; often requires more engineering to match vLLM’s dynamic batching ergonomics.

## 13) Common Pitfalls

- **Underestimating KV memory** at target concurrency/lengths.
- **Overly large step sizes** that inflate p95/p99 latency.
- **Mismatched tokenization** breaks prefix cache reuse.
- **Too-small page size** causing allocator overhead; **too-large** causing wasted space.
- **Frequent graph recapture** from shape volatility (e.g., toggling logits processors).

## 14) Summary

PagedAttention turns the KV cache into a **virtual, pageable address space**, letting attention kernels treat fragmented physical memory as logically contiguous. Combined with **continuous batching**, **chunked prefill**, and **GPU graphs**, vLLM reaches high throughput while maintaining predictable latency under real-world, mixed workloads. Correct **page sizing**, **token-per-step budgeting**, and **prefix caching** are the main knobs to hit your SLOs.
