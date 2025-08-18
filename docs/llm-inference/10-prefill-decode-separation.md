# Prefill/Decode Separation

## Overview

Autoregressive large language model (LLM) inference naturally splits into two execution phases:

1. **Prefill Phase**

   - The model processes the input prompt (context tokens) in parallel.
   - All tokens are embedded, projected through the transformer stack, and stored into the KV cache.
   - This phase is compute-intensive but benefits from full parallelization across the sequence length.

2. **Decode Phase**
   - The model generates tokens one at a time using cached past key/value states.
   - Each new token reuses the KV cache, drastically reducing computation but enforcing sequential execution.
   - This phase is latency-sensitive since each token requires a full forward pass through the transformer layers.

Separating prefill and decode paths is critical for optimizing inference performance, enabling better scheduling, batching, and system-level resource utilization.

## Motivation

Without explicit separation, inference engines often treat all forward passes uniformly, missing opportunities to:

- **Batch Efficiently**: Prefill requests can be grouped by sequence length, while decode requests are batched by active sessions.
- **Reduce Kernel Launch Overhead**: Dedicated kernel graphs for each phase minimize redundant scheduling work.
- **Improve Throughput**: Prefill-heavy workloads (e.g., summarization with long prompts) and decode-heavy workloads (e.g., chat completion) can be handled independently, avoiding resource contention.
- **Enable Advanced Features**: Techniques like speculative decoding and continuous batching rely on this separation to overlap prefill and decode efficiently.

## Execution Model

### Prefill Path

- Input: `N` tokens (entire prompt or appended context).
- Computation: Fully parallel attention and MLP across sequence.
- Output: KV cache initialized up to `N` positions + logits for the final token.

### Decode Path

- Input: Single token (or small micro-batch of generated tokens).
- Computation: Attention uses cached keys/values from prefill; only the new position is updated.
- Output: Logits for the next token + updated KV cache for one position.

## Implementation Strategies

1. **Separate Graph Compilation**

   - Capture CUDA/HIP graphs for prefill and decode independently.
   - Prefill graphs scale with sequence length; decode graphs are length-invariant.

2. **Dynamic Batching**

   - Prefill: Group requests with similar input lengths.
   - Decode: Merge active decoding requests across users into a single step.

3. **Scheduling Policies**

   - Run long prefill operations in parallel with short decode steps to prevent starvation.
   - Prioritize latency-sensitive decode requests over bulk prefill.

4. **Memory Management**
   - Allocate KV cache once during prefill, extend incrementally during decode.
   - Separate allocator pools may be used to avoid fragmentation between phases.

## Trade-offs

- **Pros**

  - Lower latency per token.
  - Better hardware utilization with distinct optimization per phase.
  - Enables speculative decoding and continuous batching.

- **Cons**
  - Increased complexity in scheduler and kernel management.
  - Requires careful memory planning to avoid overhead when switching between paths.
  - Prefill-decode imbalance may lead to idle resources if not scheduled properly.

## Example: Continuous Batching with Separation

1. Batch multiple prefill requests until a threshold is reached.
2. Immediately start decode for each finished request.
3. Interleave new prefill jobs with ongoing decode jobs.
4. Maintain high utilization by ensuring decodes (short kernels) are never starved by long prefills.

## Key Takeaways

- Prefill/Decode separation is a **core design principle** in modern LLM inference engines.
- It enables **latency reduction**, **throughput maximization**, and **system scalability**.
- Frameworks like **vLLM**, **TGI**, and **DeepSpeed-MII** implement explicit prefill/decode execution paths to support advanced scheduling and batching policies.
