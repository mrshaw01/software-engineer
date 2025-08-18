# Batching & Prefill vs. Decode

Efficient batching is critical to high-throughput LLM inference. In an autoregressive transformer model, inference is divided into two distinct phases: **prefill** and **decode**. Each has different computational characteristics and batching considerations.

## Prefill Phase

### Definition

The prefill phase processes the full input context (prompt) for a given request. It is equivalent to a forward pass over the entire input sequence up to the point where generation begins.

### Characteristics

- High arithmetic intensity (large number of tokens per request).
- Each token attends to all previous tokens in the prompt.
- **Attention complexity**: O(N²), where N is the prompt length.
- Key/Value (KV) caches are populated during this stage.

### Batching Strategy

- Group requests with similar prompt lengths to reduce padding waste.
- Schedule infrequent requests (e.g. long prompts) carefully to avoid blocking short ones.
- Prefill is **amortizable**: batching improves utilization significantly.
- Use static padding or dynamic padding to align inputs in a batch.

## Decode Phase

### Definition

The decode phase handles token-by-token generation. At each step, only the most recent token is processed while attending to the full cached context from prefill.

### Characteristics

- Low arithmetic intensity (1 token per step).
- Attention complexity: O(1) per new token, but memory-bound due to KV cache reads.
- KV cache is reused and appended with new key/value pairs.

### Batching Strategy

- Decode is **latency-critical** due to its token-by-token nature.
- Batch together requests that are actively generating tokens.
- Requires careful dynamic batching since generation length varies per request.
- Token outputs can diverge across batch members, introducing challenges in shared compute reuse.

## Joint Scheduling Considerations

Efficient inference requires separate batching logic for prefill and decode stages:

| Stage   | Primary Goal    | Optimal Batching                                  |
| ------- | --------------- | ------------------------------------------------- |
| Prefill | Throughput      | Static batching with length-based grouping        |
| Decode  | Latency & reuse | Dynamic batching with grouping by generation step |

### Key Techniques

- **Two-queue system**: One queue for prefill, one for decode, with independent schedulers.
- **KV Cache reuse**: Prefill populates cache, decode reuses it.
- **Prefill-Decode split kernels**: Specialized kernels for each stage improve hardware utilization.
- **Max concurrency**: Use asynchronous scheduling to keep both stages full.

## Example: Decode Bottleneck

In a system serving many short prompts and long generations, decode becomes the throughput limiter. Optimizations include:

- Grouped attention (for cache-friendly memory access).
- CUDA graphs for repeated decode steps.
- Token bucket grouping (e.g. vLLM’s scheduling).

## Summary

The separation of prefill and decode enables more efficient compute scheduling and batching. By tailoring strategies to each phase's characteristics, LLM serving systems can achieve significantly better throughput and latency trade-offs.
