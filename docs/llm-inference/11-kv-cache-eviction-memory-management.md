# KV Cache Eviction & Memory Management

## Overview

The Key-Value (KV) cache is central to efficient autoregressive inference in large language models. It stores the hidden representations (keys and values) of all previously processed tokens, avoiding redundant recomputation in subsequent decoding steps. However, as context lengths increase into tens or hundreds of thousands of tokens, KV cache memory grows linearly and can become the primary bottleneck. Efficient eviction and memory management strategies are therefore required to maintain both performance and scalability.

## KV Cache Growth

For a transformer model, each attention head stores:

- **Keys**: `[sequence_length, head_dim]`
- **Values**: `[sequence_length, head_dim]`

Across all layers and heads, memory usage is:

$Memory ≈ num\_layers × num\_heads × sequence\_length × head\_dim × sizeof(dtype)$

For large models, this can quickly exceed GPU memory limits, especially with FP16 precision.

## Eviction Strategies

### 1. Least Recently Used (LRU)

- Evict tokens that have not been attended to in recent steps.
- Pros: Straightforward implementation.
- Cons: Lacks semantic awareness; may remove important context.

### 2. Age-Based / Sliding Window

- Retain only the most recent `W` tokens (window size).
- Pros: Predictable memory usage, simple to manage.
- Cons: Long-range dependencies outside the window are lost.

### 3. Importance-Based

- Selective eviction of tokens with low attention contribution (measured via attention weights or heuristics).
- Pros: Preserves semantically critical context.
- Cons: More complex and computationally expensive.

### 4. Hybrid Approaches

- Combine sliding-window for base stability with importance-based retention of sparse long-term anchors.
- Example: **Attention Sinks** or **Landmark Tokens** that persist beyond normal eviction windows.

## Memory Management Techniques

### 1. Cache Compaction

- Defragment KV memory by compacting active tokens into contiguous regions.
- Reduces wasted memory from evicted slots.
- Requires efficient indexing and remapping of attention positions.

### 2. Dynamic Shaping

- Allocate cache tensors dynamically to match current sequence lengths instead of worst-case maximums.
- Improves memory utilization in batch scenarios with varying prompt sizes.

### 3. Quantized KV Cache

- Store KV in lower precision (e.g., FP8, INT8).
- Reduces memory footprint and bandwidth usage.
- Requires careful calibration to avoid accuracy degradation.

### 4. Memory Pooling

- Pre-allocate large memory pools shared across requests.
- Minimizes fragmentation and reduces GPU allocator overhead during runtime.

### 5. Multi-Level Storage

- GPU memory holds the active sliding window.
- Older cache segments offloaded to CPU or NVMe, with retrieval on demand.
- Suitable for ultra-long context (>100k tokens).

## Trade-offs

- **Performance vs. Accuracy**: Sliding window ensures speed but may harm long-context understanding. Importance-based retention is more accurate but costly.
- **Simplicity vs. Complexity**: LRU and sliding window are easy to implement, while hybrid and multi-level designs introduce scheduling and synchronization overhead.
- **Latency vs. Throughput**: Aggressive eviction improves per-request latency but may reduce throughput in batch settings due to frequent remapping.

## Practical Examples

- **GPT-3 / LLaMA**: Use full KV caching, limited by sequence length.
- **MPT / LongChat**: Employ sliding-window attention to enable long contexts.
- **vLLM**: Implements **PagedAttention**, treating KV memory as virtualized pages for efficient compaction and sharing.
- **FlashAttention-2**: Works with block-sparse layouts, complementing sliding-window and eviction strategies.

## Summary

Effective KV cache eviction and memory management are crucial for scaling LLM inference to long contexts and high throughput. Techniques range from simple sliding windows to sophisticated paged caches with hybrid eviction. System designers must carefully balance memory footprint, computational overhead, and accuracy to optimize inference performance in real-world deployments.
