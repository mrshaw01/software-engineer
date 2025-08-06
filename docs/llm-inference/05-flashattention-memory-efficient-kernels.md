# FlashAttention & Memory-Efficient Kernels

Efficient attention computation is essential for optimizing LLM inference, where memory bandwidth and compute cost are key bottlenecks. This section covers FlashAttention and other kernel-level optimizations that significantly reduce memory usage and improve performance, especially during the autoregressive decode phase.

## FlashAttention: Overview

**FlashAttention** is a memory-efficient attention algorithm that computes exact softmax attention while avoiding the quadratic memory overhead of standard implementations.

### Motivation

Standard attention scales with O(n²) memory and compute due to the intermediate attention score matrix. FlashAttention addresses this by:

- Fusing operations (matmul + softmax + dropout)
- Avoiding materialization of large intermediate matrices
- Leveraging tiling and recomputation to fit within GPU SRAM

### Key Ideas

- **Tiling over sequence length**: Inputs are split into blocks, and attention is computed per block using on-chip memory.
- **Recomputation strategy**: Forward pass computes softmax-normalized outputs without storing full attention matrix; backward recomputes needed values.
- **Fused kernels**: Attention is implemented using a single kernel for QKᵗ, softmax, and final matmul with V.

### Performance Benefits

- **Memory Usage**: Reduced from O(n²) to O(n) in practice.
- **Speed**: Often 2–4× faster than standard attention (depending on sequence length and hardware).
- **Numerical Stability**: Maintains exact output by computing softmax in a stable log-sum-exp form.

## FlashAttention Variants

| Variant           | Description                                         |
| ----------------- | --------------------------------------------------- |
| FlashAttention v1 | Original implementation (CUDA kernel with tiling)   |
| FlashAttention v2 | Improved vectorization, support for dropout, causal |
| FlashDecoding     | Optimized for autoregressive decoding (streaming)   |

## Integration with Transformer Inference

During inference, attention can be split into two modes:

- **Prefill phase** (context embedding): full attention with FlashAttention (non-causal or causal)
- **Decode phase** (token-by-token): uses single-token KV lookup; FlashDecoding or fused attention kernels are used

## Memory-Efficient Kernel Techniques

Besides FlashAttention, other techniques improve kernel-level memory and compute efficiency:

### 1. Fused MLP Kernels

- Combine linear → activation → linear in a single kernel
- Reduces memory traffic between layers
- Common fusions: GELU + Linear, SiLU + Linear

### 2. Fused LayerNorm

- Fuses mean/variance computation and normalization
- Reduces memory reads/writes
- Often fused with adjacent matmuls in transformer blocks

### 3. Tiling and Vectorization

- Break large tensor ops into blocks that fit into SRAM or L2 cache
- Use warp- or thread-level parallelism
- Enables data reuse and improves occupancy

### 4. Shared Memory and SRAM Utilization

- Manual usage of shared memory in CUDA/HIP kernels
- Avoids expensive global memory access
- Common in fused attention and softmax kernels

## Tradeoffs

| Technique         | Benefit               | Tradeoff                       |
| ----------------- | --------------------- | ------------------------------ |
| FlashAttention    | Speed & memory saving | Limited to supported hardware  |
| Kernel Fusion     | Less memory traffic   | Harder to debug, less modular  |
| Tiling & Blocking | SRAM reuse            | Requires tuning for dimensions |
| Recomputation     | Low memory footprint  | Slightly higher compute cost   |

## Implementation Notes

- FlashAttention is implemented in Triton, CUDA, and used in frameworks like HuggingFace, vLLM, and PyTorch 2.0.
- Memory-efficient fused kernels are often custom-written in CUDA or scheduled via Triton.
- PagedAttention in vLLM builds on FlashAttention principles, with virtualized contiguous attention cache layout.

## References

- Dao et al. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- Triton language for custom kernel authoring: https://github.com/openai/triton
- vLLM’s FlashDecoding: https://github.com/vllm-project/vllm
