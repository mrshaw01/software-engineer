# Transformer Inference Internals

Transformer models form the backbone of modern large language models (LLMs). Understanding how they behave during inference is crucial for optimizing runtime performance and minimizing latency. This document focuses on the internal components and computation patterns of transformers during autoregressive inference.

## Overview

Transformer inference involves executing a stack of layers composed of attention and feed-forward modules. During inference, the model processes tokens sequentially (one at a time in the decode phase), which differs from the parallelism used during training. Performance depends heavily on how efficiently these layer computations are implemented and reused.

## Transformer Layer Structure

Each transformer layer consists of the following core components:

1. **Multi-Head Attention (MHA)**

   - Splits input into multiple heads.
   - Performs scaled dot-product attention for each head.
   - Merges outputs of all heads.

2. **Feed-Forward Network (FFN)**

   - Typically a two-layer MLP with activation (e.g., GELU).
   - Operates independently per token.

3. **Residual Connections**

   - Add the input to each sub-layer’s output to preserve gradients and stabilize training.

4. **Layer Normalization**

   - Applied before or after sub-layer (pre-norm vs. post-norm).
   - Ensures numerical stability during computation.

## Attention Computation in Inference

### 1. **Self-Attention**

During autoregressive inference, each token attends only to past tokens. This requires:

- **Masked Attention**: Prevents attending to future positions.
- **Caching**: Stores key and value projections from previous tokens for reuse.

### 2. **KV Cache Access**

At each decode step:

- The query for the new token is computed.
- Keys and values for all previous tokens are retrieved from the cache.
- Attention scores are calculated against cached keys.
- Output is computed by weighting cached values with the scores.

## Rotary Positional Embeddings (RoPE)

Many LLMs (e.g., LLaMA) replace absolute positional encodings with rotary embeddings:

- Applies a rotation in complex space to queries and keys.
- Requires tracking the absolute position index of each token during inference.
- Enables extrapolation beyond training sequence length.

## Causal Masking

Causal masking is enforced during self-attention:

- In **training**, this is applied as a matrix mask.
- In **inference**, masking is implicit since only one token is processed at a time, and past tokens are cached.

## Residual and Layer Norm Behavior

Inference systems must preserve:

- Correct **layer norm statistics** for single-token inputs.
- **Residual path accumulation** without degradation due to reduced numerical precision (e.g., bfloat16, int8).
- Certain models fuse residual, attention, and MLP into one block to reduce memory bandwidth usage.

## Model-Specific Quirks

Inference pipelines must adapt to specific model behaviors:

- **GPT-style models**: Pre-layer norm, rotary embeddings, ALiBi (Attention with Linear Biases).
- **T5-style models**: Encoder-decoder with relative positional encoding.
- **Mixtral/MoE models**: Include routing logic for expert selection and sparse compute.

## Layer Execution Pipeline

Each transformer layer typically follows this execution pattern at inference:

```
Input Token → LayerNorm
            → QKV Projection
            → Rotary Embedding
            → Attention Scores (with cached KV)
            → Attention Output
            → Residual Add + LayerNorm
            → MLP → Residual Add
```

## Inference Considerations

| Component      | Optimization Goals                                       |
| -------------- | -------------------------------------------------------- |
| Attention      | Use efficient kernels (e.g., FlashAttention, fused ops)  |
| KV Cache       | Compact layout, reuse across steps, batch-aligned access |
| Layer Norm     | Fuse with adjacent ops where possible                    |
| MLP            | Fused GEMMs, quantized matrix multiplications            |
| Residual Paths | Minimize redundant memory loads/stores                   |

## Summary

Understanding the internal structure of transformer layers and their runtime behavior during inference is key to achieving low-latency, high-throughput deployments. Efficient inference hinges on minimizing redundant computation, leveraging caching, and carefully managing numerical stability across layers.
