# DeepSeek-V3: Introduction Notes

## Context and Motivation

- **LLMs Evolution**: Rapid advancements towards AGI by Anthropic, Google, OpenAI (2024).
- **Open-Source Progress**: Significant strides by DeepSeek, LLaMA, Qwen, Mistral series to close gaps with closed-source models.
- **DeepSeek-V3 Goal**: Scale up open-source MoE models to push capability boundaries.

## Model Overview

- **Model Type**: Mixture-of-Experts (MoE)
- **Total Parameters**: 671B
- **Activated per Token**: 37B

## Architectural Innovations

1. **Multi-head Latent Attention (MLA)**
   - For efficient inference.
   - Validated in DeepSeek-V2.
2. **DeepSeekMoE**
   - Cost-effective training architecture.
   - Proven performance stability in prior versions.

## New Strategies

- **Auxiliary-Loss-Free Load Balancing**
  - Balances expert loads without performance degradation.
- **Multi-Token Prediction Objective**
  - Enhances benchmark performance by predicting multiple tokens simultaneously.

## Training Optimizations

- **FP8 Mixed Precision Training**
  - First large-scale validation of FP8 for accelerated training and reduced memory.
  - Low-precision training tied to hardware advances (e.g. InfiniBand, NVLink).
- **DualPipe Algorithm**
  - Efficient pipeline parallelism with minimal bubbles.
  - Overlaps computation and communication to reduce all-to-all communication overhead.
- **Efficient Cross-Node Communication**
  - Optimized all-to-all kernels for full IB and NVLink utilization.
- **Memory Footprint Optimization**
  - Enables training without costly tensor parallelism.

## Training Dataset and Stability

- **Pre-training Tokens**: 14.8 trillion diverse, high-quality tokens.
- **Training Stability**: No irrecoverable loss spikes or rollbacks throughout.

## Context Length Extension

1. **Stage 1**: Max context length extended to 32K.
2. **Stage 2**: Further extended to 128K.

## Post-Training

- **Stages**:
  - Supervised Fine-Tuning (SFT)
  - Reinforcement Learning (RL)
- **Distillation**: Transfers reasoning capability from DeepSeek-R1 series.
- **Balance**: Maintains accuracy and generation length.

## Evaluation Results

- **Benchmarks**:
  - DeepSeek-V3-Base: Strongest open-source base model, especially in code and math.
  - Chat Version: Outperforms other open-source models; comparable to GPT-4o, Claude-3.5-Sonnet.

## Training Costs Summary

| **Stage**         | **GPU Hours (H800)** | **Cost (USD)** |
| ----------------- | -------------------- | -------------- |
| Pre-Training      | 2,664K               | $5.328M        |
| Context Extension | 119K                 | $0.238M        |
| Post-Training     | 5K                   | $0.01M         |
| **Total**         | **2.788M**           | **$5.576M**    |

- **Efficiency**:
  - 180K GPU hours per trillion tokens (~3.7 days on 2048 H800 GPUs).
  - Full pre-training completed in < 2 months.

> **Note**: Costs exclude prior research, architecture ablations, and data experiments.
