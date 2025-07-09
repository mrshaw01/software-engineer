# Comprehensive Report on DeepSeek V3 Architecture

## Introduction

DeepSeek V3 is a state-of-the-art Mixture-of-Experts (MoE) language model containing a total of 671 billion parameters, with 37 billion parameters activated per token. It integrates advanced architectures and innovative training strategies to optimize performance and efficiency.

## Architectural Overview

### Basic Architecture

DeepSeek V3 is built upon the Transformer framework and incorporates two key components:

1. **Multi-Head Latent Attention (MLA)**
2. **DeepSeekMoE** with an auxiliary-loss-free load balancing strategy

These components enable efficient inference and economical training.

### Multi-Head Latent Attention (MLA)

MLA introduces low-rank joint compression for attention keys, values, and queries to reduce Key-Value (KV) cache sizes during inference:

- **Latent Compression:**

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?c_{KV_t}=W_{DKV}h_t"/>
</p>
<p align="center">
  <img src="https://latex.codecogs.com/svg.image?k^C_t=W_{UK}c_{KV_t},\quad%20v^C_t=W_{UV}c_{KV_t}"/>
</p>
<p align="center">
  <img src="https://latex.codecogs.com/svg.image?k^R_t=\text{RoPE}(W_{KR}h_t)"/>
</p>

- **Final Key-Value Pair:**

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?k_{t,i}=[k^C_{t,i};k^R_t]"/>
</p>

- **Query Compression:**

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?c_{Q_t}=W_{DQ}h_t"/>
</p>
<p align="center">
  <img src="https://latex.codecogs.com/svg.image?q^C_t=W_{UQ}c_{Q_t},\quad%20q^R_t=\text{RoPE}(W_{QR}c_{Q_t})"/>
</p>
<p align="center">
  <img src="https://latex.codecogs.com/svg.image?q_{t,i}=[q^C_{t,i};q^R_{t,i}]"/>
</p>

- **Attention Output:**

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?o_{t,i}=\sum_{j=1}^{t}\text{Softmax}_j\left(\frac{q_{t,i}^Tk_{j,i}}{\sqrt{d_h+d_h^R}}\right)v_{j,i}^C"/>
</p>

### DeepSeekMoE

This MoE architecture includes shared and routed experts to enhance efficiency:

- **Feed-Forward Networks:**

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?h_t'=u_t+\sum_{i=1}^{N_s}FFN_i^{(s)}(u_t)+\sum_{i=1}^{N_r}g_{i,t}FFN_i^{(r)}(u_t)"/>
</p>

- **Routing Gating Mechanism:**

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?g_{i,t}=\frac{g'_{i,t}}{\sum_{j=1}^{N_r}g'_{j,t}}"/>
</p>
<p align="center">
  <img src="https://latex.codecogs.com/svg.image?g'_{i,t}=\begin{cases}s_{i,t},&s_{i,t}+b_i\in\text{TopK}\{s_{j,t}+b_j\},K_r\\0,&\text{otherwise}\end{cases}"/>
</p>

### Auxiliary-Loss-Free Load Balancing

To prevent load imbalance without compromising model performance, DeepSeek V3 dynamically adjusts routing biases ($b_i$) based on expert load monitoring, eliminating the need for auxiliary losses.

## Multi-Token Prediction (MTP)

DeepSeek V3 uses a sequential multi-token prediction strategy, enhancing data efficiency and token prediction capabilities:

- **Prediction Modules:**
  Each module sequentially predicts tokens, maintaining causal chains across layers.

- **Loss Calculation:**

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\mathcal{L}_{MTP}=\frac{\lambda}{D}\sum_{k=1}^{D}\mathcal{L}_{MTP}^k"/>
</p>

## Training Framework

### DualPipe Algorithm

DualPipe significantly improves training efficiency by reducing pipeline bubbles and overlapping computations and communications.

- **Pipeline Bubbles:** DualPipe exhibits fewer bubbles compared to traditional methods.
- **Memory Efficiency:** It balances memory usage effectively despite maintaining multiple parameter copies.

### FP8 Mixed Precision Training

DeepSeek V3 introduces FP8 mixed precision training, reducing GPU memory usage and enhancing computational speed:

- **Fine-Grained Quantization:** Tile/block-wise scaling effectively manages activation and weight outliers.
- **Precision Improvements:** FP8 operations utilize increased accumulation precision and adaptive quantization.

## Infrastructure

DeepSeek V3 operates on clusters comprising 2048 NVIDIA H800 GPUs interconnected via NVLink and InfiniBand, optimizing cross-node communication and computation parallelism.

## Inference and Deployment

The inference strategy separates prefilling and decoding:

- **Prefilling Stage:** Utilizes TP4, DP8, and EP32 with redundant expert deployment.
- **Decoding Stage:** Employs TP4, DP80, and EP320 for maximum throughput and minimal latency.

## Evaluation Results

DeepSeek V3 demonstrates superior performance across multiple benchmarks, notably excelling in math and coding tasks, and showing comparable capabilities to leading closed-source models.

## Conclusion

DeepSeek V3 showcases significant advancements in efficient training, inference, and high performance across complex language modeling tasks, setting a new benchmark for open-source language models.
