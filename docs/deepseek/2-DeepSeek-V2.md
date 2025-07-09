# Comprehensive Report on DeepSeek-V2 Architecture

## Overview

DeepSeek-V2 is an innovative, Mixture-of-Experts (MoE) language model characterized by economical training, efficient inference, and powerful performance. It possesses 236 billion total parameters, with 21 billion parameters activated per token, and supports a context length of 128K tokens.

## Architectural Design

DeepSeek-V2 is built upon the Transformer framework but integrates two major innovations:

1. **Multi-head Latent Attention (MLA)**: Enhances inference efficiency by significantly compressing the Key-Value (KV) cache.
2. **DeepSeekMoE**: Enables economical training through sparse computation and efficient expert parallelism.

## Multi-Head Latent Attention (MLA)

MLA addresses the inefficiency of standard Multi-Head Attention (MHA) by drastically reducing KV cache size without compromising performance.

### Mathematical Formulation

#### Standard MHA

The standard Multi-Head Attention mechanism computes queries $q_t$, keys $k_t$, and values $v_t$ from input embeddings $h_t$:

$$q_t = W^Q h_t \quad k_t = W^K h_t \quad v_t = W^V h_t$$

Multi-head attention output $u_t$ is then obtained as:

$$u_t = W^O \text{concat}(o_{t,1}, o_{t,2}, \dots, o_{t,n_h})$$

where:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?o_{t,i}=\sum\limits_{j=1}^{t}\text{softmax}_j\left(\frac{q_{t,i}^T%20k_{j,i}}{\sqrt{d_h}}\right)v_{j,i}"/>
</p>

#### Low-Rank KV Joint Compression

MLA compresses keys and values into a low-dimensional latent vector to optimize cache efficiency:

$$c_t^{KV} = W^{DKV} h_t \quad k_t^C = W^{UK} c_t^{KV} \quad v_t^C = W^{UV} c_t^{KV}$$

During inference, MLA caches only $c_t^{KV}$, drastically reducing KV cache storage.

#### Decoupled Rotary Position Embedding (RoPE)

MLA employs a decoupled RoPE strategy to integrate positional information without compromising efficiency:

$$q_t^R = \text{RoPE}(W^{QR} c_t^Q) \quad k_t^R = \text{RoPE}(W^{KR} h_t)$$

Concatenation of compressed and positional embeddings gives the final query-key representations:

$$q_{t,i} = [q_{t,i}^C; q_{t,i}^R] \quad k_{t,i} = [k_{t,i}^C; k_t^R]$$

The attention output integrates these embeddings:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?o_{t,i}=\sum_{j=1}^{t}\text{softmax}_j\left(\frac{q_{t,i}^T%20k_{j,i}}{\sqrt{d_h%20+%20d_h^R}}\right)v_{j,i}^C"/>
</p>

## DeepSeekMoE Architecture

### Basic Architecture

DeepSeekMoE partitions feed-forward network (FFN) experts into shared and routed experts:

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?h_t'=u_t+%5Csum_%7Bi=1%7D%5E%7BN_s%7DFFN_i%5E%7B(s)%7D(u_t)+%5Csum_%7Bi=1%7D%5E%7BN_r%7Dg_%7Bi,t%7DFFN_i%5E%7B(r)%7D(u_t)"/>
</p>

where:

$$
g_{i,t} =
\begin{cases}
s_{i,t}, & \text{if } s_{i,t} \in \text{TopK}(\{s_{j,t}\}, K_r) \\\\
0, & \text{otherwise}
\end{cases}
$$

$$s_{i,t} = \text{softmax}_i(u_t^T e_i)$$

### Routing Strategies and Balancing Mechanisms

- **Device-Limited Routing**: Limits MoE-related communication to at most M devices per token.
- **Balance Losses**: Includes expert-level, device-level, and communication balance losses to maintain efficient parallelism.
- **Token-Dropping Strategy**: Drops tokens with low affinity scores, reducing computational waste.

## Training and Optimization

- **Dataset**: 8.1T-token multi-source, multilingual corpus with enhanced Chinese content.
- **Hyper-Parameters**: 60 Transformer layers, 128 attention heads, and expert parallelism on 8 devices.
- **Framework**: HAI-LLM with optimized communication and CUDA kernels.
- **Long Context Extension**: Employs YaRN technique, extending effective context window length to 128K tokens.

## Performance and Efficiency

- **Inference Efficiency**: MLA significantly reduces KV cache by 93.3%, boosting maximum throughput by up to 5.76x compared to DeepSeek 67B.
- **Training Cost Reduction**: Saves 42.5% of GPU training hours compared to dense models.
- **Evaluation Benchmarks**: Achieves superior or competitive performance across diverse tasks including MMLU, GSM8K, HumanEval, and multilingual datasets.

## Fine-Tuning and Alignment

- **Supervised Fine-Tuning (SFT)**: Conducted on 1.5M high-quality conversational instances.
- **Reinforcement Learning (RL)**: Employs Group Relative Policy Optimization (GRPO), focusing separately on reasoning tasks (math and code) and general preference alignment.
- **Evaluation Results**: Demonstrates top-tier open-ended conversational performance, surpassing other open-source chat models on benchmarks like AlpacaEval and MT-Bench.

## Conclusion

DeepSeek-V2 stands out due to its balanced approach, combining the strengths of dense and sparse architectures to deliver both efficiency and superior performance, marking a significant advancement in economical and efficient large language model training and inference.
