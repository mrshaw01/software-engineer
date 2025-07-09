# Comprehensive Report on DeepSeek V1 LLM Architecture

## Introduction

DeepSeek LLM is an open-source large language model (LLM) designed with a long-term perspective, focusing on scaling laws for optimized model performance. This report summarizes the architecture, mathematical formulations, hyperparameter selection, scaling strategies, alignment techniques, and evaluation outcomes of DeepSeek V1.

## Architectural Design

DeepSeek V1 is fundamentally based on the Transformer decoder-only architecture, with specific enhancements and modifications to optimize both pre-training and inference:

### Micro-architecture

- **Pre-Norm Structure:** Employs RMSNorm for normalization.
- **Feed-Forward Network (FFN):** Uses SwiGLU activation with an intermediate dimension of $\frac{8}{3} d_{model}$.
- **Positional Encoding:** Rotary Embeddings are used to encode positional information.
- **Attention Mechanism:** For the larger 67B parameter model, Grouped-Query Attention (GQA) is employed instead of traditional Multi-Head Attention (MHA) for computational efficiency.

### Macro-architecture

- **Model Sizes:** Primarily developed in two sizes, 7B and 67B parameters.
- **Layer Configurations:**

  - 7B model: 30 layers.
  - 67B model: 95 layers, optimizing depth rather than width for better partitioning during training and inference.

| Params | Layers | Model Dimension | Heads | KV Heads | Context Length | Batch Size | Learning Rate | Tokens |
| ------ | ------ | --------------- | ----- | -------- | -------------- | ---------- | ------------- | ------ |
| 7B     | 30     | 4096            | 32    | 32       | 4096           | 2304       | 4.2e-4        | 2.0T   |
| 67B    | 95     | 8192            | 64    | 8        | 4096           | 4608       | 3.2e-4        | 2.0T   |

## Mathematical Formulations and Scaling Laws

DeepSeek extensively revisits and refines scaling laws, improving model and data allocation for optimal performance.

### Optimal Hyperparameter Scaling

Empirical scaling laws are defined as:

$$\eta_{opt} = 0.3118 \cdot C^{-0.1250}, \quad B_{opt} = 0.2920 \cdot C^{0.3271}$$

where:

- $\eta_{opt}$: optimal learning rate
- $B_{opt}$: optimal batch size
- $C$: compute budget

### Optimal Model and Data Scaling

A refined representation, **non-embedding FLOPs/token ($M$)**, provides a precise model scale measure:

$$M_{opt} = 0.1715 \cdot C^{0.5243}, \quad D_{opt} = 5.8316 \cdot C^{0.4757}$$

where:

- $M_{opt}$: optimal model scale
- $D_{opt}$: optimal data scale

## Training Infrastructure

DeepSeek utilizes the HAI-LLM framework integrating:

- Data parallelism, tensor parallelism, sequence parallelism, and 1F1B pipeline parallelism.
- ZeRO-1 for optimizer state partitioning.
- Flash Attention for efficient hardware usage.
- Gradient accumulation in FP32 precision for stability.

## Alignment Techniques

DeepSeek implements a two-stage alignment:

### Supervised Fine-Tuning (SFT)

- Approximately 1.5 million multilingual instruction data instances.
- Specific fine-tuning on general language tasks, math, and code exercises.
- Fine-tuning epochs differ between 7B (4 epochs) and 67B (2 epochs) due to overfitting.

### Direct Preference Optimization (DPO)

- Enhances model conversational and safety skills.
- Training employs generated response preferences across various multilingual prompts.

## Evaluation Results

Evaluations across multiple benchmarks demonstrate DeepSeek’s superior performance:

### Public Benchmarks

- **General Performance:** DeepSeek 67B notably outperforms LLaMA-2 70B in math, code, and reasoning tasks.
- **Chat Models:** Significant improvement in reasoning tasks (GSM8K, HumanEval) after SFT and DPO.

### Open-ended Evaluations

- **Chinese Open-ended Evaluation (AlignBench):** DeepSeek 67B Chat surpasses GPT-3.5-turbo across fundamental and reasoning tasks.
- **English Open-ended Evaluation (MT-Bench):** DeepSeek 67B Chat DPO achieves a near-GPT-4 performance.

### Held-out Evaluations

- **LeetCode and Hungarian Exam:** Superior results relative to contemporary models, indicating robust generalization.

### Safety Evaluation

- Demonstrates high safety compliance across various sensitive and risky categories through rigorous manual verification.

## Conclusion

DeepSeek V1 successfully leverages enhanced scaling laws, a refined Transformer architecture, and rigorous training methodologies to establish a high-performing, safe, and open-source LLM framework. It demonstrates strong capabilities in reasoning, mathematics, coding, and open-ended conversation, laying a robust foundation for further advancements in large-scale language modeling.
