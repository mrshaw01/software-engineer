# Megatron-LM and Tensor Parallel MLP

Megatron-LM is a **large-scale language model framework** based on GPT-2 architecture, extended to support more layers and parameters through **model parallelism**. It primarily uses **tensor parallelism** to efficiently train massive models across multiple GPUs.

## 1. Megatron-LM Overview

- Built on **Transformer architecture** with **multi-head attention layers**.
- GPT-2 based, but with significantly **larger number of parameters and layers**.
- Uses **pre-layer normalization** (different from GPT-2’s post-layer norm).
- Efficient scaling using **tensor parallelism** for MLP and attention layers.

## 2. Multi-Head Attention in Megatron-LM

- Each transformer layer consists of:

  - **Self-Attention module** (multi-head attention)
  - **MLP (feed-forward) module**
  - **Residual connections + LayerNorm**

- Tensor parallelism is applied to both **attention** and **MLP blocks** to distribute computations.

## 3. Tensor-Parallel MLP

### Input/Parameters/Output

- Input: **X**
- Parameters: **A**, **B** (weight matrices)
- Output: **Z**

### Two Phases:

1. **Phase 1:** $Y = GELU(XA)$
2. **Phase 2:** $Z = Dropout(YB)$

## 4. Parallel Computation

- Partition **A** and **B** across GPUs (e.g., $A = [A_1, A_2]$, $B = [B_1, B_2]$).
- Each GPU computes partial results:

  - GPU 1: $Z_1 = GELU(XA_1)B_1$
  - GPU 2: $Z_2 = GELU(XA_2)B_2$

- Final output:

$$
Z = Dropout(Z_1 + Z_2)
$$

- Aggregation is done using **AllReduce communication**.

## 5. Mathematical Formulation

### Phase 1:

$$Y = GELU(XA) = GELU([XA_1, XA_2]) = [GELU(XA_1), GELU(XA_2)]$$

### Phase 2:

$$Z = Dropout(YB) = Dropout(Y_1B_1 + Y_2B_2) = Dropout(Z_1 + Z_2)$$

## Tensor-Parallel Multi-Head Attention (MHA)

### Setup

- **Input:** `X`
- **Parameters:** $ W^Q, W^K, W^V, B $
- **Output:** `Z`

### Execution Phases

1. **Phase 1:**
   $$Y = \text{Dropout}(\text{Softmax}(QK^T) V)$$
   where:

   - $ Q = XW^Q $
   - $ K = XW^K $
   - $ V = XW^V $

2. **Phase 2:**
   $$Z = \text{Dropout}(Y B)$$

### Key Challenge

- $$
  \text{Softmax}(Q_1 K_1^T + Q_2 K_2^T) \neq
  \text{Softmax}(Q_1 K_1^T) + \text{Softmax}(Q_2 K_2^T)
  $$
- To parallelize, **Q, K, and V must be partitioned along the head dimension**.

## Dimension Analysis in MHA

- $ Q, K, V \in \mathbb{R}^{S \times H} $
- When there are multiple heads:
  - Partition columns of $ Q, K, V $ by **head dimension**.
  - Each GPU processes **a subset of heads**.

## Pitfalls

- Each GPU must have the same **copy of input (X)** and **output (Z)** before and after computation.
- At the first forward pass, **input X must be broadcast** across GPUs:
  - **Implicit way:** Load identical data on each GPU.
  - **Explicit way:** Use `Broadcast()` collective communication.

## Communication Overheads

- **Tensor-parallel MHA:** 2× AllReduce (1 forward + 1 backward)
- **Tensor-parallel MLP:** 2× AllReduce (1 forward + 1 backward)

## Key Takeaways

- **Megatron LM leverages tensor parallelism to scale transformers efficiently.**
- **MLP and MHA layers are split across GPUs to distribute computation and memory usage.**
- **Collective communication (AllReduce, Broadcast) is essential for aggregating partial results.**
