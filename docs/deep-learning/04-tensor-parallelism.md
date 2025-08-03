# Tensor Parallelism (TP)

Tensor Parallelism is a **form of model parallelism** that splits the computation of tensor operations (e.g., matrix multiplications) across multiple GPUs. Instead of replicating the entire model on each device (like data parallelism), tensor parallelism **partitions the model parameters and computation**.

## 1. Motivation

- Deep learning models have grown so large that they often cannot fit into the memory of a single GPU.
- For such large models, we **parallelize the model itself** instead of the data.
- Tensor Parallelism is one of the main approaches for model parallelism, along with **Pipeline Parallelism (PP)**, **Expert Parallelism (EP)**, and **Fully Sharded Data Parallelism (FSDP)**.

## 2. Concept

- **Goal:** Exploit parallelism inside tensor operations, such as matrix multiplication, by splitting parameters across GPUs.
- **Steps:**

  1. Partition model parameters (e.g., weight matrices) across GPUs.
  2. Each GPU computes its part of the operation using the partitioned parameters.
  3. Synchronize and aggregate intermediate results to produce final outputs.

## 3. Example: Matrix Multiplication

For `C = A × B`:

- Assume `A`, `B`, `C` are 2×2 matrices.
- Partition `B` and `C` column-wise across GPUs, while keeping the same copy of `A` on each GPU.

```
C00  C01       B00  B01
C10  C11       B10  B11
```

- GPU 0 computes `C0 = A × B0`
- GPU 1 computes `C1 = A × B1`

This approach can be extended to larger matrices and more GPUs by splitting rows/columns accordingly.

## 4. Steps in Tensor Parallel Training

1. **Partition parameters** across multiple GPUs.
2. **Perform computations in parallel** using the local parameter partitions.
3. **Synchronize results** (e.g., using `all_gather` or `reduce_scatter`) to get the final outputs.

## 5. Advantages & Challenges

### Advantages

- Enables training extremely large models that cannot fit in a single GPU.
- Reduces memory consumption per GPU.
- Often combined with **Data Parallelism** and **Pipeline Parallelism** for 3D parallelism.

### Challenges

- Requires **extra communication** between GPUs to aggregate intermediate results.
- Implementation complexity increases.
- Speedup depends on tensor partitioning efficiency and interconnect bandwidth.

## 6. Popular Libraries Supporting TP

- **Megatron-LM** – Implements tensor parallelism for GPT-like models.
- **DeepSpeed** – Supports tensor parallel training in combination with ZeRO and pipeline parallelism.
- **PyTorch Distributed Tensor (DTensor)** – Experimental support for sharding tensors across devices.

Tensor Parallelism is an essential technique for **scaling model training across multiple GPUs** and is widely used in training **large language models (LLMs)** like GPT, BERT, and beyond.
