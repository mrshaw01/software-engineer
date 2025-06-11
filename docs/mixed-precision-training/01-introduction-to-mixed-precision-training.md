# Part 1: Introduction to Mixed Precision Training

## 1. Introduction

Using numerical formats with lower precision than 32-bit floating point offers several practical benefits:

- **Reduced memory usage**: Lower precision formats consume less memory, allowing for training and deployment of larger models.
- **Lower memory bandwidth**: Smaller data formats mean faster data transfers.
- **Faster computation**: Especially on GPUs with Tensor Core support, reduced precision significantly accelerates mathematical operations.

**Mixed precision training** achieves these benefits while maintaining task-specific accuracy. It works by using 16-bit floating point (FP16) for most operations and retaining 32-bit floating point (FP32) only where necessary, such as in gradient accumulation or weight updates.

## 2. What is Mixed Precision Training?

Mixed precision training involves the combined use of FP16 and FP32 in a deep learning workflow. This approach provides significant computational speedup—often up to 3x—on hardware that supports FP16 computation, such as NVIDIA GPUs with Tensor Cores (starting from the Volta architecture).

To enable mixed precision training:

- The model must be modified to use FP16 where appropriate.
- **Loss scaling** must be added to preserve small gradient values that might otherwise underflow in FP16.

This technique became practically feasible with CUDA 8 and the NVIDIA Deep Learning SDK, initially on Pascal GPUs.

## 3. Why Use Lower Precision?

Training deep neural networks with lower precision:

- **Decreases memory consumption**: FP16 uses half the memory of FP32, allowing larger models or mini-batches.
- **Reduces training time**: FP16 cuts memory access time and provides higher arithmetic throughput—up to 8x on supported GPUs.

Mixed precision with proper loss scaling achieves comparable accuracy to full precision while offering computational benefits.
