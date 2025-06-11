# Mixed Precision Training

This folder contains a structured overview of mixed precision training for deep learning, with a focus on FP16, BF16, and hardware-specific scenarios such as NPUs.

Mixed precision training improves performance and memory efficiency by using lower-precision formats (like FP16 or BF16) without sacrificing model accuracy. This technique is especially powerful when supported by hardware accelerators like GPUs or NPUs.

## Document Overview

### [01-introduction-to-mixed-precision-training.md](01-introduction-to-mixed-precision-training.md)

**Introduction to Mixed Precision Training**
An overview of why mixed precision matters, what benefits it brings (e.g. speed, memory savings), and how it maintains accuracy by selectively using FP32 where needed.

### [02-fp16-vs-bf16-formats.md](02-fp16-vs-bf16-formats.md)

**FP16 and BF16 Formats**
A deep dive into the two main 16-bit floating-point formats used in mixed precision. Explains layout, dynamic range, precision tradeoffs, and when to prefer one over the other.

### [03-training-with-mixed-precision-fp16.md](03-training-with-mixed-precision-fp16.md)

**Training with Mixed Precision Using FP16**
Covers practical challenges and solutions when training with FP16, including loss scaling, overflow detection, and maintaining FP32 master weights.

### [04-mixed-precision-training-scenarios-npu.md](04-mixed-precision-training-scenarios-npu.md)

**Mixed Precision Scenarios on NPUs**
Explores how mixed precision support varies across NPU architectures, and what operations must be supported depending on the use case (e.g., training vs inference, classification vs generative models).

## Use Cases

- Speed up training time with minimal accuracy loss
- Fit larger models or batch sizes into limited memory
- Improve hardware utilization on modern accelerators
- Understand precision requirements per layer or op

## Related

- [NVIDIA Mixed Precision Training Guide](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/)
- [Mixed Precision Training — Micikevicius et al., 2017 (arXiv:1710.03740)](https://arxiv.org/abs/1710.03740)
