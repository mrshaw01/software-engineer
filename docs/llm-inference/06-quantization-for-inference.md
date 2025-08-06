# Quantization for Inference

Quantization is a key optimization technique for reducing the memory footprint and compute requirements of large language models (LLMs) during inference. It involves representing model parameters and activations with lower-precision data types (e.g., int8, bfloat8) rather than full-precision float32.

This section outlines common quantization strategies, trade-offs, and their practical implications for LLM inference systems.

## 1. Motivation

Large models incur high latency and memory costs at inference time. Quantization addresses this by:

- Reducing model size (weights and activations)
- Lowering memory bandwidth usage
- Accelerating matrix multiplications with hardware-friendly formats (e.g., int8 GEMM)
- Improving batch throughput, especially on memory-bound devices

## 2. Quantization Types

### 2.1 Post-Training Quantization (PTQ)

PTQ converts a pretrained model to lower precision without retraining.

- **Advantages**: Fast, simple, no training needed
- **Drawbacks**: May lead to accuracy degradation, especially for sensitive layers

Common PTQ methods:

- Static quantization: Requires calibration dataset to estimate activation ranges
- Dynamic quantization: Estimates activation ranges on-the-fly during inference
- Weight-only quantization: Only quantizes model weights (e.g., GPTQ)

### 2.2 Quantization-Aware Training (QAT)

QAT simulates quantization effects during training so the model learns to be robust to quantization noise.

- **Advantages**: Higher accuracy, especially for lower-bit quantization
- **Drawbacks**: Requires retraining with access to training data

## 3. Data Types for Quantization

| Data Type  | Bit Width | Use Case                                |
| ---------- | --------- | --------------------------------------- |
| float32    | 32-bit    | Baseline                                |
| float16    | 16-bit    | Common on GPUs                          |
| bfloat16   | 16-bit    | TPU-friendly                            |
| int8       | 8-bit     | PTQ/QAT inference                       |
| bfloat8    | 8-bit     | Emerging, low-error                     |
| int4, int2 | ≤ 4-bit   | Aggressive compression (e.g., LLM.int4) |

## 4. Quantization Granularity

- **Per-tensor quantization**: Single scale/zero-point per tensor; fast but less accurate
- **Per-channel quantization**: Separate scale/zero-point per channel; more accurate, especially for MLP weights
- **Group-wise quantization**: Trade-off between per-tensor and per-channel; balances complexity and accuracy

## 5. Common Quantization Formats

- **GPTQ** (Post-training quantization using Hessian-based rounding)
- **AWQ** (Activation-aware weight quantization)
- **SmoothQuant** (Scales activations/weights jointly for better quantization compatibility)
- **INT8 + FP16 hybrid** (Keeps sensitive layers in FP16, others in INT8)

## 6. Quantization in Practice

### 6.1 LLM Quantization Examples

| Model            | Format   | Comment                       |
| ---------------- | -------- | ----------------------------- |
| LLaMA 2-13B      | GPTQ     | 4-bit, widely used            |
| Falcon 40B       | AWQ      | Better accuracy than GPTQ     |
| GPT-3 (research) | INT8 QAT | Good balance of size/accuracy |

### 6.2 Framework Support

| Framework    | Quantization Support                      |
| ------------ | ----------------------------------------- |
| PyTorch      | `torch.quantization`, `bitsandbytes`      |
| Transformers | `AutoGPTQ`, `optimum`, `bitsandbytes`     |
| TensorRT     | INT8/FP16 support for optimized execution |
| ONNX Runtime | INT8 inference via `quantize_static` APIs |

## 7. Accuracy vs Performance Tradeoff

Quantization introduces a tradeoff:

- **Lower precision = faster inference**
- **But may lead to accuracy drop**, especially in generative tasks

To mitigate:

- Keep first/last layers in higher precision (e.g., FP16)
- Use per-channel quantization
- Employ QAT if retraining is feasible

## 8. Hardware Implications

Modern accelerators often have native support for mixed-precision inference:

| Hardware    | Optimized For       |
| ----------- | ------------------- |
| NVIDIA GPUs | INT8, FP16, TF32    |
| AMD GPUs    | INT8, BF16          |
| TPUs        | BF16, INT8          |
| NPUs        | Custom (e.g., BFP8) |

## 9. Best Practices

- Calibrate using representative input data
- Profile accuracy before and after quantization
- Layer-wise sensitivity analysis (e.g., keep attention layers in higher precision)
- Use quantization-friendly activation functions (e.g., ReLU over GELU)

## 10. References

- [GPTQ: Accurate Post-training Quantization for Generative Transformers](https://arxiv.org/abs/2210.17323)
- [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)
- [SmoothQuant: Accurate and Efficient Post-Training Quantization](https://arxiv.org/abs/2211.10438)
- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
