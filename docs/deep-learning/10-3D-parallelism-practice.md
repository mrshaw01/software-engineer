# 3D Parallelism in Practice

Large Language Models (LLMs) such as GPT-4, LLaMA, and PaLM require **massive computational resources** for training due to their billions or even trillions of parameters. To train these models efficiently, modern training systems employ **3D parallelism**, which combines:

- **Data Parallelism (DP)**
- **Tensor Parallelism (TP)**
- **Pipeline Parallelism (PP)**

This approach enables scaling training to thousands of GPUs while optimizing memory usage and compute efficiency.

## Key Concepts

### **1. Data Parallelism (DP)**

- Each GPU (or group of GPUs) holds a **full copy of the model**.
- Different GPUs process **different mini-batches of data**.
- Gradients are synchronized (e.g., via `torch.distributed.all_reduce`).

### **2. Tensor Parallelism (TP)**

- A single model layer is **split across multiple GPUs**.
- Each GPU computes a partial result of the layer (e.g., matrix multiplications).
- Commonly used for very large layers like attention and feedforward layers.

### **3. Pipeline Parallelism (PP)**

- The **model is divided into stages** across GPUs.
- Each stage processes a portion of the model’s layers.
- Micro-batches are used to **fill the pipeline** and maximize GPU utilization.

## 3D Parallelism Formula

Total number of GPUs used:

$$\text{GPUs} = DP \times TP \times PP$$

For example:

- **DP = 4**, **TP = 2**, **PP = 2** → **Total GPUs = 16**

## Frameworks Supporting 3D Parallelism

- **[DeepSpeed](https://github.com/microsoft/DeepSpeed)** → ZeRO, Pipeline + Tensor Parallelism.
- **[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)** → Tensor Parallelism + Pipeline Parallelism.
- **[PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html)** → Optimized data parallelism with sharded states.

## Example: 3D Parallel Training with Megatron-DeepSpeed

```python
import torch
import deepspeed
from megatron.training import pretrain

# Initialize distributed environment
device = torch.device("cuda")
torch.distributed.init_process_group(backend="nccl")

# Define parallelism sizes
args = {
    "tensor_model_parallel_size": 2,  # TP
    "pipeline_model_parallel_size": 2,  # PP
    "data_parallel_size": 4,  # DP
}

# DeepSpeed configuration (JSON)
ds_config = {
    "train_batch_size": 1024,
    "train_micro_batch_size_per_gpu": 8,
    "zero_optimization": {
        "stage": 1
    },
    "bf16": {"enabled": True},
    "gradient_accumulation_steps": 4,
}

# Launch training
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=pretrain(args),
    model_parameters=pretrain(args).parameters(),
    config=ds_config
)

for step, batch in enumerate(data_loader):
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
```

## Workflow

1. **TP** splits large layers (e.g., attention heads, FFN hidden dimensions) across GPUs.
2. **PP** divides layers into sequential stages.
3. **DP** replicates each pipeline stage+TP group for data parallel training.

## Example Parallelism Setup

| DP  | TP  | PP  | GPUs | Use Case                 |
| --- | --- | --- | ---- | ------------------------ |
| 4   | 1   | 1   | 4    | Standard DDP             |
| 2   | 2   | 2   | 8    | Balanced 3D parallelism  |
| 8   | 4   | 2   | 64   | Large-scale GPT training |

## Best Practices

- Start with **tensor parallelism** for very large layers.

- Add **pipeline parallelism** to reduce per-GPU memory.

- Use **ZeRO** or **FSDP** to optimize data parallelism and optimizer states.

- Profile **throughput vs. memory trade-offs** before finalizing DP/TP/PP ratios.

## References

- [Megatron-LM Paper](https://arxiv.org/abs/1909.08053)
- [PipeDream Paper](https://arxiv.org/abs/1806.03377)
- [DeepSpeed ZeRO](https://arxiv.org/abs/1910.02054)
