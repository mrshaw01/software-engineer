# Fully Sharded Data Parallelism (FSDP)

Fully Sharded Data Parallelism (FSDP) is a **memory-optimized distributed training strategy** that shards **model parameters, gradients, and optimizer states** across devices. It is built on ideas from **ZeRO-DP (Zero Redundancy Optimizer)**, enabling **scalable training of trillion-parameter models** while maintaining high compute efficiency.

## Motivation

### Limitations of Data Parallelism (DP)

- Each GPU stores a **full copy of model parameters, gradients, and optimizer states**.

- Memory consumption for a model of size $\Psi$ (number of parameters):

  $$\text{Memory} = (2 + 2 + K) \Psi \text{ bytes}$$

  where:

  - **2 Ψ bytes** for parameters (fp16 + fp32 copy).
  - **2 Ψ bytes** for gradients.
  - **K Ψ bytes** for optimizer states (e.g., Adam has $K = 12$).

- Inefficient for **very large models** due to redundant copies.

### Model Parallelism (MP)

- Scales to larger models but incurs **significant communication overhead** for activations and gradients.

### Goal of FSDP

Combine:

- **Compute efficiency of DP**
- **Memory efficiency of MP**

## ZeRO-DP Foundation

ZeRO-DP introduces **three partitioning stages**:

1. **Optimizer State Partitioning ($P_\text{OS}$)**

   - Shards optimizer states across GPUs.
   - Each GPU updates only its shard of optimizer states.
   - **AllGather()** parameters at the end of each step.

2. **Gradient Partitioning ($P_\text{OS+G}$)**

   - Gradients are partitioned during the backward pass.
   - **ReduceScatter()** distributes reduced gradients across GPUs.

3. **Parameter Partitioning ($P_\text{OS+G+P}$)**

   - Parameters themselves are sharded.
   - **Broadcast()** or **AllGather()** parameters **on demand** before computation.
   - Frees parameters after each forward/backward pass.

The memory requirement with all three stages becomes approximately:

$$\text{Memory per GPU} \approx \frac{(2 + 2 + K)\Psi}{N}$$

where $N$ is the number of GPUs.

## FSDP Overview

FSDP is essentially **ZeRO Stage 3** with **layer-wise sharding and just-in-time parameter gathering**.

### Key Operations

#### **Forward Pass**

- For each layer, parameters are **AllGather()** from shards.
- After the layer computation, the full parameters are **freed immediately**.

#### **Backward Pass**

- Gradients are computed per layer.
- **ReduceScatter()** partitions and reduces gradients across GPUs.

#### **Optimizer Step**

- Each GPU updates **its shard of optimizer states and parameters**.
- **AllGather()** is only used when a full parameter update is required.

## Memory Savings

### Without Sharding

$$M_\text{DP} = (2 + 2 + K)\Psi$$

### With FSDP (ZeRO Stage 3)

$$M_\text{FSDP} = \frac{(2 + 2 + K)\Psi}{N}$$

Where:

- $\Psi$ = model size (parameters)
- $K$ = optimizer state factor (Adam: 12)
- $N$ = number of GPUs

Thus, for $N = 8$, **memory usage per GPU is reduced by 8×**.

## Communication Patterns

| Stage                            | Operation           |
| -------------------------------- | ------------------- |
| Parameter Gathering              | **AllGather()**     |
| Gradient Aggregation             | **ReduceScatter()** |
| Parameter Update Synchronization | **AllGather()**     |

During training:

1. **Broadcast/AllGather** full parameters layer-wise for computation.
2. **ReduceScatter** distributes reduced gradients.
3. Parameters are **freed immediately after usage**, lowering peak memory.

## Mixed Precision Training

- **Parameters and activations stored as fp16.**
- A **master fp32 copy** of parameters/optimizer states is kept for numerical stability.

This further reduces memory while maintaining training stability.

## FSDP in PyTorch

```python
import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1024, 4096)
        self.layer2 = nn.Linear(4096, 1024)

    def forward(self, x):
        return self.layer2(torch.relu(self.layer1(x)))

# Wrap the model with FSDP
model = MyModel().cuda()
fsdp_model = FSDP(model)

optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-4)

for data, labels in dataloader:
    optimizer.zero_grad()
    outputs = fsdp_model(data.cuda())
    loss = loss_fn(outputs, labels.cuda())
    loss.backward()
    optimizer.step()
```

## Comparison: DP vs ZeRO vs FSDP

| Feature                  | DP   | ZeRO Stage 1–3           | FSDP                  |
| ------------------------ | ---- | ------------------------ | --------------------- |
| Parameter Copy per GPU   | Full | Partial (Stage 3 shards) | Sharded (layer-wise)  |
| Gradient Sharding        | ❌   | Stage 2+                 | ✅                    |
| Optimizer State Sharding | ❌   | Stage 1+                 | ✅                    |
| Memory Efficiency        | Low  | Medium–High              | High                  |
| Peak Memory Usage        | High | Reduced                  | Minimal (freed early) |
| Mixed Precision Support  | Yes  | Yes                      | Yes                   |
