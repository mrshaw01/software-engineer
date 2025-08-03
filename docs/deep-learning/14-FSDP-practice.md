# FSDP Practice Code Examples

This file provides **hands-on practice** with Fully Sharded Data Parallelism (FSDP) in PyTorch.

## 1️⃣ Basic FSDP Training Example

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1024, 4096)
        self.layer2 = nn.Linear(4096, 1024)

    def forward(self, x):
        return self.layer2(torch.relu(self.layer1(x)))

def main():
    dist.init_process_group("nccl")

    torch.cuda.set_device(dist.get_rank())
    model = MyModel().cuda()

    fsdp_model = FSDP(model)
    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    for step in range(10):
        inputs = torch.randn(16, 1024).cuda()
        labels = torch.randn(16, 1024).cuda()

        optimizer.zero_grad()
        outputs = fsdp_model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        if dist.get_rank() == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    main()
```

Run with:

```bash
torchrun --nproc_per_node=2 fsdp_basic.py
```

## 2️⃣ FSDP with Mixed Precision

```python
from torch.distributed.fsdp import MixedPrecision

mixed_precision_policy = MixedPrecision(
    param_dtype=torch.float16,
    reduce_dtype=torch.float16,
    buffer_dtype=torch.float16,
)

fsdp_model = FSDP(model, mixed_precision=mixed_precision_policy)
```

## 3️⃣ Auto Wrapping Large Layers

```python
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

auto_wrap_policy = size_based_auto_wrap_policy(min_num_params=1e6)

fsdp_model = FSDP(model, auto_wrap_policy=auto_wrap_policy)
```

This wraps submodules with more than 1 million parameters in separate FSDP units.

## 4️⃣ Activation Checkpointing + FSDP

```python
from torch.utils.checkpoint import checkpoint_wrapper

model.layer1 = checkpoint_wrapper(model.layer1)
model.layer2 = checkpoint_wrapper(model.layer2)

fsdp_model = FSDP(model)
```

Activation checkpointing reduces memory by recomputing activations during backward pass.

## 5️⃣ Saving and Loading FSDP Models

```python
# Saving
state_dict = fsdp_model.state_dict()
torch.save(state_dict, "fsdp_model.pth")

# Loading
fsdp_model.load_state_dict(torch.load("fsdp_model.pth"))
```
