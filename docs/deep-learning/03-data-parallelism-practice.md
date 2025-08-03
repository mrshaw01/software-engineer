# Data Parallelism Practice with PyTorch DistributedDataParallel (DDP)

This guide demonstrates how to implement data parallelism in PyTorch using **Distributed Data Parallel (DDP)**.

## 1. Overview

PyTorch provides `torch.distributed` and `torch.nn.parallel.DistributedDataParallel` to efficiently scale training across multiple GPUs or nodes.

Key steps:

1. Initialize the process group (`torch.distributed.init_process_group`).
2. Create a model and wrap it with `DistributedDataParallel`.
3. Use `DistributedSampler` for dataset sharding.
4. Launch multiple processes using `torch.multiprocessing.spawn` or a cluster manager like **SLURM**.

## 2. Basic Single-Node Multi-GPU Example

```python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

# Define a simple model
class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)

# Initialize distributed environment
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train_ddp(rank, world_size):
    setup(rank, world_size)

    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    for epoch in range(5):
        inputs = torch.randn(20, 10).to(rank)
        labels = torch.randn(20, 10).to(rank)

        optimizer.zero_grad()
        outputs = ddp_model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        if rank == 0:
            print(f"Epoch {epoch}, Rank {rank}, Loss: {loss.item()}")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)
```

## 3. Multi-Node Multi-GPU Training (SLURM)

### Launch Script Example (2 nodes, 4 GPUs per node)

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4

export MASTER_PORT=12355
export WORLD_SIZE=8
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

source ~/miniconda/etc/profile.d/conda.sh
conda activate myenv

srun python train.py
```

### Inside `train.py`

Replace `setup()` with:

```python
def setup(rank, world_size):
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank % torch.cuda.device_count())
```

## 4. Using DistributedSampler

To prevent data duplication across GPUs, use `DistributedSampler`:

```python
from torch.utils.data import DataLoader, DistributedSampler

train_dataset = MyDataset()
sampler = DistributedSampler(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)

for epoch in range(num_epochs):
    sampler.set_epoch(epoch)  # Ensures proper shuffling each epoch
    for inputs, labels in train_loader:
        ...  # training loop
```

## 5. Key Points

- Call `sampler.set_epoch(epoch)` inside the training loop.
- Move model to GPU **before** wrapping with DDP.
- Use `dist.barrier()` for synchronization when needed.
- Use NCCL backend for multi-GPU training on NVIDIA GPUs.
