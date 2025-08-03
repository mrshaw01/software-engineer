# 06 – Tensor Parallelism Practice

This section provides a **hands-on example of implementing Tensor Parallelism (TP)** in PyTorch, inspired by the approach used in **Megatron-LM**.

## Setup Distributed Environment

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def init_distributed(rank, world_size):
    dist.init_process_group(
        backend="nccl",  # Best for NVIDIA GPUs
        init_method="env://",  # Uses environment variables MASTER_ADDR, MASTER_PORT
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)
    print(f"Rank {rank}/{world_size} initialized.")
```

## Tensor Parallel Linear Layer

Instead of storing the **full weight matrix on each GPU**, we split the **output dimension** across GPUs.

```python
import torch.nn as nn

class TensorParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, process_group):
        super().__init__()
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group)
        self.rank = dist.get_rank(process_group)

        assert out_features % self.world_size == 0
        self.local_out_features = out_features // self.world_size

        # Each GPU stores only its partition of the weight
        self.weight = nn.Parameter(torch.randn(in_features, self.local_out_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(self.local_out_features))

    def forward(self, x):
        local_out = x @ self.weight + self.bias
        # Gather results from all GPUs
        outputs = [torch.zeros_like(local_out) for _ in range(self.world_size)]
        dist.all_gather(outputs, local_out, group=self.process_group)
        return torch.cat(outputs, dim=-1)
```

## Tensor Parallel MLP Block

An MLP block in a transformer consists of two linear layers with an activation function in between.

```python
class TensorParallelMLP(nn.Module):
    def __init__(self, hidden_size, ffn_hidden_size, process_group):
        super().__init__()
        self.dense_in = TensorParallelLinear(hidden_size, ffn_hidden_size, process_group)
        self.activation = nn.GELU()
        self.dense_out = TensorParallelLinear(ffn_hidden_size, hidden_size, process_group)

    def forward(self, x):
        x = self.dense_in(x)
        x = self.activation(x)
        x = self.dense_out(x)
        return x
```

## Tensor Parallel Multi-Head Attention

We split the **attention heads** across GPUs so that each GPU computes attention for a subset of heads.

```python
class TensorParallelSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, process_group):
        super().__init__()
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group)

        assert num_heads % self.world_size == 0
        self.local_heads = num_heads // self.world_size
        self.head_dim = hidden_size // num_heads

        self.qkv_proj = TensorParallelLinear(hidden_size, 3 * hidden_size, process_group)
        self.out_proj = TensorParallelLinear(hidden_size, hidden_size, process_group)

    def forward(self, x):
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        b, seq_len, _ = q.shape
        q = q.view(b, seq_len, self.local_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, seq_len, self.local_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, seq_len, self.local_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = attn_weights @ v

        attn_output = attn_output.transpose(1, 2).reshape(b, seq_len, -1)
        return self.out_proj(attn_output)
```

## Example Usage

```python
import torch.multiprocessing as mp

def run_training(rank, world_size):
    init_distributed(rank, world_size)
    process_group = dist.new_group(ranks=list(range(world_size)))

    hidden_size = 1024
    ffn_hidden_size = 4096
    num_heads = 16

    model = nn.Sequential(
        TensorParallelSelfAttention(hidden_size, num_heads, process_group),
        TensorParallelMLP(hidden_size, ffn_hidden_size, process_group)
    ).cuda(rank)

    x = torch.randn(8, 32, hidden_size).cuda(rank)
    output = model(x)

    print(f"Rank {rank}: Output shape {output.shape}")

if __name__ == "__main__":
    world_size = 2
    mp.spawn(run_training, args=(world_size,), nprocs=world_size, join=True)
```

## Key Concepts

- **Weight partitioning** → Reduces memory usage by splitting weights across GPUs.
- **Head partitioning** → Each GPU computes attention for a subset of heads.
- **All-Gather & All-Reduce** → Used to synchronize outputs across GPUs.
- **Scales to massive models** → Used in **Megatron-LM, GPT-NeoX, BLOOM** for trillion-parameter models.
