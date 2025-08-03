# Distributed MNIST Training with Fully Sharded Data Parallel (FSDP)

This guide demonstrates **distributed training** on the **MNIST dataset** using **PyTorch Fully Sharded Data Parallel (FSDP)**.
FSDP is a memory-efficient distributed training strategy that **shards model parameters, gradients, and optimizer states across devices**, enabling training of larger models within limited GPU memory.

## Prerequisites

```bash
pip install torch torchvision
```

> ⚠️ Ensure that you are running on a machine with multiple GPUs or across multiple nodes connected via NCCL backend.

## How to Run

On a single node with 2 GPUs:

```bash
torchrun --nproc_per_node=2 train_fsdp_mnist.py
```

On multiple nodes:

```bash
torchrun \
  --nnodes=2 \
  --node_rank=0 \
  --nproc_per_node=4 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=<MASTER_IP>:29500 \
  train_fsdp_mnist.py
```

## Training Script

```python
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# ------------------------------
# 1️⃣ Distributed Setup
# ------------------------------
def setup_distributed():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_distributed():
    destroy_process_group()

# ------------------------------
# 2️⃣ Model Definition
# ------------------------------
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.fc(self.flatten(x))

# ------------------------------
# 3️⃣ Training Function
# ------------------------------
def train_one_epoch(model, loader, optimizer, criterion, epoch, rank):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    if rank == 0:
        print(f"[Epoch {epoch}] Training Loss: {running_loss/len(loader):.4f}")

# ------------------------------
# 4️⃣ Evaluation Function
# ------------------------------
@torch.no_grad()
def evaluate(model, loader, rank):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    if rank == 0:
        print(f"Validation Accuracy: {100 * correct / total:.2f}%")

# ------------------------------
# 5️⃣ Main Function
# ------------------------------
def main():
    setup_distributed()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")

    # Dataset & Dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2, pin_memory=True)

    # Model, Loss, Optimizer
    model = SimpleNN().to(device)
    model = FSDP(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training Loop
    for epoch in range(5):
        train_one_epoch(model, train_loader, optimizer, criterion, epoch, rank)

    evaluate(model, test_loader, rank)

    cleanup_distributed()

if __name__ == "__main__":
    main()
```

## References

- [PyTorch FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)
- [PyTorch Distributed Overview](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
