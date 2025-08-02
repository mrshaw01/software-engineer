### **Data Parallelism Overview**

Data parallelism is a strategy used in computing to divide a large task into smaller, independent tasks that can be processed simultaneously across multiple processors or machines. It is especially effective for tasks that involve processing large datasets with repetitive operations.

### **Analogy: Preparing a Banquet**

- Imagine **P chefs** preparing **N meals**.
- Each chef produces **N/P complete meals** independently.
- As the number of meals (**N**) increases, more chefs (**P**) can be added, provided there are enough resources (stoves, cutting boards, etc.).
- This setup makes scaling relatively easy.

### **Data Parallelism in Deep Neural Networks (DNNs)**

1. **Characteristics**

   - Deep learning operations (layers) follow fixed computation patterns.
   - Most primitive operations are independent along the mini-batch dimension **N**.
   - The main exception is operations like **batch normalization**, which require statistics across the batch.

2. **How It Works**

   - **Broadcast** network weights to all workers (e.g., GPUs) so that all start with synchronized models.
   - **Scatter** the input mini-batch across workers, each processing a different subset.
   - After computation, **gather** the outputs.

### **Inference with Data Parallelism**

- Each worker processes a portion of the input data independently.
- The layers (e.g., convolution, bias addition, ReLU) can be chained without communication between workers.
- The outputs from all workers are combined at the end.

### **Training with Data Parallelism**

Training consists of:

1. **Forward pass**
2. **Backpropagation**
3. **Parameter update**

- **Forward pass & Backpropagation:**
  Parallelization is similar to inference; layers are processed in parallel across workers without communication.

- **Parameter Update:**
  After each iteration, network weights must remain synchronized across all workers.

  - Before updating parameters, **gradients from all workers are averaged using AllReduce collective communication**.
  - This ensures all workers have the same updated weights for the next iteration.

### **Data Parallelism in PyTorch**

PyTorch provides built-in support for data parallelism through the `torch.distributed` package, enabling efficient distributed training across multiple GPUs and nodes.

### **Key Components**

1. **`torch.distributed` Package**

   - Provides APIs for distributed training.
   - Supports multiple communication backends: **NCCL, MPI, Gloo**.

     - **NCCL** – Best for NVIDIA GPUs.
     - **Gloo/MPI** – Suitable for CPUs.

2. **Main Libraries**

   - **Collective Communication (`c10d`)**

     - Low-level APIs for communication primitives (e.g., `all_reduce`).

   - **Distributed Data-Parallel (DDP)**

     - Simplifies multi-GPU training by automatically handling gradient synchronization.

   - **RPC-based Distributed Training**

     - Rarely used for model parallel or parameter server approaches.

### **Important Functions**

- **Initialize Process Group**

  ```python
  torch.distributed.init_process_group(backend="nccl")
  ```

  - Creates a default global process group for communication.
  - Each process in the group participates in collective operations.

- **Create New Group**

  ```python
  group = torch.distributed.new_group(...)
  ```

  - Allows creating a subgroup of processes for communication.

- **AllReduce**

  ```python
  dist.all_reduce(tensor, op=ReduceOp.SUM, async_op=False)
  ```

  - Aggregates values across processes (e.g., sum, mean).
  - `async_op=True` returns a handle for asynchronous execution.

### **Sync vs Async Operations**

- **Synchronous (`async_op=False`)**

  - Function returns after the operation is complete.

- **Asynchronous (`async_op=True`)**

  - Returns a handle; allows overlapping computation with communication.
  - Use `handle.wait()` to ensure completion.

### **DistributedDataParallel (DDP)**

DDP simplifies data parallelism by:

- Replicating the model on each GPU.
- Splitting input data automatically.
- Synchronizing gradients with AllReduce.

Example:

```python
torch.cuda.set_device(i)
torch.distributed.init_process_group(backend='nccl', world_size=N)
model = DistributedDataParallel(model, device_ids=[i], output_device=i)
```

### **Launching Multiple Processes**

Each process requires environment variables to communicate:

- `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, etc.

**Example (SLURM):**

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4

export MASTER_PORT=12345
export WORLD_SIZE=8
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

source ~/miniconda/etc/profile.d/conda.sh
conda activate myenv
srun python my_training_script.py
```

### **Collective Communication Example**

```python
tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
dist.all_reduce(tensor, op=ReduceOp.SUM)
print(tensor)  # Output will be summed across ranks
```
