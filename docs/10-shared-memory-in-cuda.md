# What is the role of shared memory in CUDA kernels?

### What is Shared Memory in CUDA?

**Shared memory** in CUDA is a small, fast, **on-chip memory** that is shared among all threads in the **same thread block**.

It acts like a **programmable cache**, enabling threads to **cooperate** and **reuse data** without repeatedly accessing slower **global memory**.

### Key Characteristics:

| Feature  | Description                       |
| -------- | --------------------------------- |
| Scope    | Visible to all threads in a block |
| Latency  | Much lower than global memory     |
| Size     | Typically 48–100 KB per SM        |
| Lifetime | Kernel-duration, block-specific   |

### Why Use Shared Memory?

1. **Avoid Redundant Global Memory Access**

   - Threads can load global data **once into shared memory** and reuse it multiple times.

2. **Enable Thread Cooperation**

   - Threads can **collaboratively compute** using shared data (e.g., reduction, matrix tiling).

3. **Improve Memory Access Patterns**

   - You can **restructure global memory access** into more efficient coalesced patterns using shared memory as a staging buffer.

### Common Use Case: Matrix Multiplication

```cpp
__shared__ float tileA[BLOCK_SIZE][BLOCK_SIZE];
__shared__ float tileB[BLOCK_SIZE][BLOCK_SIZE];

tileA[ty][tx] = A[row][k];
tileB[ty][tx] = B[k][col];
__syncthreads();
```

- Tiles of A and B are loaded into shared memory.
- Threads compute partial results collaboratively, reducing global memory traffic.

### Optimization Tips:

- **Avoid bank conflicts**: Shared memory is split into banks. If multiple threads access the same bank, access becomes serialized.
- **Pad arrays** to avoid conflicts in some layouts.

### Summary:

> **Shared memory** in CUDA is a low-latency, on-chip memory used to **speed up data reuse and collaboration within thread blocks**. It’s essential for performance-critical kernels, like matrix multiplications and reductions, where global memory latency would otherwise be a bottleneck.
