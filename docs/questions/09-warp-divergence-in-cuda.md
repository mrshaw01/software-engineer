# Describe the concept of warp divergence and its impact on GPU performance.

### What is Warp Divergence?

On NVIDIA GPUs, a **warp** is a group of **32 threads** that execute **in lockstep** — meaning all threads in the warp execute the **same instruction** at the **same time**.

**Warp divergence** occurs when threads in the same warp **take different execution paths** due to conditional logic (e.g., `if/else`, `switch`, loops).

### Example of Warp Divergence:

```cpp
int tid = threadIdx.x;

if (tid % 2 == 0) {
    // Half the warp executes this
    do_even_work();
} else {
    // The other half waits
    do_odd_work();
}
```

> Result: Threads must **serialize** — first, all even threads run while odd ones are idle, then vice versa.

### Impact on Performance:

- GPU **serializes** divergent paths within a warp.
- This **reduces effective parallelism** — while some threads are active, others are **idle**.
- Performance degradation depends on how many divergent paths there are and how unbalanced the workload is.

### How to Minimize Warp Divergence:

1. **Avoid divergent conditionals** inside warps when possible.

   - Use **predication** (evaluate both branches but commit results selectively).

2. **Reorganize data** so threads in the same warp follow the same code path.
3. Replace conditionals with **bitwise ops**, **select functions**, or **lookup tables** if possible.
4. Use **warp-level primitives** like `__shfl_sync()` or `__ballot_sync()` for communication without branching.

### Summary:

> **Warp divergence** happens when threads in a warp take **different execution paths**, causing them to **serialize execution** and lose parallel efficiency. Reducing divergence is key to writing high-performance CUDA code — especially in control-flow-heavy kernels.
