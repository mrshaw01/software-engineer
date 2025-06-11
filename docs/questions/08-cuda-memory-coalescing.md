# How does memory coalescing work in CUDA, and why is it important?

**Memory coalescing** in CUDA is a technique that enables threads in a **warp** to access global memory **efficiently** by grouping their memory requests into as few transactions as possible.

### What is a Warp?

- A **warp** = 32 threads that execute in **lockstep** on NVIDIA GPUs.
- All threads in a warp issue memory instructions together.

### What Is Memory Coalescing?

When threads in a warp access **consecutive and properly aligned** memory addresses, the memory controller can **coalesce** (combine) those individual memory accesses into a **single memory transaction**.

#### Example: Coalesced Access

```cpp
// Each thread i accesses A[i]
int idx = threadIdx.x;
float value = A[idx];
```

If `A` is stored contiguously in memory, this access is **coalesced**.

#### ❌ Example: Non-Coalesced Access

```cpp
// Each thread i accesses A[i * stride]
int idx = threadIdx.x;
float value = A[idx * 7];  // Irregular stride
```

This leads to **non-coalesced** (scattered) accesses → **multiple slow transactions**.

### Why Is It Important?

- **Global memory is slow** compared to shared or register memory.
- Coalesced access uses fewer memory transactions → **higher bandwidth utilization**.
- Non-coalesced access causes:

  - More memory fetches
  - Higher latency
  - Lower throughput
  - Poor occupancy and performance

### Best Practices for Coalescing

- Ensure **thread i accesses memory location i** (or i + constant stride).
- Use **Structure of Arrays (SoA)** instead of **Array of Structures (AoS)** to enable linear access.
- Align memory to **128-bit or 256-bit** boundaries (depending on architecture).
- Process data in **tiles** and load it into **shared memory** for reuse.

### Summary:

> **Memory coalescing** combines multiple global memory requests from a **warp** into **fewer transactions**, drastically improving memory bandwidth efficiency. It's critical for achieving high performance in CUDA kernels. Coalesced memory access patterns are often the difference between a slow and a fast CUDA program.
