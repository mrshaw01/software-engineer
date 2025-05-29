# How do memory access patterns affect performance on modern CPUs?

Memory access patterns have a **significant impact** on performance in modern CPUs due to the **memory hierarchy** (registers → L1/L2/L3 cache → DRAM) and **cache line-based prefetching**.

### Key Concepts

#### 1. **Cache Locality**

Modern CPUs load data in **cache lines** (typically 64 bytes). Performance is best when accesses:

- Are **contiguous**
- Exhibit **spatial and temporal locality**

### ⚠️ Poor Memory Access Patterns

#### ❌ Strided Access:

```python
# Access every k-th element
for i in range(0, len(arr), 8):
    process(arr[i])
```

- Causes **cache misses**, as accessed elements are far apart in memory.
- Prevents efficient use of hardware prefetchers.

#### ❌ Random Access:

```python
for i in random_indices:
    process(arr[i])
```

- Completely defeats caching and leads to many **DRAM fetches**.

### Optimized Patterns

#### ✔️ Contiguous Access (Row-major in NumPy by default):

```python
for i in range(len(arr)):
    process(arr[i])
```

- Maximizes **cache line reuse**
- Enables **hardware prefetching**

### 🔄 Structures of Arrays vs Arrays of Structures

- **Structure of Arrays (SoA)** improves performance over **Array of Structures (AoS)** in many cases:

```python
# AoS (less cache-friendly)
class Particle:
    x, y, z, mass

# SoA (better memory layout)
x_array = [...]
y_array = [...]
```

This is especially true in SIMD/vectorized code.

### ⚙️ Optimization Techniques

- **Loop blocking / tiling**: Break work into cache-sized chunks.
- **Alignment**: Use aligned data structures to match cache line size.
- **SIMD**: Use vector instructions (AVX, SSE) that benefit from contiguous data.
- **Prefetching**: Manually prefetch or design for hardware prefetchers.

### 🧪 Example in NumPy

```python
# Fast: access along rows (C-order)
np.sum(array, axis=1)

# Slow: access along columns (C-order)
np.sum(array, axis=0)
```

### Summary:

> On modern CPUs, **memory access patterns affect cache efficiency**. Contiguous, sequential access patterns maximize **cache hits and prefetching**, while strided or random access leads to **cache misses and performance degradation**. Structuring data and algorithms to optimize for **spatial locality** is essential for high-performance CPU code.
