# What is Numba and how can it speed up Python code?

**Numba** is a **Just-In-Time (JIT) compiler** for Python that translates a subset of Python and NumPy code into **fast machine code** using **LLVM**, enabling significant speedups — especially for numerical computations.

### How Numba Works

- You write Python functions as usual.
- Add a **`@jit` decorator** (or `@njit` for no Python fallback).
- Numba compiles the function to **native code at runtime**, eliminating Python interpreter overhead.

### Example

```python
from numba import njit
import numpy as np

@njit
def sum_squared(x):
    total = 0.0
    for i in range(x.shape[0]):
        total += x[i] * x[i]
    return total

x = np.random.rand(1000000)
print(sum_squared(x))
```

> This runs **10x–100x faster** than a pure Python loop.

### Why Numba Is Fast

| Feature             | Benefit                                    |
| ------------------- | ------------------------------------------ |
| LLVM Backend        | Generates efficient low-level machine code |
| Loop Optimization   | Eliminates Python loop overhead            |
| SIMD & Parallelism  | Auto-vectorization and multithreading      |
| Type Specialization | Compiles to fast, type-specific code       |

### Key Features

- **`@jit`, `@njit`**: Compile numerical functions
- **`parallel=True`**: Enable automatic multithreading
- **`prange`**: Parallel loop hint
- **`numba.cuda`**: Write CUDA kernels in Python for GPU execution
- **Supports NumPy**: Works seamlessly with NumPy arrays and ufuncs

### Limitations

- Only supports a **subset of Python** (no lists of lists, no classes unless compiled with `@jitclass`).
- Best suited for **numerical, array-oriented code**.
- Doesn't accelerate everything (e.g., I/O, string processing).

### Summary:

> **Numba** compiles Python functions to **fast native code** using LLVM, enabling performance near that of C or Fortran — especially for **numerical loops, array operations, and CUDA kernels**. It’s an excellent tool for speeding up Python without leaving the Python ecosystem.
