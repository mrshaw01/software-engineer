# What is CuPy, and how is it related to NumPy?

**CuPy** is a **GPU-accelerated numerical computing library** that is designed to be a **drop-in replacement for NumPy**, but with operations executed on **NVIDIA GPUs using CUDA**.

### Key Relationship to NumPy:

| Feature           | **NumPy**                   | **CuPy**                                    |
| ----------------- | --------------------------- | ------------------------------------------- |
| Execution target  | CPU                         | GPU (NVIDIA, via CUDA)                      |
| API compatibility | Native                      | Nearly identical (mirrors NumPy API)        |
| Performance       | Fast on CPU                 | Much faster on GPU for large arrays         |
| Use case          | General numerical computing | GPU-accelerated computation (DL, HPC, etc.) |

### How CuPy Works:

- Implements most of NumPy’s API: `cupy.array`, `cupy.dot`, `cupy.fft`, etc.
- Under the hood, it uses **CUDA libraries** like:

  - **cuBLAS** for linear algebra
  - **cuFFT** for fast Fourier transforms
  - **cuRAND** for random number generation
  - **Thrust/CUB** for scan/sort operations

### Example: Minimal Code Change

```python
# NumPy version (CPU)
import numpy as np
x = np.random.rand(10000)
y = np.sqrt(x)

# CuPy version (GPU)
import cupy as cp
x = cp.random.rand(10000)
y = cp.sqrt(x)
```

> Just switch `np` → `cp`, and your code runs on the GPU.

### Notes and Caveats:

- CuPy arrays (`cp.ndarray`) live in **GPU memory**; you must transfer data between host and device explicitly:

```python
x_cpu = cp.asnumpy(x_gpu)  # to CPU
x_gpu = cp.asarray(x_cpu)  # to GPU
```

- Not **every** NumPy function is supported, but most are — especially those involving array math, linear algebra, FFT, and random.

- Works well with **SciPy-compatible functions** (e.g., `cupyx.scipy.linalg`), and can be combined with frameworks like **PyTorch**, **Dask**, or **RAPIDS**.

### Summary:

> **CuPy is a GPU-powered version of NumPy**, offering a nearly identical API while running computations on **NVIDIA GPUs**. It enables users to write **high-performance CUDA-accelerated numerical code in Python** with minimal changes to their existing NumPy workflows.
