# Introduction to CUDA

## CUDA Overview

- Parallel programming model for NVIDIA GPUs

### Compute Unified Device Architecture (CUDA)

- CUDA 1.0 released in Nov 2006
- CUDA 12.3 released in Oct 2023

### Popularity

- Most widely used GPU programming model

  - NVIDIA GPUs dominate the market
  - Highly vendor-dependent (NVIDIA)
  - AMD provides HIP, a similar GPU programming model

## GPU Parallelism

### Concept of Massive Parallelism

- **Use Case**: Graphics rendering

  - Processes millions of triangles and pixels
  - Requires massive parallelism

- **Architecture**:

  - Each core is simple
  - A large number of simple shader cores are used

### Architecture for Parallelism (continued)

- GPU design supports massively parallel applications
- A single GPU chip has thousands of cores

  - High throughput from many simple cores

## General-Purpose GPU (GPGPU)

### What is GPGPU?

- General Purpose Graphic Processing Unit
- GPUs used for general applications, not just graphics

### Comparison: CPU vs GPU

#### CPU

- Latency-optimized
- Complex control logic
- Large memory capacity

#### GPU

- Throughput-optimized
- Many simple computations
- High memory bandwidth

### Note

- GPUs are not always faster than CPUs

  - Best suited for GPU-friendly tasks: graphics, scientific computing, deep learning

## GPU Performance Trends

- Floating-point performance continues to rise rapidly

  - **NVIDIA A100**:

    - 19.5 TFLOPS (FP32)
    - 312 TFLOPS (FP16)

  - **NVIDIA H100**:

    - 66.9 TFLOPS (FP32)
    - 989.4 TFLOPS (FP16)

## Real-World Performance Gains

| Application                                | CPU Time | CPU+GPU Time | Speedup     |
| ------------------------------------------ | -------- | ------------ | ----------- |
| Computation Chemistry (UIUC)               | 4.6 days | 27 mins      | 245× faster |
| Neurological Modeling (Evolved Machines)   | 2.7 days | 30 mins      | 130× faster |
| Cell Phone RF Simulation (Nokia, Motorola) | 8 hours  | 13 mins      | 37× faster  |
| 3D CT Ultrasound (Techniscan)              | 3 hours  | 16 mins      | 13× faster  |

## GPU System Architecture

- Typical system: 1~2 CPUs and 1~8 GPUs
- Connected via PCIe interconnect
- CPUs use GPUs as co-processors

## GPU Architecture

- GPUs contain multiple **Streaming Multiprocessors (SMs)**

  - Example: V100 has 80 SMs

- Each SM includes:

  - Arithmetic units
  - Caches
  - Warp schedulers
  - Other compute resources

## Inside the Streaming Multiprocessor (SM)

### CUDA Cores

- Basic arithmetic units
- V100 SM includes:

  - 32× FP64 CUDA cores
  - 64× FP32 CUDA cores
  - 64× INT CUDA cores

### Tensor Cores

- High-performance, mixed-precision units

  - E.g., FP32 × FP32 → FP16

- Common in deep learning

  - Useful where perfect accuracy is not required

### L1 Data Cache and Shared Memory

- High-bandwidth cache
- Also functions as scratch-pad memory

## SIMT Model: How SMs Execute

- **SIMT (Single Instruction Multiple Threads)** architecture
- **Warp** is the basic execution unit (not a thread)

  - A warp consists of 32 consecutive threads

- All threads in a warp execute the same instruction simultaneously

  - Share the same fetch-decode unit
