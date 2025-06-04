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

## CUDA Programming Model

- **CUDA programming model**

  - Defines kernels, threads, execution, and memory models
  - Required to write programs for NVIDIA GPUs

- **CUDA API**

  - C/C++ based API (Fortran is also supported)
  - Interfaces for memory management, streams, and device control

- **CUDA runtime**

  - Execution environment for CUDA
  - Provided by NVIDIA as a runtime library

## Key Terms in CUDA Programming

- **Host**

  - Processor running the main program (typically CPU)

- **Device**

  - Hardware that runs CUDA kernels (typically NVIDIA GPU)

- **Kernel**

  - Function executed on the device
  - Written in CUDA C/C++
  - Executed in parallel by many CUDA threads

## CUDA Applications

- **Structure**

  - Host Program (runs on CPU)
  - CUDA Kernel (runs on GPU)

```cpp
// Host Program
int main() {
  cudaMalloc(...);
  cudaMemcpy(...);
  kernel_1<<<...>>>();
  kernel_2<<<...>>>();
  cudaMemcpy(...);
}
```

```cpp
// CUDA Kernels
__global__ void kernel_1(...) { ... }
__global__ void kernel_2(...) { ... }
```

- **Host Program**

  - Written in C/C++
  - Requests tasks (computation, data transfer, sync) via CUDA APIs

- **Kernel**

  - Basic device-executable code unit
  - Also written in CUDA C/C++

- **Execution**

  - Host and kernels can run in parallel
  - Multiple GPUs can also operate in parallel

## CUDA Toolkit

- Development tools for CUDA programming

  - Free to download

- **Compiler (nvcc)**

  - NVIDIA Compiler Collection
  - Includes compiler and linker
  - Interface similar to gcc/clang

- **Debugger**

  - `cuda-memcheck`, `cuda-gdb`

- **Profiler**

  - `nsys` (Nsight Systems), `ncu` (Nsight Compute)

- **Other tools**

  - Various utilities bundled in the toolkit

## CUDA C/C++

- **Programming language for CUDA kernels**

  - Based on ISO C++ (C++14, C++17) with some extensions
  - Standard Template Library (STL) not supported in device code

    - Use libraries like `thrust` instead

- **Language Extensions**

  - Vector types: `char2`, `int4`, `float4`, ...
  - Function/memory qualifiers: `__global__`, `__device__`, `__host__`, `__constant__`, ...
  - Synchronization: `__syncthreads()`, `__thread_fence()`
  - Built-in functions: `threadIdx`, `sin()`, `atomicAdd()`, `printf()`, ...

## CUDA C/C++ Program

- **Ordinary C/C++ Program**

  - Defines functions
  - Main function runs first, calls other functions

- **CUDA C/C++ Program**

  - Regular C/C++ with kernel functions
  - Host function launches kernels

    ```cpp
    kernel<<<...>>>(...);
    ```

  - Multiple threads run the kernel in parallel

## Hello, World! Example

```cpp
// hello_world.cu
#include <cstdio>

int main() {
  printf("Hello, World!\n");
  return 0;
}
```

```sh
$ nvcc -o hello_world hello_world.cu
$ ./hello_world
```

## Hello, Parallel World!

- Define a kernel to print "Hello, World!" from GPU threads

```cpp
__global__ void hello_world() {
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  printf("Device(GPU) Thread %d: Hello, World!\n", tidx);
}

int main() {
  hello_world<<<2, 4>>>();
  cudaDeviceSynchronize();
  return 0;
}
```

```sh
$ nvcc -o hello_parallel_world hello_parallel_world.cu
$ ./hello_parallel_world
```

## How CUDA Applications Work

### On Host (CPU)

1. Create objects for kernel execution

   - Stream creation
   - Device memory allocation

2. Execute kernel functions

   - Transfer input data to device memory
   - Launch kernel(s)
   - Transfer output back to host memory

### On Device (GPU)

- Execute the launched kernel(s) in parallel
