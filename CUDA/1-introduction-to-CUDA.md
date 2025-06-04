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

## CUDA Application Example - Addition

### CUDA Application for simple addition

```cpp
__global__ void kernel_add(const int *a, const int *b, int *c) {
    *c = *a + *b;
}

int main() {
    int a = 1, b = 2, c;
    int *d_a, *d_b, *d_c;

    // 1. Allocate device memory
    cudaMalloc(&d_a, sizeof(int));
    cudaMalloc(&d_b, sizeof(int));
    cudaMalloc(&d_c, sizeof(int));

    // 2. Transfer input data to device memory
    cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);

    // 3. Execute kernel
    kernel_add<<<1,1>>>(d_a, d_b, d_c);

    // 4. Transfer output data to host memory
    cudaMemcpy(&c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
    printf("c: %d\n", c);
}
```

## Compilation Process in CUDA

- **Separate compilation**
  Compile twice for GPU code (kernels) and CPU code (host functions)

```text
GPU compile                              CPU compile
------------                            -------------
.cu                                     .cu
 ↓                                        ↓
C++ Preprocessor                        C++ Preprocessor
 ↓ .cpp1.i                                ↓ .cpp4.ii
cicc                                    cudafe++
 ↓ .ptx (IR)                              ↓ .cudafe1.cpp
ptxas                                   C++ Compiler
 ↓ .cubin (binary)                        ↓ .o / .obj
fatbinary
```

- **CUDA C/C++ kernel → ptx assembly → cubin → fatbin**

### ptx (Parallel Thread Execution)

- Assembly for GPU
- Specify a virtual GPU architecture at the ISA level
- Example:
  `nvcc --gpu-architecture=compute_70`

### cubin (CUDA Binary)

- Object file for GPU
- Specify a real GPU architecture
- Example:
  `nvcc --gpu-code=sm_70`

### fatbin

- Collection of cubin and ptx files
- Save multiple versions using fatbin
- Supports multiple GPU architectures in a single binary

## Practice 1: Figuring Out Device Information

### 1. Identify GPUs on the Practice Server

Use the following commands to check GPU types and counts:

```bash
nvidia-smi
nvidia-smi -q
```

### 2. Submit Commands via `srun`

Use `srun` to execute the same commands on compute nodes:

```bash
srun nvidia-smi
srun nvidia-smi -q
```

### 3. Analyze Theoretical GPU Performance

Check the following hardware specifications:

- **Number of SMs** (Streaming Multiprocessors)
- **Number of CUDA cores**
- **FP16 / FP32 / FP64 performance** (Theoretical peak FLOPS)
- **Memory capacity** (in GB)
- **Memory bandwidth** (in GB/s)

### 4. Reference Architecture Documentation

Look up detailed specs in the official whitepaper:

> Search for: `"V100 whitepaper"`
> URL: [https://www.nvidia.com/en-us/data-center/v100/](https://www.nvidia.com/en-us/data-center/v100/)

## Practice 2: Hello, World!

Run “Hello, World!” programs on both CPU and GPU to understand basic CUDA compilation and execution.

### Skeleton Path

```bash
/home/scratch/getp/hello_world
```

> ⚠️ Copy the files to your home directory before compiling.

### Compilation

```bash
nvcc -o hello_world hello_world.cu
nvcc -o hello_parallel_world hello_parallel_world.cu
```

### Execution

```bash
srun ./hello_world
srun ./hello_parallel_world
```

### Sample Output

```bash
$ srun ./hello_world
Host(CPU): Hello World!

$ srun ./hello_parallel_world
Device(GPU) Thread 4: Hello, World!
Device(GPU) Thread 5: Hello, World!
Device(GPU) Thread 6: Hello, World!
Device(GPU) Thread 7: Hello, World!
Device(GPU) Thread 0: Hello, World!
Device(GPU) Thread 1: Hello, World!
Device(GPU) Thread 2: Hello, World!
Device(GPU) Thread 3: Hello, World!
```

### Discussion

- **Why is the output not in order?**
  CUDA threads execute in parallel and independently. Output order depends on thread scheduling, which is nondeterministic.

- **What happens if `cudaDeviceSynchronize()` is not called?**
  The host may exit before the GPU completes execution, causing incomplete or missing output.

### Optional Exploration

- Vary the **number of threads**

  - Change the block size
  - Adjust the number of threads per block

- Observe output patterns and thread execution order

## Practice 3: Compile and Linking

Compile a mixed-source program with both C++ and CUDA code.

### Skeleton Path

```bash
/home/scratch/getp/compile_and_linking
```

> Contains `file1.cpp`, `file2.cu`, and `compile.sh`

### Steps

```bash
$ ./compile.sh
$ srun ./main
```

### Expected Output

```
(Host) Welcome to GETP!
(Device) Welcome to GETP!
```

### Follow-Up Questions (With Answers)

- **Where is the fatbin stored?**
  The fatbin (fat binary) is embedded inside the final executable by `nvcc`. You can inspect it with `cuobjdump --dump-fatbin`.

- **How is kernel code linked during runtime?**
  The CUDA runtime uses the fatbin section to load and launch device code on the GPU. The device code is dynamically linked into the GPU context during kernel launch.

- **What is `nvlink`?**
  `nvlink` is **not** a dynamic linker.

  > It is a high-speed **interconnect** developed by NVIDIA that allows fast data transfer between GPUs or between CPU and GPU.
  > For dynamic linking in CUDA, `cuModuleLoad()` and `cuModuleGetFunction()` are used instead.
