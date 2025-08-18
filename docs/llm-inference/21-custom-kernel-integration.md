# Custom Kernel Integration

This section explains how to design, implement, and integrate custom GPU kernels (CUDA/HIP) into an LLM inference stack. It covers kernel APIs, portability, launch configuration, graph capture, autotuning, testing, profiling, and packaging for production.

## 21.1 When to Write a Custom Kernel

Prefer libraries (cuBLAS, cuDNN, rocBLAS, MIOpen, CUTLASS/Triton) for dense GEMM/convolution. A custom kernel is justified when you need:

- **Fusions** that vendor libraries do not provide (e.g., RMSNorm + bias + activation + dequant).
- **Irregular memory access** (scatter/gather, block-sparse).
- **Quantized paths** with nonstandard formats (NF4, mixed per-channel scales, grouped scales).
- **Small/skinny shapes** where GEMM underutilizes the GPU.
- **Latency-critical epilogues** (e.g., fast-token decode micro-kernels).

## 21.2 Kernel API Design (Stable ABI)

Define a minimal, stable C ABI to decouple kernel code from higher layers (PyTorch/vLLM/etc.). Keep the interface explicit about shapes, strides, and data types.

```c
// abi.h
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  DT_F16 = 0, DT_BF16 = 1, DT_F32 = 2, DT_I8 = 3
} dtype_t;

typedef struct {
  int64_t n;        // number of rows/elements
  int64_t d;        // hidden size or vector width
  int64_t stride;   // leading dimension (in elements)
} tensor_desc_t;

typedef struct {
  void* stream;     // cudaStream_t or hipStream_t as void*
  void* workspace;  // optional temporary buffer
  size_t workspace_bytes;
} launch_ctx_t;

// Return 0 on success; nonzero on error.
int rmsnorm_launch(
  void* y, const void* x, const void* weight,
  tensor_desc_t desc, float eps, dtype_t dt, launch_ctx_t ctx);

#ifdef __cplusplus
}
#endif
```

Guidelines:

- Use **plain pointers** and explicit descriptors.
- Include an opaque **stream** and **workspace**.
- Avoid global state; all inputs provided per-call.
- Keep **return codes** for robust error handling.

## 21.3 Cross-Platform Portability (CUDA & HIP)

Use a single codebase compiled for both backends. Wrap differences with lightweight macros.

```cpp
// backend.h
#pragma once
#if defined(USE_CUDA)
  #include <cuda.h>
  #include <cuda_fp16.h>
  #define DEV __device__
  #define GLOB __global__
  #define ALIGN(x) __align__(x)
  #define WARP_SIZE 32
  using stream_t = cudaStream_t;
  #define LAUNCH(kernel, grid, block, smem, stream, ...) \
      (kernel)<<<(grid),(block),(smem),(stream)>>>(__VA_ARGS__)
  #define ASSERT_LAST_ERROR() do { auto e = cudaGetLastError(); if (e) return (int)e; } while(0)
#elif defined(USE_HIP)
  #include <hip/hip_runtime.h>
  #include <hip/hip_fp16.h>
  #define DEV __device__
  #define GLOB __global__
  #define ALIGN(x) __attribute__((aligned(x)))
  #define WARP_SIZE 64
  using stream_t = hipStream_t;
  #define LAUNCH(kernel, grid, block, smem, stream, ...) \
      hipLaunchKernelGGL(kernel, grid, block, smem, stream, __VA_ARGS__)
  #define ASSERT_LAST_ERROR() do { auto e = hipGetLastError(); if (e) return (int)e; } while(0)
#else
  #error "Define USE_CUDA or USE_HIP"
#endif
```

Compilation flags:

- CUDA: `-O3 --use_fast_math -Xptxas -O3 -Xptxas -dlcm=ca`
- HIP: `-O3 -ffast-math --amdgpu-target=gfx90a,gfx942` (match your hardware)

## 21.4 Example: High-Throughput RMSNorm (Fused)

A fused kernel that computes RMSNorm and writes the normalized output (optionally with scale). It uses block-wise parallel reduction and vectorized loads/stores.

```cpp
// rmsnorm.cu/hip
#include "backend.h"
#include <stdint.h>

// Vectorized accessor
template<typename T, int V>
struct Vec { T x[V]; };

template<int V> struct fp_traits {};
template<> struct fp_traits<2> { using f16v = __half2; };
template<> struct fp_traits<4> { struct f16x4 { __half x[4]; }; };

template<int V>
DEV inline float to_float_acc(__half v) {
  return __half2float(v);
}

template<>
DEV inline float to_float_acc<2>(__half2 v2) {
#if defined(USE_CUDA)
  return __half2float(__low2half(v2)) + __half2float(__high2half(v2));
#else
  return __half2float(v2.x) + __half2float(v2.y);
#endif
}

template<int V>
GLOB void rmsnorm_kernel_f16(
    __half* __restrict__ y,
    const __half* __restrict__ x,
    const __half* __restrict__ gamma,
    int64_t n, int64_t d, int64_t stride,
    float eps)
{
  // Each block handles one row; each thread reduces over columns.
  int64_t row = blockIdx.x;
  if (row >= n) return;

  const __half* xrow = x + row * stride;
  __half* yrow       = y + row * stride;

  // Thread-local sum of squares
  float ssq = 0.f;

  // Stride in vector units
  int tid = threadIdx.x;
  int nthreads = blockDim.x;
  for (int64_t col = tid; col < d; col += nthreads) {
    __half xv = xrow[col];
    float xf = __half2float(xv);
    ssq += xf * xf;
  }

  // Block reduction (shared memory)
  extern __shared__ float smem[];
  smem[tid] = ssq;
  __syncthreads();

  // Simple tree reduction
  for (int offset = nthreads >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) smem[tid] += smem[tid + offset];
    __syncthreads();
  }

  float inv_rms = rsqrtf(smem[0] / (float)d + eps);

  // Normalize + scale
  for (int64_t col = tid; col < d; col += nthreads) {
    float xf = __half2float(xrow[col]);
    float g  = gamma ? __half2float(gamma[col]) : 1.f;
    yrow[col] = __float2half(xf * inv_rms * g);
  }
}

int rmsnorm_launch(
  void* y, const void* x, const void* weight,
  tensor_desc_t desc, float eps, dtype_t dt, launch_ctx_t lctx)
{
  if (dt != DT_F16) return 100; // only f16 in this example

  const int64_t n = desc.n, d = desc.d, ld = desc.stride;
  if (n <= 0 || d <= 0 || ld < d) return 101;

  stream_t stream = (stream_t)lctx.stream;
  const int block = 256;
  const dim3 grid((unsigned int)n);
  size_t smem = block * sizeof(float);

  LAUNCH((rmsnorm_kernel_f16<1>), grid, block, smem, stream,
     (__half*)y, (const __half*)x, (const __half*)weight,
     n, d, ld, eps);

  ASSERT_LAST_ERROR();
  return 0;
}
```

Notes:

- One block per row (good for decode-time batch sizes and moderate `d`).
- Shared-memory reduction; tune `block` for the target GPU.
- Support BF16/FP32/I8 via `dtype_t` dispatch (omitted for brevity).
- Optional fusion points: add bias, activation, de/quantization, residual add.

## 21.5 C++ Launcher and Python Binding (PyTorch)

Provide a thin C++ wrapper that loads the ABI and exposes a PyTorch custom op. This keeps PyTorch optional—your engine can also call the C ABI directly.

```cpp
// torch_binding.cpp
#include <torch/extension.h>
#include "abi.h"

torch::Tensor rmsnorm_torch(torch::Tensor x, torch::Tensor gamma, double eps) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA/HIP tensor");
  TORCH_CHECK(x.scalar_type() == torch::kHalf, "only f16 in this example");
  TORCH_CHECK(x.dim() == 2, "x: [n, d]");
  auto n = x.size(0), d = x.size(1);
  auto y = torch::empty_like(x);

#if defined(USE_CUDA)
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#elif defined(USE_HIP)
  hipStream_t stream = at::hip::getCurrentHIPStream();
#endif

  tensor_desc_t desc{n, d, (int64_t)x.stride(0)};
  launch_ctx_t ctx{(void*)stream, nullptr, 0};

  dtype_t dt = DT_F16;
  int rc = rmsnorm_launch(
    y.data_ptr(), x.data_ptr(), gamma.defined()? gamma.data_ptr(): nullptr,
    desc, (float)eps, dt, ctx);
  TORCH_CHECK(rc == 0, "rmsnorm_launch failed with code ", rc);
  return y;
}

TORCH_LIBRARY(llmops, m) {
  m.def("rmsnorm(Tensor x, Tensor gamma, float eps) -> Tensor");
}

TORCH_LIBRARY_IMPL(llmops, CUDA, m) {
  m.impl("rmsnorm", rmsnorm_torch);
}
```

**CMakeLists.txt** (unified for CUDA/HIP):

```cmake
cmake_minimum_required(VERSION 3.20)
project(llm_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
option(USE_CUDA "Build with CUDA" ON)
option(USE_HIP  "Build with HIP" OFF)

find_package(Torch REQUIRED)

if (USE_CUDA)
  enable_language(CUDA)
  add_definitions(-DUSE_CUDA)
  set(BACKEND_SOURCES rmsnorm.cu torch_binding.cpp)
  set_source_files_properties(rmsnorm.cu PROPERTIES LANGUAGE CUDA)
  add_library(llm_kernels SHARED ${BACKEND_SOURCES})
  target_compile_options(llm_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)
elseif(USE_HIP)
  # Using hip as CXX compiler
  set(CMAKE_CXX_COMPILER hipcc)
  add_definitions(-DUSE_HIP)
  add_library(llm_kernels SHARED rmsnorm.cu torch_binding.cpp)
  target_compile_options(llm_kernels PRIVATE -O3 -ffast-math)
else()
  message(FATAL_ERROR "Select USE_CUDA or USE_HIP")
endif()

target_include_directories(llm_kernels PRIVATE ${TORCH_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(llm_kernels "${TORCH_LIBRARIES}")
set_property(TARGET llm_kernels PROPERTY CXX_STANDARD 17)
```

Python usage:

```python
import torch
import importlib
torch.ops.load_library("./libllm_kernels.so")
y = torch.ops.llmops.rmsnorm(x, gamma, 1e-5)
```

## 21.6 Launch Configuration & Occupancy

Checklist:

- Choose **block size** to balance latency and occupancy; start with 128–512 threads.
- Ensure **coalesced** global memory access (contiguous in the fastest-moving dimension).
- Use **vectorized loads/stores** when aligned (e.g., `float2/float4`, `__half2`).
- Prefer **shared-memory** reductions; avoid atomics on global memory for hot paths.
- Avoid excessive register pressure; watch SASS/ISA and occupancy reports.

## 21.7 CUDA/HIP Graph Capture Compatibility

To enable graph capture for steady-state decode loops:

- No dynamic allocations inside the kernel or launcher.
- Fixed shapes and workspace sizes during capture.
- Use the provided **stream**; do not create/destroy streams inside.
- For CUDA:

  - Ensure all API calls in the launch path are **stream-capture safe**.

- For HIP:

  - Use ROCm stream capture support on versions that implement it; otherwise, prebuild launch graphs.

Pseudo-code:

```cpp
// capture_once_then_replay.cpp
void* graph_exec = nullptr;
if (!graph_exec) {
  // Begin capture on stream
  // enqueue rmsnorm_launch + other kernels
  // End capture -> instantiate graph_exec
}
// On each token: replay graph on the same stream
```

## 21.8 Autotuning

Select launch parameters at runtime and cache by shape/dtype/arch.

```cpp
struct Key { int d; int device; dtype_t dt; };
struct Tuned { int block; };

Tuned tune_rmsnorm(int d, int device, dtype_t dt) {
  int candidates[] = {128, 256, 512};
  float best = 1e30f; Tuned out{256};
  for (int b: candidates) {
    // time N warmup + M measure launches
    // pick the lowest latency
  }
  return out;
}
```

Persist results to a small JSON alongside the engine; reload at startup.

## 21.9 Numerical Considerations

- Prefer **F32 accumulation** for RMS/variance; write in target dtype.
- Clamp/epsilon: pick `eps` \~ `1e-5` to avoid denorms while preserving accuracy.
- When quantized:

  - Apply **per-channel scales** before normalization if mathematically required.
  - Beware of overflow when dequantizing `int8` to `fp16` (use F32 intermediates).

## 21.10 Testing & Verification

**Correctness**

- Reference implementation in Python (Torch) for bitwise/ULP comparisons.
- Test across shapes: small/large `d`, non-contiguous strides, broadcasted weights.
- Validate on both backends if you claim portability.

**Performance**

- Microbench with event timing (exclude H2D/D2H).
- Compare against:

  - Pure PyTorch layer (baseline).
  - Fused alternatives (e.g., Triton kernel).

Example unit test (PyTorch):

```python
import torch, math
torch.ops.load_library("./libllm_kernels.so")

def rmsnorm_ref(x, w, eps):
    ssq = (x.float() * x.float()).mean(dim=-1, keepdim=True)
    inv = torch.rsqrt(ssq + eps)
    y = x * inv
    return y * w if w is not None else y

for n in [1, 4, 32]:
    for d in [64, 256, 4096]:
        x = torch.randn(n, d, device="cuda", dtype=torch.float16)
        w = torch.randn(d, device="cuda", dtype=torch.float16)
        y = torch.ops.llmops.rmsnorm(x, w, 1e-5)
        ref = rmsnorm_ref(x, w, 1e-5).half()
        max_abs = (y - ref).abs().max().item()
        assert max_abs < 2e-2, (n, d, max_abs)
```

## 21.11 Profiling & Debugging

**Profiling**

- CUDA: Nsight Systems/Compute, `compute-sanitizer`.
- ROCm: `rocprof`, `rocminfo`, `rocm-smi`, Radeon GPU Profiler.

**Debugging**

- Check errors after launches (`ASSERT_LAST_ERROR()`).
- Use device-side asserts in debug builds.
- Guard shared memory usage and out-of-bounds with explicit checks when validating.
- For memory issues: CUDA `compute-sanitizer --tool memcheck`, ROCm `rocgdb` and `rocm-smi`.

## 21.12 Integration into an Inference Engine

- **Operator registry**: Map high-level graph nodes to kernel calls (by op name, dtype, shape).
- **Static workspace planner**: Query `workspace_bytes` per op and allocate once per session.
- **Streams**: Use per-request streams; serialize dependent ops; overlap independent ops.
- **KV cache**: Keep cache pointers/strides in descriptors; avoid implicit global state.
- **Error propagation**: Convert nonzero kernel return codes into engine exceptions with context.

## 21.13 Packaging & Distribution

- Build **fat binaries** for multiple SM/GCN targets (e.g., `sm_80,sm_90` or `gfx90a,gfx942`).
- Version the ABI; bump on breaking changes.
- Provide both:

  - A shared library exposing the C ABI.
  - Optional framework bindings (PyTorch custom op).

- Document environment variables for runtime tuning (e.g., `LLM_RMSNORM_BLOCK=256`).

## 21.14 Production Checklist

- [ ] Correctness vs. reference across dtypes and shapes.
- [ ] Stable ABI with explicit descriptors and stream.
- [ ] No hidden allocations; graph-capture safe.
- [ ] Autotuned launch params cached by arch.
- [ ] Profiling shows improvement vs. baseline (≥10–20%).
- [ ] Backpressure-safe under concurrent requests.
- [ ] Packaged for target architectures with CI validation.
- [ ] Clear fallbacks (library/Triton) behind a feature flag.

## 21.15 Next Steps

- Extend the example to support **BF16/FP32** and **per-channel quant** (I8→F16).
- Add optional fusions: **residual add**, **SiLU**, **gate** (SwiGLU epilogue).
- Provide a portable Triton variant to compare against the handwritten kernel.

**Files in this section**

- `abi.h` – Stable C ABI for kernel launches
- `backend.h` – CUDA/HIP portability macros
- `rmsnorm.cu` – RMSNorm kernel (CUDA/HIP)
- `torch_binding.cpp` – Optional PyTorch custom op binding
- `CMakeLists.txt` – Unified build configuration
