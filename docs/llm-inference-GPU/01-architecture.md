# GPU Architecture Orientation

## Summary

This module introduces the execution and memory model shared by modern NVIDIA (CUDA) and AMD (ROCm/HIP) GPUs, with a focus on what matters to LLM inference. You will learn how threads are grouped and scheduled, how memory hierarchies constrain throughput, and how to reason about arithmetic intensity and roofline limits. A minimal, runnable bandwidth microbenchmark is provided (single-source CUDA/HIP) to validate coalescing, occupancy, and effective DRAM throughput. By the end, you should be able to estimate whether a kernel is compute- or bandwidth-bound and verify it with profilers.

## Why It Matters for LLM Inference

- **Prefill (encoder-style GEMMs)** are usually **compute-bound** on Tensor/MFMA cores; tiling and epilog fusions dominate.
- **Decode (autoregressive step)** is often **memory-/latency-bound**, dominated by KV-cache reads/writes and small GEMMs. Latency hiding via occupancy and cache locality are critical.
- Understanding warp/wave execution, memory coalescing, and cache/DRAM bandwidth clarifies why techniques like **Paged KV**, **quantization**, and **persistent kernels** materially move tokens/sec.

## Key Concepts and Formulas

**Execution groups**

- NVIDIA: warp = 32 threads; many warps run on an SM (streaming multiprocessor).
- AMD: wavefront typically = 64 threads (Wave64) on a CU (compute unit). (Some modes support Wave32.)

**Latency hiding and occupancy**

- A scheduler swaps among warps/waves to hide memory latency. Enough **active warps/waves** are required.

- Upper bound on active blocks per SM/CU:

  $$B_{active} = \min\left(\left\lfloor\frac{R_{SM}}{R_{thr}\cdot T}\right\rfloor,\;\left\lfloor\frac{S_{SM}}{S_{blk}}\right\rfloor,\;\left\lfloor\frac{T_{SM}}{T}\right\rfloor,\;B_{max}\right)$$

  where $R_{SM}$ registers/SM, $S_{SM}$ shared/LDS per SM, $T_{SM}$ max threads/SM, $B_{max}$ architectural block limit, $R_{thr}$ registers/thread, $S_{blk}$ shared/LDS per block, $T$ threads/block.

- **Occupancy**: $Occ = \frac{W_{active}}{W_{max}}$ where $W_{active} = B_{active}\cdot \lceil T/W_{size} \rceil$ and $W_{size}=32$ (NVIDIA) or typically 64 (AMD).

**Arithmetic intensity (AI)**

- $AI = \frac{\text{FLOPs}}{\text{Bytes moved}}$.
- Roofline bound: $GFLOP/s \le \min(\text{Peak FLOP/s},\;\text{Peak BW} \times AI)$.

**Streaming SAXPY (y = a·x + y) example**

- Per element (FP32): 2 FLOPs (mul+add); 12 bytes DRAM traffic (load x, load y, store y). AI = $2/12 = 1/6$ FLOP/byte ≈ 0.1667. If Peak BW = 1 TB/s, bound is ≈ 166 GFLOP/s.
- Effective bandwidth measurement:

  $$BW_{eff} = \frac{N\cdot B_{iter}\cdot I}{t}\ ,\quad B_{iter}=12\ \text{bytes/elem(FP32)}$$

  where $N$ elements, $I$ iterations, $t$ seconds.

**KV cache sizing (decode path)**

- Bytes: $B\cdot L\cdot T\cdot H_{kv}\cdot d\cdot 2\cdot b$ where batch $B$, layers $L$, sequence length $T$, KV heads $H_{kv}$, head dim $d$, 2 for K and V, and $b$ bytes/element.
- Numeric example (BF16/FP16): $B=1, L=32, T=8192, H_{kv}=8, d=128, b=2$ → **1,073,741,824 bytes ≈ 1 GiB**.

**Coalescing and transaction size (rule of thumb)**

- A full warp (32) of FP32 scalars maps to 128-byte segments on NVIDIA; a Wave64 maps to 256-byte segments on AMD. Use aligned vector types (e.g., `float4`, `int4`, `__half2`) to reduce transactions and pressure.

**Shared memory/LDS**

- Organized in banks (commonly 32). To avoid bank conflicts, map thread `t` to consecutive addresses: `base + t` or `base + k*t` with `k` co-prime to bank count.

## GPU Deep Dive

### NVIDIA specifics

- **Warps (32 threads)** scheduled by multiple warp schedulers per SM.
- **Tensor Cores** accelerate matrix MACs; prefer tiles in multiples of 8/16 depending on dtype.
- **Memory hierarchy**: per-SM L1/shared, chip-wide L2, then DRAM/HBM. Coalesced 128B transactions from warps maximize throughput.
- **Occupancy limiters**: registers/thread, shared memory/block, threads/block, blocks/SM. Use `nvcc --ptxas-options=-v` to see register usage.

### AMD specifics

- **Wavefronts (typically 64 threads)** on CUs; issue control differs but the model (hide latency via many waves) is similar.
- **MFMA/XDLOPs** provide matrix MAC acceleration; tile sizes differ from NVIDIA.
- **LDS (shared memory)** is software-managed scratchpad. Global memory coalescing targets 256B aligned segments for Wave64.
- **Occupancy limiters**: VGPR/SGPR usage per thread, LDS per block, waves per CU. Use `hipcc -save-temps` and `rocprof` to inspect usage.

## Implementation

A minimal, **single-source** CUDA/HIP SAXPY bandwidth benchmark with vectorized memory access. Build with either `nvcc` or `hipcc`.

### Code: `topics/01-architecture/code/arch_bw_saxpy.cpp`

```cpp
// Single-source CUDA/HIP SAXPY bandwidth microbenchmark (FP32)
// Build (CUDA): nvcc -O3 -std=c++17 -arch=${SM_ARCH} -lineinfo topics/01-architecture/code/arch_bw_saxpy.cpp -o bw_saxpy
// Build (ROCm): hipcc -O3 -std=c++17 --offload-arch=${GFX_ARCH} topics/01-architecture/code/arch_bw_saxpy.cpp -o bw_saxpy
// Run: ./bw_saxpy -n 16777216 -i 50 -a 1.5

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cmath>
#include <cstring>
#include <string>

#if defined(__HIP_PLATFORM_AMD__)
  #include <hip/hip_runtime.h>
  #define DEVFN __global__
  #define API_CHECK(x) do { auto e = (x); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); std::abort(); } } while(0)
  #define Event hipEvent_t
  #define EventCreate hipEventCreate
  #define EventRecord hipEventRecord
  #define EventElapsed hipEventElapsedTime
  #define EventDestroy hipEventDestroy
  #define Malloc hipMalloc
  #define Free hipFree
  #define MemcpyH2D(dst,src,sz) API_CHECK(hipMemcpy(dst,src,sz, hipMemcpyHostToDevice))
  #define MemcpyD2H(dst,src,sz) API_CHECK(hipMemcpy(dst,src,sz, hipMemcpyDeviceToHost))
  #define DeviceSync() API_CHECK(hipDeviceSynchronize())
  #define LaunchKernel(kernel, grid, block, shmem, stream, ...) \
      hipLaunchKernelGGL(kernel, dim3(grid), dim3(block), shmem, stream, __VA_ARGS__)
#else
  #include <cuda_runtime.h>
  #define DEVFN __global__
  #define API_CHECK(x) do { auto e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); std::abort(); } } while(0)
  #define Event cudaEvent_t
  #define EventCreate cudaEventCreate
  #define EventRecord cudaEventRecord
  #define EventElapsed cudaEventElapsedTime
  #define EventDestroy cudaEventDestroy
  #define Malloc cudaMalloc
  #define Free cudaFree
  #define MemcpyH2D(dst,src,sz) API_CHECK(cudaMemcpy(dst,src,sz, cudaMemcpyHostToDevice))
  #define MemcpyD2H(dst,src,sz) API_CHECK(cudaMemcpy(dst,src,sz, cudaMemcpyDeviceToHost))
  #define DeviceSync() API_CHECK(cudaDeviceSynchronize())
  #define LaunchKernel(kernel, grid, block, shmem, stream, ...) \
      (kernel)<<<grid, block, shmem, stream>>>(__VA_ARGS__)
#endif

#ifndef WARP_SIZE
  #if defined(__HIP_PLATFORM_AMD__)
    #define WAVE_SIZE 64
  #else
    #define WARP_SIZE 32
  #endif
#endif

// Vectorized SAXPY over float4 for coalesced 16B accesses per thread
DEVFN void saxpy_vec4_kernel(const float a, const float* __restrict__ x,
                             float* __restrict__ y, size_t n) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t n4 = n / 4; // number of float4 elements

  const float4* __restrict__ x4 = reinterpret_cast<const float4*>(x);
  float4* __restrict__ y4 = reinterpret_cast<float4*>(y);

  if (tid < n4) {
    float4 xv = x4[tid];
    float4 yv = y4[tid];
    yv.x = fmaf(a, xv.x, yv.x);
    yv.y = fmaf(a, xv.y, yv.y);
    yv.z = fmaf(a, xv.z, yv.z);
    yv.w = fmaf(a, xv.w, yv.w);
    y4[tid] = yv;
  }

  // Handle tail with grid-stride on scalar path
  size_t tail_start = n4 * 4;
  for (size_t i = tail_start + tid; i < n; i += (size_t)gridDim.x * blockDim.x) {
    y[i] = fmaf(a, x[i], y[i]);
  }
}

static void usage(const char* prog){
  fprintf(stderr, "Usage: %s [-n elements] [-i iters] [-a alpha]\n", prog);
}

int main(int argc, char** argv) {
  size_t n = (1ull<<24); // 16,777,216 elements (~64 MiB per array)
  int iters = 50;
  float alpha = 1.5f;

  for (int i=1; i<argc; ++i) {
    if (!strcmp(argv[i], "-n") && i+1<argc) { n = strtoull(argv[++i], nullptr, 10); }
    else if (!strcmp(argv[i], "-i") && i+1<argc) { iters = atoi(argv[++i]); }
    else if (!strcmp(argv[i], "-a") && i+1<argc) { alpha = atof(argv[++i]); }
    else { usage(argv[0]); return 1; }
  }

  const size_t bytes = n * sizeof(float);
  std::vector<float> hx(n, 1.0f), hy(n, 2.0f);

  float *dx=nullptr, *dy=nullptr;
  API_CHECK(Malloc((void**)&dx, bytes));
  API_CHECK(Malloc((void**)&dy, bytes));
  MemcpyH2D(dx, hx.data(), bytes);
  MemcpyH2D(dy, hy.data(), bytes);

  const int block = 256;
  const size_t n4 = n / 4;
  const int grid = (int)((n4 ? (n4 + block - 1) / block : (n + block - 1) / block));

  Event start, stop; API_CHECK(EventCreate(&start)); API_CHECK(EventCreate(&stop));
  DeviceSync();
  API_CHECK(EventRecord(start));
  for (int it=0; it<iters; ++it) {
    LaunchKernel(saxpy_vec4_kernel, grid, block, 0, 0, alpha, dx, dy, n);
  }
  DeviceSync();
  API_CHECK(EventRecord(stop));
  API_CHECK(EventDestroy(start)); // ensure stop contains elapsed

  float ms=0.0f; API_CHECK(EventElapsed(&ms, start, stop));
  // Copy back for validation
  MemcpyD2H(hy.data(), dy, bytes);
  Free(dx); Free(dy);

  // Validate: y = 2 + iters*alpha
  const float expected = 2.0f + iters * alpha;
  double max_abs_err = 0.0;
  for (size_t i=0;i<n;++i) max_abs_err = std::max(max_abs_err, (double)std::fabs(hy[i] - expected));

  const double seconds = ms / 1e3;
  const double bytes_per_iter = 12.0 * (double)n; // FP32: load x (4), load y (4), store y (4)
  const double bw_GBps = (iters * bytes_per_iter) / seconds / 1e9;
  const double gflops = (iters * 2.0 * (double)n) / seconds / 1e9; // 2 FLOPs/elem

  printf("n=%zu iters=%d alpha=%.3f\n", n, iters, alpha);
  printf("time=%.3f ms  BW=%.2f GB/s  GFLOP/s=%.2f  max_abs_err=%.3g\n", ms, bw_GBps, gflops, max_abs_err);
  return (max_abs_err < 1e-5) ? 0 : 2;
}
```

### Build and Run

CUDA (example for H100):

```
nvcc -O3 -std=c++17 -arch=sm_90 -lineinfo topics/01-architecture/code/arch_bw_saxpy.cpp -o bw_saxpy
./bw_saxpy -n 33554432 -i 100 -a 1.25
```

ROCm (example for MI300):

```
hipcc -O3 -std=c++17 --offload-arch=gfx942 topics/01-architecture/code/arch_bw_saxpy.cpp -o bw_saxpy
./bw_saxpy -n 33554432 -i 100 -a 1.25
```

## Profiling and Validation

### NVIDIA

Nsight Compute (key metrics):

```
ncu --target-processes all \
  --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,smsp__throughput.avg.pct_of_peak_sustained_active, \
           l1tex__t_bytes.sum.per_second,lts__t_bytes.sum.per_second \
  ./bw_saxpy -n 33554432 -i 100
```

Expect high DRAM/L2 throughput, near-peak memory pipes, and minimal stalls on `long_scoreboard` when coalesced.

Nsight Systems (timeline):

```
nsys profile -o nsys_bw ./bw_saxpy -n 33554432 -i 100
```

You should see tightly packed kernel launches with little gap between iterations.

### AMD

`rocprof` summary stats:

```
rocprof --stats ./bw_saxpy -n 33554432 -i 100
```

Inspect achieved memory BW and waves/CU. Low waves or LDS bank conflicts indicate occupancy or access issues.

## Performance Checklist

1. Vectorized loads/stores (`float4`/`int4`) with 16-byte alignment.
2. Full-warps/waves per block (e.g., 256 or 512 threads) for efficient scheduling.
3. Occupancy not blocked by registers/shared-LDS. If low, reduce `threads/block` or refactor to lower register pressure.
4. Coalesced access: thread `t` reads `x[base+t]`, no striding that causes gather/scatter.
5. Avoid unnecessary syncs; no `__syncthreads()` in purely streaming kernels.
6. Measure `BW_eff` and compare against device peak; target ≥70% for large `n`.
7. Check L2 vs DRAM counters to confirm hitting DRAM roof when expected.
8. Validate numerics (max abs error < 1e-5 for FP32 SAXPY here).
9. Compile with `-lineinfo` to map stalls to source lines in profilers.
10. Pin CPU memory (optional) to reduce H2D/D2H time when profiling end-to-end.

## Troubleshooting

| Symptom                                | Likely cause               | Fix                                                                       |
| -------------------------------------- | -------------------------- | ------------------------------------------------------------------------- |
| BW far below peak                      | Uncoalesced loads/stores   | Use contiguous indexing and vector types; avoid misaligned pointers       |
| Many `long_scoreboard` stalls (NVIDIA) | Memory latency not hidden  | Increase active warps (reduce regs/thread or block size), use prefetching |
| Waves/CU < 4 (AMD)                     | VGPR/SGPR pressure         | Refactor to reduce live ranges; compile with `-O3` and inspect VGPR use   |
| Bank conflicts in shared/LDS           | Strided or modulo patterns | Re-tile or pad shared arrays to multiples of bank count                   |
| Divergence within warp/wave            | Branchy code               | Separate paths, use predication or warp-/wave-uniform control             |
| Tail elements incorrect                | Missing tail handling      | Add grid-stride scalar path or dedicated tail kernel                      |
| Kernel launch overhead dominates       | Too-small workload         | Increase batch or fuse iterations; use CUDA/HIP Graphs for decode loops   |
| Validation error grows with iters      | Numeric instabilities      | Use FMA (as here), consider BF16/FP32 accumulators                        |
| Profiler shows L2 thrashing            | Poor locality              | Tile to reuse in L2/shared; batch requests                                |
| H2D/D2H dominates wall time            | Counting transfers         | Isolate kernel time with events; use pinned memory for IO tests           |

## Acceptance Criteria

- Code builds on both CUDA 12.x and ROCm/HIP 6.x with the commands above.
- Running `./bw_saxpy -n 33554432 -i 100` completes in seconds and reports max abs error < 1e-5.
- Effective DRAM BW ≥ 70% of peak on a high-bandwidth GPU (when `n` ≥ 2^25), indicating good coalescing and occupancy.
- Nsight Compute or rocprof confirms high memory pipe utilization with minimal structural hazards.

## Further Work

- Extend to BF16/FP16 with `__half2`/BF16 vectorization; add fused dequant for INT8 to emulate typical LLM inference paths.
- Add a compute-bound microbenchmark (tensor-core/MFMA GEMM tile) and compare roofline position versus this bandwidth-bound SAXPY.
- Demonstrate persistent-kernel decode loop and CUDA/HIP Graphs to reduce launch overhead.
- Add KV-cache read/write microbenchmarks with varying head dims and page sizes to quantify locality benefits.
- Incorporate NCCL/RCCL overlap patterns to set expectations for multi-GPU throughput in prefill versus decode.
