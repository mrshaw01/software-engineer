# Roofline Modeling & Back-of-Envelope Math

## 2. Summary

This module shows how to predict LLM inference performance using the roofline model and quick, defensible arithmetic. You will compute arithmetic intensity (FLOPs per byte), identify whether an operation is compute-bound or memory-bound, and validate predictions with a runnable CUDA/HIP microbenchmark. We work through prefill and decode paths, derive numeric examples for GEMM, attention, and KV-cache traffic, and provide commands to profile on NVIDIA (CUDA) and AMD (ROCm/HIP) GPUs. The outcome is a reproducible method to estimate tokens/sec and choose the right optimization lever before writing kernels.

## 3. Why It Matters for LLM Inference

- **Prefill (long contexts, large batches):** dominated by large GEMMs with high arithmetic intensity → usually **compute-bound**.
- **Decode (batch ≤ few, 1 token/step):** dominated by GEMV, attention over KV cache, and small matmuls → much lower arithmetic intensity → often **memory-bound**.
- Correctly placing each hotspot on the roofline guides you to the highest-ROI optimization: kernel fusion and tensor cores for compute-bound; layout, paging, and bandwidth for memory-bound.

## 4. Key Concepts and Formulas

Let

- $P_{\text{peak}}$ = peak compute throughput (FLOP/s).
- $B_{\text{peak}}$ = peak memory bandwidth (B/s).
- $\text{AI} = \frac{\text{FLOPs}}{\text{Bytes moved to/from DRAM}}$.

**Roofline bound:**

$$
P_{\text{achievable}} \le \min\left(P_{\text{peak}},\ \text{AI}\cdot B_{\text{peak}}\right)
$$

**Dense GEMM** $C_{M\times N} = A_{M\times K} B_{K\times N}$
FLOPs $= 2MNK$. Bytes (first-order, DRAM once): $s(MK + KN + MN)$ where $s$ is bytes/elem.

$$
\text{AI}_\text{GEMM} = \frac{2MNK}{s(MK+KN+MN)}
$$

**GEMV** $y_{M} = A_{M\times K}x_{K}$
FLOPs $= 2MK$. Bytes $\approx s(MK + M + K) \approx sMK $ (weights dominate).

$$
\text{AI}_\text{GEMV} \approx \frac{2}{s}\quad(\text{e.g., }s{=}2 \text{ bytes for FP16} \Rightarrow \text{AI}\approx 1\ \text{FLOP/B})
$$

**Decode attention (single token) with GQA**:
Let $H_q$ query heads, $H_{kv}$ KV heads, head dim $d$, cache length $T$, element size $s$.

- FLOPs (QK^T and AV): $\approx 4 H_q d T$.
- Bytes (K and V reads): $\approx 2 s H_{kv} d T$. (Q read is negligible.)

$$
\text{AI}_\text{attn, decode} \approx \frac{4 H_q d T}{2 s H_{kv} d T} = \frac{2 H_q}{s H_{kv}}
$$

For FP16 $s{=}2$, $H_q/H_{kv}{=}4$ (typical GQA), $\text{AI}\approx 4\ \text{FLOP/B}$.

**SAXPY** $y = a x + y$ (per-element, FP32): 2 FLOPs, 3 memory ops → $\text{AI} = 2 / (12\,\text{B}) \approx 0.1667 \ \text{FLOP/B}$.

## 5. GPU Deep Dive

### NVIDIA specifics

- **Execution:** 32-thread warps on SMs; FP16/BF16 tensor cores give highest GEMM throughput.
- **Memory:** HBM2e/HBM3 global memory; L2 is shared per GPU; per-SM L1 + shared memory. Coalesced 128-B transactions per memory channel are critical.
- **Optimization lever:** Use tensor cores (mma.sync), maximize occupancy and L2 reuse for prefill; for decode, prioritize KV layout (paged) and coalesced streaming.

### AMD specifics

- **Execution:** 64-lane wavefronts on CUs; MFMA/XDLOPs are the matrix engines.
- **Memory:** HBM stacks; large L2; LDS (shared memory) per CU with bank considerations.
- **Optimization lever:** Target MFMA paths; align/tile for LDS reuse; tune VGPR usage; for decode, ensure KV cache paging and stride-1 access per wavefront.

## 6. Implementation

Minimal single-source program that:

1. Measures a **memory-bound** kernel (SAXPY).
2. Measures a **compute-tilted** FMA kernel (register-heavy).
3. Prints measured bandwidth, GFLOP/s, and a small **predicted roofline** table.

### File: `topics/03-roofline-modeling/code/roofline.cu`

```cpp
// Single-source CUDA/HIP roofline microbench
// Build (CUDA):
//   nvcc -O3 -std=c++17 -arch=${SM_ARCH} -lineinfo topics/03-roofline-modeling/code/roofline.cu -o roofline
// Build (ROCm):
//   hipcc -O3 -std=c++17 --offload-arch=${GFX_ARCH} topics/03-roofline-modeling/code/roofline.cu -o roofline
// Run:
//   ./roofline [N] [iters] [repeats]
// Defaults: N=1<<26 elements (~256 MiB for two arrays FP32), iters=1024, repeats=5

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

#if defined(__HIP_PLATFORM_AMD__)
  #include <hip/hip_runtime.h>
  #define DEVFN __global__
  #define API_CHECK(x) do { auto e = (x); if (e != hipSuccess) { \
    fprintf(stderr,"HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e)); std::abort(); } } while(0)
  #define LAUNCH(dimGrid, dimBlock, shmem, stream, kernel, ...) \
    hipLaunchKernelGGL(kernel, dimGrid, dimBlock, shmem, stream, __VA_ARGS__)
  #define DEVICE_MALLOC(ptr, bytes) API_CHECK(hipMalloc(ptr, bytes))
  #define DEVICE_FREE(ptr) API_CHECK(hipFree(ptr))
  #define HTOD(dst, src, bytes) API_CHECK(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice))
  #define DTOH(dst, src, bytes) API_CHECK(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost))
  #define DEVICE_SYNC() API_CHECK(hipDeviceSynchronize())
  #define EVENT_T hipEvent_t
  #define EVENT_CREATE(ev) API_CHECK(hipEventCreate(&ev))
  #define EVENT_RECORD(ev, stream) API_CHECK(hipEventRecord(ev, stream))
  #define EVENT_ELAPSED_MS(ms, start, stop) API_CHECK(hipEventElapsedTime(&ms, start, stop))
  #define EVENT_DESTROY(ev) API_CHECK(hipEventDestroy(ev))
  #define STREAM_T hipStream_t
  #define STREAM_CREATE(s) API_CHECK(hipStreamCreate(&s))
  #define STREAM_DESTROY(s) API_CHECK(hipStreamDestroy(s))
#else
  #include <cuda_runtime.h>
  #define DEVFN __global__
  #define API_CHECK(x) do { auto e = (x); if (e != cudaSuccess) { \
    fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::abort(); } } while(0)
  #define LAUNCH(dimGrid, dimBlock, shmem, stream, kernel, ...) \
    kernel<<<dimGrid, dimBlock, shmem, stream>>>(__VA_ARGS__)
  #define DEVICE_MALLOC(ptr, bytes) API_CHECK(cudaMalloc(ptr, bytes))
  #define DEVICE_FREE(ptr) API_CHECK(cudaFree(ptr))
  #define HTOD(dst, src, bytes) API_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice))
  #define DTOH(dst, src, bytes) API_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost))
  #define DEVICE_SYNC() API_CHECK(cudaDeviceSynchronize())
  #define EVENT_T cudaEvent_t
  #define EVENT_CREATE(ev) API_CHECK(cudaEventCreate(&ev))
  #define EVENT_RECORD(ev, stream) API_CHECK(cudaEventRecord(ev, stream))
  #define EVENT_ELAPSED_MS(ms, start, stop) API_CHECK(cudaEventElapsedTime(&ms, start, stop))
  #define EVENT_DESTROY(ev) API_CHECK(cudaEventDestroy(ev))
  #define STREAM_T cudaStream_t
  #define STREAM_CREATE(s) API_CHECK(cudaStreamCreate(&s))
  #define STREAM_DESTROY(s) API_CHECK(cudaStreamDestroy(s))
#endif

// Vector SAXPY: y = a*x + y
DEVFN void saxpy_kernel(int n, float a, const float* __restrict__ x, float* __restrict__ y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a * x[i] + y[i];
  }
}

// Register FMA storm: z = fma(z, c, d) repeated `iters` times, minimal DRAM traffic
DEVFN void fma_kernel(int n, int iters, const float* __restrict__ x, float* __restrict__ y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float z = x[i];
    float c = 1.000123f;
    float d = 0.999771f;
    #pragma unroll 4
    for (int t = 0; t < iters; ++t) {
      z = fmaf(z, c, d); // 1 FMA = 2 FLOPs
    }
    y[i] = z;
  }
}

double run_ms(dim3 grid, dim3 block, STREAM_T stream,
              void (*kernel_saxpy)(int,float,const float*,float*),
              void (*kernel_fma)(int,int,const float*,float*),
              int which, int n, int iters,
              float a, const float* dx, float* dy, int repeats) {
  // which: 0->saxpy, 1->fma
  EVENT_T start, stop;
  EVENT_CREATE(start); EVENT_CREATE(stop);
  // warmup
  for (int w = 0; w < 2; ++w) {
    if (which == 0) { LAUNCH(grid, block, 0, stream, saxpy_kernel, n, a, dx, dy); }
    else            { LAUNCH(grid, block, 0, stream, fma_kernel, n, iters, dx, dy); }
  }
  DEVICE_SYNC();

  double best_ms = 1e30;
  for (int r = 0; r < repeats; ++r) {
    EVENT_RECORD(start, stream);
    if (which == 0) { LAUNCH(grid, block, 0, stream, saxpy_kernel, n, a, dx, dy); }
    else            { LAUNCH(grid, block, 0, stream, fma_kernel, n, iters, dx, dy); }
    EVENT_RECORD(stop, stream);
    DEVICE_SYNC();
    float ms = 0.f;
    EVENT_ELAPSED_MS(ms, start, stop);
    if (ms < best_ms) best_ms = ms;
  }
  EVENT_DESTROY(start); EVENT_DESTROY(stop);
  return best_ms;
}

int main(int argc, char** argv) {
  int n = (argc > 1) ? std::stoi(argv[1]) : (1<<26); // elements
  int iters = (argc > 2) ? std::stoi(argv[2]) : 1024;
  int repeats = (argc > 3) ? std::stoi(argv[3]) : 5;

  size_t bytes = size_t(n) * sizeof(float);
  float *dx = nullptr, *dy = nullptr;
  DEVICE_MALLOC((void**)&dx, bytes);
  DEVICE_MALLOC((void**)&dy, bytes);

  std::vector<float> hx(n, 1.0f), hy(n, 2.0f);
  HTOD(dx, hx.data(), bytes);
  HTOD(dy, hy.data(), bytes);

  STREAM_T stream; STREAM_CREATE(stream);
  dim3 block(256);
  dim3 grid((n + block.x - 1) / block.x);

  // 1) SAXPY (memory-bound)
  double ms_saxpy = run_ms(grid, block, stream,
                           nullptr, nullptr, 0, n, iters, 3.14159f, dx, dy, repeats);
  double flops_saxpy = 2.0 * n; // mul + add
  double bytes_saxpy = (2 + 1) * double(n) * sizeof(float); // x load + y load + y store
  double gflops_saxpy = flops_saxpy / (ms_saxpy * 1e6);
  double gbps_saxpy   = bytes_saxpy / (ms_saxpy * 1e6);

  // 2) FMA (compute-tilted)
  double ms_fma = run_ms(grid, block, stream,
                         nullptr, nullptr, 1, n, iters, 0.f, dx, dy, repeats);
  double flops_fma = 2.0 * n * iters; // FMA = 2 FLOPs
  double bytes_fma = (1 + 1) * double(n) * sizeof(float); // load + store
  double gflops_fma = flops_fma / (ms_fma * 1e6);
  double ai_saxpy = flops_saxpy / bytes_saxpy;       // ~0.1667
  double ai_fma   = flops_fma   / bytes_fma;         // ~iters/4 for FP32

  // Measured peaks for roofline prediction:
  double B_peak_meas = gbps_saxpy * 1e9; // B/s
  double P_peak_meas = gflops_fma * 1e9; // FLOP/s

  printf("Elements: %d, iters(FMA): %d, repeats: %d\n", n, iters, repeats);
  printf("SAXPY:  time = %.3f ms, BW = %.1f GB/s, Perf = %.1f GFLOP/s, AI = %.4f FLOP/B\n",
         ms_saxpy, gbps_saxpy, gflops_saxpy, ai_saxpy);
  printf("FMA  :  time = %.3f ms, Perf = %.1f GFLOP/s, AI = %.2f FLOP/B (approx)\n",
         ms_fma, gflops_fma, ai_fma);

  // Small predicted roofline table:
  const double ai_list[] = {0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32};
  printf("\nPredicted roofline (using measured peaks):\n");
  printf("  AI(FLOP/B) | Pred GFLOP/s  | Bound\n");
  for (double ai : ai_list) {
    double pred = std::min(P_peak_meas, ai * B_peak_meas) / 1e9; // GFLOP/s
    const char* bound = (ai * B_peak_meas < P_peak_meas) ? "Memory" : "Compute";
    printf("  %9.3f | %12.1f | %s\n", ai, pred, bound);
  }

  STREAM_DESTROY(stream);
  DEVICE_FREE(dx); DEVICE_FREE(dy);
  return 0;
}
```

### Build commands

- CUDA:

```
nvcc -O3 -std=c++17 -arch=${SM_ARCH} -lineinfo topics/03-roofline-modeling/code/roofline.cu -o roofline
# Example: export SM_ARCH=sm_90
```

- ROCm/HIP:

```
hipcc -O3 -std=c++17 --offload-arch=${GFX_ARCH} topics/03-roofline-modeling/code/roofline.cu -o roofline
# Example: export GFX_ARCH=gfx942
```

### Run

```
./roofline                 # N=2^26, iters=1024, repeats=5
./roofline 67108864 2048   # N=64M, iters=2048
```

## 7. Profiling and Validation

### NVIDIA

- **Nsight Compute (CLI):**

```
nv-nsight-cu-cli --kernel-name-base regex:'saxpy_kernel' \
  --metrics dram__bytes.sum,sm__sass_thread_inst_executed_op_fadd_pred_on.sum,sm__throughput.avg.pct_of_peak_sustained_active \
  ./roofline

nv-nsight-cu-cli --kernel-name-base regex:'fma_kernel' \
  --metrics sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,sm__inst_executed.sum \
  ./roofline
```

Interpretation:

- `dram__bytes.sum / time` should align with the printed GB/s for SAXPY.
- `sm__pipe_fma_cycles_active...` high percentage indicates compute saturation for FMA.

### AMD

- **rocprof (stats + timeline):**

```
rocprof --stats ./roofline
rocprof --hip-trace --timestamp on ./roofline
```

Interpretation:

- Kernel duration (timeline) vs. bytes moved approximates bandwidth.
- Expect FMA kernel to show minimal memory transactions relative to runtime and high VALU/MFMA utilization (with omniperf/omnitrace if available).

## 8. Performance Checklist

- [ ] **SAXPY bandwidth ≥ 60%** of the best of 3 runs (stability check).
- [ ] **FMA GFLOP/s increases** with `iters` and plateaus (compute-limited regime).
- [ ] Predicted roofline table classifies SAXPY as **memory-bound** (e.g., AI≈0.167).
- [ ] Changing `N` (problem size) does not change plateau behavior (roofline stable).
- [ ] With pinned host memory and larger `N`, measured bandwidth does **not** regress.

## 9. Troubleshooting

| Symptom                                | Likely cause                       | Fix                                                                                               |
| -------------------------------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------- |
| SAXPY GB/s very low                    | Uncoalesced access, too small `N`  | Use large `N` (≥ 2^26), keep stride-1, ensure blockDim=256–1024                                   |
| FMA GFLOP/s doesn’t scale with `iters` | Register/spill or clock throttling | Reduce `iters`, check occupancy, watch thermals/power limits                                      |
| Run crashes on HIP with “HSA error”    | Offload arch mismatch              | Set `--offload-arch=${GFX_ARCH}` matching your GPU (e.g., `gfx90a`, `gfx942`)                     |
| Nsight/rocprof shows short kernels     | Warmup only measured               | Ensure benchmark uses post-warmup best-of-N; don’t profile warmups                                |
| CUDA illegal memory access             | `N` too large for allocation       | Reduce `N` or free other GPU memory                                                               |
| Inconsistent GB/s between runs         | Background activity / clocks       | Set persistence/auto-boost, close other GPU jobs, use `nvidia-smi -ac` or `rocm-smi` if permitted |

## 10. Acceptance Criteria

- Document explains roofline mechanics and places **prefill vs. decode** on the roofline with numeric examples.
- Program **compiles** with `nvcc` and `hipcc`, runs in seconds, and prints:

  - SAXPY GB/s and GFLOP/s with AI≈0.1667.
  - FMA GFLOP/s with AI≈iters/4 and a clear compute plateau.
  - A predicted roofline table with correct memory/compute regime tagging.

- Profiling commands included with at least two critical counters and expected interpretation.
- Checklist and troubleshooting cover at least 5 concrete issues and remedies.

## 11. Further Work

- Add a **GEMM microbench** (CUTLASS / hipBLASLt) to measure tensor-core/MFMA rooflines directly.
- Extend kernels to **FP16/BF16** and **INT8** (with fused dequant) to study datatype-driven AI shifts.
- Integrate a **KV-cache stream** microbench (paged layout) to emulate decode attention bandwidth pressure.
- Plot measured roofline (CSV → matplotlib) and overlay real model operator points from a trace.
- Use **CUDA/HIP Graphs** to reduce launch overhead and observe decode-regime improvements.

## Back-of-Envelope Examples (Checked)

1. **Prefill QKV GEMM (FP16),** batch $B{=}8$, seq $L{=}2048$, hidden $D{=}4096$:
   $M{=}BL=16384$, $K{=}D=4096$, $N{=}3D=12288$, $s{=}2$ B.
   FLOPs $= 2MNK = 2 \cdot 16384 \cdot 4096 \cdot 12288 = 1{,}649{,}267{,}441{,}664$.
   Bytes $= s(MK+KN+MN) = 2(16384\cdot4096 + 4096\cdot12288 + 16384\cdot12288) = 637{,}534{,}208$ B $= 0.59375$ GiB.

$$
\text{AI} \approx 1.649\times10^{12} / 6.375\times10^{8} \approx \mathbf{2587\ FLOP/B} \quad\Rightarrow\ \text{compute-bound.}
$$

2. **Decode GEMV (FP16),** $D{=}4096$:
   FLOPs $= 2D^2 = 33{,}554{,}432$.
   Bytes $\approx sD^2 = 2\cdot 16{,}777{,}216 = 33{,}554{,}432$ B (weights dominate).

$$
\text{AI} \approx \mathbf{1\ FLOP/B} \quad\Rightarrow\ \text{likely memory-bound on modern GPUs.}
$$

3. **Decode Attention with GQA,** $H_q{=}32$, $H_{kv}{=}8$, $d{=}128$, $T{=}8192$, FP16:
   FLOPs $\approx 4 H_q d T = 4\cdot 32\cdot 128\cdot 8192 = 134{,}217{,}728$.
   Bytes $\approx 2 s H_{kv} d T = 2\cdot 2\cdot 8\cdot 128\cdot 8192 = 33{,}554{,}432$ B.

$$
\text{AI} \approx \mathbf{4\ FLOP/B} \quad\Rightarrow\ \text{memory-sensitive; KV layout and bandwidth dominate.}
$$

4. **SAXPY (FP32):** AI $= 2\ \text{FLOPs} / 12\ \text{B} = \mathbf{0.1667\ FLOP/B}$ → firmly memory-bound.
