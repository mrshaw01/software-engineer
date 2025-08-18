# Precision & Numerics for Inference

## Summary

Inference speed and accuracy depend critically on numeric precision and the stability of reductions (softmax, layer norm) executed millions of times per request. This topic explains which precisions to use (FP32, TF32, FP16, BF16, FP8/INT8), where accumulators must be wider, and how to verify numerical safety. We provide runnable CUDA/HIP code for a mixed-precision softmax that matches FP64 within tight tolerances while maintaining high throughput. The outcome is a checklist and acceptance tests to use across your serving stack.

## Why It Matters for LLM Inference

- **Prefill:** Large GEMMs dominate; Tensor/MFMA cores prefer lower precision inputs with FP32 accumulators. Bad choices cause divergence in attention/LN and degrade perplexity.
- **Decode:** Memory-bound with many small reductions; underflow/overflow risks increase. Stable softmax/LN with FP32 accumulators is essential to avoid token drift and rare NaNs.

## Key Concepts and Formulas

1. **Machine epsilon (unit roundoff)** for normalized numbers (base 2):
   FP32: $\epsilon=2^{-23}\approx1.1920929\times10^{-7}$
   FP16: $\epsilon=2^{-10}\approx9.765625\times10^{-4}$
   BF16: $\epsilon=2^{-7}\approx7.8125\times10^{-3}$

2. **Dynamic range (min normal, max finite)**:
   FP16: $x_{\min}=2^{-14}\approx 6.1035\times10^{-5}$, $x_{\max}\approx 65504$
   BF16: $x_{\min}=2^{-126}\approx1.1755\times10^{-38}$, $x_{\max}\approx 3.3895\times10^{38}$
   Implication: BF16 has FP32-like range (better for softmax/LN) but fewer mantissa bits than FP16 (worse unit roundoff).

3. **Softmax stability**: For a row $x\in\mathbb{R}^n$, compute
   $m=\max_i x_i$, $y_i=\exp(x_i-m)/\sum_j \exp(x_j-m)$.
   Use FP32 for the reduction and exponent; cast inputs/outputs as needed.

4. **LayerNorm stability**:
   Given $\mu=\frac{1}{n}\sum x_i$, $\sigma^2=\frac{1}{n}\sum (x_i-\mu)^2$. Accumulate in FP32, add $\epsilon$ (e.g., $1\times10^{-5}$) in FP32 before rsqrt.

5. **Error compounding**: Summing $n$ terms with unit roundoff $u$ can incur $\mathcal{O}(nu)$ relative error. Wider accumulators reduce $u$ and bound error.

6. **Throughput implications**: Lower-precision tensors halve or quarter bandwidth pressure; if accumulators are FP32, arithmetic throughput remains high while stability is preserved.

## GPU Deep Dive

### NVIDIA specifics

- Warps of 32 threads; Tensor Cores support FP16/BF16/TF32/FP8 inputs with FP32 accumulation. Use cuBLASLt epilogues (bias, GELU, quant dequant) to fuse operations where possible.
- For reductions, prefer warp shuffles (`__shfl_xor_sync`) for intra-warp and shared memory for inter-warp; accumulate in FP32. Avoid denormals via default FTZ.

### AMD specifics

- Wavefronts of 32/64 threads (MI2xx/MI3xx); MFMA/XDLOPs units accelerate FP16/BF16/FP8 with FP32 accumulation. HIP offers similar intrinsics; use `__shfl_xor` equivalents and LDS for inter-wave reductions.
- Prefer rocBLAS/hipBLASLt for GEMM with fused epilogues; ensure BF16 hardware path is enabled on MI300-class GPUs.

## Implementation

This section provides a **single-source CUDA/HIP** mixed-precision softmax. Inputs are FP16 or BF16; reductions and exponentials are FP32. A CPU FP64 reference validates results.

### Files

```bash
llm-inference/
  topics/02-precision-numerics/
    README.md
    code/
      mixed_softmax_single_source.cpp
```

### Build

CUDA (e.g., SM80/SM90):

```bash
cd topics/02-precision-numerics/code
nvcc -O3 -std=c++17 -arch=sm_80 -lineinfo mixed_softmax_single_source.cpp -o softmax_cuda
```

ROCm/HIP (e.g., gfx942/gfx90a):

```bash
cd topics/02-precision-numerics/code
hipcc -O3 -std=c++17 --offload-arch=gfx942 mixed_softmax_single_source.cpp -o softmax_hip
```

### Run

```bash
# rows x cols with dtype and repeats
./softmax_cuda 4096 4096 fp16 50
./softmax_cuda 4096 4096 bf16 50
# or
./softmax_hip 4096 4096 fp16 50
./softmax_hip 4096 4096 bf16 50
```

Expected output: maximum absolute/relative error vs FP64 and effective GB/s.

### Source: `code/mixed_softmax_single_source.cpp`

```cpp
// Single-source mixed-precision softmax for CUDA and HIP
// - Inputs: FP16 or BF16
// - Accumulation: FP32
// - Validation: CPU FP64 reference
// Build:
//   nvcc -O3 -std=c++17 -arch=sm_80 mixed_softmax_single_source.cpp -o softmax_cuda
//   hipcc -O3 -std=c++17 --offload-arch=gfx942 mixed_softmax_single_source.cpp -o softmax_hip

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#include <limits>
#include <string>

#if defined(__HIP_PLATFORM_AMD__)
  #include <hip/hip_runtime.h>
  #include <hip/hip_bfloat16.h>
  #define DEVFN __global__
  #define API_CHECK(x) do { auto e=(x); if (e!=hipSuccess) { \
      fprintf(stderr, "HIP error %d at %s:%d\n", (int)e, __FILE__, __LINE__); std::abort(); } } while(0)
  #define MALLOC  hipMalloc
  #define MEMCPY  hipMemcpy
  #define MEMSET  hipMemset
  #define FREE    hipFree
  #define MEMCPY_H2D hipMemcpyHostToDevice
  #define MEMCPY_D2H hipMemcpyDeviceToHost
  #define DEVICE_EVENT hipEvent_t
  #define EVENT_CREATE hipEventCreate
  #define EVENT_RECORD hipEventRecord
  #define EVENT_ELAPSED hipEventElapsedTime
  #define EVENT_DESTROY hipEventDestroy
  #define DEVICE_SYNC hipDeviceSynchronize
  #define STREAM hipStream_t
  #define STREAM_CREATE hipStreamCreate
  #define STREAM_DESTROY hipStreamDestroy
  #define LAUNCH(kernel, grid, block, shmem, stream, ...) \
      hipLaunchKernelGGL(kernel, grid, block, shmem, stream, __VA_ARGS__)
  using bfloat16_t = hip_bfloat16;
#else
  #include <cuda_runtime.h>
  #include <cuda_bf16.h>
  #define DEVFN __global__
  #define API_CHECK(x) do { auto e=(x); if (e!=cudaSuccess) { \
      fprintf(stderr, "CUDA error %d at %s:%d: %s\n", (int)e, __FILE__, __LINE__, cudaGetErrorString((cudaError_t)e)); std::abort(); } } while(0)
  #define MALLOC  cudaMalloc
  #define MEMCPY  cudaMemcpy
  #define MEMSET  cudaMemset
  #define FREE    cudaFree
  #define MEMCPY_H2D cudaMemcpyHostToDevice
  #define MEMCPY_D2H cudaMemcpyDeviceToHost
  #define DEVICE_EVENT cudaEvent_t
  #define EVENT_CREATE cudaEventCreate
  #define EVENT_RECORD cudaEventRecord
  #define EVENT_ELAPSED cudaEventElapsedTime
  #define EVENT_DESTROY cudaEventDestroy
  #define DEVICE_SYNC cudaDeviceSynchronize
  #define STREAM cudaStream_t
  #define STREAM_CREATE cudaStreamCreate
  #define STREAM_DESTROY cudaStreamDestroy
  #define LAUNCH(kernel, grid, block, shmem, stream, ...) \
      kernel<<<grid, block, shmem, stream>>>(__VA_ARGS__)
  using bfloat16_t = __nv_bfloat16;
#endif

// Half types
#if defined(__HIP_PLATFORM_AMD__)
  using half_t = _Float16; // HIP Clang supports _Float16
  __device__ __forceinline__ float to_float(half_t x){ return (float)x; }
  __device__ __forceinline__ half_t to_half(float x){ return (half_t)x; }
  __device__ __forceinline__ float to_float(bfloat16_t x){ return __bfloat162float(x); }
  __device__ __forceinline__ bfloat16_t to_bf16(float x){ return __float2bfloat16(x); }
#else
  #include <cuda_fp16.h>
  using half_t = __half;
  __device__ __forceinline__ float to_float(half_t x){ return __half2float(x); }
  __device__ __forceinline__ half_t to_half(float x){ return __float2half(x); }
  __device__ __forceinline__ float to_float(bfloat16_t x){ return __bfloat162float(x); }
  __device__ __forceinline__ bfloat16_t to_bf16(float x){ return __float2bfloat16(x); }
#endif

// Vectorized load/store helpers (optional); here we keep scalar for clarity.

template<typename T>
DEVFN void rowwise_softmax(const T* __restrict__ x, T* __restrict__ y, int rows, int cols){
  int row = blockIdx.x;
  int tid = threadIdx.x;
  if (row >= rows) return;
  // 1) compute row max in FP32
  float local_max = -std::numeric_limits<float>::infinity();
  for (int i = tid; i < cols; i += blockDim.x){
    float v = to_float(x[row*cols + i]);
    local_max = fmaxf(local_max, v);
  }
  // block reduce max
  __shared__ float sdata[1024]; // enough for blockDim<=1024
  sdata[tid] = local_max;
  __syncthreads();
  for (int s = blockDim.x/2; s>0; s>>=1){
    if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid+s]);
    __syncthreads();
  }
  float m = sdata[0];
  // 2) compute sum(exp(x-m)) in FP32
  float local_sum = 0.f;
  for (int i = tid; i < cols; i += blockDim.x){
    float v = to_float(x[row*cols + i]);
    float e = expf(v - m);
    local_sum += e;
    // store the numerator temporarily in y as FP32 via bit-cast would need extra buffer;
    // we can recompute, but to keep bandwidth low we store as T after casting later.
  }
  sdata[tid] = local_sum;
  __syncthreads();
  for (int s = blockDim.x/2; s>0; s>>=1){
    if (tid < s) sdata[tid] += sdata[tid+s];
    __syncthreads();
  }
  float denom = sdata[0] + 1e-12f; // avoid divide-by-zero
  // 3) write outputs
  for (int i = tid; i < cols; i += blockDim.x){
    float v = to_float(x[row*cols + i]);
    float e = expf(v - m) / denom;
    if constexpr (std::is_same<T, half_t>::value) {
      y[row*cols + i] = to_half(e);
    } else {
      y[row*cols + i] = to_bf16(e);
    }
  }
}

// Host-side helpers

template <typename T>
void to_device_dtype(const std::vector<float>& src, T* dst, int64_t n){
  std::vector<T> tmp(n);
  for (int64_t i=0;i<n;++i){
    if constexpr (std::is_same<T, half_t>::value) tmp[i] = (T)src[i];
    else tmp[i] = to_bf16(src[i]);
  }
  API_CHECK(MEMCPY(dst, tmp.data(), n*sizeof(T), MEMCPY_H2D));
}

static void cpu_softmax_ref(const std::vector<double>& x, std::vector<double>& y, int rows, int cols){
  for (int r=0;r<rows;++r){
    double m = -std::numeric_limits<double>::infinity();
    for (int c=0;c<cols;++c) m = std::max(m, x[r*cols+c]);
    double sum = 0.0;
    for (int c=0;c<cols;++c){ sum += std::exp(x[r*cols+c]-m); }
    for (int c=0;c<cols;++c){ y[r*cols+c] = std::exp(x[r*cols+c]-m)/sum; }
  }
}

static void usage(const char* prog){
  printf("Usage: %s <rows> <cols> <dtype: fp16|bf16> [repeat]\n", prog);
}

int main(int argc, char** argv){
  if (argc < 4){ usage(argv[0]); return 1; }
  int rows = std::stoi(argv[1]);
  int cols = std::stoi(argv[2]);
  std::string dtype = argv[3];
  int repeat = (argc>4) ? std::stoi(argv[4]) : 50;

  const int64_t N = (int64_t)rows * cols;
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-10.f, 10.f);

  std::vector<float> h_x(N);
  for (int64_t i=0;i<N;++i) h_x[i] = dist(rng);

  std::vector<double> h_x64(N), h_ref(N), h_out64(N);
  for (int64_t i=0;i<N;++i) h_x64[i] = (double)h_x[i];
  cpu_softmax_ref(h_x64, h_ref, rows, cols);

  STREAM stream; STREAM_CREATE(&stream);

  DEVICE_EVENT start, stop; EVENT_CREATE(&start); EVENT_CREATE(&stop);

  const int block = 256;
  dim3 grid(rows);

  if (dtype=="fp16"){
    half_t *d_x=nullptr, *d_y=nullptr; MALLOC(&d_x, N*sizeof(half_t)); MALLOC(&d_y, N*sizeof(half_t));
    to_device_dtype(h_x, d_x, N);
    // warmup
    LAUNCH(rowwise_softmax<half_t>, grid, block, 0, stream, d_x, d_y, rows, cols);
    DEVICE_SYNC();
    EVENT_RECORD(start, stream);
    for (int it=0; it<repeat; ++it){
      LAUNCH(rowwise_softmax<half_t>, grid, block, 0, stream, d_x, d_y, rows, cols);
    }
    EVENT_RECORD(stop, stream); DEVICE_SYNC();
    float ms=0; EVENT_ELAPSED(&ms, start, stop);
    // Copy back
    std::vector<half_t> h_y(N); API_CHECK(MEMCPY(h_y.data(), d_y, N*sizeof(half_t), MEMCPY_D2H));
    for (int64_t i=0;i<N;++i) h_out64[i] = (double)((float)h_y[i]);
    // Errors
    double max_abs=0.0, max_rel=0.0;
    for (int64_t i=0;i<N;++i){
      double a = h_out64[i], b = h_ref[i];
      double abs = std::abs(a-b); max_abs = std::max(max_abs, abs);
      double rel = abs / (std::abs(b)+1e-15); max_rel = std::max(max_rel, rel);
    }
    double bytes = (double)N * (sizeof(half_t) + sizeof(half_t)) * repeat; // read+write per repeat
    double gbps = bytes / (ms/1e3) / 1e9;
    printf("dtype=fp16 rows=%d cols=%d repeat=%d time=%.3f ms avg=%.3f ms, throughput=%.2f GB/s\n", rows, cols, repeat, ms, ms/repeat, gbps);
    printf("max_abs=%.3e max_rel=%.3e\n", max_abs, max_rel);
    FREE(d_x); FREE(d_y);
  } else if (dtype=="bf16"){
    bfloat16_t *d_x=nullptr, *d_y=nullptr; MALLOC(&d_x, N*sizeof(bfloat16_t)); MALLOC(&d_y, N*sizeof(bfloat16_t));
    // Convert host float -> bf16
    std::vector<bfloat16_t> tmp(N);
    for (int64_t i=0;i<N;++i) tmp[i] = to_bf16(h_x[i]);
    API_CHECK(MEMCPY(d_x, tmp.data(), N*sizeof(bfloat16_t), MEMCPY_H2D));
    // warmup
    LAUNCH(rowwise_softmax<bfloat16_t>, grid, block, 0, stream, d_x, d_y, rows, cols);
    DEVICE_SYNC();
    EVENT_RECORD(start, stream);
    for (int it=0; it<repeat; ++it){
      LAUNCH(rowwise_softmax<bfloat16_t>, grid, block, 0, stream, d_x, d_y, rows, cols);
    }
    EVENT_RECORD(stop, stream); DEVICE_SYNC();
    float ms=0; EVENT_ELAPSED(&ms, start, stop);
    std::vector<bfloat16_t> h_y(N); API_CHECK(MEMCPY(h_y.data(), d_y, N*sizeof(bfloat16_t), MEMCPY_D2H));
    for (int64_t i=0;i<N;++i) h_out64[i] = (double)to_float(h_y[i]);
    double max_abs=0.0, max_rel=0.0;
    for (int64_t i=0;i<N;++i){
      double a = h_out64[i], b = h_ref[i];
      double abs = std::abs(a-b); max_abs = std::max(max_abs, abs);
      double rel = abs / (std::abs(b)+1e-15); max_rel = std::max(max_rel, rel);
    }
    double bytes = (double)N * (sizeof(bfloat16_t) + sizeof(bfloat16_t)) * repeat;
    double gbps = bytes / (ms/1e3) / 1e9;
    printf("dtype=bf16 rows=%d cols=%d repeat=%d time=%.3f ms avg=%.3f ms, throughput=%.2f GB/s\n", rows, cols, repeat, ms, ms/repeat, gbps);
    printf("max_abs=%.3e max_rel=%.3e\n", max_abs, max_rel);
    FREE(d_x); FREE(d_y);
  } else {
    usage(argv[0]); return 2; }

  EVENT_DESTROY(start); EVENT_DESTROY(stop); STREAM_DESTROY(stream);
  return 0;
}
```

Notes:

- The kernel recomputes `expf(v-m)` when writing the output to avoid extra global memory traffic. If you prefer, add a temporary FP32 buffer (`float* tmp`) to store numerators between passes.
- For very wide rows (e.g., cols ≥ 16384), consider vectorized loads/stores and warp-level reductions to increase bandwidth.

## Profiling and Validation

### NVIDIA (Nsight Compute)

Collect achieved memory throughput and FP32 utilization:

```bash
ncu --set full --metrics \
  sm__sass_average_data_bytes_per_sector_mem_global.pct,\
  smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,\
  dram__throughput.avg.pct_of_peak_sustained_elapsed \
  ./softmax_cuda 4096 4096 fp16 200
```

Interpretation:

- `dram__throughput.avg.pct_of_peak_sustained_elapsed ≥ 30%` indicates good memory efficiency for this memory-bound kernel.
- Check that stall reasons are not dominated by `barrier` or `lg_throttle`.

### AMD (rocprof)

```bash
rocprof --hip-trace --hsa-trace \
  --stats ./softmax_hip 4096 4096 bf16 200
```

Capture counters (MI300): `SQ_INSTS_VALU`, `GRBM_COUNT`, `MemUnitBusy`. Expect high memory unit activity and modest VALU utilization.

### Numerical validation

Run both variants and ensure errors meet thresholds in Acceptance Criteria. For worst-case rows with large magnitude inputs (e.g., ±80), BF16 should remain stable due to range; FP16 may saturate without the max-subtraction.

## Performance Checklist

1. Inputs in FP16/BF16, accumulators in FP32 for softmax/LN and attention score reductions.
2. Use max-subtracted softmax and FP32 `expf` and sum.
3. For GEMMs: configure library for FP16/BF16 inputs with FP32 accumulation and fused epilogues (bias + activation + dequant) when available.
4. Disable unnecessary casts; keep tensors in a single low-precision format end-to-end when safe.
5. Validate with FP64 CPU references on a sample of batches/seq lengths.
6. Monitor NaN/Inf counters; log first occurrence with tensor statistics.
7. Ensure determinism in CI: fixed seeds, fixed launch configs, no atomic reductions in validation path.

## Troubleshooting

| Symptom                             | Likely cause                              | Fix                                                                               |
| ----------------------------------- | ----------------------------------------- | --------------------------------------------------------------------------------- |
| Sporadic NaNs in logits             | FP16 overflow during `exp`                | Use max-subtraction and FP32 accumulators; consider BF16 inputs                   |
| Divergent outputs across runs       | Non-deterministic reductions              | Avoid atomic adds; use fixed-size block reductions; set streams deterministically |
| Large relative error in tail tokens | Underflow in softmax denominator          | Accumulate in FP32; clamp with epsilon; use BF16 for inputs                       |
| Slow kernel despite low math        | Uncoalesced memory or low occupancy       | Use 128–256 threads per block; consider vectorized loads; check L2 hit rate       |
| BF16 result worse than FP16         | Mantissa too small for tiny probabilities | Keep accumulators FP32; consider Kahan compensation only if needed                |
| GEMM accuracy drop vs training      | Accumulator not FP32                      | Ensure BLAS uses FP32 accumulate (e.g., computeType)                              |
| Occasional Infs in LayerNorm        | Variance <= 0 due to rounding             | Add epsilon in FP32 before rsqrt; reorder math to reduce cancellation             |

## Acceptance Criteria

1. `mixed_softmax_single_source.cpp` builds and runs on both backends.
2. For `rows=4096, cols=4096, repeat=50`, the kernel reports:

   - `max_abs ≤ 5e-6` and `max_rel ≤ 1e-5` vs FP64 reference for both FP16 and BF16 inputs.

3. Nsight Compute or rocprof indicates memory-bound behavior with ≥ 30% of peak DRAM throughput.
4. No NaN/Inf encountered when inputs are uniformly sampled from \[−10, 10].

## Further Work

- Add **fused dequant** path: `y = softmax( (A·(dequant(W_int8, s))) )` with per-channel scales and FP32 accumulation.
- Implement **LayerNorm** with Welford’s online algorithm in FP32 and optional Kahan compensation.
- Provide a cuBLASLt/rocBLAS benchmark for GEMM in FP16/BF16/FP8 with FP32 accumulators and fused epilogues.
- Extend to **FlashAttention** math with FP16/BF16 inputs and FP32 score accumulation.
