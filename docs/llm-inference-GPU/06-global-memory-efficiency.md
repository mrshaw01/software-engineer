# Global Memory Efficiency

## Summary

Global memory traffic dominates the runtime of LLM inference, especially during decode where arithmetic intensity is low. This topic provides concrete methods to maximize effective bandwidth: coalesced and vectorized accesses, cache-friendly layouts, tiling into shared memory/LDS, and avoiding redundant reads. We include runnable CUDA/HIP microbenchmarks that quantify GB/s for naïve vs. optimized kernels and a fused bias+activation example reflecting real decode-path work.

## Why It Matters for LLM Inference

- **Prefill** often reaches high tensor core utilization, but still wastes time if QKV/MLP tensors are fetched with poor coalescing or cache thrashing.
- **Decode** is usually **memory-bound**: small GEMMs and elementwise ops shuttle activations/KV via HBM repeatedly. Each uncoalesced or redundant read directly reduces tokens/s.
- Improving global memory efficiency increases both **throughput (tokens/s)** and reduces **tail latency** for small batches.

## Key Concepts and Formulas

- **Coalescing**: threads in a warp/wavefront should access a contiguous, properly aligned segment (typically 128B or 256B). Misaligned or strided access inflates transactions.
- **Vectorized I/O**: use 128-bit types (`float4`, `int4`, `__half2x2`) to cut instruction count and improve L2/L1 transaction efficiency.
- **Arithmetic Intensity (AI)**: `AI = FLOPs / Bytes`. Kernels with `AI < 10` on modern GPUs are often memory-bound. Example for bias+ReLU:

  - Compute: 1 add + 1 max ≈ 2 FLOPs per element.
  - Traffic (naïve): read `x` (4B), read `bias` (4B), write `y` (4B) ⇒ 12B.
  - `AI ≈ 2 / 12 ≈ 0.167 FLOPs/B` ⇒ strongly memory-bound.

- **Achieved Bandwidth**: `B_achieved = Bytes_moved / time`. For copy of `N` floats: `Bytes_moved = 2 * N * 4` (read + write).
- **Alignment**: prefer 16B alignment (or 32B when available). `cudaMalloc/hipMalloc` typically return ≥256B aligned pointers.

## GPU Deep Dive

### NVIDIA specifics

- **Warps and transactions**: 32-thread warps generate 32×4B = 128B per float load when perfectly coalesced.
- **L2 and L1/TEX**: L2 is shared across SMs; L1/TEX per-SM. Strided patterns amplify sector misses.
- **Tensor Cores**: do not help if memory limits throughput; feed them with fused epilogues and vectorized global I/O.

### AMD specifics

- **Wavefronts**: 32 or 64 threads depending on arch (MI200/MI300 use 64). Coalescing rules analogous; contiguous 256B segments are ideal.
- **LDS**: high-bandwidth on-chip scratchpad; use to stage reused vectors (e.g., bias, small constants) to avoid repeated HBM fetches.
- **MFMA/XDLOPs**: as on NVIDIA, compute units stall without efficient global I/O.

## Implementation

We provide a single-source C++17 file that compiles under CUDA (NVCC) or ROCm/HIP (HIPCC). It benchmarks:

1. Scalar copy (naïve)
2. Vectorized copy (`float4`)
3. Bias+ReLU (naïve)
4. Bias+ReLU with shared/LDS tiling and vectorization

Files:

- `topics/06-global-memory-efficiency/code/global_mem_efficiency.cu` (single-source)

### Build

CUDA (set your SM):

```
nvcc -O3 -std=c++17 -arch=${SM_ARCH} -lineinfo topics/06-global-memory-efficiency/code/global_mem_efficiency.cu -o gmem
```

Example: `SM_ARCH=sm_80` (A100), `sm_90` (H100), `sm_89` (L40S).

ROCm (set your GFX):

```
hipcc -O3 -std=c++17 --offload-arch=${GFX_ARCH} topics/06-global-memory-efficiency/code/global_mem_efficiency.cu -o gmem
```

Example: `GFX_ARCH=gfx90a` (MI200), `gfx942` (MI300X).

### Run

```
./gmem --n 268435456 --hidden 4096 --iters 100
```

- `n`: number of elements (floats). 268,435,456 ≈ 1 Gi elements ⇒ 4 GiB per buffer; reduce if memory-limited.
- `hidden`: bias length (broadcast period).
- `iters`: inner-loop repeats to amplify timing signal.

### Code: `global_mem_efficiency.cu`

```cpp
// Single-source CUDA/HIP microbenchmarks for global memory efficiency
// Build with NVCC or HIPCC (see README). C++17.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cmath>
#include <cstring>
#include <string>
#include <chrono>

#if defined(__HIP_PLATFORM_AMD__)
  #include <hip/hip_runtime.h>
  #define DEVFN __global__
  #define API_CHECK(x) do { auto e = (x); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %d at %s:%d\n", e, __FILE__, __LINE__); abort(); } } while(0)
  using DeviceStream = hipStream_t;
  using DeviceEvent = hipEvent_t;
  static inline void EventCreate(DeviceEvent* e){ API_CHECK(hipEventCreate(e)); }
  static inline void EventRecord(DeviceEvent e, DeviceStream s){ API_CHECK(hipEventRecord(e,s)); }
  static inline float EventElapsed(DeviceEvent a, DeviceEvent b){ float ms; API_CHECK(hipEventElapsedTime(&ms,a,b)); return ms; }
  static inline void EventDestroy(DeviceEvent e){ API_CHECK(hipEventDestroy(e)); }
  static inline void StreamCreate(DeviceStream* s){ API_CHECK(hipStreamCreate(s)); }
  static inline void StreamDestroy(DeviceStream s){ API_CHECK(hipStreamDestroy(s)); }
  static inline void DeviceSync(){ API_CHECK(hipDeviceSynchronize()); }
  template <typename T> static inline void* Malloc(size_t bytes){ void* p=nullptr; API_CHECK(hipMalloc(&p, bytes)); return p; }
  static inline void MemcpyH2D(void* d, const void* h, size_t b){ API_CHECK(hipMemcpy(d,h,b, hipMemcpyHostToDevice)); }
  static inline void MemcpyD2H(void* h, const void* d, size_t b){ API_CHECK(hipMemcpy(h,d,b, hipMemcpyDeviceToHost)); }
  static inline void Free(void* p){ API_CHECK(hipFree(p)); }
  #define LAUNCH(kernel, grid, block, shmem, stream, ...) \
    hipLaunchKernelGGL(kernel, grid, block, shmem, stream, __VA_ARGS__)
#else
  #include <cuda_runtime.h>
  #define DEVFN __global__
  #define API_CHECK(x) do { auto e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %d at %s:%d: %s\n", e, __FILE__, __LINE__, cudaGetErrorString(e)); abort(); } } while(0)
  using DeviceStream = cudaStream_t;
  using DeviceEvent = cudaEvent_t;
  static inline void EventCreate(DeviceEvent* e){ API_CHECK(cudaEventCreate(e)); }
  static inline void EventRecord(DeviceEvent e, DeviceStream s){ API_CHECK(cudaEventRecord(e,s)); }
  static inline float EventElapsed(DeviceEvent a, DeviceEvent b){ float ms; API_CHECK(cudaEventElapsedTime(&ms,a,b)); return ms; }
  static inline void EventDestroy(DeviceEvent e){ API_CHECK(cudaEventDestroy(e)); }
  static inline void StreamCreate(DeviceStream* s){ API_CHECK(cudaStreamCreate(s)); }
  static inline void StreamDestroy(DeviceStream s){ API_CHECK(cudaStreamDestroy(s)); }
  static inline void DeviceSync(){ API_CHECK(cudaDeviceSynchronize()); }
  template <typename T> static inline void* Malloc(size_t bytes){ void* p=nullptr; API_CHECK(cudaMalloc(&p, bytes)); return p; }
  static inline void MemcpyH2D(void* d, const void* h, size_t b){ API_CHECK(cudaMemcpy(d,h,b, cudaMemcpyHostToDevice)); }
  static inline void MemcpyD2H(void* h, const void* d, size_t b){ API_CHECK(cudaMemcpy(h,d,b, cudaMemcpyDeviceToHost)); }
  static inline void Free(void* p){ API_CHECK(cudaFree(p)); }
  #define LAUNCH(kernel, grid, block, shmem, stream, ...) \
    kernel<<<grid, block, shmem, stream>>>(__VA_ARGS__)
#endif

// Utility: round-up division
static inline uint64_t div_up(uint64_t a, uint64_t b){ return (a + b - 1) / b; }

// 1) Naive scalar copy: each thread processes one element per loop stride
DEVFN void copy_scalar(const float* __restrict__ x, float* __restrict__ y,
                       uint64_t n, int iters){
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
  for(int it=0; it<iters; ++it){
    for(uint64_t i = idx; i < n; i += stride){
      y[i] = x[i];
    }
  }
}

// 2) Vectorized copy: 128-bit via float4, with tail handling
DEVFN void copy_vec4(const float* __restrict__ x, float* __restrict__ y,
                     uint64_t n, int iters){
  const uint64_t n4 = n / 4; // vectorizable chunks
  auto vx = reinterpret_cast<const float4*>(x);
  auto vy = reinterpret_cast<float4*>(y);
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
  for(int it=0; it<iters; ++it){
    for(uint64_t i = idx; i < n4; i += stride){
      float4 t = vx[i];
      vy[i] = t;
    }
    // tail (scalar)
    for(uint64_t i = n4*4 + idx; i < n; i += stride){
      y[i] = x[i];
    }
  }
}

// 3) Bias+ReLU naive: y[i] = max(0, x[i] + bias[i % hidden])
DEVFN void bias_relu_naive(const float* __restrict__ x, const float* __restrict__ bias,
                           float* __restrict__ y, uint64_t n, int hidden, int iters){
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
  for(int it=0; it<iters; ++it){
    for(uint64_t i = idx; i < n; i += stride){
      float v = x[i] + bias[i % hidden];
      y[i] = v > 0.f ? v : 0.f;
    }
  }
}

// 4) Bias+ReLU tiled + vectorized: stage bias in shared/LDS and use float4
template<int TILE>
DEVFN void bias_relu_tiled_vec4(const float* __restrict__ x, const float* __restrict__ bias,
                                float* __restrict__ y, uint64_t n, int hidden, int iters){
  extern __shared__ float sbias[]; // TILE floats
  const int tid = threadIdx.x;
  const int block = blockIdx.x;
  const int threads = blockDim.x;

  const uint64_t elems_per_block = (uint64_t)TILE * 1024ULL; // heuristic chunk per block per iter
  uint64_t block_start = (uint64_t)block * elems_per_block;

  for(int it=0; it<iters; ++it){
    for(uint64_t base = block_start; base < n; base += (uint64_t)gridDim.x * elems_per_block){
      // stage bias: cover TILE elements of bias cyclically
      for(int o = tid; o < TILE; o += threads){
        sbias[o] = bias[o % hidden];
      }
      __syncthreads();

      // process chunk using float4
      uint64_t start = base + tid * 4ULL; // starting element index for this thread (vectorized)
      uint64_t end = min(base + elems_per_block, n);
      for(uint64_t i = start; i + 3 < end; i += (uint64_t)threads * 4ULL){
        float4 xv = *reinterpret_cast<const float4*>(&x[i]);
        // bias index maps per element; use modulo within TILE window
        int bi0 = (int)((i + 0) % TILE);
        int bi1 = (int)((i + 1) % TILE);
        int bi2 = (int)((i + 2) % TILE);
        int bi3 = (int)((i + 3) % TILE);
        float4 bv = {sbias[bi0], sbias[bi1], sbias[bi2], sbias[bi3]};
        float4 out = {xv.x + bv.x, xv.y + bv.y, xv.z + bv.z, xv.w + bv.w};
        // ReLU
        out.x = out.x > 0.f ? out.x : 0.f;
        out.y = out.y > 0.f ? out.y : 0.f;
        out.z = out.z > 0.f ? out.z : 0.f;
        out.w = out.w > 0.f ? out.w : 0.f;
        *reinterpret_cast<float4*>(&y[i]) = out;
      }
      __syncthreads();
    }
  }
}

// Host harness
struct Args { uint64_t n = 1ull<<28; int hidden=4096; int iters=50; int block=256; int grid=0; };

static Args parse(int argc, char** argv){
  Args a; for(int i=1;i<argc;++i){
    if(!strcmp(argv[i],"--n") && i+1<argc) a.n = std::strtoull(argv[++i],nullptr,10);
    else if(!strcmp(argv[i],"--hidden") && i+1<argc) a.hidden = std::atoi(argv[++i]);
    else if(!strcmp(argv[i],"--iters") && i+1<argc) a.iters = std::atoi(argv[++i]);
    else if(!strcmp(argv[i],"--block") && i+1<argc) a.block = std::atoi(argv[++i]);
    else if(!strcmp(argv[i],"--grid") && i+1<argc) a.grid = std::atoi(argv[++i]);
  } return a;
}

static double gbps_copy(uint64_t n, int iters, double ms){
  // read + write per iter
  double bytes = (double)iters * (double)n * 2.0 * sizeof(float);
  return (bytes / (ms/1e3)) / 1e9;
}

static double gbps_bias(uint64_t n, int iters, double ms){
  // Effective traffic model: read x + read bias (amortized) + write y
  // For naive: count full bias read per element. For tiled: bias amortized.
  // Here report lower bound assuming ideal bias reuse → x+y only (8B/elt) + small bias.
  double bytes = (double)iters * (double)n * (2.0 * sizeof(float));
  return (bytes / (ms/1e3)) / 1e9;
}

int main(int argc, char** argv){
  Args a = parse(argc, argv);
  if(a.grid==0){
    // default: enough blocks to cover device reasonably
    int dev=0;
#if defined(__HIP_PLATFORM_AMD__)
    hipDeviceProp_t prop; API_CHECK(hipGetDeviceProperties(&prop, dev));
    a.grid = prop.multiProcessorCount * 8; // heuristic
#else
    cudaDeviceProp prop; API_CHECK(cudaGetDeviceProperties(&prop, dev));
    a.grid = prop.multiProcessorCount * 8; // heuristic
#endif
  }
  printf("n=%llu hidden=%d iters=%d grid=%d block=%d\n", (unsigned long long)a.n, a.hidden, a.iters, a.grid, a.block);

  size_t bytes = a.n * sizeof(float);
  size_t bbytes = a.hidden * sizeof(float);
  std::vector<float> hx(a.n, 1.0f), hb(a.hidden, 0.5f), hy(a.n, 0.0f);

  float* dx = (float*)Malloc<float>(bytes);
  float* dy = (float*)Malloc<float>(bytes);
  float* db = (float*)Malloc<float>(bbytes);
  MemcpyH2D(dx, hx.data(), bytes);
  MemcpyH2D(db, hb.data(), bbytes);

  DeviceStream stream; StreamCreate(&stream);
  DeviceEvent e0,e1; EventCreate(&e0); EventCreate(&e1);

  // Warmup
  LAUNCH(copy_scalar, a.grid, a.block, 0, stream, dx, dy, a.n, 1);
  DeviceSync();

  auto run = [&](const char* name, auto kernel, size_t shmem){
    EventRecord(e0, stream);
    kernel; // launch already encoded in caller
    EventRecord(e1, stream);
    DeviceSync();
    float ms = EventElapsed(e0, e1);
    return ms;
  };

  // 1) scalar copy
  float ms_copy_scalar = 0.f; {
    EventRecord(e0, stream);
    LAUNCH(copy_scalar, a.grid, a.block, 0, stream, dx, dy, a.n, a.iters);
    EventRecord(e1, stream); DeviceSync(); ms_copy_scalar = EventElapsed(e0,e1);
    printf("copy_scalar: %.3f ms, %.2f GB/s\n", ms_copy_scalar, gbps_copy(a.n,a.iters,ms_copy_scalar));
  }

  // 2) vec4 copy
  float ms_copy_vec4 = 0.f; {
    EventRecord(e0, stream);
    LAUNCH(copy_vec4, a.grid, a.block, 0, stream, dx, dy, a.n, a.iters);
    EventRecord(e1, stream); DeviceSync(); ms_copy_vec4 = EventElapsed(e0,e1);
    printf("copy_vec4:   %.3f ms, %.2f GB/s\n", ms_copy_vec4, gbps_copy(a.n,a.iters,ms_copy_vec4));
  }

  // 3) bias relu naive
  float ms_bias_naive = 0.f; {
    EventRecord(e0, stream);
    LAUNCH(bias_relu_naive, a.grid, a.block, 0, stream, dx, db, dy, a.n, a.hidden, a.iters);
    EventRecord(e1, stream); DeviceSync(); ms_bias_naive = EventElapsed(e0,e1);
    printf("bias_relu_naive: %.3f ms, ~%.2f GB/s (lower-bound)\n", ms_bias_naive, gbps_bias(a.n,a.iters,ms_bias_naive));
  }

  // 4) bias relu tiled vec4 (TILE must divide hidden for perfect reuse; OK if not)
  float ms_bias_tiled = 0.f; {
    const int TILE = 4096; // set near hidden for ideal reuse
    EventRecord(e0, stream);
    LAUNCH((bias_relu_tiled_vec4<TILE>), a.grid, a.block, TILE*sizeof(float), stream, dx, db, dy, a.n, a.hidden, a.iters);
    EventRecord(e1, stream); DeviceSync(); ms_bias_tiled = EventElapsed(e0,e1);
    printf("bias_relu_tiled_vec4: %.3f ms, ~%.2f GB/s (lower-bound)\n", ms_bias_tiled, gbps_bias(a.n,a.iters,ms_bias_tiled));
  }

  // Validate simple correctness on small sample
  MemcpyD2H(hy.data(), dy, std::min<size_t>(bytes, 1024*sizeof(float)));
  for(size_t i=0;i<std::min<size_t>(a.n, 1024);++i){ float ref = hx[i] + hb[i % a.hidden]; ref = ref>0.f?ref:0.f; if (fabs(hy[i]-ref) > 1e-6f){
      fprintf(stderr, "Validation failed at %zu: got %f expected %f\n", i, hy[i], ref); return 1; }}

  Free(dx); Free(dy); Free(db);
  EventDestroy(e0); EventDestroy(e1); StreamDestroy(stream);
  printf("OK\n");
  return 0;
}
```

## Profiling and Validation

### NVIDIA

- Nsight Compute (per-kernel):

```
ncu --set full --kernel-name regex:copy_.* ./gmem --n 134217728 --iters 100
ncu --set full --kernel-name regex:bias_.* ./gmem --n 134217728 --hidden 4096 --iters 100
```

Key metrics to inspect:

- `dram__throughput.avg.pct_of_peak_sustained_elapsed` ≥ 70% on `copy_vec4`.
- `l2_tex__throughput.avg.pct_of_peak_sustained_elapsed` higher on vectorized vs. scalar.
- `smsp__inst_executed_pipe_lsu.sum` reduced (fewer memory ops) for vectorized version.

### AMD

- rocprof (timing + memory)

```
rocprof --stats ./gmem --n 134217728 --iters 100
```

Key counters to monitor (availability varies by ROCm):

- Achieved HBM BW from timeline should approach ≥70% device peak on `copy_vec4`.
- `SQ_WAVES` stable and high occupancy; low VALU stalls for memory-bound kernels.

### Validation Thresholds

- Vectorized copy must deliver ≥1.4× the scalar copy GB/s on the same problem size.
- Tiled `bias_relu` must outperform naïve by ≥1.2× on typical `hidden∈{2048,4096,8192}`.

## Performance Checklist

- Ensure pointers are 16B-aligned; default `cudaMalloc/hipMalloc` satisfies this.
- Use `float4`/`int4` vectorized loads/stores when data size allows.
- Favor SoA layouts for per-token features to enable contiguous per-thread accesses.
- Stage small, reused vectors (bias, LayerNorm scale) in shared/LDS.
- Batch multiple elementwise ops into a single pass (fused epilogue) to avoid extra HBM traffic.
- Tune `grid` to `~8×SM/CU` and `block` to 128–512 for bandwidth tests; validate no register spills.

## Troubleshooting

| Symptom                              | Likely cause                             | Fix                                                                                                 |
| ------------------------------------ | ---------------------------------------- | --------------------------------------------------------------------------------------------------- |
| GB/s plateaus far below 50% of peak  | Uncoalesced or misaligned accesses       | Reinterpret as `float4`; ensure leading dimension is multiple of 4 elements; pad tensors            |
| Vectorized kernel slower than scalar | Tail-handling dominates, low N           | Increase problem size or specialize kernels for multiples of 4                                      |
| Nsight shows L2 sector misses high   | Strided access pattern                   | Change layout to make per-thread contiguous; transpose offline or use tiled shared-memory transpose |
| Occupancy too low (<25%)             | Excess registers or large shared memory  | Reduce unrolling, use smaller tiles, or increase blocks                                             |
| Validation mismatch                  | Race or aliasing with `reinterpret_cast` | Ensure alignment, avoid overlapping src/dst, add `__restrict__`                                     |
| HIP build fails on `float4`          | Missing include or name clash            | Use `typedef struct {float x,y,z,w;} float4;` or include `<hip/hip_runtime.h>`                      |
| Spiky latency                        | PCIe paging or power states              | Pin host memory and repeat runs; set persistence mode; warm up                                      |

## Acceptance Criteria

- Code compiles under NVCC 12.x and HIPCC 6.x with the provided commands.
- `copy_vec4` achieves ≥70% of device peak bandwidth on large `n` (≥128M elements) or ≥1.4× over scalar baseline.
- `bias_relu_tiled_vec4` outperforms naïve by ≥1.2× for `hidden≥2048`.
- Nsight/rocprof runs complete and demonstrate reduced memory op count and higher L2 throughput on vectorized kernels.
- CPU vs. GPU results match within `1e-6` on sampled outputs.

## Further Work

- Add FP16/BF16 variants using `__half2`/`ushort2` to double effective element throughput.
- Introduce asynchronous copies (`cp.async` on NVIDIA) and `lds_direct` patterns on AMD to overlap global fetch with compute.
- Evaluate cache line sizes and prefetch distance per-arch; specialize tiles for MI300 vs. H100.
- Extend fused epilogues: bias + activation + residual + quantize-dequant to minimize HBM traffic.
- Integrate with a small attention/MLP forward to show end-to-end tokens/s impact.
