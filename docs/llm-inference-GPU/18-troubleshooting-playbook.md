# Troubleshooting Playbook for GPU LLM Inference (NVIDIA CUDA / AMD ROCm)

## 2. Summary

This playbook provides a systematic approach to diagnosing and fixing correctness, performance, and stability issues in LLM inference on NVIDIA and AMD GPUs. It prioritizes fast isolation, minimal repros, and metrics-driven validation. You will learn a consistent workflow (reproduce → isolate → measure → fix → verify) and use a compact CUDA/HIP microbenchmark to detect classic memory and launch-path pitfalls. The result is a repeatable method to restore expected tokens/sec and latency while preserving numerical correctness.

## 3. Why It Matters for LLM Inference

Prefill is bandwidth-bound and sensitive to global-memory efficiency and fusion; decode is launch/latency-bound and sensitive to kernel overheads, KV cache locality, and graphs/persistent execution. Failures (NaNs, OOM, stalls) often appear only under production batching, prompting targeted repros and counters. A disciplined playbook reduces MTTR, avoids cargo-cult fixes, and keeps SLOs intact.

## 4. Key Concepts and Formulas

- **Arithmetic intensity (AI)**: `AI = FLOPs / Bytes`. Low AI (e.g., copy, LN) is bandwidth-bound; high AI (GEMM) is compute-bound.
- **Achieved bandwidth (copy)**:
  For an elementwise copy (read + write), bytes moved per element = `2 * sizeof(T)`.
  `GBps = (N_effective * 2 * sizeof(T)) / time_seconds / 1e9`.
- **Token latency decomposition**:
  `t_token ≈ t_sched + t_launch + t_kernel + t_sync + t_comm`.
  Decode improvements favor reducing `t_launch` (graphs/persistent), improving KV hit locality, and avoiding pageable H2D/DtoH on critical paths.
- **KV cache footprint (bytes)** (for reference): `B ≈ 2 * L * H * Dh * dtype_size * batch`, where L=seq len, H=heads, Dh=head dim; factor 2 for K and V.

## 5. GPU Deep Dive

### NVIDIA (CUDA)

- **Execution**: warps of 32 threads; SM-scheduling; Tensor Cores for FP16/BF16/FP8; L1 per-SM + unified L2.
- **Hot counters**: global load/store efficiency, achieved occupancy, DRAM throughput, L2 hit rate, tensor core utilization, warp stall reasons (memory throttle, barrier, math pipe).
- **Common pitfalls**: uncoalesced loads, misalignment (lack of `float4`/`__half2`), small-kernel storm (decode), pageable transfers, missing graphs/persistence.

### AMD (ROCm/HIP)

- **Execution**: wavefronts of 64; Compute Units (CUs); MFMA/XDLOPs for matrix ops; LDS analogous to shared memory.
- **Hot counters**: SQ waves/occupancy, VALU vs MFMA mix, L2 hit/miss, DRAM BW, LDS bank conflicts, wave stall reasons.
- **Common pitfalls**: same memory issues; ensure `--offload-arch` matches GFX; pin host memory; avoid fine-grain allocations on hot paths; verify RCCL topology.

## 6. Implementation (Runnable CUDA/HIP)

Minimal single-source bandwidth/stride diagnostic. It exposes three typical issues:

1. uncoalesced access via stride, 2) misalignment vs `float4` vectorization, 3) timing with events and correctness check.

**Path**: `topics/18-troubleshooting-playbook/code/bandwidth_stride.cpp`

```cpp
// SPDX-License-Identifier: MIT
// bandwidth_stride.cpp : CUDA/HIP single-source memory pattern diagnostic.
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
  #define API_CHECK(x) do { auto _e=(x); if(_e!=hipSuccess){fprintf(stderr,"HIP error %d:%s at %s:%d\n",(int)_e, hipGetErrorString(_e), __FILE__, __LINE__); std::abort();} } while(0)
  #define DEV_ALLOC hipMalloc
  #define DEV_FREE  hipFree
  #define MEMCPY_H2D(dst,src,bytes) API_CHECK(hipMemcpy(dst,src,bytes,hipMemcpyHostToDevice))
  #define MEMCPY_D2H(dst,src,bytes) API_CHECK(hipMemcpy(dst,src,bytes,hipMemcpyDeviceToHost))
  #define EVENT_CREATE(e) API_CHECK(hipEventCreate(&e))
  #define EVENT_RECORD(e,s) API_CHECK(hipEventRecord(e,s))
  #define EVENT_SYNC(e) API_CHECK(hipEventSynchronize(e))
  #define EVENT_ELAPSED(ms,a,b) API_CHECK(hipEventElapsedTime(&ms,a,b))
  #define EVENT_DESTROY(e) API_CHECK(hipEventDestroy(e))
  #define STREAM hipStream_t
  #define STREAM_CREATE(s) API_CHECK(hipStreamCreate(&s))
  #define STREAM_DESTROY(s) API_CHECK(hipStreamDestroy(s))
  #define LAUNCH(grid,block,shmem,stream,...) hipLaunchKernelGGL(__VA_ARGS__, grid, block, shmem, stream)
#else
  #include <cuda_runtime.h>
  #define DEVFN __global__
  #define API_CHECK(x) do { auto _e=(x); if(_e!=cudaSuccess){fprintf(stderr,"CUDA error %d:%s at %s:%d\n",(int)_e, cudaGetErrorString(_e), __FILE__, __LINE__); std::abort();} } while(0)
  #define DEV_ALLOC cudaMalloc
  #define DEV_FREE  cudaFree
  #define MEMCPY_H2D(dst,src,bytes) API_CHECK(cudaMemcpy(dst,src,bytes,cudaMemcpyHostToDevice))
  #define MEMCPY_D2H(dst,src,bytes) API_CHECK(cudaMemcpy(dst,src,bytes,cudaMemcpyDeviceToHost))
  #define EVENT_CREATE(e) API_CHECK(cudaEventCreate(&e))
  #define EVENT_RECORD(e,s) API_CHECK(cudaEventRecord(e,s))
  #define EVENT_SYNC(e) API_CHECK(cudaEventSynchronize(e))
  #define EVENT_ELAPSED(ms,a,b) API_CHECK(cudaEventElapsedTime(&ms,a,b))
  #define EVENT_DESTROY(e) API_CHECK(cudaEventDestroy(e))
  #define STREAM cudaStream_t
  #define STREAM_CREATE(s) API_CHECK(cudaStreamCreate(&s))
  #define STREAM_DESTROY(s) API_CHECK(cudaStreamDestroy(s))
  #define LAUNCH(grid,block,shmem,stream,...) (__VA_ARGS__)<<<grid,block,shmem,stream>>>
#endif

static inline bool is_aligned_16(const void* p){ return (reinterpret_cast<uintptr_t>(p) & 0xF) == 0; }

DEVFN void copy_coalesced_scalar(const float* __restrict__ src, float* __restrict__ dst, size_t n){
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  for(size_t idx=i; idx<n; idx += (size_t)blockDim.x*gridDim.x){
    dst[idx] = src[idx];
  }
}

DEVFN void copy_coalesced_vec4(const float* __restrict__ src, float* __restrict__ dst, size_t n_vec4){
  const float4* __restrict__ s4 = reinterpret_cast<const float4*>(src);
  float4* __restrict__ d4 = reinterpret_cast<float4*>(dst);
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  for(size_t idx=i; idx<n_vec4; idx += (size_t)blockDim.x*gridDim.x){
    d4[idx] = s4[idx];
  }
}

DEVFN void copy_strided_read(const float* __restrict__ src, float* __restrict__ dst, size_t n_eff, int stride){
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  for(size_t idx=i; idx<n_eff; idx += (size_t)blockDim.x*gridDim.x){
    dst[idx] = src[(size_t)idx * (size_t)stride];
  }
}

DEVFN void count_nans(const float* __restrict__ x, size_t n, unsigned long long* __restrict__ out){
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long local = 0;
  for(size_t idx=i; idx<n; idx += (size_t)blockDim.x*gridDim.x){
    float v = x[idx];
    if(!(v==v)) local++;
  }
#if defined(__HIP_PLATFORM_AMD__)
  atomicAdd(out, local);
#else
  atomicAdd(out, local);
#endif
}

struct Args {
  size_t n = (1ull<<26);   // elements
  int stride = 1;          // src read stride
  int iters = 100;
  int offset_bytes = 0;    // misalignment test
  bool check_nan = false;
};

static void usage(const char* prog){
  printf("Usage: %s [--n N] [--stride S] [--iters K] [--offset-bytes B] [--check-nan]\n", prog);
}

int main(int argc, char** argv){
  Args a;
  for(int i=1;i<argc;i++){
    if(!strcmp(argv[i],"--n") && i+1<argc) a.n = strtoull(argv[++i],nullptr,10);
    else if(!strcmp(argv[i],"--stride") && i+1<argc) a.stride = atoi(argv[++i]);
    else if(!strcmp(argv[i],"--iters") && i+1<argc) a.iters = atoi(argv[++i]);
    else if(!strcmp(argv[i],"--offset-bytes") && i+1<argc) a.offset_bytes = atoi(argv[++i]);
    else if(!strcmp(argv[i],"--check-nan")) a.check_nan = true;
    else { usage(argv[0]); return 1; }
  }
  if(a.stride < 1) a.stride = 1;

  const size_t n_eff = (size_t)a.n / (size_t)a.stride;
  const size_t bytes_src = a.n * sizeof(float) + (size_t)a.offset_bytes + 16; // headroom for alignment tests
  const size_t bytes_dst = n_eff * sizeof(float) + (size_t)a.offset_bytes + 16;

  // Host init
  std::vector<float> h_src(a.n);
  for(size_t i=0;i<a.n;i++){ h_src[i] = (float)((i % 131) - 65) / 67.0f; }
  std::vector<float> h_dst(n_eff, 0.0f);

  // Device alloc (over-allocate for optional pointer offset)
  void* d_src_raw = nullptr; void* d_dst_raw = nullptr;
  API_CHECK(DEV_ALLOC(&d_src_raw, bytes_src));
  API_CHECK(DEV_ALLOC(&d_dst_raw, bytes_dst));

  // Apply offset for misalignment testing
  char* d_src_c = reinterpret_cast<char*>(d_src_raw);
  char* d_dst_c = reinterpret_cast<char*>(d_dst_raw);
  float* d_src = reinterpret_cast<float*>(d_src_c + a.offset_bytes);
  float* d_dst = reinterpret_cast<float*>(d_dst_c + a.offset_bytes);

  // H2D
  MEMCPY_H2D(d_src, h_src.data(), a.n * sizeof(float));

  // Launch config
  STREAM stream; STREAM_CREATE(stream);
  dim3 block(256);
  dim3 grid( std::min<size_t>( (n_eff + block.x - 1)/block.x, 120u*1024u ) );

  // Warmup + choose kernel
  bool can_vec4 = (a.stride==1) && is_aligned_16(d_src) && is_aligned_16(d_dst) && ((a.n % 4)==0);
  if(a.stride==1){
    if(can_vec4){
      LAUNCH(grid, block, 0, stream, copy_coalesced_vec4, d_src, d_dst, a.n/4);
    }else{
      LAUNCH(grid, block, 0, stream, copy_coalesced_scalar, d_src, d_dst, a.n);
    }
  }else{
    LAUNCH(grid, block, 0, stream, copy_strided_read, d_src, d_dst, n_eff, a.stride);
  }
#if defined(__HIP_PLATFORM_AMD__)
  API_CHECK(hipGetLastError());
#else
  API_CHECK(cudaGetLastError());
#endif

  // Time
#if defined(__HIP_PLATFORM_AMD__)
  hipEvent_t e0,e1;
#else
  cudaEvent_t e0,e1;
#endif
  EVENT_CREATE(e0); EVENT_CREATE(e1);
  EVENT_RECORD(e0, stream);
  for(int it=0; it<a.iters; ++it){
    if(a.stride==1){
      if(can_vec4){
        LAUNCH(grid, block, 0, stream, copy_coalesced_vec4, d_src, d_dst, a.n/4);
      }else{
        LAUNCH(grid, block, 0, stream, copy_coalesced_scalar, d_src, d_dst, a.n);
      }
    }else{
      LAUNCH(grid, block, 0, stream, copy_strided_read, d_src, d_dst, n_eff, a.stride);
    }
  }
  EVENT_RECORD(e1, stream);
  EVENT_SYNC(e1);
  float ms=0.0f; EVENT_ELAPSED(ms, e0, e1);
  EVENT_DESTROY(e0); EVENT_DESTROY(e1);

  // D2H + verify
  MEMCPY_D2H(h_dst.data(), d_dst, n_eff * sizeof(float));
  double max_abs_err = 0.0;
  for(size_t i=0;i<n_eff;i++){
    float ref = h_src[(size_t)i * (size_t)a.stride];
    max_abs_err = std::max(max_abs_err, (double)std::fabs(ref - h_dst[i]));
  }

  // Optional NaN scan
  unsigned long long* d_nan_ct = nullptr;
  unsigned long long h_nan_ct = 0;
  if(a.check_nan){
    API_CHECK(DEV_ALLOC((void**)&d_nan_ct, sizeof(unsigned long long)));
#if defined(__HIP_PLATFORM_AMD__)
    API_CHECK(hipMemset(d_nan_ct, 0, sizeof(unsigned long long)));
#else
    API_CHECK(cudaMemset(d_nan_ct, 0, sizeof(unsigned long long)));
#endif
    LAUNCH(grid, block, 0, stream, count_nans, d_dst, n_eff, d_nan_ct);
    MEMCPY_D2H(&h_nan_ct, d_nan_ct, sizeof(unsigned long long));
  }

  STREAM_DESTROY(stream);
  DEV_FREE(d_src_raw); DEV_FREE(d_dst_raw);
  if(d_nan_ct) DEV_FREE(d_nan_ct);

  // Report
  const double sec = ms / 1e3;
  const double bytes = (double)n_eff * 2.0 * sizeof(float) * (double)a.iters;
  const double GBps = bytes / sec / 1e9;
  printf("n=%zu stride=%d iters=%d offset_bytes=%d vec4=%s\n", a.n, a.stride, a.iters, a.offset_bytes, can_vec4?"yes":"no");
  printf("time=%.3f ms, achieved=%.2f GB/s, max_abs_err=%.3e\n", ms, GBps, max_abs_err);
  if(a.check_nan) printf("nan_count=%llu\n", (unsigned long long)h_nan_ct);
  return 0;
}
```

### Build

```bash
# CUDA
nvcc -O3 -std=c++17 -arch=${SM_ARCH} -lineinfo topics/18-troubleshooting-playbook/code/bandwidth_stride.cpp -o bw_diag

# ROCm
hipcc -O3 -std=c++17 --offload-arch=${GFX_ARCH} topics/18-troubleshooting-playbook/code/bandwidth_stride.cpp -o bw_diag
```

### Run (examples)

```bash
# Coalesced, aligned, vectorized (expect high GB/s)
./bw_diag --n 134217728 --iters 100

# Uncoalesced read (expect large drop)
./bw_diag --n 134217728 --stride 8 --iters 100

# Misalignment test (disables float4 path; expect drop)
./bw_diag --n 134217728 --offset-bytes 4 --iters 100

# Sanity scan for NaNs after a suspicious pipeline step
./bw_diag --n 67108864 --check-nan
```

### Numeric Example (checked)

Let `n=134,217,728`, `iters=100`, `stride=1`.
Elements moved per iter: `N_effective = n = 134,217,728`.
Bytes per iter: `2 * N * sizeof(float) = 2 * 134,217,728 * 4 = 1,073,741,824 B ≈ 1.074 GB`.
Over 100 iters: `≈ 107.374 GB`.
If elapsed time = `90 ms` → `0.09 s`, then `GBps ≈ 107.374 / 0.09 ≈ 1193.0 GB/s`.

## 7. Profiling and Validation

### Functional debugging

- Force sync to surface errors deterministically:
  `CUDA_LAUNCH_BLOCKING=1` or `HIP_LAUNCH_BLOCKING=1`.
- Memory correctness: `compute-sanitizer --tool memcheck ./your_app` (CUDA).
  ROCm: `rocgdb` for stepping; `rocprof --hsa-trace --hip-trace` for API tracing.

### Performance profiling

- **CUDA, timeline**: `nsys profile -t cuda,osrt,nvtx -o trace ./your_app`.
- **CUDA, kernels**: `ncu --set full --target-processes all ./your_app`. Focus on:

  - DRAM throughput (GB/s), L2 hit rate, Global Load/Store Efficiency (%),
  - Achieved Occupancy (%), Warp Stall Reasons, Tensor Core Utilization (if GEMM).

- **ROCm, API+kernel**: `rocprof --hip-trace --hsa-trace -o prof.csv ./your_app`.
  Stats mode: `rocprof --stats ./your_app`. Track:

  - DRAM BW, L2 hit rate, Waves per CU, VALU/MFMA issue, LDS bank conflicts.

### Pass thresholds (microbenchmark)

- Coalesced, aligned copy: ≥ 60–80% of theoretical peak DRAM BW for your SKU.
- Strided (e.g., 8): expect ≥ 5×–20× slower than coalesced → validates diagnosis.
- Misaligned (offset 4): ≥ 1.2×–2.5× slower than aligned vectorized path.

## 8. Performance Checklist

- Memory

  - [ ] No unintended stride or gather in hot loops (check GB/s via `bw_diag`).
  - [ ] Align pointers to 16B or 32B; use vector types (`float4`, `__half2`).
  - [ ] Pinned H2D/DtoH for request/response; avoid transfers on decode path.
  - [ ] Preallocate KV/cache/temps; avoid allocator thrash and fragmentation.

- Kernels

  - [ ] Prefer fused epilogues; keep AI high for prefill GEMMs.
  - [ ] Decode uses CUDA/HIP Graphs or persistent kernels; batch small ops.
  - [ ] Block sizes respect occupancy vs register pressure; avoid oversubscription stalls.

- Numerics

  - [ ] BF16/FP16 with FP32 accumulate for softmax, LayerNorm, rotary; epsilon set.
  - [ ] No implicit dtype up/downcasts on hot paths; deterministic flags set for A/B tests.

- Multi-GPU

  - [ ] NCCL/RCCL detects topology; env avoids PCIe-only when NVLink/XGMI is present.
  - [ ] Overlap compute/comm; stream priorities make sense.

- Serving

  - [ ] Batching policy and max context lengths match memory plan.
  - [ ] No synchronous CPU work in decode loop; I/O threads are pinned and non-blocking.

## 9. Troubleshooting (Symptom → Likely Cause → Fix)

| Symptom                              | Likely Cause                            | Fix                                                                                                 |
| ------------------------------------ | --------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Tokens/sec regressed after refactor  | Uncoalesced loads or lost vectorization | Run `bw_diag` with stride/offset; restore `float4` path; realign buffers.                           |
| Spiky decode latency                 | Kernel launch storm; pageable host I/O  | Enable CUDA/HIP Graphs or persistent kernels; pin host memory; coalesce RPCs.                       |
| OOM at high batch/context            | Fragmentation, KV cache growth          | Pool allocator, page KV, reduce max context, quantize KV (e.g., FP8/INT8) with accuracy checks.     |
| NaNs in logits                       | FP16 overflow in LN/softmax; bad scale  | Use BF16 or FP32 accumulators; clamp/epsilon; check RMSNorm constants.                              |
| Low SM/CU occupancy                  | Too many registers or tiny grids        | Tune block size, unroll pragmas; limit registers (`-maxrregcount` / launch bounds); adjust tiling.  |
| High L2 misses                       | Strided or random access                | Re-tile; swap layout to sequence-major for decode; prefetch; use shared/LDS staging.                |
| PCIe saturation                      | Host staging or cross-socket NUMA       | Pin memory; bind threads to NUMA domain; overlap with compute; reduce copy size on decode.          |
| NCCL/RCCL hangs                      | Rank/env mismatch; blocked PCIe peer    | Verify env; check visibility; fall back to ring/tree; pin device order; test with all_reduce bench. |
| Graph capture fails                  | Unsupported runtime API in capture      | Move allocations/I/O out of capture; guard streams; use capture-friendly allocators.                |
| Intermittent “illegal memory access” | Use-after-free or OOB                   | Memcheck; add guard allocations; enable launch blocking; binary search last good change.            |

## 10. Acceptance Criteria

- The provided `bw_diag` builds on both CUDA and ROCm with the listed flags.
- Coalesced aligned case achieves ≥ 60% of device peak DRAM bandwidth on a modern GPU.
- Strided and misaligned cases show clear, repeatable regressions relative to the coalesced baseline.
- Memcheck passes on the microbenchmark; optional NaN scan reports zero NaNs.
- The checklist and troubleshooting table lead to a concrete fix in a real inference service (e.g., restoring tokens/sec or stabilizing latency) when applied.

## 11. Further Work

- Add a GEMM + fused bias+activation microbenchmark to correlate tensor-core counters with end-to-end prefill throughput.
- Extend to KV-cache paging patterns and measure L2 hit rate vs layout variants.
- Provide decode-path persistent-kernel example with mailbox and back-to-back token generation.
- Integrate optional INT8/FP8 dequant staging to show bandwidth vs accuracy trade-offs.
- Automate a “first response” script: environment dump, driver/runtime versions, clocks, numa/pci topology, and a short counter capture.
