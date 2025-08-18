# Observability: Timelines & Counters

## 2. Summary

This module shows how to make GPU inference workloads observable with **timelines** (NVTX/roctx ranges) and **counters** (Nsight Systems/Compute and rocprof). You’ll learn which spans and metrics matter for prefill and decode, how to instrument kernels with negligible overhead, and how to turn traces into actionable performance deltas. A minimal CUDA/HIP program emits ranges for prefill, KV movement, and decode, measures TTFT and tokens/sec, and can be profiled on NVIDIA and AMD GPUs.

## 3. Why It Matters for LLM Inference

Prefill is often compute-bound and throughput-oriented; decode is latency-bound with small kernels and launch overhead sensitivity. Without timelines you can’t separate time in host vs. device or attribute stalls to phases (e.g., KV gathers). Without counters you can’t tell if you’re FLOP-limited, bandwidth-limited, or under-occupied. Proper spans and metrics make bottlenecks obvious and guide fixes (batching, fusion, graphs, persistent kernels).

## 4. Key Concepts and Formulas

- **TTFT (time-to-first-token)** ≈ `T_prefill + T_first_decode`.
- **Steady-state tokens/sec** ≈ `(N_decode_tokens − 1) / T_decode_after_first`.
- **Arithmetic intensity (AI)** = FLOPs / bytes.
  Example (toy kernel): if each element performs `R` FMAs (2 FLOPs) and loads/stores 8 bytes (read+write of 4-byte float), then
  `AI ≈ (2R) / 8 = R / 4  [FLOPs/byte]`.
  With `R=32`, `AI ≈ 8` → typically compute-leaning; with `R=2`, `AI ≈ 0.5` → memory-leaning.
- **Occupancy**: active warps per SM (NVIDIA) or wavefronts per CU (AMD) vs. maximum; low occupancy often implies register/shared-memory pressure or tiny launches.
- **Launch overhead** dominates small decode steps; mitigate with capture/graphs or persistent kernels.

## 5. GPU Deep Dive

**NVIDIA specifics**

- Warps of 32 threads schedule on SMs; tensor cores accelerate matrix math.
- Use NVTX to annotate host spans; Nsight Systems correlates CUDA API, kernels, and NVTX. Nsight Compute provides SpeedOfLight/Memory/Compute sections for utilization, bandwidth, occupancy, and cache behavior.

**AMD specifics**

- Wavefronts of 64 threads on CUs; MFMA/XDLOPs accelerate matrix math; LDS is the on-chip scratchpad.
- Use `roctx` for spans; `rocprof` gathers HIP/HSA traces and stats. Counters highlight waves, VALU/MFMA mix, cache behavior, and memory throughput.

## 6. Implementation

The following single-source implementation builds on both CUDA and HIP. It emits NVTX/roctx ranges for **PREFILL**, **KV_GATHER**, and **DECODE/STEP_X**, measures TTFT and tokens/sec, and uses GPU events for timing.

Place this file at:

```
topics/15-observability-timelines-and-counters/code/timeline_counters.cu
```

```cpp
// timeline_counters.cu
// Single-source CUDA/HIP + NVTX/roctx instrumentation for timelines & counters.
// Build (NVIDIA):
//   nvcc -O3 -std=c++17 -arch=sm_80 -lineinfo timeline_counters.cu -lnvToolsExt -o obs_cuda
// Build (AMD ROCm):
//   hipcc -O3 -std=c++17 --offload-arch=gfx942 timeline_counters.cu -lroctx64 -o obs_hip
//
// Run (defaults shown):
//   ./obs_cuda --batch 1 --ctx 1024 --hidden 4096 --decode 128 --prefill_rounds 32 --decode_rounds 6 --repeat 20
//
// Output:
//   TTFT_ms, steady_tokens_per_s, and kernel timings. Use Nsight/rocprof to see ranges.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

#if defined(__HIP_PLATFORM_AMD__)
  #include <hip/hip_runtime.h>
  #include <roctx.h>
  #define DEVFN __global__
  #define API_CHECK(x) do { auto _e = (x); if (_e != hipSuccess) { \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(_e), __FILE__, __LINE__); exit(1);} } while(0)
  #define MALLOC  hipMalloc
  #define FREE    hipFree
  #define MEMSET  hipMemset
  #define EVENT   hipEvent_t
  #define EVENT_CREATE(e) API_CHECK(hipEventCreate(&(e)))
  #define EVENT_REC(e,s) API_CHECK(hipEventRecord((e), (s)))
  #define EVENT_ELAPSE(a,b,ms) API_CHECK(hipEventElapsedTime(&(ms),(a),(b)))
  #define STREAM  hipStream_t
  #define STREAM_CREATE(s) API_CHECK(hipStreamCreate(&(s)))
  #define STREAM_SYNC(s) API_CHECK(hipStreamSynchronize((s)))
  #define DEVICE_SYNCH() API_CHECK(hipDeviceSynchronize())
  #define RANGE_PUSH(name) roctxRangePushA(name)
  #define RANGE_POP()      roctxRangePop()
  #define LAUNCH(kernel,grid,block,stream,...) do { \
      hipLaunchKernelGGL(kernel, grid, block, 0, stream, __VA_ARGS__); \
    } while(0)
#else
  #include <cuda_runtime.h>
  #include <nvToolsExt.h>
  #define DEVFN __global__
  #define API_CHECK(x) do { auto _e = (x); if (_e != cudaSuccess) { \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); exit(1);} } while(0)
  #define MALLOC  cudaMalloc
  #define FREE    cudaFree
  #define MEMSET  cudaMemset
  #define EVENT   cudaEvent_t
  #define EVENT_CREATE(e) API_CHECK(cudaEventCreate(&(e)))
  #define EVENT_REC(e,s) API_CHECK(cudaEventRecord((e), (s)))
  #define EVENT_ELAPSE(a,b,ms) API_CHECK(cudaEventElapsedTime(&(ms),(a),(b)))
  #define STREAM  cudaStream_t
  #define STREAM_CREATE(s) API_CHECK(cudaStreamCreate(&(s)))
  #define STREAM_SYNC(s) API_CHECK(cudaStreamSynchronize((s)))
  #define DEVICE_SYNCH() API_CHECK(cudaDeviceSynchronize())
  #define RANGE_PUSH(name) nvtxRangePushA(name)
  #define RANGE_POP()      nvtxRangePop()
  #define LAUNCH(kernel,grid,block,stream,...) do { \
      kernel<<<grid, block, 0, stream>>>(__VA_ARGS__); \
    } while(0)
#endif

struct Args {
  int batch = 1;
  int ctx   = 1024;   // prompt length
  int hidden= 4096;   // hidden width proxy
  int decode= 128;    // number of generated tokens
  int prefill_rounds = 32; // compute rounds in prefill (higher => more compute)
  int decode_rounds  = 6;  // compute rounds in decode (lower => more latency sensitivity)
  int repeat = 20;         // repetitions for stable timing
};

static inline int next_pow2(int x){ int p=1; while(p<x) p<<=1; return p; }

DEVFN void prefill_kernel(float* __restrict out,
                          const float* __restrict in,
                          int n_elems, int rounds)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = idx; i < n_elems; i += stride) {
    float v = in[i];
    // Compute-leaning: many FMAs per element to emulate GEMM epilogue work.
    // Deterministic math: fixed loop count.
    #pragma unroll 4
    for (int r = 0; r < rounds; ++r) {
      v = fmaf(v, 1.0009765625f, 0.000244140625f); // ~1+1/1024 and +1/4096
    }
    out[i] = v;
  }
}

DEVFN void kv_gather_kernel(float* __restrict dst,
                            const float* __restrict kv,
                            int n_elems, int stride)
{
  // Memory-leaning: strided loads to emulate KV gathers over time dimension.
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n_elems;
  for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
    int src = (i * stride) % total; // pseudo-gather pattern
    dst[i] = kv[src];
  }
}

DEVFN void decode_kernel(float* __restrict out,
                         const float* __restrict in,
                         int n_elems, int rounds)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = idx; i < n_elems; i += stride) {
    float v = in[i];
    #pragma unroll 1
    for (int r = 0; r < rounds; ++r) {
      v = fmaf(v, 0.99951171875f, 0.0001220703125f);
    }
    out[i] = v;
  }
}

static void parse(int argc, char** argv, Args& a){
  for (int i=1;i<argc;i++){
    auto eq = std::strchr(argv[i],'=');
    auto opt = std::string(argv[i]);
    auto val = [&](int def){ return eq? std::atoi(eq+1):def; };
    if (opt.rfind("--batch",0)==0) a.batch = val(a.batch);
    else if (opt.rfind("--ctx",0)==0) a.ctx = val(a.ctx);
    else if (opt.rfind("--hidden",0)==0) a.hidden = val(a.hidden);
    else if (opt.rfind("--decode",0)==0) a.decode = val(a.decode);
    else if (opt.rfind("--prefill_rounds",0)==0) a.prefill_rounds = val(a.prefill_rounds);
    else if (opt.rfind("--decode_rounds",0)==0) a.decode_rounds = val(a.decode_rounds);
    else if (opt.rfind("--repeat",0)==0) a.repeat = val(a.repeat);
  }
}

int main(int argc, char** argv){
  Args a; parse(argc, argv, a);
  STREAM stream; STREAM_CREATE(stream);

  const int n_tokens_prompt = a.batch * a.ctx;
  const int n_elems = n_tokens_prompt * a.hidden;
  const int n_elems_decode = a.batch * a.hidden;
  const size_t bytes_prompt = size_t(n_elems) * sizeof(float);
  const size_t bytes_decode = size_t(n_elems_decode) * sizeof(float);

  float *d_in=nullptr, *d_out=nullptr, *d_kv=nullptr, *d_tmp=nullptr;
  API_CHECK(MALLOC(&d_in,  bytes_prompt));
  API_CHECK(MALLOC(&d_out, bytes_prompt));
  API_CHECK(MALLOC(&d_kv,  bytes_prompt));
  API_CHECK(MALLOC(&d_tmp, bytes_prompt));
  MEMSET(d_in,  0, bytes_prompt);
  MEMSET(d_out, 0, bytes_prompt);
  MEMSET(d_kv,  0, bytes_prompt);
  MEMSET(d_tmp, 0, bytes_prompt);

  const int block = 256;
  const int grid_prefill = next_pow2((n_elems + block - 1) / block);
  const int grid_decode  = next_pow2((n_elems_decode + block - 1) / block);
  const int grid_kv      = next_pow2((n_elems + block - 1) / block);

  EVENT e_start, e_after_prefill, e_after_first, e_end, e_tmp;
  EVENT_CREATE(e_start);
  EVENT_CREATE(e_after_prefill);
  EVENT_CREATE(e_after_first);
  EVENT_CREATE(e_end);
  EVENT_CREATE(e_tmp);

  // Warmup
  RANGE_PUSH("WARMUP");
  LAUNCH(prefill_kernel, grid_prefill, block, stream, d_out, d_in, n_elems, 2);
  LAUNCH(kv_gather_kernel, grid_kv, block, stream, d_tmp, d_kv, n_elems, a.hidden);
  LAUNCH(decode_kernel, grid_decode, block, stream, d_out, d_out, n_elems_decode, 2);
  STREAM_SYNC(stream);
  RANGE_POP();

  float best_ttft_ms = 1e30f;
  float best_tokens_per_s = 0.f;

  for (int rep=0; rep<a.repeat; ++rep) {
    EVENT_REC(e_start, stream);

    RANGE_PUSH("PREFILL");
    LAUNCH(prefill_kernel, grid_prefill, block, stream, d_out, d_in, n_elems, a.prefill_rounds);
    RANGE_POP();
    EVENT_REC(e_after_prefill, stream);

    // First decode step
    RANGE_PUSH("DECODE/STEP_0");
    LAUNCH(kv_gather_kernel, grid_kv, block, stream, d_tmp, d_kv, n_elems, a.hidden);
    LAUNCH(decode_kernel, grid_decode, block, stream, d_out, d_out, n_elems_decode, a.decode_rounds);
    RANGE_POP();
    EVENT_REC(e_after_first, stream);

    // Remaining decode steps
    RANGE_PUSH("DECODE/STEADY");
    for (int t=1; t<a.decode; ++t) {
      char name[64];
      std::snprintf(name, sizeof(name), "DECODE/STEP_%d", t);
      RANGE_PUSH(name);
      LAUNCH(kv_gather_kernel, grid_kv, block, stream, d_tmp, d_kv, n_elems, a.hidden);
      LAUNCH(decode_kernel, grid_decode, block, stream, d_out, d_out, n_elems_decode, a.decode_rounds);
      RANGE_POP();
    }
    RANGE_POP();

    EVENT_REC(e_end, stream);
    STREAM_SYNC(stream);

    float t_prefill_ms=0.f, t_first_ms=0.f, t_total_ms=0.f;
    EVENT_ELAPSE(e_start, e_after_prefill, t_prefill_ms);
    EVENT_ELAPSE(e_after_prefill, e_after_first, t_first_ms);
    EVENT_ELAPSE(e_start, e_end, t_total_ms);

    const float ttft_ms = t_prefill_ms + t_first_ms;
    const float steady_decode_tokens = (float)std::max(0, a.decode - 1);
    const float t_steady_ms = t_total_ms - ttft_ms;
    const float tokens_per_s = (t_steady_ms > 0.0f) ? (1000.0f * steady_decode_tokens / t_steady_ms) : 0.0f;

    if (ttft_ms < best_ttft_ms) best_ttft_ms = ttft_ms;
    if (tokens_per_s > best_tokens_per_s) best_tokens_per_s = tokens_per_s;
  }

  printf("TTFT_ms=%.3f tokens_per_s=%.2f (best of %d reps)\n",
         best_ttft_ms, best_tokens_per_s, a.repeat);

  FREE(d_in); FREE(d_out); FREE(d_kv); FREE(d_tmp);
  return 0;
}
```

### Build Commands

NVIDIA:

```bash
cd topics/15-observability-timelines-and-counters/code
nvcc -O3 -std=c++17 -arch=sm_80 -lineinfo timeline_counters.cu -lnvToolsExt -o obs_cuda
```

AMD ROCm:

```bash
cd topics/15-observability-timelines-and-counters/code
hipcc -O3 -std=c++17 --offload-arch=gfx942 timeline_counters.cu -lroctx64 -o obs_hip
```

### Run (example)

```bash
# NVIDIA
./obs_cuda --batch=1 --ctx=1024 --hidden=4096 --decode=128 --prefill_rounds=32 --decode_rounds=6 --repeat=20
# AMD
./obs_hip  --batch=1 --ctx=1024 --hidden=4096 --decode=128 --prefill_rounds=32 --decode_rounds=6 --repeat=20
```

## 7. Profiling and Validation

### NVIDIA

1. **Nsight Systems** (timeline with NVTX):

```bash
nsys profile -t cuda,nvtx --force-overwrite true -o nsys_obs \
  ./obs_cuda --batch=1 --ctx=2048 --hidden=4096 --decode=256 --repeat=10
# Open the .qdrep in Nsight Systems GUI. You should see PREFILL, DECODE/STEP_X ranges and kernel/API bars aligned.
```

2. **Nsight Compute** (counters per kernel):

```bash
# Profile representative kernels; sections are stable across versions.
ncu --set full --section SpeedOfLight --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis \
    --target-processes all --launch-skip 0 --launch-count 1 \
    ./obs_cuda --batch=1 --ctx=2048 --hidden=4096 --decode=64 --repeat=1
```

Interpretation:

- **SpeedOfLight**: Achieved occupancy and SM/Tensor utilization. Prefill should show higher compute utilization than decode.
- **MemoryWorkloadAnalysis**: For `kv_gather_kernel`, expect higher DRAM throughput and lower cache hit rate vs. prefill/decode.
- **ComputeWorkloadAnalysis**: Instruction mix; prefill should have more FMA.

### AMD

1. **rocprof** (HIP/HSA trace + stats):

```bash
rocprof --hip-trace --hsa-trace --timestamp on --stats --output-dir rocprof_out \
  ./obs_hip --batch=1 --ctx=2048 --hidden=4096 --decode=256 --repeat=10
# Inspect CSVs for kernel durations, wavefront counts, memory BW indicators.
```

2. **Optional**: For deeper counters, enumerate available events first (varies per ASIC):

```bash
rocprof --list-basic > available_metrics.txt
# Then select a small set and re-run with: rocprof --events <EVENT1>,<EVENT2>,... <cmd>
```

Interpretation:

- Decode kernels should be short with visible per-step overhead.
- KV gather should report more memory stress vs. compute kernels.
- Wavefront occupancy should be lower for tiny decode kernels.

### Validations to perform

- NVTX/roctx ranges appear and nest correctly around kernels.
- TTFT and tokens/sec print consistent values across repetitions (≤10% variance).
- Prefill kernels are more compute-utilized than KV gather; KV gather shows higher memory pressure.

## 8. Performance Checklist

- Spans:

  - [ ] One **PREFILL** range encloses all prefill kernels.
  - [ ] Each decode step emits **DECODE/STEP_k** range; steady region groups them.
  - [ ] KV movement has a dedicated **KV_GATHER** label (or appears inside step range as in the sample).

- Metrics:

  - [ ] Prefill: compute utilization noticeably higher than memory BW utilization.
  - [ ] KV gather: DRAM BW is elevated; cache hit rate lower than prefill/decode.
  - [ ] Decode: kernel durations small; many launches; host API visible.

- Outcomes:

  - [ ] TTFT measured and reported.
  - [ ] Steady-state tokens/sec measured and reported.
  - [ ] Variance across runs ≤ 10% (same settings, warmed caches).

- Hygiene:

  - [ ] `-lineinfo` (CUDA) enabled for source correlation; `--stats` used on ROCm.
  - [ ] Build targets match your GPU arch (`-arch=sm_X` or `--offload-arch=gfxXYZ`).

## 9. Troubleshooting (symptom → likely cause → fix)

- **No NVTX/roctx ranges in timeline** → Missing link libs or ranges outside profiled region → Link `-lnvToolsExt` (CUDA) or `-lroctx64` (ROCm); ensure ranges bracket active work.
- **All kernels look serialized** → Host syncs or implicit sync via default stream → Use a dedicated stream; avoid `DeviceSynchronize()` inside loops; check API trace for sync points.
- **Decode throughput extremely low** → Launch overhead or tiny grids → Increase `decode_rounds`, fuse steps, adopt CUDA/HIP Graphs or persistent kernels in real pipeline.
- **Prefill shows memory-bound** → Rounds too low or poor tiling → Increase compute rounds in this demo; in real code ensure tensor-core paths and better blocking.
- **Counters missing in Nsight/rocprof** → Unsupported metric set for device → Use stable sections (`SpeedOfLight`, `MemoryWorkloadAnalysis`) on Nsight; on ROCm list metrics and pick supported ones.
- **High variance across runs** → DVFS/clock drift or cold caches → Fix power mode, run warmups, increase `--repeat`, pin clocks if permissible.

## 10. Acceptance Criteria

- The code builds and runs on both CUDA 12.x and ROCm 6.x with the provided commands.
- Running the binary prints `TTFT_ms` and `tokens_per_s`.
- Nsight Systems (CUDA) and rocprof (ROCm) show timeline ranges for PREFILL and per-step DECODE.
- Nsight Compute shows that:

  - Prefill kernels have higher compute utilization than KV gather.
  - KV gather shows higher DRAM activity than prefill/decode.

- Repeated runs with identical arguments keep variance ≤ 10%.

## 11. Further Work

- Add **CUDA/HIP Graphs** capture for decode loop to quantify launch-overhead reduction.
- Convert decode to a **persistent kernel** to measure per-token latency improvements.
- Instrument **H2D/D2H** memory spans and CPU post-processing (tokenization, sampling) with NVTX/roctx.
- Integrate **hardware counters to roofline** plotting (simple Python harness).
- Record **per-layer spans** (LN, QKV GEMM, attention, MLP) in a real model to triage hotspots.
