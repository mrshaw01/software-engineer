## Mixture-of-Experts (MoE) Inference

### Summary

Mixture-of-Experts (MoE) layers replace a dense MLP with many sparsely-activated expert MLPs. A learned router selects top-k experts per token, which reduces compute per token while increasing model capacity. This note explains MoE routing, capacity management, communication patterns (all-to-all), and GPU execution on NVIDIA (CUDA) and AMD (ROCm/HIP). A minimal, runnable kernel demonstrates top-1 routing and per-expert MLP on GPU, with guidance for profiling and validation.

### Why It Matters for LLM Inference

In prefill, MoE’s router and dispatch are amortized over long sequences; MLP compute dominates. In decode, router+dispatch latency and potential all-to-all synchronization become critical, often limiting tokens/sec. Correct batching, capacity control, and overlap of communication with expert compute are essential for stable low-latency serving.

## Key Concepts and Formulas

Let:

- $T$: tokens in a batch step,
- $H$: hidden size,
- $r$: expansion ratio for expert FFN ($H_\text{ffn}=rH$; commonly 4),
- $E$: number of experts,
- $k$: router top-k (1 or 2),
- $b$: bytes per activation element (2 for BF16/FP16, 4 for FP32),
- $f_\text{remote}$: fraction of routed tokens that leave the local device in expert parallel.

### Expert Compute (per token, per selected expert)

Two GEMMs (ignoring bias/activation fusion):

$$
\text{FLOPs}_\text{MLP} \approx 2 H H_\text{ffn} + 2 H_\text{ffn} H \approx 4 H H_\text{ffn} = 4 r H^2.
$$

For top-k routing, multiply by $k$. Example: $H=4096, r=4, k=2 \Rightarrow 4\times 4\times 4096^2 \times 2 \approx 536$ MFLOPs/token.

### Router Cost

A token-wise projection to $E$ logits: $\mathcal{O}(E H)$ per token. With $E$ up to 64–128, router cost is smaller than MLP but matters in decode.

### Capacity and Dropping

Let capacity factor $C\ge 1$. Per-expert capacity:

$$
\text{cap} = \left\lceil C \cdot \frac{kT}{E} \right\rceil.
$$

Tokens exceeding capacity may be dropped, rerouted, or padded; inference should choose $C$ so that drops are rare at target workloads.

### Communication (Expert Parallel, per step)

If tokens (activations) must move to devices hosting their experts, all-to-all volume:

$$
\text{Bytes}_\text{A2A} \approx k \cdot T \cdot H \cdot b \cdot f_\text{remote}.
$$

Example: $T=1024,H=4096,k=2,b=2,f_\text{remote}=0.75$ ⇒ $\approx 2\times1024\times4096\times2\times0.75 \approx 12.6$ MB per direction per MoE layer.

Arithmetic intensity (compute/comm) rough check (per token):

$$
\frac{\text{FLOPs}}{\text{Bytes}} \approx \frac{4 r H^2 k}{kHb}=\frac{4 r H}{b}.
$$

For $H=4096,r=4,b=2 \Rightarrow \approx 32768$ FLOPs/byte—MLP compute is ample; the risk is latency and synchronization in decode, not bandwidth saturation (unless many experts or small H).

## GPU Deep Dive

### NVIDIA

- **Warps/SMs**: 32-thread warps on SMs; keep per-token loops coalesced over hidden dimension to drive memory throughput.
- **Tensor Cores**: Use BF16/FP16 with FP32 accumulate for MLP GEMMs; align to 8/16 element multiples.
- **Memory**: Stage token activations in L2; group tokens by expert to improve temporal locality of expert weights.
- **Comm**: Use NCCL all-to-all; overlap via separate streams and `cudaEvent` synchronization; consider CUDA Graphs for decode to contain launch overhead.

### AMD

- **Wavefronts/CUs**: 64-lane wavefronts on Compute Units; prefer LDS tiling that matches MFMA tile sizes.
- **MFMA/XDLOPs**: Drive BF16/FP16 MFMA paths; ensure alignment and vectorized loads (`float4`, `__half2`).
- **Memory (LDS)**: Pack expert-grouped tokens to increase reuse; use `hipExtStreamCreateWithPriority` to separate A2A and compute streams.
- **Comm**: Use RCCL all-to-all with `rocTX`/`rocprof` timeline correlation; overlap with expert compute.

## Implementation

This minimal kernel demonstrates:

1. top-1 routing on GPU,
2. per-expert MLP (two matvecs) on GPU,
3. optional CPU reference check.

It is single-source and compiles with **nvcc** or **hipcc**.

### Build

CUDA:

```bash
nvcc -O3 -std=c++17 -arch=${SM_ARCH:-sm_80} -lineinfo topics/14-moe-inference/code/moe_top1_unified.cu -o moe_top1
```

ROCm/HIP:

```bash
hipcc -O3 -std=c++17 --offload-arch=${GFX_ARCH:-gfx942} topics/14-moe-inference/code/moe_top1_unified.cu -o moe_top1_hip
```

Run (defaults shown; keep sizes modest for CPU check):

```bash
./moe_top1 --tokens=1024 --hidden=256 --ffn=512 --experts=8 --check=1
# or
./moe_top1_hip --tokens=1024 --hidden=256 --ffn=512 --experts=8 --check=1
```

### File: `topics/14-moe-inference/code/moe_top1_unified.cu`

```cpp
// Single-source CUDA/HIP demo: top-1 MoE routing + MLP (BF16/FP16-ready via float baseline)
#if defined(__HIP_PLATFORM_AMD__)
  #include <hip/hip_runtime.h>
  #define DEVAPI __global__
  #define API_CHECK(x) do { auto e=(x); if(e!=hipSuccess){fprintf(stderr,"HIP error %d at %s:%d\n",e,__FILE__,__LINE__); abort();} } while(0)
  #define DEV_MALLOC hipMalloc
  #define DEV_FREE hipFree
  #define MEMCPY_HTOD(dst,src,sz) hipMemcpy(dst,src,sz,hipMemcpyHostToDevice)
  #define MEMCPY_DTOH(dst,src,sz) hipMemcpy(dst,src,sz,hipMemcpyDeviceToHost)
  #define MEMCPY_D2D(dst,src,sz)  hipMemcpy(dst,src,sz,hipMemcpyDeviceToDevice)
  #define DEVICE_SYNC() hipDeviceSynchronize()
  #define EVENT_T hipEvent_t
  #define EVENT_CREATE(e) hipEventCreate(&(e))
  #define EVENT_RECORD(e,s) hipEventRecord((e), (s))
  #define EVENT_ELAPSED(ms, start, stop) hipEventElapsedTime(&(ms), (start), (stop))
  #define STREAM_T hipStream_t
  #define STREAM_CREATE(s) hipStreamCreate(&(s))
  #define STREAM_DESTROY(s) hipStreamDestroy((s))
#else
  #include <cuda_runtime.h>
  #define DEVAPI __global__
  #define API_CHECK(x) do { auto e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA error %d (%s) at %s:%d\n",e,cudaGetErrorString((cudaError_t)e),__FILE__,__LINE__); abort();} } while(0)
  #define DEV_MALLOC cudaMalloc
  #define DEV_FREE cudaFree
  #define MEMCPY_HTOD(dst,src,sz) cudaMemcpy(dst,src,sz,cudaMemcpyHostToDevice)
  #define MEMCPY_DTOH(dst,src,sz) cudaMemcpy(dst,src,sz,cudaMemcpyDeviceToHost)
  #define MEMCPY_D2D(dst,src,sz)  cudaMemcpy(dst,src,sz,cudaMemcpyDeviceToDevice)
  #define DEVICE_SYNC() cudaDeviceSynchronize()
  #define EVENT_T cudaEvent_t
  #define EVENT_CREATE(e) cudaEventCreate(&(e))
  #define EVENT_RECORD(e,s) cudaEventRecord((e), (s))
  #define EVENT_ELAPSED(ms, start, stop) cudaEventElapsedTime(&(ms), (start), (stop))
  #define STREAM_T cudaStream_t
  #define STREAM_CREATE(s) cudaStreamCreate(&(s))
  #define STREAM_DESTROY(s) cudaStreamDestroy((s))
#endif

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>
#include <cstring>
#include <string>

static inline float gelu(float x) {
  // Tanh approximation
  const float kAlpha = 0.7978845608028654f; // sqrt(2/pi)
  const float kBeta = 0.044715f;
  float y = x*(1.0f + tanhf(kAlpha*(x + kBeta*x*x*x)));
  return 0.5f*y;
}

DEVAPI
void router_top1(const float* __restrict__ x,      // [T,H]
                 const float* __restrict__ wg,     // [E,H]
                 int* __restrict__ expert_of_tok,  // [T]
                 int T, int H, int E)
{
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  const float* xt = x + t * H;

  float best = -1e30f;
  int best_e = 0;
  // Dot with each expert router row
  for (int e = 0; e < E; ++e) {
    const float* we = wg + e * H;
    float acc = 0.f;
    // Unroll by 4
    int h=0;
    for (; h+3 < H; h+=4) {
      acc += xt[h+0]*we[h+0] + xt[h+1]*we[h+1] + xt[h+2]*we[h+2] + xt[h+3]*we[h+3];
    }
    for (; h < H; ++h) acc += xt[h]*we[h];
    if (acc > best) { best = acc; best_e = e; }
  }
  expert_of_tok[t] = best_e;
}

DEVAPI
void moe_mlp_top1(const float* __restrict__ x,     // [T,H]
                  const float* __restrict__ w1,    // [E,Hff,H] row-major: k,j
                  const float* __restrict__ w2,    // [E,H,Hff] row-major: i,k
                  const int*  __restrict__ expert_of_tok,
                  float* __restrict__ tmp_ffn,     // [T,Hff]
                  float* __restrict__ y,           // [T,H]
                  int T, int H, int Hff)
{
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  const float* xt = x + t * H;
  int e = expert_of_tok[t];

  const float* w1_e = w1 + (size_t)e * Hff * H;
  const float* w2_e = w2 + (size_t)e * H * Hff;
  float* tmp = tmp_ffn + (size_t)t * Hff;
  // tmp = GeLU(w1_e * x)
  for (int k = 0; k < Hff; ++k) {
    const float* w1_row = w1_e + (size_t)k * H;
    float acc = 0.f;
    int j=0;
    for (; j+3 < H; j+=4) {
      acc += w1_row[j+0]*xt[j+0] + w1_row[j+1]*xt[j+1] + w1_row[j+2]*xt[j+2] + w1_row[j+3]*xt[j+3];
    }
    for (; j < H; ++j) acc += w1_row[j]*xt[j];
    tmp[k] = gelu(acc);
  }
  // y = w2_e * tmp
  float* yt = y + (size_t)t * H;
  for (int i = 0; i < H; ++i) {
    const float* w2_row = w2_e + (size_t)i * Hff;
    float acc = 0.f;
    int k=0;
    for (; k+3 < Hff; k+=4) {
      acc += w2_row[k+0]*tmp[k+0] + w2_row[k+1]*tmp[k+1] + w2_row[k+2]*tmp[k+2] + w2_row[k+3]*tmp[k+3];
    }
    for (; k < Hff; ++k) acc += w2_row[k]*tmp[k];
    yt[i] = acc;
  }
}

struct Args {
  int T = 1024, H = 256, Hff = 512, E = 8;
  int check = 1;
};

static Args parse_args(int argc, char** argv) {
  Args a;
  for (int i=1;i<argc;++i) {
    std::string s(argv[i]);
    auto pick = [&](const char* k, int& dst){
      auto p = s.find(k);
      if (p==0) dst = std::atoi(s.c_str()+strlen(k));
    };
    pick("--tokens=", a.T);
    pick("--hidden=", a.H);
    pick("--ffn=",    a.Hff);
    pick("--experts=",a.E);
    pick("--check=",  a.check);
  }
  return a;
}

static void cpu_reference(const std::vector<float>& x, const std::vector<float>& wg,
                          const std::vector<float>& w1, const std::vector<float>& w2,
                          std::vector<int>& expert_of_tok, std::vector<float>& y,
                          int T,int H,int Hff,int E)
{
  // Router
  for (int t=0;t<T;++t){
    const float* xt = &x[t*H];
    float best=-1e30f; int best_e=0;
    for (int e=0;e<E;++e){
      const float* we = &wg[e*H];
      float acc=0.f;
      for (int j=0;j<H;++j) acc += xt[j]*we[j];
      if (acc>best){ best=acc; best_e=e; }
    }
    expert_of_tok[t] = best_e;
  }
  // MLP
  std::vector<float> tmp(T*Hff);
  for(int t=0;t<T;++t){
    int e = expert_of_tok[t];
    const float* xt=&x[t*H];
    const float* w1e=&w1[(size_t)e*Hff*H];
    const float* w2e=&w2[(size_t)e*H*Hff];
    float* tmpt=&tmp[t*Hff];
    for(int k=0;k<Hff;++k){
      const float* row=&w1e[(size_t)k*H];
      float acc=0.f; for(int j=0;j<H;++j) acc+=row[j]*xt[j];
      tmpt[k]=gelu(acc);
    }
    float* yt=&y[t*H];
    for(int i=0;i<H;++i){
      const float* row=&w2e[(size_t)i*Hff];
      float acc=0.f; for(int k=0;k<Hff;++k) acc+=row[k]*tmpt[k];
      yt[i]=acc;
    }
  }
}

int main(int argc, char** argv){
  Args a = parse_args(argc, argv);
  printf("MoE demo: T=%d H=%d Hff=%d E=%d check=%d\n", a.T, a.H, a.Hff, a.E, a.check);

  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-0.5f,0.5f);

  size_t x_sz   = (size_t)a.T * a.H;
  size_t wg_sz  = (size_t)a.E * a.H;
  size_t w1_sz  = (size_t)a.E * a.Hff * a.H;
  size_t w2_sz  = (size_t)a.E * a.H * a.Hff;
  size_t y_sz   = (size_t)a.T * a.H;
  size_t tmp_sz = (size_t)a.T * a.Hff;

  std::vector<float> hx(x_sz), hwg(wg_sz), hw1(w1_sz), hw2(w2_sz);
  for(auto& v: hx)  v = dist(rng);
  for(auto& v: hwg) v = dist(rng);
  for(auto& v: hw1) v = dist(rng);
  for(auto& v: hw2) v = dist(rng);

  float *dx=nullptr, *dwg=nullptr, *dw1=nullptr, *dw2=nullptr, *dy=nullptr, *dtmp=nullptr;
  int *dexp=nullptr;
  API_CHECK( DEV_MALLOC(&dx,   x_sz  * sizeof(float)) );
  API_CHECK( DEV_MALLOC(&dwg,  wg_sz * sizeof(float)) );
  API_CHECK( DEV_MALLOC(&dw1,  w1_sz * sizeof(float)) );
  API_CHECK( DEV_MALLOC(&dw2,  w2_sz * sizeof(float)) );
  API_CHECK( DEV_MALLOC(&dy,   y_sz  * sizeof(float)) );
  API_CHECK( DEV_MALLOC(&dtmp, tmp_sz* sizeof(float)) );
  API_CHECK( DEV_MALLOC(&dexp, a.T   * sizeof(int)) );

  MEMCPY_HTOD(dx,  hx.data(),  x_sz  * sizeof(float));
  MEMCPY_HTOD(dwg, hwg.data(), wg_sz * sizeof(float));
  MEMCPY_HTOD(dw1, hw1.data(), w1_sz * sizeof(float));
  MEMCPY_HTOD(dw2, hw2.data(), w2_sz * sizeof(float));

  STREAM_T stream; STREAM_CREATE(stream);
  EVENT_T e0,e1,e2; EVENT_CREATE(e0); EVENT_CREATE(e1); EVENT_CREATE(e2);

  int block = 128;
  int grid  = (a.T + block - 1)/block;

  EVENT_RECORD(e0, stream);
  router_top1<<<grid, block, 0, stream>>>(dx, dwg, dexp, a.T, a.H, a.E);
  EVENT_RECORD(e1, stream);
  moe_mlp_top1<<<grid, block, 0, stream>>>(dx, dw1, dw2, dexp, dtmp, dy, a.T, a.H, a.Hff);
  EVENT_RECORD(e2, stream);

  DEVICE_SYNC();
  float ms_router=0.f, ms_mlp=0.f, ms_tot=0.f;
  EVENT_ELAPSED(ms_router, e0, e1);
  EVENT_ELAPSED(ms_mlp,    e1, e2);
  EVENT_ELAPSED(ms_tot,    e0, e2);

  printf("Timing (ms): router=%.3f, mlp=%.3f, total=%.3f\n", ms_router, ms_mlp, ms_tot);
  double toks_per_sec = (double)a.T * 1000.0 / ms_tot;
  printf("Throughput: %.2f tokens/sec (single MoE layer)\n", toks_per_sec);

  std::vector<float> y(y_sz);
  MEMCPY_DTOH(y.data(), dy, y_sz*sizeof(float));

  if (a.check){
    std::vector<int> exp_ref(a.T);
    std::vector<float> y_ref(y_sz, 0.f);
    cpu_reference(hx, hwg, hw1, hw2, exp_ref, y_ref, a.T, a.H, a.Hff, a.E);
    // Compare
    double max_abs=0.0, mean_abs=0.0;
    for (size_t i=0;i<y_sz;++i){
      double d = std::abs((double)y_ref[i] - (double)y[i]);
      max_abs = std::max(max_abs, d);
      mean_abs += d;
    }
    mean_abs /= (double)y_sz;
    printf("CPU check: mean_abs=%.3e, max_abs=%.3e\n", mean_abs, max_abs);
    // basic acceptance
    if (!(mean_abs < 5e-5 && max_abs < 5e-3)) {
      fprintf(stderr, "Validation failed.\n");
      return 2;
    }
  }

  DEV_FREE(dx); DEV_FREE(dwg); DEV_FREE(dw1); DEV_FREE(dw2);
  DEV_FREE(dy); DEV_FREE(dtmp); DEV_FREE(dexp);
  STREAM_DESTROY(stream);
  return 0;
}
```

Notes:

- The kernel is intentionally simple and correctness-oriented. It highlights how per-token expert selection dictates expert-specific weight access, motivating grouping by expert and GEMM libraries (CUTLASS/cublasLt / rocBLAS/hipBLASLt) in production.
- For reproducibility, all random initializations use a fixed seed (123).

## Profiling and Validation

### NVIDIA

Prefill/Decode capture:

```bash
nsys profile -t cuda,osrt --stats=true ./moe_top1 --tokens=1024 --hidden=256 --ffn=512 --experts=8
```

Kernel metrics:

```bash
ncu --target-processes all \
    --metrics sm__sass_thread_inst_executed_op_fma_pred_on.sum,smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active, \
             lts__t_sectors_srcunit_tex_op_read.sum,dram__throughput.avg.pct_of_peak_sustained_active,launch__occupancy_limit_active_warps_pct \
    ./moe_top1 --tokens=1024 --hidden=256 --ffn=512 --experts=8
```

Interpretation:

- `*tensor_cycles_active*` (or FP32 FMAs here) indicates compute utilization (target: >40% for MLP kernel in this scalar demo; much higher with GEMM paths).
- `dram__throughput.*` and `lts__*read*` show memory pressure. Grouping tokens by expert should reduce L2 misses.
- `launch__occupancy_*` helps identify register/spill limits.

### AMD

Timeline and counters:

```bash
rocprof --hip-trace --hsa-trace --timestamp on ./moe_top1_hip --tokens=1024 --hidden=256 --ffn=512 --experts=8
rocprof --stats ./moe_top1_hip --tokens=1024 --hidden=256 --ffn=512 --experts=8
```

Look for:

- Kernel duration split (router vs MLP).
- Wavefront occupancy and MFMA usage (with GEMM fused paths).
- LDS bank conflicts (should be minimal in this scalar demo; revisit when tiling).

### Validation

The program compares GPU output with a CPU reference and prints mean/max absolute error. Acceptance thresholds (float path) are set at `mean_abs < 5e-5` and `max_abs < 5e-3`. For FP16/BF16 variants, relax thresholds (documented per datatype).

## Performance Checklist

- Routing

  - [ ] Router kernel < 20% of layer time in decode at target batch size.
  - [ ] Top-k implemented with vectorized loads over $H$; logits reduced in registers.

- Dispatch & Layout

  - [ ] Tokens grouped by expert prior to MLP to improve expert-weight locality.
  - [ ] Capacity factor $C$ chosen so drops < 0.5% at P95 workload.

- Compute

  - [ ] Expert MLP uses GEMM (cublasLt/hipBLASLt) with BF16/FP16 input and FP32 accumulate.
  - [ ] Fused GeLU/bias epilogue enabled.

- Communication (if expert parallel)

  - [ ] NCCL/RCCL all-to-all overlapped with expert compute on separate streams.
  - [ ] Bucket sizes tuned to keep >80% link utilization without fragmenting compute.

- Launch Overhead

  - [ ] Decode uses CUDA/HIP Graphs or persistent kernels to reduce per-step launches.

- Memory

  - [ ] Expert weights aligned (128B) and allocated from a pool; activations coalesced.
  - [ ] L2 residency for hot experts observed (L2 hit rate up after grouping).

## Troubleshooting (Symptom → Likely Cause → Fix)

- Low tokens/sec in decode → Kernel launch overhead → Use CUDA/HIP Graphs or a persistent decode kernel; fuse router+pack.
- Router dominates time → E or H too large for naive dot → Use GEMV/GEMM for router, vectorize loads, pre-normalize.
- Expert imbalance (hot experts) → Inadequate capacity factor → Increase $C$, enable second-choice fallback, or apply token-dropping with logging.
- Poor L2 hit rate in MLP → Tokens not grouped by expert → Pack tokens by expert and run per-expert GEMM batches.
- A2A stalls → Single stream serialization → Separate comm/compute streams, insert events, tune bucket sizes.
- DRAM bandwidth capped → Misaligned loads / scalar code → Use vector types (`float4`, `__half2`) and aligned allocations.
- Numerical instability in GeLU → FP16 underflow → Use BF16 or FP32 accumulate; scale inputs; clamp in epilogue.
- ROCm build slow or fails → GFX arch mismatch → Set `--offload-arch=gfx942` (MI300A/X) or appropriate target.
- Validation drift vs CPU → Different activation approximation → Ensure both use the same GeLU formula; match accumulation precision.

## Acceptance Criteria

- This document explains routing, capacity, and comm/compute trade-offs with formulas and numeric examples.
- The provided **single-source CUDA/HIP** program compiles and runs in seconds on a modern GPU and validates against a CPU reference.
- Profiling commands for Nsight and rocprof are included with 2–3 critical counters and interpretation.
- A practical checklist and troubleshooting table enable engineers to verify and remediate performance issues.

## Further Work

- Implement top-2 routing with capacity and combine weights; compare accuracy vs latency.
- Replace scalar MLP with library GEMMs (cublasLt/hipBLASLt) and a fused bias+GeLU epilogue.
- Add optional NCCL/RCCL all-to-all to demonstrate overlap with compute (two streams + events).
- Introduce expert batching and on-the-fly token packing using CUDA/HIP cooperative groups.
- Evaluate FP8 quantized expert weights with on-the-fly dequantization fused into GEMM epilogues.

## Cross-Topic Links

- 05: Prefill vs. Decode Performance Split
- 08: GEMM Paths & Fused Epilogues
- 13: Multi-GPU Topology & Parallelism
- 16: Throughput, Latency & Batching Policy
- 21: End-to-End Optimization Plan
