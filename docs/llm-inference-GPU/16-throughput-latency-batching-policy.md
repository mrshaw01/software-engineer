# Throughput, Latency & Batching Policy

## Summary

This module explains how batching policies shape user‑perceived latency and system throughput in LLM inference. We formulate the core trade‑offs (TTFT vs tokens/sec), derive queueing‑theoretic bounds, and show how prefill and decode have different optimal policies. A minimal CUDA/HIP benchmark demonstrates how batch size and scheduling window affect measured performance. The module ends with a concrete, SLO‑driven batching policy you can adopt and verify.

## Why It Matters for LLM Inference

Prefill is compute‑heavy and benefits from large, homogeneous batches; decode is step‑serial and sensitive to TTFT, so over‑batching increases latency. Serving systems must dynamically balance admission windows, max batch sizes, and token interleaving to hit both throughput and latency SLOs. Correct policy separates interactive (decode‑dominated) from bulk/offline (prefill‑dominated) traffic, and continuously adapts to arrival rate.

## Key Concepts and Formulas

1. **Definitions**

   - **TTFT** (Time‑to‑First‑Token): wall time from request arrival to first generated token emission.
   - **E2E Latency**: time from arrival to completion (first token + remaining tokens).
   - **Throughput**: tokens/sec or requests/sec at steady state.
   - **Batching Window (Δ)**: time we wait to collect requests before launching a kernel/step.
   - **Max Batch (Bmax)**: upper bound on concurrent sequences per kernel.

2. **Little’s Law** (stable system): $L = \lambda W$.

   - With arrival rate $\lambda$ (req/s) and average time in system $W$, the average number in system $L$ is fixed.

3. **TTFT Decomposition**

   $$
   \mathrm{TTFT} \approx \underbrace{\mathrm{admission\ wait}}_{\le \Delta} + \underbrace{\mathrm{queue\ wait}}_{\text{prior steps}} + \underbrace{t_{\text{step}}(b)}_{\text{1st decode step}}
   $$

   For fixed‑window admission, expected admission wait is $\mathbb{E}[\text{wait}] \approx \Delta/2$ (uniform arrival assumption).

4. **Decode Step Time Model**
   Empirically, per‑step time grows sublinearly with batch size $b$:

   $$
   t_{\text{step}}(b) \approx t_0 + c\,b^{\alpha},\quad 0<\alpha\le 1
   $$

   where $t_0$ is launch/overhead and $c$ reflects arithmetic + memory costs. Your hardware/kernel determines $\alpha$.

5. **Prefill vs Decode**

   - **Prefill** FLOPs per token $\propto L$ (context length). High arithmetic intensity → bigger batches and sequence‑length bucketing maximize FLOP/s.
   - **Decode** FLOPs per token are roughly constant; serial dependence penalizes long windows. Optimal batches are modest; admission windows should be short for interactive SLOs.

6. **SLO‑Aware Window**
   If your p95 TTFT SLO is $S$ ms and measured $t_{\text{step}}(b)$ is known, choose

   $$
   \Delta \le S - t_{\text{step}}(b) - \text{queue\ safety\ margin}
   $$

7. **Tokens/Sec** (measured)

   $$
   \mathrm{throughput} = \frac{\sum_{i} T_i}{T_{\text{gpu}}}
   $$

   where $T_i$ tokens are generated for request $i$ and $T_{\text{gpu}}$ is total on‑GPU time. For user‑perceived throughput, include admission time in the denominator.

8. **Numeric Example** (instantiate)

   - Arrival rate $\lambda=80$ req/s, tokens per request $T=32$, $B_{\max}=16$, $\Delta=6$ ms.
   - Measured $t_{\text{step}}(b{=}16)=1.9$ ms, $t_{\text{step}}(b{=}4)=0.9$ ms.
   - Expected admission wait $\approx 3$ ms; TTFT at $b{=}16$ $\approx 3 + 1.9 = 4.9$ ms (plus any queueing). Reducing $\Delta$ to 2 ms lowers TTFT by \~1 ms but may reduce average batch to \~8 under the same $\lambda$.

## GPU Deep Dive

### NVIDIA

- **Warps/SMs**: 32‑thread warps scheduled across SMs; high throughput comes from sufficient thread‑level parallelism (TLP) and occupancy. Decode kernels are typically small; persistent kernels or CUDA Graphs can amortize launch overhead.
- **Tensor Cores**: Exploit for prefill GEMMs and fused epilogues; in decode, smaller matrices limit Tensor Core efficiency—favor fusion and cache reuse.
- **Memory Hierarchy**: KV cache access is memory‑bound; maximize L2 reuse, coalesce loads, and align to 32‑byte sectors.

### AMD

- **Wavefronts/CUs**: 64‑lane wavefronts on Compute Units; ensure workgroup sizes align with 64‑lane multiples. Decode benefits from persistent‑wavefront designs to reduce dispatch overhead.
- **MFMA/XDLOPs**: Use for prefill GEMMs. For decode, use LDS tiling and vectorized loads (e.g., `float4`, `__half2`) to sustain bandwidth.
- **L2/LDS**: Keep hot state in LDS when feasible; page‑friendly KV layouts reduce VRAM pressure and TLB misses.

## Implementation

We provide a minimal decode microbenchmark that:

1. simulates admission with a fixed window (Δ) and $B_{\max}$,
2. launches one GPU kernel per decode step for the active batch,
3. measures tokens/sec, TTFT distribution, and E2E latency.

Two compilable backends are provided.

### File: `topics/16-throughput-latency-batching-policy/code/decode_batch_bench.cu`

```cpp
// Minimal CUDA decode-step batching benchmark
// Build: nvcc -O3 -std=c++17 -arch=${SM_ARCH} -lineinfo decode_batch_bench.cu -o decode_batch_bench
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cassert>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) do { cudaError_t e=(x); if(e!=cudaSuccess){\
  fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)
#endif

struct Req { double arrival; int remaining; double first_token=-1.0; double done=-1.0; bool started=false; };

__global__ void decode_step_kernel(float* out, int work_iters) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    float acc = 0.f;
    // Each block simulates one sequence (one token step)
    // Work scales with work_iters; loop is unrolled-less to keep portable
    for (int i = tid; i < work_iters; i += blockDim.x) {
        float x = (i + 1) * 1.000173f + bid * 0.5f;
        // some flops
        for (int k = 0; k < 8; ++k) {
            acc = fmaf(x, x + k, acc);
        }
    }
    // In-block reduction to avoid compiler eliminating work
    __shared__ float s[256];
    s[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid+stride];
        __syncthreads();
    }
    if (tid==0) out[bid] = s[0];
}

int main(int argc, char** argv){
    int Nreq = 200;           // total simulated requests
    double lambda = 80.0;     // arrivals per second
    int tokens = 32;          // tokens per request
    int Bmax = 16;            // max batch size
    double window_ms = 6.0;   // admission window [ms]
    int work_iters = 1<<14;   // per-step compute knob
    int block_threads = 256;  // threads per block

    for(int i=1;i<argc;++i){
        auto a = std::string(argv[i]);
        auto get = [&](const char* k){ return a.rfind(k,0)==0 ? atof(a.substr(strlen(k)).c_str()) : NAN; };
        if(a.rfind("--nreq=",0)==0) Nreq = (int)get("--nreq=");
        else if(a.rfind("--lambda=",0)==0) lambda = get("--lambda=");
        else if(a.rfind("--tokens=",0)==0) tokens = (int)get("--tokens=");
        else if(a.rfind("--bmax=",0)==0) Bmax = (int)get("--bmax=");
        else if(a.rfind("--window_ms=",0)==0) window_ms = get("--window_ms=");
        else if(a.rfind("--work=",0)==0) work_iters = (int)get("--work=");
        else if(a.rfind("--threads=",0)==0) block_threads = (int)get("--threads=");
    }
    assert(block_threads<=1024);

    // Generate Poisson arrivals
    std::mt19937 rng(123);
    std::exponential_distribution<double> expd(lambda);
    std::vector<Req> reqs(Nreq);
    double t=0.0; // simulated wall time (ms)
    double now = 0.0;
    for(int i=0;i<Nreq;++i){
        double gap_s = expd(rng); // seconds between arrivals
        now += gap_s*1000.0; // ms
        reqs[i].arrival = now;
        reqs[i].remaining = tokens;
    }

    float *d_out=nullptr; CHECK_CUDA(cudaMalloc(&d_out, sizeof(float)*Bmax));

    // Scheduling loop: interleaved decode steps; accept new arrivals up to Bmax each step
    std::vector<int> active; active.reserve(Bmax);
    size_t next_req = 0; // next not-yet-admitted request index
    double sim_time_ms = 0.0; // includes admission window + measured GPU time
    double total_gpu_ms = 0.0;

    while(true){
        // Admit at least one if none active and requests remain
        if(active.empty() && next_req < reqs.size()){
            double admit_until = std::max(sim_time_ms, reqs[next_req].arrival) + window_ms;
            // include all arrivals until admit_until, up to Bmax
            while(next_req < reqs.size() && reqs[next_req].arrival <= admit_until && (int)active.size()<Bmax){
                active.push_back((int)next_req);
                reqs[next_req].started = true;
                next_req++;
            }
            // advance simulated time by the window
            sim_time_ms = admit_until;
        } else {
            // top-up between steps (shorter window: half)
            double admit_until = sim_time_ms + window_ms*0.25;
            while(next_req < reqs.size() && reqs[next_req].arrival <= admit_until && (int)active.size()<Bmax){
                active.push_back((int)next_req);
                reqs[next_req].started = true;
                next_req++;
            }
            sim_time_ms = std::max(sim_time_ms, admit_until);
        }

        if(active.empty()) break; // done

        // Launch one decode step for all active sequences
        int b = (int)active.size();
        cudaEvent_t s,e; CHECK_CUDA(cudaEventCreate(&s)); CHECK_CUDA(cudaEventCreate(&e));
        CHECK_CUDA(cudaEventRecord(s));
        decode_step_kernel<<<b, block_threads>>>(d_out, work_iters);
        CHECK_CUDA(cudaEventRecord(e)); CHECK_CUDA(cudaEventSynchronize(e));
        float ms=0.f; CHECK_CUDA(cudaEventElapsedTime(&ms,s,e));
        CHECK_CUDA(cudaEventDestroy(s)); CHECK_CUDA(cudaEventDestroy(e));

        total_gpu_ms += ms;
        sim_time_ms += ms; // add GPU time to simulated wall time

        // Update request states
        for(int idx=0; idx<b; ++idx){
            int rid = active[idx];
            if(reqs[rid].first_token < 0.0) reqs[rid].first_token = sim_time_ms; // completes first step now
            reqs[rid].remaining -= 1;
        }
        // Remove finished
        active.erase(std::remove_if(active.begin(), active.end(), [&](int rid){
            if(reqs[rid].remaining<=0){ reqs[rid].done = sim_time_ms; return true; }
            return false;
        }), active.end());
    }

    // Compute metrics
    std::vector<double> ttft_ms, e2e_ms;
    ttft_ms.reserve(Nreq); e2e_ms.reserve(Nreq);
    for(auto &r: reqs){
        if(r.first_token>=0){
            ttft_ms.push_back(r.first_token - r.arrival);
            e2e_ms.push_back(r.done - r.arrival);
        }
    }
    auto pct = [](std::vector<double>& v, double p){
        std::sort(v.begin(), v.end());
        if(v.empty()) return 0.0; size_t k = (size_t)std::clamp((p*(v.size()-1)), 0.0, (double)(v.size()-1));
        return v[k];
    };

    double mean_ttft = std::accumulate(ttft_ms.begin(), ttft_ms.end(), 0.0)/ttft_ms.size();
    double p95_ttft = pct(ttft_ms, 0.95);
    double mean_e2e = std::accumulate(e2e_ms.begin(), e2e_ms.end(), 0.0)/e2e_ms.size();
    double p95_e2e = pct(e2e_ms, 0.95);

    long long total_tokens = (long long)tokens * (long long)Nreq;
    double tps_gpu = (double)total_tokens / (total_gpu_ms/1000.0);

    printf("CONFIG nreq=%d lambda=%.1f/s tokens=%d Bmax=%d window_ms=%.2f work=%d\n", Nreq, lambda, tokens, Bmax, window_ms, work_iters);
    printf("GPU time: %.3f ms  | tokens: %lld  | tokens/sec (GPU): %.2f\n", total_gpu_ms, total_tokens, tps_gpu);
    printf("TTFT mean=%.3f ms  p95=%.3f ms\n", mean_ttft, p95_ttft);
    printf("E2E  mean=%.3f ms  p95=%.3f ms\n", mean_e2e, p95_e2e);

    CHECK_CUDA(cudaFree(d_out));
    return 0;
}
```

### File: `topics/16-throughput-latency-batching-policy/code/decode_batch_bench_hip.cpp`

```cpp
// Minimal HIP decode-step batching benchmark (AMD ROCm)
// Build: hipcc -O3 -std=c++17 --offload-arch=${GFX_ARCH} decode_batch_bench_hip.cpp -o decode_batch_bench_hip
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cassert>

#ifndef CHECK_HIP
#define CHECK_HIP(x) do { hipError_t e=(x); if(e!=hipSuccess){\
  fprintf(stderr,"HIP error %s:%d: %s\n",__FILE__,__LINE__,hipGetErrorString(e)); exit(1);} } while(0)
#endif

struct Req { double arrival; int remaining; double first_token=-1.0; double done=-1.0; bool started=false; };

__global__ void decode_step_kernel(float* out, int work_iters) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    float acc = 0.f;
    for (int i = tid; i < work_iters; i += blockDim.x) {
        float x = (i + 1) * 1.000173f + bid * 0.5f;
        #pragma unroll 1
        for (int k = 0; k < 8; ++k) {
            acc = fmaf(x, x + k, acc);
        }
    }
    __shared__ float s[256];
    s[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid+stride];
        __syncthreads();
    }
    if (tid==0) out[bid] = s[0];
}

int main(int argc, char** argv){
    int Nreq = 200; double lambda = 80.0; int tokens = 32; int Bmax = 16; double window_ms = 6.0; int work_iters = 1<<14; int block_threads=256;
    for(int i=1;i<argc;++i){
        auto a = std::string(argv[i]);
        auto get = [&](const char* k){ return a.rfind(k,0)==0 ? atof(a.substr(strlen(k)).c_str()) : NAN; };
        if(a.rfind("--nreq=",0)==0) Nreq = (int)get("--nreq=");
        else if(a.rfind("--lambda=",0)==0) lambda = get("--lambda=");
        else if(a.rfind("--tokens=",0)==0) tokens = (int)get("--tokens=");
        else if(a.rfind("--bmax=",0)==0) Bmax = (int)get("--bmax=");
        else if(a.rfind("--window_ms=",0)==0) window_ms = get("--window_ms=");
        else if(a.rfind("--work=",0)==0) work_iters = (int)get("--work=");
        else if(a.rfind("--threads=",0)==0) block_threads = (int)get("--threads=");
    }
    assert(block_threads<=1024);

    std::mt19937 rng(123);
    std::exponential_distribution<double> expd(lambda);
    std::vector<Req> reqs(Nreq);
    double now = 0.0;
    for(int i=0;i<Nreq;++i){ now += expd(rng)*1000.0; reqs[i].arrival = now; reqs[i].remaining = tokens; }

    float *d_out=nullptr; CHECK_HIP(hipMalloc(&d_out, sizeof(float)*Bmax));

    std::vector<int> active; active.reserve(Bmax);
    size_t next_req=0; double sim_time_ms=0.0; double total_gpu_ms=0.0;

    while(true){
        if(active.empty() && next_req < reqs.size()){
            double admit_until = std::max(sim_time_ms, reqs[next_req].arrival) + window_ms;
            while(next_req < reqs.size() && reqs[next_req].arrival <= admit_until && (int)active.size()<Bmax){
                active.push_back((int)next_req); reqs[next_req].started=true; next_req++; }
            sim_time_ms = admit_until;
        } else {
            double admit_until = sim_time_ms + window_ms*0.25;
            while(next_req < reqs.size() && reqs[next_req].arrival <= admit_until && (int)active.size()<Bmax){
                active.push_back((int)next_req); reqs[next_req].started=true; next_req++; }
            sim_time_ms = std::max(sim_time_ms, admit_until);
        }
        if(active.empty()) break;

        hipEvent_t s,e; CHECK_HIP(hipEventCreate(&s)); CHECK_HIP(hipEventCreate(&e));
        CHECK_HIP(hipEventRecord(s));
        hipLaunchKernelGGL(decode_step_kernel, dim3((int)active.size()), dim3(block_threads), 0, 0, d_out, work_iters);
        CHECK_HIP(hipEventRecord(e)); CHECK_HIP(hipEventSynchronize(e));
        float ms=0.f; CHECK_HIP(hipEventElapsedTime(&ms,s,e));
        CHECK_HIP(hipEventDestroy(s)); CHECK_HIP(hipEventDestroy(e));
        total_gpu_ms += ms; sim_time_ms += ms;

        for(int rid: active){ if(reqs[rid].first_token<0.0) reqs[rid].first_token=sim_time_ms; reqs[rid].remaining -= 1; }
        active.erase(std::remove_if(active.begin(), active.end(), [&](int rid){ if(reqs[rid].remaining<=0){ reqs[rid].done=sim_time_ms; return true;} return false; }), active.end());
    }

    std::vector<double> ttft_ms, e2e_ms; ttft_ms.reserve(Nreq); e2e_ms.reserve(Nreq);
    for(auto &r: reqs){ if(r.first_token>=0){ ttft_ms.push_back(r.first_token - r.arrival); e2e_ms.push_back(r.done - r.arrival);} }
    auto pct = [](std::vector<double>& v, double p){ std::sort(v.begin(), v.end()); if(v.empty()) return 0.0; size_t k=(size_t)std::clamp((p*(v.size()-1)),0.0,(double)(v.size()-1)); return v[k]; };
    double mean_ttft = std::accumulate(ttft_ms.begin(), ttft_ms.end(), 0.0)/ttft_ms.size();
    double p95_ttft = pct(ttft_ms, 0.95);
    double mean_e2e  = std::accumulate(e2e_ms.begin(), e2e_ms.end(), 0.0)/e2e_ms.size();
    double p95_e2e   = pct(e2e_ms, 0.95);
    long long total_tokens = (long long)tokens * (long long)Nreq;
    double tps_gpu = (double)total_tokens / (total_gpu_ms/1000.0);

    printf("CONFIG nreq=%d lambda=%.1f/s tokens=%d Bmax=%d window_ms=%.2f work=%d\n", Nreq, lambda, tokens, Bmax, window_ms, work_iters);
    printf("GPU time: %.3f ms  | tokens: %lld  | tokens/sec (GPU): %.2f\n", total_gpu_ms, total_tokens, tps_gpu);
    printf("TTFT mean=%.3f ms  p95=%.3f ms\n", mean_ttft, p95_ttft);
    printf("E2E  mean=%.3f ms  p95=%.3f ms\n", mean_e2e, p95_e2e);

    CHECK_HIP(hipFree(d_out));
    return 0;
}
```

### Build Commands

CUDA:

```
SM_ARCH=sm_80  # A100 example; change to your GPU
nvcc -O3 -std=c++17 -arch=${SM_ARCH} -lineinfo topics/16-throughput-latency-batching-policy/code/decode_batch_bench.cu -o decode_batch_bench
```

ROCm/HIP:

```
GFX_ARCH=gfx942  # MI300X example; set to your GPU (e.g., gfx90a for MI200)
hipcc -O3 -std=c++17 --offload-arch=${GFX_ARCH} topics/16-throughput-latency-batching-policy/code/decode_batch_bench_hip.cpp -o decode_batch_bench_hip
```

### Run Examples

Interactive SLO, shorter window:

```
./decode_batch_bench --lambda=60 --tokens=32 --bmax=8 --window_ms=3 --work=16384
```

Bulk/offline, larger batches:

```
./decode_batch_bench --lambda=120 --tokens=512 --bmax=64 --window_ms=20 --work=16384
```

Compare GPU tokens/sec and p95 TTFT across settings. Expect larger batches to improve tokens/sec and increase TTFT.

## Profiling and Validation

### NVIDIA (Nsight Systems/Compute)

- **Timeline**: kernel cadence and gaps

```
nsys profile -o nsys_decode --stats=true ./decode_batch_bench --lambda=80 --bmax=16 --window_ms=6
```

- **Kernel metrics** (Nsight Compute):

```
ncu --set full --kernel-name regex:decode_step_kernel ./decode_batch_bench --lambda=80 --bmax=16 --window_ms=6
```

Key counters to inspect:

- `sm__warps_active.avg.pct_of_peak_sustained_active` (≥ 40% desirable in this toy)
- `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active` (compute saturation)
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` (< 30% indicates compute‑bound here)
- `launch__occupancy_limit_active_warps` (confirm occupancy not limited by regs)

### AMD (rocprof / Omnitrace)

```
rocprof --stats --hsa-trace --timestamp on ./decode_batch_bench_hip --lambda=80 --bmax=16 --window_ms=6
```

Inspect:

- `GRBM_GUI_ACTIVE` (GPU busy time fraction)
- `SQ_WAVES` and `VALUUtilization` (compute saturation)
- `L2CacheHit` and `MemUnitBusy` (memory pressure)

### Validation Procedure

1. Fix `--work` and sweep `--bmax` ∈ {1, 2, 4, 8, 16, 32} with $\Delta=6$ ms.
2. Record tokens/sec (GPU) and p95 TTFT.
3. Verify monotonic increase of tokens/sec with $b$ and non‑decreasing TTFT.

## Performance Checklist

- Sequence bucketing by length for prefill; ragged/paged batching for decode.
- Admission window $\Delta$ set by SLO: start at 2–8 ms for interactive, 10–50 ms for bulk.
- Separate queues: prefill‑heavy vs decode‑heavy; ensure decode has preemption or priority.
- Limit $B_{\max}$ based on measured `t_step(b)` curve; avoid LMK “cliff” where L2 spills.
- Use persistent kernels or CUDA/HIP Graphs to amortize launch overhead at small $b$.
- Cap live KV cache bytes with paging; avoid OOM stalls that break latency guarantees.
- Telemetry: p50/p90/p95 TTFT, tokens/sec, queue depth, GPU busy %, admission window used.

## Troubleshooting

| Symptom                                        | Likely Cause                                 | Fix                                                                               |
| ---------------------------------------------- | -------------------------------------------- | --------------------------------------------------------------------------------- |
| TTFT spikes at low load                        | Window too large; batching waiting dominates | Reduce $\Delta$ for interactive queue; enable immediate dispatch for batch size 1 |
| Tokens/sec plateaus early                      | Kernel not scaling with $b$; memory‑bound    | Improve tiling, vectorized loads, fused ops; ensure L2 reuse                      |
| p95 TTFT > SLO despite small window            | Queueing due to mixed prefill & decode       | Isolate queues; prioritize decode; limit prefill concurrency                      |
| High kernel gaps on timeline                   | Launch overhead; CPU scheduler contention    | Use persistent kernels or CUDA/HIP Graphs; pin scheduler thread                   |
| OOM or paging storms                           | KV cache unmanaged                           | Implement KV paging with hard caps; spill long tails                              |
| Throughput unstable                            | Arrival bursts not smoothed                  | Use short EMA of $\lambda$ to adjust $\Delta$ and $B_{\max}$                      |
| Good GPU tokens/sec but poor perceived latency | Admission and network time excluded          | Track wall‑clock including admission, RPC, and tokenizer                          |

## Acceptance Criteria

- Both CUDA and HIP benchmarks compile and run under 5 seconds with defaults on a modern GPU.
- Sweeping $b$ from 1→16 increases GPU tokens/sec by ≥2× on this microbenchmark.
- With $\Delta$ reduced from 6 ms → 2 ms at fixed load, p95 TTFT decreases measurably (≥15%).
- Nsight/rocprof timelines show minimal gaps (<2 ms) between successive decode steps at $b≥8$.

## SLO‑Driven Batching Policy (Reference)

1. Maintain two queues: `interactive` and `bulk`. Decode tokens from `interactive` have priority.
2. Estimate arrival rate $\hat{\lambda}$ with EMA over 250–500 ms.
3. Choose `Δ` and `Bmax` per queue:

   - `interactive`: `Δ = clamp(SLO_p95 - t_step(b) - margin, 1, 8) ms`, `Bmax` in \[4, 16].
   - `bulk`: `Δ` in \[10, 50] ms, `Bmax` in \[16, 128] depending on VRAM.

4. Between steps, top‑up active batch from waiting requests up to `Bmax`.
5. Preempt long prefill waves if interactive queue builds beyond a depth threshold.
6. Periodically re‑fit the measured `t_step(b)` curve and adjust `Bmax`.

### Pseudocode

```python
# runs on CPU scheduler
while True:
    q = interactive if q_depth(interactive) > 0 else bulk
    b = min(q.size_within(Δ_ms), Bmax)
    if b == 0:
        sleep(Δ_ms/2); continue
    active = q.pop(b)
    for step in range(steps):
        launch_decode_kernel(active)
        admit_topup = q.pop_within(Δ_topup_ms, Bmax - len(active))
        active.extend(admit_topup)
        active = [r for r in active if not r.finished()]
```

## Further Work

- Integrate with real attention kernels (FP16/BF16 with accumulation) and measure Tensor Core/MFMA utilization vs $b$.
- Add CUDA/HIP Graphs and a persistent‑kernel variant to quantify launch amortization.
- Extend simulator with network + tokenizer time for end‑to‑end SLO accounting.
- Multi‑GPU: overlap NCCL/RCCL all‑gather with compute; batch across replicas with rendezvous every N tokens.
