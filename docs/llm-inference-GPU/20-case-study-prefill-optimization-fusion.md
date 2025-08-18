# Case Study: Prefill Optimization & Fusion

## Summary

Prefill dominates latency for long-context LLMs. The path includes heavy GEMMs and bandwidth-bound elementwise layers (bias, activation, residual add, layer norm) plus attention softmax. This case study demonstrates how fusing post-GEMM epilogues and attention softmax reduces global memory traffic and kernel launch overhead. We provide runnable CUDA/HIP code, a validation harness, and a profiling checklist to quantify gains.

## Why It Matters for LLM Inference

Prefill (processing the prompt) is throughput-oriented and benefits from high arithmetic intensity in GEMMs. However, the non-GEMM stages are often memory-bound and launch-bound. Fusing bias+activation+residual+layernorm and scale+mask+softmax minimizes reads/writes and reduces launches. The net effect shortens end-to-end prefill latency, enabling higher tokens-per-second at large sequence lengths.

## Key Concepts and Formulas

- Arithmetic intensity (AI): `AI = FLOPs / Bytes`. For elementwise ops, AI is low; performance is capped by memory bandwidth.
- Memory traffic model (per element, FP16 stored = 2 bytes):

  - Unfused bias→GELU→residual→LayerNorm (two-pass LN):

    - Bias add: 6 B (read y, b; write)
    - GELU: 4 B (read; write)
    - Residual add: 6 B (read y, r; write)
    - LayerNorm: ≈10 B (first read; second pass read+γ+β; write)
    - Total ≈ 26 B/elem.

  - Fused epilogue with on-chip staging (shared memory):

    - Read y, b, r once (6 B), γ and β once (4 B), single write (2 B)
    - Total ≈ 12 B/elem.

- Numeric example (B=1, S=4096, H=4096 → M=S×B rows, H cols):

  - Elements: 4096×4096 = 16,777,216
  - Unfused traffic ≈ 416 MiB; fused ≈ 192 MiB; \~2.17× less global bytes.

- Softmax with scale and causal mask can be implemented as two reductions (max, sum) with one global read and one global write per element when staging a row in shared memory.

## GPU Deep Dive

### NVIDIA specifics

- Warps of 32 threads, SMs with Tensor Cores. Prefer vectorized loads (`float4`, `__half2`) and coalesced row-major access. Use `__expf`, fast-math (when acceptable), and dynamic shared memory. Nsight Compute counters: DRAM Throughput, L2 Hit Rate, Achieved Occupancy, Warp Stall Reasons.

### AMD specifics

- Wavefronts of 64 threads on CUs with MFMA/XDLOPs. Use LDS (shared memory) to stage row tiles. Prefer 128-bit vectorized accesses (`float4`). rocprof counters: Memory BW, L2 Cache Hit Rate, SQ_WAVES, VALU Busy.

## Implementation

The single-source program below provides:

1. A fused epilogue kernel: bias + GELU + residual add + LayerNorm (two-pass, values staged in shared memory/LDS).
2. A fused attention row softmax: scale + causal mask + softmax.
3. CPU references, correctness checks, and timing.

> Constraints: Designed for rows up to a few thousand elements (e.g., H≤8192, S≤8192). For larger rows, tile the row in segments.

### Code: `topics/20-case-study-prefill-optimization-fusion/code/prefill_fusion.cu`

```cpp
// Single-source CUDA/HIP program for prefill fusion case study
// Build (NVIDIA): nvcc -O3 -std=c++17 -arch=sm_80 -lineinfo prefill_fusion.cu -o prefill_fusion
// Build (AMD):    hipcc -O3 -std=c++17 --offload-arch=gfx942 prefill_fusion.cu -o prefill_fusion

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <cassert>

#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#define DEVFN __global__
#define API_CHECK(x) do { auto _e = (x); if (_e != hipSuccess) { \
  fprintf(stderr, "HIP error %d at %s:%d\n", (int)_e, __FILE__, __LINE__); abort(); } } while(0)
#define DEV_MEMCPY(dst,src,n,kind) API_CHECK(hipMemcpy(dst,src,n,kind))
#define DEV_MALLOC(p,n) API_CHECK(hipMalloc(p,n))
#define DEV_FREE(p) API_CHECK(hipFree(p))
#define DEV_DEVICE_SYNC() API_CHECK(hipDeviceSynchronize())
#define DEV_EVENT_T hipEvent_t
#define DEV_EVENT_CREATE(e) API_CHECK(hipEventCreate(e))
#define DEV_EVENT_RECORD(e) API_CHECK(hipEventRecord(e, 0))
#define DEV_EVENT_ELAPSED_MS(ms, start, stop) API_CHECK(hipEventElapsedTime(&ms, start, stop))
#define DEV_EVENT_DESTROY(e) API_CHECK(hipEventDestroy(e))
#define DEV_GET_LAST_ERROR() API_CHECK(hipGetLastError())
#else
#include <cuda_runtime.h>
#define DEVFN __global__
#define API_CHECK(x) do { auto _e = (x); if (_e != cudaSuccess) { \
  fprintf(stderr, "CUDA error %d (%s) at %s:%d\n", (int)_e, cudaGetErrorString(_e), __FILE__, __LINE__); abort(); } } while(0)
#define DEV_MEMCPY(dst,src,n,kind) API_CHECK(cudaMemcpy(dst,src,n,kind))
#define DEV_MALLOC(p,n) API_CHECK(cudaMalloc(p,n))
#define DEV_FREE(p) API_CHECK(cudaFree(p))
#define DEV_DEVICE_SYNC() API_CHECK(cudaDeviceSynchronize())
#define DEV_EVENT_T cudaEvent_t
#define DEV_EVENT_CREATE(e) API_CHECK(cudaEventCreate(e))
#define DEV_EVENT_RECORD(e) API_CHECK(cudaEventRecord(e, 0))
#define DEV_EVENT_ELAPSED_MS(ms, start, stop) API_CHECK(cudaEventElapsedTime(&ms, start, stop))
#define DEV_EVENT_DESTROY(e) API_CHECK(cudaEventDestroy(e))
#define DEV_GET_LAST_ERROR() API_CHECK(cudaGetLastError())
#endif

// Fast GELU (tanh approximation)
__device__ __forceinline__ float gelu(float x) {
    const float kAlpha = 0.7978845608028654f;   // sqrt(2/pi)
    const float kBeta  = 0.035677408136300125f; // 0.044715 * sqrt(2/pi)
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(kAlpha * (x + kBeta * x3)));
}

// Fused epilogue per-row: (y + b) -> GELU -> + residual -> LayerNorm
// Inputs: y[M,H], b[H], r[M,H], gamma[H], beta[H]; Output: out[M,H]
// Two-pass LN, but intermediate values are staged in shared memory to avoid global writes.
DEVFN void fused_bias_gelu_residual_ln(const float* __restrict__ y,
                                       const float* __restrict__ b,
                                       const float* __restrict__ r,
                                       const float* __restrict__ gamma,
                                       const float* __restrict__ beta,
                                       float* __restrict__ out,
                                       int H, float eps) {
    extern __shared__ float smem[]; // size: H + 2*blockDim.x
    float* rowbuf = smem;           // [H] fused values
    float* redbuf = smem + H;       // [2*blockDim.x] reduction scratch

    int row = blockIdx.x; // one block per row
    const float* yrow = y + (size_t)row * H;
    const float* rrow = r + (size_t)row * H;
    float* outrow = out + (size_t)row * H;

    // Pass 1: compute fused value and accumulate sum/sumsq
    float thread_sum = 0.f;
    float thread_sumsq = 0.f;

    for (int j = threadIdx.x; j < H; j += blockDim.x) {
        float v = yrow[j] + b[j];
        v = gelu(v);
        v = v + rrow[j];
        rowbuf[j] = v; // stage
        thread_sum += v;
        thread_sumsq += v * v;
    }

    // Reduce within block
    redbuf[threadIdx.x] = thread_sum;
    redbuf[threadIdx.x + blockDim.x] = thread_sumsq;
    __syncthreads();

    // Parallel tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            redbuf[threadIdx.x] += redbuf[threadIdx.x + stride];
            redbuf[threadIdx.x + blockDim.x] += redbuf[threadIdx.x + blockDim.x + stride];
        }
        __syncthreads();
    }

    float mean = redbuf[0] / (float)H;
    float var  = redbuf[blockDim.x] / (float)H - mean * mean;
    float inv_std = rsqrtf(var + eps);

    // Pass 2: normalize and write
    for (int j = threadIdx.x; j < H; j += blockDim.x) {
        float xhat = (rowbuf[j] - mean) * inv_std;
        outrow[j] = xhat * gamma[j] + beta[j];
    }
}

// Fused attention softmax per row: scale + causal mask + softmax
// scores[M, N] -> probs[M, N] ; if causal, mask columns j > row
DEVFN void fused_scale_mask_softmax(const float* __restrict__ scores,
                                    float* __restrict__ probs,
                                    int N, float scale, int causal) {
    extern __shared__ float smem[]; // [N] row staging + [blockDim.x] reduction
    float* row = smem;
    float* red = smem + N; // [blockDim.x]

    int row_id = blockIdx.x;
    const float* srow = scores + (size_t)row_id * N;
    float* prow = probs + (size_t)row_id * N;

    // Load/scale/mask; compute local max
    float tmax = -INFINITY;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        float v = srow[j] * scale;
        if (causal && j > row_id) v = -INFINITY;
        row[j] = v;
        tmax = fmaxf(tmax, v);
    }

    // Reduce max
    red[threadIdx.x] = tmax;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    float rmax = red[0];

    // Compute sum of exp
    float tsum = 0.f;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        float e = (row[j] == -INFINITY) ? 0.f : expf(row[j] - rmax);
        row[j] = e; // reuse to avoid extra storage
        tsum += e;
    }

    red[threadIdx.x] = tsum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    float rsum = red[0] + 1e-30f; // avoid div-by-zero

    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        float p = row[j] / rsum;
        prow[j] = (causal && j > row_id) ? 0.f : p;
    }
}

// ---------------- CPU references for validation ----------------
static inline float gelu_ref(float x){
    const float kAlpha = 0.7978845608028654f;
    const float kBeta  = 0.035677408136300125f;
    float x3 = x*x*x;
    return 0.5f * x * (1.0f + std::tanh(kAlpha*(x + kBeta*x3)));
}

void ref_fused_epilogue(const std::vector<float>& y, const std::vector<float>& b,
                        const std::vector<float>& r, const std::vector<float>& gamma,
                        const std::vector<float>& beta, std::vector<float>& out,
                        int M, int H, float eps){
    for(int i=0;i<M;++i){
        const float* yrow=&y[(size_t)i*H];
        const float* rrow=&r[(size_t)i*H];
        float* outrow=&out[(size_t)i*H];
        std::vector<float> tmp(H);
        double sum=0.0, sumsq=0.0;
        for(int j=0;j<H;++j){
            float v = gelu_ref(yrow[j] + b[j]) + rrow[j];
            tmp[j]=v; sum += v; sumsq += (double)v*v;
        }
        float mean = (float)(sum / H);
        float var  = (float)(sumsq / H - (double)mean*mean);
        float inv_std = 1.0f/std::sqrt(var + eps);
        for(int j=0;j<H;++j){
            float xhat = (tmp[j]-mean)*inv_std;
            outrow[j] = xhat*gamma[j] + beta[j];
        }
    }
}

void ref_softmax(const std::vector<float>& scores, std::vector<float>& probs,
                 int M, int N, float scale, bool causal){
    for(int i=0;i<M;++i){
        const float* srow=&scores[(size_t)i*N];
        float* prow=&probs[(size_t)i*N];
        float m=-INFINITY;
        for(int j=0;j<N;++j){
            float v = srow[j]*scale;
            if(causal && j>i) v = -INFINITY;
            prow[j]=v; m=std::max(m,v);
        }
        double sum=0.0;
        for(int j=0;j<N;++j){
            float e = (prow[j]==-INFINITY)?0.f:std::exp(prow[j]-m);
            prow[j]=e; sum+=e;
        }
        for(int j=0;j<N;++j){ prow[j]=(prow[j]/(float)(sum+1e-30)); if(causal && j>i) prow[j]=0.f; }
    }
}

// Utility: random init
void fill_random(std::vector<float>& v, float scale=1.0f){
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-scale, scale);
    for(auto& x: v) x = dist(rng);
}

int main(int argc, char** argv){
    // Problem sizes (adjustable via env or argv if desired)
    int B = 1;           // batch
    int S = 2048;        // sequence length
    int H = 4096;        // hidden size per row
    int M = B * S;       // rows
    int N = S;           // row length for softmax

    if(argc==4){ B=atoi(argv[1]); S=atoi(argv[2]); H=atoi(argv[3]); M=B*S; N=S; }

    const float eps = 1e-5f;
    const float softmax_scale = 1.0f/std::sqrt((float)H/ (float)B + 1e-9f); // illustrative
    const bool causal = true;

    size_t bytes_y = (size_t)M*H*sizeof(float);
    size_t bytes_vecH = (size_t)H*sizeof(float);
    size_t bytes_soft = (size_t)M*N*sizeof(float);

    std::vector<float> h_y(M*H), h_b(H), h_r(M*H), h_g(H), h_bt(H), h_out_ref(M*H), h_out_gpu(M*H);
    std::vector<float> h_scores(M*N), h_probs_ref(M*N), h_probs_gpu(M*N);

    fill_random(h_y, 1.0f); fill_random(h_b, 0.1f); fill_random(h_r, 1.0f); fill_random(h_g, 0.5f); fill_random(h_bt, 0.5f);
    fill_random(h_scores, 0.5f);

    // CPU references
    ref_fused_epilogue(h_y, h_b, h_r, h_g, h_bt, h_out_ref, M, H, eps);
    ref_softmax(h_scores, h_probs_ref, M, N, softmax_scale, causal);

    // Device alloc
    float *d_y, *d_b, *d_r, *d_g, *d_bt, *d_out;
    float *d_scores, *d_probs;
    DEV_MALLOC((void**)&d_y, bytes_y);
    DEV_MALLOC((void**)&d_r, bytes_y);
    DEV_MALLOC((void**)&d_out, bytes_y);
    DEV_MALLOC((void**)&d_b, bytes_vecH);
    DEV_MALLOC((void**)&d_g, bytes_vecH);
    DEV_MALLOC((void**)&d_bt, bytes_vecH);
    DEV_MALLOC((void**)&d_scores, bytes_soft);
    DEV_MALLOC((void**)&d_probs, bytes_soft);

    DEV_MEMCPY(d_y, h_y.data(), bytes_y, hipMemcpyHostToDevice);
    DEV_MEMCPY(d_r, h_r.data(), bytes_y, hipMemcpyHostToDevice);
    DEV_MEMCPY(d_b, h_b.data(), bytes_vecH, hipMemcpyHostToDevice);
    DEV_MEMCPY(d_g, h_g.data(), bytes_vecH, hipMemcpyHostToDevice);
    DEV_MEMCPY(d_bt, h_bt.data(), bytes_vecH, hipMemcpyHostToDevice);
    DEV_MEMCPY(d_scores, h_scores.data(), bytes_soft, hipMemcpyHostToDevice);

    // Kernel launch params
    dim3 block(256);
    dim3 grid_e(M);
    size_t shmem_e = (size_t)H*sizeof(float) + 2*block.x*sizeof(float);

    dim3 grid_s(M);
    size_t shmem_s = (size_t)N*sizeof(float) + block.x*sizeof(float);

    // Warm-up
    fused_bias_gelu_residual_ln<<<grid_e, block, shmem_e>>>(d_y, d_b, d_r, d_g, d_bt, d_out, H, eps);
    fused_scale_mask_softmax<<<grid_s, block, shmem_s>>>(d_scores, d_probs, N, softmax_scale, causal?1:0);
    DEV_GET_LAST_ERROR(); DEV_DEVICE_SYNC();

    // Timing
    const int iters = 50;
    DEV_EVENT_T start, stop; DEV_EVENT_CREATE(&start); DEV_EVENT_CREATE(&stop);

    // Epilogue timing
    DEV_EVENT_RECORD(start);
    for(int it=0; it<iters; ++it){
        fused_bias_gelu_residual_ln<<<grid_e, block, shmem_e>>>(d_y, d_b, d_r, d_g, d_bt, d_out, H, eps);
    }
    DEV_EVENT_RECORD(stop); DEV_DEVICE_SYNC();
    float ms_e=0.f; DEV_EVENT_ELAPSED_MS(ms_e, start, stop); ms_e/=iters;

    // Softmax timing
    DEV_EVENT_RECORD(start);
    for(int it=0; it<iters; ++it){
        fused_scale_mask_softmax<<<grid_s, block, shmem_s>>>(d_scores, d_probs, N, softmax_scale, causal?1:0);
    }
    DEV_EVENT_RECORD(stop); DEV_DEVICE_SYNC();
    float ms_s=0.f; DEV_EVENT_ELAPSED_MS(ms_s, start, stop); ms_s/=iters;

    // Copy back
    DEV_MEMCPY(h_out_gpu.data(), d_out, bytes_y, hipMemcpyDeviceToHost);
    DEV_MEMCPY(h_probs_gpu.data(), d_probs, bytes_soft, hipMemcpyDeviceToHost);

    // Validate
    auto l2 = [](const std::vector<float>& a, const std::vector<float>& b){
        double se=0.0, sn=0.0; size_t n=a.size();
        for(size_t i=0;i<n;++i){ double d=(double)a[i]-b[i]; se+=d*d; sn+= (double)a[i]*a[i]; }
        return std::sqrt(se/(sn+1e-30));
    };

    double rel_ep = l2(h_out_ref, h_out_gpu);
    double rel_sm = l2(h_probs_ref, h_probs_gpu);

    // Bandwidth model (global bytes moved)
    double bytes_per_elem_e_fused = 12.0; // from analysis
    double total_bytes_e = (double)M * H * bytes_per_elem_e_fused;
    double gbps_e = (total_bytes_e / (ms_e/1000.0)) / 1e9;

    double bytes_per_elem_s = 8.0; // 4B read + 4B write per element (staged once); mask/scale in registers
    double total_bytes_s = (double)M * N * bytes_per_elem_s;
    double gbps_s = (total_bytes_s / (ms_s/1000.0)) / 1e9;

    printf("Sizes: B=%d S=%d H=%d -> M=%d rows\n", B,S,H,M);
    printf("Epilogue fused: avg %.3f ms, est BW %.1f GB/s, rel-L2 %.3e\n", ms_e, gbps_e, rel_ep);
    printf("Softmax fused:  avg %.3f ms, est BW %.1f GB/s, rel-L2 %.3e\n", ms_s, gbps_s, rel_sm);

    DEV_EVENT_DESTROY(start); DEV_EVENT_DESTROY(stop);

    DEV_FREE(d_y); DEV_FREE(d_r); DEV_FREE(d_out); DEV_FREE(d_b); DEV_FREE(d_g); DEV_FREE(d_bt);
    DEV_FREE(d_scores); DEV_FREE(d_probs);
    return 0;
}
```

### Build Commands

- NVIDIA:

```
nvcc -O3 -std=c++17 -arch=sm_80 -lineinfo topics/20-case-study-prefill-optimization-fusion/code/prefill_fusion.cu -o prefill_fusion
```

- AMD:

```
hipcc -O3 -std=c++17 --offload-arch=gfx942 topics/20-case-study-prefill-optimization-fusion/code/prefill_fusion.cu -o prefill_fusion
```

### Run

```
./prefill_fusion                 # defaults B=1 S=2048 H=4096
./prefill_fusion 1 4096 4096     # larger example
```

## Profiling and Validation

### NVIDIA (Nsight Systems/Compute)

- Timeline (launch overhead):

```
nsys profile -t cuda,osrt -o trace_prefill ./prefill_fusion
```

- Kernel metrics:

```
ncu --target-processes all \
    --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed, \
             lts__t_bytes.sum,sm__warps_active.avg.pct_of_peak_sustained_active, \
             sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active \
    ./prefill_fusion
```

Interpretation: Fused epilogue should approach memory roofline (high DRAM %), moderate-to-high L2 bytes, and reasonable occupancy.

### AMD (rocprof)

```
rocprof --hip-trace --timestamp on --stats ./prefill_fusion
```

Focus counters: memory BW utilization, L2 cache hit rate, SQ_WAVES (enough waves in flight). Expect high memory BW for fused epilogue and softmax.

### Numerical Checks

The program reports relative L2 error between GPU and CPU references. Acceptance: rel-L2 ≤ 1e-5 for both kernels.

## Performance Checklist

- Vectorized, coalesced global loads/stores (row-major).
- One global read per input tensor and one global write for outputs.
- Shared memory within per-row block budget (H·4 B + O(threads)).
- Block size tuned (e.g., 256–512) with ≥50% achieved occupancy.
- Launch count minimized (2 kernels here vs 5–6 unfused operators).
- Nsight/rocprof: DRAM throughput ≥70% of peak for epilogue; ≥60% for softmax.
- Validation: rel-L2 ≤ 1e-5.

## Troubleshooting

| Symptom                                         | Likely cause                                    | Fix                                                                                |
| ----------------------------------------------- | ----------------------------------------------- | ---------------------------------------------------------------------------------- |
| rel-L2 is large or NaNs in LN                   | Negative variance due to precision/cancellation | Accumulate in float, add epsilon, verify reduction; disable fast-math if necessary |
| Kernel launch fails with too much shared memory | H too large for per-row staging                 | Tile the row (process H in segments) or reduce block size                          |
| Low bandwidth (<50% peak)                       | Uncoalesced access or insufficient waves/warps  | Ensure contiguous row access, increase blockDim, launch more rows concurrently     |
| Divergence in softmax                           | Incorrect causal mask index                     | Verify row_id vs column index condition                                            |
| Occupancy very low                              | Excess registers or large shmem                 | Tune block size, split kernel, use `-maxrregcount` (CUDA)                          |
| Watchdog timeout                                | Very large M,N on display GPU                   | Reduce problem size or run on compute-only GPU                                     |
| Different results across runs                   | Non-deterministic math flags                    | Set deterministic build flags; avoid atomics; fix seeds                            |

## Acceptance Criteria

- Program compiles with indicated commands on CUDA 12.x or ROCm/HIP 6.x.
- Validation passes: rel-L2 ≤ 1e-5 for both fused epilogue and softmax.
- Profiling shows reduced kernel launches vs unfused baseline and high memory BW.
- For B=1, S=4096, H=4096: estimated global bytes \~192 MiB for fused epilogue (vs \~416 MiB unfused), derived from the traffic model.

## Further Work

- Integrate GEMM epilogue fusion via cuBLASLt/rocBLAS (bias+activation in GEMM epilogue) to remove extra reads of y.
- Vectorize with `float4` loads and consider `__half2` path with FP32 accumulation.
- Tile very large rows (H>8192) and overlap reductions using warp shuffles.
- Fuse dropout and residual scaling when training/finetuning paths are considered (inference usually omits dropout).
- Capture prefill subgraph with CUDA/HIP Graphs to eliminate launch overhead for repeated shapes.
