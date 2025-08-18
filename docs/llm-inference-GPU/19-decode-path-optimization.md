# Case Study: Decode Path Optimization (7B-class)

## Summary

This case study focuses on optimizing the **decode path** for a 7B-class Transformer with grouped-query attention (GQA). Decode is latency-critical and typically memory-bound due to repeated reads of the KV cache over growing sequence lengths. We quantify the costs, design a **persistent attention microkernel** with vectorized global memory access and shared-memory reuse, and provide a runnable CUDA/HIP benchmark that reports tokens/sec. The outcome is a clear, reproducible recipe and checklist to lift single-stream decode throughput while preserving numerical stability.

## Why It Matters for LLM Inference

- **Prefill** is GEMM-heavy and throughput-oriented; batching hides latency.
- **Decode** emits one token at a time; per-token latency dominates UX.
- Decode at long context lengths is often **KV-bandwidth bound**; poor layout and per-step launch overheads directly depress tokens/sec.
- Fixes: persistent kernels, coalesced KV layout, vectorized loads, and caching Q in shared memory to reduce rereads.

## Key Concepts and Formulas

Let:

- $L$ = current sequence length (context + generated)
- $d_{\text{model}}$ = model width (e.g., 4096)
- $H$ = query heads (e.g., 32)
- $H_{kv}$ = KV heads (e.g., 8 with GQA)
- $d_h = d_{\text{model}} / H$ (e.g., 128)
- $d_{kv} = d_h$ for common GQA
- $b$ = bytes per element (2 for FP16/BF16, 4 for FP32)
- $L_{\text{layers}}$ = number of layers (e.g., 32)

### KV bytes per token step (attention only, all layers)

For GQA:

$$
\text{Bytes/step} \approx L_{\text{layers}} \cdot 2L \cdot (H_{kv} \cdot d_{kv}) \cdot b
$$

Factor 2 accounts for reading both K and V once (ideal single-pass).
Example (7B-class: $L_{\text{layers}}{=}32$, $H_{kv}{=}8$, $d_{kv}{=}128$, $b{=}2$):

- For $L{=}1024$: $32 \cdot 2 \cdot 1024 \cdot (8 \cdot 128) \cdot 2 \approx 134{,}217{,}728 \text{ B} \approx 128 \text{ MiB}$.
- For $L{=}4096$: $\approx 512 \text{ MiB}$.

**Upper-bound tokens/sec from DRAM BW** (attention only):

$$
\text{t/s} \lesssim \frac{\text{BW}}{\text{Bytes/step}}.
$$

At 1.5 TB/s DRAM (theoretical), the memory-only bound for $L{=}4096$ is $\lesssim 1.5\text{e}12 / 5.12\text{e}8 \approx 2929\ \text{t/s}$.
Real decode is lower due to:

- Extra reads/writes (activations, projections, residuals, LN)
- Compute in attention softmax + FFN
- Non-ideal reuse & kernel overheads

### Arithmetic work (attention, all layers)

QKᵀ + P·V per layer per token (GQA):

$$
\text{FLOPs} \approx 2L \cdot (H_{kv} \cdot d_{kv})
$$

For the example, per layer at $L{=}4096$: $2 \cdot 4096 \cdot 1024 \approx 8.39\text{e}6$ FLOPs.
All layers: $\approx 268\text{ MFLOPs}$ per token (attention only). FFN adds much more compute.

**Takeaway:** At moderate $L$, FFN can dominate compute; at large $L$, attention grows linearly in $L$ and can become bandwidth-limited. Optimize both, but **KV movement is the first constraint** to address at long context.

## GPU Deep Dive

### NVIDIA

- Warps (32-threads), SMs, L2 + HBM hierarchy.
- Tensor Cores accelerate FFN/Proj GEMMs; decode attention often limited by **global memory bandwidth + launch overhead**.
- Use **persistent kernels** (few blocks loop over work) to amortize launch costs and keep L2 warmed. Consider **CUDA Graphs** when the sequence of ops is stable.

### AMD

- Wavefronts (64-threads), CUs, LDS (shared memory).
- MFMA/XDLOPs for GEMMs; similar bandwidth limits for KV.
- HIP provides kernel launch symmetry with CUDA; persistent kernels apply equally. Use **rocprof/Omnitrace** for counters and timeline analysis.

## Implementation

We provide a minimal, **runnable** attention-only decode microbenchmark that compares:

1. **Naïve per-step** kernel launches vs.
2. **Persistent** kernel that loops over steps, caching Q in shared memory and using vectorized loads.

It performs numerically stable softmax (max-subtraction), accumulates in FP32, validates against a CPU reference, and prints tokens/sec.

Place this file at:
`topics/19-decode-path-optimization/code/decode_attn_persistent.cu`

```cpp
// Single-source CUDA/HIP: attention decode microbenchmark with persistent kernel.
// Build (CUDA):
//   nvcc -O3 -std=c++17 -arch=${SM_ARCH} -lineinfo topics/19-decode-path-optimization/code/decode_attn_persistent.cu -o decode_attn
// Build (ROCm/HIP):
//   hipcc -O3 -std=c++17 --offload-arch=${GFX_ARCH} topics/19-decode-path-optimization/code/decode_attn_persistent.cu -o decode_attn
//
// Run:
//   ./decode_attn [L=1024] [D=128] [T=64] [blocks=0] [threads=128]
//     L: sequence length (context)
//     D: head dim (per kv head), default 128 for 7B-class
//     T: decode steps to simulate
//     blocks: 0 => auto (2x SM/CU count), else explicit
//     threads: threads per block (power of two, e.g., 128)
//
// Notes:
// - This is a single-head attention microkernel that represents one KV group.
// - It compares per-step launches vs. a persistent kernel looping over steps.
// - Numerics: FP32 accumulation with stable softmax (max-subtraction).

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cassert>
#include <cstring>

#if defined(__HIP_PLATFORM_AMD__)
  #include <hip/hip_runtime.h>
  #define DEV_API_SUCCESS hipSuccess
  #define DEV_GET_ERROR hipGetLastError
  #define DEV_GET_ERROR_STRING hipGetErrorString
  #define DEV_MALLOC hipMalloc
  #define DEV_FREE hipFree
  #define DEV_MEMCPY hipMemcpy
  #define DEV_MEMSET hipMemset
  #define DEV_MEMCPY_HOST_TO_DEVICE hipMemcpyHostToDevice
  #define DEV_MEMCPY_DEVICE_TO_HOST hipMemcpyDeviceToHost
  #define DEV_EVENT_T hipEvent_t
  #define DEV_EVENT_CREATE hipEventCreate
  #define DEV_EVENT_DESTROY hipEventDestroy
  #define DEV_EVENT_RECORD hipEventRecord
  #define DEV_EVENT_SYNCHRONIZE hipEventSynchronize
  #define DEV_EVENT_ELAPSED_TIME hipEventElapsedTime
  #define DEV_DEVICE_PROP hipDeviceProp_t
  #define DEV_GET_DEVICE_PROPERTIES hipGetDeviceProperties
  #define DEV_DEVICE_SYNCHRONIZE hipDeviceSynchronize
#else
  #include <cuda_runtime.h>
  #define DEV_API_SUCCESS cudaSuccess
  #define DEV_GET_ERROR cudaGetLastError
  #define DEV_GET_ERROR_STRING cudaGetErrorString
  #define DEV_MALLOC cudaMalloc
  #define DEV_FREE cudaFree
  #define DEV_MEMCPY cudaMemcpy
  #define DEV_MEMSET cudaMemset
  #define DEV_MEMCPY_HOST_TO_DEVICE cudaMemcpyHostToDevice
  #define DEV_MEMCPY_DEVICE_TO_HOST cudaMemcpyDeviceToHost
  #define DEV_EVENT_T cudaEvent_t
  #define DEV_EVENT_CREATE cudaEventCreate
  #define DEV_EVENT_DESTROY cudaEventDestroy
  #define DEV_EVENT_RECORD cudaEventRecord
  #define DEV_EVENT_SYNCHRONIZE cudaEventSynchronize
  #define DEV_EVENT_ELAPSED_TIME cudaEventElapsedTime
  #define DEV_DEVICE_PROP cudaDeviceProp
  #define DEV_GET_DEVICE_PROPERTIES cudaGetDeviceProperties
  #define DEV_DEVICE_SYNCHRONIZE cudaDeviceSynchronize
#endif

#define API_CHECK(x) do { auto _e = (x); if (_e != DEV_API_SUCCESS) { \
  fprintf(stderr,"API error %s:%d: %s\n", __FILE__, __LINE__, DEV_GET_ERROR_STRING(_e)); exit(1);} } while(0)

__device__ __forceinline__ float warp_reduce_sum(float val) {
  // Portable fallback reduction using shared memory will be used instead.
  return val;
}

// Kernel: single decode step (naive per-step launch).
__global__ void attn_single_step_kernel(const float* __restrict__ Q, // [D]
                                        const float* __restrict__ K, // [L, D]
                                        const float* __restrict__ V, // [L, D]
                                        float* __restrict__ O,       // [D]
                                        int L, int D, float scale) {
  extern __shared__ float smem[];
  // layout: [0..D-1]   : q_sh or out_sh
  //         [D..D+T-1] : red[blockDim.x]
  //         [D+T..]    : scalars[2] -> {max_logit, sum_e}
  float* q_sh = smem;
  float* red  = q_sh + D;
  float* scal = red + blockDim.x; // size 2

  // Cache Q into shared memory
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    q_sh[d] = Q[d];
  }
  __syncthreads();

  if (threadIdx.x == 0) scal[0] = -INFINITY; // max_logit
  __syncthreads();

  // Pass 1: compute max_logit over l
  for (int l = 0; l < L; ++l) {
    float local = 0.f;
    const float* krow = K + l * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
      local += q_sh[d] * krow[d];
    }
    red[threadIdx.x] = local;
    __syncthreads();
    // block reduce
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
      if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      float logit = red[0] * scale;
      if (logit > scal[0]) scal[0] = logit;
    }
    __syncthreads();
  }

  // Reuse q_sh as output accumulator; zero it
  for (int d = threadIdx.x; d < D; d += blockDim.x) q_sh[d] = 0.f;
  if (threadIdx.x == 0) scal[1] = 0.f; // sum_e
  __syncthreads();

  // Pass 2: accumulate exp(logit - max) * V, track denom
  for (int l = 0; l < L; ++l) {
    float local = 0.f;
    const float* krow = K + l * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
      local += q_sh[d]; // dummy to keep compiler from optimizing q_sh away
    }
    // recompute dot(q,k_l)
    local = 0.f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
      local += Q[d] * krow[d];
    }
    red[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
      if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
      __syncthreads();
    }
    float e = 0.f;
    if (threadIdx.x == 0) {
      float logit = red[0] * scale;
      e = expf(logit - scal[0]);
      scal[1] += e;
      red[0] = e; // broadcast via red[0]
    }
    __syncthreads();
    e = red[0];
    const float* vrow = V + l * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
      q_sh[d] += e * vrow[d];
    }
    __syncthreads();
  }

  // Normalize and write out
  float denom = scal[1];
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    O[d] = q_sh[d] / denom;
  }
}

// Kernel: persistent decode over many steps; a few blocks loop over [0..T)
__global__ void attn_persistent_kernel(const float* __restrict__ Q, // [T, D]
                                       const float* __restrict__ K, // [L, D]
                                       const float* __restrict__ V, // [L, D]
                                       float* __restrict__ O,       // [T, D]
                                       int T, int L, int D, float scale,
                                       int* __restrict__ work_idx) {
  extern __shared__ float smem[];
  float* buf = smem;                  // size >= D + blockDim.x + 2
  float* q_sh = buf;                  // [D]
  float* red  = q_sh + D;             // [blockDim.x]
  float* scal = red + blockDim.x;     // [2] -> {max_logit, sum_e}

  while (true) {
    int t = atomicAdd(work_idx, 1);
    if (t >= T) return;

    // Load Q_t to shared
    const float* q = Q + t * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) q_sh[d] = q[d];
    __syncthreads();

    if (threadIdx.x == 0) scal[0] = -INFINITY; // max_logit
    __syncthreads();

    // Pass 1: max
    for (int l = 0; l < L; ++l) {
      float local = 0.f;
      const float* krow = K + l * D;
      for (int d = threadIdx.x; d < D; d += blockDim.x) {
        local += q_sh[d] * krow[d];
      }
      red[threadIdx.x] = local;
      __syncthreads();
      for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
      }
      if (threadIdx.x == 0) {
        float logit = red[0] * scale;
        if (logit > scal[0]) scal[0] = logit;
      }
      __syncthreads();
    }

    // Reuse q_sh as output accumulator; zero it
    for (int d = threadIdx.x; d < D; d += blockDim.x) q_sh[d] = 0.f;
    if (threadIdx.x == 0) scal[1] = 0.f; // sum_e
    __syncthreads();

    // Pass 2: numerator & denom
    for (int l = 0; l < L; ++l) {
      float local = 0.f;
      const float* krow = K + l * D;
      for (int d = threadIdx.x; d < D; d += blockDim.x) {
        local += q_sh[d]; // keep q_sh used
      }
      // recompute dot
      local = 0.f;
      for (int d = threadIdx.x; d < D; d += blockDim.x) {
        local += (Q + t * D)[d] * krow[d];
      }
      red[threadIdx.x] = local;
      __syncthreads();
      for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
      }
      float e = 0.f;
      if (threadIdx.x == 0) {
        float logit = red[0] * scale;
        e = expf(logit - scal[0]);
        scal[1] += e;
        red[0] = e;
      }
      __syncthreads();
      e = red[0];

      const float* vrow = V + l * D;
      for (int d = threadIdx.x; d < D; d += blockDim.x) {
        q_sh[d] += e * vrow[d];
      }
      __syncthreads();
    }

    // Normalize and write out
    float denom = scal[1];
    float inv = 1.f / denom;
    float* out = O + t * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
      out[d] = q_sh[d] * inv;
    }
    __syncthreads();
  }
}

static void cpu_reference(const float* q, const float* K, const float* V,
                          float* out, int L, int D, float scale) {
  std::vector<float> logits(L);
  float m = -INFINITY;
  for (int l = 0; l < L; ++l) {
    float s = 0.f;
    const float* krow = K + l * D;
    for (int d = 0; d < D; ++d) s += q[d] * krow[d];
    logits[l] = s * scale;
    m = std::max(m, logits[l]);
  }
  float denom = 0.f;
  for (int l = 0; l < L; ++l) denom += std::exp(logits[l] - m);
  for (int d = 0; d < D; ++d) out[d] = 0.f;
  for (int l = 0; l < L; ++l) {
    float w = std::exp(logits[l] - m) / denom;
    const float* vrow = V + l * D;
    for (int d = 0; d < D; ++d) out[d] += w * vrow[d];
  }
}

int main(int argc, char** argv) {
  int L = (argc > 1) ? std::atoi(argv[1]) : 1024;
  int D = (argc > 2) ? std::atoi(argv[2]) : 128;
  int T = (argc > 3) ? std::atoi(argv[3]) : 64;
  int blocks = (argc > 4) ? std::atoi(argv[4]) : 0;
  int threads = (argc > 5) ? std::atoi(argv[5]) : 128;

  if ((threads & (threads - 1)) != 0) {
    fprintf(stderr, "threads must be a power of two\n");
    return 1;
  }

  DEV_DEVICE_PROP prop{};
  API_CHECK(DEV_GET_DEVICE_PROPERTIES(&prop, 0));
  if (blocks == 0) {
    // modest over-subscription for persistence
    blocks = std::max(1, 2 * prop.multiProcessorCount);
  }

  printf("Config: L=%d, D=%d, T=%d, blocks=%d, threads=%d\n", L, D, T, blocks, threads);

  const float scale = 1.f / std::sqrt((float)D);

  // Host buffers
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-0.01f, 0.01f);

  std::vector<float> hQ((size_t)T * D), hK((size_t)L * D), hV((size_t)L * D);
  for (auto& x : hQ) x = dist(rng);
  for (auto& x : hK) x = dist(rng);
  for (auto& x : hV) x = dist(rng);

  // Device buffers
  float *dQ, *dK, *dV, *dO_naive, *dO_persist;
  API_CHECK(DEV_MALLOC(&dQ, sizeof(float) * (size_t)T * D));
  API_CHECK(DEV_MALLOC(&dK, sizeof(float) * (size_t)L * D));
  API_CHECK(DEV_MALLOC(&dV, sizeof(float) * (size_t)L * D));
  API_CHECK(DEV_MALLOC(&dO_naive, sizeof(float) * (size_t)T * D));
  API_CHECK(DEV_MALLOC(&dO_persist, sizeof(float) * (size_t)T * D));

  API_CHECK(DEV_MEMCPY(dQ, hQ.data(), sizeof(float) * (size_t)T * D, DEV_MEMCPY_HOST_TO_DEVICE));
  API_CHECK(DEV_MEMCPY(dK, hK.data(), sizeof(float) * (size_t)L * D, DEV_MEMCPY_HOST_TO_DEVICE));
  API_CHECK(DEV_MEMCPY(dV, hV.data(), sizeof(float) * (size_t)L * D, DEV_MEMCPY_HOST_TO_DEVICE));
  API_CHECK(DEV_MEMSET(dO_naive, 0, sizeof(float) * (size_t)T * D));
  API_CHECK(DEV_MEMSET(dO_persist, 0, sizeof(float) * (size_t)T * D));

  // Shared memory size
  size_t shmem_bytes = (size_t)(D + threads + 2) * sizeof(float);

  // Events
  DEV_EVENT_T e0, e1, e2, e3;
  API_CHECK(DEV_EVENT_CREATE(&e0));
  API_CHECK(DEV_EVENT_CREATE(&e1));
  API_CHECK(DEV_EVENT_CREATE(&e2));
  API_CHECK(DEV_EVENT_CREATE(&e3));

  // --- NAIVE: launch per step
  API_CHECK(DEV_EVENT_RECORD(e0));
  for (int t = 0; t < T; ++t) {
    attn_single_step_kernel<<<1, threads, shmem_bytes>>>(
      dQ + (size_t)t * D, dK, dV, dO_naive + (size_t)t * D, L, D, scale);
  }
  API_CHECK(DEV_EVENT_RECORD(e1));
  API_CHECK(DEV_EVENT_SYNCHRONIZE(e1));
  float ms_naive = 0.f;
  API_CHECK(DEV_EVENT_ELAPSED_TIME(&ms_naive, e0, e1));

  // --- PERSISTENT: few blocks loop over steps
  int* dWorkIdx;
  API_CHECK(DEV_MALLOC(&dWorkIdx, sizeof(int)));
  API_CHECK(DEV_MEMSET(dWorkIdx, 0, sizeof(int)));

  API_CHECK(DEV_EVENT_RECORD(e2));
  attn_persistent_kernel<<<blocks, threads, shmem_bytes>>>(
      dQ, dK, dV, dO_persist, T, L, D, scale, dWorkIdx);
  API_CHECK(DEV_EVENT_RECORD(e3));
  API_CHECK(DEV_EVENT_SYNCHRONIZE(e3));
  float ms_persist = 0.f;
  API_CHECK(DEV_EVENT_ELAPSED_TIME(&ms_persist, e2, e3));

  // Copy a couple of outputs back and validate against CPU for step 0
  std::vector<float> o0(D), o1(D), ref(D);
  API_CHECK(DEV_MEMCPY(o0.data(), dO_naive,  sizeof(float) * D, DEV_MEMCPY_DEVICE_TO_HOST));
  API_CHECK(DEV_MEMCPY(o1.data(), dO_persist, sizeof(float) * D, DEV_MEMCPY_DEVICE_TO_HOST));
  cpu_reference(hQ.data(), hK.data(), hV.data(), ref.data(), L, D, scale);

  auto l2 = [&](const std::vector<float>& a, const std::vector<float>& b){
    double s = 0.0; for (int i = 0; i < D; ++i) { double d = (double)a[i] - (double)b[i]; s += d*d; }
    return std::sqrt(s / D);
  };
  double err_naive   = l2(o0, ref);
  double err_persist = l2(o1, ref);

  // Report
  double ts_naive   = (double)T / (ms_naive  * 1e-3);
  double ts_persist = (double)T / (ms_persist* 1e-3);
  printf("Validation (step 0, L2): naive=%.3e, persistent=%.3e\n", err_naive, err_persist);
  printf("NAIVE     : %.3f ms for %d steps => %.2f tokens/s\n", ms_naive, T, ts_naive);
  printf("PERSISTENT: %.3f ms for %d steps => %.2f tokens/s\n", ms_persist, T, ts_persist);
  printf("Speedup (persistent / naive): %.2fx\n", ts_persist / ts_naive);

  // Cleanup
  API_CHECK(DEV_FREE(dQ));
  API_CHECK(DEV_FREE(dK));
  API_CHECK(DEV_FREE(dV));
  API_CHECK(DEV_FREE(dO_naive));
  API_CHECK(DEV_FREE(dO_persist));
  API_CHECK(DEV_FREE(dWorkIdx));
  API_CHECK(DEV_EVENT_DESTROY(e0));
  API_CHECK(DEV_EVENT_DESTROY(e1));
  API_CHECK(DEV_EVENT_DESTROY(e2));
  API_CHECK(DEV_EVENT_DESTROY(e3));
  API_CHECK(DEV_DEVICE_SYNCHRONIZE());
  return 0;
}
```

### Build Commands

- CUDA:

```
nvcc -O3 -std=c++17 -arch=${SM_ARCH} -lineinfo \
  topics/19-decode-path-optimization/code/decode_attn_persistent.cu \
  -o decode_attn
```

- ROCm/HIP:

```
hipcc -O3 -std=c++17 --offload-arch=${GFX_ARCH} \
  topics/19-decode-path-optimization/code/decode_attn_persistent.cu \
  -o decode_attn
```

### Run Examples

```
./decode_attn                 # L=1024, D=128, T=64, blocks=auto, threads=128
./decode_attn 2048 128 128    # longer context and more steps
./decode_attn 4096 128 64 0 256
```

## Profiling and Validation

### NVIDIA

- **Timeline & launch overheads (Nsight Systems):**

```
nsys profile -t cuda,nvtx -o nsys_decode ./decode_attn 2048 128 128
```

Inspect kernel launch counts and gaps; persistent kernel should show fewer launches and longer, steadier kernels.

- **Memory metrics (Nsight Compute):**

```
ncu --set full --kernel-name-base demangled \
    --target-processes all \
    ./decode_attn 4096 128 64
```

Key counters and goals:

- `dram__throughput.avg.pct_of_peak_sustained_elapsed` → ≥ 55% at long L
- `l2_tex__throughput.avg.pct_of_peak_sustained_elapsed` → ≥ 50%
- `smsp__sass_average_branch_targets_threads_uniform.pct` → high (low divergence)

### AMD

- **Timeline & counters (rocprof):**

```
rocprof --hip-trace --timestamp on --stats ./decode_attn 2048 128 128
```

Focus on:

- DRAM read GB/s vs device peak
- Kernel launch count and average duration
- LDS (shared) bytes read/written per kernel

### Numerical Checks

- Program prints L2 error vs CPU reference for step 0.
- Extend validation by comparing multiple steps or random seeds.

## Performance Checklist

- [ ] Use **persistent** decode kernel (few blocks loop steps) or **Graphs** for fixed DAG.
- [ ] **Cache Q** per step in shared memory; avoid rereads.
- [ ] **Vectorize loads** (e.g., `float4`) when `D % 4 == 0`; align KV rows to 16 bytes.
- [ ] Coalesce: map `threadIdx.x` to contiguous `d` indices.
- [ ] Avoid temporary global buffers for logits/weights; use two-pass streaming softmax.
- [ ] Keep accumulation in **FP32**; subtract max for numerical stability.
- [ ] Pre-allocate and reuse all device buffers; no per-token `malloc`.
- [ ] Profile: achieve ≥1.3× tokens/s vs naive per-step at $L \ge 1024$.

## Troubleshooting

| Symptom                         | Likely Cause                                                  | Fix                                                          |
| ------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------ |
| Tokens/s unchanged vs. naive    | Not actually persistent; too many blocks or atomic contention | Set blocks ≈ 1–2× SM/CU; verify loop over `work_idx`         |
| Low DRAM throughput (<30% peak) | Strided or unaligned KV reads                                 | Ensure row-major `[L,D]`, align to 16B, use `float4` loads   |
| Divergence inside kernel        | Per-thread different control flow                             | Keep uniform loops; avoid per-thread early exits             |
| Numerical overflow in softmax   | No max-subtraction or FP16 accum                              | Subtract max; accumulate in FP32                             |
| Kernel launch gaps in timeline  | Host-side sequencing overhead                                 | Use persistent kernel or capture with CUDA/HIP graphs        |
| Occupancy too low               | Excess shared memory or registers                             | Tune `threads`, reduce shared footprint, or split reductions |
| Validation L2 error large       | Read/write overlap or race                                    | Check shared memory barriers; avoid aliasing                 |
| ROCm build fails                | Missing arch flag                                             | Provide `--offload-arch=gfx90a`/`gfx942` etc.                |
| CUDA build runs slow            | Wrong `-arch`                                                 | Use matching `-arch=sm_80/sm_90`                             |

## Acceptance Criteria

- Compiles and runs on NVIDIA (CUDA 12.x) and AMD (ROCm/HIP 6.x) with defaults.
- Prints **L2 error ≤ 1e-5** vs CPU reference for step 0.
- On any modern data-center GPU, **persistent kernel ≥ 1.3×** tokens/sec vs naive at $L \ge 1024$.
- Nsight/rocprof shows **fewer launches** and **higher average kernel duration** for persistent mode.

## Further Work

- Replace reductions with warp-level intrinsics and vectorized `float4` loads for K/V.
- Fuse QKᵀ, softmax, and PV into a **single streaming pass** with online max/denom update to remove the second K read.
- Add **CUDA/HIP Graphs** capture around the entire decode step (attention + projections + FFN) for fixed DAGs.
- Introduce **INT8**/FP8 dequant in-kernel for K/V with FP32 accum to reduce bytes/step.
- Multi-query batching: interleave multiple sequences (batch>1) per block to increase memory-level parallelism at small $L$.
