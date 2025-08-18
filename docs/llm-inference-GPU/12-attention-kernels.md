# Attention Kernels & Memory-Efficient Variants

## Summary

Attention dominates both compute and memory traffic in LLM inference. Naïve scaled dot-product attention (SDPA) materializes the N×N score matrix, stressing HBM and L2. Memory-efficient variants (e.g., “FlashAttention-style” streaming softmax) avoid materialization and maximize on-chip reuse of K/V tiles. This note explains the math, maps it to NVIDIA/AMD hardware, and provides runnable CUDA and HIP kernels with a lightweight benchmark and validation against a CPU reference.

## Why It Matters for LLM Inference

- **Prefill**: Long sequence (large N) with wide tiles—attention is bandwidth-bound unless we reuse K/V aggressively.
- **Decode**: Short effective N per step but tight latency; persistent/graphs help, and streaming softmax removes large intermediates and kernel launches.

## Key Concepts and Formulas

Let sequence length $N$, head dimension $d$, heads $H$. For a single head:

- Scores: $S = \frac{1}{\sqrt{d}} QK^{\top}$ with $Q,K\in \mathbb{R}^{N\times d}$.
- Softmax (row-wise): $P_{i,:} = \mathrm{softmax}(S_{i,:})$.
- Output: $O = P V$, $V\in \mathbb{R}^{N\times d}$.

### Numerical stability (log-sum-exp, LSE)

Online softmax keeps per-row running maximum $m$ and normalization $l$:

- For tile $t$ with scores $s^{(t)}_j$,
  $m'=\max(m,\max_j s^{(t)}_j)$,
  $l' = e^{m-m'}l + \sum_j e^{s^{(t)}_j - m'}$,
  $\mathrm{acc}' = e^{m-m'}\mathrm{acc} + \sum_j e^{s^{(t)}_j - m'} V_j$.
  After all tiles: $O_i = \mathrm{acc}/l$.

### Memory traffic (per head, naïve vs. tiled)

- Naïve materialization (SDPA):
  Read $Q,K,V$: $3Nd$ elems; write $O$: $Nd$; **plus** materialize $S$: $N^2$ elems and read it again → prohibitive for large $N$.
- Tiled streaming (tile sizes $M\times K$):
  For each $K$-tile, reuse across $M$ queries in an SM/CU: K/V tile read once from HBM, reused $M$ times in on-chip memory. Approx traffic $\approx Nd$ (Q) + $\lceil N/K \rceil \cdot Kd$ (K/V once per tile) + $Nd$ (O), ignoring write-alloc and metadata.

### Arithmetic intensity (AI)

Per tile (per row): dot-products cost $K\cdot d$ MACs; reads \~$K\cdot d$ for K and $K\cdot d$ for V, plus $d$ for Q (amortized across tiles).
AI improves with reuse of K/V across $M$ rows: effective bytes per MAC drop by $\approx M$.

## GPU Deep Dive

### NVIDIA (SMs, Warps, Tensor Cores)

- **Warps** of 32 threads; schedule favors coherent, 128-byte aligned, vectorized loads (e.g., `float4`, `__half2`).
- **Shared memory/L1** (\~100 KB/SM on Ada/Hopper) is critical for K/V tiles.
- **Tensor Cores** (HMMA/BF16/FP16/FP8) accelerate dot products; our minimal kernel uses FP32 accumulations for clarity, but production variants use MMA fragments + fused epilogues.

### AMD (CUs, Wavefronts, MFMA/XDLOPs, LDS)

- **Wavefronts** of 64 threads; coalescing aligns to 128B/256B segments.
- **LDS** (Local Data Share) provides high-bandwidth on-chip storage; mirror CUDA shared memory use.
- **MFMA/XDLOPs** matrix ops enable high-throughput GEMMs; here we demonstrate scalar FMA for portability with FP32 accumulations.

## Implementation

We provide two GPU kernels per backend:

1. **Naïve row-wise streaming**: each thread computes one query row, reading K/V directly from global (no on-chip K/V reuse).
2. **Memory-efficient tiled (Flash-style)**: blocks cooperatively stage K/V tiles into on-chip memory (Shared/LDS) and apply online softmax.

Both support FP16 (default) and optional BF16 (`-DUSE_BF16`).

### Build Targets

- CUDA 12.x, C++17
  `nvcc -O3 -std=c++17 -arch=${SM_ARCH} -lineinfo topics/12-attention-kernels/code/flash_attn_cuda.cu -o flash_attn_cuda`
- ROCm/HIP 6.x, C++17
  `hipcc -O3 -std=c++17 --offload-arch=${GFX_ARCH} topics/12-attention-kernels/code/flash_attn_hip.cpp -o flash_attn_hip`

Run: `./flash_attn_cuda` or `./flash_attn_hip`

### CUDA: Minimal, Runnable Kernels + Harness

`topics/12-attention-kernels/code/flash_attn_cuda.cu`

```cpp
// Minimal SDPA kernels: CUDA 12.x
// Builds: nvcc -O3 -std=c++17 -arch=${SM_ARCH} -lineinfo flash_attn_cuda.cu -o flash_attn_cuda
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#ifdef USE_BF16
#include <cuda_bf16.h>
#endif
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>

#ifndef BLOCK_M
#define BLOCK_M 64   // queries per block
#endif
#ifndef BLOCK_K
#define BLOCK_K 64   // keys per tile
#endif
#ifndef D_HEAD
#define D_HEAD 64    // head dimension
#endif

#if defined(USE_BF16)
using htype = __nv_bfloat16;
__device__ __forceinline__ float to_float(htype x){ return __bfloat162float(x); }
__device__ __forceinline__ htype to_htype(float x){ return __float2bfloat16(x); }
#else
using htype = __half;
__device__ __forceinline__ float to_float(htype x){ return __half2float(x); }
__device__ __forceinline__ htype to_htype(float x){ return __float2half(x); }
#endif

#define API_CHECK(x) do { auto e = (x); if (e != cudaSuccess){ \
  printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); abort();}} while(0)

__global__ void attn_naive_rowwise(
    const htype* __restrict__ Q,
    const htype* __restrict__ K,
    const htype* __restrict__ V,
    htype* __restrict__ O,
    int N, int d, float scale)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x; // query row
  if (i >= N) return;
  // online softmax state
  float m = -INFINITY;
  float l = 0.f;
  float acc[D_HEAD]; // assume d <= D_HEAD compile-time cap
#pragma unroll
  for (int t=0;t<D_HEAD;++t) acc[t]=0.f;

  // Pointer to Q_i
  const htype* q = Q + i * d;

  // First pass over keys in tiles to improve locality (still from global)
  for (int base=0; base<N; base+=BLOCK_K){
    int kTile = min(BLOCK_K, N - base);

    // compute tile max
    float tile_max = -INFINITY;
    for (int j=0; j<kTile; ++j){
      const htype* kj = K + (base + j)*d;
      float dot = 0.f;
#pragma unroll
      for (int t=0;t<D_HEAD;++t){
        if (t<d) dot += to_float(q[t]) * to_float(kj[t]);
      }
      tile_max = fmaxf(tile_max, dot * scale);
    }
    float m_new = fmaxf(m, tile_max);
    float alpha = expf(m - m_new);
    // accumulate with normalized exps
    float l_new = alpha * l;
#pragma unroll
    for (int t=0;t<D_HEAD;++t) if (t<d) acc[t] *= alpha;

    for (int j=0; j<kTile; ++j){
      const htype* kj = K + (base + j)*d;
      const htype* vj = V + (base + j)*d;
      float dot = 0.f;
#pragma unroll
      for (int t=0;t<D_HEAD;++t){
        if (t<d) dot += to_float(q[t]) * to_float(kj[t]);
      }
      float p = expf(dot * scale - m_new);
      l_new += p;
#pragma unroll
      for (int t=0;t<D_HEAD;++t){
        if (t<d) acc[t] += p * to_float(vj[t]);
      }
    }
    m = m_new; l = l_new;
  }

  htype* o = O + i*d;
#pragma unroll
  for (int t=0;t<D_HEAD;++t){
    if (t<d) o[t] = to_htype(acc[t] / l);
  }
}

extern __shared__ htype smem[]; // K and V tiles back-to-back

__global__ void attn_tiled_flash(
    const htype* __restrict__ Q,
    const htype* __restrict__ K,
    const htype* __restrict__ V,
    htype* __restrict__ O,
    int N, int d, float scale)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x; // query row per thread
  if (i >= N) return;

  htype* smemK = smem;
  htype* smemV = smem + BLOCK_K * d;

  float m = -INFINITY, l = 0.f;
  float acc[D_HEAD]; // FP32 accum
#pragma unroll
  for (int t=0;t<D_HEAD;++t) acc[t]=0.f;

  const htype* q = Q + i*d;

  for (int base=0; base<N; base+=BLOCK_K){
    const int kTile = min(BLOCK_K, N - base);
    // Cooperative load of K/V tiles into shared
    int tLinear = threadIdx.x;
    int tile_elems = kTile * d;
    for (int idx = tLinear; idx < tile_elems; idx += blockDim.x){
      smemK[idx] = K[(base * d) + idx];
      smemV[idx] = V[(base * d) + idx];
    }
    __syncthreads();

    // compute max on this tile
    float tile_max = -INFINITY;
    for (int j=0;j<kTile;++j){
      const htype* kj = smemK + j*d;
      float dot = 0.f;
#pragma unroll
      for (int t=0;t<D_HEAD;++t){
        if (t<d) dot += to_float(q[t]) * to_float(kj[t]);
      }
      tile_max = fmaxf(tile_max, dot * scale);
    }
    float m_new = fmaxf(m, tile_max);
    float alpha = expf(m - m_new);

#pragma unroll
    for (int t=0;t<D_HEAD;++t) if (t<d) acc[t] *= alpha;
    float l_new = alpha * l;

    for (int j=0;j<kTile;++j){
      const htype* kj = smemK + j*d;
      const htype* vj = smemV + j*d;
      float dot = 0.f;
#pragma unroll
      for (int t=0;t<D_HEAD;++t){
        if (t<d) dot += to_float(q[t]) * to_float(kj[t]);
      }
      float p = expf(dot * scale - m_new);
      l_new += p;
#pragma unroll
      for (int t=0;t<D_HEAD;++t){
        if (t<d) acc[t] += p * to_float(vj[t]);
      }
    }
    m = m_new; l = l_new;
    __syncthreads();
  }

  htype* o = O + i*d;
#pragma unroll
  for (int t=0;t<D_HEAD;++t){
    if (t<d) o[t] = to_htype(acc[t] / l);
  }
}

// ---------------- Harness: init, run, check ----------------
static void cpu_reference(const std::vector<float>& Qf,
                          const std::vector<float>& Kf,
                          const std::vector<float>& Vf,
                          std::vector<float>& Of,
                          int N, int d, float scale, int rows_check)
{
  for (int i=0;i<rows_check;++i){
    // max
    float m = -INFINITY;
    for (int j=0;j<N;++j){
      float dot=0.f;
      for (int t=0;t<d;++t) dot += Qf[i*d+t]*Kf[j*d+t];
      m = std::max(m, dot*scale);
    }
    // sum and acc
    float l=0.f;
    std::vector<float> acc(d, 0.f);
    for (int j=0;j<N;++j){
      float dot=0.f;
      for (int t=0;t<d;++t) dot += Qf[i*d+t]*Kf[j*d+t];
      float p=std::exp(dot*scale - m);
      l += p;
      for (int t=0;t<d;++t) acc[t]+= p*Vf[j*d+t];
    }
    for (int t=0;t<d;++t) Of[i*d+t] = acc[t]/l;
  }
}

int main(){
  const int N = 1024;       // sequence length
  const int d = D_HEAD;     // head dim (must <= D_HEAD)
  const float scale = 1.0f / std::sqrt((float)d);

  size_t bytes = (size_t)N*d*sizeof(htype);
  htype *dQ,*dK,*dV,*dO;
  API_CHECK(cudaMalloc(&dQ, bytes));
  API_CHECK(cudaMalloc(&dK, bytes));
  API_CHECK(cudaMalloc(&dV, bytes));
  API_CHECK(cudaMalloc(&dO, bytes));

  // Host init (FP32 then cast)
  std::mt19937 rng(0);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  std::vector<float> Qf(N*d), Kf(N*d), Vf(N*d);
  for (int i=0;i<N*d;++i){ Qf[i]=dist(rng); Kf[i]=dist(rng); Vf[i]=dist(rng); }
  std::vector<htype> Qh(N*d), Kh(N*d), Vh(N*d);
  for (int i=0;i<N*d;++i){
#if defined(USE_BF16)
    Qh[i]=__float2bfloat16(Qf[i]);
    Kh[i]=__float2bfloat16(Kf[i]);
    Vh[i]=__float2bfloat16(Vf[i]);
#else
    Qh[i]=__float2half(Qf[i]);
    Kh[i]=__float2half(Kf[i]);
    Vh[i]=__float2half(Vf[i]);
#endif
  }
  API_CHECK(cudaMemcpy(dQ, Qh.data(), bytes, cudaMemcpyHostToDevice));
  API_CHECK(cudaMemcpy(dK, Kh.data(), bytes, cudaMemcpyHostToDevice));
  API_CHECK(cudaMemcpy(dV, Vh.data(), bytes, cudaMemcpyHostToDevice));

  dim3 block(BLOCK_M);
  dim3 grid((N + BLOCK_M - 1)/BLOCK_M);
  size_t smem_bytes = (size_t)BLOCK_K * d * sizeof(htype) * 2;

  // Warmup
  attn_tiled_flash<<<grid,block,smem_bytes>>>(dQ,dK,dV,dO,N,d,scale);
  API_CHECK(cudaDeviceSynchronize());

  // Time both kernels
  cudaEvent_t s,e; API_CHECK(cudaEventCreate(&s)); API_CHECK(cudaEventCreate(&e));
  float ms_naive=0.f, ms_tiled=0.f;

  API_CHECK(cudaEventRecord(s));
  attn_naive_rowwise<<<grid,block>>>(dQ,dK,dV,dO,N,d,scale);
  API_CHECK(cudaEventRecord(e));
  API_CHECK(cudaEventSynchronize(e));
  API_CHECK(cudaEventElapsedTime(&ms_naive,s,e));

  API_CHECK(cudaEventRecord(s));
  attn_tiled_flash<<<grid,block,smem_bytes>>>(dQ,dK,dV,dO,N,d,scale);
  API_CHECK(cudaEventRecord(e));
  API_CHECK(cudaEventSynchronize(e));
  API_CHECK(cudaEventElapsedTime(&ms_tiled,s,e));

  // Copy output
  std::vector<htype> Oh(N*d);
  API_CHECK(cudaMemcpy(Oh.data(), dO, bytes, cudaMemcpyDeviceToHost));

  // Verify first 128 rows vs CPU reference
  std::vector<float> Oref(N*d, 0.f);
  cpu_reference(Qf,Kf,Vf,Oref,N,d,scale,/*rows_check=*/128);

  // Compute max abs error on checked rows
  double max_abs_err=0.0;
  for (int i=0;i<128*d;++i){
#if defined(USE_BF16)
    double diff = std::abs((double)Oref[i] - (double)__bfloat162float(Oh[i]));
#else
    double diff = std::abs((double)Oref[i] - (double)__half2float(Oh[i]));
#endif
    if (diff>max_abs_err) max_abs_err=diff;
  }

  printf("N=%d d=%d  naive=%.3f ms  tiled=%.3f ms  speedup=%.2fx  max_abs_err(128 rows)=%.3e\n",
         N,d,ms_naive,ms_tiled, ms_naive/std::max(ms_tiled,1e-6f), max_abs_err);

  cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
  return 0;
}
```

### HIP: Minimal, Runnable Kernels + Harness

`topics/12-attention-kernels/code/flash_attn_hip.cpp`

```cpp
// Minimal SDPA kernels: ROCm/HIP 6.x
// Builds: hipcc -O3 -std=c++17 --offload-arch=${GFX_ARCH} flash_attn_hip.cpp -o flash_attn_hip
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#ifdef USE_BF16
#include <hip/hip_bfloat16.h>
#endif
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>

#ifndef BLOCK_M
#define BLOCK_M 64
#endif
#ifndef BLOCK_K
#define BLOCK_K 64
#endif
#ifndef D_HEAD
#define D_HEAD 64
#endif

#if defined(USE_BF16)
using htype = hip_bfloat16;
__device__ __forceinline__ float to_float(htype x){ return __bfloat16_to_float(x); }
__device__ __forceinline__ htype to_htype(float x){ return __float_to_bfloat16(x); }
#else
using htype = __half;
__device__ __forceinline__ float to_float(htype x){ return __half2float(x); }
__device__ __forceinline__ htype to_htype(float x){ return __float2half(x); }
#endif

#define API_CHECK(x) do { auto e = (x); if (e != hipSuccess){ \
  printf("HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); abort();}} while(0)

__global__ void attn_naive_rowwise(
    const htype* __restrict__ Q,
    const htype* __restrict__ K,
    const htype* __restrict__ V,
    htype* __restrict__ O,
    int N, int d, float scale)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  float m = -INFINITY, l = 0.f;
  float acc[D_HEAD]; for (int t=0;t<D_HEAD;++t) acc[t]=0.f;

  const htype* q = Q + i*d;

  for (int base=0;base<N;base+=BLOCK_K){
    int kTile = min(BLOCK_K, N - base);
    float tile_max = -INFINITY;
    for (int j=0;j<kTile;++j){
      const htype* kj = K + (base + j)*d;
      float dot = 0.f;
      for (int t=0;t<d;++t) dot += to_float(q[t]) * to_float(kj[t]);
      tile_max = fmaxf(tile_max, dot*scale);
    }
    float m_new = fmaxf(m, tile_max);
    float alpha = expf(m - m_new);
    for (int t=0;t<d;++t) acc[t]*=alpha;
    float l_new = alpha*l;

    for (int j=0;j<kTile;++j){
      const htype* kj = K + (base + j)*d;
      const htype* vj = V + (base + j)*d;
      float dot=0.f;
      for (int t=0;t<d;++t) dot += to_float(q[t]) * to_float(kj[t]);
      float p = expf(dot*scale - m_new);
      l_new += p;
      for (int t=0;t<d;++t) acc[t]+= p*to_float(vj[t]);
    }
    m=m_new; l=l_new;
  }
  htype* o = O + i*d;
  for (int t=0;t<d;++t) o[t] = to_htype(acc[t]/l);
}

extern __shared__ htype smem[]; // K then V

__global__ void attn_tiled_flash(
    const htype* __restrict__ Q,
    const htype* __restrict__ K,
    const htype* __restrict__ V,
    htype* __restrict__ O,
    int N, int d, float scale)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  htype* smemK = smem;
  htype* smemV = smem + BLOCK_K * d;

  float m=-INFINITY, l=0.f;
  float acc[D_HEAD]; for (int t=0;t<D_HEAD;++t) acc[t]=0.f;

  const htype* q = Q + i*d;

  for (int base=0;base<N;base+=BLOCK_K){
    int kTile = min(BLOCK_K, N - base);
    int tile_elems = kTile*d;
    for (int idx = threadIdx.x; idx < tile_elems; idx += blockDim.x){
      smemK[idx] = K[(base*d) + idx];
      smemV[idx] = V[(base*d) + idx];
    }
    __syncthreads();

    float tile_max = -INFINITY;
    for (int j=0;j<kTile;++j){
      const htype* kj = smemK + j*d;
      float dot=0.f;
      for (int t=0;t<d;++t) dot += to_float(q[t]) * to_float(kj[t]);
      tile_max = fmaxf(tile_max, dot*scale);
    }
    float m_new = fmaxf(m, tile_max);
    float alpha = expf(m - m_new);
    for (int t=0;t<d;++t) acc[t]*=alpha;
    float l_new = alpha*l;

    for (int j=0;j<kTile;++j){
      const htype* kj = smemK + j*d;
      const htype* vj = smemV + j*d;
      float dot=0.f;
      for (int t=0;t<d;++t) dot += to_float(q[t]) * to_float(kj[t]);
      float p = expf(dot*scale - m_new);
      l_new += p;
      for (int t=0;t<d;++t) acc[t]+= p*to_float(vj[t]);
    }
    m=m_new; l=l_new;
    __syncthreads();
  }
  htype* o = O + i*d;
  for (int t=0;t<d;++t) o[t] = to_htype(acc[t]/l);
}

// CPU reference and harness
static void cpu_reference(const std::vector<float>& Qf,
                          const std::vector<float>& Kf,
                          const std::vector<float>& Vf,
                          std::vector<float>& Of,
                          int N, int d, float scale, int rows_check)
{
  for (int i=0;i<rows_check;++i){
    float m=-INFINITY;
    for (int j=0;j<N;++j){
      float dot=0.f; for (int t=0;t<d;++t) dot += Qf[i*d+t]*Kf[j*d+t];
      m = std::max(m, dot*scale);
    }
    float l=0.f; std::vector<float> acc(d,0.f);
    for (int j=0;j<N;++j){
      float dot=0.f; for (int t=0;t<d;++t) dot += Qf[i*d+t]*Kf[j*d+t];
      float p = std::exp(dot*scale - m);
      l+=p; for (int t=0;t<d;++t) acc[t]+= p*Vf[j*d+t];
    }
    for (int t=0;t<d;++t) Of[i*d+t]=acc[t]/l;
  }
}

int main(){
  const int N=1024, d=D_HEAD; const float scale = 1.0f/std::sqrt((float)d);
  size_t bytes=(size_t)N*d*sizeof(htype);
  htype *dQ,*dK,*dV,*dO;
  API_CHECK(hipMalloc(&dQ,bytes));
  API_CHECK(hipMalloc(&dK,bytes));
  API_CHECK(hipMalloc(&dV,bytes));
  API_CHECK(hipMalloc(&dO,bytes));

  std::mt19937 rng(0); std::uniform_real_distribution<float> dist(-0.5f,0.5f);
  std::vector<float> Qf(N*d),Kf(N*d),Vf(N*d);
  for(int i=0;i<N*d;++i){ Qf[i]=dist(rng); Kf[i]=dist(rng); Vf[i]=dist(rng); }
  std::vector<htype> Qh(N*d),Kh(N*d),Vh(N*d);
  for(int i=0;i<N*d;++i){
#if defined(USE_BF16)
    Qh[i]=__float_to_bfloat16(Qf[i]);
    Kh[i]=__float_to_bfloat16(Kf[i]);
    Vh[i]=__float_to_bfloat16(Vf[i]);
#else
    Qh[i]=__float2half(Qf[i]); Kh[i]=__float2half(Kf[i]); Vh[i]=__float2half(Vf[i]);
#endif
  }
  API_CHECK(hipMemcpy(dQ,Qh.data(),bytes,hipMemcpyHostToDevice));
  API_CHECK(hipMemcpy(dK,Kh.data(),bytes,hipMemcpyHostToDevice));
  API_CHECK(hipMemcpy(dV,Vh.data(),bytes,hipMemcpyHostToDevice));

  dim3 block(BLOCK_M);
  dim3 grid((N + BLOCK_M - 1)/BLOCK_M);
  size_t smem_bytes = (size_t)BLOCK_K * d * sizeof(htype) * 2;

  // Warmup
  hipLaunchKernelGGL(attn_tiled_flash, grid, block, smem_bytes, 0, dQ,dK,dV,dO,N,d,scale);
  API_CHECK(hipDeviceSynchronize());

  hipEvent_t s,e; hipEventCreate(&s); hipEventCreate(&e);
  float ms_naive=0.f, ms_tiled=0.f;

  hipEventRecord(s);
  hipLaunchKernelGGL(attn_naive_rowwise, grid, block, 0, 0, dQ,dK,dV,dO,N,d,scale);
  hipEventRecord(e); hipEventSynchronize(e); hipEventElapsedTime(&ms_naive,s,e);

  hipEventRecord(s);
  hipLaunchKernelGGL(attn_tiled_flash, grid, block, smem_bytes, 0, dQ,dK,dV,dO,N,d,scale);
  hipEventRecord(e); hipEventSynchronize(e); hipEventElapsedTime(&ms_tiled,s,e);

  std::vector<htype> Oh(N*d);
  API_CHECK(hipMemcpy(Oh.data(), dO, bytes, hipMemcpyDeviceToHost));

  std::vector<float> Oref(N*d,0.f);
  cpu_reference(Qf,Kf,Vf,Oref,N,d,scale,/*rows_check=*/128);

  double max_abs_err=0.0;
  for (int i=0;i<128*d;++i){
#if defined(USE_BF16)
    double diff = std::abs((double)Oref[i] - (double)__bfloat16_to_float(Oh[i]));
#else
    double diff = std::abs((double)Oref[i] - (double)__half2float(Oh[i]));
#endif
    if (diff>max_abs_err) max_abs_err=diff;
  }

  printf("N=%d d=%d  naive=%.3f ms  tiled=%.3f ms  speedup=%.2fx  max_abs_err(128 rows)=%.3e\n",
         N,d,ms_naive,ms_tiled, ms_naive/std::max(ms_tiled,1e-6f), max_abs_err);

  hipFree(dQ); hipFree(dK); hipFree(dV); hipFree(dO);
  return 0;
}
```

## Profiling and Validation

### NVIDIA

- Nsight Compute (per-kernel):

```
ncu --set full --kernel-name regex:attn_.* ./flash_attn_cuda
```

Key metrics to inspect:

- `sm__throughput.avg.pct_of_peak_sustained_active` (target ≥ 35% for tiled on consumer GPUs with d=64).
- `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second` and `lts__t_bytes.sum.per_second`: tiled should reduce DRAM GB/s vs naive at same N.
- `launch__occupancy_limit_active_warps` and `smsp__inst_executed_per_warp`: confirm good occupancy and balanced instruction mix.

### AMD

- rocprof timeline + counters:

```
rocprof --hip-trace --timestamp on -o trace.json ./flash_attn_hip
rocprof --stats --hsa-trace --timestamp on ./flash_attn_hip
```

Counters to check (names vary by ASIC):

- L2 hit rate (aim higher with tiled vs naive).
- `SQ_WAVES`, `VALUInsts`, `SALUInsts` for instruction mix; wave occupancy ≥ 40% typically OK given shared/LDS.

### Functional validation

The harness prints `max_abs_err` vs CPU on first 128 rows. Targets:

- FP16/BF16 with FP32 accum: `max_abs_err ≤ 3e-2` at `N=1024, d=64`.

## Performance Checklist

- [ ] Block size `BLOCK_M` is multiple of warp/wavefront (32 on NVIDIA, 64 on AMD); default 64 fits both.
- [ ] `BLOCK_K * d * sizeof(type) * 2` fits in Shared/LDS (<= 48–100 KB typical).
- [ ] FP32 accumulation enabled; scaling factor $1/\sqrt{d}$ applied.
- [ ] No bank conflicts: prefer `d` multiple of 32; if not, consider padding `d_pad = ((d+31)/32)*32`.
- [ ] Global loads 128B-aligned; consider `reinterpret_cast<const float4*>` for vectorized loads if `d % 8 == 0`.
- [ ] Error threshold passes (≤ 3e-2).
- [ ] Tiled kernel shows ≥ 1.5× speedup over naive at `N≥1024, d=64`.

## Troubleshooting

| Symptom                  | Likely Cause                           | Fix                                                                     |
| ------------------------ | -------------------------------------- | ----------------------------------------------------------------------- |
| `illegal memory access`  | Shared memory size too small           | Increase dynamic shared bytes: `smem_bytes = BLOCK_K*d*sizeof(type)*2`. |
| Very low speedup         | `BLOCK_M` too small; poor reuse        | Increase `BLOCK_M` (64/128), ensure occupancy still healthy.            |
| Numerical `nan`/`inf`    | Missing LSE rescale or overflow in exp | Ensure online softmax formula and FP32 accumulators.                    |
| Low occupancy            | Large registers/shared usage           | Reduce `BLOCK_M` or unrolling; check `-maxrregcount` only if necessary. |
| L2/DRAM BW too high      | No coalescing/padding                  | Align pointers; pad `d` to 32; use `float4`/`__half2` loads.            |
| ROCm build fails on BF16 | Missing header/type                    | Use `-DUSE_BF16` only with ROCm ≥ 6.x; include `<hip/hip_bfloat16.h>`.  |
| CUDA BF16 compile error  | Toolkit < 11.0 or header missing       | Require CUDA ≥ 11.0; include `<cuda_bf16.h>`.                           |
| Accuracy poor vs CPU     | Different dtype on host                | Generate host FP32, cast once to device dtype; verify scaling.          |

## Acceptance Criteria

- Documentation explains math, kernel structure, and hardware mapping with numeric examples.
- CUDA and HIP programs compile and run in seconds on modern GPUs.
- Harness prints speed and error; `max_abs_err ≤ 3e-2` on first 128 rows at `N=1024, d=64`.
- Tiled kernel demonstrates **≥ 1.5×** speedup over naive on the same hardware configuration (indicative; absolute may vary).
- Profiling instructions included with 2–3 critical counters and interpretation guidance.

## Further Work

- **Tensor-core/MFMA path**: tile $d$ to MMA fragments and fuse epilogues (bias, dropout, causal masking).
- **Causal/ALiBi/LoRA**: integrate masking in the streaming loop without materialization.
- **Grouped-Q/KV (GQA/MQA)**: stride heads to maximize K/V reuse across groups.
- **Quantized K/V**: INT8/FP8 with on-the-fly dequant in the tile loop.
- **Persistent decode**: single long-lived kernel with device-side loops and CUDA/HIP Graphs to amortize launch overhead.
- **Split-K + multi-CTA reduction**: for very large $N$, split key dimension and reduce partial outputs.
