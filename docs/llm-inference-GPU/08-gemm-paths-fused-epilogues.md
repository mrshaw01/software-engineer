# GEMM Paths & Fused Epilogues

## Summary

General Matrix Multiply (GEMM) dominates transformer inference compute, both in attention projections and MLP layers. The choice of GEMM path (library vs. custom kernels) and the ability to fuse epilogues (bias, activation, de/quantization, scaling) directly affect latency, bandwidth, and utilization. This note explains GEMM variants for prefill and decode, quantifies memory/compute trade-offs, and provides runnable CUDA and HIP examples that implement fused bias+GELU epilogues. We also outline profiling and pass/fail checks to validate gains.

## Why It Matters for LLM Inference

- **Prefill**: large M and K (sequence·batch × hidden). Compute-bound when Tensor Cores/MFMA are saturated. Fusing epilogues raises arithmetic intensity, cutting extra global reads/writes.
- **Decode**: tiny M (≈ batch). Often memory- or launch-bound. Persistent kernels, GEMM re-use, and light-weight fused epilogues reduce overheads. Library paths with heuristics can underperform if shapes are skinny; custom or grouped GEMM can help.

## Key Concepts and Formulas

Let A ∈ ℝ^{M×K}, B ∈ ℝ^{K×N}, C ∈ ℝ^{M×N}. Using FP16 elements (s = 2 bytes). FLOPs ≈ 2MNK.

- **Bytes (no fusion)** ≈ s·(MK + KN + 3MN). Terms: read A, read B, write C, read C for epilogue, write C again (bias/activation), bias N is negligible.
- **Bytes (fused epilogue)** ≈ s·(MK + KN + MN + N). We remove ≈ 2·s·MN bytes.
- **Arithmetic Intensity (AI)** = FLOPs / Bytes.

### Numeric example (Prefill scale)

M = 8·2048 = 16384, K = 4096, N = 4096, s = 2 bytes.

- FLOPs = 2·M·N·K = 5.49755813888e11.
- Bytes (no fuse) = 2·(MK + KN + 3MN) = 570,425,344 ≈ 544 MiB.
- Bytes (fused) = 2·(MK + KN + MN) = 301,989,888 ≈ 288 MiB.
- **Savings** ≈ 268,435,456 bytes ≈ 256 MiB.
- AI(no fuse) ≈ 963.9 FLOP/byte, AI(fused) ≈ 1819.2 FLOP/byte (≈1.9× improvement).

### Numeric example (Decode scale)

M = batch = 8, K = 4096, N = 4096.

- Bytes saved by fusion ≈ 2·s·MN = 2·2·(8·4096) = 131,072 bytes ≈ 128 KiB.
- Relative gain is modest; launch overhead and weight reuse dominate. Use persistent kernels or batched/Grouped GEMM.

## GPU Deep Dive

### NVIDIA (CUDA)

- **Warps/SMs**: 32-thread warps, schedule Tensor Core HMMA on SMs. Use WMMA or CUTLASS/cublasLt.
- **Tensor Cores**: Peak when tiles match MMA shapes (e.g., 16×16×16 FP16→FP32). Align K to multiples of 16.
- **Memory**: Prefer coalesced row-major/col-major with 128-bit loads (e.g., `__half2`/`float4`). Stage tiles in shared memory, avoid bank conflicts.
- **Library path**: `cublasLtMatmul` supports epilogues (bias, ReLU, GELU, scale, aux). Good for prefill; decode benefits from persistent kernels or CUDA Graphs.

### AMD (ROCm/HIP)

- **Wavefronts/CUs**: 64-lane wavefronts on Compute Units. MFMA/XDLOPs drive matrix math throughput.
- **LDS**: Stage A/B tiles; avoid bank conflicts; vectorize (e.g., `half2`/`int4`).
- **Library path**: `hipBLASLt` provides GEMM with epilogues on ROCm; otherwise, write tiled MFMA kernels or use rocWMMA. For skinny decode shapes, persistent wavefront kernels reduce launches.

## Implementation

This section provides two runnable baselines with a **fused bias+GELU epilogue**. The CUDA sample uses WMMA; the HIP sample uses a tiled kernel with LDS. Both validate against a CPU reference and print GFLOP/s.

### CUDA: WMMA GEMM with Fused Bias+GELU

**File:** `topics/08-gemm-paths-fused-epilogues/code/gemm_wmma_bias_gelu.cu`

```cpp
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

// GELU approximation
__device__ __forceinline__ float gelu(float x) {
    const float c = 0.7978845608028654f; // sqrt(2/pi)
    return 0.5f * x * (1.f + tanhf(c * (x + 0.044715f * x * x * x)));
}

// Each block computes one 16x16 C tile with one warp.
// Grid dims: (N/16, M/16). K must be multiple of 16.
__global__ void wmma_gemm_bias_gelu(const half* __restrict__ A, const half* __restrict__ B,
                                    const half* __restrict__ bias, half* __restrict__ C,
                                    int M, int N, int K) {
    int tile_m = blockIdx.y; // [0, M/16)
    int tile_n = blockIdx.x; // [0, N/16)

    wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16,16,16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    // Base pointers for this tile
    const half* A_base = A + (tile_m * 16) * K;
    const half* B_base = B + (tile_n * 16);

    for (int k0 = 0; k0 < K; k0 += 16) {
        // Load 16x16 tiles
        wmma::load_matrix_sync(a_frag, A_base + k0, K);
        wmma::load_matrix_sync(b_frag, B_base + k0 * N, N);
        // MMA accumulate
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Fuse bias + GELU and store
    // Load into a register buffer, add bias per column, apply GELU.
    float c_tile[16*16];
    wmma::store_matrix_sync(c_tile, c_frag, 16, wmma::mem_row_major);

    // Bias is length-N (one per column)
    int row0 = tile_m * 16;
    int col0 = tile_n * 16;
    for (int i = 0; i < 16; ++i) {
        int row = row0 + i;
        if (row >= M) break;
        for (int j = 0; j < 16; ++j) {
            int col = col0 + j;
            if (col >= N) break;
            float v = c_tile[i*16 + j] + __half2float(bias[col]);
            v = gelu(v);
            C[row * N + col] = __float2half_rn(v);
        }
    }
}

static void cpu_ref(const std::vector<half>& A, const std::vector<half>& B,
                    const std::vector<half>& bias, std::vector<half>& C,
                    int M, int N, int K) {
    auto h2f = [](half h){ return __half2float(h); };
    auto gelu_host = [](float x){ const float c=0.7978845608028654f; return 0.5f*x*(1.f+tanhf(c*(x+0.044715f*x*x*x))); };
    std::vector<float> acc(M*N, 0.f);
    for (int m=0;m<M;++m){
        for (int k=0;k<K;++k){
            float a = h2f(A[m*K+k]);
            for (int n=0;n<N;++n){
                acc[m*N+n] += a * h2f(B[k*N+n]);
            }
        }
    }
    for (int m=0;m<M;++m){
        for (int n=0;n<N;++n){
            float v = acc[m*N+n] + __half2float(bias[n]);
            C[m*N+n] = __float2half_rn(gelu_host(v));
        }
    }
}

int main(int argc, char** argv){
    int M = 1024, N = 1024, K = 1024; // multiples of 16
    if (argc==4){ M=atoi(argv[1]); N=atoi(argv[2]); K=atoi(argv[3]); }

    if (M%16||N%16||K%16){ fprintf(stderr, "M,N,K must be multiples of 16\n"); return 1; }

    size_t szA = (size_t)M*K, szB=(size_t)K*N, szC=(size_t)M*N, szBias=N;
    std::vector<half> hA(szA), hB(szB), hC(szC), hBias(szBias), hRef(szC);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.f,1.f);
    for(size_t i=0;i<szA;++i) hA[i] = __float2half(dist(rng));
    for(size_t i=0;i<szB;++i) hB[i] = __float2half(dist(rng));
    for(int i=0;i<N;++i) hBias[i] = __float2half(dist(rng));

    half *dA,*dB,*dC,*dBias;
    cudaMalloc(&dA, szA*sizeof(half));
    cudaMalloc(&dB, szB*sizeof(half));
    cudaMalloc(&dC, szC*sizeof(half));
    cudaMalloc(&dBias, szBias*sizeof(half));
    cudaMemcpy(dA,hA.data(),szA*sizeof(half),cudaMemcpyHostToDevice);
    cudaMemcpy(dB,hB.data(),szB*sizeof(half),cudaMemcpyHostToDevice);
    cudaMemcpy(dBias,hBias.data(),szBias*sizeof(half),cudaMemcpyHostToDevice);

    dim3 grid(N/16, M/16);
    dim3 block(32,1,1); // one warp

    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    wmma_gemm_bias_gelu<<<grid, block>>>(dA,dB,dBias,dC,M,N,K);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms=0; cudaEventElapsedTime(&ms,t0,t1);

    cudaMemcpy(hC.data(),dC,szC*sizeof(half),cudaMemcpyDeviceToHost);

    // correctness
    cpu_ref(hA,hB,hBias,hRef,M,N,K);
    double max_abs=0, max_rel=0;
    for(size_t i=0;i<szC;++i){
        float a = __half2float(hC[i]);
        float b = __half2float(hRef[i]);
        double abs_err = fabs(a-b);
        double rel_err = abs_err / (fabs(b)+1e-6);
        if (abs_err>max_abs) max_abs=abs_err;
        if (rel_err>max_rel) max_rel=rel_err;
    }

    double flops = 2.0*(double)M*N*K; // FMA counts as 2 flops
    double gflops = flops/(ms*1e6);
    printf("CUDA WMMA fused bias+GELU: M=%d N=%d K=%d => %.2f GFLOP/s, time=%.3f ms\n", M,N,K,gflops,ms);
    printf("MaxAbsErr=%.3e MaxRelErr=%.3e\n", max_abs, max_rel);

    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dBias);
    return (max_rel<5e-3 && max_abs<5e-2) ? 0 : 2; // FP16 tolerance
}
```

**Build (CUDA 12.x):**

```bash
SM_ARCH=sm_80   # e.g., sm_80=A100, sm_89=H100, sm_90=Blackwell
nvcc -O3 -std=c++17 -arch=${SM_ARCH} -lineinfo \
  topics/08-gemm-paths-fused-epilogues/code/gemm_wmma_bias_gelu.cu -o gemm_wmma_bias_gelu
```

**Run:**

```bash
./gemm_wmma_bias_gelu 4096 4096 4096
```

### HIP: Tiled GEMM with Fused Bias+GELU (LDS)

**File:** `topics/08-gemm-paths-fused-epilogues/code/gemm_tiled_bias_gelu_hip.cpp`

```cpp
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>

__device__ __forceinline__ float gelu(float x) {
    const float c = 0.7978845608028654f; // sqrt(2/pi)
    return 0.5f * x * (1.f + tanhf(c * (x + 0.044715f * x * x * x)));
}

// 16x16 tiles with BK=16; one block computes one C tile.
// Grid: (N/16, M/16). All dims must be multiples of 16 for simplicity.
__global__ void gemm_tiled_bias_gelu(const __half* __restrict__ A,
                                     const __half* __restrict__ B,
                                     const __half* __restrict__ bias,
                                     __half* __restrict__ C,
                                     int M, int N, int K) {
    const int TILE=16;
    __shared__ __half As[TILE][TILE];
    __shared__ __half Bs[TILE][TILE];

    int tile_m = blockIdx.y; // tile row
    int tile_n = blockIdx.x; // tile col
    int row = tile_m * TILE + threadIdx.y;
    int col = tile_n * TILE + threadIdx.x;

    float acc = 0.f;

    for (int k0 = 0; k0 < K; k0 += TILE) {
        // Load tiles
        As[threadIdx.y][threadIdx.x] = A[row*K + (k0 + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(k0 + threadIdx.y)*N + col];
        __syncthreads();

        // Compute partial products
        #pragma unroll
        for (int k=0;k<TILE;++k){
            float a = __half2float(As[threadIdx.y][k]);
            float b = __half2float(Bs[k][threadIdx.x]);
            acc += a*b;
        }
        __syncthreads();
    }

    // Fused bias + GELU
    float v = acc + __half2float(bias[col]);
    v = gelu(v);
    C[row*N + col] = __float2half_rn(v);
}

static void cpu_ref(const std::vector<__half>& A, const std::vector<__half>& B,
                    const std::vector<__half>& bias, std::vector<__half>& C,
                    int M, int N, int K) {
    auto h2f = [](__half h){ return __half2float(h); };
    auto gelu_host = [](float x){ const float c=0.7978845608028654f; return 0.5f*x*(1.f+tanhf(c*(x+0.044715f*x*x*x))); };
    std::vector<float> acc(M*N, 0.f);
    for (int m=0;m<M;++m){
        for (int k=0;k<K;++k){
            float a = h2f(A[m*K+k]);
            for (int n=0;n<N;++n){
                acc[m*N+n] += a * h2f(B[k*N+n]);
            }
        }
    }
    for (int m=0;m<M;++m){
        for (int n=0;n<N;++n){
            float v = acc[m*N+n] + __half2float(bias[n]);
            C[m*N+n] = __float2half_rn(gelu_host(v));
        }
    }
}

int main(int argc, char** argv){
    int M=512,N=512,K=512; // multiples of 16
    if (argc==4){ M=atoi(argv[1]); N=atoi(argv[2]); K=atoi(argv[3]); }
    if (M%16||N%16||K%16){ fprintf(stderr, "M,N,K must be multiples of 16\n"); return 1; }

    size_t szA=(size_t)M*K, szB=(size_t)K*N, szC=(size_t)M*N, szBias=N;
    std::vector<__half> hA(szA), hB(szB), hC(szC), hBias(szBias), hRef(szC);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.f,1.f);
    for(size_t i=0;i<szA;++i) hA[i] = __float2half(dist(rng));
    for(size_t i=0;i<szB;++i) hB[i] = __float2half(dist(rng));
    for(int i=0;i<N;++i) hBias[i] = __float2half(dist(rng));

    __half *dA,*dB,*dC,*dBias;
    hipMalloc(&dA, szA*sizeof(__half));
    hipMalloc(&dB, szB*sizeof(__half));
    hipMalloc(&dC, szC*sizeof(__half));
    hipMalloc(&dBias, szBias*sizeof(__half));
    hipMemcpy(dA, hA.data(), szA*sizeof(__half), hipMemcpyHostToDevice);
    hipMemcpy(dB, hB.data(), szB*sizeof(__half), hipMemcpyHostToDevice);
    hipMemcpy(dBias, hBias.data(), szBias*sizeof(__half), hipMemcpyHostToDevice);

    dim3 grid(N/16, M/16);
    dim3 block(16,16);

    hipEvent_t t0,t1; hipEventCreate(&t0); hipEventCreate(&t1);
    hipEventRecord(t0);
    hipLaunchKernelGGL(gemm_tiled_bias_gelu, grid, block, 0, 0, dA,dB,dBias,dC,M,N,K);
    hipEventRecord(t1); hipEventSynchronize(t1);
    float ms=0; hipEventElapsedTime(&ms,t0,t1);

    hipMemcpy(hC.data(), dC, szC*sizeof(__half), hipMemcpyDeviceToHost);

    cpu_ref(hA,hB,hBias,hRef,M,N,K);
    double max_abs=0, max_rel=0;
    for(size_t i=0;i<szC;++i){
        float a = __half2float(hC[i]);
        float b = __half2float(hRef[i]);
        double abs_err=fabs(a-b);
        double rel_err=abs_err/(fabs(b)+1e-6);
        if(abs_err>max_abs) max_abs=abs_err;
        if(rel_err>max_rel) max_rel=rel_err;
    }

    double flops = 2.0*(double)M*N*K;
    double gflops = flops/(ms*1e6);
    printf("HIP tiled fused bias+GELU: M=%d N=%d K=%d => %.2f GFLOP/s, time=%.3f ms\n", M,N,K,gflops,ms);
    printf("MaxAbsErr=%.3e MaxRelErr=%.3e\n", max_abs, max_rel);

    hipFree(dA); hipFree(dB); hipFree(dC); hipFree(dBias);
    return (max_rel<5e-3 && max_abs<5e-2) ? 0 : 2;
}
```

**Build (ROCm 6.x):**

```bash
GFX_ARCH=gfx942  # MI300-class; use gfx90a for MI200, etc.
hipcc -O3 -std=c++17 --offload-arch=${GFX_ARCH} \
  topics/08-gemm-paths-fused-epilogues/code/gemm_tiled_bias_gelu_hip.cpp -o gemm_tiled_bias_gelu_hip
```

**Run:**

```bash
./gemm_tiled_bias_gelu_hip 1024 1024 1024
```

### Library Paths (reference snippets)

Below show how to set up fused epilogues with vendor libraries. These are illustrative and may require version-specific headers.

**CUDA / cuBLASLt (bias+GELU):**

```cpp
// ... create handle, matmul desc, set compute type FP16->FP32
cublasLtHandle_t lt; cublasLtCreate(&lt);
cublasLtMatmulDesc_t op; cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_16F);
int epi = CUBLASLT_EPILOGUE_BIAS_GELU;
cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi));
cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &dBias, sizeof(dBias));
// ... create A/B/C layouts (order row/col), pick algo, call cublasLtMatmul(...)
```

**ROCm / hipBLASLt (bias+GELU):**

```cpp
// hipblasLtHandle_t handle; hipblasLtMatmulDesc_t op; etc.
hipblasLtEpilogue_t epi = HIPBLASLT_EPILOGUE_BIAS_GELU;
hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi));
hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &dBias, sizeof(dBias));
// hipblasLtMatmul(...)
```

## Profiling and Validation

### NVIDIA

- **Nsight Compute (kernel-level):**

```bash
ncu --target-processes all \
    --metrics sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active, \
             sm__inst_executed_pipe_tensor_op_hmma.sum, \
             lts__t_bytes.sum,dram__bytes.sum \
    ./gemm_wmma_bias_gelu 4096 4096 4096
```

Expect high HMMA active (>70%) for square 4k shapes; dram bytes should be ≈ `2*(MK + KN + MN)`·s when fused.

- **Nsight Systems (timeline & launches):** capture to confirm single kernel covers epilogue.

### AMD

- **rocprof (timeline + counters):**

```bash
rocprof --hip-trace --hsa-trace ./gemm_tiled_bias_gelu_hip 1024 1024 1024
```

- **Metrics (if available):** check `SQ_WAVES`, `GRBM_GUI_ACTIVE`, and memory bytes. Expect fewer global bytes than a non-fused two-kernel pipeline.

### Correctness

- CPU reference comparison with FP16 tolerance: `MaxRelErr < 5e-3` and `MaxAbsErr < 5e-2`.

## Performance Checklist

- [ ] M,N,K aligned to MMA/MFMA tile sizes (e.g., multiple of 16 for FP16).
- [ ] Global loads/stores 128-bit aligned; use vector types when adding real kernels.
- [ ] Single fused kernel writes C exactly once.
- [ ] Tensor Core/MFMA utilization ≥ 70% (prefill-sized GEMM on modern GPUs).
- [ ] Decode path uses persistent kernel or Graphs when M is small.
- [ ] Strides/leading dims set to avoid transposes or reorders inside kernel.
- [ ] Workspace and algorithm selection fixed for reproducibility.

## Troubleshooting

| Symptom                             | Likely Cause                               | Fix                                                                      |
| ----------------------------------- | ------------------------------------------ | ------------------------------------------------------------------------ |
| Low tensor core utilization         | K not multiple of 16; wrong WMMA/TF32 mode | Pad K; choose correct fragment types; ensure FP16 inputs and FP32 accum  |
| High DRAM bytes vs expectation      | Epilogue not fused; extra kernel writes    | Verify single kernel in timeline; check library epilogue enum/attributes |
| Bank conflicts in shared/LDS        | Poor tile mapping, no padding              | Pad LDS tiles (e.g., TILE+1 stride); align loads to 128 bits             |
| Launch-bound at decode              | Too many small GEMMs                       | Use persistent kernels, CUDA/HIP Graphs, or group requests               |
| Mismatched outputs vs reference     | Activation mismatch or precision           | Match GELU variant, use FP32 accum; relax tolerance for FP16             |
| hipcc build fails on half types     | Missing hip_fp16 include or flags          | `#include <hip/hip_fp16.h>`; compile with ROCm 6.x                       |
| cuBLASLt/hipBLASLt epilogue ignored | Unsupported combination                    | Check library versions; fall back to custom fused kernel                 |

## Acceptance Criteria

- CUDA and HIP examples compile and run within seconds on A100/H100 or MI200/MI300-class GPUs.
- Nsight Compute / rocprof confirm that fused kernels write C once and reduce DRAM bytes vs. a two-kernel baseline.
- Correctness thresholds: `MaxRelErr < 5e-3` and `MaxAbsErr < 5e-2` for FP16→FP32.
- For `M=N=K=4096`, CUDA sample achieves ≥ 65% HMMA active on supported GPUs.

## Further Work

- Add INT8/FP8 paths with fused dequant (scales per tensor or per channel) in epilogue.
- Implement persistent decode kernels with KV-resident Q,K,V projections and fused RoPE.
- Integrate CUTLASS/rocWMMA variants to compare library vs. custom kernels.
- Evaluate grouped GEMM for MoE and QKV fusion (A×\[Wq|Wk|Wv]).
- Add allocator/paging-aware weight prefetch to sustain Tensor Core/MFMA.
