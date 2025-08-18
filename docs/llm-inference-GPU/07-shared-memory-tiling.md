# Shared Memory/LDS & Tiling

## Summary

Shared Memory (NVIDIA) and LDS (AMD) are low-latency, programmer-managed memories on GPU SMs/CUs. Correct tiling uses these spaces to maximize on-chip data reuse, reduce DRAM traffic, and avoid bank conflicts. This note explains the tiling math, bank layout, coalescing rules, and provides a runnable half-precision (FP16) tiled GEMM kernel that accumulates in FP32 for both CUDA and HIP. You’ll profile shared-memory efficiency, confirm low bank conflicts, and use a checklist to validate improvements.

## Why It Matters for LLM Inference

Decode steps are memory-bound: attention and MLP layers repeatedly read weights and KV cache with modest compute per byte. Tiling into Shared/LDS increases arithmetic intensity (reuse per global byte), which is essential for:

- Attention: block-tiling Q·Kᵀ and V aggregation.
- MLP: W₁/W₂ GEMMs where reusing tiles lowers DRAM bandwidth pressure and improves latency.
- Decode micro-batching: persistent kernels rely on on-chip reuse to hide latency.

## Key Concepts and Formulas

1. **Arithmetic Intensity (AI)** for a tiled GEMM block
   For a block computing `C[bM×bN] += A[bM×bK] · B[bK×bN]` with data element size `B` bytes:

   - FLOPs per block step (over `bK`): `F = 2 · bM · bN · bK`
   - Global bytes loaded per step (ignoring C write): `G = (bM·bK + bK·bN) · B`
   - AI (FLOPs/byte):
     `AI_step = F/G = (2 · bM · bN) / (B · (bM + bN))` (independent of `bK`).
     Example (FP16, `B=2`, `bM=bN=64`): `AI_step = 2·64·64 / (2·(64+64)) = 8192 / 256 = 32 FLOPs/byte`.

2. **Shared/LDS capacity constraint**
   `S_used = (bM·bK + (bK·(bN+pad))) · B` must fit beneath the per-block shared/LDS budget.
   Example (FP16, `bM=bN=64`, `bK=32`, `pad=1`):
   `S_used = (64·32 + 32·(64+1))·2 = (2048 + 2080)·2 = 8256 bytes`.

3. **Occupancy vs. shared memory**
   Blocks per SM/CU limited by registers and shared/LDS:
   `blocks_per_sm_by_smem = floor(S_total_per_sm / S_used_per_block)`.
   Keep ≥2 resident blocks per SM/CU to hide latency.

4. **Bank conflicts**

   - Both NVIDIA and AMD expose 32 banks, typically 4 bytes wide. Two FP16 values share a bank entry.
   - Access pattern `addr/4 mod 32` maps threads to banks.
   - Use vector widths of 4 bytes (e.g., `__half2` or `float`) and padding (`+1`) on one tile dimension to avoid worst-case modulo patterns.

5. **Coalescing (global)**

   - Make global loads/stores contiguous in the fastest-varying dimension.
   - For `A` (row-major), load along `K`; for `B`, load along `N`.

## GPU Deep Dive

### NVIDIA specifics

- **Warps**: 32 threads; shared memory has 32 banks × 4B.
- **Tensor Cores**: Best with WMMA and aligned fragments; this example uses classic SIMT with FP32 accumulators for clarity.
- **Occupancy**: Balanced by registers/thread and shared memory/block; inspect with Nsight Compute.

### AMD specifics

- **Wavefronts**: 64 threads; LDS also 32 banks × 4B; avoid conflicts across 64 threads.
- **MFMA/XDLOPs**: Matrix cores on RDNA/CDNA; we use SIMT here to keep the example portable.
- **rocprof/Omnitrace**: Collect LDS counters and wavefront occupancy.

## Implementation

The following single-source file compiles with **both** `nvcc` and `hipcc`. It implements an FP16×FP16→FP32 tiled GEMM with bank-conflict-safe LDS layout (padding on B-tile). It includes a CPU reference, simple correctness check, and timing.

Place under:

```
llm-inference/topics/07-shared-memory-tiling/code/tiled_gemm_shared.cu
```

### Build

CUDA:

```
nvcc -O3 -std=c++17 -arch=sm_80 -lineinfo tiled_gemm_shared.cu -o tiled_gemm_shared
```

ROCm:

```
hipcc -O3 -std=c++17 --offload-arch=gfx90a tiled_gemm_shared.cu -o tiled_gemm_shared_hip
```

### Run

```
./tiled_gemm_shared          # defaults M=N=K=1024
./tiled_gemm_shared 2048 1024 4096
```

### Code: `tiled_gemm_shared.cu`

```cpp
// Single-source CUDA/HIP tiled GEMM using Shared/LDS tiling.
// FP16 inputs, FP32 accumulate, bank-conflict-safe LDS layout.
#if defined(__HIP_PLATFORM_AMD__)
  #include <hip/hip_runtime.h>
  #include <hip/hip_fp16.h>
  #define API_CHECK(x) do { auto e = (x); if (e != hipSuccess) { fprintf(stderr,"HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(e)); abort(); } } while(0)
  #define DEV_ALLOC(ptr,bytes) API_CHECK(hipMalloc((void**)&(ptr),(bytes)))
  #define DEV_FREE(ptr) API_CHECK(hipFree(ptr))
  #define MEMCPY_H2D(dst,src,bytes) API_CHECK(hipMemcpy((dst),(src),(bytes), hipMemcpyHostToDevice))
  #define MEMCPY_D2H(dst,src,bytes) API_CHECK(hipMemcpy((dst),(src),(bytes), hipMemcpyDeviceToHost))
  #define GET_LAST_ERROR() API_CHECK(hipGetLastError())
  #define DEVICE_SYNCH() API_CHECK(hipDeviceSynchronize())
  #define EVENT_T      hipEvent_t
  #define EVENT_CREATE(e) API_CHECK(hipEventCreate(&(e)))
  #define EVENT_RECORD(e, s) API_CHECK(hipEventRecord((e), (s)))
  #define EVENT_SYNCH(e) API_CHECK(hipEventSynchronize((e)))
  #define EVENT_ELAPSED_MS(ms, start, stop) API_CHECK(hipEventElapsedTime(&(ms), (start), (stop)))
  #define STREAM_T hipStream_t
  #define STREAM_DEFAULT 0
  #define LAUNCH_KERNEL(kernel, grid, block, shmem, stream, ...) \
      hipLaunchKernelGGL(kernel, grid, block, shmem, stream, __VA_ARGS__)
  using half_t = __half;
#else
  #include <cuda_runtime.h>
  #include <cuda_fp16.h>
  #define API_CHECK(x) do { auto e = (x); if (e != cudaSuccess) { fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); abort(); } } while(0)
  #define DEV_ALLOC(ptr,bytes) API_CHECK(cudaMalloc((void**)&(ptr),(bytes)))
  #define DEV_FREE(ptr) API_CHECK(cudaFree(ptr))
  #define MEMCPY_H2D(dst,src,bytes) API_CHECK(cudaMemcpy((dst),(src),(bytes), cudaMemcpyHostToDevice))
  #define MEMCPY_D2H(dst,src,bytes) API_CHECK(cudaMemcpy((dst),(src),(bytes), cudaMemcpyDeviceToHost))
  #define GET_LAST_ERROR() API_CHECK(cudaGetLastError())
  #define DEVICE_SYNCH() API_CHECK(cudaDeviceSynchronize())
  #define EVENT_T      cudaEvent_t
  #define EVENT_CREATE(e) API_CHECK(cudaEventCreate(&(e)))
  #define EVENT_RECORD(e, s) API_CHECK(cudaEventRecord((e), (s)))
  #define EVENT_SYNCH(e) API_CHECK(cudaEventSynchronize((e)))
  #define EVENT_ELAPSED_MS(ms, start, stop) API_CHECK(cudaEventElapsedTime(&(ms), (start), (stop)))
  #define STREAM_T cudaStream_t
  #define STREAM_DEFAULT 0
  #define LAUNCH_KERNEL(kernel, grid, block, shmem, stream, ...) \
      kernel<<<grid, block, shmem, stream>>>(__VA_ARGS__)
  using half_t = __half;
#endif

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

// Tile/block parameters (tuned for clarity and portability)
constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 32;
constexpr int TM = 4;  // each thread computes TM x TN outputs
constexpr int TN = 4;
constexpr int PAD_N = 1; // padding on B tile N-dim to avoid bank conflicts

// Convert helpers (device)
__device__ __forceinline__ float h2f(half_t h) {
#if defined(__HIP_PLATFORM_AMD__)
    return __half2float(h);
#else
    return __half2float(h);
#endif
}

__device__ __forceinline__ half_t f2h(float f) {
#if defined(__HIP_PLATFORM_AMD__)
    return __float2half(f);
#else
    return __float2half(f);
#endif
}

// Kernel: C[M×N] = A[M×K] * B[K×N], A/B in FP16, C in FP32
__global__ void gemm_tiled_shared(const half_t* __restrict__ A,
                                  const half_t* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int N, int K)
{
    // Shared/LDS tiles
    __shared__ half_t Asub[BLOCK_M][BLOCK_K];
    __shared__ half_t Bsub[BLOCK_K][BLOCK_N + PAD_N];

    const int tidx = threadIdx.x;  // 0..15
    const int tidy = threadIdx.y;  // 0..15

    const int blockRow = blockIdx.y; // along M
    const int blockCol = blockIdx.x; // along N

    // Start indices in C for this thread's microtile
    const int row0 = blockRow * BLOCK_M + tidy * TM;
    const int col0 = blockCol * BLOCK_N + tidx * TN;

    // Register tile accumulators
    float acc[TM][TN];
#pragma unroll
    for (int i=0;i<TM;i++)
#pragma unroll
        for (int j=0;j<TN;j++)
            acc[i][j] = 0.0f;

    // Loop over K in BLOCK_K chunks
    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        // Cooperative load of A block: [BLOCK_M x BLOCK_K]
        for (int i = tidy; i < BLOCK_M; i += blockDim.y) {
            for (int k = tidx; k < BLOCK_K; k += blockDim.x) {
                int a_row = blockRow * BLOCK_M + i;
                int a_col = k0 + k;
                half_t val = f2h(0.0f);
                if (a_row < M && a_col < K) {
                    val = A[a_row * K + a_col];
                }
                Asub[i][k] = val;
            }
        }
        // Cooperative load of B block: [BLOCK_K x (BLOCK_N+PAD_N)]
        for (int k = tidy; k < BLOCK_K; k += blockDim.y) {
            for (int j = tidx; j < BLOCK_N; j += blockDim.x) {
                int b_row = k0 + k;
                int b_col = blockCol * BLOCK_N + j;
                half_t val = f2h(0.0f);
                if (b_row < K && b_col < N) {
                    val = B[b_row * N + b_col];
                }
                Bsub[k][j] = val;
            }
        }
        __syncthreads();

        // Compute this thread's TMxTN micro-tile
#pragma unroll
        for (int kk = 0; kk < BLOCK_K; ++kk) {
            half_t a_reg[TM];
#pragma unroll
            for (int i=0;i<TM;i++) {
                int r = tidy*TM + i;
                a_reg[i] = Asub[r][kk];
            }
            half_t b_reg[TN];
#pragma unroll
            for (int j=0;j<TN;j++) {
                int c = tidx*TN + j;
                b_reg[j] = Bsub[kk][c];
            }
#pragma unroll
            for (int i=0;i<TM;i++) {
                float a_f = h2f(a_reg[i]);
#pragma unroll
                for (int j=0;j<TN;j++) {
                    acc[i][j] += a_f * h2f(b_reg[j]);
                }
            }
        }
        __syncthreads();
    }

    // Write back
#pragma unroll
    for (int i=0;i<TM;i++) {
        int r = row0 + i;
        if (r >= M) continue;
#pragma unroll
        for (int j=0;j<TN;j++) {
            int c = col0 + j;
            if (c >= N) continue;
            C[r * N + c] = acc[i][j];
        }
    }
}

// Host reference GEMM (float)
static void gemm_ref(const std::vector<float>& Af,
                     const std::vector<float>& Bf,
                     std::vector<float>& Cf,
                     int M, int N, int K)
{
    for (int i=0;i<M;i++) {
        for (int j=0;j<N;j++) {
            float sum = 0.0f;
            for (int k=0;k<K;k++) {
                sum += Af[i*K+k] * Bf[k*N+j];
            }
            Cf[i*N+j] = sum;
        }
    }
}

int main(int argc, char** argv) {
    int M = 1024, N = 1024, K = 1024;
    if (argc == 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }

    // Derive block/grid sizes (must match TM/TN/BLOCK_M/BLOCK_N choice)
    dim3 block(16,16,1);
    dim3 grid((N + BLOCK_N - 1)/BLOCK_N, (M + BLOCK_M - 1)/BLOCK_M, 1);

    printf("GEMM: C[%d x %d] = A[%d x %d] * B[%d x %d]\n", M,N,M,K,K,N);
    printf("Tiles: BM=%d BN=%d BK=%d, TM=%d TN=%d\n", BLOCK_M,BLOCK_N,BLOCK_K,TM,TN);

    // Host allocations
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> A_hf(M*K), B_hf(K*N), C_ref(M*N), C_out(M*N);
    for (auto &x : A_hf) x = dist(rng);
    for (auto &x : B_hf) x = dist(rng);

    // Convert to half on host
    std::vector<half_t> A_h(M*K), B_h(K*N);
    for (int i=0;i<M*K;i++) A_h[i] = f2h(A_hf[i]);
    for (int i=0;i<K*N;i++) B_h[i] = f2h(B_hf[i]);

    // Device allocations
    half_t *A_d=nullptr, *B_d=nullptr;
    float *C_d=nullptr;
    DEV_ALLOC(A_d, sizeof(half_t)*M*K);
    DEV_ALLOC(B_d, sizeof(half_t)*K*N);
    DEV_ALLOC(C_d, sizeof(float)*M*N);

    MEMCPY_H2D(A_d, A_h.data(), sizeof(half_t)*M*K);
    MEMCPY_H2D(B_d, B_h.data(), sizeof(half_t)*K*N);

    // Warmup
    LAUNCH_KERNEL(gemm_tiled_shared, grid, block, 0, STREAM_DEFAULT, A_d, B_d, C_d, M,N,K);
    GET_LAST_ERROR();
    DEVICE_SYNCH();

    // Timed run
    EVENT_T e0, e1;
    EVENT_CREATE(e0);
    EVENT_CREATE(e1);
    EVENT_RECORD(e0, STREAM_DEFAULT);
    LAUNCH_KERNEL(gemm_tiled_shared, grid, block, 0, STREAM_DEFAULT, A_d, B_d, C_d, M,N,K);
    EVENT_RECORD(e1, STREAM_DEFAULT);
    EVENT_SYNCH(e1);
    float ms=0.0f;
    EVENT_ELAPSED_MS(ms, e0, e1);

    MEMCPY_D2H(C_out.data(), C_d, sizeof(float)*M*N);

    // Reference & check
    gemm_ref(A_hf, B_hf, C_ref, M,N,K);
    double max_abs = 0.0, max_rel = 0.0;
    for (int i=0;i<M*N;i++) {
        double diff = std::abs((double)C_out[i] - (double)C_ref[i]);
        max_abs = std::max(max_abs, diff);
        double denom = std::max(1e-7, std::abs((double)C_ref[i]));
        max_rel = std::max(max_rel, diff/denom);
    }

    double flops = 2.0 * (double)M * (double)N * (double)K;
    double gflops = flops / (ms * 1e6);
    printf("Time: %.3f ms, Throughput: %.2f GFLOP/s\n", ms, gflops);
    printf("Max abs err: %.3e, Max rel err: %.3e\n", max_abs, max_rel);
    printf("PASS = %s (tol abs<=1e-2 or rel<=1e-2)\n",
           (max_abs<=1e-2 || max_rel<=1e-2) ? "true":"false");

    DEV_FREE(A_d); DEV_FREE(B_d); DEV_FREE(C_d);
    return 0;
}
```

## Profiling and Validation

### NVIDIA (Nsight Compute)

Collect shared/L1/L2 efficiency and bank conflict indicators:

```
ncu --kernel-name regex:gemm_tiled_shared \
    --metrics smsp__pipe_lsu_mem_shared_op_{ld,st}.active,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_{ld,st}.sum,\
smsp__sass_average_data_bytes_per_sector_mem_shared.pct,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
dram__throughput.avg.pct_of_peak_sustained_active \
    ./tiled_gemm_shared 1024 1024 1024
```

Expectations:

- Shared load/store activity high during compute.
- `l1tex__data_bank_conflicts...` near zero (padding in Bsub).
- DRAM throughput low relative to flops (good reuse).

### AMD (rocprof / Omnitrace)

Collect LDS and wavefront stats (exact counters vary by GPU):

```
rocprof --hip-trace --stats ./tiled_gemm_shared_hip 1024 1024 1024

# With events (if available on your ASIC):
rocprof --events SQ_INSTS_LDS,TA_FLAT_READ,TA_FLAT_WRITE,GRBM_GUI_ACTIVE \
        ./tiled_gemm_shared_hip 1024 1024 1024
```

Expectations:

- Significant `SQ_INSTS_LDS` vs. flat global ops.
- Stable GUI active and healthy waves per CU.

### Pass thresholds

- Correctness: max(abs) ≤ 1e-2 **or** max(rel) ≤ 1e-2.
- Shared/LDS bank conflicts: effectively zero (≤ 1% of shared transactions conflicted).
- Achieved occupancy (Nsight Compute `sm__warps_active`): ≥ 50%.
- DRAM throughput not saturating while GFLOP/s increases vs. a naive no-tiling baseline (≥ 1.5× typical on midrange GPUs).

## Performance Checklist

- Tiling: `BLOCK_M = TM·blockDim.y`, `BLOCK_N = TN·blockDim.x`.
- Shared/LDS usage fits per-block budget with ≥2 blocks/SM(CU).
- Coalesced loads: A along K, B along N.
- Bank conflict mitigation: padding on B tile second dimension.
- Use FP32 accumulation for numerical stability; verify against CPU reference.
- Determinism: fixed RNG seed; single warmup + timed run encapsulated by events.
- Build with `-O3 -lineinfo` (CUDA) / `-O3` (HIP) for reasonable SASS/ISA.

## Troubleshooting

| Symptom                       | Likely cause                            | Fix                                                                          |
| ----------------------------- | --------------------------------------- | ---------------------------------------------------------------------------- |
| Very low GFLOP/s              | No tiling reuse or poor occupancy       | Verify tile sizes; reduce BLOCK_K or per-block smem; ensure ≥2 blocks/SM(CU) |
| High bank conflicts           | Access stride maps threads to same bank | Add/adjust padding; use 4-byte vector accesses where possible                |
| Global load inefficiency      | Non-coalesced loads                     | Reorient cooperative loads so fastest dimension is contiguous                |
| Numerical mismatch            | FP16 accumulation                       | Accumulate in FP32; check input scaling and reference                        |
| Kernel launch failure on HIP  | Too large shared memory                 | Reduce tiles or compile with appropriate LDS limits                          |
| Occupancy capped at 25%       | Register pressure                       | Reduce unroll, TM/TN; inspect compiler register report                       |
| Unstable timings              | DVFS/clock changes                      | Fix power mode; run multiple iterations and take median                      |
| Good GEMM but poor end-to-end | Downstream memory stalls                | Apply same tiling to attention kernels; overlap copies with compute          |
| Bank conflicts only on AMD    | Wavefront 64 patterns                   | Re-evaluate padding with 64-thread access; ensure modulo-32 banks are spread |
| Nsight “invalid metric”       | GPU/driver mismatch                     | Use metrics sets compatible with your compute capability                     |

## Acceptance Criteria

- Markdown explains tiling, bank conflicts, and provides numeric AI example.
- The single-source file compiles with both `nvcc` and `hipcc` and runs in seconds on a modern GPU.
- Profiling instructions provided with at least three interpretable metrics/counters.
- Checklist actionable; troubleshooting maps at least 8 symptoms to fixes.
- Correctness within stated tolerance and throughput reported (GFLOP/s).

## Further Work

- Introduce **double-buffering** in Shared/LDS to overlap global loads with compute.
- Vectorized loads/stores (`__half2`, `int4`) with alignment checks.
- Tensor-Core/MFMA variants (WMMA on NVIDIA, MFMA on AMD) with accumulator tiling.
- Apply the same tiling skeleton to **QKᵀ** and **softmax·V** attention paths with row/col-major layouts and causal masks.
- Persistent-kernel decode path with register-resident fragments and CUDA/HIP Graphs for reduced launch overhead.
