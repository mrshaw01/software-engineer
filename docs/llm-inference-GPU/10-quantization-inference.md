# Quantization for Inference

## 2. Summary

Quantization reduces numeric precision to shrink memory, increase arithmetic density, and lower bandwidth for LLM inference. This topic explains data formats (INT8, FP8), scale/zero-point schemes, and how to fuse dequantization into GEMM epilogues for minimal overhead. We provide runnable CUDA/HIP code for an INT8×INT8→INT32 GEMM with fused per-channel dequantization to FP16/FP32, plus profiling guidance. The outcome is a repeatable baseline you can extend to attention/MLP paths and KV-cache compression.

## 3. Why It Matters for LLM Inference

Prefill is GEMM-bound with large activation tensors; quantized weights reduce HBM traffic and improve tensor core utilization. Decode is KV/bandwidth-bound; quantized KV reduces cache footprint and DRAM/L2 traffic. In both regimes, dequant must be fused to avoid erasing gains. Proper per-channel scaling minimizes accuracy loss, while persistent/graph execution hides launch overhead.

## 4. Key Concepts and Formulas

### 4.1 Quantization/Dequantization

For symmetric INT8 (preferred for weights):

- Scale: $s = \frac{\max(|x|)}{127}$
- Quantize: $q = \mathrm{clip}(\mathrm{round}(x/s), -127, 127)$
- Dequantize: $\hat{x} = s \cdot q$

Asymmetric INT8 (more common for activations):

- Scale: $s = \frac{x_{\max} - x_{\min}}{255}$, zero-point: $z = \mathrm{round}(-x_{\min}/s)$
- Quantize: $q = \mathrm{clip}(\mathrm{round}(x/s) + z, 0, 255)$
- Dequantize: $\hat{x} = s \cdot (q - z)$

Error model (round-to-nearest): $e = \hat{x} - x \sim \mathcal{U}(-\tfrac{s}{2}, \tfrac{s}{2}) \Rightarrow \mathbb{E}[e^2]=\tfrac{s^2}{12}$.

### 4.2 Per-tensor vs. Per-channel

- Per-tensor scale: one $s$ per tensor. Cheapest but higher error.
- Per-channel scale: one $s$ per output channel/column (typical for GEMM B matrix). Better accuracy; store $N$ scales for $K\times N$ weight matrix.

### 4.3 Fused Dequant in GEMM Epilogues

For $C = A B + b$, with $A_q \in \mathbb{Z}^{M\times K}$, $B_q \in \mathbb{Z}^{K\times N}$, scales $s_A$ (per-tensor) and $s_{B,j}$ (per-column):

- Accumulate: $S_{ij} = \sum_{k} A_{q,ik} \cdot B_{q,kj}$ in INT32
- Dequantize+Bias: $C_{ij} = \alpha \cdot S_{ij} \cdot (s_A \cdot s_{B,j}) + b_j$ in FP16/FP32
- Optionally fuse activation $\phi(\cdot)$ (e.g., ReLU/GELU)

### 4.4 Numeric Example (KV Cache Size)

7B-class model, $L=4096$, $n_\text{layers}=32$, $n_\text{heads}=32$, $d_\text{head}=128$:
Elements for K and V across all layers/tokens:

$$
E = 32 \cdot 4096 \cdot 32 \cdot 128 \cdot 2 = 1{,}073{,}741{,}824
$$

- FP16 KV: $2.147\,\text{GB}$
- INT8 KV: $1.074\,\text{GB}$ (≈2× reduction)
  Bandwidth reductions track this ratio during decode.

## 5. GPU Deep Dive

### NVIDIA

- Warps (32 threads), SMs, Tensor Cores.
- INT8 paths: IMMA / DP4A; FP8 (E4M3/E5M2) on Hopper/Blackwell Tensor Cores.
- Prefer NHWC/row-major layouts that enable vectorized loads (`char4`, `int4`) and DP4A dot-4 accumulations.
- Accumulate in INT32; dequant in FP16/BF16 with fused bias/activation in epilogue.

### AMD

- Wavefronts (64 threads), Compute Units, MFMA/XDLOPs; LDS for tiling.
- INT8 MFMA and FP8 are supported on recent CDNA (e.g., MI2xx/MI3xx) via rocWMMA/inline MFMA.
- Use LDS tiling and coalesced `int4` loads; accumulate in INT32, fuse dequant+epilogue.

## 6. Implementation

We provide a minimal, dependency-light, **single-source** GEMM microbenchmark that:

1. Generates FP32 matrices on host,
2. Performs symmetric per-tensor quantization for $A$ and per-column for $B$,
3. Computes $C$ on GPU using INT8×INT8→INT32 with fused dequant to FP32,
4. Validates against FP32 reference and reports GOPS.

### File: `topics/10-quantization-inference/code/int8_gemm_qdq.cpp`

```cpp
// Single-source CUDA/HIP INT8 GEMM with fused dequant epilogue.
// Build (CUDA):
//   nvcc -O3 -std=c++17 -arch=sm_80 -lineinfo -x cu topics/10-quantization-inference/code/int8_gemm_qdq.cpp -o int8_qdq
// Build (ROCm):
//   hipcc -O3 -std=c++17 --offload-arch=gfx942 topics/10-quantization-inference/code/int8_gemm_qdq.cpp -o int8_qdq_hip

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <limits>
#include <cstring>

#if defined(__HIP_PLATFORM_AMD__)
  #include <hip/hip_runtime.h>
  #define cudaMalloc hipMalloc
  #define cudaFree hipFree
  #define cudaMemcpy hipMemcpy
  #define cudaMemcpyHostToDevice hipMemcpyHostToDevice
  #define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
  #define cudaDeviceSynchronize hipDeviceSynchronize
  #define cudaEvent_t hipEvent_t
  #define cudaEventCreate hipEventCreate
  #define cudaEventRecord hipEventRecord
  #define cudaEventSynchronize hipEventSynchronize
  #define cudaEventElapsedTime hipEventElapsedTime
  #define cudaGetLastError hipGetLastError
  #define cudaError_t hipError_t
  #define cudaSuccess hipSuccess
  #define cudaGetErrorString hipGetErrorString
  #define cudaStream_t hipStream_t
  #define cudaStreamCreate hipStreamCreate
  #define cudaStreamDestroy hipStreamDestroy
  #define cudaGetDeviceProperties hipGetDeviceProperties
  #define cudaDeviceProp hipDeviceProp_t
#endif

#if !defined(__HIP_PLATFORM_AMD__)
  #include <cuda_runtime.h>
  #if __CUDACC_VER_MAJOR__ >= 8
    #include <sm_61_intrinsics.h> // for __dp4a when available
  #endif
#endif

#define API_CHECK(expr) do { auto e = (expr); if (e != cudaSuccess) { \
  fprintf(stderr, "API error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::abort(); }} while(0)

static inline float randn(std::mt19937 &rng) {
  static thread_local std::normal_distribution<float> dist(0.0f, 1.0f);
  return dist(rng);
}

static void reference_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
  #pragma omp parallel for
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float acc = 0.f;
      for (int k = 0; k < K; ++k) acc += A[i*K + k] * B[k*N + j];
      C[i*N + j] = acc;
    }
  }
}

static float max_abs(const float* x, int n) {
  float m = 0.f;
  for (int i = 0; i < n; ++i) m = std::max(m, std::fabs(x[i]));
  return m;
}

static void quantize_symmetric_A(const float* A, int8_t* Aq, int M, int K, float &sA) {
  sA = max_abs(A, M*K) / 127.f;
  if (sA == 0.f) sA = 1.f; // avoid div-by-zero
  for (int i = 0; i < M*K; ++i) {
    float v = A[i] / sA;
    int q = (int) lrintf(v);
    q = std::max(-127, std::min(127, q));
    Aq[i] = (int8_t) q;
  }
}

static void quantize_symmetric_B_per_col(const float* B, int8_t* Bq, int K, int N, std::vector<float> &sB) {
  sB.resize(N);
  for (int j = 0; j < N; ++j) {
    float m = 0.f;
    for (int k = 0; k < K; ++k) m = std::max(m, std::fabs(B[k*N + j]));
    float sj = m / 127.f;
    if (sj == 0.f) sj = 1.f;
    sB[j] = sj;
    for (int k = 0; k < K; ++k) {
      float v = B[k*N + j] / sj;
      int q = (int) lrintf(v);
      q = std::max(-127, std::min(127, q));
      Bq[k*N + j] = (int8_t) q;
    }
  }
}

// One C element per thread, naive but simple. Optionally uses DP4A on NVIDIA.
__global__ void int8_gemm_fused_dequant(const int8_t* __restrict__ A, const int8_t* __restrict__ B,
                                        float* __restrict__ C,
                                        const float sA, const float* __restrict__ sB_col,
                                        int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;

  int32_t acc = 0;

  int k = 0;

#if !defined(__HIP_PLATFORM_AMD__) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
  // Use DP4A on NVIDIA for 4-way int8 dot if K multiple of 4.
  const int K4 = (K / 4) * 4;
  const int8_t* arow = A + row * K;
  const int8_t* bcol = B + col; // column-major access in row-major layout -> stride N
  for (; k < K4; k += 4) {
    int a4;
    memcpy(&a4, arow + k, 4); // pack 4 int8
    // pack 4 B bytes at offsets (k..k+3, col) -> strided by N
    int b4 =  (int)(uint8_t)(*(bcol + (k+0)*N)) |
             ((int)(uint8_t)(*(bcol + (k+1)*N)) << 8) |
             ((int)(uint8_t)(*(bcol + (k+2)*N)) << 16)|
             ((int)(uint8_t)(*(bcol + (k+3)*N)) << 24);
    acc = __dp4a(a4, b4, acc);
  }
#else
  // Portable path: scalar MAC
  for (; k < K; ++k) {
    int a = (int) A[row*K + k];
    int b = (int) B[k*N + col];
    acc += a * b;
  }
#endif
  // Tail for DP4A path if K not multiple of 4
#if !defined(__HIP_PLATFORM_AMD__) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
  for (; k < K; ++k) {
    int a = (int) A[row*K + k];
    int b = (int) B[k*N + col];
    acc += a * b;
  }
#endif

  float scale = sA * sB_col[col];
  float out = (float)acc * scale; // fused dequant epilogue; add bias/act here if desired
  C[row * N + col] = out;
}

int main(int argc, char** argv) {
  // Problem sizes (override via args M N K)
  int M = 512, N = 512, K = 1024;
  if (argc == 4) { M = std::atoi(argv[1]); N = std::atoi(argv[2]); K = std::atoi(argv[3]); }

  printf("INT8 GEMM with fused dequant: M=%d N=%d K=%d\n", M, N, K);

  std::mt19937 rng(123); // deterministic
  std::vector<float> A(M*K), B(K*N), C_ref(M*N), C_gpu(M*N);
  for (int i = 0; i < M*K; ++i) A[i] = 0.5f * randn(rng);
  for (int i = 0; i < K*N; ++i) B[i] = 0.5f * randn(rng);

  // Reference FP32
  reference_gemm(A.data(), B.data(), C_ref.data(), M, N, K);

  // Quantize
  std::vector<int8_t> Aq(M*K), Bq(K*N);
  float sA = 1.f;
  quantize_symmetric_A(A.data(), Aq.data(), M, K, sA);
  std::vector<float> sB(N);
  quantize_symmetric_B_per_col(B.data(), Bq.data(), K, N, sB);

  // Device allocations
  int8_t *dA=nullptr, *dB=nullptr;
  float *dC=nullptr, *dSB=nullptr;
  API_CHECK(cudaMalloc(&dA, (size_t)M*K*sizeof(int8_t)));
  API_CHECK(cudaMalloc(&dB, (size_t)K*N*sizeof(int8_t)));
  API_CHECK(cudaMalloc(&dC, (size_t)M*N*sizeof(float)));
  API_CHECK(cudaMalloc(&dSB, (size_t)N*sizeof(float)));
  API_CHECK(cudaMemcpy(dA, Aq.data(), (size_t)M*K, cudaMemcpyHostToDevice));
  API_CHECK(cudaMemcpy(dB, Bq.data(), (size_t)K*N, cudaMemcpyHostToDevice));
  API_CHECK(cudaMemcpy(dSB, sB.data(), (size_t)N*sizeof(float), cudaMemcpyHostToDevice));

  dim3 block(16, 16);
  dim3 grid((N + block.x - 1)/block.x, (M + block.y - 1)/block.y);

  // Warmup
  for (int w = 0; w < 10; ++w) {
    int8_gemm_fused_dequant<<<grid, block>>>(dA, dB, dC, sA, dSB, M, N, K);
  }
  API_CHECK(cudaDeviceSynchronize());
  API_CHECK(cudaMemcpy(C_gpu.data(), dC, (size_t)M*N*sizeof(float), cudaMemcpyDeviceToHost));

  // Validate
  double num=0.0, den=0.0, max_abs_err=0.0;
  for (int i = 0; i < M*N; ++i) {
    double diff = (double)C_gpu[i] - (double)C_ref[i];
    num += diff*diff;
    den += (double)C_ref[i]*(double)C_ref[i] + 1e-12;
    max_abs_err = std::max(max_abs_err, std::abs(diff));
  }
  double rel_l2 = std::sqrt(num/den);
  printf("Validation: rel_L2=%.6f, max_abs_err=%.6f\n", rel_l2, max_abs_err);

  // Benchmark
  const int iters = 50;
  cudaEvent_t t0, t1;
  API_CHECK(cudaEventCreate(&t0));
  API_CHECK(cudaEventCreate(&t1));
  API_CHECK(cudaEventRecord(t0));
  for (int it = 0; it < iters; ++it) {
    int8_gemm_fused_dequant<<<grid, block>>>(dA, dB, dC, sA, dSB, M, N, K);
  }
  API_CHECK(cudaEventRecord(t1));
  API_CHECK(cudaEventSynchronize(t1));
  float ms = 0.f;
  API_CHECK(cudaEventElapsedTime(&ms, t0, t1));
  float ms_per = ms / iters;

  // Each MAC ~ 2 ops.
  double ops = 2.0 * (double)M * (double)N * (double)K;
  double gops = ops / (ms_per * 1e6);
  printf("Time: %.3f ms  Throughput: %.1f GOPS\n", ms_per, gops);

  cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dSB);
  return (rel_l2 < 0.03) ? 0 : 1;
}
```

### Build

- CUDA:

```
nvcc -O3 -std=c++17 -arch=sm_80 -lineinfo -x cu topics/10-quantization-inference/code/int8_gemm_qdq.cpp -o int8_qdq
```

- ROCm:

```
hipcc -O3 -std=c++17 --offload-arch=gfx942 topics/10-quantization-inference/code/int8_gemm_qdq.cpp -o int8_qdq_hip
```

### Run

```
# Defaults M=512 N=512 K=1024
./int8_qdq
# or custom sizes
./int8_qdq 1024 1024 2048
```

Expected output (indicative):

```
Validation: rel_L2=0.018xxx, max_abs_err=...
Time: 1.23 ms  Throughput: 870.0 GOPS
```

## 7. Profiling and Validation

### NVIDIA

- Nsight Compute (kernel metrics):

```
ncu --target-processes all \
  --metrics sm__sass_thread_inst_executed_op_integer.sum,\
sm__inst_executed_pipe_tensor.sum,sm__warps_active.avg.pct_of_peak_sustained_active,\
l1tex__t_bytes.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed \
  ./int8_qdq
```

Interpretation:

- `*_op_integer.sum` should dominate (INT8 path).

- `warps_active` > 60% indicates decent occupancy.

- `dram__throughput.*` lower vs FP16 baseline implies bandwidth savings.

- Nsight Systems (timeline/overlap):

```
nsys profile -t cuda,osrt -o prof_int8 ./int8_qdq
```

### AMD

- rocprof:

```
rocprof --hip-trace --stats ./int8_qdq_hip
```

Track:

- VALU utilization (INT arithmetic), kernel time, HBM throughput deltas vs FP16 baseline.
- For deeper counters, use Omnitrace/rocprofiler v2 to inspect LDS/TCP traffic.

### Numerical Validation

- Program returns non-zero exit if `rel_L2 >= 0.03`. Adjust thresholds when moving to per-tensor/per-group schemes.

## 8. Performance Checklist

- Use per-channel scales for weight matrix (B) in GEMM.
- Fuse dequant (and bias/activation) into GEMM epilogue.
- Vectorize loads (`char4`/`int4`) and consider DP4A/MFMA when available.
- Ensure memory alignment (at least 16 bytes) and contiguous strides.
- Pre-allocate and reuse buffers; avoid host-device re-quant per step.
- For decode, quantize KV cache with paging; verify head-wise scales.
- Validate numerics: rel_L2 < 3% vs FP32; no NaNs post-activation.
- Determinism: fixed seeds, avoid stochastic rounding unless explicitly enabled.

## 9. Troubleshooting

| Symptom                     | Likely Cause                      | Fix                                                                                |
| --------------------------- | --------------------------------- | ---------------------------------------------------------------------------------- |
| rel_L2 > 5%                 | Per-tensor scale too coarse       | Switch to per-channel or per-group scales; clip outliers (e.g., 99.9th percentile) |
| Max error huge              | Integer overflow in INT32 acc     | Reduce K tile size or use INT32 accumulate; verify DP4A packing                    |
| Kernel slow vs FP16         | Scalar inner loop                 | Enable DP4A (CUDA) or MFMA (AMD); tile into shared memory/LDS                      |
| Bandwidth no improvement    | Dequant as separate kernel        | Fuse dequant+bias+activation into GEMM epilogue                                    |
| Artifacts in attention      | Activation mean shift             | Use asymmetric activation quant (zero-point) or SmoothQuant-style rescaling        |
| Decode tokens/sec unchanged | KV still FP16                     | Quantize KV with per-head scales; ensure paging aligns with cache line size        |
| Non-deterministic outputs   | Random seeds or atomic reductions | Fix seeds; avoid atomics; use deterministic reduction order                        |
| NaNs with FP8               | Exponent underflow/overflow       | Choose E4M3 vs E5M2 appropriately; clamp scales; use BF16 accumulators             |

## 10. Acceptance Criteria

- Code builds and runs on both CUDA 12.x (e.g., `sm_80`) and ROCm/HIP 6.x (e.g., `gfx942`).
- Validation passes with `rel_L2 < 0.03` on M=512, N=512, K=1024.
- Nsight Compute or rocprof shows dominant INT arithmetic and reduced DRAM throughput vs FP16 reference (when you instrument a comparable FP16 kernel).
- Throughput is reported in GOPS; kernel runtime within a few milliseconds for the default size.

## 11. Further Work

- Group-wise quantization (e.g., per 64 columns) to reduce scale memory.
- FP8 (E4M3/E5M2) weights/activations with BF16 accumulation on Tensor Cores/MFMA.
- Quantized attention: Q/K/V projection quant, INT8 softmax preconditioning (logits scaling), and KV-cache INT8 with dequant in dot(Q,Kᵀ).
- Outlier-aware methods (AWQ/GPTQ/RPTQ/SmoothQuant) for better accuracy at low bit-widths.
- Fused dequant + bias + activation + residual in one epilogue using cutlass/rocWMMA or custom kernels.
- Persistent-kernel decode path with quantized matvec and graph capture to minimize launch overhead.

### Notes on Precision & Stability

- Accumulate in INT32 even with INT8 operands; promote to FP16/BF16 only at epilogue.
- For softmax/layernorm, retain higher precision (FP16/BF16) or use blockwise scaling to mitigate underflow/overflow.
- When moving to FP8, prefer BF16 accumulators and test both E4M3 (more mantissa) and E5M2 (more range) for activations vs weights.

## Repository Placement

```
llm-inference/
  topics/
    10-quantization-inference/
      README.md
      code/
        int8_gemm_qdq.cpp
```
