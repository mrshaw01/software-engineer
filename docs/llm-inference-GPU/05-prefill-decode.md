# Prefill vs. Decode Performance Split

## Summary

Prefill (a.k.a. context or prompt processing) and decode (token-by-token generation) stress GPUs very differently. Prefill exploits large matrix multiplications and sequence-wide attention, yielding high arithmetic intensity and excellent GPU utilization. Decode exposes limited parallelism per new token and becomes memory-bandwidth and latency bound due to repeated KV-cache reads. This note quantifies both paths, gives back-of-envelope math, and provides a runnable CUDA/HIP microbenchmark that separates compute-bound prefill from bandwidth-bound decode.

## Why It Matters for LLM Inference

- **Latency:** User-visible latency during generation is dominated by decode step time per token.
- **Throughput:** Prefill can be batched and parallelized; decode faces inherent sequential dependence.
- **Capacity planning:** KV-cache bandwidth and memory footprint dominate decode scaling, while FLOP capacity dominates prefill.
- **Optimization focus:** Prefill → GEMM fusion and math throughput; Decode → KV layout, paging, graphs/persistent kernels, quantized cache.

## Key Concepts and Formulas

Let:

- Layers $L$, hidden size $H$, heads $h$, KV-heads $h_{kv}$ (GQA), head dim $d=H/h$,
- Sequence length $T$, batch $B$ (per-sequence analysis uses $B=1$),
- Element size $s$ bytes (BF16/FP16 $s=2$, FP8 $s=1$).

### Prefill FLOPs (single forward of $T$ tokens, 1 sequence)

- Linear projections and MLP (approx):

  $$
  \text{FLOPs}_{\text{lin+MLP}} \approx L \cdot \left( (6H^2) + (2H^2) + (16H^2) \right)
  = 24L H^2
  $$

  (QKV: $6H^2$, OUT: $2H^2$, MLP (GELU/SiLU 2-proj) ≈ $16H^2$).

- Attention scores $QK^\top$ and apply to $V$ (Flash/MHA math intact):

  $$
  \text{FLOPs}_{\text{attn, prefill}} \approx L \cdot 2T^2 d \cdot h
  $$

- **Total prefill:**

  $$
  \text{FLOPs}_{\text{prefill}} \approx 24L H^2 + 2L T^2 d h
  $$

### Decode FLOPs per generated token

- Linear/MLP cost is nearly the same as prefill per token:

  $$
  \text{FLOPs}_{\text{lin+MLP, decode}} \approx 24L H^2
  $$

- Attention with accumulated cache (scores + apply):

  $$
  \text{FLOPs}_{\text{attn, decode}} \approx L \cdot 4 T d \cdot h
  $$

- **Total decode per token:**

  $$
  \text{FLOPs}_{\text{decode/token}} \approx 24L H^2 + 4L T d h
  $$

### KV-cache bytes per decoded token

Per layer, reading $K$ and $V$ for all past tokens:

$$
\text{Bytes}_{\text{KV, per layer}} \approx 2 \cdot T \cdot d \cdot h_{kv} \cdot s
$$

All layers:

$$
\text{Bytes}_{\text{KV, token}} \approx 2 L T d h_{kv} s
$$

### Numeric example (7B-class model)

Assume $L{=}32$, $H{=}4096$, $h{=}32\Rightarrow d{=}128$, $h_{kv}{=}8$, $T{=}2048$, BF16 $s{=}2$ bytes.

- **Prefill FLOPs**:

  - Lin+MLP: $24LH^2 = 24\times32\times 4096^2 = 24\times32\times 16{,}777{,}216 = 12{,}884{,}901{,}376$ ≈ **12.88 GFLOPs**
  - Attention: $2LT^2dh = 2\times32\times 2048^2 \times 128 \times 32 = 1{,}112{,}396{,}544{,}288$ ≈ **1.112 TFLOPs**
  - **Total prefill** ≈ **1.125 TFLOPs** for the whole 2,048-token context (≈ **0.543 GFLOPs/token** on average).

- **Decode FLOPs per token**:

  - Lin+MLP: $24LH^2 = 12.88 \text{ GFLOPs}$
  - Attention: $4LTdh = 4\times32\times 2048\times 128 \times 32 = 33{,}554{,}432$ ≈ **0.0336 GFLOPs**
  - **Total decode** ≈ **12.91 GFLOPs/token**

- **KV bytes per token**:

  $$
  2LTdh_{kv}s = 2\times 32\times 2048\times 128\times 8\times 2
  = 268{,}435{,}456\ \text{bytes} \approx \mathbf{256\ MiB/token}
  $$

Interpretation: prefill is **compute-heavy**; decode is **memory-heavy**. With \~256 MiB of cache traffic per token, a 1.5 TB/s HBM link caps at ≈ 5.7 k tokens/s if fully isolated, and real systems achieve far less due to concurrency, scheduling, and non-KV work.

## GPU Deep Dive

### NVIDIA

- **Prefill:** High occupancy and tensor core utilization; large GEMMs saturate SM tensor pipes. Fuse bias/activation and use CUTLASS/cuBLASLt epilogues for fewer global writes.
- **Decode:** Small-batch matvecs + long linear scans of KV. Primary limits: DRAM bandwidth, L2 hit rate, and kernel launch overhead. Use CUDA Graphs or persistent kernels. Align KV to 128B, interleave by head, and prefer SoA for coalesced reads.

### AMD

- **Prefill:** MFMA/XDLOPs deliver peak throughput; LDS tiling critical to keep VALU fed. Use rocWMMA/Composable-Kernel or rocBLAS with fused epilogues.
- **Decode:** KV scans pressure L2/HBM. Layout and paging via fine-grained HIP allocations or pooled allocators; persistent kernels with wavefront-friendly striding and async copies (`cp.async`-like on CDNA3) improve overlap.

## Implementation

A minimal, dependency-light microbenchmark that:

- Models **prefill** as compute-intensive FMAs over a $[T\times H]$ tensor.
- Models **decode** as bandwidth-intensive sequential scans over a $[T\times d\times h_{kv}]$ KV cache for one token.

It builds with either `nvcc` (CUDA) or `hipcc` (ROCm) from the same source.

**File:** `topics/05-prefill-decode/code/prefill_decode_bench.cu`

```cpp
// Single-source CUDA/HIP microbench for prefill vs decode.
// Build (CUDA): nvcc -O3 -std=c++17 -arch=${SM_ARCH} -lineinfo prefill_decode_bench.cu -o bench
// Build (ROCm): hipcc -O3 -std=c++17 --offload-arch=${GFX_ARCH} prefill_decode_bench.cu -o bench

#if defined(__HIP_PLATFORM_AMD__)
  #include <hip/hip_runtime.h>
  #define DEVFN __global__
  #define API_CHECK(x) do { auto e=(x); if(e!=hipSuccess){fprintf(stderr,"HIP err %d at %s:%d\n",e,__FILE__,__LINE__); abort();} } while(0)
  #define LAUNCH(kernel,grid,block,shmem,stream,...) hipLaunchKernelGGL(kernel,grid,block,shmem,stream,__VA_ARGS__)
  using stream_t = hipStream_t;
  using event_t  = hipEvent_t;
  inline void create_event(event_t* e){ API_CHECK(hipEventCreate(e)); }
  inline void record_event(event_t e, stream_t s){ API_CHECK(hipEventRecord(e,s)); }
  inline float elapsed_ms(event_t a, event_t b){ float ms; API_CHECK(hipEventElapsedTime(&ms,a,b)); return ms; }
  inline void sync(stream_t s){ API_CHECK(hipStreamSynchronize(s)); }
  inline void dev_malloc(void** p, size_t n){ API_CHECK(hipMalloc(p,n)); }
  inline void dev_free(void* p){ API_CHECK(hipFree(p)); }
  inline void h2d(void* d, const void* h, size_t n){ API_CHECK(hipMemcpy(d,h,n,hipMemcpyHostToDevice)); }
#else
  #include <cuda_runtime.h>
  #define DEVFN __global__
  #define API_CHECK(x) do { auto e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA err %d at %s:%d\n",e,__FILE__,__LINE__); abort();} } while(0)
  #define LAUNCH(kernel,grid,block,shmem,stream,...) kernel<<<grid,block,shmem,stream>>>(__VA_ARGS__)
  using stream_t = cudaStream_t;
  using event_t  = cudaEvent_t;
  inline void create_event(event_t* e){ API_CHECK(cudaEventCreate(e)); }
  inline void record_event(event_t e, stream_t s){ API_CHECK(cudaEventRecord(e,s)); }
  inline float elapsed_ms(event_t a, event_t b){ float ms; API_CHECK(cudaEventElapsedTime(&ms,a,b)); return ms; }
  inline void sync(stream_t s){ API_CHECK(cudaStreamSynchronize(s)); }
  inline void dev_malloc(void** p, size_t n){ API_CHECK(cudaMalloc(p,n)); }
  inline void dev_free(void* p){ API_CHECK(cudaFree(p)); }
  inline void h2d(void* d, const void* h, size_t n){ API_CHECK(cudaMemcpy(d,h,n,cudaMemcpyHostToDevice)); }
#endif

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

struct Params {
  int L = 32;       // layers
  int H = 4096;     // hidden size
  int h = 32;       // attention heads
  int h_kv = 8;     // KV heads (GQA)
  int T = 2048;     // sequence length for prefill / past length for decode
  int iters_fma = 256; // compute stretch for prefill
};

__host__ __device__ inline int div_up(int a,int b){ return (a+b-1)/b; }

// Compute-heavy "prefill" kernel: per element performs many FMAs to raise intensity.
DEVFN void prefill_fma_kernel(const float* __restrict__ X,
                              const float* __restrict__ W,
                              float* __restrict__ Y,
                              int n_elems, int iters_fma) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_elems) return;
  float x = X[i];
  float w = W[i % 1024]; // small weight tile to simulate reuse
  float acc = 0.f;
  #pragma unroll 4
  for (int k = 0; k < iters_fma; ++k) {
    acc = fmaf(x, w, acc);
    // Light data dependency to avoid compiler removing work
    w = fmaf(w, 1.00001f, 0.0001f);
  }
  Y[i] = acc;
}

// Bandwidth-heavy "decode" kernel: stream through K and V once per layer.
DEVFN void decode_kv_scan_kernel(const float* __restrict__ K,
                                 const float* __restrict__ V,
                                 const float* __restrict__ q,
                                 float* __restrict__ out,
                                 int T, int d, int h_kv) {
  // One thread processes one (head, dim) lane; simple striding to cover all.
  int lane = blockIdx.x * blockDim.x + threadIdx.x;
  int lanes = h_kv * d;
  if (lane >= lanes) return;
  int head = lane / d;
  int dim  = lane % d;

  const float* K_base = K + head * (size_t)T * d + dim;
  const float* V_base = V + head * (size_t)T * d + dim;
  float qv = q[head * d + dim];
  float acc = 0.f;

  // Stream over time; two global reads (K,V) per step.
  for (int t = 0; t < T; ++t) {
    float k = K_base[t * d];
    float v = V_base[t * d];
    // Minimal math per byte -> memory bound
    float score = qv * k;
    acc = fmaf(score, v, acc);
  }
  out[head * d + dim] = acc;
}

int main(int argc, char** argv) {
  Params p;
  if (argc == 7) {
    p.L = std::atoi(argv[1]);
    p.H = std::atoi(argv[2]);
    p.h = std::atoi(argv[3]);
    p.h_kv = std::atoi(argv[4]);
    p.T = std::atoi(argv[5]);
    p.iters_fma = std::atoi(argv[6]);
  }

  const int d = p.H / p.h;
  const size_t n_prefill = (size_t)p.T * p.H;
  const size_t n_kv = (size_t)p.T * d * p.h_kv;

  // Host buffers
  std::vector<float> h_X(n_prefill, 1.0f), h_W(1024, 0.5f), h_Y(n_prefill, 0.f);
  std::vector<float> h_K(n_kv, 1.0f), h_V(n_kv, 1.0f), h_q(p.h_kv * d, 0.1f), h_out(p.h_kv * d, 0.f);

  // Device buffers
  float *X, *W, *Y, *K, *V, *q, *out;
  dev_malloc((void**)&X,  n_prefill*sizeof(float));
  dev_malloc((void**)&W,  h_W.size()*sizeof(float));
  dev_malloc((void**)&Y,  n_prefill*sizeof(float));
  dev_malloc((void**)&K,  n_kv*sizeof(float));
  dev_malloc((void**)&V,  n_kv*sizeof(float));
  dev_malloc((void**)&q,  h_q.size()*sizeof(float));
  dev_malloc((void**)&out,h_out.size()*sizeof(float));
  h2d(X,  h_X.data(), n_prefill*sizeof(float));
  h2d(W,  h_W.data(), h_W.size()*sizeof(float));
  h2d(K,  h_K.data(), n_kv*sizeof(float));
  h2d(V,  h_V.data(), n_kv*sizeof(float));
  h2d(q,  h_q.data(), h_q.size()*sizeof(float));

  stream_t stream; API_CHECK(
  #if defined(__HIP_PLATFORM_AMD__)
    hipStreamCreate(&stream)
  #else
    cudaStreamCreate(&stream)
  #endif
  );

  // --- Prefill timing ---
  event_t p_start, p_stop; create_event(&p_start); create_event(&p_stop);
  record_event(p_start, stream);
  {
    int threads = 256;
    int blocks = div_up((int)n_prefill, threads);
    for (int l = 0; l < p.L; ++l) {
      LAUNCH(prefill_fma_kernel, blocks, threads, 0, stream, X, W, Y, (int)n_prefill, p.iters_fma);
    }
  }
  record_event(p_stop, stream);
  sync(stream);
  float prefill_ms = elapsed_ms(p_start, p_stop);

  // --- Decode timing (per token) ---
  event_t d_start, d_stop; create_event(&d_start); create_event(&d_stop);
  record_event(d_start, stream);
  {
    int threads = 256;
    int lanes = p.h_kv * d;
    int blocks = div_up(lanes, threads);
    for (int l = 0; l < p.L; ++l) {
      LAUNCH(decode_kv_scan_kernel, blocks, threads, 0, stream, K, V, q, out, p.T, d, p.h_kv);
    }
  }
  record_event(d_stop, stream);
  sync(stream);
  float decode_ms = elapsed_ms(d_start, d_stop);

  // Simple correctness guards
  float guard = 0.f;
  for (float v : h_out) guard += v;
  printf("Guard (sum host-out before copy) = %f (expected 0.000000)\n", guard);
  // Copy one small result to validate non-nan
  API_CHECK(
  #if defined(__HIP_PLATFORM_AMD__)
    hipMemcpy(h_out.data(), out, h_out.size()*sizeof(float), hipMemcpyDeviceToHost)
  #else
    cudaMemcpy(h_out.data(), out, h_out.size()*sizeof(float), cudaMemcpyDeviceToHost)
  #endif
  );
  float chk = 0.f; for (float v: h_out) chk += v;
  if (!std::isfinite(chk)) { fprintf(stderr,"Decode output invalid.\n"); return 1; }

  // Estimate bytes moved by decode KV (float here = 4B; adjust for BF16/FP8 when mapping)
  double bytes_decode = (double)p.L * (double)p.T * (double)d * (double)p.h_kv * 2 /*K+V*/ * sizeof(float);
  double gbps = (bytes_decode / (decode_ms/1e3)) / 1e9;

  printf("Params: L=%d H=%d h=%d h_kv=%d d=%d T=%d iters_fma=%d\n",
         p.L,p.H,p.h,p.h_kv,d,p.T,p.iters_fma);
  printf("Prefill time: %.3f ms for full context -> avg tokens/s (prefill) ≈ %.2f\n",
         prefill_ms, (double)p.T / (prefill_ms/1e3));
  printf("Decode time (per token): %.3f ms -> tokens/s (decode) ≈ %.2f\n",
         decode_ms, 1000.0 / decode_ms);
  printf("Decode effective bandwidth (float32 model of KV): %.2f GB/s (note: BF16 halves bytes).\n", gbps);

  dev_free(X); dev_free(W); dev_free(Y);
  dev_free(K); dev_free(V); dev_free(q); dev_free(out);
  return 0;
}
```

### Build and Run

```bash
# CUDA
nvcc -O3 -std=c++17 -arch=${SM_ARCH} -lineinfo topics/05-prefill-decode/code/prefill_decode_bench.cu -o bench_cuda
./bench_cuda            # defaults (L=32,H=4096,h=32,h_kv=8,T=2048,iters_fma=256)
./bench_cuda 32 4096 32 8 4096 512  # heavier run

# ROCm
hipcc -O3 -std=c++17 --offload-arch=${GFX_ARCH} topics/05-prefill-decode/code/prefill_decode_bench.cu -o bench_hip
./bench_hip
```

Notes:

- The decode kernel reports **float32** bandwidth; for BF16 KV-cache, halve the byte count.
- Increase `iters_fma` to push prefill further into compute-bound territory.
- Increase `T` to raise decode KV traffic linearly.

## Profiling and Validation

### NVIDIA

- **Nsight Systems (launch overhead, graphs):**

  ```bash
  nsys profile --stats=true ./bench_cuda
  ```

  Look for many short kernels in decode; consider CUDA Graphs.

- **Nsight Compute (key counters):**

  ```bash
  ncu --set full --kernels ::prefill_fma_kernel --target-processes all ./bench_cuda
  ncu --set full --kernels ::decode_kv_scan_kernel --target-processes all ./bench_cuda
  ```

  Inspect:

  - `sm__throughput.avg.pct_of_peak_sustained_elapsed`: high in prefill, lower in decode.
  - `dram__throughput.avg.pct_of_peak_sustained_elapsed` and `lts__t_sectors_srcunit_tex_op_read.sum`: high in decode.
  - `launch__occupancy_limit_active_warps` / `smsp__warps_active.avg.pct_of_peak_sustained_active`: lower in decode.

### AMD

- **rocprof (timeline + counters):**

  ```bash
  rocprof --timestamp on --hip-trace --hsa-trace ./bench_hip
  rocprof --stats --hsa-trace --hip-trace --kernels decode_kv_scan_kernel ./bench_hip
  ```

  Inspect:

  - Memory: `TC_BYTES_[RD|WR]` or derived bandwidth; high in decode.
  - Compute: `SQ_INSTS_VALU`, `SQ_WAVES` high in prefill.
  - Cache: high L2 miss rate during decode scans indicates layout issues.

### Validation

- Prefill: tensor/multiply utilization should be substantially higher than DRAM utilization.
- Decode: DRAM/L2 throughput should approach device limits while SM/VALU utilization lags.

## Performance Checklist

- [ ] **KV layout**: contiguous by head and time; 128-byte aligned base and strides.
- [ ] **Element size**: BF16/FP8 for KV; validate numerical stability of softmax and LN with FP32 accumulation.
- [ ] **Allocator**: preallocate KV pools; avoid per-token alloc/free.
- [ ] **Graphs/Persistent**: capture decode step into CUDA/HIP Graphs or use persistent kernels to amortize launch overhead.
- [ ] **Batching**: micro-batch decode across sequences where latency budget allows; avoid over-batching that harms tail latency.
- [ ] **Fused epilogues**: prefill GEMMs with fused bias/activation/quant-dequant.
- [ ] **Quantized KV**: INT8/FP8 KV with on-the-fly dequant in attention block.
- [ ] **Paging**: Paged KV to keep working set hot; avoid TLB thrash and oversubscription.

## Troubleshooting (Symptom → Likely Cause → Fix)

- **Low tokens/s (decode)** → KV reads uncoalesced → Reorder to SoA by head/time; pad to 128B.
- **High kernel launch overhead** → Many small decode kernels → Use CUDA/HIP Graphs; persistent kernels.
- **HBM below 50% peak in decode** → L2 misses + bank conflicts → Align/pad strides; use larger request sizes; prefetch.
- **NaNs in attention** → Low-precision accumulators → Accumulate in FP32; clamp logits; use fused softmax.
- **Prefill underutilized** → GEMM shapes suboptimal → Use tuned libraries (cuBLASLt/rocBLAS) or CUTLASS/CK with correct tile shapes.
- **Out-of-memory** → KV cache too large → Reduce $T$, use GQA (smaller $h_{kv}$), quantize KV, enable paging/eviction.
- **Long TTFT** → Serialization in input pipeline → Overlap tokenization/IO with GPU compute; warm kernels with graphs.
- **Jittery latency** → Power/clock throttling or memory contention → Lock clocks, isolate processes, NUMA-pin host threads.

## Acceptance Criteria

- Code builds with `nvcc` and `hipcc` and completes in seconds on a modern GPU.
- Prefill kernel shows significantly higher compute utilization than memory utilization.
- Decode kernel shows significantly higher memory throughput than compute utilization.
- Reported decode “GB/s” is within 30–80% of device peak in an isolated run (heuristic).
- Changing $T$ changes decode time approximately linearly; changing `iters_fma` changes prefill time approximately linearly.

## Further Work

- Replace prefill FMAs with real GEMMs (CUTLASS/rocBLAS) and fused epilogues.
- Implement attention with softmax and masking; measure numerical stability across FP16/BF16/FP8.
- Add KV paging benchmark with varying page sizes and L2 residency tracking.
- Add CUDA/HIP Graph versions and a persistent kernel variant for decode.
- Extend to multi-GPU (NCCL/RCCL) with tensor/pipeline parallelism and overlap of all-reduce with compute.

**Repository placement**

```
llm-inference/
  topics/
    05-prefill-decode/
      README.md                 # this file
      code/
        prefill_decode_bench.cu
```

**Quick sanity target**

- Default parameters approximate a 7B-class model and demonstrate the compute-vs-bandwidth split without external libraries. Adjust $T$ and `iters_fma` to observe slope changes immediately.
