# KV Cache Sizing, Bandwidth & Layout

## Summary

Key–Value (KV) cache dominates memory footprint and bandwidth during autoregressive LLM inference, especially in the decode path. This note provides exact sizing formulas, back‑of‑envelope bandwidth limits, and practical layout choices that maximize HBM efficiency on NVIDIA (CUDA) and AMD (ROCm/HIP) GPUs. A small, runnable benchmark is included to measure read throughput for different layouts using vectorized global loads. The outcome is a reproducible method to size the cache, estimate tokens/sec ceilings, and adopt a layout that sustains near-peak DRAM bandwidth.

## Why It Matters for LLM Inference

- **Prefill:** writes K and V once per position; bandwidth is amortized across many tokens.
- **Decode:** for each new token, every layer rereads all **past** K and V (O(T)), making decode strongly **memory‑bound**. Optimizing KV layout and alignment directly increases tokens/sec.

## Key Concepts and Formulas

Let

- `L` = number of transformer layers
- `B` = batch size (sequences)
- `T_max` = maximum cached sequence length (context capacity)
- `T_ctx` = current context length during decode (≤ T_max)
- `H` = attention heads (queries)
- `H_kv` = KV heads (MQA/GQA: `H_kv ≤ H`)
- `D_h` = head dimension (per head)
- `s` = bytes per element (FP16/BF16: 2; FP8/INT8: 1)

### KV cache size (bytes)

For separate K and V arrays:

```
Size_KV = B · L · T_max · (2 · H_kv · D_h · s)
```

**Example (numerically checked):** `B=1, L=32, H_kv=8, D_h=128, T_max=4096, s=2` →

```
Size_KV = 1 · 32 · 4096 · (2 · 8 · 128 · 2) = 536,870,912 bytes ≈ 0.5 GiB
```

Compare:

- Full multi‑head KV (`H_kv=32`): 2.0 GiB
- MQA (`H_kv=1`): 0.0625 GiB

### Per‑token decode traffic (bytes)

Each new token at context `T_ctx`:

- Read all prior K and V per layer: `Read = L · (2 · H_kv · D_h · T_ctx · s)`
- Write current token’s K and V per layer: `Write = L · (2 · H_kv · D_h · s)`

**Example:** `L=32, H_kv=8, D_h=128, s=2, T_ctx=4095` →

```
Read  = 32 · (2 · 8 · 128 · 4095 · 2) = 536,739,840 bytes ≈ 0.537 GB
Write = 32 · (2 · 8 · 128 · 2)       =     131,072 bytes ≈ 0.131 MB
```

Decode is dominated by reads (>99.9%). A peak‑bandwidth roofline for tokens/sec is:

```
TPS_bw ≈ BW_peak / Read
```

For `BW_peak = 1.6 TB/s`: `TPS_bw ≈ 1600 GB/s / 0.537 GB ≈ 2980 tok/s` (upper bound; compute/overheads reduce this).

### Arithmetic intensity (per layer, per token)

Approximate FLOPs for attention (ignoring small terms):

```
F ≈ 2 · H · D_h · T_ctx           (Q·K^T)
  + 2 · H · D_h · T_ctx           (Attn·V)
  = 4 · H · D_h · T_ctx
```

Bytes read per layer: `B_layer = 2 · H_kv · D_h · T_ctx · s`.

**Example:** `H=32, H_kv=8, D_h=128, T_ctx=4096, s=2` →

```
F = 4 · 32 · 128 · 4096 = 67,108,864 FLOPs
B_layer ≈ 2 · 8 · 128 · 4096 · 2 = 16,777,216 bytes
AI = F / B_layer ≈ 4 FLOPs/byte  → strongly memory‑bound
```

Reducing `H_kv` (GQA/MQA) lowers bytes but keeps FLOPs (per `H`) nearly unchanged → higher arithmetic intensity and better tokens/sec.

## GPU Deep Dive

### NVIDIA (CUDA)

- Warp size 32; favor 128‑byte sector‑aligned, fully coalesced global transactions.
- Ensure `D_h · s` is padded to a multiple of 128 bytes for vectorized loads (`int4`/`float4` or `__half2` pairs). Prefer SoA layouts that minimize strided walks across heads.
- L2 is large but not enough for long contexts. Expect low L2 hit rates at large `T_ctx`; the goal is sustained HBM bandwidth.
- Use `cudaMallocAsync` with a pool for stable pointer addresses; preallocate KV slabs.

### AMD (ROCm/HIP)

- Wavefront 64; align accesses so contiguous lanes read contiguous 16‑byte chunks.
- LDS (shared memory) is limited; focus on global memory coalescing and MFMA‑backed GEMMs for Q/K/V projections.
- Use `hipMallocAsync` and a pool allocator where available; ensure page‑friendly strides to play well with RCCL and migration.

## Implementation

We provide a single‑source CUDA/HIP micro‑benchmark that allocates K and V caches and measures decode‑like streaming reads across layouts.

**Layouts** (row‑major, last index contiguous):

- `layout=0`: `[L][B][H_kv][T_max][D_h]` (KV‑head major)
- `layout=1`: `[L][B][T_max][H_kv][D_h]` (time major)

Both use `D_h` contiguous; the stride between successive `t` differs, which changes coalescing for decode reads.

### File: `topics/04-kv-cache-sizing-bandwidth-layout/code/kv_layout_bench.cu`

```cpp
// Single-source CUDA/HIP benchmark for KV cache streaming reads.
// Build (CUDA): nvcc -O3 -std=c++17 -arch=${SM_ARCH} kv_layout_bench.cu -o kv_bench
// Build (ROCm): hipcc -O3 -std=c++17 --offload-arch=${GFX_ARCH} kv_layout_bench.cu -o kv_bench

#if defined(__HIP_PLATFORM_AMD__)
  #include <hip/hip_runtime.h>
  #define API_CHECK(x) do { auto _e=(x); if(_e!=hipSuccess){fprintf(stderr,"HIP err %d at %s:%d\n",_e,__FILE__,__LINE__); abort();} } while(0)
  #define DEVFN __global__
  #define DEV_MEMCPY hipMemcpy
  #define DEV_EVENT hipEvent_t
  #define EVENT_CREATE hipEventCreate
  #define EVENT_RECORD hipEventRecord
  #define EVENT_SYNC hipEventSynchronize
  #define EVENT_ELAPSED hipEventElapsedTime
  #define DEV_DEVICE_SYNC hipDeviceSynchronize
  #define DEV_MALLOC hipMalloc
  #define DEV_FREE hipFree
  #define DEV_MEMSET hipMemset
  #define DEV_GET_LAST_ERROR hipGetLastError
  #define DEV_STR "HIP"
#else
  #include <cuda_runtime.h>
  #define API_CHECK(x) do { auto _e=(x); if(_e!=cudaSuccess){fprintf(stderr,"CUDA err %d at %s:%d\n",_e,__FILE__,__LINE__); abort();} } while(0)
  #define DEVFN __global__
  #define DEV_MEMCPY cudaMemcpy
  #define DEV_EVENT cudaEvent_t
  #define EVENT_CREATE cudaEventCreate
  #define EVENT_RECORD cudaEventRecord
  #define EVENT_SYNC cudaEventSynchronize
  #define EVENT_ELAPSED cudaEventElapsedTime
  #define DEV_DEVICE_SYNC cudaDeviceSynchronize
  #define DEV_MALLOC cudaMalloc
  #define DEV_FREE cudaFree
  #define DEV_MEMSET cudaMemset
  #define DEV_GET_LAST_ERROR cudaGetLastError
  #define DEV_STR "CUDA"
#endif

#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cstring>

struct Args {
  int L=32, B=1, Hkv=8, Dh=128;  // topology
  int T_max=4096, T_ctx=4095;    // context
  int layout=0;                  // 0:[L][B][Hkv][T][D], 1:[L][B][T][Hkv][D]
  int bytes=2;                   // element size (2=FP16/BF16)
  int iters=5;                   // timing repeats
};

static inline size_t ceil_div(size_t a, size_t b){ return (a + b - 1)/b; }

// Return byte offset for (l,b,hkv,t,d) in the chosen layout.
__host__ __device__ inline size_t offset_bytes(const Args a, size_t l, size_t b, size_t hkv, size_t t, size_t d){
  if(a.layout==0){ // [L][B][Hkv][T][D]
    size_t idx = ((((size_t)l*a.B + b)*a.Hkv + hkv)*a.T_max + t)*a.Dh + d;
    return idx * (size_t)a.bytes;
  } else {         // [L][B][T][Hkv][D]
    size_t idx = ((((size_t)l*a.B + b)*a.T_max + t)*a.Hkv + hkv)*a.Dh + d;
    return idx * (size_t)a.bytes;
  }
}

// Kernel: each block processes one (layer, hkv) tile; threads stream 16B chunks (int4) across T_ctx for both K and V.
DEVFN void kv_stream_read(const uint8_t* __restrict__ K, const uint8_t* __restrict__ V, Args a,
                          uint64_t* __restrict__ sink){
  const int l = blockIdx.x / a.Hkv;      // layer
  const int hkv = blockIdx.x % a.Hkv;    // kv-head
  const int b = 0;                       // batch fixed to 1 for simplicity

  // Per-step bytes for one (l,hkv): D_h * bytes
  const size_t step_bytes = (size_t)a.Dh * (size_t)a.bytes;
  // How many 16B chunks per step per array (K or V):
  const size_t chunks_per_step = step_bytes / 16; // assume Dh*bytes multiple of 16

  uint64_t acc = 0;
  for(int t=0; t<a.T_ctx; ++t){
    // Base offsets for K and V at this (l,b,hkv,t)
    size_t baseKb = offset_bytes(a, l, b, hkv, t, 0);
    size_t baseVb = baseKb; // identical layout for V

    const int4* __restrict__ K4 = reinterpret_cast<const int4*>(K + baseKb);
    const int4* __restrict__ V4 = reinterpret_cast<const int4*>(V + baseVb);

    for(size_t c = threadIdx.x; c < chunks_per_step; c += blockDim.x){
      int4 k = K4[c];
      int4 v = V4[c];
      acc += (uint64_t)k.x + k.y + k.z + k.w + v.x + v.y + v.z + v.w;
    }
  }

  // Reduce acc within block (naive XOR to avoid optimizing away)
  if(threadIdx.x==0){ sink[blockIdx.x] = acc ^ (uint64_t)(l*1315423911u + hkv*2654435761u); }
}

int main(int argc, char** argv){
  Args a;
  for(int i=1;i<argc;i++){
    if(!strncmp(argv[i],"--L=",4)) a.L=atoi(argv[i]+4);
    else if(!strncmp(argv[i],"--B=",4)) a.B=atoi(argv[i]+4);
    else if(!strncmp(argv[i],"--Hkv=",6)) a.Hkv=atoi(argv[i]+6);
    else if(!strncmp(argv[i],"--Dh=",5)) a.Dh=atoi(argv[i]+5);
    else if(!strncmp(argv[i],"--Tmax=",7)) a.T_max=atoi(argv[i]+7);
    else if(!strncmp(argv[i],"--Tctx=",7)) a.T_ctx=atoi(argv[i]+7);
    else if(!strncmp(argv[i],"--layout=",9)) a.layout=atoi(argv[i]+9);
    else if(!strncmp(argv[i],"--bytes=",8)) a.bytes=atoi(argv[i]+8);
    else if(!strncmp(argv[i],"--iters=",8)) a.iters=atoi(argv[i]+8);
  }

  if(a.B!=1){ fprintf(stderr,"B>1 not yet implemented in this microbench\n"); return 1; }
  if(((size_t)a.Dh*(size_t)a.bytes) % 16 != 0){
    fprintf(stderr,"Dh*bytes must be multiple of 16 for int4 vector loads.\n");
    return 2;
  }

  const size_t per_token_per_lh_bytes = (size_t)a.Dh * (size_t)a.bytes; // for K or V
  const size_t per_lh_bytes = (size_t)a.T_max * per_token_per_lh_bytes;  // for K or V
  const size_t tiles = (size_t)a.L * (size_t)a.Hkv;                      // layer*kvhead tiles

  const size_t bytes_one = tiles * per_lh_bytes;      // one of {K,V}
  const size_t total_bytes_alloc = 2*bytes_one;       // K + V

  uint8_t *dK=nullptr, *dV=nullptr; uint64_t *dSink=nullptr;
  API_CHECK( DEV_MALLOC(&dK, bytes_one) );
  API_CHECK( DEV_MALLOC(&dV, bytes_one) );
  API_CHECK( DEV_MALLOC(&dSink, tiles*sizeof(uint64_t)) );
  API_CHECK( DEV_MEMSET(dK, 1, bytes_one) );
  API_CHECK( DEV_MEMSET(dV, 2, bytes_one) );

  DEV_EVENT evStart, evStop; EVENT_CREATE(&evStart); EVENT_CREATE(&evStop);

  dim3 grid((unsigned)tiles);
  dim3 block(256);

  // Warmup
  kv_stream_read<<<grid, block>>>(dK, dV, a, dSink);
  API_CHECK( DEV_DEVICE_SYNC() );

  float msAccum=0.0f;
  for(int it=0; it<a.iters; ++it){
    EVENT_RECORD(evStart, 0);
    kv_stream_read<<<grid, block>>>(dK, dV, a, dSink);
    EVENT_RECORD(evStop, 0);
    EVENT_SYNC(evStop);
    float ms=0.0f; EVENT_ELAPSED(&ms, evStart, evStop);
    msAccum += ms;
  }

  const double msAvg = msAccum / a.iters;
  // Total bytes read during kernel: for each (L,Hkv) tile, we read T_ctx·(Dh·bytes) from K and same from V.
  const double bytes_read = (double)tiles * (double)a.T_ctx * (double)per_token_per_lh_bytes * 2.0;
  const double gbps = (bytes_read / 1e9) / (msAvg / 1e3);

  printf("Backend=%s layout=%d L=%d Hkv=%d Dh=%d Tctx=%d bytes=%d\n", DEV_STR, a.layout, a.L, a.Hkv, a.Dh, a.T_ctx, a.bytes);
  printf("Alloc(K+V)=%.3f GiB, BytesRead=%.3f GB, Time=%.3f ms, Throughput=%.1f GB/s\n",
         total_bytes_alloc / 1073741824.0, bytes_read/1e9, msAvg, gbps);

  // Rudimentary check to keep the compiler from removing loads
  std::vector<uint64_t> host(tiles);
  API_CHECK( DEV_MEMCPY(host.data(), dSink, tiles*sizeof(uint64_t), cudaMemcpyDeviceToHost) );
  uint64_t xorv=0; for(size_t i=0;i<tiles;i++) xorv ^= host[i];
  printf("Checksum=0x%016llx\n", (unsigned long long)xorv);

  API_CHECK( DEV_FREE(dK) );
  API_CHECK( DEV_FREE(dV) );
  API_CHECK( DEV_FREE(dSink) );
  return 0;
}
```

### Build

CUDA:

```
nvcc -O3 -std=c++17 -arch=${SM_ARCH} topics/04-kv-cache-sizing-bandwidth-layout/code/kv_layout_bench.cu -o kv_bench
```

ROCm:

```
hipcc -O3 -std=c++17 --offload-arch=${GFX_ARCH} topics/04-kv-cache-sizing-bandwidth-layout/code/kv_layout_bench.cu -o kv_bench
```

### Run Examples

```
# Default: L=32, Hkv=8, Dh=128, Tctx=4095, bytes=2, layout=0
./kv_bench

# Compare layouts
./kv_bench --layout=0
./kv_bench --layout=1

# Heavier KV (full multi-head KV)
./kv_bench --Hkv=32

# Sliding window decode (smaller Tctx)
./kv_bench --Tctx=1024
```

Expect higher GB/s with the layout that yields the most contiguous strides in the looped `t` dimension on your GPU.

## Profiling and Validation

### NVIDIA Nsight Compute

```
ncu --set full --target-processes all \
    --metrics dram__bytes.sum,lts__t_bytes.sum,sm__pipe_lsu_mem_shared_op_st.sum,sm__warps_active.avg.pct_of_peak_sustained_active \
    ./kv_bench --layout=0
```

Interpretation:

- `dram__bytes.sum`: total DRAM traffic; should match modelled bytes ±5%.
- `lts__t_bytes.sum`: traffic through L2; large with low hit rate at big `T_ctx` is expected.
- `sm__warps_active.*`: look for high occupancy but note memory‑bound nature.

### AMD rocprof

```
rocprof --stats --hsa-trace ./kv_bench --layout=0
```

Review:

- Memory throughput counters (e.g., `SQ_INSTS_VALU`, `TCC_READ/TCC_WRITE` derived stats) for sustained bandwidth.
- Kernel duration versus `BytesRead` → GB/s close to device peak is desired.

### Validation

- Kernel computes a checksum to prevent dead‑code elimination.
- Cross‑check theoretical `BytesRead = L·H_kv·T_ctx·D_h·s·2` against profiler results.

## Performance Checklist

- [ ] `D_h · s` aligned to 128 bytes (vectorized `int4` loads). If not, pad `D_h_pad = ceil((D_h·s)/128)·(128/s)`.
- [ ] `H_kv` chosen via GQA/MQA to fit KV cache in budget and reduce read bytes.
- [ ] KV layout puts the looped decode dimension (`t`) on a stride that coalesces across lanes.
- [ ] Preallocate KV with a pool; avoid per‑step allocations.
- [ ] Sliding window or paged KV reduces effective `T_ctx` under long contexts.
- [ ] Nsight/rocprof shows ≥70% of peak DRAM bandwidth for the microbench.

## Troubleshooting

| Symptom                              | Likely Cause                            | Fix                                                                      |
| ------------------------------------ | --------------------------------------- | ------------------------------------------------------------------------ |
| GB/s far below peak                  | Misaligned `D_h·s`                      | Pad `D_h` to 128‑byte multiples; use vector loads (`int4`)               |
| Large variance across runs           | Page faults/allocations                 | Preallocate with `cudaMallocAsync/hipMallocAsync`; warm up kernel        |
| Kernel is compute‑bound              | Small `T_ctx`                           | Increase `T_ctx` to expose memory bandwidth regime                       |
| L2 hit rate unexpectedly high        | Cache reuse due to tiny problem         | Increase `T_ctx`/problem size to realistic values                        |
| PCIe spillover                       | Using pageable host memory              | Avoid host transfers in hot path; use device‑resident KV                 |
| OOM                                  | KV too large                            | Reduce `T_max` or `H_kv`; consider quantized KV (INT8/FP8)               |
| Divergent performance across layouts | Stride interacts with DRAM/partitioning | Choose the layout that maximizes sequential accesses in your decode loop |

## Acceptance Criteria

- The knowledge file explains sizing, bandwidth, and layout with numeric examples and an explicit roofline.
- `kv_layout_bench.cu` builds with `nvcc` and `hipcc` and runs in seconds.
- Nsight/rocprof confirms total bytes within ±5% of the model and shows sustained bandwidth.
- On a modern GPU, measured throughput ≥70% of device copy bandwidth for `layout=best` with default parameters.

## Further Work

- Integrate a true attention microkernel to measure end‑to‑end decode step (Q·K^T, softmax, Attn·V) under different layouts.
- Add FP8/INT8 KV with on‑the‑fly dequant (fused with dot‑product) for reduced bytes and higher AI.
- Implement paged KV with fixed‑size blocks to explore TLB behavior and eviction policies.
- Experiment with persistent decode kernels and CUDA/HIP Graphs to remove launch overhead at small batch.
