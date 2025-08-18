# KV Paging & Memory Pools

## 2. Summary

Key/value (KV) cache dominates memory during LLM inference, especially in long-context, decode-heavy workloads. Paging organizes KV into fixed-size blocks and manages them via a pool to avoid device allocations in the hot path, reduce fragmentation, enable prefix sharing, and support overcommit/eviction. This note defines KV paging precisely, derives sizes and limits, and ships a runnable CUDA/HIP microbenchmark that appends tokens into paged KV blocks, validating correctness and timing pool vs. naïve allocation. You can use it as a baseline before integrating into attention kernels or a full serving stack.

## 3. Why It Matters for LLM Inference

- **Prefill:** Bulk writes; large, sequential stores. Paging lets you reuse blocks across requests (prefix cache) and avoid huge contiguous buffers.
- **Decode:** Steady per-token appends; hot path must not call `cuda/hipMalloc`. Paging gives O(1) block lookup and amortized-zero allocation cost after warmup.
- **Operational:** Memory pools cap fragmentation and allow admission control and eviction when load spikes.

## 4. Key Concepts and Formulas

### KV Size (per sequence, full retention)

Let:

- $L$: layers, $H$: heads, $d$: head dimension, $T$: tokens in context,
- $b$: bytes per element (e.g., FP16/BF16 = 2),
- factor 2 for K and V.

$$
\text{KV\_bytes} = 2 \cdot L \cdot H \cdot d \cdot T \cdot b
$$

Examples (BF16/FP16, $b=2$):

- 7B-class (e.g., $L{=}32,H{=}32,d{=}128,T{=}8192$):
  $\;2\cdot32\cdot32\cdot128\cdot8192\cdot2 = 4{,}294{,}967{,}296 \approx 4\text{ GiB}$.
- 70B-class ($L{=}80,H{=}64,d{=}128,T{=}8192$):
  $= 21{,}474{,}836{,}480 \approx 20\text{ GiB}$.

### Paging

Choose a **block size** $B$ tokens per page (typical 16–64). For a single head-layer:

- Page bytes (per K **or** V): $B \cdot d \cdot b$.
- With K+V kept separate, per page total = $2 \cdot B \cdot d \cdot b$.

For the 7B example with $B{=}16, d{=}128, b{=}2$:

- Per K page = $16\cdot128\cdot2 = 4096$ bytes (4 KiB); K+V = 8 KiB.
- Pages per head-layer = $\lceil T/B \rceil = 512$.

Paging doesn’t reduce peak memory if you keep all tokens, but it:

- eliminates hot-path allocations,
- enables **prefix sharing** (page refcounts),
- eases **eviction/overcommit** policies.

### Mapping

For token index $t$:

- Page index $p = \lfloor t/B \rfloor$, in-page offset $o = t - pB$.
- Addressing uses a `page_table[head][layer][p] -> page_id` plus $o$.

## 5. GPU Deep Dive

### NVIDIA

- **Warps/SMs:** Write KV with coalesced 128B sectors; align pages to ≥128B.
- **Tensor Cores:** Not used for KV writes, but keep decode kernels persistent to minimize launch overhead.
- **L2/L1:** 4–16 KiB page sizes align well with L2 line strides and TLB behavior; avoid tiny pages (<1 KiB).
- **Allocator:** Prefer custom pool or `cudaMemPool_t`+`cudaMallocAsync` for robustness.

### AMD

- **Wavefronts/CUs:** Similar coalescing concerns; use multiples of 64 elements for smooth mapping to lanes.
- **MFMA/XDLOPs:** Not relevant to KV writes directly; matters for GEMMs.
- **LDS:** KV cache lives in global memory; LDS is for staging/fusion only.
- **Allocator:** `hipMallocAsync` and HIP mempools (ROCm 5.4+) or custom pool.

## 6. Implementation

A minimal, runnable example that:

1. Implements a fixed-size page pool (no per-token device mallocs).
2. Appends $T$ tokens; each token writes its K and V into the proper page and offset.
3. Validates content.
4. Times pool vs naïve page allocation.

### Files

- `topics/11-kv-paging-memory-pools/code/kv_paging_pool.cu` (single-source CUDA/HIP)

```cpp
// kv_paging_pool.cu
// Build (CUDA): nvcc -O3 -std=c++17 -arch=${SM_ARCH} -lineinfo kv_paging_pool.cu -o kv_pool
// Build (ROCm): hipcc -O3 -std=c++17 --offload-arch=${GFX_ARCH} kv_paging_pool.cu -o kv_pool

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <chrono>
#include <cassert>

#if defined(__HIP_PLATFORM_AMD__)
  #include <hip/hip_runtime.h>
  #define DEV_MALLOC hipMalloc
  #define DEV_FREE hipFree
  #define DEV_MEMCPY hipMemcpy
  #define DEV_MEMCPY_DEVICE_TO_DEVICE hipMemcpyDeviceToDevice
  #define DEV_MEMCPY_HOST_TO_DEVICE hipMemcpyHostToDevice
  #define DEV_MEMCPY_DEVICE_TO_HOST hipMemcpyDeviceToHost
  #define DEV_EVENT_T hipEvent_t
  #define DEV_EVENT_CREATE hipEventCreate
  #define DEV_EVENT_RECORD hipEventRecord
  #define DEV_EVENT_SYNCHRONIZE hipEventSynchronize
  #define DEV_EVENT_ELAPSED_TIME hipEventElapsedTime
  #define DEV_DEVICE_SYNCHRONIZE hipDeviceSynchronize
  #define API_CHECK(x) do { auto _e=(x); if (_e!=hipSuccess){fprintf(stderr,"HIP error %s:%d: %s\n",__FILE__,__LINE__, hipGetErrorString(_e)); std::abort();}} while(0)
#else
  #include <cuda_runtime.h>
  #define DEV_MALLOC cudaMalloc
  #define DEV_FREE cudaFree
  #define DEV_MEMCPY cudaMemcpy
  #define DEV_MEMCPY_DEVICE_TO_DEVICE cudaMemcpyDeviceToDevice
  #define DEV_MEMCPY_HOST_TO_DEVICE cudaMemcpyHostToDevice
  #define DEV_MEMCPY_DEVICE_TO_HOST cudaMemcpyDeviceToHost
  #define DEV_EVENT_T cudaEvent_t
  #define DEV_EVENT_CREATE cudaEventCreate
  #define DEV_EVENT_RECORD cudaEventRecord
  #define DEV_EVENT_SYNCHRONIZE cudaEventSynchronize
  #define DEV_EVENT_ELAPSED_TIME cudaEventElapsedTime
  #define DEV_DEVICE_SYNCHRONIZE cudaDeviceSynchronize
  #define API_CHECK(x) do { auto _e=(x); if (_e!=cudaSuccess){fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__, cudaGetErrorString(_e)); std::abort();}} while(0)
#endif

// Kernel: write one token's K or V vector into a page at in-page index.
__global__ void write_token_u16(uint16_t* page, int elements_per_token, int in_page_idx, uint16_t value) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  size_t base = static_cast<size_t>(in_page_idx) * elements_per_token;
  for (int i = tid; i < elements_per_token; i += stride) {
    page[base + i] = value;
  }
}

struct PagePool {
  uint8_t* base = nullptr;
  size_t page_size = 0;     // bytes
  int capacity = 0;         // number of pages
  std::vector<int> free_list; // host-managed
  std::vector<uint8_t*> page_ptrs; // optional direct pointers

  void init(size_t page_bytes, int capacity_pages) {
    page_size = page_bytes;
    capacity = capacity_pages;
    size_t total = page_size * (size_t)capacity;
    API_CHECK(DEV_MALLOC((void**)&base, total));
    free_list.reserve(capacity);
    for (int i = capacity - 1; i >= 0; --i) free_list.push_back(i);
    page_ptrs.resize(capacity, nullptr);
  }
  void destroy() {
    if (base) { DEV_FREE(base); base = nullptr; }
    free_list.clear(); page_ptrs.clear();
    page_size = 0; capacity = 0;
  }
  int alloc_page() {
    if (free_list.empty()) return -1;
    int id = free_list.back(); free_list.pop_back();
    page_ptrs[id] = base + (size_t)id * page_size;
    return id;
  }
  void free_page(int id) {
    page_ptrs[id] = nullptr;
    free_list.push_back(id);
  }
  uint8_t* ptr(int id) { return page_ptrs[id]; }
};

static inline uint64_t kv_bytes_full(int L, int H, int d, int T, int bytes_per_elem) {
  return (uint64_t)2 * L * H * (uint64_t)d * T * bytes_per_elem;
}

struct Args {
  int L=1, H=1, d=128, T=8192, B=16; // layers, heads, d_head, tokens, page block size
  bool use_pool = true;              // vs naive per-page malloc
  int validate_samples = 4;          // number of tokens to validate
};

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i=1;i<argc;i++){
    std::string s(argv[i]);
    auto get = [&](int& dst){ if (i+1<argc) dst = std::atoi(argv[++i]); };
    if (s=="--layers") get(a.L);
    else if (s=="--heads") get(a.H);
    else if (s=="--dhead") get(a.d);
    else if (s=="--tokens") get(a.T);
    else if (s=="--block") get(a.B);
    else if (s=="--mode") {
      if (i+1<argc) { std::string m(argv[++i]); a.use_pool = (m!="naive"); }
    } else if (s=="--validate") get(a.validate_samples);
  }
  return a;
}

int main(int argc, char** argv) {
  Args a = parse_args(argc, argv);
  const int bytes_per_elem = 2; // uint16_t as FP16/BF16 stand-in
  const size_t elems_per_token = (size_t)a.d;
  const size_t page_elems = (size_t)a.B * elems_per_token; // per K or V
  const size_t page_bytes = page_elems * bytes_per_elem;
  const int pages_per_head_layer = (a.T + a.B - 1) / a.B;
  const int total_pages = a.L * a.H * pages_per_head_layer;

  printf("[config] L=%d H=%d d=%d T=%d B=%d mode=%s\n",
    a.L, a.H, a.d, a.T, a.B, a.use_pool ? "pool":"naive");

  printf("[math] Full KV size (bf16/fp16): %.3f GiB\n",
    kv_bytes_full(a.L,a.H,a.d,a.T,bytes_per_elem)/1024.0/1024.0/1024.0);

  // K and V pools or naive arrays
  PagePool poolK, poolV;
  std::vector<uint16_t*> naiveK(total_pages, nullptr), naiveV(total_pages, nullptr);

  if (a.use_pool) {
    // Pre-allocate K and V pools with enough pages.
    poolK.init(page_bytes, total_pages);
    poolV.init(page_bytes, total_pages);
  }

  // Page tables: (layer, head, page_index) -> page_id or pointer index
  // We store integer handle per K/V; for pool, handle==id; for naive, we store pointer in arrays.
  auto index3 = [&](int l, int h, int p) { return (l * a.H + h) * pages_per_head_layer + p; };

  // Timing
  DEV_EVENT_T beg, end;
  API_CHECK(DEV_EVENT_CREATE(&beg));
  API_CHECK(DEV_EVENT_CREATE(&end));
  API_CHECK(DEV_EVENT_RECORD(beg));

  int allocated_pages = 0;
  // Append tokens for all heads*layers (simulate decode)
  for (int t = 0; t < a.T; ++t) {
    int p = t / a.B;
    int o = t % a.B;
    uint16_t val = (uint16_t)(t & 0xFFFF);

    for (int l = 0; l < a.L; ++l) {
      for (int h = 0; h < a.H; ++h) {
        int idx = index3(l,h,p);

        uint16_t* k_page = nullptr;
        uint16_t* v_page = nullptr;

        if (a.use_pool) {
          if (poolK.ptr(idx) == nullptr) {
            int idK = poolK.alloc_page();
            int idV = poolV.alloc_page();
            if (idK != idx || idV != idx) {
              // For simplicity we expect 1:1 id==idx; if not, index via ptr() anyway.
            }
            allocated_pages += 2;
          }
          k_page = reinterpret_cast<uint16_t*>(poolK.ptr(idx));
          v_page = reinterpret_cast<uint16_t*>(poolV.ptr(idx));
        } else {
          if (naiveK[idx] == nullptr) {
            API_CHECK(DEV_MALLOC((void**)&naiveK[idx], page_bytes));
            API_CHECK(DEV_MALLOC((void**)&naiveV[idx], page_bytes));
            allocated_pages += 2;
          }
          k_page = naiveK[idx];
          v_page = naiveV[idx];
        }

        // Launch small writes for this token into the page at offset o.
        int threads = 128;
        int blocks  = (int)((elems_per_token + threads - 1) / threads);
        write_token_u16<<<blocks, threads>>>(k_page, (int)elems_per_token, o, val);
        write_token_u16<<<blocks, threads>>>(v_page, (int)elems_per_token, o, val);
      }
    }
  }

  API_CHECK(DEV_EVENT_RECORD(end));
  API_CHECK(DEV_EVENT_SYNCHRONIZE(end));
  float ms=0.0f; API_CHECK(DEV_EVENT_ELAPSED_TIME(&ms, beg, end));
  API_CHECK(DEV_DEVICE_SYNCHRONIZE());

  double tokens_total = (double)a.T * a.L * a.H;
  double toks_per_s = tokens_total / (ms * 1e-3);
  printf("[timing] elapsed = %.3f ms, tokens = %.0f, tokens/s = %.1f\n", ms, tokens_total, toks_per_s);
  printf("[alloc] pages allocated (K+V) = %d, page_bytes=%zu (%.1f KiB)\n", allocated_pages, page_bytes, page_bytes/1024.0);

  // Validation: sample a few tokens (begin/mid/end)
  int checks = a.validate_samples;
  if (checks > 0) {
    std::vector<int> sample_t;
    sample_t.push_back(0);
    sample_t.push_back(std::min(a.T-1, a.B-1));
    sample_t.push_back(std::min(a.T-1, a.B));
    sample_t.push_back(a.T-1);
    while ((int)sample_t.size() > checks) sample_t.pop_back();

    std::vector<uint16_t> hostbuf(elems_per_token);
    bool ok = true;
    for (int t : sample_t) {
      int p = t / a.B;
      int o = t % a.B;
      uint16_t expect = (uint16_t)(t & 0xFFFF);

      int l = a.L-1, h = a.H-1; // spot-check last layer/head
      int idx = index3(l,h,p);

      uint16_t* k_page = a.use_pool ? reinterpret_cast<uint16_t*>(poolK.ptr(idx)) : naiveK[idx];
      API_CHECK(DEV_MEMCPY(hostbuf.data(), k_page + (size_t)o*elems_per_token, elems_per_token*sizeof(uint16_t), DEV_MEMCPY_DEVICE_TO_HOST));

      for (size_t i=0;i<elems_per_token;i++) {
        if (hostbuf[i] != expect) { ok = false; break; }
      }
      if (!ok) {
        printf("[validate] FAIL at t=%d (expected %u)\n", t, (unsigned)expect);
        break;
      }
    }
    printf("[validate] %s\n", ok ? "OK" : "FAILED");
  }

  // Cleanup
  if (a.use_pool) { poolK.destroy(); poolV.destroy(); }
  else {
    for (auto p : naiveK) if (p) API_CHECK(DEV_FREE(p));
    for (auto p : naiveV) if (p) API_CHECK(DEV_FREE(p));
  }
  return 0;
}
```

#### Build

- CUDA:

```
nvcc -O3 -std=c++17 -arch=${SM_ARCH} -lineinfo topics/11-kv-paging-memory-pools/code/kv_paging_pool.cu -o kv_pool
# e.g., SM_ARCH=sm_90
```

- ROCm/HIP:

```
hipcc -O3 -std=c++17 --offload-arch=${GFX_ARCH} topics/11-kv-paging-memory-pools/code/kv_paging_pool.cu -o kv_pool
# e.g., GFX_ARCH=gfx942
```

#### Run

```
# default: L=1,H=1,d=128,T=8192,B=16, mode=pool
./kv_pool
./kv_pool --mode naive           # compare against naive per-page malloc/free
./kv_pool --layers 32 --heads 32 --dhead 128 --tokens 8192 --block 16
```

## 7. Profiling and Validation

### Functional validation

- Program prints `validate OK`. If it fails, see Troubleshooting.

### Performance runs

- **Nsight Systems (timeline)**

  ```
  nsys profile -t cuda,nvtx -o nsys_kv ./kv_pool --layers 32 --heads 32 --tokens 8192 --block 16
  ```

  Expect: no device allocations during the main loop in pool mode.

- **Nsight Compute (counters)**

  ```
  ncu --set regex --metrics \
    sm__warps_active.avg.pct_of_peak_sustained_active,\
    lts__t_bytes_op_write.sum,\
    dram__bytes.sum \
    ./kv_pool --layers 32 --heads 32 --tokens 8192 --block 16
  ```

  Check: high store throughput, minimal divergence; compare pool vs naive.

- **ROCm rocprof**

  ```
  rocprof --hip-trace --hsa-trace --stats \
    ./kv_pool --layers 32 --heads 32 --tokens 8192 --block 16
  ```

  Check: HIP allocations should not appear inside the token loop in pool mode.

### Pass thresholds (guide)

- Pool mode achieves ≥3× tokens/s vs naive mode on the same configuration.
- No `cuda/hipMalloc` calls observed inside the decode loop (after first few iterations).

## 8. Performance Checklist

- Choose **page size** $2\cdot B\cdot d\cdot b$ between 4–32 KiB (K+V separate pages yields half each); keep multiples of 128B.
- No device allocations or `new/delete` in the hot decode loop.
- Pre-size page pool for expected max resident pages (or enable eviction).
- Coalesced stores: write per-token vectors with at least 128 threads; avoid scalar byte stores.
- Keep a compact `page_table` (int32 ids) and avoid pointer chasing on device.
- For prefix sharing: implement per-page **refcount** and copy-on-write for the last page.
- Batch page fills when possible (group tokens across heads) to reduce launches.
- Use async execution graphs or persistent kernels in full pipelines to amortize launch cost.

## 9. Troubleshooting

| Symptom                   | Likely Cause                                  | Fix                                                              |
| ------------------------- | --------------------------------------------- | ---------------------------------------------------------------- |
| Validation FAIL           | Wrong in-page offset math                     | Ensure $o = t \bmod B$, base index $o\cdot d$                    |
| Tokens/s similar to naive | Pool capacity too small or hidden allocations | Pre-size pool; verify timeline shows no device mallocs           |
| Kernel underutilized      | Too few threads                               | Use ≥128 threads; scale blocks with $d$                          |
| L2/DRAM throughput low    | Unaligned or tiny pages                       | Use 4–16 KiB pages per K; align to 128B                          |
| OOM with many sequences   | No eviction/admission control                 | Implement LRU/clock eviction; cap pages per tenant               |
| Fragmentation over time   | Mixed-size allocations                        | Fixed-size pages only in the hot path; segregate by dtype/d_head |
| ROCm build fails          | Wrong `--offload-arch`                        | Set correct GFX arch (e.g., `gfx942`, `gfx90a`)                  |
| CUDA build fails          | Wrong `-arch`                                 | Use correct SM (e.g., `sm_90`, `sm_80`)                          |

## 10. Acceptance Criteria

- Knowledge file explains paging/pools with formulas and numeric examples.
- `kv_pool` compiles on CUDA 12.x and ROCm/HIP 6.x.
- `kv_pool` runs <5 seconds for `L=32,H=32,d=128,T=8192,B=16` on a modern GPU.
- Validation passes; pool mode ≥3× tokens/s over naive mode (indicative on most setups).
- Nsight/rocprof shows no device allocations in the decode loop in pool mode.

## 11. Further Work

- **Prefix Cache:** Add page refcounts + copy-on-write on the tail page to share prefixes across sequences (vLLM-style).
- **Eviction:** Implement LRU/clock with per-sequence residency limits and async scrubbing.
- **Packing:** Multi-sequence compaction to defragment sparse last pages.
- **Datatype Specialization:** Use `__half2`/`bf16x2` vector stores, and INT8/FP8 with fused dequant in attention.
- **Execution Graphs:** Wrap the per-token writes in CUDA/HIP Graphs or a persistent kernel for lower launch overhead.
- **Multi-GPU:** Distribute page pools per device with NCCL/RCCL-aware paging of cross-device KV (e.g., tensor-parallel heads).

### Quick Start Commands

```bash
# CUDA
nvcc -O3 -std=c++17 -arch=sm_90 -lineinfo topics/11-kv-paging-memory-pools/code/kv_paging_pool.cu -o kv_pool
./kv_pool --layers 32 --heads 32 --dhead 128 --tokens 8192 --block 16

# ROCm
hipcc -O3 -std=c++17 --offload-arch=gfx942 topics/11-kv-paging-memory-pools/code/kv_paging_pool.cu -o kv_pool
./kv_pool --layers 32 --heads 32 --dhead 128 --tokens 8192 --block 16
```
