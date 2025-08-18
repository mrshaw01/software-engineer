# Multi-GPU Topology & Parallelism

## 2. Summary

This topic explains how to map LLM inference workloads onto multi-GPU systems while respecting the physical interconnect topology (PCIe, NVLink/NVSwitch, xGMI/Infinity Fabric) and software collectives (NCCL/RCCL). You will learn which forms of parallelism (data, tensor, pipeline, expert) fit which topologies, how to estimate communication time, and how to validate with lightweight GPU benchmarks. The deliverables include runnable CUDA/HIP code to measure peer-to-peer bandwidth/latency and a single-process multi-GPU NCCL/RCCL all-reduce microbenchmark.

## 3. Why It Matters for LLM Inference

Inference decode is latency-sensitive and issues many small collectives when tensor parallelism is used (two all-reduces per transformer layer). Prefill is throughput-oriented and tolerates larger microbatches and activation transfers. Mapping parallel groups to “fast” topology islands (NVSwitch or on-package xGMI) and keeping high-latency links (cross-socket PCIe, inter-node) for coarse parallelism (data or pipeline) is often the difference between linear scaling and regressions.

## 4. Key Concepts and Formulas

Let:

- $N$: GPUs in the group (e.g., tensor-parallel degree).
- $S$: message size in bytes for a collective.
- $B$: sustained link bandwidth (bytes/s) available to the algorithm.
- $L$: per-message latency (s).

Common collective cost models:

- **Ring all-reduce:**

  $$
  T_{\text{ring}} \approx \frac{2(N-1)}{N} \cdot \frac{S}{B} + (N-1)\cdot L
  $$

- **Tree all-reduce:**

  $$
  T_{\text{tree}} \approx 2\lceil \log_2 N \rceil \cdot \left(\frac{S}{B} + L\right)
  $$

Tensor-parallel communication (Megatron-style) per layer per token (decode):

- Two all-reduces of activation vectors of length $H$ (hidden size) in activation dtype $d$ bytes (e.g., 2 for FP16/BF16):

  $$
  S_{\text{TP}} \approx 2 \cdot H \cdot d \quad \text{bytes/token/layer}
  $$

Numeric example: $H=4096, d=2\Rightarrow S_{\text{TP}}=16{,}384$ B ≈ 16 KB per layer. For 32 layers ≈ 512 KB per token, so **latency $L$** dominates.

Pipeline-parallel boundary transfer (prefill): activations of shape $[B, S_{\text{mb}}, H]$ cross stage boundaries (one forward hop per boundary). Communication volume scales with microbatch size; use larger microbatches during prefill to amortize.

MoE expert parallel (inference): per token all-to-all of hidden vectors for the routed experts; volume roughly $E_{\text{routed}}\cdot H\cdot d$ and **requires** low-diameter fabrics.

## 5. GPU Deep Dive

### NVIDIA

- **Compute units:** SMs; 32-thread warps.
- **Interconnects:**

  - PCIe (host/root-complex dependent).
  - NVLink/NVSwitch: low-latency, high-bisection fabrics; multiple NVLink lanes per GPU into NVSwitch for all-to-all.

- **Collectives:** NCCL chooses Ring/Tree/CollNet and protocol (LL, LL128, Simple).
- **Hints:** Keep tensor-parallel groups within a single NVSwitch island; use NCCL environment knobs for algorithm/protocol selection when needed.

### AMD

- **Compute units:** CUs; 64-lane wavefronts; MFMA/XDLOPs for tensor ops; LDS for on-chip SRAM.
- **Interconnects:**

  - PCIe;
  - xGMI/Infinity Fabric; on MI300-class, many GPUs and CPU dies share package fabrics.

- **Collectives:** RCCL is API-compatible with NCCL, tuned for xGMI/IF and IB.
- **Hints:** Place tensor-parallel groups intra-package; use RCCL (NCCL-compatible API) and ensure ROCm P2P is enabled.

## 6. Implementation

Two minimal tools:

1. **Peer-to-Peer bandwidth/latency** (no NCCL/RCCL required).
2. **Single-process multi-GPU all-reduce** using NCCL (NVIDIA) or RCCL (AMD) with identical `nccl.h` API.

### 6.1 Single-source P2P bandwidth/latency

`topics/13-multi-gpu-topology-parallelism/code/p2p_peer_bw.cpp`

```cpp
// Single-source CUDA/HIP P2P bandwidth + latency microbench
// Build (CUDA): nvcc -O3 -std=c++17 p2p_peer_bw.cpp -o p2p_bw
// Build (ROCm): hipcc -O3 -std=c++17 p2p_peer_bw.cpp -o p2p_bw
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <chrono>
#include <string>

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  #include <hip/hip_runtime.h>
  #define API_PREFIX hip
  #define CALL(x) do { auto _e = (x); if (_e != hipSuccess) { \
      fprintf(stderr,"HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_e)); std::exit(1);} } while(0)
#else
  #include <cuda_runtime.h>
  #define API_PREFIX cuda
  #define CALL(x) do { auto _e = (x); if (_e != cudaSuccess) { \
      fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); std::exit(1);} } while(0)
#endif

static inline double gbps(double bytes, double secs) {
  return (bytes / secs) / 1e9;
}

int main(int argc, char** argv) {
  int dev_src = 0, dev_dst = 1;
  size_t bytes = (size_t)256 << 20; // 256 MiB default
  int iters = 50, warmup = 5;

  if (argc > 1) dev_src = std::atoi(argv[1]);
  if (argc > 2) dev_dst = std::atoi(argv[2]);
  if (argc > 3) bytes   = (size_t)std::atoll(argv[3]); // raw bytes
  if (argc > 4) iters   = std::atoi(argv[4]);

  int ndev = 0;
  CALL(API_PREFIXGetDeviceCount(&ndev));
  if (dev_src < 0 || dev_src >= ndev || dev_dst < 0 || dev_dst >= ndev || dev_src == dev_dst) {
    fprintf(stderr, "Need two distinct device ids in [0,%d). Usage: %s [src] [dst] [bytes] [iters]\n", ndev, argv[0]);
    return 2;
  }

  int access = 0;
  CALL(API_PREFIXDeviceCanAccessPeer(&access, dev_dst, dev_src));
  printf("P2P capability dst->src: %s\n", access ? "YES" : "NO");
  if (access) { CALL(API_PREFIXSetDevice(dev_dst)); CALL(API_PREFIXDeviceEnablePeerAccess(dev_src, 0)); }
  CALL(API_PREFIXDeviceCanAccessPeer(&access, dev_src, dev_dst));
  printf("P2P capability src->dst: %s\n", access ? "YES" : "NO");
  if (access) { CALL(API_PREFIXSetDevice(dev_src)); CALL(API_PREFIXDeviceEnablePeerAccess(dev_dst, 0)); }

  void *src=nullptr, *dst=nullptr;
  CALL(API_PREFIXSetDevice(dev_src));
  CALL(API_PREFIXMalloc(&src, bytes));
  CALL(API_PREFIXMemset(src, 1, bytes));
  CALL(API_PREFIXSetDevice(dev_dst));
  CALL(API_PREFIXMalloc(&dst, bytes));
  CALL(API_PREFIXMemset(dst, 2, bytes));

  // Large-copy bandwidth
  double best_gbps = 0.0, sum_gbps = 0.0;
  for (int i=0;i<warmup+iters;i++) {
    auto t0 = std::chrono::high_resolution_clock::now();
    CALL(API_PREFIXMemcpyPeer(dst, dev_dst, src, dev_src, bytes));
    CALL(API_PREFIXDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    double g = gbps((double)bytes, secs);
    if (i>=warmup){ sum_gbps += g; if (g>best_gbps) best_gbps = g; }
  }
  printf("P2P memcpy %zu bytes: avg %.2f GB/s, best %.2f GB/s over %d iters\n",
         (size_t)bytes, sum_gbps/iters, best_gbps, iters);

  // Small-copy latency (4 KB)
  size_t small = 4<<10;
  int reps = 10000;
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i=0;i<reps;i++) {
    CALL(API_PREFIXMemcpyPeer(dst, dev_dst, src, dev_src, small));
  }
  CALL(API_PREFIXDeviceSynchronize());
  auto t1 = std::chrono::high_resolution_clock::now();
  double secs = std::chrono::duration<double>(t1 - t0).count();
  double usec = secs * 1e6 / reps;
  printf("P2P memcpy %zu bytes: latency %.2f us (avg over %d)\n", small, usec, reps);

  CALL(API_PREFIXFree(src));
  CALL(API_PREFIXFree(dst));
  return 0;
}
```

### 6.2 Single-process NCCL/RCCL all-reduce microbench

`topics/13-multi-gpu-topology-parallelism/code/nccl_allreduce_bench.cpp`

```cpp
// Single-process, multi-GPU all-reduce using NCCL (NVIDIA) or RCCL (AMD).
// Build (CUDA): nvcc -O3 -std=c++17 nccl_allreduce_bench.cpp -lnccl -o ar_bench
// Build (ROCm): hipcc -O3 -std=c++17 nccl_allreduce_bench.cpp -lrccl -o ar_bench
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <chrono>
#include <string>
#include <thread>

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  #include <hip/hip_runtime.h>
  #define DEV_API hip
  #define CALL(x) do{ auto e=(x); if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__, hipGetErrorString(e)); std::exit(1);} }while(0)
#else
  #include <cuda_runtime.h>
  #define DEV_API cuda
  #define CALL(x) do{ auto e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__, cudaGetErrorString(e)); std::exit(1);} }while(0)
#endif

#include <nccl.h> // RCCL provides a drop-in "nccl.h" header

#define NCCLCHECK(cmd) do { ncclResult_t r = (cmd); if (r != ncclSuccess) { \
  fprintf(stderr, "NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r)); std::exit(1);} } while(0)

struct DeviceCtx {
  int dev;
  void* buf;
  size_t bytes;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  hipStream_t stream;
#else
  cudaStream_t stream;
#endif
  ncclComm_t comm;
};

int main(int argc, char** argv) {
  int nDevs = 0;
  CALL(DEV_APIGetDeviceCount(&nDevs));
  if (nDevs < 2) { fprintf(stderr,"Need >=2 GPUs\n"); return 2; }

  int ng = std::min(8, nDevs);
  size_t bytes = (size_t)128 << 20; // 128 MiB default
  int iters = 40, warmup = 10;
  std::string dtype = "float32";

  if (argc > 1) ng    = std::atoi(argv[1]);
  if (argc > 2) bytes = (size_t)std::atoll(argv[2]);
  if (argc > 3) iters = std::atoi(argv[3]);

  if (ng < 2 || ng > nDevs) { fprintf(stderr,"Invalid ng=%d (have %d)\n", ng, nDevs); return 3; }

  std::vector<int> devs(ng);
  for (int i=0;i<ng;i++) devs[i] = i;

  std::vector<DeviceCtx> ctx(ng);
  for (int i=0;i<ng;i++) {
    ctx[i].dev = devs[i];
    CALL(DEV_APISetDevice(ctx[i].dev));
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    CALL(hipStreamCreate(&ctx[i].stream));
#else
    CALL(cudaStreamCreate(&ctx[i].stream));
#endif
    CALL(DEV_APIMalloc(&ctx[i].buf, bytes));
    CALL(DEV_APIMemset(ctx[i].buf, 0, bytes));
    ctx[i].bytes = bytes;
  }

  std::vector<ncclComm_t> comms(ng);
  NCCLCHECK(ncclCommInitAll(comms.data(), ng, devs.data()));
  for (int i=0;i<ng;i++) ctx[i].comm = comms[i];

  ncclDataType_t dt = ncclFloat32;
  size_t count = bytes / sizeof(float);

  auto run_once = [&](bool measure){
    if (measure) { ; }
    NCCLCHECK(ncclGroupStart());
    for (int i=0;i<ng;i++) {
      NCCLCHECK(ncclAllReduce(ctx[i].buf, ctx[i].buf, count, dt, ncclSum, ctx[i].comm, ctx[i].stream));
    }
    NCCLCHECK(ncclGroupEnd());
    for (int i=0;i<ng;i++) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
      CALL(hipStreamSynchronize(ctx[i].stream));
#else
      CALL(cudaStreamSynchronize(ctx[i].stream));
#endif
    }
  };

  // Warmup
  for (int i=0;i<warmup;i++) run_once(false);

  // Timed
  double best=1e30, agg=0.0;
  for (int i=0;i<iters;i++) {
    auto t0 = std::chrono::high_resolution_clock::now();
    run_once(true);
    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    if (secs < best) best = secs;
    agg += secs;
  }

  // Effective algorithmic traffic per GPU (ring model): 2*(N-1)/N * S
  double eff_bytes_per_gpu = 2.0 * (ng-1) / ng * (double)bytes;
  double avg_bw = (eff_bytes_per_gpu / (agg / iters)) / 1e9;
  double best_bw = (eff_bytes_per_gpu / best) / 1e9;
  printf("AllReduce ng=%d, payload=%zu bytes per GPU, iters=%d\n", ng, (size_t)bytes, iters);
  printf("Avg time: %.6f s, Best time: %.6f s\n", agg / iters, best);
  printf("Effective BW per GPU: avg %.2f GB/s, best %.2f GB/s (ring-equivalent)\n", avg_bw, best_bw);

  for (int i=0;i<ng;i++) {
    NCCLCHECK(ncclCommDestroy(ctx[i].comm));
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    CALL(hipStreamDestroy(ctx[i].stream));
#else
    CALL(cudaStreamDestroy(ctx[i].stream));
#endif
    CALL(DEV_APIFree(ctx[i].buf));
  }
  return 0;
}
```

#### Build commands

```
# CUDA
nvcc -O3 -std=c++17 topics/13-multi-gpu-topology-parallelism/code/p2p_peer_bw.cpp -o p2p_bw
nvcc -O3 -std=c++17 topics/13-multi-gpu-topology-parallelism/code/nccl_allreduce_bench.cpp -lnccl -o ar_bench

# ROCm / HIP
hipcc -O3 -std=c++17 topics/13-multi-gpu-topology-parallelism/code/p2p_peer_bw.cpp -o p2p_bw
hipcc -O3 -std=c++17 topics/13-multi-gpu-topology-parallelism/code/nccl_allreduce_bench.cpp -lrccl -o ar_bench
```

#### Example runs

```
# P2P bandwidth/latency between GPU 0 -> 1 with 256 MiB transfer
./p2p_bw 0 1 $((256<<20)) 50

# All-reduce across 4 GPUs, 128 MiB payload per GPU, 40 iters
./ar_bench 4 $((128<<20)) 40
```

## 7. Profiling and Validation

### NVIDIA

- **Nsight Systems (timeline of collectives + kernels):**

  ```
  nsys profile -o nsys_ar --stats=true ./ar_bench 8 $((128<<20)) 40
  ```

  Look for back-to-back `ncclAllReduce` on fast links (NVLink/NVSwitch). Verify minimal CPU gaps.

- **Nsight Compute (kernel-level NCCL protocol):**

  ```
  ncu --set full --target-processes all ./ar_bench 8 $((128<<20)) 10
  ```

  Metrics of interest: `dram__throughput`, `nvlink__throughput`, SM stall reasons (should be low, comm kernels are lightweight).

- **Topology check:**

  ```
  nvidia-smi topo -m
  ```

  Ensure all tensor-parallel ranks reside within a low-distance island (NVSwitch).

### AMD

- **rocprof (timeline + counters):**

  ```
  rocprof --timestamp on --hsa-trace --hip-trace --roctx-trace --obj-tracking on ./ar_bench 8 $((128<<20)) 40
  ```

  Inspect HIP memcpy P2P throughput and RCCL kernels.

- **Topology check:**

  ```
  rocminfo | grep -A2 'Agent'    # inventories GPUs
  rocm-smi --showtopo
  ```

  Confirm TP groups within a single package or xGMI island.

### Validation thresholds (suggested)

- P2P `p2p_bw`: sustained ≥ 70% of expected fabric peak for large copies; small-copy latency ≤ 5–15 µs on NVLink/xGMI (PCIe may be higher).
- All-reduce `ar_bench`: best effective BW per GPU within 60–80% of ring model using measured `B` from `p2p_bw`.

## 8. Performance Checklist

- Map **tensor parallel (TP)** ranks to the fastest intra-node island (NVSwitch/xGMI).
- Use **pipeline parallel (PP)** across slower edges (cross-socket PCIe, inter-node IB).
- Keep **data parallel (DP)** inter-node when necessary; overlap reduce-scatter/all-gather with compute.
- Pin processes/threads to NUMA-local CPUs for their GPU.
- Enable GPU **P2P** and verify with `p2p_bw`.
- Lock NCCL/RCCL algorithm/protocol if auto-tuning is unstable:

  - `NCCL_ALGO=Ring|Tree`
  - `NCCL_PROTO=LL|LL128|Simple`
  - `NCCL_MIN_NCHANNELS` (set 4–16 as needed)
  - `NCCL_DEBUG=INFO` (or `TRACE` temporarily)

- For inter-node: verify IB interface selection (`NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`, `RCCL_IFNAME`), GDR settings, and MTU.

## 9. Troubleshooting

| Symptom                               | Likely Cause                                           | Fix                                                                                                                                                 |
| ------------------------------------- | ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| P2P bandwidth low, high latency       | P2P disabled, cross-root complex routing, IOMMU issues | Enable peer access; place GPUs under same root complex; check BIOS ACS/IOMMU; retest with `p2p_bw`.                                                 |
| NCCL hangs at init                    | Wrong interface or blocked by firewall                 | Set `NCCL_SOCKET_IFNAME=ib0` (or correct NIC), open required ports, set `NCCL_DEBUG=INFO`.                                                          |
| All-reduce BW halves when adding GPUs | Crossing topology island                               | Restrict group to one island; choose PP/DP across islands.                                                                                          |
| Spiky decode latency                  | Many small collectives + protocol mismatch             | Force `NCCL_PROTO=LL` and `NCCL_ALGO=Tree` for small messages; increase CUDA/HIP graph or persistent kernel usage to reduce launches (see Topic 9). |
| CPU pegged, poor overlap              | CPU affinity and driver/runtime contention             | Pin threads per rank; reduce `NCCL_NTHREADS` or adjust channels; ensure IRQs affine to local NUMA.                                                  |
| RCCL slower than expected             | GDR not enabled or IF width limited                    | Verify xGMI links and ROCm version; test interconnect with `p2p_bw`; update ROCm.                                                                   |
| Inter-node BW far below IB spec       | MTU or GID mismatch                                    | Set MTU=9000; `NCCL_IB_GID_INDEX`; ensure RoCEv2/IB configured consistently.                                                                        |
| Unstable throughput run-to-run        | Auto-tuner variance                                    | Pin `NCCL_ALGO/PROTO`, warm up longer, avoid frequency scaling throttling.                                                                          |
| Memory OOM at high TP                 | Activation duplication                                 | Use sequence/context parallel where applicable; fuse all-reduces with epilogues to reduce buffers.                                                  |

## 10. Acceptance Criteria

- Both programs compile with commands provided and run within seconds on a modern multi-GPU node.
- `p2p_bw` reports plausible large-copy throughput and µs-scale small-copy latency on fast fabrics.
- `ar_bench` scales to ≥4 GPUs and reports effective per-GPU bandwidth; best result within 60–80% of measured P2P peak using ring model.
- Documentation explains mapping of TP/PP/DP/MoE to topology with concrete formulas and at least one numeric instantiation.

## 11. Further Work

- Add **inter-node** variant using MPI ranks for NCCL-across-nodes testing.
- Implement **reduce-scatter/all-gather** microbench to reflect tensor-parallel schedules.
- Integrate **CUDA/HIP Graphs** capture around collective sequences to reduce launch overhead (decode path).
- Add **expert-parallel all-to-all** microbench to study token routing for MoE.
- Collect **TTFT and tokens/s** for a small LLM under varying parallel mappings and correlate with these microbench results.

## Back-of-Envelope Worked Examples

1. **Tensor Parallel Decode (latency sensitivity)**
   Hidden $H=4096$, dtype FP16 (2 B). Per layer, per token:
   $S_{\text{TP}} = 2 \cdot 4096 \cdot 2 = 16{,}384$ B = 16 KB.
   For 32 layers, total per token: $32 \cdot 16\text{KB} = 512\text{KB}$.
   On a fast fabric with $B = 200$ GB/s, $L = 3$ µs, $N=8$:

$$
T_{\text{ring}} \approx \frac{2(7)}{8}\cdot \frac{16\,384}{200\times 10^9} + 7\cdot 3\mu s
\approx 1.4\times 10^{-7} \text{s} + 21\mu s \approx 21.14\mu s
$$

Latency dominates. Keeping TP within NVSwitch/xGMI is critical.

2. **Pipeline Prefill (throughput sensitivity)**
   Microbatch $B=16$, sequence $S_{\text{mb}}=1024$, $H=4096$, FP16.
   Boundary tensor ≈ $B \cdot S_{\text{mb}} \cdot H \cdot d \approx 16\cdot 1024\cdot4096\cdot2 \approx 134{,}217{,}728$ B ≈ 128 MB.
   At $B=100$ GB/s, transfer time ≈ 1.28 ms per boundary—amortized over many tokens; PP can cross slower links if needed.

## Deployment Notes (NCCL/RCCL)

- Prefer automatic topology but be ready to pin:
  `NCCL_ALGO=Ring` (large messages), `NCCL_PROTO=LL` (small messages), `NCCL_MIN_NCHANNELS=4–16`.
- Select NICs/IFs explicitly in multi-homed systems:
  `NCCL_SOCKET_IFNAME=ib0`, `NCCL_IB_HCA=mlx5_0,mlx5_1`.
- Logging: `NCCL_DEBUG=INFO`, `NCCL_TOPO_DUMP_FILE=/tmp/nccl_topo.xml` to inspect model.
- RCCL uses the same API; link with `-lrccl` and validate with `rocprof`.
