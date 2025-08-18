# Deployment & Operations on GPUs

## Summary

Efficient GPU deployment for LLM inference requires reproducible builds, topology-aware scheduling, deterministic runtime configuration, and robust observability. This document provides a production-focused guide to running inference on NVIDIA (CUDA) and AMD (ROCm/HIP) GPUs: container baselines, node tuning, NUMA/PCIe/NVLink/IF links, power/thermal management, health checks, and safe rollout/rollback. It includes a minimal, runnable CUDA/HIP healthcheck + graph-capture microbenchmark to validate nodes before admitting live traffic. Outcomes: faster TTFT, predictable p99 latency, and fewer operational incidents.

## Why It Matters for LLM Inference

Decode is latency-sensitive and sensitive to kernel launch overheads and cache locality; prefill is bandwidth-dominant and benefits from large, bursty GEMMs. Operationally: warm pools, pinned memory, CUDA/HIP Graphs, and stream priorities stabilize p95–p99. Capacity planning hinges on KV cache footprint and interconnect bandwidth; misconfigured clocks, NUMA, or MIG/SR-IOV partitions can silently halve throughput.

## Key Concepts and Formulas

- **Capacity per node (tokens/s):**
  $C_{node} = \sum_{g=1}^{G} C_g(\text{batch}, L_{ctx}, d_{model}, q) \times U_{sched}$
  where $U_{sched}$ is scheduler utilization (0–1), $q$ is quantization state.
- **KV memory per request (bytes):**
  $M_{KV} = 2 \cdot L_{ctx} \cdot h \cdot d_{head} \cdot b_{dtype}$
  Example: 32 heads, d*head=128, L=8{,}192, BF16 (2 bytes): $M*{KV}=2·8192·32·128·2 = 134{,}217{,}728$ B ≈ 128 MiB per sequence.
- **Node admission check:** ensure $\sum M_{KV}$ + model params + activations + fragmentation < VRAM – safety margin (5–10%).
- **NUMA placement:** bind inference worker threads to the CPU NUMA domain that hosts the GPU’s closest PCIe root complex.

## GPU Deep Dive

### NVIDIA specifics

- **Execution:** 32-thread warps, SM residency depends on registers/shared memory and block size. Tensor Cores accelerate FP16/BF16/FP8; use cooperative kernels or CUDA Graphs for decode.
- **Memory:** L2 is shared; page-locked (pinned) host memory for H2D/D2H. Use `cudaMallocAsync` pools to reduce fragmentation and improve concurrency.
- **Ops features:** MIG for partitioning A100/H100 class GPUs; DCGM for telemetry; `nvidia-smi -pm 1` to enable persistence; application clocks for stability.

### AMD specifics

- **Execution:** 64-thread wavefronts, CUs/SAs; MFMA/XDLOPs for matrix math. LDS is the shared memory analogue; alignment matters for vectorized loads/stores.
- **Memory:** ROCm HIP runtime, HSA queues. Use `hipMallocAsync` and stream-ordered memory pools available in ROCm 6.x. Pinned memory via `hipHostMalloc`.
- **Ops features:** SR-IOV for partitioning on supported MI parts; ROCm SMI/`amd-smi` for telemetry; performance states set to ‘high’ for stable clocks.

## Implementation

A minimal healthcheck and microbenchmark that:

1. enumerates device(s) and prints key properties; 2) allocates memory via async pool; 3) captures a graph (H2D → kernel → D2H) and replays it; 4) reports throughput and basic pass/fail. Single-source, compiles with either NVCC or HIPCC.

**File:** `topics/17-deployment-ops/code/ops_healthcheck.cpp`

```cpp
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstring>

#if defined(__HIP_PLATFORM_AMD__)
  #include <hip/hip_runtime.h>
  #define DEVFN __global__
  #define API_CHECK(x) do { auto _e=(x); if (_e!=hipSuccess){\
    fprintf(stderr,"HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_e)); exit(1);} } while(0)
  using stream_t = hipStream_t; using event_t = hipEvent_t; using device_prop_t = hipDeviceProp_t;
  #define StreamCreateWithPriority hipStreamCreateWithPriority
  #define StreamDestroy hipStreamDestroy
  #define StreamBeginCapture hipStreamBeginCapture
  #define StreamEndCapture hipStreamEndCapture
  #define Graph hipGraph_t
  #define GraphExec hipGraphExec_t
  #define GraphInstantiate hipGraphInstantiate
  #define GraphLaunch hipGraphLaunch
  #define GraphDestroy hipGraphDestroy
  #define GraphExecDestroy hipGraphExecDestroy
  #define EventCreate hipEventCreate
  #define EventRecord hipEventRecord
  #define EventElapsed hipEventElapsedTime
  #define EventDestroy hipEventDestroy
  #define MemcpyAsync hipMemcpyAsync
  #define MemHostToDevice hipMemcpyHostToDevice
  #define MemDeviceToHost hipMemcpyDeviceToHost
  #define GetDeviceCount hipGetDeviceCount
  #define GetDeviceProperties hipGetDeviceProperties
  #define SetDevice hipSetDevice
  #define MallocAsync hipMallocAsync
  #define FreeAsync hipFreeAsync
  #define HostMalloc hipHostMalloc
  #define HostFree hipHostFree
  #define StreamSynchronize hipStreamSynchronize
  #define PeekAtLastError hipPeekAtLastError
#else
  #include <cuda_runtime.h>
  #define DEVFN __global__
  #define API_CHECK(x) do { auto _e=(x); if (_e!=cudaSuccess){\
    fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); exit(1);} } while(0)
  using stream_t = cudaStream_t; using event_t = cudaEvent_t; using device_prop_t = cudaDeviceProp;
  #define StreamCreateWithPriority cudaStreamCreateWithPriority
  #define StreamDestroy cudaStreamDestroy
  #define StreamBeginCapture cudaStreamBeginCapture
  #define StreamEndCapture cudaStreamEndCapture
  using Graph = cudaGraph_t; using GraphExec = cudaGraphExec_t;
  #define GraphInstantiate cudaGraphInstantiate
  #define GraphLaunch cudaGraphLaunch
  #define GraphDestroy cudaGraphDestroy
  #define GraphExecDestroy cudaGraphExecDestroy
  #define EventCreate cudaEventCreate
  #define EventRecord cudaEventRecord
  #define EventElapsed cudaEventElapsedTime
  #define EventDestroy cudaEventDestroy
  #define MemcpyAsync cudaMemcpyAsync
  #define MemHostToDevice cudaMemcpyHostToDevice
  #define MemDeviceToHost cudaMemcpyDeviceToHost
  #define GetDeviceCount cudaGetDeviceCount
  #define GetDeviceProperties cudaGetDeviceProperties
  #define SetDevice cudaSetDevice
  #define MallocAsync cudaMallocAsync
  #define FreeAsync cudaFreeAsync
  #define HostMalloc cudaHostAlloc
  #define HostFree cudaFreeHost
  #define StreamSynchronize cudaStreamSynchronize
  #define PeekAtLastError cudaPeekAtLastError
#endif

static inline void print_prop(const device_prop_t& p, int id){
  printf("GPU %d: %s\n", id, p.name);
  printf("  MultiProcessorCount: %d\n", p.multiProcessorCount);
  printf("  Memory Clock (kHz): %d, Memory Bus Width (bits): %d\n", p.memoryClockRate, p.memoryBusWidth);
  printf("  SharedMemPerBlock (B): %zu, MaxThreadsPerBlock: %d\n", (size_t)p.sharedMemPerBlock, p.maxThreadsPerBlock);
}

DEVFN void fma_kernel(const float* __restrict__ A, float* __restrict__ B, int n, int iters){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n){
    float x = A[idx];
    #pragma unroll 4
    for(int t=0;t<iters;++t){
      x = fmaf(x, 1.0001f, 0.0001f);
    }
    B[idx] = x;
  }
}

int main(int argc, char** argv){
  int dev_count=0; API_CHECK(GetDeviceCount(&dev_count));
  if(dev_count<=0){ fprintf(stderr, "No GPU device found.\n"); return 2; }
  int dev_id = (argc>1)? std::atoi(argv[1]) : 0; if(dev_id>=dev_count) dev_id=0;
  API_CHECK(SetDevice(dev_id));
  device_prop_t prop{}; API_CHECK(GetDeviceProperties(&prop, dev_id));
  print_prop(prop, dev_id);

  // Parameters
  const int N = (argc>2)? std::atoi(argv[2]) : (1<<20); // elements
  const int iters = (argc>3)? std::atoi(argv[3]) : 512; // work per element
  const int launches = (argc>4)? std::atoi(argv[4]) : 100; // graph replays

  // Create priority stream
  int leastPri=0, greatestPri=0; // CUDA/HIP both fill these
#if defined(__HIP_PLATFORM_AMD__)
  hipDeviceGetStreamPriorityRange(&leastPri, &greatestPri);
#else
  cudaDeviceGetStreamPriorityRange(&leastPri, &greatestPri);
#endif
  stream_t stream{}; API_CHECK(StreamCreateWithPriority(&stream, /*flags=*/0, /*priority=*/greatestPri));

  size_t bytes = size_t(N)*sizeof(float);
  // Pinned host buffers for deterministic transfers
  float* hA=nullptr; float* hB=nullptr; API_CHECK(HostMalloc((void**)&hA, bytes)); API_CHECK(HostMalloc((void**)&hB, bytes));
  for(int i=0;i<N;++i) hA[i] = 0.5f + 0.001f * (i%100);

  // Async device allocations
  float* dA=nullptr; float* dB=nullptr; API_CHECK(MallocAsync((void**)&dA, bytes, stream)); API_CHECK(MallocAsync((void**)&dB, bytes, stream));

  // Warmup kernel config
  dim3 block(256); dim3 grid((N + block.x - 1)/block.x);

  // Capture graph: H2D -> kernel -> D2H
  API_CHECK(StreamBeginCapture(stream));
  API_CHECK(MemcpyAsync(dA, hA, bytes, MemHostToDevice, stream));
#if defined(__HIP_PLATFORM_AMD__)
  hipLaunchKernelGGL(fma_kernel, grid, block, 0, stream, dA, dB, N, iters);
#else
  fma_kernel<<<grid, block, 0, stream>>>(dA, dB, N, iters);
#endif
  API_CHECK(MemcpyAsync(hB, dB, bytes, MemDeviceToHost, stream));
  Graph graph{}; API_CHECK(StreamEndCapture(stream, &graph));

  GraphExec graphExec{}; API_CHECK(GraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  event_t start, stop; API_CHECK(EventCreate(&start)); API_CHECK(EventCreate(&stop));
  API_CHECK(StreamSynchronize(stream));

  API_CHECK(EventRecord(start, stream));
  for(int i=0;i<launches;++i){ API_CHECK(GraphLaunch(graphExec, stream)); }
  API_CHECK(EventRecord(stop, stream));
  API_CHECK(StreamSynchronize(stream));

  float ms=0.f; API_CHECK(EventElapsed(&ms, start, stop));
  double avg_ms = ms / launches;
  double gb = (2.0 * bytes) / 1e9; // H2D + D2H per launch
  double gbps = gb / (avg_ms/1e3);
  double elems_per_s = double(N) / (avg_ms/1e3);

  // Basic validation
  if (std::isnan(hB[0])){ fprintf(stderr, "Validation failed: NaN detected.\n"); return 3; }

  printf("Healthcheck OK\n");
  printf("Launches: %d, N: %d, iters: %d\n", launches, N, iters);
  printf("Avg latency: %.3f ms, H2D+D2H: %.3f GB/s, Elem/s: %.0f\n", avg_ms, gbps, elems_per_s);

  // Cleanup
  API_CHECK(FreeAsync(dA, stream)); API_CHECK(FreeAsync(dB, stream));
  API_CHECK(StreamSynchronize(stream));
  API_CHECK(HostFree(hA)); API_CHECK(HostFree(hB));
  API_CHECK(GraphExecDestroy(graphExec)); API_CHECK(GraphDestroy(graph));
  API_CHECK(EventDestroy(start)); API_CHECK(EventDestroy(stop)); API_CHECK(StreamDestroy(stream));

  // Surface last error if any latent issue
  auto last = PeekAtLastError();
#if defined(__HIP_PLATFORM_AMD__)
  if(last!=hipSuccess){ fprintf(stderr, "HIP latent error: %s\n", hipGetErrorString(last)); return 4; }
#else
  if(last!=cudaSuccess){ fprintf(stderr, "CUDA latent error: %s\n", cudaGetErrorString(last)); return 4; }
#endif
  return 0;
}
```

### Build commands

CUDA:

```bash
nvcc -O3 -std=c++17 -arch=${SM_ARCH:-sm_90} -lineinfo topics/17-deployment-ops/code/ops_healthcheck.cpp -o ops_healthcheck
```

ROCm/HIP:

```bash
hipcc -O3 -std=c++17 --offload-arch=${GFX_ARCH:-gfx942} topics/17-deployment-ops/code/ops_healthcheck.cpp -o ops_healthcheck
```

### Run

```bash
# args: [device_id] [N_elems] [iters] [launches]
./ops_healthcheck 0 1048576 512 100
```

Expected output includes device properties and average latency/GB/s. Non-zero exit indicates the node should be quarantined.

## Profiling and Validation

### NVIDIA

- **Nsight Systems (timeline):**

```bash
nsys profile -t cuda,osrt -o nsys_ops ./ops_healthcheck 0 1048576 512 200
```

Check: one-time graph instantiate, then repeated `cudaGraphLaunch` with minimal CPU gaps.

- **Nsight Compute (metrics):**

```bash
ncu --set full --kernel-name fma_kernel ./ops_healthcheck 0 1048576 512 10
```

Counters to watch: `sm__throughput.avg.pct_of_peak_sustained_elapsed`, `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum.per_cycle_active`, `dram__throughput.avg.pct_of_peak_sustained_elapsed`.

### AMD

- **rocprof (timeline + counters):**

```bash
rocprof --hsa-trace --hip-trace --timestamp on --stats \
  --kernels fma_kernel --obj-tracking on --out-dir rocprof_ops ./ops_healthcheck 0 1048576 512 200
```

Counters: CU occupancy, LDS usage (should be near 0 here), DRAM busy %, and H2D/D2H bandwidth consistency.

**Pass thresholds (example, tune per hardware):**

- Graph replay cadence variance < 5% across 200 launches.
- GB/s within 70% of theoretical PCIe/NVLink/IF cap for pinned memory transfers.
- No ECC/RAS faults during run.

## Performance Checklist

- Containers pin exact CUDA/ROCm toolkits; base image digest is recorded.
- `nvidia-smi -pm 1` or `amd-smi set --power-performance high` applied; thermals < 85°C under steady state.
- CPU affinity and NUMA binding set to the GPU’s locality; IRQ balancing does not migrate GPU interrupts.
- HugeTLB enabled for host pools feeding pinned buffers; IOMMU set to passthrough when appropriate.
- Pinned host buffers for H2D/D2H; `cuda/hipMallocAsync` pools enabled; fragmentation monitored.
- CUDA/HIP Graphs used on decode path; warmup performed at startup.
- MIG/SR-IOV partitions sized to fit KV and batch targets; no overcommit.
- DCGM/SMI exporters feeding centralized monitoring; alerts on clocks throttling, ECC/RAS, and fan failures.
- Blue/green rollout with canary pods and load shedding; per-pod readiness gates based on healthcheck metrics.

## Troubleshooting

| Symptom                                   | Likely cause                         | Fix                                                                      |
| ----------------------------------------- | ------------------------------------ | ------------------------------------------------------------------------ |
| p99 spikes during decode                  | CPU thread migration / NUMA mismatch | Pin threads and memory to local NUMA; disable IRQ migration for GPU NICs |
| Throughput halved after upgrade           | Driver–runtime mismatch              | Align driver and container toolkit versions; rebuild images              |
| Random `Xid` (NVIDIA) or RAS errors (AMD) | Unstable clocks or thermals          | Set persistence/app clocks; ensure airflow; reduce power cap             |
| Host–device copies slow                   | Non-pinned memory                    | Use `cuda/hipHostMalloc`; reuse buffers                                  |
| Fragmentation OOMs despite free VRAM      | Many small `cuda/hipMalloc`          | Consolidate into async pool; preallocate arenas                          |
| High TTFT after deploy                    | Graphs not warmed                    | Run warmup on startup; cache plans and cuBLAS/rocBLAS handles            |
| Cross-node variability                    | Different BIOS/IOMMU/PCIe ASPM       | Standardize firmware; disable ASPM on GPU/root ports                     |
| NCCL/RCCL hangs                           | Firewall / IB/NVSwitch misconfig     | Verify fabric; set env for iface selection; test ring/allreduce          |
| MIG/SR-IOV pods crashloop                 | Resource name mismatch               | Match device plugin resource names and limits                            |
| Kernel launch jitter                      | Power management oscillation         | Lock performance state; monitor SM/CU clocks                             |

## Acceptance Criteria

- `ops_healthcheck` builds on both CUDA 12.x and ROCm 6.x toolchains.
- On a modern datacenter GPU, 100 graph replays complete in < 2 s with < 5% cadence variance.
- Pinned memory transfers achieve ≥ 10 GB/s on PCIe Gen4 x16 or higher (adjust for platform).
- No runtime errors; exit code 0; basic output non-NaN and stable across runs.
- Deployment checklist items are satisfied and documented for each node pool.

## Further Work

- Integrate DCGM/SMI exporters and set SLO-driven alerts (tokens/s, TTFT, p95/p99 per model).
- Implement a richer readiness probe that measures small decode loops with real model weights.
- Add MIG/SR-IOV-aware scheduler hints and QoS classes for multi-tenant isolation.
- Automate blue/green rollouts with traffic shadowing and state drain for KV caches.
- Extend healthcheck to verify NCCL/RCCL bandwidth and GPUDirect RDMA paths.

## Deployment Snippets

### Minimal Dockerfiles

CUDA base:

```dockerfile
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY topics/17-deployment-ops/code/ops_healthcheck.cpp ./
RUN nvcc -O3 -std=c++17 -arch=${SM_ARCH:-sm_90} -lineinfo ops_healthcheck.cpp -o /usr/local/bin/ops_healthcheck
CMD ["/usr/local/bin/ops_healthcheck", "0", "1048576", "512", "100"]
```

ROCm base:

```dockerfile
FROM rocm/dev-ubuntu-22.04:6.1
ENV HIP_VISIBLE_DEVICES=0
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY topics/17-deployment-ops/code/ops_healthcheck.cpp ./
RUN hipcc -O3 -std=c++17 --offload-arch=${GFX_ARCH:-gfx942} ops_healthcheck.cpp -o /usr/local/bin/ops_healthcheck
CMD ["/usr/local/bin/ops_healthcheck", "0", "1048576", "512", "100"]
```

### Kubernetes Pod (NVIDIA/AMD variants)

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-healthcheck
spec:
  restartPolicy: Never
  containers:
    - name: ops
      image: your-registry/ops-healthcheck:latest
      resources:
        limits:
          # For NVIDIA plugin
          nvidia.com/gpu: 1
          # For AMD plugin use:
          # amd.com/gpu: 1
      env:
        - name: OMP_PROC_BIND
          value: "true"
        - name: OMP_PLACES
          value: "cores"
      securityContext:
        runAsNonRoot: true
        allowPrivilegeEscalation: false
```

### Node Tuning (run with care)

```bash
# NVIDIA persistence
nvidia-smi -pm 1
# AMD performance state (example; verify on your platform)
amd-smi set --power-performance high || rocm-smi --setperflevel high
# NUMA binding example for worker
numactl --physcpubind=0-15 --membind=0 ./server_binary
```
