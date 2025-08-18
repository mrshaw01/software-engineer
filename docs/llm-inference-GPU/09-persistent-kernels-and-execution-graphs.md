# Persistent Kernels & Execution Graphs

## 2. Summary

This topic shows how to reduce launch overhead and improve decode responsiveness in LLM inference using two techniques: (1) persistent kernels that stay resident on the GPU and pull work from a device-side queue, and (2) CUDA/HIP execution graphs that capture a sequence of operations and replay them with minimal CPU intervention. You will build runnable examples for NVIDIA (CUDA) and AMD (ROCm/HIP), profile them, and quantify gains under decode-like micro-workloads.

## 3. Why It Matters for LLM Inference

Decode frequently executes many short, latency-sensitive kernels per token (layernorms, projections, small attention primitives, dequant, epilogues). Kernel launches can cost microseconds each; multiplied by O(10²) kernels/token, launch overhead alone can approach milliseconds, directly limiting tokens/sec and p50 latency. Persistent kernels collapse many launches into one; execution graphs amortize launch overhead and improve CPU–GPU submission efficiency. Prefill benefits less (kernels are bigger and fewer), but graphs still help by lowering CPU overhead at high batch/sequence.

## 4. Key Concepts and Formulas

- Kernel launch overhead: let $t_L$ be host→device launch latency (typically \~5–20 μs depending on stack and platform).
- Per-token launch cost if $K$ kernels/token: $T_{\text{launch}} \approx K \cdot t_L$.
- If compute per token is $T_{\text{comp}}$, fractional overhead is $f \approx \frac{K \cdot t_L}{T_{\text{comp}} + K \cdot t_L}$.
- Execution graph replaces $K$ launches by one graph launch: residual $T^{\text{graph}}_{\text{launch}}$ is usually a small multiple of $t_L$ but independent of $K$.
- Persistent kernel replaces $K$ launches by a single long-running kernel, making $T_{\text{launch}}$ ≈ one-time.
- Example instantiation: $K=150$, $t_L=6\,\mu s$ → $T_{\text{launch}}\approx 0.90\,ms$. If $T_{\text{comp}}=18\,ms$, launch overhead ≈ 4.8%. If $T_{\text{comp}}=6\,ms$ (small models), overhead ≈ 13%. Both techniques can meaningfully reduce this.

## 5. GPU Deep Dive

### NVIDIA

- **Warps/SMs**: 32-thread warps scheduled on SMs; persistent kernels often launch ≈1 block/SM to avoid preemption and ensure forward progress.
- **Tensor Cores**: When persistent workers call GEMM micro-kernels, ensure shapes are Tensor Core–friendly (e.g., multiples of 8/16 in FP16/BF16) to avoid silent path downgrades.
- **Memory hierarchy**: Use L2 residency where possible. Persistent design helps by reusing coefficients and metadata without re-launch stalls.
- **Execution Graphs**: CUDA Graphs capture/cuGraphInstantiate -> cuGraphLaunch; prefer stream capture for simplicity and parameter updates with graph node param setters.

### AMD

- **Wavefronts/CUs**: 64-thread wavefronts on Compute Units (CUs). Launch ≈1 block/CU in persistent mode (or a tuned fraction) to balance occupancy and fairness.
- **MFMA/XDLOPs**: Ensure matrix tiles match MFMA-friendly sizes; persistent workers should batch micro-ops to hit MFMA paths.
- **LDS**: Exploit LDS for small working sets (layernorm stats, epilogue vectors). Persistent kernels avoid re-initialization overhead per launch.
- **HIP Graphs**: `hipGraph` mirrors CUDA Graphs API; stream capture then instantiate `hipGraphExec_t` and replay.

## 6. Implementation

Two minimal, runnable single-source examples (guarded for CUDA/HIP):

- `persistent_decode.cu`: persistent workers pulling “token steps” and applying a simple AXPY to emulate per-token micro-work.
- `graphs_axpy.cu`: baseline loop vs. stream-captured execution graph performing N short kernels.

### Code: topics/09-persistent-kernels-and-execution-graphs/code/persistent_decode.cu

```cpp
// Single-source CUDA/HIP persistent "decode-like" kernel demo.
// Builds with: nvcc or hipcc (see commands below).
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <chrono>

#if defined(__HIP_PLATFORM_AMD__)
  #include <hip/hip_runtime.h>
  #define DEVAPI hip
  #define API_CHECK(x) do { auto _e = (x); if (_e != hipSuccess) { \
    fprintf(stderr,"HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_e)); exit(1);} } while(0)
  #define DEVICE_PROP hipDeviceProp_t
  #define GET_PROP(dev,prop) hipGetDeviceProperties(&(prop),(dev))
  #define MEMCPY_DEV_TO_DEV hipMemcpyDeviceToDevice
  #define MEMCPY_HOST_TO_DEV hipMemcpyHtoD
  #define MEMCPY_DEV_TO_HOST hipMemcpyDtoH
  #define MEMCPY hipMemcpy
  #define EVENT_CREATE hipEventCreate
  #define EVENT_RECORD hipEventRecord
  #define EVENT_SYNC hipEventSynchronize
  #define EVENT_ELAPSED hipEventElapsedTime
  #define STREAM_CREATE hipStreamCreate
  #define STREAM_DESTROY hipStreamDestroy
  #define DEVICE_SYNC hipDeviceSynchronize
#else
  #include <cuda_runtime.h>
  #define DEVAPI cuda
  #define API_CHECK(x) do { auto _e = (x); if (_e != cudaSuccess) { \
    fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); exit(1);} } while(0)
  #define DEVICE_PROP cudaDeviceProp
  #define GET_PROP(dev,prop) cudaGetDeviceProperties(&(prop),(dev))
  #define EVENT_CREATE cudaEventCreate
  #define EVENT_RECORD cudaEventRecord
  #define EVENT_SYNC cudaEventSynchronize
  #define EVENT_ELAPSED cudaEventElapsedTime
  #define STREAM_CREATE cudaStreamCreate
  #define STREAM_DESTROY cudaStreamDestroy
  #define DEVICE_SYNC cudaDeviceSynchronize
#endif

struct Control {
  int total_steps;
  int step;     // incremented atomically by blocks
  int stop;     // not used in this demo, reserved
};

__global__ void persistent_axpy(const float* __restrict__ x,
                                float* __restrict__ y,
                                int n, float a, Control* ctrl) {
  extern __shared__ int smem[];
  while (true) {
    int claimed = 0;
    if (threadIdx.x == 0) {
      // Each block claims one "token step"
#if defined(__HIP_PLATFORM_AMD__) || !defined(__CUDA_ARCH__)
      claimed = atomicAdd(&(ctrl->step), 1);
#else
      claimed = atomicAdd(&(ctrl->step), 1);
#endif
    }
    __shared__ int s;
    if (threadIdx.x == 0) s = claimed;
    __syncthreads();
    int step_local = s;
    if (step_local >= ctrl->total_steps || ctrl->stop) break;

    // Grid-stride loop: all blocks cooperate on the same compute shape per step.
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
      y[i] = a * x[i] + y[i];
    }
    __syncthreads();
    // Loop to claim next step.
  }
}

static double checksum(const std::vector<float>& v) {
  long double acc = 0.0;
  for (float x : v) acc += x;
  return (double)acc;
}

int main(int argc, char** argv) {
  int n = 1<<20;          // vector length
  int steps = 500;        // number of "tokens"
  float a = 0.001f;       // small workload per token
  int user_blocks = 0;    // 0 = auto: 1 block per SM/CU
  int threads = 256;

  for (int i=1; i<argc; ++i) {
    if (sscanf(argv[i], "--n=%d", &n)==1) continue;
    if (sscanf(argv[i], "--steps=%d", &steps)==1) continue;
    if (sscanf(argv[i], "--blocks=%d", &user_blocks)==1) continue;
    if (sscanf(argv[i], "--threads=%d", &threads)==1) continue;
    if (sscanf(argv[i], "--a=%f", &a)==1) continue;
  }

  DEVICE_PROP prop{};
  API_CHECK(GET_PROP(0, prop));
  int sms = prop.multiProcessorCount;
  int blocks = (user_blocks>0)? user_blocks : sms;

  std::vector<float> hx(n, 1.0f), hy(n, 0.0f), hy_ref(n, 0.0f);
  // CPU reference: y += steps * a * x
  for (int i=0;i<n;++i) hy_ref[i] = hy[i] + steps * a * hx[i];

  float *dx=nullptr, *dy=nullptr;
  Control *dctrl=nullptr;
#if defined(__HIP_PLATFORM_AMD__)
  API_CHECK(hipMalloc((void**)&dx, n*sizeof(float)));
  API_CHECK(hipMalloc((void**)&dy, n*sizeof(float)));
  API_CHECK(hipMalloc((void**)&dctrl, sizeof(Control)));
  API_CHECK(hipMemcpy(dx, hx.data(), n*sizeof(float), hipMemcpyHostToDevice));
  API_CHECK(hipMemcpy(dy, hy.data(), n*sizeof(float), hipMemcpyHostToDevice));
#else
  API_CHECK(cudaMalloc((void**)&dx, n*sizeof(float)));
  API_CHECK(cudaMalloc((void**)&dy, n*sizeof(float)));
  API_CHECK(cudaMalloc((void**)&dctrl, sizeof(Control)));
  API_CHECK(cudaMemcpy(dx, hx.data(), n*sizeof(float), cudaMemcpyHostToDevice));
  API_CHECK(cudaMemcpy(dy, hy.data(), n*sizeof(float), cudaMemcpyHostToDevice));
#endif

  Control hctrl{steps, 0, 0};
#if defined(__HIP_PLATFORM_AMD__)
  API_CHECK(hipMemcpy(dctrl, &hctrl, sizeof(Control), hipMemcpyHostToDevice));
#else
  API_CHECK(cudaMemcpy(dctrl, &hctrl, sizeof(Control), cudaMemcpyHostToDevice));
#endif

  // Timing
#if defined(__HIP_PLATFORM_AMD__)
  hipEvent_t e0, e1; API_CHECK(EVENT_CREATE(&e0)); API_CHECK(EVENT_CREATE(&e1));
  API_CHECK(EVENT_RECORD(e0));
  hipLaunchKernelGGL(persistent_axpy, dim3(blocks), dim3(threads), 0, 0, dx, dy, n, a, dctrl);
  API_CHECK(DEVICE_SYNC());
  API_CHECK(EVENT_RECORD(e1));
  API_CHECK(EVENT_SYNC(e1));
  float ms=0; API_CHECK(EVENT_ELAPSED(&ms, e0, e1));
#else
  cudaEvent_t e0, e1; API_CHECK(EVENT_CREATE(&e0)); API_CHECK(EVENT_CREATE(&e1));
  API_CHECK(EVENT_RECORD(e0));
  persistent_axpy<<<blocks, threads, 0>>>(dx, dy, n, a, dctrl);
  API_CHECK(DEVICE_SYNC());
  API_CHECK(EVENT_RECORD(e1));
  API_CHECK(EVENT_SYNC(e1));
  float ms=0; API_CHECK(EVENT_ELAPSED(&ms, e0, e1));
#endif

  // Copy back and validate
#if defined(__HIP_PLATFORM_AMD__)
  API_CHECK(hipMemcpy(hy.data(), dy, n*sizeof(float), hipMemcpyDeviceToHost));
#else
  API_CHECK(cudaMemcpy(hy.data(), dy, n*sizeof(float), cudaMemcpyDeviceToHost));
#endif

  double err = 0.0;
  for (int i=0;i<n;++i) err = fmax(err, fabs((double)hy[i] - (double)hy_ref[i]));
  double tok_per_s = (steps) / (ms / 1000.0);

  printf("Persistent kernel: n=%d steps=%d blocks=%d threads=%d\n", n, steps, blocks, threads);
  printf("Elapsed: %.3f ms, tokens/s: %.2f, max_abs_err=%.3e\n", ms, tok_per_s, err);

#if defined(__HIP_PLATFORM_AMD__)
  hipFree(dx); hipFree(dy); hipFree(dctrl);
#else
  cudaFree(dx); cudaFree(dy); cudaFree(dctrl);
#endif
  return (err < 1e-5) ? 0 : 1;
}
```

### Code: topics/09-persistent-kernels-and-execution-graphs/code/graphs_axpy.cu

```cpp
// Single-source CUDA/HIP execution graph demo vs. baseline loop.
// Captures N short kernels and replays the graph.
// Builds with: nvcc or hipcc (see commands below).
#include <cstdio>
#include <vector>
#include <cmath>

#if defined(__HIP_PLATFORM_AMD__)
  #include <hip/hip_runtime.h>
  #define DEVAPI hip
  #define API_CHECK(x) do { auto _e = (x); if (_e != hipSuccess) { \
    fprintf(stderr,"HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_e)); exit(1);} } while(0)
  using GraphT = hipGraph_t;
  using GraphExecT = hipGraphExec_t;
  #define STREAM_CREATE hipStreamCreate
  #define STREAM_DESTROY hipStreamDestroy
  #define STREAM_BEGIN_CAPTURE hipStreamBeginCapture
  #define STREAM_END_CAPTURE hipStreamEndCapture
  #define GRAPH_INSTANTIATE hipGraphInstantiate
  #define GRAPH_LAUNCH hipGraphLaunch
  #define DEVICE_SYNC hipDeviceSynchronize
  #define EVENT_CREATE hipEventCreate
  #define EVENT_RECORD hipEventRecord
  #define EVENT_SYNC hipEventSynchronize
  #define EVENT_ELAPSED hipEventElapsedTime
#else
  #include <cuda_runtime.h>
  #define DEVAPI cuda
  #define API_CHECK(x) do { auto _e = (x); if (_e != cudaSuccess) { \
    fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); exit(1);} } while(0)
  using GraphT = cudaGraph_t;
  using GraphExecT = cudaGraphExec_t;
  #define STREAM_CREATE cudaStreamCreate
  #define STREAM_DESTROY cudaStreamDestroy
  #define STREAM_BEGIN_CAPTURE cudaStreamBeginCapture
  #define STREAM_END_CAPTURE cudaStreamEndCapture
  #define GRAPH_INSTANTIATE cudaGraphInstantiate
  #define GRAPH_LAUNCH cudaGraphLaunch
  #define DEVICE_SYNC cudaDeviceSynchronize
  #define EVENT_CREATE cudaEventCreate
  #define EVENT_RECORD cudaEventRecord
  #define EVENT_SYNC cudaEventSynchronize
  #define EVENT_ELAPSED cudaEventElapsedTime
#endif

__global__ void axpy_kernel(const float* __restrict__ x,
                            float* __restrict__ y,
                            int n, float a) {
  for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += gridDim.x*blockDim.x) {
    y[i] += a * x[i];
  }
}

static float time_ms(std::function<void()> fn) {
#if defined(__HIP_PLATFORM_AMD__)
  hipEvent_t e0,e1; API_CHECK(EVENT_CREATE(&e0)); API_CHECK(EVENT_CREATE(&e1));
  API_CHECK(EVENT_RECORD(e0));
  fn();
  API_CHECK(EVENT_RECORD(e1)); API_CHECK(EVENT_SYNC(e1));
  float ms=0; API_CHECK(EVENT_ELAPSED(&ms,e0,e1)); return ms;
#else
  cudaEvent_t e0,e1; API_CHECK(EVENT_CREATE(&e0)); API_CHECK(EVENT_CREATE(&e1));
  API_CHECK(EVENT_RECORD(e0));
  fn();
  API_CHECK(EVENT_RECORD(e1)); API_CHECK(EVENT_SYNC(e1));
  float ms=0; API_CHECK(EVENT_ELAPSED(&ms,e0,e1)); return ms;
#endif
}

int main(int argc, char** argv) {
  int n = 1<<20;
  int steps = 500;         // number of short kernels
  float a = 0.001f;
  int blocks = 128, threads = 128;

  for (int i=1;i<argc;++i) {
    if (sscanf(argv[i], "--n=%d",&n)==1) continue;
    if (sscanf(argv[i], "--steps=%d",&steps)==1) continue;
    if (sscanf(argv[i], "--blocks=%d",&blocks)==1) continue;
    if (sscanf(argv[i], "--threads=%d",&threads)==1) continue;
    if (sscanf(argv[i], "--a=%f",&a)==1) continue;
  }

  std::vector<float> hx(n, 1.0f), hy(n, 0.0f), ref(n, 0.0f);
  float *dx=nullptr, *dy=nullptr;
#if defined(__HIP_PLATFORM_AMD__)
  API_CHECK(hipMalloc(&dx, n*sizeof(float)));
  API_CHECK(hipMalloc(&dy, n*sizeof(float)));
  API_CHECK(hipMemcpy(dx, hx.data(), n*sizeof(float), hipMemcpyHostToDevice));
  API_CHECK(hipMemcpy(dy, hy.data(), n*sizeof(float), hipMemcpyHostToDevice));
#else
  API_CHECK(cudaMalloc(&dx, n*sizeof(float)));
  API_CHECK(cudaMalloc(&dy, n*sizeof(float)));
  API_CHECK(cudaMemcpy(dx, hx.data(), n*sizeof(float), cudaMemcpyHostToDevice));
  API_CHECK(cudaMemcpy(dy, hy.data(), n*sizeof(float), cudaMemcpyHostToDevice));
#endif

  // Baseline: launch short kernels in a loop
  float baseline_ms = time_ms([&](){
#if defined(__HIP_PLATFORM_AMD__)
    for (int s=0;s<steps;++s) {
      hipLaunchKernelGGL(axpy_kernel, dim3(blocks), dim3(threads), 0, 0, dx, dy, n, a);
    }
    API_CHECK(DEVICE_SYNC());
#else
    for (int s=0;s<steps;++s) {
      axpy_kernel<<<blocks,threads>>>(dx,dy,n,a);
    }
    API_CHECK(DEVICE_SYNC());
#endif
  });

  // Graph: capture the same sequence and replay once
  float graph_ms = 0.0f;
  {
#if defined(__HIP_PLATFORM_AMD__)
    hipStream_t stream; API_CHECK(STREAM_CREATE(&stream));
    GraphT graph{}; GraphExecT exec{};
    API_CHECK(STREAM_BEGIN_CAPTURE(stream, hipStreamCaptureModeGlobal));
    for (int s=0;s<steps;++s) {
      hipLaunchKernelGGL(axpy_kernel, dim3(blocks), dim3(threads), 0, stream, dx, dy, n, a);
    }
    API_CHECK(STREAM_END_CAPTURE(stream, &graph));
    API_CHECK(GRAPH_INSTANTIATE(&exec, graph, nullptr, nullptr, 0));
    graph_ms = time_ms([&](){ API_CHECK(GRAPH_LAUNCH(exec, stream)); API_CHECK(DEVICE_SYNC()); });
    hipGraphDestroy(graph); hipGraphExecDestroy(exec);
    API_CHECK(STREAM_DESTROY(stream));
#else
    cudaStream_t stream; API_CHECK(STREAM_CREATE(&stream));
    GraphT graph{}; GraphExecT exec{};
    API_CHECK(STREAM_BEGIN_CAPTURE(stream, cudaStreamCaptureModeGlobal));
    for (int s=0;s<steps;++s) {
      axpy_kernel<<<blocks,threads,0,stream>>>(dx,dy,n,a);
    }
    API_CHECK(STREAM_END_CAPTURE(stream, &graph));
    API_CHECK(GRAPH_INSTANTIATE(&exec, graph, nullptr, nullptr, 0));
    graph_ms = time_ms([&](){ API_CHECK(GRAPH_LAUNCH(exec, stream)); API_CHECK(DEVICE_SYNC()); });
    cudaGraphDestroy(graph); cudaGraphExecDestroy(exec);
    API_CHECK(STREAM_DESTROY(stream));
#endif
  }

#if defined(__HIP_PLATFORM_AMD__)
  API_CHECK(hipMemcpy(hy.data(), dy, n*sizeof(float), hipMemcpyDeviceToHost));
#else
  API_CHECK(cudaMemcpy(hy.data(), dy, n*sizeof(float), cudaMemcpyDeviceToHost));
#endif
  for (int i=0;i<n;++i) ref[i] = steps * a * hx[i];
  double max_err = 0.0;
  for (int i=0;i<n;++i) max_err = fmax(max_err, fabs((double)hy[i] - (double)ref[i]));

  double base_tok_s  = steps / (baseline_ms / 1000.0);
  double graph_tok_s = steps / (graph_ms    / 1000.0);
  printf("AXPY short-kernel demo: n=%d steps=%d blocks=%d threads=%d\n", n, steps, blocks, threads);
  printf("Baseline loop: %.3f ms, tokens/s %.2f\n", baseline_ms, base_tok_s);
  printf("Graph replay : %.3f ms, tokens/s %.2f (speedup %.2fx)\n", graph_ms, graph_tok_s, baseline_ms/graph_ms);
  printf("max_abs_err=%.3e\n", max_err);
  return (max_err < 1e-5) ? 0 : 1;
}
```

### Build commands

NVIDIA (CUDA 12.x):

```bash
# Persistent kernel
nvcc -O3 -std=c++17 -arch=${SM_ARCH} -lineinfo topics/09-persistent-kernels-and-execution-graphs/code/persistent_decode.cu -o bin/persistent_decode
# Graphs
nvcc -O3 -std=c++17 -arch=${SM_ARCH} -lineinfo topics/09-persistent-kernels-and-execution-graphs/code/graphs_axpy.cu -o bin/graphs_axpy
```

AMD ROCm/HIP 6.x:

```bash
# Retrieve your GFX arch (e.g., gfx90a, gfx942, gfx1100)
hipcc -O3 -std=c++17 --offload-arch=${GFX_ARCH} topics/09-persistent-kernels-and-execution-graphs/code/persistent_decode.cu -o bin/persistent_decode_hip
hipcc -O3 -std=c++17 --offload-arch=${GFX_ARCH} topics/09-persistent-kernels-and-execution-graphs/code/graphs_axpy.cu -o bin/graphs_axpy_hip
```

### Run examples

```bash
# Persistent: emulate 500 tokens, 1 block/SM (auto)
./bin/persistent_decode --n=1048576 --steps=500
# Graphs: compare baseline vs graph for 500 short kernels
./bin/graphs_axpy --n=1048576 --steps=500
```

## 7. Profiling and Validation

### NVIDIA

- **Nsight Systems (timeline & launch overhead)**:

  ```bash
  nsys profile --stats=true -o nsys_persist ./bin/persistent_decode --steps=500
  nsys profile --stats=true -o nsys_graphs ./bin/graphs_axpy --steps=500
  ```

  Inspect: process CPU time vs GPU time, kernel submission count, graph launch node(s).

- **Nsight Compute (per-kernel metrics)**:

  ```bash
  ncu --set full --target-processes all ./bin/persistent_decode --steps=500
  ncu --set full --target-processes all ./bin/graphs_axpy --steps=500
  ```

  Focus counters:

  - `sm__throughput.avg.pct_of_peak_sustained_active` ≥ 30% for this toy workload.
  - `gpu__time_duration.sum` reduced for graph vs baseline.
  - Kernel launch count drastically lower (view in Nsight Systems summary).

### AMD

- **rocprof (HIP/HSA traces)**:

  ```bash
  rocprof --hip-trace --hsa-trace --timestamp on --stats ./bin/persistent_decode_hip --steps=500
  rocprof --hip-trace --hsa-trace --timestamp on --stats ./bin/graphs_axpy_hip --steps=500
  ```

  Check HIP API call counts (kernel launches) and total GPU time. Confirm fewer launches and shorter host-side submission time for the graph run.

- **Omnitrace (optional)**: capture timelines to verify graph node replay vs per-kernel launch.

**Pass thresholds (for this demo on modern GPUs):**

- Graphs: ≥1.2× speedup vs baseline when `steps` is large and per-kernel work is small (e.g., `n=2^20`, `steps≥300`).
- Persistent: ≥1.2× vs baseline loop under similar settings; often matches or beats graph replay for very small `n`.

## 8. Performance Checklist

- [ ] Use ≈1 block per SM/CU in persistent mode (`--blocks=0` auto) to reduce scheduler interference.
- [ ] Ensure grid-stride loops for full-device participation per “token step”.
- [ ] Keep per-step work small enough to expose launch overhead; otherwise expect limited gains.
- [ ] For graphs, prefer **stream capture** with a single `GraphExec` replay.
- [ ] Validate numerical equivalence vs CPU or analytic reference (max abs error < 1e-5 here).
- [ ] Measure tokens/sec and kernel launch counts before/after.
- [ ] On NVIDIA, confirm no silent Tensor Core downgrades if you extend this to GEMMs.

## 9. Troubleshooting

| Symptom                         | Likely cause                                         | Fix                                                                          |
| ------------------------------- | ---------------------------------------------------- | ---------------------------------------------------------------------------- |
| Persistent kernel never exits   | `total_steps`/`stop` misuse or atomic not advancing  | Ensure `ctrl->step` increments and `total_steps` set on device before launch |
| Low occupancy in persistent run | Too few blocks or large shared memory usage          | Start with 1 block/SM and 128–256 threads; minimize shared memory            |
| Graph capture fails             | Illegal API during capture (e.g., sync)              | Move blocking calls outside capture region                                   |
| No speedup from graphs          | Per-kernel work too large; submission not bottleneck | Reduce kernel size or increase `steps` to stress launch overhead             |
| HIP build errors                | ROCm version lacks Graph features                    | Use ROCm ≥5.2 and update HIP; or disable graph demo                          |
| Validation mismatch             | Race in device control or wrong ref calc             | Keep control struct immutable during run; verify AXPY closed form            |
| Spiky timeline                  | CPU thread preemption                                | Pin CPU thread or use `taskset`/`numactl`; keep the run simple               |
| SM starvation                   | Too few steps vs blocks                              | Ensure `steps` ≫ blocks so all blocks do useful work                         |

## 10. Acceptance Criteria

- Documentation explains persistent kernels and execution graphs with decode implications and concrete math.
- Both examples compile and run on CUDA 12.x and ROCm/HIP 6.x.
- Profiling commands included for Nsight Systems/Compute and rocprof; user can observe reduced kernel launch counts.
- Numeric validation passes (`max_abs_err < 1e-5`); tokens/sec reported.
- With `n=2^20, steps=500`, the graph run shows ≥1.2× speedup vs baseline on at least one supported GPU, and the persistent run achieves ≥1.2× vs baseline.

## 11. Further Work

- Replace AXPY with a fused epilogue microkernel (bias+gelu+residual) to better mimic decode tail.
- Integrate CUDA Graph node parameter updates to vary per-step scalars without re-instantiation.
- Use cooperative groups (`cg::this_grid().sync()`) on CUDA to coordinate per-step gridwide barriers (requires cooperative launch).
- Extend persistent workers to fetch work from a **device queue** fed by the host for multi-request decode.
- Add end-to-end decode harness: KV cache reads, rotary, attention, MLP epilogues; measure TTFT and tokens/sec with/without graphs/persistent.
