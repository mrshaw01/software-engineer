# CUDA/HIP Graphs for Fast Execution

High-performance inference systems rely on reducing the overhead of kernel launches and maximizing reuse of static execution patterns. CUDA Graphs (NVIDIA) and HIP Graphs (AMD ROCm) enable this by capturing and reusing sequences of GPU operations. This section explains how graph-based execution improves latency and throughput during LLM inference.

## Overview

During autoregressive inference, the decoding loop involves repetitive, latency-sensitive kernel executions (e.g., attention, matmul, normalization). Launching these kernels individually incurs overhead due to CPU–GPU synchronization and kernel setup. CUDA/HIP graphs allow pre-recording a computation graph and launching it as a single entity with significantly less overhead.

## Key Concepts

### CUDA/HIP Graph

A **graph** is a DAG (Directed Acyclic Graph) of GPU operations (e.g., memory copies, kernel executions) that is recorded once and can be replayed many times.

- **Capture Phase**: The application records a sequence of GPU operations into a graph object.
- **Instantiation Phase**: The graph is compiled into an executable representation.
- **Launch Phase**: The executable graph is launched, skipping host-side launch overhead.

### Graph Types

- **Stream Capture**: Dynamically captures operations submitted to a CUDA stream into a graph.
- **Manual Graph Construction**: API-level control to build nodes and dependencies explicitly.

## Benefits for LLM Inference

### 1. Reduced Launch Overhead

Graph replay is faster than issuing individual kernel launches, especially for small batch sizes or fast-decoding loops (e.g., greedy sampling).

### 2. Reuse of Execution Paths

Since the decoding loop structure is mostly static (e.g., KV cache access, rotary embedding, attention projection), the entire loop can be encapsulated in a graph.

### 3. Deterministic Execution

Graph-based scheduling removes launch-time variability, improving latency jitter in production environments.

## Integration into LLM Serving

### Prefill vs. Decode Phase

- **Prefill Phase**: Less suited for graph capture due to variable-length context and dynamic padding.
- **Decode Phase**: Ideal for CUDA/HIP graphs since the shape is typically `[batch, 1]`, and logic is fixed.

### Workflow in vLLM (example)

1. Decode step is captured once using `cudaStreamBeginCapture()`.
2. All necessary kernels (e.g., attention, matmul, normalization, sampling) are recorded.
3. Graph is instantiated and stored per model config or shape signature.
4. Future decode invocations reuse the captured graph with different data bindings.

## Considerations

### Limitations

- Graphs are static: input/output shapes and execution logic must not change across invocations.
- Graph capture may fail if unsupported APIs or conditional branching is involved.
- Compilation overhead in `cudaGraphInstantiate()` can be high; best amortized over many reuses.

### Memory Binding

- Device memory for inputs/outputs must be reused across graph launches.
- Changing bindings requires re-instantiating or using updateable parameters (e.g., `cudaGraphExecKernelNodeSetParams`).

### Graph Pooling

Serving systems often maintain a pool of instantiated graphs indexed by:

- `batch_size`
- `sequence_length`
- `KV_cache_shape`

This avoids redundant capture/instantiation and enables amortized fast-path execution.

## Example: Pseudocode

```cpp
// 1. Begin stream capture
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

// 2. Submit kernels
run_attention_kernel<<<...>>>(...);
run_mlp_kernel<<<...>>>(...);

// 3. End and instantiate graph
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&graphExec, graph, ...);

// 4. Launch graph in decode loop
cudaGraphLaunch(graphExec, stream);
```

## Performance Impact

| Optimization         | Benefit (typical)               |
| -------------------- | ------------------------------- |
| Kernel launch bypass | \~1.5–2× lower latency          |
| Static scheduling    | Reduced latency jitter          |
| Graph reuse          | High throughput in batch decode |

In benchmarks for LLM inference (e.g., vLLM, FasterTransformer), decode steps see up to **40–50% speedup** when using CUDA Graphs vs. naive kernel launches, particularly at small batch sizes.

## Summary

CUDA and HIP Graphs are essential for optimizing the repeated decode step in LLM inference. By capturing and reusing computation graphs, serving systems can significantly reduce launch overhead, stabilize latency, and scale efficiently. Proper integration requires careful control over input shapes and memory reuse, but the performance gains justify this effort in production-grade inference systems.
