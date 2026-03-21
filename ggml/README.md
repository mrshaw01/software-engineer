# GGML

A knowledge base for understanding GGML, including its architecture, core concepts, and implementation details.

# Overview

GGML is a tensor library for machine learning. It provides tensor operations, automatic differentiation, optimization utilities, quantization support, and a backend system for running computation on different devices. The standalone `ggml` repository contains the core library, and the project is under active development.

## System Architecture

GGML can be viewed as a layered system that separates public APIs, core tensor and graph logic, backend abstraction, and device-specific implementations.

### 1. Application and API Layer

- `include/ggml.h`: core tensor types and tensor operation APIs
- `include/ggml-backend.h`: backend, device, buffer, and scheduler APIs

This layer defines the public interface used by applications. It includes core types such as `ggml_tensor` and `ggml_context`, as well as backend-facing abstractions such as `ggml_backend_t`, `ggml_backend_buffer_type_t`, and `ggml_backend_sched_t`.

### 2. Core Library Layer

- `src/ggml.c`: tensor operations and graph-related core logic
- `src/ggml-alloc.c`: memory allocation helpers
- `src/ggml-threading.cpp`: threading support
- `src/ggml-quants.c`: quantization routines
- `src/ggml-opt.cpp`: optimization-related logic

This layer implements the core tensor functionality of GGML, including graph construction and execution primitives, memory management, threading, quantization, and optimization utilities. It is largely hardware-agnostic.

### 3. Backend Abstraction Layer

- `include/ggml-backend.h`: backend API definitions
- `src/ggml-backend.cpp`: backend implementation logic
- `src/ggml-backend-reg.cpp`: backend registration and loading

This layer provides the abstraction that lets GGML target different devices through a common interface. The backend scheduler can coordinate multiple backends together, handling tensor placement, buffer allocation, and copies between backends.

### 4. Hardware Backend Layer

- `src/ggml-cpu`
- `src/ggml-cuda`
- `src/ggml-vulkan`
- `src/ggml-metal`
- `src/ggml-sycl`
- `src/ggml-opencl`
- `src/ggml-cann`
- other backends such as `ggml-hip`, `ggml-musa`, `ggml-openvino`, `ggml-webgpu`, `ggml-rpc`, `ggml-zdnn`, and `ggml-zendnn`

These directories contain device-specific implementations for CPUs, NVIDIA GPUs, Apple GPUs, Vulkan-capable GPUs, SYCL targets, Huawei accelerators, and other platforms.

## Core Components

### Tensor System

- `include/ggml.h`
- `src/ggml.c`

GGML represents tensors with `struct ggml_tensor`, the central data structure used for both plain data tensors and nodes in a computation graph. The structure stores the tensor’s element type, shape, byte strides, graph metadata, and data pointer. It also tracks the source tensors that produced the current tensor, which is how GGML encodes graph dependencies.

Key fields in `struct ggml_tensor` include:

- `type`: tensor element type, such as `GGML_TYPE_F32` or a quantized format
- `buffer`: backend buffer associated with the tensor
- `ne[GGML_MAX_DIMS]`: number of elements in each dimension
- `nb[GGML_MAX_DIMS]`: stride in bytes for each dimension
- `op`: operation that produces this tensor when it is a graph node
- `op_params`: packed parameters for the operation
- `flags`: tensor flags such as input, output, parameter, or compute
- `src[GGML_MAX_SRC]`: source tensors for the operation
- `view_src` and `view_offs`: metadata for tensor views
- `data`: pointer to the tensor’s underlying storage
- `name` and `extra`: optional metadata and backend-specific extra state

The `ne[]` and `nb[]` arrays are especially important because they let GGML represent both contiguous and non-contiguous tensors. This makes views, transposes, and other layout-transforming operations possible without always copying data.

Tensors are created from a `ggml_context`, which acts as the object arena for tensor and graph metadata. A context is initialized through `ggml_init(struct ggml_init_params)`. The initialization parameters include `mem_size`, `mem_buffer`, and `no_alloc`. When `no_alloc` is enabled, GGML does not allocate memory for tensor data immediately, which is useful when tensor storage will be assigned later by graph allocation logic or backend-managed buffers.

In practice, this means tensor creation and graph construction are separated from final data placement. You can first build the tensor objects and computation graph inside a context, then let allocators or backend buffers decide where the actual tensor data should live. This separation is one of the core design choices that allows GGML to support multiple execution backends efficiently.
