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

### Type System and Quantization

- `src/ggml.c`
- `src/ggml-quants.c`
- `src/ggml-cpu/quants.c`

GGML uses a type-traits system to describe both plain numeric types and quantized tensor formats. In `src/ggml.c`, the core table is `type_traits[GGML_TYPE_COUNT]`, and each entry records metadata such as the type name, block size, byte size, whether the type is quantized, and, when available, conversion hooks such as `to_float` and `from_float_ref`. Public helpers such as `ggml_blck_size()`, `ggml_type_size()`, `ggml_is_quantized()`, and `ggml_get_type_traits()` expose this metadata to the rest of the library.

For standard dense types such as `F32`, `F16`, `BF16`, and integer types, the block size is typically 1. For quantized formats, the block size matches the packing format used by that representation. For example, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, and `Q8_0` are registered as quantized types with explicit dequantization and reference quantization functions in the type-traits table.

The repo includes several families of quantized types. The classic block quantization family includes formats such as `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, and `Q8_1`. The K-quant family includes `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, and `Q8_K`. The type enum also includes IQ-family formats such as `IQ2_XXS`, `IQ2_XS`, `IQ3_XXS`, `IQ1_S`, `IQ4_NL`, `IQ3_S`, `IQ2_S`, `IQ4_XS`, and `IQ1_M`, as well as newer formats like `TQ1_0`, `TQ2_0`, `MXFP4`, and `NVFP4`.

In `src/ggml-quants.c`, GGML provides the reference implementations for quantization and dequantization. This includes functions such as `quantize_row_q4_0_ref()` for the classic formats and `quantize_row_q2_K_ref()` for K-quant formats. In the CPU quantization file, the K-quant section is explicitly described as “2–6 bit quantization in super-blocks,” which reflects the higher-level packing used by the K-family formats.

One important detail is that not every type exposes both conversion hooks. Many entries provide both `to_float` and `from_float_ref`, but some formats only provide one side in the core traits table. For example, `IQ2_XXS` and `IQ2_XS` currently expose `to_float` but have `from_float_ref = NULL`, while `Q8_1` has `from_float_ref` but no `to_float` entry in the shown type-traits definition. So it is more accurate to say that GGML supports per-type conversion hooks where they are implemented, rather than saying every type provides both unconditionally.

The CPU-specific quantization layer in `src/ggml-cpu/quants.c` builds on the reference routines and provides execution-oriented kernels for quantization, dequantization, and dot products on quantized blocks. For example, it wraps functions such as `quantize_row_q4_0()` and `quantize_row_q2_K()`, and it also contains specialized dot-product kernels such as `ggml_vec_dot_iq2_xxs_q8_K_generic()` for mixed-format arithmetic on packed blocks. This is where architecture-tuned CPU paths are organized, while other hardware backends can provide their own optimized implementations.
