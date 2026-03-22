# GGML

A knowledge base for understanding GGML, including its architecture, core concepts, and implementation details.

- [Tensor Operations and Computation Graphs](./1_TensorOperationsAndComputationGraphs.md)
- [Quantization System](./2_QuantizationSystem.md)
- [Memory Management and Allocation](./3_MemoryManagementAndAllocation.md)
- [Backend Interface and Abstraction](./4_BackendInterfaceAndAbstraction.md)
- [Graph Scheduler and Multi-Backend Execution](./5_GraphSchedulerAndMultiBackendExecution.md)

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

## Backend System

The backend layer lets GGML execute the same computation graph across different devices through a common interface. The public API is defined in `include/ggml-backend.h`, while the core implementation lives in `src/ggml-backend.cpp` and the registry and dynamic-loading logic live in `src/ggml-backend-reg.cpp`.

- `include/ggml-backend.h`
- `src/ggml-backend.cpp`
- `src/ggml-backend-reg.cpp`

### Backend Registration

GGML maintains a global backend registry that separates a backend family from the concrete devices it exposes.

The public API includes functions such as:

- `ggml_backend_register()`
- `ggml_backend_device_register()`
- `ggml_backend_reg_count()`
- `ggml_backend_reg_get()`
- `ggml_backend_dev_count()`
- `ggml_backend_dev_get()`

In this model:

- `ggml_backend_reg_t` represents a registered backend family
- `ggml_backend_dev_t` represents a concrete device exposed by that backend
- `ggml_backend_buffer_type_t` describes a buffer or allocation strategy associated with a backend or device
- `ggml_backend_dev_init()` creates an executable backend instance from a device

Each backend device can report metadata and capabilities such as:

- device name and description
- device type
- free and total memory
- async execution support
- host-buffer support
- buffer-from-host-pointer support
- event support and synchronization capabilities

GGML also supports dynamic backend loading. When `GGML_BACKEND_DL` is enabled, backend libraries can be discovered and loaded at runtime, and `GGML_BACKEND_DIR` can be used to specify the directory containing those backend libraries. Runtime helpers such as `ggml_backend_load_all()` and `ggml_backend_load_all_from_path()` are used to probe and load available backends.

### Multi-Backend Scheduling

GGML provides a scheduler, `ggml_backend_sched_t`, to coordinate execution across multiple backends. It is created with `ggml_backend_sched_new()`, which takes an ordered list of backends and buffer types. Lower backend indices have higher priority.

At a high level, the scheduler works as follows:

1. **Backend assignment**
   Assign graph nodes to backends based on existing tensor placement, backend priority, and whether the backend supports the operation.

2. **Graph splitting**
   Split the full computation graph into backend-specific subgraphs so each split can run on a single backend. In the current implementation, `ggml_backend_sched_split_graph()` performs backend assignment and graph partitioning.

3. **Memory allocation**
   Use scheduler allocation APIs such as `ggml_backend_sched_reserve()` and related graph-allocation logic to place tensors into appropriate backend buffers.

4. **Cross-backend copies**
   Insert or manage tensor copies when data must move between backends. GGML exposes backend copy helpers such as `ggml_backend_tensor_copy()` and `ggml_backend_tensor_copy_async()` for this purpose.

5. **Execution and synchronization**
   Execute each split on its assigned backend, then synchronize through backend or scheduler synchronization primitives, including backend events when supported.

One important nuance is that it is better to describe the scheduler as a **backend-assignment and graph-splitting system** rather than as a fixed “5-pass algorithm.” The current implementation uses multiple internal assignment passes, but the stable architectural idea is: assign nodes, split the graph, allocate buffers, manage copies, and execute across backends.

## Build System

- `CMakeLists.txt`
- `src/CMakeLists.txt`
- `src/ggml-cpu/CMakeLists.txt`

GGML uses CMake as its primary build system. The top-level `CMakeLists.txt` defines the global build options, `src/CMakeLists.txt` builds the core libraries and backend targets, and `src/ggml-cpu/CMakeLists.txt` contains the architecture-specific logic for the CPU backend.

### Configuration Options

The top-level build file exposes a large set of options that control which backends are built, how they are packaged, and which CPU instruction sets are enabled.

Common option groups include:

- **Backend selection**: options such as `GGML_CUDA`, `GGML_VULKAN`, `GGML_METAL`, `GGML_SYCL`, `GGML_RPC`, `GGML_OPENCL`, `GGML_WEBGPU`, `GGML_CANN`, and others enable specific backends.
- **Dynamic backend loading**: `GGML_BACKEND_DL` enables building backends as dynamic libraries, and `GGML_BACKEND_DIR` specifies the directory used to load them at runtime.
- **General build optimization**: `GGML_NATIVE`, `GGML_LTO`, and `GGML_CCACHE` control native tuning, link-time optimization, and ccache usage.
- **CPU ISA features**: options such as `GGML_AVX`, `GGML_AVX2`, `GGML_AVX512`, `GGML_F16C`, `GGML_FMA`, `GGML_AMX_TILE`, `GGML_AMX_INT8`, and `GGML_AMX_BF16` control x86 CPU feature flags.
- **CPU multi-variant builds**: `GGML_CPU_ALL_VARIANTS` enables building multiple CPU backend variants and explicitly requires `GGML_BACKEND_DL`.

### CPU Backend Variants

The CPU backend build logic is implemented in `src/ggml-cpu/CMakeLists.txt` through `ggml_add_cpu_backend_variant_impl(tag_name)`. This function creates either a default `ggml-cpu` target or a tagged target named `ggml-cpu-<tag>`, then applies architecture-specific source files, compile options, and compile definitions.

When `GGML_CPU_ALL_VARIANTS` is enabled, GGML can build multiple CPU backend variants instead of only one. In the current repo, this is implemented as a feature- and architecture-driven mechanism rather than a small fixed list of hard-coded variants. For example:

- on **x86**, the selected backend variant is shaped by options such as `GGML_AVX2`, `GGML_AVX512`, `GGML_F16C`, `GGML_FMA`, and AMX-related flags;
- on **ARM**, the build logic raises the target ISA as needed, for example to `armv8.2-a`, `armv8.6-a`, or `armv9.2-a`, depending on enabled features such as dot product, FP16 vector arithmetic, SVE, i8mm, SVE2, and SME;
- on **s390x**, the `GGML_CPU_ALL_VARIANTS` path can cross-compile a range of machine targets from z15 through z17.

So it is more accurate to describe `GGML_CPU_ALL_VARIANTS` as a **multi-target CPU backend build mode** than as a fixed list like `ggml-cpu-haswell` or `ggml-cpu-skylakex`. The current CMake logic builds tagged CPU backend targets and assigns the appropriate ISA flags for the target architecture.

### Backend Library Registration

Backend targets are created in `src/CMakeLists.txt` through `ggml_add_backend_library(backend)`. That helper switches behavior depending on whether dynamic backend loading is enabled.

- When **`GGML_BACKEND_DL` is ON**, the backend is built as a `MODULE` library, marked with `GGML_BACKEND_DL`, added as a dependency of `ggml`, and installed either to `GGML_BACKEND_DIR` or `CMAKE_INSTALL_BINDIR`.
- When **`GGML_BACKEND_DL` is OFF**, the backend is added as a regular library target, linked into the main `ggml` target, and installed as part of the normal build.

Each backend library links against `ggml-base`, includes the parent source directory, and, when shared libraries are being built, receives the `GGML_BACKEND_BUILD` and `GGML_BACKEND_SHARED` compile definitions. Backend libraries also inherit version metadata through `VERSION ${GGML_VERSION}` and `SOVERSION ${GGML_VERSION_MAJOR}` where supported.

At runtime, `src/ggml-backend-reg.cpp` registers compiled-in backends such as CUDA, Metal, SYCL, Vulkan, WebGPU, OpenCL, CANN, BLAS, RPC, OpenVINO, and CPU through the backend registry constructor. This is the bridge between the CMake build configuration and the runtime backend discovery system.

## Memory Management

- `include/ggml.h`
- `include/ggml-alloc.h`
- `src/ggml-alloc.c`

GGML separates **graph construction** from **physical tensor storage**. A `ggml_context` owns tensor and graph metadata, while the allocator layer decides when and where actual tensor data is placed. The core context setup is defined by `struct ggml_init_params`, which includes `mem_size`, `mem_buffer`, and `no_alloc`. GGML also exposes `ggml_get_no_alloc()` and `ggml_set_no_alloc()` to control this behavior explicitly.

### No-Alloc Context

A common GGML pattern is to create a `ggml_context` with `no_alloc = true` so the program can build tensor objects and the computation graph without immediately allocating backing storage. This lets GGML first determine tensor shapes, graph dependencies, and execution structure, and then allocate memory later through allocator or backend-buffer APIs. That separation is fundamental to how GGML supports graph planning and multi-backend execution.

### Context and Tensor Allocation Helpers

The public allocation API is declared in `include/ggml-alloc.h`. It includes:

- `ggml_tallocr`: a simple tensor allocator over an existing backend buffer
- `ggml_gallocr_t`: the graph allocator
- `ggml_backend_alloc_ctx_tensors_from_buft()`: allocate all tensors in a context from a buffer type
- `ggml_backend_alloc_ctx_tensors()`: allocate all tensors in a context using a backend

`ggml_tallocr` is the simple, linear allocator: it wraps a backend buffer, tracks alignment and the current offset, and allocates tensor storage sequentially inside that buffer.

### Graph Allocator

The main graph allocator is `ggml_gallocr_t`. The public API includes `ggml_gallocr_new()`, `ggml_gallocr_new_n()`, `ggml_gallocr_reserve()`, `ggml_gallocr_reserve_n()`, `ggml_gallocr_alloc_graph()`, and `ggml_gallocr_get_buffer_size()`. The header also documents an important usage pattern: reserve against a worst-case graph first, then allocate concrete graphs later to reduce reallocations. It also documents two special tensor flags used by the allocator: `ggml_set_input()` marks tensors that are allocated at the beginning of the graph in non-overlapping addresses, and `ggml_set_output()` marks tensors that are never freed or overwritten.

Internally, `ggml_gallocr` works by traversing the graph, allocating leaf tensors and explicit inputs early, tracking each tensor’s remaining children and view relationships, and freeing storage once a tensor is no longer needed. In the current implementation, this is driven by counters such as `n_children` and `n_views`, and tensors are released through `ggml_gallocr_free_node()` when those counts drop to zero. So the high-level idea is **lifetime-based reuse**, but it is more accurate to describe the implementation as dependency-count and view-aware reuse rather than a separate explicit “forward pass / backward pass” liveness algorithm.

### Backend Buffers

Actual storage is represented through backend buffer abstractions such as `ggml_backend_buffer_type_t` and `ggml_backend_buffer_t`. The graph allocator can reserve one or more backend buffers, and when storage is materialized it uses backend buffer allocation helpers to create physical buffers and place tensors into them. The helper `ggml_backend_alloc_ctx_tensors_from_buft()` is especially useful when you want a whole context allocated from a specific buffer type.

### Dynamic Allocator

Inside `src/ggml-alloc.c`, GGML also implements an internal dynamic allocator, `ggml_dyn_tallocr`, which is used by the graph allocator’s virtual-buffer machinery. This is **not** the main public API, but it is an important implementation detail. The allocator maintains free-block lists per chunk, with `MAX_FREE_BLOCKS = 256`, and supports up to `GGML_VBUFFER_MAX_CHUNKS = 16`. Allocation uses a best-fitting free block search before falling back to the last block, which may grow the chunk size.

When physical buffers are finally created, GGML builds a virtual buffer object that may span multiple backend buffer chunks. Each chunk is allocated with `ggml_backend_buft_alloc_buffer()`, assigned a usage such as `GGML_BACKEND_BUFFER_USAGE_COMPUTE`, and later used to place tensors at computed chunk-relative offsets.

### Summary

In practice, GGML memory management is built around four layers:

1. **Context metadata** in `ggml_context`
2. **Simple linear allocation** through `ggml_tallocr`
3. **Graph-aware lifetime reuse** through `ggml_gallocr`
4. **Backend-owned physical storage** through backend buffer types and buffers

This design lets GGML build graphs cheaply, delay allocation until execution planning is known, and aggressively reuse memory across graph nodes to reduce peak memory usage.

## Synchronization with `llama.cpp` and `whisper.cpp`

- `scripts/sync-llama-am.sh`
- `scripts/sync-llama.sh`
- `scripts/sync-llama.last`
- `scripts/sync-whisper-am.sh`
- `scripts/sync-whisper.sh`
- `scripts/sync-whisper.last`

The standalone GGML repository is developed in parallel with downstream projects, especially `llama.cpp` and `whisper.cpp`. The GGML README explicitly notes that some development currently happens in those repos, and the `scripts/` directory contains dedicated sync scripts and last-synced commit markers for both downstream sources. Recent GGML history also includes explicit `sync : llama.cpp` and `sync : whisper.cpp` commits, showing that this workflow is actively used.

### Sync Model

GGML does **not** use a fully automatic continuous synchronization service. According to the maintainer, changes are synced **manually** across `ggml`, `llama.cpp`, and `whisper.cpp` using the Bash scripts in the `scripts` folder, typically every once in a while rather than on every commit.

### How the Sync Scripts Work

The sync scripts implement a patch-based import workflow that preserves commit authorship and history while adapting downstream paths back into the standalone GGML repo. The process works roughly as follows.

1. **Commit tracking**
   The files `scripts/sync-llama.last` and `scripts/sync-whisper.last` store the last downstream commit that has already been synchronized. Each script reads the corresponding `.last` file before scanning for new commits.

2. **Change detection**
   The scripts run `git log` from the recorded commit up to `HEAD` in the downstream repo to identify candidate commits for import.

3. **Commit filtering**
   The scripts exclude commits that already look back-ported. For example, the `whisper` sync script filters out commits tagged like `ggml/<num>` and `llama/<num>`, while the `llama` sync script filters out `ggml/<num>` and `whisper/<num>`. This avoids re-importing changes that were already synchronized through another path.

4. **Patch generation**
   For each selected commit, the scripts use `git format-patch --stdout` to generate mail-style patches for the GGML-related paths only. This keeps the imported patch focused on the shared GGML code instead of unrelated downstream project files.

5. **Path and metadata rewriting**
   Since GGML code lives under `ggml/` inside `llama.cpp` and `whisper.cpp` but at the repository root inside standalone `ggml`, the scripts rewrite patch paths with `sed`, for example mapping `ggml/src/...` to `src/...` and `ggml/include/...` to `include/...`. They also rewrite PR references in patch subjects from forms like `(#1234)` to namespaced markers such as `(llama/1234)` or `(whisper/1234)`.

6. **History-preserving application**
   The transformed patch is applied with `git am`, which preserves commit metadata such as author and commit message structure more faithfully than a manual copy or squash import.

7. **Advance the sync point**
   After a successful import, the script writes the latest downstream commit hash back into the corresponding `.last` file so the next sync starts from the new boundary.

### Architectural Role

This workflow lets `llama.cpp` and `whisper.cpp` evolve quickly in their application context while still feeding shared GGML changes back into the standalone repository. At the same time, the maintainer has noted that GGML is not yet versioned strongly enough to guarantee cross-project binary compatibility, so this sync mechanism should be understood as a practical source-sync workflow rather than a strict release or ABI compatibility system.

## Testing Infrastructure

GGML includes a broad test suite under `tests/`, covering backend behavior, tensor operators, quantization, math kernels, optimizers, and other focused functionality. The current test directory includes files such as `test-backend-ops.cpp`, `test-quantize-fns.cpp`, `test-quantize-perf.cpp`, `test-mul-mat.cpp`, `test-opt.cpp`, and many smaller operator-specific tests.

Key files for the core validation workflow include:

- `tests/test-backend-ops.cpp`
- `tests/test-quantize-fns.cpp`
- `tests/test-quantize-perf.cpp`

### Backend Operation Testing

`tests/test-backend-ops.cpp` is the main cross-backend validation driver. The file header states that it checks forward-pass consistency across backends, can validate backward gradients against finite-difference estimates in gradient mode, and can also run in performance mode. The program loads all available backends with `ggml_backend_load_all()`, enumerates backend devices, initializes each backend, and runs the selected test mode against them.

### Quantization Testing

`tests/test-quantize-fns.cpp` is the correctness test for quantization-related routines, while `tests/test-quantize-perf.cpp` is the performance-oriented benchmark for quantization paths. Together, these cover the functional validation and speed measurement of GGML’s quantization code paths.

### Test Modes

`test-backend-ops.cpp` supports several modes through its command-line interface. Internally, these correspond to `MODE_TEST`, `MODE_PERF`, `MODE_GRAD`, and `MODE_SUPPORT`, selected by the commands `test`, `perf`, `grad`, and `support`. It also supports extra options such as `--list-ops`, `--show-coverage`, `--output`, and `--test-file`.

The modes are used as follows:

- **Test mode**: checks correctness by comparing results across backends.
- **Perf mode**: measures execution performance for supported operations.
- **Grad mode**: compares backpropagated gradients with gradients estimated by finite differences.
- **Support mode**: reports backend support coverage for operations.

Overall, GGML’s testing infrastructure combines correctness checks, backend consistency validation, gradient verification, and targeted microbenchmarks, giving the project both regression coverage and performance visibility as backends and kernels evolve.)

## Key Data Structures

### `ggml_tensor`

**Defined in:** `include/ggml.h`

`ggml_tensor` is the core data structure in GGML. It represents both raw tensors and intermediate nodes in a computation graph. The header documentation describes it as storing the tensor’s size, data type, memory buffer, and pointers to its source tensors. It also uses `ne[]` for element counts and `nb[]` for byte strides, which allows GGML to represent non-contiguous layouts such as views, transposes, and permutations. Tensor data is accessed through the `data` pointer.

In practice, this makes `ggml_tensor` more than just a storage object: it is also the unit of graph construction. When an operation such as `ggml_add()` or `ggml_mul_mat()` is created, the result is another `ggml_tensor` whose source links encode the dependency structure of the graph.

### `ggml_backend`

**Defined in:** `include/ggml-backend.h`

`ggml_backend` is the abstract execution handle for a backend. In the public API it appears as an opaque type:

- `typedef struct ggml_backend * ggml_backend_t;`

This means user code interacts with it through API functions rather than by accessing struct fields directly. The backend interface includes functions for naming and freeing a backend, allocating buffers, setting and getting tensor data, synchronizing execution, planning graph execution, and computing graphs synchronously or asynchronously.

Conceptually, `ggml_backend_t` is the runtime object that represents “where and how computation runs,” such as a CPU backend, CUDA backend, Vulkan backend, or another device-specific execution engine.

### `ggml_cgraph`

**Defined in:** `include/ggml.h`

`ggml_cgraph` is the computation-graph object used to hold a graph of tensor operations before execution. The public examples in `ggml.h` show the standard workflow:

1. create a graph with `ggml_new_graph(ctx)`
2. add nodes with `ggml_build_forward_expand(gf, output_tensor)`
3. execute the graph later through a compute API

Once built, a `ggml_cgraph` can be executed either through the classic graph compute path or through the backend system. The backend API exposes functions such as `ggml_backend_graph_plan_create()`, `ggml_backend_graph_compute()`, `ggml_backend_graph_compute_async()`, and scheduler-based graph execution APIs that all take a `struct ggml_cgraph *` as input.

So, at a high level:

- `ggml_tensor` is the fundamental tensor and graph-node object
- `ggml_backend` is the execution backend handle
- `ggml_cgraph` is the graph container that organizes tensor operations for execution
