# Memory Management and Allocation

This document explains GGML’s memory-management system, which separates **tensor and graph definition** from **physical data allocation**. The design centers on a `ggml_context` for metadata, a graph allocator for lifetime-based reuse, and backend buffer types for the final physical storage. The relevant interfaces are in `include/ggml.h`, `include/ggml-alloc.h`, and `include/ggml-backend.h`, with the main allocator implementation in `src/ggml-alloc.c`.

## Architecture Overview

GGML memory management is best understood as four cooperating layers:

1. **Tensor definition layer**
   A `ggml_context` owns tensor and graph metadata. It is initialized with `struct ggml_init_params`, which includes `mem_size` and `mem_buffer`, and GGML exposes `ggml_get_no_alloc()` / `ggml_set_no_alloc()` to control whether tensor data is allocated during graph construction. This layer defines tensor objects and graph structure, not the final execution buffers.

2. **Graph memory-planning layer**
   `ggml_gallocr_t` is the graph allocator. Its public API includes `ggml_gallocr_new()`, `ggml_gallocr_reserve()`, `ggml_gallocr_reserve_n()`, `ggml_gallocr_alloc_graph()`, and buffer-size queries. The intended workflow is: optionally reserve against a worst-case graph, then allocate the actual graph before execution.

3. **Physical allocation layer**
   Backend buffer types and backend buffers represent the actual storage used at execution time. GGML exposes helpers such as `ggml_backend_buft_alloc_buffer()` and context-allocation helpers such as `ggml_backend_alloc_ctx_tensors_from_buft()`. This is the layer where planned tensor storage becomes real memory.

4. **Allocator implementation layer**
   Under the hood, `src/ggml-alloc.c` contains both a simple tensor allocator (`ggml_tallocr`) and an internal dynamic allocator (`ggml_dyn_tallocr`). The dynamic allocator manages free-block lists inside allocator chunks, while the graph allocator uses tensor dependency and view tracking to decide when storage can be reused.

## Context and No-Allocation Mode

- `include/ggml.h`

`ggml_context` is the entry point for defining tensors and computation graphs. It is created with `ggml_init(struct ggml_init_params params)`, and the initialization parameters are:

| Field        | Type     | Purpose                                                  |
| ------------ | -------- | -------------------------------------------------------- |
| `mem_size`   | `size_t` | Size of the context memory pool in bytes                 |
| `mem_buffer` | `void *` | Metadata buffer; if `NULL`, GGML allocates it internally |
| `no_alloc`   | `bool`   | Do not allocate memory for tensor data                   |

GGML also exposes `ggml_get_no_alloc(ctx)` and `ggml_set_no_alloc(ctx, no_alloc)` to query or change this behavior.

A tensor defined in this context is represented by `struct ggml_tensor`. The public tensor structure includes:

- `type`
- `buffer`
- `ne[GGML_MAX_DIMS]` for shape
- `nb[GGML_MAX_DIMS]` for byte strides
- `op`
- `src[]`
- `view_src` and `view_offs`
- `data`

This separation is important: the context owns the tensor metadata and graph structure, while the final placement of tensor payloads can be deferred to the allocator and backend-buffer layers.

## Graph Allocator (`ggml_gallocr`)

- `include/ggml-alloc.h`
- `src/ggml-alloc.c`

`ggml_gallocr` is GGML’s graph allocator. The public API exposes:

- `ggml_gallocr_new(...)`
- `ggml_gallocr_new_n(...)`
- `ggml_gallocr_reserve(...)`
- `ggml_gallocr_reserve_n(...)`
- `ggml_gallocr_alloc_graph(...)`

The header also documents the intended usage pattern: optionally reserve buffers from a worst-case graph first, then allocate the actual graph before execution. It further notes two allocator-relevant tensor flags: `ggml_set_input()` places inputs at the beginning of the graph in non-overlapping addresses, and `ggml_set_output()` keeps outputs from being freed or overwritten.

### Allocation Strategy

The allocator does not work from a standalone “first-use / last-use” table in the public API. In `src/ggml-alloc.c`, the graph-allocation logic is organized around graph traversal, hash-based bookkeeping, child-count tracking, and view tracking:

1. reset the allocator hash tables
2. allocate graph leaf tensors first
3. count children and views for graph nodes
4. allocate nodes
5. decrement parent child-counts as nodes are processed
6. free parent storage when its child count and view count both reach zero

That last step is the key reuse mechanism: once a tensor is no longer needed by any remaining node and is not held alive by any view, its storage can be returned to the allocator and reused.

### Internal Dynamic Allocator

Inside `src/ggml-alloc.c`, `ggml_gallocr` relies on an internal dynamic allocator, `ggml_dyn_tallocr`. Its storage model is chunk-based:

- each chunk stores a free-block array
- each chunk tracks `n_free_blocks`
- the allocator can span up to `GGML_VBUFFER_MAX_CHUNKS`
- each chunk stores up to `MAX_FREE_BLOCKS` free blocks

The implementation uses helpers such as `ggml_dyn_tallocr_new(...)`, `ggml_dyn_tallocr_alloc(...)`, and `ggml_dyn_tallocr_free_bytes(...)` to manage reusable address ranges during graph allocation.

### Memory Reuse Model

A useful mental model is:

- **context** defines tensor objects and graph structure
- **graph allocator** decides when tensor storage can be reused
- **dynamic allocator** manages reusable address ranges
- **backend buffers** provide the final physical storage

This is why GGML can build a full graph first, then place tensors into a compact memory layout that reuses storage across non-overlapping lifetimes.

### Dynamic Allocator (`ggml_dyn_tallocr`)

- `src/ggml-alloc.c`

`ggml_dyn_tallocr` is the internal allocator used by the graph allocator to manage a virtual address space before physical backend buffers are created. It tracks aligned allocations as `(chunk, offset)` addresses, where each chunk represents one future backend-buffer segment. The allocator state includes `alignment`, `max_chunk_size`, an array of chunk pointers, and `n_chunks`. Each chunk stores a sorted list of free blocks plus its `max_size`, which records the highest offset reached in that chunk. GGML limits this structure to `GGML_VBUFFER_MAX_CHUNKS = 16`, and each chunk can store up to `MAX_FREE_BLOCKS = 256` free blocks.

#### Chunk and Free-Block Management

Each chunk is represented by a `tallocr_chunk`:

- `free_blocks[MAX_FREE_BLOCKS]`
- `n_free_blocks`
- `max_size`

Free blocks are stored in ascending address order. `ggml_dyn_tallocr_insert_block(...)` inserts a block at the correct sorted position, and `ggml_dyn_tallocr_remove_block(...)` removes one by shifting later entries left. Keeping the free-block list sorted makes adjacent-block merging simpler during free operations.

When a new chunk is created with `ggml_dyn_tallocr_new_chunk(...)`, GGML initializes it with one free block at offset `0`. Its initial size is `MAX(min_size, alloc->max_chunk_size)`. There are two explicit exceptions documented in the source: a chunk may exceed `max_chunk_size` if a single tensor is larger than the maximum, or if GGML is running out of chunk slots. On the last available chunk, GGML sets the initial free-block size to `SIZE_MAX/2` so allocation can continue if the backend is able to provide the larger memory region.

#### Best-Fit Allocation Algorithm

`ggml_dyn_tallocr_alloc(...)` first aligns the requested size, then searches for a placement in two stages:

1. **Search non-terminal free blocks across all chunks**
   GGML scans every chunk and looks at every free block except the final block in each chunk. Among blocks large enough to satisfy the request, it chooses the smallest one. This is the main best-fit step and is intended to reduce fragmentation.

2. **Fallback to the last free block of each chunk**
   If no regular free block fits, GGML considers the last free block in each chunk. For these blocks it computes a `reuse_factor = chunk->max_size - block->offset - size`. The comments in the source explain the interpretation:

   - negative: extra memory must be grown
   - zero: exact fit
   - positive: leftover unused space remains

   GGML uses this value to choose the most suitable tail block, preferring reuse or the closest fit depending on the sign.

3. **Create a new chunk if needed**
   If no existing chunk can satisfy the allocation, GGML creates a new chunk and allocates from it. After allocation, the chunk’s `max_size` is updated to reflect the highest address used so far.

#### Free-Block Merging

`ggml_dyn_tallocr_free_bytes(...)` also aligns the freed size, then attempts to merge the freed range with adjacent free blocks in the same chunk. The implementation follows three cases:

1. **Merge with the previous block**
   If an existing free block ends exactly where the freed range begins, GGML extends that block forward. It then checks whether the enlarged block now touches the next free block and merges that as well.

2. **Merge with the next block**
   If the freed range ends exactly where an existing free block begins, GGML moves that block’s offset backward and enlarges it. It then checks whether this new block now touches the previous free block and merges backward if possible.

3. **Insert a new free block**
   If neither adjacent-merge case applies, GGML inserts a new free block into the sorted free-block list.

#### Relationship to Physical Buffers

The dynamic allocator only plans addresses. Later, `ggml_vbuffer_alloc(...)` materializes real backend buffers using each chunk’s `max_size`, allocating one backend buffer per chunk and then placing tensors at the computed `(chunk, offset)` addresses. This is how the virtual allocation plan becomes actual backend-owned storage.

## Backend Buffer System

- `include/ggml-backend.h`
- `include/ggml-alloc.h`
- `src/ggml-backend.cpp`
- `src/ggml-alloc.c`

The backend-buffer layer is the bridge between GGML’s allocator logic and the actual memory owned by a device or runtime. The public API separates **buffer types** from **buffers**:

- `ggml_backend_buffer_type_t` describes an allocation mechanism
- `ggml_backend_buffer_t` is an allocated buffer instance
- `ggml_backend_dev_t` exposes the device associated with a buffer type

The main public helpers are `ggml_backend_buft_alloc_buffer(...)`, `ggml_backend_buft_get_alignment(...)`, `ggml_backend_buft_get_max_size(...)`, `ggml_backend_buft_get_alloc_size(...)`, `ggml_backend_buffer_get_base(...)`, `ggml_backend_buffer_get_size(...)`, `ggml_backend_buffer_get_alignment(...)`, `ggml_backend_buffer_get_alloc_size(...)`, and `ggml_backend_buffer_set_usage(...)`.

### Buffer Type Hierarchy

A useful way to read the backend-buffer system is:

1. **Device**
   `ggml_backend_dev_t` represents a device and exposes properties such as memory size, device type, and capabilities. Devices can also expose a default buffer type and a host buffer type through `ggml_backend_dev_buffer_type(...)` and `ggml_backend_dev_host_buffer_type(...)`.

2. **Buffer type**
   `ggml_backend_buffer_type_t` describes how allocation works for that device or memory class. Allocation happens through `ggml_backend_buft_alloc_buffer(...)`, and the buffer type also reports alignment, maximum size, whether it is host-accessible, and the allocation size required for a given tensor layout.

3. **Buffer instance**
   `ggml_backend_buffer_t` is the actual allocated storage object. It exposes size, alignment, usage, host accessibility, optional base-address access, tensor initialization, tensor read/write helpers, clear/reset helpers, and buffer destruction.

### Buffer Interface Operations

Internally, `src/ggml-backend.cpp` constructs buffers from an interface table (`ggml_backend_buffer_i`) plus a backend-specific context pointer. The public wrapper functions call into that interface when the operation is implemented. The interface supports these operations: `free_buffer`, `get_base`, `init_tensor`, `memset_tensor`, `set_tensor`, `get_tensor`, `cpy_tensor`, `clear`, and `reset`.

| Operation         | Purpose                                           | Notes                                                                  |
| ----------------- | ------------------------------------------------- | ---------------------------------------------------------------------- |
| `get_base()`      | Return a base address                             | Optional for buffers that do not have one contiguous host-visible base |
| `init_tensor()`   | Initialize tensor state inside the buffer         | Optional                                                               |
| `set_tensor()`    | Write tensor data into the buffer                 | Used through `ggml_backend_tensor_set(...)`                            |
| `get_tensor()`    | Read tensor data from the buffer                  | Used through `ggml_backend_tensor_get(...)`                            |
| `memset_tensor()` | Fill part of a tensor with one byte value         | Optional                                                               |
| `cpy_tensor()`    | Copy one tensor into another buffer-native layout | Optional                                                               |
| `clear()`         | Clear the whole buffer                            | Used through `ggml_backend_buffer_clear(...)`                          |
| `reset()`         | Reset backend buffer state                        | Optional                                                               |
| `free_buffer()`   | Release backend-owned resources                   | Optional at the interface level, but used when present                 |

### Multi-Buffer Support

GGML also implements a logical **multi-buffer** in `src/ggml-backend.cpp`. The helper `ggml_backend_multi_buffer_alloc_buffer(...)` creates one logical buffer from an array of existing buffers. This is used when one allocation plan spans multiple physical buffer chunks instead of one contiguous allocation.

The multi-buffer wrapper is intentionally limited:

- `get_base` is `NULL`
- `init_tensor`, `memset_tensor`, `set_tensor`, `get_tensor`, and `cpy_tensor` are `NULL`
- `clear` is implemented by forwarding to each child buffer
- usage propagation is implemented through `ggml_backend_multi_buffer_set_usage(...)`

So a multi-buffer is best understood as a coordination object over several backend buffers, not as one ordinary flat host-visible buffer.

### Tensor Allocator (`ggml_tallocr`)

For simple allocation from an already-created backend buffer, GGML provides `ggml_tallocr` in `include/ggml-alloc.h`. Its state is:

- `buffer`
- `base`
- `alignment`
- `offset`

and the public API is:

- `ggml_tallocr_new(buffer)`
- `ggml_tallocr_alloc(&talloc, tensor)`

`ggml_tallocr_alloc(...)` computes the tensor’s required size with `ggml_backend_buffer_get_alloc_size(...)`, pads it to the allocator alignment, checks capacity against `ggml_backend_buffer_get_size(...)`, assigns storage, and advances the linear `offset`. This makes `ggml_tallocr` a simple sequential allocator: it is straightforward and efficient, but it does not perform lifetime-based reuse the way `ggml_gallocr` does.

### Memory Layout and Alignment

Alignment is backend-defined, not hard-coded by the generic GGML API. Both buffer types and allocated buffers expose alignment queries through:

- `ggml_backend_buft_get_alignment(...)`
- `ggml_backend_buffer_get_alignment(...)`

Similarly, tensor allocation size is backend-aware:

- `ggml_backend_buft_get_alloc_size(buft, tensor)`
- `ggml_backend_buffer_get_alloc_size(buffer, tensor)`

This matters because tensor layout in GGML is not always contiguous. `struct ggml_tensor` stores both `ne[]` and `nb[]`, and element addressing follows the stride-based rule

```c
offset = i0*nb[0] + i1*nb[1] + i2*nb[2] + i3*nb[3]
```

so buffer size and placement must respect layout, padding, and backend alignment rather than assuming a single flat dense array.

### Allocation Workflow

A typical allocation pipeline in GGML is:

1. **Define tensors and build the graph** inside a `ggml_context`.
2. **Create a graph allocator** with `ggml_gallocr_new(...)` or `ggml_gallocr_new_n(...)`.
3. **Reserve memory** with `ggml_gallocr_reserve(...)` or `ggml_gallocr_reserve_n(...)`, optionally using a worst-case graph first.
4. **Allocate the graph** with `ggml_gallocr_alloc_graph(...)`. For compute buffers, the allocator materializes backend virtual buffers with usage `GGML_BACKEND_BUFFER_USAGE_COMPUTE`.
5. **Read or write tensor contents** through `ggml_backend_tensor_set(...)`, `ggml_backend_tensor_get(...)`, `ggml_backend_tensor_memset(...)`, or tensor-copy helpers.

GGML also provides utility helpers to allocate all tensors in a `ggml_context` directly from one buffer type or backend:

- `ggml_backend_alloc_ctx_tensors_from_buft_size(...)`
- `ggml_backend_alloc_ctx_tensors_from_buft(...)`
- `ggml_backend_alloc_ctx_tensors(...)`

### Key API Functions

#### Graph allocator API

| Function                                                                  | Purpose                                            |
| ------------------------------------------------------------------------- | -------------------------------------------------- |
| `ggml_gallocr_new(buft)`                                                  | Create a graph allocator for one buffer type       |
| `ggml_gallocr_new_n(bufts, n)`                                            | Create a graph allocator for multiple buffer types |
| `ggml_gallocr_reserve(galloc, graph)`                                     | Reserve buffer space from a graph                  |
| `ggml_gallocr_reserve_n(galloc, graph, node_buffer_ids, leaf_buffer_ids)` | Reserve with explicit multi-buffer placement       |
| `ggml_gallocr_alloc_graph(galloc, graph)`                                 | Allocate graph storage                             |
| `ggml_gallocr_get_buffer_size(galloc, buffer_id)`                         | Query reserved/allocated buffer size               |

#### Backend buffer API

| Function                                             | Purpose                                                         |
| ---------------------------------------------------- | --------------------------------------------------------------- |
| `ggml_backend_buft_alloc_buffer(buft, size)`         | Allocate a backend buffer from a buffer type                    |
| `ggml_backend_buffer_get_base(buffer)`               | Get the base pointer when the buffer exposes one                |
| `ggml_backend_tensor_set(tensor, data, off, sz)`     | Write into tensor storage                                       |
| `ggml_backend_tensor_get(tensor, data, off, sz)`     | Read from tensor storage                                        |
| `ggml_backend_tensor_memset(tensor, value, off, sz)` | Fill part of a tensor                                           |
| `ggml_backend_buffer_clear(buffer, value)`           | Clear the whole buffer                                          |
| `ggml_backend_tensor_copy(src, dst)`                 | Copy between tensors, with host-buffer and backend-copy support |

### View Operations

View-style tensor operations do not imply a new physical allocation. GGML represents these through:

- tensor fields `view_src` and `view_offs`
- view/layout ops such as `GGML_OP_VIEW`, `GGML_OP_RESHAPE`, `GGML_OP_PERMUTE`, and `GGML_OP_TRANSPOSE`
- APIs such as `ggml_view_tensor(...)`

This is why the allocator tracks both normal dependency counts and view relationships: a tensor’s memory cannot be reused while a live view still references it.

### Buffer Usage Hints

GGML defines three usage hints for backend buffers:

- `GGML_BACKEND_BUFFER_USAGE_ANY`
- `GGML_BACKEND_BUFFER_USAGE_WEIGHTS`
- `GGML_BACKEND_BUFFER_USAGE_COMPUTE`

The public API exposes `ggml_backend_buffer_set_usage(...)` and `ggml_backend_buffer_get_usage(...)`, and the allocator uses `GGML_BACKEND_BUFFER_USAGE_COMPUTE` when materializing graph compute buffers. Multi-buffers propagate usage to all child buffers.

### Buffer from Host Pointer

Some devices advertise support for creating buffers from an existing host pointer. This is reflected in `struct ggml_backend_dev_caps` through the `buffer_from_host_ptr` capability and exposed through:

- `ggml_backend_dev_buffer_from_host_ptr(device, ptr, size, max_tensor_size)`

GGML also exposes host-buffer queries and host-buffer-type discovery, so this path is useful for backends that can wrap external host memory instead of always allocating a fresh backend-owned block.
