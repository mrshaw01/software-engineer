# Tensor Operations and Computation Graphs

This page documents GGML’s core tensor representation, the tensor operation system used to build models, and the computation graph abstraction that links those operations together. GGML’s public API and most of the core graph-building logic are centered in `include/ggml.h`, with execution logic and many operator implementations in `src/ggml.c`.

## Overview

GGML uses a deferred computation model: users create tensors, apply operations to produce new tensors, build a computation graph from the final result, and only then execute the graph. The header documentation explicitly shows this pattern with `ggml_new_graph(ctx)`, `ggml_build_forward_expand(gf, f)`, and a later compute call such as `ggml_graph_compute_with_ctx(...)`. The same docs also note that each tensor operation produces a new tensor, and that GGML provides forward and backward computation functions for tensor operators.

This design gives GGML several important properties:

- **Deferred execution**: define operations first, compute later
- **Automatic differentiation support**: the graph structure supports forward and backward evaluation
- **Backend flexibility**: the same graph can later be executed through CPU or backend-based execution paths
- **Memory efficiency**: graph construction can be separated from allocation and execution planning

## Tensor Structure

### Core Tensor Representation

The fundamental data structure is `struct ggml_tensor` in `include/ggml.h`. In the current GGML repo, it stores the tensor type, backend buffer pointer, shape, byte strides, operation metadata, source tensors, view metadata, raw data pointer, name, and backend-specific extra state.

A `ggml_tensor` therefore serves two roles at once:

1. it describes a tensor’s storage and layout
2. it can also represent a node in a computation graph, by recording which operation produced it and which source tensors it depends on

### Key Fields

The most important fields of `ggml_tensor` are:

- `type` — tensor element type, such as `GGML_TYPE_F32` or a quantized format
- `buffer` — backend buffer associated with the tensor
- `ne[GGML_MAX_DIMS]` — number of elements in each dimension
- `nb[GGML_MAX_DIMS]` — stride in bytes for each dimension
- `op` — the operation that produces this tensor when it is a graph node
- `op_params` — operation-specific parameters
- `flags` — tensor flags
- `src[GGML_MAX_SRC]` — source tensors for the operation
- `view_src` and `view_offs` — metadata for tensor views
- `data` — pointer to the tensor’s storage
- `name` — optional tensor name
- `extra` — backend-specific extra data, for example CUDA-related state

One important correction relative to some older GGML descriptions: in the current public `ggml_tensor` definition, there is no public `backend` enum field and no public `grad` field inside the struct. Device placement is represented through the backend buffer layer, and automatic differentiation is exposed through graph and optimization APIs rather than a standalone `grad` member in the public tensor struct.

### Tensor Layout and Strides

- `include/ggml.h`
- `src/ggml.c`

GGML supports up to 4-dimensional tensors, and it supports non-contiguous layouts through the `nb[]` stride array. The header documents the stride rules directly:

- `nb[0] = ggml_type_size(type)`
- `nb[1] = nb[0] * (ne[0] / ggml_blck_size(type)) + padding`
- `nb[i] = nb[i-1] * ne[i-1]` for higher dimensions

That means the byte address of element `[i0, i1, i2, i3]` can be understood as:

```c
data + i0*nb[0] + i1*nb[1] + i2*nb[2] + i3*nb[3]
```

This stride-based representation is what allows GGML to support layout-changing operations and views without always copying data first. The GGML header specifically calls out more complex operations such as `ggml_permute()` as examples of this more general tensor-operator model.
