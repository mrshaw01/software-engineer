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

## Operation System

### Operation Types

GGML defines its operator set primarily through `enum ggml_op`, and it also has separate enums for unary activations (`enum ggml_unary_op`) and GLU variants (`enum ggml_glu_op`). The current operator list spans basic arithmetic, reductions, normalization, matrix operations, layout transforms, indexing, masking, convolution, attention, sequence-model-specific kernels, loss functions, and optimizer steps.

A practical way to group the current GGML operators is:

- **Element-wise and arithmetic:** `GGML_OP_ADD`, `ADD_ID`, `ADD1`, `ACC`, `SUB`, `MUL`, `DIV`, `SQR`, `SQRT`, `LOG`, `SIN`, `COS`, `CLAMP`, plus unary ops such as `ABS`, `NEG`, `TANH`, `RELU`, `SIGMOID`, `GELU`, `SILU`, `EXP`, `EXPM1`, and `SOFTPLUS`.
- **Reduction and selection:** `GGML_OP_SUM`, `SUM_ROWS`, `CUMSUM`, `MEAN`, `ARGMAX`, `COUNT_EQUAL`, `ARGSORT`, and `TOP_K`.
- **Normalization:** `GGML_OP_NORM`, `RMS_NORM`, `RMS_NORM_BACK`, `GROUP_NORM`, and `L2_NORM`.
- **Linear algebra:** `GGML_OP_MUL_MAT`, `MUL_MAT_ID`, `OUT_PROD`, `SCALE`, and `SOLVE_TRI`.
- **Transformation and layout:** `GGML_OP_CONT`, `RESHAPE`, `VIEW`, `PERMUTE`, `TRANSPOSE`, `CONCAT`, `REPEAT`, `REPEAT_BACK`, `SET`, and `CPY`.
- **Indexing, masking, and transformer-style ops:** `GGML_OP_GET_ROWS`, `GET_ROWS_BACK`, `SET_ROWS`, `DIAG`, `DIAG_MASK_INF`, `DIAG_MASK_ZERO`, `SOFT_MAX`, `ROPE`, `ROPE_BACK`, `FLASH_ATTN_EXT`, `FLASH_ATTN_BACK`, `GET_REL_POS`, and `ADD_REL_POS`.
- **Convolution, pooling, and image-style ops:** `GGML_OP_CONV_TRANSPOSE_1D`, `IM2COL`, `IM2COL_BACK`, `IM2COL_3D`, `CONV_2D`, `CONV_3D`, `CONV_2D_DW`, `CONV_TRANSPOSE_2D`, `POOL_1D`, `POOL_2D`, `POOL_2D_BACK`, `UPSCALE`, `PAD`, and `PAD_REFLECT_1D`.
- **Specialized model kernels:** `GGML_OP_TIMESTEP_EMBEDDING`, `SSM_CONV`, `SSM_SCAN`, `RWKV_WKV6`, `RWKV_WKV7`, `GATED_LINEAR_ATTN`, `GATED_DELTA_NET`, `GLU`, `CROSS_ENTROPY_LOSS`, `CROSS_ENTROPY_LOSS_BACK`, `OPT_STEP_ADAMW`, and `OPT_STEP_SGD`.

### Operation Function Signatures

- `include/ggml.h`
- `src/ggml.c`

Many GGML operators follow a simple constructor-style API: they take a `ggml_context *`, one or more input tensors, and return a new `struct ggml_tensor *`. For example, `ggml_add`, `ggml_sub`, `ggml_mul`, and `ggml_div` are binary tensor operators, while `ggml_sqr`, `ggml_sqrt`, `ggml_log`, and `ggml_softplus` are unary tensor operators. GGML also provides in-place variants for many of these operations.

```c
struct ggml_tensor * ggml_<op_name>(
    struct ggml_context * ctx,
    struct ggml_tensor * a,      // first input
    struct ggml_tensor * b,      // second input, if needed
    ...                          // op-specific parameters
);
```

This pattern is most obvious for unary and binary operators, but GGML also extends it to more specialized APIs such as indexed addition, matrix multiplication, normalization, masking, reshaping, permutation, convolution, and attention-related operations. The public header examples and the GPT-2 example both show this style of graph construction in practice.

### Operation Creation and Graph Building

- `include/ggml.h`
- `src/ggml.c`

GGML uses deferred execution. The header documentation explicitly states that defining expressions such as `ggml_mul(ctx, x, x)` or `ggml_add(ctx, a, b)` does not perform computation immediately. Instead, each operation produces a new tensor node that becomes part of the computation graph, and the actual computation happens later when the graph is executed.

The representation of that graph is stored directly in `struct ggml_tensor`. Each tensor has an `op` field that records the operator kind and a `src[GGML_MAX_SRC]` array that stores pointers to the source tensors. The GGML header even shows the canonical example:

```c
struct ggml_tensor * c = ggml_add(ctx, a, b);
// assert(c->src[0] == a);
// assert(c->src[1] == b);
```

That is the core mechanism GGML uses to build graphs incrementally as tensor expressions are created.

In the usual flow, you first create tensors and compose operations, then create a `ggml_cgraph`, and finally add the output node with `ggml_build_forward_expand(gf, result)`. Only after that does GGML run the graph through a compute path.

### View Operations

- `include/ggml.h`
- `src/ggml.c`

GGML has explicit support for view-style and layout-style operations. At the tensor level, `enum ggml_op` includes `GGML_OP_VIEW`, `RESHAPE`, `PERMUTE`, and `TRANSPOSE`. At the tensor-structure level, `struct ggml_tensor` contains `view_src` and `view_offs`, in addition to `ne[]` and `nb[]`, which is how GGML represents shared-storage views and custom layouts.

The public API includes `ggml_view_tensor(...)`, and the repo also exposes shaped view helpers such as `ggml_view_1d(...)` and `ggml_view_2d(...)`. In the GPT-2 example, GGML uses `ggml_view_1d`, `ggml_view_2d`, `ggml_reshape_3d`, and `ggml_permute` to slice KV-cache memory and reinterpret layout without rebuilding tensors from scratch.

So the right mental model is: a view operation creates a new tensor object that points back to existing storage through `view_src` and an offset/layout description, while `ne[]` and `nb[]` describe the new logical shape and strides. That is what enables zero-copy or low-copy transformations such as slicing, reshaping, permutation, and transpose-style layout reinterpretation.
