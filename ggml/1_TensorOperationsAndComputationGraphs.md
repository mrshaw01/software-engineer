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

## Computation Graph Construction

### Graph Structure

- `include/ggml.h`
- `src/ggml.c`
- `src/ggml-opt.cpp`

`ggml_cgraph` is GGML’s computation graph object. The public API exposes graph construction through functions such as `ggml_new_graph(...)`, `ggml_new_graph_custom(...)`, and `ggml_build_forward_expand(...)`. The workflow is: define tensor expressions first, build a graph from the final output tensor, and execute it later.

The graph stores more than a single output pointer. The implementation uses graph fields such as `size`, `n_nodes`, `n_leafs`, `nodes`, `leafs`, `visited_hash_set`, `grads`, and `grad_accs`, which shows that the graph contains ordered execution nodes, leaf tensors, traversal state, and gradient-related bookkeeping.

In GGML, tensor dependencies are represented through each tensor’s `src[]` pointers, while `ggml_cgraph` stores the ordered result of traversing those dependencies so the graph can be executed in dependency order.

### Graph Building Process

- `include/ggml.h`
- `src/ggml.c`

The main entry point for forward graph construction is `ggml_build_forward_expand(gf, output_tensor)`. The documented usage pattern is:

1. create tensors in a `ggml_context`
2. compose tensor operations
3. create a `ggml_cgraph`
4. expand the graph from the final output tensor
5. execute the graph later

At a high level, graph construction works like this:

- start from the output tensor
- follow dependencies through `src[]`
- collect leaf tensors separately from computed nodes
- build an ordered node list so dependencies come before the nodes that consume them
- maintain visited-state and optional gradient mappings inside the graph

This organization ensures that sequential graph execution can process nodes in dependency order.

### Forward and Backward Graphs

GGML supports both forward and backward graph workflows. The public header notes that users can define a function graph once and compute forward or backward graphs multiple times using the same memory buffer. It also documents `ggml_set_param(...)` for marking tensors as optimization parameters.

The implementation also includes gradient-related graph state through `grads` and `grad_accs`, which is used when duplicating graphs and preserving gradient mappings.

### Graph Planning and Execution

- `include/ggml-backend.h`

After construction, a `ggml_cgraph` can be planned and executed through the backend layer. The backend API includes:

- `ggml_backend_graph_plan_create(...)`
- `ggml_backend_graph_plan_free(...)`
- `ggml_backend_graph_plan_compute(...)`
- `ggml_backend_graph_compute(...)`
- `ggml_backend_graph_compute_async(...)`

So the overall model is:

- `ggml_tensor` objects encode dependencies
- `ggml_build_forward_expand(...)` collects them into a `ggml_cgraph`
- the graph stores ordered nodes, leaves, and optional gradient metadata
- backend APIs plan and execute the graph on one or more devices

## Graph Execution Model

### Execution Context

- `include/ggml-cpu.h`
- `include/ggml-backend.h`
- `src/ggml-cpu/ggml-cpu.c`

GGML exposes two graph-execution paths: a CPU execution path in `include/ggml-cpu.h` and a backend execution path in `include/ggml-backend.h`. The CPU path is built around `struct ggml_cplan`, which contains `work_size`, `work_data`, `n_threads`, `threadpool`, an abort callback, and a `use_ref` flag. The backend path exposes graph-plan, direct graph-compute, async graph-compute, and scheduler-based execution APIs.

### CPU Execution APIs

The CPU execution API consists of:

- `ggml_graph_plan(...)`
- `ggml_graph_compute(...)`
- `ggml_graph_compute_with_ctx(...)`

`ggml_graph_plan(...)` returns a `ggml_cplan` describing the temporary workspace required for execution. If `plan.work_size > 0`, the caller allocates `plan.work_data` before calling `ggml_graph_compute(...)`. The helper `ggml_graph_compute_with_ctx(...)` provides the same execution path but allocates the work buffer from the `ggml_context` instead of requiring a separate caller-managed buffer.

### Backend Execution APIs

The backend execution API consists of:

- `ggml_backend_graph_plan_create(...)`
- `ggml_backend_graph_plan_free(...)`
- `ggml_backend_graph_plan_compute(...)`
- `ggml_backend_graph_compute(...)`
- `ggml_backend_graph_compute_async(...)`

For multi-backend execution, GGML also provides the scheduler API:

- `ggml_backend_sched_reserve(...)`
- `ggml_backend_sched_alloc_graph(...)`
- `ggml_backend_sched_graph_compute(...)`
- `ggml_backend_sched_graph_compute_async(...)`
- `ggml_backend_sched_synchronize(...)`

This separates classic CPU graph execution from backend-managed execution on one or more devices.

### Execution Flow

A repo-aligned execution flow is:

1. **Build the graph** with `ggml_new_graph(...)` and `ggml_build_forward_expand(...)`.
2. **Choose an execution path**:
   - CPU path with `ggml_graph_plan(...)` and `ggml_graph_compute(...)`, or
   - backend path with `ggml_backend_graph_compute(...)` or the backend scheduler APIs.
3. **Prepare temporary resources**:
   - for the CPU path, allocate `cplan.work_data` if needed, or use `ggml_graph_compute_with_ctx(...)`;
   - for the scheduler path, reserve and allocate graph buffers as needed.
4. **Execute the graph** in dependency order through the selected execution API.

### Dispatch and Parallelism

On the CPU path, parallelism is configured through `ggml_cplan.n_threads` and `ggml_cplan.threadpool`. GGML also exposes threadpool management functions such as `ggml_threadpool_new(...)`, `ggml_threadpool_free(...)`, `ggml_threadpool_pause(...)`, and `ggml_threadpool_resume(...)`. For the CPU backend object, GGML provides `ggml_backend_cpu_set_n_threads(...)` and `ggml_backend_cpu_set_threadpool(...)`.

Type-specific CPU execution is also part of the public CPU interface. `include/ggml-cpu.h` exposes `struct ggml_type_traits_cpu`, which includes hooks such as `from_float`, `vec_dot`, `vec_dot_type`, and `nrows`, and provides access through `ggml_get_type_traits_cpu(...)`. This is the CPU-side type-dispatch layer used by quantized and non-quantized kernels.

### Example: Building and Executing a Graph

The following example follows the graph-construction pattern documented in `include/ggml.h` and uses the declared CPU execution API from `include/ggml-cpu.h`. The same compute call pattern also appears in `examples/gpt-2/main-ctx.cpp`.

```c
// 1. Initialize context
struct ggml_init_params params = {
    .mem_size   = 16 * 1024 * 1024,
    .mem_buffer = NULL,
};

struct ggml_context * ctx = ggml_init(params);

// 2. Create tensors
struct ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

ggml_set_param(ctx, x);

// 3. Build the expression: f(x) = a*x^2 + b
struct ggml_tensor * x2 = ggml_mul(ctx, x, x);
struct ggml_tensor * f  = ggml_add(ctx, ggml_mul(ctx, a, x2), b);

// 4. Build the forward graph
struct ggml_cgraph * gf = ggml_new_graph(ctx);
ggml_build_forward_expand(gf, f);

// 5. Set values
ggml_set_f32(x, 2.0f);
ggml_set_f32(a, 3.0f);
ggml_set_f32(b, 4.0f);

// 6. Execute
ggml_graph_compute_with_ctx(ctx, gf, n_threads);

// 7. Read result
float result = ggml_get_f32_1d(f, 0);
// result = 16
```

This graph has:

- **leaf tensors**: `x`, `a`, `b`
- **intermediate tensor**: `x2 = x * x`
- **output tensor**: `f = a * x2 + b`

The important point is that graph construction and graph execution are separate steps: tensor operations define the graph first, and execution happens only when a compute API is called.

## Testing and Validation

The `tests/` directory contains both broad backend validation and focused operator tests. In addition to `test-backend-ops.cpp`, the suite includes files such as `test-quantize-fns.cpp`, `test-quantize-perf.cpp`, `test-opt.cpp`, `test-conv2d.cpp`, `test-pool.c`, and other operator-specific tests.

### Cross-Backend Validation

`tests/test-backend-ops.cpp` is the main cross-backend validation driver. The file header states that it checks whether multiple backends produce consistent results for the same GGML operations during the forward pass. It also supports an optional gradient-validation path that compares backpropagation results against finite-difference estimates, and a performance mode for benchmarking.

The program supports four main execution modes:

- `test` → correctness checking
- `perf` → performance measurement
- `grad` → gradient validation
- `support` → backend support inspection

It also supports options such as `--list-ops`, `--show-coverage`, `--output`, and `--test-file`.

### Backend Enumeration

The test runner loads and enumerates available backends with `ggml_backend_load_all()`, walks the device list through `ggml_backend_dev_count()` and `ggml_backend_dev_get()`, initializes each backend with `ggml_backend_dev_init()`, and reports status per backend. It also queries backend memory information and, when supported, sets backend thread counts through the backend registry interface.

### Quantization Validation

Quantization is tested separately from general backend operator execution. The `tests/` tree includes `test-quantize-fns.cpp` for quantization-function correctness and `test-quantize-perf.cpp` for quantization performance benchmarking. This keeps quantization validation explicit instead of relying only on the broader backend-op test driver.

### Role in the Development Workflow

Together, these tests provide three complementary checks:

- functional correctness across backends
- numerical validation for gradients
- targeted performance measurement for both operators and quantization paths

This makes the test suite useful both for regression checking and for backend bring-up or optimization work.
