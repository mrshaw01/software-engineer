# Graph Scheduler and Multi-Backend Execution

This document describes the graph scheduler (`ggml_backend_sched`), which enables transparent execution of GGML computation graphs across multiple heterogeneous backend devices. The scheduler automatically:

- assigns operations to suitable backends based on operation support, backend priority, and tensor placement
- partitions graphs into backend-specific subgraphs called **splits**
- manages tensor copies across backend boundaries
- allocates compute buffers across one or more backend buffer types
- supports pipeline-style execution with multiple copies and backend events

## Scheduler Architecture

The scheduler sits above the backend, device, and buffer abstractions. Its job is to take one GGML computation graph and turn it into a coordinated multi-backend execution plan.

At a high level, it does four things:

1. **Backend assignment**
   Decide which backend should execute each node.

2. **Graph splitting**
   Partition the node list into contiguous backend-specific subgraphs.

3. **Copy management**
   Insert and manage tensor copies when a split depends on data produced on another backend.

4. **Allocation and execution**
   Allocate the required compute buffers, then execute each split on its assigned backend.

### Scheduler Data Structure

The scheduler is represented by `ggml_backend_sched_t`, an opaque handle publicly, with an internal structure in `src/ggml-backend.cpp`.

Important internal fields include:

- `n_backends`
  Number of backends managed by the scheduler.

- `backends[]`
  Ordered backend list. Lower indices have higher priority.

- `bufts[]`
  Buffer types associated with each backend.

- `galloc`
  Graph allocator used for backend-aware graph memory allocation.

- `hash_set`
  Hash table used to track tensors across the graph.

- `hv_tensor_backend_ids`
  Backend assignment per tensor.

- `hv_tensor_copies`
  Per-tensor copies across backends and pipeline copies.

- `node_backend_ids` / `leaf_backend_ids`
  Backend assignment arrays for graph nodes and leafs.

- `prev_node_backend_ids` / `prev_leaf_backend_ids`
  Previous assignments retained across graphs.

- `graph`
  Internal copied graph with modified inputs for split execution.

- `splits`
  Array of backend-specific graph partitions.

- `n_splits`
  Number of splits produced for the last graph.

- `n_copies`, `cur_copy`, `next_copy`
  Pipeline-parallel copy management state.

- `events[backend][copy]`
  Event objects used for synchronization across backends and copies.

- `graph_inputs[]` / `n_graph_inputs`
  Tracked graph inputs.

- `ctx`
  Internal no-alloc context used when splitting the graph.

- `callback_eval` / `callback_eval_user_data`
  Optional evaluation callback support.

- `context_buffer` / `context_buffer_size`
  Internal metadata buffer for the scheduler’s own temporary graph context.

- `op_offload`
  Enables backend offload hints during assignment.

### Split Representation

Each split is represented internally by `struct ggml_backend_sched_split` and contains:

- `backend_id`
  Which backend executes the split.

- `i_start`, `i_end`
  Range of nodes in the original graph covered by the split.

- `inputs[]`
  Tensors that must be available before the split executes.

- `n_inputs`
  Number of split inputs.

- `graph`
  A graph view for this split.

This means a split is not a separate user graph. It is a view into part of the original graph plus the extra metadata needed to execute it on one backend.

### Key Configuration Constants

The scheduler implementation defines three important limits:

- `GGML_SCHED_MAX_BACKENDS = 16`
  Maximum number of backends in one scheduler.

- `GGML_SCHED_MAX_SPLIT_INPUTS = 30`
  Maximum number of tracked input tensors per split.

- `GGML_SCHED_MAX_COPIES = 4`
  Maximum number of pipeline copies tracked per backend.

These constants bound the scheduler’s internal arrays for backend lists, split inputs, copies, and events.

## Public Scheduler API

The public scheduler interface is declared in `include/ggml-backend.h`.

### Construction and Lifetime

- `ggml_backend_sched_new(backends, bufts, n_backends, graph_size, parallel, op_offload)`
  Create a scheduler for a backend set.

- `ggml_backend_sched_free(sched)`
  Destroy the scheduler.

Backends with lower indices are given higher priority.

### Reservation and Allocation

- `ggml_backend_sched_reserve_size(sched, measure_graph, sizes)`
  Compute required buffer sizes from a measure graph.

- `ggml_backend_sched_reserve(sched, measure_graph)`
  Reserve scheduler buffers from a graph.

- `ggml_backend_sched_alloc_graph(sched, graph)`
  Allocate graph storage on the scheduler.

### Query Functions

- `ggml_backend_sched_get_n_backends(sched)`
- `ggml_backend_sched_get_backend(sched, i)`
- `ggml_backend_sched_get_n_splits(sched)`
- `ggml_backend_sched_get_n_copies(sched)`
- `ggml_backend_sched_get_buffer_type(sched, backend)`
- `ggml_backend_sched_get_buffer_size(sched, backend)`

### Manual Placement Hooks

- `ggml_backend_sched_set_tensor_backend(sched, tensor, backend)`
- `ggml_backend_sched_get_tensor_backend(sched, tensor)`

These let callers pin specific tensors to specific backends before allocation.

### Split and Execution Functions

- `ggml_backend_sched_split_graph(sched, graph)`
- `ggml_backend_sched_graph_compute(sched, graph)`
- `ggml_backend_sched_graph_compute_async(sched, graph)`
- `ggml_backend_sched_synchronize(sched)`

### Reset and Observation

- `ggml_backend_sched_reset(sched)`
  Reset all assignments and allocator state.

- `ggml_backend_sched_set_eval_callback(sched, callback, user_data)`
  Set a node-evaluation callback.

## Backend Assignment Model

The scheduler chooses backends using three main signals:

1. **Operation support**
   Whether the backend supports the node’s operation.

2. **Tensor location**
   Whether an input or weight is already allocated in a buffer compatible with that backend.

3. **Backend priority**
   Lower-index backends are preferred when multiple backends can run the same work.

The scheduler also considers backend offload hints when `op_offload` is enabled.

## Graph Splitting Model

After backend assignment, the scheduler partitions the graph into contiguous regions that can run on a single backend.

A split boundary appears when:

- the backend assignment changes between adjacent nodes
- a backend needs data produced elsewhere
- copies must be inserted between producer and consumer backends

Each split is then executed as a backend-specific graph view.

## Copy Management

When a split depends on tensors located on another backend, the scheduler creates and tracks copied tensors. Internally, tensor copies are indexed by:

- original tensor
- target backend
- pipeline copy id

This allows one logical tensor to have multiple backend-local materializations.

The scheduler uses these copies to:

- move weights or intermediate tensors between backends
- keep backend-local execution efficient
- support overlapped pipeline-style execution

## Pipeline Parallelism

The scheduler can operate with multiple copies when `parallel` is enabled at construction time.

Pipeline support is visible in the fields:

- `n_copies`
- `cur_copy`
- `next_copy`
- `events[backend][copy]`

This allows different copies of the graph state to move through backend execution with event-based synchronization between stages.

## Internal Graph Context

During graph splitting, the scheduler creates an internal `ggml_context` with:

- a scheduler-owned metadata buffer
- `no_alloc = true`

This context is used to build copied graph views and split-local graph structures without allocating tensor payloads. Actual payload allocation is handled later through the graph allocator and backend buffer types.

## Typical Scheduler Flow

A typical end-to-end scheduler flow is:

1. create several backends
2. create the scheduler with `ggml_backend_sched_new(...)`
3. optionally reserve memory with a worst-case graph
4. optionally assign selected tensors to specific backends
5. build the runtime graph
6. let the scheduler split and allocate it
7. execute with `ggml_backend_sched_graph_compute(...)`
8. synchronize if needed

## Practical Mental Model

A useful way to think about the scheduler is:

- the **graph** describes what must be computed
- the **scheduler** decides where each part should run
- the **allocator** decides where each tensor should live
- the **copy manager** moves tensors across backend boundaries
- the **backend APIs** execute each split on the selected device

## Five-Pass Graph Splitting Algorithm

The scheduler split routine does two jobs at once:

1. assign each executable node to a backend
2. partition the graph into backend-specific splits and identify the tensors that must be copied between them

The algorithm works over the graph’s `nodes[]` array and treats view-style ops specially. View ops do not represent standalone compute work, so they are generally skipped during backend assignment and split-boundary detection.

### Pre-Step: Scheduler Reset and Working Context

Before the five passes begin, the scheduler resets its split state and rebuilds an internal no-allocation context used to construct split-local graph views.

This setup phase typically does the following:

- clear previous split count
- clear tracked graph inputs
- reset internal copy bookkeeping
- recreate the scheduler-owned `ggml_context` with `no_alloc = true`
- prepare backend-id arrays for nodes and leafs
- preserve any explicit user placement that was already set on tensors

This gives the scheduler a clean workspace for assignment and split construction.

### Pass 1: Pre-allocated Tensor Assignment

**Goal:** seed backend assignments from tensors that already have a concrete location.

This pass gives the scheduler its initial anchors. It looks for tensors that are already tied to a backend, usually because:

- the tensor already has an allocated buffer
- the tensor was explicitly assigned by the user
- a view tensor inherits storage from a source tensor that is already placed

Typical behavior in this pass:

- if a leaf tensor already lives in a backend buffer, record that backend
- if a node consumes a source tensor that is already allocated, try assigning the node to that same backend
- only keep that assignment if the backend supports the operation
- leave unsupported nodes unassigned for later passes

Why this matters:

- weights often already live on a chosen backend
- KV-cache or persistent buffers may already be placed
- user-pinned tensors should influence scheduling early

So pass 1 does not try to solve the whole graph. It establishes trustworthy placement seeds from already-placed tensors.

### Pass 2: Expand Adjacent Assignments

**Goal:** grow the seeded assignments into neighboring nodes to form larger contiguous backend regions.

This pass tries to avoid over-fragmenting the graph. Once some nodes are assigned in pass 1, the scheduler expands those assignments to adjacent unassigned nodes whenever the backend can execute them.

The implementation does this as an adjacency expansion pass, with an important policy:

- non-lowest-priority backends are expanded preferentially
- the lowest-priority backend, typically CPU, is not eagerly expanded the same way
- unsupported nodes stay unassigned instead of forcing a bad placement

Conceptually, the pass behaves like this:

#### Downward expansion

Walk forward through the graph:

- keep track of the most recent active backend assignment
- when the current node is unassigned, try assigning it to that backend
- skip view ops
- if the active backend becomes the lowest-priority backend, stop propagating it aggressively

#### Upward expansion

Walk backward through the graph:

- perform the same idea in reverse
- this fills in short gaps that were missed by the forward sweep
- again, only assign if the backend supports the operation

Why this pass exists:

- it reduces backend switching
- it keeps GPU-capable regions contiguous
- it prevents CPU from “winning” too early just because it supports almost everything

The result is a graph that already looks much more clustered by backend, but still leaves ambiguous nodes unresolved.

### Pass 3: Upgrade to Higher-Priority Backends

**Goal:** move nodes to a better backend when that move is safe with respect to buffer compatibility.

After pass 2, some nodes may be assigned to a backend that works, but is not the best choice. Pass 3 tries to upgrade those nodes to a higher-priority backend, typically moving work away from a lower-priority backend when the tensor-storage situation allows it.

The key condition here is **buffer compatibility**.

The scheduler does not simply ask, “can backend B run this op?”
It also asks, “can backend B work with the tensor’s existing or planned buffer type?”

Typical checks in this pass:

- if the tensor is already allocated, inspect its actual buffer type
- if it is not yet allocated, inspect the backend buffer type associated with its assigned placement
- verify that the candidate higher-priority backend supports that buffer type
- verify that the operation itself is supported

Why this matters:

- multiple backends may be able to use the same host-memory buffer type
- a node assigned to a lower-priority backend may be safely upgraded without introducing extra copies
- this is especially useful for backends that share host memory semantics

In practice, this pass improves placement quality without breaking buffer compatibility.

### Pass 4: Assign Remaining Unassigned Nodes

**Goal:** finish backend assignment for all nodes that still have no placement.

After the first three passes, the remaining unassigned nodes are usually the hard cases:

- ops not supported by the backend being expanded
- nodes between differently placed regions
- nodes whose placement depends on the locations of their sources

This pass resolves those nodes by looking at source tensors and local graph context.

Typical logic in this pass:

- inspect the node’s source tensors
- look at the backends already assigned to those sources
- prefer a backend that:
  - supports the op
  - is compatible with the relevant buffer type
  - keeps the node near its producers or consumers
- if several choices are possible, prefer the higher-priority backend
- if a current local backend region already exists, prefer staying within it when valid

This pass is where the scheduler stops being speculative and makes a complete assignment decision for every executable node.

By the end of pass 4:

- every non-view compute node is expected to have a backend assignment
- the graph is ready to be partitioned into splits

### Pass 5: Create Split Structures

**Goal:** convert the per-node backend assignments into executable backend-specific subgraphs.

Once every compute node has a backend, the scheduler walks the graph and creates `split` records.

A split is a contiguous run of nodes executed on one backend. Each split stores:

- the backend id
- start node index
- end node index
- a list of external input tensors needed by that split
- a graph view representing just that region

#### Step 5.1: Find the first real split backend

The scheduler starts by finding the first non-view node and uses its assigned backend as the backend for the first split.

View ops are skipped because they do not define standalone execution regions.

#### Step 5.2: Extend the split while the backend is stable

As the scheduler advances through nodes:

- if the next compute node has the same backend, keep it in the current split
- if the backend changes, terminate the current split and start a new one

#### Step 5.3: Detect split inputs

For each node in the split, inspect its sources.

If a source tensor comes from:

- a different backend, or
- an incompatible weight buffer, or
- a tensor that must be materialized locally for this split,

then that source is added to the split’s `inputs[]` list.

These tensors are the boundary values that must exist on the split backend before execution can begin.

#### Step 5.4: Insert copy requirements

When a split needs a tensor produced or stored elsewhere, the scheduler creates or reuses a backend-local copy entry for that tensor.

This copy tracking is indexed by:

- original tensor
- destination backend
- pipeline copy slot

That lets the scheduler support:

- ordinary inter-backend copies
- repeated execution with pipeline copies
- reuse of previously created split-local tensors

#### Step 5.5: Start a new split when necessary

A new split is created when any of these conditions hold:

- the node backend changes
- a source weight lives on a different incompatible backend
- the split input list reaches its configured limit
- continuing would make the copy structure or dependency boundary invalid

This keeps each split executable as one clean backend-local subgraph.

## Why the Algorithm Uses Five Passes

The five-pass structure is deliberate.

### Pass 1

Seeds the graph with strong placement evidence from tensors that already have a location.

### Pass 2

Expands those seeds into nearby nodes to reduce fragmentation.

### Pass 3

Improves placement quality by moving work to higher-priority compatible backends.

### Pass 4

Resolves the ambiguous leftovers using source-driven placement.

### Pass 5

Turns the finished assignment map into executable splits and copy boundaries.

If GGML tried to do all of this in one traversal, it would either:

- assign too much work to CPU too early
- create too many tiny splits
- miss opportunities to keep work on a higher-priority backend
- or insert unnecessary copies

The multi-pass approach gives the scheduler enough context to make better placement decisions.

## Practical Mental Model

A good way to think about the algorithm is:

- **Pass 1:** “What is already anchored somewhere?”
- **Pass 2:** “What nearby work can stay with those anchors?”
- **Pass 3:** “Can any of that work move to a better backend safely?”
- **Pass 4:** “What still has no home, and where should it go?”
- **Pass 5:** “Now that every node has a backend, where are the execution boundaries and required copies?”

That is the core scheduling loop that turns one GGML graph into a multi-backend execution plan.

## Split Execution and Tensor Copying

### Execution Flow

Once the scheduler has produced splits, execution proceeds split by split in backend order, while respecting cross-backend dependencies and copy requirements.

A high-level execution flow is:

1. build or reuse backend-local tensor copies required by each split
2. make sure split inputs are available on the destination backend
3. launch the split graph on that backend
4. optionally record backend events for pipeline copies
5. move to the next split when its dependencies are satisfied

In practice, a split executes as a backend-local subgraph:

- one backend
- one contiguous node range
- one list of external input tensors that must be present before execution

This means the scheduler does not execute the original graph monolithically. It executes a sequence of backend-specific graph fragments.

There are two main entry points:

- `ggml_backend_sched_graph_compute(...)`
- `ggml_backend_sched_graph_compute_async(...)`

The synchronous function is the blocking execution path. The asynchronous function allows backend work and copies to remain queued until explicit synchronization.

A typical compute sequence is:

1. `ggml_backend_sched_split_graph(...)`
2. `ggml_backend_sched_alloc_graph(...)`
3. `ggml_backend_sched_graph_compute(...)` or `ggml_backend_sched_graph_compute_async(...)`
4. `ggml_backend_sched_synchronize(...)` if the caller needs completion

### Copy Identification and Execution

Tensor copying is driven by split boundaries.

A copy is needed when a split consumes a tensor that is not already usable on the split backend. This usually happens in cases such as:

- the source tensor was produced by a different backend
- the tensor is a persistent weight placed on another backend
- the tensor exists in a buffer type not directly usable by the destination backend
- the scheduler is running with multiple pipeline copies and needs per-copy materialization

For each split, the scheduler tracks a list of split inputs. These are tensors that must be available before that split can run. A split input may be:

- a leaf tensor such as a weight or input tensor
- an intermediate tensor produced by an earlier split
- a tensor view whose source storage must be materialized on the destination backend

The scheduler then maps each required split input to a backend-local tensor copy.

Conceptually, the copy mapping is indexed by:

- original tensor
- destination backend
- copy slot for pipeline execution

This allows the scheduler to reuse already-created backend-local copies instead of recreating them every time.

The important distinction is:

- **original tensor**: the logical tensor from the main graph
- **copied tensor**: a backend-local materialization used by one split

The split graph is built from these local tensors so each backend sees tensors in its own address space and buffer model.

### Asynchronous Copy Mechanism

The scheduler uses backend-level asynchronous copy support when available.

The relevant backend API is:

- `ggml_backend_tensor_copy_async(backend_src, backend_dst, src, dst)`

The semantics are:

- the copy is queued after the currently pending work on the source backend
- the destination backend waits for the copy before using the destination tensor
- if async copy is not supported, GGML falls back to a synchronous copy path

This gives the scheduler a uniform copy mechanism across backends.

For pipeline execution, the scheduler also maintains backend events per backend and per copy slot. These events let the scheduler:

- record when a split has completed on one backend
- make another backend wait before consuming the copied tensor
- overlap compute and transfer across different split copies

So the asynchronous model is not just “copy later.” It is:

- queue copy after producer work
- establish dependency on the destination side
- continue issuing backend work as allowed by the event graph

This is what enables pipeline-style overlapping instead of strict global synchronization after every split.

## Memory Management Integration

### Allocator Configuration

The scheduler is tightly coupled with the graph allocator.

Its allocator-facing state includes:

- the backend list
- one buffer type per backend
- a graph allocator
- per-node and per-leaf backend ids
- copy-tracking structures for tensors materialized on multiple backends

At construction time, the scheduler receives:

- `backends`
- `bufts`
- `n_backends`
- `graph_size`
- `parallel`
- `op_offload`

These configuration inputs determine:

- which backends are available
- which buffer types will be used for compute allocation
- whether multiple pipeline copies should be maintained
- whether backend offload hints should influence placement

The allocator itself is created to understand one or more backend buffer types, not just one flat memory pool.

### Allocation Process

Allocation happens after or alongside graph splitting.

The typical flow is:

1. assign node and leaf backends
2. split the graph
3. determine which tensors belong to which backend buffer id
4. reserve or allocate graph memory through the graph allocator
5. create backend buffers for the required chunks
6. attach tensors and views to those buffers

There are two major reservation paths:

- `ggml_backend_sched_reserve_size(...)`
- `ggml_backend_sched_reserve(...)`

These are used to size scheduler-managed buffers from a measure graph, often before the real runtime graph is executed.

Then the main allocation path is:

- `ggml_backend_sched_alloc_graph(...)`

This is the point where the scheduler turns backend assignments and split structure into actual tensor placement.

The scheduler does not allocate memory by itself using ad hoc rules. Instead, it translates scheduling decisions into graph-allocator inputs and lets the allocator materialize backend buffers accordingly.

### Node and Leaf Buffer Assignment

The scheduler maintains separate backend assignment arrays for:

- graph nodes
- graph leafs

This distinction matters because leaf tensors often represent:

- model weights
- graph inputs
- externally managed tensors
- persistent state such as cache tensors

while nodes are computed tensors produced during graph execution.

During allocation, the scheduler maps each tensor to a buffer id based on:

- explicit user placement, if any
- existing tensor buffer placement
- backend assignment chosen by the scheduler
- compatibility between tensor storage and backend buffer type

In effect:

- **leaf buffer assignment** determines where persistent or input tensors live
- **node buffer assignment** determines where intermediate compute tensors are allocated

This separation is important because a graph can mix:

- long-lived weight storage
- short-lived compute storage
- cross-backend copied tensors
- view tensors that inherit storage from another tensor

The allocator must preserve all of those relationships while still maximizing reuse.

A useful way to think about the integration is:

- the scheduler decides **which backend should own execution**
- the allocator decides **which buffer should hold the tensor**
- the copy manager ensures **the right tensor version is present on the right backend at the right time**

That is how GGML turns one logical graph into a multi-backend, copy-aware, buffer-allocated execution plan.

## Pipeline Parallelism

### Pipeline Architecture

The scheduler supports a pipeline-style execution mode by keeping multiple logical copies of split inputs and copy tensors in flight at the same time.

At the scheduler level, the key fields are:

- `n_copies`
- `cur_copy`
- `events[backend][copy]`

When pipeline mode is enabled, the scheduler does **not** execute only one global set of split inputs. Instead, it maintains several copy slots. Each slot represents one pipeline stage state for tensors that must be materialized on different backends.

The model is:

- one logical graph
- multiple split-local tensor-copy sets
- one active copy slot selected by `cur_copy`
- one event per backend per copy slot

This lets the scheduler overlap:

- execution of one split on one backend
- tensor copies for another split
- reuse of split-local inputs from previous iterations

#### Copy-slot organization

The scheduler indexes copied tensors by:

- original tensor
- destination backend
- copy slot

So one tensor may have:

- its original storage
- one local copy on backend A for slot 0
- one local copy on backend A for slot 1
- one local copy on backend B for slot 0
- and so on

This is why pipeline mode increases memory usage: copied tensors are no longer singletons. They can exist multiple times so that different in-flight graph evaluations do not overwrite each other.

#### Input handling in pipeline mode

Input tensors are treated specially.

When a tensor is marked with `GGML_TENSOR_FLAG_INPUT` and the scheduler uses more than one copy slot, the scheduler creates per-copy versions of that input. The current copy slot can reuse the original input tensor, while the other slots use duplicated tensor layouts. Those duplicated tensors are marked as both input and output so the allocator keeps them alive and does not overwrite them during graph allocation.

This gives the pipeline two important properties:

1. the caller can keep feeding new input values without immediately destroying in-flight copies
2. backend-local split execution can continue using older copies while a newer request is being prepared

#### Why copies are per-backend and per-slot

A single copy dimension is not enough.

- **Per-backend** copies are needed because split execution is backend-local.
- **Per-slot** copies are needed because pipeline execution reuses the same logical graph across multiple overlapping iterations.

Without both dimensions, one iteration could overwrite tensor data still needed by another backend or another in-flight pipeline slot.

### Event-Based Synchronization

Pipeline synchronization is built around backend events.

Each backend and copy slot can have one event object:

- `events[backend][copy]`

These events are used to coordinate safe reuse of copied tensors and to avoid global synchronization after every split.

#### Event lifecycle inside split execution

For a split with inputs, the scheduler does the following:

1. determine which copied input tensor should be used for the current copy slot
2. make sure that destination storage is safe to overwrite
3. copy or queue-copy the source tensor into the split-local tensor copy
4. run the split graph on the split backend
5. record an event for that backend and copy slot once the split has consumed its inputs

This event becomes the synchronization point for later reuse of that same copy slot.

#### Two synchronization cases

There are two important cases during input-copy handling.

##### 1. User-provided graph inputs

If the input tensor is a graph input coming from the user, the scheduler copies it immediately before execution. The reason is simple: the caller may overwrite that input buffer as soon as the API call returns or even while backend work is still pending. So the scheduler first waits until the destination copy slot is safe, then performs the copy right away.

##### 2. Internal or persistent tensors

If the input is not a direct user input, the scheduler can treat reuse more like backend-managed storage. In that case, it waits for the split backend to finish using the destination copy slot before overwriting it, then tries to perform an asynchronous backend copy.

So the synchronization rule is:

- **user input**: copy early to protect against caller overwrite
- **internal tensor**: wait for destination reuse safety, then copy through backend logic

#### Async copy path and fallback

When possible, the scheduler uses backend asynchronous copy support. If that is not available, it falls back to a synchronous path.

The fallback still preserves correctness because the scheduler already uses events or backend synchronization to guarantee that:

- the source data is ready
- the destination copy slot is no longer in use
- later compute sees the copied tensor only after the transfer is complete

So async copy improves overlap, but correctness does not depend on it.

#### Event recording

After a split finishes, the scheduler records an event for the current backend and current copy slot when the split has inputs. That event means:

- this backend has consumed or produced data for this pipeline slot
- later reuse of the same copy slot on this backend must wait for that event

This avoids unnecessary global barriers. Instead of synchronizing all backends after every split, the scheduler uses fine-grained per-backend, per-slot dependency tracking.

#### Copy-slot rotation

At the end of split execution, the scheduler advances:

- `cur_copy = (cur_copy + 1) % n_copies`

This is what turns the copy-slot system into a pipeline. Each new graph execution uses the next slot, while previous slots may still be finishing work on one or more backends.

### Pipeline Initialization

Pipeline support is enabled when the scheduler is created with `parallel = true`.

#### Scheduler construction behavior

During `ggml_backend_sched_new(...)`, the scheduler:

1. stores the backend list
2. stores the associated buffer types
3. sets `n_copies`
4. creates per-backend event objects when multiple copies are enabled
5. creates the graph allocator with all scheduler buffer types
6. resets scheduler state

The copy count is chosen as:

- `1` when pipeline mode is disabled
- `GGML_SCHED_MAX_COPIES` when pipeline mode is enabled

So pipeline mode is not a dynamic “scale from one to any number” feature at runtime. It is a fixed multi-copy mode chosen at scheduler construction time.

#### Event creation during initialization

When `n_copies > 1`, the scheduler allocates an event for every:

- backend
- copy slot

This means event initialization cost is front-loaded into scheduler creation rather than paid inside the hot execution loop.

If a backend cannot provide an event object, the event entry may be `NULL`. In that case, the scheduler falls back to backend-wide synchronization for that path.

#### Relationship to backend ordering

The scheduler constructor requires:

- at least one backend
- the last backend to be CPU

This ordering rule matters because the scheduler treats lower-index backends as higher priority during assignment, while CPU serves as the general fallback backend.

Pipeline parallelism does not change that placement rule. It only changes how split-local data is copied and reused during execution.

#### Practical enablement conditions

At the scheduler level, pipeline mode is simply the `parallel` construction flag.

At the application level, pipeline mode is usually enabled only when the participating devices can actually benefit from it. In practice this means:

- multiple non-CPU devices
- asynchronous execution support
- event support

If those capabilities are missing, pipeline mode degenerates toward ordinary split execution because the scheduler must synchronize more aggressively.

#### Interaction with memory usage

Pipeline mode increases memory usage for two reasons:

1. copied tensors may exist once per backend and per copy slot
2. event and bookkeeping state also scale with copy count

So pipeline mode is a throughput optimization, not a free win. It trades extra memory for the ability to overlap:

- copy of one iteration
- compute of another iteration
- reuse of split-local storage without destructive interference

### Mental Model

A useful way to think about scheduler pipeline parallelism is:

- **splits** define backend-local work units
- **copy slots** let multiple iterations coexist safely
- **events** protect each backend-local copy slot from being overwritten too early
- **cur_copy** rotates through the available pipeline slots
- **async copy + async compute** create the overlap that makes the pipeline useful

## Scheduler API Usage

### Initialization and Configuration

The scheduler is created with:

```c
ggml_backend_sched_t ggml_backend_sched_new(
    ggml_backend_t * backends,
    ggml_backend_buffer_type_t * bufts,
    int n_backends,
    size_t graph_size,
    bool parallel,
    bool op_offload
);
```

Parameters:

- `backends`
  Ordered array of execution backends. Lower indices have higher priority.

- `bufts`
  Optional array of buffer types, one per backend. If `NULL`, the scheduler uses each backend’s default buffer type.

- `n_backends`
  Number of backends managed by the scheduler.

- `graph_size`
  Size hint for the scheduler’s internal graph metadata context.

- `parallel`
  Enables multi-copy pipeline mode.

- `op_offload`
  Enables backend offload hints during backend assignment.

A common configuration pattern is:

- GPU backends first
- CPU backend last
- `parallel = false` for simple execution
- `parallel = true` when pipeline copies and event-based overlap are desired
- `op_offload = true` when backend offload hints should influence assignment

Example:

```c
ggml_backend_t backends[] = { backend_gpu0, backend_gpu1, backend_cpu };

ggml_backend_sched_t sched = ggml_backend_sched_new(
    backends,
    NULL,
    3,
    GGML_DEFAULT_GRAPH_SIZE,
    false,
    true
);
```

The scheduler lifetime is explicit:

```c
ggml_backend_sched_free(sched);
```

### Basic Usage Pattern

The scheduler is meant to be used in a loop where graph metadata may be rebuilt, but backend-managed allocation is reused whenever possible.

A typical pattern is:

1. create backends
2. create the scheduler
3. optionally reserve memory from a worst-case graph
4. build a graph
5. execute it with the scheduler
6. rebuild or reset only when graph allocation changes

Minimal flow:

```c
ggml_backend_sched_t sched = ggml_backend_sched_new(
    backends, NULL, n_backends, GGML_DEFAULT_GRAPH_SIZE, false, true
);

struct ggml_cgraph * graph = build_graph(...);

ggml_backend_sched_graph_compute(sched, graph);
```

Important behavior:

- on the first execution, the graph may be allocated automatically
- later executions can reuse the allocation when the graph structure and allocation assumptions remain compatible
- if graph inputs or graph structure change in a way that requires reallocation, the scheduler should be reset and the graph rebuilt

### Memory Reservation

The scheduler supports reserving memory ahead of time from a measure graph.

Two APIs are provided:

```c
void ggml_backend_sched_reserve_size(
    ggml_backend_sched_t sched,
    struct ggml_cgraph * measure_graph,
    size_t * sizes
);

bool ggml_backend_sched_reserve(
    ggml_backend_sched_t sched,
    struct ggml_cgraph * measure_graph
);
```

Use cases:

- reserve from a worst-case graph shape
- pre-size backend compute buffers before the real execution graph arrives
- reduce or avoid reallocations during the main compute loop

Typical usage:

```c
struct ggml_cgraph * reserve_graph = build_max_graph(...);
ggml_backend_sched_reserve(sched, reserve_graph);
```

If you want the raw size requirement per backend, use `ggml_backend_sched_reserve_size(...)`.

You can also inspect the resulting buffer size later with:

```c
size_t sz = ggml_backend_sched_get_buffer_size(sched, backend);
```

### Graph Execution

The scheduler provides three main execution-related entry points:

```c
void ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph);

bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph);

enum ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph);

enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph);

void ggml_backend_sched_synchronize(ggml_backend_sched_t sched);
```

There are two common execution styles.

#### Style 1: Automatic allocate-and-run

```c
struct ggml_cgraph * graph = build_graph(...);
ggml_backend_sched_graph_compute(sched, graph);
```

This is the simplest path. The scheduler will split, allocate if needed, and execute.

#### Style 2: Explicit split and allocation

```c
struct ggml_cgraph * graph = build_graph(...);

ggml_backend_sched_split_graph(sched, graph);
ggml_backend_sched_alloc_graph(sched, graph);

// fill input tensors here
ggml_backend_tensor_set(input, data, 0, size);

ggml_backend_sched_graph_compute(sched, graph);
```

This pattern is useful when:

- inputs must be written after allocation but before execution
- you want explicit control over allocation timing
- you want to inspect the split structure before running

For asynchronous execution:

```c
ggml_backend_sched_graph_compute_async(sched, graph);
// do other work
ggml_backend_sched_synchronize(sched);
```

### Manual Backend Assignment

The scheduler can place tensors automatically, but it also provides manual placement hooks:

```c
void ggml_backend_sched_set_tensor_backend(
    ggml_backend_sched_t sched,
    struct ggml_tensor * node,
    ggml_backend_t backend
);

ggml_backend_t ggml_backend_sched_get_tensor_backend(
    ggml_backend_sched_t sched,
    struct ggml_tensor * node
);
```

This lets you pin a node or tensor to a specific backend before allocation.

Example:

```c
struct ggml_tensor * node = ggml_mul_mat(ctx, a, b);
ggml_backend_sched_set_tensor_backend(sched, node, backend_gpu0);
```

Manual assignment is useful when:

- you want one op to stay on a particular accelerator
- you already know a tensor should execute near its weights
- you are debugging scheduler placement
- you want deterministic placement for a specific hotspot

Important rule:

- if you change manual assignments after a graph has already been allocated, call `ggml_backend_sched_reset(...)` and rebuild / reallocate the graph before executing again

### Split Introspection

The scheduler exposes several query functions for introspection:

```c
int ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched);
ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, int i);

int ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched);
int ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched);

ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(
    ggml_backend_sched_t sched,
    ggml_backend_t backend
);

size_t ggml_backend_sched_get_buffer_size(
    ggml_backend_sched_t sched,
    ggml_backend_t backend
);
```

These let you inspect:

- how many backends are active
- which backend is at each priority index
- how many splits were created for the last graph
- how many pipeline copies the scheduler is maintaining
- which buffer type is associated with a backend
- how much compute-buffer space was reserved for a backend

Example:

```c
int n_splits = ggml_backend_sched_get_n_splits(sched);

for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
    ggml_backend_t backend = ggml_backend_sched_get_backend(sched, i);
    size_t buf_size = ggml_backend_sched_get_buffer_size(sched, backend);
}
```

A practical use for split introspection is:

- checking whether your graph is overly fragmented
- checking whether a backend is receiving any work
- checking whether pipeline mode created multiple copy slots
- estimating compute-buffer pressure per backend

### Evaluation Callbacks

The scheduler supports per-node observation through an evaluation callback.

The callback type is:

```c
typedef bool (*ggml_backend_sched_eval_callback)(
    struct ggml_tensor * t,
    bool ask,
    void * user_data
);
```

It is installed with:

```c
void ggml_backend_sched_set_eval_callback(
    ggml_backend_sched_t sched,
    ggml_backend_sched_eval_callback callback,
    void * user_data
);
```

The callback has two modes.

#### Ask phase

When `ask == true`, the scheduler is asking whether the user wants to observe that node.

This exists so the scheduler can decide whether several nodes may still be batched together, or whether it needs to expose a node boundary for observation.

In other words:

- `ask == true` means: _do you care about this node?_

#### Observe phase

When `ask == false`, the scheduler passes the tensor to the user for observation.

In this phase:

- returning `true` means continue execution
- returning `false` means cancel graph execution

This makes the callback useful for:

- intermediate tensor inspection
- debugging graph execution
- dumping selected node outputs
- implementing early-stop or custom validation logic

Example pattern:

```c
bool my_eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask) {
        return should_observe_tensor(t);
    }

    inspect_tensor(t);

    return true; // continue execution
}

ggml_backend_sched_set_eval_callback(sched, my_eval_callback, my_user_data);
```

### Reset Semantics

The reset API is important enough to treat as part of normal scheduler usage:

```c
void ggml_backend_sched_reset(ggml_backend_sched_t sched);
```

Reset does all of the following:

- clears assignments
- resets allocator state
- invalidates previously allocated graph tensors
- requires the caller to discard old graph tensors and create new ones

Use reset when:

- changing backend assignments
- switching to a different graph allocation pattern
- rebuilding inputs or nodes that require a fresh allocation
- moving from one graph instance to another incompatible one

Incorrect usage is:

- keep old allocated graph tensors alive after reset
- change node placement and then reuse stale tensors

Correct usage is:

1. reset scheduler
2. rebuild graph tensors
3. allocate or compute again

### End-to-End Example

A complete scheduler-driven execution pattern looks like this:

```c
ggml_backend_t backends[] = { backend_gpu0, backend_gpu1, backend_cpu };

ggml_backend_sched_t sched = ggml_backend_sched_new(
    backends,
    NULL,
    3,
    GGML_DEFAULT_GRAPH_SIZE,
    false,
    true
);

// Optional worst-case reservation
struct ggml_cgraph * reserve_graph = build_max_graph(...);
ggml_backend_sched_reserve(sched, reserve_graph);

// Build actual graph
struct ggml_cgraph * graph = build_graph(...);

// Optional manual placement
ggml_backend_sched_set_tensor_backend(sched, some_node, backend_gpu0);

// Explicit allocation before writing inputs
ggml_backend_sched_alloc_graph(sched, graph);

// Fill inputs
ggml_backend_tensor_set(input_tensor, input_data, 0, input_size);

// Execute
enum ggml_status st = ggml_backend_sched_graph_compute(sched, graph);

// Clean up
ggml_backend_sched_free(sched);
```

### Practical Guidance

A good rule of thumb is:

- use `ggml_backend_sched_graph_compute(...)` for the simple path
- use `ggml_backend_sched_reserve(...)` when you know the worst-case graph size
- use `ggml_backend_sched_alloc_graph(...)` when inputs must be filled after allocation
- use `ggml_backend_sched_set_tensor_backend(...)` only for deliberate placement control
- use `ggml_backend_sched_get_n_splits(...)` and buffer-size queries to understand scheduler behavior
- use `ggml_backend_sched_set_eval_callback(...)` for debug observation or custom stopping logic
- call `ggml_backend_sched_reset(...)` before changing placement or allocating an incompatible new graph

## Implementation Details

### Hash-Based Tensor Tracking

The scheduler uses a hash-based indirection layer so it can track per-tensor scheduling state without modifying the tensor objects themselves.

At the core of this mechanism is:

- `struct ggml_hash_set hash_set`

This hash set stores tensor pointers and assigns each tracked tensor a stable integer slot. The scheduler then uses that slot to index parallel state arrays.

The main internal mapping macros are:

```c
#define hash_id(tensor) ggml_hash_find_or_insert(&sched->hash_set, tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#define tensor_id_copy(id, backend_id, copy_id) \
    sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + \
                            (backend_id) * sched->n_copies + \
                            (copy_id)]
#define tensor_copy(tensor, backend_id, copy_id) \
    tensor_id_copy(hash_id(tensor), backend_id, copy_id)
```

This gives the scheduler three important properties:

1. **Stable per-tensor indexing**
   A tensor can be looked up once and then referenced by integer id across the whole scheduling pipeline.

2. **Separate scheduling metadata**
   Backend assignment and copied-tensor tracking live in scheduler-owned arrays instead of inside `ggml_tensor`.

3. **Multi-backend and multi-copy support**
   One logical tensor can map to:

   - one backend assignment
   - zero or more backend-local copies
   - one copy per backend and per pipeline slot

The main arrays driven by the hash id are:

- `hv_tensor_backend_ids`
  backend assignment per tracked tensor

- `hv_tensor_copies`
  backend-local copied tensors per tracked tensor, backend, and copy slot

This is why the scheduler can keep the original graph unchanged while still building backend-local split graphs and copied-tensor materializations.

A useful mental model is:

- `hash_set` answers: _which scheduler slot belongs to this tensor?_
- `hv_tensor_backend_ids` answers: _which backend owns this tensor logically?_
- `hv_tensor_copies` answers: _which backend-local materialization exists for this tensor right now?_

### View Operation Handling

The scheduler treats view-style operations specially.

Internally, it defines:

```c
static bool ggml_is_view_op(enum ggml_op op) {
    return op == GGML_OP_VIEW ||
           op == GGML_OP_RESHAPE ||
           op == GGML_OP_PERMUTE ||
           op == GGML_OP_TRANSPOSE;
}
```

These ops are not scheduled like ordinary compute nodes. Instead, they are treated as layout reinterpretations of existing storage.

There are two separate aspects to view handling.

#### 1. View ops are skipped during split construction

During graph splitting, view ops are skipped when the scheduler is deciding:

- which nodes define real backend work
- where split boundaries should be created

This prevents the scheduler from creating artificial split boundaries for nodes that do not represent standalone computation.

So in scheduler terms, view ops are metadata transformations, not compute anchors.

#### 2. View tensors inherit backend constraints from `view_src`

When the scheduler tries to infer placement for a tensor, it checks:

- the tensor’s own buffer first
- then `view_src`
- then other graph-placement clues

If `tensor->view_src != NULL`, the scheduler uses the source tensor’s buffer and backend information to determine placement constraints.

This matters in several places:

##### Backend inference from existing storage

If a view source already lives in a backend buffer, the scheduler prefers that backend for the view tensor as well.

##### Pre-allocated view tensors are not movable

If either:

- `tensor->buffer` exists, or
- `tensor->view_src->buffer` exists,

then the tensor is effectively pre-allocated, and the scheduler does not treat it as freely movable.

##### Buffer compatibility checks also follow `view_src`

When checking whether a backend can use a tensor, the scheduler first looks at:

```c
ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
```

So a view tensor inherits buffer compatibility constraints from the storage it aliases.

#### Practical consequence

View ops reduce scheduling complexity in one sense, because they do not create new compute work. But they increase storage-coupling complexity, because the scheduler must preserve aliasing relationships and avoid assigning a view independently of the source storage it depends on.

That is why view tensors influence:

- backend inference
- buffer compatibility
- split input materialization
- allocation reuse safety

### Debug Output

The scheduler contains several debug-oriented fields:

- `debug`
- `debug_realloc`
- `debug_graph_size`
- `debug_prev_graph_size`

These fields are part of the scheduler state rather than part of the public API.

#### Split-level debug logging

When debug logging is enabled, the scheduler can print one summary line per split, including:

- split index
- backend name
- number of inputs
- split input tensor names
- split input tensor sizes

The log shape is roughly:

- `## SPLIT #<n>: <backend> # <n_inputs> inputs`
- followed by the input list

This is the highest-level scheduler debug output and is useful for seeing whether the graph is fragmented across too many backends.

#### Node-level debug logging

When `sched->debug > 1`, the scheduler prints per-node information such as:

- node index
- op name
- tensor name
- tensor size
- assigned backend name
- cause string
- use count
- compute flag
- source tensor names and source backends

This is the most useful debug mode when diagnosing:

- unexpected CPU placement
- backend fragmentation
- excess copies
- missing backend support for an op

#### Cause tracking

The file also contains an internal cause-tracing mechanism:

- `SET_CAUSE(node, ...)`
- `GET_CAUSE(node)`

This is currently compiled out behind `#if 0`, so by default it does not collect detailed reason strings. But the code structure shows that the scheduler is designed to support explicit “why was this node assigned here?” tracing.

The cause strings shown in the code include markers such as:

- `1.dst`
- `1.vsrc`
- `1.inp`
- `1.off`
- `1.wgt...`
- `2.sup`

These correspond to different assignment or propagation reasons across the scheduler passes.

#### Warning path for unsupported buffer-type/op combinations

In non-release builds, the scheduler also logs a warning if no backend can both:

- support the tensor’s buffer type, and
- run the operation

In that case, the warning explains that the weight will need to be copied.

This is an important debug signal because it tells you that:

- backend assignment alone was not enough
- a cross-backend materialization will be required due to buffer incompatibility

#### Reallocation-oriented debug state

The `debug_realloc`, `debug_graph_size`, and `debug_prev_graph_size` fields exist to help investigate graph reallocation behavior.

These fields are not the main scheduling logic, but they are useful when debugging cases where:

- graph structure changed
- reserved memory no longer matches the graph
- scheduler allocation had to be rebuilt unexpectedly

### Backend Priority and Selection

Backend priority is one of the scheduler’s core rules.

Internally, backend priority is defined by backend index:

- lower backend id = higher priority

This is visible both in the scheduler implementation and in the public scheduler construction API.

#### Priority lookup

The helper:

```c
static int ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend)
```

returns the backend index in the scheduler’s ordered backend list. That index is the backend’s priority rank.

So the array order passed to `ggml_backend_sched_new(...)` is not just storage order. It is the scheduler’s preference order.

#### Highest-priority compatible backend from existing storage

When a tensor already has storage, the scheduler uses:

```c
ggml_backend_sched_backend_from_buffer(...)
```

This function:

1. gets the tensor’s buffer, or the source buffer for a view tensor
2. iterates backends from lowest index to highest index
3. returns the first backend that:

   - supports that buffer type
   - supports the operation

So if a tensor is already allocated, the scheduler tries to keep it on the highest-priority backend that can legally use that storage.

#### Input placement rule

Graph input tensors are assigned specially:

```c
cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
```

So the scheduler assumes the last backend is CPU and uses that backend as the default anchor for input tensors.

This is one reason the common backend ordering convention is:

- accelerators first
- CPU last

#### Weight-locality rule

If an operation uses a tensor that behaves like a weight, the scheduler prefers to run that op on the same backend as the weight.

This improves locality by avoiding unnecessary weight copies.

The logic is:

- inspect source tensors
- if a weight source is already tied to a backend
- prefer that backend for the op

#### Offload hint rule

If `op_offload` is enabled, the scheduler can override the simple “run with the weight” rule in one important case:

- the source backend is the last backend, assumed CPU
- the source buffer is host-accessible
- a higher-priority backend supports the op
- that backend’s `offload_op` hook returns true

Then the scheduler prefers the higher-priority backend.

This is a practical rule for cases like:

- weights stored in host memory
- op should still run on an accelerator
- copying is preferable to executing on CPU

#### Buffer compatibility rule

Even if a backend supports an op, that is not enough.

The scheduler also checks buffer compatibility through `ggml_backend_supports_buft(...)`.

This is critical because a backend may be able to execute the math but still be unable to use the tensor’s existing buffer type directly.

So backend selection is always constrained by both:

- **operator support**
- **buffer-type support**

#### Lowest-priority backend behavior during propagation

During adjacent assignment expansion, the scheduler treats the lowest-priority backend specially.

When the currently propagated backend is the last backend, the scheduler stops using it as an aggressive expansion anchor. In practice, this means CPU is treated more as a fallback backend than as the first backend to spread across the graph.

This design helps prevent the graph from collapsing onto CPU too early just because CPU supports most ops.

### Summary

The implementation details that matter most are:

- **hash-based tensor tracking** gives every tensor a stable scheduler id
- **view handling** preserves aliasing and avoids scheduling view ops as standalone compute
- **debug support** exists at split level, node level, and reallocation level
- **backend priority** is encoded directly by backend order, with lower indices meaning higher priority
- **selection is never based on op support alone**; buffer compatibility and existing tensor placement are part of every important decision
