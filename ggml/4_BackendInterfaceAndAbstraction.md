# Backend Interface and Abstraction

This document describes GGML's backend abstraction layer, which provides a polymorphic interface for executing tensor operations on heterogeneous hardware. The abstraction layer allows multiple backend implementations such as CPU, CUDA, Vulkan, Metal, SYCL, OpenCL, CANN, and others to be used through the same public API.

## Abstraction Layers

The backend system is organized into five cooperating abstraction layers:

1. **Registry layer**
   Process-wide discovery and enumeration of backend families.

2. **Device layer**
   Representation of concrete hardware devices and their capabilities.

3. **Buffer type layer**
   Description of allocation properties for a device or memory class.

4. **Buffer layer**
   Concrete allocated memory objects used to store tensor data.

5. **Backend layer**
   Execution context used to run graphs, move data, and synchronize work.

## Abstraction Layer Relationships

| Layer       | Type                         | Responsibility                         | Lifetime                               |
| ----------- | ---------------------------- | -------------------------------------- | -------------------------------------- |
| Registry    | `ggml_backend_reg_t`         | Backend discovery and enumeration      | Process-wide singleton                 |
| Device      | `ggml_backend_dev_t`         | Hardware representation and properties | Static, per-device                     |
| Buffer Type | `ggml_backend_buffer_type_t` | Memory allocation properties           | Static, per-device or per-memory-class |
| Buffer      | `ggml_backend_buffer_t`      | Allocated memory instance              | User-managed                           |
| Backend     | `ggml_backend_t`             | Execution stream or execution context  | User-managed                           |

## Backend Registry Interface

### Registry Interface Methods

The registry layer represents a backend family such as CPU, CUDA, Vulkan, or Metal. A registry entry is identified by `ggml_backend_reg_t` and provides the methods needed for backend discovery and enumeration.

Core registry-facing APIs:

- `ggml_backend_reg_name(reg)`
  Returns the backend family name.

- `ggml_backend_reg_dev_count(reg)`
  Returns the number of devices exposed by that backend family.

- `ggml_backend_reg_dev_get(reg, index)`
  Returns the device at a given index.

- `ggml_backend_reg_get_proc_address(reg, name)`
  Returns a backend-specific extension symbol by name.

The extension lookup mechanism is important because some backend features are not part of the fixed common API. GGML defines several common extension function types that may be retrieved through `ggml_backend_reg_get_proc_address(...)`, including:

- `ggml_backend_split_buffer_type_t`
- `ggml_backend_set_n_threads_t`
- `ggml_backend_dev_get_extra_bufts_t`
- `ggml_backend_set_abort_callback_t`
- `ggml_backend_get_features_t`

This allows the common backend API to stay small while still letting specific backends expose extra functionality.

### Global Registry Access

GGML also provides process-wide registry management and enumeration APIs.

Backend-family registration:

- `ggml_backend_register(reg)`
- `ggml_backend_device_register(device)`

Registry enumeration:

- `ggml_backend_reg_count()`
- `ggml_backend_reg_get(index)`
- `ggml_backend_reg_by_name(name)`

Device enumeration:

- `ggml_backend_dev_count()`
- `ggml_backend_dev_get(index)`
- `ggml_backend_dev_by_name(name)`
- `ggml_backend_dev_by_type(type)`

Convenience backend initialization:

- `ggml_backend_init_by_name(name, params)`
- `ggml_backend_init_by_type(type, params)`
- `ggml_backend_init_best()`

Dynamic loading:

- `ggml_backend_load(path)`
- `ggml_backend_unload(reg)`
- `ggml_backend_load_all()`
- `ggml_backend_load_all_from_path(dir_path)`

These functions make the registry layer the entry point for backend discovery, backend loading, and device lookup.

## Backend Device Interface

### Device Types and Capabilities

A device is represented by `ggml_backend_dev_t`. It models one concrete hardware target and serves as the bridge between backend discovery and backend execution.

GGML defines four device categories:

- `GGML_BACKEND_DEVICE_TYPE_CPU`
  CPU device using system memory

- `GGML_BACKEND_DEVICE_TYPE_GPU`
  GPU device using dedicated memory

- `GGML_BACKEND_DEVICE_TYPE_IGPU`
  Integrated GPU device using host memory

- `GGML_BACKEND_DEVICE_TYPE_ACCEL`
  Accelerator intended to be used together with the CPU backend

This device classification is used by enumeration helpers such as `ggml_backend_dev_by_type(...)` and by property queries exposed through the device interface.

### Device Capabilities Structure

Device capabilities are described by `struct ggml_backend_dev_caps`:

```c
struct ggml_backend_dev_caps {
    bool async;                // asynchronous operations
    bool host_buffer;          // pinned host buffer support
    bool buffer_from_host_ptr; // wrapping an existing host pointer
    bool events;               // event synchronization support
};
```

## Backend Buffer Type Interface

### Buffer Type Interface Definition

A backend buffer type describes how memory should be allocated for a specific device or memory class. It is represented publicly as `ggml_backend_buffer_type_t`, while the actual implementation is provided by a backend-specific interface table behind that opaque handle.

Conceptually, a buffer type provides:

- allocation of a backend buffer of a given size
- alignment requirements
- maximum supported allocation size
- tensor-specific allocation size calculation
- host accessibility information
- association with a device

This separation allows one device to expose multiple allocation strategies, such as:

- a default device-local buffer type
- a host buffer type
- other specialized memory types

### Public Buffer Type API

The main public APIs for buffer types are:

- `ggml_backend_buft_name(buft)`
  Returns the buffer type name.

- `ggml_backend_buft_alloc_buffer(buft, size)`
  Allocates a backend buffer from that buffer type.

- `ggml_backend_buft_get_alignment(buft)`
  Returns the alignment required by this memory type.

- `ggml_backend_buft_get_max_size(buft)`
  Returns the maximum supported allocation size.

- `ggml_backend_buft_get_alloc_size(buft, tensor)`
  Returns the allocation size required for a tensor in this memory type.

- `ggml_backend_buft_supports_backend(buft, backend)`
  Returns whether the buffer type can be used with the given backend.

- `ggml_backend_buft_supports_buft(buft, other)`
  Returns whether this buffer type is compatible with another buffer type.

- `ggml_backend_buft_is_host(buft)`
  Returns whether this buffer type is host-accessible.

- `ggml_backend_buft_get_device(buft)`
  Returns the device that owns this buffer type.

These APIs make the buffer type layer the main place where GGML asks, “how should memory be allocated for this device?”

## Backend Buffer Interface

### Buffer Interface Definition

A backend buffer is a concrete allocated memory object represented by `ggml_backend_buffer_t`. Internally, it is backed by a backend-specific interface table plus a backend-owned context pointer.

A buffer can provide operations such as:

- get base pointer
- initialize tensor metadata
- write tensor contents
- read tensor contents
- memset tensor contents
- copy tensor contents
- clear the full buffer
- reset buffer state
- free the buffer

Not every backend implements every optional operation. GGML uses wrapper functions so callers interact with a consistent public API even when the underlying backend capabilities differ.

### Public Buffer API

The main public APIs for buffers are:

- `ggml_backend_buffer_name(buffer)`
  Returns the buffer name.

- `ggml_backend_buffer_free(buffer)`
  Releases the buffer.

- `ggml_backend_buffer_get_base(buffer)`
  Returns the base pointer when the buffer exposes one.

- `ggml_backend_buffer_get_size(buffer)`
  Returns the allocated size.

- `ggml_backend_buffer_get_alignment(buffer)`
  Returns the buffer alignment.

- `ggml_backend_buffer_get_max_size(buffer)`
  Returns the maximum size supported by this buffer implementation.

- `ggml_backend_buffer_get_alloc_size(buffer, tensor)`
  Returns the required allocation size for a tensor in this buffer.

- `ggml_backend_buffer_get_type(buffer)`
  Returns the owning buffer type.

- `ggml_backend_buffer_get_device(buffer)`
  Returns the associated device.

- `ggml_backend_buffer_is_host(buffer)`
  Returns whether the buffer is host-accessible.

- `ggml_backend_buffer_clear(buffer, value)`
  Fills the entire buffer with one byte value.

- `ggml_backend_buffer_reset(buffer)`
  Resets backend-specific buffer state when supported.

### Buffer Usage Hints

GGML defines three usage hints for backend buffers:

- `GGML_BACKEND_BUFFER_USAGE_ANY`
- `GGML_BACKEND_BUFFER_USAGE_WEIGHTS`
- `GGML_BACKEND_BUFFER_USAGE_COMPUTE`

These hints let a backend distinguish between:

- long-lived read-mostly buffers such as model weights
- temporary reusable compute buffers
- general-purpose buffers

The public APIs are:

- `ggml_backend_buffer_set_usage(buffer, usage)`
- `ggml_backend_buffer_get_usage(buffer)`

The graph allocator uses compute usage for temporary graph buffers, while long-lived model tensors may be placed in weight-oriented buffers.

### Tensor Data Transfer

GGML provides tensor-level read and write helpers so callers do not need to reason about raw backend memory layouts directly.

Main APIs:

- `ggml_backend_tensor_set(tensor, data, offset, size)`
  Copies host data into tensor storage.

- `ggml_backend_tensor_get(tensor, data, offset, size)`
  Copies tensor data back to host memory.

- `ggml_backend_tensor_memset(tensor, value, offset, size)`
  Fills part of a tensor with one byte value.

- `ggml_backend_tensor_copy(src, dst)`
  Copies tensor contents from one tensor to another.

- `ggml_backend_tensor_copy_async(backend, src, dst)`
  Starts an asynchronous tensor copy when the backend supports it.

These helpers let GGML move tensor data across host and device memory while preserving backend-specific layout rules.

## Backend Interface (Execution Streams)

A backend execution context is represented by `ggml_backend_t`. This is the object that answers, “where and how does computation run?”

Conceptually, a backend owns:

- an execution stream or command context
- buffer allocation methods
- graph planning and execution methods
- synchronization behavior
- backend-specific offload decisions

### Core Backend API

The main public backend APIs include:

- `ggml_backend_name(backend)`
  Returns the backend name.

- `ggml_backend_free(backend)`
  Destroys the backend execution context.

- `ggml_backend_get_device(backend)`
  Returns the device that created the backend.

- `ggml_backend_alloc_buffer(backend, size)`
  Allocates a buffer using the backend’s default buffer type.

- `ggml_backend_get_default_buffer_type(backend)`
  Returns the default buffer type for this backend.

- `ggml_backend_graph_plan_create(backend, graph)`
  Creates a backend-specific graph execution plan.

- `ggml_backend_graph_plan_free(backend, plan)`
  Destroys a graph plan.

- `ggml_backend_graph_plan_compute(backend, plan)`
  Executes a prepared graph plan.

- `ggml_backend_graph_compute(backend, graph)`
  Executes a graph directly.

- `ggml_backend_graph_compute_async(backend, graph)`
  Starts asynchronous graph execution.

- `ggml_backend_synchronize(backend)`
  Waits for queued work to complete.

### Tensor Transfer Through the Backend

The backend also exposes asynchronous transfer helpers:

- `ggml_backend_tensor_set_async(backend, tensor, data, offset, size)`
- `ggml_backend_tensor_get_async(backend, tensor, data, offset, size)`

These APIs allow data movement and execution to share the same backend execution context.

### Backend Support Queries

GGML also lets higher layers query backend capabilities at the execution level:

- `ggml_backend_supports_op(backend, op)`
  Returns whether the backend can execute an operation.

- `ggml_backend_supports_buft(backend, buft)`
  Returns whether the backend supports a given buffer type.

- `ggml_backend_offload_op(backend, op)`
  Returns whether an operation should be offloaded to this backend.

These methods are used by higher-level scheduling and placement logic.

### Events

Backends can also participate in event-based synchronization through `ggml_backend_event_t`.

Main event APIs:

- `ggml_backend_event_new(device)`
- `ggml_backend_event_free(event)`
- `ggml_backend_event_record(event, backend)`
- `ggml_backend_event_wait(event, backend)`
- `ggml_backend_event_synchronize(event)`

This allows coordination across execution contexts without exposing backend-specific event types directly.

### Mental Model

A useful mental model is:

- **buffer type** decides how memory can be allocated
- **buffer** is the actual allocated memory
- **backend** is the execution context that uses those buffers to run graphs and move data

That split is the core of GGML’s backend abstraction.

## Polymorphic Interface Pattern

### Interface Structure Pattern

GGML uses a type-erased handle plus interface-table pattern across the backend layer.

Public code works with opaque handle types such as:

- `ggml_backend_reg_t`
- `ggml_backend_dev_t`
- `ggml_backend_buffer_type_t`
- `ggml_backend_buffer_t`
- `ggml_backend_t`
- `ggml_backend_event_t`

Each handle hides a backend-specific implementation object. Internally, the object stores:

- a pointer to an interface table
- a backend-specific context pointer
- metadata such as name, device, size, or usage depending on the object kind

This gives GGML runtime polymorphism in plain C-style APIs:

- the public API stays stable
- each backend provides its own implementation
- optional methods can exist without changing the common interface

### Example: Buffer Type Polymorphic Dispatch

A buffer type is exposed as `ggml_backend_buffer_type_t`, but allocation is dispatched through the backend-specific implementation attached to that buffer type.

The public wrapper:

- `ggml_backend_buft_alloc_buffer(buft, size)`

calls the allocation method implemented by that backend’s buffer-type interface.

The same pattern applies to other buffer-type operations:

- `ggml_backend_buft_name(...)`
- `ggml_backend_buft_get_alignment(...)`
- `ggml_backend_buft_get_max_size(...)`
- `ggml_backend_buft_get_alloc_size(...)`
- `ggml_backend_buft_is_host(...)`
- `ggml_backend_buft_get_device(...)`

So user code sees one API, while CPU, CUDA, Vulkan, Metal, and other backends provide different implementations behind it.

### Optional Interface Methods

Some interface methods are required for all implementations, while others are optional.

Typical optional methods include:

- tensor initialization hooks
- tensor memset hooks
- direct tensor copy hooks
- reset hooks
- base-pointer access for non-host-visible buffers
- backend-specific extension entry points

GGML handles missing optional methods through wrapper logic or fallback behavior. This lets backends expose only the operations that make sense for their memory model and execution model.

## Device Discovery and Enumeration

### Device Enumeration Functions

GGML provides process-wide enumeration for both backend families and concrete devices.

Backend-family enumeration:

- `ggml_backend_reg_count()`
- `ggml_backend_reg_get(index)`
- `ggml_backend_reg_by_name(name)`

Device enumeration:

- `ggml_backend_dev_count()`
- `ggml_backend_dev_get(index)`
- `ggml_backend_dev_by_name(name)`
- `ggml_backend_dev_by_type(type)`

Per-registry device enumeration:

- `ggml_backend_reg_dev_count(reg)`
- `ggml_backend_reg_dev_get(reg, index)`

These functions allow code to:

- discover which backend families are registered
- inspect which devices each backend exposes
- select devices by name or by device category

### Backend Initialization Helpers

GGML provides direct helpers to create an execution backend from discovered devices:

- `ggml_backend_dev_init(device, params)`
- `ggml_backend_init_by_name(name, params)`
- `ggml_backend_init_by_type(type, params)`
- `ggml_backend_init_best()`

These are convenience layers over the device model.

Typical usage patterns are:

- initialize a backend from a known device
- initialize the first device with a given name
- initialize the first device of a given type
- initialize the best available backend, preferring GPU and falling back to CPU

### Dynamic Backend Loading

GGML supports dynamic backend discovery through shared libraries.

The public APIs are:

- `ggml_backend_load(path)`
- `ggml_backend_unload(reg)`
- `ggml_backend_load_all()`
- `ggml_backend_load_all_from_path(dir_path)`

This allows the process to:

- load a backend implementation from a shared library
- register it into the global registry
- enumerate its devices like any statically registered backend
- unload it later if it was dynamically loaded

This design keeps the runtime discovery path consistent whether a backend is built in or loaded from disk.

## Events and Synchronization

### Event API

GGML exposes backend events through the opaque handle `ggml_backend_event_t`.

The public event APIs are:

- `ggml_backend_event_new(device)`
- `ggml_backend_event_free(event)`
- `ggml_backend_event_record(event, backend)`
- `ggml_backend_event_synchronize(event)`
- `ggml_backend_event_wait(backend, event)`

These APIs provide a backend-neutral synchronization primitive for asynchronous execution.

### Event Usage Pattern

A typical event flow is:

1. create an event for a device
2. submit work to a backend on that device
3. record the event after the queued work
4. either:
   - wait for the event on the host with `ggml_backend_event_synchronize(...)`, or
   - make another backend wait on that event with `ggml_backend_event_wait(...)`

This allows GGML to coordinate asynchronous execution and data movement without exposing backend-specific event objects.

### Device-Level Event Support

Event support is not assumed for every device. It is reported through the device capability structure:

- `struct ggml_backend_dev_caps`
  - `async`
  - `host_buffer`
  - `buffer_from_host_ptr`
  - `events`

The `events` flag tells higher layers whether event-based synchronization is supported on that device.

This means code can:

- discover a device
- inspect its capabilities
- enable event-based execution only when the device supports it
- fall back to backend synchronization when needed

## Multi-Buffer Support

### Multi-Buffer Creation

GGML supports a logical buffer that is composed from multiple physical backend buffers. In the backend implementation, this is created with:

- `ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers)`

The multi-buffer stores:

- the number of child buffers
- the child buffer array
- the total logical size, computed as the sum of child-buffer sizes

This is useful when one logical allocation spans multiple chunks rather than one contiguous physical buffer.

### Multi-Buffer Interface

The multi-buffer implementation intentionally exposes only a minimal interface.

Implemented behavior:

- `free_buffer`
- `clear`

Not implemented in the multi-buffer interface:

- `get_base`
- `init_tensor`
- `memset_tensor`
- `set_tensor`
- `get_tensor`
- `cpy_tensor`
- `reset`

So a multi-buffer behaves like a coordination wrapper over several backend buffers, not like one normal flat host-visible allocation.

### Detection and Usage

The backend implementation includes an internal helper:

- `ggml_backend_buffer_is_multi_buffer(buffer)`

It identifies multi-buffers by checking the installed buffer interface.

Usage hints are propagated automatically:

- `ggml_backend_buffer_set_usage(buffer, usage)` updates `buffer->usage`
- if the buffer is a multi-buffer, it forwards that usage to all child buffers

This makes multi-buffer behavior transparent for higher-level code that only sets usage on the logical buffer.

## Tensor Allocation Interface

### Tensor Allocation Function

For simple linear allocation from a pre-existing backend buffer, GGML provides `ggml_tallocr`.

Public pieces:

- `struct ggml_tallocr`
- `ggml_tallocr_new(buffer)`
- `ggml_tallocr_alloc(&talloc, tensor)`

`ggml_tallocr_alloc(...)` works like this:

1. compute tensor allocation size with `ggml_backend_buffer_get_alloc_size(...)`
2. pad the size to allocator alignment
3. check that enough capacity remains in the buffer
4. compute `addr = base + offset`
5. advance the allocator offset
6. attach the tensor to the buffer at that address

This is a sequential allocator. It is simple and fast, but it does not do lifetime-based reuse.

GGML also provides context-wide helpers that allocate all tensors in a context:

- `ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, buft)`
- `ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft)`
- `ggml_backend_alloc_ctx_tensors(ctx, backend)`

### View Tensor Initialization

View tensors are handled specially during backend allocation.

When a tensor has `view_src != NULL`, GGML does not allocate independent storage for it. Instead, during graph allocation, if the source tensor already has backend-managed storage, GGML initializes the view with:

- `ggml_backend_view_init(tensor)`

That means view tensors inherit storage from their source tensor and only adjust metadata such as offset and layout, instead of creating a new payload allocation.

### CPU Buffer Type

For CPU execution, the public CPU backend header exposes:

- `ggml_backend_cpu_init()`
- `ggml_backend_is_cpu(...)`
- `ggml_backend_cpu_set_n_threads(...)`
- `ggml_backend_cpu_set_threadpool(...)`
- `ggml_backend_cpu_set_abort_callback(...)`
- `ggml_backend_cpu_set_use_ref(...)`
- `ggml_backend_cpu_reg()`

The common public path to obtain a CPU buffer type is through the generic backend and device APIs:

- initialize a CPU backend
- call `ggml_backend_get_default_buffer_type(backend)`

or

- discover a CPU device
- call `ggml_backend_dev_buffer_type(device)`

So the CPU buffer type participates in the same generic backend-buffer abstraction as every other backend.

## Buffer Usage and Operation Support

### Operation Support Query

GGML exposes operation-support checks at both the backend and device levels:

- `ggml_backend_supports_op(backend, op)`
- `ggml_backend_dev_supports_op(device, op)`

At the backend layer, the call simply forwards to the backend’s device. This keeps support checks device-centric while still exposing a convenient backend-level API.

### Buffer Type Compatibility

GGML does not expose a standalone “buffer type can work with buffer type” compatibility API in the public header. Instead, compatibility is expressed through backend or device support queries:

- `ggml_backend_supports_buft(backend, buft)`
- `ggml_backend_dev_supports_buft(device, buft)`

In practice, this means compatibility is asked from the execution side:

- can this backend use this buffer type?
- can this device use this buffer type?

rather than from the buffer type itself.

### Offload Operation Hint

GGML also exposes an offload hint:

- `ggml_backend_offload_op(backend, op)`
- `ggml_backend_dev_offload_op(device, op)`

At the backend layer, this is forwarded to the device. At the device layer, the offload callback is optional; if a device does not provide one, GGML returns `false`.

So `offload_op` is a hint mechanism, not a mandatory placement rule.

### Usage in Scheduler

The scheduler uses both operation support and buffer compatibility when assigning tensors and ops to backends.

Important scheduler-facing APIs include:

- `ggml_backend_sched_new(...)`
- `ggml_backend_sched_set_tensor_backend(...)`
- `ggml_backend_sched_get_tensor_backend(...)`
- `ggml_backend_sched_split_graph(...)`
- `ggml_backend_sched_alloc_graph(...)`
- `ggml_backend_sched_graph_compute(...)`

The header’s scheduler example also shows an important policy:

- tensors allocated in a buffer marked with `GGML_BACKEND_BUFFER_USAGE_WEIGHTS` are assigned preferably to run on the same backend as that buffer

Inside the scheduler implementation, buffer compatibility is checked by looking at:

- the tensor’s existing buffer, if already allocated
- otherwise the buffer type associated with its assigned backend
- then calling `ggml_backend_supports_buft(...)` for the candidate backend

Operation support is checked with `ggml_backend_supports_op(...)`.

So scheduler placement is driven by three main signals:

1. whether the backend supports the operation
2. where the tensor is already allocated or assigned
3. whether the candidate backend supports the relevant buffer type
