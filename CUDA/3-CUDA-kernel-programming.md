# CUDA Kernel Programming

## CUDA Applications

- A CUDA application consists of:

  - **Host program** (runs on CPU)
  - **CUDA kernels** (run on GPU)

```cpp
// Host Program
int main() {
    ...
    cudaMalloc(...);
    cudaMemcpy(...);
    kernel_1<<<...>>>();
    kernel_2<<<...>>>();
    cudaMemcpy(...);
    ...
}

// CUDA Kernels
__global__ void kernel_1(...) {
    ...
}
__global__ void kernel_2(...) {
    ...
}
```

## How to Define a Kernel

- **Kernel specifier**: `__global__`

  - Indicates the function is a kernel function.

- **Return type** must be `void`.
- **Kernel functions** can call other `__device__` functions.

```cpp
__global__ void square(int *input, int *output) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    output[id] = input[id] * input[id];
}
```

## Function Space Specifier

- Defines where the function resides and who can call it.
- CUDA supports three specifiers:

### `__host__`

- Runs on host processor.
- Callable by host functions.
- Default for ordinary C/C++ functions.

### `__global__`

- A kernel function.
- Runs on the device (GPU).
- Callable by host function (or rarely by a kernel function in dynamic parallelism).

### `__device__`

- Runs on the device.
- Callable only by device/kernel functions.

## Function Calls in CUDA

```cpp
__device__ float foo(float A, float B) {
    ...
}

__device__ int bar(...) {
    int result;
    ... foo(A, B); ...
    return result;
}

__global__ void your_kernel(...) {
    ... bar(...) ...
}

// Optional: __host__
int main() {
    ... your_kernel<<<...>>>(...) ...
}
```

- Device functions (`__device__`) can call other device functions.
- Kernel functions (`__global__`) can call device functions.
- Host functions can launch kernel functions.

## Memory Space Specifier

- Specifies the memory space where a variable resides.
- CUDA supports three memory spaces:

### `__host__`

- On CPU memory.
- Accessed by host functions.
- Dynamically allocated with `malloc()` or `cudaMallocHost()`.

### `__device__`

- On GPU memory.
- Accessed by kernel/device functions.
- Dynamically allocated with `cudaMalloc()`.

### `__constant__`

- On GPU constant memory.
- Accessed by kernel/device functions.
- Cannot be dynamically allocated.

## Memory Space Specifiers Example

```cpp
__host__ float foo[128];     // Host memory
__device__ float bar[128];   // Device memory
__constant__ float baz[128]; // Constant memory

__global__ void kernel(...) {
    float c = bar[0] + baz[1]; // Access device memory
}

void function(...) {
    float d = foo[0];          // Access host memory
}
```

## Summary of CUDA Function/Memory Specifiers

### Function Space Specifier

| Specifier    | Caller                          | Runs on        |
| ------------ | ------------------------------- | -------------- |
| `__host__`   | Host function                   | Host processor |
| `__global__` | Host function \*(rarely device) | GPU            |
| `__device__` | Kernel/device function          | GPU            |

### Memory Space Specifier

| Specifier      | Resides in           | Dynamically Allocatable? |
| -------------- | -------------------- | ------------------------ |
| `__host__`     | Host processor (CPU) | Yes (`malloc()`)         |
| `__device__`   | Devices (GPU)        | Yes (`cudaMalloc()`)     |
| `__constant__` | Devices (GPU)        | No                       |

## How to Write Kernel Functions

- CUDA kernel = C/C++ + CUDA-specific extensions

### CUDA Extensions:

1. Built-in thread coordinate variables
2. Other CUDA extensions:

   - Vector types
   - Struct types
   - Built-in device functions

## Thread Space in Kernels

### Thread Hierarchy

- CUDA allows a **3-dimensional thread hierarchy**:

  - `typedef struct {int x, y, z;} dim3;`
  - Used for `gridDim`, `blockDim`, etc.

- Thread space is a 3D grid:

  - Each thread has up to 3 coordinates: `(x, y, z)`

#### Built-in Thread Coordinate Variables

- `dim3 threadIdx, blockIdx, blockDim, gridDim`
- Used to uniquely identify a thread
- Accessible in kernel/device functions

#### Thread and Grid Structure

Each thread belongs to:

1. A **thread block** (identified by `blockIdx`)
2. A **grid** of blocks (identified by `gridDim`)

| Variable    | Description                    |
| ----------- | ------------------------------ |
| `threadIdx` | Thread’s position in its block |
| `blockIdx`  | Block’s position in the grid   |
| `blockDim`  | Dimensions of a block          |
| `gridDim`   | Dimensions of the grid         |

#### Global Thread ID (3D)

To compute the global thread index:

```cpp
int globalX = blockIdx.x * blockDim.x + threadIdx.x;
int globalY = blockIdx.y * blockDim.y + threadIdx.y;
int globalZ = blockIdx.z * blockDim.z + threadIdx.z;
```

### Visual Example: 2D Thread Space

- `blockDim = (5, 4)`
- `gridDim = (3, 3)`
- Total threads = `(15, 12)` (i.e., `5×3`, `4×3`)

#### For the red thread:

- `threadIdx = (3, 1)`
- `blockIdx = (1, 1)`
- **Global ID**: `(8, 5)`

  - `8 = 5 × 1 + 3`
  - `5 = 4 × 1 + 1`

### Thread Space in 1D vs 2D

#### 1-D Case:

- `blockDim = {5}`
- `gridDim = {6}`

Thread 17 is located at:

```
block 2, thread index 2
17 = 5 * 3 + 2
```

#### 2-D Case:

- `blockDim = (5, 4)`
- `gridDim = (3, 3)`

Examples:

- Block(0, 0) has thread(0, 0)
- Block(2, 1) has thread(2, 1) and thread(2, 2)

## How to Figure out Thread Indices

- Most important step: **figure out the global ID**
- Threads typically compute output elements independently (except in reductions)

### Accessing Input Data

Use global indices `i`, `j`, `k`:

```cpp
__global__ void foo3d() {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;
    ...
}
```

### Vector Addition Example

```cpp
__global__ void vecadd_kernel(int N, int *a, int *b, int *c) {
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx > N) return;
    c[tidx] = a[tidx] + b[tidx];
}

int main() {
    ...
    vecadd_kernel<<<2, 32>>>(N, a, b, c);
    ...
    return 0;
}
```

### Thread ID Mapping Table

For `<<<2, 32>>>`:

#### blockIdx.x = 0:

| Global ID | threadIdx.x | blockIdx.x |
| --------- | ----------- | ---------- |
| 0         | 0           | 0          |
| 1         | 1           | 0          |
| ...       | ...         | ...        |
| 31        | 31          | 0          |

#### blockIdx.x = 1:

| Global ID | threadIdx.x | blockIdx.x |
| --------- | ----------- | ---------- |
| 32        | 0           | 1          |
| 33        | 1           | 1          |
| ...       | ...         | ...        |
| 63        | 31          | 1          |

## Scalar Type

- CUDA supports scalar types similar to ordinary C/C++.

| Type                       | Description                                          |
| -------------------------- | ---------------------------------------------------- |
| `bool`                     | true or false                                        |
| `char`                     | Signed 8-bit integer                                 |
| `unsigned char`, `uchar`   | Unsigned 8-bit integer                               |
| `short`                    | Signed 16-bit integer                                |
| `unsigned short`, `ushort` | Unsigned 16-bit integer                              |
| `int`                      | Signed 32-bit integer                                |
| `unsigned int`, `uint`     | Unsigned 32-bit integer                              |
| `long`                     | Signed 64-bit integer                                |
| `unsigned long`, `ulong`   | Unsigned 64-bit integer                              |
| `float`                    | IEEE754 32-bit floating point                        |
| `double`                   | IEEE754 64-bit floating point                        |
| `half`                     | IEEE754-2008 16-bit floating point                   |
| `size_t`                   | Type of `sizeof`; 64-bit or 32-bit depending on arch |
| `void`                     | Void type                                            |

## Vector Type

- CUDA supports vector types.
- Format: `typeN`, where:

  - `type` = `char`, `uchar`, `short`, `ushort`, `int`, `uint`, `long`, `ulong`, `longlong`, `ulonglong`, `float`, `double`
  - `N` = 1, 2, 3, or 4

| CUDA Vector Type |
| ---------------- |
| `charn`          |
| `ucharn`         |
| `shortn`         |
| `ushortn`        |
| `intn`           |
| `uintn`          |
| `longn`          |
| `ulongn`         |
| `floatn`         |
| `doublen`        |

### Vector Variables and Pointers

```cpp
float4 a, b, c;
float4 *p;
```

### Core Vector Operations

| Operation Type   | Example                                            |
| ---------------- | -------------------------------------------------- |
| Creation         | `float4 make_float4(x, y, z, w);`                  |
| Assignment       | `a = b;`, `a = p[i];`, `p[i] = a;`                 |
| Vector Literals  | `a = {1.0f, 2.0f, 3.0f, 4.0f};`                    |
| Component Access | `a.x = t;`, `t = a.x;` (components: x → y → z → w) |
| Arithmetics      | `c.x = a.x + b.x;`, `c.y = a.y + b.y;`, ...        |

## Usage of Vector Type

**Copy Kernel Using Vector Types (e.g., `int4`)**

```cpp
__global__ void device_copy_vector4_kernel(int* d_in, int* d_out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < N/4; i += blockDim.x * gridDim.x) {
        reinterpret_cast<int4*>(d_out)[i] = reinterpret_cast<int4*>(d_in)[i];
    }
}
```

## Struct Type

- CUDA supports C/C++ structs and classes.
- **Recommendation**: use simple structs.
- To call a member function from a kernel, use `__device__` specifier.
- Use both `__host__` and `__device__` for dual-usage functions.

```cpp
struct S {
    char c; int i;
    __host__ __device__ void setCharacter(char c) { this->c = c; }
    __host__ __device__ void setInteger(int i) { this->i = i; }
};

__global__ void kernel_function(S *var, char c) {
    var->setCharacter(c);
    ...
}

void host_function(S *var, int i) {
    var->setInteger(i);
    ...
}
```

## Built-in Device Functions

CUDA provides built-in functions callable from kernel/device functions:

- `printf`
- Math functions
- Atomic functions
- Synchronization functions
- Memory fences
- Warp functions
- Image functions
- No explicit include is needed

## Built-in Device Functions: `printf`

- `printf` is available from CUDA 4.0
- Useful for debugging
- **Caution**:

  - No guaranteed order of output
  - Possible data races
  - Can hinder compiler optimization

```cpp
__global__ void kernel() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from CUDA thread %d\n", tid);
}
```

## Built-in Device Functions: Math Functions

- Math functions from `math.h` are usable in kernels.
- Be mindful of type correctness.

```cpp
__global__ void diff(int *in1, int *in2, int *out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = fabs((double)(in1[idx] - in2[idx]));
}
```
