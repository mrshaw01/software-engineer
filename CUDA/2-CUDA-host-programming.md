# CUDA Host Programming

## CUDA Runtime API

### CUDA Runtime API

- Programming interface that a host program calls
- To utilize CUDA devices and related functionalities
- Many chores are hidden and transparent to user

  - Context management, dynamic kernel linking

### CUDA Driver API

- User can explicitly control context and dynamic linking
- Not as popular as Runtime API due to its complexity

### Core Files

- `cuda_runtime.h`: Runtime API function and variable declarations

  - `cuda.h`: Driver API function and variable declarations

- `libcudart.so`: Runtime API library

  - e.g., `libcuda.so`: Driver API library

- Automatic include and linking if you use `nvcc`

## CUDA Runtime API: Core Features

1. **Error handling**
2. **Device management**
3. **Memory management**
4. **Execution control**
5. Stream management
6. Event management
7. Miscellaneous

- Official documentation:
  [https://docs.nvidia.com/cuda/cuda-runtime-api/index.html](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html)

## Error Handling

- Every CUDA API call returns an error code

```cpp
enum cudaError_t {
    cudaSuccess,
    cudaErrorInvalidValue
};
```

- When the API call succeeds:

  - Returns `cudaSuccess` (= 0)

- Otherwise:

  - Returns an error code other than `cudaSuccess` (!= 0)
  - e.g., `cudaErrorInvalidValue`, `cudaErrorMemoryAllocation`

**We have to check the returned error code for every API call**

## Error Handling APIs

```cpp
cudaError_t cudaGetLastError(void)
```

- Returns the last error from a runtime call

```cpp
const char* cudaGetErrorName(cudaError_t error)
```

- Returns the name of the error in a C-string

```cpp
const char* cudaGetErrorString(cudaError_t error)
```

- Returns the description of the error in a C-string

## Error Handling Macros

- When an error occurs, prints the file name and the line number to standard error and exits the program
- The most popular way of handling errors in CUDA

**Definition of Error Handling Macro**

```cpp
#define CHECK_CUDA(call)                                      \
    do {                                                      \
        cudaError_t status_ = call;                           \
        if (status_ != cudaSuccess) {                         \
            fprintf(stderr, "CUDA error (%s:%d): %s:%s\n",    \
                __FILE__, __LINE__,                           \
                cudaGetErrorName(status_),                    \
                cudaGetErrorString(status_));                 \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)
```

**Usage Example**

```cpp
CHECK_CUDA(cudaMalloc(&ptr, 1024));
```

## Device Management

- CUDA supports APIs for multiple GPUs

  - Device ID: Each device is identified with an integer from 0
  - e.g., 4 devices → 0, 1, 2, and 3

- We can obtain information for each device through API calls

- **Caution**: The number of devices in the system and the number of available devices for CUDA programs can differ

  - `CUDA_VISIBLE_DEVICES` controls the number of available devices

**Example**

```bash
CUDA_VISIBLE_DEVICES=2,3 ./main
```

- Uses device 2 and 3
- Device 2 and 3 become device 0 and 1 in the CUDA program:

  - 2 → 0, 3 → 1

- The program `main` accesses only these two devices

## Device Management APIs

```cpp
cudaError_t cudaGetDeviceCount(int *count)
```

- Sets `*count` to the number of devices

```cpp
cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device)
```

- Sets `*prop` to the properties of the device

**Example: Get Device Properties from All Devices**

```cpp
#define MAX_GPUS 8
int count;
cudaDeviceProp props[MAX_GPUS];
cudaGetDeviceCount(&count);
for (int d = 0; d < count; ++d) {
    cudaGetDeviceProperties(&props[d], d);
    /* Do something with props[d] */
}
```

## cudaDeviceProp Struct

Core member variables:

| Member Variable       | Type     | Description                           |
| --------------------- | -------- | ------------------------------------- |
| `name`                | `char*`  | Device name                           |
| `multiProcessorCount` | `int`    | The number of SMs                     |
| `maxThreadsPerBlock`  | `int`    | Maximum number of threads per block   |
| `totalGlobalMem`      | `size_t` | Size of global memory (device memory) |
| `sharedMemPerBlock`   | `size_t` | **Size of shared memory per block**   |

## Device Management APIs (cont'd)

```cpp
cudaError_t cudaGetDevice(int *device)
```

- Sets `*device` to the current device ID

```cpp
cudaError_t cudaSetDevice(int device)
```

- Sets the current device ID to `device`

```cpp
cudaError_t cudaDeviceSynchronize(void)
```

- Waits until all tasks on the current device are finished
- Host process blocks, which impacts performance
- Call this function **only when really necessary**

**Example: Launch and Wait for a Kernel on Device 0**

```cpp
int device;
cudaGetDevice(&device);
if (device != 0)
    cudaSetDevice(0);
my_kernel<<<...>>>(...);
cudaDeviceSynchronize();
```

## Memory Management

- Host process calls CUDA APIs to allocate device memory

  - Similar to `malloc()` for heap memory
  - Host must reclaim allocated memory (e.g., `cudaFree()`)
  - Example: `cudaMalloc()`, `cudaFree()`

- Device memory and host memory are in **separate address spaces**

  - Host cannot access device memory
  - Kernel cannot access host memory

| Function Type   | Host Memory | Device Memory |
| --------------- | ----------- | ------------- |
| Host function   | O           | X             |
| Kernel function | X           | O             |

- Communication between host and device memory requires CUDA APIs

  - Example: `cudaMemcpy()`

## Device Memory

- Device memory = global memory + constant memory
- Used as an argument for CUDA API functions or kernels
- Data transfer to/from device memory is done using CUDA APIs

**Typical Data Flow**

1. Memory allocation
2. Host to Device (H2D) transfer (`cudaMemcpy`)
3. Argument passed to kernel
4. Kernel accesses memory like array
5. Device to Host (D2H) transfer (`cudaMemcpy`)

## Memory Management APIs

```cpp
cudaError_t cudaMalloc(void **devPtr, size_t size)
```

- Allocates device memory of `size` bytes and sets `*devPtr`
- Uses current device (`cudaGetDevice()`)

```cpp
cudaError_t cudaFree(void *devPtr)
```

- Frees the memory pointed to by `devPtr`

```cpp
cudaError_t cudaMemcpy(void *dst, const void *src, size_t size, cudaMemcpyKind kind)
```

- Copies `size` bytes of data from `src` to `dst`
- `cudaMemcpyKind` specifies direction:

  - `cudaMemcpyHostToDevice`: Host → Device
  - `cudaMemcpyDeviceToHost`: Device → Host

## Usage of Device Memory

```cpp
int *ptr;
int *devPtr;

ptr = (int *) malloc(sizeof(int) * 1024);
cudaMalloc(&devPtr, sizeof(int) * 1024);

/* Do something with ptr */

cudaMemcpy(devPtr, ptr, sizeof(int) * 1024, cudaMemcpyHostToDevice);
my_kernel<<<...>>>(devPtr, ...);
cudaMemcpy(ptr, devPtr, sizeof(int) * 1024, cudaMemcpyDeviceToHost);

cudaFree(devPtr);
free(ptr);
```

## Execution Control - Kernel Launch

```cpp
kernel_name<<<gridDim, blockDim, sharedMem, stream>>>(...params);
```

- **Parameters**:

  - `dim3 gridDim`: Grid dimensions
  - `dim3 blockDim`: Block dimensions
  - `size_t sharedMem` (optional): Shared memory size (default: 0)
  - `cudaStream_t stream` (optional): Stream to use (default: 0)

- **No return value**

  - Check errors with `cudaGetLastError()`

- **Asynchronous**

  - Use `cudaDeviceSynchronize()` to ensure completion

## Thread Hierarchy

- **Thread**: Basic software unit (\~CPU thread)
- **Warp**: Basic hardware unit

  - 1 warp = 32 threads
  - Threads in a warp execute in lockstep

- **Thread Block**: Group of cooperating threads
- **Grid**: Group of thread blocks

## Size of Thread Blocks and Grids

```cpp
kernel_name<<<gridDim, blockDim, sharedMem, stream>>>(...params);
```

```cpp
struct dim3 {
    int x;
    int y;
    int z;
};
```

- Grids and blocks define thread space
- Use `dim3` for both `gridDim` and `blockDim`
- Thread space is 3D (can use 1D or 2D by setting others to 1)
- Example use cases:

  - 1D: Vector addition
  - 2D: Matrix multiplication
  - 3D: Image processing

## Example of Thread Space Configuration

- **1D Case**: `blockDim = {5}`, `gridDim = {6}`

  - 6 blocks × 5 threads = 30 threads

- **2D Case**: `blockDim = {5, 4}`, `gridDim = {3, 3}`

  - 9 blocks, each with 20 threads = 180 threads

## Example of Kernel Launch

1. `blockDim = {128}`, `gridDim = {4}`
   → 128 × 4 = 512 threads

   ```cpp
   dim3 blockDim(128);
   dim3 gridDim(4);
   foo<<<gridDim, blockDim>>>();
   ```

2. `blockDim = {32, 2}`, `gridDim = {2, 8}`
   → (32×2) × (2×8) = 1024 threads

   ```cpp
   dim3 blockDim(32, 2);
   dim3 gridDim(2, 8);
   foo<<<gridDim, blockDim>>>();
   ```

3. `blockDim = {4, 4, 4}`, `gridDim = {2, 4, 8}`
   → (4×4×4) × (2×4×8) = 4096 threads

   ```cpp
   dim3 blockDim(4, 4, 4);
   dim3 gridDim(2, 4, 8);
   foo<<<gridDim, blockDim>>>();
   ```

## How to Set up the Thread Space?

1. **Determine total threads**

   - 1 thread per output element
   - Can loop to process multiple outputs
   - Avoid multiple threads per output (complex reduction)

2. **Choose thread block size**

   - `blockDim`: Number of threads per block
   - Prefer multiples of warp size (32)

3. **Choose grid size**

   - `gridDim`: Number of blocks
   - Total threads = `gridDim * blockDim`

**Example**:

```cpp
blockDim = {16, 16}
gridDim.x = W / 16
gridDim.y = H / 16
```
