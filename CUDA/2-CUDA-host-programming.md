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
