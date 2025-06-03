## MPI Overview

- **Message Passing Interface (MPI)**

  - Provides a standard for writing message-passing programs
  - Portable, efficient, flexible

- **Target Systems**

  - Designed for parallel computers, clusters, and heterogeneous networks
  - Provides access to advanced parallel hardware for end users, library writers, and tool developers

- **Specification of Message-Passing Libraries**

  - Defines what such a library should be
  - Includes an API for such libraries
  - MPI standard:

    - [http://www.mpi-forum.org](http://www.mpi-forum.org)
    - [http://www-unix.mcs.anl.gov/mpi/](http://www-unix.mcs.anl.gov/mpi/)

- **Language Bindings**

  - C, C++, and FORTRAN

## MPI Timeline and Features

- **Timeline**

  - 1994: MPI 1.0
  - 1995: MPI 1.1 – Revision and clarification to MPI 1.0 (Major milestone, C, FORTRAN)
  - 1997: MPI 1.2 – Corrections and clarifications to MPI 1.1
  - 1997: MPI 2.0 – Major extension to MPI 1.1 (C++, C, FORTRAN)

    - Partially implemented in most libraries
    - A few full implementations (e.g., ANL MPICH2)

  - 2012: MPI 3.0
  - 2015: MPI 3.1
  - 2021: MPI 4.0

- **Key Features**

  - De facto standard for parallel computing (industry-standard)
  - Portability across virtually all HPC platforms
  - High performance and scalability
  - Rich functionality:

    - MPI 1.1: 125 functions
    - MPI 2: 152 functions

## MPI Programming Model

- **Architecture Support**

  - Initially designed for distributed memory architectures
  - Supports SMPs combined over networks (hybrid distributed/shared memory)
  - Implementations adapted for both distributed and shared memory systems

- **Hardware Compatibility**

  - Runs on virtually any platform: Distributed Memory, Shared Memory, Hybrid
  - Programming model remains distributed regardless of the physical architecture

- **Parallelism**

  - All parallelism is explicit
  - The programmer must identify and implement parallelism using MPI constructs

- **Communication Model**

  - Message passing programming model
  - Each process has a separate address space
  - Inter-process communication involves:

    - Synchronization
    - Data transfer between address spaces

  - Communication is cooperative:

    - Data is sent by one process, received by another
    - Point-to-point communications
    - Receiver must explicitly participate in memory changes

- **Process Model**

  - Number of CPUs is statically determined

    - MPI 2 supports dynamic process creation (limited availability)
    - Typically, one-to-one mapping of MPI processes to processors

  - **SPMD (Single Program, Multiple Data)**

    - All processes run the same program but act on different data

  - **MPMD (Multiple Program, Multiple Data)**

    - Each process can run a different program
    - MPI supports MPMD launch mode

## MPI Program Structure

- MPI include file
- MPI environment initialization
- Message passing calls
- MPI environment termination

## MPI Hello World

```c
#include "mpi.h"
#include <stdio.h>

int main(int argc, char **argv)
{
    int my_rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    printf("Hello, I am %d of %d\n", my_rank, size);
    MPI_Finalize();
    return 0;
}
```

Output:

```
Hello, I am 0 of 4
Hello, I am 2 of 4
Hello, I am 1 of 4
Hello, I am 3 of 4
```

## Running MPI Programs

- `mpiexec <args>` is part of MPI-2, recommended but not required
- Starting an MPI program depends on the MPI implementation

Example:

```sh
mpicc -o hello hello.c
mpirun -np 4 ./hello
```

- Runs 4 processes (not necessarily on 4 processors)

## MPI Naming Conventions

- All names have `MPI_` prefix
- In C: mixed uppercase/lowercase

  ```c
  ierr = MPI_Xxx(arg1, arg2, ...);
  ```

- MPI constants are all uppercase

  - `MPI_COMM_WORLD`, `MPI_SUCCESS`, `MPI_DOUBLE`, `MPI_SUM`, ...

## Error Handling

- An error causes all processes to abort

  - User can make routines return error code instead
  - In C++, exceptions are thrown (MPI-2)

```c
ierr = MPI_...;
if (ierr == MPI_SUCCESS) {
    // everything is fine
}
```

## Environment

- Elements of an application:

  - $N$ processes numbered from 0 to $N - 1$
  - Communication paths between processes

    - A _communicator_ defines the group of processes that can communicate
    - Most MPI routines require a `communicator` argument

- All processes in the computation form the communicator `MPI_COMM_WORLD`

  - Predefined by MPI and available anywhere
  - Subgroups/subcommunicators can be created within `MPI_COMM_WORLD`

- `MPI_Comm_size()`

  - Returns number of processes $N$

- `MPI_Comm_rank()`

  - Returns rank (ID) of calling process within communicator
  - Used for source/destination in communications

- A process can belong to multiple communicators with different ranks

## MPI Initialization

- `int MPI_Init(int *argc, char ***argv)`

  - Initializes MPI environment
  - Must be called before any other MPI routine (except `MPI_Initialized`)
  - Can be called only once

- `int MPI_Initialized(int *flag)`

  - Checks if `MPI_Init()` has been called

## Termination

- `int MPI_Finalize(void)`

  - Cleans up MPI environment
  - Must be called before program exits
  - No other MPI routine can be called afterward

    - Exception: `MPI_Initialized()`, `MPI_Get_version()`, `MPI_Finalized()`

- `int MPI_Abort(MPI_Comm comm, int errcode)`

  - Abnormal termination
  - Attempts to abort all tasks

## MPI Initialization and Termination (Example)

```c
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int flag;
    MPI_Initialized(&flag);
    if (flag != 0) {
        // ...
    }

    MPI_Finalize();
    return 0;
}
```

## MPI Implementations

- MPI is a specification for APIs, not an implementation

  - Defines the C interface of `MPI_Send`

    ```c
    int MPI_Send(const void *buf, int count, MPI_Datatype datatype,
                 int dest, int tag, MPI_Comm comm);
    ```

  - Defines semantics (e.g., blocking send)
  - Does not define how `MPI_Send` is implemented

- Common MPI implementations:

  - **OpenMPI**
  - **MPICH** (Argonne National Lab)
  - **MVAPICH2** (MPICH derivative from Ohio State University)
  - **Intel MPI** (MPICH derivative from Intel)

## Compiling MPI Programs

- Compiling an MPI program with `gcc` or `g++` without MPI-specific flags causes undefined reference errors

  - Program is not linked to the MPI library

```sh
$ g++ -o hello_world hello_world.cpp
# results in multiple "undefined reference to `MPI_...`" errors
```

- Use `mpicc` or `mpic++` instead

  - These are wrappers that insert the correct flags for linking
  - Use `--showme` to view actual flags added

```sh
$ mpic++ -o hello_world hello_world.cpp       # works
$ mpic++ -o hello_world hello_world.cpp --showme
g++ -o hello_world hello_world.cpp -I/usr/local/include ...
```

## MPI Hello World with Hostname

- `MPI_Get_processor_name(...)` returns the hostname of the node

  - Useful for debugging

```c
#include <cstdio>
#include <mpi.h>

int main() {
    MPI_Init(NULL, NULL);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char hostname[MPI_MAX_PROCESSOR_NAME];
    int hostnamelen;
    MPI_Get_processor_name(hostname, &hostnamelen);

    printf("[%s] Hello, I am rank %d of size %d world!\n", hostname, rank, size);

    MPI_Finalize();
    return 0;
}
```

## Executing MPI Programs

- MPI programs can be executed directly:

  ```sh
  $ ./hello_world
  [a5] Hello, I am rank 0 of size 1 world!
  ```

- Use `mpirun` or `mpiexec` to control number of processes:

  ```sh
  $ mpirun ./hello_world
  ```

- If no process count is specified, the number of processes defaults to the number of physical cores

```sh
$ mpirun ./hello_world
[a5] Hello, I am rank 19 of size 32 world!
[a5] Hello, I am rank 22 of size 32 world!
...
```

- Control processes per node using `-npernode`:

  ```sh
  $ mpirun -npernode 4 ./hello_world
  ```

- Specify nodes with `-H`:

  ```sh
  $ mpirun -H a4,a5 ./hello_world
  [a5] Hello, I am rank 0 of size 2 world!
  [a4] Hello, I am rank 1 of size 2 world!
  ```

- Specify number of processes per node using colon (`:`):

  ```sh
  $ mpirun -H a4:4,a5:4 ./hello_world
  ```

## Executing in SLURM (salloc)

- In a SLURM environment, use `salloc` to allocate resources and run programs:

```sh
# Allocate 1 node, 1 process per node
$ salloc -N 1 mpirun -mca btl ^openib -npernode 1 ./hello_world

# Allocate 2 nodes, 2 processes per node
$ salloc -N 2 mpirun -mca btl ^openib -npernode 2 ./hello_world
```

## Error Handling Macro

- Always check error codes returned by MPI functions

  - Time for debugging will dramatically decrease

- `MPI_Error_string` returns a description for the error code

```c
#include <mpi.h>
#include <stdio.h>

#define CHECK_MPI(call)                                \
  do {                                                 \
    int code = call;                                   \
    if (code != MPI_SUCCESS) {                         \
      char estr[MPI_MAX_ERROR_STRING];                 \
      int elen;                                        \
      MPI_Error_string(code, estr, &elen);             \
      fprintf(stderr, "MPI error (%s:%d): %s\n",       \
              __FILE__, __LINE__, estr);               \
      MPI_Abort(MPI_COMM_WORLD, code);                 \
    }                                                  \
  } while (0)

int main(int argc, char **argv) {
  int my_rank, size;
  CHECK_MPI(MPI_Init(&argc, &argv));
  CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));
  CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &size));
  printf("Hello, I am %d of %d\n", my_rank, size);
  CHECK_MPI(MPI_Finalize());
  return 0;
}
```

## Common Misconceptions: Threads vs. Processes

- Threads within the same process share the address space
- Processes do not share the address space
- Threads can access values written by another thread if the address is known

```c
// pthread example
A[i] = i;
pthread_create(..., thread_func, ...);
s += A[i];

// OpenMP example
A[i] = i;
#pragma omp parallel
{
  s += A[i];
}
```

- In MPI, communication is required if passing data between processes

```c
// Rank 0 process
A[i] = i;
MPI_Barrier();
// Rank 0 process can access A
```

```c
// Rank 1 process
MPI_Barrier();
// Rank 1 process cannot access A
```

## Common Misconceptions: SPMD

- MPI follows SPMD (Single Program, Multiple Data) model

  - This does not mean all processes do the same work
  - You can implement MPI programs in MPMD style

```c
int rank;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

if (rank == 0) {
  // Program 1
}

if (rank == 1) {
  // Program 2
}
```

## Practice: MPI Hello World

- Implement MPI Hello World program

  - Print host names of each node the process executes on
  - Do error handling

- Write Makefile by yourself

- Try launching processes in different configurations with `salloc`:

  - Multiple processes in a single node
  - Multiple processes in multiple nodes

## MPI Communications

- **Point-to-point communications**

  - Involves a sender and a receiver (one process to another)

- **Collective communications**

  - All processes within a communicator participate
  - Examples: Barrier, reduction operations, gather, ...

## MPI Data Types

- Recursively defined:

  - Predefined types (e.g., `MPI_INT`, `MPI_DOUBLE_PRECISION`)
  - Contiguous array of MPI datatypes
  - Strided block of datatypes
  - Indexed array of blocks of datatypes
  - Arbitrary structure of datatypes

- MPI functions to construct custom datatypes:

  - Array of (int, float) pairs
  - Row of a matrix stored column-wise

## Basic MPI Data Types

| MPI datatype       | C datatype        |
| ------------------ | ----------------- |
| MPI_CHAR           | signed char       |
| MPI_SHORT          | signed short      |
| MPI_INT            | signed int        |
| MPI_LONG           | signed long       |
| MPI_UNSIGNED_CHAR  | unsigned char     |
| MPI_UNSIGNED_SHORT | unsigned short    |
| MPI_UNSIGNED       | unsigned int      |
| MPI_UNSIGNED_LONG  | unsigned long int |
| MPI_DOUBLE         | double            |
| MPI_FLOAT          | float             |
| MPI_LONG_DOUBLE    | long double       |
| MPI_BYTE           |                   |
| MPI_PACKED         |                   |

## MPI Tags

- Messages are sent with a user-defined integer tag

  - Assists the receiving process in identifying the message
  - Messages can be screened using specific tags
  - Not screened when specifying `MPI_ANY_TAG`

## Blocking Send

```c
int MPI_Send(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
```

- `buf`: address of send buffer (where the message is)
- `count`: number of data items
- `datatype`: type of data items
- `dest`: rank of destination process
- `tag`: message tag
- `comm`: communicator, usually `MPI_COMM_WORLD`

### Behavior

- `MPI_Send()` is blocking

  - When it returns, data has been delivered to the system
  - The user can safely access or overwrite the send buffer
  - The message may not have been received yet by the destination process

## Blocking Receive

```c
int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)
```

- `buf`: address of receive buffer
- `count`: number of elements in the receive buffer
- `datatype`: data type of receive buffer elements
- `source`: rank of source process or `MPI_ANY_SOURCE`
- `tag`: message tag or `MPI_ANY_TAG`
- `status`: status object containing additional information about the received message or `MPI_STATUS_IGNORE`

### Behavior

- `MPI_Recv()` is blocking

  - Waits until a matching message is received
  - After it returns, data is in the buffer and ready for use

- Receiving fewer than `count` items is OK; receiving more is an error

## MPI_Recv Status

- `MPI_Status` structure has 3 members:

  - `status.MPI_TAG`: tag of received message
  - `status.MPI_SOURCE`: source rank of message
  - `status.MPI_ERROR`: error code

```c
int MPI_Get_count(MPI_Status *status, MPI_Datatype datatype, int *count)
```

- Returns the length of the received message

### Example

```c
MPI_Status status = {0}; // MPI_ERROR is not set when there is no error
MPI_Recv(buf, count, MPI_FLOAT, 0, 1234, MPI_COMM_WORLD, &status);
printf("Source = %d, Tag = %d, Error = %d\n",
    status.MPI_SOURCE,
    status.MPI_TAG,
    status.MPI_ERROR);
```

## Send-Recv Example

### Separate Programs

**Rank 0**

```c
int count = 10000;
float *buf = (float*)malloc(count * sizeof(float));

MPI_Send(buf, count, MPI_FLOAT, 1, 1234, MPI_COMM_WORLD);
```

**Rank 1**

```c
int count = 10000;
float *buf = (float*)malloc(count * sizeof(float));
MPI_Status status;

MPI_Recv(buf, count, MPI_FLOAT, 0, 1234, MPI_COMM_WORLD, &status);
```

### Combined Program

```c
int count = 10000;
float *buf = (float*)malloc(count * sizeof(float));

if (rank == 0) {
    MPI_Send(buf, count, MPI_FLOAT, 1, 1234, MPI_COMM_WORLD);
}

if (rank == 1) {
    MPI_Status status;
    MPI_Recv(buf, count, MPI_FLOAT, 0, 1234, MPI_COMM_WORLD, &status);
}
```

## MPI is Simple

Many parallel programs can be written using just six functions:

- `MPI_Init()`
- `MPI_Finalize()`
- `MPI_Comm_size()`
- `MPI_Comm_rank()`
- `MPI_Send()`
- `MPI_Recv()`

## Deadlocks

Be careful to avoid deadlocks.

- Carefully sequence all messages

**Example of Potential Deadlocks**

- Both processes calling `MPI_Send()` before `MPI_Recv()`
- Mismatched message order (tags or send/recv order)

## Buffering

- Send and matching receive operations may not be synchronized in reality

  - Behavior is implementation-dependent
  - Typically, a system buffer holds data in transit

# MPI Send and Receive Modes

## Send Modes

### Standard Send

```c
int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);
```

### Synchronous Send

```c
int MPI_Ssend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);
```

- Completes only when matching receive is posted and has started receiving.
- Ensures synchronization without extra primitives like barriers.
- Guarantees order: code after `MPI_Ssend` executes only after matching receive begins.

### Buffered Send

```c
int MPI_Bsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);
```

- Completion is independent of a matching receive.
- Local operation; user must explicitly allocate a buffer.

Example:

```c
int pack_size;
MPI_Pack_size(count, MPI_FLOAT, MPI_COMM_WORLD, &pack_size);
int buffer_size = pack_size + MPI_BSEND_OVERHEAD;
float* buf = (float*)malloc(buffer_size);
MPI_Buffer_attach(buf, buffer_size);

MPI_Bsend(A, count, MPI_FLOAT, 1, 1234, MPI_COMM_WORLD);
```

### Ready Send

```c
int MPI_Rsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);
```

- Requires that the matching receive is already posted.
- No handshake; may perform better in some cases.
- If posted before the receive, results in undefined behavior.

Synchronization example:

```c
MPI_Barrier(MPI_COMM_WORLD);
MPI_Rsend(...);
```

## Receive Mode

Only one mode:

```c
int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status);
```

## Delivery Order

- If a sender sends two messages to the same destination that match the same receive, the first message will be received.
- If a receiver posts two receives matching the same message, the first posted will receive it.
- If a receive matches two sends from different sources, it may receive either.

Example:

```c
if (rank == 0) {
    MPI_Bsend(buf1, count, MPI_DOUBLE, 1, tag, comm);
    MPI_Bsend(buf2, count, MPI_DOUBLE, 1, tag, comm);
} else if (rank == 1) {
    MPI_Recv(buf1, count, MPI_DOUBLE, 0, MPI_ANY_TAG, comm, &status);
    MPI_Recv(buf2, count, MPI_DOUBLE, 0, tag, comm, &status);
}
```

## Sendrecv

- Avoids deadlock by combining send and receive in one call.
- Implements non-blocking send and recv internally, then waits.

```c
int MPI_Sendrecv(
    void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag,
    void *recvbuf, int recvcount, MPI_Datatype recvtype, int source, int recvtag,
    MPI_Comm comm, MPI_Status *status);
```

## Non-blocking Communication

### Overview

- **Blocking send/recv**: returns after operation completes.
- **Non-blocking send/recv**: returns immediately; must use `MPI_Wait` or `MPI_Test`.

### Prefixes

- Non-blocking versions use `I` prefix:

  - `MPI_Isend`, `MPI_Issend`, `MPI_Ibsend`, `MPI_Irsend`, `MPI_Irecv`

- Can mix blocking and non-blocking (e.g. `MPI_Send` with `MPI_Irecv`).

### Benefits

- Avoids idle time and some deadlocks.
- Enables overlapping computation with communication.

### Non-blocking Communication Example

#### Send

```c
MPI_Isend(buf, count, type, dest, tag, comm, &request);
MPI_Wait(&request, &status);
```

#### Receive

```c
MPI_Irecv(buf, count, type, dest, tag, comm, &request);
MPI_Wait(&request, &status);
```

### Completion Check

#### Using `MPI_Test`

```c
MPI_Test(MPI_Request *request, int *flag, MPI_Status *status);
```

- Sets `flag = true` if complete, false otherwise.

Other variants:

- `MPI_TestAll`, `MPI_TestAny`, `MPI_TestSome`

#### Using `MPI_Wait`

```c
MPI_Wait(MPI_Request *request, MPI_Status *status);
```

- Blocks until complete.

Other variants:

- `MPI_WaitAll`, `MPI_WaitAny`, `MPI_WaitSome`

### Non-blocking Example with Work Overlap

#### Rank 0

```c
int count = 10000;
float *buf = (float*)malloc(count * sizeof(float));

MPI_Request request;
MPI_Isend(buf, count, MPI_FLOAT, 1, 1234, MPI_COMM_WORLD, &request);

// Useful work can be done here

MPI_Status status = {0};
MPI_Wait(&request, &status);
```

#### Rank 1

```c
int count = 10000;
float *buf = (float*)malloc(count * sizeof(float));

MPI_Request request;
MPI_Irecv(buf, count, MPI_FLOAT, 0, 1234, MPI_COMM_WORLD, &request);

// Useful work can be done here

MPI_Status status = {0};
MPI_Wait(&request, &status);
```

# MPI Point-to-Point Communication Practice

## Objective

Implement an MPI program demonstrating point-to-point communication between two ranks:

- Rank 0 creates and initializes an array (a[i] = i)
- Rank 1 receives and verifies the array
- Use blocking `MPI_Send` and `MPI_Recv`

## Performance Measurement

- Time the communication using `MPI_Wtime` or `get_time()` from util.cpp
- Calculate network bandwidth:
  ```
  bandwidth = data size (bytes) / communication time (seconds)
  ```

## Advanced Tasks

1. Experiment with different send modes:

   - Standard (`MPI_Send`)
   - Synchronous (`MPI_Ssend`)
   - Buffered (`MPI_Bsend`)
   - Ready (`MPI_Rsend`)

2. Implement non-blocking communication:
   - `MPI_Isend` and `MPI_Irecv`
   - Use `MPI_Wait` or `MPI_Test` for completion checks

## Key Concepts

- Blocking vs. non-blocking communication
- Different send mode semantics
- Performance measurement techniques
- Bandwidth calculation for network assessment

# MPI Collective Communication

## Overview

- Cooperative communication among all processes in a communicator
- Must be called by all processes in the communicator
- Provides optimized patterns for common parallel operations
- Available in both blocking and non-blocking versions

## Key Operations

### Broadcast (MPI_Bcast)

- One-to-all data distribution from root process
- Signature: `MPI_Bcast(void *buf, int count, MPI_Datatype datatype, int root, MPI_Comm comm)`

### Reduction (MPI_Reduce)

- Combines data from all processes to one target
- Supports operations: MPI_SUM, MPI_MAX, MPI_MIN, etc.
- Signature: `MPI_Reduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm)`

### All-Reduce (MPI_Allreduce)

- Combines values and distributes result to all processes
- Signature: `MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)`

### Gather (MPI_Gather)

- Collects data from all processes to root
- Signature: `MPI_Gather(void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)`

### Scatter (MPI_Scatter)

- Distributes data from root to all processes
- Signature: `MPI_Scatter(void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)`

### All-to-All (MPI_Alltoall)

- Each process sends distinct data to every other process
- Signature: `MPI_Alltoall(void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)`

### Reduce-Scatter (MPI_Reduce_scatter)

- Combines reduction followed by scatter operation
- Signature: `MPI_Reduce_scatter(const void *sendbuf, void *recvbuf, const int recvcounts[], MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)`

### Scan (MPI_Scan)

- Computes inclusive prefix reduction
- Signature: `MPI_Scan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)`

### Barrier (MPI_Barrier)

- Synchronizes all processes in communicator
- Signature: `MPI_Barrier(MPI_Comm comm)`

# MPI Advanced Collective Communication

## In-Place Operations

- **MPI_IN_PLACE** allows reusing buffers for memory efficiency
- Usage example for MPI_Reduce:
  ```c
  MPI_Reduce((rank == root) ? MPI_IN_PLACE : sendbuf,
             recvbuf, count, datatype, op, root, comm);
  ```
- Benefits:
  - Avoids extra memory allocation
  - Reduces memory copies
  - Particularly useful for root process in reduction operations

## Variable-Length Data Operations

### Scatterv/Gatherv

- Handle uneven data distributions across processes
- **MPI_Scatterv** parameters:

  - `sendcounts[]`: Array specifying how many elements to send to each process
  - `displs[]`: Array of displacements indicating where each process's data begins
  - `recvcount`: Number of elements to receive (must match corresponding sendcount)

- **MPI_Gatherv** parameters:
  - `recvcounts[]`: Array specifying how many elements to receive from each process
  - `displs[]`: Array of displacements for placement in receive buffer

### Usage Example

```c
// Scatterv example
int sendcounts[4] = {10, 20, 30, 40};  // Different amounts for 4 processes
int displs[4] = {0, 10, 30, 60};       // Starting offsets in send buffer
MPI_Scatterv(sendbuf, sendcounts, displs, MPI_INT,
             recvbuf, recvcount, MPI_INT, root, comm);
```

## Key Differences from Regular Collectives

| Feature        | Regular (Scatter/Gather) | Vector (Scatterv/Gatherv) |
| -------------- | ------------------------ | ------------------------- |
| Data Size      | Uniform per process      | Variable per process      |
| Buffer Control | Simple                   | Requires offsets/counts   |
| Flexibility    | Limited                  | High                      |
| Memory Usage   | Predictable              | More complex management   |

## Best Practices

1. Use `MPI_IN_PLACE` for memory-critical applications
2. Prefer vector versions when data distribution is irregular
3. Always verify displacement arrays to prevent buffer overflows
4. Consider using derived datatypes for complex non-contiguous data patterns
