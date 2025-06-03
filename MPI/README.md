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
