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
