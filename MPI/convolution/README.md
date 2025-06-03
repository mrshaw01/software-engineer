# The Problem: Convolution Im2col

- **Input**: The input tensor \$I \in \mathbb{R}^{N \times C \times H \times W}\$ and the filter tensor \$F \in \mathbb{R}^{K \times C \times R \times S}\$ are located on the root process.
- **Output**: The output tensor \$O \in \mathbb{R}^{N \times K \times OH \times OW}\$.

# Strategy

1. **Distribute the input along the \$N\$ (batch) dimension.**
2. **Divide the images equally among all processes.**
   Each process computes its portion of the output and sends it back to the root process.
3. **Within each process, distribute images across GPUs.**
   Each GPU computes its assigned part of the output and returns it to the host.

To divide `N` images into `M` parts and compute the `[begin, end)` range for a given `interval_id`, use:

```cpp
void splitIntoIntervals(int N, int M, int interval_id, int &begin, int &end) {
    int interval_size = N / M;
    int remainder = N % M;

    begin = interval_id * interval_size + std::min(interval_id, remainder);
    end = begin + interval_size + (interval_id < remainder ? 1 : 0);

    if (end > N) end = N;
}
```

This function ensures balanced and contiguous image distribution across processes and devices.

# Im2col

- The `gpu_im2col` kernel is used to transform the input into a matrix suitable for efficient matrix multiplication.

# Matmul + Implicit Reshape

- The `gpu_matmul_kernel` is used to perform matrix multiplication.
- Techniques:

  - **Tiling and unrolling**: Improves shared memory usage and ensures coalesced memory access.
  - **Implicit reshape**: Output is written directly to its final layout in global memory, avoiding costly post-processing reshapes.

# Pipeline with CUDA Streams

- On each device, input is partitioned into blocks to enable **pipelined execution** that overlaps data transfer and computation.
- Pipeline stages per stream:

  1. Transfer input from host to device
  2. `im2col`
  3. `matmul + implicit reshape`
  4. Transfer output from device to host

- Each CUDA stream manages one such pipeline for a block of input.
- This approach avoids idle time between stages and maximizes throughput when dependencies between stages are minimal.

# Choosing Optimal Hyperparameters

- **Unrolling factor**: Tried values from `1 * TILE_SIZE` to `10 * TILE_SIZE`.
  The best performance was achieved with `6 * TILE_SIZE` across the base test case and various random scenarios.
- **Number of nodes**: Chosen based on the shape and size of the specific problem instance.

# Base Test Case

```bash
salloc -N [num_nodes] --exclusive \
mpirun --bind-to none -mca btl ^openib -npernode 1 \
./main -n 10 128 512 64 64 2048 9 9 1 1 1 1 1 1 > output
```

- Output is redirected to a file named `output` using `>`.
- The program supports arbitrary values of `[num_nodes]`, as the input batch is evenly distributed across processes.
- **Important constraint**: Only `-npernode 1` is supported. This design follows a model where:

  - Each **MPI process corresponds to one node**.
  - Each process can spawn **multiple threads** internally for computation.

This approach simplifies distribution and ensures that intra-node parallelism is handled via threading, while inter-node distribution is handled via MPI.

# Performance

```bash
$ salloc -N 4 --exclusive mpirun --bind-to none -mca btl ^openib -npernode 1 ./main -n 10 128 512 64 64 2048 9 9 1 1 1 1 1 1 > output

Problem size: N = 128, C = 512, H = 64, W = 64, K = 2048, R = 9, S = 9
              pad_h = 1, pad_w = 1, stride_h = 1, stride_w = 1
              dilation_h = 1, dilation_w = 1
Number of iterations: 10
Print tensor: off
Validation: off

              OH: 58, OW: 58

Hello world from processor a0, rank 0 out of 4, mpi_Nbegin=0, mpiNend=32
Hello world from processor a3, rank 3 out of 4, mpi_Nbegin=96, mpiNend=128
Hello world from processor a1, rank 1 out of 4, mpi_Nbegin=32, mpiNend=64
Hello world from processor a2, rank 2 out of 4, mpi_Nbegin=64, mpiNend=96
Initializing... done!
Initializing... done!
Initializing... done!
Initializing... done!
[rank 0] Initializing Tensors...
done!
Calculating...(iter=0) 1.001533 sec
Calculating...(iter=1) 0.780736 sec
Calculating...(iter=2) 0.778776 sec
Calculating...(iter=3) 0.777010 sec
Calculating...(iter=4) 0.777679 sec
Calculating...(iter=5) 0.777492 sec
Calculating...(iter=6) 0.776477 sec
Calculating...(iter=7) 0.776771 sec
Calculating...(iter=8) 0.774829 sec
Calculating...(iter=9) 0.776981 sec
Avg. time: 0.799828 sec
Avg. throughput: 91450.078824 GFLOPS
```

```bash
$ salloc -N 3 --exclusive mpirun --bind-to none -mca btl ^openib -npernode 1 ./main -n 10 128 512 64 64 2048 9 9 1 1 1 1 1 1 > output

Problem size: N = 128, C = 512, H = 64, W = 64, K = 2048, R = 9, S = 9
              pad_h = 1, pad_w = 1, stride_h = 1, stride_w = 1
              dilation_h = 1, dilation_w = 1
Number of iterations: 10
Print tensor: off
Validation: off

              OH: 58, OW: 58

Hello world from processor a2, rank 2 out of 3, mpi_Nbegin=86, mpiNend=128
Hello world from processor a0, rank 0 out of 3, mpi_Nbegin=0, mpiNend=43
Hello world from processor a1, rank 1 out of 3, mpi_Nbegin=43, mpiNend=86
Initializing... done!
Initializing... done!
Initializing... done!
[rank 0] Initializing Tensors...
done!
Calculating...(iter=0) 1.155043 sec
Calculating...(iter=1) 0.906965 sec
Calculating...(iter=2) 0.905342 sec
Calculating...(iter=3) 0.902356 sec
Calculating...(iter=4) 0.902952 sec
Calculating...(iter=5) 0.905167 sec
Calculating...(iter=6) 0.903159 sec
Calculating...(iter=7) 0.905086 sec
Calculating...(iter=8) 0.902400 sec
Calculating...(iter=9) 0.902527 sec
Avg. time: 0.929100 sec
Avg. throughput: 78726.068440 GFLOPS
```

```bash
$ salloc -N 2 --exclusive mpirun --bind-to none -mca btl ^openib -npernode 1 ./main -n 10 128 512 64 64 2048 9 9 1 1 1 1 1 1 > output

Problem size: N = 128, C = 512, H = 64, W = 64, K = 2048, R = 9, S = 9
              pad_h = 1, pad_w = 1, stride_h = 1, stride_w = 1
              dilation_h = 1, dilation_w = 1
Number of iterations: 10
Print tensor: off
Validation: off

              OH: 58, OW: 58

Hello world from processor a0, rank 0 out of 2, mpi_Nbegin=0, mpiNend=64
Hello world from processor a1, rank 1 out of 2, mpi_Nbegin=64, mpiNend=128
Initializing... done!
Initializing... done!
[rank 0] Initializing Tensors...
done!
Calculating...(iter=0) 1.436517 sec
Calculating...(iter=1) 1.123397 sec
Calculating...(iter=2) 1.122187 sec
Calculating...(iter=3) 1.124591 sec
Calculating...(iter=4) 1.121765 sec
Calculating...(iter=5) 1.122278 sec
Calculating...(iter=6) 1.122050 sec
Calculating...(iter=7) 1.121910 sec
Calculating...(iter=8) 1.121434 sec
Calculating...(iter=9) 1.122647 sec
Avg. time: 1.153878 sec
Avg. throughput: 63390.054778 GFLOPS
```

```bash
$ salloc -N 1 --exclusive mpirun --bind-to none -mca btl ^openib -npernode 1 ./main -n 10 64 512 64 64 2048 9 9 1 1 1 1 1 1 > output

Problem size: N = 64, C = 512, H = 64, W = 64, K = 2048, R = 9, S = 9
              pad_h = 1, pad_w = 1, stride_h = 1, stride_w = 1
              dilation_h = 1, dilation_w = 1
Number of iterations: 10
Print tensor: off
Validation: off

              OH: 58, OW: 58

Hello world from processor a3, rank 0 out of 1, mpi_Nbegin=0, mpiNend=64
Initializing... done!
[rank 0] Initializing Tensors...
done!
Calculating...(iter=0) 0.927416 sec
Calculating...(iter=1) 0.905828 sec
Calculating...(iter=2) 0.907298 sec
Calculating...(iter=3) 0.907683 sec
Calculating...(iter=4) 0.906843 sec
Calculating...(iter=5) 0.905612 sec
Calculating...(iter=6) 0.906051 sec
Calculating...(iter=7) 0.905331 sec
Calculating...(iter=8) 0.905972 sec
Calculating...(iter=9) 0.906117 sec
Avg. time: 0.908415 sec
Avg. throughput: 40259.331045 GFLOPS
```

# Validation

```bash
$ salloc -N 4 --exclusive mpirun --bind-to none -mca btl ^openib -npernode 1 ./main -v -n 3 46 55 24 56 19 2 2 2 3 3 3 2 4 > output

Problem size: N = 46, C = 55, H = 24, W = 56, K = 19, R = 2, S = 2
              pad_h = 2, pad_w = 3, stride_h = 3, stride_w = 3
              dilation_h = 2, dilation_w = 4
Number of iterations: 3
Print tensor: off
Validation: on

              OH: 9, OW: 20

Hello world from processor a0, rank 0 out of 4, mpi_Nbegin=0, mpiNend=12
Hello world from processor a3, rank 3 out of 4, mpi_Nbegin=35, mpiNend=46
Hello world from processor a1, rank 1 out of 4, mpi_Nbegin=12, mpiNend=24
Hello world from processor a2, rank 2 out of 4, mpi_Nbegin=24, mpiNend=35
Initializing... done!
Initializing... done!
Initializing... done!
[rank 0] Initializing Tensors...
Initializing... done!
done!
Calculating...(iter=0) 0.005421 sec
Calculating...(iter=1) 0.000983 sec
Calculating...(iter=2) 0.000929 sec
Validation Result: VALID
Avg. time: 0.002444 sec
Avg. throughput: 28.319653 GFLOPS
```

```bash
$ salloc -N 3 --exclusive mpirun --bind-to none -mca btl ^openib -npernode 1 ./main -v -n 3 15 49 77 57 19 2 1 3 3 2 3 1 4 > output

Problem size: N = 15, C = 49, H = 77, W = 57, K = 19, R = 2, S = 1
              pad_h = 3, pad_w = 3, stride_h = 2, stride_w = 3
              dilation_h = 1, dilation_w = 4
Number of iterations: 3
Print tensor: off
Validation: on

              OH: 41, OW: 21

Hello world from processor a0, rank 0 out of 3, mpi_Nbegin=0, mpiNend=5
Hello world from processor a2, rank 2 out of 3, mpi_Nbegin=10, mpiNend=15
Hello world from processor a1, rank 1 out of 3, mpi_Nbegin=5, mpiNend=10
Initializing... done!
Initializing... done!
Initializing... done!
[rank 0] Initializing Tensors...
done!
Calculating...(iter=0) 0.005618 sec
Calculating...(iter=1) 0.000888 sec
Calculating...(iter=2) 0.000845 sec
Validation Result: VALID
Avg. time: 0.002450 sec
Avg. throughput: 19.627066 GFLOPS
```

```bash
$ salloc -N 2 --exclusive mpirun --bind-to none -mca btl ^openib -npernode 1 ./main -v -n 3 37 210 100 57 19 2 1 3 3 11 8 8 7 > output

Problem size: N = 37, C = 210, H = 100, W = 57, K = 19, R = 2, S = 1
              pad_h = 3, pad_w = 3, stride_h = 11, stride_w = 8
              dilation_h = 8, dilation_w = 7
Number of iterations: 3
Print tensor: off
Validation: on

              OH: 9, OW: 8

Hello world from processor a1, rank 0 out of 2, mpi_Nbegin=0, mpiNend=19
Hello world from processor a2, rank 1 out of 2, mpi_Nbegin=19, mpiNend=37
Initializing... done!
[rank 0] Initializing Tensors...
Initializing... done!
done!
Calculating...(iter=0) 0.016221 sec
Calculating...(iter=1) 0.006300 sec
Calculating...(iter=2) 0.006240 sec
Validation Result: VALID
Avg. time: 0.009587 sec
Avg. throughput: 4.434919 GFLOPS
```

```bash
$ salloc -N 1 --exclusive mpirun --bind-to none -mca btl ^openib -npernode 1 ./main -v -n 3 37 139 110 29 19 2 1 3 3 6 6 2 5 > output


Problem size: N = 37, C = 139, H = 110, W = 29, K = 19, R = 2, S = 1
              pad_h = 3, pad_w = 3, stride_h = 6, stride_w = 6
              dilation_h = 2, dilation_w = 5
Number of iterations: 3
Print tensor: off
Validation: on

              OH: 19, OW: 6

Hello world from processor a1, rank 0 out of 1, mpi_Nbegin=0, mpiNend=37
Initializing... done!
[rank 0] Initializing Tensors...
done!
Calculating...(iter=0) 0.002605 sec
Calculating...(iter=1) 0.001437 sec
Calculating...(iter=2) 0.001331 sec
Validation Result: VALID
Avg. time: 0.001791 sec
Avg. throughput: 24.880469 GFLOPS
```
