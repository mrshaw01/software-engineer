# Distributed Matrix Multiplication Benchmark

This project benchmarks high-performance tiled matrix multiplication using MPI across multiple nodes and multiple threads per node. It includes validation and performance measurement for various matrix sizes, unrolling strategies, and hyper-parameter choices.

## Base Test Case

```bash
salloc -N 2 --exclusive \
mpirun --bind-to none -mca btl ^openib -npernode 1 \
./main -t 32 -n 10 8192 8192 8192
```

## Highest Performance for Base Case

```bash
./run_performance_base.sh
```

Achieved peak performance of:

- **2175.07 GFLOPS**
- **Average time**: 0.505 sec
- **Matrix size**: 8192 × 8192 × 8192
- **Threads per node**: 32
- **Nodes**: 2

## Techniques Used

- Tiled matrix multiplication
- Loop unrolling
- SIMD vectorization
- Multi-threading with OpenMP
- Distributed processing with MPI

## Hyper-parameter Choices

### Tiling (obtained via random search)

```cpp
ITILESIZE = 96
JTILESIZE = 512
KTILESIZE = 768
```

### Unrolling (empirically chosen)

```cpp
for (k = kk; k < boundk - 5; k += 6) // unroll by 6
```

### Threads per node

Use physical core count: `-t 32`

## Performance for Various Matrix Sizes

```bash
./run_performance.sh
```

| M × N × K          | Time (s) | Throughput (GFLOPS) |
| ------------------ | -------- | ------------------- |
| 8192 × 8192 × 8192 | 0.511    | 2152.03             |
| 4096 × 4096 × 4096 | 0.069    | 1988.30             |
| 2048 × 2048 × 2048 | 0.014    | 1198.32             |
| 1024 × 1024 × 1024 | 0.003    | 690.51              |
| 5678 × 7891 × 1234 | 0.068    | 1623.02             |
| 7891 × 1234 × 5678 | 0.067    | 1650.22             |
| 1234 × 5678 × 7891 | 0.080    | 1390.25             |

## Validation

Run with different per-node configurations:

```bash
./run_validation_npernode1.sh   # For -npernode 1
./run_validation_npernode2.sh   # For -npernode 2
./run_validation_npernode5.sh   # For -npernode 5
```

```bash
./run_validation_npernode1.sh
```

| Job Allocation | Problem Size (M x N x K) | Threads | Iterations | Avg. Time (sec) | Avg. Throughput (GFLOPS) | Result |
| -------------- | ------------------------ | ------- | ---------- | --------------- | ------------------------ | ------ |
| 37550          | 293 x 399 x 123          | 32      | 3          | 0.001578        | 18.224894                | VALID  |
| 37551          | 3 x 699 x 10             | 32      | 3          | 0.001639        | 0.025583                 | VALID  |
| 37552          | 331 x 21 x 129           | 32      | 3          | 0.001736        | 1.032990                 | VALID  |
| 37554          | 2000 x 1 x 2000          | 32      | 3          | 0.002461        | 3.250244                 | VALID  |
| 37555          | 323 x 429 x 111          | 32      | 3          | 0.001822        | 16.886570                | VALID  |
| 37557          | 1 x 2000 x 2000          | 32      | 3          | 0.001373        | 5.825422                 | VALID  |
| 37559          | 2000 x 2000 x 1          | 32      | 3          | 0.002159        | 3.704806                 | VALID  |
| 37560          | 64 x 64 x 64             | 32      | 3          | 0.001733        | 0.302493                 | VALID  |
| 37562          | 128 x 128 x 128          | 32      | 3          | 0.001727        | 2.428295                 | VALID  |
| 37563          | 256 x 256 x 256          | 32      | 3          | 0.002863        | 11.719985                | VALID  |

```bash
./run_validation_npernode2.sh
```

| Job Allocation | Problem Size (M x N x K) | Threads | Iterations | Avg. Time (sec) | Avg. Throughput (GFLOPS) | Result |
| -------------- | ------------------------ | ------- | ---------- | --------------- | ------------------------ | ------ |
| 37602          | 293 x 399 x 123          | 32      | 3          | 0.001979        | 14.533072                | VALID  |
| 37603          | 3 x 699 x 10             | 32      | 3          | 0.005234        | 0.008013                 | VALID  |
| 37606          | 331 x 21 x 129           | 32      | 3          | 0.002362        | 0.759173                 | VALID  |
| 37608          | 2000 x 1 x 2000          | 32      | 3          | 0.002953        | 2.709426                 | VALID  |
| 37609          | 323 x 429 x 111          | 32      | 3          | 0.001929        | 15.949316                | VALID  |
| 37610          | 1 x 2000 x 2000          | 32      | 3          | 0.001639        | 4.881118                 | VALID  |
| 37611          | 2000 x 2000 x 1          | 32      | 3          | 0.003368        | 2.375311                 | VALID  |
| 37612          | 64 x 64 x 64             | 32      | 3          | 0.002009        | 0.261022                 | VALID  |
| 37614          | 128 x 128 x 128          | 32      | 3          | 0.002068        | 2.028463                 | VALID  |
| 37615          | 256 x 256 x 256          | 32      | 3          | 0.001997        | 16.799127                | VALID  |

```bash
./run_validation_npernode5.sh
```

| Job Allocation | Problem Size (M x N x K) | Threads | Iterations | Avg. Time (sec) | Avg. Throughput (GFLOPS) | Result |
| -------------- | ------------------------ | ------- | ---------- | --------------- | ------------------------ | ------ |
| 37942          | 293 x 399 x 123          | 32      | 3          | 0.002078        | 13.839963                | VALID  |
| 37943          | 3 x 699 x 10             | 32      | 3          | 0.000174        | 0.241081                 | VALID  |
| 37944          | 331 x 21 x 129           | 32      | 3          | 0.006903        | 0.259782                 | VALID  |
| 37945          | 2000 x 1 x 2000          | 32      | 3          | 0.003359        | 2.381380                 | VALID  |
| 37946          | 323 x 429 x 111          | 32      | 3          | 0.003430        | 8.967726                 | VALID  |
| 37948          | 1 x 2000 x 2000          | 32      | 3          | 0.000951        | 8.408930                 | VALID  |
| 37949          | 2000 x 2000 x 1          | 32      | 3          | 0.003870        | 2.067135                 | VALID  |
| 37950          | 64 x 64 x 64             | 32      | 3          | 0.000527        | 0.995484                 | VALID  |
| 37952          | 128 x 128 x 128          | 32      | 3          | 0.001060        | 3.958044                 | VALID  |
| 37954          | 256 x 256 x 256          | 32      | 3          | 0.001906        | 17.604656                | VALID  |

## Notes

- The first iteration is excluded from average timing due to initialization overhead.
- All experiments are run under exclusive node allocation using SLURM (`salloc`).
- For best performance, tune matrix size, tile size, and unrolling based on hardware.

## Files

- `main`: Executable binary for matrix multiplication
- `run_performance.sh`: Performance benchmarking across multiple sizes
- `run_validation_*.sh`: Functional correctness testing across different node configurations
