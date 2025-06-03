# MPI Parallel Computation of π

## Mathematical Foundation

The integral approximation method for calculating π:

```
π = ∫₀¹ (4/(1+x²)) dx ≈ (1/n) * Σ [4/(1+xᵢ²)] for i=1 to n
```

where:

- `n` = number of intervals
- `xᵢ` = midpoint of the i-th interval

## MPI Implementation Strategy

### Process Workflow

1. **Master Process (Rank 0)**

   - Gets number of intervals `n` from user
   - Broadcasts `n` to all processes
   - Collects partial results via reduction
   - Computes and displays final π approximation

2. **Worker Processes**
   - Receive `n` via broadcast
   - Compute their portion of the integral
   - Contribute to final reduction

### Key MPI Operations Used

- `MPI_Bcast`: Distribute `n` to all processes
- `MPI_Reduce`: Combine partial results with summation

## Code Implementation

```c
#include "mpi.h"
#include <math.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    int n, myid, numprocs, i;
    double PI25DT = 3.141592653589793238462643;
    double mypi, pi, h, sum, x;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    while (1) {
        if (myid == 0) {
            printf("Enter the number of intervals: (0 quits) ");
            scanf("%d", &n);
        }
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (n == 0) break;

        h = 1.0 / (double)n;
        sum = 0.0;
        for (i = myid + 1; i <= n; i += numprocs) {
            x = h * ((double)i - 0.5);
            sum += 4.0 / (1.0 + x * x);
        }
        mypi = h * sum;

        MPI_Reduce(&mypi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (myid == 0) {
            printf("pi is approximately %.16f, Error is %.16f\n",
                   pi, fabs(pi - PI25DT));
        }
    }

    MPI_Finalize();
    return 0;
}
```

## Parallelization Approach

- **Work Distribution**: Each process computes every `numprocs`-th interval
  - Process `k` computes intervals: k+1, k+1+numprocs, k+1+2\*numprocs, etc.
- **Numerical Integration**: Midpoint rule used for each interval
- **Result Aggregation**: All partial sums combined using `MPI_SUM` reduction

## Performance Considerations

1. **Load Balancing**: Equal work distribution when `n` is divisible by `numprocs`
2. **Communication Overhead**: Single broadcast and reduction operations
3. **Numerical Accuracy**: More intervals → better approximation of π

## Example Execution

```
Enter the number of intervals: (0 quits) 1000000
pi is approximately 3.1415926535897936, Error is 0.0000000000000000
```
