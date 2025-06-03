#include "matmul.h"
#include "util.h"
#include <cstdio>
#include <cstdlib>
#include <immintrin.h>
#include <math.h>
#include <mpi.h>

#define ITILESIZE (96)
#define JTILESIZE (512)
#define KTILESIZE (768)

typedef struct {
    float *_A;
    float *_B;
    float *_C;
    int _M;
    int _N;
    int _K;
    int _mpi_rank;
} ThreadParams;

void *matmul_thread(void *params) {
    ThreadParams *threadParams = (ThreadParams *)params;
    float *A = threadParams->_A, *B = threadParams->_B, *C = threadParams->_C;
    int M = threadParams->_M, N = threadParams->_N, K = threadParams->_K;

    if (threadParams->_mpi_rank)
        zero_mat(C, M, N);

    int kk, ii, jj;
    for (kk = 0; kk < K; kk += KTILESIZE)
        for (ii = 0; ii < M; ii += ITILESIZE)
            for (jj = 0; jj < N; jj += JTILESIZE) {
                int k, i, j;
                int boundk = std::min(kk + KTILESIZE, K);
                int boundi = std::min(ii + ITILESIZE, M);
                int boundj = std::min(jj + JTILESIZE, N);
                for (k = kk; k < boundk - 5; k += 6)
                    for (i = ii; i < boundi; i++) {
                        __m256 a0 = _mm256_set1_ps(A[i * K + k + 0]);
                        __m256 a1 = _mm256_set1_ps(A[i * K + k + 1]);
                        __m256 a2 = _mm256_set1_ps(A[i * K + k + 2]);
                        __m256 a3 = _mm256_set1_ps(A[i * K + k + 3]);
                        __m256 a4 = _mm256_set1_ps(A[i * K + k + 4]);
                        __m256 a5 = _mm256_set1_ps(A[i * K + k + 5]);
                        for (j = jj; j < boundj - 7; j += 8) {
                            __m256 b0 = _mm256_loadu_ps(B + (k + 0) * N + j);
                            __m256 b1 = _mm256_loadu_ps(B + (k + 1) * N + j);
                            __m256 b2 = _mm256_loadu_ps(B + (k + 2) * N + j);
                            __m256 b3 = _mm256_loadu_ps(B + (k + 3) * N + j);
                            __m256 b4 = _mm256_loadu_ps(B + (k + 4) * N + j);
                            __m256 b5 = _mm256_loadu_ps(B + (k + 5) * N + j);
                            __m256 c = _mm256_loadu_ps(C + i * N + j);
                            c = _mm256_fmadd_ps(a0, b0, c);
                            c = _mm256_fmadd_ps(a1, b1, c);
                            c = _mm256_fmadd_ps(a2, b2, c);
                            c = _mm256_fmadd_ps(a3, b3, c);
                            c = _mm256_fmadd_ps(a4, b4, c);
                            c = _mm256_fmadd_ps(a5, b5, c);
                            _mm256_storeu_ps(C + i * N + j, c);
                        }
                        for (; j < boundj; j++) {
                            C[i * N + j] += A[i * K + (k + 0)] * B[(k + 0) * N + j];
                            C[i * N + j] += A[i * K + (k + 1)] * B[(k + 1) * N + j];
                            C[i * N + j] += A[i * K + (k + 2)] * B[(k + 2) * N + j];
                            C[i * N + j] += A[i * K + (k + 3)] * B[(k + 3) * N + j];
                            C[i * N + j] += A[i * K + (k + 4)] * B[(k + 4) * N + j];
                            C[i * N + j] += A[i * K + (k + 5)] * B[(k + 5) * N + j];
                        }
                    }
                for (; k < boundk; k++)
                    for (i = ii; i < boundi; i++)
                        for (j = jj; j < boundj; j++)
                            C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
    pthread_exit(NULL);
}

void matmul(float *_A, float *_B, float *_C, int _M, int _N, int _K, int _num_threads, int _mpi_rank,
            int _mpi_world_size) {
    // Calculate the size of each interval
    int M_start, M_end;
    splitIntoIntervals(_M, _mpi_world_size, _mpi_rank, M_start, M_end);
    MPI_Request ireq;

    if (_mpi_rank == 0) {
        for (int i = 1; i < _mpi_world_size; i++) {
            int start, end;
            splitIntoIntervals(_M, _mpi_world_size, i, start, end);
            if (end - start > 0) {
                MPI_Isend(_B, _K * _N, MPI_FLOAT, i, 8888, MPI_COMM_WORLD, &ireq);
                MPI_Isend(_A + start * _K, (end - start) * _K, MPI_FLOAT, i, 9999, MPI_COMM_WORLD, &ireq);
            }
        }
    } else if (M_end - M_start > 0) {
        MPI_Recv(_B, _K * _N, MPI_FLOAT, 0, 8888, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(_A, (M_end - M_start) * _K, MPI_FLOAT, 0, 9999, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (M_end - M_start > 0) {
        // Local matrix multiplication
        pthread_t threads[_num_threads];
        ThreadParams params[_num_threads];
        pthread_attr_t attr;
        cpu_set_t cpu;

        // Initialize and start each thread
        for (int i = 0; i < _num_threads; ++i) {
            int start, end;
            splitIntoIntervals(M_end - M_start, _num_threads, i, start, end);
            if (end - start > 0) {
                params[i]._A = _A + start * _K;
                params[i]._B = _B;
                params[i]._C = _C + start * _N;
                params[i]._M = end - start;
                params[i]._N = _N;
                params[i]._K = _K;
                params[i]._mpi_rank = _mpi_rank;
                CPU_ZERO(&cpu);
                CPU_SET(i, &cpu);
                pthread_attr_init(&attr); // Initialize thread attributes
                pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpu);
                pthread_create(&threads[i], &attr, matmul_thread, (void *)&params[i]);
            }
        }
        // Wait for all threads to finish
        for (int i = 0; i < _num_threads; ++i) {
            int start, end;
            splitIntoIntervals(M_end - M_start, _num_threads, i, start, end);
            if (end - start > 0)
                pthread_join(threads[i], NULL);
        }
    }

    if (_mpi_rank == 0) {
        for (int i = 1; i < _mpi_world_size; i++) {
            int start, end;
            splitIntoIntervals(_M, _mpi_world_size, i, start, end);
            if (end - start > 0)
                MPI_Recv(_C + start * _N, (end - start) * _N, MPI_FLOAT, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else if (M_end - M_start > 0)
        MPI_Isend(_C, (M_end - M_start) * _N, MPI_FLOAT, 0, _mpi_rank, MPI_COMM_WORLD, &ireq);
}
