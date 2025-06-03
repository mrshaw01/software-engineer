#include "util.h"
#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <thread>

#include "convolution.cuh"

#define CHECK_CUDA(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t status_ = call;                                                                                    \
        if (status_ != cudaSuccess) {                                                                                  \
            fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, cudaGetErrorName(status_),              \
                    cudaGetErrorString(status_));                                                                      \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

#define NGPU 4
static half *I_gpu[NGPU], *A_gpu[NGPU], *B_gpu[NGPU];
static float *C_gpu[NGPU];
#define DEFAULT_BLOCKS 16
#define DIM 768
#define TILE_SIZE 16

__global__ void gpu_im2col(half *I_gpu, half *B_gpu, int N, int C, int H, int W, int R, int S, int pad_h, int pad_w,
                           int stride_h, int stride_w, int dilation_h, int dilation_w) {
    const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
    const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N * OH * OW)
        return;
    const int n = tid / (OH * OW);
    const int oh = (tid / OW) % OH;
    const int ow = tid % OW;

#pragma unroll
    for (int c = 0; c < C; ++c)
        for (int r = 0; r < R; ++r)
            for (int s = 0; s < S; ++s) {
                const int h = oh * stride_h - pad_h + r * dilation_h;
                const int w = ow * stride_w - pad_w + s * dilation_w;
                if (h >= 0 && h < H && w >= 0 && w < W)
                    B_gpu[((c * R * S) + (r * S) + s) * (N * OH * OW) + (n * OH * OW + oh * OW + ow)] =
                        I_gpu[n * C * H * W + c * H * W + h * W + w];
            }
}

__global__ void gpu_matmul_kernel(half *A, half *B, float *C, int M, int N, int K, int OHW) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE * 6 + ty;
    int col = bx * TILE_SIZE * 6 + tx;

    __shared__ float As[TILE_SIZE * 6][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE * 6];

    float sum[6][6] = {0.0f};
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        As[ty + 0 * TILE_SIZE][tx] = (row + 0 * TILE_SIZE < M && t * TILE_SIZE + tx < K)
                                         ? A[(row + 0 * TILE_SIZE) * K + TILE_SIZE * t + tx]
                                         : (half)0.0f;
        As[ty + 1 * TILE_SIZE][tx] = (row + 1 * TILE_SIZE < M && t * TILE_SIZE + tx < K)
                                         ? A[(row + 1 * TILE_SIZE) * K + TILE_SIZE * t + tx]
                                         : (half)0.0f;
        As[ty + 2 * TILE_SIZE][tx] = (row + 2 * TILE_SIZE < M && t * TILE_SIZE + tx < K)
                                         ? A[(row + 2 * TILE_SIZE) * K + TILE_SIZE * t + tx]
                                         : (half)0.0f;
        As[ty + 3 * TILE_SIZE][tx] = (row + 3 * TILE_SIZE < M && t * TILE_SIZE + tx < K)
                                         ? A[(row + 3 * TILE_SIZE) * K + TILE_SIZE * t + tx]
                                         : (half)0.0f;
        As[ty + 4 * TILE_SIZE][tx] = (row + 4 * TILE_SIZE < M && t * TILE_SIZE + tx < K)
                                         ? A[(row + 4 * TILE_SIZE) * K + TILE_SIZE * t + tx]
                                         : (half)0.0f;
        As[ty + 5 * TILE_SIZE][tx] = (row + 5 * TILE_SIZE < M && t * TILE_SIZE + tx < K)
                                         ? A[(row + 5 * TILE_SIZE) * K + TILE_SIZE * t + tx]
                                         : (half)0.0f;

        Bs[ty][tx + 0 * TILE_SIZE] = (col + 0 * TILE_SIZE < N && t * TILE_SIZE + ty < K)
                                         ? B[(TILE_SIZE * t + ty) * N + col + 0 * TILE_SIZE]
                                         : (half)0.0f;
        Bs[ty][tx + 1 * TILE_SIZE] = (col + 1 * TILE_SIZE < N && t * TILE_SIZE + ty < K)
                                         ? B[(TILE_SIZE * t + ty) * N + col + 1 * TILE_SIZE]
                                         : (half)0.0f;
        Bs[ty][tx + 2 * TILE_SIZE] = (col + 2 * TILE_SIZE < N && t * TILE_SIZE + ty < K)
                                         ? B[(TILE_SIZE * t + ty) * N + col + 2 * TILE_SIZE]
                                         : (half)0.0f;
        Bs[ty][tx + 3 * TILE_SIZE] = (col + 3 * TILE_SIZE < N && t * TILE_SIZE + ty < K)
                                         ? B[(TILE_SIZE * t + ty) * N + col + 3 * TILE_SIZE]
                                         : (half)0.0f;
        Bs[ty][tx + 4 * TILE_SIZE] = (col + 4 * TILE_SIZE < N && t * TILE_SIZE + ty < K)
                                         ? B[(TILE_SIZE * t + ty) * N + col + 4 * TILE_SIZE]
                                         : (half)0.0f;
        Bs[ty][tx + 5 * TILE_SIZE] = (col + 5 * TILE_SIZE < N && t * TILE_SIZE + ty < K)
                                         ? B[(TILE_SIZE * t + ty) * N + col + 5 * TILE_SIZE]
                                         : (half)0.0f;

        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum[0][0] += (float)(As[ty + 0 * TILE_SIZE][k] * Bs[k][tx + 0 * TILE_SIZE]);
            sum[0][1] += (float)(As[ty + 0 * TILE_SIZE][k] * Bs[k][tx + 1 * TILE_SIZE]);
            sum[0][2] += (float)(As[ty + 0 * TILE_SIZE][k] * Bs[k][tx + 2 * TILE_SIZE]);
            sum[0][3] += (float)(As[ty + 0 * TILE_SIZE][k] * Bs[k][tx + 3 * TILE_SIZE]);
            sum[0][4] += (float)(As[ty + 0 * TILE_SIZE][k] * Bs[k][tx + 4 * TILE_SIZE]);
            sum[0][5] += (float)(As[ty + 0 * TILE_SIZE][k] * Bs[k][tx + 5 * TILE_SIZE]);

            sum[1][0] += (float)(As[ty + 1 * TILE_SIZE][k] * Bs[k][tx + 0 * TILE_SIZE]);
            sum[1][1] += (float)(As[ty + 1 * TILE_SIZE][k] * Bs[k][tx + 1 * TILE_SIZE]);
            sum[1][2] += (float)(As[ty + 1 * TILE_SIZE][k] * Bs[k][tx + 2 * TILE_SIZE]);
            sum[1][3] += (float)(As[ty + 1 * TILE_SIZE][k] * Bs[k][tx + 3 * TILE_SIZE]);
            sum[1][4] += (float)(As[ty + 1 * TILE_SIZE][k] * Bs[k][tx + 4 * TILE_SIZE]);
            sum[1][5] += (float)(As[ty + 1 * TILE_SIZE][k] * Bs[k][tx + 5 * TILE_SIZE]);

            sum[2][0] += (float)(As[ty + 2 * TILE_SIZE][k] * Bs[k][tx + 0 * TILE_SIZE]);
            sum[2][1] += (float)(As[ty + 2 * TILE_SIZE][k] * Bs[k][tx + 1 * TILE_SIZE]);
            sum[2][2] += (float)(As[ty + 2 * TILE_SIZE][k] * Bs[k][tx + 2 * TILE_SIZE]);
            sum[2][3] += (float)(As[ty + 2 * TILE_SIZE][k] * Bs[k][tx + 3 * TILE_SIZE]);
            sum[2][4] += (float)(As[ty + 2 * TILE_SIZE][k] * Bs[k][tx + 4 * TILE_SIZE]);
            sum[2][5] += (float)(As[ty + 2 * TILE_SIZE][k] * Bs[k][tx + 5 * TILE_SIZE]);

            sum[3][0] += (float)(As[ty + 3 * TILE_SIZE][k] * Bs[k][tx + 0 * TILE_SIZE]);
            sum[3][1] += (float)(As[ty + 3 * TILE_SIZE][k] * Bs[k][tx + 1 * TILE_SIZE]);
            sum[3][2] += (float)(As[ty + 3 * TILE_SIZE][k] * Bs[k][tx + 2 * TILE_SIZE]);
            sum[3][3] += (float)(As[ty + 3 * TILE_SIZE][k] * Bs[k][tx + 3 * TILE_SIZE]);
            sum[3][4] += (float)(As[ty + 3 * TILE_SIZE][k] * Bs[k][tx + 4 * TILE_SIZE]);
            sum[3][5] += (float)(As[ty + 3 * TILE_SIZE][k] * Bs[k][tx + 5 * TILE_SIZE]);

            sum[4][0] += (float)(As[ty + 4 * TILE_SIZE][k] * Bs[k][tx + 0 * TILE_SIZE]);
            sum[4][1] += (float)(As[ty + 4 * TILE_SIZE][k] * Bs[k][tx + 1 * TILE_SIZE]);
            sum[4][2] += (float)(As[ty + 4 * TILE_SIZE][k] * Bs[k][tx + 2 * TILE_SIZE]);
            sum[4][3] += (float)(As[ty + 4 * TILE_SIZE][k] * Bs[k][tx + 3 * TILE_SIZE]);
            sum[4][4] += (float)(As[ty + 4 * TILE_SIZE][k] * Bs[k][tx + 4 * TILE_SIZE]);
            sum[4][5] += (float)(As[ty + 4 * TILE_SIZE][k] * Bs[k][tx + 5 * TILE_SIZE]);

            sum[5][0] += (float)(As[ty + 5 * TILE_SIZE][k] * Bs[k][tx + 0 * TILE_SIZE]);
            sum[5][1] += (float)(As[ty + 5 * TILE_SIZE][k] * Bs[k][tx + 1 * TILE_SIZE]);
            sum[5][2] += (float)(As[ty + 5 * TILE_SIZE][k] * Bs[k][tx + 2 * TILE_SIZE]);
            sum[5][3] += (float)(As[ty + 5 * TILE_SIZE][k] * Bs[k][tx + 3 * TILE_SIZE]);
            sum[5][4] += (float)(As[ty + 5 * TILE_SIZE][k] * Bs[k][tx + 4 * TILE_SIZE]);
            sum[5][5] += (float)(As[ty + 5 * TILE_SIZE][k] * Bs[k][tx + 5 * TILE_SIZE]);
        }
        __syncthreads();
    }

    for (int i = 0; i < 6; i++) {
        int rowi = row + i * TILE_SIZE;
        for (int j = 0; j < 6; j++) {
            int colj = col + j * TILE_SIZE;
            if (rowi < M && colj < N)
                C[((colj / OHW) * M + rowi) * OHW + (colj % OHW)] = sum[i][j];
        }
    }
}

void convolution_thread(half *_I, half *_F, float *_O, int N, int C, int H, int W, int K, int R, int S, int pad_h,
                        int pad_w, int stride_h, int stride_w, int dilation_h, int dilation_w, int gpu_id) {
    cudaSetDevice(gpu_id);
    int G_Nbegin, G_Nend;
    splitIntoIntervals(N, NGPU, gpu_id, G_Nbegin, G_Nend);
    int G_Nsub = G_Nend - G_Nbegin;

    half *I = _I, *F = _F;
    float *O = _O;
    const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
    const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;

    const int MM = K;
    const int KK = C * R * S;

    // Create stream_F
    cudaStream_t stream_F;
    cudaStreamCreate(&stream_F);
    cudaMemcpyAsync(A_gpu[gpu_id], F, sizeof(half) * MM * KK, cudaMemcpyHostToDevice, stream_F);

    // Define the number of blocks
    int BLOCKS = min(G_Nsub, DEFAULT_BLOCKS);
    int Nbegin[BLOCKS], Nend[BLOCKS];

    // Create streams
    cudaStream_t streams[BLOCKS];
    for (int i = 0; i < BLOCKS; ++i) {
        splitIntoIntervals(G_Nsub, BLOCKS, i, Nbegin[i], Nend[i]);
        int Nsub = Nend[i] - Nbegin[i];
        cudaStreamCreate(&streams[i]);
        // Data transfer: A_gpu and I_gpu
        cudaMemcpyAsync(&I_gpu[gpu_id][Nbegin[i] * C * H * W], &I[(G_Nbegin + Nbegin[i]) * C * H * W],
                        sizeof(half) * Nsub * C * H * W, cudaMemcpyHostToDevice, streams[i]);
        // Kernel launch: im2col
        dim3 bDim(DIM);
        dim3 gDim((Nsub * OH * OW + DIM - 1) / DIM);
        gpu_im2col<<<gDim, bDim, 0, streams[i]>>>(&I_gpu[gpu_id][Nbegin[i] * C * H * W],
                                                  &B_gpu[gpu_id][C * R * S * Nbegin[i] * OH * OW], Nsub, C, H, W, R, S,
                                                  pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);
    }
    for (int i = 0; i < BLOCKS; ++i) {
        int Nsub = Nend[i] - Nbegin[i];
        int NNsub = Nsub * OH * OW;
        // Synchronization to get A
        if (i == 0)
            cudaStreamSynchronize(stream_F);
        // Kernel launch: matmul
        dim3 blockDim(TILE_SIZE, TILE_SIZE);
        dim3 gridDim((NNsub + 6 * TILE_SIZE - 1) / (6 * TILE_SIZE), (MM + 6 * TILE_SIZE - 1) / (6 * TILE_SIZE));
        gpu_matmul_kernel<<<gridDim, blockDim, 0, streams[i]>>>(
            A_gpu[gpu_id], &B_gpu[gpu_id][C * R * S * Nbegin[i] * OH * OW], &C_gpu[gpu_id][K * Nbegin[i] * OH * OW], MM,
            NNsub, KK, OH * OW);
        // Data transfer: C_gpu to O
        cudaMemcpyAsync(&O[K * (G_Nbegin + Nbegin[i]) * OH * OW], &C_gpu[gpu_id][K * Nbegin[i] * OH * OW],
                        sizeof(float) * MM * NNsub, cudaMemcpyDeviceToHost, streams[i]);
    }

    cudaStreamDestroy(stream_F);
    // Synchronize all streams
    for (int i = 0; i < BLOCKS; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
}

void convolution(half *_I, half *_F, float *_O, int N, int C, int H, int W, int K, int R, int S, int pad_h, int pad_w,
                 int stride_h, int stride_w, int dilation_h, int dilation_w, int _mpi_rank, int _mpi_world_size) {
    const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
    const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;

    int mpi_rank = _mpi_rank;
    int mpi_world_size = _mpi_world_size;
    int mpi_Nbegin, mpi_Nend;
    splitIntoIntervals(N, mpi_world_size, mpi_rank, mpi_Nbegin, mpi_Nend);

    MPI_Datatype mpi_type_float16;
    MPI_Type_contiguous(2, MPI_BYTE, &mpi_type_float16);
    MPI_Type_commit(&mpi_type_float16);

    MPI_Request sendF, sendI;

    if (mpi_rank == 0) {
        for (int i = 1; i < mpi_world_size; i++) {
            int begin, end;
            splitIntoIntervals(N, mpi_world_size, i, begin, end);
            if (end > begin) {
                MPI_Isend(_F, K * C * R * S, mpi_type_float16, i, 7777, MPI_COMM_WORLD, &sendF);
                MPI_Isend(&_I[begin * C * H * W], (end - begin) * C * H * W, mpi_type_float16, i, 8888, MPI_COMM_WORLD,
                          &sendI);
            }
        }
    } else if (mpi_Nend > mpi_Nbegin) {
        MPI_Recv(_F, K * C * R * S, mpi_type_float16, 0, 7777, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(_I, (mpi_Nend - mpi_Nbegin) * C * H * W, mpi_type_float16, 0, 8888, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (mpi_Nend > mpi_Nbegin) {
        std::thread threads[NGPU];
        for (int i = 0; i < NGPU; i++)
            threads[i] = std::thread(convolution_thread, _I, _F, _O, (mpi_Nend - mpi_Nbegin), C, H, W, K, R, S, pad_h,
                                     pad_w, stride_h, stride_w, dilation_h, dilation_w, i);
        for (int i = 0; i < NGPU; i++)
            threads[i].join();
    }

    MPI_Request sendbackO;
    if (mpi_rank != 0) {
        if (mpi_Nend > mpi_Nbegin)
            MPI_Isend(_O, (mpi_Nend - mpi_Nbegin) * K * OH * OW, MPI_FLOAT, 0, 9999, MPI_COMM_WORLD, &sendbackO);
    } else {
        for (int i = 1; i < mpi_world_size; i++) {
            int begin, end;
            splitIntoIntervals(N, mpi_world_size, i, begin, end);
            if (end > begin)
                MPI_Recv(&_O[begin * K * OH * OW], (end - begin) * K * OH * OW, MPI_FLOAT, i, 9999, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
        }
    }
}

void convolution_initialize(int N, int C, int H, int W, int K, int R, int S, int pad_h, int pad_w, int stride_h,
                            int stride_w, int dilation_h, int dilation_w, int _mpi_rank, int _mpi_world_size) {
    int mpi_rank = _mpi_rank;
    int mpi_world_size = _mpi_world_size;
    int mpi_Nbegin, mpi_Nend;
    splitIntoIntervals(N, mpi_world_size, mpi_rank, mpi_Nbegin, mpi_Nend);
    if (mpi_Nend > mpi_Nbegin) {
        const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
        const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;

        const int MM = K;
        const int KK = C * R * S;

        for (int i = 0; i < NGPU; i++) {
            cudaSetDevice(i);
            int G_Nbegin, G_Nend;
            splitIntoIntervals(mpi_Nend - mpi_Nbegin, NGPU, i, G_Nbegin, G_Nend);
            int G_Nsub = G_Nend - G_Nbegin;
            int G_NNsub = G_Nsub * OH * OW;
            CHECK_CUDA(cudaMalloc(&I_gpu[i], sizeof(half) * G_Nsub * C * H * W));
            CHECK_CUDA(cudaMalloc(&A_gpu[i], sizeof(half) * MM * KK));
            CHECK_CUDA(cudaMalloc(&B_gpu[i], sizeof(half) * KK * G_NNsub));
            CHECK_CUDA(cudaMalloc(&C_gpu[i], sizeof(float) * MM * G_NNsub));
        }
    }
}

void convolution_cleanup(half *_I, half *_F, float *_O, int N, int C, int H, int W, int K, int R, int S, int pad_h,
                         int pad_w, int stride_h, int stride_w, int dilation_h, int dilation_w) {
    for (int i = 0; i < NGPU; i++) {
        CHECK_CUDA(cudaFree(I_gpu[i]));
        CHECK_CUDA(cudaFree(A_gpu[i]));
        CHECK_CUDA(cudaFree(B_gpu[i]));
        CHECK_CUDA(cudaFree(C_gpu[i]));
    }
}
