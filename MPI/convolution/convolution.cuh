#pragma once

#include <cuda_fp16.h>

void convolution(half *_I, half *_F, float *_O, int N, int C, int H, int W, int K, int R, int S, int pad_h, int pad_w,
                 int stride_h, int stride_w, int dilation_h, int dilation_w, int _mpi_rank, int _mpi_world_size);

void convolution_initialize(int N, int C, int H, int W, int K, int R, int S, int pad_h, int pad_w, int stride_h,
                            int stride_w, int dilation_h, int dilation_w, int _mpi_rank, int _mpi_world_size);

void convolution_cleanup(half *_I, half *_F, float *_O, int N, int C, int H, int W, int K, int R, int S, int pad_h,
                         int pad_w, int stride_h, int stride_w, int dilation_h, int dilation_w);
