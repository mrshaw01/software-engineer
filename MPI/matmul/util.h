#pragma once

void splitIntoIntervals(int N, int M, int interval_id, int &begin, int &end);

void timer_start(int i);

double timer_stop(int i);

void check_matmul(float *A, float *B, float *C, int M, int N, int K);

void print_mat(float *m, int R, int C);

void alloc_mat(float **m, int R, int C);

void rand_mat(float *m, int R, int C);

void zero_mat(float *m, int R, int C);
