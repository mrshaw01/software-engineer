#!/bin/bash

salloc -N 2 --exclusive                              \
  mpirun --bind-to none -mca btl ^openib -npernode 1 \
  ./main -t 32 -n 10 8192 8192 8192

salloc -N 2 --exclusive                              \
  mpirun --bind-to none -mca btl ^openib -npernode 1 \
  ./main -t 32 -n 10 4096 4096 4096

salloc -N 2 --exclusive                              \
  mpirun --bind-to none -mca btl ^openib -npernode 1 \
  ./main -t 32 -n 10 2048 2048 2048

salloc -N 2 --exclusive                              \
  mpirun --bind-to none -mca btl ^openib -npernode 1 \
  ./main -t 32 -n 10 1024 1024 1024

salloc -N 2 --exclusive                              \
  mpirun --bind-to none -mca btl ^openib -npernode 1 \
  ./main -t 32 -n 10 5678 7891 1234

salloc -N 2 --exclusive                              \
  mpirun --bind-to none -mca btl ^openib -npernode 1 \
  ./main -t 32 -n 10 7891 1234 5678

salloc -N 2 --exclusive                              \
  mpirun --bind-to none -mca btl ^openib -npernode 1 \
  ./main -t 32 -n 10 1234 5678 7891
