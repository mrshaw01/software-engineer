# !/bin/bash

salloc -N 2 --exclusive                              \
  mpirun --bind-to none -mca btl ^openib -npernode 2 \
  ./main -v -t 32 -n 3 293 399 123

salloc -N 2 --exclusive                              \
  mpirun --bind-to none -mca btl ^openib -npernode 2 \
  ./main -v -t 32 -n 3 3 699 10

salloc -N 2 --exclusive                              \
  mpirun --bind-to none -mca btl ^openib -npernode 2 \
  ./main -v -t 32 -n 3 331 21 129

salloc -N 2 --exclusive                              \
  mpirun --bind-to none -mca btl ^openib -npernode 2 \
  ./main -v -t 32 -n 3 2000 1 2000

salloc -N 2 --exclusive                              \
  mpirun --bind-to none -mca btl ^openib -npernode 2 \
  ./main -v -t 32 -n 3 323 429 111
  
salloc -N 2 --exclusive                              \
  mpirun --bind-to none -mca btl ^openib -npernode 2 \
  ./main -v -t 32 -n 3 1 2000 2000

salloc -N 2 --exclusive                              \
  mpirun --bind-to none -mca btl ^openib -npernode 2 \
  ./main -v -t 32 -n 3 2000 2000 1

salloc -N 2 --exclusive                              \
  mpirun --bind-to none -mca btl ^openib -npernode 2 \
  ./main -v -t 32 -n 3 64 64 64

salloc -N 2 --exclusive                              \
  mpirun --bind-to none -mca btl ^openib -npernode 2 \
  ./main -v -t 32 -n 3 128 128 128

salloc -N 2 --exclusive                              \
  mpirun --bind-to none -mca btl ^openib -npernode 2 \
  ./main -v -t 32 -n 3 256 256 256

