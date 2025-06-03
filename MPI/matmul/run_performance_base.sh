#!/bin/bash

salloc -N 2 --exclusive                              \
  mpirun --bind-to none -mca btl ^openib -npernode 1 \
  ./main -t 32 -n 10 8192 8192 8192
