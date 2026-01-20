#!/bin/bash

# Load necessary modules for Intel MKL
module load tbb/2021.12
module load compiler-rt/2024.1.0
module load mkl/2024.1
module load gcc/13.2.0-gcc-8.5.0-sfeapnb

# Compile the code
gcc -o orthogonalization_exec -O3 orthogonalization.c -m64 -I"${MKLROOT}/include" -L"${MKLROOT}/lib" -lmkl_rt -Wl,--no-as-needed -lpthread -lm -ldl

echo "Compilation completed! Executable: orthogonalization_exec"
