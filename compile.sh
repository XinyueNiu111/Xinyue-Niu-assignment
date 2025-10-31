#!/bin/bash
# Load environment
module load tbb/2021.12
module load compiler-rt/2024.1.0
module load mkl/2024.1
module load gcc/13.2.0-gcc-8.5.0-sfeapnb

# Compile Step 1
echo "Compiling Step 1..."
gcc -O3 1.c -o step1_vectors \
    -I"${MKLROOT}/include" -L"${MKLROOT}/lib" \
    -lmkl_rt -lpthread -lm -ldl

# Compile Step 2
echo "Compiling Step 2..."
gcc -O3 step2_orthogonalization.c -o step2_orthogonalization \
    -I"${MKLROOT}/include" -L"${MKLROOT}/lib" \
    -lmkl_rt -lpthread -lm -ldl

echo "Compilation finished successfully."

