#!/bin/bash
#SBATCH -J step2_orthogonalization      
#SBATCH -o ./step2_orthogonalization.out  
#SBATCH -e ./step2_orthogonalization.err 
#SBATCH --partition=compute               
#SBATCH --nodes=1                         
#SBATCH --ntasks=1                        
#SBATCH --time=00:05:00                   

# load
module load tbb/2021.12
module load compiler-rt/2024.1.0
module load mkl/2024.1
module load gcc/13.2.0-gcc-8.5.0-sfeapnb

# compile
echo "Compiling..."
gcc -O3 step2_orthogonalization.c -o step2_orthogonalization \
    -I"${MKLROOT}/include" -L"${MKLROOT}/lib" \
    -lmkl_rt -lpthread -lm -ldl

# run
echo "Running..."
./step2_orthogonalization

echo "Job finished."

