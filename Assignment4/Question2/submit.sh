#!/bin/bash

#SBATCH -J Nilpotent_Job
#SBATCH -o ./%x.out
#SBATCH -e ./%x.err
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --get-user-env
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Load modules
module load tbb/2021.12
module load compiler-rt/2024.1.0
module load mkl/2024.1
module load gcc/13.2.0-gcc-8.5.0-sfeapnb

# Execute the program
./nilpotent_exec 1>./nilpotent.out 2>./nilpotent.err
