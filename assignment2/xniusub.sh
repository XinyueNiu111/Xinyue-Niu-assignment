#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=echo_output.txt
#SBATCH --error=echo_error.txt

echo "Hello!"
