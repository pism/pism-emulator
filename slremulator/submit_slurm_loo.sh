#!/bin/bash

# Loop over all kernels and submit scripts
for kernel in "exp" "mat32" "mat52" "expquad"; do
    sbatch create_slurm_loo.sh $method
done
