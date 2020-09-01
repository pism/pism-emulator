#!/bin/bash
#SBATCH --partition=t2sma
ll
#SBATCH --ntasks=28
#SBATCH --tasks-per-node=28
#SBATCH --time=6:00:00
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=pism-loo.%j

umask 007

cd $SLURM_SUBMIT_DIR

ulimit -l unlimited
ulimit -s unlimited
ulimit

chmod u+x run_validation_loo.py

method=$1
/usr/bin/scl enable rh-python36 ./run_validation_loo.py --n_procs 28 -m $method
/usr/bin/scl enable rh-python36 ./run_validation_loo.py --n_procs 28 -m $method --step_bic
