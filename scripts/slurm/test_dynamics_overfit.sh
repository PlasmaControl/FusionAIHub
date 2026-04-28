#!/bin/bash
#SBATCH --job-name=dyn_overfit
#SBATCH --output=logs/%j_dyn_overfit.out
#SBATCH --error=logs/%j_dyn_overfit.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=4G

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

srun pixi run python ../training/test_dynamics_overfit_rollout.py
