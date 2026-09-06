#!/bin/bash
#SBATCH --job-name=ae-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out

# Task 7b: score every ae_seldnet_*.pt on the 60 validation shots and draw the
# six-shot figure set. One whole-record forward pass per shot per checkpoint;
# measured at ~2 min per checkpoint on an A100, so an hour is generous.

set -euo pipefail

ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker
REPO=/scratch/gpfs/nc1514/FusionAIHub
export PYTHONPATH=$REPO/src
export LABELMAKER_GIT_SHA=$(git -C "$REPO" rev-parse HEAD)
export OMP_NUM_THREADS=4
export MPLCONFIGDIR=$ROOT/runs/mplconfig

echo "job ${SLURM_JOB_ID} on $(hostname) at $(date -Is)"

srun "$ROOT/envs/phase3/bin/python" -u "$REPO/scripts/labelmaker/ae_evaluate.py" \
  --dataset-dir "$ROOT/ae/dataset" \
  --models-dir "$ROOT/models/d3d_ae_activity_seldnet" \
  --fig-dir "$REPO/outputs/labelmaker/ae/training"

echo "finished at $(date -Is)"
