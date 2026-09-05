#!/bin/bash
#SBATCH --job-name=ae-mask-vs-ann
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out

# Task 6: measure big_tf_unet's coherent mask against the 180 annotated AE shots.
#
# Environment: the tokeye repo's own venv, used read-only (torch 2.9.1+cu128,
# tokeye importable). pyarrow is not installed there and the tokeye tree is
# read-only, so pyarrow lives in a --target directory on group storage that is
# put on PYTHONPATH. Nothing is written into /scratch/gpfs/nc1514/tokeye.

set -euo pipefail

REPO=/scratch/gpfs/nc1514/FusionAIHub
PY=/scratch/gpfs/nc1514/tokeye/.venv/bin/python
export PYTHONPATH=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/ae/pylibs
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1   # the checkpoint is loaded from a local path

nvidia-smi || true
echo "job $SLURM_JOB_ID on $(hostname) at $(date -Is)"

srun "$PY" "$REPO/outputs/labelmaker/ae/scripts/mask_vs_annotation.py" \
    --stage all \
    --device cuda \
    --width 0 \
    --out-root /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/ae \
    --fig-dir "$REPO/outputs/labelmaker/ae/figures" \
    --json-out "$REPO/outputs/labelmaker/ae/mask_vs_annotation.json" \
    --dpi 130

echo "finished at $(date -Is)"
