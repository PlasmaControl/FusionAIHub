#!/bin/bash
#SBATCH --job-name=elm-tiers-cpu
#SBATCH --account=nstx-u
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/runs/slurm/%x_%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/runs/slurm/%x_%j.out

# CPU steps of scripts/labeler/elm_tiers.py (refine, evaluate); the arguments are passed through.
#   sbatch scripts/labeler/elm_tiers_cpu.sbatch refine
set -euo pipefail
REPO=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub
export LABELER_ROOT="${LABELER_ROOT:-$REPO/.cache/labeler}"
export ELM_TIERS_ROOT="${ELM_TIERS_ROOT:-$LABELER_ROOT/elm_tiers}"
export PYTHONPATH="$REPO/src" LABELER_NO_FETCH=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"
echo "job ${SLURM_JOB_ID:-none} on $(hostname) at $(date -Is): $*"
"$REPO/.pixi/envs/stellarai/bin/python" -u scripts/labeler/elm_tiers.py "$@"
echo "finished at $(date -Is)"
