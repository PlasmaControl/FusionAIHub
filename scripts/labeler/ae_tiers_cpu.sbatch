#!/bin/bash
#SBATCH --job-name=ae-tiers-cpu
#SBATCH --account=nstx-u
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/runs/slurm/%x_%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/runs/slurm/%x_%j.out

# CPU steps of scripts/labeler/ae_tiers.py (refine, audit, evaluate); arguments passed through.
#   sbatch scripts/labeler/ae_tiers_cpu.sbatch refine --pred teachers --oof 'teacher_f*' --name r1
set -euo pipefail
REPO=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub
export LABELER_ROOT="${LABELER_ROOT:-$REPO/.cache/labeler}"
export AE_TIERS_ROOT="${AE_TIERS_ROOT:-$LABELER_ROOT/ae_tiers}"
export PYTHONPATH="$REPO/src" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
cd "$REPO"
echo "job ${SLURM_JOB_ID:-none} on $(hostname) at $(date -Is): $*"
"$REPO/.pixi/envs/stellarai/bin/python" -u scripts/labeler/ae_tiers.py --root "$AE_TIERS_ROOT" "$@"
echo "finished at $(date -Is)"
