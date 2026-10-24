#!/bin/bash
#SBATCH --job-name=elm_dsm
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/runs/slurm/%A_%a.out
# CPU only. The graph is 124 -> 128 -> three (128, 3) heads in float64; the fork's
# loop is dense matmuls that thread well and never needed a GPU. The 32 GB is for
# the 1.57 GB split pickle plus the float64 copies of it, not for the model.
#
# Array task 0 is the `all124` column set (upstream's inputs), task 1 is `no_bes`
# (the 60 non-BES columns of the split pickle - slots 0-11 and 76-123 of
# new_diagnostic_order - the set the FAITH corpus can actually fill). Same rows,
# same split, same seed: only the input width differs.
set -euo pipefail

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
REPO=/scratch/gpfs/nc1514/FusionAIHub
ENV="${LABELER_PIXI_ENV:-/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labeler}"  # another cluster: export LABELER_PIXI_ENV
# The cluster's production root (labeler.env.CLUSTERS), whatever LABELER_ROOT the caller
# exports: the phase3 interpreter and this job's outputs live there.
ROOT="$(env -u LABELER_ROOT "$ENV/bin/python" "$REPO/src/labeler/env.py" LABELER_ROOT "/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler")"
export LABELER_GIT_SHA=$(git -C "$REPO" rev-parse HEAD)
export PYTHONPATH="$REPO/src"

SETS=(all124 no_bes)
COLUMN_SET=${SETS[$SLURM_ARRAY_TASK_ID]}

mkdir -p "$ROOT/runs/slurm"

srun "$ROOT/envs/phase3/bin/python" -u "$REPO/scripts/labeler/elm_dsm_train.py" \
  --out-dir "$ROOT/models/d3d_elm_time_to_event_dsm" \
  --column-set "$COLUMN_SET" \
  --iters 1000 --max-hours 4.5 \
  --k 3 --hidden 128 --dropout 0.2 --lr 1e-3 --batch-size 1024 --seed 0
