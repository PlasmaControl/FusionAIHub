#!/bin/bash
#SBATCH -A fus187
#SBATCH -J eval_per_bin
#SBATCH -o logs/%j_eval_per_bin_stage1.out
#SBATCH -e logs/%j_eval_per_bin_stage1.err
#SBATCH -t 1:00:00
#SBATCH -p extended
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
set -e

# One-off experimental plot: Stage 1 best.pt applied to shot 200729
# with per-(channel, freq-bin) spec normalisation computed from THIS
# shot only. Saves a static PNG comparing GT vs pred spectrograms for
# ECE, CO2, BES on the highest-variance channel.
#
# Stage 1 was trained with channel-wise spec normalisation, so feeding
# per-bin normalised inputs is off-distribution — this is the
# experiment we want to see before committing to a full per-bin
# retraining.
#
# Usage:
#   sbatch scripts/slurm_frontier/eval_per_bin_stage1.sh \
#       [checkpoint] [shot_h5] [output_png]
#
# Defaults match the d=1024 / 48L Stage 1 best.pt and shot 200729.

CHECKPOINT="${1:-/lustre/orion/fus187/proj-shared/models/e2e_stage1_d1024_48L/e2e_stage1_best.pt}"
SHOT="${2:-/lustre/orion/fus187/proj-shared/foundation_model/200729_processed.h5}"
OUTPUT="${3:-eval_runs/animations/200729_per_bin_stage1.png}"

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi
if [ ! -f "$SHOT" ]; then
    echo "ERROR: shot file not found: $SHOT" >&2
    exit 1
fi

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${PROJECT_DIR}"
mkdir -p logs "$(dirname "$OUTPUT")"

export MASTER_PORT=29542
source scripts/slurm_frontier/_frontier_common.sh

echo "[eval_per_bin] checkpoint : $CHECKPOINT"
echo "[eval_per_bin] shot       : $SHOT"
echo "[eval_per_bin] output     : $OUTPUT"

python -u scripts/training/eval_per_bin_stage1.py \
    --checkpoint "$CHECKPOINT" \
    --shot "$SHOT" \
    --output "$OUTPUT" \
    --batch_size 8 \
    --num_workers 2

echo "[eval_per_bin] result: $OUTPUT"
