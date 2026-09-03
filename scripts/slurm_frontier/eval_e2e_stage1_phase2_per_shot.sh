#!/bin/bash
#SBATCH -A fus187
#SBATCH -J eval_s1_p2_1
#SBATCH -o logs/%j_eval_e2e_stage1_phase2_per_shot.out
#SBATCH -e logs/%j_eval_e2e_stage1_phase2_per_shot.err
#SBATCH -t 1:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
set -e

# Phase 2.1 per-shot summary plots. Single-GPU re-inference on the
# (top-N + bottom-N) shots selected by Phase 1's top_bottom_shots.csv.gz,
# then renders a 2×2 grid per (shot, modality):
#   TL = per-window MAE timeseries, TR = best window GT/pred,
#   BL = worst window GT/pred,      BR = MAE histogram.
#
# Usage (positional):
#   sbatch scripts/slurm_frontier/eval_e2e_stage1_phase2_per_shot.sh \
#       eval_runs/stage1_phase1_e2e_stage1_best_4609988 \
#       /lustre/orion/fus187/proj-shared/models/e2e_stage1/e2e_stage1_best.pt
#
# Env overrides:
#   EVAL_MAX_SHOTS_TO_PLOT  default 0 (= all unique selected shots).
#                           Small int caps it for smoke runs.
#   EVAL_BATCH_SIZE         default 128.

OUTPUT_DIR="${1:-${EVAL_OUTPUT_DIR:-}}"
CHECKPOINT="${2:-${EVAL_CHECKPOINT:-}}"
if [ -z "$OUTPUT_DIR" ] || [ -z "$CHECKPOINT" ]; then
    echo "Usage: sbatch $0 <output_dir> <checkpoint_path>" >&2
    echo "Example:" >&2
    echo "  sbatch $0 eval_runs/stage1_phase1_e2e_stage1_best_4609988 \\" >&2
    echo "      /lustre/orion/fus187/proj-shared/models/e2e_stage1/e2e_stage1_best.pt" >&2
    exit 1
fi
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "ERROR: output_dir not found: $OUTPUT_DIR" >&2
    exit 1
fi
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${PROJECT_DIR}"
mkdir -p logs

# Distinct port from the training jobs even though we don't init DDP.
export MASTER_PORT=29521
source scripts/slurm_frontier/_frontier_common.sh

EVAL_MAX_SHOTS_TO_PLOT="${EVAL_MAX_SHOTS_TO_PLOT:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"

echo "[eval_s1_p2_1] output_dir          : $OUTPUT_DIR"
echo "[eval_s1_p2_1] checkpoint          : $CHECKPOINT"
echo "[eval_s1_p2_1] max_shots_to_plot   : $EVAL_MAX_SHOTS_TO_PLOT  (0 = all)"
echo "[eval_s1_p2_1] batch_size          : $EVAL_BATCH_SIZE"

python scripts/training/eval_e2e_phase2_per_shot.py \
    --output_dir "$OUTPUT_DIR" \
    --checkpoint "$CHECKPOINT" \
    --data_dir /lustre/orion/fus187/proj-shared/foundation_model \
    --stats_path /lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt \
    --batch_size $EVAL_BATCH_SIZE \
    --max_shots_to_plot $EVAL_MAX_SHOTS_TO_PLOT

echo "[eval_s1_p2_1] plots in: $OUTPUT_DIR/plots"
