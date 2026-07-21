#!/bin/bash
#SBATCH -A fus187
#SBATCH -J eval_s2_p2_1
#SBATCH -o logs/%j_eval_e2e_stage2_phase2_per_shot.out
#SBATCH -e logs/%j_eval_e2e_stage2_phase2_per_shot.err
#SBATCH -t 4:00:00
#SBATCH -p extended
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
set -e

# Phase 2.1 per-shot summary plots for a Stage 2 (delta-rollout)
# checkpoint. Single-GPU re-inference on the (top-N + bottom-N) shots
# selected by Phase 1's top_bottom_shots.csv.gz, then renders a 2×2
# grid per (shot, modality). Plot panels show the **k=K final-step
# rollout prediction** vs GT.
#
# Walltime: K-step rollout makes per-shot inference ~K× slower than
# Stage 1 — 1h base → 4h here. Tweak via EVAL_K (override) and
# EVAL_BATCH_SIZE if d=1024.
#
# Usage (positional):
#   sbatch scripts/slurm_frontier/eval_e2e_stage2_phase2_per_shot.sh \
#       eval_runs/stage2_phase1_e2e_stage2_delta_best_<jobid> \
#       /lustre/orion/fus187/proj-shared/models/e2e_stage2_delta_48L/e2e_stage2_delta_best.pt
#
# Env overrides:
#   EVAL_MAX_SHOTS_TO_PLOT  default 0 (= all unique selected shots).
#   EVAL_BATCH_SIZE         default 128 (use 32-64 for d=1024).
#   EVAL_K                  default 0 (autodetect from checkpoint).

OUTPUT_DIR="${1:-${EVAL_OUTPUT_DIR:-}}"
CHECKPOINT="${2:-${EVAL_CHECKPOINT:-}}"
if [ -z "$OUTPUT_DIR" ] || [ -z "$CHECKPOINT" ]; then
    echo "Usage: sbatch $0 <output_dir> <stage2_checkpoint_path>" >&2
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

export MASTER_PORT=29526
source scripts/slurm_frontier/_frontier_common.sh

EVAL_MAX_SHOTS_TO_PLOT="${EVAL_MAX_SHOTS_TO_PLOT:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
EVAL_K="${EVAL_K:-0}"

echo "[eval_s2_p2_1] output_dir          : $OUTPUT_DIR"
echo "[eval_s2_p2_1] checkpoint          : $CHECKPOINT"
echo "[eval_s2_p2_1] max_shots_to_plot   : $EVAL_MAX_SHOTS_TO_PLOT  (0 = all)"
echo "[eval_s2_p2_1] batch_size          : $EVAL_BATCH_SIZE"
echo "[eval_s2_p2_1] K (0=auto)          : $EVAL_K"

python scripts/training/eval_e2e_phase2_per_shot.py \
    --output_dir "$OUTPUT_DIR" \
    --checkpoint "$CHECKPOINT" \
    --data_dir /lustre/orion/fus187/proj-shared/foundation_model \
    --stats_path /lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt \
    --batch_size $EVAL_BATCH_SIZE \
    --max_shots_to_plot $EVAL_MAX_SHOTS_TO_PLOT \
    --K $EVAL_K

echo "[eval_s2_p2_1] plots in: $OUTPUT_DIR/plots"
