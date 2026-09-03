#!/bin/bash
#SBATCH -A fus187
#SBATCH -J eval_anim
#SBATCH -o logs/%j_eval_e2e_animation.out
#SBATCH -e logs/%j_eval_e2e_animation.err
#SBATCH -t 2:00:00
#SBATCH -p extended
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
set -e

# Single-shot animation generator: tangtv video on top + 4×4 growing
# time traces below. Driven by scripts/training/eval_e2e_animation.py.
#
# Usage (positional):
#   sbatch scripts/slurm_frontier/eval_e2e_animation.sh \
#       <checkpoint> <shot_id> [output_dir]
#
# Optional env overrides:
#   EVAL_FPS                  default 4
#   EVAL_STRIDE               default 1 (frames per window)
#   EVAL_K                    default 0 = autodetect from checkpoint
#   EVAL_BATCH_SIZE           default 64
#   EVAL_VIDEO_SMOOTH_SIGMA   default 1.5 — Gaussian σ (px) applied to
#                             predicted video over (H, W) only. Suppresses
#                             the 12×12 patch-boundary checkerboard
#                             from independent per-patch decoding. Set 0
#                             to disable; 3.0+ for stronger smoothing.

CHECKPOINT="${1:-}"
SHOT_ID="${2:-}"
OUTPUT_DIR="${3:-eval_runs/animations}"
if [ -z "$CHECKPOINT" ] || [ -z "$SHOT_ID" ]; then
    echo "Usage: sbatch $0 <checkpoint_path> <shot_id> [output_dir]" >&2
    exit 1
fi
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${PROJECT_DIR}"
mkdir -p logs "${OUTPUT_DIR}"

export MASTER_PORT=29540
source scripts/slurm_frontier/_frontier_common.sh

EVAL_FPS="${EVAL_FPS:-4}"
EVAL_STRIDE="${EVAL_STRIDE:-1}"
EVAL_K="${EVAL_K:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
EVAL_VIDEO_SMOOTH_SIGMA="${EVAL_VIDEO_SMOOTH_SIGMA:-1.5}"
EVAL_MODE="${EVAL_MODE:-both}"

echo "[eval_anim] checkpoint     : $CHECKPOINT"
echo "[eval_anim] shot_id        : $SHOT_ID"
echo "[eval_anim] output_dir     : $OUTPUT_DIR"
echo "[eval_anim] fps/stride/K   : $EVAL_FPS / $EVAL_STRIDE / $EVAL_K"
echo "[eval_anim] vid smooth σ   : $EVAL_VIDEO_SMOOTH_SIGMA"
echo "[eval_anim] mode           : $EVAL_MODE"

python scripts/training/eval_e2e_animation.py \
    --checkpoint "$CHECKPOINT" \
    --data_dir /lustre/orion/fus187/proj-shared/foundation_model \
    --stats_path /lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt \
    --shot_id "$SHOT_ID" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$EVAL_BATCH_SIZE" \
    --num_workers 2 \
    --fps "$EVAL_FPS" \
    --stride "$EVAL_STRIDE" \
    --K "$EVAL_K" \
    --video_smooth_sigma "$EVAL_VIDEO_SMOOTH_SIGMA" \
    --mode "$EVAL_MODE"

echo "[eval_anim] result in: $OUTPUT_DIR/${SHOT_ID}_animation.mp4"
