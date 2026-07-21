#!/bin/bash
#SBATCH -A fus187
#SBATCH -J eval_s1_p3
#SBATCH -o logs/%j_eval_e2e_stage1_phase3_stitched.out
#SBATCH -e logs/%j_eval_e2e_stage1_phase3_stitched.err
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
set -e

# Phase 3 stitched-window plots — Phase 3.0 (TS + spectrogram) followed
# by Phase 3.1 (video grid + mp4). Single-GPU re-inference per shot;
# stashes 3 segments × 80 windows each (~4 s of shot wall-time at
# 0 / 33 / 66 % of shot length).
#   3.0 → line plots for TS modalities, per-channel stacked heatmaps
#         for spectrograms.
#   3.1 → 5×6 grid PNG per (shot, segment) + 1 mp4 per shot (3-panel
#         GT|model|diff, native 60 fps, libx264 via bundled ffmpeg) for
#         video (tangtv) modalities.
#
# Usage:
#   sbatch scripts/slurm_frontier/eval_e2e_stage1_phase3_stitched.sh \
#       eval_runs/stage1_phase1_e2e_stage1_best_4609988 \
#       /lustre/orion/fus187/proj-shared/models/e2e_stage1/e2e_stage1_best.pt
#
# Env overrides:
#   EVAL_MAX_SHOTS_TO_PLOT  default 0 (= all top/bottom-selected). Small
#                           int caps via coverage-aware ordering.
#   EVAL_BATCH_SIZE         default 128.
#   EVAL_PHASES             default "3.0 3.1". Set to "3.0" or "3.1"
#                           to run only one sub-phase.
#   EVAL_SKIP_MP4           Phase 3.1 only — set to 1 for grid PNGs
#                           without mp4 encoding.

OUTPUT_DIR="${1:-${EVAL_OUTPUT_DIR:-}}"
CHECKPOINT="${2:-${EVAL_CHECKPOINT:-}}"
if [ -z "$OUTPUT_DIR" ] || [ -z "$CHECKPOINT" ]; then
    echo "Usage: sbatch $0 <output_dir> <checkpoint_path>" >&2
    exit 1
fi
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "ERROR: output_dir not found: $OUTPUT_DIR" >&2; exit 1
fi
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT" >&2; exit 1
fi

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${PROJECT_DIR}"
mkdir -p logs

# Distinct port from Phase 2.1 (29521); we don't init DDP but the
# variable still gets read by _frontier_common.sh.
export MASTER_PORT=29522
source scripts/slurm_frontier/_frontier_common.sh

EVAL_MAX_SHOTS_TO_PLOT="${EVAL_MAX_SHOTS_TO_PLOT:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
EVAL_PHASES="${EVAL_PHASES:-3.0 3.1}"

P31_EXTRA=()
if [ "${EVAL_SKIP_MP4:-0}" = "1" ]; then
    P31_EXTRA+=("--skip_mp4")
fi
# Optional shot restriction (Phase 3.1 only; Phase 3.0 doesn't yet
# support it). Pass a space-separated list of shot IDs via EVAL_ONLY_SHOTS.
if [ -n "${EVAL_ONLY_SHOTS:-}" ]; then
    P31_EXTRA+=("--only_shots")
    for s in $EVAL_ONLY_SHOTS; do
        P31_EXTRA+=("$s")
    done
fi

echo "[eval_s1_p3] output_dir         : $OUTPUT_DIR"
echo "[eval_s1_p3] checkpoint         : $CHECKPOINT"
echo "[eval_s1_p3] max_shots_to_plot  : $EVAL_MAX_SHOTS_TO_PLOT  (0 = all)"
echo "[eval_s1_p3] batch_size         : $EVAL_BATCH_SIZE"
echo "[eval_s1_p3] phases             : $EVAL_PHASES"
echo "[eval_s1_p3] skip_mp4           : ${EVAL_SKIP_MP4:-0}"

for phase in $EVAL_PHASES; do
    case "$phase" in
        3.0)
            echo ""
            echo "[eval_s1_p3] === Phase 3.0 (TS + spectrogram) ==="
            python scripts/training/eval_e2e_phase3_stitched.py \
                --output_dir "$OUTPUT_DIR" \
                --checkpoint "$CHECKPOINT" \
                --data_dir /lustre/orion/fus187/proj-shared/foundation_model \
                --stats_path /lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt \
                --batch_size $EVAL_BATCH_SIZE \
                --max_shots_to_plot $EVAL_MAX_SHOTS_TO_PLOT
            ;;
        3.1)
            echo ""
            echo "[eval_s1_p3] === Phase 3.1 (video grid + mp4) ==="
            python scripts/training/eval_e2e_phase3_1_video.py \
                --output_dir "$OUTPUT_DIR" \
                --checkpoint "$CHECKPOINT" \
                --data_dir /lustre/orion/fus187/proj-shared/foundation_model \
                --stats_path /lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt \
                --batch_size $EVAL_BATCH_SIZE \
                --max_shots_to_plot $EVAL_MAX_SHOTS_TO_PLOT \
                "${P31_EXTRA[@]}"
            ;;
        *)
            echo "[eval_s1_p3] WARNING: unknown phase '$phase' (expected 3.0 or 3.1)" >&2
            ;;
    esac
done

echo ""
echo "[eval_s1_p3] plots / mp4s in: $OUTPUT_DIR/plots"
