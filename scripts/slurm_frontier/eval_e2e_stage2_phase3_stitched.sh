#!/bin/bash
#SBATCH -A fus187
#SBATCH -J eval_s2_p3
#SBATCH -o logs/%j_eval_e2e_stage2_phase3_stitched.out
#SBATCH -e logs/%j_eval_e2e_stage2_phase3_stitched.err
#SBATCH -t 8:00:00
#SBATCH -p extended
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
set -e

# Phase 3 stitched-window plots + Phase 3.1 video for a Stage 2
# (delta-rollout) checkpoint. Per-shot K-step rollout (K autodetected
# from ckpt['args']['K_max']) stashes the final-step (k=K) prediction
# at each stitched segment.
#   3.0 → line plots for TS modalities, per-channel stacked heatmaps
#         for spectrograms — all showing the model's k=K prediction.
#   3.1 → 5×6 grid PNG per (shot, segment) + 1 continuous mp4 per shot
#         (n_channels × 3 layout: GT | k=K prediction | |diff|, native
#         60 fps, libx264) for video modalities.
#
# Walltime: K=10 rollout makes per-shot inference ~K× slower than
# Stage 1 — 2h base → 8h here.
#
# Usage:
#   sbatch scripts/slurm_frontier/eval_e2e_stage2_phase3_stitched.sh \
#       eval_runs/stage2_phase1_e2e_stage2_delta_best_<jobid> \
#       /lustre/orion/fus187/proj-shared/models/e2e_stage2_delta_48L/e2e_stage2_delta_best.pt
#
# Env overrides:
#   EVAL_MAX_SHOTS_TO_PLOT  default 0 (= all top/bottom-selected).
#   EVAL_BATCH_SIZE         default 128 (use 32-64 for d=1024).
#   EVAL_PHASES             default "3.0 3.1". Set to one to skip the other.
#   EVAL_SKIP_MP4           Phase 3.1 only — set to 1 to skip mp4 encoding.
#   EVAL_K                  default 0 (autodetect from checkpoint).

OUTPUT_DIR="${1:-${EVAL_OUTPUT_DIR:-}}"
CHECKPOINT="${2:-${EVAL_CHECKPOINT:-}}"
if [ -z "$OUTPUT_DIR" ] || [ -z "$CHECKPOINT" ]; then
    echo "Usage: sbatch $0 <output_dir> <stage2_checkpoint_path>" >&2
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

export MASTER_PORT=29527
source scripts/slurm_frontier/_frontier_common.sh

EVAL_MAX_SHOTS_TO_PLOT="${EVAL_MAX_SHOTS_TO_PLOT:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
EVAL_PHASES="${EVAL_PHASES:-3.0 3.1}"
EVAL_K="${EVAL_K:-0}"

P31_EXTRA=()
if [ "${EVAL_SKIP_MP4:-0}" = "1" ]; then
    P31_EXTRA+=("--skip_mp4")
fi

echo "[eval_s2_p3] output_dir         : $OUTPUT_DIR"
echo "[eval_s2_p3] checkpoint         : $CHECKPOINT"
echo "[eval_s2_p3] max_shots_to_plot  : $EVAL_MAX_SHOTS_TO_PLOT  (0 = all)"
echo "[eval_s2_p3] batch_size         : $EVAL_BATCH_SIZE"
echo "[eval_s2_p3] phases             : $EVAL_PHASES"
echo "[eval_s2_p3] skip_mp4           : ${EVAL_SKIP_MP4:-0}"
echo "[eval_s2_p3] K (0=auto)         : $EVAL_K"

for phase in $EVAL_PHASES; do
    case "$phase" in
        3.0)
            echo ""
            echo "[eval_s2_p3] === Phase 3.0 (TS + spectrogram, k=K view) ==="
            python scripts/training/eval_e2e_phase3_stitched.py \
                --output_dir "$OUTPUT_DIR" \
                --checkpoint "$CHECKPOINT" \
                --data_dir /lustre/orion/fus187/proj-shared/foundation_model \
                --stats_path /lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt \
                --batch_size $EVAL_BATCH_SIZE \
                --max_shots_to_plot $EVAL_MAX_SHOTS_TO_PLOT \
                --K $EVAL_K
            ;;
        3.1)
            echo ""
            echo "[eval_s2_p3] === Phase 3.1 (video grid + mp4, k=K view) ==="
            python scripts/training/eval_e2e_phase3_1_video.py \
                --output_dir "$OUTPUT_DIR" \
                --checkpoint "$CHECKPOINT" \
                --data_dir /lustre/orion/fus187/proj-shared/foundation_model \
                --stats_path /lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt \
                --batch_size $EVAL_BATCH_SIZE \
                --max_shots_to_plot $EVAL_MAX_SHOTS_TO_PLOT \
                --K $EVAL_K \
                "${P31_EXTRA[@]}"
            ;;
        *)
            echo "[eval_s2_p3] WARNING: unknown phase '$phase' (expected 3.0 or 3.1)" >&2
            ;;
    esac
done

echo ""
echo "[eval_s2_p3] plots / mp4s in: $OUTPUT_DIR/plots"
