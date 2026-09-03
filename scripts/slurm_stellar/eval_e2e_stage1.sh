#!/bin/bash
#SBATCH --job-name=eval_s1
#SBATCH --output=logs/%j_eval_e2e_stage1.out
#SBATCH --error=logs/%j_eval_e2e_stage1.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# #SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=32G

# Stage 1 evaluation: load a frozen checkpoint, run K=1 over the full val
# set, and dump per-modality MAE / dir_cos / mag_ratio / per-channel CSV /
# plots / summary.md / metrics.json. Works for both Phase A
# (runs/e2e_stage1/) and Phase C (runs/c_stage1/) checkpoints.
#
# Usage (positional args; env vars NOT inherited through sbatch):
#   sbatch eval_e2e_stage1.sh runs/e2e_stage1/e2e_stage1_best.pt
#   sbatch eval_e2e_stage1.sh runs/c_stage1/c_stage1_best.pt tangtv
#
# Arg 1: checkpoint path (required)
# Arg 2: video modality name, e.g. "tangtv" (optional; needed for Phase C)

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

CHECKPOINT="${1:-}"
USE_VIDEO="${2:-}"

if [ -z "$CHECKPOINT" ]; then
    echo "Usage: sbatch $0 <checkpoint_path> [video_modality]" >&2
    echo "Example:" >&2
    echo "  sbatch $0 runs/e2e_stage1/e2e_stage1_best.pt" >&2
    echo "  sbatch $0 runs/c_stage1/c_stage1_best.pt tangtv" >&2
    exit 1
fi
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

DATA_DIR="/scratch/gpfs/EKOLEMEN/foundation_model"
STATS_PATH="/scratch/gpfs/ps9551/FusionAIHub/scripts/slurm/preprocessing_stats.pt"

# ── Output dir derived from checkpoint name + job id ───────────────
CKPT_DIR="$(dirname "$CHECKPOINT")"
CKPT_STEM="$(basename "$CHECKPOINT" .pt)"
OUTPUT_DIR="${CKPT_DIR}/eval_${CKPT_STEM}_${SLURM_JOB_ID}"

VIDEO_FLAG=""
if [ -n "$USE_VIDEO" ]; then
    VIDEO_FLAG="--use_video $USE_VIDEO"
fi

echo "Checkpoint:  $CHECKPOINT"
echo "Output dir:  $OUTPUT_DIR"
echo "Use video:   ${USE_VIDEO:-(none)}"

srun pixi run python ../training/eval_e2e_stage1.py \
    --checkpoint "$CHECKPOINT" \
    --data_dir "$DATA_DIR" \
    --stats_path "$STATS_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --val_fraction 0.1 \
    --seed 42 \
    --chunk_duration_s 0.05 \
    --step_size_s 0.01 \
    --warmup_s 1.0 \
    --batch_size 128 \
    --num_workers 4 \
    --n_plot_samples 4 \
    --max_batches 20 \
    $VIDEO_FLAG