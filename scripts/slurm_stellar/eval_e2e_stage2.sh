#!/bin/bash
#SBATCH --job-name=eval_s2
#SBATCH --output=logs/%j_eval_e2e_stage2.out
#SBATCH --error=logs/%j_eval_e2e_stage2.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# #SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=32G

# Stage 2 (delta-loss) evaluation: load a frozen checkpoint, run a K-step
# autoregressive rollout over the val set, dump per-step / per-modality MAE,
# direction_cos, magnitude_ratio + per-channel CSV + plots + summary.md +
# metrics.json. PASS/FAIL on Stage 2 gates:
#   G1 model<copy at k=1
#   G2 model<copy at k=K
#   G3 dir_cos > 0 at every k
#   G4 mag_ratio in [0.3, 3.0] at every k
#
# Usage (positional args):
#   sbatch eval_e2e_stage2.sh runs/e2e_stage2_delta/e2e_stage2_delta_best.pt
#   sbatch eval_e2e_stage2.sh <checkpoint> <video_modality>
#
# Arg 1: checkpoint path (required)
# Arg 2: video modality name, e.g. "tangtv" (optional; for any C-Stage 2)

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

CHECKPOINT="${1:-}"
USE_VIDEO="${2:-}"

if [ -z "$CHECKPOINT" ]; then
    echo "Usage: sbatch $0 <checkpoint_path> [video_modality]" >&2
    echo "Example:" >&2
    echo "  sbatch $0 runs/e2e_stage2_delta/e2e_stage2_delta_best.pt" >&2
    exit 1
fi
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

DATA_DIR="/scratch/gpfs/EKOLEMEN/foundation_model"
STATS_PATH="/scratch/gpfs/ps9551/FusionAIHub/scripts/slurm/preprocessing_stats.pt"

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

srun pixi run python ../training/eval_e2e_stage2.py \
    --checkpoint "$CHECKPOINT" \
    --data_dir "$DATA_DIR" \
    --stats_path "$STATS_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --K 10 \
    --val_fraction 0.1 \
    --seed 42 \
    --chunk_duration_s 0.05 \
    --step_size_s 0.01 \
    --warmup_s 1.0 \
    --batch_size 128 \
    --num_workers 4 \
    --n_plot_samples 4 \
    --min_disp_norm 0.01 \
    --mag_ratio_lo 0.3 \
    --mag_ratio_hi 3.0 \
    --max_batches 20 \
    $VIDEO_FLAG
