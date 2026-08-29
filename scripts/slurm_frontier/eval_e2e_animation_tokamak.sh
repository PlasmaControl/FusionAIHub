#!/bin/bash
#SBATCH -A fus187
#SBATCH -J eval_anim_tok
#SBATCH -o logs/%j_eval_e2e_animation_tokamak.out
#SBATCH -e logs/%j_eval_e2e_animation_tokamak.err
#SBATCH -t 2:00:00
#SBATCH -p extended
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
set -e

# Tokamak-themed animation: digital twin (predictions, left) +
# reactor (GT, right) PNG backgrounds, cam frames overlaid at
# upper/lower divertor positions, time traces (Te/ne/Ti) between
# cams, ECE/CO2 spectrograms on the outer columns.
#
# Usage (positional):
#   sbatch scripts/slurm_frontier/eval_e2e_animation_tokamak.sh \
#       <checkpoint> [shot_id] [output_dir]
#
# Optional env overrides:
#   EVAL_BATCH_SIZE     default 64
#   EVAL_K              default 0 (autodetect from checkpoint)
#   EVAL_ROLLOUT_STEP   default 0 (1-step-ahead, Stage 1 default).
#                        Set to -1 for K-step-ahead (autoregressive
#                        Stage 2 visualisation). The output mp4 name
#                        is suffixed with stepN where N=rollout_step+1.

CHECKPOINT="${1:-}"
SHOT_ID="${2:-200729}"
OUTPUT_DIR="${3:-eval_runs/animations}"
if [ -z "$CHECKPOINT" ]; then
    echo "Usage: sbatch $0 <checkpoint_path> [shot_id] [output_dir]" >&2
    exit 1
fi
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${PROJECT_DIR}"
mkdir -p logs "${OUTPUT_DIR}"

export MASTER_PORT=29541
source scripts/slurm_frontier/_frontier_common.sh

# Persistent MIOpen kernel cache for EVAL/RENDER jobs — override the per-job,
# node-local /tmp cache that _frontier_common.sh sets (right for 64-rank
# training, wasteful for short renders). Renders are 1 GPU / few ranks, so the
# home/Lustre cache contention that motivated the /tmp redirect doesn't apply.
# A FIXED shared path lets every render REUSE the compiled kernels instead of
# recompiling the ~40-min MIOpen set each run. First render populates it; all
# later renders of the same arch/eval-shapes start in minutes.
export MIOPEN_USER_DB_PATH="/lustre/orion/fus187/proj-shared/ps9551/.miopen_eval_cache"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
EVAL_K="${EVAL_K:-0}"
EVAL_ROLLOUT_STEP="${EVAL_ROLLOUT_STEP:-0}"
# EVAL_EXTRA_ARGS: free-form passthrough to the python script.
# DEFAULT now includes --no_spec_fusion (2026-06-13): the soft-mask GT
# fusion ("presentation fix") is DEACTIVATED by default so renders show
# the RAW model spec output. To re-enable the presentation fusion for a
# polished render, override with EVAL_EXTRA_ARGS="" sbatch ...
# Use ${VAR-default} (single dash) NOT ${VAR:-default}: the colon form
# substitutes the default for BOTH unset AND empty, so EVAL_EXTRA_ARGS=""
# (to request the fused presentation render) would wrongly fall back to
# --no_spec_fusion. The single-dash form honors an explicit empty value.
EVAL_EXTRA_ARGS="${EVAL_EXTRA_ARGS---no_spec_fusion}"
# EVAL_DATA_DIR: which processed-shot directory to read (GT + inference both
# use it). Default = main foundation_model set; override for shots elsewhere,
# e.g. EVAL_DATA_DIR=/lustre/orion/proj-shared/fus187/additional_data (199xxx).
EVAL_DATA_DIR="${EVAL_DATA_DIR:-/lustre/orion/fus187/proj-shared/foundation_model}"

echo "[eval_anim_tok] checkpoint     : $CHECKPOINT"
echo "[eval_anim_tok] shot_id        : $SHOT_ID"
echo "[eval_anim_tok] data_dir       : $EVAL_DATA_DIR"
echo "[eval_anim_tok] output_dir     : $OUTPUT_DIR"
echo "[eval_anim_tok] batch / K      : $EVAL_BATCH_SIZE / $EVAL_K"
echo "[eval_anim_tok] rollout_step   : $EVAL_ROLLOUT_STEP"
echo "[eval_anim_tok] extra_args     : $EVAL_EXTRA_ARGS"

python scripts/training/eval_e2e_animation_tokamak.py \
    --checkpoint "$CHECKPOINT" \
    --data_dir "$EVAL_DATA_DIR" \
    --stats_path /lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt \
    --shot_id "$SHOT_ID" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$EVAL_BATCH_SIZE" \
    --num_workers 2 \
    --K "$EVAL_K" \
    --rollout_step "$EVAL_ROLLOUT_STEP" \
    ${EVAL_EXTRA_ARGS}

echo "[eval_anim_tok] result in: $OUTPUT_DIR/_tokamak_animation_step<N>.mp4"
echo "                 (N = rollout_step + 1; rollout_step=-1 → N=K)"
