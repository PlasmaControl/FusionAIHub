#!/bin/bash
# Frontier CPU-only launcher for scripts/build_dataset_cache.py.
# Builds the dataset indexing caches (video-presence + per-file chunk counts)
# in parallel so subsequent train_e2e jobs hit them at __init__ time and skip
# the indexing wall entirely.
#
# Usage:
#   # Smoke (100 files):
#   MAX_FILES=100 sbatch scripts/slurm_frontier/build_dataset_cache.sh
#
#   # Full pass, persist cache for training jobs to reuse:
#   sbatch scripts/slurm_frontier/build_dataset_cache.sh
#
#   # Don't allocate a GPU node at all — source _frontier_settings.sh (which
#   # activates the pixi `frontier` env) on a login or compute node and call
#   # python directly:
#   python scripts/build_dataset_cache.py --max_files 100
#
# Common env overrides:
#   MAX_FILES=<int>        # cap on training files (default: unset = all)
#   DATA_DIR=<path>        # override data root
#   CACHE_DIR=<path>       # where to write the indexing caches (default:
#                          #   /lustre/orion/fus187/proj-shared/foundation_model_meta,
#                          #   matches the train_e2e_stage1.py default so
#                          #   subsequent training jobs reuse the cache)
#   NO_CACHE=1             # skip cache write (pure timing measurement)
#
#SBATCH -A fus187
#SBATCH -J build_dataset_cache
#SBATCH -o logs/%j_build_dataset_cache.out
#SBATCH -e logs/%j_build_dataset_cache.err
#SBATCH -t 0:30:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=0
#SBATCH --cpus-per-task=16
set -uo pipefail

# SLURM stages the submit script under /var/spool/slurmd/... so BASH_SOURCE
# is useless for locating the repo. Use SLURM_SUBMIT_DIR — submit from the
# repo root: `cd <repo> && sbatch scripts/slurm_frontier/build_dataset_cache.sh`.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f "${PROJECT_DIR}/scripts/slurm_frontier/_frontier_settings.sh" ]; then
    echo "ERROR: SLURM_SUBMIT_DIR (${PROJECT_DIR}) is not the repo root." >&2
    echo "       cd into the FusionAIHub repo before sbatch." >&2
    exit 1
fi
cd "${PROJECT_DIR}"
mkdir -p logs

# shellcheck disable=SC1091
source scripts/slurm_frontier/_frontier_settings.sh

DATA_DIR="${DATA_DIR:-/lustre/orion/fus187/proj-shared/foundation_model}"
CACHE_DIR="${CACHE_DIR:-/lustre/orion/fus187/proj-shared/foundation_model_meta}"
# Must mirror train_e2e_stage1.sh's --use_video so the produced lengths cache
# is keyed on the same (post-filter) path list training will see. Set empty
# to skip the filter — but then the cache won't be reusable by --use_video
# training runs.
USE_VIDEO="${USE_VIDEO:-tangtv}"

MAX_FILES_FLAG=""
[ -n "${MAX_FILES:-}" ] && MAX_FILES_FLAG="--max_files $MAX_FILES"

CACHE_FLAG="--cache_dir $CACHE_DIR"
[ "${NO_CACHE:-0}" = "1" ] && CACHE_FLAG="--no_cache"

VIDEO_FLAG=""
[ -n "${USE_VIDEO}" ] && VIDEO_FLAG="--use_video $USE_VIDEO"

# Stage selector. PREDICTION_HORIZON_S and CACHE_NAME_PREFIX must agree:
# the lengths cache contents depend on prediction_horizon_s, so we name
# the cache file per stage to avoid one stage overwriting another.
PREDICTION_HORIZON_S="${PREDICTION_HORIZON_S:-0.05}"
CACHE_NAME_PREFIX="${CACHE_NAME_PREFIX:-lengths_e2e_stage1}"

echo "[build_dataset_cache] data_dir=$DATA_DIR cache=$CACHE_DIR \
use_video=${USE_VIDEO:-none} max_files=${MAX_FILES:-all} \
prediction_horizon_s=${PREDICTION_HORIZON_S} prefix=${CACHE_NAME_PREFIX}"

python -u scripts/build_dataset_cache.py \
    --data_dir "$DATA_DIR" \
    --prediction_horizon_s "$PREDICTION_HORIZON_S" \
    --cache_name_prefix "$CACHE_NAME_PREFIX" \
    $CACHE_FLAG \
    $VIDEO_FLAG \
    $MAX_FILES_FLAG
