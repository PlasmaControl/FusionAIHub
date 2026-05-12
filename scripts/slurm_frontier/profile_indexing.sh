#!/bin/bash
# Frontier CPU-only launcher for scripts/profile_indexing.py.
# Times the file-length indexing pass that train_e2e jobs do in build_datasets,
# and reports files/sec throughput. Optionally pre-populates a lengths cache
# so future training jobs skip the indexing wall entirely.
#
# Usage:
#   # Smoke (100 files, ~1 min):
#   MAX_FILES=100 sbatch scripts/slurm_frontier/profile_indexing.sh
#
#   # Full pass, persist cache for training jobs to reuse:
#   sbatch scripts/slurm_frontier/profile_indexing.sh
#
#   # Don't allocate a GPU node at all — source _frontier_common.sh (which
#   # activates the pixi `frontier` env) on a login or compute node and call
#   # python directly:
#   python scripts/profile_indexing.py --max_files 100
#
# Common env overrides:
#   MAX_FILES=<int>        # cap on training files (default: unset = all)
#   DATA_DIR=<path>        # override data root
#   CACHE_DIR=<path>       # where to write the lengths cache (default:
#                          #   /lustre/orion/fus187/proj-shared/foundation_model_meta,
#                          #   matches the train_e2e_stage1.py default so
#                          #   subsequent training jobs reuse the cache)
#   NO_CACHE=1             # skip cache write (pure profile)
#
#SBATCH -A fus187
#SBATCH -J e2e_idx_profile
#SBATCH -o logs/%j_idx_profile.out
#SBATCH -e logs/%j_idx_profile.err
#SBATCH -t 8:00:00
#SBATCH -p extended
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=0
#SBATCH --cpus-per-task=8
set -uo pipefail

# SLURM stages the submit script under /var/spool/slurmd/... so BASH_SOURCE
# is useless for locating the repo. Use SLURM_SUBMIT_DIR — submit from the
# repo root: `cd <repo> && sbatch scripts/slurm_frontier/profile_indexing.sh`.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f "${PROJECT_DIR}/scripts/slurm_frontier/_frontier_common.sh" ]; then
    echo "ERROR: SLURM_SUBMIT_DIR (${PROJECT_DIR}) is not the repo root." >&2
    echo "       cd into the FusionAIHub repo before sbatch." >&2
    exit 1
fi
cd "${PROJECT_DIR}"
mkdir -p logs

# shellcheck disable=SC1091
source scripts/slurm_frontier/_frontier_common.sh

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

echo "[idx_profile] data_dir=$DATA_DIR cache=$CACHE_DIR use_video=${USE_VIDEO:-none} max_files=${MAX_FILES:-all}"

python -u scripts/profile_indexing.py \
    --data_dir "$DATA_DIR" \
    $CACHE_FLAG \
    $VIDEO_FLAG \
    $MAX_FILES_FLAG
