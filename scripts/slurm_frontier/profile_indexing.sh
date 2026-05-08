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
#   # Don't allocate a GPU node at all by calling python directly after `conda
#   # activate $CONDA_ENV_PATH` from a login or compute node:
#   python scripts/profile_indexing.py --max_files 100
#
# Common env overrides:
#   MAX_FILES=<int>        # cap on training files (default: unset = all)
#   DATA_DIR=<path>        # override data root
#   CACHE_DIR=<path>       # where to write the lengths cache (default:
#                          #   runs/lengths_cache_e2e_stage1/, persists for
#                          #   subsequent training jobs)
#   NO_CACHE=1             # skip cache write (pure profile)
#
#SBATCH -A fus187
#SBATCH -J e2e_idx_profile
#SBATCH -o logs/%j_idx_profile.out
#SBATCH -e logs/%j_idx_profile.err
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=0
#SBATCH --cpus-per-task=8
set -uo pipefail

PROJECT_DIR=/lustre/orion/fus187/scratch/nchen/FusionAIHub
cd "$PROJECT_DIR"
mkdir -p logs

# shellcheck disable=SC1091
source scripts/slurm_frontier/_frontier_common.sh

DATA_DIR="${DATA_DIR:-/lustre/orion/fus187/proj-shared/foundation_model}"
CACHE_DIR="${CACHE_DIR:-runs/lengths_cache_e2e_stage1}"

MAX_FILES_FLAG=""
[ -n "${MAX_FILES:-}" ] && MAX_FILES_FLAG="--max_files $MAX_FILES"

CACHE_FLAG="--cache_dir $CACHE_DIR"
[ "${NO_CACHE:-0}" = "1" ] && CACHE_FLAG="--no_cache"

echo "[idx_profile] data_dir=$DATA_DIR cache=$CACHE_DIR max_files=${MAX_FILES:-all}"

python -u scripts/profile_indexing.py \
    --data_dir "$DATA_DIR" \
    $CACHE_FLAG \
    $MAX_FILES_FLAG
