#!/bin/bash
# Offline pre-build of the K-anneal B horizon-specific lengths cache.
# SINGLE process, NO torch.distributed / NCCL — so the ~87-min full-set scan
# can never trip NCCL's 10-min watchdog (which is why it MUST run offline,
# not inside a multi-rank training job).
#
# Usage:
#   sbatch scripts/slurm_frontier/prebuild_lengths_cache.sbatch
#   # then (multi-partition eligibility):
#   scontrol update job=<id> Partition=extended,batch,g1
#
#SBATCH -A fus187
#SBATCH -J prebuild_lengths
#SBATCH -o /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.out
#SBATCH -e /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
set -uo pipefail

FMH=/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub
cd "$FMH"
mkdir -p logs
# shellcheck disable=SC1091
source scripts/slurm_frontier/_frontier_common.sh

# B-specific horizon-specific cache dir (NOT the shared foundation_model_meta).
CACHE_DIR="${CACHE_DIR:-/lustre/orion/fus187/proj-shared/models/e2e_g3fix_kanneal_v2/lengths_h0.7}"
TRAIN_HORIZON="${TRAIN_HORIZON:-0.7}"   # Lever #1 block-0 = K*chunk+pred = 10*0.05+0.2
VAL_HORIZON="${VAL_HORIZON:-0.2}"       # trainer's val span (model horizon; validate() single-step)

echo "[prebuild.sbatch] host=$(hostname) cache_dir=$CACHE_DIR train_h=$TRAIN_HORIZON val_h=$VAL_HORIZON"

# Run in the LOGIN/COMPUTE node's single process (no srun → no distributed).
python -u scripts/data_preparation/prebuild_lengths_cache.py \
    --cache_dir "$CACHE_DIR" \
    --train_horizon "$TRAIN_HORIZON" \
    --val_horizon "$VAL_HORIZON"

echo "=== PREBUILD DONE (exit $?) ==="
