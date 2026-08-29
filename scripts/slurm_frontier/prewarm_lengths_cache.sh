#!/bin/bash
#SBATCH -A fus187
#SBATCH -J prewarm_lengths
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -t 3:00:00
#SBATCH -p extended
# NOTE: the single-process ALL-shots scan is ~1.8 h, so -t MUST be >=3h. The
# batch partition caps at 2h (rejects this) -> use extended, or after submit
# `scontrol update job=<id> Partition=g1` for 48h headroom.
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
set -e
# Single-process pre-warm of the ALL-shots lengths cache (no DDP -> no NCCL
# watchdog). See scripts/training/prewarm_lengths_cache.py.
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
export MASTER_PORT=29571
source scripts/slurm_frontier/_frontier_common.sh
python scripts/training/prewarm_lengths_cache.py
echo "=== PREWARM DONE (exit $?) ==="
