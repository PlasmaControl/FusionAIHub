#!/bin/bash
#SBATCH -A fus187
#SBATCH -J decstab
#SBATCH -o logs/%j_decstab.out
#SBATCH -e logs/%j_decstab.err
#SBATCH -t 1:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
set -e
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"; cd "${PROJECT_DIR}"; mkdir -p logs analysis/mode_audit
export MASTER_PORT=29575
source scripts/slurm_frontier/_frontier_common.sh
export MIOPEN_USER_DB_PATH="/lustre/orion/fus187/proj-shared/ps9551/.miopen_eval_cache"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"; mkdir -p "$MIOPEN_USER_DB_PATH"
MODALITIES="${MODALITIES:-ece,co2}" NWIN_PER_SHOT="${NWIN_PER_SHOT:-400}" \
OUT_DIR="${OUT_DIR:-analysis/mode_audit}" python analysis/mode_audit/decoded_stability.py
echo "[decstab] done"
