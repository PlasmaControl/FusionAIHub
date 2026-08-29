#!/bin/bash
#SBATCH -A fus187
#SBATCH -J denoise_a1
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -t 1:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
set -e
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"; cd "${PROJECT_DIR}"; mkdir -p logs analysis/mode_audit/denoise
export MASTER_PORT=29601
source scripts/slurm_frontier/_frontier_common.sh
export MIOPEN_USER_DB_PATH="/lustre/orion/fus187/proj-shared/ps9551/.miopen_eval_cache"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"; mkdir -p "$MIOPEN_USER_DB_PATH"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
python analysis/mode_audit/denoise_a1_viz.py
echo "[denoise_a1] done"
