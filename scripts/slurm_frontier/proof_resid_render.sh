#!/bin/bash
#SBATCH -A fus187
#SBATCH -J resid_proof
#SBATCH -o logs/%j_resid_proof.out
#SBATCH -e logs/%j_resid_proof.err
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
set -e
# Focused residual-FSQ mode-prediction proof render (spectro-only overfit model).
# Env: CKPT, MODALITIES, SHOTS, NCOL, OUT_DIR (all have defaults in the .py).
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${PROJECT_DIR}"
mkdir -p logs
export MASTER_PORT=29547
source scripts/slurm_frontier/_frontier_common.sh
# Reuse the shared eval MIOpen cache (same arch as the comparison render → warm).
export MIOPEN_USER_DB_PATH="/lustre/orion/fus187/proj-shared/ps9551/.miopen_eval_cache"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"
python scripts/training/proof_resid_render.py
echo "[resid_proof] done"
