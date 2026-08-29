#!/bin/bash
#SBATCH -A fus187
#SBATCH -J desc_strat_eval
#SBATCH -o logs/%j_desc_strat_eval.out
#SBATCH -e logs/%j_desc_strat_eval.err
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
# Forecast-layer (descriptor-head) mode-skill eval, stratified by window activity.
# Usage: sbatch descriptor_stratified_eval.sh <ckpt> [n_shots] [n_batches]
set -euo pipefail
CKPT="${1:?ckpt required}"
N_SHOTS="${2:-40}"
N_BATCHES="${3:-200}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${PROJECT_DIR}"
mkdir -p logs
export MASTER_PORT=29562
source scripts/slurm_frontier/_frontier_common.sh
# reuse the shared eval MIOpen cache (arch already compiled by the renders)
export MIOPEN_USER_DB_PATH="/lustre/orion/fus187/proj-shared/ps9551/.miopen_eval_cache"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"

echo "[desc_strat_eval] ckpt=$CKPT  n_shots=$N_SHOTS  n_batches=$N_BATCHES"
python analysis/mode_audit/descriptor_stratified_eval.py \
    --ckpt "$CKPT" \
    --n_shots "$N_SHOTS" \
    --n_batches "$N_BATCHES" \
    --anchor_beta "${ANCHOR_BETA:-6.0}" \
    --prediction_horizon_s "${PRED_HORIZON_S:-0.05}" \
    --out "${OUT_JSON:-analysis/mode_audit/descriptor_stratified_eval.json}" \
    ${FIGURE_ARGS:-}
