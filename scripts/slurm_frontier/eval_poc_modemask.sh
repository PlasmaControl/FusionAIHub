#!/bin/bash
#SBATCH -A fus187
#SBATCH -J poc_modemask_eval
#SBATCH -o logs/%j_poc_modemask_eval.out
#SBATCH -e logs/%j_poc_modemask_eval.err
#SBATCH -t 0:40:00
#SBATCH -p extended
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
set -e
# POC held-out verdict: does the LEARNED mode-mask beat persistence on held-out
# shots? See scripts/training/poc_modemask_eval.py.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${PROJECT_DIR}"
export MASTER_PORT=29544
source scripts/slurm_frontier/_frontier_common.sh
export MIOPEN_USER_DB_PATH="/lustre/orion/fus187/proj-shared/ps9551/.miopen_eval_cache"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
export EVAL_CKPT="${EVAL_CKPT:?set EVAL_CKPT}"
export EVAL_MAX_FILES="${EVAL_MAX_FILES:-400}"
export EVAL_VAL_SHOTS="${EVAL_VAL_SHOTS:-15}"
export EVAL_FIG_TAG="${EVAL_FIG_TAG:-poc}"
export EVAL_OUT="${EVAL_OUT:-eval_runs/poc_modemask}"
echo "[poc-eval] ckpt=$EVAL_CKPT max_files=$EVAL_MAX_FILES val_shots=$EVAL_VAL_SHOTS tag=$EVAL_FIG_TAG mode=${EVAL_MODE:-verdict}"
if [ "${EVAL_MODE:-verdict}" = "maskfit" ]; then
    python scripts/training/test_mask_head_fit.py
else
    python scripts/training/poc_modemask_eval.py
fi
