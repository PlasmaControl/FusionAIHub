#!/bin/bash
#SBATCH -A fus187
#SBATCH -J mode_audit3
#SBATCH -o logs/%j_mode_audit3.out
#SBATCH -e logs/%j_mode_audit3.err
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
set -e
# IGNITE mode-loss audit Task 3 (k1 triad + codeacc split). Needs the world model.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${PROJECT_DIR}"
mkdir -p logs analysis/mode_audit
export MASTER_PORT=29563
source scripts/slurm_frontier/_frontier_common.sh
export MIOPEN_USER_DB_PATH="/lustre/orion/fus187/proj-shared/ps9551/.miopen_eval_cache"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"
CKPT="${CKPT:-/lustre/orion/fus187/proj-shared/models/e2e_step2_fsq_finer/e2e_stage1_latest.pt}" \
MOD="${MOD:-ece}" SHOTS="${SHOTS:-200729,190996,204811}" N_MODE_WIN="${N_MODE_WIN:-20}" \
OUT_DIR="${OUT_DIR:-analysis/mode_audit}" \
python analysis/mode_audit/triad_task.py
echo "[mode_audit3] done"
