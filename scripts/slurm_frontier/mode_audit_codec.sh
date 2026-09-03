#!/bin/bash
#SBATCH -A fus187
#SBATCH -J mode_audit012
#SBATCH -o logs/%j_mode_audit012.out
#SBATCH -e logs/%j_mode_audit012.err
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
set -e
# IGNITE mode-loss audit, codec-side tasks 0/1/2 (diagnostic only, no training).
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${PROJECT_DIR}"
mkdir -p logs analysis/mode_audit
export MASTER_PORT=29561
source scripts/slurm_frontier/_frontier_common.sh
export MIOPEN_USER_DB_PATH="/lustre/orion/fus187/proj-shared/ps9551/.miopen_eval_cache"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"
CODEC_DIR="${CODEC_DIR:-/lustre/orion/fus187/proj-shared/models/fsq_resid_p8_all}" \
MODALITIES="${MODALITIES:-ece,co2,bes,mhr}" \
SHOTS_FILE="${SHOTS_FILE:-/lustre/orion/fus187/proj-shared/models/codec_shots.txt}" \
NWIN_PER_SHOT="${NWIN_PER_SHOT:-800}" \
OUT_DIR="${OUT_DIR:-analysis/mode_audit}" \
python analysis/mode_audit/codec_tasks.py
echo "[mode_audit012] done"
