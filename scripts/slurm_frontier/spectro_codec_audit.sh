#!/bin/bash
#SBATCH -A fus187
#SBATCH -J codec_audit
#SBATCH -o logs/%j_codec_audit.out
#SBATCH -e logs/%j_codec_audit.err
#SBATCH -t 1:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
set -e
# Codec audit: (1) code histogram (imbalance) + (2) faithfulness splice test.
# No world model — frozen codec + data only. Env: CODEC_DIR, MODALITIES, SHOT, OUT_DIR.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${PROJECT_DIR}"
mkdir -p logs
export MASTER_PORT=29553
source scripts/slurm_frontier/_frontier_common.sh
export MIOPEN_USER_DB_PATH="/lustre/orion/fus187/proj-shared/ps9551/.miopen_eval_cache"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"
CODEC_DIR="${CODEC_DIR:-/lustre/orion/fus187/proj-shared/models/fsq_resid_p8_all}" \
MODALITIES="${MODALITIES:-ece,co2,bes,mhr}" \
SHOT="${SHOT:-200729}" \
OUT_DIR="${OUT_DIR:-eval_runs/codec_audit}" \
python scripts/training/spectro_codec_audit.py
echo "[codec_audit] done"
