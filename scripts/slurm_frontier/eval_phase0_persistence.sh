#!/bin/bash
#SBATCH -A fus187
#SBATCH -J phase0_persist
#SBATCH -o logs/%j_phase0_persistence.out
#SBATCH -e logs/%j_phase0_persistence.err
#SBATCH -t 0:40:00
#SBATCH -p extended
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
set -e
# Phase-0 validation render: persistence-conditioned spectrogram forecast on the
# current production model (μ + input-window persistence mask, no GT, no arch
# change, no training). See scripts/training/phase0_persistence_forecast.py.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${PROJECT_DIR}"
export MASTER_PORT=29543
source scripts/slurm_frontier/_frontier_common.sh
export MIOPEN_USER_DB_PATH="/lustre/orion/fus187/proj-shared/ps9551/.miopen_eval_cache"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
export EVAL_CKPT="${EVAL_CKPT:?set EVAL_CKPT}"
export EVAL_SHOT="${EVAL_SHOT:-200729}"
export EVAL_MODALITY="${EVAL_MODALITY:-ece}"
export EVAL_OUT="${EVAL_OUT:-eval_runs/phase0_persistence}"
echo "[phase0] ckpt=$EVAL_CKPT shot=$EVAL_SHOT modality=$EVAL_MODALITY out=$EVAL_OUT"
python scripts/training/phase0_persistence_forecast.py
