#!/bin/bash
#SBATCH -A fus187
#SBATCH -J codec_dec
#SBATCH -o logs/%j_codec_dec.out
#SBATCH -e logs/%j_codec_dec.err
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
set -e
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${PROJECT_DIR}"
mkdir -p logs
export MASTER_PORT=29561
source scripts/slurm_frontier/_frontier_common.sh
export MIOPEN_USER_DB_PATH="/lustre/orion/fus187/proj-shared/ps9551/.miopen_eval_cache"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"
# env passed via --export: MODALITY, FINETUNE_FROM, BG_SUBTRACT, ADV_LAMBDA,
# FM_LAMBDA, SPEC_RECON_WEIGHT, FT_STEPS, EVAL_SHOTS, OUT_DIR, ...
python scripts/training/train_fsq_codec.py
echo "[codec_dec] done"
