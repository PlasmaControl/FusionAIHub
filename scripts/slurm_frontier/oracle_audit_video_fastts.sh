#!/bin/bash
#SBATCH -A fus187
#SBATCH -J oracle_aud
#SBATCH -o logs/%j_oracle_aud.out
#SBATCH -e logs/%j_oracle_aud.err
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
# Pre-training ORACLE audit (stability + persistence gate) for the two remaining
# unmeasured production inputs: tangtv (video) + filterscopes (fast-TS).
# DIAGNOSTIC ONLY. Does NOT touch the running chain or shared cache.
# Env (required): TARGET(video|fastts) CODEC_PT [MODALITY for video]
# Env (optional): SHOTS_IN SHOTS_OUT NWIN_PER_SHOT SHIFT_SAMP OUT_DIR
set -e
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"; cd "${PROJECT_DIR}"
mkdir -p logs eval_runs/oracle_audit
export MASTER_PORT="${MASTER_PORT:-29611}"
source scripts/slurm_frontier/_frontier_common.sh
# isolated MIOpen cache — do NOT share the running chain's cache
export MIOPEN_USER_DB_PATH="/lustre/orion/fus187/proj-shared/ps9551/.miopen_oracle_audit_cache"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"; mkdir -p "$MIOPEN_USER_DB_PATH"

TARGET="${TARGET:?set TARGET=video|fastts}"
CODEC_PT="${CODEC_PT:?set CODEC_PT=/path/to/codec.pt}"
OUT_DIR="${OUT_DIR:-eval_runs/oracle_audit}"
NWIN_PER_SHOT="${NWIN_PER_SHOT:-400}"

echo "[oracle_aud] TARGET=$TARGET MODALITY=${MODALITY:-<n/a>} CODEC_PT=$CODEC_PT OUT_DIR=$OUT_DIR"
TARGET="$TARGET" CODEC_PT="$CODEC_PT" MODALITY="${MODALITY:-tangtv_lower}" \
  SHOTS_IN="${SHOTS_IN:-190996,191001,191652,200417,200729,204808,204811,204812}" \
  SHOTS_OUT="${SHOTS_OUT:-200226,200722,201664,201797}" \
  NWIN_PER_SHOT="$NWIN_PER_SHOT" SHIFT_SAMP="${SHIFT_SAMP:-5}" OUT_DIR="$OUT_DIR" \
  python eval_runs/oracle_audit/oracle_video_fastts.py
echo "[oracle_aud] done"
