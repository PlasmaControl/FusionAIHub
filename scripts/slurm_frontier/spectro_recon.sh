#!/bin/bash
#SBATCH -A fus187
#SBATCH -J spectro_recon
#SBATCH -o logs/%j_spectro_recon.out
#SBATCH -e logs/%j_spectro_recon.err
#SBATCH -t 1:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
set -e
# Full-window spectrogram RECONSTRUCTION benchmark (encode -> FSQ -> decode vs GT),
# per spectro modality, comparing the 3 codec families side by side:
#   production raw (patch 32/16) | residual (patch 32/16) | finer residual (patch 8/16).
# Reports reconstruction corr per codec + renders GT | recon(each) | diff on the strongest-mode channel.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${PROJECT_DIR}"
mkdir -p logs
export MASTER_PORT=29551
source scripts/slurm_frontier/_frontier_common.sh
export MIOPEN_USER_DB_PATH="/lustre/orion/fus187/proj-shared/ps9551/.miopen_eval_cache"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"
SHOT="${SHOT:-200729}"
M=/lustre/orion/fus187/proj-shared/models
for MOD in ece co2 bes mhr; do
  echo "===================== RECON ${MOD} (shot ${SHOT}) ====================="
  MODALITY=$MOD SHOT=$SHOT \
  CODEC_PATHS="$M/fsq_spectro_codecs_tok96/spectro_codec_${MOD}.pt,$M/fsq_spectro_residual_codecs/spectro_codec_${MOD}.pt,$M/fsq_resid_p8_all/spectro_codec_${MOD}.pt" \
  OUT_DIR="eval_runs/codec_recon_real/${MOD}" \
  python scripts/training/spectro_recon.py || echo "[WARN] ${MOD} recon failed"
done
echo "[spectro_recon] done"
