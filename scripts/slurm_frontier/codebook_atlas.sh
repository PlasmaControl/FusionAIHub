#!/bin/bash
#SBATCH -A fus187
#SBATCH -J codebook_atlas
#SBATCH -o logs/%j_codebook_atlas.out
#SBATCH -e logs/%j_codebook_atlas.err
#SBATCH -t 1:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
set -e
# Codebook atlas hero figure for a frozen FSQ spectro codec.
# Env: MODALITY (ece|co2|bes|mhr), SHOTS (comma list), NWIN_PER_SHOT, K_CLUSTERS,
#      CODEC_PATH, OUT_DIR. Defaults target the tok96 spectro codecs.
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
export MASTER_PORT=29561
source scripts/slurm_frontier/_frontier_common.sh
# shared MIOpen cache (reuse compiled conv kernels across atlas runs)
export MIOPEN_USER_DB_PATH="/lustre/orion/fus187/proj-shared/ps9551/.miopen_eval_cache"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"

echo "[codebook_atlas] script=${RECON_SCRIPT:-codebook_atlas.py} MODALITY=${MODALITY:-ece} SHOT(S)=${SHOT:-${SHOTS:-200729}}"
python scripts/training/${RECON_SCRIPT:-codebook_atlas.py}
