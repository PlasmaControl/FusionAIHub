#!/bin/bash
#SBATCH -A fus187
#SBATCH -J fsq_codec
#SBATCH -o /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.out
#SBATCH -e /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.err
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gres=gpu:1
# Phase 1a: pre-train + freeze the adversarial FSQ spectro codec (per modality).
# See scripts/training/train_fsq_codec.py. Configure via env (MODALITY, EVAL_SHOTS,
# FSQ_DIM, PATCH_F/T, AE_STEPS, N_WINDOWS, adversarial config, OUT_DIR).
FMH=/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub
cd "$FMH"
source scripts/slurm_frontier/_frontier_common.sh
# fresh per-job /tmp MIOpen cache (avoids the FIND_MODE poison / shared-cache issues);
# MIOPEN_SHARED=1 to reuse the warm cache, MIOPEN_FAST=1 for FIND_MODE=2 (risky).
if [ -n "${MIOPEN_SHARED:-}" ]; then
    export MIOPEN_USER_DB_PATH="/lustre/orion/fus187/proj-shared/ps9551/.miopen_eval_cache"
    export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"; mkdir -p "$MIOPEN_USER_DB_PATH"
fi
[ -n "${MIOPEN_FAST:-}" ] && export MIOPEN_FIND_MODE=2
export PYTHONPATH="$FMH/src${PYTHONPATH:+:$PYTHONPATH}"
echo "[fsq_codec] host=$(hostname) modality=${MODALITY:-ece} \
shots=${EVAL_SHOTS_FILE:-${EVAL_SHOTS:-200729}} \
steps=${AE_STEPS:-6000} out=${OUT_DIR:-eval_runs/fsq_codec_${MODALITY:-ece}}"
python scripts/training/train_fsq_codec.py
echo "=== FSQ CODEC (Phase 1a) DONE (exit $?) ==="
