#!/bin/bash
#SBATCH -A fus187
#SBATCH -J fsq_video
#SBATCH -o /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.out
#SBATCH -e /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.err
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gres=gpu:1
# POC: FSQ (VQ-style) video codec for a tangtv divertor view. See
# scripts/training/poc_fsq_video.py. Env: MODALITY, EVAL_SHOTS, FSQ_DIM/L,
# AE_STEPS, N_WINDOWS, DECODER, OUT_DIR.
FMH=/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub
cd "$FMH"
source scripts/slurm_frontier/_frontier_common.sh
export PYTHONPATH="$FMH/src${PYTHONPATH:+:$PYTHONPATH}"
# Shared MIOpen kernel cache (persist compiled convs across runs -> avoid the
# ROCm kernel-search HANG on new decoder conv shapes; same as render/atlas jobs).
export MIOPEN_USER_DB_PATH="/lustre/orion/fus187/proj-shared/ps9551/.miopen_eval_cache"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"
echo "[fsq_video] host=$(hostname) modality=${MODALITY:-tangtv_lower} shots=${EVAL_SHOTS:-def} \
steps=${AE_STEPS:-4000} decoder=${DECODER:-resize_conv} out=${OUT_DIR:-eval_runs/fsq_video_${MODALITY:-tangtv_lower}}"
python scripts/training/poc_fsq_video.py
echo "=== FSQ VIDEO CODEC (POC) DONE (exit $?) ==="
