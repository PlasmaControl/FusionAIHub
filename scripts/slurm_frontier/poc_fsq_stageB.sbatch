#!/bin/bash
#SBATCH -A fus187
#SBATCH -J fsq_stageB
#SBATCH -o /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.out
#SBATCH -e /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.err
#SBATCH -t 1:30:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gres=gpu:1
# FSQ Stage-B POC: train+freeze FSQ-AE, train code predictor (CE), eval code
# prediction vs persistence on a held-out temporal split. See
# scripts/training/poc_fsq_stageB.py. Configure via env (EVAL_SHOT, FSQ_DIM,
# AE_STEPS, PRED_STEPS, N_WINDOWS, VAL_FRAC, N_CHANNELS, OUT_DIR).
FMH=/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub
cd "$FMH"
source scripts/slurm_frontier/_frontier_common.sh
# _frontier_common sets a cold per-job /tmp MIOpen cache. MIOPEN_SHARED=1 reuses
# the warm persistent cache; default = fresh cache (avoids the FIND_MODE=2
# runtime-hang / poisoned-kernel issue seen in the Stage-A jobs). Normal
# find-mode (no MIOPEN_FAST) => correct kernels, no runtime stall.
if [ -n "${MIOPEN_SHARED:-}" ]; then
    export MIOPEN_USER_DB_PATH="/lustre/orion/fus187/proj-shared/ps9551/.miopen_eval_cache"
    export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
    mkdir -p "$MIOPEN_USER_DB_PATH"
fi
[ -n "${MIOPEN_FAST:-}" ] && export MIOPEN_FIND_MODE=2
export PYTHONPATH="$FMH/src${PYTHONPATH:+:$PYTHONPATH}"
echo "[fsq_stageB] host=$(hostname) shot=${EVAL_SHOT:-200729} fsq_dim=${FSQ_DIM:-24} \
ae_steps=${AE_STEPS:-3000} pred_steps=${PRED_STEPS:-4000} out=${OUT_DIR:-eval_runs/fsq_stageB}"
python scripts/training/poc_fsq_stageB.py
echo "=== FSQ STAGE-B POC DONE (exit $?) ==="
