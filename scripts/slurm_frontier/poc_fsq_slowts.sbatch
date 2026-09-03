#!/bin/bash
#SBATCH -A fus187
#SBATCH -J fsq_slowts
#SBATCH -o /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.out
#SBATCH -e /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.err
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gres=gpu:1
# Adversarial FSQ codecs for slow time-series (7 modalities in one job).
FMH=/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub
cd "$FMH"
source scripts/slurm_frontier/_frontier_common.sh
export PYTHONPATH="$FMH/src${PYTHONPATH:+:$PYTHONPATH}"
echo "[fsq_slowts] host=$(hostname) mods=${MODALITIES:-all} shots=${MAX_SHOTS:-200}"
python scripts/training/poc_fsq_slowts.py
echo "=== FSQ SLOW-TS (POC) DONE (exit $?) ==="
