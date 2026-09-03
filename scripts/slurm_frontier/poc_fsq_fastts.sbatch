#!/bin/bash
#SBATCH -A fus187
#SBATCH -J fsq_fastts
#SBATCH -o /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.out
#SBATCH -e /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.err
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gres=gpu:1
# POC: FSQ codec for fast time-series (filterscopes). Exploratory. See
# scripts/training/poc_fsq_fastts.py.
FMH=/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub
cd "$FMH"
source scripts/slurm_frontier/_frontier_common.sh
export PYTHONPATH="$FMH/src${PYTHONPATH:+:$PYTHONPATH}"
echo "[fsq_fastts] host=$(hostname) shots=${EVAL_SHOTS:-def} steps=${AE_STEPS:-4000}"
python scripts/training/poc_fsq_fastts.py
echo "=== FSQ FAST-TS (POC) DONE (exit $?) ==="
