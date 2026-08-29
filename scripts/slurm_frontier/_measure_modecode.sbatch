#!/bin/bash
#SBATCH -A fus187
#SBATCH -J modecode_rate
#SBATCH -o /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.out
#SBATCH -e /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.err
#SBATCH -t 0:45:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gres=gpu:1
FMH=/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub
cd "$FMH"
source scripts/slurm_frontier/_frontier_common.sh
python scripts/training/measure_modecode_rate.py \
  "${CKPT:-/lustre/orion/fus187/proj-shared/models/e2e_stage1_allshots_b32/e2e_stage1_best.pt}"
