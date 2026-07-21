#!/bin/bash
#SBATCH -A fus187
#SBATCH -J probe_fit
#SBATCH -o /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.out
#SBATCH -e /lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub/logs/%x_%j.err
#SBATCH -t 0:30:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gres=gpu:1
FMH=/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub
cd "$FMH"
source scripts/slurm_frontier/_frontier_common.sh
python scripts/training/probe_fit.py
