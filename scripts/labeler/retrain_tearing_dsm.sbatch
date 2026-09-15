#!/bin/bash
#SBATCH --job-name=tm_dsm_continue
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out
# CPU only: the model is 114k float64 parameters over 778k rows; the fork's
# training loop is dense matmuls that thread well and never needed a GPU
# (upstream's own run spent 7.5 s per epoch).
#
# Measured on job 2923879 (jobstats): 300 epochs in 29 min 38 s; CPU utilisation
# climbed from 66.5% at 2.5 min (data loading is single threaded) through 80.3%,
# 87.8% and 91.2% to 95.1% over the whole run; memory high-water 4.6 GB of the
# 32 GB requested. So the 8 cores earn their place and the right-sized request
# is --cpus-per-task 8, --mem 8G, --time 01:00:00. The directives below are left
# at what actually ran, so the card and PROVENANCE.json describe this file.
set -euo pipefail

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker
REPO=/scratch/gpfs/nc1514/FusionAIHub
export LABELMAKER_GIT_SHA=$(git -C "$REPO" rev-parse HEAD)

srun "$ROOT/envs/phase3/bin/python" -u "$REPO/scripts/labelmaker/retrain_tearing_dsm.py" \
  --out-dir "$ROOT/models/d3d_tearing_time_to_event_dsm_continued" \
  --lr 1e-4 --batch-size 1000 --max-epochs 300 --patience 5 --seed 0
