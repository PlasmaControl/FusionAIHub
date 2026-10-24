#!/bin/bash
#SBATCH --job-name=tm_dsm_continue
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/runs/slurm/%j.out
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
REPO=/scratch/gpfs/nc1514/FusionAIHub
ENV="${LABELER_PIXI_ENV:-/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labeler}"  # another cluster: export LABELER_PIXI_ENV
# The cluster's production root (labeler.env.CLUSTERS), whatever LABELER_ROOT the caller
# exports: the phase3 interpreter and this job's outputs live there.
ROOT="$(env -u LABELER_ROOT "$ENV/bin/python" "$REPO/src/labeler/env.py" LABELER_ROOT "/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler")"
export LABELER_GIT_SHA=$(git -C "$REPO" rev-parse HEAD)

srun "$ROOT/envs/phase3/bin/python" -u "$REPO/scripts/labeler/retrain_tearing_dsm.py" \
  --out-dir "$ROOT/models/d3d_tearing_time_to_event_dsm_continued" \
  --lr 1e-4 --batch-size 1000 --max-epochs 300 --patience 5 --seed 0
