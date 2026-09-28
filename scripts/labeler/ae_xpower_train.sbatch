#!/bin/bash
#SBATCH --job-name=ae-xpower-train
#SBATCH --partition=pppl
#SBATCH --qos=pppl-short-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=2800M
#SBATCH --time=01:30:00
#SBATCH --array=0-2
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%A_%a.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%A_%a.out
# Scheduler: training jobs 2942484_2, 2942491_0 and 2942491_1 ran as submitted:
# QOS pppl-short-stellar, partition serial (sacct). Only evaluate (2942492) and
# gallery (2942516-2942518) went to partition all / QOS stellar-debug.

# The AE frame model, one array task per candidate, in `train.candidates(VERSION)`
# order (VERSION defaults to v1):
#   0 band80-mhd3    1 band0-mhd3    2 band0-mhd10
# v2 is chosen by cross-validation instead (ae_xpower_cv.sbatch, then
# ae_xpower_final.sbatch); `train --version v2 --candidate` is refused.
# `python -m labeler.ae.xpower.train --candidate <name>` reads the owner's saved AE labels
# ($LABELER_LABEL_TABLES/alfven_eigenmode/review/labels.csv, copied into the model directory),
# the review store's AE rows ($LABELER_ROOT/spectrograms/alfven_eigenmode) and TokEye's masks
# ($LABELER_ROOT/ae/masks), and writes $LABELER_ROOT/models/ae_xpower/$VERSION/<name>/
# {model.pt, split.csv, training.json, review/labels.csv}. Nothing is fetched.
# CPU only: the model has under 100k parameters, and a GPU would idle far below the 70 % gate.
#
#   pilot:      PILOT=20 sbatch --array=0 <this>   (20 shots, 2 epochs, into runs/ae_xpower/pilot/$VERSION)
#   full run:   sbatch <this>   (2800M, sized from the full runs below)
#   gate each:  python -m labeler.jobstats --job-id <array id>_<k> --cpu-only
#
# ------------------------------------------ sizing (measured 2026-09-26, login node, 4 threads)
#   probe, 16 train + 16 val shots, 15 epochs: 304 % of 4 threads (76 %); val F1 0.95.
#   pilot, 8 shots, 2 epochs: 8-10 s wall, max RSS 1.1-1.4 GB.
#   one batch of 16 crops x 64 frames: 0.47 s, so 88 train shots x 8 crops = 44 batches, 21 s an
#   epoch for band80 (175 bins) and ~1.5 x that for band0 (257 bins); 60 epochs: 22-32 min.
#   mem=2800M     full runs peaked at 2311 MB (band80-mhd3) and 2470 MB
#                 (band0-mhd3; sacct MaxRSS 2,454,492K at 2400M).
#                 1700M ran out of memory (MaxRSS ~1.74 GB when killed).
#                 The 20-shot pilot's peak (1182 MB) under-predicts the 104-shot
#                 run: size from the full runs, not from the pilot rule.
#   time=01:30:00  32 min x 2.8.
set -euo pipefail
REPO="${REPO:-/scratch/gpfs/nc1514/FusionAIHub}"
ENV=/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker
ROOT="$("$ENV/bin/python" "$REPO/src/labeler/env.py" LABELER_ROOT "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export LABELER_ROOT="$ROOT" PYTHONPATH="$REPO/src"
export LABELER_LABEL_TABLES="${LABELER_LABEL_TABLES:-/scratch/gpfs/nc1514/FusionAIHub/data/events}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"
VERSION="${VERSION:-v1}"  # v1 by default, so v1's commands still mean v1
K="${SLURM_ARRAY_TASK_ID:-0}"
CANDIDATE="$("$ENV/bin/python" -c "from labeler.ae.xpower.train import candidates; print(list(candidates('$VERSION'))[$K])")"
echo "job ${SLURM_JOB_ID:-none} task $K ($VERSION $CANDIDATE) on $(hostname) at $(date -Is), root $ROOT"
srun "$ENV/bin/python" -m labeler.ae.xpower.train --version "$VERSION" \
    --candidate "$CANDIDATE" ${PILOT:+--pilot "$PILOT"}
echo "finished at $(date -Is)"
