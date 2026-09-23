#!/bin/bash
#SBATCH --job-name=labeler-spectrograms
#SBATCH --partition=pppl
#SBATCH --qos=pppl-short-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=13G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out

# The review page's AE rows for every alfven_eigenmode roster shot, one worker per cpu:
# `python -m labeler.events.review.build --event alfven_eigenmode` writes
# $LABELER_ROOT/spectrograms/alfven_eigenmode/<shot>.h5 (38-52 MB each, about 8 GB for 180).
# Every roster shot's CO2 is already in $LABELER_ROOT/raw, so nothing is fetched and no fdp
# is needed. A shot that fails prints its error on its own line and the rest carry on.
# Shots already built are skipped; FORCE=1 rebuilds them.
#
#   pilot:      SHOTS="170659 170660 170661 170662 170663 170664 170665 170666" sbatch <this>
#   the rest:   sbatch <this>
#   gate each:  python -m labeler.jobstats --job-id <id> --cpu-only
#
# ------------------------------------------------ sizing (measured 2026-09-22, login node)
#   1 worker, 4 shots:  7.0-7.4 s/shot, 99 % of one core, max RSS 1.44 GB
#   4 workers, 8 shots: 16.7 s wall, 369 % of one core (92 % of 4), process-tree RSS peak
#                       5.63 GB, i.e. 1.41 GB per worker; the parent holds ~0.1 GB
#   The work is CPU-bound: a 312 MB raw file reads in well under a second.
#   cpus-per-task=8   8 workers; the idle start and the last round cost a few percent.
#   pilot 2939069     8 shots, 8 workers: 13 s wall, MaxRSS 9.84 GiB (1.23 per worker), i.e.
#                     65.6 % of the 15G it asked for; too short for jobstats to give a CPU number.
#   mem=13G           the pilot's 9.84 GiB peak x 1.3 (76 % of it at that peak).
#   time=00:15:00     180 x 7.5 s / 8 = 169 s; x 2 for cold reads and contention, plus start.
#   The pilot runs at this same shape, so its MaxRSS is the peak the full run will meet.
set -euo pipefail
REPO="${REPO:-/scratch/gpfs/nc1514/FusionAIHub}"
ENV=/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker
ROOT="$("$ENV/bin/python" "$REPO/src/labeler/env.py" LABELER_ROOT "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export LABELER_ROOT="$ROOT" PYTHONPATH="$REPO/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"
echo "job ${SLURM_JOB_ID:-none} on $(hostname) at $(date -Is), root $ROOT"
# SHOTS is left unquoted on purpose: it is a list of shot numbers.
# shellcheck disable=SC2086
srun "$ENV/bin/python" -m labeler.events.review.build --event alfven_eigenmode \
    --workers "${SLURM_CPUS_PER_TASK:-8}" ${SHOTS:+--shots $SHOTS} ${FORCE:+--force}
echo "finished at $(date -Is)"
