#!/bin/bash
#SBATCH --job-name=frames-prepare
#SBATCH --partition=pppl
#SBATCH --qos=pppl-short-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%A_%a.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%A_%a.out

# One frame model's legacy stores (--stores), or every shot's features, one shard a task:
#   python -m labeler.frames.prepare --method $METHOD [--stores] --index I --count N --workers W
# --stores builds each shot outside the roster a store under
# $LABELER_ROOT/frames/stores/<event>/ (review.build(out=), never spectrograms/; a
# stale tearing-mode store is rebuilt); without it, $LABELER_ROOT/frames/features/<method>/
# <shot>.npz, a dropped shot's reason beside it. Each run writes its record under
# records/. Task I of an array of N is shard I of N (SLURM_ARRAY_TASK_ID and
# SLURM_ARRAY_TASK_COUNT, D67); without an array, one shard. PILOT=1 takes each shard's
# first 20 shots (LIMIT, 20 unless set); SHOTS names shots; FORCE=1 rebuilds built stores.
# Nothing is fetched. Exits 1 when a store could not be built. Sawteeth need no stores.
#
#   pilot:      METHOD=ntm_frames PILOT=1 sbatch <this> --stores
#   full run:   METHOD=ntm_frames sbatch --array=0-7 <this> --stores
#               METHOD=elm_frames sbatch <this> --stores   (and hmode_frames: one task)
#               then the features: METHOD=M sbatch --array=0-3 <this>
#   gate:       python -m labeler.jobstats --job-id <id> --cpu-only --pilot   (a pilot)
#               python -m labeler.jobstats --job-id <array id>_<k> --cpu-only
#
# ------------------------------------------ sizing (measured 2026-09-26 for the stores' builds; the rest estimates)
#   a store's build, one worker (review_build.sbatch): tearing modes 5.1 s and 1.12 GB,
#   ELMs 1.2 s and 0.35 GB, H-mode 0.5 s and 0.32 GB. Stores to build (D68): NTM about
#   4,839 x 1.9 MB, ELM about 200 x 10 MB, H-mode 98 x 2.6 MB, 11.4 GB in all.
#   cpus-per-task=8   NTM over 8 tasks: 605 shots x 5.1 s / 8 workers = 6.4 min a task;
#                     ELM 200 x 1.2 s / 8 = 30 s in one task.
#   mem=12G           8 x 1.12 GB x 1.2 for the NTM builds; the features are estimated
#                     below 1 GB a worker (not measured); reset from the pilot's MaxRSS.
#   time=00:30:00     6.4 min x 4 for cold GPFS reads; features estimated faster.
set -euo pipefail
REPO="${REPO:-/scratch/gpfs/nc1514/FusionAIHub}"
ENV=/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker
ROOT="$("$ENV/bin/python" "$REPO/src/labeler/env.py" LABELER_ROOT "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export LABELER_ROOT="$ROOT" PYTHONPATH="$REPO/src"
export LABELER_LABEL_TABLES="${LABELER_LABEL_TABLES:-/scratch/gpfs/nc1514/FusionAIHub/data/events}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE LABELER_NO_FETCH=1
cd "$REPO"
: "${METHOD:?set METHOD: elm_frames, hmode_frames, ntm_frames or sawtooth_frames}"
if [ -n "${PILOT:-}" ]; then LIMIT="${LIMIT:-20}"; fi
echo "job ${SLURM_JOB_ID:-none} $METHOD shard ${SLURM_ARRAY_TASK_ID:-0} of ${SLURM_ARRAY_TASK_COUNT:-1} ($*) on $(hostname) at $(date -Is), root $ROOT"
srun "$ENV/bin/python" -m labeler.frames.prepare --method "$METHOD" \
    --index "${SLURM_ARRAY_TASK_ID:-0}" --count "${SLURM_ARRAY_TASK_COUNT:-1}" \
    --workers "${SLURM_CPUS_PER_TASK:-8}" ${LIMIT:+--limit "$LIMIT"} ${SHOTS:+--shots $SHOTS} \
    ${FORCE:+--force} "$@"
echo "finished at $(date -Is)"
