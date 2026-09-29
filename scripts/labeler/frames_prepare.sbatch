#!/bin/bash
#SBATCH --job-name=frames-prepare
#SBATCH --partition=pppl
#SBATCH --qos=pppl-short-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=00:45:00
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
#               METHOD=elm_frames sbatch --mem=3G <this> --stores   (and hmode_frames)
#               then the features: METHOD=M sbatch --mem=3G --array=0-3 <this>
#   gate:       python -m labeler.jobstats --job-id <id> --cpu-only --pilot   (a pilot)
#               python -m labeler.jobstats --job-id <array id>_<k> --cpu-only
#
# ------------------------------------------ sizing (measured 2026-09-29: 20-shot pilots, 8 workers)
#   stores (--stores): NTM 2947617 20 built in 22 s, MaxRSS 8.1 GiB (1.06 a worker),
#   5.5 s CPU a shot; ELM 2947621 20 built in 9 s, 2.1 GiB; H-mode 2947616 18 built
#   in 38 s, 2.3 GiB. Features: 2947622-2947625, 20 shots each in 8-9 s, 2.1 GiB (the
#   workers' imports; a shot adds little). The ELM pilot 2947615 took 20 roster shots
#   and built nothing, so its re-pilot named 20 others (SHOTS).
#   cpus-per-task=8   NTM stores over 8 tasks: about 605 shots x 5.5 s / 8 = 7 min a task.
#   mem=12G           the NTM stores (8.1 GiB, 67 %); the rest take --mem=3G on the
#                     command line: ELM and H-mode stores, and every method's features.
#   time=00:45:00     estimated: NTM's 7 min x 4 for cold GPFS reads; above 30 min, as
#                     30 min or less goes to all/stellar-debug (2 running jobs a user,
#                     10 submitted), where the pilots 2947615-2947625 went.
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
