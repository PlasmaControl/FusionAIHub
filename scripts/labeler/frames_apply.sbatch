#!/bin/bash
#SBATCH --job-name=frames-apply
#SBATCH --partition=pppl
#SBATCH --qos=pppl-short-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%A_%a.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%A_%a.out

# One frame model over the roster or the population, as suggestions, one shard a task:
#   python -m labeler.frames.apply --method $METHOD --roster|--population --index I --count N --workers W
# writes $LABELER_ROOT/suggestions/<method>/v2/shards/<set>-I-of-N.{npz,failed.jsonl,json}.
# The roster is read from its review stores; the population's non-roster shots with the
# spec's groups on disk are built in memory, no store written (D48). Task I of an array
# of N is shard I of N (SLURM_ARRAY_TASK_ID and SLURM_ARRAY_TASK_COUNT, D67). PILOT=1
# takes each shard's first 20 shots (LIMIT, 20 unless set) into shards/pilot/, which the
# merge never reads; SHOTS names shots, a pilot too. Nothing is fetched. It runs whatever
# the bar (D50); the merge, once the test is scored, writes the table, its meta (the
# bar's verdict, tier suggestions, D47, D49) and summary.csv, and for hmode_frames
# lmode_frames' too (D38):
#   LABELER_NO_FETCH=1 python -m labeler.frames.apply --method $METHOD --merge   (the login node)
#
#   pilot:      METHOD=elm_frames PILOT=1 sbatch <this> --population
#   full run:   METHOD=elm_frames sbatch <this> --roster
#               METHOD=elm_frames sbatch --array=0-7 <this> --population
#               then the merge, above, or METHOD=elm_frames sbatch --time=00:10:00 <this> --merge
#   gate:       python -m labeler.jobstats --job-id <id> --cpu-only --pilot   (a pilot)
#               python -m labeler.jobstats --job-id <array id>_<k> --cpu-only
#
# ------------------------------------------ sizing (measured 2026-09-26 for the rows' builds; the rest estimates)
#   a shot's rows, built as a store is (review_build.sbatch): tearing modes 5.1 s and
#   1.12 GB, sawtooth 2.6 s and 0.51 GB, ELMs 1.2 s and 0.35 GB, H-mode 0.5 s and
#   0.32 GB; the model's pass is estimated well under a second a shot.
#   population: 4,872 shots less the blind and the roster, about 4,400 before the
#   group check (estimate).
#   cpus-per-task=8   NTM over 8 tasks: 550 shots x 5.1 s / 8 workers = 5.8 min a task;
#                     the roster, 450 stores read, in one task.
#   mem=12G           8 x 1.12 GB x 1.2 for NTM; reset from the pilot's MaxRSS.
#   time=00:45:00     5.8 min x 7 for cold GPFS reads of the corpus.
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
srun "$ENV/bin/python" -m labeler.frames.apply --method "$METHOD" \
    --index "${SLURM_ARRAY_TASK_ID:-0}" --count "${SLURM_ARRAY_TASK_COUNT:-1}" \
    --workers "${SLURM_CPUS_PER_TASK:-8}" ${LIMIT:+--limit "$LIMIT"} ${SHOTS:+--shots $SHOTS} "$@"
echo "finished at $(date -Is)"
