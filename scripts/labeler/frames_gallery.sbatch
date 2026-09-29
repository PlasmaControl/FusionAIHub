#!/bin/bash
#SBATCH --job-name=frames-gallery
#SBATCH --partition=pppl
#SBATCH --qos=pppl-short-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out

# One frame model's pictures, one a test shot and one a roster shot:
#   python -m labeler.frames.gallery --method $METHOD [--limit N]
# writes $LABELER_ROOT/gallery/<event>/<method>-v1/{test,roster}/<shot>.jpg: the
# features the model reads, the target where the shot has one, P and the threshold,
# titled as suggestions. PILOT=1 draws the first 20 of each (LIMIT, 20 unless set),
# which the full run draws over. The stores and features are only read; nothing is
# fetched.
#
#   pilot:      METHOD=elm_frames PILOT=1 sbatch --time=00:10:00 <this>
#   full run:   METHOD=elm_frames sbatch <this>
#   gate:       python -m labeler.jobstats --job-id <id> --cpu-only --pilot   (a pilot)
#               python -m labeler.jobstats --job-id <id> --cpu-only
#
# ------------------------------------------ sizing (measured 2026-09-29, synthetic tree only; the rest estimates)
#   measured: the test tree's 2 ELM pictures, 1.3 s on the login node (0.65 s each).
#   pictures: the 450-shot roster and the test shots (NTM's about 760), about 300 MB
#   for the four methods (spec §3.5).
#   cpus-per-task=1   one process: matplotlib and a store read a picture.
#   mem=4G            estimate: one store's rows and one figure at a time.
#   time=01:00:00     estimate: 1,200 pictures at about 1 s, x 3 for cold store reads.
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
echo "job ${SLURM_JOB_ID:-none} $METHOD on $(hostname) at $(date -Is), root $ROOT"
srun "$ENV/bin/python" -m labeler.frames.gallery --method "$METHOD" ${LIMIT:+--limit "$LIMIT"}
echo "finished at $(date -Is)"
