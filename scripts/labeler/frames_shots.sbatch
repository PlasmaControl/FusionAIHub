#!/bin/bash
#SBATCH --job-name=frames-shots
#SBATCH --partition=pppl
#SBATCH --qos=pppl-short-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out

# One frame model's shots and their split (round three, Task 2.7):
#   python -m labeler.frames.shots --method $METHOD [--force]
# writes $LABELER_ROOT/frames/shots/v2/<method>.{csv,json} (frames.VERSION's; v1's
# stay in frames/shots/) and the owner's frozen saves, <method>.owner.csv, laid over
# the original labels (F2; no owner split). FORCE=1 makes the split again, refused
# once a model is trained on it. The corpus, the raw cache and the label tables are only read; nothing is
# fetched. ELM, H/L and sawtooth take seconds on the login node; the tearing-mode
# split reads the archive's 28,521 grids and takes 6-9 minutes, so it runs here.
# METHOD is one of elm_frames, hmode_frames, ntm_frames, sawtooth_frames.
#
#   run:        METHOD=ntm_frames sbatch <this>
#   again:      METHOD=ntm_frames FORCE=1 sbatch <this>
#   gate:       python -m labeler.jobstats --job-id <id> --cpu-only
#
# ------------------------------------------ sizing (measured 2026-09-29, login node; the rest estimates)
#   the NTM split on the login node: 6-9 min and 0.42 GB max RSS (Task 2.7), one
#   process.
#   cpus-per-task=1   one process.
#   mem=8G            estimate, well over the measured 0.42 GB max RSS (Task 2.7,
#                     login node): 28,521 small grids and the corpus's metadata;
#                     reset to the run's MaxRSS x 1.2 (Task 2.12).
#   time=00:30:00     at least 15 min for the NTM split's 6-9 min, cold GPFS reads.
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
echo "job ${SLURM_JOB_ID:-none} $METHOD on $(hostname) at $(date -Is), root $ROOT"
srun "$ENV/bin/python" -m labeler.frames.shots --method "$METHOD" ${FORCE:+--force}
echo "finished at $(date -Is)"
