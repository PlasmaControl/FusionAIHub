#!/bin/bash
#SBATCH --job-name=frames-train
#SBATCH --partition=pppl
#SBATCH --qos=pppl-short-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out

# One frame model's training on its train shots, stopped and thresholded on its val shots:
#   python -m labeler.frames.train --method $METHOD [--pilot] [--limit N] [--seed S]
# writes $LABELER_ROOT/models/<method>/v1/{training.json,model.pt}; a model is trained
# once on its split. PILOT=1 trains 2 epochs into runs/frames/pilot/<method>/, which a
# pilot may replace; LIMIT takes the first N train and N val shots (a pilot too); SEED
# is the run's seed (20260923 unless set). Torch takes SLURM_CPUS_PER_TASK threads.
# CPU only: the model has under 40k parameters, and a GPU would idle far below the gate.
# The features are only read; nothing is fetched.
#
#   pilot:      METHOD=elm_frames PILOT=1 LIMIT=20 sbatch --time=00:15:00 <this>
#   full run:   METHOD=elm_frames sbatch <this>
#   gate:       python -m labeler.jobstats --job-id <id> --cpu-only --pilot   (a pilot)
#               python -m labeler.jobstats --job-id <id> --cpu-only
#
# ------------------------------------------ sizing (measured 2026-09-29, synthetic tree only; the rest estimates)
#   measured: 2 epochs over the test tree's 3 ELM shots, 6.8 s on the login node.
#   The features in memory, float16: NTM about 5,077 shots x 100 kB = 0.5 GB, sawteeth
#   448 x 250 kB = 0.1 GB, ELM and H/L less (estimates from the widths and windows).
#   cpus-per-task=8   torch's threads; the pilot's CPU use resets it.
#   mem=8G            estimate: the features twice over plus torch; reset from the
#                     pilot's MaxRSS x 1.2.
#   time=02:00:00     estimate: 30 epochs at most, early stopping at patience 5; NTM's
#                     13,600 crops an epoch is the longest; reset from the pilot's
#                     seconds an epoch (training.json's history).
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
srun "$ENV/bin/python" -m labeler.frames.train --method "$METHOD" \
    ${PILOT:+--pilot} ${LIMIT:+--limit "$LIMIT"} ${SEED:+--seed "$SEED"}
echo "finished at $(date -Is)"
