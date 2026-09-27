#!/bin/bash
#SBATCH --job-name=ae-xpower-train
#SBATCH --partition=pppl
#SBATCH --qos=pppl-short-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --time=01:30:00
#SBATCH --array=0-2
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%A_%a.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%A_%a.out

# The AE frame model, one array task per candidate, in `train.CANDIDATES` order:
#   0 band80-mhd3    1 band0-mhd3    2 band0-mhd10
# `python -m labeler.ae.xpower.train --candidate <name>` reads the owner's saved AE labels
# ($LABELER_LABEL_TABLES/alfven_eigenmode/review/labels.csv, copied into the model directory),
# the review store's AE rows ($LABELER_ROOT/spectrograms/alfven_eigenmode) and TokEye's masks
# ($LABELER_ROOT/ae/masks), and writes $LABELER_ROOT/models/ae_xpower/v1/<name>/
# {model.pt, split.csv, training.json, review/labels.csv}. Nothing is fetched.
# CPU only: the model has under 100k parameters, and a GPU would idle far below the 70 % gate.
#
#   pilot:      PILOT=20 sbatch --array=0 <this>   (20 shots, 2 epochs, into runs/ae_xpower/pilot)
#   full run:   sbatch --mem=<from the pilot> <this>
#   gate each:  python -m labeler.jobstats --job-id <array id>_<k> --cpu-only
#
# ------------------------------------------ sizing (measured 2026-09-26, login node, 4 threads)
#   probe, 16 train + 16 val shots, 15 epochs: 304 % of 4 threads (76 %); val F1 0.95.
#   pilot, 8 shots, 2 epochs: 8-10 s wall, max RSS 1.1-1.4 GB.
#   one batch of 16 crops x 64 frames: 0.47 s, so 88 train shots x 8 crops = 44 batches, 21 s an
#   epoch for band80 (175 bins) and ~1.5 x that for band0 (257 bins); 60 epochs: 22-32 min.
#   mem=2G         1.4 GB x 1.3 plus ~0.2 GB for 104 shots' inputs. Reset it from the pilot's
#                  training.json `peak_rss_mb` (plan Task 2.9): peak x 1.25 for the full shots,
#                  and jobstats, which samples every 30 s, then still sees 70 % or more.
#   time=01:30:00  32 min x 2.8.
set -euo pipefail
REPO="${REPO:-/scratch/gpfs/nc1514/FusionAIHub-round-two}"
ENV=/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker
ROOT="$("$ENV/bin/python" "$REPO/src/labeler/env.py" LABELER_ROOT "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export LABELER_ROOT="$ROOT" PYTHONPATH="$REPO/src"
export LABELER_LABEL_TABLES="${LABELER_LABEL_TABLES:-/scratch/gpfs/nc1514/FusionAIHub/data/events}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"
K="${SLURM_ARRAY_TASK_ID:-0}"
CANDIDATE="$("$ENV/bin/python" -c "from labeler.ae.xpower.train import CANDIDATES; print(list(CANDIDATES)[$K])")"
echo "job ${SLURM_JOB_ID:-none} task $K ($CANDIDATE) on $(hostname) at $(date -Is), root $ROOT"
srun "$ENV/bin/python" -m labeler.ae.xpower.train --candidate "$CANDIDATE" ${PILOT:+--pilot "$PILOT"}
echo "finished at $(date -Is)"
