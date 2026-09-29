#!/bin/bash
#SBATCH --job-name=ae-seg-train
#SBATCH --partition=pppl
#SBATCH --qos=pppl-short-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out

# The AE segmentation model, then its one scoring on the test shots:
#   python -m labeler.ae.seg.train     the reviewed pseudo-masks
#       ($LABELER_ROOT/segmentation/alfven_eigenmode/pseudo-v1, built on the login node by
#       `python -m labeler.ae.seg.pseudo`), the owner's mask decisions
#       ($LABELER_LABEL_TABLES/alfven_eigenmode/review/masks.jsonl, if there are any), the review
#       store's AE rows and the chosen ae_xpower model's split -> $LABELER_ROOT/models/ae_seg/v1/
#       {model.pt, split.csv, training.json}
#   python -m labeler.ae.seg.evaluate  the test shots, against the recipe and TokEye baselines
#       -> evaluation.{json,md} beside the model
# Nothing is fetched. CPU only: SegNet has ~0.28 M parameters, and a GPU would idle far below
# the 70 % gate.
#
#   pilot:      PILOT=12 sbatch <this>   (12 shots, 2 epochs, into runs/ae_seg/pilot; 3 test shots)
#   full run:   sbatch --mem=<from the pilot> <this>
#   gate:       python -m labeler.jobstats --job-id <id> --cpu-only
#
# ------------------------------------------ sizing (measured 2026-09-26, login node, 4 threads)
#   pilot, 8 train + 4 val shots, 2 epochs: 5.3-5.8 s an epoch, 15.5 s wall, 335 % of 4 threads
#   (84 %), max RSS 1.39 GB.
#   probe, 16 train + 4 val shots, 12 epochs: ~120 s; validation Dice 0.90 at its threshold.
#   88 train shots x 8 crops = 88 batches of 8, ~48 s an epoch; 40 epochs at most: ~32 min
#   (patience 8 usually stops it sooner). evaluate, 3 test shots: 5.3 s, 1.18 GB, 258 %; the 58
#   test shots take ~2 min.
#   mem=2G         1.39 GB x 1.25, plus ~0.1 GB for 104 shots' inputs; reset it from the pilot's
#                  training.json `peak_rss_mb` x 1.25, so jobstats, which samples every 30 s,
#                  still sees 70 % or more.
#   time=01:30:00  32 min x 2.8.
set -euo pipefail
REPO="${REPO:-/scratch/gpfs/nc1514/FusionAIHub}"
ENV=/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker
ROOT="$("$ENV/bin/python" "$REPO/src/labeler/env.py" LABELER_ROOT "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export LABELER_ROOT="$ROOT" PYTHONPATH="$REPO/src"
export LABELER_LABEL_TABLES="${LABELER_LABEL_TABLES:-/scratch/gpfs/nc1514/FusionAIHub/data/events}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE LABELER_NO_FETCH=1
cd "$REPO"
echo "job ${SLURM_JOB_ID:-none} on $(hostname) at $(date -Is), root $ROOT"
MODELS="$ROOT/models/ae_seg/v1"
if [ -n "${PILOT:-}" ]; then MODELS="$ROOT/runs/ae_seg/pilot"; fi
srun "$ENV/bin/python" -m labeler.ae.seg.train ${PILOT:+--pilot "$PILOT"}
srun "$ENV/bin/python" -m labeler.ae.seg.evaluate --models "$MODELS" ${PILOT:+--limit 3}
echo "finished at $(date -Is)"
