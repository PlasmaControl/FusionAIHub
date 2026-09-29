#!/bin/bash
#SBATCH --job-name=ae-xpower-evaluate
#SBATCH --partition=pppl
#SBATCH --qos=pppl-short-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2500M
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out
# Scheduler: pppl/pppl-short-stellar -> all/stellar-debug; sacct 2942492,2942516-18.

# After the train array: choose a candidate on the validation shots, then score the test shots once.
#   python -m labeler.ae.xpower.evaluate --choose   -> models/ae_xpower/$VERSION/chosen.json
#   python -m labeler.ae.xpower.evaluate --test     -> models/ae_xpower/$VERSION/evaluation.{json,md}
# VERSION defaults to v1. For v2 (after ae_xpower_final.sbatch), --choose only checks
# chosen.json against cv/choice.json and writes nothing, and the test also reports
# v1's 58 test shots beside the full 60 (the bar is judged on the 60 only).
# A full scoring (no LIMIT) is refused under runs/; a pilot there takes LIMIT, 20 or fewer,
# and for v2 only a pilot choice's model (a copy of the final model there is refused).
# v2 is scored only in models/ae_xpower/v2 or under runs/; a copy elsewhere is refused.
# Each candidate is scored against the labels it was trained with (its review/labels.csv copy).
#
#   after the array:  sbatch --dependency=afterok:<array id> <this>
#   pilot:            MODELS=$LABELER_ROOT/runs/ae_xpower/pilot/$VERSION LIMIT=3 sbatch <this>
#   gate:             python -m labeler.jobstats --job-id <id> --cpu-only
#
# ------------------------------------------ sizing (measured 2026-09-26, login node, 1 thread)
#   pilot, 3 test shots, every method (SELDNet included): 14 s wall, max RSS 1.85 GB.
#   mem=2500M       1.85 GB x 1.35; the scores held per shot are a few kB.
#   time=00:30:00   3 x 16 validation shots x ~1.5 s + 58 test shots x ~4 s = ~5 min, x 6.
set -euo pipefail
REPO="${REPO:-/scratch/gpfs/nc1514/FusionAIHub}"
ENV=/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker
ROOT="$("$ENV/bin/python" "$REPO/src/labeler/env.py" LABELER_ROOT "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export LABELER_ROOT="$ROOT" PYTHONPATH="$REPO/src"
export LABELER_LABEL_TABLES="${LABELER_LABEL_TABLES:-/scratch/gpfs/nc1514/FusionAIHub/data/events}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HDF5_USE_FILE_LOCKING=FALSE LABELER_NO_FETCH=1
cd "$REPO"
VERSION="${VERSION:-v1}"  # v1 by default, so v1's commands still mean v1
echo "job ${SLURM_JOB_ID:-none} ($VERSION) on $(hostname) at $(date -Is), root $ROOT"
srun "$ENV/bin/python" -m labeler.ae.xpower.evaluate --choose --version "$VERSION" \
    ${MODELS:+--models "$MODELS"} ${LIMIT:+--limit "$LIMIT"}
srun "$ENV/bin/python" -m labeler.ae.xpower.evaluate --test --version "$VERSION" \
    ${MODELS:+--models "$MODELS"} ${LIMIT:+--limit "$LIMIT"}
echo "finished at $(date -Is)"
