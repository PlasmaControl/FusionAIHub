#!/bin/bash
#SBATCH --job-name=ae-xpower-evaluate
#SBATCH --partition=pppl
#SBATCH --qos=pppl-short-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=01:00:00
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
# v4 (VERSION=v4, v2 on 60-250 kHz) is scored as v2, on v2's 60 test shots (refused
# otherwise): its record carries test_reuse (a second use of v2's test shots) and the
# second look at v2's. v2's own evaluation.json is untouched and still refuses a rescoring.
# Each candidate is scored against the labels it was trained with (its review/labels.csv copy).
#
#   after the array:  sbatch --dependency=afterok:<array id> <this>
#   pilot:            MODELS=$LABELER_ROOT/runs/ae_xpower/pilot/$VERSION LIMIT=3 sbatch <this>
#   gate:             python -m labeler.jobstats --job-id <id> --cpu-only
#
# ------------------------------------------ sizing (measured 2026-09-29, v3)
#   v3 scores whole windows from masks-full, about 3 times v2's 0-2 s frames. At 2500M its
#   first pilot (2944500) ran out of memory; the second (2944515, 3 test shots at 8G) took
#   1 min 7 s, --test's MaxRSS 4,707,100 KiB (4.49 GiB). The full test (2944534, 60 shots):
#   15 min 50 s, MaxRSS 4,984,784 KiB (4.75 GiB), CPU 96.5 %, memory 73.3 %.
#   v4 (an estimate, not measured): v2's 0-2 s test on 60-250 kHz; v2's test on the same 60
#                   shots (E, 2943080) had MaxRSS 2392 MiB, x 1.11 x 1.15 is about 3100M:
#                   pass --mem=3100M (the header's 6G is v3's whole windows).
#   mem=6G          the pilot's 4.49 GiB x 1.3; the test holds one whole shot at a time.
#   time=01:00:00   the full test's 16 min x 3.75.
# before v3 (measured 2026-09-26, login node, 1 thread):
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
