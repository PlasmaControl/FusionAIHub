#!/bin/bash
#SBATCH --job-name=elm-inst2-cpu
#SBATCH --partition=pppl
#SBATCH --qos=pppl-short-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/runs/slurm/%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/runs/slurm/%j.out

# The CPU stages of elm-inst2 (scripts/labeler/elm_inst2.py), one per job:
#   STAGE=prepare   his shots' inputs, merged onsets and run-day folds (before elm_inst2.sbatch)
#   STAGE=evaluate  the development record outputs/labeler/elm/inst2/evaluation.json
#   STAGE=key       the challenge key, once: refuses when the run folder's key.lock or
#                   key_merit_scores.json, or this checkout's outputs/labeler/elm/inst2/key.json,
#                   exists
#   SCRIPT=elm_inst2_typed.py STAGE=evaluate|key   round 2 (type-2 prices, size typing):
#                   evaluation_typed.json, then the key once more under key2.lock
#
#   run:   export LABELER_ROOT=... LABELER_LABEL_TABLES=... SHOT_DESIGN_DATA_ROOT=...
#          STAGE=prepare sbatch --output=$LABELER_ROOT/runs/slurm/%j.out \
#              --error=$LABELER_ROOT/runs/slurm/%j.out scripts/labeler/elm_inst2_cpu.sbatch
#          REPO=<checkout> runs another checkout's code (the default is the main one)
#   gate:  python -m labeler.jobstats --job-id <id> --cpu-only
#
# ------------------------------------------ sizing
#   2977038 (prepare): 11 s, MaxRSS 0.4 GB. 2977055 (evaluate): 34 s, MaxRSS 1.8 GB, CPU
#   94 %. 2977056 (key): 6 s, MaxRSS 0.3 GB. Round 2 (SCRIPT=elm_inst2_typed.py): 2977224
#   and 2977240 (evaluate): 1 min 36 s, MaxRSS 1.9 GB; 2977241 (key): 14 s, MaxRSS 0.4 GB.
#   All too short for jobstats; one CPU, mem 3G.
set -euo pipefail
REPO="${REPO:-/scratch/gpfs/nc1514/FusionAIHub}"
: "${LABELER_ROOT:?export LABELER_ROOT before sbatch}"
: "${LABELER_LABEL_TABLES:?export LABELER_LABEL_TABLES before sbatch}"
STAGE="${STAGE:?STAGE=prepare|evaluate|key}"
SCRIPT="${SCRIPT:-elm_inst2.py}"
export PYTHONPATH="$REPO/src" LABELER_NO_FETCH=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
echo "job ${SLURM_JOB_ID:-none} on $(hostname) at $(date -Is), $SCRIPT $STAGE, repo $REPO at $(git rev-parse --short HEAD)"
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml \
    -e labeler python -u "scripts/labeler/$SCRIPT" "$STAGE"
echo "done at $(date -Is)"
