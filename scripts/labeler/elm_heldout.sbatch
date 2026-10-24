#!/bin/bash
#SBATCH --job-name=elm-heldout
#SBATCH --partition=pppl
#SBATCH --qos=pppl-short-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=900M
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/runs/slurm/%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/runs/slurm/%j.out

# The frozen ELM models on Ricardo Shousha's review of 2026-10-23 (docs/labeler/elm_heldout.md):
#   python scripts/labeler/elm_heldout_evaluate.py
# elm-ours (five cv2 fold models), elm-inst (five inst1 fold models) and ELM-O on his 81
# shots, the clock and elm_jump from their tables, and the old reviewed set as the
# reference panel; no model is refit. CPU only. Nothing is fetched here: his shots'
# inputs are fetched on the login node first (scripts/labeler/elmo_fetch.py --review-pin
# 2026-10-23). The Stellar roots must be exported, and the log path overridden, because
# the #SBATCH --output above names a FusionAIHub/.cache/labeler that Stellar does not have:
#   export LABELER_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labeler
#   export SHOT_DESIGN_DATA_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/shot_design
#   export LABELER_LABEL_TABLES=/scratch/gpfs/EKOLEMEN/nc1514/events
#   LOG="--output=$LABELER_ROOT/runs/slurm/%j.out --error=$LABELER_ROOT/runs/slurm/%j.out"
#
#   full run:   sbatch $LOG <this>
#   pilot:      ARGS="--shots 192045 206709 --skip-old --no-run-days --out-dir <scratch>" \
#                 sbatch $LOG --time=00:15:00 <this>
#   a checkout without outputs/labeler/elm (the records are git-ignored) adds
#                 REPO=<checkout> ARGS="--published <checkout with them>/outputs/labeler/elm"
#   gate:       python -m labeler.jobstats --job-id <id> --cpu-only
#
# sizing (2026-10-24): the pilot (job 2977011, 4 of his shots and the old panel) 22 s,
# 0.75 GB. The first full runs, 8 CPUs and 4G (job 2977012) and 2 CPUs and 1200M (job
# 2977014), failed the jobstats gate (CPU 24.9 % and 63.1 %, memory 22.9 % and 55.9 %):
# the run is one serial pass of file reads and small U-Nets, so it asks for one CPU. With the
# input screen added (the reference levels of the 119 reviewed shots), jobs 2977028 (1000M,
# cold file cache, 5 min 54 s: CPU 36.3 %, memory 69 %) and 2977032 (1100M: CPU 77.6 %,
# memory 60.8 %) failed; job 2977033 at 900M took 2 min 43 s and passed (CPU 78.6 %, memory
# 72.8 %). The memory the gate reads is a sampled mean, and sacct's MaxRSS follows the
# allowance (894, 992 and 1065 MiB at 900, 1000 and 1100M) because it counts reclaimable file
# pages, so a larger request only lowers the gate's figure: keep 900M, which has run twice
# (jobs 2977016 and 2977033) without an OOM.
set -euo pipefail
REPO="${REPO:-/scratch/gpfs/nc1514/FusionAIHub}"
ENV="${ENV:-/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labeler}"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export LABELER_ROOT="${LABELER_ROOT:?export the Stellar roots first}" PYTHONPATH="$REPO/src"
export LABELER_LABEL_TABLES="${LABELER_LABEL_TABLES:?}" SHOT_DESIGN_DATA_ROOT="${SHOT_DESIGN_DATA_ROOT:?}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HDF5_USE_FILE_LOCKING=FALSE LABELER_NO_FETCH=1
cd "$REPO"
echo "job ${SLURM_JOB_ID:-none} on $(hostname) at $(date -Is), root $LABELER_ROOT, repo $REPO"
# shellcheck disable=SC2086
srun "$ENV/bin/python" scripts/labeler/elm_heldout_evaluate.py \
    --workers "${SLURM_CPUS_PER_TASK:-8}" ${ARGS:-}
echo "finished at $(date -Is)"
