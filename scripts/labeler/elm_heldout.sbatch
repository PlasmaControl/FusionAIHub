#!/bin/bash
#SBATCH --job-name=elm-heldout
#SBATCH --partition=pppl
#SBATCH --qos=pppl-short-stellar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
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
#   sbatch --output=$LABELER_ROOT/runs/slurm/%j.out --error=$LABELER_ROOT/runs/slurm/%j.out <this>
#
#   pilot:      ARGS="--shots 192045 206709 --skip-old --no-run-days --out-dir <scratch>" \
#                 sbatch --time=00:15:00 <this>
#   full run:   sbatch <this>
#   gate:       python -m labeler.jobstats --job-id <id> --cpu-only
#
# sizing (2026-10-24, login-node SLURM pilot job 2977011: 4 of his shots and the old
# panel, 22 s, 0.75 GB MaxRSS): the full run adds 77 shots of ten small U-Nets on one
# CPU thread each and ELM-O on the 34 with BES, so --mem=4G (pilot x 5) and a 30 minute
# limit (the logs scan for the run days reads 0.6 GB once).
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
