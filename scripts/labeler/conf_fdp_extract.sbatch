#!/bin/bash
#SBATCH --job-name=conf-fdp-extract
#SBATCH --account=nstx-u
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/runs/slurm/%x_%j.out
#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/runs/slurm/%x_%j.out

# Staged sparse MDSplus trees -> confine-ours 1 ms grids (scripts/labeler/conf_fdp_label.py extract),
# no network. The stage subcommand submits this per batch with LIST=<root>/batches/bNNNN.csv.
#   LIST=... [KEEP=1] sbatch scripts/labeler/conf_fdp_extract.sbatch
set -euo pipefail
REPO=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub
export LABELER_ROOT="${LABELER_ROOT:-$REPO/.cache/labeler}"
export CONF_FDP_ROOT="${CONF_FDP_ROOT:-$LABELER_ROOT/conf_fdp_1ms}"
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 LABELER_NO_FETCH=1
unset BEARER_TOKEN
cd "$REPO"
echo "job ${SLURM_JOB_ID:-none} on $(hostname) at $(date -Is): list $LIST, root $CONF_FDP_ROOT"
ARGS=(--root "$CONF_FDP_ROOT" extract --list "$LIST" --workers "${SLURM_CPUS_PER_TASK:-8}")
[ -n "${KEEP:-}" ] && ARGS+=(--keep-inputs)
"$REPO/.pixi/envs/fdp/bin/python" -u scripts/labeler/conf_fdp_label.py "${ARGS[@]}"
echo "finished at $(date -Is)"
