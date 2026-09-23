#!/bin/bash
# Sourced by simulate_batch.sbatch, encode_batch.sbatch and build_batch.sbatch
# under scripts/shot_design/. The Stellar twin of
# scripts/slurm_frontier/_shot_design_common.sh: the interpreter is called directly ($PY),
# never through `pixi run`, because pixi's activation would silently replace an exported
# SHOT_DESIGN_DATA_ROOT with the production root (docs/shot-design/overview.md, "Scratch
# databases and the pixi activation env").
#
# SHOT_DESIGN_DATA_ROOT is REQUIRED from the caller: a batch run writes a database, frame
# codes, designs and simulations, and none of that may land in the production root
# (/scratch/gpfs/EKOLEMEN/nc1514/ideate) without the owner's say-so. The batch roots live
# beside it under /scratch/gpfs/EKOLEMEN/nc1514/ideate/experiments/<name>.
#
set -euo pipefail
REPO="${REPO:-/scratch/gpfs/nc1514/FusionAIHub}"
cd "$REPO"
: "${SHOT_DESIGN_DATA_ROOT:?set SHOT_DESIGN_DATA_ROOT to the batch root (never the production root)}"
case "$SHOT_DESIGN_DATA_ROOT" in
  /scratch/gpfs/EKOLEMEN/nc1514/ideate|/scratch/gpfs/EKOLEMEN/nc1514/ideate/)
    echo "refusing to run a batch against the production data root $SHOT_DESIGN_DATA_ROOT" >&2
    exit 2 ;;
esac
export SHOT_DESIGN_DATA_ROOT
export LABELER_ROOT="${LABELER_ROOT:-/scratch/gpfs/EKOLEMEN/nc1514/labelmaker}"
export SHOT_DESIGN_CORPUS="${SHOT_DESIGN_CORPUS:-/scratch/gpfs/EKOLEMEN/foundation_model}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
ROOT="$SHOT_DESIGN_DATA_ROOT"
PY="${SHOT_DESIGN_PY:-$REPO/.pixi/envs/shot-design/bin/python}"
PY_CPU="${SHOT_DESIGN_PY_CPU:-$REPO/.pixi/envs/shot-design-cpu/bin/python}"
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$ROOT/runs/slurm"
echo "job ${SLURM_JOB_ID:-none} on $(hostname) at $(date -Is); data root $ROOT"
