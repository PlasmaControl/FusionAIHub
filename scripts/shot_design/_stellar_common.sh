#!/bin/bash
# Sourced by simulate_batch.sbatch, encode_batch.sbatch and build_batch.sbatch
# under scripts/shot_design/. The Stellar twin of
# scripts/slurm_frontier/_shot_design_common.sh: the interpreter is called directly ($PY),
# with the roots exported here, so a job does not depend on pixi's activation.
#
# SHOT_DESIGN_DATA_ROOT is REQUIRED from the caller: a batch run writes a database, frame
# codes, designs and simulations, and none of that may land in a production root without the
# owner's say-so. Both are refused, Stellar's (/scratch/gpfs/EKOLEMEN/nc1514/shot_design) and
# StellarAI's (/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/shot_design), however the path
# is spelled: the resolved paths are compared, so a trailing or doubled slash, a `/./` or `..`
# and a symlink all land on the same answer. The batch roots live beside the production root
# under experiments/<name>.
#
set -euo pipefail
REPO="${REPO:-/scratch/gpfs/nc1514/FusionAIHub}"
cd "$REPO"
: "${SHOT_DESIGN_DATA_ROOT:?set SHOT_DESIGN_DATA_ROOT to the batch root (never the production root)}"
batch_root="$(realpath -m -- "$SHOT_DESIGN_DATA_ROOT")"
for production in /scratch/gpfs/EKOLEMEN/nc1514/shot_design /scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/shot_design; do
  if [ "$batch_root" = "$(realpath -m -- "$production")" ]; then
    echo "refusing to run a batch against the production data root $SHOT_DESIGN_DATA_ROOT" >&2
    exit 2
  fi
done
export SHOT_DESIGN_DATA_ROOT
PY="${SHOT_DESIGN_PY:-$REPO/.pixi/envs/shot-design/bin/python}"
PY_CPU="${SHOT_DESIGN_PY_CPU:-$REPO/.pixi/envs/shot-design-cpu/bin/python}"
LABELER_ROOT="$("$PY" "$REPO/src/labeler/env.py" LABELER_ROOT "/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler")"
export LABELER_ROOT
export SHOT_DESIGN_CORPUS="${SHOT_DESIGN_CORPUS:-/scratch/gpfs/EKOLEMEN/foundation_model}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
ROOT="$SHOT_DESIGN_DATA_ROOT"
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$ROOT/runs/slurm"
echo "job ${SLURM_JOB_ID:-none} on $(hostname) at $(date -Is); data root $ROOT"
