#!/bin/bash
# Sourced by every scripts/slurm_frontier/shot_design_*.sh. Single-GPU or CPU jobs: no RCCL plugin.
#
# The three roots and the paths file are exported here rather than left to the
# `shot-design-frontier` pixi activation, because these scripts run the interpreter
# directly ($PY) instead of through `pixi run` -- the activation never fires.
set -euo pipefail
REPO="${REPO:-/lustre/orion/fus187/scratch/${USER}/FusionAIHub}"
cd "$REPO"
export RCCL_PLUGIN=0
source "$REPO/scripts/slurm_frontier/_frontier_settings.sh"
export SHOT_DESIGN_PATHS="$REPO/configs/shot_design/paths.frontier.yaml"
export SHOT_DESIGN_DATA_ROOT=/lustre/orion/fus187/proj-shared/nchen/shot_design
export LABELER_ROOT=/lustre/orion/fus187/proj-shared/nchen/labeler
export SHOT_DESIGN_CORPUS=/lustre/orion/fus187/proj-shared/foundation_model
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
ROOT="$SHOT_DESIGN_DATA_ROOT"
# _frontier_settings.sh prepends the `frontier` env to PATH; PY names the
# `shot-design-frontier` interpreter explicitly so the two envs cannot be confused.
PY="$REPO/.pixi/envs/shot-design-frontier/bin/python"
mkdir -p "$ROOT/runs/slurm"
echo "job ${SLURM_JOB_ID:-none} on $(hostname) at $(date -Is)"
rocm-smi --showproductname 2>/dev/null | grep -m1 'Card series' || true
