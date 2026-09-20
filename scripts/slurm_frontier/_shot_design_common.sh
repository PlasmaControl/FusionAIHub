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
# Module loads, RCCL/MIOpen and the distributed endpoint are for a compute node: on a login node
# (`scripts/shot_design/{blurb_frontier,demo_frontier,demo_frontier_collect}.sh` source this
# file too) _frontier_settings.sh dies on `SLURM_JOB_ID` unbound and would call scontrol on an
# empty node list. The shot-design env needs none of it for CPU work.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    source "$REPO/scripts/slurm_frontier/_frontier_settings.sh"
fi
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
