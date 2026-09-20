#!/bin/bash
#SBATCH -A fus187
#SBATCH -p batch
#SBATCH -J sd-simulate
# No -q debug here on purpose: this job is -t 01:00:00 on batch, already inside the
# debug QOS's own 2 h cap, and Frontier's debug QOS allows only ONE submitted job per
# user (QOSMaxSubmitJobsPU=1). The three-design demo loop and the UI's second
# concurrent simulation both submit this wrapper again while an earlier one is still
# queued/running, and debug refuses that second sbatch outright. A one-off run that
# wants the shorter debug queue can still ask for it explicitly:
#   sbatch -q debug scripts/slurm_frontier/shot_design_simulate.sh <ident>
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=closest
#SBATCH -c 7
#SBATCH -t 01:00:00
#SBATCH --output=/lustre/orion/fus187/proj-shared/nchen/shot_design/runs/slurm/%j.out
# sbatch runs a spool COPY of this file, so `dirname "$0"` is not the repo; the submit
# directory is (every wrapper is submitted from the repo root). Local runs fall back.
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/scripts/slurm_frontier/_shot_design_common.sh"
IDENT="${1:?design ident}"
srun "$PY" -m shot_design simulate "$IDENT" --device cuda "${@:2}"
