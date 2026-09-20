#!/bin/bash
#SBATCH -A fus187
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -J sd-simulate
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
