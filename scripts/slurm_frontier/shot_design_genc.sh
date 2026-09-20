#!/bin/bash
#SBATCH -A fus187
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -J sd-genc
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=closest
#SBATCH -c 7
#SBATCH -t 01:00:00
#SBATCH --output=/lustre/orion/fus187/proj-shared/nchen/shot_design/runs/slurm/%j.out
# The G-ENC gate against the pinned v4 generation: re-encode `g_enc.py`'s default
# five shots and compare each one against production's own frame-codes cache, one
# GCD, debug QOS -- a five-shot re-encode is minutes, not a training job. READ-ONLY:
# the cache is only ever compared against, never written to.
#
# sbatch runs a spool COPY of this file, so `dirname "$0"` is not the repo; the submit
# directory is (every wrapper is submitted from the repo root). Local runs fall back.
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/scripts/slurm_frontier/_shot_design_common.sh"
srun "$PY" scripts/shot_design/g_enc.py --device cuda --out "$ROOT/runs/genc_v4.json"
