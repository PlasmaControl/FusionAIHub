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
# `sbatch` runs a COPY of this file from a spool directory, so `dirname "$0"` is
# that spool dir, not this repo -- confirmed on job 5514036 ("_shot_design_common.sh:
# No such file or directory"); `$SLURM_SUBMIT_DIR` is the directory `sbatch` was
# invoked from instead, which the other shot_design_*.sh wrappers assume equals
# this one without checking (none has a real sbatch run in `sacct` history yet,
# only `--test-only`). Falls back to `dirname "$0"` for a local, non-sbatch run.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    source "$SLURM_SUBMIT_DIR/scripts/slurm_frontier/_shot_design_common.sh"
else
    source "$(dirname "$0")/_shot_design_common.sh"
fi
srun "$PY" scripts/shot_design/g_enc.py --device cuda --out "$ROOT/runs/genc_v4.json"
