#!/bin/bash
#SBATCH -A fus187
#SBATCH -p extended
#SBATCH -J sd-build
#SBATCH -N 1
#SBATCH -t 06:00:00
#SBATCH --output=/lustre/orion/fus187/proj-shared/nchen/shot_design/runs/slurm/%j.out
# `extended`, not `batch`: Frontier caps a 1-91 node job on `batch` at 2 h, and
# `sbatch --test-only` rejects this 6 h job there ("Requested walltime 360 greater than
# limit 120 minutes"). `extended` is the house partition for small long jobs -- it is what
# eval_dynamics.sh runs a one-node job on.
# Full rebuild of the shot database for one shot list (atomic: db.tmp, then a swap).
# A PRODUCTION WRITE -- it replaces $SHOT_DESIGN_DATA_ROOT/db. Env overrides: SHOT_LIST,
# BUILD_ARGS (e.g. `--reader corpus --no-encode`, `--limit 20` for a pilot).
# sbatch runs a spool COPY of this file, so `dirname "$0"` is not the repo; the submit
# directory is (every wrapper is submitted from the repo root). Local runs fall back.
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/scripts/slurm_frontier/_shot_design_common.sh"
srun -n1 -c56 "$PY" -m shot_design build --workers 48 \
    --list "${SHOT_LIST:-recommender_frontier_v1}" ${BUILD_ARGS:-}
