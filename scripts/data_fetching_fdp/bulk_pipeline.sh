#!/bin/bash
# Orchestrate from the login host; the two heavy stages use CPU Slurm jobs.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export WORKERS="${WORKERS:-4}"
exec pixi run --frozen -e stellarai python \
    scripts/data_fetching_fdp/bulk_pipeline.py "$@"
