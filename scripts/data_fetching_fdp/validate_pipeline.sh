#!/bin/bash
# Run on a login node: TigerData archival is performed here; heavy Python
# stages get separate CPU-only Slurm allocations through --slurm.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1
export WORKERS="${WORKERS:-8}"
exec pixi run --frozen -e stellarai python \
    scripts/data_fetching_fdp/validate_pipeline.py validate --slurm "$@"
