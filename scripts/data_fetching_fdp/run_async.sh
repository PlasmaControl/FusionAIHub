#!/bin/bash
# Both directions share reservations and the same four archive workers.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export WORKERS="${WORKERS:-4}"
# Concurrent fetches per direction. The login node caps a user at 188 GiB and each
# fetch stages several GB in /dev/shm, so 8 per direction got the driver OOM-killed.
PER_DIRECTION="${PER_DIRECTION:-4}"
exec pixi run --frozen -e stellarai python \
    scripts/data_fetching_fdp/async_pipeline.py \
    --ascending scratch/fdp_bulk_205000_20261028 \
    --descending scratch/fdp_bulk_below190000_20261028 \
    --control scratch/fdp_async_control --per-direction "$PER_DIRECTION" --archive-workers 4
