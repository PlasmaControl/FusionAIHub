#!/bin/bash
# Build once on the login node for the WHOLE list before any GPU task starts.
# No GPU or SLURM allocation. Text goes in text/; the site accounting client
# is staged unchanged in runs/ because it is not installed on compute nodes.
set -euo pipefail
REPO="${REPO:-/scratch/gpfs/nc1514/FusionAIHub}"
ROOT="$("/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/python" "$REPO/src/labeler/env.py" LABELER_ROOT "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")"
SHOT_FILE="${SHOT_FILE:-$ROOT/recommender_v1.txt}"
export LABELER_ROOT="$ROOT" PYTHONPATH="$REPO/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
export PIXI_CACHE_DIR="${PIXI_CACHE_DIR:-/tmp/l12-pixi}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/l12-cache}"
cd "$REPO"
# Resolve the installed client on the LOGIN node. Keep its site configuration
# private, alongside the three unchanged Python files it needs. No installation
# or writes to the phase3/pixi environments or /usr/local are required.
JOBSTATS_SOURCE="$(dirname "$(readlink -f "$(command -v jobstats)")")"
install -d -m 700 "$ROOT/runs/slurm/jobstats-client"
install -m 600 "$JOBSTATS_SOURCE/config.py" "$JOBSTATS_SOURCE/jobstats.py" \
    "$JOBSTATS_SOURCE/output_formatters.py" "$ROOT/runs/slurm/jobstats-client/"
install -m 700 "$JOBSTATS_SOURCE/jobstats" "$ROOT/runs/slurm/jobstats-client/jobstats"
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml \
    -e labelmaker python "$REPO/scripts/labeler/tokeye_masks.py" \
    --shot-file "$SHOT_FILE" --root "$ROOT" --device cpu --build-text-subset
