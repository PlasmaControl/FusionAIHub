#!/bin/bash
# Train listed arm:seed runs one after another on one head-node GPU.
# Usage: bash scripts/labeler/ae_supervision_swap_head.sh GPU arm:seed [arm:seed ...]
# Each run is bounded by `timeout` (AESWAP_TIMEOUT, default 6h) and logs to
# $OUT/head-<arm>-seed<seed>-r1.log. The head node is shared: one process per GPU.
set -uo pipefail
GPU=${1:?GPU index required}
shift
REPO=${REPO:-/scratch/gpfs/nc1514/FusionAIHub}
ENV="${LABELER_PIXI_ENV:-/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labeler}"  # another cluster: export LABELER_PIXI_ENV
# The cluster's production root (labeler.env.CLUSTERS), whatever LABELER_ROOT the caller
# exports: the phase3 interpreter and round4/ live there.
LABELER_ROOT="$(env -u LABELER_ROOT "$ENV/bin/python" "$REPO/src/labeler/env.py" LABELER_ROOT "/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler")" || exit 1
LABELER_LABEL_TABLES="$("$ENV/bin/python" "$REPO/src/labeler/env.py" LABELER_LABEL_TABLES "/scratch/gpfs/EKOLEMEN/D3D/events")" || exit 1
export LABELER_ROOT LABELER_LABEL_TABLES
# Short on purpose: an AF_UNIX socket path must stay under 108 bytes.
export TMPDIR="$LABELER_ROOT/scratch/ae-sw"
export LABELER_NO_FETCH=1 PYTHONPATH="$REPO/src" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT="$LABELER_ROOT/round4/aeswap"
PYTHON=${AESWAP_PYTHON:-$LABELER_ROOT/envs/phase3/bin/python}
mkdir -p "$TMPDIR"
cd "$REPO"
for RUN in "$@"; do
    ARM=${RUN%%:*}
    SEED=${RUN##*:}
    echo "head start $ARM seed $SEED on GPU $GPU $(date -Is)"
    CUDA_VISIBLE_DEVICES=$GPU timeout "${AESWAP_TIMEOUT:-6h}" "$PYTHON" -u \
        scripts/labeler/ae_supervision_swap.py train \
        --supervision "$ARM" --seed "$SEED" > "$OUT/head-$ARM-seed$SEED-r1.log" 2>&1
    echo "head finished $ARM seed $SEED exit $? $(date -Is)"
    bash "$LABELER_ROOT/scratch/bin/tmpsweep.sh"
done
