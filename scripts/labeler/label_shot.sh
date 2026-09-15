#!/usr/bin/env bash
# One command, one shot (or several): every implemented labeler model runs on
# the shot and you get the labels as HDF5 and as a numpy .npz, a JSON summary and
# one figure with a panel per prediction.
#
#   scripts/labeler/label_shot.sh 187199            # one shot
#   scripts/labeler/label_shot.sh 187199 186545     # several
#   LABELER_DEMO_OUT=/some/dir scripts/labeler/label_shot.sh 199597
#
# Output, per shot, under $LABELER_DEMO_OUT (default outputs/labelmaker/analysis in
# this repo; the cached features and the canonical labels file stay under $LABELER_ROOT):
#   <shot>/<shot>_labels.h5        every label:  <model>/<label>/{xdata,ydata}
#   <shot>/<shot>_labels.npz       the same as numpy: time_s + "<model>/<label>"
#   <shot>/<shot>_analysis.json    per label: rows, valid fraction, peak, first
#                                  threshold crossing, archived-truth scores
#   <shot>/<shot>_labels.png       the figure
#
# Which models and labels: src/labeler/analyze_default.yaml (pass your own
# with LABELER_DEMO_CONFIG=path). Fetching features that are not in the
# corpus needs an fdp token (`pixi run -e labelmaker fdp login`, once).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="$("/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/python" "$REPO/src/labeler/env.py" LABELER_ROOT "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")"
OUT="$("/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/python" "$REPO/src/labeler/env.py" LABELER_DEMO_OUT "$REPO/outputs/labelmaker/analysis")"
CONFIG="$("/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/python" "$REPO/src/labeler/env.py" LABELER_DEMO_CONFIG "$REPO/src/labeler/analyze_default.yaml")"

if [ "$#" -lt 1 ]; then
    echo "usage: $(basename "$0") SHOT [SHOT ...]" >&2
    exit 2
fi
for shot in "$@"; do
    case "$shot" in
        ''|*[!0-9]*) echo "not a shot number: $shot" >&2; exit 2 ;;
    esac
done

cd "$REPO"
FORCE="$("/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/python" "$REPO/src/labeler/env.py" LABELER_DEMO_FORCE)"
export LABELER_ROOT="$ROOT"
pixi run -q -e labelmaker fdp run python -m labeler.run analyze \
    --shots "$@" --config "$CONFIG" --out "$OUT" ${FORCE:+--force}

# Put the canonical label file next to the figure, and a numpy copy of it.
pixi run -q -e labelmaker python - "$ROOT" "$OUT" "$@" <<'PY'
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

root, out, *shots = sys.argv[1:]
for shot in shots:
    src = Path(root) / "labels" / f"{shot}_labels.h5"
    dst_dir = Path(out) / shot
    if not src.exists() or not dst_dir.exists():
        print(f"  {shot}: no labels written (see the analyze log above)")
        continue
    h5 = dst_dir / src.name
    shutil.copyfile(src, h5)
    arrays: dict[str, np.ndarray] = {}
    with h5py.File(h5, "r") as f:
        for slug, group in f.items():
            for name, ds in group.items():
                if "xdata" in ds and "time_s" not in arrays:
                    arrays["time_s"] = ds["xdata"][()]
                arrays[f"{slug}/{name}"] = ds["ydata"][()]
    npz = h5.with_suffix(".npz")
    np.savez(npz, **arrays)
    n_labels = sum(1 for k in arrays if "/" in k and not k.endswith(("_valid", "_spread")))
    print(f"  {shot}: {h5}")
    print(f"  {shot}: {npz}  ({n_labels} labels, {len(arrays['time_s'])} rows)")
PY
