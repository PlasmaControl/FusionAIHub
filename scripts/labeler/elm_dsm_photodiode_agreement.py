#!/usr/bin/env python
"""Compare the upstream WPQH photodiode export with a fresh fetch on the swap shots.

The reduced-input DSM detection adaptation took PCPHD02/03 from a fresh fetch on 111
reviewed shots and from the upstream export (`dalpha_wpqh.pkl`) on the 8 legacy-swap
shots. `elm_dsm_fetch.py --native-photodiodes --fresh-shots ...` fetched those 8
fresh. This script reads both sources for each shot and column, and records how the
raw samples and the 50 ms row means the detector reads compare, so the scale of the
two sources can be judged without retraining. Offline: no network access.

    python scripts/labeler/elm_dsm_photodiode_agreement.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from labeler.config import Paths, git_sha, sha256_of
from labeler.elm import dsm
from labeler.timebase import window_mean

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/labeler/elm/dsm/swap_photodiode_agreement.json"
SWAP_SHOTS = (189885, 190637, 190643, 192721, 192732, 192751, 196541, 200385)
COLUMNS = ("pcphd02", "pcphd03")


def compare(upstream: dict, fresh_path: Path) -> dict:
    t0, y0 = np.ravel(upstream["times"]), np.ravel(upstream["data"])
    with np.load(fresh_path) as z:
        t1, y1 = np.ravel(z["x"]), np.ravel(z["y"])
    out = {
        "upstream_samples": int(t0.size),
        "fresh_samples": int(t1.size),
        "same_length": bool(t0.size == t1.size),
        "fresh_sha256": sha256_of(fresh_path),
    }
    if t0.size == t1.size:
        out["max_abs_time_difference_ms"] = float(np.max(np.abs(t0 - t1)))
        out["max_abs_value_difference"] = float(np.max(np.abs(y0 - y1)))
        out["identical_samples"] = bool(np.array_equal(y0, y1))
    rows = dsm.ROW_T_MS
    a = window_mean(t0, y0, rows - dsm.WINDOW_MS, dsm.WINDOW_MS)
    b = window_mean(t1, y1, rows - dsm.WINDOW_MS, dsm.WINDOW_MS)
    both = np.isfinite(a) & np.isfinite(b)
    out["rows_both_finite"] = int(both.sum())
    out["rows_finite_upstream_only"] = int((np.isfinite(a) & ~np.isfinite(b)).sum())
    out["rows_finite_fresh_only"] = int((~np.isfinite(a) & np.isfinite(b)).sum())
    if both.any():
        a, b = a[both], b[both]
        scale = max(float(np.median(np.abs(a))), 1e-12)
        out["median_abs_row_mean_upstream"] = float(np.median(np.abs(a)))
        out["median_abs_row_mean_fresh"] = float(np.median(np.abs(b)))
        out["max_abs_row_mean_difference"] = float(np.max(np.abs(a - b)))
        out["max_abs_row_mean_difference_over_median_abs"] = float(
            np.max(np.abs(a - b)) / scale
        )
        out["row_mean_correlation"] = (
            float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None
        )
    return out


def main() -> int:
    paths = Paths.from_env()
    upstream = dsm.upstream_photodiodes()
    store = paths.root / "round4/elm/dsm/native_photodiodes"
    shots = {}
    for shot in SWAP_SHOTS:
        shots[str(shot)] = {
            name: compare(upstream[str(shot)][name], store / f"{shot}_{name}.npz")
            for name in COLUMNS
        }
    rows = [r for s in shots.values() for r in s.values()]
    summary = {
        "records": len(rows),
        "identical_samples": sum(bool(r.get("identical_samples")) for r in rows),
        "max_abs_value_difference": max(
            r.get("max_abs_value_difference", float("nan")) for r in rows
        ),
        "max_abs_row_mean_difference": max(
            r.get("max_abs_row_mean_difference", float("nan")) for r in rows
        ),
        "max_abs_row_mean_difference_over_median_abs": max(
            r.get("max_abs_row_mean_difference_over_median_abs", float("nan"))
            for r in rows
        ),
        "min_row_mean_correlation": min(
            r["row_mean_correlation"]
            for r in rows
            if r.get("row_mean_correlation") is not None
        ),
    }
    record = {
        "git": git_sha(full=True),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "script_sha256": sha256_of(Path(__file__)),
        "purpose": "scale agreement of the upstream PCPHD02/03 export with a fresh "
        "fetch on the eight legacy-swap shots",
        "upstream": "/projects/EKOLEMEN/wpqh_elm_hiro/data/dalpha_wpqh.pkl",
        "fresh_store": str(store),
        "row_grid": "dsm.ROW_T_MS rows, 50 ms mean ending at each row stamp",
        "summary": summary,
        "shots": shots,
    }
    OUT.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
