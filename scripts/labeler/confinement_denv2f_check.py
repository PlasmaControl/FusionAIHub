#!/usr/bin/env python
"""What ``\\BCI::DENV2F`` is, from the traces on disk: a density, not a raw voltage.

confine-ours reads this channel as its line-density input. The fetch script first
called it a CO2 interferometer voltage; its values (1e13 to 3e14) are not volts, so
this compares its flat-top level (the median over 2-4 s) with the Thomson core density
(the highest channel's median over the same time) of the same shots in the corpus:
the rank correlation, and the ratio to the Thomson peak in cm^-3. Reads the fetched
0D files and the corpus' ``ts_core_density``; fetches nothing. Output:
``outputs/labeler/confinement/ours/denv2f_check.json``.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
LABELER = Path(
    os.environ.get("LABELER_ROOT", "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")
)
ZEROD = LABELER / "round4/conf/zerod"
CORPUS = Path("/scratch/gpfs/EKOLEMEN/foundation_model")
OUT = REPO / "outputs/labeler/confinement/ours/denv2f_check.json"
WINDOW_MS = (2000.0, 4000.0)


def rank(values: np.ndarray) -> np.ndarray:
    return values.argsort().argsort().astype(float)


def flat_top(path: Path) -> float | None:
    """Median of DENV2F over the window, or None if the shot has no such trace."""
    with np.load(path) as d:
        if "dens" not in d.files or d["dens"].size < 500:
            return None
        v = d["dens"].astype(float)
        t = float(d["dens_t0_ms"]) + np.arange(v.size)
    sel = (t >= WINDOW_MS[0]) & (t <= WINDOW_MS[1]) & np.isfinite(v)
    return float(np.median(v[sel])) if sel.sum() >= 100 else None


def thomson_peak(path: Path) -> float | None:
    """Median over the window of the highest Thomson core channel (m^-3)."""
    with h5py.File(path) as h:
        if "ts_core_density" not in h:
            return None
        t = h["ts_core_density"]["xdata"][:] * 1000.0
        y = np.asarray(h["ts_core_density"]["ydata"][:], float)
    sel = (t >= WINDOW_MS[0]) & (t <= WINDOW_MS[1])
    if sel.sum() < 5:
        return None
    peak = float(np.nanmedian(np.nanmax(y[:, sel], axis=0)))
    return peak if peak > 0 else None


def main() -> int:
    rows = []
    for f in sorted(ZEROD.glob("*.npz")):
        shot = int(f.stem)
        corpus = CORPUS / f"{shot}_processed.h5"
        if not corpus.exists():
            continue
        dens, peak = flat_top(f), thomson_peak(corpus)
        if dens is not None and peak is not None:
            rows.append((shot, dens, peak))
    a = np.array(rows)
    ratio = a[:, 1] / (a[:, 2] * 1e-6)
    pct = [5, 25, 50, 75, 95]
    result = {
        "git": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "window_ms": list(WINDOW_MS),
        "shots": len(a),
        "denv2f_percentiles": dict(zip(pct, np.percentile(a[:, 1], pct).tolist())),
        "thomson_peak_m3_percentiles": dict(
            zip(pct, np.percentile(a[:, 2], pct).tolist())
        ),
        "spearman_denv2f_vs_thomson_peak": float(
            np.corrcoef(rank(a[:, 1]), rank(a[:, 2]))[0, 1]
        ),
        "ratio_to_thomson_peak_cm3": dict(zip(pct, np.percentile(ratio, pct).tolist())),
        "reading": (
            "values of 1e13 to 3e14 and a rank correlation with the Thomson density "
            "mean a density-like trace in cm^-3, not a voltage; its scale is about "
            "twice the Thomson peak in cm^-3, so it stays nominal"
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "reading"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
