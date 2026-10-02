"""Curated RWM onsets and an explicitly unconfirmed high-beta magnetic screen.

N1RMS alone cannot distinguish an RWM from tearing or applied fields. The
screen therefore writes uncertain candidates, never present or absent. Curated
point annotations keep their times without acquiring an invented end time.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd

from ..config import Paths
from . import coverage, equilibrium
from .interval_tables import validate_intervals
from .verify import NoDataError

UNASSESSED = 4
NO_WALL_FACTOR = 4.0
MIN_MS = 10.0
NOISE_MAD = 6.0
RULE = {
    **equilibrium.RULE,
    "screen": "n1rms_high_beta",
    "beta_n_over_li": NO_WALL_FACTOR,
    "amplitude": "above the flat-top median + 6 median absolute deviations",
    "noise_mad": NOISE_MAD,
    "min_ms": MIN_MS,
    "classification": "uncertain candidates only; the rest is unassessed",
    "database": "onsets only, with no duration or negative coverage inferred",
    "limitations": "N1RMS is not RWM-specific and includes applied-field response",
    "timing": "postprocessed RMS; sampled excursions are not physical growth times",
}


def onset_table(paths: Paths) -> pd.DataFrame:
    """The latest curated format table, retaining point times and its identity."""
    directory = paths.label_tables / "resistive_wall_mode" / "format"
    path = max(directory.glob("*_format_*.csv"), key=lambda p: p.name, default=None)
    if path is None:
        return pd.DataFrame(columns=["shot", "t_ms", "source"])
    frame = validate_intervals(pd.read_csv(path))
    points = frame[(frame.category == 1) & (frame.t_start == frame.t_end)]
    details = defaultdict(list)
    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    for source in meta.get("made_from", []):
        raw_path = paths.label_tables / source["raw_file"]
        if not raw_path.is_file():
            continue
        original = pd.read_csv(raw_path)
        for row in original.to_dict("records"):
            key = (int(row["SHOT"]), round(float(row["ONSET_TIME"]), 9))
            details[key].append({"ntor": int(row["NTOR"]),
                                 "mode_type": str(row["MODE_TYPE"]),
                                 "raw_source": source["source"]})
    rows = []
    for row in points.itertuples():
        key = (int(row.shot), round(float(row.t_start), 9))
        attributes = details[key].pop(0) if details[key] else {}
        rows.append({"shot": int(row.shot), "t_ms": float(row.t_start),
                     "source": path.name, **attributes})
    return pd.DataFrame(rows, columns=["shot", "t_ms", "source", "ntor",
                                       "mode_type", "raw_source"])


def onsets(shot: int, paths: Paths) -> list[dict]:
    frame = onset_table(paths)
    found = frame.loc[frame.shot == int(shot)].drop(columns="shot")
    return [{k: int(v) if k == "ntor" else v for k, v in row.items() if pd.notna(v)}
            for row in found.to_dict("records")]


def _nearest(array, target_ms):
    """Nearest finite samples, only within half a native step of their own clock."""
    t = array.x * 1000
    values = np.asarray(array.y[0], dtype=float)
    # sample_runs validates the clock and the scalar shape before sampling.
    equilibrium.sample_runs(t, array.y)
    index = np.searchsorted(t, target_ms)
    left, right = np.maximum(index - 1, 0), np.minimum(index, len(t) - 1)
    index = np.where(np.abs(t[right] - target_ms) < np.abs(t[left] - target_ms),
                     right, left)
    out = values[index].copy()
    out[np.abs(t[index] - target_ms) > 0.5 * np.median(np.diff(t)) + 1e-6] = np.nan
    return out


def detect(shot: int, paths: Paths, window=None):
    from .spans import Found

    info = {"onsets": onsets(shot, paths)}
    try:
        amplitude, measured, inputs = equilibrium.gated(shot, "n1rms", paths, window)
        beta = equilibrium.signal(shot, "betan", paths)
        li = equilibrium.signal(shot, "li", paths)
        inputs.update(betan=dict(beta.attrs), li=dict(li.attrs))
    except (NoDataError, KeyError, ValueError, OSError) as error:
        return Found((), (), {**info, "not_run": str(error)})
    t, y = amplitude.x * 1000, amplitude.y[0]
    beta_y, li_y = _nearest(beta, t), _nearest(li, t)
    valid = np.isfinite(y) & np.isfinite(beta_y) & np.isfinite(li_y) & (li_y > 0)
    for array in (beta, li):
        measured = coverage.intersect_intervals(
            measured, equilibrium.sample_runs(array.x * 1000, array.y)
        )
    inside = np.zeros(len(t), dtype=bool)
    for a, b in measured:
        inside |= (t >= a) & (t <= b)
    baseline = y[inside & valid]
    if not baseline.size:
        return Found((), (), {**info, "not_run": "no overlapping finite screen inputs"})
    median = float(np.median(baseline))
    mad = float(np.median(np.abs(baseline - median)))
    floor = median + NOISE_MAD * max(mad, np.finfo(float).eps * max(1, abs(median)))
    candidate = valid & inside & (y > floor) & (beta_y > NO_WALL_FACTOR * li_y)
    runs = equilibrium.sample_runs(t, np.where(candidate, y, np.nan))
    spans = tuple((a, b, 2) for a, b in runs if b - a >= MIN_MS - 1e-6)
    # Nearest sampling can miss the support of a coarse EFIT sample; do not
    # claim that every N1RMS sample inside the display hull was assessed.
    observed = coverage.intersect_intervals(
        measured, equilibrium.sample_runs(t, np.where(valid, y, np.nan))
    )
    return Found(spans, observed, {**info, "screen": "n1rms_high_beta",
                                  "inputs": inputs,
                                  "amplitude_threshold_g": floor})


def database_windows(paths: Paths) -> pd.DataFrame:
    """Display windows for curated shots missing from the frozen review cohort.

    These are view extents, not RWM durations or observation coverage.
    """
    rows = []
    for shot, frame in onset_table(paths).groupby("shot", sort=True):
        rows.append([int(shot), min(0, int(np.floor(frame.t_ms.min())) - 500),
                     int(np.ceil(frame.t_ms.max())) + 500])
    return pd.DataFrame(rows, columns=["shot", "window_start_ms", "window_end_ms"])
