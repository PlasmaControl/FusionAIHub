"""Suggestion tables: a method's labels for shots nobody has reviewed.

A suggestion table has the format-table columns (`shot, category, t_start,
t_end, confidence`, ms) and the catalog states as categories (0 absent,
1 present, 2 uncertain, 3 not observable); each shot's rows tile its window.
It lives at `$LABELER_ROOT/suggestions/<method>/<version>/<event>_suggest_<method>_<version>.csv`
beside a `.meta.json` naming what made it. It is a suggestion, not a label
(v1 spec §3): the review page can start from one (`review/source.json`), and
only what a reviewer saves becomes a label.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Paths, atomic_path
from ..scoring.frames import FRAME_MS
from .interval_tables import validate_intervals

COLUMNS = ("shot", "category", "t_start", "t_end", "confidence")


def table_path(paths: Paths, event: str, method: str, version: str) -> Path:
    name = f"{event}_suggest_{method}_{version}.csv"
    return paths.root / "suggestions" / method / version / name


def frame_rows(
    shot: int,
    first: int,
    states: Sequence[int],
    confidence: Sequence[float] | None = None,
) -> list[list]:
    """One row per run of equal frame states, frames `first ..` of `FRAME_MS`.

    A run's confidence is the mean of `confidence` over it, blank without one.
    """
    states = np.asarray(states, dtype=np.int64)
    if not len(states):
        return []
    edges = np.flatnonzero(np.diff(states)) + 1
    starts, stops = np.r_[0, edges], np.r_[edges, len(states)]
    rows = []
    for a, b in zip(starts, stops):
        conf = "" if confidence is None else round(float(np.mean(confidence[a:b])), 4)
        rows.append(
            [
                int(shot),
                int(states[a]),
                (first + a) * FRAME_MS,
                (first + b) * FRAME_MS,
                conf,
            ]
        )
    return rows


def span_rows(
    shot: int, window, spans: Iterable[tuple[float, float, int]]
) -> list[list]:
    """Rows tiling `window` (ms): `spans` `(start, stop, state)` painted in order
    over absent, clipped to the window, whole ms."""
    lo, hi = int(np.floor(window[0])), int(np.ceil(window[1]))
    cells = np.zeros(max(0, hi - lo), dtype=np.int64)
    for a, b, state in spans:
        start = max(int(np.floor(a)), lo) - lo
        stop = min(int(np.ceil(b)), hi) - lo
        if stop > start:
            cells[start:stop] = int(state)
    if not len(cells):
        return []
    edges = np.flatnonzero(np.diff(cells)) + 1
    starts, stops = np.r_[0, edges], np.r_[edges, len(cells)]
    return [
        [int(shot), int(cells[a]), lo + int(a), lo + int(b), ""]
        for a, b in zip(starts, stops)
    ]


def write_table(path, rows: Iterable[Sequence], meta: dict) -> pd.DataFrame:
    """Validate and write the table, sorted by shot and start, and its meta."""
    frame = pd.DataFrame(list(rows), columns=list(COLUMNS))
    frame = frame.sort_values(["shot", "t_start"], kind="stable", ignore_index=True)
    validate_intervals(frame.replace({"confidence": {"": np.nan}}))
    path = Path(path)
    with atomic_path(path) as tmp:
        frame.to_csv(tmp, index=False)
    with atomic_path(path.with_suffix(".meta.json")) as tmp:
        tmp.write_text(
            json.dumps(
                {**meta, "rows": len(frame), "shots": int(frame.shot.nunique())},
                indent=1,
            )
            + "\n"
        )
    return frame
