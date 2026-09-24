"""Point events: disruption times, ELM times and sawtooth crash times.

A points table has one row per point: `shot, phenomenon, kind, t_ms, attrs,
window_start_ms, window_end_ms`. `t_ms` keeps the signal's own resolution, so it
is a float. The two window columns are set, together, for a point checked inside
a blind window `[start, end)`, and are blank otherwise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ...config import atomic_path
from ..databases import DatabaseError
from ..interval_tables import attrs_text
from .states import PHENOMENA

POINT_COLUMNS = (
    "shot",
    "phenomenon",
    "kind",
    "t_ms",
    "attrs",
    "window_start_ms",
    "window_end_ms",
)
WINDOW_COLUMNS = ("window_start_ms", "window_end_ms")


def validate_points(frame: pd.DataFrame) -> pd.DataFrame:
    """The table with its types fixed; `DatabaseError` names the first problem."""
    if tuple(frame.columns) != POINT_COLUMNS:
        raise DatabaseError(f"Expected columns {POINT_COLUMNS}")
    result = frame.copy()
    shots = pd.to_numeric(result.shot, errors="coerce")
    if not (np.isfinite(shots) & (shots >= 0) & (shots % 1 == 0)).all():
        raise DatabaseError("shot must be a nonnegative integer")
    result["shot"] = shots.astype("int64")
    for row in result[["phenomenon", "kind"]].drop_duplicates().itertuples():
        kinds = PHENOMENA[row.phenomenon].points if row.phenomenon in PHENOMENA else ()
        if row.kind not in kinds:
            raise DatabaseError(f"{row.phenomenon} has no point kind {row.kind!r}")
    result["t_ms"] = pd.to_numeric(result.t_ms, errors="coerce")
    if not np.isfinite(result.t_ms).all():
        raise DatabaseError("t_ms must be a finite number")
    result["attrs"] = result["attrs"].map(attrs_text)
    lo = pd.to_numeric(result.window_start_ms, errors="coerce")
    hi = pd.to_numeric(result.window_end_ms, errors="coerce")
    if (lo.isna() != hi.isna()).any():
        raise DatabaseError("window_start_ms and window_end_ms are set together")
    checked = lo.notna()
    if not (lo[checked] < hi[checked]).all():
        raise DatabaseError("a point window must end after it starts")
    t = result.t_ms[checked]
    if not ((t >= lo[checked]) & (t < hi[checked])).all():
        raise DatabaseError("a checked point must lie inside its window")
    result["window_start_ms"], result["window_end_ms"] = lo, hi
    return result


def read_points(path) -> pd.DataFrame:
    """A points table; a missing file is an empty one."""
    path = Path(path)
    if not path.is_file():
        return pd.DataFrame(columns=list(POINT_COLUMNS))
    blank = {column: [""] for column in WINDOW_COLUMNS}
    frame = pd.read_csv(
        path, dtype={"attrs": str}, keep_default_na=False, na_values=blank
    )
    return validate_points(frame)


def write_points(frame: pd.DataFrame, path) -> None:
    """Validate, sort by shot, phenomenon, time and kind, and write atomically."""
    frame = validate_points(frame).sort_values(
        ["shot", "phenomenon", "t_ms", "kind"], kind="stable"
    )
    with atomic_path(path) as tmp:
        frame.to_csv(tmp, index=False)
