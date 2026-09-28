"""Point events: disruption times, ELM times and sawtooth crash times.

A points table has one row per point: `shot, phenomenon, kind, t_ms, attrs,
window_start_ms, window_end_ms`. `t_ms` keeps the signal's own resolution, so it
is a float. The two window columns are set, together, for a point checked inside
a blind window `[start, end)`, and are blank otherwise.
"""

from __future__ import annotations

import csv
import io
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


def validate_csv_fields(path) -> list[int]:
    """Refuse ragged records and duplicate headers before pandas changes them.

    Skip empty records and return each record's starting physical line number.
    Quoted commas and newlines belong to their field, not to the table structure.
    Seekable text/binary streams are inspected without changing their position.
    """
    row, width = 0, None
    lines = []
    try:
        if hasattr(path, "read"):
            position = path.tell()
            try:
                data = path.read()
            finally:
                path.seek(position)
            data = data.decode("utf-8-sig") if isinstance(data, bytes) else data
        else:
            data = Path(path).read_text(encoding="utf-8-sig")
        with io.StringIO(data, newline="") as source:
            reader = csv.reader(source, strict=True)
            while True:
                row = reader.line_num + 1
                fields = next(reader, None)
                if fields is None:
                    break
                if not fields:
                    continue
                lines.append(row)
                if width is None:
                    width = len(fields)
                    repeated = sorted(
                        {name for name in fields if fields.count(name) > 1}
                    )
                    if repeated:
                        raise pd.errors.ParserError(
                            f"{path}: row {row}: duplicate headers: "
                            + ", ".join(repeated)
                        )
                elif len(fields) != width:
                    raise pd.errors.ParserError(
                        f"{path}: row {row}: expected {width} fields, got {len(fields)}"
                    )
    except csv.Error as error:
        raise pd.errors.ParserError(
            f"{path}: row {row}: error tokenizing CSV: {error}"
        ) from error
    return lines


def validate_points(frame: pd.DataFrame) -> pd.DataFrame:
    """The table with its types fixed; `DatabaseError` names the first problem."""
    if tuple(frame.columns) != POINT_COLUMNS:
        raise DatabaseError(f"Expected columns {POINT_COLUMNS}")
    result = frame.copy()
    shots = pd.to_numeric(result.shot, errors="coerce")
    if not (
        np.isfinite(shots) & (shots >= 0) & (shots < 2**63) & (shots % 1 == 0)
    ).all():
        raise DatabaseError("shot must be a nonnegative integer")
    result["shot"] = shots.astype("int64")
    for row in result[["phenomenon", "kind"]].drop_duplicates().itertuples():
        kinds = PHENOMENA[row.phenomenon].points if row.phenomenon in PHENOMENA else ()
        if row.kind not in kinds:
            raise DatabaseError(f"{row.phenomenon} has no point kind {row.kind!r}")
    for column in ("phenomenon", "kind"):
        result[column] = result[column].astype(str)
    result["t_ms"] = pd.to_numeric(result.t_ms, errors="coerce").astype("float64")
    if not np.isfinite(result.t_ms).all():
        raise DatabaseError("t_ms must be a finite number")
    repeated = result.duplicated(["shot", "phenomenon", "kind", "t_ms"])
    if repeated.any():
        row = result.loc[repeated].iloc[0]
        raise DatabaseError(
            f"repeated point: shot {row.shot}, {row.phenomenon}, "
            f"{row.kind} at {row.t_ms:g} ms"
        )
    result["attrs"] = result["attrs"].map(attrs_text).astype(str)
    for column in WINDOW_COLUMNS:
        raw = result[column]
        values = pd.to_numeric(raw, errors="coerce").astype("float64")
        blank = raw.isna() | (raw == "")
        if (~blank & ~np.isfinite(values)).any():
            raise DatabaseError(f"{column} must be a finite number when set")
        result[column] = values
    lo, hi = result.window_start_ms, result.window_end_ms
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
        return validate_points(pd.DataFrame(columns=list(POINT_COLUMNS)))
    validate_csv_fields(path)
    blank = {column: [""] for column in WINDOW_COLUMNS}
    frame = pd.read_csv(
        path,
        dtype={"attrs": str},
        keep_default_na=False,
        na_values=blank,
        index_col=False,
    )
    return validate_points(frame)


def write_points(frame: pd.DataFrame, path) -> None:
    """Validate, sort by shot, phenomenon, time and kind, and write atomically."""
    frame = validate_points(frame).sort_values(
        ["shot", "phenomenon", "t_ms", "kind"], kind="stable"
    )
    with atomic_path(path) as tmp:
        frame.to_csv(tmp, index=False)
