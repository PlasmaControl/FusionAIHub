"""The checks a catalog table must pass; writers and the exporter refuse on any.

Each check takes tables and returns findings, one per problem, so a report lists
every problem at once, and `require` turns a non-empty list into an error.

- `tiling`: each shot's rows, in time order, tile its window: no gap, no overlap,
  no row without length;
- `states`: every state is 0-3;
- `attrs`: attribute keys and values are the phenomenon's;
- `windows`: each shot's window lies inside its allowed window (the default
  assessed window from Ip); shots with no allowed window are not catalog shots
  (the AE180 relabels) and are left alone;
- `points`: the points table is well formed and each point lies inside its
  shot's assessed window `[start, end)`.

`python -m labeler.events.catalog.check` runs them over every labels and points
table under each category's `review/`.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ...config import Paths
from ..databases import DatabaseError
from ..interval_tables import ATTRS_COLUMN, parse_attrs, validate_intervals
from .points import read_points, validate_points
from .states import PHENOMENA, STATE_NAMES, attr_problems

Windows = Mapping[int, tuple[float, float]]


@dataclass(frozen=True)
class Finding:
    """One problem: which check, which table, which shot, and what."""

    check: str
    where: str
    shot: int | None
    detail: str

    def __str__(self) -> str:
        shot = "" if self.shot is None else f" shot {self.shot}:"
        return f"{self.where}: {self.check}:{shot} {self.detail}"


class CatalogError(ValueError):
    """A catalog table failed its checks; the message lists the findings."""


def require(findings: list[Finding], limit: int = 20) -> None:
    """Raise `CatalogError` listing the findings, if there are any."""
    if findings:
        shown = "\n".join(str(f) for f in findings[:limit])
        more = len(findings) - limit
        raise CatalogError(shown + (f"\n... and {more} more" if more > 0 else ""))


def _ms(t: float) -> str:
    return f"{t:g} ms"


def tiling(frame: pd.DataFrame, where: str = "labels") -> list[Finding]:
    out = []
    for shot, rows in frame.groupby("shot", sort=True):
        rows = rows.sort_values(["t_start", "t_end"], kind="stable")
        starts, ends = rows.t_start.to_numpy(), rows.t_end.to_numpy()
        for a in starts[ends <= starts]:
            detail = f"a row at {_ms(a)} has no length"
            out.append(Finding("tiling", where, int(shot), detail))
        for end, start, next_end in zip(ends[:-1], starts[1:], ends[1:]):
            if start > end:
                detail = f"a gap from {_ms(end)} to {_ms(start)}"
            elif start < end:
                overlap_end = _ms(min(end, next_end))
                detail = f"an overlap from {_ms(start)} to {overlap_end}"
            else:
                continue
            out.append(Finding("tiling", where, int(shot), detail))
    return out


def states(frame: pd.DataFrame, where: str = "labels") -> list[Finding]:
    bad = frame[~frame.category.isin(list(STATE_NAMES))]
    return [
        Finding(
            "states",
            where,
            int(r.shot),
            f"state {r.category} at {_ms(r.t_start)} is not 0-3",
        )
        for r in bad.itertuples()
    ]


def attrs(frame: pd.DataFrame, category: str, where: str = "labels") -> list[Finding]:
    if ATTRS_COLUMN not in frame:
        return []
    return [
        Finding("attrs", where, int(r.shot), f"at {_ms(r.t_start)}: {problem}")
        for r in frame.itertuples()
        for problem in attr_problems(category, parse_attrs(getattr(r, ATTRS_COLUMN)))
    ]


def assessed(frame: pd.DataFrame) -> dict[int, tuple[float, float]]:
    """Each shot's assessed window `(first start, last end)`, ms."""
    spans = frame.groupby("shot").agg(lo=("t_start", "min"), hi=("t_end", "max"))
    return {int(s): (float(r.lo), float(r.hi)) for s, r in spans.iterrows()}


def windows(
    frame: pd.DataFrame, allowed: Windows, where: str = "labels"
) -> list[Finding]:
    out = []
    for shot, (lo, hi) in assessed(frame).items():
        span = allowed.get(shot)
        if span is not None and (lo < span[0] or hi > span[1]):
            detail = f"window {lo:g}-{hi:g} ms is outside {span[0]:g}-{span[1]:g}"
            out.append(Finding("windows", where, shot, detail + " ms"))
    return out


def points(
    frame: pd.DataFrame, labels: pd.DataFrame, category: str, where: str = "points"
) -> list[Finding]:
    try:
        frame = validate_points(frame)
    except DatabaseError as error:
        return [Finding("points", where, None, str(error))]
    out = []
    spans = assessed(labels)
    for r in frame.itertuples():
        shot, at = int(r.shot), f"{r.kind} at {_ms(r.t_ms)}"
        if r.phenomenon != category:
            detail = f"{at} is a {r.phenomenon} point"
            out.append(Finding("points", where, shot, detail))
            continue
        span = spans.get(shot)
        if span is None or not span[0] <= r.t_ms < span[1]:
            detail = f"{at} is outside the assessed window"
            out.append(Finding("points", where, shot, detail))
        for problem in attr_problems(category, parse_attrs(r.attrs)):
            out.append(Finding("attrs", where, shot, f"{at}: {problem}"))
    return out


def check_table(
    labels: pd.DataFrame,
    category: str,
    *,
    allowed: Windows | None = None,
    points_frame: pd.DataFrame | None = None,
    where: str = "labels",
) -> list[Finding]:
    """Every check on one labels table (and its points table, if any)."""
    try:
        labels = validate_intervals(labels)
    except DatabaseError as error:
        return [Finding("schema", where, None, str(error))]
    found = tiling(labels, where) + states(labels, where)
    found += attrs(labels, category, where)
    if allowed is not None:
        found += windows(labels, allowed, where)
    if points_frame is not None:
        beside = where.replace("labels", "points")
        found += points(points_frame, labels, category, beside)
    return found


def check_category(
    event_dir: Path, *, allowed: Windows | None = None
) -> tuple[list[Finding], int]:
    """Check every `labels.csv` under `<event_dir>/review/`, and its `points.csv`."""
    event_dir = Path(event_dir)
    found, n = [], 0
    for path in sorted((event_dir / "review").rglob("labels.csv")):
        n += 1
        beside = path.with_name("points.csv")
        found += check_table(
            pd.read_csv(path, dtype={ATTRS_COLUMN: str}, keep_default_na=False),
            event_dir.name,
            allowed=allowed,
            points_frame=_raw_points(beside) if beside.is_file() else None,
            where=str(path.relative_to(event_dir.parent)),
        )
    return found, n


def _raw_points(path: Path) -> pd.DataFrame:
    """A points table as written, so `points` reports what is wrong with it."""
    try:
        return read_points(path)
    except DatabaseError:
        return pd.read_csv(path, dtype={"attrs": str}, keep_default_na=False)


def read_windows(path: Path) -> dict[int, tuple[float, float]]:
    """Allowed windows from a table with `shot, window_start_ms, window_end_ms`."""
    frame = pd.read_csv(path)
    return {
        int(r.shot): (float(r.window_start_ms), float(r.window_end_ms))
        for r in frame.itertuples()
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m labeler.events.catalog.check")
    parser.add_argument("categories", nargs="*", default=sorted(PHENOMENA))
    parser.add_argument("--root", type=Path, help="default: LABELER_LABEL_TABLES")
    parser.add_argument(
        "--windows", type=Path, help="a CSV of shot, window_start_ms, window_end_ms"
    )
    args = parser.parse_args(argv)
    for category in args.categories:
        if category not in PHENOMENA:
            parser.error(f"{category} is not a catalog phenomenon")
    root = Paths.from_env().label_tables if args.root is None else args.root
    allowed = None if args.windows is None else read_windows(args.windows)
    found, tables = [], 0
    for category in args.categories:
        more, n = check_category(root / category, allowed=allowed)
        found += more
        tables += n
    for finding in found:
        print(finding)
    print(f"{len(found)} finding(s) in {tables} table(s)")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
