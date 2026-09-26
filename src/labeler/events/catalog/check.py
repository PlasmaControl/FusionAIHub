"""Catalog checks; writers and the exporter are to refuse on any finding.

Each check takes tables and returns findings, one per problem, so a report lists
every problem at once, and `require` turns a non-empty list into an error.

- `tiling`: each shot's rows, in time order, tile its window: no gap, no overlap,
  no row without length;
- `states`: every state is 0-3;
- `attrs`: attribute keys and values are the phenomenon's;
- `windows`: each shot's window lies inside its allowed window (the default
  assessed window from Ip); shots with no allowed window are not catalog shots
  (the AE180 relabels) and are left alone;
- `points`: the points table is well formed, each point's attrs are checked,
  and each point lies inside its shot's assessed window `[start, end)`;
  a disruption's t_D, t80 and t20 agree with each other and its present span.

`python -m labeler.events.catalog.check` runs them over every labels and points
table under each category's `review/`.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from numbers import Integral, Real
from pathlib import Path

import pandas as pd

from ...config import Paths
from ..databases import DatabaseError
from ..interval_tables import ATTRS_COLUMN, parse_attrs, validate_intervals
from .points import POINT_COLUMNS, read_points, validate_csv_fields, validate_points
from .states import PHENOMENA, STATE_NAMES, attr_problems

Windows = Mapping[int, tuple[float, float]]
CSV_ERRORS = (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError)


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
        if shot not in allowed:
            continue
        span = allowed[shot]
        problem = _allowed_problem(shot, span)
        if problem:
            out.append(Finding("windows", where, shot, problem))
        elif lo < span[0] or hi > span[1]:
            detail = f"window {lo:g}-{hi:g} ms is outside {span[0]:g}-{span[1]:g}"
            out.append(Finding("windows", where, shot, detail + " ms"))
    return out


def _allowed_problem(shot, span) -> str | None:
    prefix = f"allowed window for key {shot!r}: "
    if (
        isinstance(shot, bool)
        or not isinstance(shot, Integral)
        or not 0 <= shot < 2**63
    ):
        return prefix + "shot must be an integer in [0, 2**63)"
    valid = False
    if isinstance(span, Sequence) and len(span) == 2:
        try:
            valid = (
                all(
                    isinstance(value, Real)
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    for value in span
                )
                and span[0] < span[1]
            )
        except OverflowError:
            pass
    if not valid:
        return prefix + "bounds must be two finite numbers with start < end"
    return None


def points(
    frame: pd.DataFrame,
    labels: pd.DataFrame | None,
    category: str,
    where: str = "points",
) -> list[Finding]:
    try:
        frame = validate_points(frame)
    except DatabaseError as error:
        return [Finding("points", where, None, str(error))]
    out = []
    spans = assessed(labels) if labels is not None else {}
    for r in frame.itertuples():
        shot, at = int(r.shot), f"{r.kind} at {_ms(r.t_ms)}"
        if r.phenomenon != category:
            detail = f"{at} is a {r.phenomenon} point"
            out.append(Finding("points", where, shot, detail))
            continue
        span = spans.get(shot)
        if labels is not None and (span is None or not span[0] <= r.t_ms < span[1]):
            detail = f"{at} is outside the assessed window"
            out.append(Finding("points", where, shot, detail))
        for problem in attr_problems(category, parse_attrs(r.attrs)):
            out.append(Finding("attrs", where, shot, f"{at}: {problem}"))
    if category == "disruption":
        out += disruption_points(frame, labels, where)
    return out


def disruption_points(
    frame: pd.DataFrame, labels: pd.DataFrame | None, where: str
) -> list[Finding]:
    """One t_D near [t80, t20]; present agrees with those bounds within 1 ms."""
    frame = frame[frame.phenomenon == "disruption"]
    present = labels[labels.category == 1] if labels is not None else None
    shots = set(frame.shot)
    if present is not None:
        shots.update(present.shot)
    out = []
    kinds = {"t_D", "t80", "t20"}
    for shot in sorted(shots):
        rows = frame[frame.shot == shot]
        counts = rows.kind.value_counts()
        for kind in sorted(kinds):
            if counts.get(kind, 0) > 1:
                out.append(Finding("points", where, int(shot), f"more than one {kind}"))
        missing = kinds - set(counts.index)
        if len(rows) and missing:
            detail = f"missing point kinds: {', '.join(sorted(missing))}"
            out.append(Finding("points", where, int(shot), detail))
        spans = present[present.shot == shot] if present is not None else None
        if spans is not None and len(spans) and {"t80", "t20"} & missing:
            detail = "present span has no t80 and t20 pair"
            out.append(Finding("points", where, int(shot), detail))
        if counts.get("t80", 0) == counts.get("t20", 0) == 1:
            lo = float(rows.loc[rows.kind == "t80", "t_ms"].iloc[0])
            hi = float(rows.loc[rows.kind == "t20", "t_ms"].iloc[0])
            if lo >= hi:
                out.append(
                    Finding("points", where, int(shot), "t80 must be before t20")
                )
            elif counts.get("t_D", 0) == 1:
                at = float(rows.loc[rows.kind == "t_D", "t_ms"].iloc[0])
                # D17: max|dIp/dt| lies within the quench plus a margin of
                # max(2 ms scoring tolerance, one quarter of the quench).
                # The quarter covers an exponential quench's earlier steepest slope.
                margin = max(2.0, (hi - lo) / 4)
                if not lo - margin <= at <= hi + margin:
                    detail = (
                        f"D17: t_D at {_ms(at)} is outside the quench {lo:g}-{hi:g} ms "
                        f"(margin {margin:.3g} ms)"
                    )
                    out.append(Finding("points", where, int(shot), detail))
            if spans is not None:
                if len(spans) != 1:
                    detail = "t80 and t20 require exactly one present span"
                elif (
                    abs(float(spans.iloc[0].t_start) - lo) > 1
                    or abs(float(spans.iloc[0].t_end) - hi) > 1
                ):
                    detail = "present span bounds must be within 1 ms of t80 and t20"
                else:
                    continue
                out.append(Finding("points", where, int(shot), detail))
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
    found = []
    beside = "points" if where == "labels" else str(Path(where).with_name("points.csv"))
    if allowed is not None:
        valid_allowed = {}
        for shot, span in allowed.items():
            problem = _allowed_problem(shot, span)
            if problem:
                found.append(Finding("windows", where, None, problem))
            else:
                valid_allowed[shot] = span
        allowed = valid_allowed
    try:
        labels = validate_intervals(labels)
    except DatabaseError as error:
        found.append(Finding("schema", where, None, str(error)))
        if points_frame is not None:
            found += points(points_frame, None, category, beside)
        return found
    found += tiling(labels, where) + states(labels, where)
    found += attrs(labels, category, where)
    if allowed is not None:
        found += windows(labels, allowed, where)
    if points_frame is None and category == "disruption":
        points_frame = pd.DataFrame(columns=list(POINT_COLUMNS))
    if points_frame is not None:
        found += points(points_frame, labels, category, beside)
    return found


def check_category(
    event_dir: Path, *, allowed: Windows | None = None
) -> tuple[list[Finding], int]:
    """Check each review directory holding labels or points, counting it once."""
    event_dir = Path(event_dir)
    found, n = [], 0
    directories = {
        path.parent
        for name in ("labels.csv", "points.csv")
        for path in (event_dir / "review").rglob(name)
        if path.is_file()
    }
    for directory in sorted(directories):
        n += 1
        path, beside = directory / "labels.csv", directory / "points.csv"
        where = str(path.relative_to(event_dir.parent))
        points_where = str(beside.relative_to(event_dir.parent))
        labels, points_frame = None, None
        if not path.is_file():
            found.append(
                Finding(
                    "points",
                    points_where,
                    None,
                    "no labels.csv beside it, so its points have no assessed window",
                )
            )
        else:
            try:
                validate_csv_fields(path)
                labels = pd.read_csv(
                    path,
                    dtype={ATTRS_COLUMN: str},
                    keep_default_na=False,
                    index_col=False,
                )
            except CSV_ERRORS as error:
                found.append(Finding("schema", where, None, str(error)))
        if beside.is_file():
            try:
                points_frame = _raw_points(beside)
            except CSV_ERRORS as error:
                found.append(Finding("schema", points_where, None, str(error)))
        if labels is not None:
            found += check_table(
                labels,
                event_dir.name,
                allowed=allowed,
                points_frame=points_frame,
                where=where,
            )
        elif points_frame is not None:
            found += points(points_frame, None, event_dir.name, points_where)
    return found, n


def _raw_points(path: Path) -> pd.DataFrame:
    """A points table as written, so `points` reports what is wrong with it."""
    try:
        return read_points(path)
    except DatabaseError:
        validate_csv_fields(path)
        return pd.read_csv(
            path, dtype={"attrs": str}, keep_default_na=False, index_col=False
        )


def read_windows(path: Path) -> dict[int, tuple[float, float]]:
    """Allowed windows from a table with `shot, window_start_ms, window_end_ms`."""
    try:
        validate_csv_fields(path)
    except pd.errors.ParserError as error:
        raise CatalogError(str(error)) from error
    except (UnicodeDecodeError, OSError) as error:
        raise CatalogError(f"{path}: row 1: {error}") from error
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, index_col=False)
    except (*CSV_ERRORS, OSError) as error:
        raise CatalogError(f"{path}: row 1: {error}") from error
    columns = ("shot", "window_start_ms", "window_end_ms")
    if not set(columns) <= set(frame.columns):
        raise CatalogError(f"{path}: row 1: expected columns {columns}")
    if frame.empty:
        raise CatalogError(f"{path}: row 1: no windows in the table")
    allowed = {}
    for row, values in enumerate(frame[list(columns)].itertuples(index=False), 2):
        prefix = f"{path}: row {row}: "
        # Parse the integer separately to preserve all int64 shot IDs exactly.
        try:
            value = Decimal(values.shot)
            if not value.is_finite() or not 0 <= value < 2**63 or value % 1:
                raise ValueError
            shot = int(value)
        except (InvalidOperation, ValueError) as error:
            raise CatalogError(
                prefix + "shot must be an integer in [0, 2**63)"
            ) from error
        bounds = []
        for column, raw in zip(columns[1:], values[1:]):
            try:
                bound = float(raw)
                if not math.isfinite(bound):
                    raise ValueError
            except ValueError as error:
                raise CatalogError(
                    prefix + f"{column} must be a finite number"
                ) from error
            bounds.append(bound)
        lo, hi = bounds
        if lo >= hi:
            raise CatalogError(prefix + "window_start_ms must be before window_end_ms")
        if shot in allowed:
            raise CatalogError(prefix + f"duplicate shot {shot}")
        allowed[shot] = (lo, hi)
    return allowed


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
    if not root.is_dir():
        parser.error(f"root is not an existing directory: {root}")
    try:
        allowed = None if args.windows is None else read_windows(args.windows)
    except CatalogError as error:
        parser.error(str(error))
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
