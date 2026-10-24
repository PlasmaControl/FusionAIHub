#!/usr/bin/env python
"""Regrid every saved review label, with backups; without --apply, write nothing."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from labeler.config import Paths, atomic_path
from labeler.events.grid import GRID_MS, snap, snap_span
from labeler.events.interval_tables import (
    INTERVAL_COLUMNS,
    WITH_ATTRS,
    attrs_text,
    parse_attrs,
    validate_intervals,
)
from labeler.events.review import labels, saves

BACKUP_ROOT = Path(
    "/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/backups/regrid_2026-10-10"
)


def _keep_attrs(label: labels.Label, rows: pd.DataFrame) -> labels.Label:
    """Keep extra row metadata on the portions its annotation still paints."""
    if label.attrs or "attrs" not in rows or not any(
        set(parse_attrs(cell)) - {"iscrowd"} for cell in rows["attrs"]
    ):
        return label
    sources = [
        (*snap_span(row.t_start, row.t_end), int(row.category), parse_attrs(row.attrs))
        for row in rows.itertuples() if row.t_end > row.t_start
    ]
    spans, cells = [], []
    for row in label.rows(0):
        _, category, start, stop, *_ = row
        default = parse_attrs(row[5]) if len(row) == 6 else {}
        matches = [
            (a, b, attrs) for a, b, c, attrs in sources
            if c == category and (
                not c or attrs.get("iscrowd") == default.get("iscrowd")
            )
            and a < stop and b > start
        ]
        edges = sorted({start, stop, *(
            edge for a, b, _ in matches for edge in (a, b) if start < edge < stop
        )})
        pieces = []
        for a, b in pairwise(edges):
            attrs = next(
                (attrs for lo, hi, attrs in reversed(matches) if lo <= a and b <= hi),
                default,
            )
            cell = attrs_text(attrs)
            if pieces and pieces[-1][1] == a and pieces[-1][2] == cell:
                pieces[-1] = (pieces[-1][0], b, cell)
            else:
                pieces.append((a, b, cell))
        for a, b, cell in pieces:
            spans.append((a, b, category))
            cells.append(cell)
    return labels.Label(label.window, tuple(spans), attrs=tuple(cells))


def _signature(frame: pd.DataFrame) -> list[tuple]:
    """Compare saved rows by values, apart from CSV numeric spelling."""
    return [
        (int(row.category), float(row.t_start), float(row.t_end),
         None if pd.isna(row.confidence) else float(row.confidence),
         attrs_text(getattr(row, "attrs", "")))
        for row in frame.itertuples()
    ]


def _span_counts(rows: pd.DataFrame, label: labels.Label | None) -> tuple[int, int]:
    evidence = "attrs" in rows and any(
        labels.EVIDENCE_ATTRS.intersection(parse_attrs(cell)) for cell in rows["attrs"]
    )
    annotations = [row for row in rows.itertuples() if row.category]
    kept = set()
    if not evidence and label is not None:
        identified = labels.normalise(
            label.window,
            [(row.t_start, row.t_end, i + 1) for i, row in enumerate(annotations)],
            iscrowd=[parse_attrs(getattr(row, "attrs", "")).get("iscrowd")
                     for row in annotations],
        )
        kept = {i for _, _, i in identified.intervals}
    claims = dropped = 0
    index = 0
    for row in rows.itertuples():
        a, b, c = row.t_start, row.t_end, int(row.category)
        if evidence:
            dropped += snap(b) <= snap(a)
            continue
        if not c:
            continue
        index += 1
        if b == a:
            dropped += 1
            continue
        claims += snap(a) == snap(b)
        dropped += index not in kept
    return int(claims), int(dropped)


def _child_violations(path: Path, found: dict[int, labels.Label]) -> list[dict]:
    """Report child annotation pieces outside the parent's snapped saved region."""
    child_dir = path.parent.parent
    if child_dir.parent.name != "subcategories":
        return []
    parent_dir = child_dir.parent.parent
    parents = labels.read_saved(parent_dir, grid_ms=GRID_MS)
    violations = []
    for shot, label in found.items():
        region = labels.region_of(parent_dir.name, parents.get(shot))
        allowed = () if region is None else region.spans
        for a, b, c in label.intervals:
            if not c:
                continue
            outside, end = [], a
            for lo, hi in allowed:
                if hi <= end or lo >= b:
                    continue
                if lo > end:
                    outside.append([end, lo])
                end = max(end, min(b, hi))
            if end < b:
                outside.append([end, b])
            if outside:
                violations.append({
                    "shot": shot, "span": [a, b, c], "outside": outside,
                    "parent": parent_dir.name,
                })
    return violations


def regrid(
    label_tables, *, apply=False, force=False, backup_root=BACKUP_ROOT,
) -> dict:
    """Prepare every replacement before writing; keep history files untouched."""
    root, backups = Path(label_tables), Path(backup_root)
    paths = sorted(
        root.rglob("review/labels.csv"),
        key=lambda path: (len(path.relative_to(root).parts), str(path.relative_to(root))),
    )
    if apply and not force:
        for path in paths:
            backup = backups / path.relative_to(root)
            if backup.exists():
                raise FileExistsError(f"backup already exists: {backup}; use --force")
    report = {
        "label_tables": str(root), "grid_ms": GRID_MS, "applied": bool(apply),
        "files": [],
    }
    replacements = []
    for path in paths:
        read_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        found = labels.read_labels(path, grid_ms=GRID_MS)
        before = validate_intervals(pd.read_csv(path, index_col=False))
        columns = WITH_ATTRS if "attrs" in before or any(
            label.iscrowd or label.attrs for label in found.values()
        ) else INTERVAL_COLUMNS
        output, changed, claims, dropped = [], [], 0, 0
        for shot, rows in before.groupby("shot", sort=True):
            shot = int(shot)
            label = found.get(shot)
            midpoint, vanished = _span_counts(rows, label)
            claims += midpoint
            dropped += vanished
            written = [] if label is None else _keep_attrs(label, rows).rows(shot)
            if columns == WITH_ATTRS:
                written = [row if len(row) == 6 else [*row, ""] for row in written]
            replacement = pd.DataFrame(written, columns=list(columns))
            if _signature(rows) != _signature(validate_intervals(replacement)):
                changed.append(shot)
            output.extend(written)
        frame = pd.DataFrame(output, columns=list(columns))
        validate_intervals(frame)
        report["files"].append({
            "path": str(path.relative_to(root)),
            "rows_before": len(before), "rows_after": len(frame),
            "shots_changed": changed, "midpoint_claims": claims,
            "spans_dropped": dropped, "child_violations": _child_violations(path, found),
        })
        replacements.append((path, frame, read_sha256))
    if apply:
        for path, frame, read_sha256 in replacements:
            backup = backups / path.relative_to(root)
            backup.parent.mkdir(parents=True, exist_ok=True)
            # The folder's merge lock keeps a save's merge from rewriting the table
            # under this write; a table that changed since it was read is not
            # overwritten (a rerun reads it again).
            with saves.lock(path.parent):
                if hashlib.sha256(path.read_bytes()).hexdigest() != read_sha256:
                    raise RuntimeError(f"{path} changed since it was read; rerun")
                # A rerun keeps the first backup: it is the only pre-grid copy.
                if not backup.exists():
                    shutil.copy2(path, backup)
                with saves.replacing(path) as tmp:
                    frame.to_csv(tmp, index=False)
        # A rerun's report goes beside the first, which lists what the grid changed.
        name = "report.json"
        if (backups / name).exists():
            name = f"report.rerun-{datetime.now(UTC):%Y%m%dT%H%M%S}.json"
        with atomic_path(backups / name) as tmp:
            tmp.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="back up and rewrite tables")
    parser.add_argument("--force", action="store_true", help="rerun over existing backups (the first backup is kept)")
    parser.add_argument("--label-tables", type=Path, default=Paths.from_env().label_tables)
    parser.add_argument("--backup-root", type=Path, default=BACKUP_ROOT)
    args = parser.parse_args()
    report = regrid(
        args.label_tables, apply=args.apply, force=args.force, backup_root=args.backup_root,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
