#!/usr/bin/env python
"""Regenerate common-schema format/ tables from untouched manifest raw/ files."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from labeler.events.databases import (
    FORMAT_COLUMNS,
    FORMAT_SCHEMA_VERSION,
    UNITS,
    DatabaseError,
    TableSpec,
    load_manifest,
    write_format_table,
)


def csv_adapter(spec: TableSpec, path: Path) -> pd.DataFrame:
    """Map manifest-declared point/interval CSV columns into the common schema."""
    try:
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise DatabaseError(f"{path}: unreadable raw CSV: {exc}") from exc
    missing = set(spec.columns) - set(raw.columns)
    if missing:
        raise DatabaseError(f"{path}: missing raw columns {sorted(missing)}")
    times = []
    for col in spec.time_cols:
        values = pd.to_numeric(raw[col], errors="coerce")
        if not values.map(math.isfinite).all():
            raise DatabaseError(f"{path}: `{col}` must contain finite times")
        times.append(values.astype(float) * UNITS[spec.t_units])
    attrs_rows = []
    for row in raw.to_dict("records"):
        attrs = {"table": spec.stem}
        for col in spec.attr_cols:
            value = row[col]
            kind = spec.attr_types.get(col, "str")
            try:
                if value == "":
                    value = None
                elif kind in ("int", "float"):
                    value = float(value)
                    if not math.isfinite(value) or (kind == "int" and value % 1):
                        raise ValueError("non-finite or non-integral attribute")
                    if kind == "int":
                        value = int(value)
            except ValueError as exc:
                raise DatabaseError(f"{path}: invalid `{col}`: {exc}") from exc
            attrs[col] = value
        attrs_rows.append(json.dumps(attrs, sort_keys=True, allow_nan=False))
    return pd.DataFrame({
        "shot": raw[spec.shot_col], "t0_s": times[0], "t1_s": times[-1],
        "phenomenon": spec.phenomenon, "evidence_kind": "database",
        "source": spec.source, "confidence": float("nan"), "attrs": attrs_rows,
    }, columns=FORMAT_COLUMNS)


def rwm_onsets_2017(spec: TableSpec, path: Path) -> pd.DataFrame:
    """Hansen 2017: millisecond point onsets, NTOR and MODE_TYPE attributes."""
    return csv_adapter(spec, path)


def rwm_onsets_2024(spec: TableSpec, path: Path) -> pd.DataFrame:
    """Hansen 2024: the same raw columns, independently registered provenance."""
    return csv_adapter(spec, path)


ADAPTERS = {
    "csv": csv_adapter,
    "rwm_onsets_2017": rwm_onsets_2017,
    "rwm_onsets_2024": rwm_onsets_2024,
}


def convert(spec: TableSpec, root: Path) -> Path:
    """Convert one table; raw bytes and a manifest timestamp determine the output."""
    if spec.converter not in ADAPTERS:
        raise DatabaseError(f"{spec.stem}: unknown converter {spec.converter!r}")
    raw_path = spec.raw_path(root)
    frame = ADAPTERS[spec.converter](spec, raw_path)
    out = root / spec.dir / "format" / f"{spec.format_stem}.csv"
    write_format_table(frame, out, {
        "schema_version": FORMAT_SCHEMA_VERSION,
        "made_from": {
            "raw_file": raw_path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        },
        # Stable provenance identifier: reproduces the committed metadata bytes.
        "made_by": f"scripts/labelmaker/labels_format.py:{spec.converter}",
        "made_at": spec.made_at,
    })
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2]
                        / "data/events", help="Root containing tables.yaml")
    parser.add_argument("--table", action="append", help="Convert only this stem")
    args = parser.parse_args(argv)
    specs = load_manifest(args.root)
    if args.table:
        unknown = set(args.table) - {s.stem for s in specs}
        if unknown:
            parser.error(f"unknown table(s): {sorted(unknown)}")
        specs = tuple(s for s in specs if s.stem in args.table)
    for spec in specs:
        print(convert(spec, args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
