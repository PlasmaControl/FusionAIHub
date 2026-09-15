"""Common label-table schema and the format/ curated-table loader.

Original lists live in raw/ and only scripts/labeler/labels_format.py
converts them. This module reads format/<format_stem>.csv, preserving its
stored evidence kind, confidence and JSON attributes. A listing is not a
coverage claim: events and source records keep NaN coverage, and a shot
absent from every table contributes neither an event nor a source row.
"""
from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..config import Paths, atomic_path
from .lexicon import LexiconError, load_lexicon
from .schema import EVIDENCE_KINDS, Event

#: One CSV projection for all format/ and extend_<producer>/ event tables.
FORMAT_COLUMNS = (
    "shot", "t0_s", "t1_s", "phenomenon", "evidence_kind", "source",
    "confidence", "attrs",
)
FORMAT_SCHEMA_VERSION = 1

#: The manifest's name inside the label-tables root.
MANIFEST = "tables.yaml"

#: What every row of a table claims, and who claims it. The prefix is kept
#: literal - `source` is a column, not a filename - and the resulting
#: `event_id` is the schema's own `{shot}-database:<stem>-{n:05d}`.
SOURCE_PREFIX = "database:"
EVIDENCE_KIND = "database"

#: What a row's times mean.
KINDS = ("point", "interval")

#: Multiplier onto seconds. Deliberately small: a unit that is not here is
#: an error and not a guess.
UNITS = {"ms": 1e-3, "s": 1.0}

#: How an `attr_cols` column is typed on the way into `attrs`. `attrs` is
#: stored as JSON, so these are the three types JSON has that a table
#: column can be.
ATTR_TYPES = ("int", "float", "str")

#: Why the source record's coverage is NaN, in one sentence, for the docs
#: and for whoever reads a curated row back. It is NOT written to the
#: sources file: `schema._source_row` refuses a `reason` on an `ok` row
#: (task Lfix-C1's contract, and it is right - a reason is what a non-`ok`
#: row owes), so the record carries `reason=""` and the NaN COVERAGE is
#: the signal. That is unambiguous on the wire because no detector writes
#: `ok` with NaN coverage: a curated source is `status="ok"`, coverage
#: NaN, and a `source` beginning `database:`.
COVERAGE_REASON = (
    "curated list; coverage unknown (a listing is not a coverage claim)"
)

_NAN = float("nan")

#: `(path, mtime_ns, size)` -> the parsed frame. A `--databases-only` run
#: asks every table about every shot, so the CSV is parsed once per process
#: and the key re-reads it if it changes underneath.
_CACHE: dict[tuple[str, int, int], pd.DataFrame] = {}


class DatabaseError(ValueError):
    """A manifest or a table that cannot be believed, with the file named."""


def validate_format(frame: pd.DataFrame, *, where: str = "table") -> pd.DataFrame:
    """Validate the common CSV schema; return a copy with numeric dtypes.

    Empty confidence means unknown, never certainty. ``attrs`` stays JSON
    text here; the events loader parses it into an object's attributes.
    """
    if tuple(frame.columns) != FORMAT_COLUMNS:
        raise DatabaseError(
            f"{where}: columns must be {FORMAT_COLUMNS}; got {tuple(frame.columns)}"
        )
    out = frame.copy()
    for col in ("shot", "t0_s", "t1_s", "confidence"):
        values = out[col].replace("", float("nan")) if col == "confidence" \
            else out[col]
        numeric = pd.to_numeric(values, errors="coerce")
        valid = numeric.map(math.isfinite)
        if col == "confidence":
            valid = (valid & numeric.between(0, 1)) | values.isna()
        elif col == "shot":
            valid &= numeric.map(lambda v: math.isfinite(v) and v % 1 == 0
                                 and -(2**63) <= v < 2**63)
        if not valid.all():
            raise DatabaseError(f"{where}: invalid `{col}` value")
        out[col] = numeric.astype("int64" if col == "shot" else "float64")
    if (out["t1_s"] < out["t0_s"]).any():
        raise DatabaseError(f"{where}: `t1_s` must not precede `t0_s`")
    ids = set(load_lexicon().ids)
    for col, allowed in (("phenomenon", ids), ("evidence_kind", EVIDENCE_KINDS)):
        if not out[col].isin(allowed).all():
            raise DatabaseError(f"{where}: invalid `{col}`; expected one of {allowed}")
    if not out["source"].map(lambda v: isinstance(v, str) and bool(v.strip())).all():
        raise DatabaseError(f"{where}: `source` must be a non-empty string")
    for value in out["attrs"]:
        try:
            attrs = json.loads(value)
            if not isinstance(attrs, dict):
                raise TypeError("expected a JSON object")
            json.dumps(attrs, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise DatabaseError(f"{where}: invalid `attrs`: {exc}") from exc
    return out


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write deterministic UTF-8 CSV, LF endings, twelve decimal places, empty NaN."""
    def cell(value):
        if isinstance(value, float):
            return "" if math.isnan(value) else f"{value:.12f}"
        return value

    with atomic_path(path) as tmp, tmp.open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(frame.columns)
        writer.writerows([cell(v) for v in row]
                         for row in frame.itertuples(index=False, name=None))


def write_meta(path: Path, meta: Mapping[str, Any]) -> None:
    """Write the sidecar beside either an events projection or a shot summary."""
    text = json.dumps(dict(meta), sort_keys=True, indent=2, allow_nan=False) + "\n"
    with atomic_path(path.with_suffix(".meta.json")) as tmp:
        tmp.write_text(text, encoding="utf-8")


def write_format_table(frame: pd.DataFrame, path: Path, meta: Mapping[str, Any]) -> None:
    """Validate, sort without deduplicating, and write the common CSV plus metadata."""
    frame = validate_format(frame, where=str(path))
    frame["attrs"] = frame["attrs"].map(
        lambda value: json.dumps(json.loads(value), sort_keys=True, allow_nan=False)
    )
    frame = frame.sort_values(list(FORMAT_COLUMNS), kind="stable")
    metadata = {**meta, "n_rows": len(frame), "n_shots": int(frame["shot"].nunique())}
    # Validate metadata before replacing either artifact.
    json.dumps(metadata, allow_nan=False)
    write_csv(frame, path)
    write_meta(path, metadata)


@dataclass(frozen=True)
class TableSpec:
    """One entry of `tables.yaml`: where a table is and what its columns mean."""

    stem: str
    dir: str
    phenomenon: str
    kind: str
    shot_col: str
    t_units: str
    provenance: str
    t_col: str = ""
    t0_col: str = ""
    t1_col: str = ""
    attr_cols: tuple[str, ...] = ()
    attr_types: Mapping[str, str] = field(default_factory=dict)
    raw_file: str = ""
    format_stem: str = ""
    converter: str = ""
    made_at: str = ""

    @property
    def source(self) -> str:
        """`database:<stem>`, the `source` of every row this table produces."""
        return f"{SOURCE_PREFIX}{self.stem}"

    @property
    def time_cols(self) -> tuple[str, ...]:
        return (self.t_col,) if self.kind == "point" else (self.t0_col,
                                                           self.t1_col)

    @property
    def columns(self) -> tuple[str, ...]:
        """Every column the CSV must have."""
        return (self.shot_col, *self.time_cols, *self.attr_cols)

    def path(self, root=None) -> Path:
        return _root(root) / self.dir / "format" / f"{self.format_stem}.csv"

    def raw_path(self, root=None) -> Path:
        return _root(root) / self.dir / "raw" / self.raw_file


def _root(root=None) -> Path:
    return Path(Paths.from_env().label_tables if root is None else root)


def _str(where: str, entry: Mapping[str, Any], key: str, *,
         required: bool = True) -> str:
    value = entry.get(key)
    if value is None and not required:
        return ""
    if not isinstance(value, str) or not value.strip():
        raise DatabaseError(
            f"{where}: `{key}` must be a non-empty string; got {value!r}"
        )
    return value.strip()


def _spec(where: str, entry: Any) -> TableSpec:
    if not isinstance(entry, Mapping):
        raise DatabaseError(f"{where}: each table is a mapping, not {type(entry)}")
    stem = _str(where, entry, "stem")
    where = f"{where}: table {stem}"
    kind = _str(where, entry, "kind")
    if kind not in KINDS:
        raise DatabaseError(f"{where}: `kind` must be one of {KINDS}; got {kind!r}")
    t_units = _str(where, entry, "t_units")
    if t_units not in UNITS:
        raise DatabaseError(
            f"{where}: `t_units` must be one of {tuple(UNITS)}; got {t_units!r}"
        )
    cols = entry.get("attr_cols") or []
    if not isinstance(cols, Sequence) or isinstance(cols, str) or \
            not all(isinstance(c, str) and c.strip() for c in cols):
        raise DatabaseError(f"{where}: `attr_cols` must be a list of column names")
    attr_cols = tuple(c.strip() for c in cols)
    types = entry.get("attr_types") or {}
    if not isinstance(types, Mapping):
        raise DatabaseError(f"{where}: `attr_types` must be a mapping")
    for col, kind_name in types.items():
        if col not in attr_cols:
            raise DatabaseError(
                f"{where}: `attr_types` types {col!r}, which is not in "
                f"`attr_cols` {attr_cols}"
            )
        if kind_name not in ATTR_TYPES:
            raise DatabaseError(
                f"{where}: `attr_types[{col}]` must be one of {ATTR_TYPES}; "
                f"got {kind_name!r}"
            )
    for key in ("stem", "dir", "raw_file", "format_stem", "converter"):
        value = _str(where, entry, key)
        if value in (".", "..") or "/" in value or "\\" in value:
            raise DatabaseError(f"{where}: `{key}` must be a single path component")
    made_at = _str(where, entry, "made_at")
    try:
        if datetime.fromisoformat(made_at).tzinfo is None:
            raise ValueError("timezone required")
    except ValueError as exc:
        raise DatabaseError(f"{where}: `made_at` must be an ISO timestamp") from exc
    return TableSpec(
        stem=stem,
        dir=_str(where, entry, "dir"),
        phenomenon=_str(where, entry, "phenomenon"),
        kind=kind,
        shot_col=_str(where, entry, "shot_col"),
        t_units=t_units,
        provenance=_str(where, entry, "provenance"),
        t_col=_str(where, entry, "t_col") if kind == "point" else "",
        t0_col=_str(where, entry, "t0_col") if kind == "interval" else "",
        t1_col=_str(where, entry, "t1_col") if kind == "interval" else "",
        attr_cols=attr_cols,
        attr_types={str(k): str(v) for k, v in types.items()},
        raw_file=entry["raw_file"],
        format_stem=entry["format_stem"],
        converter=entry["converter"],
        made_at=made_at,
    )


def load_manifest(root=None) -> tuple[TableSpec, ...]:
    """Every table `tables.yaml` declares; every problem names the file.

    Checked rather than trusted, for the same reason `lexicon.py` checks
    itself: this file is edited by hand, by whoever was sent the next CSV,
    and its mistakes are silent. A `phenomenon` that is not a lexicon id
    would produce event rows that join to nothing; a stem that is not
    unique would give two tables one source and make a re-run of either
    replace the other's rows.
    """
    path = _root(root) / MANIFEST
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise DatabaseError(f"label-table manifest not readable: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise DatabaseError(f"{path}: the manifest is a mapping, not {type(raw)}")
    version = raw.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise DatabaseError(f"{path}: `version` must be a positive integer")
    entries = raw.get("tables")
    if entries is None:
        entries = []
    if not isinstance(entries, Sequence) or isinstance(entries, str):
        raise DatabaseError(f"{path}: `tables` must be a list")
    specs = tuple(_spec(str(path), entry) for entry in entries)
    seen: set[str] = set()
    outputs: set[tuple[str, str]] = set()
    for spec in specs:
        if spec.stem in seen:
            raise DatabaseError(
                f"{path}: two tables share the stem {spec.stem!r}; the stem is "
                f"the source, so they would replace each other's rows"
            )
        seen.add(spec.stem)
        output = (spec.dir, spec.format_stem)
        if output in outputs:
            raise DatabaseError(f"{path}: duplicate format_stem output {output}")
        outputs.add(output)
    if specs:
        try:
            ids = set(load_lexicon().ids)
        except LexiconError as exc:                     # pragma: no cover
            raise DatabaseError(f"{path}: the lexicon is unreadable: {exc}") from exc
        for spec in specs:
            if spec.phenomenon not in ids:
                raise DatabaseError(
                    f"{path}: table {spec.stem!r} names the phenomenon "
                    f"{spec.phenomenon!r}, which is not a lexicon id "
                    f"{tuple(sorted(ids))}"
                )
    return specs


def read_table(spec: TableSpec, root=None) -> pd.DataFrame:
    """A validated common-schema frame; attrs remains JSON text until event loading."""
    return _read_cached(spec, root).copy()


def _read_cached(spec: TableSpec, root=None) -> pd.DataFrame:
    path = spec.path(root)
    try:
        stat = path.stat()
    except OSError as exc:
        raise DatabaseError(f"table {spec.stem!r} is not readable at {path}: "
                            f"{exc}") from exc
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    hit = _CACHE.get(key)
    if hit is None:
        hit = _CACHE[key] = _parse(path)
    for col, expected in (("source", spec.source), ("phenomenon", spec.phenomenon)):
        if not (hit[col] == expected).all():
            raise DatabaseError(f"{path}: `{col}` must match manifest {expected!r}")
    return hit


def _parse(path: Path) -> pd.DataFrame:
    try:
        raw = pd.read_csv(path, keep_default_na=False)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError,
            pd.errors.EmptyDataError) as exc:
        raise DatabaseError(f"{path}: unreadable as CSV: {exc}") from exc
    return validate_format(raw, where=str(path))


def shots(spec: TableSpec, root=None) -> frozenset[int]:
    """Which shots this table names. Membership only - not a coverage claim."""
    return frozenset(int(s) for s in _read_cached(spec, root)["shot"].tolist())


def known_sources(root=None) -> tuple[str, ...]:
    """`database:<stem>` for every declared table.

    Documentation, like `schema.KNOWN_SOURCES` itself, and computed rather
    than listed there because the manifest changes without a code edit.
    """
    return tuple(spec.source for spec in load_manifest(root))


def events_for_shot(
    shot: int,
    specs: Sequence[TableSpec] | None = None,
    root=None,
) -> tuple[list[Event], list[dict]]:
    """This shot's curated rows, and the record of which tables named it.

    Returns `(events, source_records)`: the events to write, and one record
    per table that CONSULTED this shot - which is to say, per table that
    lists it. A table that does not list the shot contributes neither, so
    the sources file never claims coverage nobody has.

    The records are `schema.write_sources` rows minus the `shot` the writer
    fills in: `status="ok"`, `reason=""`, `n_events`, NaN coverage, and
    `(diag, channel, pass_name) = ("", -1, "")`, which is the key a curated
    table owns since it reads no diagnostic.
    """
    shot = int(shot)
    specs = load_manifest(root) if specs is None else tuple(specs)
    events: list[Event] = []
    records: list[dict] = []
    for spec in specs:
        frame = _read_cached(spec, root)
        rows = frame[frame["shot"] == shot]
        if rows.empty:
            continue
        for row in rows.to_dict("records"):
            attrs = json.loads(row["attrs"])
            events.append(
                Event(
                    shot=shot,
                    source=row["source"],
                    phenomenon=row["phenomenon"],
                    t0_s=float(row["t0_s"]),
                    t1_s=float(row["t1_s"]),
                    confidence=float(row["confidence"]),
                    diag="",
                    channel=-1,
                    pass_name="",
                    attrs=attrs,
                    t_cov0_s=_NAN,
                    t_cov1_s=_NAN,
                    evidence_kind=row["evidence_kind"],
                )
            )
        records.append({
            "source": spec.source,
            "status": "ok",
            # Empty by contract - see COVERAGE_REASON. The NaN coverage
            # below is what says the examined interval is unknown.
            "reason": "",
            "n_events": len(rows),
            "t_cov0_s": _NAN,
            "t_cov1_s": _NAN,
            "diag": "",
            "channel": -1,
            "pass_name": "",
        })
    return events, records


__all__ = [
    "COVERAGE_REASON",
    "EVIDENCE_KIND",
    "FORMAT_COLUMNS",
    "FORMAT_SCHEMA_VERSION",
    "MANIFEST",
    "SOURCE_PREFIX",
    "DatabaseError",
    "TableSpec",
    "events_for_shot",
    "known_sources",
    "load_manifest",
    "read_table",
    "shots",
    "validate_format",
    "write_csv",
    "write_format_table",
    "write_meta",
]
