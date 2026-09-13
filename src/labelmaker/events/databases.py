"""Curated label tables: somebody's list, read as events.

A table is knowledge we did not compute and cannot re-derive - Jeremy
Hansen's RWM onsets, Jalal's WPQH windows, David Smith's manual ELMs. It
arrives as a CSV, it stays exactly as its author sent it (`data/labels/`,
never `src/`), and everything that reconciles it with labelmaker's schema
is in `data/labels/tables.yaml` and in this module. Adding a table is a
YAML entry; if it needs a code edit, the manifest is missing a field and
that is the thing to add.

**A listing is not a coverage claim.** Every event here carries
`t_cov0_s = t_cov1_s = NaN`, because a database that names a shot says
nothing about which interval of that shot anybody examined. So a shot
ABSENT from a table is not a negative, and `events_for_shot` writes no
event row AND no source record for it: "ran and found nothing" is a claim,
and nobody made it. Writing `status="ok", n_events=0` for every shot in
the corpus against every table would be a false coverage claim over 16,909
shots, which is exactly the mistake the sources file exists to prevent.

**A curated list has no calibrated probability**, so `confidence` is NaN
too. A 1.0 would let a ranker treat a human list as a perfectly confident
detector.

The other conversions are as boring as they should be: times become
seconds on load and never on disk (`t_units` is what says which unit the
file is in, so a table already in seconds needs no code change), the extra
columns are carried into `attrs` verbatim beside the table's stem, and the
phenomenon comes from the manifest rather than from the file name - the
directory `resistive_wall_mode` is prose and `rwm` is the join key.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..config import Paths
from .lexicon import LexiconError, load_lexicon
from .schema import Event

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

#: Why the source record's coverage is NaN. It is a `reason` on an `ok`
#: row, which is unusual and is the point: the table ran, it found what it
#: found, and the interval it examined is unknown rather than empty.
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
        return _root(root) / self.dir / f"{self.stem}.csv"


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
    for spec in specs:
        if spec.stem in seen:
            raise DatabaseError(
                f"{path}: two tables share the stem {spec.stem!r}; the stem is "
                f"the source, so they would replace each other's rows"
            )
        seen.add(spec.stem)
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


def _typed(spec: TableSpec, col: str, value: Any) -> Any:
    """One `attrs` value, as the manifest says to read it."""
    want = spec.attr_types.get(col, "str")
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if want == "int":
        return int(value)
    if want == "float":
        return float(value)
    return str(value)


def read_table(spec: TableSpec, root=None) -> pd.DataFrame:
    """The table as `shot`, `t0_s`, `t1_s` and its own extra columns.

    Seconds, integers and finite times are established HERE, so an error
    names the CSV and the column. Left alone: the row ORDER and the
    duplicates - two onsets at the same time is a claim the table makes.
    """
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
        hit = _CACHE[key] = _parse(spec, path)
    return hit


def _parse(spec: TableSpec, path: Path) -> pd.DataFrame:
    try:
        raw = pd.read_csv(path)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise DatabaseError(f"{path}: unreadable as CSV: {exc}") from exc
    missing = [c for c in spec.columns if c not in raw.columns]
    if missing:
        raise DatabaseError(
            f"{path}: the manifest names {missing}, which the file does not "
            f"have; it has {list(raw.columns)}"
        )
    shot = pd.to_numeric(raw[spec.shot_col], errors="coerce")
    if not shot.notna().all() or not (shot % 1 == 0).all():
        raise DatabaseError(
            f"{path}: every `{spec.shot_col}` must be an integer shot number"
        )
    scale = UNITS[spec.t_units]
    times = []
    for col in spec.time_cols:
        t = pd.to_numeric(raw[col], errors="coerce")
        if not t.notna().all() or not t.map(math.isfinite).all():
            raise DatabaseError(
                f"{path}: every `{col}` must be a finite number of "
                f"{spec.t_units}"
            )
        times.append(t.astype("float64") * scale)
    t0 = times[0]
    t1 = times[-1]
    if (t1 < t0).any():
        raise DatabaseError(
            f"{path}: `{spec.time_cols[-1]}` precedes `{spec.time_cols[0]}` "
            f"on {int((t1 < t0).sum())} row(s)"
        )
    out = pd.DataFrame({
        "shot": shot.astype("int64"),
        "t0_s": t0,
        "t1_s": t1,
    })
    for col in spec.attr_cols:
        out[col] = [_typed(spec, col, v) for v in raw[col]]
    return out


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
            attrs = {col: row[col] for col in spec.attr_cols}
            attrs["table"] = spec.stem
            events.append(
                Event(
                    shot=shot,
                    source=spec.source,
                    phenomenon=spec.phenomenon,
                    t0_s=float(row["t0_s"]),
                    t1_s=float(row["t1_s"]),
                    confidence=_NAN,
                    diag="",
                    channel=-1,
                    pass_name="",
                    attrs=attrs,
                    t_cov0_s=_NAN,
                    t_cov1_s=_NAN,
                    evidence_kind=EVIDENCE_KIND,
                )
            )
        records.append({
            "source": spec.source,
            "status": "ok",
            "reason": COVERAGE_REASON,
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
    "MANIFEST",
    "SOURCE_PREFIX",
    "DatabaseError",
    "TableSpec",
    "events_for_shot",
    "known_sources",
    "load_manifest",
    "read_table",
    "shots",
]
