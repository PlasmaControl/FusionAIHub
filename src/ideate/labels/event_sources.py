"""`events/<shot>_sources.parquet`: which detector looked at what, and over which span.

WHY THE TABLE EXISTS. `events.parquet` records what somebody FOUND. Nothing in it records what
somebody LOOKED AT and found nothing, and the two are not the same claim: a shot with no tearing
rows is a shot where either no tearing mode occurred or no tearing detector ever ran, and today
the database cannot tell those apart -- `get_events(198658)` returned `n: 0` with an empty
`caveats` list for a shot that is not in the 500-shot database at all. Completion and coverage
have to be persisted even when a detector emits zero events, and this is the table that does it.

THE CONTRACT. labelmaker WRITES it, ideate READS it. One row per `(source, diag, channel, pass)`
that ran or was deliberately skipped, with `status` in {ok, skipped, error} and `reason` saying
why for the last two. `t_cov0_s`/`t_cov1_s` are that source's OWN coverage -- not a span borrowed
from a sibling input, which is the second half of the same defect: actuator events all received
the union of the gas, NBI and RMP time axes, so an NBI event carried coverage from -10 to 94.9 s
because the gas recorder happened to run that long.

`ideate labels join` ingests every file that exists into `db/event_sources.parquet` with the same
columns. A shot whose file is absent contributes NO ROWS, and that absence is what `get_events`
reports as `unprocessed`: nobody has run a detector over this shot, so its empty event list is
not evidence of a quiet shot.

TWO ROWS THAT ARE NOT COVERAGE. An `ok` row whose `t_cov` pair is NaN ran and recorded no span
(`unknown_coverage_rows`), and an `ok` row from a non-diagnostic source -- `text`, `database`,
`database:<stem>` -- is a logbook hit or a curated listing rather than a measurement
(`is_non_diagnostic`). Neither can make a window observed, and a reply has to say which of the
two it is holding instead of counting either as "a source that looked".

`write_sources` is the fixture writer for the contract: labelmaker's tests and ideate's build the
same table through it, so the two sides cannot drift into two shapes of the same file.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

#: Column order and dtype of `events/<shot>_sources.parquet` and of `db/event_sources.parquet`.
#: `object` is pandas' dtype for python strings, matching `labelmaker.events.schema.DTYPES`.
SOURCES_DTYPES: dict[str, str] = {
    "shot": "int32",
    "source": "object",
    "status": "object",
    "reason": "object",
    "t_cov0_s": "float64",
    "t_cov1_s": "float64",
    "n_events": "int32",
    "diag": "object",
    "channel": "int16",
    "pass_name": "object",
    "run_id": "object",
    "git_sha": "object",
    "written_at": "object",
}
SOURCES_COLUMNS: tuple[str, ...] = tuple(SOURCES_DTYPES)

#: `ok` -- the source ran to completion over `[t_cov0_s, t_cov1_s]`, emitting `n_events` (which
#: may be 0, and that is the point of the table). `skipped` -- it was not run, `reason` says why
#: (no such diagnostic on this shot, out of scope for the pass). `error` -- it was run and
#: failed, `reason` carries the failure. Only `ok` rows contribute coverage.
STATUSES: tuple[str, ...] = ("ok", "skipped", "error")

#: Sources whose `ok` row is NOT an observation of the plasma. `text` runs the lexicon over the
#: shot's own logbook entries and labelmaker records that it ran, over the shot's span -- but
#: "the word was looked for" says nothing about what any diagnostic showed, which is the same
#: policy `labelmaker.events.windows.DIAGNOSTIC_EVIDENCE` states for rows, applied here to the
#: source that writes them. `database` is the same kind of claim from the other direction: a
#: curated table is a list somebody published of shots that had a thing, and a listing is not a
#: measurement of this shot -- its rows carry NaN coverage for exactly that reason. Excluded from
#: `has_observed_products` and from coverage: a shot with a logbook, a curated entry and no
#: detector run is `unprocessed`, not "0 detections inside coverage".
NON_DIAGNOSTIC_SOURCES: tuple[str, ...] = ("text", "database")

#: The same rule for the per-table sources labelmaker writes as `database:<stem>` (one row per
#: curated file). A PREFIX, not a substring: `databases_of_rwm` would be somebody's detector.
NON_DIAGNOSTIC_SOURCE_PREFIXES: tuple[str, ...] = ("database:",)


def is_non_diagnostic(source: str) -> bool:
    """Is this source's `ok` row something other than a diagnostic having looked at the plasma?"""
    name = str(source)
    return name in NON_DIAGNOSTIC_SOURCES or name.startswith(NON_DIAGNOSTIC_SOURCE_PREFIXES)

SUFFIX = "_sources.parquet"


def sources_file(events_dir: Path | str, shot: int) -> Path:
    """`<events_dir>/<shot>_sources.parquet`. `events_dir` is `labelmaker.Paths.events`."""
    return Path(events_dir) / f"{int(shot)}{SUFFIX}"


def empty_sources() -> pd.DataFrame:
    """No rows, the contract's columns, the contract's dtypes."""
    return pd.DataFrame({name: pd.Series(dtype=dt) for name, dt in SOURCES_DTYPES.items()})


def _frame(rows: Sequence[Mapping]) -> pd.DataFrame:
    if not rows:
        return empty_sources()
    out = pd.DataFrame(list(rows))
    missing = [c for c in SOURCES_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(
            f"{SUFFIX} rows are missing {', '.join(missing)}; the columns are "
            f"{', '.join(SOURCES_COLUMNS)}"
        )
    extra = [c for c in out.columns if c not in SOURCES_DTYPES]
    if extra:
        raise ValueError(f"{SUFFIX} rows carry unknown columns {', '.join(extra)}")
    bad = sorted(set(out["status"]) - set(STATUSES))
    if bad:
        raise ValueError(f"status must be one of {STATUSES}, got {', '.join(bad)}")
    return out[list(SOURCES_COLUMNS)].astype(SOURCES_DTYPES)


def write_sources(path: Path | str, rows: Sequence[Mapping]) -> Path:
    """Write one shot's source table. The fixture writer for the contract; see the docstring."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _frame(rows).to_parquet(path, index=False)
    return path


def source_row(
    shot: int,
    source: str,
    *,
    status: str = "ok",
    reason: str = "",
    t_cov0_s: float = float("nan"),
    t_cov1_s: float = float("nan"),
    n_events: int = 0,
    diag: str = "",
    channel: int = -1,
    pass_name: str = "",
    run_id: str = "",
    git_sha: str = "",
    written_at: str = "",
) -> dict:
    """One contract row with every column present. Keyword-only past `source` on purpose: a
    positional coverage pair is exactly how a start and an end get swapped."""
    if status not in STATUSES:
        raise ValueError(f"status {status!r} is not one of {STATUSES}")
    return {
        "shot": int(shot), "source": str(source), "status": str(status), "reason": str(reason),
        "t_cov0_s": float(t_cov0_s), "t_cov1_s": float(t_cov1_s), "n_events": int(n_events),
        "diag": str(diag), "channel": int(channel), "pass_name": str(pass_name),
        "run_id": str(run_id), "git_sha": str(git_sha), "written_at": str(written_at),
    }


def read_sources(path: Path | str) -> pd.DataFrame:
    """One shot's source table, or an empty typed frame.

    A missing file is the normal case and means "nobody has processed this shot", which the
    caller reports as `unprocessed` -- so it must not be an error here, exactly as
    `labelmaker.events.schema.read_events` treats a missing events file.
    """
    path = Path(path)
    if not path.exists():
        return empty_sources()
    return pd.read_parquet(path)[list(SOURCES_COLUMNS)].astype(SOURCES_DTYPES)


def sources_union(shots: Iterable[int], *, events_dir: Path | str) -> pd.DataFrame:
    """Every existing `<shot>_sources.parquet` in one typed frame, sorted by (shot, source, diag).

    A shot with no file contributes no rows. That is the whole signal: `db/event_sources.parquet`
    holding nothing for a shot is how the database says nobody looked, and a row of zeros would
    destroy the difference between "looked, saw nothing" and "never looked".
    """
    parts = []
    for shot in sorted({int(s) for s in shots}):
        df = read_sources(sources_file(events_dir, shot))
        if not df.empty:
            parts.append(df)
    if not parts:
        return empty_sources()
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["shot", "source", "diag", "channel", "pass_name"], kind="stable")
        .reset_index(drop=True)
        .astype(SOURCES_DTYPES)
    )


def shot_summary(sources: pd.DataFrame, shot: int) -> dict:
    """`{n_sources, n_sources_ok, n_sources_skipped, n_sources_error, has_observed_products}`.

    `has_observed_products` is "somebody ran a detector over this shot and it completed" -- at
    least one `ok` row from a source `is_non_diagnostic` rejects. A shot whose every source is
    `skipped` has been considered and not examined, which is nearer to unprocessed than to
    observed and is counted as such; so is a shot on which only the `text` lexicon ran or only a
    curated table lists it. `n_sources_ok` still counts every `ok` row, `text` included: it is a
    count of what ran.

    `n_sources_unknown_coverage` counts the diagnostic `ok` rows whose `t_cov` pair is NaN: they
    RAN and nothing records over what span, so they can neither cover nor un-cover any window
    (`covers`). They are not a gap in the coverage, they are the absence of a coverage claim, and
    the number is here so a reply can say so instead of counting them as observation.
    """
    rows = sources[sources["shot"] == int(shot)] if len(sources) else sources
    counts = {s: int((rows["status"] == s).sum()) for s in STATUSES}
    return {
        "n_sources": len(rows),
        "n_sources_ok": counts["ok"],
        "n_sources_skipped": counts["skipped"],
        "n_sources_error": counts["error"],
        "n_sources_unknown_coverage": len(unknown_coverage_rows(rows)),
        "has_observed_products": bool(len(_observing(rows))),
    }


def _observing(sources: pd.DataFrame) -> pd.DataFrame:
    """The rows that completed AND are a diagnostic's: `ok`, and not `is_non_diagnostic`."""
    if not len(sources):
        return sources
    keep = (sources["status"] == "ok") & ~sources["source"].map(is_non_diagnostic)
    return sources[keep]


def _finite(sources: pd.DataFrame) -> np.ndarray:
    lo = sources["t_cov0_s"].to_numpy(dtype=float)
    hi = sources["t_cov1_s"].to_numpy(dtype=float)
    return np.isfinite(lo) & np.isfinite(hi)


def unknown_coverage_rows(sources: pd.DataFrame) -> pd.DataFrame:
    """The diagnostic `ok` rows whose coverage is NaN: they ran, over nothing anybody recorded.

    This is not an empty span and not a skipped source. Real shot 198658 wrote one --
    `actuator / ech_power_total / ok / NaN / NaN / 0 events`, the actuator's trace being all-NaN
    -- and a reply that counted it as coverage would assert an observation over a window nothing
    is recorded as having looked at. A caller reports these rows by name.
    """
    ok = _observing(sources)
    if not len(ok):
        return ok
    return ok[~_finite(ok)]


def observing_rows(
    sources: pd.DataFrame, t0_s: float | None = None, t1_s: float | None = None
) -> pd.DataFrame:
    """The diagnostic `ok` rows with FINITE coverage overlapping `[t0_s, t1_s]`. Either bound may
    be None, meaning unbounded; both None is "every source that recorded a span".

    Per source, never the hull of all of them: two passes over [1,2] and [5,6] have not looked at
    3-4 s, and answering that from their outer span is the borrowed-coverage defect this table
    exists to fix. `retrieval.phenomena` clips per source for the same reason.
    """
    ok = _observing(sources)
    if not len(ok):
        return ok
    keep = _finite(ok)
    if t0_s is not None:
        keep &= ok["t_cov1_s"].to_numpy(dtype=float) >= float(t0_s)
    if t1_s is not None:
        keep &= ok["t_cov0_s"].to_numpy(dtype=float) <= float(t1_s)
    return ok[keep]


def coverage_span(sources: pd.DataFrame) -> tuple[float, float] | None:
    """The HULL `(min t_cov0, max t_cov1)` of the `ok` DIAGNOSTIC rows, or None when no such row
    recorded a span.

    The outer span of what ran, for reporting ("the covered stretch runs 0 to 6 s"). It is
    deliberately not offered per event, and it is not what `covers` asks: handing one source's
    span to another source's rows is the defect this module exists to fix, and the hull would
    hand every source the union of all of them. A `text` or `database:` row's span is not a
    diagnostic having looked, so it does not count (`is_non_diagnostic`).
    """
    ok = observing_rows(sources)
    if ok.empty:
        return None
    return float(ok["t_cov0_s"].min()), float(ok["t_cov1_s"].max())


def covers(sources: pd.DataFrame, t0_s: float | None, t1_s: float | None) -> bool | None:
    """Does SOME ONE `ok` diagnostic source's own coverage overlap `[t0_s, t1_s]`?

    None when no such source recorded a span at all -- which is not False: a source that ran and
    recorded no coverage says nothing either way, and the caller has to report that rather than
    turn it into an observation (see `unknown_coverage_rows`).

    Overlap, not containment -- the same rule `get_events` filters rows by, so a window the
    filter would return an event from is never reported as uncovered.
    """
    if observing_rows(sources).empty:
        return None
    return not observing_rows(sources, t0_s, t1_s).empty


__all__ = [
    "NON_DIAGNOSTIC_SOURCES",
    "NON_DIAGNOSTIC_SOURCE_PREFIXES",
    "SOURCES_COLUMNS",
    "SOURCES_DTYPES",
    "STATUSES",
    "SUFFIX",
    "coverage_span",
    "covers",
    "empty_sources",
    "is_non_diagnostic",
    "observing_rows",
    "read_sources",
    "shot_summary",
    "source_row",
    "sources_file",
    "sources_union",
    "unknown_coverage_rows",
    "write_sources",
]
