"""What one event is, and how a shot's events are stored.

One parquet file per shot, one row per event, `COLUMNS` in this order and
`DTYPES` exactly - a consumer reading a thousand shots concatenates them
without a dtype surprise. Validation lives in `Event.__post_init__` rather
than in the writer, so a detector that computes a backwards interval or a
confidence of 3.4 fails where the mistake was made, not a stage later.

Provenance is per row, not per file: `source` and `evidence_kind` say who
claims the event and what kind of claim it is, and `run_id`/`git_sha`/
`written_at` say which run wrote it. A forecast is not an observation, so it
carries a `horizon_s` and nothing else may. `evidence_kind` is not
decoration: `events/windows.py` refuses to compute a diagnostic feature
from anything but a `detector` or `heuristic` row of the family's own
source, which is what keeps an operator's logbook line and a model's
forecast out of the ELM count.

`events/<shot>_sources.parquet` is the second file, written by the same
`Paths.events` directory and read by `read_sources`: one row per
`(source, diag, channel, pass)` that RAN or was SKIPPED, with the coverage
it ran over and how many events it produced. An events file states what
was found; a detector that ran and found nothing writes nothing to it, and
without the sources file "observed silence" and "never processed" are the
same empty query.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import atomic_path, git_sha

#: What kind of claim a row is. A forecast is not an observation.
EVIDENCE_KINDS = (
    "detector", "heuristic", "forecast", "text", "human", "database", "model",
)

#: The spectrogram pass an event was found on; "" where none applies.
PASS_NAMES = ("", "wide", "zoom")

#: The producers we expect. Documentation, not a constraint: a new detector
#: may name itself anything non-empty, and this tuple is what the rest of the
#: pipeline knows how to interpret.
KNOWN_SOURCES = (
    "tokeye_track", "tokeye_transient", "ece_sawtooth", "dalpha_lh",
    "elm_clock", "actuator", "qh_proxy", "text", "model", "label_forecast",
    "database", "qmin_rule",
)

#: Column order of `events/<shot>_events.parquet`.
COLUMNS = (
    "shot", "event_id", "source", "evidence_kind", "phenomenon",
    "t0_s", "t1_s", "f0_khz", "f1_khz", "confidence", "horizon_s",
    "diag", "channel", "pass_name", "attrs", "t_cov0_s", "t_cov1_s",
    "run_id", "git_sha", "written_at",
)

#: Dtype of every column. `object` is pandas' dtype for python strings; the
#: reader casts to this map so a file written by an older pyarrow still
#: compares equal.
DTYPES = {
    "shot": "int32",
    "event_id": "object",
    "source": "object",
    "evidence_kind": "object",
    "phenomenon": "object",
    "t0_s": "float64",
    "t1_s": "float64",
    "f0_khz": "float32",
    "f1_khz": "float32",
    "confidence": "float32",
    "horizon_s": "float32",
    "diag": "object",
    "channel": "int16",
    "pass_name": "object",
    "attrs": "object",
    "t_cov0_s": "float64",
    "t_cov1_s": "float64",
    "run_id": "object",
    "git_sha": "object",
    "written_at": "object",
}

_NAN = float("nan")


def _attrs_json(attrs: Mapping[str, Any]) -> str:
    """`attrs` as the stored JSON string, or ValueError.

    `allow_nan=False` because the default writes bare `NaN`/`Infinity`, which
    is not JSON: the string round-trips through python and through nothing
    else. A non-finite attribute is a bug in the detector that computed it,
    and this is the earliest place it can be said so.
    """
    try:
        return json.dumps(dict(attrs), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"attrs must be JSON-serialisable: {exc}") from exc


@dataclass(frozen=True)
class Event:
    """One phenomenon occurrence, as claimed by one source.

    Times are seconds and half-open `[t0_s, t1_s)`; a point event (an L-H
    transition, an ELM) has `t1_s == t0_s`. `t_cov0_s`/`t_cov1_s` are the
    coverage of the diagnostic the source looked at, so "no event here" can
    be told apart from "nobody looked".
    """

    shot: int
    source: str
    phenomenon: str
    t0_s: float
    t1_s: float
    f0_khz: float = _NAN
    f1_khz: float = _NAN
    confidence: float = _NAN
    diag: str = ""
    channel: int = -1
    pass_name: str = ""
    attrs: Mapping[str, Any] = field(default_factory=dict)
    t_cov0_s: float = _NAN
    t_cov1_s: float = _NAN
    evidence_kind: str = "detector"
    horizon_s: float = _NAN

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("source must not be empty")
        if not self.phenomenon:
            raise ValueError("phenomenon must not be empty")
        if not (math.isfinite(self.t0_s) and math.isfinite(self.t1_s)):
            raise ValueError(
                f"t0_s and t1_s must be finite; got {self.t0_s}, {self.t1_s}"
            )
        if self.t1_s < self.t0_s:
            raise ValueError(
                f"t1_s must not precede t0_s; got {self.t0_s}, {self.t1_s}"
            )
        # An event that ends after its own coverage is a claim about a
        # stretch nobody measured. On the three pilot shots 9, 11 and 15
        # track rows did exactly that, by up to 2.052 ms, because a
        # stitched transform's support runs past the record it was
        # computed from. That is edge padding, not plasma, and the fix is
        # `coverage.clip_to_coverage` at the point the row is built -
        # which is why this is a refusal here rather than a repair in the
        # writer: a row nobody clipped must fail where it was made.
        if (
            math.isfinite(self.t1_s)
            and math.isfinite(self.t_cov1_s)
            and self.t1_s > self.t_cov1_s
        ):
            raise ValueError(
                f"t1_s must not exceed t_cov1_s; got {self.t1_s} against "
                f"{self.t_cov1_s} (clip with events.coverage.clip_to_coverage)"
            )
        if self.evidence_kind not in EVIDENCE_KINDS:
            raise ValueError(
                f"evidence_kind {self.evidence_kind!r} not in {EVIDENCE_KINDS}"
            )
        if self.pass_name not in PASS_NAMES:
            raise ValueError(f"pass_name {self.pass_name!r} not in {PASS_NAMES}")
        if math.isfinite(self.confidence) and not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must lie in [0, 1]; got {self.confidence}"
            )
        if (
            math.isfinite(self.f0_khz)
            and math.isfinite(self.f1_khz)
            and self.f1_khz < self.f0_khz
        ):
            raise ValueError(
                f"f1_khz must not precede f0_khz; got {self.f0_khz}, {self.f1_khz}"
            )
        forecast = self.evidence_kind == "forecast"
        if math.isfinite(self.horizon_s) != forecast:
            raise ValueError(
                "horizon_s is required for a forecast and meaningless otherwise; "
                f"got evidence_kind={self.evidence_kind!r}, "
                f"horizon_s={self.horizon_s}"
            )
        _attrs_json(self.attrs)


def _empty_frame() -> pd.DataFrame:
    """No events, right columns, right dtypes."""
    return pd.DataFrame({name: pd.Series(dtype=DTYPES[name]) for name in COLUMNS})


def _rows(shot: int, events: Sequence[Event], *, run_id: str) -> pd.DataFrame:
    """The new rows, `event_id` numbered per source in `t0_s` order."""
    now = datetime.now(UTC).isoformat(timespec="seconds")
    sha = git_sha()
    ordered = sorted(events, key=lambda e: e.t0_s)     # stable: ties keep order
    counters: dict[str, int] = {}
    rows = []
    for e in ordered:
        n = counters.get(e.source, 0)
        counters[e.source] = n + 1
        rows.append(
            {
                "shot": int(shot),
                "event_id": f"{int(shot)}-{e.source}-{n:05d}",
                "source": e.source,
                "evidence_kind": e.evidence_kind,
                "phenomenon": e.phenomenon,
                "t0_s": float(e.t0_s),
                "t1_s": float(e.t1_s),
                "f0_khz": float(e.f0_khz),
                "f1_khz": float(e.f1_khz),
                "confidence": float(e.confidence),
                "horizon_s": float(e.horizon_s),
                "diag": e.diag,
                "channel": int(e.channel),
                "pass_name": e.pass_name,
                "attrs": _attrs_json(e.attrs),
                "t_cov0_s": float(e.t_cov0_s),
                "t_cov1_s": float(e.t_cov1_s),
                "run_id": run_id,
                "git_sha": sha,
                "written_at": now,
            }
        )
    if not rows:
        return _empty_frame()
    return pd.DataFrame(rows, columns=list(COLUMNS)).astype(DTYPES)


def write_events(
    path,
    shot: int,
    events: Sequence[Event],
    *,
    run_id: str,
    merge: bool = True,
    sources: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Write one shot's events atomically; return the file's new contents.

    A write owns the sources it names - those of `events`, plus any listed in
    `sources` - and replaces every row of theirs. With `merge`, rows from any
    other source are carried forward untouched, so the sawtooth heuristic and
    the TokEye tracker can write the same file in either order. `sources` is
    how a source is cleared: `events=[]`, `sources=["tokeye_track"]` says the
    tracker ran and found nothing, which is not the same as never running.
    """
    path = Path(path)
    events = list(events)
    wrong = sorted({int(e.shot) for e in events if int(e.shot) != int(shot)})
    if wrong:
        raise ValueError(f"events for shot {wrong} in a write of shot {shot}")
    owned = {e.source for e in events}
    if sources is not None:
        owned |= set(sources)
    new = _rows(shot, events, run_id=run_id)
    parts = [new]
    if merge and path.exists():
        old = read_events(path)
        parts.insert(0, old[~old["source"].isin(owned)])
    kept = [p for p in parts if not p.empty]
    out = pd.concat(kept, ignore_index=True) if kept else _empty_frame()
    out = (
        out.sort_values(["source", "t0_s", "event_id"], kind="stable")
        .reset_index(drop=True)
        .astype(DTYPES)
    )
    with atomic_path(path) as tmp:
        out.to_parquet(tmp, index=False)
    return out


def read_events(path, *, source: str | None = None,
                phenomenon: str | None = None) -> pd.DataFrame:
    """A shot's events, optionally one source or one phenomenon.

    A missing file reads as an empty frame with the right columns: `events/`
    is absent for every shot the mask job has not reached yet, and a caller
    joining labels to events must not have to care.
    """
    path = Path(path)
    if not path.exists():
        out = _empty_frame()
    else:
        out = pd.read_parquet(path)[list(COLUMNS)].astype(DTYPES)
    if source is not None:
        out = out[out["source"] == source]
    if phenomenon is not None:
        out = out[out["phenomenon"] == phenomenon]
    return out.reset_index(drop=True)


# ------------------------------------------------- the per-source record

#: What a source's run came to. `ok` is "it ran to completion", however
#: many events that turned out to be - zero included.
SOURCE_STATUSES = ("ok", "skipped", "error")

#: Column order of `events/<shot>_sources.parquet`. CONTRACT: shot_design's
#: consumer is built against this list, so a column may be appended and
#: none may be renamed, reordered or dropped.
SOURCE_COLUMNS = (
    "shot", "source", "status", "reason", "t_cov0_s", "t_cov1_s",
    "n_events", "diag", "channel", "pass_name", "run_id", "git_sha",
    "written_at", "intervals", "min_gap_s",
)

#: Dtype of every column of the sources file.
SOURCE_DTYPES = {
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
    "intervals": "object",
    "min_gap_s": "float64",
}

#: What makes two rows the same record. A source is not enough: the
#: tracker runs once per `(diag, channel, pass)` block and each of those
#: has its own coverage and its own answer.
SOURCE_KEY = ("source", "diag", "channel", "pass_name")


def _empty_sources() -> pd.DataFrame:
    return pd.DataFrame(
        {name: pd.Series(dtype=SOURCE_DTYPES[name]) for name in SOURCE_COLUMNS}
    )


def _source_row(shot: int, record: Mapping[str, Any], *, run_id: str,
                sha: str, now: str) -> dict:
    """One validated row, provenance filled in."""
    status = str(record.get("status", "ok"))
    if status not in SOURCE_STATUSES:
        raise ValueError(f"status {status!r} not in {SOURCE_STATUSES}")
    source = str(record.get("source", ""))
    if not source:
        raise ValueError("source must not be empty")
    reason = str(record.get("reason", "") or "")
    if status == "ok" and reason:
        raise ValueError(
            f"an ok source carries no reason; got {reason!r} for {source!r}"
        )
    if status != "ok" and not reason:
        raise ValueError(f"a {status} source must say why; {source!r} does not")
    pass_name = str(record.get("pass_name", "") or "")
    if pass_name not in PASS_NAMES:
        raise ValueError(f"pass_name {pass_name!r} not in {PASS_NAMES}")
    n_events = int(record.get("n_events", 0))
    if n_events < 0:
        raise ValueError(f"n_events must not be negative; got {n_events}")
    t0 = float(record.get("t_cov0_s", _NAN))
    t1 = float(record.get("t_cov1_s", _NAN))
    if math.isfinite(t0) and math.isfinite(t1) and t1 < t0:
        raise ValueError(f"t_cov1_s must not precede t_cov0_s; got {t0}, {t1}")
    intervals = record.get("intervals")
    min_gap_s = float(record.get("min_gap_s", _NAN))
    if intervals is not None:
        from .coverage import Coverage

        cov = Coverage(json.loads(intervals), min_gap_s)
        t0, t1 = cov.hull
    return {
        "shot": int(shot),
        "source": source,
        "status": status,
        "reason": reason,
        "t_cov0_s": t0,
        "t_cov1_s": t1,
        "n_events": n_events,
        "diag": str(record.get("diag", "") or ""),
        "channel": int(record.get("channel", -1)),
        "pass_name": pass_name,
        "run_id": run_id,
        "git_sha": sha,
        "written_at": now,
        "intervals": intervals,
        "min_gap_s": min_gap_s,
    }


def write_sources(
    path,
    shot: int,
    records: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    merge: bool = True,
) -> pd.DataFrame:
    """Write one shot's per-source completion record; return the new file.

    One row per `(source, diag, channel, pass_name)` that RAN or was
    SKIPPED. This is the file that answers "did anybody look", which an
    events file cannot: a detector that ran and found nothing writes no
    event row, and is indistinguishable in `events/<shot>_events.parquet`
    from a detector that never ran. `status == "ok"` with `n_events == 0`
    is the observed silence; `status == "skipped"` with a `reason` is the
    absence of an observation; no row at all is "not processed".

    An `ok` row carries `reason=""` and a `skipped` or `error` row must
    carry one, and `_source_row` REFUSES the other two combinations: a
    reason on an `ok` row would be a caveat hidden in a field consumers
    read as provenance, and a curated table's "a listing is not a coverage
    claim" belongs in its NaN `t_cov` pair and the docs, not there.

    Merged and atomic like `write_events`, on the same reasoning and with
    the same key: a re-run of one channel replaces that channel's rows and
    leaves every other source's alone.
    """
    path = Path(path)
    now = datetime.now(UTC).isoformat(timespec="seconds")
    sha = git_sha()
    rows = [
        _source_row(shot, r, run_id=run_id, sha=sha, now=now) for r in records
    ]
    keys = {tuple(r[k] for k in SOURCE_KEY) for r in rows}
    if len(keys) != len(rows):
        raise ValueError("two records for one (source, diag, channel, pass)")
    new = (
        pd.DataFrame(rows, columns=list(SOURCE_COLUMNS)).astype(SOURCE_DTYPES)
        if rows else _empty_sources()
    )
    parts = [new]
    if merge and path.exists():
        old = read_sources(path)
        if not old.empty:
            keep = ~pd.MultiIndex.from_frame(old[list(SOURCE_KEY)]).isin(keys)
            parts.insert(0, old[keep])
    kept = [p for p in parts if not p.empty]
    out = pd.concat(kept, ignore_index=True) if kept else _empty_sources()
    out = (
        out.sort_values(list(SOURCE_KEY), kind="stable")
        .reset_index(drop=True)
        .astype(SOURCE_DTYPES)
    )
    with atomic_path(path) as tmp:
        out.to_parquet(tmp, index=False)
    return out


def read_sources(path, *, source: str | None = None) -> pd.DataFrame:
    """A shot's per-source record, or an empty typed frame where there is none.

    Missing reads as empty for the same reason `read_events` does: most of
    a campaign has not been processed yet, and a caller joining coverage to
    events must not have to care. An empty frame here means "no source has
    been recorded for this shot", which is exactly the "not processed"
    state the file exists to distinguish.
    """
    path = Path(path)
    if not path.exists():
        out = _empty_sources()
    else:
        out = pd.read_parquet(path)
        # Null intervals preserve the fact that an older writer recorded
        # only a hull; consumers must disclose its unknown interior gaps.
        if "intervals" not in out:
            out["intervals"] = None
        if "min_gap_s" not in out:
            out["min_gap_s"] = _NAN
        out = out[list(SOURCE_COLUMNS)].astype(SOURCE_DTYPES)
    if source is not None:
        out = out[out["source"] == source]
    return out.reset_index(drop=True)


def index_rows(path) -> list[dict]:
    """One summary row per (shot, source, phenomenon), for `events_index`."""
    df = read_events(path)
    rows: list[dict] = []
    for (shot, source, phenomenon), g in df.groupby(
        ["shot", "source", "phenomenon"], sort=True
    ):
        latest = g.sort_values("written_at", kind="stable").iloc[-1]
        rows.append(
            {
                "shot": int(shot),
                "source": str(source),
                "phenomenon": str(phenomenon),
                "n_events": len(g),
                "t0_min_s": float(g["t0_s"].min()),
                "t1_max_s": float(g["t1_s"].max()),
                "confidence_max": float(g["confidence"].max()),
                "evidence_kind": str(latest["evidence_kind"]),
                "run_id": str(latest["run_id"]),
                "written_at": str(latest["written_at"]),
            }
        )
    return rows


def intervals(events, phenomenon: str) -> np.ndarray:
    """`(n, 2)` float64 `[t0_s, t1_s]` for one phenomenon, sorted by start.

    Takes either the `Event`s a detector just produced or a frame read back
    from disk, because both callers exist and neither should have to convert.
    """
    if isinstance(events, pd.DataFrame):
        sel = events.loc[events["phenomenon"] == phenomenon, ["t0_s", "t1_s"]]
        pairs = sel.to_numpy(dtype=np.float64).reshape(-1, 2)
    else:
        pairs = np.array(
            [[e.t0_s, e.t1_s] for e in events if e.phenomenon == phenomenon],
            dtype=np.float64,
        ).reshape(-1, 2)
    return pairs[np.argsort(pairs[:, 0], kind="stable")]
