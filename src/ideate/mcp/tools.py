"""The tools themselves: plain functions in, JSON-serialisable dicts out.

Three rules hold for every function here, and they are the reason this module exists rather than
the CLI being wrapped:

* **the signature and the docstring are the schema.** An assistant sees the argument names, their
  types, their defaults and the docstring, and nothing else. So the annotations are real (a
  missing one becomes a slot the model has to guess at, and `test_mcp` asserts against that) and
  the docstrings are written for the caller, not for the maintainer.
* **`caveats` is always present.** A retrieval result with no caveats and a retrieval result
  whose caveats were dropped look identical, and the second is how "no shot matched" becomes
  "there are no such shots". Every return carries the key, possibly empty.
* **nothing raises to the transport.** An exception on an MCP call reaches the model as a
  protocol error -- in practice `is_error=True` and the string "Error executing tool <name>",
  with no `caveats` key and nothing to act on. Every failure comes back as
  `{"error": <a sentence the caller can act on>, "caveats": [...]}` -- the same sentences the CLI
  prints, because they are the ones that say what to run next. Each tool turns the failures it
  ANTICIPATES into their own sentence; `never_raises`, applied where the tools are registered,
  is what makes the promise hold for the ones nobody anticipated (a database caught mid-publish,
  an `events.parquet` written to some other schema).

The database is loaded once per directory and cached (`_load_db`), because `ShotDB.load` reads
every table and embedding matrix and a server answers many calls. A rebuild under a running
server is therefore not seen until `reset_cache()`; that is the trade, and it is the right one
for a process whose whole job is answering questions about a database that changes daily at most.
"""

from __future__ import annotations

import functools
import json
import math
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeVar

from .. import config

#: The segment names `schema.SegName` allows. Named here so an error can list them.
SEGMENTS = ("full", "ramp_up", "flat_top", "ramp_down")

#: What a model is likely to type, and what it means. `flattop` is the common one -- the CLI
#: refuses it with an argparse error, which is right for a person at a terminal and useless to an
#: assistant that cannot see the error before it has already spent the call.
SEGMENT_ALIASES = {"flattop": "flat_top", "flat-top": "flat_top", "rampup": "ramp_up",
                   "ramp-up": "ramp_up", "rampdown": "ramp_down", "ramp-down": "ramp_down"}

#: Said when there is no `events.parquet`. Exact, because a caller keys on it.
NO_EVENTS = "no events table yet (labelmaker events not joined)"

_FORECAST_CAVEAT = (
    "{n} forecast row(s) are in `forecasts`, not in `events`: a forecast is a model's claim "
    "about what was about to happen, not an observation of what did"
)

_DATABASE_CAVEAT = (
    "{n} row(s) are in `database_intervals`, not in `events`: a curated list names a shot and a "
    "time, not a measurement. Its coverage is null because nobody recorded which interval of the "
    "shot was examined, so a shot's ABSENCE from such a list is not a negative"
)

_TIMELESS_CAVEAT = (
    "{n} row(s) have no recorded time and were not considered for the window: unknown when, "
    "which is not the same as outside it"
)

#: What `ShotDB.load` opens once `_db()` has seen the manifest. A build publishes per file, so a
#: manifest sitting next to a missing table is a database mid-rebuild, not a broken one -- and
#: that is the difference between "wait and retry" and "something is wrong".
DB_FILES = ("shots.parquet", "segments.parquet", "pca.json")

F = TypeVar("F", bound=Callable[..., dict])


def _incomplete_db() -> list[str]:
    """A caveat naming the likely cause when the database is half-published, else nothing."""
    try:
        db_dir = config.load_paths().db_dir
        if not (db_dir / "manifest.json").exists():
            return []
        missing = [name for name in DB_FILES if not (db_dir / name).exists()]
    except Exception:  # noqa: BLE001 - a guard may not raise on its way to reporting a failure
        return []
    if not missing:
        return []
    return [
        (
            f"the database at {db_dir} has a manifest but no {', '.join(missing)}: it is being "
            f"rebuilt or is incomplete -- retry, or run `ideate build`"
        )
    ]


def never_raises(fn: F) -> F:
    """`fn` with every escaping exception turned into this module's error dict.

    Applied where the tools are REGISTERED (`server.TOOLS`), not at definition, for two reasons:
    the functions stay plain functions that raise for their own tests, and there is one place
    that says the promise holds rather than one `except` per tool that a new tool can forget.

    `functools.wraps` is load-bearing, not tidiness: the signature and the docstring ARE the tool
    schema, and a bare `*args` wrapper would register every tool with no arguments and no
    description. The message names the exception TYPE rather than guessing at a cause; when the
    state is one we can recognise -- a database caught mid-publish -- the caveat says so.
    """

    @functools.wraps(fn)
    def guarded(*args, **kwargs) -> dict:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - catching everything is the whole point
            return _error(f"{type(exc).__name__}: {exc}", _incomplete_db())

    return guarded


def reset_cache() -> None:
    """Forget every loaded database. Call after a rebuild, and between tests."""
    _load_db.cache_clear()


@lru_cache(maxsize=4)
def _load_db(db_dir: str):
    from ..shotdb.store import ShotDB

    return ShotDB.load(Path(db_dir))


def _db():
    """`(db, None)` or `(None, error_dict)` -- the loaded database, or why there isn't one.

    The error is the sentence `ideate query` prints for the same condition, so an assistant that
    relays it to a person gives them the command that fixes it.
    """
    paths = config.load_paths()
    if not (paths.db_dir / "manifest.json").exists():
        return None, _error(
            f"no database at {paths.db_dir} -- run `ideate build --list poc_v1` first"
        )
    return _load_db(str(paths.db_dir)), None


def _error(message: str, caveats: list[str] | None = None) -> dict:
    return {"error": message, "caveats": list(caveats or [])}


def _segment(name: str, caveats: list[str]) -> str | None:
    """`name` as a `SegName`, appending a caveat when it had to be corrected. None if unknown."""
    got = SEGMENT_ALIASES.get(name.strip().lower(), name.strip())
    if got not in SEGMENTS:
        return None
    if got != name:
        caveats.append(f"segment {name!r} read as {got!r}")
    return got


def _range(value: Any):
    """One constraint as a `Range`. `{"lo": x, "hi": y}`, or a two-element `[lo, hi]`.

    Both, because a model writes both and neither is wrong; either bound may be null for a
    one-sided constraint.
    """
    from ..schema import Range

    if isinstance(value, dict):
        return Range(lo=value.get("lo"), hi=value.get("hi"))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return Range(lo=value[0], hi=value[1])
    raise ValueError(
        f"constraint {value!r} is neither {{'lo': .., 'hi': ..}} nor a two-element [lo, hi]"
    )


# --------------------------------------------------------------------------------- the tools


def search_shots(
    text: str = "",
    ref_shot: int | None = None,
    segment: str = "flat_top",
    constraints: dict | None = None,
    actuators: dict | None = None,
    require_labels: list[str] | None = None,
    avoid_labels: list[str] | None = None,
    n: int = 10,
) -> dict:
    """Find DIII-D shots resembling a description, a reference shot, or a set of conditions.

    Every argument is optional but at least one of `text`, `ref_shot`, `constraints` and
    `actuators` must say something: they are the search channels, and with none of them the
    search has nothing to rank on (the reply then says so in `caveats`).

    Args:
        text: free text about the physics wanted -- "QH-mode at low torque", "locked mode after
            an RMP ramp". Matched against the operator logbook and the mini-proposal titles.
        ref_shot: a shot number to find neighbours of. It is never returned as its own neighbour.
        segment: which part of the discharge to compare, one of "full", "ramp_up", "flat_top",
            "ramp_down". "flattop" is understood and corrected.
        constraints: hard filters on segment scalars, `{"ip_mean": {"lo": 1.0e6, "hi": 1.5e6}}`
            (a two-element `[lo, hi]` works too, and either bound may be null). Column names are
            the ones `describe_shot` returns under `record.segments[].raw`/`.derived`. A shot
            whose value was never recorded never satisfies a constraint.
        actuators: a proposed actuator setting, `{"nbi.total": 5.0e6, "ech.total": 8.0e5}`, in SI
            units. Used both to rank and to check the proposal against DIII-D's operating limits,
            which is answered in `proposal_flags` even when no shot resembles it.
        require_labels: labels every result must carry ("QH", "H", "L", or an operational label).
        avoid_labels: labels no result may carry ("dud", "disrupted").
        n: how many results to return.

    Returns:
        `{"results": [...], "n": int, "proposal_flags": [...], "caveats": [str], "query": {...}}`
        or `{"error": str, "caveats": [str]}`. Each result carries the shot, its score, a
        one-paragraph description, the labels, the outcome and an `explanation` naming which
        channel found it and how far each constraint was from the query.
    """
    from ..retrieval import rank as rank_mod
    from ..retrieval.phenomena import PhenomenaError
    from ..schema import QueryState

    caveats: list[str] = []
    seg = _segment(segment, caveats)
    if seg is None:
        return _error(f"unknown segment {segment!r}; the segments are {', '.join(SEGMENTS)}")
    db, err = _db()
    if err:
        return err
    try:
        state = QueryState(
            text=text or None,
            ref_shot=ref_shot,
            segment=seg,
            constraints={k: _range(v) for k, v in (constraints or {}).items()},
            actuators={k: float(v) for k, v in (actuators or {}).items()},
            require_labels=set(require_labels or ()),
            avoid_labels=set(avoid_labels or ()),
            n=int(n),
        )
    except (ValueError, TypeError) as exc:
        return _error(str(exc), caveats)
    if state.ref_shot is not None and state.ref_shot not in db.shots.index:
        held = sorted(int(s) for s in db.shots.index)
        span = f"{held[0]}-{held[-1]}" if held else "(empty)"
        return _error(
            f"shot {state.ref_shot} is not in the database ({len(held)} shots, {span}). "
            f"Add it with `ideate add {state.ref_shot}`.",
            caveats,
        )
    if state.ref_shot is not None and f"{state.ref_shot}:{seg}" not in db.segments.index:
        caveats.append(
            f"shot {state.ref_shot} has no {seg} segment, so the reference channel is not ranking"
        )
    try:
        found = rank_mod.search(state, db)
        report = _pool_report(state, db)
    except PhenomenaError as exc:
        return _error(str(exc), [*caveats, str(exc)])
    except KeyError as exc:
        return _error(
            f"{exc.args[0]}. Columns are the ones describe_shot returns for a shot.", caveats
        )
    except (ValueError, TypeError) as exc:  # a bad query is a message, not a crash
        return _error(f"{type(exc).__name__}: {exc}", caveats)

    fired = {name: len(ranking) for name, ranking in found.rankings.items()}
    caveats.extend(db.label_filter_caveats(state.segment, state.avoid_labels))
    if report["candidates"] == 0:
        caveats.append("nothing passed the filters")
    elif not any(fired.values()):
        # Not the same as "nothing matched": no channel had anything to search ON. A model told
        # only that the list is empty will rephrase, which cannot help.
        caveats.append(
            "no channel had anything to search on -- give ref_shot, text, constraints or "
            "actuators"
        )
    elif not found.items:
        caveats.append(f"{report['candidates']} candidate(s) passed the filters, none ranked")
    if report["nan_excluded"]:
        caveats.append(
            "excluded for having no recorded value: "
            + ", ".join(f"{col} ({k} shots)" for col, k in sorted(report["nan_excluded"].items()))
        )
    return {
        "query": {
            "text": state.text,
            "ref_shot": state.ref_shot,
            "segment": state.segment,
            "n": state.n,
            "candidates": report["candidates"],
            "channels": fired,
        },
        "results": [item.model_dump(mode="json") for item in found.items],
        "n": len(found.items),
        "proposal_flags": [f.model_dump(mode="json") for f in found.proposal_flags],
        "caveats": caveats,
    }


def describe_shot(shot: int, segment: str = "flat_top") -> dict:
    """Everything the database holds about one shot: the prose description and the full record.

    Args:
        shot: the DIII-D shot number.
        segment: which part of the discharge the description leads with, one of "full",
            "ramp_up", "flat_top", "ramp_down". "flattop" is understood and corrected.

    Returns:
        `{"shot": int, "segment": str, "description": str, "record": {...},
        "frame_codes": {...}, "caveats": [str]}` or `{"error": str, "caveats": [str]}`.
        `description` is the same paragraph a search result carries. `record` is the whole stored
        record: the segments with every scalar, the labels and their source, the outcome, and the
        operator logbook entries verbatim. `frame_codes` says whether this shot has an IGNITE
        frame-code cache and, from its provenance sidecar, which device and thread count encoded
        it -- the codes are not bit-identical across either, so two shots encoded differently are
        not necessarily comparable.

    Quote the logbook from `record.human.log_entries` only, and verbatim. The description is
    generated text about the numbers; it is not something anyone said.
    """
    caveats: list[str] = []
    seg = _segment(segment, caveats)
    if seg is None:
        return _error(f"unknown segment {segment!r}; the segments are {', '.join(SEGMENTS)}")
    db, err = _db()
    if err:
        return err
    shot = int(shot)
    if shot not in db.shots.index:
        held = sorted(int(s) for s in db.shots.index)
        span = f"{held[0]}-{held[-1]}" if held else "(empty)"
        return _error(
            f"shot {shot} is not in the database ({len(held)} shots, {span}). "
            f"Add it with `ideate add {shot}`.",
            caveats,
        )
    from ..retrieval import describe as describe_mod

    rec = db.get(shot)
    if rec.segment(seg) is None:
        caveats.append(f"shot {shot} has no {seg} segment; the description falls back to `full`")
    codes = _frame_codes(shot, caveats)
    return {
        "shot": shot,
        "segment": seg,
        "description": describe_mod.describe(rec, seg, db=db),
        "record": rec.model_dump(mode="json"),
        "frame_codes": codes,
        "caveats": caveats,
    }


def _frame_codes(shot: int, caveats: list[str]) -> dict:
    """What the IGNITE frame-code cache for this shot is, and what made it.

    The device belongs in the answer because the codes are not device-independent: measured on
    two shots, cuda and cpu -- and cpu at four threads against cpu at eight, same machine, same
    input -- give one to four different tokens. Two caches encoded on different devices are
    therefore not necessarily comparable, and the cache file itself records nothing about how it
    was made (the payload is four keys the checkpoint validates; provenance is the JSON sibling).
    """
    from ..design import provenance
    from ..shotdb.build import frame_codes_dirs

    try:
        dirs = frame_codes_dirs(config.load_paths())
    except Exception:  # noqa: BLE001 - a provenance lookup may not break the description
        return {"present": False, "device": None, "path": None}
    for d in dirs:
        path = d / f"{int(shot)}.pt"
        if not path.exists():
            continue
        side = provenance.read_sidecar(d, shot) or {}
        if not side.get("device"):
            caveats.append(
                f"the frame-code cache for shot {shot} has no provenance sidecar: the device and "
                f"thread count it was encoded on are not recorded, and the codes are not "
                f"bit-reproducible across either"
            )
        return {
            "present": True,
            "path": str(path),
            "device": side.get("device"),
            "torch_threads": side.get("torch_threads"),
            "encoded_at": side.get("encoded_at"),
            "ignite_revision": side.get("ignite_revision"),
            "provenance_backfilled": side.get("backfilled"),
        }
    return {"present": False, "device": None, "path": None}


#: The four states a `get_events` reply can be in. They are not degrees of the same thing: the
#: first three say the database cannot answer the question asked, and only the last is an answer.
EVENT_STATES = ("unindexed", "unprocessed", "uncovered", "observed")

_UNPROCESSED_CAVEAT = (
    "no observed-event product for shot {shot}: no relevant detector source is recorded as "
    "having completed over it. Returned event rows do not establish that a registered covering "
    "source ran. Absence is not evidence -- this is not a quiet shot, it is an unexamined one"
)

_NO_DETECTION_CAVEAT = (
    "{n} source(s) ran over shot {shot} and reported 0 detections inside their coverage"
    "{window}. This IS an observation of nothing happening, unlike an unprocessed shot"
)

#: Which `evidence_kind` values decide `status`. An allow-list, the same two
#: `labelmaker.events.windows.DIAGNOSTIC_EVIDENCE` and `retrieval.phenomena.OBSERVED_KINDS`
#: name: a `forecast` is a model's estimate, a `text` row is a word in a logbook, a `database`
#: row is an entry in a curated table and a `human`/`model` row is neither a diagnostic nor a
#: heuristic. None of them is somebody having looked at this shot's plasma, so none of them may
#: turn an unexamined shot into an observed one.
OBSERVED_KINDS: tuple[str, ...] = ("detector", "heuristic")

#: Which `evidence_kind` values get a LIST OF THEIR OWN in `get_events`, and are therefore not
#: in `events`. A different question from `OBSERVED_KINDS`, which decides `status`, and not its
#: complement: `OBSERVED_KINDS` is an allow-list ("did somebody look at the plasma"), this is a
#: deny-list ("is this claim a different KIND of claim that must be reported separately"). A
#: `human` or `model` row is neither observed nor separately listed - it stays in `events` with
#: its `evidence_kind` on it, which is the status quo, and giving it a list is a decision
#: nobody has made yet.
OWN_LIST_KINDS: tuple[str, ...] = ("forecast", "text", "database")

#: Said whenever a diagnostic source completed without recording its coverage -- whether or not
#: any other source did record some. The row is a real fact ("it ran") that establishes nothing
#: about any window, and a reader who saw only `n_sources_ok` would count it as a source that
#: looked. Real shot 198658: `actuator/ech_power_total`, ok, NaN..NaN, 0 events.
_UNKNOWN_COVERAGE_CAVEAT = (
    "{n} source(s) ran over shot {shot} and recorded NO coverage -- {names}: ran; coverage "
    "unknown -- says nothing about this window, neither that it was looked at nor that it "
    "was not"
)

_UNCOVERED_UNKNOWN_CAVEAT = (
    "no source with recorded coverage ran over shot {shot}{window}: the sources that completed "
    "did not record what span they read, so nothing establishes that anybody looked -- an empty "
    "result here is not an observation of nothing happening"
)

_TEXT_CAVEAT = (
    "{n} row(s) are in `text_mentions`, not in `events`: a text row is a LEXICON HIT in the "
    "operator logbook -- somebody wrote a word -- and is not an assertion that the phenomenon "
    "occurred, nor a claim about what any diagnostic showed"
)


def get_events(
    shot: int,
    phenomenon: str | None = None,
    t0_s: float | None = None,
    t1_s: float | None = None,
) -> dict:
    """Time-resolved events for one shot: what a detector saw, what a logbook mentioned, what a curated list names, and separately what a model forecast.

    Args:
        shot: the DIII-D shot number.
        phenomenon: keep only this phenomenon ("tearing", "elm", "sawtooth", "disruption", ...).
        t0_s: keep only events overlapping the window starting here (seconds from shot start).
        t1_s: ... and ending here. Either bound may be given alone. `t0_s` must be less than
            `t1_s` and both must be finite; a reversed or non-finite window is an error, never a
            silently empty result.

    Returns:
        `{"shot": int, "status": str, "events": [...], "n": int, "text_mentions": [...],
        "n_text_mentions": int, "database_intervals": [...], "n_database": int,
        "forecasts": [...], "n_forecasts": int, "coverage": {...}, "caveats": [str]}` or
        `{"error": str, "status": "unindexed", "caveats": [str]}`.

    READ `status` FIRST. It is one of four, and an empty `events` means something different in
    each:

    * `unindexed` -- the shot is not in the database at all. Nothing was ever loaded for it.
    * `unprocessed` -- the shot is in the database, but no relevant detector is recorded as having run
      over it. Its empty event list is not evidence that the shot was quiet.
    * `uncovered` -- relevant detectors ran, but none is recorded as having covered the window
      you asked about: either their spans lie elsewhere (the caveat names the span that IS
      covered) or they completed without recording a span at all (the caveat names them). A
      source that ran and recorded no coverage can neither cover nor un-cover a window.
    * `observed` -- a relevant detector's OWN coverage overlaps the window. An empty `events` here
      is a real observation of nothing, and the caveats say how many sources reported it --
      counting only the sources whose coverage overlaps the window, not everything that ran.

    `events`, `text_mentions`, `database_intervals` and `forecasts` are four different kinds of
    claim and must stay apart when you report them. An `events` row is somebody's claim about
    what a DIAGNOSTIC showed, with `source` saying who and `confidence` how sure. A `forecasts`
    row (evidence_kind `forecast`) is a model's estimate of what was ABOUT to happen, computed
    from a risk curve and a threshold; reporting one as an observation is how "shot 190591
    disrupted at 3.2 s" gets written from a probability. A `text_mentions` row (evidence_kind
    `text`) is a lexicon hit in the operator logbook -- somebody wrote a word at some point in a
    shift -- which is evidence that the word was written and not that the phenomenon occurred.
    A `database_intervals` row (evidence_kind `database`) is a row of a curated table somebody
    sent us: it names a shot and a time, not a measurement, its `confidence` is null because a
    human list has no calibrated probability, and its `coverage` is null because nobody recorded
    which interval of the shot was examined -- so a shot's ABSENCE from such a list is not a
    negative, and a curated row never makes `status` `observed`.

    Each row carries `t_cov0_s`/`t_cov1_s`, the coverage of the diagnostic that was looked at, and
    `coverage` carries the per-source table, so "nothing was seen here" can be told from "nobody
    looked here".
    """
    import pandas as pd

    caveats: list[str] = []
    shot = int(shot)
    window, err = _window(t0_s, t1_s)
    if err:
        return err
    t0_s, t1_s = window

    db, err = _db()
    if err:
        return err
    if shot not in db.shots.index:
        held = sorted(int(s) for s in db.shots.index)
        span = f"{held[0]}-{held[-1]}" if held else "(empty)"
        # The same sentence `describe_shot` gives, and the same shape of reply: the reviewer's
        # 198658 call got `n: 0` with no caveats from this tool while `describe_shot` correctly
        # said the shot was absent, and an assistant reading the two together learnt that the
        # shot was in the database and quiet.
        return {
            **_error(
                f"shot {shot} is not in the database ({len(held)} shots, {span}). "
                f"Add it with `ideate add {shot}`.",
                caveats,
            ),
            "status": "unindexed",
        }

    from ..labels import event_sources as es
    from ..retrieval import phenomena as ph

    relevant = None
    entry = None
    if phenomenon:
        entry = ph.registry().get(phenomenon)
        relevant = entry.covering_sources if entry is not None else ()
        if entry is None:
            caveats.append(
                f'unknown phenomenon id {phenomenon!r}; the registry has {sorted(ph.registry())}; '
                'absence cannot be interpreted as a detector observation'
            )
        elif not relevant:
            caveats.append(ph.NO_DETECTOR.format(id=phenomenon))
    sources = es.for_shot(db, shot, relevant)
    sources = _accepted_phenomenon_rows(sources, entry, caveats, "coverage")
    if "event_sources" in db.load_errors:
        caveats.append(f"could not read event_sources.parquet: {db.load_errors['event_sources']}")
    summary = _sources_summary(sources, shot)

    paths = config.load_paths()
    path = paths.db_dir / "events.parquet"
    if not path.exists():
        caveats.append(NO_EVENTS)
        df = None
    else:
        try:
            df = pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001 - a corrupt table is a message, not a stack trace
            return _error(f"could not read {path}: {type(exc).__name__}: {exc}")
        df = df[df["shot"] == shot].copy()
        # Read-only compatibility for pre-L-A products. Keep the burst
        # accessible under its actual class without rewriting the store.
        legacy = (df["source"] == "tokeye_transient") & (df["phenomenon"] == "elm")
        df.loc[legacy, "phenomenon"] = "transient"

    all_rows = df if df is not None else None
    if phenomenon and all_rows is not None:
        all_rows = all_rows[all_rows["phenomenon"] == phenomenon]
        all_rows = _accepted_phenomenon_rows(all_rows, entry, caveats, "event")

    nan_excluded = 0
    if all_rows is not None and (t0_s is not None or t1_s is not None):
        # A NaN compares False against either bound, so a row whose times were never recorded
        # falls out of a windowed call looking exactly like a row that did not overlap. Count
        # them and say so: "we do not know when this happened" is not "it did not happen then",
        # and the caller cannot tell the two apart from an absence.
        timeless = all_rows["t0_s"].isna() | all_rows["t1_s"].isna()
        nan_excluded = int(timeless.sum())
        all_rows = all_rows[~timeless]
        if nan_excluded:
            caveats.append(_TIMELESS_CAVEAT.format(n=nan_excluded))
    if all_rows is not None:
        # Overlap, not containment: an event that straddles the edge of the window happened in
        # the window, and a point event (t1 == t0, an L-H transition) is inside a window that
        # touches it.
        if t0_s is not None:
            all_rows = all_rows[all_rows["t1_s"] >= float(t0_s)]
        if t1_s is not None:
            all_rows = all_rows[all_rows["t0_s"] <= float(t1_s)]
        all_rows = all_rows.sort_values(["t0_s", "event_id"], kind="stable")
        rows = [_event_row(rec) for rec in all_rows.to_dict("records")]
    else:
        rows = []

    events = [r for r in rows if r.get("evidence_kind") not in OWN_LIST_KINDS]
    text_mentions = [r for r in rows if r.get("evidence_kind") == "text"]
    database = [r for r in rows if r.get("evidence_kind") == "database"]
    forecasts = [r for r in rows if r.get("evidence_kind") == "forecast"]

    status = es.coverage_state(sources, t0_s, t1_s)
    coverage = _coverage_block(sources, summary)
    window_text = "" if t0_s is None and t1_s is None else f" over [{t0_s}, {t1_s}] s"
    unknown = _unknown_coverage_sources(sources)
    if status == "unprocessed":
        caveats.append(_UNPROCESSED_CAVEAT.format(shot=shot))
    elif status == "uncovered" and coverage["t_cov0_s"] is None:
        caveats.append(_UNCOVERED_UNKNOWN_CAVEAT.format(shot=shot, window=window_text))
    elif status == "uncovered":
        span = coverage["t_cov0_s"], coverage["t_cov1_s"]
        caveats.append(
            f"the window [{t0_s}, {t1_s}] s is outside every source's coverage of shot {shot}, "
            f"which runs {span[0]} to {span[1]} s -- nobody looked there, so an empty result "
            f"says nothing about the window you asked about"
        )
    elif status == "observed" and not events:
        # The count is the sources whose OWN finite coverage overlaps the window, not
        # `n_sources_ok`: that counts the logbook lexicon, a curated table and every
        # unknown-coverage row, none of which reported 0 detections inside any coverage.
        caveats.append(
            _NO_DETECTION_CAVEAT.format(
                n=_n_covering(sources, t0_s, t1_s) or "an unrecorded number of",
                shot=shot,
                window=window_text,
            )
        )
    if unknown:
        caveats.append(
            _UNKNOWN_COVERAGE_CAVEAT.format(
                n=len(unknown), shot=shot, names=", ".join(unknown)
            )
        )
    if summary["n_sources_error"]:
        caveats.append(
            f"{summary['n_sources_error']} source(s) FAILED on shot {shot}: whatever they would "
            f"have seen is missing from this reply"
        )
    if forecasts:
        caveats.append(_FORECAST_CAVEAT.format(n=len(forecasts)))
    if text_mentions:
        caveats.append(_TEXT_CAVEAT.format(n=len(text_mentions)))
    if database:
        caveats.append(_DATABASE_CAVEAT.format(n=len(database)))
    return {
        "shot": shot,
        "status": status,
        "events": events,
        "n": len(events),
        "text_mentions": text_mentions,
        "n_text_mentions": len(text_mentions),
        "database_intervals": database,
        "n_database": len(database),
        "forecasts": forecasts,
        "n_forecasts": len(forecasts),
        "coverage": coverage,
        "nan_excluded": nan_excluded,
        "caveats": caveats,
    }


def _accepted_phenomenon_rows(frame, entry, caveats: list[str], kind: str):
    """Apply the registry's shared diagnostic gate and name every excluded source/diag group."""
    if entry is None or not entry.coverage_diags or frame.empty:
        return frame
    keep = []
    excluded: dict[tuple[str, str], int] = {}
    for row in frame.to_dict("records"):
        accepted = entry.accepts_row(row)
        keep.append(accepted)
        if not accepted:
            key = str(row.get("source")), str(row.get("diag"))
            excluded[key] = excluded.get(key, 0) + 1
    for (source, diag), n in sorted(excluded.items()):
        caveats.append(
            f"{n} {kind} row(s) excluded for {entry.id}: {source}/{diag} does not satisfy "
            f"coverage_diags {list(entry.coverage_diags)}; legacy diagnostic rows establish "
            "neither coverage nor hits"
        )
    return frame.loc[keep]


def _window(t0_s, t1_s) -> tuple[tuple[float | None, float | None], dict | None]:
    """`((t0, t1), None)` or `((None, None), error)`. A reversed window is a mistake, not a query.

    A reversed or NaN window used to come back as a successful empty result, which reads exactly
    like "nothing happened in that interval" -- for an interval that does not exist.
    """
    for name, value in (("t0_s", t0_s), ("t1_s", t1_s)):
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            return (None, None), _error(f"{name}={value!r} is not a number of seconds")
        if not math.isfinite(value):
            return (None, None), _error(
                f"{name}={value} is not finite; give a time in seconds from shot start",
                ["a non-finite window is not an empty window: nothing was searched"],
            )
    if t0_s is not None and t1_s is not None and float(t0_s) >= float(t1_s):
        return (None, None), _error(
            f"the window [{t0_s}, {t1_s}] s is empty or reversed: t0_s must be less than t1_s. "
            f"For events at one instant give a window around it, or one bound alone.",
            ["no rows were searched, so this is not a report that the window was quiet"],
        )
    return (
        (None if t0_s is None else float(t0_s), None if t1_s is None else float(t1_s)),
        None,
    )


def _sources_summary(sources, shot: int) -> dict:
    from ..labels import event_sources as es

    return es.shot_summary(sources, shot)


def _unknown_coverage_sources(sources) -> list[str]:
    """The names of the diagnostic `ok` rows that recorded no coverage, deduplicated, in order."""
    from ..labels import event_sources as es

    out: list[str] = []
    for rec in es.unknown_coverage_rows(sources).to_dict("records"):
        name = str(rec["source"])
        label = f"{name}/{rec['diag']}" if str(rec["diag"]) else name
        if label not in out:
            out.append(label)
    return out


def _n_covering(sources, t0_s, t1_s) -> int:
    """How many diagnostic sources' own finite coverage overlaps the window."""
    from ..labels import event_sources as es

    return len(es.observing_rows(sources, t0_s, t1_s))


def _coverage_block(sources, summary) -> dict:
    from ..labels import event_sources as es

    span = es.coverage_span(sources)
    return {
        **summary,
        "t_cov0_s": None if span is None else span[0],
        "t_cov1_s": None if span is None else span[1],
        "sources": [
            {
                "source": str(r["source"]),
                "status": str(r["status"]),
                "reason": str(r["reason"]),
                "diag": str(r["diag"]),
                "channel": int(r["channel"]),
                "pass_name": str(r["pass_name"]),
                "t_cov0_s": None if _missing(r["t_cov0_s"]) else float(r["t_cov0_s"]),
                "t_cov1_s": None if _missing(r["t_cov1_s"]) else float(r["t_cov1_s"]),
                "n_events": int(r["n_events"]),
            }
            for r in sources.to_dict("records")
        ],
    }


def _event_row(rec: dict) -> dict:
    """One events row as JSON: numpy scalars unwrapped, NaN as null, `attrs` decoded.

    `attrs` is stored as a JSON string (the schema's own choice, so a parquet column can hold a
    per-detector shape); handing that string on would make every caller parse it, and one of
    them would forget.
    """
    out: dict[str, Any] = {}
    for key, value in rec.items():
        if hasattr(value, "item"):
            value = value.item()
        out[key] = None if _missing(value) else value
    raw = out.get("attrs")
    if isinstance(raw, str):
        try:
            out["attrs"] = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            out["attrs"] = {"_unparsed": raw}
    return out



def _missing(value: Any) -> bool:
    """True for anything JSON has no word for: NaN, `pd.NA`, `NaT`, `None`.

    Non-finite floats were the reachable case (`f0_khz`/`horizon_s` are NaN on most rows) and
    were the only one handled; a `pd.NA` or a `NaT` -- what a nullable-integer or a datetime
    column hands back, which this schema will grow -- went through untouched and would raise
    inside the JSON encoder, which is to say inside the transport.
    """
    import pandas as pd

    if value is None:
        return True
    if isinstance(value, float):
        return not math.isfinite(value)
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):  # an array or a list: not a missing scalar
        return False


def _pool_report(q, db) -> dict:
    """`candidates` and `nan_excluded` for one query, without a second pass over the table.

    `rank.search_report` computes the hard-filter mask, walks the constrained columns again for
    the NaN counts, and compares the segment column a third time for a `segment_rows` this tool
    never used -- all on top of the pass each channel inside `search` makes for itself (which is
    deliberate; see `channels.py`: filtering after fusion would let masked rows eat a channel's
    k budget). A query that constrains NOTHING admits every row of its segment, so the mask it
    would build is knowable without building one, and the common call -- a text or reference
    search -- now costs the channels' passes and no more.
    """
    from ..retrieval import channels

    in_segment = (db.segments["segment"] == q.segment).to_numpy(dtype=bool)
    filtering = (
        q.constraints or q.require_labels or q.avoid_labels or q.exclude_shots or q.exclude_runs
    )
    candidates = (
        int(channels.hard_filter(q, db).sum()) if filtering else int(in_segment.sum())
    )
    return {"candidates": candidates, "nan_excluded": channels.nan_excluded(q, db)}
