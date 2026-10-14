"""Candidate tables for the taxonomy's MHD and Alfven-eigenmode children.

A candidate table is a suggestion table (`suggestions`): tiled intervals with the
catalog's states, 0 absent, 1 present, 2 uncertain, 3 not observable, beside a
`.meta.json` that says how it was made. The review page can start from one
(`review/source.json`); that pointer is not written here. Four methods:

- `n_amplitude` (`mhd_n1`, `mhd_n2`, `mhd_n3`): the toroidal-n amplitude of the
  MPI66M probes (`panels.mhd.amplitude`, dB above the n map's floor) at or above
  the threshold for at least `MIN_MS`, gaps up to `MAX_GAP_MS` merged. Each shot is
  calibrated against its own null (the same map with the probes' angles permuted:
  `panels.mhd`): a cell must fit better than the null's 99th percentile and lie on
  a coherent line, and the threshold of a shot is `THRESHOLD_DB` or the null's own
  amplitude for that n, whichever is larger; the meta records each shot's.
  `mhd_n1` also holds the 1/1 crash spans of `crash`, since a sawtooth is an n=1
  mode the amplitude row may miss. If a table would flag more than `TOO_MUCH` or
  less than `TOO_LITTLE` of the roster's plasma time, the threshold is changed once
  (`retuned`) and the meta records both.
- `crash` (`mhd_n1.m1`): runs of at least `spans.MIN_RUN` sawtooth crashes
  (`panels.mhd_1_1.crash_times`) no more than `spans.SAWTOOTH_MAX_GAP_MS` apart,
  padded by `spans.PAD_MS`: the sawtooth method's own run rule.
- `q_rule` (`mhd_n1.m2`, `mhd_n1.m3`, `mhd_n2.m3`, `mhd_n3.m4`): the tearing-mode
  interval table's present intervals (`TM_INTERVALS`), each with its n, given the
  smallest integer m > n with m/n at least the interval's median EFIT01 q_min
  (`choose_m`), kept only when (m, n) is one of `ALLOWED`; an interval with no
  q_min, or another (m, n), is dropped and counted in the meta.
- `ece_boxes` (`alfven_eigenmode.tae`, `alfven_eigenmode.rsae`): the 26 shots'
  hand-drawn ECE boxes of that type, their time spans united. The files do not
  say what was inspected, so a time with no box of the type is unknown: the
  table's window is where any box of the shot lies, and what is not a box of the
  type is uncertain.
- `mhd_modes` (all eight MHD events, version v5): `labeler.mhd`'s mode tracks, rules
  1-4, the frozen label model and the rotating-resonance gate, run once per shot and
  cached under `$LABELER_ROOT/mhd/labeler/<pipeline.CACHE_VERSION>/`, so the eight events
  share one run; `labeler.mhd.rows` turns a shot's tracks into each event's rows. A parent
  `mhd_nK` is every track with that n but fixed-frequency lines; `mhd_n1` also holds the
  present spans of `MHD_CRASH` (q-gated ECE sawtooth crashes: 1/1, so n = 1), the union
  merged and length-checked by the tracks' span rule. A child m/n is its resonant tracks
  (present) and its unresolved ones (uncertain), inside the parent's present time;
  `mhd_n1.m1` also holds the crash's present spans inside mhd_n1 (a resonant 1/1 track's
  confidence kept where both are), and time with no q = 1 surface
  (`labeler.sawtooth.qgate`) is absent where m1 is observable; 2/1 and 3/1 are set against
  the n = 1 tracks alone (the crash is 1/1 evidence only). The equilibrium is EFIT02, else
  the EFIT02ER or EFIT01 fallback (`labeler.mhd.equilibrium`; the meta records each shot's
  tree); only a shot with no tree makes the children not observable, no Mirnov all eight
  (bar crash time for the n = 1 pair). On EFIT01 a q_min-above track is uncertain for its
  class, never absent. Earlier tables and caches stay on disk. Its roster is the `mhd_n*`
  rosters with the silver and gold lists (`mhd_roster`), never a blind cohort shot (the
  run refuses without `catalog/cohort.csv`); `--event all` writes the eight tables in one
  pass, and `--cached` builds rows from the cache alone, skipping (and naming) a shot whose
  cache key no longer holds. An existing version is never written into unless `--refresh`
  is given.

    python -m labeler.events.candidates --method n_amplitude --event mhd_n1 \\
        [--shots S ...] [--limit N] [--version v1] [--workers N] [--force]
    python -m labeler.events.candidates --method mhd_modes --event all --workers 16
    python -m labeler.events.candidates --method mhd_modes --event all --cached \\
        --version v5

The roster of `mhd_n*` events is `mhd_n1/shots.csv` under the label tables. Without
`--fetch` nothing is fetched (`LABELER_NO_FETCH` is set): a shot with an input
missing is skipped and named in the meta's `skipped`. A run merges its shots into
the version's table, replacing the shots it ran.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Paths, git_sha
from . import equilibrium, spans, suggestions
from .raw import NO_FETCH_ENV
from .verify import NoDataError

log = logging.getLogger(__name__)

VERSION = "v1"

#: `n_amplitude`: a mode is present where its amplitude is at least this many dB
#: above the n map's floor for at least `MIN_MS`, joining gaps up to `MAX_GAP_MS`.
THRESHOLD_DB = 12.0
MIN_MS = 20.0
MAX_GAP_MS = 20.0
#: The share of a roster's plasma time a table may flag: outside these the
#: threshold is changed once. Over `TOO_MUCH` it is raised by `RETUNE_UP`, under
#: `TOO_LITTLE` lowered by `RETUNE_DOWN`.
TOO_MUCH = 0.8
TOO_LITTLE = 0.01
RETUNE_UP = 1.5
RETUNE_DOWN = 0.5

#: `crash`: the sawtooth method's run rule (`spans`).
CRASH_RUN = {
    "min_count": spans.MIN_RUN,
    "max_gap_ms": spans.SAWTOOTH_MAX_GAP_MS,
    "pad_ms": spans.PAD_MS,
}

#: `q_rule`: the tearing-mode children and their (m, n).
Q_RULE_EVENTS = {
    "mhd_n1.m2": (2, 1),
    "mhd_n1.m3": (3, 1),
    "mhd_n2.m3": (3, 2),
    "mhd_n3.m4": (4, 3),
}
ALLOWED = frozenset(Q_RULE_EVENTS.values())
TM_INTERVALS = "neoclassical_tearing_mode/extend_tm_interval/tm_interval.csv"

#: `ece_boxes`: the event's box type in the label files.
ECE_BOX_TYPES = {"alfven_eigenmode.tae": "TAE", "alfven_eigenmode.rsae": "RSAE"}
ECE_LABELS = "alfven_eigenmode/raw/ece_all_labels"

#: `mhd_modes`: the eight events (`labeler.mhd.rows.EVENTS`), the method's own
#: version, and the crash table whose present spans mhd_n1 and its 1/1 child take.
MHD_EVENTS = (
    "mhd_n1",
    "mhd_n2",
    "mhd_n3",
    "mhd_n1.m1",
    "mhd_n1.m2",
    "mhd_n1.m3",
    "mhd_n2.m3",
    "mhd_n3.m4",
)
METHOD_VERSION = {"mhd_modes": "v5"}
MHD_CRASH = ("crash", "v3")
MHD_LISTS = ("silver/silver_shots.csv", "gold/gold_shots.csv")

N_EVENTS = {"mhd_n1": 1, "mhd_n2": 2, "mhd_n3": 3}
EVENTS = {
    "n_amplitude": tuple(N_EVENTS),
    "crash": ("mhd_n1.m1",),
    "q_rule": tuple(Q_RULE_EVENTS),
    "ece_boxes": tuple(ECE_BOX_TYPES),
    "mhd_modes": MHD_EVENTS,
}
PRESENT, UNCERTAIN = 1, 2


# --- n_amplitude and crash -----------------------------------------------------


def threshold_runs(
    t_ms, level, *, threshold=THRESHOLD_DB, min_ms=MIN_MS, max_gap_ms=MAX_GAP_MS
) -> list[tuple[float, float]]:
    """The intervals where `level` (one value per column at `t_ms`) is at least
    `threshold` for `min_ms` or longer, joining gaps of up to `max_gap_ms`.

    A column covers half a step either side of its time; a nan column is not
    above. Two runs are joined when the gap between their columns' edges is at
    most `max_gap_ms`; a joined run shorter than `min_ms` is dropped.
    """
    t = np.asarray(t_ms, dtype=np.float64)
    if len(t) < 2:
        return []
    half = (t[-1] - t[0]) / (len(t) - 1) / 2
    with np.errstate(invalid="ignore"):
        above = np.asarray(level, dtype=np.float64) >= threshold
    if not above.any():
        return []
    first = np.flatnonzero(above & ~np.r_[False, above[:-1]])
    last = np.flatnonzero(above & ~np.r_[above[1:], False])
    joined: list[list[float]] = []
    for a, b in zip(t[first] - half, t[last] + half, strict=True):
        if joined and a - joined[-1][1] <= max_gap_ms:
            joined[-1][1] = float(b)
        else:
            joined.append([float(a), float(b)])
    return [(a, b) for a, b in joined if b - a >= min_ms]


@dataclass
class Levels:
    """One shot's n amplitudes `(3, T)` dB at `t_ms`, its window and its crashes.

    `null_db` is the shot's null amplitude per n (nan where the null found none)
    and `gate` the fit its cells needed (`panels.mhd.Amplitude`); without
    `null_db` the threshold stands as given.
    """

    t_ms: np.ndarray
    amplitude: np.ndarray
    window: tuple[float, float]
    window_source: str
    crashes: np.ndarray | None = None
    crash_error: str = ""
    null_db: np.ndarray | None = None
    gate: float | None = None


def levels(shot, paths, *, crashes: bool) -> Levels:
    """The shot's n=1..3 amplitudes over its record and, with `crashes`, its 1/1
    crash times (an unreadable ECE is `crash_error`, not a failure)."""
    from .panels import mhd, mhd_1_1
    from .panels import neoclassical_tearing_mode as ntm
    from .panels._shared import MISSING, plasma_window

    s = ntm.spectra(shot, paths=paths)
    modes, score = ntm.mode_numbers(s.spec, list(ntm.PROBES.values()), with_score=True)
    got = mhd.amplitude(s, modes, score)
    window = plasma_window(shot, paths)
    source = "catalog"
    if window is None:
        if not s.plasma.any():
            raise NoDataError(f"shot {shot}: no plasma columns in the Mirnov record")
        window, source = (
            (float(s.t_ms[s.plasma][0]), float(s.t_ms[s.plasma][-1])),
            "mirnov",
        )
    found, error = None, ""
    if crashes:
        try:
            found = mhd_1_1.crash_times(shot, paths=paths)
        except (*MISSING, ValueError) as problem:
            error = f"{type(problem).__name__}: {problem}"
    return Levels(
        s.t_ms,
        got.db,
        (float(window[0]), float(window[1])),
        source,
        found,
        error,
        got.null_db,
        got.gate,
    )


def _levels_job(args):
    shot, paths, crashes = args
    try:
        return levels(shot, paths, crashes=crashes), ""
    except Exception as error:  # noqa: BLE001 - one shot's failure is its own
        log.warning("shot %s: could not run: %s", shot, error)
        return None, f"{type(error).__name__}: {error}"


def crash_spans(times_ms) -> list[tuple[float, float]]:
    """The 1/1 runs: `CRASH_RUN`'s rule over the crash times."""
    return spans.runs(times_ms, **CRASH_RUN)


def flagged_fraction(rows) -> float:
    """The share of the tiled time that is present (state 1)."""
    total = present = 0.0
    for _, state, a, b, _ in rows:
        total += b - a
        present += (b - a) if state == PRESENT else 0.0
    return present / total if total else 0.0


def shot_threshold(got: Levels, k, threshold) -> float:
    """The amplitude `k` must reach on this shot: `threshold`, or the shot's null
    amplitude for `k` where that is larger."""
    if got.null_db is None or not np.isfinite(got.null_db[k - 1]):
        return float(threshold)
    return float(max(threshold, got.null_db[k - 1]))


def n_amplitude_rows(shot, got: Levels, k, threshold, *, with_crashes) -> list[list]:
    """The shot's rows for toroidal number `k`: present where `k`'s amplitude holds
    at `threshold` (raised to the shot's null, `shot_threshold`) and, for n=1,
    where the 1/1 crash runs are."""
    found = threshold_runs(
        got.t_ms, got.amplitude[k - 1], threshold=shot_threshold(got, k, threshold)
    )
    painted = [(a, b, PRESENT) for a, b in found]
    if with_crashes and got.crashes is not None:
        painted += [(a, b, PRESENT) for a, b in crash_spans(got.crashes)]
    return suggestions.span_rows(shot, got.window, painted)


# --- q_rule --------------------------------------------------------------------


def choose_m(n: int, q_min: float) -> int:
    """The smallest integer m > n with m / n at least `q_min`.

    A tearing mode of toroidal number n sits on the rational surface q = m/n, and
    the surface nearest the axis that the profile reaches is the first one at or
    above q_min.
    """
    return max(int(n) + 1, math.ceil(n * q_min - 1e-9))


def child_event(n: int, q_min: float) -> str | None:
    """The `Q_RULE_EVENTS` event of this mode, or None when its (m, n) is not one."""
    pair = (choose_m(n, q_min), int(n))
    return next((event for event, mn in Q_RULE_EVENTS.items() if mn == pair), None)


def tm_intervals(paths: Paths) -> pd.DataFrame:
    """The tearing-mode interval table's present spans (one per interval, whole,
    with their n) and its other rows' bounds: columns `shot, t_start, t_end, n`."""
    path = paths.label_tables / TM_INTERVALS
    frame = pd.read_csv(path)
    attrs = frame["attrs"].map(lambda v: json.loads(v) if isinstance(v, str) else {})
    frame["n"] = attrs.map(lambda v: v.get("n"))
    frame["crowd"] = attrs.map(lambda v: int(v.get("iscrowd", 0)))
    return frame


def median_qmin(shot, a, b, paths, *, fetch=False) -> float | None:
    """The median EFIT01 q_min over `[a, b]` ms, or None where it has no sample."""
    try:
        array = equilibrium.signal(int(shot), "qmin", paths, fetch=fetch)
    except (NoDataError, ValueError, KeyError, OSError):
        return None
    t, y = np.asarray(array.x) * 1000.0, np.asarray(array.y)[0]
    inside = (t >= a) & (t <= b) & np.isfinite(y) & (y > 0)
    return float(np.median(y[inside])) if inside.any() else None


def q_rule_rows(shot, frame, event, paths, *, fetch=False):
    """`(rows, info)` for one shot of `frame` (`tm_intervals`)."""
    mine = frame[frame.shot == shot]
    window = (float(mine.t_start.min()), float(mine.t_end.max()))
    kept, info = [], {"intervals": []}
    for row in mine[(mine.category == PRESENT) & (mine.crowd == 1)].itertuples():
        a, b, n = float(row.t_start), float(row.t_end), int(row.n)
        q = median_qmin(shot, a, b, paths, fetch=fetch)
        entry = {"t_start": a, "t_end": b, "n": n, "q_min": q}
        if q is None:
            entry["dropped"] = "no q_min"
        else:
            entry["m"] = choose_m(n, q)
            if child_event(n, q) is None:
                entry["dropped"] = f"({entry['m']}, {n}) is not in the taxonomy"
            else:
                entry["event"] = child_event(n, q)
        info["intervals"].append(entry)
        if entry.get("event") == event:
            kept.append((a, b, PRESENT))
    gaps = [
        (float(r.t_start), float(r.t_end), 3)
        for r in mine[mine.category == 3].itertuples()
    ]
    return suggestions.span_rows(shot, window, [*gaps, *kept]), info


# --- ece_boxes -----------------------------------------------------------------


def ece_box_rows(shot, boxes: pd.DataFrame, box_type: str) -> list[list]:
    """`shot`'s rows for `box_type`: its boxes' time spans united (present) over
    the stretch where any box of the shot lies, the rest of it uncertain."""
    window = (
        float(np.floor(boxes.t1.min() * 1000.0)),
        float(np.ceil(boxes.t2.max() * 1000.0)),
    )
    mine = boxes[boxes["type"] == box_type]
    painted = [(*window, UNCERTAIN)]
    painted += [(a * 1000.0, b * 1000.0, PRESENT) for a, b in zip(mine.t1, mine.t2)]
    return suggestions.span_rows(shot, window, painted)


# --- the run -------------------------------------------------------------------


@dataclass
class Result:
    rows: list[list] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    per_shot: dict[str, dict] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def roster(paths: Paths, event: str) -> list[int]:
    """The event's roster, `<parent>/shots.csv` under the label tables."""
    parent = event.split(".")[0]
    path = paths.label_tables / parent / "shots.csv"
    return sorted(int(s) for s in pd.read_csv(path).shot)


def run_n_amplitude(event, shots, paths, *, threshold=None, workers=1) -> Result:
    k = N_EVENTS[event]
    crashes = event == "mhd_n1"
    jobs = [(s, paths, crashes) for s in shots]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            done = list(pool.map(_levels_job, jobs, chunksize=1))
    else:
        done = [_levels_job(job) for job in jobs]
    got = {s: lv for s, (lv, _) in zip(shots, done, strict=True) if lv is not None}
    result = Result(
        skipped={
            str(s): why for s, (lv, why) in zip(shots, done, strict=True) if lv is None
        }
    )
    first = THRESHOLD_DB if threshold is None else float(threshold)

    def draw(value):
        return {
            s: n_amplitude_rows(s, lv, k, value, with_crashes=crashes)
            for s, lv in got.items()
        }

    def fraction(tables):
        return flagged_fraction([row for rows in tables.values() for row in rows])

    tables = draw(first)
    share, final, retuned = fraction(tables), first, False
    if threshold is None and got and (share > TOO_MUCH or share < TOO_LITTLE):
        final = first * (RETUNE_UP if share > TOO_MUCH else RETUNE_DOWN)
        tables, retuned = draw(final), True
    result.rows = [row for rows in tables.values() for row in rows]
    result.per_shot = {
        str(s): {
            "window": list(lv.window),
            "window_source": lv.window_source,
            "threshold_db": round(shot_threshold(lv, k, final), 2),
            **(
                {"null_db": round(float(lv.null_db[k - 1]), 2)}
                if lv.null_db is not None and np.isfinite(lv.null_db[k - 1])
                else {}
            ),
            **({"fit_gate": round(float(lv.gate), 3)} if lv.gate is not None else {}),
            **({"crashes": len(lv.crashes)} if lv.crashes is not None else {}),
            **({"crash_error": lv.crash_error} if lv.crash_error else {}),
        }
        for s, lv in got.items()
    }
    result.meta = {
        "rule": {
            "n": k,
            "amplitude": "loudest 1-30 kHz cell whose best n is k, whose fit is >= 0.95 "
            "and above a phase-scrambled null's 99th percentile, on a line of 3+ "
            "columns within 1 frequency bin, dB above the n map's floor "
            "(panels.mhd.amplitude); null-calibrated per shot",
            "threshold_db": final,
            "threshold_per_shot": "max(threshold_db, the shot's null amplitude for n); "
            "per_shot[shot].threshold_db",
            "min_ms": MIN_MS,
            "max_gap_ms": MAX_GAP_MS,
            **(
                {"unions": "the 1/1 crash runs of the `crash` method"}
                if crashes
                else {}
            ),
        },
        "threshold_db": final,
        "first_threshold_db": first,
        "flagged_fraction_first": round(share, 4),
        "flagged_fraction": round(fraction(tables), 4),
        "retuned": retuned,
        "retune_rule": f"once, x{RETUNE_UP} over {TOO_MUCH:.0%} flagged, x{RETUNE_DOWN} "
        f"under {TOO_LITTLE:.0%}; none when --threshold-db is given",
    }
    return result


def run_crash(event, shots, paths, *, workers=1) -> Result:
    from .panels import mhd_1_1
    from .panels._shared import MISSING, plasma_window

    result = Result()
    for shot in shots:
        window = plasma_window(shot, paths)
        if window is None:
            result.skipped[str(shot)] = "no catalog window"
            continue
        try:
            times = mhd_1_1.crash_times(shot, paths=paths)
        except (*MISSING, ValueError) as error:
            result.skipped[str(shot)] = f"{type(error).__name__}: {error}"
            continue
        runs = crash_spans(times)
        result.rows += suggestions.span_rows(
            shot, window, [(a, b, PRESENT) for a, b in runs]
        )
        result.per_shot[str(shot)] = {"crashes": len(times), "runs": len(runs)}
    result.meta = {
        "rule": {
            **CRASH_RUN,
            "crashes": "labeler.events.crash (ECE step in noise units, threshold 3, "
            "elm_z 5, together 1 ms, lasting 20), inside the Ip flat-top",
        }
    }
    return result


def run_q_rule(event, shots, paths, *, fetch=False, workers=1) -> Result:
    frame = tm_intervals(paths)
    have = {int(s) for s in frame.shot}
    result = Result()
    dropped = {"no q_min": 0, "not in the taxonomy": 0}
    kept = 0
    for shot in shots:
        if shot not in have:
            result.skipped[str(shot)] = "not in the tearing-mode interval table"
            continue
        rows, info = q_rule_rows(shot, frame, event, paths, fetch=fetch)
        result.rows += rows
        result.per_shot[str(shot)] = info
        for entry in info["intervals"]:
            reason = entry.get("dropped", "")
            if "no q_min" in reason:
                dropped["no q_min"] += 1
            elif reason:
                dropped["not in the taxonomy"] += 1
            kept += entry.get("event") == event
    result.meta = {
        "rule": {
            "m": "the smallest integer m > n with m/n >= the interval's median EFIT01 q_min",
            "event": f"(m, n) = {Q_RULE_EVENTS[event]}",
            "kept_only_if": sorted(ALLOWED),
            "source": TM_INTERVALS,
        },
        "intervals_kept": kept,
        "intervals_dropped": dropped,
    }
    return result


def run_ece_boxes(event, shots, paths, *, workers=1) -> Result:
    from .source_formatters import read_ae_ece

    folder = paths.label_tables / ECE_LABELS
    box_type = ECE_BOX_TYPES[event]
    result = Result()
    records = {shot: boxes for shot, boxes in read_ae_ece(folder)}
    for shot in sorted(records) if not shots else sorted(set(shots) & set(records)):
        boxes = records[shot]
        if not (boxes["type"] == box_type).any():
            result.skipped[str(shot)] = f"no {box_type} box"
            continue
        result.rows += ece_box_rows(shot, boxes, box_type)
        result.per_shot[str(shot)] = {
            "boxes": int((boxes["type"] == box_type).sum()),
            "window_ms": [
                float(np.floor(boxes.t1.min() * 1000)),
                float(np.ceil(boxes.t2.max() * 1000)),
            ],
        }
    result.meta = {
        "rule": {
            "type": box_type,
            "present": f"the union of the shot's {box_type} boxes' time spans, any channel "
            "or frequency",
            "uncertain": "the rest of the stretch where any box of the shot lies: a time "
            "with no box is unknown, not absent",
            "source": ECE_LABELS,
        },
    }
    return result


def mhd_roster(paths: Paths) -> list[int]:
    """`mhd_modes`' shots: the `mhd_n*` rosters, the silver list and the gold list."""
    shots = set()
    for event in N_EVENTS:
        shots |= set(roster(paths, event))
    for name in MHD_LISTS:
        path = paths.label_tables / name
        if path.is_file():
            shots |= {int(s) for s in pd.read_csv(path).shot}
    return sorted(shots)


def _mhd_job(args):
    """One shot through `labeler.mhd.pipeline` and its q = 1 gate: `(shot, Shot or None,
    why, seconds, no-q=1 spans or None, gate error)`. The pipeline reruns a shot whose cache
    key no longer holds; `cached` never runs it: a shot whose cached key differs from
    `pipeline.key` now is skipped as stale, never used silently, and a shot with no window
    keeps that reason."""
    shot, paths, force, cached = args
    from ..mhd import pipeline

    start = time.time()
    try:
        window, source = pipeline.window(shot, paths)
        if window is None:
            return shot, None, f"no window: {source}", time.time() - start, None, ""
        if cached:
            got = pipeline.load(shot, paths)
            if got is None:
                return shot, None, "not in the cache", time.time() - start, None, ""
            stale = pipeline.stale(got, pipeline.key(shot, paths, window))
            if stale:
                why = f"stale cache: the key differs in {', '.join(stale)}"
                return shot, None, why, time.time() - start, None, ""
        else:
            got = pipeline.run_shot(shot, paths, window, source, force=force)
    except Exception as error:  # noqa: BLE001 - one shot's failure is its own
        log.warning("shot %s: could not run: %s", shot, error)
        why = f"{type(error).__name__}: {error}"
        return shot, None, why, time.time() - start, None, ""
    no_q1, gate_error = _gate(shot, paths)
    return shot, got, "", time.time() - start, no_q1, gate_error


def _no_q1(shot, paths):
    """The sawtooth stream's no-q=1 spans (`labeler.sawtooth.qgate`); raises when the gate
    cannot be read."""
    from ..sawtooth import qgate

    found = qgate.shot_spans(shot, paths)
    return [(a, b) for a, b, state, _ in found if state == qgate.NO_Q1]


def _gate(shot, paths) -> tuple[list | None, str]:
    """`(no-q=1 spans, "")`, or `(None, the error)` when the gate cannot be read: the meta
    names the shot (`q_gate.errors`) and its m1 rows take no no-q=1 absent."""
    try:
        return _no_q1(shot, paths), ""
    except Exception as error:  # noqa: BLE001 - one shot's gate failure is its own
        log.warning("shot %s: no q = 1 gate: %s", shot, error)
        return None, f"{type(error).__name__}: {error}"


def mhd_crash_spans(paths: Paths) -> dict[int, list[tuple[float, float]]] | None:
    """`MHD_CRASH`'s present spans per shot, or None when the table is not there."""
    path = suggestions.table_path(paths, "mhd_n1.m1", *MHD_CRASH)
    if not path.is_file():
        return None
    frame = pd.read_csv(path, keep_default_na=False)
    frame = frame[frame.category.astype(int) == PRESENT]
    out: dict[int, list[tuple[float, float]]] = {}
    for row in frame.itertuples(index=False):
        out.setdefault(int(row.shot), []).append((float(row.t_start), float(row.t_end)))
    return out


def _mhd_crash_source(paths: Paths, crash) -> dict | str:
    """The crash table the n = 1 events took (path, sha256, mtime, its q gate), for the meta."""
    path = suggestions.table_path(paths, "mhd_n1.m1", *MHD_CRASH)
    if crash is None:
        return f"no {MHD_CRASH[0]} {MHD_CRASH[1]} table: tracks only"
    meta_path = path.with_suffix(".meta.json")
    gate = (
        json.loads(meta_path.read_text()).get("gate") if meta_path.is_file() else None
    )
    return {
        "table": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "mtime": time.strftime(
            "%Y-%m-%dT%H:%M:%S%z", time.localtime(path.stat().st_mtime)
        ),
        "shots": len(crash),
        "q_gate": gate,
    }


def _qgate_rule():
    try:
        from ..sawtooth import qgate
    except ImportError:
        return None
    return getattr(qgate, "RULE", None)


def run_mhd_modes(
    events, shots, paths, *, workers=1, force=False, cached=False
) -> dict[str, Result]:
    """`mhd_modes` for `events` over `shots`: one pipeline run per shot (cached), every
    event's rows from it. `cached`: read the cache only."""
    from ..mhd import freeze, rows
    from ..mhd.pipeline import cache_dir, tree

    start = time.time()
    jobs = [(int(s), paths, force, cached) for s in shots]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            done = list(pool.map(_mhd_job, jobs, chunksize=1))
    else:
        done = [_mhd_job(job) for job in jobs]
    elapsed = time.time() - start
    crash = mhd_crash_spans(paths)
    gated = [job[0] for job in done if job[1] is not None and job[4] is not None]
    gate_errors = {str(job[0]): job[5] for job in done if job[1] is not None and job[5]}
    frozen = freeze.load()
    results = {event: Result() for event in events}
    clipped = not_in_m1 = crash_only = 0.0
    trees: dict[str, int] = {}
    for shot, got, why, seconds, no_q1, _ in done:
        if got is None:
            for result in results.values():
                result.skipped[str(shot)] = why
            continue
        crash_here = (crash or {}).get(shot, [])
        cells = rows.shot_cells(got, crash=crash_here, no_q1=no_q1)
        clipped += rows.clipped_ms(cells["mhd_n1"], crash_here)
        not_in_m1 += rows.clipped_ms(cells["mhd_n1.m1"], crash_here)
        crash_only += rows.crash_only_ms(got, crash_here) if crash_here else 0.0
        trees[str(tree(got.info))] = trees.get(str(tree(got.info)), 0) + 1
        for event, result in results.items():
            result.rows += cells[event].rows(shot)
            result.per_shot[str(shot)] = _mhd_shot_info(
                got, event, cells[event], seconds
            )
    for event, result in results.items():
        result.meta = {
            "rule": _mhd_rule(event),
            "freeze": {
                "file": "src/labeler/mhd/freeze.json",
                "sha": freeze.sha(),
                "fit_shots": frozen["fit"]["shots"],
                "fit_shots_sha256": frozen["fit"]["shots_sha256"],
                "law": frozen["law"],
            },
            "cache": str(cache_dir(paths)) + (" (read only)" if cached else ""),
            "pipeline_seconds": round(elapsed, 1),
            "workers": workers,
            "trees_run": dict(sorted(trees.items())),
        }
        if event in ("mhd_n1", "mhd_n1.m1"):
            result.meta["crash"] = _mhd_crash_source(paths, crash)
            if crash is not None:
                result.meta["crash"] |= {
                    "only_crash_ms": round(crash_only, 1),
                    "outside_parent_ms": round(clipped, 1),
                    "not_in_m1_ms": round(not_in_m1, 1),
                    "checks": "outside_parent_ms: crash present time mhd_n1 does not "
                    "hold (runs under 20 ms the union's span rule dropped); not_in_m1_ms: "
                    "crash present time m1 does not hold (those runs, and any inside "
                    "no-q=1 time)",
                }
        if event == "mhd_n1.m1":
            result.meta["q_gate"] = (
                {
                    "applied_shots": len(gated),
                    "rule": _qgate_rule(),
                    "absent": "no-q=1 time, except where m1 is not observable",
                    "errors": dict(
                        sorted(gate_errors.items(), key=lambda kv: int(kv[0]))
                    ),
                }
                if gated or gate_errors
                else "not applied (no shot had a q = 1 gate)"
            )
    return results


def _mhd_rule(event) -> dict:
    from ..mhd import rows

    crash = f"{MHD_CRASH[0]} {MHD_CRASH[1]}"
    equilibrium = (
        "EFIT02, else the EFIT02ER or EFIT01 fallback file (labeler.mhd.equilibrium), "
        "each with >= 2 time slices; the per-shot meta records the tree"
    )
    if event in rows.PARENTS:
        out = {
            "n": rows.PARENTS[event],
            "present": "every track with this n but fixed-frequency lines (harmonics and "
            "non-resonant modes included), frames +-2 ms, gaps <= 20 ms merged, "
            "spans < 20 ms dropped; confidence the largest median n-fit score",
            "not_observable": "outside the Mirnov record's finite stretch and over "
            "its NaN gaps; all of a shot without a readable Mirnov record or four live "
            "toroidal probes",
            "equilibrium": equilibrium,
        }
        if event == "mhd_n1":
            out["also_present"] = (
                f"{crash}'s present spans (q-gated ECE sawtooth crashes, 1/1 so n = 1), "
                "whatever the Mirnov shows there; no confidence"
            )
            out["union"] = (
                "the tracks' span rule on the union: absent gaps <= 20 ms between its "
                "present runs are present (no confidence), and a run under 20 ms with "
                "its merges and no track time is dropped"
            )
        return out
    name = rows.CHILDREN[event]
    out = {
        "m/n": name,
        "present": f"tracks with resonance status m/n and label {name} (confidence "
        "P(resonant)), the parent's span rule, inside the parent's present time",
        "uncertain": f"tracks whose status is {', '.join(rows.UNSURE)} and whose 90% "
        f"credible set holds {name} (confidence its posterior); on an EFIT01 shot also "
        f"q_min-above tracks whose candidate is {name} (never absent)",
        "not_observable": "the parent's present time on a shot with no equilibrium "
        "tree, or with no slice within 30 ms, or over tracks the model could not run "
        "on; and the parent's not-observable time",
        "absent": "the rest",
        "equilibrium": equilibrium,
    }
    if event in ("mhd_n1.m2", "mhd_n1.m3"):
        out["parent"] = (
            f"mhd_n1's n = 1 tracks alone: {crash} is 1/1 evidence only, so its "
            "crash-only time is absent for this class where the Mirnov saw no n = 1 "
            "track and not observable where the Mirnov saw nothing"
        )
    if event == "mhd_n1.m1":
        out["also_present"] = (
            f"{crash}'s present spans inside mhd_n1's present time (only a run the "
            "union's span rule dropped is left out); a resonant 1/1 track's confidence "
            "is kept where a crash covers it too"
        )
        out["absent_no_q1"] = (
            "labeler.sawtooth.qgate no-q=1 time, except where m1 is not observable"
        )
    return out


def _mhd_shot_info(got, event, cells, seconds) -> dict:
    """The shot's record in the event's meta: inputs, time per state and, for a parent, the
    statuses of its modelled n tracks; for a child, its labelled and uncertain tracks."""
    from ..mhd import rows
    from ..mhd.equilibrium import TRUSTED
    from ..mhd.pipeline import UNSURE, tree

    info, T = got.info, got.tracks
    counts: dict[str, int] = {}
    if len(T):
        n = int(event.split(".")[0][-1])
        mine = T[(T.n.astype(int) == n) & T.modelled.astype(bool)]
        if event in rows.PARENTS:
            counts = {str(k): int(v) for k, v in mine.status.value_counts().items()}
        else:
            name = rows.CHILDREN[event]
            held = mine.set90.astype(str).str.split("|").apply(lambda s: name in s)
            unsure = mine.status.isin(UNSURE) & held
            if tree(info) not in TRUSTED:
                unsure |= (mine.status == "q_min above") & (mine.cand == name)
            counts = {
                "m/n": int(((mine.status == "m/n") & (mine.label == name)).sum()),
                "uncertain": int(unsure.sum()),
            }
    return {
        "window": info["window"],
        "window_source": info["window_source"],
        "mirnov": info.get("status", ""),
        "tree": tree(info),
        "rotation": bool(info.get("inputs", {}).get("rotation")),
        "tracks": int(info.get("tracks", 0)),
        "modelled": int(info.get("modelled", 0)),
        "seconds": cells.seconds(),
        "status_counts": counts,
        "run_s": round(float(seconds), 1),
    }


def _mhd_totals(per_shot) -> dict:
    """The table's time per state (s) and its tracks' statuses, over every shot in it."""
    seconds: dict[str, float] = {}
    statuses: dict[str, int] = {}
    trees: dict[str, int] = {}
    for info in per_shot.values():
        for state, value in info.get("seconds", {}).items():
            seconds[state] = round(seconds.get(state, 0.0) + value, 3)
        for status, count in info.get("status_counts", {}).items():
            statuses[status] = statuses.get(status, 0) + int(count)
        tree = str(info.get("tree", "efit02" if info.get("efit02") else None))
        trees[tree] = trees.get(tree, 0) + 1
    return {
        "totals_s": seconds,
        "status_counts": statuses,
        "trees": dict(sorted(trees.items())),
    }


def run_mhd_all(
    paths,
    *,
    events=MHD_EVENTS,
    shots=None,
    version=None,
    workers=1,
    force=False,
    cached=False,
    refresh=False,
) -> dict:
    """`mhd_modes` for `events` (all eight by default) in one pass; event -> its meta.
    A version whose table exists is refused unless `refresh` (merge into it)."""
    version = version or METHOD_VERSION["mhd_modes"]
    there = [
        str(table_path(paths, event, "mhd_modes", version))
        for event in events
        if table_path(paths, event, "mhd_modes", version).is_file()
    ]
    if there and not refresh:
        raise FileExistsError(
            f"mhd_modes {version} is already written ({', '.join(there)}); a version is "
            "never overwritten: pick a new --version, or pass --refresh to merge into it"
        )
    from .review.silver import blind_shots

    cohort = paths.label_tables / "catalog" / "cohort.csv"
    if not cohort.is_file():  # cohort_manifest.yaml's output: without it, fail closed
        raise FileNotFoundError(
            f"{cohort} is missing: the blind cohort shots cannot be kept out of mhd_modes"
        )
    blind = blind_shots(paths.label_tables)  # the catalog's blind test shots: never
    shots = [s for s in (shots or mhd_roster(paths)) if int(s) not in blind]
    results = run_mhd_modes(
        events, shots, paths, workers=workers, force=force, cached=cached
    )
    return {
        event: write(paths, event, "mhd_modes", version, result, shots_run=shots)
        for event, result in results.items()
    }


def table_path(paths: Paths, event: str, method: str, version: str) -> Path:
    return suggestions.table_path(paths, event, method, version)


def write(paths, event, method, version, result: Result, *, shots_run) -> dict:
    """Merge `result` into the version's table (its shots replace the old rows) and
    write the meta; return the meta."""
    path = table_path(paths, event, method, version)
    meta_path = path.with_suffix(".meta.json")
    rows, old_meta = list(result.rows), {}
    rerun = {int(s) for s in shots_run}
    if path.is_file():
        old = pd.read_csv(path, keep_default_na=False)
        rows = old[~old.shot.isin(rerun)].values.tolist() + rows
        old_meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    skipped = {
        k: v for k, v in old_meta.get("skipped", {}).items() if int(k) not in rerun
    } | result.skipped
    per_shot = {
        k: v for k, v in old_meta.get("per_shot", {}).items() if int(k) not in rerun
    } | result.per_shot
    meta = {
        "event": event,
        "method": method,
        "version": version,
        "git_sha": git_sha(),
        "categories": {
            "0": "absent",
            "1": "present",
            "2": "uncertain",
            "3": "not_observable",
        },
        **result.meta,
        **(_mhd_totals(per_shot) if method == "mhd_modes" else {}),
        "skipped": dict(sorted(skipped.items(), key=lambda kv: int(kv[0]))),
        "per_shot": dict(sorted(per_shot.items(), key=lambda kv: int(kv[0]))),
    }
    frame = suggestions.write_table(path, rows, meta)
    return {
        **meta,
        "table": str(path),
        "rows": len(frame),
        "shots": int(frame.shot.nunique()),
    }


def run(
    method,
    event,
    paths,
    *,
    shots=None,
    version=None,
    threshold=None,
    fetch=False,
    workers=1,
    refresh=False,
) -> dict:
    """Compute `method` for `event` over `shots` (the roster by default) and write
    its table; the meta. `refresh`: mhd_modes may merge into an existing version."""
    if event not in EVENTS[method]:
        raise ValueError(f"{method} is for {EVENTS[method]}, not {event!r}")
    version = version or METHOD_VERSION.get(method, VERSION)
    if method == "mhd_modes":
        return run_mhd_all(
            paths,
            events=(event,),
            shots=shots,
            version=version,
            workers=workers,
            refresh=refresh,
        )[event]
    shots = (
        list(shots)
        if shots
        else ([] if method == "ece_boxes" else roster(paths, event))
    )
    if method == "n_amplitude":
        result = run_n_amplitude(
            event, shots, paths, threshold=threshold, workers=workers
        )
    elif method == "crash":
        result = run_crash(event, shots, paths)
    elif method == "q_rule":
        result = run_q_rule(event, shots, paths, fetch=fetch)
    else:
        result = run_ece_boxes(event, shots, paths)
    ran = sorted(
        set(shots)
        | {int(s) for s in result.per_shot}
        | {int(s) for s in result.skipped}
    )
    return write(paths, event, method, version, result, shots_run=ran)


def _version(text: str) -> str:
    if not re.fullmatch(r"v\d+", text):
        raise argparse.ArgumentTypeError(f"not a version such as v1 or v2: {text!r}")
    return text


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--method", required=True, choices=sorted(EVENTS))
    parser.add_argument("--event", required=True)
    parser.add_argument("--shots", type=int, nargs="+", help="only these shots")
    parser.add_argument(
        "--limit", type=int, default=0, help="only the first N of the roster"
    )
    parser.add_argument(
        "--version",
        type=_version,
        help=f"default {VERSION}, mhd_modes {METHOD_VERSION['mhd_modes']}",
    )
    parser.add_argument("--threshold-db", type=float, help="n_amplitude: no retune")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="let a missing input be fetched (needs fdp)",
    )
    parser.add_argument(
        "--force", action="store_true", help="mhd_modes: rerun cached shots"
    )
    parser.add_argument(
        "--cached",
        action="store_true",
        help="mhd_modes: the per-shot cache only, nothing recomputed (a stale shot is "
        "skipped and named)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="mhd_modes: merge into an existing version (refused without it)",
    )
    args = parser.parse_args(argv)
    if args.method == "mhd_modes":
        if args.event != "all" and args.event not in MHD_EVENTS:
            parser.error(f"--method mhd_modes is for all, {', '.join(MHD_EVENTS)}")
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        os.environ.setdefault(NO_FETCH_ENV, "1")
        paths = Paths.from_env()
        shots = args.shots or (mhd_roster(paths)[: args.limit] if args.limit else None)
        metas = run_mhd_all(
            paths,
            events=MHD_EVENTS if args.event == "all" else (args.event,),
            shots=shots,
            version=args.version,
            workers=args.workers,
            force=args.force,
            cached=args.cached,
            refresh=args.refresh,
        )
        print(
            json.dumps(
                {
                    event: {k: meta[k] for k in ("table", "rows", "shots", "totals_s")}
                    | {"skipped": len(meta["skipped"])}
                    for event, meta in metas.items()
                },
                indent=1,
            )
        )
        return 0
    if args.event not in EVENTS[args.method]:
        parser.error(f"--method {args.method} is for {', '.join(EVENTS[args.method])}")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not args.fetch:
        os.environ.setdefault(NO_FETCH_ENV, "1")
    paths = Paths.from_env()
    shots = args.shots
    if args.limit and not shots and args.method != "ece_boxes":
        shots = roster(paths, args.event)[: args.limit]
    meta = run(
        args.method,
        args.event,
        paths,
        shots=shots,
        version=args.version,
        threshold=args.threshold_db,
        fetch=args.fetch,
        workers=args.workers,
    )
    print(
        json.dumps(
            {k: v for k, v in meta.items() if k not in ("per_shot", "skipped")},
            indent=1,
        )
    )
    print(f"skipped: {len(meta['skipped'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
