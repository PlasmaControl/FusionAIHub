"""Span suggestions for the ELM, H-mode, sawtooth and tearing-mode editors.

v1's detectors find these phenomena as points and transitions:
`transients.elm_clock_events` the ELMs on a D-alpha filterscope,
`heuristics.lh_transitions` the L-H and H-L transitions from D-alpha, density
and beam power, `heuristics.sawtooth_events` the crashes on the ECE array. This
module turns one shot's into the spans the review page edits, over the shot's
catalog window, and writes them as a suggestion table (`suggestions`) that the
page opens unreviewed shots on:

- H-mode (method `dalpha_lh`): present from each L-H to the next H-L, or to the
  end of the stretch the inputs measured. An H-L with no L-H before it makes the
  time back to the transition before it, or to the start of that stretch,
  uncertain: the shot was in H-mode then, but the detector did not see it begin.
- ELMing (`elm_clock`): each run of at least `MIN_RUN` ELMs with gaps of at most
  `ELM_MAX_GAP_MS`, padded by `PAD_MS` on both sides, less any time the H-mode
  method saw the shot in L-mode: ELMs are an H-mode phenomenon, and the clock
  also counts L-mode D-alpha spikes. A shot the H-mode method cannot run on
  keeps its runs whole.
- Sawteeth (`ece_sawtooth`): the same rule over the crashes, with gaps of at
  most `SAWTOOTH_MAX_GAP_MS`.

ELM and sawtooth runs form only from the events inside the shot's window and
after the plasma starts (`plasma_start`): the first time the 25 ms centred mean
of |Ip| inside the window reaches `RAMP_FRACTION` of its plateau, the catalog's
flat-top rule (`catalog.window`), with Ip from the corpus or the raw cache. A
shot with no Ip starts `RAMP_FALLBACK_MS` into its window. Events before the
start are dropped before the runs are grouped, so the ramp-up's crash-like
steps and spikes neither make a run nor join one; the time before is absent.
- Tearing modes (`window`): no method yet, so the window alone, all absent. It
  gives the page the catalog window to open each shot on.

Inside the window, time the detector's inputs did not measure is not observable
(3) and the rest is absent unless a span says otherwise. A shot the detector
could not run on is not observable throughout, and the table's meta keeps why.
Inputs come from the corpus, else the raw cache; nothing is fetched.

    pixi run -e labelmaker python -m labeler.events.spans --event edge_localized_mode

runs over the frozen cohort's non-blind shots in queue order (`--limit N` takes
the first N, `--shots` names them; `--windows population` takes the population
instead) and merges into the method's table. A shot already in the table is
skipped unless `--force`, so a rerun never changes what a reviewer was shown.
`--gold` also scores the method's drafts on the event's gold shots (`gold`) and
writes the score into the table's meta, where a later run without it keeps it.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DEFAULT_LABEL_TABLES, Paths, git_sha
from ..scoring.frames import Assessment
from . import coverage, heuristics, suggestions, transients
from .catalog.cohort import read_cohort
from .catalog.states import ABSENT, NOT_OBSERVABLE, PRESENT, UNCERTAIN
from .catalog.window import FLATTOP_FRACTION, FLATTOP_MEAN_MS, _centred_mean
from .review import agreement, labels
from .verify import NoDataError, corpus_signal

log = logging.getLogger(__name__)

VERSION = "v1"
MIN_RUN = 3
PAD_MS = 5.0
ELM_MAX_GAP_MS = 200.0
SAWTOOTH_MAX_GAP_MS = 300.0
#: The resolutions `pipeline` measures each detector's coverage at.
ELM_MIN_GAP_S = transients.MIN_DISTANCE_MS * 1e-3
SAWTOOTH_MIN_GAP_S = heuristics.STEP_SPAN_MS * 1e-3
LH_MIN_GAP_S = heuristics.LH_DROP_WINDOW_MS * 1e-3
#: The plasma starts where the centred `RAMP_MEAN_MS` mean of |Ip| inside the
#: window first reaches `RAMP_FRACTION` of its `PLATEAU_PERCENTILE`th
#: percentile: the catalog's own flat-top rule (rule 4). Of the fractions tried
#: on the 450 queue shots (0.5 to 0.8), 0.8 leaves 5 sawtooth drafts and no ELM
#: draft present before 300 ms; at 0.5, 128 sawtooth drafts still are.
RAMP_FRACTION = FLATTOP_FRACTION
RAMP_MEAN_MS = FLATTOP_MEAN_MS
PLATEAU_PERCENTILE = 95
#: Without Ip, the start is this far into the window: on the 450 queue shots,
#: all of which have Ip, the start above falls a median 725 ms in (5-95 %:
#: 372-1427 ms).
RAMP_FALLBACK_MS = 700.0
IP_MISSING = (NoDataError, KeyError, OSError)
START_RULE = {
    "ip_fraction": RAMP_FRACTION,
    "ip_mean_ms": RAMP_MEAN_MS,
    "plateau_percentile": PLATEAU_PERCENTILE,
    "fallback_ms": RAMP_FALLBACK_MS,
    "ip": "the corpus's 'ip', else the raw cache's",
    "events": "only those in [start, window end] form runs",
}

Window = tuple[int, int]
Interval = tuple[float, float]


@dataclass(frozen=True)
class Found:
    """What a method saw on one shot, in ms: its spans, and where it could see."""

    spans: tuple[tuple[float, float, int], ...]
    measured: tuple[Interval, ...]


def runs(
    times_ms: Iterable[float], *, max_gap_ms: float, min_count: int, pad_ms: float
) -> list[Interval]:
    """Each run of at least `min_count` points no more than `max_gap_ms` apart,
    as `(first - pad_ms, last + pad_ms)`."""
    t = np.sort(np.asarray(list(times_ms), dtype=np.float64))
    if not len(t):
        return []
    groups = np.split(t, np.flatnonzero(np.diff(t) > max_gap_ms) + 1)
    return [
        (float(g[0]) - pad_ms, float(g[-1]) + pad_ms)
        for g in groups
        if len(g) >= min_count
    ]


def hmode_spans(
    marks: Iterable[tuple[float, bool]], measured: Sequence[Interval]
) -> list[tuple[float, float, int]]:
    """H-mode spans from `(t_ms, is_lh)` transitions, stretch by stretch."""
    marks = sorted(marks)
    spans = []
    for lo, hi in measured:
        start = None  # the L-H that opened the H-mode we are in
        last = lo  # the transition before this one, or the stretch's start
        for t, is_lh in (m for m in marks if lo <= m[0] <= hi):
            if is_lh:
                start = t if start is None else start
            elif start is not None:
                spans.append((start, t, PRESENT))
                start = None
            else:
                spans.append((last, t, UNCERTAIN))
            last = t
        if start is not None:
            spans.append((start, hi, PRESENT))
    return spans


def clip(spans, measured: Sequence[Interval]) -> list[tuple[float, float, int]]:
    """Each span's parts inside the measured stretches."""
    return [
        (max(a, lo), min(b, hi), state)
        for a, b, state in spans
        for lo, hi in measured
        if max(a, lo) < min(b, hi)
    ]


def minus(intervals: Sequence[Interval], holes: Iterable[Interval]) -> list:
    """`intervals` without the time in `holes`."""
    holes = sorted(holes)
    out = []
    for lo, hi in intervals:
        cursor = lo
        for a, b in holes:
            if b <= cursor or a >= hi:
                continue
            if a > cursor:
                out.append((cursor, a))
            cursor = max(cursor, b)
        if cursor < hi:
            out.append((cursor, hi))
    return out


def shot_rows(shot: int, window: Window, found: Found | None) -> list[list]:
    """The shot's rows tiling `window`: not observable where the method could not
    see, absent where it could, and its spans."""
    lo, hi = window
    painted = [(lo, hi, NOT_OBSERVABLE)]
    if found is not None:
        seen = clip([(a, b, ABSENT) for a, b in found.measured], [(lo, hi)])
        painted += seen
        painted += clip(found.spans, [(a, b) for a, b, _ in seen])
    return suggestions.span_rows(shot, window, painted)


def ramp_start(t_ms, ip, window) -> float | None:
    """The first time inside `window` that the centred mean of |Ip| reaches
    `RAMP_FRACTION` of its plateau, or None if nothing there can be measured."""
    t = np.asarray(t_ms, dtype=np.float64).ravel()
    y = np.abs(np.asarray(ip, dtype=np.float64).ravel())
    inside = (t >= window[0]) & (t <= window[1])
    t, y = t[inside], y[inside]
    if t.size < 3:
        return None
    means = _centred_mean(y, round(RAMP_MEAN_MS / 2 / float(np.median(np.diff(t)))))
    finite = np.isfinite(means)
    if not finite.any():
        return None
    plateau = float(np.percentile(means[finite], PLATEAU_PERCENTILE))
    if not plateau > 0:
        return None
    return float(t[np.flatnonzero(finite & (means >= RAMP_FRACTION * plateau))[0]])


def plasma_start(shot: int, paths: Paths, window: Window) -> tuple[float, str]:
    """`(start_ms, how)`: `ramp_start` on the shot's Ip, else `RAMP_FALLBACK_MS`
    into the window, with why."""
    try:
        t_s, y = read(shot, "ip", paths)
        start = ramp_start(t_s * 1000.0, y[0], window)
        why = None if start is not None else "Ip never measured inside the window"
    except IP_MISSING as error:
        start, why = None, f"{type(error).__name__}: {error}"
    if start is not None:
        return start, "ip"
    fallback = f"window start + {RAMP_FALLBACK_MS:g} ms: {why}"
    return float(window[0]) + RAMP_FALLBACK_MS, fallback


def in_plasma(times_ms: Iterable[float], start: float, window: Window) -> list:
    """The events from the plasma's start to the window's end."""
    return [t for t in times_ms if start <= t <= window[1]]


def from_start(intervals: Iterable[Interval], start: float) -> list[Interval]:
    """`intervals` from `start` on: a run's padding does not reach before it."""
    return [(max(a, start), b) for a, b in intervals if b > start]


def read(shot: int, group: str, paths: Paths, channels=None):
    """`(t_s, y)` of one group from the corpus, else the raw cache; never fetched."""
    for root in (paths.corpus, paths.raw_cache):
        try:
            array = corpus_signal(shot, group, channels=channels, corpus=root)
        except NoDataError:
            continue
        return np.asarray(array.x, dtype=np.float64) / 1000.0, array.y
    raise NoDataError(f"shot {shot}: no {group!r} in the corpus or the raw cache")


def _ms(cov: coverage.Coverage) -> tuple[Interval, ...]:
    return tuple((lo * 1000.0, hi * 1000.0) for lo, hi in cov.intervals)


def _dalpha(shot: int, paths: Paths):
    return read(shot, "filterscopes", paths, range(heuristics.N_DALPHA_CHANNELS))


def dalpha_channel(y, shot: int) -> int:
    """The filterscope row the ELM clock reads: the first of `y`'s (FS01, FS02,
    ...) with two finite samples in a row, as `pipeline` picks. The ELM editor
    draws the same one."""
    finite = (np.isfinite(row[:-1]) & np.isfinite(row[1:]) for row in y)
    channel = next((i for i, ok in enumerate(finite) if ok.any()), -1)
    if channel < 0:
        raise NoDataError(f"shot {shot}: no finite D-alpha in filterscopes 0-7")
    return channel


def detect_elm(shot: int, paths: Paths, window: Window | None = None) -> Found:
    """ELM runs; with a `window`, only from the plasma's start (`plasma_start`)."""
    t_s, y = _dalpha(shot, paths)
    channel = dalpha_channel(y, shot)
    cov = coverage.Coverage.measured(t_s, y[channel], min_gap_s=ELM_MIN_GAP_S)
    found = transients.elm_clock_events(y[channel], t_s, shot=shot, channel=channel)
    elms = [e.t0_s * 1000 for e in found if e.phenomenon == transients.ELM_PHENOMENON]
    if window is not None:
        start, _ = plasma_start(shot, paths, window)
        elms = in_plasma(elms, start, window)
    spans = runs(elms, max_gap_ms=ELM_MAX_GAP_MS, min_count=MIN_RUN, pad_ms=PAD_MS)
    spans = minus(spans, lmode(shot, paths))
    if window is not None:
        spans = from_start(spans, start)
    return Found(tuple((a, b, PRESENT) for a, b in spans), _ms(cov))


def lmode(shot: int, paths: Paths) -> list[Interval]:
    """Where the H-mode method saw the shot in L-mode; none if it cannot run."""
    try:
        hmode = detect_hmode(shot, paths)
    except Exception as error:  # noqa: BLE001 - no gate is the fallback
        log.info("shot %d: no H-mode gate for the ELMs: %s", shot, error)
        return []
    return minus(hmode.measured, [(a, b) for a, b, _ in hmode.spans])


def detect_hmode(shot: int, paths: Paths, window: Window | None = None) -> Found:
    """H-mode spans over what the inputs measured; `window` is not used: the
    transitions set their own start."""
    dalpha_t, dalpha_y = _dalpha(shot, paths)
    ne_t, ne_y = read(shot, "co2", paths, [0])
    pinj_t, pinj_y = read(shot, "pinj", paths)
    cov = coverage.Coverage.measured(
        dalpha_t, dalpha_y, min_gap_s=LH_MIN_GAP_S
    ).intersect(
        coverage.Coverage.measured(ne_t, ne_y[0], min_gap_s=LH_MIN_GAP_S),
        coverage.Coverage.measured(pinj_t, pinj_y, min_gap_s=LH_MIN_GAP_S),
    )
    found = heuristics.lh_transitions(
        dalpha_t,
        dalpha_y,
        ne_t_s=ne_t,
        ne_y=ne_y[0],
        betan_t_s=None,
        betan_y=None,
        pinj_t_s=pinj_t,
        pinj_y=np.asarray(pinj_y, dtype=np.float64).sum(axis=0) * 1e-3,  # W -> kW
        shot=shot,
        t_cov=cov.hull,
    )
    kinds = {heuristics.LH_PHENOMENON: True, heuristics.HL_PHENOMENON: False}
    marks = [
        (e.t0_s * 1000, kinds[e.phenomenon]) for e in found if e.phenomenon in kinds
    ]
    measured = _ms(cov)
    return Found(tuple(hmode_spans(marks, measured)), measured)


def detect_sawtooth(shot: int, paths: Paths, window: Window | None = None) -> Found:
    """Sawtooth runs; with a `window`, only from the plasma's start."""
    t_s, y = read(shot, "ece", paths)
    cov = coverage.Coverage.measured(t_s, y, min_gap_s=SAWTOOTH_MIN_GAP_S)
    found = heuristics.sawtooth_events(y, t_s, shot=shot, t_cov=cov.hull)
    crashes = [
        e.t0_s * 1000 for e in found if e.phenomenon == heuristics.SAWTOOTH_PHENOMENON
    ]
    if window is not None:
        start, _ = plasma_start(shot, paths, window)
        crashes = in_plasma(crashes, start, window)
    spans = runs(
        crashes, max_gap_ms=SAWTOOTH_MAX_GAP_MS, min_count=MIN_RUN, pad_ms=PAD_MS
    )
    if window is not None:
        spans = from_start(spans, start)
    return Found(tuple((a, b, PRESENT) for a, b in spans), _ms(cov))


def detect_window(shot: int, paths: Paths, window: Window | None = None) -> Found:
    """No method: the window, all absent, for an editor with nothing to suggest."""
    return Found((), ((-np.inf, np.inf),))


@dataclass(frozen=True)
class Method:
    """An editor's method: `detect(shot, paths, window)`, its inputs, and the
    constants its rule uses, as the table's meta records them."""

    event: str
    name: str
    detect: Callable[[int, Paths, Window | None], Found]
    inputs: tuple[str, ...]
    rule: dict = field(default_factory=dict)


METHODS = {
    m.event: m
    for m in (
        Method(
            "edge_localized_mode",
            "elm_clock",
            detect_elm,
            ("filterscopes", "co2", "pinj", "ip"),
            {
                "min_run": MIN_RUN,
                "pad_ms": PAD_MS,
                "max_gap_ms": ELM_MAX_GAP_MS,
                "min_gap_s": ELM_MIN_GAP_S,
                "start": START_RULE,
                "l_mode": "less the time the dalpha_lh method saw in L-mode",
            },
        ),
        Method(
            "high_confinement_mode",
            "dalpha_lh",
            detect_hmode,
            ("filterscopes", "co2", "pinj"),
            {"min_gap_s": LH_MIN_GAP_S},
        ),
        Method(
            "sawtooth_oscillation",
            "ece_sawtooth",
            detect_sawtooth,
            ("ece", "ip"),
            {
                "min_run": MIN_RUN,
                "pad_ms": PAD_MS,
                "max_gap_ms": SAWTOOTH_MAX_GAP_MS,
                "min_gap_s": SAWTOOTH_MIN_GAP_S,
                "start": START_RULE,
            },
        ),
        Method("neoclassical_tearing_mode", "window", detect_window, ()),
    )
}


def cohort_path(paths: Paths) -> Path:
    """The frozen cohort, `catalog/cohort.csv` (v1 Task 5.2 copies it there).

    The label tables' copy when they hold one, else this checkout's own. The
    review page's label tables are the main checkout's `data/events/`, which
    holds the cohort only once v1 is merged there; the two are one frozen file.
    """
    path = paths.label_tables / "catalog" / "cohort.csv"
    return path if path.is_file() else DEFAULT_LABEL_TABLES / "catalog" / "cohort.csv"


def queue(paths: Paths) -> pd.DataFrame:
    """The cohort's non-blind shots in review-queue order, with their windows."""
    cohort = read_cohort(cohort_path(paths))
    cohort = cohort[~cohort.blind].sort_values("queue_rank", kind="stable")
    return cohort[["shot", "window_start_ms", "window_end_ms"]].reset_index(drop=True)


def population(paths: Paths) -> pd.DataFrame:
    """Every population shot but the blind ones, by shot, with their windows."""
    frame = pd.read_csv(paths.catalog / "population.csv")
    blind = set(read_cohort(cohort_path(paths)).query("blind").shot)
    frame = frame[~frame.shot.isin(blind)].sort_values("shot", kind="stable")
    return frame[["shot", "window_start_ms", "window_end_ms"]].reset_index(drop=True)


def suggest(method: Method, shot: int, window: Window, paths: Paths):
    """`(rows, reason)`: the shot's rows, and why the method could not run, or None."""
    try:
        found, reason = method.detect(int(shot), paths, tuple(window)), None
    except Exception as error:  # noqa: BLE001 - one shot's failure is its own
        log.warning("shot %d: %s could not run: %s", shot, method.name, error)
        found, reason = None, f"{type(error).__name__}: {error}"
    return shot_rows(int(shot), window, found), reason


def _suggest(args):
    return suggest(*args)


def gold(method: Method, paths: Paths, reference=None) -> dict:
    """The method's drafts scored on the event's gold shots, frame by frame.

    The gold shots are the roster's (`<event>/shots.csv`) tier-gold rows; their
    labels are `reference`, a format table, by default the event's saved labels
    (`review/labels.csv`). Each gold label's window is suggested as the page
    would and scored with the agreement gate's frame counts (`agreement.pooled`):
    precision and recall of the drafts' present frames. A gold shot with no label
    is listed under `missing`; one the method could not run on is scored as
    drafted (not observable throughout) and listed under `could_not_run`.
    """
    event_dir = paths.label_tables / method.event
    reference = labels.labels_path(event_dir) if reference is None else reference
    roster = pd.read_csv(event_dir / "shots.csv", dtype={"tier": str})
    shots = sorted(int(s) for s in roster.shot[roster.tier == "gold"])
    saved = labels.read_labels(reference)
    missing = {str(s): "no gold label" for s in shots if s not in saved}
    pairs, failed = [], {}
    for shot in (s for s in shots if s in saved):
        label = saved[shot]
        rows, reason = suggest(method, shot, label.window, paths)
        if reason:
            failed[str(shot)] = reason
        estimate = Assessment.from_rows([(a, b, state) for _, state, a, b, _ in rows])
        pairs.append((Assessment.from_label(label), estimate))
    return {
        "reference": str(reference),
        "git_sha": git_sha(),
        "gold_shots": len(shots),
        "shots": len(pairs),
        **agreement.pooled(pairs),
        "missing": missing,
        "could_not_run": failed,
    }


def run(
    method: Method,
    targets: pd.DataFrame,
    paths: Paths,
    *,
    windows: str,
    force: bool = False,
    workers: int = 1,
    scored: dict | None = None,
) -> dict:
    """Suggest `targets`' shots and merge them into the method's table; `scored`
    (`gold`) replaces the meta's gold score, which is otherwise kept."""
    path = suggestions.table_path(paths, method.event, method.name, VERSION)
    meta_path = path.with_suffix(".meta.json")
    old = pd.read_csv(path, keep_default_na=False) if path.is_file() else None
    old_meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    done = set() if old is None or force else set(old.shot)
    todo = targets[~targets.shot.isin(done)]
    no_window = todo.window_start_ms.isna() | todo.window_end_ms.isna()
    skipped = {str(int(s)): "no catalog window" for s in todo.shot[no_window]}
    todo = todo[~no_window]
    jobs = [
        (method, int(r.shot), (int(r.window_start_ms), int(r.window_end_ms)), paths)
        for r in todo.itertuples()
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_suggest, jobs, chunksize=1))
    else:
        results = [_suggest(job) for job in jobs]
    rows = [row for shot_rows_, _ in results for row in shot_rows_]
    skipped |= {str(job[1]): why for job, (_, why) in zip(jobs, results) if why}
    rerun = {job[1] for job in jobs} | {int(s) for s in skipped}
    if old is not None:
        rows = old[~old.shot.isin(rerun)].values.tolist() + rows
    kept = {k: v for k, v in old_meta.get("skipped", {}).items() if int(k) not in rerun}
    meta = {
        "event": method.event,
        "method": method.name,
        "version": VERSION,
        "git_sha": git_sha(),
        "inputs": list(method.inputs),
        "rule": method.rule,
        "windows": windows,
        "skipped": dict(sorted({**kept, **skipped}.items())),
    }
    scored = old_meta.get("gold") if scored is None else scored
    if scored is not None:
        meta["gold"] = scored
    frame = suggestions.write_table(path, rows, meta)
    return {
        **meta,
        "table": str(path),
        "shots_run": len(jobs),
        "skipped_run": len(skipped),
        "rows": len(frame),
        "shots": int(frame.shot.nunique()),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--event", required=True, choices=sorted(METHODS))
    parser.add_argument("--windows", choices=("cohort", "population"), default="cohort")
    parser.add_argument("--shots", type=int, nargs="+", help="only these shots")
    parser.add_argument("--limit", type=int, default=0, help="only the first N")
    parser.add_argument("--force", action="store_true", help="rerun shots done")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--gold",
        nargs="?",
        const="",
        metavar="TABLE",
        help="score the drafts on the gold shots' labels (TABLE, else the saved)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    paths = Paths.from_env()
    targets = queue(paths) if args.windows == "cohort" else population(paths)
    if args.shots:
        unknown = sorted(set(args.shots) - set(targets.shot))
        if unknown:
            parser.error(f"not in the {args.windows} (or blind): {unknown}")
        targets = targets[targets.shot.isin(args.shots)]
    if args.limit:
        targets = targets.head(args.limit)
    method = METHODS[args.event]
    scored = None
    if args.gold is not None:
        scored = gold(method, paths, Path(args.gold) if args.gold else None)
    summary = run(
        method,
        targets,
        paths,
        windows=args.windows,
        force=args.force,
        workers=args.workers,
        scored=scored,
    )
    keys = ("table", "shots_run", "skipped_run", "rows", "shots")
    print(json.dumps({k: summary[k] for k in keys} | {"gold": summary.get("gold")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
