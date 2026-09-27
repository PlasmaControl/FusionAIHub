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
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DEFAULT_LABEL_TABLES, Paths, git_sha
from . import coverage, heuristics, suggestions, transients
from .catalog.cohort import read_cohort
from .catalog.states import ABSENT, NOT_OBSERVABLE, PRESENT, UNCERTAIN
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


def detect_elm(shot: int, paths: Paths) -> Found:
    t_s, y = _dalpha(shot, paths)
    channel = dalpha_channel(y, shot)
    cov = coverage.Coverage.measured(t_s, y[channel], min_gap_s=ELM_MIN_GAP_S)
    found = transients.elm_clock_events(y[channel], t_s, shot=shot, channel=channel)
    elms = [e.t0_s * 1000 for e in found if e.phenomenon == transients.ELM_PHENOMENON]
    spans = runs(elms, max_gap_ms=ELM_MAX_GAP_MS, min_count=MIN_RUN, pad_ms=PAD_MS)
    spans = minus(spans, lmode(shot, paths))
    return Found(tuple((a, b, PRESENT) for a, b in spans), _ms(cov))


def lmode(shot: int, paths: Paths) -> list[Interval]:
    """Where the H-mode method saw the shot in L-mode; none if it cannot run."""
    try:
        hmode = detect_hmode(shot, paths)
    except Exception as error:  # noqa: BLE001 - no gate is the fallback
        log.info("shot %d: no H-mode gate for the ELMs: %s", shot, error)
        return []
    return minus(hmode.measured, [(a, b) for a, b, _ in hmode.spans])


def detect_hmode(shot: int, paths: Paths) -> Found:
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


def detect_sawtooth(shot: int, paths: Paths) -> Found:
    t_s, y = read(shot, "ece", paths)
    cov = coverage.Coverage.measured(t_s, y, min_gap_s=SAWTOOTH_MIN_GAP_S)
    found = heuristics.sawtooth_events(y, t_s, shot=shot, t_cov=cov.hull)
    crashes = [
        e.t0_s * 1000 for e in found if e.phenomenon == heuristics.SAWTOOTH_PHENOMENON
    ]
    spans = runs(
        crashes, max_gap_ms=SAWTOOTH_MAX_GAP_MS, min_count=MIN_RUN, pad_ms=PAD_MS
    )
    return Found(tuple((a, b, PRESENT) for a, b in spans), _ms(cov))


def detect_window(shot: int, paths: Paths) -> Found:
    """No method: the window, all absent, for an editor with nothing to suggest."""
    return Found((), ((-np.inf, np.inf),))


@dataclass(frozen=True)
class Method:
    event: str
    name: str
    detect: Callable[[int, Paths], Found]
    inputs: tuple[str, ...]


METHODS = {
    m.event: m
    for m in (
        Method(
            "edge_localized_mode",
            "elm_clock",
            detect_elm,
            ("filterscopes", "co2", "pinj"),
        ),
        Method(
            "high_confinement_mode",
            "dalpha_lh",
            detect_hmode,
            ("filterscopes", "co2", "pinj"),
        ),
        Method("sawtooth_oscillation", "ece_sawtooth", detect_sawtooth, ("ece",)),
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
        found, reason = method.detect(int(shot), paths), None
    except Exception as error:  # noqa: BLE001 - one shot's failure is its own
        log.warning("shot %d: %s could not run: %s", shot, method.name, error)
        found, reason = None, f"{type(error).__name__}: {error}"
    return shot_rows(int(shot), window, found), reason


def _suggest(args):
    return suggest(*args)


def run(
    method: Method,
    targets: pd.DataFrame,
    paths: Paths,
    *,
    windows: str,
    force: bool = False,
    workers: int = 1,
) -> dict:
    """Suggest `targets`' shots and merge them into the method's table."""
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
        "rule": {
            "min_run": MIN_RUN,
            "pad_ms": PAD_MS,
            "elm_max_gap_ms": ELM_MAX_GAP_MS,
            "sawtooth_max_gap_ms": SAWTOOTH_MAX_GAP_MS,
        },
        "windows": windows,
        "skipped": dict(sorted({**kept, **skipped}.items())),
    }
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
    summary = run(
        METHODS[args.event],
        targets,
        paths,
        windows=args.windows,
        force=args.force,
        workers=args.workers,
    )
    keys = ("table", "shots_run", "skipped_run", "rows", "shots")
    print(json.dumps({k: summary[k] for k in keys}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
