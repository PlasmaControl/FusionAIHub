"""ELMy spans from the review page's own ELM onsets: the `elm_jump` method.

The page's "ELM onset" markers are jump-and-hold onsets on D-alpha summed over the
filterscope chords that see the outer strike point, upper or lower divertor
(`panels.edge_localized_mode`: `dalpha`, `flat_top`, `jump_and_hold`, `onsets`,
after D. Eldon's OMFIT ELM module). This method turns one shot's into the spans the
ELM editor opens on, as a suggestion table (`suggestions`):

- present: each run of at least `MIN_COUNT` onsets no more than `MAX_GAP_MS` apart,
  padded by `PAD_MS` on both sides (`spans.runs`), less the time the shot's best
  confinement layer calls L-mode (2) or QH-mode (3) (`gate`). ELMs are an H-mode
  phenomenon; QH-mode is ELM-free by definition, and its edge harmonic oscillation
  can fire the jump detector;
- not observable: where the D-alpha trace has no finite sample, or repeats one
  value for at least `spans.DEAD_MS` (a dead channel), and the whole window of a
  shot whose D-alpha cannot be read (named in the meta's `skipped`);
- absent: the rest of the window.

The gate paints the shot's confinement layers on the 10 ms grid as the silver
export does (`review.silver.paint`: each frame takes the first layer that assessed
it, in the order the owner's saves, folded saves, format tables, extend tables, the
review source, other suggestions; confine-ours' `roster_review.csv` and
`gold_shots.csv` are extend tables there; the time an owner's save leaves unmarked
inside its window reads as the event's default class, L-mode (2), since 2026-10-17,
so it is a hole too), then confine-ours' tables the silver page
does not read (`CONFINE_OURS`: `roster.csv`, in another schema). Nothing the D-alpha
L-H detector (`dalpha_lh`) drew is read: not a table named for it, and not the rows
another table took from it (`dalpha_lh_shots`: the confinement review source,
`regimes` v1, opens every shot without curated intervals on the detector's rows).
The clock's gate is that detector, which reads the same lower-divertor filterscope
and removes real ELMy time on lower-divertor shots. The frames the gate paints
L-mode or QH-mode are the holes; a shot no such layer covers has no gate.

The window is the shot's catalog window (`panels._shared.plasma_window`), else its
own Ip window (`frames.targets.window_for`); a shot with neither is skipped. Onsets
stand only inside the Ip flat-top (`flat_top`), as on the page.

The meta's `per_shot` keeps, by shot: the D-alpha row read (`dalpha`: the strike
sum or the spans' filterscope), the divertor `side` ("upper", "lower", or
"fallback" with `why`), the `chords`, `onsets`, the gate layers that decided a
frame (`gate`) and how much present time the gate removed (`gate_removed_ms`), and
the onsets' normalised jump (`jump_median`, `jump_max`). Each onset's normalised jump, the largest value of the
jump-and-hold trace in the `panels.edge_localized_mode.MERGE_MS` after it (a size
attribute on that shot's own scale, no subcategory), is in the onsets table beside
the suggestion table (`onsets_path`: shot, t_ms, jump, in_span).

    python -m labeler.events.elm_jump [--shots S ...] [--limit N] [--version v2] \\
        [--workers N] [--force]

runs over the ELM roster, the silver list and the gold list, less the blind cohort
shots, and merges into the version's table: a shot already in it is skipped unless
`--force`. A table whose meta records other rules than `RULE` is not run into: name
a new `--version`. Nothing is fetched unless `--fetch`. The onsets table is written
after the suggestion table has been validated and written.

v1 (2026-10-14) read the regimes rows on shots without curated intervals, which
are the detector's: on 10 of its shots they kept a lower layer's L-mode from
cutting present time (1.8 s in all). v2 leaves them out and adds 25 silver shots
whose filterscopes were fetched after v1 ran.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Paths, atomic_path, git_sha
from . import coverage, rosters, spans, suggestions
from .raw import NO_FETCH_ENV, raw_signal
from .verify import NoDataError

log = logging.getLogger(__name__)

EVENT = "edge_localized_mode"
METHOD = "elm_jump"
VERSION = "v2"
#: The run rule: onsets no more than `MAX_GAP_MS` apart, at least `MIN_COUNT` of
#: them, padded by `PAD_MS`.
MAX_GAP_MS = 50.0
MIN_COUNT = 5
PAD_MS = 5.0
#: The confinement categories an ELMy span may not cross: L-mode and QH-mode.
GATED = {2: "low", 3: "qh"}
#: Nothing drawn from the D-alpha L-H detector is the gate: a layer whose name holds
#: this, or a table's rows on the shots it took from a table whose path holds it.
NOT_A_GATE = "dalpha_lh"
#: confine-ours' tables, read where the silver page has no confinement layer.
CONFINE_OURS = (
    "confinement/extend_confine_ours/roster.csv",
    "confinement/extend_confine_ours/gold_shots.csv",
)
GOLD_FILE = "gold/gold_shots.csv"
PRESENT = 1

RULE = {
    "onsets": "panels.edge_localized_mode jump-and-hold on D-alpha summed over the "
    "strike-point chords (upper or lower divertor from EFIT01 strike points; the "
    "spans' filterscope where they cannot be read), inside the Ip flat-top",
    "max_gap_ms": MAX_GAP_MS,
    "min_count": MIN_COUNT,
    "pad_ms": PAD_MS,
    "gate": "less the 10 ms frames painted L-mode (2) or QH-mode (3) by the silver "
    "confinement layers (first assessing layer wins) and then confine-ours' tables; "
    "never dalpha_lh output: no dalpha_lh layer, and no rows a table drew from a "
    "dalpha_lh table (regimes v1 on shots outside its curated table)",
    "not_observable": f"no finite D-alpha sample (gaps over "
    f"{spans.ELM_MIN_GAP_S * 1000:g} ms) or one value repeated for "
    f"{spans.DEAD_MS:g} ms or more",
    "window": "catalog window (plasma_window), else the shot's Ip window "
    "(frames.targets.window_for)",
}


def table_path(paths: Paths, version: str = VERSION) -> Path:
    return suggestions.table_path(paths, EVENT, METHOD, version)


def onsets_path(paths: Paths, version: str = VERSION) -> Path:
    """The per-onset table beside the suggestion table."""
    path = table_path(paths, version)
    return path.with_name(path.stem + "_onsets.csv")


# --- one shot --------------------------------------------------------------------


@dataclass
class Shot:
    """One shot's draft: its rows, its onsets and what the meta keeps."""

    rows: list[list]
    onsets: pd.DataFrame
    info: dict = field(default_factory=dict)


def onset_jumps(x_ms, jump, times_ms, within_ms: float) -> np.ndarray:
    """The largest finite `jump` in `[t, t + within_ms)` for each onset `t` (nan
    where that stretch holds none)."""
    x = np.asarray(x_ms, dtype=np.float64)
    j = np.asarray(jump, dtype=np.float64)
    out = np.full(len(times_ms), np.nan)
    for k, t in enumerate(times_ms):
        a, b = np.searchsorted(x, [t, t + within_ms])
        part = j[a : max(b, a + 1)]
        part = part[np.isfinite(part)]
        if part.size:
            out[k] = float(part.max())
    return out


def elmy_runs(times_ms) -> list[tuple[float, float]]:
    """The present spans before the gate: `RULE`'s runs of onsets."""
    return spans.runs(
        times_ms, max_gap_ms=MAX_GAP_MS, min_count=MIN_COUNT, pad_ms=PAD_MS
    )


def gate_holes(values, t0_ms: int) -> list[tuple[float, float]]:
    """The `(start, end)` ms of the runs of 10 ms frames from `t0_ms` whose painted
    confinement category (`values`) is L-mode or QH-mode."""
    from . import grid

    hole = np.isin(np.asarray(values), list(GATED))
    edges = np.flatnonzero(np.diff(np.r_[0, hole.astype(np.int8), 0]))
    return [
        (float(t0_ms + a * grid.GRID_MS), float(t0_ms + b * grid.GRID_MS))
        for a, b in zip(edges[::2], edges[1::2], strict=True)
    ]


def dalpha_lh_shots(path) -> frozenset[int]:
    """The shots on which the table at `path` holds rows it took from a
    `NOT_A_GATE` table.

    A table whose meta's `sources` name such a table holds its rows on every shot
    of its own that its other sources do not hold. The confinement review source,
    `regimes` v1, is one: a shot with curated intervals opens on them and any other
    on the D-alpha L-H detector's rows (its meta's `rule`), so it holds the
    detector's rows on every shot but the curated table's. Where another source
    cannot be read, every shot counts. A table with no meta, or whose meta names no
    such source, holds none.
    """
    path = Path(path)
    try:
        stats = path.stat(), path.with_suffix(".meta.json").stat()
    except OSError:
        return frozenset()
    return _dalpha_lh_shots(str(path), *(s.st_mtime_ns for s in stats))


@lru_cache(maxsize=32)
def _dalpha_lh_shots(path: str, _mtime: int, _meta_mtime: int) -> frozenset[int]:
    try:
        meta = json.loads(Path(path).with_suffix(".meta.json").read_text())
    except ValueError as error:
        log.warning("%s: unreadable meta, its sources unknown: %s", path, error)
        return frozenset()
    sources = meta.get("sources") if isinstance(meta, dict) else None
    entries = sources.values() if isinstance(sources, dict) else sources or ()
    tables = [
        str(e.get("table", "")) if isinstance(e, dict) else str(e)
        for e in entries
        if isinstance(e, dict | str)
    ]
    if not any(NOT_A_GATE in table for table in tables):
        return frozenset()
    shots = set(pd.read_csv(path, usecols=["shot"]).shot.astype(int))
    everything = frozenset(shots)
    for other in tables:
        if NOT_A_GATE in other:
            continue
        try:
            shots -= set(pd.read_csv(other, usecols=["shot"]).shot.astype(int))
        except (OSError, ValueError) as error:
            log.warning(
                "%s: source %r unreadable (%s): every shot counts", path, other, error
            )
            return everything
    return frozenset(shots)


def confinement_layers(paths: Paths, shot: int) -> list:
    """The shot's gate layers in the order they win: the silver page's confinement
    layers, less what `NOT_A_GATE` drew (a layer named for it, or a table's rows on
    a shot it took from it: `dalpha_lh_shots`), then confine-ours' tables
    (`CONFINE_OURS`) where the silver page does not read them."""
    from . import grid
    from .review import labels, silver

    try:
        found = silver.layers(paths, "confinement", int(shot))
    except Exception as error:  # noqa: BLE001 - the tables below still count
        log.warning("shot %s: no silver confinement layers: %s", shot, error)
        found = []
    found = [
        layer
        for layer in found
        if NOT_A_GATE not in layer.name
        and int(shot) not in dalpha_lh_shots(silver.layer_file(paths, layer))
    ]
    for relative in CONFINE_OURS:
        name = f"extend:{Path(relative).with_suffix('').as_posix()}"
        if any(layer.name == name for layer in found):
            continue  # the silver page reads it already
        frame = _confine_ours(paths.label_tables / relative)
        mine = frame[(frame.shot == int(shot)) & (frame.category > 0)]
        if not len(mine):
            continue
        spans_ = tuple(
            (*grid.snap_span(r.t_start, r.t_end), int(r.category))
            for r in mine.itertuples()
        )
        window = (min(a for a, _, _ in spans_), max(b for _, b, _ in spans_))
        found.append(silver.Layer("extend", name, labels.Label(window, spans_)))
    return found


def gate(paths: Paths, shot: int) -> tuple[list[str], list[tuple[float, float]]]:
    """`(sources, holes)`: the gate layers that decided some 10 ms frame, and the
    stretches the frames they decided call L-mode or QH-mode.

    Each frame takes the category of the first layer that assessed it
    (`review.silver.paint`, as the silver grid export does), so a save or a curated
    table wins over a model's draft where both speak, and the draft still fills
    the time they leave out.
    """
    from . import grid
    from .review import silver

    found = confinement_layers(paths, shot)
    if not found:
        return [], []
    t0, hi = silver.time_range([found])
    values, taken = silver.paint(found, t0, (hi - t0) // grid.GRID_MS)
    used = sorted({int(k) for k in taken[taken >= 0]})
    return [found[k].name for k in used], gate_holes(values, t0)


_TABLES: dict[Path, pd.DataFrame] = {}


def _confine_ours(path: Path) -> pd.DataFrame:
    if path not in _TABLES:
        _TABLES[path] = (
            pd.read_csv(path, usecols=["shot", "category", "t_start", "t_end"])
            if path.is_file()
            else pd.DataFrame(columns=["shot", "category", "t_start", "t_end"])
        )
    return _TABLES[path]


def _ip_on(x_ms, shot, paths):
    """Ip on the clock `x_ms`, or None where the shot has none."""
    from .panels._shared import MISSING

    try:
        ip = raw_signal(int(shot), "ip", paths=paths)
    except MISSING:
        return None
    return np.interp(x_ms, ip.x, ip.y[0], np.nan, np.nan)


def side_of(d) -> tuple[str, list[str]]:
    """`(side, chords)` of a `panels.edge_localized_mode.Dalpha`."""
    from .panels import edge_localized_mode as panel

    if d.title != panel.DALPHA_STRIKE_TITLE:
        return "fallback", [d.legend]
    chords = d.caveat.split(" summed", 1)[0].split("+")
    return ("upper" if any(c.endswith("UP") for c in chords) else "lower"), chords


def draft(shot: int, window, paths: Paths) -> Shot:
    """The shot's rows over `window` (ms), its onsets and its meta entry.

    Raises what reading D-alpha raises (`NoDataError`, `ValueError`, ...) when the
    shot has no D-alpha trace to read.
    """
    from .panels import edge_localized_mode as panel

    d = panel.dalpha(shot, paths=paths)
    x = np.asarray(d.x, dtype=np.float64)
    y = np.asarray(d.y, dtype=np.float64)
    ip = _ip_on(x, shot, paths)
    keep = None if ip is None else panel.flat_top(x, ip)
    try:
        jump = panel.jump_and_hold(x, y, keep=keep)
    except NoDataError:
        jump = np.full(len(x), np.nan)
    times = panel.onsets(x, jump, keep=keep) if np.isfinite(jump).any() else []
    times = np.asarray(times, dtype=np.float64)
    sizes = onset_jumps(x, jump, times, panel.MERGE_MS)
    side, chords = side_of(d)

    present = elmy_runs(times)
    sources, holes = gate(paths, shot)
    gated = spans.minus(present, holes)
    before = sum(b - a for a, b in present)
    after = sum(b - a for a, b in gated)

    cov = coverage.Coverage.measured(x / 1000.0, y, min_gap_s=spans.ELM_MIN_GAP_S)
    dead = spans.dead_stretches(x, y)
    stretches = [(a * 1000.0, b * 1000.0) for a, b in cov.intervals]
    measured = tuple(spans.minus(stretches, dead))
    found = spans.Found(tuple((a, b, PRESENT) for a, b in gated), measured)
    rows = spans.shot_rows(int(shot), tuple(window), found)

    lo, hi = window
    inside = (times >= lo) & (times <= hi)
    in_span = np.array([any(a <= t <= b for a, b in gated) for t in times], dtype=bool)
    onsets = pd.DataFrame(
        {
            "shot": int(shot),
            "t_ms": np.round(times[inside], 3),
            "jump": np.round(sizes[inside], 4),
            "in_span": in_span[inside],
        }
    )
    info = {
        "dalpha": d.title,
        "side": side,
        "chords": chords,
        "onsets": int(inside.sum()),
        "gate": sources,
        "gate_removed_ms": round(before - after, 1),
    }
    if side == "fallback":
        info["why"] = d.caveat[:200]
    if inside.any() and np.isfinite(sizes[inside]).any():
        info["jump_median"] = round(float(np.nanmedian(sizes[inside])), 4)
        info["jump_max"] = round(float(np.nanmax(sizes[inside])), 4)
    dead_ms = sum(max(0.0, min(b, hi) - max(a, lo)) for a, b in dead)
    if dead_ms > 0:
        info["dead_ms"] = round(dead_ms, 1)
    return Shot(rows, onsets, info)


def _job(args):
    shot, window, paths = args
    try:
        return draft(shot, window, paths), ""
    except Exception as error:  # noqa: BLE001 - one shot's failure is its own
        log.warning("shot %s: elm_jump could not run: %s", shot, error)
        return None, f"{type(error).__name__}: {error}"


# --- the run ---------------------------------------------------------------------


def shot_list(paths: Paths) -> list[int]:
    """The ELM roster, the silver list and the gold list, less the blind shots."""
    from .review import silver

    roster = rosters.read_roster(rosters.roster_path(EVENT, root=paths.label_tables))
    shots = set(roster.shot.astype(int))
    shots |= set(silver.read_silver(paths.label_tables).shot.astype(int))
    gold = paths.label_tables / GOLD_FILE
    if gold.is_file():
        shots |= set(pd.read_csv(gold).shot.astype(int))
    return sorted(shots - silver.blind_shots(paths.label_tables))


def catalog_windows(paths: Paths) -> dict[int, tuple[float, float]]:
    """The cohort's and the population's rule-4 windows by shot, as `plasma_window`
    reads them (the cohort's first)."""
    out: dict[int, tuple[float, float]] = {}
    for table in (spans.population, spans.queue):
        try:
            frame = table(paths)
        except (OSError, ValueError, KeyError) as error:
            log.warning("no %s windows: %s", table.__name__, error)
            continue
        for row in frame.itertuples(index=False):
            lo, hi = float(row.window_start_ms), float(row.window_end_ms)
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                out[int(row.shot)] = (lo, hi)
    return out


def windows(paths: Paths, shots) -> tuple[dict, dict, dict]:
    """`(windows, sources, skipped)`: each shot's window, where it came from
    ("catalog" or "ip"), and why a shot has none."""
    from ..frames.targets import window_for  # it imports spans

    catalog = catalog_windows(paths)
    got, source, skipped = {}, {}, {}
    for shot in shots:
        try:
            got[shot], source[shot] = window_for(
                int(shot), paths, None, catalog=catalog
            )
        except (ValueError, NoDataError) as error:
            skipped[str(shot)] = f"no window: {error}"
    return got, source, skipped


class RuleChanged(ValueError):
    """A table drafted under other rules than `RULE`."""


def check_rule(path: Path) -> None:
    meta_path = path.with_suffix(".meta.json")
    if not path.is_file():
        return
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    if meta.get("rule") != json.loads(json.dumps(RULE)):
        raise RuleChanged(
            f"{meta_path} records other rules than elm_jump's now; its table is left "
            "as it is: name a new --version"
        )


def run(paths: Paths, *, shots=None, version=VERSION, workers=1, force=False) -> dict:
    """Draft `shots` (`shot_list` by default) into the version's table; its meta."""
    path = table_path(paths, version)
    check_rule(path)
    shots = sorted(int(s) for s in (shots if shots else shot_list(paths)))
    meta_path = path.with_suffix(".meta.json")
    old_rows = pd.read_csv(path, keep_default_na=False) if path.is_file() else None
    old_meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    old_onsets = (
        pd.read_csv(onsets_path(paths, version))
        if onsets_path(paths, version).is_file()
        else None
    )
    have = set() if old_rows is None else set(old_rows.shot.astype(int))
    todo = [s for s in shots if force or s not in have]
    got, source, skipped = windows(paths, todo)
    jobs = [(s, got[s], paths) for s in todo if s in got]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            done = list(pool.map(_job, jobs, chunksize=4))
    else:
        done = [_job(job) for job in jobs]

    rows, onsets, per_shot = [], [], {}
    for (shot, window, _), (result, why) in zip(jobs, done, strict=True):
        entry = {
            "window": [float(window[0]), float(window[1])],
            "window_source": source[shot],
        }
        if result is None:
            # As the span methods do: a shot it could not run on is not observable.
            rows += spans.shot_rows(shot, tuple(window), None)
            skipped[str(shot)] = why
            per_shot[str(shot)] = entry
            continue
        rows += result.rows
        onsets.append(result.onsets)
        per_shot[str(shot)] = {**entry, **result.info}

    ran = {int(s) for s in todo}
    if old_rows is not None:
        rows = old_rows[~old_rows.shot.isin(ran)].values.tolist() + rows
    if old_onsets is not None:
        onsets.insert(0, old_onsets[~old_onsets.shot.isin(ran)])
    keep = {k: v for k, v in old_meta.get("skipped", {}).items() if int(k) not in ran}
    skipped = keep | skipped
    per_shot = {
        k: v for k, v in old_meta.get("per_shot", {}).items() if int(k) not in ran
    } | per_shot
    table = (
        pd.concat(onsets, ignore_index=True)
        if onsets
        else pd.DataFrame(columns=["shot", "t_ms", "jump", "in_span"])
    ).sort_values(["shot", "t_ms"], kind="stable", ignore_index=True)
    sides = pd.Series([v.get("side", "") for v in per_shot.values()]).value_counts()
    meta = {
        "event": EVENT,
        "method": METHOD,
        "version": version,
        "git_sha": git_sha(),
        "categories": {
            "0": "absent",
            "1": "present",
            "2": "uncertain",
            "3": "not_observable",
        },
        "rule": RULE,
        "onsets_table": onsets_path(paths, version).name,
        "onsets_columns": {
            "jump": "largest normalised jump-and-hold value in the "
            "MERGE_MS after the onset (rise over the shot's flat-top p99.5 rise); "
            "a size on the shot's own scale, not a class",
            "in_span": "the onset lies in a present span of the table",
        },
        "sides": {k: int(v) for k, v in sides.items() if k},
        "skipped": dict(sorted(skipped.items(), key=lambda kv: int(kv[0]))),
        "per_shot": dict(sorted(per_shot.items(), key=lambda kv: int(kv[0]))),
    }
    # The table is validated before anything is written, and the onsets follow it,
    # so a table that fails validation leaves both files as they were.
    frame = suggestions.write_table(path, rows, meta)
    with atomic_path(onsets_path(paths, version)) as tmp:
        table.to_csv(tmp, index=False)
    return {
        **meta,
        "table": str(path),
        "rows": len(frame),
        "shots": int(frame.shot.nunique()),
    }


def _version(text: str) -> str:
    if not re.fullmatch(r"v\d+", text):
        raise argparse.ArgumentTypeError(f"not a version such as v1 or v2: {text!r}")
    return text


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shots", type=int, nargs="+", help="only these shots")
    parser.add_argument("--limit", type=int, default=0, help="only the first N")
    parser.add_argument("--version", type=_version, default=VERSION)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="redraft drafted shots")
    parser.add_argument(
        "--fetch", action="store_true", help="let a missing input be fetched (fdp)"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not args.fetch:
        os.environ.setdefault(NO_FETCH_ENV, "1")
    paths = Paths.from_env()
    shots = args.shots or shot_list(paths)
    if args.limit:
        shots = shots[: args.limit]
    try:
        meta = run(
            paths,
            shots=shots,
            version=args.version,
            workers=args.workers,
            force=args.force,
        )
    except RuleChanged as error:
        parser.error(str(error))
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
