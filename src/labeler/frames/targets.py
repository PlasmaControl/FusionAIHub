"""Per-bin targets for the frame models, and each shot's window (spec §3.2).

A target is one state per bin `[k * bin_ms, (k + 1) * bin_ms)` on the shot's
own clock:
- UNKNOWN (-1): not labelled, or not observable; masked;
- ABSENT (0) and PRESENT_T (1);
- UNCERTAIN_T (2): labelled uncertain, and masked like UNKNOWN.

The legacy targets are 50 ms format grids (`interval_tables.read_label_grid`),
one cell `[t, t + 50)` per `time_ms`: Hiro's ELM onsets, Jalal Butt's H and L
grids and Jaemin Seo's tearing archive, which holds every sampled 0 (D41). A
cell's state is its maximum over rho, unknown where every rho is (spec §3.2),
and a bin of several cells takes their maximum, so an onset anywhere in it
makes it present. The sawtooth's target is an interval table, the
suggestions of the ece_sawtooth v3 detector (ECE and SXR crashes; D56 as
amended: v2's over-called, the owner found on 2026-09-29), at
`suggestions/ece_sawtooth/v3/`, whose 10 ms frames (`scoring.frames`) are
pooled to bins the same way. `target_shots` and `target_bins` read a spec's
original target, whichever kind it is.

From v2 a shot's target is its original's bins with the owner's saved label
over them (`merged`, F2; the owner's words of 2026-09-29 20:20, "just use the
original labels and shots. if i reviewed them then override or add to it"). On
every bin where the owner's label is not UNKNOWN (UNCERTAIN_T included), the
owner's state replaces the original's; elsewhere the original's stands. A shot
the owner saved that the original lacks is added with the owner's label alone.
`merged` is the one rule: `shots` splits by it, `prepare` writes its states
into the features, and `train`, `evaluate` and `gallery` read those states.
`flips` counts what the owner's label changed, bin by bin (F14: the owner's
ELM saves are spans of ELMy time over Hiro's onsets), for `shots`' meta and
`evaluate`'s record; `original_pin` is the original's file and sha256 where it
is one file, the table, for `shots` to pin and `prepare` to check.
"""

from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path

import numpy as np

from ..ae.xpower import data
from ..config import Paths
from ..events import spans, suggestions
from ..events.catalog.states import NOT_OBSERVABLE, PRESENT, UNCERTAIN
from ..events.catalog.window import IP_GROUP, assessed_window, restrike_end
from ..events.interval_tables import SAMPLE_MS, read_label_grid
from ..events.panels._shared import plasma_window
from ..events.review.labels import read_labels
from ..events.verify import NoDataError
from ..scoring.frames import FRAME_MS, OUTSIDE
from . import EventSpec, grid_path

log = logging.getLogger(__name__)

UNKNOWN, ABSENT, PRESENT_T, UNCERTAIN_T = -1, 0, 1, 2
#: Each legacy target's format grids (`grid_path`), by event; P(H) reads Jalal
#: Butt's H grids and his L grids beside them (D38).
GRIDS = {
    "hiro_onsets": ("edge_localized_mode",),
    "jalal_butt_hl": ("high_confinement_mode", "low_confinement_mode"),
    "tearing_archive": ("neoclassical_tearing_mode",),
}
#: Each interval-table target's suggestion table, `(method, version)` (D56).
TABLES = {"ece_sawtooth_v3": ("ece_sawtooth", "v3")}


def _per(bin_ms: float, unit_ms: float, units: str) -> int:
    """How many `unit_ms` one bin holds; a bin must hold a whole number."""
    per = float(bin_ms) / unit_ms
    if not (per >= 1 and per == math.floor(per)):
        raise ValueError(
            f"bin_ms {bin_ms:g} is not a whole number of {unit_ms:g} ms {units}"
        )
    return int(per)


def _grid_bins(npz_path, bin_ms) -> tuple[np.ndarray, np.ndarray]:
    """A legacy grid's bin indices, its first bin to its last, and their states."""
    _per(bin_ms, SAMPLE_MS, "cells")
    grid = read_label_grid(Path(npz_path))
    starts = np.asarray(grid["time_ms"], dtype=np.float64)
    label = np.asarray(grid["label"], dtype=np.float64)
    known = label[~np.isnan(label)]
    if not np.isin(known, (0.0, 1.0)).all():
        values = sorted(set(known.tolist()) - {0.0, 1.0})
        raise ValueError(f"{npz_path}: a legacy grid's cells are 0 or 1, not {values}")
    cells = np.fmax.reduce(label, axis=1)  # NaN only where every rho is
    k = np.floor(starts / bin_ms).astype(np.int64)
    if np.any(starts + SAMPLE_MS > (k + 1) * bin_ms + 1e-6):
        raise ValueError(f"{npz_path}: a cell straddles a {bin_ms:g} ms bin's edge")
    if not len(k):
        return k, np.zeros(0, dtype=np.int8)
    axis = np.arange(k[0], k[-1] + 1)
    pooled = np.full(len(axis), np.nan)
    np.fmax.at(pooled, k - axis[0], cells)
    states = np.where(pooled > 0, PRESENT_T, ABSENT)
    return axis, np.where(np.isnan(pooled), UNKNOWN, states).astype(np.int8)


def legacy_bins(npz_path, bin_ms) -> tuple[np.ndarray, np.ndarray]:
    """`(bin starts ms, states)` of a legacy 0/1 grid, from its first bin to its
    last: the maximum over rho and over the bin's cells, UNKNOWN where none of
    them is known."""
    k, states = _grid_bins(npz_path, bin_ms)
    return k * float(bin_ms), states


def _placed(axis, k, states) -> np.ndarray:
    """`states` at bins `k` on `axis`, UNKNOWN elsewhere."""
    out = np.full(len(axis), UNKNOWN, dtype=np.int8)
    out[k - axis[0]] = states
    return out


def hl_bins(h_path, l_path, bin_ms) -> tuple[np.ndarray, np.ndarray]:
    """`(bin starts ms, states)` of Jalal Butt's H and L grids as P(H)'s target
    (D38), over the bins either grid has: H only is PRESENT_T, L only ABSENT and
    both (a transition) UNCERTAIN_T; neither, H and L both 0 included, is
    UNKNOWN."""
    kh, h = _grid_bins(h_path, bin_ms)
    kl, l_ = _grid_bins(l_path, bin_ms)
    ks = np.concatenate([kh, kl])
    if not len(ks):
        return np.zeros(0), np.zeros(0, dtype=np.int8)
    axis = np.arange(ks.min(), ks.max() + 1)
    is_h = _placed(axis, kh, h) == PRESENT_T
    is_l = _placed(axis, kl, l_) == PRESENT_T
    states = np.select(
        [is_h & is_l, is_h, is_l], [UNCERTAIN_T, PRESENT_T, ABSENT], UNKNOWN
    )
    return axis * float(bin_ms), states.astype(np.int8)


def table_bins(label, window, bin_ms) -> tuple[np.ndarray, np.ndarray]:
    """`(bin starts ms, states)` of one shot's interval-table `Label` over the
    whole bins inside `window` (ms): its 10 ms frames (`ae.xpower.data.targets`),
    a bin taking their highest state. A bin with a frame off the label's window
    or not observable is UNKNOWN; an uncertain one is UNCERTAIN_T."""
    per = _per(bin_ms, FRAME_MS, "frames")
    first, n = data.window_frames(window)
    k0, k1 = -(-first // per), (first + n) // per
    nb = max(0, k1 - k0)
    frames = data.targets(label, k0 * per, nb * per).reshape(nb, per)
    top = frames.max(axis=1)
    unknown = (frames == OUTSIDE).any(axis=1) | (top == NOT_OBSERVABLE)
    states = np.select(
        [unknown, top == UNCERTAIN, top == PRESENT],
        [UNKNOWN, UNCERTAIN_T, PRESENT_T],
        ABSENT,
    )
    return (k0 + np.arange(nb)) * float(bin_ms), states.astype(np.int8)


def label_bins(label, bin_ms) -> tuple[np.ndarray, np.ndarray]:
    """`table_bins` over the label's own window."""
    return table_bins(label, label.window, bin_ms)


def merged(original, label, bin_ms) -> tuple[np.ndarray, np.ndarray] | None:
    """`(bin starts ms, states)` of a shot's target (F2): `original`, its
    `target_bins` or None, with the owner's saved `Label` (or None) over it.

    The bins run from the first either has to the last. Where the owner's
    `label_bins` state is not UNKNOWN it replaces the original's; elsewhere the
    original's stands, UNKNOWN off its bins. None when neither is given."""
    if label is None:
        return original
    over_starts, over = label_bins(label, bin_ms)
    if original is None:
        return over_starts, over
    starts, states = original
    ko = np.rint(np.asarray(starts, dtype=np.float64) / bin_ms).astype(np.int64)
    kl = np.rint(np.asarray(over_starts, dtype=np.float64) / bin_ms).astype(np.int64)
    ks = np.concatenate([ko, kl])
    if not len(ks):
        return np.zeros(0), np.zeros(0, dtype=np.int8)
    axis = np.arange(ks.min(), ks.max() + 1)
    out = _placed(axis, ko, np.asarray(states, dtype=np.int8))
    owned = _placed(axis, kl, over)
    out = np.where(owned != UNKNOWN, owned, out).astype(np.int8)
    return axis * float(bin_ms), out


def bin_range(window, bin_ms: float) -> tuple[int, int]:
    """`(k0, k1)`: the whole bins `k0 .. k1 - 1` of `bin_ms` inside the whole 10 ms
    frames of `window`."""
    per = _per(bin_ms, FRAME_MS, "frames")
    first, n = data.window_frames(window)
    return -(-first // per), (first + n) // per


def on_bins(target, k0: int, k1: int, bin_ms: float) -> np.ndarray:
    """A target's states, `(bin starts ms, states)` or None, on bins `k0 .. k1 -
    1`; UNKNOWN where it has none, and everywhere for None."""
    out = np.full(max(0, k1 - k0), UNKNOWN, dtype=np.int8)
    if target is None:
        return out
    starts, states = target
    k = np.rint(np.asarray(starts, dtype=np.float64) / bin_ms).astype(np.int64)
    states = np.asarray(states, dtype=np.int8)
    inside = (k >= k0) & (k < k1)
    out[k[inside] - k0] = states[inside]
    return out


STATE_NAMES = {
    UNKNOWN: "unknown",
    ABSENT: "absent",
    PRESENT_T: "present",
    UNCERTAIN_T: "uncertain",
}
_STATE_OF = {name: state for state, name in STATE_NAMES.items()}
#: Each change the owner's label can make to a bin (`flips`): the original's
#: state to the owner's, which is never UNKNOWN.
FLIP_KEYS = tuple(
    f"{STATE_NAMES[a]}_to_{STATE_NAMES[b]}"
    for a in STATE_NAMES
    for b in (ABSENT, PRESENT_T, UNCERTAIN_T)
    if a != b
)


def flips(original, owner, mask=None) -> dict[str, int]:
    """What the owner's label did to a shot's target (F14), over the bins in
    `mask` (default all): `original` and `owner` are states on the same bins
    (`on_bins`), UNKNOWN where each has none. `bins` counts the bins, `owned`
    those the owner's label decides (not UNKNOWN: `merged` takes its state), and
    each of `FLIP_KEYS` the owned bins whose state it changed: "absent_to_present"
    is 0 to 1, "present_to_absent" 1 to 0, "*_to_uncertain" a bin taken out of
    the score, "unknown_to_*" one the original had no state for."""
    original = np.asarray(original, dtype=np.int8)
    owner = np.asarray(owner, dtype=np.int8)
    mask = np.ones(len(original), bool) if mask is None else np.asarray(mask, bool)
    owned = mask & (owner != UNKNOWN)
    out = {"bins": int(mask.sum()), "owned": int(owned.sum())}
    for key in FLIP_KEYS:
        a, b = (_STATE_OF[name] for name in key.split("_to_"))
        out[key] = int(np.sum(owned & (original == a) & (owner == b)))
    return out


def flip_total(parts) -> dict[str, int]:
    """`flips`' counts summed, with `shots`, how many were summed."""
    parts = list(parts)
    out = {"shots": len(parts), "bins": 0, "owned": 0} | dict.fromkeys(FLIP_KEYS, 0)
    for part in parts:
        for key, value in part.items():
            out[key] += value
    return out


def _table(paths: Paths, spec: EventSpec) -> Path:
    return suggestions.table_path(paths, spec.event, *TABLES[spec.target])


def original_pin(paths: Paths, spec: EventSpec) -> dict:
    """The original target's file and its sha256, where it is one file (an
    interval table, `TABLES`), for `shots` to pin and `prepare` to check
    (F16); a legacy target is a grid per shot, so it pins nothing and says
    so. A table not there has no sha256."""
    if spec.target not in TABLES:
        folders = [
            str(grid_path(paths, event, 0).parent) for event in GRIDS[spec.target]
        ]
        return {
            "path": None,
            "sha256": None,
            "grids": folders,
            "note": "a grid per shot: no single file to pin",
        }
    table = _table(paths, spec)
    sha = hashlib.sha256(table.read_bytes()).hexdigest() if table.is_file() else None
    return {"path": str(table), "sha256": sha}


def target_shots(paths: Paths, spec: EventSpec) -> list[int]:
    """The shots `spec`'s target has, a grid or a table row each, in order; raises
    when the grids' folder or the table is missing."""
    if spec.target in GRIDS:
        found = set()
        for event in GRIDS[spec.target]:
            folder = grid_path(paths, event, 0).parent
            if not folder.is_dir():
                raise FileNotFoundError(f"{spec.target}: no grids at {folder}")
            found |= {int(path.stem) for path in folder.glob("*.npz")}
        return sorted(found)
    table = _table(paths, spec)
    if not table.is_file():
        raise FileNotFoundError(f"{spec.target}: no table at {table}")
    return sorted(read_labels(table))


def target_bins(paths: Paths, spec: EventSpec, shot: int):
    """`(bin starts ms, states)` of one shot's original target over all of it:
    its grid's bins (`legacy_bins`, `hl_bins`), or its table label's window's."""
    if spec.target == "jalal_butt_hl":
        h, l_ = (grid_path(paths, event, shot) for event in GRIDS[spec.target])
        return hl_bins(h, l_, spec.bin_ms)
    if spec.target in GRIDS:
        (event,) = GRIDS[spec.target]
        return legacy_bins(grid_path(paths, event, shot), spec.bin_ms)
    return label_bins(read_labels(_table(paths, spec))[int(shot)], spec.bin_ms)


def labelled(states) -> bool:
    """Whether any bin is labelled, not UNKNOWN (D60's `labelled_shots`)."""
    return bool(np.any(np.asarray(states) != UNKNOWN))


def hull(starts, states, bin_ms) -> tuple[int, int] | None:
    """The labelled bins' hull in whole ms, `(first start, last start +
    bin_ms)`; None when every bin is UNKNOWN."""
    known = np.flatnonzero(np.asarray(states) != UNKNOWN)
    if not known.size:
        return None
    starts = np.asarray(starts, dtype=np.float64)
    return math.floor(starts[known[0]]), math.ceil(starts[known[-1]] + bin_ms)


def catalog_windows(paths: Paths) -> dict[int, tuple[float, float]]:
    """Every shot's `plasma_window`, the two tables read once: the cohort's
    first (`spans.queue`), then the population's; a table that cannot be read
    is passed over."""
    found: dict[int, tuple[float, float]] = {}
    for table in (spans.queue, spans.population):
        try:
            frame = table(paths)
        except (OSError, ValueError, KeyError) as error:
            log.warning("no %s windows: %s", table.__name__, error)
            continue
        for row in frame.itertuples(index=False):
            start, end = float(row.window_start_ms), float(row.window_end_ms)
            if np.isfinite(start) and np.isfinite(end) and end > start:
                found.setdefault(int(row.shot), (start, end))
    return found


def window_for(
    shot: int, paths: Paths, grid_hull, *, catalog=None
) -> tuple[tuple[int, int], str]:
    """The shot's window in whole ms, and where it came from (`window_from`).

    In spec §3.2's order: the cohort's or the population's rule-4 window
    (`plasma_window`, or `catalog`, `catalog_windows`' answer, when given;
    rounded inward; it leaves the blind shots out), "catalog"; else
    `assessed_window` on the corpus's or the raw cache's Ip (`spans.read`, never
    fetched), ended at a restrike as the catalog's log ends it (`restrike_end`),
    "ip"; else `grid_hull`, the label grid's `hull` or None, "labels". An Ip
    record that cannot be read is logged, and the window falls back.
    """
    if catalog is None:
        window = plasma_window(shot, paths)
    else:
        window = catalog.get(int(shot))
    if window is not None:
        lo, hi = math.ceil(window[0]), math.floor(window[1])
        if lo < hi:
            return (lo, hi), "catalog"
    try:
        t_s, ip = spans.read(shot, IP_GROUP, paths)
    except NoDataError:
        pass
    except (KeyError, OSError) as error:
        log.warning("shot %s: its Ip cannot be read, so no Ip window: %s", shot, error)
    else:
        t_ms, ip_a = t_s * 1000.0, ip[0]
        assessed = assessed_window(t_ms, ip_a)
        if assessed is not None:
            end = restrike_end(t_ms, ip_a, assessed)
            if end is not None and end > assessed[0]:
                assessed = (assessed[0], end)
            return assessed, "ip"
    if grid_hull is not None:
        lo, hi = grid_hull
        return (int(lo), int(hi)), "labels"
    raise ValueError(
        f"shot {shot}: no window: no catalog window, no plasma in an Ip record, "
        "and no labelled bin"
    )
