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
`ece_sawtooth` v2 suggestions (D56), whose 10 ms frames (`scoring.frames`) are
pooled to bins the same way.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..ae.xpower import data
from ..config import Paths
from ..events import spans
from ..events.catalog.states import NOT_OBSERVABLE, PRESENT, UNCERTAIN
from ..events.catalog.window import IP_GROUP, assessed_window
from ..events.interval_tables import SAMPLE_MS, read_label_grid
from ..events.panels._shared import plasma_window
from ..scoring.frames import FRAME_MS, OUTSIDE

UNKNOWN, ABSENT, PRESENT_T, UNCERTAIN_T = -1, 0, 1, 2


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


def window_for(shot: int, paths: Paths, grid_hull) -> tuple[tuple[int, int], str]:
    """The shot's window in whole ms, and where it came from (`window_from`).

    In spec §3.2's order: the cohort's or the population's rule-4 window
    (`plasma_window`, rounded inward; it leaves the blind shots out),
    "catalog"; else `assessed_window` on the corpus's or the raw cache's Ip
    (`spans.read`, never fetched), "ip"; else `grid_hull`, the label grid's
    `hull` or None, "labels".
    """
    catalog = plasma_window(shot, paths)
    if catalog is not None:
        lo, hi = math.ceil(catalog[0]), math.floor(catalog[1])
        if lo < hi:
            return (lo, hi), "catalog"
    try:
        t_s, ip = spans.read(shot, IP_GROUP, paths)
    except spans.INPUT_MISSING:
        pass
    else:
        assessed = assessed_window(t_s * 1000.0, ip[0])
        if assessed is not None:
            return assessed, "ip"
    if grid_hull is not None:
        lo, hi = grid_hull
        return (int(lo), int(hi)), "labels"
    raise ValueError(
        f"shot {shot}: no window: no catalog window, no plasma in an Ip record, "
        "and no labelled bin"
    )
