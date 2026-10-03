"""The reviewed ELM spans as dense per-millisecond targets and as scored 50 ms bins.

`data/events/edge_localized_mode/review/labels.csv` holds, per shot, spans in
ms of the shot's clock in four states: absent (0), present (1), uncertain (2)
and not observable (3). A present span is one ELM (`iscrowd` 0, a few tens of
ms wide) or a crowd (`iscrowd` 1), an ELMing period whose ELMs are not
separated. `kind` names the five cases: `absent`, `individual`, `crowd`,
`uncertain`, `not_observable`.

**Dense targets** (`dense`): one value per 1 ms cell of the input grid,
1 inside a present span, 0 inside an absent one and `IGNORE` (-1) elsewhere
(uncertain, not observable, outside every span). A second array holds the ELM
onsets, the start of each *individual* span as a Gaussian of width
`ONSET_SIGMA_MS`; it is 0 inside absent spans, ignored inside crowds (their
onsets are not marked, so a crowd's ELMs are unlabelled, not absent) and in
uncertain or unlabelled time.

**Scored bins** (`scored_bins`): exactly the rule of
`scripts/labeler/elmo_benchmark.bin_table`, so the ELM-O and `elm_clock` rows of
the benchmark and every new row sit on the same bins. A bin is the 50 ms cell
`[50k, 50k + 50)` of the shot's clock; it is scored when it lies wholly inside
one absent, individual or crowd span of at least half the analysed time, and
wholly inside analysed time. Its truth is 1 in a present span, 0 in an absent
one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

from . import inputs

BIN_MS = 50.0
IGNORE = -1
ONSET_SIGMA_MS = 2.0
KINDS = ("absent", "individual", "crowd", "uncertain", "not_observable")
SCORED_KINDS = ("absent", "individual", "crowd")
#: `review/labels.csv` in the repository (the owner's labels, 119 shots).
REVIEW_CSV = Path("data/events/edge_localized_mode/review/labels.csv")


def review_table(path: str | Path) -> pd.DataFrame:
    """The review spans, with `kind` (see the module docstring) and `crowd` columns."""
    table = pd.read_csv(path)
    crowd = table["attrs"].map(
        lambda a: int(json.loads(a).get("iscrowd", 0)) if isinstance(a, str) else 0
    )
    table["crowd"] = crowd
    table["kind"] = np.select(
        [
            table.category == 0,
            (table.category == 1) & (crowd == 1),
            table.category == 1,
            table.category == 2,
        ],
        ["absent", "crowd", "individual", "uncertain"],
        default="not_observable",
    )
    return table.sort_values(["shot", "t_start"]).reset_index(drop=True)


@dataclass(frozen=True)
class Dense:
    """One shot's targets on the 1 ms grid starting at `inputs.GRID0_MS`."""

    state: np.ndarray  # int8 (n_ms,): 1 present, 0 absent, IGNORE
    onset: np.ndarray  # float32 (n_ms,): Gaussian at individual starts, in [0, 1]
    onset_mask: np.ndarray  # bool (n_ms,): where the onset target is defined
    window: tuple[float, float]  # first start, last end of the shot's spans (ms)


def dense(spans: pd.DataFrame, n_ms: int) -> Dense:
    """Dense targets of one shot's review spans (columns `t_start`, `t_end`, `kind`)."""
    state = np.full(n_ms, IGNORE, dtype=np.int8)
    onset_mask = np.zeros(n_ms, dtype=bool)
    crowd = np.zeros(n_ms, dtype=bool)
    onset = np.zeros(n_ms, dtype=np.float32)
    g0 = inputs.GRID0_MS
    centres = g0 + np.arange(n_ms) + 0.5
    for r in spans.itertuples():
        a = int(np.clip(np.ceil(r.t_start - g0 - 1e-9), 0, n_ms))
        b = int(np.clip(np.floor(r.t_end - g0 + 1e-9), 0, n_ms))
        if b <= a:
            continue
        if r.kind == "absent":
            state[a:b] = 0
            onset_mask[a:b] = True
        elif r.kind in ("individual", "crowd"):
            state[a:b] = 1
            onset_mask[a:b] = r.kind == "individual"
            crowd[a:b] = r.kind == "crowd"
    # the onset of an individual ELM: a Gaussian at the span's start, defined
    # over the span and the ONSET_SIGMA_MS tails beside it
    for r in spans.itertuples():
        if r.kind != "individual":
            continue
        lo = max(0, int(np.floor(r.t_start - g0 - 5 * ONSET_SIGMA_MS)))
        hi = min(n_ms, int(np.ceil(r.t_start - g0 + 5 * ONSET_SIGMA_MS)))
        if hi <= lo:
            continue
        bump = np.exp(-0.5 * ((centres[lo:hi] - r.t_start) / ONSET_SIGMA_MS) ** 2)
        onset[lo:hi] = np.maximum(onset[lo:hi], bump.astype(np.float32))
        onset_mask[lo:hi] = True
    # a crowd's ELMs are unlabelled, not absent: no onset is defined inside one
    onset_mask &= ~crowd
    onset[~onset_mask] = 0.0
    window = (float(spans.t_start.min()), float(spans.t_end.max()))
    return Dense(state, onset, onset_mask, window)


@dataclass(frozen=True)
class Bins:
    """The scored 50 ms bins of one shot: left edges, truth, and where they came from."""

    t0: np.ndarray  # float (m,) left edge, ms
    truth: np.ndarray  # int8 (m,) 1 present, 0 absent
    kind: np.ndarray  # object (m,) absent / individual / crowd
    span: np.ndarray  # int (m,) index of the span in the shot's table rows


def merge_intervals(starts, stops, tol: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Analysed time as separate intervals: those that touch (within `tol` ms) join."""
    out0: list[float] = []
    out1: list[float] = []
    for a, b in sorted(zip(map(float, starts), map(float, stops))):
        if out1 and a <= out1[-1] + tol:
            out1[-1] = max(out1[-1], b)
        else:
            out0.append(a)
            out1.append(b)
    return np.array(out0), np.array(out1)


def scored_bins(spans: pd.DataFrame, cov0: np.ndarray, cov1: np.ndarray) -> Bins:
    """One shot's scored bins given its analysed intervals `[cov0, cov1)` in ms.

    `spans` are the shot's review rows in time order. A span with less than half
    its length analysed has no bins; the others contribute every 50 ms bin that
    lies wholly inside both the span and one analysed interval.
    """
    t0: list[np.ndarray] = []
    truth: list[np.ndarray] = []
    kind: list[np.ndarray] = []
    index: list[np.ndarray] = []
    cov0, cov1 = np.asarray(cov0, dtype=float), np.asarray(cov1, dtype=float)
    for i, row in enumerate(spans.itertuples()):
        if row.kind not in SCORED_KINDS or row.t_end <= row.t_start:
            continue
        length = row.t_end - row.t_start
        analysed = np.clip(
            np.minimum(cov1, row.t_end) - np.maximum(cov0, row.t_start), 0, None
        ).sum()
        if analysed < 0.5 * length:
            continue
        edges = (
            np.arange(
                int(np.ceil(row.t_start / BIN_MS)), int(np.floor(row.t_end / BIN_MS))
            )
            * BIN_MS
        )
        k = np.searchsorted(cov0, edges, side="right") - 1
        inside = (k >= 0) & (cov1[np.maximum(k, 0)] >= edges + BIN_MS)
        edges = edges[inside]
        t0.append(edges)
        truth.append(np.full(edges.size, int(row.kind != "absent"), dtype=np.int8))
        kind.append(np.full(edges.size, row.kind, dtype=object))
        index.append(np.full(edges.size, i, dtype=np.int64))
    if not t0:
        return Bins(
            np.zeros(0), np.zeros(0, np.int8), np.zeros(0, object), np.zeros(0, int)
        )
    return Bins(
        np.concatenate(t0),
        np.concatenate(truth),
        np.concatenate(kind),
        np.concatenate(index),
    )


def bin_scores(fine: np.ndarray, bins: Bins) -> np.ndarray:
    """The mean of a 1 ms trace `fine` (from `inputs.GRID0_MS`) over each bin."""
    first = (bins.t0 - inputs.GRID0_MS).astype(np.int64)
    width = int(BIN_MS)
    if bins.t0.size and (first.min() < 0 or first.max() + width > fine.size):
        raise ValueError("a scored bin lies outside the trace")
    csum = np.r_[0.0, np.cumsum(np.asarray(fine, dtype=np.float64))]
    return (csum[first + width] - csum[first]) / width


def hard_hits(starts: np.ndarray, stops: np.ndarray, bins: Bins) -> np.ndarray:
    """Whether any span `[starts, stops)` (ms, sorted, not overlapping) touches each bin."""
    if not len(starts):
        return np.zeros(bins.t0.size, dtype=bool)
    starts, stops = np.asarray(starts, float), np.asarray(stops, float)
    i = np.minimum(np.searchsorted(stops, bins.t0, side="right"), len(starts) - 1)
    return (stops[i] > bins.t0) & (starts[i] < bins.t0 + BIN_MS)


def runs_of(
    fine: np.ndarray, threshold: float, width_ms: int = int(BIN_MS)
) -> tuple[np.ndarray, np.ndarray]:
    """Where the `width_ms` moving mean of a 1 ms trace is at or above `threshold`.

    The scored bins' score is this mean at the bins' own offsets; here it is taken
    at every millisecond, so a detected span is the stretch of shot time the same
    score and threshold call present. Returns `(starts, stops)` in ms of the shot's
    clock (`inputs.GRID0_MS` is the trace's first edge).
    """
    smooth = uniform_filter1d(
        np.asarray(fine, dtype=np.float64), width_ms, mode="nearest"
    )
    on = np.r_[False, smooth >= threshold, False].astype(np.int8)
    edge = np.flatnonzero(np.diff(on))
    return inputs.GRID0_MS + edge[0::2].astype(float), inputs.GRID0_MS + edge[
        1::2
    ].astype(float)
