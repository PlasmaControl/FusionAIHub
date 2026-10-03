"""The shared vocabulary of the detachment label: states, votes, one bin grid.

Three indicators (Afrac, Prad,div and the TangTV front height) are each turned
into the same four arrays on one time grid - a value, a validity mask, the reason
a bin is invalid, and a vote - so a label model can treat them as interchangeable
labelling functions. This module holds only what all of them share.

**States** are coded as in the sibling review stream: 0 absent, 1 attached,
2 detached, 3 marfe, 4 uncertain. A CSV row never carries 0 (time with no row was
not assessed). **Votes** are 1/2/3 or `ABSTAIN` = -1; an indicator abstains where it
is invalid and where its value sits between two thresholds (a transition band is
not a state).

**The bin grid** is `BIN_MS` wide. 50 ms was chosen over 10 or 20 ms because it is
the catalog's own label grid (`events.yaml`: `sample_interval_ms: 50`), because the
slowest input, the TangTV inversion, arrives every 17-33 ms and EFIT every 20 ms
(30 Hz full camera frames give about 1.5 independent frames per bin; the corpus
50 fps grid is resampled, so explicit widened ELM masks are essential), and
because the published low-pass constants (Eldon 2022: 10-50 ms; Chen 2026: Prad
leads DZ by about 50 ms) are of that order, so a finer grid would resolve nothing
the indicators can see.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ABSENT, ATTACHED, DETACHED, MARFE, UNCERTAIN = 0, 1, 2, 3, 4
ABSTAIN = -1
STATE_NAMES = {
    ABSENT: "absent",
    ATTACHED: "attached",
    DETACHED: "detached",
    MARFE: "marfe",
    UNCERTAIN: "uncertain",
}
VOTE_STATES = (ATTACHED, DETACHED, MARFE)

#: Width of one label bin in milliseconds. Chosen in the module docstring.
BIN_MS = 50.0


@dataclass(frozen=True)
class Indicator:
    """One indicator on the bin grid; all arrays have the bin grid's length."""

    name: str
    #: Value of the physical quantity the votes are cast on (NaN where unknown).
    value: np.ndarray
    #: False where the indicator may not speak; `reason` says why.
    valid: np.ndarray
    reason: np.ndarray
    #: ABSTAIN, ATTACHED, DETACHED or MARFE; always ABSTAIN where not valid.
    vote: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.value)
        for field in ("valid", "reason", "vote"):
            if len(getattr(self, field)) != n:
                raise ValueError(f"{self.name}: {field} length differs from value")
        bad = self.vote[~self.valid]
        if np.any(bad != ABSTAIN):
            raise ValueError(f"{self.name}: a vote on an invalid bin")


def bin_edges(t0_ms: float, t1_ms: float, width_ms: float = BIN_MS) -> np.ndarray:
    """Edges of the bins covering [t0, t1], aligned to multiples of the width."""
    start = np.floor(t0_ms / width_ms) * width_ms
    stop = np.ceil(t1_ms / width_ms) * width_ms
    return np.arange(start, stop + width_ms * 0.5, width_ms)


def bin_centres(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])


def bin_median(
    t_ms: np.ndarray,
    y: np.ndarray,
    edges: np.ndarray,
    *,
    keep: np.ndarray | None = None,
    min_count: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Median of `y` per bin over the samples with `keep`, and the sample count.

    A bin with fewer than `min_count` kept finite samples is NaN. The median, not
    the mean, because an ELM or a strike-point flash puts a spike on a minority of
    a bin's samples and the median ignores a minority.
    """
    t_ms = np.asarray(t_ms, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y) & np.isfinite(t_ms)
    if keep is not None:
        ok &= np.asarray(keep, dtype=bool)
    n_bins = len(edges) - 1
    out = np.full(n_bins, np.nan)
    count = np.zeros(n_bins, dtype=int)
    index = np.searchsorted(edges, t_ms[ok], side="right") - 1
    values = y[ok]
    inside = (index >= 0) & (index < n_bins)
    index, values = index[inside], values[inside]
    if index.size == 0:
        return out, count
    order = np.argsort(index, kind="stable")
    index, values = index[order], values[order]
    starts = np.flatnonzero(np.r_[True, index[1:] != index[:-1]])
    stops = np.r_[starts[1:], index.size]
    for s, e in zip(starts, stops, strict=True):
        count[index[s]] = e - s
        if e - s >= min_count:
            out[index[s]] = np.median(values[s:e])
    return out, count


def bin_mean(
    t_ms: np.ndarray,
    y: np.ndarray,
    edges: np.ndarray,
    *,
    keep: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean of `y` per bin over the samples with `keep`, and the sample count.

    For powers: a modulated neutral beam (190109 blips at 50 % duty) has a median
    of zero in a bin whose mean is the delivered power, and it is the energy that
    heats the plasma and feeds the radiation. A bin without a kept sample is NaN.
    """
    t_ms = np.asarray(t_ms, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y) & np.isfinite(t_ms)
    if keep is not None:
        ok &= np.asarray(keep, dtype=bool)
    n_bins = len(edges) - 1
    index = np.searchsorted(edges, t_ms[ok], side="right") - 1
    values = y[ok]
    inside = (index >= 0) & (index < n_bins)
    count = np.bincount(index[inside], minlength=n_bins)
    total = np.bincount(index[inside], weights=values[inside], minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan), count


def elm_at(t_ms: np.ndarray, elm_t_ms, elm_flag) -> np.ndarray:
    """ELM flag of the nearest D-alpha sample for every time in `t_ms`."""
    t_ms = np.asarray(t_ms, dtype=float)
    if elm_t_ms is None:
        return np.zeros(len(t_ms), dtype=bool)
    elm_t_ms = np.asarray(elm_t_ms, dtype=float)
    after = np.clip(np.searchsorted(elm_t_ms, t_ms), 0, len(elm_t_ms) - 1)
    before = np.clip(after - 1, 0, len(elm_t_ms) - 1)
    nearer_before = np.abs(elm_t_ms[before] - t_ms) <= np.abs(elm_t_ms[after] - t_ms)
    index = np.where(nearer_before, before, after)
    return np.asarray(elm_flag, dtype=bool)[index]


def bin_fraction(t_ms: np.ndarray, flag: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Fraction of samples per bin where `flag` is true (NaN for an empty bin)."""
    t_ms = np.asarray(t_ms, dtype=float)
    flag = np.asarray(flag, dtype=float)
    n_bins = len(edges) - 1
    index = np.searchsorted(edges, t_ms, side="right") - 1
    inside = (index >= 0) & (index < n_bins)
    total = np.bincount(index[inside], minlength=n_bins).astype(float)
    hit = np.bincount(index[inside], weights=flag[inside], minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, hit / total, np.nan)


def assemble(
    name: str,
    value: np.ndarray,
    valid: np.ndarray,
    reason: np.ndarray,
    vote: np.ndarray,
) -> Indicator:
    """Build an `Indicator`, forcing ABSTAIN wherever the bin is invalid."""
    value = np.asarray(value, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    reason = np.where(valid, "", np.asarray(reason, dtype=object)).astype(object)
    vote = np.where(valid, np.asarray(vote), ABSTAIN).astype(np.int8)
    return Indicator(name, value, valid, reason, vote)
