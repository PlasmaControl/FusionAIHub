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
slowest input, the TangTV inversion, arrives every 16.7 ms on disk (Chen 2026's
camera records 60 Hz interlaced fields as 30 Hz full frames, and the inversions are
60 Hz; the corpus's resampled raw frames are 20 ms apart) and EFIT every 20 ms, so
a 50 ms bin holds about 3 inversions (2.5 raw frames; Chen's camera integrates
ELMs, while current and radiation use narrow ELM masks), and
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
    """Nearest D-alpha ELM flag; unknown times are conservatively masked.

    Consumers also use `elm_bin_known` to distinguish an ELM from unknown
    coverage, rather than labelling missing D-alpha as verified clean time.
    """
    t_ms = np.asarray(t_ms, dtype=float)
    if elm_t_ms is None or elm_flag is None or len(elm_t_ms) < 2:
        return np.ones(len(t_ms), dtype=bool)
    elm_t_ms = np.asarray(elm_t_ms, dtype=float)
    if len(elm_flag) != len(elm_t_ms):
        return np.ones(len(t_ms), dtype=bool)
    after = np.clip(np.searchsorted(elm_t_ms, t_ms), 0, len(elm_t_ms) - 1)
    before = np.clip(after - 1, 0, len(elm_t_ms) - 1)
    nearer_before = np.abs(elm_t_ms[before] - t_ms) <= np.abs(elm_t_ms[after] - t_ms)
    index = np.where(nearer_before, before, after)
    known = np.isfinite(t_ms) & (t_ms >= elm_t_ms[0]) & (t_ms <= elm_t_ms[-1])
    flags = np.asarray(elm_flag, dtype=float)
    known &= np.isfinite(flags[index])
    step = float(np.median(np.diff(elm_t_ms)))
    known &= np.abs(elm_t_ms[index] - t_ms) <= max(step, 1.0)
    return ~known | (flags[index] > 0)


def sample_windows_known(
    t_ms, available, starts, stops, *, closed_right=False
) -> np.ndarray:
    """Complete sample availability in half-open windows, without extrapolation.

    Native bins are half-open; centered means use closed_right=True to include
    the upper endpoint in both availability and the reduction.
    Half a native sample is allowed at record boundaries. A missing sample or
    an internal gap longer than two native samples invalidates intersecting
    windows. Availability is independent of whether a measured value is zero.
    """
    t = np.asarray(t_ms, dtype=float)
    available = np.asarray(available, dtype=bool)
    starts, stops = np.asarray(starts), np.asarray(stops)
    known = np.zeros(starts.shape, bool)
    if (
        len(t) < 2
        or len(available) != len(t)
        or not np.isfinite(t).all()
        or np.any(np.diff(t) <= 0)
    ):
        return known
    step = float(np.median(np.diff(t)))
    known = (starts >= t[0] - step / 2) & (stops <= t[-1] + step / 2)
    lo = np.searchsorted(t, starts, side="left")
    hi = np.searchsorted(t, stops, side="right" if closed_right else "left")
    missing = np.r_[0, np.cumsum(~available)]
    known &= (hi > lo) & (missing[hi] == missing[lo])
    for i in np.flatnonzero(np.diff(t) > max(2 * step, 2.0)):
        known &= ~((starts < t[i + 1]) & (stops > t[i]))
    return known


def elm_bin_known(edges, elm_t_ms, elm_flag) -> np.ndarray:
    """Complete D-alpha coverage of each bin, including internal record gaps.

    Half a native sample at each boundary is allowed. Gaps longer than two
    samples (at least 2 ms) make every bin they intersect unknown.
    """
    n = len(edges) - 1
    if elm_t_ms is None or elm_flag is None:
        return np.zeros(n, bool)
    return sample_windows_known(elm_t_ms, np.isfinite(elm_flag), edges[:-1], edges[1:])


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
