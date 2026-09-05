"""Per-shot scoring ported from survival_tm/metrics_helpers.py, in seconds.

Two upstream quirks are corrected: thresholds are inclusive (risk >= threshold,
not get_classification's strict >), and reverted runs use the absolute next-zero
index, not its window-relative index. Warning times are onset minus row time,
including always-on traces (upstream receives time-to-event rather than time).
Jumps are reverted 0 -> 1 runs lasting at least 0.4 s, not shorter excursions.
IPCW ranking uses sorting/searchsorted: O(n log n) time and O(n) memory.
"""
from dataclasses import dataclass

import numpy as np

JUMP_MIN_DURATION_S = 0.4


@dataclass(frozen=True)
class ShotAlarm:
    verdict: str
    alarm: bool
    warning_time_s: float | None
    jumps: int
    n_excursions: int
    n_rows: int


def shot_alarm(t, risk, valid, *, threshold, onset_s) -> ShotAlarm:
    """Score finite valid rows in time order; empty traces cannot be scored."""
    t, risk = np.asarray(t, float), np.asarray(risk, float)
    keep = np.asarray(valid, bool) & np.isfinite(t) & np.isfinite(risk)
    t, risk = t[keep], risk[keep]
    order = np.argsort(t, kind='stable')
    t, label = t[order], risk[order] >= threshold
    if not t.size:
        raise ValueError('no valid finite rows')
    called = bool(label[-1])
    event = onset_s is not None and np.isfinite(onset_s)
    verdict = ('TP' if called else 'FN') if event else ('FP' if called else 'TN')
    starts = np.flatnonzero(~label[:-1] & label[1:]) + 1
    ends = np.flatnonzero(label[:-1] & ~label[1:]) + 1
    reverted = starts[starts < (ends[-1] if ends.size else -1)]
    stops = ends[np.searchsorted(ends, reverted)]
    jumps = int(np.count_nonzero(t[stops] - t[reverted] >= JUMP_MIN_DURATION_S))
    warning = None
    if verdict == 'TP':
        off = np.flatnonzero(~label)
        start = int(off[-1] + 1) if off.size else 0
        warning = float(onset_s - t[start])
    return ShotAlarm(verdict, called, warning, jumps, int(reverted.size), int(t.size))


def any_row_call(risk, valid, *, threshold) -> bool:
    risk = np.asarray(risk, float)
    return bool(np.any(np.asarray(valid, bool) & np.isfinite(risk) & (risk >= threshold)))


def pool_rates(verdicts) -> dict:
    verdicts = list(verdicts)
    counts = {key: verdicts.count(key) for key in ('TP', 'FN', 'TN', 'FP')}
    quiet, tearing = counts['FP'] + counts['TN'], counts['FN'] + counts['TP']
    return {'fpr': counts['FP'] / quiet if quiet else None,
            'fnr': counts['FN'] / tearing if tearing else None, 'counts': counts}


def horizon_integral(horizons_s, values) -> float:
    h, v = np.asarray(horizons_s, float), np.asarray(values, float)
    return float(np.sum(np.diff(h) * (v[:-1] + v[1:]) / 2))


def km_censoring(T, e):
    """Right-continuous reverse KM; events precede censorings at tied times.

    `e` indicates the event, not censoring. G is one before the first time.
    """
    t, event = np.asarray(T, float), np.asarray(e, bool)
    times, inverse, counts = np.unique(t, return_inverse=True, return_counts=True)
    events = np.bincount(inverse, weights=event, minlength=times.size)
    censored = counts - events
    at_risk = t.size - np.r_[0, np.cumsum(counts)[:-1]] - events
    factors = np.ones(times.size)
    np.divide(censored, at_risk, out=factors, where=at_risk > 0)
    factors[at_risk == 0] = 0
    survival = np.r_[1., np.cumprod(1 - factors)]

    def evaluate(time):
        return survival[np.searchsorted(times, time, side='right')]

    return evaluate


def ipcw_auc(T, e, score, horizon_s) -> float | None:
    """Cumulative/dynamic AUC; undefined populations or zero G return None.

    Cases are observed events at T <= h, controls have T > h. Controls all
    carry 1/G(h), which cancels in the weighted fraction. Early censored
    rows estimate G but are neither cases nor controls. Score ties get half.
    """
    t, event, score = np.asarray(T, float), np.asarray(e, bool), np.asarray(score, float)
    good = np.isfinite(t) & np.isfinite(score) & (t >= 0)
    t, event, score = t[good], event[good], score[good]
    cases, controls = event & (t <= horizon_s), t > horizon_s
    if not cases.any() or not controls.any():
        return None
    g = km_censoring(t, event)
    gc = g(t[cases])
    if g(horizon_s) <= 0 or np.any(gc <= 0):
        return None
    control = np.sort(score[controls])
    lower = np.searchsorted(control, score[cases], side='left')
    upper = np.searchsorted(control, score[cases], side='right')
    weights = 1 / gc
    return float(np.sum(weights * (lower + upper) / 2) / (weights.sum() * control.size))
