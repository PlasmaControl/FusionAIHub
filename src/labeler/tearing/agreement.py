"""Agreement of the whole-interval labels with the lab's onset labels.

A reference (Seo's archive, the survival labels) says, per shot it covers, when a
tearing mode began, or that none did. The interval label should hold every such onset
inside or at the start of one of its intervals, and each interval it draws should have
a reference onset in it or at its start. `compare_onsets` counts both and measures how
far apart the two onsets are; `summarize` reduces them to the numbers the documentation
quotes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from . import rule as tearing_rule

#: How far from an interval's edges a reference onset still counts as at its start
#: (ms). The archives sample every 20 to 25 ms.
TOLERANCE_MS = 100.0
#: Onset-time errors the summary reports the share within (ms).
WITHIN_MS = (25.0, 50.0, 100.0, 250.0)


@dataclass(frozen=True)
class Reference:
    """One shot's reference: its onset (None: no mode) and the time it covers (ms)."""

    shot: int
    onset_ms: float | None
    lo_ms: float
    hi_ms: float


def compare_onsets(
    references,
    intervals: pd.DataFrame,
    *,
    tol_ms: float = TOLERANCE_MS,
    ns: tuple[int, ...] | None = (1,),
):
    """`(onsets, compared)`: each reference onset and each interval, with their match.

    `intervals` has `shot`, `n`, `t_start`, `t_end`; only those of toroidal number `ns`
    (every one for None) are used. An onset is matched to the interval that holds it
    within `tol_ms` of its edges, the one whose start is nearest if several do;
    `error_ms` is the reference onset less that start (positive: the interval began
    first). A compared interval is one that starts inside a reference's coverage, and
    `has_onset` says whether any reference onset of its shot lies in it or at its start.
    When `intervals` has `onset_window_start_ms` (the start of the same-n weak track
    that led into the interval), `window_start_ms` is that time for the matched
    interval and `in_onset_window` says whether the reference onset lies between it
    and the interval's start, each widened by `tol_ms` (`in_onset_window_strict` is the
    same without the widening): the labelled onset is a point
    (the interval's start, where the RMS crossed a tenth of the peak), its window the
    span in which the mode could have begun. A reference onset later than the interval
    start (plus `tol_ms`) is matched but not inside the window.
    """
    if ns is not None:
        intervals = intervals[intervals.n.isin(ns)]
    by_shot = {s: g for s, g in intervals.groupby("shot")}
    onsets, compared = [], []
    for ref in references:
        mine = by_shot.get(ref.shot)
        starts = np.empty(0) if mine is None else mine.t_start.to_numpy(float)
        ends = np.empty(0) if mine is None else mine.t_end.to_numpy(float)
        if mine is not None and "onset_window_start_ms" in mine:
            windows = mine.onset_window_start_ms.to_numpy(float)
            windows = np.where(np.isfinite(windows), windows, starts)
        else:
            windows = starts
        if ref.onset_ms is not None:
            t = float(ref.onset_ms)
            inside = (starts - tol_ms <= t) & (t <= ends + tol_ms)
            hit = bool(inside.any())
            k = (
                int(np.argmin(np.where(inside, np.abs(t - starts), np.inf)))
                if hit
                else 0
            )
            onsets.append(
                {
                    "shot": ref.shot,
                    "onset_ms": t,
                    "matched": hit,
                    "error_ms": t - float(starts[k]) if hit else np.nan,
                    "start_ms": float(starts[k]) if hit else np.nan,
                    "end_ms": float(ends[k]) if hit else np.nan,
                    "window_start_ms": float(windows[k]) if hit else np.nan,
                    "in_onset_window": bool(
                        hit and windows[k] - tol_ms <= t <= starts[k] + tol_ms
                    ),
                    "in_onset_window_strict": bool(
                        hit and windows[k] <= t <= starts[k]
                    ),
                    "n_intervals": len(starts),
                }
            )
        for i in range(len(starts)):
            if not (ref.lo_ms - tol_ms <= starts[i] <= ref.hi_ms + tol_ms):
                continue
            has = ref.onset_ms is not None and bool(
                starts[i] - tol_ms <= ref.onset_ms <= ends[i] + tol_ms
            )
            compared.append(
                {
                    "shot": ref.shot,
                    "t_start": float(starts[i]),
                    "t_end": float(ends[i]),
                    "has_onset": has,
                    "reference_has_mode": ref.onset_ms is not None,
                }
            )
    return pd.DataFrame(onsets), pd.DataFrame(compared)


def summarize(onsets: pd.DataFrame, compared: pd.DataFrame, references) -> dict:
    """The agreement numbers: onsets matched and missed, the error, stray intervals."""
    out: dict = {"n_shots_covered": len(references)}
    n = len(onsets)
    matched = onsets[onsets.matched] if n else onsets
    out["reference_onsets"] = n
    out["matched"] = len(matched)
    out["missed"] = int(n - len(matched))
    out["matched_fraction"] = float(len(matched) / n) if n else None
    if len(matched):
        err = matched.error_ms.to_numpy(float)
        out["error_ms"] = {
            "median": float(np.median(err)),
            "q25": float(np.percentile(err, 25)),
            "q75": float(np.percentile(err, 75)),
            "min": float(err.min()),
            "max": float(err.max()),
            "mean_abs": float(np.abs(err).mean()),
            "within_ms": {
                f"{int(w)}": float((np.abs(err) <= w).mean()) for w in WITHIN_MS
            },
            "reference_after_interval_start_fraction": float((err > 0).mean()),
            "reference_after_interval_start_by_more_than_tolerance": int(
                (err > TOLERANCE_MS).sum()
            ),
            "reference_inside_onset_window": int(matched.in_onset_window.sum()),
            "reference_inside_onset_window_strict": int(
                matched.in_onset_window_strict.sum()
            ),
            "reference_inside_onset_window_fraction": float(
                matched.in_onset_window.mean()
            ),
        }
    out["compared_intervals"] = len(compared)
    if len(compared):
        without = compared[~compared.has_onset]
        out["intervals_with_an_onset"] = int(compared.has_onset.sum())
        out["intervals_without_an_onset"] = len(without)
        out["intervals_without_an_onset_fraction"] = float(len(without) / len(compared))
        out["intervals_without_an_onset_on_reference_mode_shots"] = int(
            without.reference_has_mode.sum()
        )
        out["intervals_without_an_onset_on_reference_quiet_shots"] = int(
            (~without.reference_has_mode).sum()
        )
    has_ref = {r.shot for r in references if r.onset_ms is not None}
    has_ours = set(compared.shot) if len(compared) else set()
    covered = {r.shot for r in references}
    out["shots"] = {
        "reference_mode_and_ours": len(has_ref & has_ours),
        "reference_mode_not_ours": len(has_ref - has_ours),
        "ours_not_reference": len((has_ours & covered) - has_ref),
        "neither": len(covered - has_ref - has_ours),
    }
    return out


#: Why a reference onset has no interval (see `miss_reason`).
REASONS = (
    "outside_window",
    "interval_starts_later",
    "short_burst",
    "below_onset_level",
)


def miss_reason(
    t_ms,
    rms_g,
    onset_ms: float,
    window,
    starts_ms,
    rule: tearing_rule.ModeRule = tearing_rule.N1_RULE,
    *,
    tol_ms: float = TOLERANCE_MS,
    look_ms: float = 300.0,
) -> str:
    """Why the rule drew no interval at a reference onset, one of `REASONS`.

    `outside_window`: the onset is outside the catalog window the rule reads.
    `interval_starts_later`: one of our intervals (`starts_ms`) begins within `look_ms`
    after the onset, more than `tol_ms` late. `short_burst`: the RMS crosses the onset
    level within `tol_ms` before to `look_ms` after the onset, but not for the rule's
    hold (found by running the rule with no hold and no duty test). `below_onset_level`:
    it never crosses the level there.
    """
    t = float(onset_ms)
    if window is not None and not (window[0] <= t <= window[1]):
        return "outside_window"
    starts = np.asarray(starts_ms, dtype=float)
    if ((starts > t + tol_ms) & (starts <= t + look_ms)).any():
        return "interval_starts_later"
    loose = replace(rule, hold_ms=0.0, min_duty=0.0, harmonic_ratio=None)
    near = [
        item
        for item in tearing_rule.mode_intervals(t_ms, rms_g, loose, window)
        if item.end_ms >= t - tol_ms and item.start_ms <= t + look_ms
    ]
    return "short_burst" if near else "below_onset_level"
