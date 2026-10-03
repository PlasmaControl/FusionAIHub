"""Hysteresis alarms and the paper's per-shot scoring.

Piccione et al. 2022 turn the forest's warning level into an alarm with a hysteresis
rule (Montes et al. 2019): the alarm fires once the level has crossed a high threshold
`k_high` and then stayed above a low threshold `k_low` for `hold_ms`. An alarm more
than 400 ms before the RWM time is too early and one less than 10 ms before it is
too late to count; here the same limits decide which alarms belong to an onset.
"""

from __future__ import annotations

import numpy as np

#: An alarm earlier than this before an onset is not a warning of it.
MAX_WARNING_MS = 400.0
#: An alarm later than this before an onset is too late to count.
MIN_WARNING_MS = 10.0


def hysteresis_alarms(t_ms, score, k_low, k_high, hold_ms, *, max_gap_ms=None):
    """Alarm times of a warning trace; the alarm latches until the level drops below `k_low`.

    A missing slice (NaN score) or a time gap above `max_gap_ms` resets the rule, so an
    alarm never spans a hole in the record.
    """
    t, s = np.asarray(t_ms, dtype=float), np.asarray(score, dtype=float)
    alarms, crossed, latched, previous = [], None, False, None
    for time, level in zip(t, s):
        gap = (
            previous is not None
            and max_gap_ms is not None
            and time - previous > max_gap_ms
        )
        previous = time
        if gap or not np.isfinite(level) or level < k_low:
            crossed, latched = None, False
            continue
        if latched:
            continue
        if crossed is None and level >= k_high:
            crossed = time
        if crossed is not None and time - crossed >= hold_ms:
            alarms.append(float(time))
            latched = True
    return alarms


def match_alarms(alarms, onsets, *, post_ms=100.0):
    """Split a shot's alarms among its onsets, as `(per_onset, unmatched)`.

    `per_onset` has one entry per onset: the warning time in ms of the earliest alarm
    in `[onset - MAX_WARNING_MS, onset - MIN_WARNING_MS]`, or None if there was none.
    An alarm is unmatched, a false alarm, unless it lies in some onset's warning range
    or in `(onset - MIN_WARNING_MS, onset + post_ms]`, where the mode is already acting.
    """
    alarms = sorted(float(a) for a in alarms)
    per_onset = []
    for onset in onsets:
        hits = [
            a for a in alarms if onset - MAX_WARNING_MS <= a <= onset - MIN_WARNING_MS
        ]
        per_onset.append(onset - hits[0] if hits else None)
    unmatched = []
    for alarm in alarms:
        explained = any(
            onset - MAX_WARNING_MS <= alarm <= onset + post_ms for onset in onsets
        )
        if not explained:
            unmatched.append(alarm)
    return per_onset, unmatched
