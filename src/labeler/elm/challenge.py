"""The DIII-D ELM detection challenge as a held-out benchmark for the ELM methods.

The challenge (D. Eldon, https://github.com/DIII-D/elm_detection_challenge) holds a
hand-timed, hand-classified ELM answer key and a merit function. Nothing of theirs is
copied here: the key is vendored unchanged as the ELM label source `elm_challenge_key`
(`labeler.events.source_formatters`), and their merit function runs from a pinned clone
outside the repository (`scripts/labeler/elm_challenge_merit.py`). This module only
turns a method's output into their submission format and adds the views of our own:

* `submission` writes detected spans as the challenge's JSON (`device`, `shot`, `time`,
  `event_flag`, `event_type`). Methods that do not classify submit the detection-only
  type `DETECTION_ONLY_TYPE`, as the challenge's samples do;
* `clock_spans` reads the ELM clock's peaks off one filterscope and widens each to the
  peak's half-prominence width, the only span the clock has;
* `onset_match` and `bin_metrics` score the same spans on the key by our boundary
  matching and by 50 ms occupancy bins.

The key is public: no method is trained or tuned on it, and the shots lie outside the
119 reviewed ones.
"""

from __future__ import annotations

import numpy as np

from ..events import raw, spans, transients
from ..events.source_formatters import ELM_CHALLENGE_STATES
from ..events.verify import NoDataError
from ..scoring import events as boundary

DEVICE = "DIII-D"
#: The detection-only event type of a method that does not classify (the challenge's
#: `type_no_classification`; the samples carry it on every event).
DETECTION_ONLY_TYPE = 5
#: Gap between a flag's last 1 and first 0 (and the reverse), ms. The merit function
#: finds an event's edges at the knots where the flag changes.
EDGE_MS = 1e-3
#: The filterscope rows the challenge's `signal_list.yml` allows: FS01-FS06 are rows
#: 0-5 of the corpus group (FS07 and FS08 are not on the list).
ALLOWED_FILTERSCOPE_ROWS = frozenset(range(6))
#: The 10 kHz grid the corpus's filterscopes sit on, which the clock was built for.
CLOCK_STEP_MS = 0.1
BIN_MS = 50.0
#: Key types whose onsets are no ELM's (uncertain: "other" and unspecified events).
UNCERTAIN_TYPES = frozenset(k for k, v in ELM_CHALLENGE_STATES.items() if v == 2)
ELM_TYPES = frozenset(k for k, v in ELM_CHALLENGE_STATES.items() if v == 1)


def separate(spans_ms) -> np.ndarray:
    """`(n, 2)` sorted spans, each ended at least `4 * EDGE_MS` before the next starts,
    so the flag returns to 0 between two events."""
    a = np.asarray(spans_ms, dtype=float).reshape(-1, 2)
    if not np.isfinite(a).all() or (a[:, 1] < a[:, 0]).any():
        raise ValueError("spans must be finite with end >= start")
    a = a[np.argsort(a[:, 0], kind="stable")]
    for i in range(len(a) - 1):
        a[i, 1] = min(a[i, 1], a[i + 1, 0] - 4 * EDGE_MS)
    return a[a[:, 1] > a[:, 0]]


def submission(
    shot: int, spans_ms, window_ms, *, method: str, filterscopes_used=(), comment=""
) -> dict:
    """One shot's detected `spans_ms` (ms) as a challenge submission record.

    The flag is 1 inside each span and 0 elsewhere; the record is bracketed by 0 at the
    case's window `window_ms` (`tstart`, `tend`) so the merit function's interpolation
    has a defined flag across it, and a shot with no span is all zeros.
    """
    a = separate(spans_ms)
    t, f, ty = [], [], []
    for start, end in a:
        t += [start - EDGE_MS, start, end, end + EDGE_MS]
        f += [0, 1, 1, 0]
        ty += [DETECTION_ONLY_TYPE] * 4
    lo, hi = float(window_ms[0]), float(window_ms[1])
    if len(t):
        lo, hi = min(lo, t[0] - 1.0), max(hi, t[-1] + 1.0)
    return {
        "comment": comment,
        "device": DEVICE,
        "shot": int(shot),
        "filterscopes_used": list(filterscopes_used),
        "detection_method": method,
        "time": [lo, *t, hi],
        "event_flag": [0, *f, 0],
        "event_type": [0, *ty, 0],
    }


def flag_spans(record: dict) -> np.ndarray:
    """The spans (ms) where a submission's flag is 1, `(n, 2)`: a rising change opens a
    span and a falling one closes it, as the merit function finds events."""
    t = np.asarray(record["time"], dtype=float)
    f = np.asarray(record["event_flag"])
    step = np.diff(f)
    starts = t[1:][step == 1]
    ends = t[1:][step == -1]
    n = min(len(starts), len(ends))
    return np.column_stack([starts[:n], ends[:n]]) if n else np.zeros((0, 2))


def resample(t_ms, y, step_ms: float = CLOCK_STEP_MS):
    """One trace on a `step_ms` grid by linear interpolation over its finite samples;
    returned unchanged if it is already no finer than that grid.

    A stretch the record has no finite samples for (more than two of its own steps
    between neighbours) and the ends beyond its first and last finite sample stay NaN:
    the clock does not smooth across a gap.
    """
    t = np.asarray(t_ms, dtype=float)
    y = np.asarray(y, dtype=float)
    own = float(np.median(np.diff(t)))
    if own >= 0.9 * step_ms:
        return t, y
    grid = np.arange(np.ceil(t[0] / step_ms), np.floor(t[-1] / step_ms) + 1) * step_ms
    ok = np.isfinite(y)
    tf = t[ok]
    out = np.interp(grid, tf, y[ok])
    i = np.clip(np.searchsorted(tf, grid), 1, len(tf) - 1)
    out[(tf[i] - tf[i - 1] > 2 * own) | (grid < tf[0]) | (grid > tf[-1])] = np.nan
    return grid, out


def clock_spans(shot: int, paths) -> tuple[np.ndarray, dict]:
    """The ELM clock's detections on one shot: `(spans_ms, info)`.

    The clock reads the first filterscope with data (`spans.dalpha_channel`) on the
    corpus's 10 kHz grid; a cached record fetched at its native rate is brought to
    that grid by `resample`. Each peak becomes the span of its half-prominence width
    centred on the peak. `info` names the channel, the record's tier, and whether the
    channel is on the challenge's `signal_list.yml`.
    """
    if raw.record_tier(shot, "filterscopes", paths=paths) is None:
        # `raw_signal` would fetch it live: a benchmark run reads what is stored
        raise NoDataError(
            f"shot {shot}: filterscopes are in neither the corpus nor the cache"
        )
    fs = raw.raw_signal(shot, "filterscopes", paths=paths)
    y = fs.y[:8]
    row = spans.dalpha_channel(y, shot)
    t, trace = resample(fs.x, y[row])
    events = transients.elm_clock_events(trace, t / 1e3, shot=shot, channel=row)
    peaks = [e for e in events if "width_ms" in e.attrs]
    centre = np.array([e.t0_s * 1e3 for e in peaks])
    half = np.array([e.attrs["width_ms"] for e in peaks]) / 2
    found = (
        np.column_stack([centre - half, centre + half])
        if len(peaks)
        else np.zeros((0, 2))
    )
    return found, {
        "channel": int(row),
        "filterscope": f"FS{row + 1:02d}",
        "tier": fs.attrs["tier"],
        "within_signal_list": row in ALLOWED_FILTERSCOPE_ROWS,
        "n_peaks": len(peaks),
    }


def key_onsets(case: dict) -> tuple[np.ndarray, np.ndarray]:
    """`(onsets, others)` of one case from `source_formatters.read_elm_challenge`: the
    nominal start of each ELM (types 1, 2, 6), and the `(start, end)` of each
    uncertain event (types 3, 4, 8)."""
    onsets = sorted(s for s, _, k in case["events"] if k in ELM_TYPES)
    others = [(s, e) for s, e, k in case["events"] if k in UNCERTAIN_TYPES]
    return np.array(onsets, dtype=float), np.array(others, dtype=float).reshape(-1, 2)


def onset_match(case: dict, found_spans, tolerance_ms: float) -> dict:
    """The onsets of `found_spans` against the key's ELM onsets inside the case window.

    Pairs are made one to one, nearest first, by `scoring.events.match`. A detection
    that matches no ELM but starts within the tolerance of an uncertain key event is
    left out of the false positives, as `scoring.events` leaves out detections beside
    the other reader's uncertain spans. Detections outside the key's window are not
    counted: the key says nothing there.
    """
    onsets, others = key_onsets(case)
    found = np.asarray(found_spans, dtype=float).reshape(-1, 2)[:, 0]
    found = found[(found >= case["tstart"]) & (found <= case["tend"])]
    m = boundary.match(onsets, found, tolerance_ms)
    loose = np.array(
        [
            bool(
                len(others)
                and (
                    (others[:, 0] - tolerance_ms <= found[i])
                    & (found[i] <= others[:, 1] + tolerance_ms)
                ).any()
            )
            for i in m.unmatched_estimate
        ],
        dtype=bool,
    )
    return {
        "tp": len(m.pairs),
        "fn": len(m.unmatched_reference),
        "fp": int((~loose).sum()),
        "excluded": int(loose.sum()),
        "offsets_ms": [float(d) for d in m.offsets],
    }


def nearest_other_gap_ms(onsets) -> tuple[np.ndarray, np.ndarray]:
    """The onsets sorted, and each one's distance (ms) to the nearest other onset
    (1e9 for a lone one)."""
    t = np.sort(np.asarray(onsets, dtype=float))
    if len(t) < 2:
        return t, np.full(len(t), 1e9)
    d = np.diff(t)
    return t, np.minimum(np.r_[1e9, d], np.r_[d, 1e9])


def nearest_signed_offsets_ms(onsets, found_starts, within_ms: float) -> np.ndarray:
    """For each onset, the detection start nearest to it minus the onset (ms), NaN
    where none lies within `within_ms`. Not one to one: a detection can serve two
    onsets, so this is a map of where detections fall, not a score."""
    onsets = np.asarray(onsets, dtype=float)
    found = np.asarray(found_starts, dtype=float)
    out = np.full(len(onsets), np.nan)
    if not len(found):
        return out
    for i, t in enumerate(onsets):
        d = found - t
        j = int(np.argmin(np.abs(d)))
        if abs(d[j]) <= within_ms:
            out[i] = d[j]
    return out


def bin_truth(grid: dict) -> np.ndarray:
    """The key's 50 ms bins as 1 (an ELM touches), 0 (absent), NaN (uncertain or not
    assessed), from a label grid read by `interval_tables.read_label_grid`."""
    return np.asarray(grid["label"], dtype=float)[:, 0]


def bins_touched(spans_ms, times_ms, bin_ms: float = BIN_MS) -> np.ndarray:
    """Which `[t, t + bin_ms)` bins any of the sorted spans touches."""
    a = np.asarray(spans_ms, dtype=float).reshape(-1, 2)
    a = a[np.argsort(a[:, 0], kind="stable")]
    t = np.asarray(times_ms, dtype=float)
    if not len(a):
        return np.zeros(len(t), dtype=bool)
    # a span touches a bin when it starts before the bin ends and ends after it starts
    running = np.maximum.accumulate(a[:, 1])
    last = np.searchsorted(a[:, 0], t + bin_ms, side="left") - 1
    return (last >= 0) & (running[np.clip(last, 0, None)] > t)


def bin_metrics_from_counts(tp, fp, fn) -> dict:
    """Precision, recall and F1 of the present class from pooled counts."""
    tp, fp, fn = int(tp), int(fp), int(fn)
    return {
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan"),
    }


def bin_metrics(truth, called) -> dict:
    """Counts and precision, recall and F1 of the present class over the bins the key
    assessed (`truth` 0 or 1)."""
    truth = np.asarray(truth, dtype=float)
    called = np.asarray(called, dtype=bool)
    keep = np.isfinite(truth)
    t, c = truth[keep] == 1, called[keep]
    tp, fp, fn = int((t & c).sum()), int((~t & c).sum()), int((t & ~c).sum())
    return {
        "bins": int(keep.sum()),
        "present_bins": int(t.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": int((~t & ~c).sum()),
        **bin_metrics_from_counts(tp, fp, fn),
    }
