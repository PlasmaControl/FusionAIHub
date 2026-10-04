"""Independent Smith-window targets and one-to-one ELM-onset evaluation.

Smith's short hand-labelled regions define positive time only inside their chosen
windows. Outside those windows is unknown. A scored 1 ms cell must lie wholly
inside at least one individual window; eligible cells are deduplicated per shot,
with any region overlap defining presence. This conservatively excludes cells
covered only by stitching across neighboring window edges.
These targets differ from the reviewed 50 ms ELMing-period occupancy benchmark.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import inputs, train

SEED = 20261003
SIGMA_MS = 1.0


def first_monotone_segment(time: np.ndarray, values: np.ndarray):
    """Retain the first chronological acquisition, never mix repeated segments.

    Cached shot 179859 has three overlapping FS time segments. Choosing by source
    acquisition order is independent of ELM labels and preserves ordinary records.
    The frozen input transformations themselves remain unchanged.
    """
    resets = np.flatnonzero(np.diff(time) <= 0)
    stop = int(resets[0] + 1) if len(resets) else len(time)
    return (
        time[:stop],
        values[:, :stop],
        {
            "time_resets": len(resets),
            "original_samples": len(time),
            "retained_samples": stop,
            "discarded_samples": len(time) - stop,
            "retained_start_ms": float(time[0]),
            "retained_stop_ms": float(time[stop - 1]),
            "rule": "first monotone acquisition in original source order; independent of labels",
        },
    )


def check_disjoint(shots, review_shots, cohort: pd.DataFrame) -> None:
    """Forbid review exposure and the fixed cohort blind test split."""
    overlap = sorted(set(shots) & set(review_shots))
    if overlap:
        raise ValueError(f"Smith training overlaps review shots: {overlap}")
    train.check_no_test(shots, cohort)


def targets(rows: pd.DataFrame, n_ms: int, grid0: float = inputs.GRID0_MS):
    """Occupancy state, Gaussian onset target and known-time mask, all `(n_ms,)`."""
    edge = grid0 + np.arange(n_ms)
    known = np.zeros(n_ms, dtype=bool)
    present = np.zeros(n_ms, dtype=bool)
    onset = np.zeros(n_ms, dtype=np.float32)
    for row in rows.itertuples():
        known |= (edge >= row.t0_ms) & (edge + 1 <= row.end_ms)
        present |= (edge < row.label_t1_ms) & (edge + 1 > row.label_t0_ms)
        bump = np.exp(-0.5 * ((edge + 0.5 - row.label_t0_ms) / SIGMA_MS) ** 2)
        onset = np.maximum(onset, bump.astype(np.float32))
    state = np.full(n_ms, -1, dtype=np.int8)
    state[known] = present[known].astype(np.int8)
    onset[~known] = 0
    return state, onset, known


def deduplicate_events(values, tolerance: float = 0.1) -> list[float]:
    """Collapse repeated detections in overlapping windows, retaining first time."""
    out = []
    for value in sorted(float(v) for v in values):
        if not out or value - out[-1] > tolerance:
            out.append(value)
    return out


def event_counts(found, truth, tolerance: float) -> tuple[int, int, int]:
    """Linear maximum-cardinality match for validation counts (no timing costs)."""
    found, truth = np.sort(found), np.sort(truth)
    i = j = tp = 0
    while i < len(found) and j < len(truth):
        if found[i] < truth[j] - tolerance:
            i += 1
        elif truth[j] < found[i] - tolerance:
            j += 1
        else:
            tp += 1
            i, j = i + 1, j + 1
    return tp, len(found) - tp, len(truth) - tp


def match_events(found, truth, tolerance: float) -> dict:
    """Maximum-cardinality chronological matching, then minimum absolute error.

    A monotone assignment is sufficient for absolute-time distances and interval
    tolerance. Dynamic programming prevents greedy nearest matching from consuming
    the only candidate of a later truth. Errors are prediction minus hand onset.
    """
    found, truth = np.sort(found), np.sort(truth)
    n, m = len(found), len(truth)
    count = np.zeros((n + 1, m + 1), dtype=np.int32)
    cost = np.zeros((n + 1, m + 1), dtype=float)
    action = np.zeros((n + 1, m + 1), dtype=np.int8)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            choices = [
                (count[i - 1, j], -cost[i - 1, j], 1),
                (count[i, j - 1], -cost[i, j - 1], 2),
            ]
            error = abs(found[i - 1] - truth[j - 1])
            if error <= tolerance:
                choices.append(
                    (count[i - 1, j - 1] + 1, -cost[i - 1, j - 1] - error, 3)
                )
            c, neg, a = max(choices)
            count[i, j], cost[i, j], action[i, j] = c, -neg, a
    errors = []
    i, j = n, m
    while i and j:
        a = action[i, j]
        if a == 3:
            errors.append(float(found[i - 1] - truth[j - 1]))
            i, j = i - 1, j - 1
        elif a == 1:
            i -= 1
        else:
            j -= 1
    tp = int(count[n, m])
    return {"tp": tp, "fp": n - tp, "fn": m - tp, "errors_ms": list(reversed(errors))}


def _rates(count):
    tp, fp, fn = count
    return {
        "precision": float(tp / (tp + fp)) if tp + fp else float("nan"),
        "recall": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else float("nan"),
    }


def event_summary(parts: list[dict], boot: np.ndarray) -> dict:
    """Pooled event counts, timing distribution and whole-shot bootstrap CIs."""
    per = np.array([[p[k] for k in ("tp", "fp", "fn")] for p in parts])
    total = per.sum(axis=0)
    errors = np.array([e for p in parts for e in p["errors_ms"]])
    positive_shots = int(sum(p["tp"] + p["fn"] > 0 for p in parts))
    out = {
        "counts": dict(zip(("tp", "fp", "fn"), total.tolist())),
        "point": _rates(total),
        "positive_bearing_shots": positive_shots,
        "descriptive_only": len(parts) < 5 or positive_shots < 5,
        "timing_error_ms": {
            "matched": len(errors),
            "sign": "prediction minus hand onset",
            "mean": float(errors.mean()) if len(errors) else None,
            "median": float(np.median(errors)) if len(errors) else None,
            "median_absolute": float(np.median(abs(errors))) if len(errors) else None,
            "quantiles": dict(
                zip(
                    ("p05", "p25", "p50", "p75", "p95"),
                    np.percentile(errors, [5, 25, 50, 75, 95]).tolist(),
                )
            )
            if len(errors)
            else {},
        },
        "replicates": len(boot),
    }
    values = {
        k: []
        for k in (
            "precision",
            "recall",
            "f1",
            "timing_mean_ms",
            "timing_median_ms",
            "timing_median_absolute_ms",
            "timing_p05_ms",
            "timing_p95_ms",
        )
    }
    for draw in boot:
        rates = _rates(per[draw].sum(axis=0))
        drawn_errors = np.array([e for i in draw for e in parts[i]["errors_ms"]])
        if len(drawn_errors):
            timing = (
                drawn_errors.mean(),
                np.median(drawn_errors),
                np.median(abs(drawn_errors)),
                *np.percentile(drawn_errors, [5, 95]),
            )
        else:
            timing = (float("nan"),) * 5
        rates.update(
            zip(
                (
                    "timing_mean_ms",
                    "timing_median_ms",
                    "timing_median_absolute_ms",
                    "timing_p05_ms",
                    "timing_p95_ms",
                ),
                timing,
            )
        )
        for k, metric_values in values.items():
            metric_values.append(rates[k])
    out["bootstrap"] = {}
    intervals = {}
    for k, v in values.items():
        v = np.asarray(v)
        finite = v[np.isfinite(v)]
        out["bootstrap"][k] = {"valid": len(finite), "undefined": len(v) - len(finite)}
        intervals[k] = (
            np.percentile(finite, [2.5, 97.5]).tolist() if len(finite) else None
        )
    if not out["descriptive_only"]:
        out["ci95"] = {
            k: v for k, v in intervals.items() if not k.startswith("timing_")
        }
        out["timing_error_ms"]["ci95"] = {
            k: v for k, v in intervals.items() if k.startswith("timing_")
        }
    return out
