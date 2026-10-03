"""Score per-bin ELM outputs against the reviewed bins, with shot-bootstrap intervals.

A method's result on one shot is a `ShotScore`: its scored 50 ms bins (`labels.Bins`
truth and kind), a continuous `score` for each bin when the method has one, the hard
`call` it makes, and the span counts of the benchmark (`SPAN_KEYS`). The numbers
here are the benchmark's own, so a method scored here sits beside ELM-O's and the
ELM clock's in `outputs/labeler/elm/elmo/evaluation.json`:

* precision, recall and F1 of the calls over the pooled bins;
* `false_alarm_bin_rate`, the share of absent bins called present;
* `crowd_bin_recall`, the share of bins inside crowd spans called present;
* `non_crowd_span_touch_recall`, the share of non-crowd present spans the method
  touches anywhere (the ELM-O rule: a detected span overlapping the labelled one);
* `absent_span_alarm_rate`, the share of absent spans it touches;
* `absent_span_alarm_rate_guard25`, the share of the same spans touched after
  removing 25 ms from each edge; empty interiors are counted separately, with
  `absent_span_interior_alarm_rate_guard25` restricted to nonempty interiors;
* AUROC and AUPRC of the continuous score over the pooled bins (ties averaged;
  AUPRC is the average precision), for methods that have one.

Intervals are percentile intervals over shot draws: `draws` gives `n` resamples of
the shots with replacement, and the same draws are used for every method scored
on the same shots so differences are paired.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

REPLICATES = 1000
SEED = 20261003
MIN_POSITIVE_SHOTS_FOR_CI = 5
SPAN_KEYS = (
    "non_crowd_spans",
    "non_crowd_span_hit",
    "absent_spans",
    "absent_span_alarm",
    "crowd_spans",
    "absent_spans_guard25_eligible",
    "absent_span_alarm_guard25",
    "absent_spans_guard25_empty",
)


def roc_auc(truth: np.ndarray, score: np.ndarray) -> float:
    """Rank AUROC; NaN for an empty class or any nonfinite score."""
    truth = np.asarray(truth).astype(bool)
    score = np.asarray(score, dtype=np.float64)
    pos, neg = int(truth.sum()), int((~truth).sum())
    if not pos or not neg or not np.isfinite(score).all():
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    new = np.r_[True, s[1:] != s[:-1]]
    first = np.flatnonzero(new)
    last = np.r_[first[1:], s.size]
    mean_rank = (first + last + 1) / 2.0  # average of 1-based ranks first+1 .. last
    ranks = np.empty(s.size)
    ranks[order] = np.repeat(mean_rank, last - first)
    return float((ranks[truth].sum() - pos * (pos + 1) / 2) / (pos * neg))


def average_precision(truth: np.ndarray, score: np.ndarray) -> float:
    """Average precision; NaN for no positives or any nonfinite score."""
    truth = np.asarray(truth).astype(bool)
    score = np.asarray(score, dtype=np.float64)
    pos = int(truth.sum())
    if not pos or not np.isfinite(score).all():
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    s, t = score[order], truth[order]
    last = np.r_[np.flatnonzero(s[1:] != s[:-1]), s.size - 1]
    tp = np.cumsum(t)[last]
    called = last + 1
    recall = tp / pos
    precision = tp / called
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def best_threshold(truth: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """The score threshold maximising F1 of `score >= threshold`, and that F1.

    Candidates are the midpoints between distinct sorted scores plus the extremes, so
    the returned threshold separates the bins it counts as called. Ties go to the
    higher threshold.
    Nonfinite scores cannot define an operating point and return two NaNs.
    """
    truth = np.asarray(truth).astype(bool)
    score = np.asarray(score, dtype=np.float64)
    if not truth.any() or not np.isfinite(score).all():
        return float("nan"), float("nan")
    order = np.argsort(-score, kind="mergesort")
    s, t = score[order], truth[order]
    last = np.r_[np.flatnonzero(s[1:] != s[:-1]), s.size - 1]
    tp = np.cumsum(t)[last]
    called = last + 1
    f1 = 2 * tp / (called + truth.sum())
    best = int(np.argmax(f1))
    below = s[last[best] + 1] if last[best] + 1 < s.size else s[last[best]] - 1.0
    return float((s[last[best]] + below) / 2), float(f1[best])


@dataclass
class ShotScore:
    """One method's output on one shot's scored bins."""

    shot: int
    truth: np.ndarray  # int8 (m,)
    kind: np.ndarray  # object (m,)
    call: np.ndarray  # bool (m,)
    score: np.ndarray | None = None  # float (m,)
    spans: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        m = len(self.truth)
        if len(self.kind) != m or len(self.call) != m:
            raise ValueError("truth, kind and call must have one entry per bin")
        if self.score is not None and len(self.score) != m:
            raise ValueError("score must have one entry per bin")


def counts(part: ShotScore) -> np.ndarray:
    """`tp, fp, fn, tn, crowd_bins, crowd_hit` then `SPAN_KEYS`, as floats."""
    t, c = part.truth.astype(bool), part.call.astype(bool)
    crowd = part.kind == "crowd"
    base = [
        (t & c).sum(),
        (~t & c).sum(),
        (t & ~c).sum(),
        (~t & ~c).sum(),
        crowd.sum(),
        (crowd & c).sum(),
    ]
    return np.array(base + [part.spans.get(k, 0) for k in SPAN_KEYS], dtype=float)


COUNT_NAMES = ("tp", "fp", "fn", "tn", "crowd_bins", "crowd_hit", *SPAN_KEYS)


def _ratio(a: float, b: float) -> float:
    return float(a / b) if b > 0 else float("nan")


def rates(total: np.ndarray) -> dict[str, float]:
    """The benchmark's hard-call rates from summed `counts`."""
    c = dict(zip(COUNT_NAMES, total))
    p = _ratio(c["tp"], c["tp"] + c["fp"])
    r = _ratio(c["tp"], c["tp"] + c["fn"])
    return {
        "precision": p,
        "recall": r,
        "f1": _ratio(2 * c["tp"], 2 * c["tp"] + c["fp"] + c["fn"]),
        "false_alarm_bin_rate": _ratio(c["fp"], c["fp"] + c["tn"]),
        "crowd_bin_recall": _ratio(c["crowd_hit"], c["crowd_bins"]),
        "non_crowd_span_touch_recall": _ratio(
            c["non_crowd_span_hit"], c["non_crowd_spans"]
        ),
        "absent_span_alarm_rate": _ratio(c["absent_span_alarm"], c["absent_spans"]),
        "absent_span_alarm_rate_guard25": _ratio(
            c["absent_span_alarm_guard25"], c["absent_spans"]
        ),
        "absent_span_interior_alarm_rate_guard25": _ratio(
            c["absent_span_alarm_guard25"], c["absent_spans_guard25_eligible"]
        ),
    }


def draws(n_shots: int, replicates: int = REPLICATES, seed: int = SEED) -> np.ndarray:
    """`(replicates, n_shots)` shot indices drawn with replacement."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_shots, size=(replicates, n_shots))


def _pool(parts: Sequence[ShotScore], index: Sequence[int]):
    sel = [parts[i] for i in index]
    truth = np.concatenate([p.truth for p in sel]) if sel else np.zeros(0)
    score = np.concatenate([p.score for p in sel]) if sel else np.zeros(0)
    return truth, score


def _ci(values: np.ndarray) -> list[float]:
    v = values[np.isfinite(values)]
    if not v.size:
        return [float("nan"), float("nan")]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def bootstrap_summary(
    values: dict[str, np.ndarray],
    *,
    n_shots: int,
    positive_shots: int,
    endpoint_shots: dict[str, int] | None = None,
) -> dict[str, object]:
    """Audit every draw; eligibility uses shots bearing each endpoint's denominator.

    An undefined draw remains in the audit denominator. Percentile intervals
    otherwise use finite draws, so their conditioning is visible rather than
    silently discarding resamples without an applicable denominator or class.
    """
    support = {key: (endpoint_shots or {}).get(key, positive_shots) for key in values}
    eligible = {
        key: n_shots >= 5 and count >= MIN_POSITIVE_SHOTS_FOR_CI
        for key, count in support.items()
    }
    return {
        "positive_shots": positive_shots,
        "descriptive_only": not any(eligible.values()),
        "endpoint_shots": support,
        "interval_eligible": eligible,
        "interval_policy": "At least five denominator-bearing physical shots "
        "per endpoint; positive-bearing shots for positive-class/ranking metrics, "
        "negative-bearing shots for false-alarm rates. Percentile intervals use "
        "finite shot-bootstrap draws, with valid/undefined counts reported.",
        "ci95": {
            key: _ci(np.asarray(draws)) if eligible[key] else None
            for key, draws in values.items()
        },
        "bootstrap_draw_counts": {
            key: {
                "valid": int(np.isfinite(draws).sum()),
                "undefined": int((~np.isfinite(draws)).sum()),
            }
            for key, draws in values.items()
        },
    }


def _endpoint_shots(parts: Sequence[ShotScore]) -> dict[str, int]:
    """Denominator support for endpoints that do not use all positive bins."""
    spans = {
        "non_crowd_span_touch_recall": "non_crowd_spans",
        "absent_span_alarm_rate": "absent_spans",
        "absent_span_alarm_rate_guard25": "absent_spans",
        "absent_span_interior_alarm_rate_guard25": "absent_spans_guard25_eligible",
    }
    return {
        "false_alarm_bin_rate": sum(bool(np.any(p.truth == 0)) for p in parts),
        "crowd_bin_recall": sum(bool(np.any(p.kind == "crowd")) for p in parts),
        **{
            metric: sum(p.spans.get(key, 0) > 0 for p in parts)
            for metric, key in spans.items()
        },
    }


def summarise(
    parts: Sequence[ShotScore], boot: np.ndarray | None = None
) -> dict[str, object]:
    """Point estimates, applicable intervals and valid/undefined draw counts.

    `boot` is `draws(len(parts))`; with none, the intervals are omitted. The
    AUROC and AUPRC are given when every part has a `score`. Intervals are null
    for fewer than five denominator-bearing shots for that endpoint.
    """
    per = np.stack([counts(p) for p in parts])
    out: dict[str, object] = {
        "shots": len(parts),
        "counts": dict(zip(COUNT_NAMES, per.sum(axis=0).astype(int).tolist())),
    }
    point = rates(per.sum(axis=0))
    scored = all(p.score is not None for p in parts)
    if scored:
        truth, score = _pool(parts, range(len(parts)))
        point["auroc"] = roc_auc(truth, score)
        point["auprc"] = average_precision(truth, score)
        point["prevalence"] = float(truth.mean())
    out["point"] = point
    if boot is None:
        return out
    reps = {k: [] for k in point if k != "prevalence"}
    for d in boot:
        total = per[d].sum(axis=0)
        r = rates(total)
        if scored:
            truth, score = _pool(parts, d)
            r["auroc"] = roc_auc(truth, score)
            r["auprc"] = average_precision(truth, score)
        for k, values in reps.items():
            values.append(r[k])
    out.update(
        bootstrap_summary(
            {k: np.asarray(v) for k, v in reps.items()},
            n_shots=len(parts),
            positive_shots=sum(bool(np.any(p.truth == 1)) for p in parts),
            endpoint_shots=_endpoint_shots(parts),
        )
    )
    out["replicates"] = len(boot)
    return out


def paired_difference(
    a: Sequence[ShotScore], b: Sequence[ShotScore], boot: np.ndarray, metric: str
) -> dict[str, object]:
    """`metric(a) - metric(b)` on the same shots (same order), with its interval."""
    if [p.shot for p in a] != [p.shot for p in b]:
        raise ValueError("the two methods must be scored on the same shots")

    ranked = metric in ("auroc", "auprc")
    per_a, per_b = (
        (None, None)
        if ranked
        else (np.stack([counts(p) for p in a]), np.stack([counts(p) for p in b]))
    )

    def value(parts, d, per):
        if metric in ("auroc", "auprc"):
            truth, score = _pool(parts, d)
            fn = roc_auc if metric == "auroc" else average_precision
            return fn(truth, score)
        total = per[d].sum(axis=0)
        return rates(total)[metric]

    all_idx = np.arange(len(a))
    point = value(a, all_idx, per_a) - value(b, all_idx, per_b)
    reps = np.array([value(a, d, per_a) - value(b, d, per_b) for d in boot])
    audit = bootstrap_summary(
        {metric: reps},
        n_shots=len(a),
        positive_shots=min(
            sum(bool(np.any(p.truth == 1)) for p in parts) for parts in (a, b)
        ),
        endpoint_shots={
            key: min(value, _endpoint_shots(b)[key])
            for key, value in _endpoint_shots(a).items()
        },
    )
    audit["ci95"] = audit["ci95"][metric]
    audit["bootstrap_draw_counts"] = audit["bootstrap_draw_counts"][metric]
    return {"value": float(point), "replicates": len(boot), **audit}
