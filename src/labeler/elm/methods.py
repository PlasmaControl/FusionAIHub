"""Each method's output on the reviewed shots, as `score.ShotScore`s.

A `ShotScore` holds a method's calls on one shot's scored bins. Three kinds of
method meet here:

* a **trace** method (`elm-ours`): a 1 ms event probability whose mean over a bin is
  the bin's score; a bin is called present at the fold's threshold, and its detected
  spans are the stretches where the 50 ms moving mean of the trace reaches it;
* a **row** method (`elm-dsm`): a score per grid row, each row summarising the 50 ms
  before its time stamp; a bin takes the score of one row (the caller says which),
  and a row at or above the threshold marks the 50 ms it summarises as detected;
* a **span** method (ELM-O, the ELM clock): detected spans, a bin is called when one
  touches it.

The span counts of the benchmark (`score.SPAN_KEYS`) are those of
`scripts/labeler/elmo_benchmark.bin_table`; `span_counts` re-states its span rule and
the benchmark script's output is the test of it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import inputs, labels, onset, score

ROW_MS = 25.0
WINDOW_MS = 50.0
EDGE_GUARD_MS = 25.0


def cover_frame(cov0, cov1) -> pd.DataFrame:
    return pd.DataFrame(
        {"t_start_ms": np.asarray(cov0, float), "t_end_ms": np.asarray(cov1, float)}
    )


def span_frame(starts, stops) -> pd.DataFrame:
    return pd.DataFrame(
        {"t_start_ms": np.asarray(starts, float), "t_end_ms": np.asarray(stops, float)}
    )


def _touches(starts, stops, lo, hi) -> np.ndarray:
    """Which of the intervals `[lo, hi)` any of the sorted disjoint spans touches."""
    if not len(starts):
        return np.zeros(len(lo), dtype=bool)
    i = np.minimum(np.searchsorted(stops, lo, side="right"), len(starts) - 1)
    return (stops[i] > lo) & (starts[i] < hi)


def span_counts(
    spans: pd.DataFrame, cover: pd.DataFrame, shot_spans: pd.DataFrame
) -> dict:
    """The benchmark's span counts for detected `spans` on one shot.

    `spans` (columns `t_start_ms`, `t_end_ms`) are the method's detected spans, sorted
    and not overlapping; `shot_spans` the shot's review rows. A labelled span with less
    than half its length analysed (`cover`) is skipped; an non-crowd or absent span
    counts as hit when a detected span touches it inside analysed time. Detections
    are intersected with this panel's coverage before both raw and guarded counts.

    Guarded alarms require a touch in [start + 25, end - 25). The guarded
    share retains the raw denominator; spans without an interior are counted
    separately, and an additional rate uses only nonempty interiors.
    """
    cov0, cov1 = labels.merge_intervals(
        cover.t_start_ms.to_numpy(float), cover.t_end_ms.to_numpy(float)
    )
    starts, stops = intersect(
        spans.t_start_ms.to_numpy(float),
        spans.t_end_ms.to_numpy(float),
        cov0,
        cov1,
    )
    out = dict.fromkeys(score.SPAN_KEYS, 0)
    for row in shot_spans.itertuples():
        if row.kind not in labels.SCORED_KINDS or row.t_end <= row.t_start:
            continue
        analysed = np.clip(
            np.minimum(cov1, row.t_end) - np.maximum(cov0, row.t_start), 0, None
        ).sum()
        if analysed < 0.5 * (row.t_end - row.t_start):
            continue
        hit = bool(
            _touches(
                starts,
                stops,
                np.array([float(row.t_start)]),
                np.array([float(row.t_end)]),
            )[0]
        )
        if row.kind == "absent":
            out["absent_spans"] += 1
            out["absent_span_alarm"] += int(hit)
            lo, hi = row.t_start + EDGE_GUARD_MS, row.t_end - EDGE_GUARD_MS
            if hi > lo:
                out["absent_spans_guard25_eligible"] += 1
                out["absent_span_alarm_guard25"] += int(
                    _touches(starts, stops, np.array([lo]), np.array([hi]))[0]
                )
            else:
                out["absent_spans_guard25_empty"] += 1
        elif row.kind == "crowd":
            out["crowd_spans"] += 1
        else:
            out["non_crowd_spans"] += 1
            out["non_crowd_span_hit"] += int(hit)
    return out


def inside(x_ms, cov0, cov1) -> np.ndarray:
    """Which times of `x_ms` lie in analysed time."""
    x = np.asarray(x_ms, dtype=float)
    k = np.searchsorted(cov0, x, side="right") - 1
    return (k >= 0) & (x < np.asarray(cov1)[np.maximum(k, 0)])


def onset_counts(found_ms, shot_spans, onset_mask, n_ms, cov0, cov1, tol):
    """`(tp, fp, fn)` of detected onsets against the shot's reviewed non-crowd starts,
    counted inside analysed time only."""
    ind = shot_spans[shot_spans.kind == "non_crowd"].t_start.to_numpy(float)
    truth = ind[inside(ind, cov0, cov1)]
    found = np.asarray(found_ms, dtype=float)
    found = found[inside(found, cov0, cov1)]
    cells = inputs.GRID0_MS + np.arange(n_ms) + 0.5
    defined = onset_mask & inside(cells, cov0, cov1)
    return onset.match(found, truth, defined, inputs.GRID0_MS, tol)


def onset_summary(per_shot: np.ndarray, boot: np.ndarray) -> dict:
    """Pooled onset precision, recall and F1 of `(shots, 3)` tp/fp/fn rows."""

    def rates(total):
        tp, fp, fn = total
        f1_denominator = 2 * tp + fp + fn
        return {
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "f1": 2 * tp / f1_denominator if f1_denominator else float("nan"),
        }

    point = rates(per_shot.sum(axis=0))
    reps = [rates(per_shot[d].sum(axis=0)) for d in boot]
    audit = score.bootstrap_summary(
        {k: np.asarray([r[k] for r in reps]) for k in point},
        n_shots=len(per_shot),
        positive_shots=int(((per_shot[:, 0] + per_shot[:, 2]) > 0).sum()),
    )
    tp, fp, fn = (int(v) for v in per_shot.sum(axis=0))
    return {
        "counts": {"tp": tp, "fp": fp, "fn": fn},
        "point": point,
        "replicates": len(boot),
        **audit,
    }


class Oof:
    """The out-of-fold predictions of one `train` run and each shot's thresholds."""

    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir)
        self.record = json.loads((self.dir / "run.json").read_text())
        self.fold_of: dict[int, int] = {}
        self.threshold: dict[int, float] = {}
        self.onset_threshold: dict[int, float] = {}
        for k in range(len(self.record["folds"])):
            info = json.loads((self.dir / f"fold{k}" / "fold.json").read_text())
            for s in info["test"]:
                self.fold_of[s] = k
                self.threshold[s] = info["threshold"]
                self.onset_threshold[s] = info["onset_threshold"]

    def trace(self, shot: int) -> np.ndarray:
        with np.load(self.dir / "pred" / f"{shot}.npz") as z:
            return z["p"].astype(np.float32)


def trace_part(shot_spans, shot, bins, cover, trace, thr) -> score.ShotScore:
    """A trace method's part: bin means, calls at `thr`, spans from the moving mean."""
    s = labels.bin_scores(trace, bins)
    starts, stops = labels.runs_of(trace, thr)
    spans = span_counts(span_frame(starts, stops), cover, shot_spans)
    return score.ShotScore(shot, bins.truth, bins.kind, s >= thr, s, spans)


def row_part(
    shot_spans, shot, bins, cover, row_t_ms, row_score, thr, *, lag_rows=0, ahead=False
):
    """A row method's part.

    `row_t_ms` are the rows' time stamps. A bin's score is the row `lag_rows` rows after
    its end: 0 is the row summarising the bin itself (each row summarises the
    `WINDOW_MS` before its stamp), `-WINDOW_MS / ROW_MS` the row ending at the bin's
    start. A row at or above `thr` marks the `WINDOW_MS` it summarises as detected, or,
    with `ahead` (an offline forward-risk score), the `WINDOW_MS` after its stamp.
    Centered upstream preprocessing can include later inputs. Bins with no such row
    are an error: the caller gives every method the same bins, restricted to the rows.
    """
    end = bins.t0 + WINDOW_MS + lag_rows * ROW_MS
    k = np.rint((end - row_t_ms[0]) / ROW_MS).astype(int)
    if k.size and (k.min() < 0 or k.max() >= len(row_t_ms)):
        raise ValueError("a scored bin has no row; restrict the bins first")
    s = np.asarray(row_score, dtype=float)[k]
    on = np.asarray(row_score) >= thr
    run0, run1 = _row_runs(row_t_ms, on, ahead)
    spans = span_counts(span_frame(run0, run1), cover, shot_spans)
    return score.ShotScore(shot, bins.truth, bins.kind, s >= thr, s, spans)


def _row_runs(row_t_ms, on, ahead=False):
    """Merged windows of the rows that are on: `[t - WINDOW_MS, t)`, or `[t, t + W)`."""
    t = np.asarray(row_t_ms, dtype=float)[np.asarray(on, dtype=bool)]
    out0: list[float] = []
    out1: list[float] = []
    for e in t:
        a = e if ahead else e - WINDOW_MS
        b = a + WINDOW_MS
        if out1 and a <= out1[-1]:
            out1[-1] = max(out1[-1], b)
        else:
            out0.append(a)
            out1.append(b)
    return np.array(out0), np.array(out1)


def span_part(shot_spans, shot, bins, cover, spans: pd.DataFrame) -> score.ShotScore:
    """A span method's part: a bin is called when a detected span touches it."""
    spans = spans.sort_values("t_start_ms")
    call = labels.hard_hits(
        spans.t_start_ms.to_numpy(float), spans.t_end_ms.to_numpy(float), bins
    )
    counts = span_counts(spans, cover, shot_spans)
    return score.ShotScore(shot, bins.truth, bins.kind, call, None, counts)


def restrict_bins(bins: labels.Bins, keep: np.ndarray) -> labels.Bins:
    return labels.Bins(
        bins.t0[keep], bins.truth[keep], bins.kind[keep], bins.span[keep]
    )


def intersect(a0, a1, b0, b1) -> tuple[np.ndarray, np.ndarray]:
    """The intersection of two sets of intervals, each sorted and not overlapping."""
    a0, a1, b0, b1 = (np.asarray(v, dtype=float) for v in (a0, a1, b0, b1))
    out0: list[float] = []
    out1: list[float] = []
    i = j = 0
    while i < len(a0) and j < len(b0):
        lo, hi = max(a0[i], b0[j]), min(a1[i], b1[j])
        if hi > lo:
            out0.append(lo)
            out1.append(hi)
        if a1[i] < b1[j]:
            i += 1
        else:
            j += 1
    return np.array(out0), np.array(out1)


def areas(parts: list[score.ShotScore], index) -> tuple[float, float]:
    """AUROC and AUPRC of the pooled bin scores of the shots `index`."""
    truth, sc = score._pool(parts, index)
    return score.roc_auc(truth, sc), score.average_precision(truth, sc)


def areas_summary(parts: list[score.ShotScore], boot: np.ndarray) -> dict:
    """AUROC and AUPRC of a continuous score, with shot-bootstrap intervals."""
    point = areas(parts, range(len(parts)))
    reps = np.array([areas(parts, d) for d in boot])
    return {
        "auroc": point[0],
        "auprc": point[1],
        **score.bootstrap_summary(
            {"auroc": reps[:, 0], "auprc": reps[:, 1]},
            n_shots=len(parts),
            positive_shots=sum(bool(np.any(p.truth == 1)) for p in parts),
        ),
    }


def kind_summary(parts: list[score.ShotScore], boot: np.ndarray) -> dict:
    """Crowd/non-crowd bin and non-crowd span recall with shot-bootstrap CIs."""
    per = []
    for part in parts:
        crowd, non_crowd = part.kind == "crowd", part.kind == "non_crowd"
        per.append(
            [
                int((crowd & part.call).sum()),
                int(crowd.sum()),
                int((non_crowd & part.call).sum()),
                int(non_crowd.sum()),
                part.spans.get("non_crowd_span_hit", 0),
                part.spans.get("non_crowd_spans", 0),
            ]
        )
    per = np.asarray(per, dtype=int)
    total = per.sum(axis=0)
    sampled = per[boot].sum(axis=1)
    out = {}
    for i, metric in enumerate(
        ("crowd_bin_recall", "non_crowd_bin_recall", "non_crowd_span_touch_recall")
    ):
        hit, count = total[2 * i : 2 * i + 2]
        numerator, denominator = sampled[:, 2 * i], sampled[:, 2 * i + 1]
        values = np.full(len(boot), np.nan)
        np.divide(numerator, denominator, out=values, where=denominator > 0)
        out[metric] = {
            "point": float(hit / count) if count else float("nan"),
            "numerator": int(hit),
            "denominator": int(count),
            **score.bootstrap_summary(
                {metric: values},
                n_shots=len(parts),
                positive_shots=int((per[:, 2 * i + 1] > 0).sum()),
            ),
        }
        out[metric]["ci95"] = out[metric]["ci95"][metric]
        out[metric]["bootstrap_draw_counts"] = out[metric]["bootstrap_draw_counts"][
            metric
        ]
    return out


def annotation_summary(
    parts: dict[str, list[score.ShotScore]],
    reviews: dict[int, pd.DataFrame],
    *,
    replicates: int = score.REPLICATES,
    seed: int = score.SEED,
) -> dict:
    """Disjoint shot groups defined by present kinds in the complete review.

    Group membership uses all reviewed rows, including those outside a panel's
    coverage or too short to contribute a scored bin. Each group's metrics use
    that panel's existing bins, spans and thresholds. Resamples draw whole shots
    within a group and are shared by methods, never individual bins or spans.
    """
    shot_lists = [[part.shot for part in values] for values in parts.values()]
    if not shot_lists or any(shots != shot_lists[0] for shots in shot_lists):
        raise ValueError("all methods must have the same shots in the same order")
    groups = {key: [] for key in ("crowd_only", "non_crowd_only", "mixed", "no_present")}
    for i, shot in enumerate(shot_lists[0]):
        kinds = set(reviews[shot].kind)
        if {"crowd", "non_crowd"} <= kinds:
            group = "mixed"
        elif "crowd" in kinds:
            group = "crowd_only"
        elif "non_crowd" in kinds:
            group = "non_crowd_only"
        else:
            group = "no_present"
        groups[group].append(i)
    out = {}
    first = next(iter(parts.values()))
    for group, indices in groups.items():
        selected = {name: [values[i] for i in indices] for name, values in parts.items()}
        summaries = {}
        if indices:
            boot = score.draws(len(indices), replicates=replicates, seed=seed)
            summaries = {
                name: score.summarise(values, boot) for name, values in selected.items()
            }
        if group == "no_present":
            for summary in summaries.values():
                for field in ("point", "ci95"):
                    summary[field]["no_present_false_positive_fraction"] = summary[
                        field
                    ]["false_alarm_bin_rate"]
                summary["bootstrap_draw_counts"][
                    "no_present_false_positive_fraction"
                ] = summary["bootstrap_draw_counts"]["false_alarm_bin_rate"]
        out[group] = {
            "shots": [first[i].shot for i in indices],
            "n_shots": len(indices),
            "bins": sum(len(first[i].truth) for i in indices),
            "applicable_metrics": (
                ["no_present_false_positive_fraction"]
                if group == "no_present"
                else ["precision", "recall", "f1", "auroc", "auprc"]
            ),
            "methods": summaries,
        }
    return out


PAIRED_METRICS = (
    "f1",
    "precision",
    "recall",
    "false_alarm_bin_rate",
    "crowd_bin_recall",
    "non_crowd_span_touch_recall",
    "absent_span_alarm_rate",
    "absent_span_alarm_rate_guard25",
)


def summarise_methods(
    parts: dict[str, list[score.ShotScore]], boot: np.ndarray, reference: str
) -> dict:
    """Every method's summary and each one's paired difference from `reference`.

    All methods must have been scored on the same shots in the same order. A
    difference of AUROC or AUPRC is given where both methods have a continuous score.
    """
    out: dict = {"methods": {}, "paired": {}}
    from .compare import DISPLAY_NAME

    for name, plist in parts.items():
        out["methods"][name] = score.summarise(plist, boot)
        out["methods"][name]["display_name"] = DISPLAY_NAME.get(name, name)
        out["methods"][name]["high_recall"] = bool(
            out["methods"][name]["point"]["recall"] >= 0.99
        )
    rule_parts = []
    for p in parts[reference]:
        spans = dict(p.spans)
        spans["non_crowd_span_hit"] = spans.get("non_crowd_spans", 0)
        spans["absent_span_alarm"] = spans.get("absent_spans", 0)
        spans["absent_span_alarm_guard25"] = spans.get(
            "absent_spans_guard25_eligible", 0
        )
        rule_parts.append(
            score.ShotScore(
                p.shot,
                p.truth,
                p.kind,
                np.ones(len(p.truth), bool),
                np.zeros(len(p.truth)),
                spans,
            )
        )
    out["methods"]["always present"] = score.summarise(rule_parts, boot)
    out["methods"]["always present"]["display_name"] = "Always-present rule"
    out["methods"]["always present"]["high_recall"] = True
    ref = parts[reference]
    for name, plist in parts.items():
        if name == reference:
            continue
        metrics = list(PAIRED_METRICS)
        if all(p.score is not None for p in ref) and all(
            p.score is not None for p in plist
        ):
            metrics += ["auroc", "auprc"]
        for metric in metrics:
            out["paired"][f"{reference} - {name}: {metric}"] = score.paired_difference(
                ref, plist, boot, metric
            )
    return out
