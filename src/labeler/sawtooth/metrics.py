"""Shot-grouped crash and presence scores, with efficient shot bootstraps."""

from __future__ import annotations

import numpy as np

from ..scoring.events import match

HISTOGRAM_BINS = 512


def classification_metrics(cells, *, unclassified=None):
    """Three-regime window scores; rows are truth and columns are prediction.

    Macro F1 averages over all three declared classes, with zero F1 for a class
    with no truth or predictions. Recall is undefined when no truth is available.
    Unclassified counts retain the original truth support and count as errors.
    """
    cells = np.asarray(cells)
    if cells.shape != (3, 3) or not np.isfinite(cells).all() or (cells < 0).any():
        raise ValueError("classification requires a finite nonnegative 3x3 matrix")
    unclassified = np.asarray(
        np.zeros(3) if unclassified is None else unclassified, dtype=float
    )
    if (
        unclassified.shape != (3,)
        or not np.isfinite(unclassified).all()
        or (unclassified < 0).any()
    ):
        raise ValueError("unclassified counts require three finite nonnegative values")
    support = cells.sum(axis=1) + unclassified
    predicted = cells.sum(axis=0)
    true_positive = cells.diagonal()
    denominators = support + predicted
    f1 = np.divide(
        2 * true_positive,
        denominators,
        out=np.zeros(3, dtype=float),
        where=denominators > 0,
    )
    return {
        "confusion": cells.astype(int).tolist(),
        "class_support": support.astype(int).tolist(),
        "predicted_class_counts": predicted.astype(int).tolist(),
        "per_class_recall": [
            float(tp / total) if total else None
            for tp, total in zip(true_positive, support, strict=True)
        ],
        "per_class_precision": [
            float(tp / total) if total else None
            for tp, total in zip(true_positive, predicted, strict=True)
        ],
        "class_shares": (support / support.sum()).tolist()
        if support.sum()
        else [None] * 3,
        "predicted_class_shares": (predicted / predicted.sum()).tolist()
        if predicted.sum()
        else [None] * 3,
        "per_class_f1": f1.tolist() if support.sum() else [None] * 3,
        "window_accuracy": float(cells.trace() / support.sum())
        if support.sum()
        else None,
        "macro_f1": float(f1.mean()) if support.sum() else None,
        "windows": int(support.sum()),
        "classified_windows": int(cells.sum()),
        "unclassified_windows": int(unclassified.sum()),
        "unclassified_by_true_class": unclassified.astype(int).tolist(),
    }


def bootstrap_classification(rows, *, replicates=1000, seed=20261003):
    """Resample whole-shot confusion matrices, including a fit-chosen majority.

    Each row carries ``cells`` and ``majority_cells`` from the same eligible
    windows. The majority class is chosen on fitting shots separately per fold.
    """
    if not rows:
        raise ValueError("classification bootstrap requires at least one shot")
    cells = np.stack([r["cells"] for r in rows])
    unclassified = np.stack([r.get("unclassified", np.zeros(3)) for r in rows])
    majority = np.stack([r["majority_cells"] for r in rows])
    result = classification_metrics(
        cells.sum(axis=0), unclassified=unclassified.sum(axis=0)
    )
    baseline = classification_metrics(majority.sum(axis=0))
    samples = {name: [] for name in ("window_accuracy", "macro_f1")}
    baseline_samples = {name: [] for name in samples}
    recall_samples = [[] for _ in range(3)]
    precision_samples = [[] for _ in range(3)]
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        selected = rng.integers(0, len(rows), size=len(rows))
        values = classification_metrics(
            cells[selected].sum(axis=0), unclassified=unclassified[selected].sum(axis=0)
        )
        majority_values = classification_metrics(majority[selected].sum(axis=0))
        for name, sample in samples.items():
            if values[name] is not None:
                sample.append(values[name])
                baseline_samples[name].append(majority_values[name])
        for sample, value in zip(
            recall_samples, values["per_class_recall"], strict=True
        ):
            if value is not None:
                sample.append(value)
        for sample, value in zip(
            precision_samples, values["per_class_precision"], strict=True
        ):
            if value is not None:
                sample.append(value)

    def interval(sample):
        return np.quantile(sample, [0.025, 0.975]).tolist() if sample else None

    baseline["ci95"] = {name: interval(v) for name, v in baseline_samples.items()}
    baseline["selection"] = "majority of natural fitting-shot windows, per fold"
    return {
        **result,
        "ci95": {
            **{name: interval(v) for name, v in samples.items()},
            "per_class_recall": [interval(v) for v in recall_samples],
            "per_class_precision": [interval(v) for v in precision_samples],
        },
        "majority_baseline": baseline,
        "shot_ids": [r["shot"] for r in rows],
        "shots": len(rows),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
    }


def bin_times(window_s, bin_ms=2.0):
    """Centers of one absolute-time bin grid within half-open waveform coverage."""
    lo, hi = window_s
    step = bin_ms / 1000
    first = int(np.ceil((lo - step / 2) / step - 1e-9))
    centers = (
        first + np.arange(max(0, int(np.ceil((hi - lo) / step)) + 1))
    ) * step + step / 2
    return centers[(centers >= lo - 1e-12) & (centers < hi)]


def event_cells(reference_s, estimate_s, tolerance_ms=2.0):
    pairing = match(
        np.asarray(reference_s) * 1000, np.asarray(estimate_s) * 1000, tolerance_ms
    )
    return np.array(
        [
            len(pairing.pairs),
            len(pairing.unmatched_estimate),
            len(pairing.unmatched_reference),
        ],
        dtype=float,
    )


def point_metrics(cells):
    tp, fp, fn = np.asarray(cells, dtype=float)
    return {
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "recall": float(tp / (tp + fn)) if tp + fn else None,
        "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else None,
    }


def interval_cells(reference, estimate, minimum_iou=0.1):
    """One-to-one interval matching, descending intersection over union."""
    reference, estimate = list(reference), list(estimate)
    possible = []
    for i, (a, b) in enumerate(reference):
        for j, (c, d) in enumerate(estimate):
            overlap = max(0.0, min(b, d) - max(a, c))
            union = max(b, d) - min(a, c)
            iou = overlap / union if union > 0 else 0.0
            if iou >= minimum_iou:
                possible.append((-iou, i, j))
    used_ref, used_est = set(), set()
    for _, i, j in sorted(possible):
        if i not in used_ref and j not in used_est:
            used_ref.add(i)
            used_est.add(j)
    return np.array(
        [len(used_ref), len(estimate) - len(used_est), len(reference) - len(used_ref)],
        dtype=float,
    )


def masked_interval_cells(reference, estimate, support, minimum_iou=0.1):
    """One-to-one span matching using only supported duration for overlap/union.

    A gap does not create another annotation or prediction. Original intervals
    with no assessed support are excluded from all matching denominators. Support
    intervals are merged first, so duplicated or overlapping cells count once.
    """
    merged = []
    for start, end in sorted((a, b) for a, b in support if b > a):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))

    def duration(start, end):
        return sum(max(0.0, min(end, b) - max(start, a)) for a, b in merged)

    references = [(a, b, duration(a, b)) for a, b in reference]
    estimates = [(a, b, duration(a, b)) for a, b in estimate]
    references = [span for span in references if span[2] > 0]
    estimates = [span for span in estimates if span[2] > 0]
    possible = []
    for i, (a, b, ref_duration) in enumerate(references):
        for j, (c, d, est_duration) in enumerate(estimates):
            overlap = duration(max(a, c), min(b, d))
            union = ref_duration + est_duration - overlap
            iou = overlap / union if union > 0 else 0.0
            if overlap > 0 and iou >= minimum_iou:
                possible.append((-iou, i, j))
    used_ref, used_est = set(), set()
    for _, i, j in sorted(possible):
        if i not in used_ref and j not in used_est:
            used_ref.add(i)
            used_est.add(j)
    return np.array(
        [
            len(used_ref),
            len(estimates) - len(used_est),
            len(references) - len(used_ref),
        ],
        dtype=float,
    )


def bootstrap_cells(rows, *, replicates=1000, seed=20261003):
    """Precision/recall/F1 with equal-probability shot bootstrap draws."""
    cells = np.stack([r["cells"] for r in rows])
    result = point_metrics(cells.sum(axis=0))
    samples = {k: [] for k in result}
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        selected = rng.integers(0, len(rows), size=len(rows))
        for k, v in point_metrics(cells[selected].sum(axis=0)).items():
            if v is not None:
                samples[k].append(v)
    return {
        "metrics": result,
        "cells": cells.sum(axis=0).astype(int).tolist(),
        "ci95": {
            k: np.quantile(v, [0.025, 0.975]).tolist() if v else None
            for k, v in samples.items()
        },
        "replicates": replicates,
        "seed": seed,
    }


def score_histogram(truth, probability):
    truth = np.asarray(truth, dtype=bool)
    probability = np.clip(np.asarray(probability, dtype=float), 0, 1)
    if truth.shape != probability.shape or not np.isfinite(probability).all():
        raise ValueError("one finite probability is required per truth bin")
    index = np.minimum((probability * HISTOGRAM_BINS).astype(int), HISTOGRAM_BINS - 1)
    return np.stack(
        [
            np.bincount(index[truth == state], minlength=HISTOGRAM_BINS)
            for state in (False, True)
        ]
    )


def presence_metrics(histogram, threshold=0.5):
    negative, positive = np.asarray(histogram, dtype=float)
    tp = np.cumsum(positive[::-1])
    fp = np.cumsum(negative[::-1])
    n, p = negative.sum(), positive.sum()
    auroc, auprc = None, None
    if p and n:
        # Within a quantization bin, positive/negative ties get half credit.
        auroc = float(
            (positive * (np.cumsum(negative) - 0.5 * negative)).sum() / (n * p)
        )
    if p:
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=tp + fp > 0)
        auprc = float((precision * positive[::-1] / p).sum())
    cut = int(np.ceil(threshold * HISTOGRAM_BINS))
    cells = [positive[cut:].sum(), negative[cut:].sum(), positive[:cut].sum()]
    return {
        "auroc": auroc,
        "auprc": auprc,
        **point_metrics(cells),
        "accuracy": float((positive[cut:].sum() + negative[:cut].sum()) / (p + n))
        if p + n
        else None,
        "positive_bins": int(p),
        "negative_bins": int(n),
    }


def binary_cells(truth, estimate):
    """Presence TP/FP/FN/TN, after observability and assessment filtering."""
    truth, estimate = np.asarray(truth, bool), np.asarray(estimate, bool)
    if truth.shape != estimate.shape:
        raise ValueError("truth and estimate must have identical shapes")
    return np.array(
        [
            (truth & estimate).sum(),
            (~truth & estimate).sum(),
            (truth & ~estimate).sum(),
            (~truth & ~estimate).sum(),
        ],
        dtype=float,
    )


def presence_from_cells(histogram, cells, threshold=0.5):
    """Retain ranking scores but use each fold's selected operating threshold."""
    scores = presence_metrics(histogram, threshold)
    cells = np.asarray(cells, float)
    tp, fp, fn, tn = cells
    scores.update(point_metrics([tp, fp, fn]))
    scores["accuracy"] = float((tp + tn) / (tp + fp + fn + tn)) if cells.sum() else None
    scores["positive_bins"] = int(tp + fn)
    scores["negative_bins"] = int(tn + fp)
    return scores


def aggregate(rows, *, replicates=1000, seed=20261003, threshold=0.5):
    """Each row is one held-out shot, with event cells and a presence histogram."""
    if not rows:
        return {"shots": 0, "crash": None, "presence": None, "ci95": {}}
    if any(r["cells"] is None for r in rows) and not all(
        r["cells"] is None for r in rows
    ):
        raise ValueError("crash availability must be the same for every shot")
    events = (
        np.stack([r["cells"] for r in rows]) if rows[0]["cells"] is not None else None
    )
    hist = np.stack([r["histogram"] for r in rows])
    presence_cells = (
        np.stack([r["presence_cells"] for r in rows])
        if all("presence_cells" in r for r in rows)
        else None
    )
    crash = point_metrics(events.sum(axis=0)) if events is not None else None
    presence = (
        presence_from_cells(hist.sum(axis=0), presence_cells.sum(axis=0), threshold)
        if presence_cells is not None
        else presence_metrics(hist.sum(axis=0), threshold)
    )
    rng = np.random.default_rng(seed)
    samples = {
        f"{kind}_{name}": []
        for kind, names in (
            ("crash", ("precision", "recall", "f1")),
            ("presence", ("auroc", "auprc", "f1", "precision", "recall", "accuracy")),
        )
        for name in names
    }
    for _ in range(replicates):
        selected = rng.integers(0, len(rows), size=len(rows))
        values = {
            "crash": (
                point_metrics(events[selected].sum(axis=0))
                if events is not None
                else None
            ),
            "presence": (
                presence_from_cells(
                    hist[selected].sum(axis=0),
                    presence_cells[selected].sum(axis=0),
                    threshold,
                )
                if presence_cells is not None
                else presence_metrics(hist[selected].sum(axis=0), threshold)
            ),
        }
        for key, sample in samples.items():
            kind, name = key.split("_", 1)
            value = values[kind][name] if values[kind] is not None else None
            if value is not None:
                sample.append(value)
    return {
        "shots": len(rows),
        "shot_ids": [r["shot"] for r in rows],
        "crash_cells": (
            events.sum(axis=0).astype(int).tolist() if events is not None else None
        ),
        "crash": crash,
        "presence": presence,
        "ci95": {
            k: np.quantile(v, [0.025, 0.975]).tolist() if v else None
            for k, v in samples.items()
        },
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "probability_histogram_bins": HISTOGRAM_BINS,
        "presence_threshold": "per-fold inner-selected"
        if presence_cells is not None
        else threshold,
    }


def paired_bootstrap(
    rows,
    baseline_rows,
    *,
    replicates=1000,
    seed=20261003,
    direction="model minus derivative-only baseline",
):
    """Model minus baseline with the same whole-shot draw on both sides.

    Shots are paired by ID, not input ordering. Crash differences stay undefined
    for a presence-only baseline. Both ranking and operating presence metrics
    use each shot's already-frozen threshold/confusion cells.
    """
    left = {r["shot"]: r for r in rows}
    right = {r["shot"]: r for r in baseline_rows}
    if (
        not left
        or set(left) != set(right)
        or len(left) != len(rows)
        or len(right) != len(baseline_rows)
    ):
        raise ValueError("paired comparison requires the same unique shots")
    shots = sorted(left)
    groups = [[side[shot] for shot in shots] for side in (left, right)]
    arrays = []
    for group in groups:
        arrays.append(
            {
                "events": np.stack([r["cells"] for r in group])
                if all(r["cells"] is not None for r in group)
                else None,
                "histogram": np.stack([r["histogram"] for r in group]),
                "presence_cells": np.stack([r["presence_cells"] for r in group]),
            }
        )

    def differences(selected):
        values = []
        for side in arrays:
            crash = (
                point_metrics(side["events"][selected].sum(axis=0))
                if side["events"] is not None
                else {"precision": None, "recall": None, "f1": None}
            )
            presence = presence_from_cells(
                side["histogram"][selected].sum(axis=0),
                side["presence_cells"][selected].sum(axis=0),
            )
            values.append(
                {
                    **{f"crash_{key}": value for key, value in crash.items()},
                    **{
                        f"presence_{key}": presence[key]
                        for key in (
                            "auroc",
                            "auprc",
                            "f1",
                            "precision",
                            "recall",
                            "accuracy",
                        )
                    },
                }
            )
        return {
            key: values[0][key] - values[1][key]
            if values[0][key] is not None and values[1][key] is not None
            else None
            for key in values[0]
        }

    result = differences(np.arange(len(shots)))
    samples = {key: [] for key in result}
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        selected = rng.integers(0, len(shots), size=len(shots))
        for key, value in differences(selected).items():
            if value is not None:
                samples[key].append(value)
    return {
        "difference": result,
        "ci95": {
            key: np.quantile(sample, [0.025, 0.975]).tolist() if sample else None
            for key, sample in samples.items()
        },
        "shot_ids": shots,
        "shots": len(shots),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "direction": direction,
        "resampling": "identical whole-shot draws for model and baseline",
    }


def spans_at(times_s, intervals):
    out = np.zeros(len(times_s), dtype=bool)
    for start, end in intervals:
        out |= (times_s >= start) & (times_s < end)
    return out
