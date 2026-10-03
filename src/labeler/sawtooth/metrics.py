"""Shot-grouped crash and presence scores, with efficient shot bootstraps."""

from __future__ import annotations

import numpy as np

from ..scoring.events import match

HISTOGRAM_BINS = 512


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
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
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


def aggregate(rows, *, replicates=1000, seed=20261003, threshold=0.5):
    """Each row is one held-out shot, with event cells and a presence histogram."""
    if not rows:
        return {"shots": 0, "crash": None, "presence": None, "ci95": {}}
    events = np.stack([r["cells"] for r in rows])
    hist = np.stack([r["histogram"] for r in rows])
    crash = point_metrics(events.sum(axis=0))
    presence = presence_metrics(hist.sum(axis=0), threshold)
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
            "crash": point_metrics(events[selected].sum(axis=0)),
            "presence": presence_metrics(hist[selected].sum(axis=0), threshold),
        }
        for key, sample in samples.items():
            kind, name = key.split("_", 1)
            value = values[kind][name]
            if value is not None:
                sample.append(value)
    return {
        "shots": len(rows),
        "shot_ids": [r["shot"] for r in rows],
        "crash_cells": events.sum(axis=0).astype(int).tolist(),
        "crash": crash,
        "presence": presence,
        "ci95": {
            k: np.quantile(v, [0.025, 0.975]).tolist() if v else None
            for k, v in samples.items()
        },
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "probability_histogram_bins": HISTOGRAM_BINS,
        "presence_threshold": threshold,
    }


def spans_at(times_s, intervals):
    out = np.zeros(len(times_s), dtype=bool)
    for start, end in intervals:
        out |= (times_s >= start) & (times_s < end)
    return out
