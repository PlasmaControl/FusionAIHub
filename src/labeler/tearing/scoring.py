"""Per-bin scores for tearing-mode detectors, and the shot bootstrap around them.

Every detector in the benchmark is judged the same way. A shot is cut into bins of
`BIN_MS` over its catalog window; a bin is positive where its centre is inside a mode
interval (`label_bins`), negative elsewhere, and ignored where the label says uncertain
or not observable. A detector gives each bin a score (`align_scores` puts a model's own
time grid on the bins). Then:

* AUROC and AUPRC are read off per-shot score histograms (`shot_stats`) on one shared
  grid of thresholds, so a bootstrap of the shots costs a matrix product;
* F1 is counted per shot at the threshold the detector was given (`shot_stats`), which
  may differ by shot (a threshold chosen on each fold's own validation shots);
* segmental F1 matches predicted to true intervals by temporal IoU
  (`segment_counts`);
* `bootstrap` resamples whole shots with replacement, 1000 times by default, so no
  bin is treated as independent of its neighbours.

Nothing here knows what a detector is; it takes arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

BIN_MS = 10.0
#: Temporal-IoU levels the segmental F1 is reported at.
TIOUS = (0.3, 0.5, 0.7)
#: A predicted or true stretch shorter than this (ms) is not a segment.
MIN_SEGMENT_MS = 50.0
#: Gaps shorter than this (ms) do not split a predicted segment.
MERGE_GAP_MS = 50.0


def shot_folds(shots, k: int = 5, seed: int = 0) -> dict[int, int]:
    """`{shot: fold}`: a seeded shuffle of the sorted shots dealt out round-robin."""
    ordered = sorted({int(s) for s in shots})
    order = np.random.default_rng(seed).permutation(len(ordered))
    return {ordered[i]: int(rank % k) for rank, i in enumerate(order)}


def inner_split(shots, seed: int, fraction: float = 0.1) -> tuple[list, list]:
    """`(train, validation)`: hold back a seeded fraction of shots, at least one."""
    shots = list(shots)
    order = np.random.default_rng(seed).permutation(len(shots))
    k = max(1, round(fraction * len(shots)))
    return [shots[i] for i in order[k:]], [shots[i] for i in order[:k]]


def shared_cv(shots, k: int = 5, seed: int = 0) -> tuple[dict, list[dict]]:
    """One cohort-wide fold plan, before any detector's input-availability filter.

    Intersect these lists with available inputs only after creating the plan. Thus
    each scored shot has the same held fold and inner role for every detector.
    """
    shots = sorted({int(s) for s in shots})
    folds = shot_folds(shots, k, seed)
    splits = []
    for fold in range(k):
        held = [s for s in shots if folds[s] == fold]
        pool = [s for s in shots if folds[s] != fold]
        train, val = inner_split(pool, 100 + fold)
        splits.append({"fold": fold, "train": train, "validation": val, "held": held})
    return folds, splits


def bin_centres(window, bin_ms: float = BIN_MS) -> np.ndarray:
    """Centres (ms) of the bins wholly inside `window`, on an absolute `bin_ms` grid."""
    lo = np.ceil(window[0] / bin_ms) * bin_ms
    hi = np.floor(window[1] / bin_ms) * bin_ms
    if hi - lo < bin_ms:
        return np.empty(0)
    return np.arange(lo, hi, bin_ms) + bin_ms / 2.0


def _inside(centres, spans) -> np.ndarray:
    mask = np.zeros(len(centres), dtype=bool)
    for a, b in spans:
        mask |= (centres >= a) & (centres <= b)
    return mask


def label_bins(rows: pd.DataFrame, centres) -> tuple[np.ndarray, np.ndarray]:
    """`(y, valid)` over bin `centres` from one shot's interval-table rows.

    `rows` has `category`, `t_start`, `t_end` (the catalog's interval schema: 1
    present, 2 uncertain, 3 not observable, 0 absent; a point has `t_start == t_end`).
    A bin is positive where its centre is in a present span, ignored (`valid` False)
    where it is in an uncertain or not-observable span, negative elsewhere.
    """
    span = rows[rows.t_end > rows.t_start]

    def of(category):
        kept = span[span.category == category]
        return _inside(centres, zip(kept.t_start, kept.t_end, strict=True))

    return of(1).astype(np.int8), ~(of(2) | of(3))


def align_scores(
    t_ms, values, centres, *, shift_ms: float = 0.0, max_gap_ms: float | None = None
) -> np.ndarray:
    """A model's scores on its own grid, linearly interpolated to bin `centres`.

    The score stamped `t` is taken to describe `t + shift_ms` (the onset CNN's output at
    `t` is the mode's presence at `t + 25 ms`). NaN samples are skipped. A bin whose
    two neighbouring samples are more than 1.5 steps apart (a step is `max_gap_ms`, by
    default the grid's median spacing) has no score (NaN), nor has one outside the
    record.
    """
    t = np.asarray(t_ms, dtype=float) + shift_ms
    v = np.asarray(values, dtype=float)
    keep = np.isfinite(t) & np.isfinite(v)
    t, v = t[keep], v[keep]
    out = np.full(len(centres), np.nan)
    if t.size < 2:
        return out
    step = float(np.median(np.diff(t))) if max_gap_ms is None else float(max_gap_ms)
    inside = (centres >= t[0]) & (centres <= t[-1])
    out[inside] = np.interp(centres[inside], t, v)
    nearest = np.searchsorted(t, centres[inside])
    right = np.clip(nearest, 0, t.size - 1)
    left = np.clip(nearest - 1, 0, t.size - 1)
    far = (t[right] - t[left]) > step * 1.5 + 1e-9
    idx = np.flatnonzero(inside)
    out[idx[far]] = np.nan
    return out


def edges_for(scores, size: int = 1024) -> np.ndarray:
    """Thresholds for the histograms: the pooled scores' quantiles, plus both ends."""
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        raise ValueError("no finite scores")
    q = np.quantile(s, np.linspace(0.0, 1.0, size + 1))
    return np.unique(np.concatenate(([s.min() - 1e-9], q, [s.max() + 1e-9])))


@dataclass
class ShotStats:
    """What one shot contributes to every metric (so the shots can be resampled)."""

    shot: int
    n_bins: int
    n_pos: int
    n_neg: int
    pos_hist: np.ndarray
    neg_hist: np.ndarray
    tp: int = 0
    fp: int = 0
    fn: int = 0
    seg: dict[float, tuple[int, int, int]] = field(default_factory=dict)


def _hist(values, edges) -> np.ndarray:
    return np.histogram(values, bins=edges)[0].astype(np.int64)


def segments(
    mask,
    *,
    valid=None,
    bin_ms: float = BIN_MS,
    min_ms: float = MIN_SEGMENT_MS,
    merge_ms: float = MERGE_GAP_MS,
) -> list[tuple[int, int]]:
    """Half-open index runs `[a, b)` of a boolean mask: gaps of at most `merge_ms`
    closed, runs shorter than `min_ms` dropped. Unavailable bins in `valid` are
    hard boundaries, including when their duration is below the gap limit."""
    m = np.asarray(mask, dtype=bool)
    available = (
        np.ones(m.shape, dtype=bool) if valid is None else np.asarray(valid, bool)
    )
    if available.shape != m.shape:
        raise ValueError("mask and valid must have the same shape")
    m = m & available
    edges = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
    runs = list(
        zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True)
    )
    gap = int(np.floor(merge_ms / bin_ms + 1e-9))
    merged: list[list[int]] = []
    for a, b in runs:
        if merged and a - merged[-1][1] <= gap and available[merged[-1][1] : a].all():
            merged[-1][1] = int(b)
        else:
            merged.append([int(a), int(b)])
    need = int(np.ceil(min_ms / bin_ms - 1e-9))
    return [(a, b) for a, b in merged if b - a >= need]


def segment_counts(
    pred, true, tiou: float, *, valid=None, **kw
) -> tuple[int, int, int]:
    """`(tp, fp, fn)` of predicted against true segments at temporal IoU `tiou`.

    Both masks are cut into segments by `segments`; each true segment is matched to at
    most one predicted segment, the best-overlapping first, and a match needs an IoU of
    at least `tiou`. Unavailable bins are hard boundaries and contribute neither
    intersection nor union to IoU. Gap closing includes only available bins.
    """
    p, t = segments(pred, valid=valid, **kw), segments(true, valid=valid, **kw)
    available = (
        np.ones(len(pred), dtype=bool)
        if valid is None
        else np.asarray(valid, dtype=bool)
    )
    pairs = []
    for i, (pa, pb) in enumerate(p):
        for j, (ta, tb) in enumerate(t):
            lo, hi = max(pa, ta), min(pb, tb)
            inter = int(available[lo:hi].sum()) if hi > lo else 0
            if inter > 0:
                union = (
                    int(available[pa:pb].sum()) + int(available[ta:tb].sum()) - inter
                )
                pairs.append((inter / union, i, j))
    pairs.sort(reverse=True)
    used_p, used_t, tp = set(), set(), 0
    for iou, i, j in pairs:
        if iou < tiou:
            break
        if i in used_p or j in used_t:
            continue
        used_p.add(i)
        used_t.add(j)
        tp += 1
    return tp, len(p) - tp, len(t) - tp


def shot_stats(
    shot, y, valid, score, edges, threshold=None, *, tious=TIOUS
) -> ShotStats:
    """One shot's contribution: histograms over `edges`, counts at `threshold`.

    Only bins that are `valid` and carry a finite score count. Segmental counts need
    a `threshold`; they are made on the shot's bins in order with the others left out
    (a bin with no score or an ignored bin is not a segment).
    """
    y = np.asarray(y).astype(bool)
    valid = np.asarray(valid, dtype=bool)
    score = np.asarray(score, dtype=float)
    use = valid & np.isfinite(score)
    pos, neg = score[use & y], score[use & ~y]
    stats = ShotStats(
        int(shot),
        int(use.sum()),
        len(pos),
        len(neg),
        _hist(pos, edges),
        _hist(neg, edges),
    )
    if threshold is not None:
        hit = np.where(use, score >= threshold, False)
        stats.tp = int((hit & use & y).sum())
        stats.fp = int((hit & use & ~y).sum())
        stats.fn = int((~hit & use & y).sum())
        for level in tious:
            stats.seg[level] = segment_counts(hit, y, level, valid=use)
    return stats


def auroc_auprc(pos_hist, neg_hist) -> tuple[float, float]:
    """AUROC (ties half) and average precision from histograms, high score first."""
    pos = np.asarray(pos_hist, dtype=float)[::-1]
    neg = np.asarray(neg_hist, dtype=float)[::-1]
    p, n = pos.sum(), neg.sum()
    if p == 0 or n == 0:
        return float("nan"), float("nan")
    tp, fp = np.cumsum(pos), np.cumsum(neg)
    tpr = np.concatenate(([0.0], tp / p))
    fpr = np.concatenate(([0.0], fp / n))
    auroc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))
    called = tp + fp
    precision = np.divide(tp, called, out=np.ones_like(tp), where=called > 0)
    recall = tp / p
    auprc = float(np.sum(np.diff(np.concatenate(([0.0], recall))) * precision))
    return auroc, auprc


def f1_of(tp, fp, fn) -> float:
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else float("nan")


def metrics(stats: list[ShotStats], weights=None) -> dict:
    """AUROC, AUPRC, F1 at the shots' thresholds, segmental F1, over weighted shots."""
    w = np.ones(len(stats)) if weights is None else np.asarray(weights, dtype=float)
    pos = (w[:, None] * np.stack([s.pos_hist for s in stats])).sum(axis=0)
    neg = (w[:, None] * np.stack([s.neg_hist for s in stats])).sum(axis=0)
    auroc, auprc = auroc_auprc(pos, neg)
    tp = float((w * [s.tp for s in stats]).sum())
    fp = float((w * [s.fp for s in stats]).sum())
    fn = float((w * [s.fn for s in stats]).sum())
    out = {
        "auroc": auroc,
        "auprc": auprc,
        "f1": f1_of(tp, fp, fn),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "prevalence": float(pos.sum() / (pos.sum() + neg.sum())),
    }
    for level in sorted({lv for s in stats for lv in s.seg}):
        counts = np.array([s.seg.get(level, (0, 0, 0)) for s in stats], dtype=float)
        a, b, c = (w[:, None] * counts).sum(axis=0)
        out[f"segf1_{level:g}"] = f1_of(a, b, c)
    return out


def bootstrap(
    stats: list[ShotStats], *, n: int = 1000, seed: int = 0, level: float = 0.95
) -> dict:
    """Point estimates and `level` percentile intervals, resampling whole shots.

    `{metric: {"value", "lo", "hi"}}`, plus `n_shots` and `replicates`. A replicate in
    which a metric is undefined (no positive bins drawn) is left out of that metric's
    interval, and `valid_replicates` says how many were kept.
    """
    if not stats:
        raise ValueError("no shots")
    point = metrics(stats)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        counts = np.bincount(
            rng.integers(0, len(stats), len(stats)), minlength=len(stats)
        )
        draws.append(metrics(stats, counts))
    alpha = (1.0 - level) / 2.0 * 100.0
    out: dict = {"n_shots": len(stats), "replicates": n}
    for key, value in point.items():
        boot = np.array([d[key] for d in draws], dtype=float)
        boot = boot[np.isfinite(boot)]
        out[key] = {
            "value": value,
            "lo": float(np.percentile(boot, alpha)) if boot.size else float("nan"),
            "hi": float(np.percentile(boot, 100.0 - alpha))
            if boot.size
            else float("nan"),
            "valid_replicates": int(boot.size),
        }
    return out


def best_threshold(stats: list[ShotStats], edges) -> float:
    """Highest candidate edge attaining maximal pooled validation F1.

    With no positive validation bins every candidate has zero F1 in this search,
    so the deterministic tie convention returns the highest histogram lower edge.
    That fallback is not an estimable F1 optimum and must be identified as such.
    """
    pos = np.sum([s.pos_hist for s in stats], axis=0)[::-1].cumsum()
    neg = np.sum([s.neg_hist for s in stats], axis=0)[::-1].cumsum()
    total = float(np.sum([s.pos_hist for s in stats]))
    f1 = np.where(pos + neg > 0, 2 * pos / np.maximum(pos + neg + total, 1e-12), 0.0)
    k = int(np.argmax(f1))
    # cumulative from the top: the k-th reversed bin starts at edges[len(edges) - 2 - k]
    return float(np.asarray(edges)[len(edges) - 2 - k])


def cv_thresholds(cohort_shots, y, valid, score, *, edges=None) -> tuple[dict, list]:
    """F1 thresholds using only each shared fold's inner validation shots.

    Histogram edges, if not supplied, are derived separately from that validation
    subset. Missing inputs are excluded; held-fold labels never choose thresholds.
    """
    _, splits = shared_cv(cohort_shots)
    thresholds, info = {}, []
    for split in splits:
        val = [
            s
            for s in split["validation"]
            if s in score and (np.asarray(valid[s]) & np.isfinite(score[s])).any()
        ]
        if not val:
            raise ValueError(f"no observable validation scores in fold {split['fold']}")
        pooled = np.concatenate(
            [
                np.asarray(score[s])[np.asarray(valid[s]) & np.isfinite(score[s])]
                for s in val
            ]
        )
        use_edges = edges
        if use_edges is None:
            use_edges = edges_for(pooled)
        stats = [shot_stats(s, y[s], valid[s], score[s], use_edges) for s in val]
        threshold = best_threshold(stats, use_edges)
        held = [s for s in split["held"] if s in score]
        thresholds.update({s: threshold for s in held})
        info.append(
            {
                **split,
                "validation_scored": val,
                "held_scored": held,
                "validation_bins_scored": sum(s.n_bins for s in stats),
                "validation_bins_positive": sum(s.n_pos for s in stats),
                "validation_bins_negative": sum(s.n_neg for s in stats),
                "validation_shots_positive": [s.shot for s in stats if s.n_pos],
                "threshold_estimable": bool(sum(s.n_pos for s in stats)),
                "threshold_status": (
                    "inner-validation F1 optimum"
                    if any(s.n_pos for s in stats)
                    else "no positive validation bins; deterministic highest-edge "
                    "fallback, not an estimable F1 optimum"
                ),
                "validation_score_quantiles": np.quantile(
                    pooled, [0.0, 0.05, 0.5, 0.95, 1.0]
                ).tolist(),
                "threshold": threshold,
            }
        )
    return thresholds, info


def evaluate(
    shots,
    y: dict,
    valid: dict,
    score: dict,
    threshold,
    *,
    edges=None,
    n: int = 1000,
    seed: int = 0,
    tious=TIOUS,
    bin_ms: float = BIN_MS,
) -> dict:
    """The benchmark's numbers for one detector on `shots`, with shot-bootstrap CIs.

    `y`, `valid` and `score` map a shot to its per-bin arrays (`label_bins`,
    `align_scores`); `threshold` is one number or a map from shot to the threshold its
    fold chose. A shot with no scored bin is left out and listed. `edges` default to
    the pooled scores' quantiles. `tious=()` leaves the segmental F1 out (for scores on
    a model's own rows, which are not `bin_ms` apart, `bin_ms` is only recorded).
    """
    pooled = np.concatenate([np.asarray(score[s], dtype=float) for s in shots])
    edges = edges_for(pooled) if edges is None else np.asarray(edges, dtype=float)
    stats, empty = [], []
    for s in shots:
        thr = threshold[s] if isinstance(threshold, dict) else threshold
        st = shot_stats(s, y[s], valid[s], score[s], edges, thr, tious=tious)
        (stats if st.n_bins else empty).append(st if st.n_bins else int(s))
    out = bootstrap(stats, n=n, seed=seed)
    out["shots_without_a_scored_bin"] = empty
    out["bins_scored"] = int(sum(s.n_bins for s in stats))
    out["bins_positive"] = int(sum(s.n_pos for s in stats))
    out["bin_ms"] = bin_ms
    out["scored_seconds"] = out["bins_scored"] * bin_ms / 1000.0
    out["segment_missing_data_policy"] = (
        "unavailable label or score bins are hard barriers to merging; "
        "excluded from IoU"
    )
    return out
