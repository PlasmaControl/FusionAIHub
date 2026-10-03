"""Shot-separated AE supervision swaps and paired 10 ms frame scores.

The legacy conversion matches ``ae_baselines_evaluate.py`` and the annotation
audit: a frame is annotated when at least half its native columns are annotated.
Dense labels use the catalog's any-touch state rule. Unknown frames have zero
training weight and are excluded from scoring. Frequency supervision is held
fixed across the three activity-supervision arms.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from labeler.events.catalog.states import ABSENT, PRESENT
from labeler.scoring.frames import FRAME_MS, OUTSIDE

RECORD_MS = 2000
N_FRAMES = RECORD_MS // FRAME_MS
SPLIT_SEED = 20261003
BOOTSTRAP_SEED = 20261004
METRICS = ("auroc", "auprc", "f1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_index(n: int) -> np.ndarray:
    """Audit grid: native-column centres evenly span the 0--2 s record."""
    if n <= 0:
        raise ValueError("a record needs columns")
    centres = (np.arange(n) + 0.5) * RECORD_MS / n
    return np.minimum((centres // FRAME_MS).astype(int), N_FRAMES - 1)


def frame_mean(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("expected finite one-dimensional column values")
    index = frame_index(values.size)
    count = np.bincount(index, minlength=N_FRAMES)
    if (count == 0).any():
        raise ValueError("native record does not cover every 10 ms frame")
    return np.bincount(index, weights=values, minlength=N_FRAMES) / count


def dense_states(label) -> np.ndarray:
    """Catalog frame states, allowing overlapping individual/crowd spans."""
    starts = np.arange(N_FRAMES) * FRAME_MS
    inside = (starts >= label.window[0]) & (starts + FRAME_MS <= label.window[1])
    states = np.where(inside, ABSENT, OUTSIDE).astype(np.int8)
    for start, stop, category in label.intervals:
        touched = inside & (starts < stop) & (starts + FRAME_MS > start)
        states[touched] = np.maximum(states[touched], category)
    return states


def clean_split(
    train: list[int], valid: list[int], gold: set[int], seed: int = SPLIT_SEED
) -> dict[str, list[int]]:
    """Fixed 100/20 split of the original 120; evaluation stays untouched."""
    if len(train) != 120 or len(valid) != 60:
        raise ValueError("expected the original 120 train / 60 validation shots")
    if len(set(train)) != 120 or len(set(valid)) != 60 or set(train) & set(valid):
        raise ValueError("shot splits must be unique and disjoint")
    if (set(train) | set(valid)) & gold:
        raise ValueError("AE shots overlap the blind gold cohort")
    ordered = np.random.default_rng(seed).permutation(sorted(train)).tolist()
    return {
        "train": sorted(ordered[:100]),
        "selection": sorted(ordered[100:]),
        "evaluation": sorted(valid),
    }


def validate_split(split: dict, gold: set[int]) -> None:
    groups = [split[k] for k in ("train", "selection", "evaluation")]
    if [len(g) for g in groups] != [100, 20, 60]:
        raise ValueError("swap split must contain 100/20/60 shots")
    flat = [s for group in groups for s in group]
    if len(set(flat)) != len(flat) or set(flat) & gold:
        raise ValueError("overlapping shots or blind gold leakage in manifest")


def activity_targets(
    active: np.ndarray,
    annotated: np.ndarray,
    freq_khz: np.ndarray,
    supervision: str,
    dense: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Native-frame y/w/ft/fw; only activity supervision changes between arms.

    Legacy/dense 10 ms targets are expanded onto the existing native input grid.
    The threeway arm retains the original native annotation/TokEye agreement.
    All arms keep the original frequency target and annotation&active weight.
    """
    active, annotated = active.astype(bool), annotated.astype(bool)
    if not (active.shape == annotated.shape == freq_khz.shape):
        raise ValueError("native label arrays differ in shape")
    agreement = active == annotated
    if supervision == "threeway":
        y = active & annotated
        weight = agreement
    elif supervision == "legacy":
        y = (frame_mean(annotated) >= 0.5)[frame_index(len(annotated))]
        weight = np.ones_like(y, dtype=bool)
    elif supervision == "dense":
        if dense is None or dense.shape != (N_FRAMES,):
            raise ValueError("dense supervision needs 200 frame states")
        states = dense[frame_index(len(annotated))]
        y = states == PRESENT
        weight = np.isin(states, (ABSENT, PRESENT))
    else:
        raise ValueError(f"unknown supervision {supervision!r}")
    finite = np.isfinite(freq_khz)
    ft = np.where(finite, (freq_khz - 80.0) / 170.0, 0.0)
    fw = annotated & active & finite
    return tuple(a.astype(np.float32) for a in (y, weight, ft, fw))


def selection_threshold(scores: np.ndarray, truth: np.ndarray) -> dict:
    """Maximise selection F1; exact ties go to the highest threshold."""
    scores, truth = np.asarray(scores), np.asarray(truth, dtype=bool)
    if scores.shape != truth.shape or not scores.size:
        raise ValueError("selection scores and truth must be nonempty and aligned")
    if not np.isfinite(scores).all() or not truth.any():
        raise ValueError("selection needs finite scores and at least one positive")
    order = np.argsort(-scores, kind="stable")
    last = np.r_[scores[order][1:] != scores[order][:-1], True]
    tp = np.cumsum(truth[order])[last]
    taken = (np.arange(len(order)) + 1)[last]
    f1 = 2 * tp / (taken + truth.sum())
    best = int(np.argmax(f1))
    return {
        "threshold": float(scores[order][last][best]),
        "f1": float(f1[best]),
        "n_frames": len(scores),
        "n_positive": int(truth.sum()),
        "rule": "maximum selection F1; ties choose the highest threshold",
    }


class ShotMetric:
    """Pooled frame metrics under shot multiplicities, with cached score order.

    Each part is (score, binary truth) for one shot. Sorting once lets bootstrap
    draws reweight entire shots without sorting repeated copies of their frames.
    """

    def __init__(self, parts: list[tuple], threshold: float):
        if not parts or any(len(s) == 0 for s, _ in parts):
            raise ValueError("each scored shot needs frames")
        score = np.concatenate([p[0] for p in parts]).astype(float)
        truth = np.concatenate([p[1] for p in parts]).astype(bool)
        if not np.isfinite(score).all() or score.shape != truth.shape:
            raise ValueError("scores must be finite and aligned with truth")
        self.n_shots = len(parts)
        self.n_frames = len(score)
        self.n_positive = int(truth.sum())
        ids = np.repeat(np.arange(len(parts)), [len(p[0]) for p in parts])
        order = np.argsort(-score, kind="stable")
        self.ids = ids[order]
        self.truth = truth[order]
        self.hard = score[order] >= threshold
        self.starts = np.r_[0, np.flatnonzero(np.diff(score[order])) + 1]

    def values(self, multiplicity: np.ndarray) -> np.ndarray:
        multiplicity = np.asarray(multiplicity, dtype=float)
        if multiplicity.shape != (self.n_shots,) or (multiplicity < 0).any():
            raise ValueError("one nonnegative multiplicity is needed per shot")
        weights = multiplicity[self.ids]
        pos = np.add.reduceat(weights * self.truth, self.starts)
        neg = np.add.reduceat(weights * ~self.truth, self.starts)
        p, n = pos.sum(), neg.sum()
        cp, cn = np.cumsum(pos), np.cumsum(neg)
        auc = (pos * (n - cn + 0.5 * neg)).sum() / (p * n) if p and n else np.nan
        precision = np.divide(cp, cp + cn, out=np.zeros_like(cp), where=cp + cn > 0)
        ap = (pos * precision).sum() / p if p else np.nan
        tp = (weights * self.truth * self.hard).sum()
        pred = (weights * self.hard).sum()
        f1 = 2 * tp / (p + pred) if p + pred else 0.0
        return np.array([auc, ap, f1], dtype=float)


def _number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def paired_scores(
    methods: dict[str, tuple[list[tuple], float]],
    replicates: int = 1000,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """95% shot-bootstrap CIs and all paired differences on identical frames.

    Callers must supply the same ordered shots/frame masks to every method.
    This is verified by per-shot truth equality before computing paired draws.
    """
    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    first = next(iter(methods.values()))[0]
    for parts, _ in methods.values():
        if len(parts) != len(first) or any(
            not np.array_equal(p[1], q[1]) for p, q in zip(parts, first, strict=True)
        ):
            raise ValueError("paired methods need identical ordered shot truths")
    engines = {name: ShotMetric(parts, t) for name, (parts, t) in methods.items()}
    rng = np.random.default_rng(seed)
    n = len(first)
    draws = np.array(
        [np.bincount(rng.integers(n, size=n), minlength=n) for _ in range(replicates)],
        dtype=float,
    )
    point = {name: e.values(np.ones(n)) for name, e in engines.items()}
    boot = {name: np.array([e.values(d) for d in draws]) for name, e in engines.items()}

    def summarize(value, samples):
        result = {}
        for i, metric in enumerate(METRICS):
            finite = samples[:, i][np.isfinite(samples[:, i])]
            result[metric] = {
                "value": _number(value[i]),
                "ci95": np.percentile(finite, [2.5, 97.5]).tolist()
                if len(finite)
                else None,
                "valid_replicates": len(finite),
            }
        return result

    names = list(engines)
    pairs = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            pairs[f"{a} minus {b}"] = summarize(point[a] - point[b], boot[a] - boot[b])
    return {
        "n_shots": n,
        "n_frames": engines[names[0]].n_frames,
        "n_positive": engines[names[0]].n_positive,
        "bootstrap": {"replicates": replicates, "seed": seed, "unit": "shot"},
        "methods": {name: summarize(point[name], boot[name]) for name in names},
        "paired_differences": pairs,
    }
