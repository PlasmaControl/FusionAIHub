"""Shot-separated AE supervision swaps and paired 10 ms frame scores.

The legacy conversion matches ``ae_baselines_evaluate.py`` and the annotation
audit: a frame is annotated when at least half its native columns are annotated.
Dense labels use the catalog's any-touch state rule. Unknown frames have zero
training weight and are excluded from scoring. Frequency supervision is held
fixed across the three activity-supervision arms.
"""

from __future__ import annotations

import hashlib
from itertools import pairwise
from pathlib import Path

import numpy as np

from labeler.events.catalog.states import ABSENT, PRESENT
from labeler.scoring.frames import FRAME_MS, OUTSIDE

RECORD_MS = 2000
N_FRAMES = RECORD_MS // FRAME_MS
SPLIT_SEED = 20261003
BOOTSTRAP_SEED = 20261004
METRICS = ("auroc", "auprc", "f1")
CONVERGENCE_RULE = {
    "applies_to": "every ae-ours record of every arm and seed, uniformly",
    "training": {
        "monitor": "combined loss on the 20 selection shots",
        "max_epochs": 30,
        "patience": 5,
        "patience_start_epoch": 10,
        "min_delta": 1e-6,
        "statement": (
            "Early stopping counts non-improving epochs only from zero-based "
            "epoch 10, so a plateau in the first ten epochs cannot end a run."
        ),
        "conformance": (
            "A record conforms when replaying this rule on its recorded "
            "selection-loss history stops it at the recorded final epoch; a "
            "record that stopped earlier is rerun with the same seed."
        ),
    },
    "screen": {
        "within_shot_sd_min": 0.005,
        "selection_auroc_min_exclusive": 0.5,
        "scope": "selected checkpoint, selection shots, its own activity target",
        "sd_statistic": "mean over selection shots of the population SD of the score",
        "statement": (
            "A record whose selected checkpoint is constant (within-shot SD below "
            "0.005) or does not rank its own target above chance (selection "
            "AUROC at most 0.5) is excluded and replaced by the arm's next unused "
            "seed, starting at 3, which is trained under the same training rule "
            "and screened again."
        ),
    },
}
EPOCH_LIMIT = CONVERGENCE_RULE["training"]["max_epochs"]


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


def replay_early_stop(
    losses: list[float],
    patience: int = CONVERGENCE_RULE["training"]["patience"],
    start: int = CONVERGENCE_RULE["training"]["patience_start_epoch"],
    min_delta: float = CONVERGENCE_RULE["training"]["min_delta"],
    max_epochs: int = EPOCH_LIMIT,
) -> dict:
    """Replay the declared early-stopping rule on a recorded selection-loss trace.

    Mirrors ``ae_train.next_bad_count``: epochs before ``start`` never count.
    """
    best, best_epoch, bad = np.inf, -1, 0
    for epoch, loss in enumerate(losses[:max_epochs]):
        if loss < best - min_delta:
            best, best_epoch, bad = loss, epoch, 0
        elif epoch >= start:
            bad += 1
            if bad >= patience:
                return {"best_epoch": best_epoch, "final_epoch": epoch, "stopped": True}
    return {
        "best_epoch": best_epoch,
        "final_epoch": min(len(losses), max_epochs) - 1,
        "stopped": False,
    }


def training_conformance(losses: list[float], **rule) -> dict:
    """True when the declared rule would have ended this history where it ended."""
    replay = replay_early_stop(losses, **rule)
    limit = rule.get("max_epochs", EPOCH_LIMIT)
    ended = replay["stopped"] or len(losses) >= limit
    conforms = ended and replay["final_epoch"] == len(losses) - 1
    return {**replay, "epochs_recorded": len(losses), "conforms": bool(conforms)}


def convergence_screen(parts: list[tuple]) -> dict:
    """Reject constant or non-discriminating selection outputs uniformly."""
    rule = CONVERGENCE_RULE["screen"]
    sd = float(np.mean([np.std(score) for score, _ in parts]))
    auc = ShotMetric(parts, 0.5).values(np.ones(len(parts)))[0]
    passed = (
        sd >= rule["within_shot_sd_min"] and auc > rule["selection_auroc_min_exclusive"]
    )
    return {
        "passed": bool(passed),
        "mean_within_shot_sd": sd,
        "selection_auroc": _number(auc),
        "n_shots": len(parts),
    }


def any_touch_prevalence(states: dict[int, np.ndarray], shots: list[int]) -> float:
    """Share of present frames among the present and absent frames of `shots`."""
    present = sum(int((states[s] == PRESENT).sum()) for s in shots)
    scored = sum(int(np.isin(states[s], (ABSENT, PRESENT)).sum()) for s in shots)
    return present / scored


def snapshot_difference(
    old: dict[int, np.ndarray],
    new: dict[int, np.ndarray],
    groups: dict[str, list[int]],
) -> dict:
    """Where two label snapshots disagree, per group of shots, in 10 ms frames.

    ``frames`` counts frames whose catalog state differs; ``net_present`` is the
    change in present frames (new minus old), so it is negative when frames were
    cleared.
    """
    out = {}
    for name, shots in groups.items():
        differing = [s for s in shots if not np.array_equal(old[s], new[s])]
        out[name] = {
            "shots": len(shots),
            "differing_shots": sorted(differing),
            "frames": int(sum((old[s] != new[s]).sum() for s in differing)),
            "net_present": int(
                sum(
                    (new[s] == PRESENT).sum() - (old[s] == PRESENT).sum()
                    for s in differing
                )
            ),
        }
    return out


def interval_changing_saves(
    entries: list[dict], after: str | None = None
) -> list[dict]:
    """Saves whose window or intervals differ from the shot's previous save.

    ``entries`` are review-history lines (``shot``, ``saved_at`` ISO text,
    ``window``, ``intervals``, optional ``name``). A shot's first save has no
    predecessor and never counts. ``after`` keeps only saves later than that
    ISO time, but each is still compared with the save just before it.
    """
    by_shot: dict[int, list[dict]] = {}
    for entry in entries:
        by_shot.setdefault(entry["shot"], []).append(entry)
    changes = []
    for shot, saves in by_shot.items():
        saves = sorted(saves, key=lambda e: e["saved_at"])
        for before, now in pairwise(saves):
            same = now["window"] == before["window"] and [
                list(i) for i in now["intervals"]
            ] == [list(i) for i in before["intervals"]]
            if not same and (after is None or now["saved_at"] > after):
                changes.append(
                    {"shot": shot, "name": now.get("name"), "saved_at": now["saved_at"]}
                )
    return sorted(changes, key=lambda c: (c["saved_at"], c["shot"]))


def grid_shift(t_ms: np.ndarray) -> dict:
    """Columns whose true 10 ms frame is not the audit's uniform-centre frame.

    ``frame_index`` assumes native columns are evenly spread over 0 to 2 s; the
    recorded column times (``t_ms``) differ from that by about a millisecond.
    """
    t_ms = np.asarray(t_ms, dtype=float)
    centres = (np.arange(t_ms.size) + 0.5) * RECORD_MS / t_ms.size
    true_frame = np.clip((t_ms // FRAME_MS).astype(int), 0, N_FRAMES - 1)
    moved = true_frame != frame_index(t_ms.size)
    return {
        "columns": int(t_ms.size),
        "columns_in_another_frame": int(moved.sum()),
        "max_abs_offset_ms": float(np.abs(t_ms - centres).max()),
        "first_ms": float(t_ms[0]),
        "last_ms": float(t_ms[-1]),
    }


def clock_prior(states: np.ndarray) -> np.ndarray:
    """Input-free per-bin rate on caller-supplied training/selection shots."""
    states = np.asarray(states)
    if states.ndim != 2 or states.shape[1] != N_FRAMES or not states.shape[0]:
        raise ValueError("clock needs shots by 200 frame states")
    observed = np.isin(states, (ABSENT, PRESENT)).sum(axis=0)
    if (observed == 0).any():
        raise ValueError("every clock bin needs an observed training frame")
    return (states == PRESENT).sum(axis=0) / observed


def within_shot_scores(
    parts: list[tuple], replicates: int = 1000, seed: int = BOOTSTRAP_SEED
) -> dict:
    """Median shot AUROC/AP, restricted to two-class shots, with shot CIs."""
    values = [
        ShotMetric([part], 0.5).values(np.ones(1))[:2]
        for part in parts
        if np.any(part[1]) and not np.all(part[1])
    ]
    result = {"n_two_class_shots": len(values), "statistic": "median across shots"}
    if not values:
        return {**result, **{m: {"value": None, "ci95": None} for m in METRICS[:2]}}
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    samples = np.median(
        values[rng.integers(len(values), size=(replicates, len(values)))], axis=1
    )
    for i, metric in enumerate(METRICS[:2]):
        result[metric] = {
            "value": float(np.median(values[:, i])),
            "ci95": np.percentile(samples[:, i], [2.5, 97.5]).tolist(),
        }
    return result


def paired_scores(
    methods: dict[str, tuple[list[tuple], float]],
    replicates: int = 1000,
    seed: int = BOOTSTRAP_SEED,
    groups: dict[str, list[str]] | None = None,
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
    result = {
        "n_shots": n,
        "n_frames": engines[names[0]].n_frames,
        "n_positive": engines[names[0]].n_positive,
        "bootstrap": {"replicates": replicates, "seed": seed, "unit": "shot"},
        "methods": {name: summarize(point[name], boot[name]) for name in names},
        "paired_differences": pairs,
    }
    if groups:
        if any(
            not members or not set(members).issubset(engines)
            for members in groups.values()
        ):
            raise ValueError("seed groups need nonempty, available members")
        counts = {len(members) for members in groups.values()} - {1}
        if len(counts) > 1:
            raise ValueError("paired groups must have matching seed counts")
        n_seeds = max(counts, default=1)
        seed_draws = rng.integers(n_seeds, size=(replicates, n_seeds))
        group_point, group_boot, group_shot = {}, {}, {}
        for name, members in groups.items():
            values = np.array([point[m] for m in members])
            samples = np.stack([boot[m] for m in members], axis=1)
            if len(members) == 1:
                values = np.repeat(values, n_seeds, axis=0)
                samples = np.repeat(samples, n_seeds, axis=1)
            group_point[name] = values
            group_boot[name] = samples[np.arange(replicates)[:, None], seed_draws].mean(
                axis=1
            )
            # Every seed kept in every draw: uncertainty from the shots alone.
            group_shot[name] = samples.mean(axis=1)

        def seed_summary(values, samples, shot_samples, count):
            summary = summarize(values.mean(axis=0), samples)
            for i, metric in enumerate(METRICS):
                summary[metric]["mean"] = summary[metric].pop("value")
                summary[metric]["sd"] = (
                    _number(values[:, i].std(ddof=1)) if count > 1 else None
                )
                finite = shot_samples[:, i][np.isfinite(shot_samples[:, i])]
                summary[metric]["ci95_shot"] = (
                    np.percentile(finite, [2.5, 97.5]).tolist() if len(finite) else None
                )
            return summary

        group_names = list(groups)
        result["seed_summary"] = {
            "groups": groups,
            "bootstrap": {
                "replicates": replicates,
                "seed": seed,
                "unit": "shot and training seed",
                "n_training_seeds": n_seeds,
                "statistic": "mean of per-seed pooled-frame metrics",
                "pairing": "identical shot draws and seed indices across arms",
                "baseline": "singleton saved models have no training-seed resampling",
                "ci95": "resamples shots and the observed seed IDs",
                "ci95_shot": "resamples shots only; every seed kept in each draw",
            },
            "methods": {
                name: seed_summary(
                    group_point[name],
                    group_boot[name],
                    group_shot[name],
                    len(members),
                )
                for name, members in groups.items()
            },
            "paired_differences": {
                f"{a} minus {b}": seed_summary(
                    group_point[a] - group_point[b],
                    group_boot[a] - group_boot[b],
                    group_shot[a] - group_shot[b],
                    max(len(groups[a]), len(groups[b])),
                )
                for i, a in enumerate(group_names)
                for b in group_names[i + 1 :]
            },
        }
    return result
