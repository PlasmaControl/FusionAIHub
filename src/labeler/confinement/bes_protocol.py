"""Window selection, shot splits and scoring for the BES confinement classifier.

The pieces of Gill et al. (2024)'s protocol that the first retrain left out, each
switchable on its own so the score gap can be attributed factor by factor:

* beam gating: the 150L beam at or above 700 kW and the 150R beam at or below 200 kW
  over the window (400 kW for WP QH windows in training only);
* transition exclusion: no window within a margin of either end of its labelled
  interval, and none in the first stretch of a high-confinement interval that follows
  L-mode (the pedestal is still building);
* a per-shot check that the channels of the 6 x 8 block carry a signal;
* the split: shot-grouped cross-validation, or the paper's split by discharge,
  stratified by each discharge's dominant regime, 72.5 / 15 / 12.5 %.

Scores are per window; a shot bootstrap needs only each shot's confusion matrix, so
``confusion_by_shot`` and ``bootstrap`` never revisit the windows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CLASSES = ("L", "H", "QH", "WP")
WP = 3
#: Gill et al.: the 150L beam views the BES array and must be on; the 150R beam spoils
#: the view.
GATE_LEFT_W = 700e3
GATE_LEFT_WP_TRAIN_W = 400e3
GATE_RIGHT_W = 200e3
#: A high-confinement interval starting within this gap of an L-mode one follows an L-H
#: transition.
CONTIGUOUS_MS = 10.0
WINDOW_MS_1MHZ = 1.024


def window_masks(
    table: pd.DataFrame,
    *,
    gate: bool,
    transition_ms: float = 0.0,
    buildup_ms: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks of the windows kept for training and for scoring.

    ``table`` is ``bes_windows.load_windows``'s. A window with no beam record fails the
    gate. The two masks differ only in the WP QH gate (400 kW in training, 700 kW in
    test).
    """
    n = len(table)
    train = np.ones(n, dtype=bool)
    score = np.ones(n, dtype=bool)
    if gate:
        right = (table.p15R_max <= GATE_RIGHT_W).to_numpy()
        left = table.p15L_min
        wp = (table.label == WP).to_numpy()
        score &= right & (left >= GATE_LEFT_W).to_numpy()
        train &= right & np.where(wp, left >= GATE_LEFT_WP_TRAIN_W, left >= GATE_LEFT_W)
    end = table.start_ms + table.dt_ms * 1024
    keep = np.ones(n, dtype=bool)
    if transition_ms > 0:
        keep &= ((table.start_ms - table.t_start) >= transition_ms).to_numpy()
        keep &= ((table.t_end - end) >= transition_ms).to_numpy()
    if buildup_ms > 0:
        after_l = (
            (table.prev_label == 0)
            & (table.gap_ms <= CONTIGUOUS_MS)
            & (table.label != 0)
        ).to_numpy()
        keep &= ~(after_l & ((table.start_ms - table.t_start) < buildup_ms).to_numpy())
    return train & keep, score & keep


def live_shots(
    table: pd.DataFrame,
    power: np.ndarray,
    channels: slice,
    *,
    dead_ratio: float = 0.01,
    max_dead_fraction: float = 0.1,
) -> set[int]:
    """Shots whose channels in ``channels`` carry a signal.

    A channel is dead on a shot when its median window power is below ``dead_ratio`` of
    the median over that block's channels; a shot passes with at most
    ``max_dead_fraction`` dead.
    """
    block = power[:, channels].astype(np.float64)
    keep = set()
    for shot, idx in table.groupby("shot").indices.items():
        med = np.median(block[idx], axis=0)
        ref = np.median(med)
        dead = (~np.isfinite(med)) | (med <= dead_ratio * ref)
        if dead.mean() <= max_dead_fraction:
            keep.add(int(shot))
    return keep


def deal(shots: list[int], signature: dict[int, str], k: int, rng) -> dict[int, int]:
    """Deal shots into k groups, shuffled inside each class-presence stratum.

    One running counter runs over the strata (largest first), so a small stratum
    spreads over the groups instead of piling into the first.
    """
    strata: dict[str, list[int]] = {}
    for s in shots:
        strata.setdefault(signature[s], []).append(s)
    groups, count = {}, 0
    for sig in sorted(strata, key=lambda g: (-len(strata[g]), g)):
        members = strata[sig]
        for j in rng.permutation(len(members)):
            groups[members[j]] = count % k
            count += 1
    return groups


def run_day_groups(dates: pd.DataFrame) -> dict[int, int]:
    """Each shot's run-day group, from the time its EFIT reconstruction was inserted.

    ``dates`` has ``shot``, ``inserted_local`` and ``consistent`` (see
    ``scripts/labeler/confinement_shot_dates_fetch.py``). ``inserted_local`` is the
    stamp MDSplus recorded, in the local (Pacific) time of the server that wrote it,
    with no time zone: the stamps fall between 08:00 and 21:59, none in the night
    hours, so they are not UTC. Shots sharing a local calendar day share a group; a
    shot without a trustworthy date joins the group of the nearest earlier shot that
    has one (the next one, for the first shots). Shots of one run day share plasma
    conditions, so folds that keep a day whole separate the sessions.
    """
    frame = dates.sort_values("shot").reset_index(drop=True)
    stamp = pd.to_datetime(frame.inserted_local, errors="coerce")
    day = stamp.dt.strftime("%Y-%m-%d").where(frame.consistent.astype(bool))
    day = day.ffill().bfill()
    codes = {d: i for i, d in enumerate(sorted(day.dropna().unique()))}
    return {int(s): codes[d] for s, d in zip(frame.shot, day, strict=True)}


def deal_groups(
    shots: list[int],
    group_of: dict[int, int],
    signature: dict[int, str],
    k: int,
    rng,
) -> dict[int, int]:
    """Deal whole groups of shots into k folds, balancing each class across the folds.

    A group's classes are those of its shots (``signature`` is a comma-separated class
    string per shot). Groups holding the rarest class go first; each goes to the fold
    that so far holds the smallest share of the classes the group carries, ties broken
    by the fold's shot count and then at random.
    """
    members: dict[int, list[int]] = {}
    for s in shots:
        members.setdefault(group_of[s], []).append(s)
    classes = {
        g: {c for s in m for c in signature[s].split(",") if c}
        for g, m in members.items()
    }
    total = {c: sum(1 for s in shots if c in signature[s].split(",")) for c in CLASSES}
    order = sorted(
        members,
        key=lambda g: (
            min(total[c] for c in classes[g]) if classes[g] else 0,
            -len(members[g]),
            rng.random(),
        ),
    )
    held = np.zeros((k, len(CLASSES)))
    count = np.zeros(k)
    fold_of: dict[int, int] = {}
    for g in order:
        mask = np.array([c in classes[g] for c in CLASSES])
        share = (
            held[:, mask] / np.maximum(np.array(list(total.values()))[mask], 1)
        ).sum(axis=1)
        best = min(range(k), key=lambda f: (round(share[f], 9), count[f], rng.random()))
        for s in members[g]:
            fold_of[s] = best
            for i, c in enumerate(CLASSES):
                held[best, i] += c in signature[s].split(",")
        count[best] += len(members[g])
    return fold_of


def class_presence(table: pd.DataFrame) -> dict[int, str]:
    """Each shot's set of classes, as a string key for stratified dealing."""
    return {
        int(s): "".join(CLASSES[i] + "," for i in sorted(set(g.label)))
        for s, g in table.groupby("shot")
    }


def cv_roles(
    table: pd.DataFrame,
    fold: int,
    *,
    folds: int = 5,
    val_groups: int = 6,
    seed: int = 20261001,
) -> np.ndarray:
    """Role of each window (0 train, 1 validation, 2 test) in cross-validation fold.

    Shot-grouped: folds are dealt within class-presence strata; a fold's test shots are
    scored by a network trained on the other folds and validated on a sixth of their
    shots.
    """
    signature = class_presence(table)
    shots = sorted(signature)
    fold_of = deal(shots, signature, folds, np.random.default_rng(seed))
    rest = [s for s in shots if fold_of[s] != fold]
    val = deal(rest, signature, val_groups, np.random.default_rng(seed + 1 + fold))
    role = {s: 2 if fold_of[s] == fold else (1 if val[s] == 0 else 0) for s in shots}
    return table.shot.map(role).to_numpy()


def dominant_regime(
    table: pd.DataFrame, mask: np.ndarray | None = None
) -> dict[int, int]:
    """Each shot's regime with the most windows (among ``mask``, else all)."""
    frame = table if mask is None else table[mask]
    counts = frame.groupby(["shot", "label"]).size().unstack(fill_value=0)
    return {int(s): int(counts.loc[s].to_numpy().argmax()) for s in counts.index}


def paper_roles(
    table: pd.DataFrame,
    dominant: dict[int, int],
    seed: int,
    *,
    test_fraction: float = 0.125,
    val_fraction: float = 0.15,
) -> np.ndarray:
    """The paper's split by discharge, stratified by dominant regime (72.5/15/12.5 %).

    Roles (0 train, 1 validation, 2 test) per window; a shot without a dominant regime
    (no window kept) is test-neutral and goes to train.
    """
    rng = np.random.default_rng(seed)
    role: dict[int, int] = {}
    for cls in sorted(set(dominant.values())):
        members = sorted(s for s, c in dominant.items() if c == cls)
        order = [members[i] for i in rng.permutation(len(members))]
        n_test = max(1, round(test_fraction * len(order))) if len(order) >= 3 else 0
        n_val = max(1, round(val_fraction * len(order))) if len(order) >= 3 else 0
        for j, s in enumerate(order):
            role[s] = 2 if j < n_test else (1 if j < n_test + n_val else 0)
    return table.shot.map(lambda s: role.get(int(s), 0)).to_numpy()


def block_roles(
    table: pd.DataFrame,
    seed: int,
    *,
    block: int = 50,
    test_fraction: float = 0.125,
    val_fraction: float = 0.15,
) -> np.ndarray:
    """A within-shot split (a leaky diagnostic, not a benchmark): runs of ``block``
    consecutive windows of one shot go to train, validation or test at random, so a test
    window has training windows of its own shot beside it.
    """
    idx = table.groupby("shot").cumcount().to_numpy() // block
    _, inverse = np.unique(
        table.shot.to_numpy().astype(np.int64) * 100_000 + idx, return_inverse=True
    )
    draw = np.random.default_rng(seed).random(inverse.max() + 1)
    role = np.where(
        draw < test_fraction,
        2,
        np.where(draw < test_fraction + val_fraction, 1, 0),
    )
    return role[inverse]


def window_keys(shots: np.ndarray, start_ms: np.ndarray) -> np.ndarray:
    """An integer id for each distinct (shot, window start) pair, in order of first
    appearance; the same window in two repeated test sets gets one id."""
    key = pd.MultiIndex.from_arrays(
        [np.asarray(shots), np.round(np.asarray(start_ms, dtype=float), 3)]
    )
    return key.factorize()[0]


def repeat_overlap(
    shots: np.ndarray, start_ms: np.ndarray, truth: np.ndarray, splits: np.ndarray
) -> dict:
    """How far the test sets of repeated random splits overlap.

    The paper's split draws a test set at random in each repeat, so a shot can be a test
    shot in several; pooling the repeats counts its windows once per repeat. Returns the
    number of (shot, repeat) tests, the distinct shots and how many repeats tested each,
    the pooled and the distinct windows, and the same per class.
    """
    shots, truth, splits = map(np.asarray, (shots, truth, splits))
    tests = pd.MultiIndex.from_arrays([shots, splits]).unique()
    times = pd.Series(tests.get_level_values(0)).value_counts()
    key = window_keys(shots, start_ms)
    classes = {}
    for i, c in enumerate(CLASSES):
        m = truth == i
        classes[c] = {
            "windows": int(m.sum()),
            "distinct_windows": int(np.unique(key[m]).size),
            "shots": int(np.unique(shots[m]).size),
        }
    return {
        "tests": len(tests),
        "shots": int(times.size),
        "shots_by_times_tested": {
            int(k): int(v) for k, v in times.value_counts().sort_index().items()
        },
        "windows": len(shots),
        "distinct_windows": int(np.unique(key).size),
        "classes": classes,
    }


def first_occurrence(shots: np.ndarray, splits: np.ndarray) -> np.ndarray:
    """Mask keeping each shot's windows from the first repeat that tested it (first in
    the sorted split names, ``split0`` before ``split1``), so every shot and window is
    counted once."""
    frame = pd.DataFrame({"shot": np.asarray(shots), "split": np.asarray(splits)})
    return (frame.split == frame.groupby("shot").split.transform("min")).to_numpy()


def average_repeats(
    shots: np.ndarray, start_ms: np.ndarray, probs: np.ndarray, truth: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One row per distinct window with the class probabilities averaged over the
    repeats that tested it: shots, window starts, probabilities, truth."""
    shots, truth = np.asarray(shots), np.asarray(truth)
    start_ms = np.asarray(start_ms, dtype=float)
    key = window_keys(shots, start_ms)
    n = int(key.max()) + 1
    count = np.bincount(key, minlength=n)
    mean = (
        np.stack(
            [np.bincount(key, weights=probs[:, j], minlength=n) for j in range(4)],
            axis=1,
        )
        / count[:, None]
    )
    first = np.full(n, -1)
    first[key[::-1]] = np.arange(len(key))[::-1]
    if not np.array_equal(truth, truth[first][key]):
        raise ValueError("a window carries different labels in two repeats")
    return shots[first], start_ms[first], mean, truth[first]


def failure_blocks(
    guess: np.ndarray,
    truth: np.ndarray,
    shots: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    below: float = 0.10,
    min_windows: int = 150,
) -> pd.DataFrame:
    """The (shot, class) blocks a classifier gets almost entirely wrong.

    A block is the windows of one true class on one shot; it is listed when it has at
    least ``min_windows`` windows (consecutive windows start 2,048 samples apart, 2.05
    ms at 1 MHz, so 150 windows span about 0.31 s) and fewer than ``below`` of them are
    called right. Columns: shot, label, windows, correct, share_correct, called (the
    class most of the windows were called, by index) and called_share.
    """
    guess, truth, shots = map(np.asarray, (guess, truth, shots))
    if mask is not None:
        guess, truth, shots = guess[mask], truth[mask], shots[mask]
    k = len(CLASSES)
    ids, inverse = np.unique(shots * k + truth, return_inverse=True)
    counts = np.bincount(inverse * k + guess, minlength=ids.size * k).reshape(-1, k)
    windows = counts.sum(axis=1)
    label = ids % k
    correct = counts[np.arange(ids.size), label]
    called = counts.argmax(axis=1)
    frame = pd.DataFrame(
        {
            "shot": ids // k,
            "label": label,
            "windows": windows,
            "correct": correct,
            "share_correct": correct / windows,
            "called": called,
            "called_share": counts.max(axis=1) / windows,
        }
    )
    keep = (frame.windows >= min_windows) & (frame.share_correct < below)
    return frame[keep].sort_values("windows", ascending=False).reset_index(drop=True)


def confusion_by_shot(
    pred: np.ndarray,
    truth: np.ndarray,
    shots: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Shot ids and their 4 x 4 confusion matrices (rows true) over the windows in
    ``mask``."""
    k = len(CLASSES)
    if mask is not None:
        pred, truth, shots = pred[mask], truth[mask], shots[mask]
    ids, inverse = np.unique(shots, return_inverse=True)
    flat = inverse * k * k + truth * k + pred
    conf = np.bincount(flat, minlength=ids.size * k * k).reshape(ids.size, k, k)
    return ids, conf


def f1_from_conf(conf: np.ndarray) -> np.ndarray:
    """Per-class F1 of a 4 x 4 confusion matrix, NaN for a class with no support."""
    tp = np.diag(conf).astype(float)
    support = conf.sum(1)
    denom = conf.sum(0) + support
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = 2 * tp / denom
    return np.where(support > 0, f1, np.nan)


def macro_f1(conf: np.ndarray) -> float:
    """Mean F1 over the classes that have windows (the paper's "average")."""
    f1 = f1_from_conf(conf)
    return float(np.nanmean(f1)) if np.isfinite(f1).any() else float("nan")


#: The paper's test set, labelled seconds per class (L, H, QH, WP QH).
PAPER_TEST_MIX = (8.3, 21.8, 17.4, 9.7)


def reweight_to_mix(conf: np.ndarray, mix: tuple[float, ...] = PAPER_TEST_MIX):
    """The 4 x 4 confusion matrix with each true class's row rescaled so the classes
    stand in the proportions ``mix`` (the total unchanged): the score the same
    classifier would have on a test set with that class mix.
    """
    conf = np.asarray(conf, dtype=float)
    support = conf.sum(axis=1)
    want = np.asarray(mix, dtype=float) / np.sum(mix) * conf.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(support > 0, want / support, 0.0)
    return conf * scale[:, None]


def bootstrap(
    conf_by_shot: np.ndarray, *, replicates: int = 1000, seed: int = 20261001
) -> dict[str, list[float]]:
    """95 % shot-bootstrap intervals of the macro and per-class F1.

    ``conf_by_shot`` is ``(S, 4, 4)``; a replicate draws S shots with replacement and
    sums their matrices.
    """
    rng = np.random.default_rng(seed)
    n = conf_by_shot.shape[0]
    macro, per = [], []
    for _ in range(replicates):
        total = conf_by_shot[rng.integers(n, size=n)].sum(axis=0)
        macro.append(macro_f1(total))
        per.append(f1_from_conf(total))
    per = np.array(per)
    out = {
        "macro_f1": [
            float(np.nanpercentile(macro, 2.5)),
            float(np.nanpercentile(macro, 97.5)),
        ]
    }
    for i, c in enumerate(CLASSES):
        col = per[:, i]
        out[f"f1_{c}"] = (
            [float(np.nanpercentile(col, 2.5)), float(np.nanpercentile(col, 97.5))]
            if np.isfinite(col).any()
            else [float("nan"), float("nan")]
        )
    return out


def paired_difference(
    conf_a: np.ndarray,
    conf_b: np.ndarray,
    *,
    replicates: int = 1000,
    seed: int = 20261001,
) -> dict:
    """Macro-F1 of ``a`` minus that of ``b`` on the same shots, with a 95 % interval.

    ``conf_a`` and ``conf_b`` are ``(S, 4, 4)`` matrices of the same ``S`` shots in the
    same order; a replicate draws the same shots for both, so the interval is of the
    paired difference.
    """
    rng = np.random.default_rng(seed)
    n = conf_a.shape[0]
    draws = np.array(
        [
            macro_f1(conf_a[i].sum(axis=0)) - macro_f1(conf_b[i].sum(axis=0))
            for i in (rng.integers(n, size=n) for _ in range(replicates))
        ]
    )
    return {
        "difference": macro_f1(conf_a.sum(axis=0)) - macro_f1(conf_b.sum(axis=0)),
        "ci95": [
            float(np.nanpercentile(draws, 2.5)),
            float(np.nanpercentile(draws, 97.5)),
        ],
        "share_not_above_zero": float(np.nanmean(draws <= 0)),
    }


def summarise(
    pred: np.ndarray,
    truth: np.ndarray,
    shots: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    replicates: int = 1000,
) -> dict:
    """Macro and per-class F1 with shot-bootstrap intervals, supports, confusion."""
    ids, by_shot = confusion_by_shot(pred, truth, shots, mask)
    total = by_shot.sum(axis=0)
    f1 = f1_from_conf(total)
    out = {
        "windows": int(total.sum()),
        "shots": int(ids.size),
        "macro_f1": macro_f1(total),
        "confusion": total.tolist(),
        "classes": {
            c: {
                "support": int(total[i].sum()),
                "shots": int((by_shot[:, i].sum(axis=1) > 0).sum()),
                "f1": None if not np.isfinite(f1[i]) else float(f1[i]),
            }
            for i, c in enumerate(CLASSES)
        },
    }
    out["ci95"] = bootstrap(by_shot, replicates=replicates)
    return out
