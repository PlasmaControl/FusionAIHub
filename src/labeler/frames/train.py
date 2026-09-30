"""Train `RowsCNN` on a method's train shots, stopped and thresholded on its val
shots (round three, Part B; spec §3.1).

    python -m labeler.frames.train --method M [--epochs N] [--limit N] [--pilot]
        [--seed S]

**Shots.** The split (`frames.shots_file`) and the features `prepare` wrote
(`frames.features_dir`), both `frames.VERSION`'s (F1): the train shots are
trained on, and the val shots stop the training and set the threshold; the
test shots are never read here. Every shot's target is its original's with the
owner's label over it (`targets.merged`, F2), as `prepare` wrote its states.
A train or val shot without features (`prepare.dropped`, or never prepared) is
left out, and the checkpoint lists it with its reason.

**Loss.** Binary cross-entropy on the spec's bins (`model.bin_logits`: the
frames' maximum for onsets, their mean for states) over the scored bins, ABSENT
or PRESENT_T with every frame observed; UNKNOWN and UNCERTAIN_T bins carry no
loss (`bin_weights`). The minority class of the scored train bins weighs their
majority-to-minority ratio, from 1 up to `pos_weight_max`, and the majority
class 1 (`class_weights`, F3): H-mode's minority is L, the absent class, which
v1's present-only weight left at 1.

**Crops.** Each epoch draws `CROPS_PER_SHOT` crops of `CROP_MS` from each
train shot with a scored bin, on its bins' edges (a shorter shot whole, padded
with unscored zeros; `crop_windows`). They start at random, unless the spec's
`balance_crops` is set (F11, the sawteeth's: 4,815 of the ece_sawtooth v2
table's 4,822 shots hold a sawtooth, so the owner's "as many shots that dont
have sawtooth as shots that do" cannot be had in whole shots, and is had in
crops): then half of a shot's crops are centred on a random scored ABSENT bin
and half on a random scored PRESENT_T bin, all on one class when the shot has
only that one, and a crop whose centre is too near an edge is moved inside the
shot. Across the train shots the crops centred on sawtooth-free time then equal
those centred on sawteeth. With balancing the class weights are taken from the
scored bins of the first epoch's crops (`crop_states`), not of the whole train
shots, so the two corrections do not compound, and held for the run, so the
val losses the stopping compares are on one weighting; every epoch's crops are
drawn by the same rule.

**Stopping and threshold.** After each epoch, the loss over the whole val
shots: the epoch with the lowest is kept, and `patience` epochs without a lower
one stop the run. The threshold is then the one of 0.10-0.90 in steps of 0.05
with the best score over the val shots' scored bins by the spec's
`threshold_rule` (F3), a tie going to the one nearest 0.5 (`pick_threshold`):
the present class's F1 ("f1"), or the mean of the present and the absent
classes' F1 ("macro_f1", H-mode's: F1(H) and F1(L), as its bar H1 reads them).

**Output.** `training.json`, then `model.pt` (`save`), in `frames.model_dir`:
the state dict, the `TrainConfig`, the threshold and its rule, both classes'
weights, whether the crops were balanced (`balance_crops`) and their count by
centre class, over the run and per epoch (`crop_centres`: "absent" and
"present", or "random"), the channels, sub-frames and width, the spec, the
split's sha256 and the shots trained on. A model is trained
once on its split, so an existing `model.pt` is refused. A pilot (`--pilot`, or
any `--limit`, which takes the first N train and N val shots) writes under
`runs/frames/pilot/<method>/`, which it may replace, for `PILOT_EPOCHS` unless
`--epochs` says otherwise. Torch takes SLURM_CPUS_PER_TASK threads (4 without).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import resource
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..ae.xpower import pilot_area
from ..config import Paths, atomic_path, git_sha
from ..scoring.frames import FRAME_MS
from . import (
    SEED,
    SPECS,
    VERSION,
    EventSpec,
    features_dir,
    model_dir,
    prepare,
    shots_file,
)
from .model import RowsCNN, bin_logits
from .targets import ABSENT, PRESENT_T

THRESHOLDS = np.round(np.arange(0.10, 0.91, 0.05), 2)
#: Each crop's length, and how many a train shot gives an epoch.
CROP_MS = 2560.0
CROPS_PER_SHOT = 4
PILOT_EPOCHS = 2


@dataclass
class TrainConfig:
    epochs: int = 30
    lr: float = 1e-3
    batch: int = 8
    pos_weight_max: float = 5.0
    patience: int = 5


@dataclass(frozen=True)
class Shot:
    """One shot's features as training reads them: `x` `(C, n_sub)`, and per bin
    its target `states` and whether every frame is `observed`."""

    shot: int
    x: np.ndarray
    states: np.ndarray
    observed: np.ndarray


def frames_per_bin(spec: EventSpec) -> int:
    return round(spec.bin_ms / FRAME_MS)


def subs_of(spec: EventSpec) -> int:
    return round(FRAME_MS / spec.sub_ms)


def pilot_dir(paths: Paths, method: str) -> Path:
    """Where a pilot writes its model."""
    return paths.runs / "frames" / "pilot" / method


def read_shot(paths: Paths, method: str, shot: int) -> Shot | None:
    """A shot's features (`prepare`'s npz); None without them."""
    path = features_dir(paths, method, VERSION) / f"{int(shot)}.npz"
    if not path.is_file():
        return None
    per = frames_per_bin(SPECS[method])
    with np.load(path) as z:
        observed = z["observed"].reshape(-1, per).all(axis=1)
        return Shot(int(shot), z["x"], z["states"], observed)


def bin_weights(
    states, observed, pos_weight: float, absent_weight: float = 1.0
) -> np.ndarray:
    """Each bin's loss weight: `absent_weight` absent, `pos_weight` present, 0
    unscored (UNKNOWN, UNCERTAIN_T, or a frame not observed)."""
    states = np.asarray(states)
    scored = np.isin(states, (ABSENT, PRESENT_T)) & np.asarray(observed, bool)
    weights = np.where(states == PRESENT_T, pos_weight, absent_weight)
    return np.where(scored, weights, 0.0).astype(np.float32)


def _loss_sum(frame_logits, states, weights, frames_per_bin: int, pool: str):
    logits = bin_logits(torch.as_tensor(frame_logits), frames_per_bin, pool)
    target = torch.as_tensor(np.asarray(states) == PRESENT_T, dtype=logits.dtype)
    weight = torch.as_tensor(np.asarray(weights), dtype=logits.dtype)
    total = F.binary_cross_entropy_with_logits(
        logits, target, weight=weight, reduction="sum"
    )
    return total, weight.sum()


def bin_loss(frame_logits, states, weights, frames_per_bin: int, pool: str):
    """The weighted mean binary cross-entropy of `(B, n)` frame logits on their
    bins' `states` `(B, n_bins)`."""
    total, weight = _loss_sum(frame_logits, states, weights, frames_per_bin, pool)
    return total / weight.clamp(min=1.0)


def class_weights(states, cap: float) -> dict[str, float]:
    """`{"absent": w, "present": w}` from the scored bins in `states` (arrays):
    the minority class weighs the majority-to-minority ratio, from 1 up to
    `cap`, and the majority 1 (F3); both 1 when a class has no bin, or on a
    tie."""
    arrays = [np.asarray(s).ravel() for s in states]
    states = np.concatenate(arrays) if arrays else np.zeros(0)
    present, absent = int(np.sum(states == PRESENT_T)), int(np.sum(states == ABSENT))
    weights = {"absent": 1.0, "present": 1.0}
    if present and absent:
        minority = "present" if present < absent else "absent"
        ratio = max(present, absent) / min(present, absent)
        weights[minority] = float(min(ratio, cap))
    return weights


#: A balanced crop's centre class, and the state its centre bin is drawn from.
CENTRES = {"absent": ABSENT, "present": PRESENT_T}
RANDOM = "random"  # an unbalanced crop's centre class


def crop_windows(shots, rng, crop_bins: int, balance: bool = False) -> list:
    """`CROPS_PER_SHOT` crops of each shot with a scored bin, as `(the shot's
    index, first bin, centre class)`. Without `balance`, each starts at random
    (`RANDOM`); with it (F11), `CROPS_PER_SHOT // 2` are centred on a random
    scored ABSENT bin ("absent") and the rest on a random scored PRESENT_T bin
    ("present"), or all on the one class a shot has, a crop moved inside the
    shot where its centre is too near an edge."""
    out = []
    for i, shot in enumerate(shots):
        scored = bin_weights(shot.states, shot.observed, 1.0) > 0
        if not scored.any():
            continue
        n = len(shot.states)
        last = max(0, n - crop_bins)
        if not balance:
            for _ in range(CROPS_PER_SHOT):
                start = int(rng.integers(0, last + 1)) if n > crop_bins else 0
                out.append((i, start, RANDOM))
            continue
        bins = {
            name: np.flatnonzero(scored & (shot.states == state))
            for name, state in CENTRES.items()
        }
        held = [name for name in CENTRES if len(bins[name])]
        if len(held) == 2:
            half = CROPS_PER_SHOT // 2
            counts = {"absent": half, "present": CROPS_PER_SHOT - half}
        else:
            counts = {held[0]: CROPS_PER_SHOT}
        for name, count in counts.items():
            for centre in rng.choice(bins[name], size=count):
                start = int(np.clip(int(centre) - crop_bins // 2, 0, last))
                out.append((i, start, name))
    return out


def centre_counts(windows) -> dict[str, int]:
    """The crops by centre class."""
    found: dict[str, int] = {}
    for _, _, name in windows:
        found[name] = found.get(name, 0) + 1
    return dict(sorted(found.items()))


def crop_states(shots, windows, crop_bins: int) -> list[np.ndarray]:
    """The scored bins' states of each crop, for `class_weights`."""
    out = []
    for i, start, _ in windows:
        shot = shots[i]
        stop = min(len(shot.states), start + crop_bins)
        scored = bin_weights(shot.states, shot.observed, 1.0)[start:stop] > 0
        out.append(shot.states[start:stop][scored])
    return out


def crops(shots, windows, crop_bins: int, per: int, subs: int, weights: dict):
    """The `windows`' crops (`crop_windows`): inputs `(B, C, crop_bins * per *
    subs)`, states and weights per bin; `weights` is `class_weights`'."""
    width = per * subs
    xs, ys, ws = [], [], []
    for i, start, _ in windows:
        shot = shots[i]
        weights_ = _weights(shot, weights)
        stop = min(len(shot.states), start + crop_bins)
        x = np.zeros((shot.x.shape[0], crop_bins * width), dtype=np.float32)
        x[:, : (stop - start) * width] = shot.x[:, start * width : stop * width]
        y = np.full(crop_bins, ABSENT, dtype=np.int8)
        y[: stop - start] = shot.states[start:stop]
        w = np.zeros(crop_bins, dtype=np.float32)
        w[: stop - start] = weights_[start:stop]
        xs.append(x), ys.append(y), ws.append(w)
    if not xs:
        raise ValueError("no train shot has a scored bin")
    return np.stack(xs), np.stack(ys), np.stack(ws)


def _weights(shot: Shot, weights: dict) -> np.ndarray:
    return bin_weights(
        shot.states, shot.observed, weights["present"], weights["absent"]
    )


def bin_probs(model: RowsCNN, x, per: int, pool: str) -> np.ndarray:
    """P per bin of one shot's `(C, n_sub)` features, the model in eval mode."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(np.asarray(x, dtype=np.float32))[None])
        return torch.sigmoid(bin_logits(logits, per, pool))[0].numpy()


def val_loss(model: RowsCNN, shots, per: int, pool: str, weights: dict) -> float:
    """The weighted loss over every scored bin of the whole val shots, `weights`
    `class_weights`'."""
    model.eval()
    total, weight = 0.0, 0.0
    with torch.no_grad():
        for shot in shots:
            logits = model(torch.from_numpy(shot.x.astype(np.float32))[None])
            weights_ = _weights(shot, weights)
            t, w = _loss_sum(logits, shot.states[None], weights_[None], per, pool)
            total, weight = total + float(t), weight + float(w)
    return total / weight if weight else math.nan


def bin_cells(prob, states, threshold: float, observed=None) -> np.ndarray:
    """`[tp, fp, fn, tn]` over the bins ABSENT or PRESENT_T and observed (a bin
    whose P is NaN is not)."""
    prob, states = np.asarray(prob, dtype=np.float64), np.asarray(states)
    scored = np.isin(states, (ABSENT, PRESENT_T)) & np.isfinite(prob)
    if observed is not None:
        scored &= np.asarray(observed, bool)
    truth, said = states == PRESENT_T, np.nan_to_num(prob, nan=0.0) >= threshold
    return np.array(
        [
            np.sum(scored & truth & said),
            np.sum(scored & ~truth & said),
            np.sum(scored & truth & ~said),
            np.sum(scored & ~truth & ~said),
        ],
        dtype=np.float64,
    )


def f1_of(cells) -> float:
    tp, fp, fn, _ = np.asarray(cells, dtype=np.float64)
    return float(2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else math.nan


#: What `pick_threshold` maximises, by `EventSpec.threshold_rule` (F3).
THRESHOLD_RULES = ("f1", "macro_f1")


def rule_score(cells, rule: str) -> float:
    """The pooled `[tp, fp, fn, tn]`'s score by `rule`: the present class's F1
    ("f1"), or the mean of it and the absent class's ("macro_f1", F1(H) and
    F1(L) for H-mode); NaN where a class's F1 is."""
    if rule not in THRESHOLD_RULES:
        raise ValueError(f"threshold rule {rule!r} is not one of {THRESHOLD_RULES}")
    present = f1_of(cells)
    if rule == "f1":
        return present
    tp, fp, fn, tn = np.asarray(cells, dtype=np.float64)
    return (present + f1_of([tn, fn, fp, tp])) / 2.0


def pick_threshold(probs, states, observed, rule: str = "f1") -> float:
    """The threshold with the best pooled score by `rule` (`rule_score`); a tie
    goes to the one nearest 0.5."""
    best, found = -1.0, 0.5
    for threshold in sorted(THRESHOLDS, key=lambda t: abs(t - 0.5)):
        cells = sum(
            bin_cells(p, s, threshold, o)
            for p, s, o in zip(probs, states, observed, strict=True)
        )
        score = rule_score(cells, rule)
        if score > best + 1e-12:
            best, found = score, float(threshold)
    return found


def _load(paths: Paths, method: str, shots, limit: int, gone: dict, left: dict):
    kept = []
    for shot in shots:
        data = read_shot(paths, method, shot)
        if data is None:
            left[str(shot)] = gone.get(int(shot), "not prepared")
            continue
        kept.append(data)
        if limit and len(kept) == limit:
            break
    return kept


def fit(
    method: str,
    paths: Paths,
    config: TrainConfig | None = None,
    *,
    out: Path | None = None,
    limit: int = 0,
    seed: int = SEED,
    log=print,
) -> Path:
    """Train the method's model (module docstring) into `out` (default
    `frames.model_dir`); the `model.pt` written."""
    config = config or TrainConfig()
    log = log or (lambda message: None)
    spec = SPECS[method]
    out = model_dir(paths, method, VERSION) if out is None else Path(out)
    pilot = pilot_area(out, paths.runs)
    if limit and not pilot:
        raise ValueError(f"{out}: a limited run writes under {paths.runs}")
    target = out / "model.pt"
    if target.exists() and not pilot:
        raise FileExistsError(f"{target}: a model is trained once on its split")
    split_bytes = shots_file(paths, method, VERSION).read_bytes()
    split = prepare.split_shots(paths, method)
    gone, left = prepare.dropped(paths, method), {}
    shots = {
        name: _load(
            paths,
            method,
            sorted(s for s, v in split.items() if v == name),
            limit,
            gone,
            left,
        )
        for name in ("train", "val")
    }
    train, val = shots["train"], shots["val"]
    per, subs = frames_per_bin(spec), subs_of(spec)
    if not any(bin_weights(s.states, s.observed, 1.0).any() for s in val):
        raise ValueError(f"{method}: no val shot has a scored bin")
    channels = {int(s.x.shape[0]) for s in train + val}
    if len(channels) != 1:
        raise ValueError(f"{method}: the shots' features have {sorted(channels)} rows")
    (channels,) = channels
    balance = spec.balance_crops
    weights = None
    if not balance:
        weights = class_weights(
            [s.states[s.observed] for s in train], config.pos_weight_max
        )
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = RowsCNN(channels, subs)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.lr)
    crop_bins = max(1, int(CROP_MS // spec.bin_ms))
    history, best, stale, centres = [], math.inf, 0, {}
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(1, config.epochs + 1):
        started = time.monotonic()
        windows = crop_windows(train, rng, crop_bins, balance)
        if weights is None:  # F11: the first epoch's crops, held for the run
            weights = class_weights(
                crop_states(train, windows, crop_bins), config.pos_weight_max
            )
        drawn = centre_counts(windows)
        for name, count in drawn.items():
            centres[name] = centres.get(name, 0) + count
        x, states, crop_w = crops(train, windows, crop_bins, per, subs, weights)
        order = rng.permutation(len(x))
        model.train()
        losses = []
        for i in range(0, len(order), config.batch):
            pick = order[i : i + config.batch]
            if len(pick) < 2:
                continue  # batch norm needs two samples
            logits = model(torch.from_numpy(x[pick]))
            loss = bin_loss(logits, states[pick], crop_w[pick], per, spec.pool)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
        loss = val_loss(model, val, per, spec.pool, weights)
        kept = bool(np.isfinite(loss) and loss < best - 1e-12)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else None,
                "val_loss": loss if np.isfinite(loss) else None,
                "kept": kept,
                "centres": drawn,
                "seconds": round(time.monotonic() - started, 1),
            }
        )
        log(json.dumps(history[-1]))
        if kept:
            best, best_state, stale = loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    model.load_state_dict(best_state)
    probs = [bin_probs(model, s.x, per, spec.pool) for s in val]
    threshold = pick_threshold(
        probs,
        [s.states for s in val],
        [s.observed for s in val],
        spec.threshold_rule,
    )
    record = {
        "method": method,
        "version": VERSION,
        "config": asdict(config),
        "threshold": threshold,
        "threshold_rule": spec.threshold_rule,
        "channels": channels,
        "subs": subs,
        "width": model.width,
        "spec": asdict(spec),
        "split_sha256": hashlib.sha256(split_bytes).hexdigest(),
        "weights": weights,
        "balance_crops": balance,
        "crop_centres": dict(sorted(centres.items())),
        "seed": seed,
        "limit": limit,
        "crop_ms": CROP_MS,
        "crops_per_shot": CROPS_PER_SHOT,
        "best_epoch": max((h["epoch"] for h in history if h["kept"]), default=0),
        "history": history,
        "shots": {"train": [s.shot for s in train], "val": [s.shot for s in val]},
        "left_out": left,
        "git_sha": git_sha(),
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    save(out, model, record)
    return target


def save(out: Path, model: RowsCNN, record: dict) -> None:
    """`training.json` (the record, its peak RSS) then `model.pt` (the record and
    the state dict), so a model is never there without its record."""
    peak = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    with atomic_path(out / "training.json") as tmp:
        tmp.write_text(json.dumps(record | {"peak_rss_mb": peak}, indent=1) + "\n")
    with atomic_path(out / "model.pt") as tmp:
        torch.save({"state_dict": model.state_dict(), **record}, tmp)


def load(path) -> tuple[RowsCNN, dict]:
    """A saved model, in eval mode, and its checkpoint."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = RowsCNN(blob["channels"], blob["subs"], blob.get("width", 32))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--method", required=True, choices=list(SPECS))
    p.add_argument(
        "--epochs",
        type=int,
        help=f"default {TrainConfig.epochs}; a pilot's {PILOT_EPOCHS}",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="the first N train and N val shots, as a pilot",
    )
    p.add_argument(
        "--pilot", action="store_true", help="into runs/frames/pilot/<method>"
    )
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args(argv)
    if args.limit < 0 or (args.epochs is not None and args.epochs < 1):
        p.error("--limit must be nonnegative and --epochs positive")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    paths = Paths.from_env()
    pilot = args.pilot or args.limit > 0
    out = (
        pilot_dir(paths, args.method)
        if pilot
        else model_dir(paths, args.method, VERSION)
    )
    epochs = args.epochs or (PILOT_EPOCHS if pilot else TrainConfig.epochs)
    try:
        path = fit(
            args.method,
            paths,
            TrainConfig(epochs=epochs),
            out=out,
            limit=args.limit,
            seed=args.seed,
            log=lambda message: print(message, flush=True),
        )
    except (OSError, ValueError) as error:
        p.error(str(error))
    print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
