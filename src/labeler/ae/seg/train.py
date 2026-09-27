"""Train SegNet on the reviewed AE pseudo-masks.

    python -m labeler.ae.seg.train [--out DIR] [--pilot N] [--epochs E]

**Data.** Every shot of the AE split with a pseudo-mask (`pseudo`): as input its
three cross-power rows at store level 8 (2.048 ms columns, 257 bins to
250 kHz), as target its pseudo-mask with the regions the reviewer rejected set
to background (`regions.reviewed_mask`). The split is the chosen AE model's
(`models/ae_xpower/v1/<chosen>/split.csv`): a shot is a test shot of both
models or of neither. Each shot is cut to the columns its mask scores, and
`MARGIN_COLS` either side for context.

**Loss.** Binary cross-entropy on the pixels the mask scores (0 or 1, never
IGNORE), AE pixels weighted by the training set's background-to-AE ratio,
capped at `pos_weight_max`, plus soft Dice over the same pixels. Each epoch
draws `crops_per_shot` crops of `crop_cols` columns from every training shot,
half of them centred on a column holding AE. Training keeps the epoch with the
best validation Dice at 0.5 and stops after `patience` epochs without a better
one; the threshold is then fixed on the validation shots (0.10-0.95 in steps of
0.05, best pooled Dice), before any test shot is scored (`evaluate`).

Writes `model.pt`, `split.csv` and `training.json` to `--out` (default
`$LABELER_ROOT/models/ae_seg/v1`; a pilot writes to `runs/ae_seg/pilot`).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import resource
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ...config import Paths, atomic_path, git_sha, sha256_of
from ...events.review import labels
from ..xpower import event_dir
from ..xpower import model_dir as ae_model_dir
from ..xpower.data import SEED, store_rows
from ..xpower.evaluate import chosen_model
from ..xpower.train import read_split
from . import EVENT, model_dir, pseudo_dir, regions
from .model import SegNet, SegNetConfig
from .pseudo import IGNORE, LEVEL, PseudoMask

THRESHOLDS = np.round(np.arange(0.10, 0.96, 0.05), 2)
MARGIN_COLS = 64


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 40
    batch: int = 8
    crop_cols: int = 256
    crops_per_shot: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    pos_weight_max: float = 5.0
    width: int = 12
    seed: int = SEED


@dataclass(frozen=True)
class Example:
    """One shot, cut to the columns its mask scores and a margin."""

    shot: int
    x: np.ndarray  # (3, n_y, n) uint8, the rows
    y: np.ndarray  # (n_y, n) uint8: 0, 1 or IGNORE
    t0_ms: float  # the time of column 0's start
    dt_ms: float
    y0_khz: float
    dy_khz: float


def load_example(
    paths: Paths, shot: int, decisions: dict, margin: int | None = MARGIN_COLS
) -> Example:
    """One shot, cut to its scored columns and `margin` either side; the whole
    shot when `margin` is None."""
    grid, values, y0, dy = store_rows(paths.spectrogram_file(EVENT, shot), LEVEL)
    file = regions.pseudo_file(paths, shot)
    pm = PseudoMask.load(file)
    if values.shape[1:] != pm.mask.shape or abs(pm.t0_ms - grid.t0_ms) > 1e-6:
        raise ValueError(f"{shot}: the pseudo-mask is not on the store's level {LEVEL}")
    y = regions.reviewed_mask(pm, decisions.get(shot), regions.file_sha256(file))
    scored = np.flatnonzero((y != IGNORE).any(axis=0))
    if not scored.size:
        raise ValueError(f"{shot}: the pseudo-mask scores no pixel")
    a, b = 0, y.shape[1]
    if margin is not None:
        a, b = max(a, scored[0] - margin), min(b, scored[-1] + 1 + margin)
    return Example(
        shot,
        np.ascontiguousarray(values[:, :, a:b]),
        np.ascontiguousarray(y[:, a:b]),
        grid.t0_ms + a * grid.dt_ms,
        grid.dt_ms,
        y0,
        dy,
    )


def crop(example: Example, start: int, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Columns `start .. start + length` of an example; past its ends, zero rows
    and an ignored target."""
    c, n_y, n = example.x.shape
    x = np.zeros((c, n_y, length), dtype=np.uint8)
    y = np.full((n_y, length), IGNORE, dtype=np.uint8)
    a, b = max(start, 0), min(start + length, n)
    if b > a:
        x[:, :, a - start : b - start] = example.x[:, :, a:b]
        y[:, a - start : b - start] = example.y[:, a:b]
    return x, y


def sample_crops(
    examples: Sequence[Example], rng, k: int, length: int
) -> tuple[np.ndarray, np.ndarray]:
    """`k` crops per example, the even-numbered ones centred on a column with AE."""
    xs, ys = [], []
    for ex in examples:
        n = ex.y.shape[1]
        ae = np.flatnonzero((ex.y == 1).any(axis=0))
        for i in range(k):
            if i % 2 == 0 and ae.size:
                start = int(rng.choice(ae)) - length // 2
            else:
                start = int(rng.integers(0, max(1, n - length + 1)))
            x, y = crop(ex, start, length)
            xs.append(x)
            ys.append(y)
    return np.stack(xs), np.stack(ys)


def pos_weight(examples: Sequence[Example], cap: float) -> float:
    """Background pixels per AE pixel over `examples`, within [1, cap]."""
    pos = sum(int((ex.y == 1).sum()) for ex in examples)
    neg = sum(int((ex.y == 0).sum()) for ex in examples)
    return float(np.clip(neg / pos if pos else cap, 1.0, cap))


def masked_loss(logits: torch.Tensor, target: torch.Tensor, weight: float):
    """BCE with AE pixels weighted `weight`, plus soft Dice; IGNORE pixels count
    in neither. `logits` `(B, H, W)`, `target` `(B, H, W)` uint8."""
    valid = (target != IGNORE).float()
    truth = (target == 1).float()
    bce = F.binary_cross_entropy_with_logits(
        logits, truth, pos_weight=torch.tensor(weight), reduction="none"
    )
    bce = (bce * valid).sum() / valid.sum().clamp(min=1.0)
    p = torch.sigmoid(logits) * valid
    dice = 1 - (2 * (p * truth).sum() + 1) / (p.sum() + (truth * valid).sum() + 1)
    return bce + dice


@torch.no_grad()
def predict(model: SegNet, x: np.ndarray) -> np.ndarray:
    """P(AE) per pixel of `(3, n_y, n)` uint8 rows, `(n_y, n)` float32."""
    model.eval()
    batch = torch.from_numpy(x[None].astype(np.float32) / 255.0)
    return torch.sigmoid(model(batch))[0, 0].numpy()


def pixel_cells(prob: np.ndarray, target: np.ndarray, threshold: float) -> np.ndarray:
    """`[tp, fp, fn, tn]` over the pixels the target scores."""
    scored = target != IGNORE
    truth, said = target == 1, prob >= threshold
    return np.array(
        [
            np.sum(scored & truth & said),
            np.sum(scored & ~truth & said),
            np.sum(scored & truth & ~said),
            np.sum(scored & ~truth & ~said),
        ],
        dtype=float,
    )


def dice_of(cells) -> float:
    tp, fp, fn, _ = np.asarray(cells, dtype=float)
    return float(2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else float("nan")


def pick_threshold(probs, targets) -> float:
    """The threshold with the best pooled Dice; ties go to the one nearest 0.5."""
    best, found = -1.0, 0.5
    for threshold in sorted(THRESHOLDS, key=lambda t: abs(t - 0.5)):
        cells = sum(pixel_cells(p, y, threshold) for p, y in zip(probs, targets))
        score = dice_of(cells)
        if score > best + 1e-12:
            best, found = score, float(threshold)
    return found


def fit(
    train: Sequence[Example],
    val: Sequence[Example],
    config: TrainConfig | None = None,
    log: Callable[[str], None] = print,
) -> tuple[SegNet, list[dict], float]:
    """The model at its best validation epoch, the per-epoch history, the threshold."""
    config = config or TrainConfig()
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    model = SegNet(SegNetConfig(width=config.width))
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    weight = pos_weight(train, config.pos_weight_max)
    history, best, stale = [], -1.0, 0
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(1, config.epochs + 1):
        started = time.monotonic()
        x, y = sample_crops(train, rng, config.crops_per_shot, config.crop_cols)
        order = rng.permutation(len(x))
        model.train()
        losses = []
        for i in range(0, len(order), config.batch):
            pick = order[i : i + config.batch]
            batch = torch.from_numpy(x[pick].astype(np.float32) / 255.0)
            loss = masked_loss(model(batch)[:, 0], torch.from_numpy(y[pick]), weight)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
        probs = [predict(model, ex.x) for ex in val]
        score = dice_of(sum(pixel_cells(p, ex.y, 0.5) for p, ex in zip(probs, val)))
        kept = bool(np.isfinite(score) and score > best)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else None,
                "val_dice": score if np.isfinite(score) else None,
                "kept": kept,
                "seconds": round(time.monotonic() - started, 1),
            }
        )
        log(json.dumps(history[-1]))
        if kept:
            best, best_state, stale = score, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    model.load_state_dict(best_state)
    probs = [predict(model, ex.x) for ex in val]
    return model, history, pick_threshold(probs, [ex.y for ex in val])


def best_epoch(history: list[dict]) -> int:
    """The epoch `fit` kept; 0 when none beat the untrained model."""
    return max((row["epoch"] for row in history if row["kept"]), default=0)


def save(
    out: Path,
    model: SegNet,
    *,
    threshold: float,
    split: dict[int, str],
    history: list[dict],
    config: TrainConfig,
    inputs: dict,
) -> None:
    """`inputs`: the sha256 of each file trained from (labels, masks, pseudo index)."""
    out.mkdir(parents=True, exist_ok=True)
    blob = {
        "state_dict": model.state_dict(),
        "config": model.config.as_dict(),
        "threshold": threshold,
        "train": asdict(config),
        "best_epoch": best_epoch(history),
        "inputs": inputs,
        "git_sha": git_sha(),
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    with atomic_path(out / "model.pt") as tmp:
        torch.save(blob, tmp)
    with atomic_path(out / "split.csv") as tmp:
        lines = ["shot,split"] + [f"{s},{v}" for s, v in sorted(split.items())]
        tmp.write_text("\n".join(lines) + "\n")
    seconds = [row["seconds"] for row in history]
    record = {
        "history": history,
        "best_epoch": best_epoch(history),
        "threshold": threshold,
        "seconds_per_epoch": round(float(np.mean(seconds)), 1) if seconds else None,
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "counts": {
            v: sum(x == v for x in split.values()) for v in sorted(set(split.values()))
        },
    }
    with atomic_path(out / "training.json") as tmp:
        tmp.write_text(json.dumps(record, indent=1) + "\n")


def load(path) -> tuple[SegNet, dict]:
    """A saved model, in eval mode, and its record."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = SegNet(SegNetConfig.from_dict(blob["config"]))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob


def _sha(path: Path) -> str | None:
    return sha256_of(path) if path.is_file() else None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, help="default $LABELER_ROOT/models/ae_seg/v1")
    p.add_argument(
        "--pilot",
        type=int,
        default=0,
        help="6-20 shots (4 of them validation), 2 epochs, to runs/ae_seg/pilot",
    )
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    args = p.parse_args(argv)
    if args.pilot and not 6 <= args.pilot <= 20:
        p.error("a pilot is 6 to 20 shots")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    paths = Paths.from_env()
    out = args.out or (
        paths.runs / "ae_seg" / "pilot" if args.pilot else model_dir(paths)
    )
    split = read_split(chosen_model(ae_model_dir(paths)).parent / "split.csv")
    have = {s for s in split if regions.pseudo_file(paths, s).is_file()}
    split = {s: v for s, v in split.items() if s in have}
    if args.pilot:
        chosen = sorted(s for s, v in split.items() if v == "train")[: args.pilot - 4]
        chosen += sorted(s for s, v in split.items() if v == "val")[:4]
        split = {s: v for s, v in split.items() if v == "test" or s in chosen}
    directory = event_dir(paths)
    decisions = regions.read_decisions(directory)
    started = time.monotonic()
    examples = {
        s: load_example(paths, s, decisions)
        for s, v in sorted(split.items())
        if v in ("train", "val")
    }
    print(f"loaded {len(examples)} shots in {time.monotonic() - started:.0f} s")
    train = [examples[s] for s, v in sorted(split.items()) if v == "train"]
    val = [examples[s] for s, v in sorted(split.items()) if v == "val"]
    config = TrainConfig(epochs=2 if args.pilot else args.epochs)
    model, history, threshold = fit(
        train, val, config, log=lambda m: print(m, flush=True)
    )
    inputs = {
        "labels_sha256": _sha(labels.labels_path(directory)),
        "masks_sha256": _sha(regions.log_path(directory)),
        "pseudo_index_sha256": _sha(pseudo_dir(paths) / "index.csv"),
    }
    save(
        out,
        model,
        threshold=threshold,
        split=split,
        history=history,
        config=config,
        inputs=inputs,
    )
    print(f"wrote {out}: best epoch {best_epoch(history)}, threshold {threshold}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
