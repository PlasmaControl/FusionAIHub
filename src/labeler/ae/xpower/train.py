"""Train FrameCNN on the owner's reviewed AE shots.

    python -m labeler.ae.xpower.train --candidate NAME [--out DIR] [--pilot N]

reads the owner's labels (`data/events/alfven_eigenmode/review/labels.csv`), the
review page's AE stores and TokEye's masks, and writes `model.pt`, `split.csv`
and `training.json` to `--out` (default `$LABELER_ROOT/models/ae_xpower/v1/NAME`).
`CANDIDATES` names the input band and MHD weight of each version.

The loss is binary cross-entropy on the frames the owner called present or
absent; an absent frame TokEye marks as MHD (`data.mhd_frames`) weighs the
candidate's `mhd_weight` times as much, so the model pays most for the mistake
the earlier models made. Training keeps the epoch with the best validation F1
and stops after `patience` epochs without a better one; the decision threshold
is then fixed on the validation shots, before any test shot is scored.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import resource
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.nn import functional as F

from ...config import Paths, atomic_path, git_sha
from ...events.catalog.states import ABSENT, PRESENT
from ...events.review import labels
from . import EVENT, VERSION, event_dir, model_dir, pilot_area, tokeye_masks
from .data import (
    BAND_KHZ,
    CONTEXT_FRAMES,
    FULL_BAND_KHZ,
    SEED,
    SUBS,
    Shot,
    band_slice,
    frame_inputs,
    load_shot,
    make_split,
    seldnet_split,
    store_rows,
)
from .model import FrameCNN, FrameCNNConfig

THRESHOLDS = np.round(np.arange(0.10, 0.91, 0.05), 2)
#: The versions `evaluate --choose` picks between on the validation shots: the
#: earlier detector's band, and the full band at two weights on MHD frames.
CANDIDATES = {
    "band80-mhd3": {"band": BAND_KHZ, "mhd_weight": 3.0},
    "band0-mhd3": {"band": FULL_BAND_KHZ, "mhd_weight": 3.0},
    "band0-mhd10": {"band": FULL_BAND_KHZ, "mhd_weight": 10.0},
}


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 60
    batch: int = 16
    crop_frames: int = 64
    crops_per_shot: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    mhd_weight: float = 3.0
    patience: int = 10
    seed: int = SEED


def frame_weights(states, mhd, mhd_weight: float, observed=None) -> np.ndarray:
    """1 on scored frames, `mhd_weight` on MHD-absent ones, 0 elsewhere.

    A frame is scored when the owner called it present or absent and, if
    `observed` is given, the rows cover it.
    """
    states = np.asarray(states)
    weights = np.isin(states, (ABSENT, PRESENT)).astype(np.float32)
    weights[(states == ABSENT) & np.asarray(mhd, bool)] = mhd_weight
    if observed is not None:
        weights[~np.asarray(observed, bool)] = 0.0
    return weights


def sample_crops(shots: Sequence[Shot], rng, k: int, length: int, mhd_weight: float):
    """`k` random `length`-frame crops of each shot: inputs, targets, weights."""
    xs, ys, ws = [], [], []
    for shot in shots:
        weights = frame_weights(shot.states, shot.mhd, mhd_weight, shot.observed)
        if shot.n < length or not weights.any():
            continue
        for start in rng.integers(0, shot.n - length + 1, size=k):
            xs.append(shot.x[..., SUBS * start : SUBS * (start + length)])
            ys.append(shot.states[start : start + length] == PRESENT)
            ws.append(weights[start : start + length])
    if not xs:
        raise ValueError("no shot has a scored frame to train on")
    return (
        np.stack(xs).astype(np.float32),
        np.stack(ys).astype(np.float32),
        np.stack(ws).astype(np.float32),
    )


def predict(model: FrameCNN, x: np.ndarray) -> np.ndarray:
    """P(AE) per frame of one shot's `(C, n_bins, SUBS * n)` input."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(np.asarray(x, dtype=np.float32))[None])
    return torch.sigmoid(logits)[0].numpy()


def probabilities(
    model: FrameCNN, rows, first: int, n: int, *, band, context: int = CONTEXT_FRAMES
) -> tuple[np.ndarray, np.ndarray]:
    """P(AE) and whether the rows cover it, for frames `first .. first + n - 1`.

    `rows` is `data.store_rows` or `data.raw_rows` output; the model sees
    `context` frames either side, as it did in training.
    """
    grid, values, y0, dy = rows
    x, observed = frame_inputs(
        values[:, band_slice(y0, dy, values.shape[1], band)],
        grid,
        first - context,
        n + 2 * context,
    )
    prob = predict(model, x)
    return prob[context : context + n], observed[context : context + n]


def frame_cells(prob, states, threshold: float) -> np.ndarray:
    """`[tp, fp, fn, tn]` over the frames the owner called present or absent."""
    states = np.asarray(states)
    scored = np.isin(states, (ABSENT, PRESENT))
    truth, said = states == PRESENT, np.asarray(prob) >= threshold
    return np.array(
        [
            np.sum(scored & truth & said),
            np.sum(scored & ~truth & said),
            np.sum(scored & truth & ~said),
            np.sum(scored & ~truth & ~said),
        ],
        dtype=float,
    )


def f1_of(cells) -> float:
    tp, fp, fn, _ = np.asarray(cells, dtype=float)
    return float(2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else float("nan")


def pick_threshold(probs, states) -> float:
    """The threshold with the best pooled F1; ties go to the one nearest 0.5."""
    best, found = -1.0, 0.5
    for threshold in sorted(THRESHOLDS, key=lambda t: abs(t - 0.5)):
        cells = sum(frame_cells(p, s, threshold) for p, s in zip(probs, states))
        score = f1_of(cells)
        if score > best + 1e-12:
            best, found = score, float(threshold)
    return found


def fit(
    train: Sequence[Shot],
    val: Sequence[Shot],
    config: TrainConfig | None = None,
    log: Callable[[str], None] = print,
) -> tuple[FrameCNN, list[dict], float]:
    """The model at its best validation epoch, the per-epoch history, the threshold."""
    config = config or TrainConfig()
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    model = FrameCNN(FrameCNNConfig(subs=SUBS))
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    history, best, stale = [], -1.0, 0
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(1, config.epochs + 1):
        started = time.monotonic()
        x, y, w = sample_crops(
            train, rng, config.crops_per_shot, config.crop_frames, config.mhd_weight
        )
        order = rng.permutation(len(x))
        model.train()
        losses = []
        for i in range(0, len(order), config.batch):
            pick = order[i : i + config.batch]
            if len(pick) < 2:
                continue  # batch norm needs two samples
            logits = model(torch.from_numpy(x[pick]))
            weight = torch.from_numpy(w[pick])
            loss = F.binary_cross_entropy_with_logits(
                logits, torch.from_numpy(y[pick]), weight=weight, reduction="sum"
            ) / weight.sum().clamp(min=1.0)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
        probs = [predict(model, shot.x) for shot in val]
        cells = sum(frame_cells(p, s.states, 0.5) for p, s in zip(probs, val))
        score = f1_of(cells)
        kept = bool(np.isfinite(score) and score > best)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else None,
                "val_f1": score if np.isfinite(score) else None,
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
    probs = [predict(model, shot.x) for shot in val]
    return model, history, pick_threshold(probs, [s.states for s in val])


def best_epoch(history: list[dict]) -> int:
    """The epoch `fit` kept; 0 when none beat the untrained model."""
    return max((row["epoch"] for row in history if row["kept"]), default=0)


def save(
    out: Path,
    model: FrameCNN,
    *,
    threshold: float,
    split: dict[int, str],
    history: list[dict],
    config: TrainConfig,
    band_khz,
    labels_file: Path,
    candidate: str = "",
    version: str = VERSION,
    labels_bytes: bytes | None = None,
    allow_replace: bool = False,
    runs: Path | None = None,
) -> None:
    refuse_checkpoint(out, allow_replace=allow_replace, runs=runs)
    if labels_bytes is None:
        labels_bytes = labels_file.read_bytes()
    out.mkdir(parents=True, exist_ok=True)
    blob = {
        "state_dict": model.state_dict(),
        "config": model.config.as_dict(),
        "candidate": candidate,
        "version": version,
        "threshold": threshold,
        "band_khz": [float(b) for b in band_khz],
        "train": asdict(config),
        "best_epoch": best_epoch(history),
        "labels_sha256": hashlib.sha256(labels_bytes).hexdigest(),
        "git_sha": git_sha(),
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    with atomic_path(out / "model.pt") as tmp:
        torch.save(blob, tmp)
    # The labels as trained on, where `labels.read_saved(out)` finds them.
    with atomic_path(labels.labels_path(out)) as tmp:
        tmp.write_bytes(labels_bytes)
    with atomic_path(out / "split.csv") as tmp:
        lines = ["shot,split"] + [f"{s},{v}" for s, v in sorted(split.items())]
        tmp.write_text("\n".join(lines) + "\n")
    record = {
        "candidate": blob["candidate"],
        "version": version,
        "band_khz": blob["band_khz"],
        "git_sha": blob["git_sha"],
        "labels_sha256": blob["labels_sha256"],
        "history": history,
        "best_epoch": best_epoch(history),
        "threshold": threshold,
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "counts": {
            v: sum(x == v for x in split.values()) for v in sorted(set(split.values()))
        },
    }
    with atomic_path(out / "training.json") as tmp:
        tmp.write_text(json.dumps(record, indent=1) + "\n")


def load(path) -> tuple[FrameCNN, dict]:
    """A saved model, in eval mode, and its record."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = FrameCNN(FrameCNNConfig.from_dict(blob["config"]))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob


def read_split(path, *, data: bytes | None = None) -> dict[int, str]:
    lines = (Path(path).read_bytes() if data is None else data).decode().split()
    return {int(a): b for a, b in (line.split(",") for line in lines[1:])}


def refuse_checkpoint(
    out: Path, *, allow_replace: bool = False, runs: Path | None = None
) -> None:
    if allow_replace and (runs is None or not pilot_area(out, runs)):
        raise ValueError(
            f"{out}: --pilot replacement requires a directory under {runs}"
        )
    file = out / "model.pt"
    if file.exists() and not allow_replace:
        raise FileExistsError(
            f"{file}: a trained candidate is not replaced; train a new version"
        )


def candidate_dir(paths: Paths, name: str, version: str = VERSION) -> Path:
    return model_dir(paths, version) / name


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--version", default=VERSION)
    p.add_argument("--candidate", choices=sorted(CANDIDATES), required=True)
    p.add_argument(
        "--out", type=Path, help="default $LABELER_ROOT/models/ae_xpower/v1/<candidate>"
    )
    p.add_argument(
        "--pilot",
        type=int,
        default=0,
        help="6-20 shots (4 of them validation), 2 epochs, to runs/ae_xpower/pilot",
    )
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    args = p.parse_args(argv)
    if args.pilot and not 6 <= args.pilot <= 20:
        p.error("a pilot is 6 to 20 shots")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    paths = Paths.from_env()
    spec = CANDIDATES[args.candidate]
    pilot_dir = paths.runs / "ae_xpower" / "pilot" / args.candidate
    out = args.out or (
        pilot_dir if args.pilot else candidate_dir(paths, args.candidate, args.version)
    )
    try:
        refuse_checkpoint(out, allow_replace=bool(args.pilot), runs=paths.runs)
    except (FileExistsError, ValueError) as error:
        p.error(str(error))
    directory = event_dir(paths)
    labels_file = labels.labels_path(directory)
    labels_bytes = labels_file.read_bytes()
    # Reuse the review parser and its validation on an immutable snapshot.
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".labels-", dir=out.parent) as snapshot:
        snapshot_file = labels.labels_path(snapshot)
        snapshot_file.parent.mkdir()
        snapshot_file.write_bytes(labels_bytes)
        saved = labels.read_saved(snapshot)
    split = make_split(saved, seldnet_split(tokeye_masks(paths)))
    config = TrainConfig(
        epochs=2 if args.pilot else args.epochs, mhd_weight=spec["mhd_weight"]
    )
    if args.pilot:
        chosen = sorted(s for s, v in split.items() if v == "train")[: args.pilot - 4]
        chosen += sorted(s for s, v in split.items() if v == "val")[:4]
        split = {s: v for s, v in split.items() if v == "test" or s in chosen}
    started = time.monotonic()
    shots = {
        s: load_shot(
            s,
            saved[s],
            store_rows(paths.spectrogram_file(EVENT, s)),
            tokeye_masks(paths),
            band=spec["band"],
        )
        for s, v in sorted(split.items())
        if v in ("train", "val")
    }
    print(
        f"loaded {len(shots)} shots in {time.monotonic() - started:.0f} s", flush=True
    )
    train = [shots[s] for s, v in sorted(split.items()) if v == "train"]
    val = [shots[s] for s, v in sorted(split.items()) if v == "val"]
    model, history, threshold = fit(
        train, val, config, log=lambda m: print(m, flush=True)
    )
    try:
        save(
            out,
            model,
            threshold=threshold,
            split=split,
            history=history,
            config=config,
            band_khz=spec["band"],
            labels_file=labels_file,
            labels_bytes=labels_bytes,
            candidate=args.candidate,
            version=args.version,
            allow_replace=bool(args.pilot),
            runs=paths.runs,
        )
    except (FileExistsError, ValueError) as error:
        p.error(str(error))
    print(
        f"wrote {out}: best epoch {best_epoch(history)}, threshold {threshold}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
