"""Train SegNet on the reviewed AE pseudo-masks.

    python -m labeler.ae.seg.train [--version V] [--out DIR] [--pilot N] [--epochs E]

**Data.** Every shot of the AE split with a pseudo-mask (`pseudo`): as input its
three cross-power rows at store level 8 (2.048 ms columns, 257 bins to
250 kHz), as target its pseudo-mask with the regions the reviewer rejected set
to background (`regions.reviewed_mask`). A decision made on other masks than
the version's own, the review page's pseudo-v1-full, reaches the target too
(`regions.transfer`), while the file clicked is still the one there (its
sha256): the pixels of the regions it rejects there are background where the
target scores them (never scored where it does not), and the rest of each
target region they touch, outside the page's regions, is IGNORE. Training
names each decision it leaves out on stderr, and `training.json` records the
shots clicked and the ones left out (`clicked_shots`, `left_out_decisions`).
`pseudo_masks.json` names each file clicked, by sha256, so `evaluate` scores
the same targets. The split
is the chosen AE model's
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

**SegNet v2** (`--version v2`, `SEG_VERSIONS["v2"]`) is the same network and
training rule on pseudo-v2 (0-250 kHz, the owner's whole windows). Its labels are
ae_xpower v3's snapshot (`models/ae_xpower/v3/review/labels.csv`, refused unless
its sha256 is v3's), never the live file; the region decisions are the live
`review/masks.jsonl`, as v1's, and reach pseudo-v2's targets through the file
clicked, as above. Its split is `pseudo.v2_split` of the
snapshot's shots, the one pseudo-v2's rules were picked on: v3's chosen model's
`split.csv` has no validation shots, but its test shots must be SegNet v2's, or
the command refuses. The blob records the band and version
(`blob_band`, `blob_version`) and `training.json` the pseudo-masks too; a blob
without them is v1's. It writes to `models/ae_seg/v2` (a pilot to
`runs/ae_seg/pilot-v2`).

**SegNet v3** (`--version v3`) is SegNet v2's recipe on pseudo-v3. It refuses
unless pseudo-v3's `rules.json` records its gate passed (`pseudo.gate_passed`),
and unless its split, `pseudo.v2_split` as v2's, is SegNet v2's own
(`models/ae_seg/v2/split.csv`), shot for shot: its test shots are v2's 60, so
its test is a second use of them, made after v2's breakdown was seen, which
`training.json` states (`test_reuse`, `reuse_note`). It writes to
`models/ae_seg/v3` (a pilot to `runs/ae_seg/pilot-v3`).

**SegNet v4** (`--version v4`) is SegNet v1's recipe on pseudo-v4 (60-250 kHz):
the live labels file and the chosen ae_xpower v1 model's split, as v1's. It
refuses unless that split is SegNet v1's own (`models/ae_seg/v1/split.csv`),
shot for shot: its test is a second use of v1's test shots, which
`training.json` states (`test_reuse`). The blob records the band and version, as
v2's. It writes to `models/ae_seg/v4` (a pilot to `runs/ae_seg/pilot-v4`).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import resource
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.nn import functional as F

from ...config import Paths, atomic_path, git_sha, sha256_of
from ...events.review import labels
from ..xpower import event_dir, read_snapshot, snapshot_file, tokeye_masks
from ..xpower import model_dir as ae_model_dir
from ..xpower.data import BAND_KHZ, SEED, store_rows
from ..xpower.evaluate import chosen_model
from ..xpower.train import read_split, refuse_checkpoint
from . import (
    EVENT,
    SEG_VERSIONS,
    VERSION,
    model_dir,
    pseudo,
    pseudo_dir,
    regions,
    reuse_note,
)
from .model import SegNet, SegNetConfig
from .pseudo import IGNORE, LEVEL, PseudoMask

THRESHOLDS = np.round(np.arange(0.10, 0.96, 0.05), 2)
MARGIN_COLS = 64


def blob_band(blob) -> tuple[float, float]:
    """The band a model was trained on and draws over, kHz: v1's blobs predate
    the key and are 80-250 kHz."""
    return tuple(float(x) for x in blob.get("band_khz", BAND_KHZ))


def blob_version(blob) -> str:
    """The SegNet version a blob records: v1's blobs predate the key."""
    return blob.get("version", "v1")


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
    paths: Paths,
    shot: int,
    decisions: dict,
    margin: int | None = MARGIN_COLS,
    *,
    pseudo_bytes: bytes | None = None,
    version: str = VERSION,
    clicked: bytes | None = None,
) -> Example:
    """One shot, cut to its scored columns and `margin` either side; the whole
    shot when `margin` is None. Its target is `version`'s pseudo-mask; `clicked`
    is the mask the shot's decision was made on when that is another
    (`regions.clicked_mask`), whose rejected regions are background too."""
    grid, values, y0, dy = store_rows(paths.spectrogram_file(EVENT, shot), LEVEL)
    file = regions.pseudo_file(paths, shot, version)
    data = file.read_bytes() if pseudo_bytes is None else pseudo_bytes
    pm = PseudoMask.load(BytesIO(data))
    if values.shape[1:] != pm.mask.shape or abs(pm.t0_ms - grid.t0_ms) > 1e-6:
        raise ValueError(f"{shot}: the pseudo-mask is not on the store's level {LEVEL}")
    decision = decisions.get(shot)
    y = regions.reviewed_mask(pm, decision, hashlib.sha256(data).hexdigest())
    if clicked is not None:
        if hashlib.sha256(clicked).hexdigest() != decision["pseudo_sha256"]:
            raise ValueError(f"{shot}: not the mask the decision was made on")
        y = regions.transfer(y, pm, PseudoMask.load(BytesIO(clicked)), decision)
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
    bundle: dict[str, bytes] | None = None,
    allow_replace: bool = False,
    runs: Path | None = None,
    band_khz=None,
    version: str | None = None,
    clicks: dict | None = None,
) -> None:
    """`inputs`: the sha256 of each file trained from (labels, masks, pseudo index).
    Given `band_khz` and `version`, which go together, the blob records both and
    `training.json` both and the version's pseudo-masks; v1's record neither.
    `clicks` goes into `training.json` as it is: `main`'s `clicked_shots` (the
    shots whose decisions on other masks reach the targets) and
    `left_out_decisions` (why each other one does not)."""
    if (band_khz is None) != (version is None):
        raise ValueError(f"{out}: a model records its band and version together")
    named, described = {}, {}
    if version is not None:
        band = [float(b) for b in band_khz]
        named = {"band_khz": tuple(band), "version": version}
        pseudo_name = SEG_VERSIONS[version].pseudo
        described = {"version": version, "pseudo": pseudo_name, "band_khz": band}
        n_test = sum(v == "test" for v in split.values())
        note = reuse_note(version, n_test)
        if note is not None:
            described["test_reuse"] = note
    refuse_checkpoint(out, allow_replace=allow_replace, runs=runs)
    lines = ["shot,split"] + [f"{s},{v}" for s, v in sorted(split.items())]
    split_bytes = ("\n".join(lines) + "\n").encode()
    inputs = {**inputs, "split_sha256": hashlib.sha256(split_bytes).hexdigest()}
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (bundle or {}).items():
        with atomic_path(out / name) as tmp:
            tmp.write_bytes(data)
    blob = {
        "state_dict": model.state_dict(),
        "config": model.config.as_dict(),
        "threshold": threshold,
        "train": asdict(config),
        "best_epoch": best_epoch(history),
        "inputs": inputs,
        "git_sha": git_sha(),
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **named,
    }
    with atomic_path(out / "model.pt") as tmp:
        torch.save(blob, tmp)
    with atomic_path(out / "split.csv") as tmp:
        tmp.write_bytes(split_bytes)
    seconds = [row["seconds"] for row in history]
    record = {
        **described,
        "inputs": inputs,
        "history": history,
        "best_epoch": best_epoch(history),
        "threshold": threshold,
        "seconds_per_epoch": round(float(np.mean(seconds)), 1) if seconds else None,
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "counts": {
            v: sum(x == v for x in split.values()) for v in sorted(set(split.values()))
        },
        **(clicks or {}),
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


def frozen_clicked(paths: Paths, shot: int, decision, manifest: dict) -> bytes | None:
    """The mask the shot's decision was made on, when `manifest` names it (a
    model's `pseudo_masks.json`, or the pseudo-masks `evaluation.json` records);
    refused when the file is no longer the one named."""
    name = regions.clicked_name(shot, decision) if decision else None
    if name not in manifest:
        return None
    path = regions.decided_file(paths, shot, decision)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest[name]:
        raise ValueError(f"{path}: not the mask clicked when the model was trained")
    return data


def read_review_bytes(labels_bytes: bytes, masks_bytes: bytes) -> tuple[dict, dict]:
    """Parse immutable review snapshots with the review modules' own parsers."""
    with TemporaryDirectory(prefix="ae-seg-review-") as directory:
        file = labels.labels_path(directory)
        file.parent.mkdir(parents=True)
        file.write_bytes(labels_bytes)
        regions.log_path(directory).write_bytes(masks_bytes)
        return labels.read_saved(directory), regions.read_decisions(directory)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--version",
        choices=sorted(SEG_VERSIONS),
        default=VERSION,
        help="v1 (the default): pseudo-v1 and the live labels; v2: pseudo-v2 and "
        "ae_xpower v3's snapshot; v3: pseudo-v3, the same snapshot and v2's split; "
        "v4: pseudo-v4 (60-250 kHz), v1's labels rule and split",
    )
    p.add_argument(
        "--out", type=Path, help="default $LABELER_ROOT/models/ae_seg/<version>"
    )
    p.add_argument(
        "--pilot",
        type=int,
        default=0,
        help="6-20 shots (4 of them validation), 2 epochs, to runs/ae_seg/pilot "
        "(v2: pilot-v2; v3: pilot-v3; v4: pilot-v4)",
    )
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    args = p.parse_args(argv)
    if args.pilot and not 6 <= args.pilot <= 20:
        p.error("a pilot is 6 to 20 shots")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    paths = Paths.from_env()
    version, spec = args.version, SEG_VERSIONS[args.version]
    v1 = version == "v1"  # v1 makes every call as it did before v2
    # v2's recipe (v2, v3): an ae_xpower snapshot and pseudo.v2_split. v1's (v1,
    # v4): the live labels and the chosen ae_xpower v1 model's split.
    v2_recipe = spec.labels is not None
    if spec.gated and not pseudo.gate_passed(paths, version):
        p.error(
            f"{pseudo_dir(paths, version) / 'rules.json'}: {spec.pseudo}'s gate "
            f"has not passed; run python -m labeler.ae.seg.pseudo --version {version}"
        )
    pilot = paths.runs / "ae_seg" / ("pilot" if v1 else f"pilot-{version}")
    out = args.out or (pilot if args.pilot else model_dir(paths, version))
    try:
        refuse_checkpoint(out, allow_replace=bool(args.pilot), runs=paths.runs)
    except (FileExistsError, ValueError) as error:
        p.error(str(error))
    directory = event_dir(paths)
    # Every version reads the live region decisions (`regions.transfer`).
    masks_file = regions.log_path(directory)
    if not v2_recipe:
        labels_file = labels.labels_path(directory)
        labels_bytes = labels_file.read_bytes()
    else:
        labels_file = snapshot_file(paths, spec.labels)
        try:
            labels_bytes, _ = read_snapshot(paths, spec.labels)
        except (OSError, ValueError) as error:
            p.error(f"label snapshot {spec.labels}: {type(error).__name__}: {error}")
    try:
        masks_bytes = masks_file.read_bytes()
        masks_hash = hashlib.sha256(masks_bytes).hexdigest()
    except FileNotFoundError:
        masks_bytes, masks_hash = b"", None
    saved, decisions = read_review_bytes(labels_bytes, masks_bytes)
    ae_models = ae_model_dir(paths, spec.ae_version)
    ae_file = chosen_model(ae_models)
    input_files = {
        "labels_sha256": labels_file,
        "masks_sha256": masks_file,
        "pseudo_index_sha256": pseudo_dir(paths, version) / "index.csv",
        "ae_model_sha256": ae_file,
        "ae_split_sha256": ae_file.parent / "split.csv",
        "ae_chosen_sha256": ae_models / "chosen.json",
    }
    initial = {
        key: _sha(path)
        for key, path in input_files.items()
        if key not in ("labels_sha256", "masks_sha256")
    }
    initial.update(
        labels_sha256=hashlib.sha256(labels_bytes).hexdigest(),
        masks_sha256=masks_hash,
    )
    split = read_split(ae_file.parent / "split.csv")
    if v2_recipe:
        # SegNet v2's own split, with validation shots (v3's has none), refused
        # unless its test shots are the chosen ae_xpower model's.
        masks = tokeye_masks(paths, spec.ae_version)
        try:
            ours = pseudo.v2_split(sorted(saved), masks)
        except (KeyError, OSError, ValueError) as error:
            p.error(f"{masks}: {type(error).__name__}: {error}")
        theirs = sorted(s for s, v in split.items() if v == "test")
        test = sorted(s for s, v in ours.items() if v == "test")
        if theirs != test:
            p.error(
                f"{ae_file.parent / 'split.csv'}: test shots {theirs} differ from "
                f"SegNet {version}'s {test}"
            )
        split = ours
    have = {s for s in split if regions.pseudo_file(paths, s, version).is_file()}
    split = {s: v for s, v in split.items() if s in have}
    if spec.test_of is not None:
        # The earlier version's split, shot for shot: the same test shots.
        earlier = model_dir(paths, spec.test_of) / "split.csv"
        try:
            before = read_split(earlier)
        except (OSError, ValueError) as error:
            p.error(f"{earlier}: {type(error).__name__}: {error}")
        if before != split:
            p.error(f"{earlier}: SegNet {spec.test_of}'s split differs from ours")
    if args.pilot:
        chosen = sorted(s for s, v in split.items() if v == "train")[: args.pilot - 4]
        chosen += sorted(s for s, v in split.items() if v == "val")[:4]
        split = {s: v for s, v in split.items() if v == "test" or s in chosen}
    missing = sorted(set(split) - saved.keys())
    if missing:
        p.error(f"{labels_file}: no archived labels for split shots {missing}")
    mask_files = {
        f"{s}.npz": regions.pseudo_file(paths, s, version) for s in sorted(split)
    }
    mask_files["index.csv"] = pseudo_dir(paths, version) / "index.csv"
    clicked, left_out = {}, []
    for s in sorted(split):
        path, why = regions.clicked_mask(paths, s, decisions.get(s), spec.pseudo)
        if why is not None:
            left_out.append(why)
        if path is not None:
            clicked[s] = path
            mask_files[regions.clicked_name(s, decisions[s])] = path
    if left_out:
        print(
            f"warning: {len(left_out)} shots' mask decisions are left out:",
            *(f"  {why}" for why in left_out),
            sep="\n",
            file=sys.stderr,
        )
    print(f"{len(clicked)} shots' mask decisions made on other masks reach the targets")
    clicks = {"clicked_shots": sorted(clicked), "left_out_decisions": left_out}
    manifest = {name: sha256_of(path) for name, path in mask_files.items()}
    manifest_bytes = (json.dumps(manifest, indent=1) + "\n").encode()
    inputs = {
        **initial,
        "masks_sha256": hashlib.sha256(masks_bytes).hexdigest(),
        "pseudo_masks_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    bundle = {
        "review/labels.csv": labels_bytes,
        "review/masks.jsonl": masks_bytes,
        "pseudo_masks.json": manifest_bytes,
    }
    started = time.monotonic()
    at = {} if v1 else {"version": version}
    examples = {}
    for s, v in sorted(split.items()):
        if v in ("train", "val"):
            more = {"clicked": clicked[s].read_bytes()} if s in clicked else {}
            examples[s] = load_example(paths, s, decisions, **at, **more)
    print(f"loaded {len(examples)} shots in {time.monotonic() - started:.0f} s")
    train = [examples[s] for s, v in sorted(split.items()) if v == "train"]
    val = [examples[s] for s, v in sorted(split.items()) if v == "val"]
    config = TrainConfig(epochs=2 if args.pilot else args.epochs)
    model, history, threshold = fit(
        train, val, config, log=lambda m: print(m, flush=True)
    )
    for key, path in input_files.items():
        if _sha(path) != initial[key]:
            print(f"training input changed: {path}; refusing to save", file=sys.stderr)
            return 1
    for name, path in mask_files.items():
        if _sha(path) != manifest[name]:
            print(f"training input changed: {path}; refusing to save", file=sys.stderr)
            return 1
    try:
        save(
            out,
            model,
            threshold=threshold,
            split=split,
            history=history,
            config=config,
            inputs=inputs,
            bundle=bundle,
            allow_replace=bool(args.pilot),
            runs=paths.runs,
            band_khz=None if v1 else spec.band_khz,
            version=None if v1 else version,
            clicks=clicks,
        )
    except (FileExistsError, ValueError) as error:
        p.error(str(error))
    print(f"wrote {out}: best epoch {best_epoch(history)}, threshold {threshold}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
