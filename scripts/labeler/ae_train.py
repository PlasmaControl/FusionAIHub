#!/usr/bin/env python
"""Task 7b: train the AE activity + frequency SELDNet.

Plain torch - the phase-3 venv has no Lightning and none is added. One run is
one ``(target, loss)`` pair; the three runs of the task are three SLURM array
tasks of ``ae_train.sbatch``.

The activity target is built at load time from the dataset's two label arrays
(task 7a, notch 0.8):

``--target threeway`` (the decision recorded in the phase-3 plan)

    frame                      target  weight
    annotated & active           1       1
    ~annotated & ~active         0       1
    active & ~annotated        (none)    0     <- the under-count region
    annotated & ~active        (none)    0     <- annotation the filter missed

    The model may call the two ignored regions active for free, which is the
    "more false positives are fine" instruction, and is never taught that a
    mask-only frame is a negative.

``--target mask`` is the comparison run: target = ``active``, weight 1
everywhere, so the effect of the three-way target is measured rather than
assumed.

The frequency head is trained only on frames whose activity target is 1 and
whose ``freq_khz`` is finite, in the normalised band coordinate
``(f_khz - 80) / 170``.

Training sees 710-frame windows; validation runs each 7820-frame record whole
in one forward pass (nothing pools time), so the reported metrics have no
window-edge effects and cover every frame of the 60 validation shots.

Usage (inside the phase-3 venv, with ``src`` on PYTHONPATH):

    python scripts/labeler/ae_train.py --target threeway --loss sce
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from labeler.ae.model import (
    AeLossConfig,
    AeSeldNet,
    AeSeldNetConfig,
    ae_loss,
    denormalise_freq,
    normalise_freq,
)
from labeler.env import getenv

DEFAULT_DATASET = Path("/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/ae/dataset")
DEFAULT_OUT = Path(
    "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/models/d3d_ae_activity_seldnet"
)


# --------------------------------------------------------------------------
# labels


def build_targets(
    active: np.ndarray, annotated: np.ndarray, freq_khz: np.ndarray, target: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(activity target, activity weight, normalised freq target, freq weight)."""
    act = active.astype(bool)
    ann = annotated.astype(bool)
    if target == "threeway":
        positive = ann & act
        negative = ~ann & ~act
        y = positive.astype(np.float32)
        w = (positive | negative).astype(np.float32)
    elif target == "mask":
        positive = act
        y = act.astype(np.float32)
        w = np.ones_like(y, dtype=np.float32)
    else:
        raise ValueError(f"unknown target {target!r}; expected 'threeway' or 'mask'")
    finite = np.isfinite(freq_khz)
    fw = (positive & finite).astype(np.float32)
    ft = np.where(finite, (freq_khz - 80.0) / 170.0, 0.0).astype(np.float32)
    return y, w, ft, fw


@dataclass(frozen=True)
class ShotLabels:
    path: Path
    shot: str
    split: str
    y: np.ndarray
    w: np.ndarray
    ft: np.ndarray
    fw: np.ndarray
    annotated: np.ndarray
    active: np.ndarray
    freq_khz: np.ndarray


def load_labels(dataset_dir: Path, target: str, limit: int | None = None) -> list[ShotLabels]:
    out: list[ShotLabels] = []
    for path in sorted(dataset_dir.glob("*.npz")):
        with np.load(path) as z:
            active = z["active"]
            annotated = z["annotated"]
            freq = z["freq_khz"]
            split = str(z["split"])
        y, w, ft, fw = build_targets(active, annotated, freq, target)
        out.append(
            ShotLabels(
                path=path,
                shot=path.stem.split("_")[0],
                split=split,
                y=y,
                w=w,
                ft=ft,
                fw=fw,
                annotated=annotated.astype(bool),
                active=active.astype(bool),
                freq_khz=freq,
            )
        )
        if limit is not None and len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------
# data


class ShotWindows(Dataset):
    """One item is ``windows_per_shot`` random windows out of one shot.

    The npz members are stored uncompressed, so reading a shot's ``spec`` is a
    20 MB sequential read (~0.1-0.3 s). Drawing several windows from each read
    is what keeps the loader off the critical path: tasks 6 and 7a were both
    host-I/O bound.
    """

    def __init__(
        self,
        shots: list[ShotLabels],
        window_frames: int,
        windows_per_shot: int,
        seed: int = 0,
    ) -> None:
        self.shots = shots
        self.window_frames = window_frames
        self.windows_per_shot = windows_per_shot
        self.epoch = 0
        self.seed = seed

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.shots)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rec = self.shots[index]
        rng = np.random.default_rng((self.seed, self.epoch, index))
        with np.load(rec.path) as z:
            spec = z["spec"]  # (4, 348, T) float16
        n_frames = spec.shape[2]
        starts = rng.integers(0, n_frames - self.window_frames + 1, self.windows_per_shot)
        w = self.window_frames
        # (K, C, T, F): transpose the stored (C, F, T) into (C, T, F).
        x = np.stack([spec[:, :, s : s + w].transpose(0, 2, 1) for s in starts])
        idx = np.stack([np.arange(s, s + w) for s in starts])
        return {
            "spec": torch.from_numpy(np.ascontiguousarray(x)),
            "y": torch.from_numpy(rec.y[idx]),
            "w": torch.from_numpy(rec.w[idx]),
            "ft": torch.from_numpy(rec.ft[idx]),
            "fw": torch.from_numpy(rec.fw[idx]),
        }


def collate_windows(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.cat([it[k] for it in items], dim=0) for k in items[0]}


def load_record(rec: ShotLabels) -> torch.Tensor:
    """The whole 7820-frame record as ``(1, C, T, F)`` float16."""
    with np.load(rec.path) as z:
        spec = z["spec"]
        x = np.ascontiguousarray(spec.transpose(0, 2, 1))
    return torch.from_numpy(x).unsqueeze(0)


# --------------------------------------------------------------------------
# metrics


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank AUROC; NaN if one class is missing."""
    pos = int(labels.sum())
    neg = int(labels.size - pos)
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    sorted_scores = scores[order]
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    # average ranks over ties
    i = 0
    while i < sorted_scores.size:
        j = i
        while j + 1 < sorted_scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[labels.astype(bool)].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def frame_metrics(
    prob: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    annotated: np.ndarray,
    active: np.ndarray,
    freq_pred_khz: np.ndarray,
    freq_true_khz: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    pred = prob >= threshold
    ann = annotated.astype(bool)
    tp = int((pred & ann).sum())
    out = {
        "n_frames": int(prob.size),
        "predicted_positive_frac": float(pred.mean()),
        "annotated_frac": float(ann.mean()),
        "active_frac": float(active.astype(bool).mean()),
        "recall_annotated": float(tp / ann.sum()) if ann.sum() else float("nan"),
        "precision_annotated": float(tp / pred.sum()) if pred.sum() else float("nan"),
        "auroc_annotated": auroc(prob, ann.astype(np.float64)),
        "auroc_active": auroc(prob, active.astype(bool).astype(np.float64)),
    }
    out["f1_annotated"] = (
        float(2 * out["recall_annotated"] * out["precision_annotated"]
              / (out["recall_annotated"] + out["precision_annotated"]))
        if out["recall_annotated"] and out["precision_annotated"]
        and math.isfinite(out["recall_annotated"]) and math.isfinite(out["precision_annotated"])
        and (out["recall_annotated"] + out["precision_annotated"]) > 0
        else float("nan")
    )
    labelled = w.astype(bool)
    out["auroc_labelled"] = (
        auroc(prob[labelled], y[labelled].astype(np.float64))
        if labelled.any()
        else float("nan")
    )
    out["labelled_frac"] = float(labelled.mean())
    conf = ann & active.astype(bool) & np.isfinite(freq_true_khz)
    out["freq_mae_khz"] = (
        float(np.abs(freq_pred_khz[conf] - freq_true_khz[conf]).mean()) if conf.any() else float("nan")
    )
    out["n_freq_frames"] = int(conf.sum())
    return out


# --------------------------------------------------------------------------
# evaluation over whole records


@torch.no_grad()
def predict_records(
    model: AeSeldNet, shots: list[ShotLabels], device: torch.device, amp_dtype
) -> dict[str, dict[str, np.ndarray]]:
    model.eval()
    out: dict[str, dict[str, np.ndarray]] = {}
    for rec in shots:
        x = load_record(rec).to(device, non_blocking=True).float()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            y = model(x)
        y = y.float()[0]
        prob = torch.sigmoid(y[:, 0]).cpu().numpy().astype(np.float64)
        freq = denormalise_freq(y[:, 1]).cpu().numpy().astype(np.float64)
        out[rec.path.stem] = {"prob": prob, "freq_khz": freq}
        del x, y
    return out


def evaluate(
    model: AeSeldNet,
    shots: list[ShotLabels],
    device: torch.device,
    amp_dtype,
    loss_cfg: AeLossConfig,
) -> dict[str, float]:
    preds = predict_records(model, shots, device, amp_dtype)
    prob = np.concatenate([preds[s.path.stem]["prob"] for s in shots])
    freq = np.concatenate([preds[s.path.stem]["freq_khz"] for s in shots])
    y = np.concatenate([s.y for s in shots])
    w = np.concatenate([s.w for s in shots])
    ann = np.concatenate([s.annotated for s in shots])
    act = np.concatenate([s.active for s in shots])
    ftrue = np.concatenate([s.freq_khz for s in shots])
    metrics = frame_metrics(prob, y, w, ann, act, freq, ftrue)
    # validation loss, on the same weighted definition the training loop uses
    logit = torch.from_numpy(np.log(np.clip(prob, 1e-7, 1 - 1e-7) / np.clip(1 - prob, 1e-7, 1)))
    out = torch.stack([logit, torch.from_numpy(normalise_freq(torch.from_numpy(freq)).numpy())], -1)
    ft = np.concatenate([s.ft for s in shots])
    fw = np.concatenate([s.fw for s in shots])
    parts = ae_loss(
        out.unsqueeze(0).float(),
        torch.from_numpy(y).unsqueeze(0),
        torch.from_numpy(w).unsqueeze(0),
        torch.from_numpy(ft).unsqueeze(0),
        torch.from_numpy(fw).unsqueeze(0),
        loss_cfg,
    )
    metrics["val_loss"] = float(parts["loss"])
    metrics["val_activity_loss"] = float(parts["activity"])
    metrics["val_freq_loss"] = float(parts["freq"])
    return metrics


# --------------------------------------------------------------------------
# provenance


def git_sha() -> str:
    sha = getenv("LABELER_GIT_SHA")
    if sha:
        return sha
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance must never fail a training run
        return "unknown"


def atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", choices=["threeway", "mask"], default="threeway")
    p.add_argument("--loss", choices=["sce", "bce"], default="sce")
    p.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--window-frames", type=int, default=710)
    p.add_argument("--windows-per-shot", type=int, default=8)
    p.add_argument("--batch-windows", type=int, default=16)
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--lambda-f", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument(
        "--pool-sizes",
        default="6,2,29",
        help=(
            "comma-separated frequency pool sizes; the product must divide the 348-bin "
            "band exactly. The default (6,2,29) takes 348 -> 1 row, which is the "
            "reference SELDNet arrangement but leaves the GRU no frequency position; "
            "'6,2' keeps 29 rows and is what the frequency head needs."
        ),
    )
    p.add_argument(
        "--tag",
        default="",
        help="suffix on the checkpoint / training.json name, for architecture variants",
    )
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit-shots", type=int, default=None, help="smoke test")
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    amp_dtype = torch.bfloat16
    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
    num_workers = args.num_workers if args.num_workers is not None else max(1, n_cpus)

    t_start = time.time()
    shots = load_labels(args.dataset_dir, args.target, limit=args.limit_shots)
    train = [s for s in shots if s.split == "train"]
    valid = [s for s in shots if s.split == "valid"]
    if not train or not valid:
        raise SystemExit(f"empty split: {len(train)} train, {len(valid)} valid")
    print(
        f"{len(train)} train / {len(valid)} valid shots, target={args.target}, "
        f"loss={args.loss}, labels loaded in {time.time() - t_start:.1f} s",
        flush=True,
    )
    label_stats = {
        "train_positive_frac": float(np.mean([s.y[s.w > 0].mean() for s in train])),
        "train_labelled_frac": float(np.mean([s.w.mean() for s in train])),
        "valid_positive_frac": float(np.mean([s.y[s.w > 0].mean() for s in valid])),
        "valid_labelled_frac": float(np.mean([s.w.mean() for s in valid])),
    }
    print("label stats:", label_stats, flush=True)

    dataset = ShotWindows(train, args.window_frames, args.windows_per_shot, seed=args.seed)
    shots_per_batch = max(1, args.batch_windows // args.windows_per_shot)
    loader = DataLoader(
        dataset,
        batch_size=shots_per_batch,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_windows,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    with np.load(train[0].path) as z:
        n_freq = int(z["spec"].shape[1])
        n_channels = int(z["spec"].shape[0])
    model_cfg = AeSeldNetConfig(
        in_channels=n_channels,
        n_freq=n_freq,
        pool_sizes=tuple(int(p) for p in args.pool_sizes.split(",")),
    )
    model = AeSeldNet(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"AeSeldNet: {n_params/1e6:.2f} M parameters, pool {model_cfg.pool_sizes}", flush=True)

    loss_cfg = AeLossConfig(
        loss=args.loss, alpha=args.alpha, beta=args.beta, lambda_f=args.lambda_f
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.max_epochs, eta_min=args.min_lr
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.target}_{args.loss}" + (f"_{args.tag}" if args.tag else "")
    ckpt_path = args.out_dir / f"ae_seldnet_{tag}.pt"
    json_path = args.out_dir / f"training_{tag}.json"

    history: list[dict] = []
    best = math.inf
    best_epoch = -1
    bad = 0
    stopped_early = False
    for epoch in range(args.max_epochs):
        dataset.set_epoch(epoch)
        model.train()
        t0 = time.time()
        sums = {"loss": 0.0, "activity": 0.0, "freq": 0.0}
        n_steps = 0
        data_wait = 0.0
        t_last = time.time()
        for batch in loader:
            data_wait += time.time() - t_last
            x = batch["spec"].to(device, non_blocking=True).float()
            y = batch["y"].to(device, non_blocking=True)
            w = batch["w"].to(device, non_blocking=True)
            ft = batch["ft"].to(device, non_blocking=True)
            fw = batch["fw"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"
            ):
                out = model(x)
            parts = ae_loss(out.float(), y, w, ft, fw, loss_cfg)
            opt.zero_grad(set_to_none=True)
            parts["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            for k in sums:
                sums[k] += float(parts[k].detach())
            n_steps += 1
            t_last = time.time()
        sched.step()
        train_time = time.time() - t0

        t1 = time.time()
        metrics = evaluate(model, valid, device, amp_dtype, loss_cfg)
        val_time = time.time() - t1
        row = {
            "epoch": epoch,
            "lr": float(opt.param_groups[0]["lr"]),
            "train_loss": sums["loss"] / max(1, n_steps),
            "train_activity_loss": sums["activity"] / max(1, n_steps),
            "train_freq_loss": sums["freq"] / max(1, n_steps),
            "train_seconds": train_time,
            "train_dataloader_wait_seconds": data_wait,
            "valid_seconds": val_time,
            "steps": n_steps,
            **{f"valid_{k}": v for k, v in metrics.items()},
        }
        history.append(row)
        print(
            f"epoch {epoch:2d} train {row['train_loss']:.4f} "
            f"val {metrics['val_loss']:.4f} "
            f"recall {metrics['recall_annotated']:.3f} "
            f"prec {metrics['precision_annotated']:.3f} "
            f"pp {metrics['predicted_positive_frac']:.3f} "
            f"auroc(lab) {metrics['auroc_labelled']:.3f} "
            f"fMAE {metrics['freq_mae_khz']:.2f} kHz "
            f"[{train_time:.0f}s train ({data_wait:.0f}s wait) / {val_time:.0f}s val]",
            flush=True,
        )

        if metrics["val_loss"] < best - 1e-6:
            best = metrics["val_loss"]
            best_epoch = epoch
            bad = 0
            tmp = ckpt_path.with_suffix(".pt.tmp")
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": model_cfg.as_dict(),
                    "loss_config": loss_cfg.as_dict(),
                    "target": args.target,
                    "epoch": epoch,
                    "val_loss": best,
                    "valid_metrics": metrics,
                    "freq_lo_khz": 80.0,
                    "freq_span_khz": 170.0,
                    "git_sha": git_sha(),
                },
                tmp,
            )
            tmp.replace(ckpt_path)
        else:
            bad += 1
            if bad >= args.patience:
                stopped_early = True
                print(f"early stop at epoch {epoch} (patience {args.patience})", flush=True)
                break

        atomic_write_json(
            json_path,
            {
                "run": tag,
                "status": "running",
                "history": history,
                "best_epoch": best_epoch,
                "best_val_loss": best,
            },
        )

    wall = time.time() - t_start
    payload = {
        "run": tag,
        "status": "finished",
        "generated": datetime.now(UTC).astimezone().strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_sha": git_sha(),
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "environment": {
            "python": platform.python_version(),
            "executable": sys.executable,
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "cpus_per_task": n_cpus,
            "num_workers": num_workers,
        },
        "model": {
            "class": "AeSeldNet",
            "config": model_cfg.as_dict(),
            "parameters": n_params,
            "pool_sizes_note": (
                f"{model_cfg.pool_sizes} multiply to {n_freq}; strided pooling needs "
                "sizes that divide the frequency axis exactly"
            ),
        },
        "loss": loss_cfg.as_dict(),
        "label": {
            "target": args.target,
            "rule": (
                "threeway: 1 on annotated&active, 0 on ~annotated&~active, weight 0 "
                "elsewhere"
                if args.target == "threeway"
                else "mask: target = active (task 7a cleaned mask, notch 0.8), weight 1"
            ),
            **label_stats,
        },
        "data": {
            "dataset_dir": str(args.dataset_dir),
            "n_train_shots": len(train),
            "n_valid_shots": len(valid),
            "window_frames": args.window_frames,
            "windows_per_shot": args.windows_per_shot,
            "batch_windows": shots_per_batch * args.windows_per_shot,
            "validation": "whole 7820-frame records, one forward pass each",
        },
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best,
        "stopped_early": stopped_early,
        "epochs_run": len(history),
        "wall_seconds": wall,
        "checkpoint": str(ckpt_path),
    }
    atomic_write_json(json_path, payload)
    print(f"best epoch {best_epoch} val_loss {best:.4f}; wall {wall/60:.1f} min", flush=True)
    print(f"wrote {ckpt_path} and {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
