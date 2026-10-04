#!/usr/bin/env python
"""Benchmark the paper's BES confinement classifier (L, H, QH, WP QH) on our labels.

The paper's model is a small 3D-convolutional network on log-spectral blocks of the
6 x 8 BES array (its test F1: L 0.94, H 0.97, QH 0.94, WP QH 0.90); the recipe is in
docs/labeler/confinement_bes_benchmark.md. Its weights and code were not found, so
the recipe is retrained here and scored on shots the network never saw.

**Data.** The corpus BES (``<shot>_processed.h5``, group ``bes``: 64 channels at
500 kHz) of the shots in the merged interval table
(``runs/labeler/confinement/v1/merged_intervals.csv``) that the corpus holds BES for.
A window is 1024 samples (2.05 ms) lying wholly inside one interval of a single
regime and starts every 2048 samples. Channels 8-55 (rows 1-6 of the 8 x 8 grid,
channel = row * 8 + column) are the 6 x 8 block; the paper does not say which rows.
The paper's pre-processing follows: band-pass 2.5-150 kHz (4th-order Butterworth,
causal), standardise with the training set's per-channel standard deviation, split
the window in 2 sub-windows of 512 and each in 2 segments of 256, FFT, log10 of the
squared magnitude of bins 0-127, average the 2 segments: 2 x 128 features per channel.
Standardising a signal subtracts 2 log10(sigma) from its log power, so the features
are stored for the raw band-passed signal and the per-channel offset is applied per
fold, from that fold's training windows.

**Network and training.** Dropout, Conv3d (10 kernels (3, 3, 5), zero padding,
groups = 2), batch norm, LeakyReLU, MaxPool3d (1, 2, 4), an MLP of two 60-unit
layers, 4 logits; cross-entropy; AdamW with one learning rate for the convolution
and a smaller one for the MLP; 60,000 steps, evaluated on a validation set every 500
with early stopping after 30 evaluations without a better loss; the checkpoint
with the best validation macro-F1 is kept. Not in the paper's excerpt, so assumed:
dropout 0.2, learning rates 1e-3 (convolution) and 1e-4 (MLP), weight decay 0.01,
batch 256, natural class frequencies, window stride, sampling rate, channel rows.

**Protocol.** Shot-grouped 5-fold cross-validation over the labelled BES shots,
folds dealt within class-presence strata. Each fold's test shots are scored by a
network trained on three other folds and validated on a sixth of the rest, so every
prediction is out of sample. Scores: per-class precision, recall and F1 (one against
the rest), macro averages (the paper's "Average"), one-vs-rest AUROC and AUPRC,
confusion matrix, 95 % intervals bootstrapped over shots; on the diagnostic-summary
model's test shots, on agreement-only windows, and on 50 ms bins beside that
model's four-class baseline.

    python scripts/labeler/confinement_bes_benchmark.py features
    python scripts/labeler/confinement_bes_benchmark.py train [--folds 0 1 2 --device cuda:0]
    python scripts/labeler/confinement_bes_benchmark.py evaluate

``features`` and ``evaluate`` need only numpy, scipy, pandas and h5py; ``train``
needs torch and a GPU (``-e shot-design``).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[2]
LABELER = Path(os.environ.get("LABELER_ROOT", "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker"))
CORPUS = Path("/scratch/gpfs/EKOLEMEN/foundation_model")
RUN = REPO / "runs/labeler/confinement/v1"
DEFAULT_WORK = LABELER / "benchmarks/confinement/bes"
DEFAULT_OUT = REPO / "outputs/labeler/confinement/bes"

CLASSES = ("L", "H", "QH", "WP")
FS = 500e3
WINDOW = 1024
STRIDE = 2048
PAD = 4096
BAND = (2.5e3, 150e3)
CH0, CH1 = 8, 56
ROWS, COLS, FREQS = 6, 8, 128
CHANNELS = CH1 - CH0
SOS = butter(4, BAND, btype="bandpass", fs=FS, output="sos")

FOLDS = 5
FOLD_SEED = 20261001
VAL_GROUPS = 6
REPLICATES = 1000
BOOT_SEED = 20261001
#: F1, precision, recall of the paper's test table.
PAPER = {
    "L": (0.94, 0.98, 0.90),
    "H": (0.97, 0.95, 0.98),
    "QH": (0.94, 0.92, 0.96),
    "WP": (0.90, 0.95, 0.86),
}
BIN_MS = 50.0


def shot_features(shot: int, rows: list[tuple]) -> dict | None:
    """Log-spectral blocks of the windows inside one shot's labelled intervals.

    ``rows`` holds (t_start ms, t_end ms, class index, status) per interval. Returns
    features (n, 2, 48, 128) float16, the windows' mean power per channel (n, 48),
    their metadata and the count of windows dropped for non-finite samples.
    """
    feats, power, meta, dropped = [], [], [], 0
    with h5py.File(CORPUS / f"{shot}_processed.h5", "r") as f:
        if "bes" not in f:
            return None
        bes = f["bes"]
        x = bes["xdata"]
        n, t0, t1 = x.shape[0], float(x[0]), float(x[-1])
        dt = (t1 - t0) / (n - 1)
        if abs(dt * FS - 1.0) > 0.01:
            raise ValueError(f"{shot}: BES step {dt * 1e6:.4f} us, not {1e6 / FS} us")
        for t_start, t_end, label, status in rows:
            lo = max(0, int(np.ceil((t_start / 1000 - t0) / dt)))
            hi = min(n, int(np.floor((t_end / 1000 - t0) / dt)))
            if hi - lo < WINDOW:
                continue
            starts = lo + STRIDE * np.arange((hi - lo - WINDOW) // STRIDE + 1)
            read = max(0, lo - PAD)
            filt = sosfilt(SOS, bes["ydata"][CH0:CH1, read:hi].astype(np.float64), axis=1)
            for c0 in range(0, len(starts), 128):
                chunk = starts[c0 : c0 + 128]
                w = filt[:, (chunk - read)[:, None] + np.arange(WINDOW)].transpose(1, 0, 2)
                ok = np.isfinite(w).all(axis=(1, 2))
                dropped += int((~ok).sum())
                if not ok.any():
                    continue
                w, chunk = w[ok], chunk[ok]
                seg = w.reshape(len(w), CHANNELS, 2, 2, 256)
                spec = np.abs(np.fft.rfft(seg, axis=-1)[..., :FREQS]) ** 2
                logs = np.log10(spec + 1e-30).mean(axis=3)
                feats.append(logs.transpose(0, 2, 1, 3).astype(np.float16))
                power.append((w**2).mean(axis=2).astype(np.float32))
                begin = 1000 * (t0 + chunk * dt)
                meta += [
                    (shot, b, b + 0.5 * WINDOW * dt * 1000, label, status) for b in begin
                ]
    if not feats:
        return {"dropped": dropped} if dropped else None
    return {
        "feats": np.concatenate(feats),
        "power": np.concatenate(power),
        "meta": meta,
        "dropped": dropped,
    }


def build_features(args: argparse.Namespace) -> None:
    """Write features.npy, power.npy, windows.csv and features.json to the work dir."""
    mi = pd.read_csv(RUN / "merged_intervals.csv")
    mi = mi[mi.regimes.isin(CLASSES)]
    mi = mi[[(CORPUS / f"{s}_processed.h5").exists() for s in mi.shot]]
    jobs = {
        int(s): [
            (r.t_start, r.t_end, CLASSES.index(r.regimes), r.status)
            for r in g.itertuples()
        ]
        for s, g in mi.groupby("shot")
    }
    started = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        done = list(pool.map(shot_features, jobs.keys(), jobs.values()))
    kept = [(s, d) for s, d in zip(jobs, done, strict=True) if d and "feats" in d]
    args.work_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.work_dir / "features.npy", np.concatenate([d["feats"] for _, d in kept]))
    np.save(args.work_dir / "power.npy", np.concatenate([d["power"] for _, d in kept]))
    windows = pd.DataFrame(
        [m for _, d in kept for m in d["meta"]],
        columns=["shot", "start_ms", "center_ms", "label", "status"],
    )
    windows.to_csv(args.work_dir / "windows.csv", index=False)
    summary = {
        "shots": len(kept),
        "windows": len(windows),
        "dropped_nonfinite": sum(d["dropped"] for d in done if d),
        "per_class_windows": {c: int((windows.label == i).sum()) for i, c in enumerate(CLASSES)},
        "per_class_shots": {
            c: int(windows[windows.label == i].shot.nunique()) for i, c in enumerate(CLASSES)
        },
        "seconds": round(time.time() - started, 1),
    }
    (args.work_dir / "features.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


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


def block_roles(windows: pd.DataFrame) -> np.ndarray:
    """Roles for the leaky diagnostic: blocks of 50 consecutive windows (0.2 s at
    500 kHz) of a shot go to train, validation or test at random (60/20/20), so a test
    window sits beside training windows of its own shot."""
    block = windows.groupby("shot").cumcount().to_numpy() // 50
    _, inverse = np.unique(windows.shot.to_numpy() * 10_000 + block, return_inverse=True)
    draw = np.random.default_rng(FOLD_SEED + 100).random(inverse.max() + 1)
    return np.where(draw < 0.6, 0, np.where(draw < 0.8, 1, 2))[inverse]


def fold_roles(windows: pd.DataFrame, fold: int) -> tuple[dict[int, int], np.ndarray]:
    """Fold of every shot, and the role (0 train, 1 validation, 2 test) of each window."""
    signature = {
        int(s): "".join(CLASSES[i] + "," for i in sorted(set(g.label)))
        for s, g in windows.groupby("shot")
    }
    shots = sorted(signature)
    folds = deal(shots, signature, FOLDS, np.random.default_rng(FOLD_SEED))
    rest = [s for s in shots if folds[s] != fold]
    val = deal(rest, signature, VAL_GROUPS, np.random.default_rng(FOLD_SEED + 1 + fold))
    role = {s: 2 if folds[s] == fold else (1 if val[s] == 0 else 0) for s in shots}
    return folds, windows.shot.map(role).to_numpy()


def macro_f1(conf) -> float:
    """Macro F1 of a 4 x 4 confusion matrix (rows true), over the classes present."""
    tp = conf.diag().double()
    denom = conf.sum(0).double() + conf.sum(1).double()
    present = conf.sum(1) > 0
    return float((2 * tp[present] / denom[present].clamp(min=1)).mean())


def run_fold(args: argparse.Namespace, fold: int) -> None:
    """Train on three folds, validate on a sixth of the rest, predict the held-out fold."""
    import torch
    from torch import nn

    class BesNet(nn.Module):
        """Dropout, grouped Conv3d, batch norm, LeakyReLU, max-pool, 2 x 60 MLP, 4 logits."""

        def __init__(self, dropout: float):
            super().__init__()
            self.drop = nn.Dropout(dropout)
            self.conv = nn.Conv3d(2, 10, (3, 3, 5), padding=(1, 1, 2), groups=2)
            self.norm = nn.BatchNorm3d(10)
            self.act = nn.LeakyReLU()
            self.pool = nn.MaxPool3d((1, 2, 4))
            flat = 10 * ROWS * (COLS // 2) * (FREQS // 4)
            self.mlp = nn.Sequential(
                nn.Linear(flat, 60), nn.LeakyReLU(), nn.Linear(60, 60),
                nn.LeakyReLU(), nn.Linear(60, len(CLASSES)),
            )

        def forward(self, x):
            x = self.pool(self.act(self.norm(self.conv(self.drop(x)))))
            return self.mlp(x.flatten(1))

    torch.manual_seed(FOLD_SEED + fold)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    windows = pd.read_csv(args.work_dir / "windows.csv")
    power = np.load(args.work_dir / "power.npy")
    role = block_roles(windows) if args.protocol == "blocks" else fold_roles(windows, fold)[1]
    prefix = "blocks_" if args.protocol == "blocks" else ""
    sigma2 = power[role == 0].astype(np.float64).mean(axis=0)
    offset = torch.tensor(-np.log10(np.maximum(sigma2, 1e-12)), dtype=torch.float32)
    offset = offset.view(1, 1, CHANNELS, 1).to(device)
    feats = torch.from_numpy(np.load(args.work_dir / "features.npy")).to(device)
    labels = torch.tensor(windows.label.to_numpy(), device=device)

    def batch(idx):
        return (feats[idx].float() + offset).view(-1, 2, ROWS, COLS, FREQS)

    @torch.no_grad()
    def predict(model, idx):
        model.eval()
        return torch.cat(
            [torch.softmax(model(batch(idx[i : i + 4096])), 1) for i in range(0, len(idx), 4096)]
        )

    train, val, test = (torch.tensor(np.flatnonzero(role == r), device=device) for r in range(3))
    model = BesNet(args.dropout).to(device)
    groups = [
        {"params": [*model.conv.parameters(), *model.norm.parameters()], "lr": args.lr_conv},
        {"params": model.mlp.parameters(), "lr": args.lr_mlp},
    ]
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    best_f1, best_loss, best_state, since, history, running = -1.0, float("inf"), None, 0, [], []
    started = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        idx = train[torch.randint(len(train), (args.batch,), device=device)]
        loss = loss_fn(model(batch(idx)), labels[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        running.append(loss.detach())
        if step % args.eval_every:
            continue
        probs = predict(model, val)
        val_loss = float(loss_fn(torch.log(probs.clamp(min=1e-9)), labels[val]))
        conf = torch.bincount(
            labels[val] * len(CLASSES) + probs.argmax(1), minlength=len(CLASSES) ** 2
        ).view(len(CLASSES), len(CLASSES))
        f1 = macro_f1(conf.cpu())
        history.append({
            "step": step,
            "train_loss": float(torch.stack(running).mean()),
            "val_loss": val_loss,
            "val_macro_f1": f1,
        })
        running = []
        if f1 > best_f1:
            best_f1, best_state = f1, copy.deepcopy(model.state_dict())
        since = 0 if val_loss < best_loss else since + 1
        best_loss = min(best_loss, val_loss)
        if step % (5 * args.eval_every) == 0:
            print(f"fold {fold} step {step} val loss {val_loss:.4f} macro-F1 {f1:.4f} "
                  f"best {best_f1:.4f} ({time.time() - started:.0f} s)", flush=True)
        if since >= args.patience:
            break
    model.load_state_dict(best_state)
    probs = predict(model, test).cpu().numpy()
    out = windows.iloc[test.cpu().numpy()].copy()
    out["fold"] = fold
    for i, c in enumerate(CLASSES):
        out[f"p_{c}"] = probs[:, i]
    out.to_csv(args.work_dir / f"{prefix}fold{fold}_predictions.csv", index=False)
    torch.save(best_state, args.work_dir / f"{prefix}fold{fold}_model.pt")
    best_step = history[int(np.argmax([h["val_macro_f1"] for h in history]))]["step"]
    record = {
        "fold": fold,
        "train_windows": len(train),
        "val_windows": len(val),
        "test_windows": len(test),
        "steps_run": step,
        "best_step": best_step,
        "best_val_macro_f1": best_f1,
        "parameters": sum(p.numel() for p in model.parameters()),
        "seconds": round(time.time() - started, 1),
        "history": history,
    }
    (args.work_dir / f"{prefix}fold{fold}_training.json").write_text(json.dumps(record))
    print(f"fold {fold} done: best step {best_step}, val macro-F1 {best_f1:.4f}", flush=True)


def auroc(score: np.ndarray, truth: np.ndarray) -> float:
    """Rank AUROC with ties averaged; NaN if one class is missing."""
    pos = int(truth.sum())
    neg = truth.size - pos
    if pos == 0 or neg == 0:
        return float("nan")
    return float((rankdata(score)[truth].sum() - pos * (pos + 1) / 2) / (pos * neg))


def auprc(score: np.ndarray, truth: np.ndarray) -> float:
    """Average precision of `score`; NaN if the class is absent."""
    if not truth.any():
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    hit = truth[order].astype(float)
    ranked = score[order]
    last = np.r_[ranked[1:] != ranked[:-1], True]
    tp = np.cumsum(hit)[last]
    precision = tp / (np.arange(hit.size) + 1)[last]
    recall = tp / hit.sum()
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def prf(pred: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Confusion matrix (rows true) and per-class precision, recall, F1."""
    k = len(CLASSES)
    conf = np.bincount(truth * k + pred, minlength=k * k).reshape(k, k)
    tp = np.diag(conf).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = tp / conf.sum(0)
        recall = tp / conf.sum(1)
        f1 = 2 * tp / (conf.sum(0) + conf.sum(1))
    return conf, precision, recall, f1


def macro(values: np.ndarray) -> float:
    """Mean over the classes that exist (NaN where a class has no support)."""
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def score(prob: np.ndarray, truth: np.ndarray, shots: np.ndarray, boot: bool = True) -> dict:
    """Per-class and macro scores of a (n, 4) probability array, with shot-bootstrap CIs."""
    pred = prob.argmax(1)
    conf, precision, recall, f1 = prf(pred, truth)
    out = {
        "windows": int(truth.size),
        "shots": int(np.unique(shots).size),
        "confusion": conf.tolist(),
        "accuracy": float((pred == truth).mean()),
        "macro": {"precision": macro(precision), "recall": macro(recall), "f1": macro(f1)},
        "classes": {},
    }
    for i, c in enumerate(CLASSES):
        one = truth == i
        out["classes"][c] = {
            "support": int(one.sum()),
            "shots": int(np.unique(shots[one]).size),
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "auroc": auroc(prob[:, i], one),
            "auprc": auprc(prob[:, i], one),
        }
    out["macro"]["auroc"] = macro(np.array([out["classes"][c]["auroc"] for c in CLASSES]))
    out["macro"]["auprc"] = macro(np.array([out["classes"][c]["auprc"] for c in CLASSES]))
    if boot:
        out["ci95"] = bootstrap(prob, truth, shots)
    return out


def bootstrap(prob: np.ndarray, truth: np.ndarray, shots: np.ndarray) -> dict:
    """95 % intervals of macro precision/recall/F1/AUROC and per-class F1 over shots."""
    rng = np.random.default_rng(BOOT_SEED)
    order = np.argsort(shots, kind="stable")
    _, first = np.unique(shots[order], return_index=True)
    parts = np.split(order, first[1:])
    keys = [f"macro_{m}" for m in ("precision", "recall", "f1", "auroc")]
    keys += [f"f1_{c}" for c in CLASSES]
    draws: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(REPLICATES):
        pick = np.concatenate([parts[j] for j in rng.integers(len(parts), size=len(parts))])
        p, t = prob[pick], truth[pick]
        _, precision, recall, f1 = prf(p.argmax(1), t)
        row = {"macro_precision": macro(precision), "macro_recall": macro(recall), "macro_f1": macro(f1)}
        row["macro_auroc"] = macro(np.array([auroc(p[:, i], t == i) for i in range(len(CLASSES))]))
        row.update({f"f1_{c}": float(f1[i]) for i, c in enumerate(CLASSES)})
        for k in keys:
            draws[k].append(row[k])
    return {
        k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
        for k, v in draws.items()
    }


def bins_vs_baseline(oof: pd.DataFrame) -> dict | None:
    """BES windows averaged over 50 ms bins against the four-class diagnostic-summary model.

    The bins are the test bins of the saved run that hold a four-class prediction and
    whose regime agrees with the windows inside them.
    """
    pred = pd.read_csv(RUN / "predictions.csv")
    cols = [f"four_class_probability_{c}" for c in CLASSES]
    pred = pred.dropna(subset=[*cols, "regime_label"])
    bins = pred[["shot", "t_start", "t_end", "regime_label", *cols]].copy()
    oof = oof.assign(bin=np.floor(oof.center_ms / BIN_MS) * BIN_MS)
    mean = oof.groupby(["shot", "bin"])[[f"p_{c}" for c in CLASSES]].mean().reset_index()
    merged = bins.merge(mean, left_on=["shot", "t_start"], right_on=["shot", "bin"])
    if merged.empty:
        return None
    merged["regime_label"] = merged.regime_label.astype(int)
    label = oof.groupby(["shot", "bin"]).label.agg(lambda v: int(v.mode().iloc[0])).reset_index()
    merged = merged.merge(label, on=["shot", "bin"])
    agree = merged.regime_label == merged.label
    dropped = int((~agree).sum())
    merged = merged[agree]
    truth = merged.label.to_numpy()
    shots = merged.shot.to_numpy()
    bes = merged[[f"p_{c}" for c in CLASSES]].to_numpy()
    base = merged[cols].to_numpy()
    out = {
        "bins": len(merged),
        "shots": int(np.unique(shots).size),
        "dropped_label_mismatch": dropped,
        "bes": score(bes, truth, shots),
        "diagnostic_summary": score(base, truth, shots),
    }
    rng = np.random.default_rng(BOOT_SEED)
    ids = np.unique(shots)
    index = {s: np.flatnonzero(shots == s) for s in ids}
    diffs = []
    for _ in range(REPLICATES):
        pick = np.concatenate([index[s] for s in rng.choice(ids, size=len(ids))])
        f_bes = macro(prf(bes[pick].argmax(1), truth[pick])[3])
        f_base = macro(prf(base[pick].argmax(1), truth[pick])[3])
        diffs.append(f_bes - f_base)
    out["macro_f1_difference"] = {
        "bes_minus_summary": out["bes"]["macro"]["f1"] - out["diagnostic_summary"]["macro"]["f1"],
        "ci95": [float(np.nanpercentile(diffs, 2.5)), float(np.nanpercentile(diffs, 97.5))],
    }
    return out


def git_sha() -> str | None:
    """Short hash of HEAD, or None outside a checkout."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
            text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def evaluate(args: argparse.Namespace) -> None:
    """Score the out-of-fold predictions and write evaluation.json."""
    oof = pd.concat(
        [pd.read_csv(args.work_dir / f"fold{k}_predictions.csv") for k in range(FOLDS)],
        ignore_index=True,
    )
    windows = pd.read_csv(args.work_dir / "windows.csv")
    if len(oof) != len(windows):
        raise SystemExit(f"{len(oof)} predictions for {len(windows)} windows: run every fold")
    split = pd.read_csv(RUN / "split.csv").set_index("shot")["split"]
    oof["split"] = oof.shot.map(split)
    columns = [f"p_{c}" for c in CLASSES]

    def block(frame: pd.DataFrame) -> dict:
        return score(frame[columns].to_numpy(), frame.label.to_numpy(), frame.shot.to_numpy())

    folds = [
        {k: v for k, v in json.loads((args.work_dir / f"fold{i}_training.json").read_text()).items()
         if k != "history"}
        for i in range(FOLDS)
    ]
    record = {
        "git": git_sha(),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "protocol": {
            "folds": FOLDS, "fold_seed": FOLD_SEED, "window_samples": WINDOW, "stride_samples": STRIDE,
            "sampling_hz": FS, "band_hz": BAND, "channels": [CH0, CH1], "replicates": REPLICATES,
            "features": json.loads((args.work_dir / "features.json").read_text()),
        },
        "folds": folds,
        "results": {
            "all_shots": block(oof),
            "summary_model_test_shots": block(oof[oof.split == "test"]),
            "agreement_only": block(oof[oof.status == "agreement"]),
        },
        "bins_vs_summary_model": bins_vs_baseline(oof),
        "paper": {c: dict(zip(("f1", "precision", "recall"), v, strict=True)) for c, v in PAPER.items()},
    }
    leaky = args.work_dir / "blocks_fold0_predictions.csv"
    if leaky.exists():
        test = pd.read_csv(leaky)
        record["diagnostic_block_split"] = score(
            test[columns].to_numpy(), test.label.to_numpy(), test.shot.to_numpy(), boot=False
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "evaluation.json").write_text(json.dumps(record, indent=1))
    for name, res in record["results"].items():
        cls = res["classes"]
        print(f"{name}: {res['shots']} shots, {res['windows']} windows, macro F1 {res['macro']['f1']:.3f}, "
              + ", ".join(f"{c} {cls[c]['f1']:.3f}" for c in CLASSES))
    if "diagnostic_block_split" in record:
        d = record["diagnostic_block_split"]
        print(f"block split (leaky): {d['windows']} test windows, macro F1 {d['macro']['f1']:.3f}")
    if record["bins_vs_summary_model"]:
        b = record["bins_vs_summary_model"]
        print(f"bins: {b['bins']} on {b['shots']} shots, macro F1 BES {b['bes']['macro']['f1']:.3f}, "
              f"summary model {b['diagnostic_summary']['macro']['f1']:.3f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    sub = ap.add_subparsers(dest="stage", required=True)
    f = sub.add_parser("features", help="windows and log-spectral features from the corpus BES")
    f.add_argument("--workers", type=int, default=16)
    t = sub.add_parser("train", help="train the cross-validation folds (GPU)")
    t.add_argument("--folds", type=int, nargs="+", default=list(range(FOLDS)))
    t.add_argument("--protocol", choices=("shots", "blocks"), default="shots",
                   help="blocks: the leaky diagnostic split of 0.2 s blocks, one run")
    t.add_argument("--device", default="cuda:0")
    t.add_argument("--steps", type=int, default=60000)
    t.add_argument("--eval-every", type=int, default=500)
    t.add_argument("--patience", type=int, default=30)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--dropout", type=float, default=0.2)
    t.add_argument("--lr-conv", type=float, default=1e-3)
    t.add_argument("--lr-mlp", type=float, default=1e-4)
    t.add_argument("--weight-decay", type=float, default=0.01)
    sub.add_parser("evaluate", help="score the out-of-fold predictions")
    args = ap.parse_args(argv)
    if args.stage == "features":
        build_features(args)
    elif args.stage == "train":
        for fold in args.folds:
            run_fold(args, fold)
    else:
        evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
