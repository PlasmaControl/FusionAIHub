#!/usr/bin/env python
"""detach-ours: can the detachment label be read from cheap 0-D signals alone?

    python scripts/labeler/detach_ours.py prep            # environment labelmaker
    CUDA_VISIBLE_DEVICES=1 $LABELER_ROOT/envs/phase3/bin/python \\
        scripts/labeler/detach_ours.py train               # torch, one GPU

A small 1-D CNN reads a 2 s window (41 bins of 50 ms) of plasma current, heating
power, line density, divertor D-alpha, the ELM share and the EFIT scalars, and
predicts the label at the centre bin: attached, detached or marfe. None of the three
indicators' own inputs is among them (no Langmuir probes, no bolometer, no camera),
though the heating power and the line density also normalise Afrac and Prad,div, so
the model is not wholly independent of them. It is trained on the bins where the
label is a certain state.

Validation is by shot: 5-fold CV over the shots that are not in the cohort's test
split, then a model trained on all of them scored on the test shots; both with
1000-replicate shot-bootstrap 95% intervals, against the majority-class predictor.
Nothing is tuned on the test shots (the epoch count and architecture are fixed).

Writes `docs/labeler/results/detachment_ours.json`; the dataset and the predictions
are under `$LABELER_ROOT/round4/detach/ours/`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from labeler.events.detachment import core, signals

REPO = Path(__file__).resolve().parents[2]
RESULT = REPO / "docs" / "labeler" / "results" / "detachment_ours.json"
HALF = 20  # bins either side of the centre: 41 bins, 2.05 s
EPOCHS = 25
FOLDS = 5
BOOTSTRAPS = 1000
SEED = 0
CHANNELS = (
    "ip_ma",
    "p_in_mw",
    "ne_rel",
    "dalpha_rel",
    "elm_share",
    "betan",
    "wmhd_mj",
    "q95",
    "kappa",
    "bcentr",
    "rxpt1",
    "zxpt1",
    "rvsod",
    "zvsod",
)
EFIT_NODES = ("betan", "wmhd", "q95", "kappa", "bcentr")


def root() -> Path:
    return Path(os.environ["LABELER_ROOT"]) / "round4" / "detach"


def shot_channels(shot: int) -> np.ndarray | None:
    """`(n_bins, len(CHANNELS))` on the shot's bin grid, NaN where unknown."""
    path = root() / "bins" / f"{shot}.npz"
    if not path.is_file():
        return None
    with np.load(path) as npz:
        bins = {k: npz[k] for k in npz.files}
    starts = bins["start_ms"]
    edges = np.r_[starts, starts[-1] + core.BIN_MS]
    n = len(starts)
    nan = np.full(n, np.nan)
    cache = signals.load_cache(shot)

    def efit(name):
        if name not in cache:
            return nan
        return core.bin_median(*cache[name], edges)[0]

    ne = bins.get("aux_ne", nan).astype(float)
    with np.errstate(all="ignore"):
        ne_rel = ne / np.nanmedian(ne) if np.isfinite(ne).any() else nan
    dalpha = nan
    got = signals.corpus_group(shot, "filterscopes", channels=range(8))
    if got is not None:
        t, y = got
        live = [r for r in y if np.isfinite(r).mean() > 0.9 and np.nanmedian(r) > 0]
        if live:
            x = np.nanmedian([np.nan_to_num(r) / np.nanmedian(r) for r in live], axis=0)
            dalpha = core.bin_median(t, x, edges)[0]
            with np.errstate(all="ignore"):
                dalpha = dalpha / np.nanmedian(dalpha)
    columns = {
        "ip_ma": bins["aux_ip_a"] / 1e6,
        "p_in_mw": bins.get("aux_p_in_w", nan) / 1e6,
        "ne_rel": ne_rel,
        "dalpha_rel": dalpha,
        "elm_share": bins.get("aux_elm_share", nan),
        "betan": efit("betan"),
        "wmhd_mj": efit("wmhd") / 1e6,
        "q95": efit("q95"),
        "kappa": efit("kappa"),
        "bcentr": efit("bcentr"),
        "rxpt1": bins.get("aux_rxpt1", nan),
        "zxpt1": bins.get("aux_zxpt1", nan),
        "rvsod": bins.get("aux_rvsod", nan),
        "zvsod": bins.get("aux_zvsod", nan),
    }
    return np.stack([np.asarray(columns[c], dtype=np.float32) for c in CHANNELS], 1)


def prep() -> None:
    labels = pd.read_csv(root() / "labels_bins.csv.gz")
    certain = labels[labels.state_lm.isin((1, 2, 3))]
    windows, target, shots, starts, split = [], [], [], [], []
    for shot, rows in certain.groupby("shot"):
        x = shot_channels(int(shot))
        if x is None:
            continue
        with np.load(root() / "bins" / f"{int(shot)}.npz") as npz:
            grid = npz["start_ms"]
        index = {float(s): i for i, s in enumerate(grid)}
        padded = np.pad(
            x, ((HALF, HALF), (0, 0)), mode="constant", constant_values=np.nan
        )
        for row in rows.itertuples():
            i = index.get(float(row.start_ms))
            if i is None:
                continue
            windows.append(padded[i : i + 2 * HALF + 1])
            target.append(int(row.state_lm) - 1)
            shots.append(int(shot))
            starts.append(float(row.start_ms))
            split.append(row.split)
        print("prep", shot, len(rows), flush=True)
    out = root() / "ours"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "dataset.npz",
        x=np.stack(windows).astype(np.float32),
        y=np.asarray(target, dtype=np.int64),
        shot=np.asarray(shots),
        start_ms=np.asarray(starts),
        split=np.asarray(split),
        channels=np.asarray(CHANNELS),
    )
    print("dataset", len(target), "windows", len(set(shots)), "shots")


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    labels = (0, 1, 2)
    f1 = {}
    for k in labels:
        tp = np.sum((y == k) & (pred == k))
        fp = np.sum((y != k) & (pred == k))
        fn = np.sum((y == k) & (pred != k))
        f1[k] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    po = float(np.mean(y == pred))
    pe = sum(np.mean(y == k) * np.mean(pred == k) for k in labels)
    present = [k for k in labels if np.any(y == k)]
    return {
        "accuracy": po,
        "kappa": float((po - pe) / (1 - pe)) if pe < 1 else 1.0,
        "macro_f1": float(np.nanmean([f1[k] for k in present])),
        "f1_attached": f1[0],
        "f1_detached": f1[1],
        "f1_marfe": f1[2],
    }


def with_ci(y, pred, shot, rng) -> dict:
    point = metrics(y, pred)
    by_shot = {s: np.flatnonzero(shot == s) for s in np.unique(shot)}
    keys = list(by_shot)
    draws = {k: [] for k in point}
    for _ in range(BOOTSTRAPS):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([by_shot[keys[j]] for j in pick])
        for k, v in metrics(y[idx], pred[idx]).items():
            draws[k].append(v)
    return {
        k: {
            "value": point[k],
            "ci95": [float(v) for v in np.nanpercentile(draws[k], [2.5, 97.5])],
        }
        for k in point
    }


def train() -> None:
    import torch
    from torch import nn

    data = np.load(root() / "ours" / "dataset.npz")
    x, y, shot, split = data["x"], data["y"], data["shot"], data["split"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(SEED)

    def standardise(train_x, *others):
        mean = np.nanmean(train_x.reshape(-1, train_x.shape[-1]), axis=0)
        std = np.nanstd(train_x.reshape(-1, train_x.shape[-1]), axis=0) + 1e-6
        out = []
        for a in (train_x, *others):
            z = (a - mean) / std
            missing = ~np.isfinite(z)
            out.append(
                np.concatenate(
                    [np.where(missing, 0.0, z), missing.astype(np.float32)], axis=2
                )
            )
        return out

    def net(channels: int) -> nn.Module:
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.body = nn.Sequential(
                    nn.Conv1d(2 * channels, 32, 5, padding=2),
                    nn.ReLU(),
                    nn.Conv1d(32, 32, 5, padding=4, dilation=2),
                    nn.ReLU(),
                    nn.Conv1d(32, 32, 5, padding=8, dilation=4),
                    nn.ReLU(),
                )
                self.head = nn.Linear(64, 3)

            def forward(self, a):
                h = self.body(a.transpose(1, 2))
                return self.head(torch.cat([h[:, :, HALF], h.mean(dim=2)], dim=1))

        return Net()

    def fit_predict(train_idx, test_idx, seed):
        torch.manual_seed(seed)
        xt, xe = standardise(x[train_idx], x[test_idx])
        yt = y[train_idx]
        weight = np.bincount(yt, minlength=3).astype(float)
        weight = np.where(weight > 0, weight.sum() / (3 * np.maximum(weight, 1)), 0)
        model = net(len(CHANNELS)).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weight, dtype=torch.float32))
        loss_fn = loss_fn.to(device)
        tx = torch.tensor(xt, device=device)
        ty = torch.tensor(yt, device=device)
        for _ in range(EPOCHS):
            order = torch.randperm(len(tx), device=device)
            for s in range(0, len(order), 256):
                b = order[s : s + 256]
                opt.zero_grad()
                loss_fn(model(tx[b]), ty[b]).backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            return model(torch.tensor(xe, device=device)).softmax(1).cpu().numpy()

    fit_shots = np.unique(shot[split != "test"])
    rng.shuffle(fit_shots)
    folds = np.array_split(fit_shots, FOLDS)
    prob = np.full((len(y), 3), np.nan)
    for k, held in enumerate(folds):
        test_mask = np.isin(shot, held)
        train_mask = (split != "test") & ~test_mask
        prob[test_mask] = fit_predict(
            np.flatnonzero(train_mask), np.flatnonzero(test_mask), SEED + k
        )
        print("fold", k, int(test_mask.sum()), flush=True)
    final = np.flatnonzero(split == "test")
    if len(final):
        prob[final] = fit_predict(np.flatnonzero(split != "test"), final, SEED + FOLDS)
    pred = prob.argmax(axis=1)
    cv = np.flatnonzero(split != "test")
    boot = np.random.default_rng(1)
    result = {
        "inputs": list(CHANNELS),
        "window_bins": 2 * HALF + 1,
        "bin_ms": core.BIN_MS,
        "epochs": EPOCHS,
        "folds": FOLDS,
        "n_windows": len(y),
        "n_shots": len(np.unique(shot)),
        "class_counts": {
            core.STATE_NAMES[k + 1]: int(np.sum(y == k)) for k in range(3)
        },
        "cv_shots": with_ci(y[cv], pred[cv], shot[cv], boot),
        "cv_majority": metrics(y[cv], np.full(len(cv), np.bincount(y[cv]).argmax())),
    }
    if len(final):
        result["test_shots"] = with_ci(y[final], pred[final], shot[final], boot)
        result["test_majority"] = metrics(
            y[final], np.full(len(final), np.bincount(y[cv]).argmax())
        )
        result["n_test_shots"] = len(np.unique(shot[final]))
    out = root() / "ours"
    np.savez_compressed(
        out / "predictions.npz", prob=prob, y=y, shot=shot, start_ms=data["start_ms"]
    )
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=1))
    print(json.dumps(result["cv_shots"], indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=("prep", "train"))
    args = parser.parse_args()
    {"prep": prep, "train": train}[args.step]()


if __name__ == "__main__":
    main()
