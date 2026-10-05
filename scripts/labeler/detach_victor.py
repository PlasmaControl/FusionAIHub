#!/usr/bin/env python
"""detach-victor: can the detachment label be read from a single camera frame?

    python scripts/labeler/detach_victor.py prep           # environment labelmaker
    CUDA_VISIBLE_DEVICES=1 $LABELER_ROOT/envs/phase3/bin/python \\
        scripts/labeler/detach_victor.py train              # torch, one GPU
    python scripts/labeler/detach_victor.py refresh         # unchanged arrays only

The model of Victor and Scotti (2024), a small CNN on raw divertor camera frames
(two 3 x 3 convolutions, 2 x 2 max pool, dropout 0.25, flatten, a hidden linear layer,
a softmax; the digest in `.tmp/label_papers/outside/` gives no widths, these are
chosen here), here with three outputs and trained on the detachment label instead of
their hand-labelled binary one: it reads one TangTV frame (corpus `tangtv` channel 2,
the lower-divertor view, nearest to the bin centre, black level removed, square-rooted,
8 x 8 block means to 30 x 90) and predicts attached, detached or marfe. The target is
the combined label where it is a certain state.

The TangTV indicator is required by the certain consensus and is read from the
same diagnostic as these frames: the combined-label score shares information with
the input. A separate diagnostic agreement score uses bins where the other two
indicators agree (the Afrac/Prad proxy pair), with TangTV withheld. This selected
reference is not independent physical truth. No certain consensus bins without a
TangTV vote are implied. The camera is also limited to its divertor geometry.

Validation is by shot: 5-fold CV over the shots outside the cohort's test split, then
a model trained on all of them scored once on the test shots; 1000-replicate shot
bootstrap 95% intervals, against the majority-class predictor. Nothing is tuned on
the test shots (epochs and architecture are fixed).

Writes `docs/labeler/results/detachment_victor.json`; the frames and the predictions
are under `$LABELER_ROOT/round4/detach/victor/`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from detach_json import dumps

REPO = Path(__file__).resolve().parents[2]
RESULT = REPO / "docs" / "labeler" / "results" / "detachment_victor.json"
BLACK = 16.0
BLOCK = 8
MAX_GAP_MS = 30.0
EPOCHS = 12
FOLDS = 5
BOOTSTRAPS = 1000
SEED = 0


def root() -> Path:
    return Path(os.environ["LABELER_ROOT"]) / "round4" / "detach"


def prep() -> None:
    import detach_tv_surrogate as tvs

    labels = pd.read_csv(root() / "labels_bins.csv.gz")
    certain = labels
    frames, target, shots, starts, split, voted = [], [], [], [], [], []
    for shot, rows in certain.groupby("shot"):
        got = tvs.corpus_frames(int(shot))
        if got is None:
            continue
        t, raw = got
        centres = rows.start_ms.to_numpy(float) + 25.0
        near = tvs.nearest(t, centres)
        keep = np.abs(t[near] - centres) <= MAX_GAP_MS
        if not keep.any():
            continue
        x = np.maximum(raw[near[keep]] - BLACK, 0.0)
        n, h, w = x.shape
        x = x.reshape(n, h // BLOCK, BLOCK, w // BLOCK, BLOCK).mean(axis=(2, 4))
        frames.append(np.sqrt(x).astype(np.float16))
        kept = rows[keep]
        target += np.where(
            kept.state_lm.isin((1, 2, 3)), kept.state_lm.to_numpy(int) - 1, -1
        ).tolist()
        shots += [int(shot)] * len(kept)
        starts += kept.start_ms.tolist()
        split += kept.split.tolist()
        voted += (kept.tangtv_vote.to_numpy(int) > 0).tolist()
        print("prep", shot, int(keep.sum()), flush=True)
    out = root() / "victor"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "dataset.npz",
        x=np.concatenate(frames)
        if frames
        else np.empty((0, 240 // BLOCK, 720 // BLOCK), dtype=np.float16),
        y=np.asarray(target, dtype=np.int64),
        shot=np.asarray(shots),
        start_ms=np.asarray(starts),
        split=np.asarray(split),
        tangtv_voted=np.asarray(voted),
    )
    print("dataset", len(target), "frames", len(set(shots)), "shots")


def train() -> None:
    import torch
    from detach_ours import (
        baseline_strata,
        dataset_fingerprint,
        fold_record,
        label_source,
        marfe_transfer,
        metrics,
        with_ci,
    )
    from torch import nn

    data = np.load(root() / "victor" / "dataset.npz")
    x, y, shot, split = data["x"], data["y"], data["shot"], data["split"]
    source = label_source(data)
    voted = data["tangtv_voted"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            h, w = 240 // BLOCK - 4, 720 // BLOCK - 4  # two valid 3 x 3 convolutions
            self.body = nn.Sequential(
                nn.Conv2d(1, 16, 3),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.25),
                nn.Flatten(),
                nn.Linear(32 * (h // 2) * (w // 2), 64),
                nn.ReLU(),
            )
            self.head = nn.Linear(64, 3)

        def forward(self, a):
            return self.head(self.body(a))

    def fit_predict(train_idx, test_idx, seed):
        if not len(train_idx):
            return np.full((len(test_idx), 3), np.nan)
        torch.manual_seed(seed)
        scale = max(
            float(np.percentile(x[train_idx][::7].astype(np.float32), 99.5)), 1e-6
        )
        yt = y[train_idx]
        weight = np.bincount(yt, minlength=3).astype(float)
        weight = np.where(weight > 0, weight.sum() / (3 * np.maximum(weight, 1)), 0)
        model = Net().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weight, dtype=torch.float32))
        loss_fn = loss_fn.to(device)
        tx = torch.tensor(x[train_idx], device=device)
        ty = torch.tensor(yt, device=device)
        for _ in range(EPOCHS):
            order = torch.randperm(len(tx), device=device)
            for s in range(0, len(order), 128):
                b = order[s : s + 128]
                a = (tx[b].float() / scale).unsqueeze(1)
                opt.zero_grad()
                loss_fn(model(a), ty[b]).backward()
                opt.step()
        model.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(test_idx), 512):
                a = torch.tensor(x[test_idx[s : s + 512]], device=device)
                out.append(model(a.float().unsqueeze(1) / scale).softmax(1).cpu())
        return torch.cat(out).numpy()

    fit_shots = np.unique(shot[split != "test"])
    np.random.default_rng(SEED).shuffle(fit_shots)
    majority_pred = np.full(len(y), -1, int)
    prob = np.full((len(y), 3), np.nan)
    fold_records = []
    for k, held in enumerate(np.array_split(fit_shots, FOLDS)):
        if not len(held):
            continue
        test_mask = np.isin(shot, held)
        train_mask = (split != "test") & (y >= 0) & ~test_mask
        prob[test_mask] = fit_predict(
            np.flatnonzero(train_mask), np.flatnonzero(test_mask), SEED + k
        )
        if train_mask.any():
            majority_pred[test_mask] = np.bincount(y[train_mask]).argmax()
        fold_records.append(
            fold_record(k, held, train_mask, test_mask, y, shot, prob, majority_pred)
        )
        print("fold", k, int(test_mask.sum()), flush=True)
    final = np.flatnonzero(split == "test")
    if len(final):
        prob[final] = fit_predict(
            np.flatnonzero((split != "test") & (y >= 0)), final, SEED + FOLDS
        )
    scored = np.isfinite(prob).all(axis=1)
    pred = np.full(len(y), -1, int)
    pred[scored] = prob[scored].argmax(axis=1)
    cv = np.flatnonzero((split != "test") & (y >= 0) & scored)
    boot = np.random.default_rng(1)
    fit_labels = y[(split != "test") & (y >= 0)]
    major = np.bincount(fit_labels).argmax() if len(fit_labels) else -1
    majority_pred[split == "test"] = major
    result = {
        "input": f"TangTV channel 2 frame, {240 // BLOCK} x {720 // BLOCK}",
        "epochs": EPOCHS,
        "folds": FOLDS,
        "fold_records": fold_records,
        "n_frames": len(y),
        "n_shots": len(np.unique(shot)),
        "n_tangtv_voted": int(voted.sum()),
        "fit_excluded_split": "test",
        "unscored_frames": int((~scored).sum()),
        "cv_shots": with_ci(y[cv], pred[cv], shot[cv], boot),
        "cv_majority": metrics(y[cv], majority_pred[cv]),
        "cv_majority_ci": with_ci(
            y[cv], majority_pred[cv], shot[cv], np.random.default_rng(1)
        ),
        "marfe_transfer": marfe_transfer(y, shot, split, fold_records),
        "dataset_fingerprint": dataset_fingerprint(data),
        "label_source": source,
        "bootstrap_comparison": "model and majority use identical shot "
        "resamples with seed 1 within each population",
        "evaluation_scope": "exploratory agreement with constructed labels; "
        "no independent physical benchmark or established learning beyond majority",
    }
    result["n_cv_frames"] = len(cv)
    result["n_cv_shots"] = len(np.unique(shot[cv]))
    result["normalization"] = (
        "99.5th percentile fitted on training frames within each fold"
    )
    result["stratified"] = baseline_strata(data, pred, boot)
    final = final[(y[final] >= 0) & scored[final]]
    free = cv[~voted[cv]]
    if len(free) > 50 and len(np.unique(y[free])) > 1:
        result["cv_no_tangtv_vote"] = {
            "n_frames": len(free),
            "n_shots": len(np.unique(shot[free])),
            **with_ci(y[free], pred[free], shot[free], boot),
            "majority": metrics(y[free], np.full(len(free), major)),
        }
    else:
        result["cv_no_tangtv_vote"] = {
            "n_frames": len(free),
            "n_shots": len(np.unique(shot[free])),
            "status": "unavailable",
            "reason": "Certain consensus requires a TangTV vote; no independent "
            "no-camera consensus accuracy reference is supplied.",
        }
    if len(final):
        result["test_shots"] = with_ci(
            y[final], pred[final], shot[final], np.random.default_rng(1)
        )
        result["test_majority"] = metrics(y[final], np.full(len(final), major))
        result["test_majority_ci"] = with_ci(
            y[final], majority_pred[final], shot[final], np.random.default_rng(1)
        )
        result["n_test_shots"] = len(np.unique(shot[final]))
    out = root() / "victor"
    np.savez_compressed(
        out / "predictions.npz",
        prob=prob,
        y=y,
        shot=shot,
        start_ms=data["start_ms"],
        split=split,
        pred=pred,
        majority_pred=majority_pred,
    )
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(dumps(result, indent=1))
    print(dumps(result["cv_shots"], indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=("prep", "train", "refresh"))
    args = parser.parse_args()
    if args.step == "refresh":
        from detach_ours import refresh

        refresh("victor")
    else:
        {"prep": prep, "train": train}[args.step]()


if __name__ == "__main__":
    main()
