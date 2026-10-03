#!/usr/bin/env python
"""detach-victor: can the detachment label be read from a single camera frame?

    python scripts/labeler/detach_victor.py prep           # environment labelmaker
    CUDA_VISIBLE_DEVICES=1 $LABELER_ROOT/envs/phase3/bin/python \\
        scripts/labeler/detach_victor.py train              # torch, one GPU

The model of Victor and Scotti (2024), a small CNN on raw divertor camera frames
(two 3 x 3 convolutions, 2 x 2 max pool, dropout 0.25, flatten, a hidden linear layer,
a softmax; the digest in `.tmp/label_papers/outside/` gives no widths, these are
chosen here), here with three outputs and trained on the detachment label instead of
their hand-labelled binary one: it reads one TangTV frame (corpus `tangtv` channel 2,
the lower-divertor view, nearest to the bin centre, black level removed, square-rooted,
8 x 8 block means to 30 x 90) and predicts attached, detached or marfe. The target is
the combined label where it is a certain state.

Two caveats are scored, not hidden. (1) The TangTV indicator is one of the label's
three voters and is itself read from these frames, so the label and the input share
information; the score is therefore also reported on the bins where TangTV did not
vote (the label there is Afrac and Prad,div alone), which is the fair test of whether
the frames know about the state. (2) The camera sees the divertor only on the
geometry it was built for; on other geometries the model is asked a question the
frames may not answer.

Validation is by shot: 5-fold CV over the shots outside the cohort's test split, then
a model trained on all of them scored once on the test shots; 1000-replicate shot
bootstrap 95% intervals, against the majority-class predictor. Nothing is tuned on
the test shots (epochs and architecture are fixed).

Writes `docs/labeler/results/detachment_victor.json`; the frames and the predictions
are under `$LABELER_ROOT/round4/detach/victor/`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

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
    certain = labels[labels.state_lm.isin((1, 2, 3))]
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
        target += (kept.state_lm.to_numpy(int) - 1).tolist()
        shots += [int(shot)] * len(kept)
        starts += kept.start_ms.tolist()
        split += kept.split.tolist()
        voted += (kept.tangtv_vote.to_numpy(int) > 0).tolist()
        print("prep", shot, int(keep.sum()), flush=True)
    out = root() / "victor"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "dataset.npz",
        x=np.concatenate(frames),
        y=np.asarray(target, dtype=np.int64),
        shot=np.asarray(shots),
        start_ms=np.asarray(starts),
        split=np.asarray(split),
        tangtv_voted=np.asarray(voted),
    )
    print("dataset", len(target), "frames", len(set(shots)), "shots")


def train() -> None:
    import torch
    from detach_ours import metrics, with_ci
    from torch import nn

    data = np.load(root() / "victor" / "dataset.npz")
    x, y, shot, split = data["x"], data["y"], data["shot"], data["split"]
    voted = data["tangtv_voted"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scale = float(np.percentile(x[::7].astype(np.float32), 99.5))

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
        torch.manual_seed(seed)
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
    prob = np.full((len(y), 3), np.nan)
    for k, held in enumerate(np.array_split(fit_shots, FOLDS)):
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
    major = np.bincount(y[cv]).argmax()
    result = {
        "input": f"TangTV channel 2 frame, {240 // BLOCK} x {720 // BLOCK}",
        "epochs": EPOCHS,
        "folds": FOLDS,
        "n_frames": len(y),
        "n_shots": len(np.unique(shot)),
        "n_tangtv_voted": int(voted.sum()),
        "cv_shots": with_ci(y[cv], pred[cv], shot[cv], boot),
        "cv_majority": metrics(y[cv], np.full(len(cv), major)),
    }
    free = cv[~voted[cv]]
    if len(free) > 50 and len(np.unique(y[free])) > 1:
        result["cv_no_tangtv_vote"] = {
            "n_frames": len(free),
            "n_shots": len(np.unique(shot[free])),
            **with_ci(y[free], pred[free], shot[free], boot),
            "majority": metrics(y[free], np.full(len(free), major)),
        }
    if len(final):
        result["test_shots"] = with_ci(y[final], pred[final], shot[final], boot)
        result["test_majority"] = metrics(y[final], np.full(len(final), major))
        result["n_test_shots"] = len(np.unique(shot[final]))
    out = root() / "victor"
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
