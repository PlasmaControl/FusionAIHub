#!/usr/bin/env python
"""`tm-ours`: a small per-bin detector on magnetics spectrogram features; its baseline.

Input: `tm_magfeatures.py`'s features of the cohort's six Mirnov probes (10 ms bins).
Target: the cohort's whole-interval tearing-mode labels (`extend_tm_interval`), bin by
bin; bins the label calls uncertain or not observable are left out of the loss and the
scores. The net is three dilated 1-D convolutions over the shot's bins (a field of about
0.3 s), a 1x1 head, three seeds averaged.

Protocol (frozen before any score was looked at):

* the 450 training and validation shots of the cohort are dealt into five shot-grouped
  folds (`scoring.shot_folds`, seed 0); each fold is scored by nets trained on the other
  four, a tenth of whose shots stop the training early and choose the F1 threshold;
* the 50 test shots are scored once, by nets trained on all 450 (``--final``), at the
  threshold their own validation shots chose;
* ``--features magnetics`` is the detector; ``--features magnetics+rms`` adds the lab's
  n = 1 and n = 2 RMS (the traces the labels are drawn from) as the ablation that says
  how much of the label is read back from them;
* ``--baseline`` is no model: the smoothed n = 1 RMS of each bin against a threshold
  (12 G, the onset level of the rule; the AUROC and AUPRC sweep every threshold).

Every number is written with its shots, bins and thresholds to
``$LABELER_ROOT/round4/tm/results/<name>.json``.

Run in the CUDA venv (the library imports need only numpy, pandas and torch)::

    PYTHONPATH=$PWD/src CUDA_VISIBLE_DEVICES=1 \\
        /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/envs/phase3/bin/python \\
        scripts/labeler/tm_ours.py --features magnetics
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from labeler.tearing import magfeatures, scoring

ROOT = Path(os.environ.get("LABELER_ROOT", "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker"))
TM = ROOT / "round4/tm"
CATALOG = REPO / "data/events/catalog"
LABELS = (
    REPO / "data/events/neoclassical_tearing_mode/extend_tm_interval/tm_interval.csv"
)
HYPER = {
    "hidden": 32,
    "kernel": 5,
    "dilations": [1, 2, 4],
    "dropout": 0.1,
    "lr": 2e-3,
    "weight_decay": 1e-4,
    "batch_shots": 8,
    "max_epochs": 80,
    "patience": 10,
    "seeds": [0, 1, 2],
    "val_fraction": 0.1,
}
FOLDS = 5
THRESHOLD_EDGES = np.linspace(0.0, 1.0, 1001)


def git_sha() -> str:
    try:
        from labeler.config import git_sha as sha

        return sha()
    except Exception:  # noqa: BLE001
        return "unknown"


def load(shots, with_rms: bool):
    """`({shot: (centres, X, y, valid)}, skipped)` for the shots with features."""
    table = pd.read_csv(LABELS)
    by_shot = {s: g for s, g in table.groupby("shot")}
    data, skipped = {}, []
    for shot in shots:
        path = TM / "magfeatures" / f"{shot}.npz"
        if not path.is_file():
            skipped.append(int(shot))
            continue
        with np.load(path) as z:
            centres, X = z["centres_ms"], z["features"].astype(np.float64)
        if with_rms:
            with np.load(TM / "signals" / f"{shot}.npz") as sig:
                extra = magfeatures.rms_bin_features(
                    sig["t_ms"], sig["n1rms"], sig["n2rms"], centres
                )
            X = np.hstack([X, extra])
        y, valid = scoring.label_bins(by_shot.get(shot, table.iloc[:0]), centres)
        valid = valid & np.isfinite(X).all(axis=1)
        data[int(shot)] = (centres, X, y, valid)
    return data, skipped


def model_class():
    from torch import nn

    class TemporalNet(nn.Module):
        def __init__(self, n_in, hidden, kernel, dilations, dropout):
            super().__init__()
            layers, width = [], n_in
            for d in dilations:
                layers += [
                    nn.Conv1d(
                        width, hidden, kernel, dilation=d, padding=d * (kernel // 2)
                    ),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
                width = hidden
            self.body = nn.Sequential(*layers)
            self.head = nn.Conv1d(hidden, 1, 1)

        def forward(self, x):
            return self.head(self.body(x)).squeeze(1)

    return TemporalNet


def standardizer(data, shots):
    rows = np.concatenate([data[s][1][data[s][3]] for s in shots])
    mean, std = rows.mean(axis=0), rows.std(axis=0)
    return mean, np.where(std > 1e-6, std, 1.0)


def batch(data, shots, mean, std, device):
    import torch

    length = max(len(data[s][0]) for s in shots)
    n_feat = data[shots[0]][1].shape[1]
    x = np.zeros((len(shots), n_feat, length), dtype=np.float32)
    y = np.zeros((len(shots), length), dtype=np.float32)
    m = np.zeros((len(shots), length), dtype=np.float32)
    for i, s in enumerate(shots):
        _, X, lab, ok = data[s]
        z = np.where(ok[:, None], (X - mean) / std, 0.0)
        x[i, :, : len(lab)] = z.T
        y[i, : len(lab)] = lab
        m[i, : len(lab)] = ok
    return tuple(torch.as_tensor(a, device=device) for a in (x, y, m))


def train_one(data, train, val, seed, device):
    """A net trained on `train` shots, early-stopped on `val`; `(net, mean, std)`."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mean, std = standardizer(data, train)
    n_in = data[train[0]][1].shape[1]
    net = model_class()(
        n_in, HYPER["hidden"], HYPER["kernel"], HYPER["dilations"], HYPER["dropout"]
    ).to(device)
    opt = torch.optim.AdamW(
        net.parameters(), lr=HYPER["lr"], weight_decay=HYPER["weight_decay"]
    )
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    vx, vy, vm = batch(data, val, mean, std, device)
    best, best_state, stale = np.inf, None, 0
    for _ in range(HYPER["max_epochs"]):
        net.train()
        order = rng.permutation(len(train))
        for i in range(0, len(order), HYPER["batch_shots"]):
            ids = [train[j] for j in order[i : i + HYPER["batch_shots"]]]
            x, y, m = batch(data, ids, mean, std, device)
            loss = (loss_fn(net(x), y) * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val_loss = (
                (loss_fn(net(vx), vy) * vm).sum() / vm.sum().clamp(min=1)
            ).item()
        if val_loss < best - 1e-5:
            best, stale = val_loss, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
            if stale >= HYPER["patience"]:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, mean, std, best


def predict(nets, data, shots, device):
    """`{shot: probability per bin}` averaged over the nets of one ensemble."""
    import torch

    out = {}
    for s in shots:
        probs = []
        for net, mean, std, _ in nets:
            x, _, _ = batch(data, [s], mean, std, device)
            with torch.no_grad():
                probs.append(torch.sigmoid(net(x))[0].cpu().numpy())
        out[s] = np.mean(probs, axis=0)
    return out


def ensemble(data, train, val, device):
    return [train_one(data, train, val, seed, device) for seed in HYPER["seeds"]]


def pick_threshold(data, val_scores, val):
    stats = [
        scoring.shot_stats(
            s, data[s][2], data[s][3], val_scores[s], THRESHOLD_EDGES, None
        )
        for s in val
    ]
    return scoring.best_threshold(stats, THRESHOLD_EDGES)


def truth_of(data, shots):
    return (
        {s: data[s][2] for s in shots},
        {s: data[s][3] for s in shots},
    )


def run_baseline(data, shots, rms_col=0):
    """The smoothed n = 1 RMS of each bin as the score, 12 G as the threshold."""
    return {s: data[s][1][:, rms_col] for s in shots}, float(np.log10(12.0))


def write(name, record):
    out = TM / "results"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.json"
    path.write_text(
        json.dumps(record, indent=1, default=float) + "\n", encoding="utf-8"
    )
    print(path)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--features", choices=("magnetics", "magnetics+rms"), default=None)
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--final", action="store_true", help="also score the 50 test shots")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args(argv)
    if not args.baseline and args.features is None:
        raise SystemExit("name --features or --baseline")

    cohort = pd.read_csv(CATALOG / "cohort.csv")
    dev = sorted(int(s) for s in cohort[cohort.split != "test"].shot)
    test = sorted(int(s) for s in cohort[cohort.split == "test"].shot)
    with_rms = args.baseline or args.features == "magnetics+rms"
    data, skipped = load(dev + test, with_rms)
    dev = [s for s in dev if s in data]
    test = [s for s in test if s in data]
    meta = {
        "git_sha": git_sha(),
        "labels": str(LABELS.relative_to(REPO)),
        "bin_ms": scoring.BIN_MS,
        "shots_without_features": skipped,
        "dev_shots": dev,
        "n_dev": len(dev),
    }

    if args.baseline:
        # the baseline needs no features of the magnetics: its n1 column is the RMS
        score, onset = run_baseline(data, dev, rms_col=-2)
        pooled = np.concatenate([score[s][data[s][3]] for s in dev])
        edges = scoring.edges_for(pooled)
        y, valid = truth_of(data, dev)
        stats = [
            scoring.shot_stats(s, y[s], valid[s], score[s], edges, None) for s in dev
        ]
        tuned = scoring.best_threshold(stats, edges)
        variants = {
            "12g": (onset, "the rule's onset level, 12 G"),
            "tuned": (tuned, "the single threshold maximising F1 on the dev shots"),
        }
        for name, shots in (("dev", dev), ("test", test if args.final else [])):
            if not shots:
                continue
            score, _ = run_baseline(data, shots, rms_col=-2)
            y, valid = truth_of(data, shots)
            for label, (thr, why) in variants.items():
                res = scoring.evaluate(
                    shots, y, valid, score, thr, n=args.bootstrap, seed=0
                )
                write(
                    f"tm_baseline_n1rms_{label}_{name}",
                    {
                        **meta,
                        "split": name,
                        "shots": shots,
                        "score": "log10 of the smoothed n=1 RMS, bin maximum",
                        "threshold_log10_g": thr,
                        "threshold_g": float(10**thr),
                        "threshold_is": why,
                        "metrics": res,
                    },
                )
        return 0

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = args.features.replace("+", "_")
    folds = scoring.shot_folds(dev, FOLDS, seed=0)
    oof, thresholds, fold_info = {}, {}, []
    for k in range(FOLDS):
        held = [s for s in dev if folds[s] == k]
        pool = [s for s in dev if folds[s] != k]
        train, val = scoring.inner_split(pool, 100 + k, HYPER["val_fraction"])
        nets = ensemble(data, train, val, device)
        val_scores = predict(nets, data, val, device)
        thr = pick_threshold(data, val_scores, val)
        oof.update(predict(nets, data, held, device))
        thresholds.update({s: thr for s in held})
        fold_info.append(
            {
                "fold": k,
                "n_train": len(train),
                "n_val": len(val),
                "n_held": len(held),
                "threshold": thr,
                "val_loss": [float(n[3]) for n in nets],
            }
        )
        print(f"fold {k}: threshold {thr:.3f}, val loss {fold_info[-1]['val_loss']}")
    y, valid = truth_of(data, dev)
    res = scoring.evaluate(dev, y, valid, oof, thresholds, n=args.bootstrap, seed=0)
    write(
        f"tm_ours_{tag}_cv",
        {
            **meta,
            "split": "dev, shot-grouped 5-fold out-of-fold",
            "hyper": HYPER,
            "folds": {str(k): v for k, v in folds.items()},
            "fold_info": fold_info,
            "features": list(magfeatures.FEATURE_NAMES)
            + (list(magfeatures.RMS_NAMES) if with_rms else []),
            "metrics": res,
        },
    )
    np.savez_compressed(
        TM / "results" / f"oof_tm_ours_{tag}.npz",
        **{f"s{s}": oof[s].astype(np.float32) for s in dev},
    )
    if args.final:
        train, val = scoring.inner_split(dev, 999, HYPER["val_fraction"])
        nets = ensemble(data, train, val, device)
        thr = pick_threshold(data, predict(nets, data, val, device), val)
        scores = predict(nets, data, test, device)
        y, valid = truth_of(data, test)
        res = scoring.evaluate(test, y, valid, scores, thr, n=args.bootstrap, seed=0)
        write(
            f"tm_ours_{tag}_test",
            {
                **meta,
                "split": "test (50 shots), nets trained on all dev shots",
                "test_shots": test,
                "threshold": thr,
                "hyper": HYPER,
                "metrics": res,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
