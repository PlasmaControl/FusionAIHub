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
* the 50 blind test shots are never opened: no test-split scoring exists in this script;
* ``--features magnetics`` is the detector; ``--features magnetics+rms`` adds the lab's
  n = 1 and n = 2 RMS (the traces the labels are drawn from) as the ablation that says
  how much of the label is read back from them;
* ``--baseline`` is no model. Two scores are written: the smoothed n = 1 RMS of each
  bin (``tm-rms``) and the two-line magnetic rule max(n = 1 / 12 G, n = 2 / 6 G)
  (``tm-rms2``: the seed levels of the label itself, no weights, no training). Each is
  scored against a threshold chosen on the same inner-validation shots as every learned
  row; the fixed 12 G (n = 1) and the fixed two-line level 1 (12 G / 6 G) are separately
  labelled extras. AUROC and AUPRC sweep every threshold.

Every number is written with its shots, bins and thresholds to
``$LABELER_ROOT/round4/tm/results/<name>.json``.

Run in the CUDA venv (the library imports need only numpy, pandas and torch)::

    PYTHONPATH=$PWD/src CUDA_VISIBLE_DEVICES=1 \\
        /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/envs/phase3/bin/python \\
        scripts/labeler/tm_ours.py --features magnetics
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for entry in (REPO / "src", Path(__file__).resolve().parent):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from tm_cv_plan import load_plan

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
        feature_ok = np.isfinite(X).all(axis=1)
        # Target uncertainty masks loss/scoring only, never the model's input context.
        z = np.where(feature_ok[:, None], (X - mean) / std, 0.0)
        x[i, :, : len(lab)] = z.T
        y[i, : len(lab)] = lab
        m[i, : len(lab)] = ok
    return tuple(torch.as_tensor(a, device=device) for a in (x, y, m))


def train_one(data, train, val, seed, device):
    """A net trained on `train` shots, early-stopped on `val`; `(net, mean, std)`."""
    import torch
    from torch import nn

    assert any(np.any(data[s][2] & data[s][3]) for s in val), (
        "positive validation required"
    )
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
    threshold = scoring.best_threshold(stats, THRESHOLD_EDGES)
    assert threshold != 0.999
    return threshold


def truth_of(data, shots):
    return (
        {s: data[s][2] for s in shots},
        {s: data[s][3] for s in shots},
    )


SEED_G = (12.0, 6.0)
#: Baseline scores: name -> (description, fixed threshold in log10 of the score's unit,
#: why that threshold). The two-line score is log10 max(n1 / 12 G, n2 / 6 G), so its
#: fixed level is 0: either line at its seed amplitude.
BASELINES = {
    "n1rms": (
        "log10 of the smoothed n=1 RMS, bin maximum",
        float(np.log10(SEED_G[0])),
        "the rule's n = 1 onset level, 12 G",
    ),
    "tworms": (
        "log10 max(n=1 RMS / 12 G, n=2 RMS / 6 G), smoothed, bin maximum",
        0.0,
        "the rule's seed levels: n = 1 at 12 G or n = 2 at 6 G",
    ),
}


def run_baseline(data, shots, kind="n1rms"):
    """Per-bin baseline scores of `kind` (`BASELINES`) for each of `shots`."""
    if kind == "n1rms":
        return {s: data[s][1][:, 0] for s in shots}
    return {
        s: np.fmax(
            data[s][1][:, 0] - np.log10(SEED_G[0]),
            data[s][1][:, 1] - np.log10(SEED_G[1]),
        )
        for s in shots
    }


def load_baseline(shots):
    """RMS availability alone defines baseline coverage; no Mirnov input required."""
    cohort = pd.read_csv(CATALOG / "cohort.csv").set_index("shot")
    table = pd.read_csv(LABELS)
    by_shot = {s: g for s, g in table.groupby("shot")}
    data, missing = {}, []
    for shot in shots:
        path = TM / "signals" / f"{shot}.npz"
        if not path.is_file():
            missing.append(shot)
            continue
        row = cohort.loc[shot]
        centres = scoring.bin_centres((row.window_start_ms, row.window_end_ms))
        with np.load(path) as z:
            features = magfeatures.rms_bin_features(
                z["t_ms"], z["n1rms"], z["n2rms"], centres
            )
        y, valid = scoring.label_bins(by_shot.get(shot, table.iloc[:0]), centres)
        data[shot] = (centres, features, y, valid & np.isfinite(features[:, 0]))
    return data, missing


def save_ensemble(nets, tag, fold, train, val, threshold):
    """Persist deployable weights with their normalisation and decision threshold."""
    import torch

    path = TM / "checkpoints" / f"tm_ours_{tag}" / f"fold_{fold}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "members": [
                {
                    "state_dict": {
                        k: v.detach().cpu() for k, v in net.state_dict().items()
                    },
                    "mean": mean,
                    "std": std,
                    "val_loss": float(loss),
                }
                for net, mean, std, loss in nets
            ],
            "hyper": HYPER,
            "threshold": threshold,
            "train_shots": train,
            "validation_shots": val,
            "labels_sha256": hashlib.sha256(LABELS.read_bytes()).hexdigest(),
        },
        path,
    )
    return str(path)


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
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args(argv)
    if not args.baseline and args.features is None:
        raise SystemExit("name --features or --baseline")

    cohort = pd.read_csv(CATALOG / "cohort.csv")
    dev_all = sorted(int(s) for s in cohort[cohort.split != "test"].shot)
    with_rms = args.baseline or args.features == "magnetics+rms"
    data, skipped = load_baseline(dev_all) if args.baseline else load(dev_all, with_rms)
    dev = [s for s in dev_all if s in data]
    meta = {
        "git_sha": git_sha(),
        "labels": str(LABELS.relative_to(REPO)),
        "bin_ms": scoring.BIN_MS,
        "shots_without_features": skipped,
        "dev_shots": dev,
        "n_dev": len(dev),
        "cv_cohort_shots": dev_all,
        "labels_sha256": hashlib.sha256(LABELS.read_bytes()).hexdigest(),
        "input_mask_policy": (
            "zero only nonfinite feature rows; target uncertainty masks loss and "
            "evaluation, never model input context"
        ),
    }

    if args.baseline:
        # the baselines need no magnetics features: their columns are the RMS
        y, valid = truth_of(data, dev)
        _, splits = load_plan(dev_all)
        for kind, (what, fixed, fixed_why) in BASELINES.items():
            score = run_baseline(data, dev, kind)
            tuned, info = scoring.cv_thresholds(dev_all, y, valid, score, splits=splits)
            for fold in info:
                fold["threshold_native"] = float(10 ** fold["threshold"])
            variants = {
                "12g" if kind == "n1rms" else "seed": (fixed, fixed_why),
                "cv_tuned": (
                    tuned,
                    "F1-maximising on shared-fold inner validation only",
                ),
            }
            for label, (thr, why) in variants.items():
                res = scoring.evaluate(
                    dev, y, valid, score, thr, n=args.bootstrap, seed=0
                )
                write(
                    f"tm_baseline_{kind}_{label}_dev",
                    {
                        **meta,
                        "split": "dev",
                        "shots": dev,
                        "score": what,
                        "threshold_log10": None if isinstance(thr, dict) else thr,
                        "threshold_native": None
                        if isinstance(thr, dict)
                        else float(10**thr),
                        "thresholds_by_shot": {str(s): float(v) for s, v in thr.items()}
                        if isinstance(thr, dict)
                        else None,
                        "folds": scoring.shared_cv(dev_all)[0],
                        "fold_info": info,
                        "threshold_is": why,
                        "metrics": res,
                    },
                )
        return 0

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.28)
    tag = args.features.replace("+", "_")
    folds, shared_splits = load_plan(dev_all)
    oof, thresholds, fold_info = {}, {}, []
    for split in shared_splits:
        k = split["fold"]
        held = [s for s in split["held"] if s in data]
        train = [s for s in split["train"] if s in data]
        val = [s for s in split["validation"] if s in data]
        nets = ensemble(data, train, val, device)
        val_scores = predict(nets, data, val, device)
        thr = pick_threshold(data, val_scores, val)
        validation_file = TM / "results" / f"validation_tm_ours_{tag}_fold{k}.npz"
        validation_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            validation_file,
            **{f"s{s}": val_scores[s].astype(np.float32) for s in val},
        )
        validation_stats = [
            scoring.shot_stats(
                s, data[s][2], data[s][3], val_scores[s], THRESHOLD_EDGES
            )
            for s in val
        ]
        oof.update(predict(nets, data, held, device))
        thresholds.update({s: thr for s in held})
        fold_info.append(
            {
                "fold": k,
                "n_train": len(train),
                "n_val": len(val),
                "n_held": len(held),
                "threshold": thr,
                "train": train,
                "validation": val,
                "held": held,
                "shared_cohort_split": split,
                "validation_scores": str(validation_file),
                "validation_bins_scored": sum(s.n_bins for s in validation_stats),
                "validation_bins_positive": sum(s.n_pos for s in validation_stats),
                "validation_bins_negative": sum(s.n_neg for s in validation_stats),
                "validation_shots_positive": [
                    s.shot for s in validation_stats if s.n_pos
                ],
                "threshold_estimable": bool(sum(s.n_pos for s in validation_stats)),
                "threshold_status": "positive-bearing inner-validation F1 optimum",
                "checkpoint": save_ensemble(nets, tag, k, train, val, thr),
                "val_loss": [float(n[3]) for n in nets],
            }
        )
        print(f"fold {k}: threshold {thr:.3f}, val loss {fold_info[-1]['val_loss']}")
    y, valid = truth_of(data, dev)
    res = scoring.evaluate(dev, y, valid, oof, thresholds, n=args.bootstrap, seed=0)
    meta["cuda_peak_allocated_gb"] = (
        torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None
    )
    write(
        f"tm_ours_{tag}_cv",
        {
            **meta,
            "split": "dev, shot-grouped 5-fold out-of-fold",
            "hyper": HYPER,
            "threshold_is": "F1-maximising on shared-fold inner validation shots only",
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
