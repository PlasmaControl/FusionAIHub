#!/usr/bin/env python
"""`tm-onsetcnn` and `tm-dsm` retrained as detectors of the whole-interval labels.

The published models forecast (the CNN the mode's presence 25 ms ahead, the DSM an onset
within 250 ms to 1 s); the Tokamak-SI labels say whether a tearing mode is present.
Here the *same architecture on the same inputs* is trained afresh on that question:

* target: a tearing mode is present at the row's time stamp (horizon 0), from the
  cohort's whole-interval labels; rows in an uncertain or not-observable span, rows the
  model calls invalid and rows outside the catalog window are left out;
* `tm-onsetcnn`: the Keras graph of the published ensemble (two BatchNormalization +
  Conv1D + MaxPooling blocks on the five 33-point profiles, a dense block joined to the
  eleven scalars, a 64-32 head with dropout), rebuilt with fresh weights; the loss is
  the cross-entropy of its tearing logit; three seeds averaged (the published ensemble
  has ten); AdamW (`HYPER`);
* `tm-dsm`: the 38-number input the published model reads (`spec.preprocess`), its
  100-1000 bias-free ReLU6 embedding, and one logit in place of the log-normal mixture
  (a horizon-0 target has no time to event); three seeds;
* protocol as `tm_ours.py`: the 450 train and validation shots in five shot-grouped
  folds (`scoring.shot_folds`, seed 0), each scored by nets trained on the other four
  with a tenth of their shots stopping the training and choosing the F1 threshold; the
  50 test shots scored once (``--final``) by nets trained on all 450;
* scores are the nets' row probabilities interpolated to the 10 ms bins
  (`detectors.rows_to_bins`), then `scoring.evaluate` with 1000 shot bootstraps.

Every number is written with its shots, bins and thresholds to
``$LABELER_ROOT/round4/tm/results/``. Run in the CUDA venv::

    PYTHONPATH=$PWD/src CUDA_VISIBLE_DEVICES=1 LABELER_ROOT=<main root> \\
        /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/envs/phase3/bin/python \\
        scripts/labeler/tm_prior_retrain.py --model cnn
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from labeler.tearing import detectors, scoring

ROOT = Path(os.environ.get("LABELER_ROOT", "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker"))
TM = ROOT / "round4/tm"
INPUTS = TM / "detector_inputs"
RESULTS = Path(os.environ.get("TM_RESULTS", TM / "results"))
CATALOG = REPO / "data/events/catalog"
LABELS = (
    REPO / "data/events/neoclassical_tearing_mode/extend_tm_interval/tm_interval.csv"
)
MODELS = {
    "cnn": ("tm-onsetcnn", "d3d_tearing_onset_cnn1d"),
    "dsm": ("tm-dsm", "d3d_tearing_time_to_event_dsm"),
}
#: Retained from the earlier development-only inner-validation search. The repaired
#: benchmark changes cohort-wide fold roles and labels, and refits these fixed settings;
#: only per-fold stopping and decision thresholds are tuned during this regeneration.
HYPER = {
    "cnn": {"lr": 3e-4, "batch_size": 128, "weight_decay": 1e-4},
    "dsm": {"lr": 1e-4, "batch_size": 128, "weight_decay": 1e-4},
    "max_epochs": 60,
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


def write(name: str, record: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(
        json.dumps(record, indent=1, default=float) + "\n", encoding="utf-8"
    )
    print(path, flush=True)
    return path


def load(kind: str, shots) -> tuple[dict, list[int]]:
    """`({shot: arrays}, shots_without_inputs)` for the shots with exported inputs."""
    slug = MODELS[kind][1]
    cohort = pd.read_csv(CATALOG / "cohort.csv").set_index("shot")
    table = pd.read_csv(LABELS)
    by_shot = {s: g for s, g in table.groupby("shot")}
    data, missing = {}, []
    for shot in shots:
        path = INPUTS / slug / f"{shot}.npz"
        if not path.is_file():
            missing.append(int(shot))
            continue
        with np.load(path) as z:
            t_ms = z["t_s"] * 1000.0
            inputs = (z["scalars"], z["profiles"]) if kind == "cnn" else (z["x"],)
            model_ok = z["valid"].copy()
        for a in inputs:
            model_ok &= np.isfinite(a.reshape(len(a), -1)).all(axis=1)
        win = (
            float(cohort.loc[shot, "window_start_ms"]),
            float(cohort.loc[shot, "window_end_ms"]),
        )
        rows = by_shot.get(shot, table.iloc[:0])
        y, label_ok = detectors.row_labels(rows, t_ms, win)
        centres = scoring.bin_centres(win)
        yb, vb = scoring.label_bins(rows, centres)
        data[int(shot)] = {
            "t_ms": t_ms,
            "inputs": inputs,
            "model_ok": model_ok,
            "y": y.astype(np.float32),
            "train_ok": model_ok & label_ok,
            "centres": centres,
            "y_bins": yb,
            "valid_bins": vb,
        }
    return data, missing


def stack(data, shots):
    """`(inputs, y)` of the rows that train: model-valid and label-valid."""
    parts = [(data[s]["inputs"], data[s]["y"], data[s]["train_ok"]) for s in shots]
    n_in = len(parts[0][0])
    inputs = tuple(
        np.concatenate([p[0][i][p[2]] for p in parts]).astype(np.float32)
        for i in range(n_in)
    )
    return inputs, np.concatenate([p[1][p[2]] for p in parts])


def make(kind: str, seed: int):
    import torch

    if kind == "cnn":
        from labeler.models.runners import keras_h5

        graph = keras_h5.load_graph(
            ROOT / "models" / MODELS[kind][1] / "best_model_0_4c.h5",
            dtype=torch.float32,
        )
        return detectors.KerasDetector(graph, column=1, seed=seed)
    from labeler.models.runners import dsm_pickle

    graph = dsm_pickle.load_dsm(ROOT / "models" / MODELS[kind][1] / "rt_fixed_rot.pkl")
    sizes = (graph.embedding[0].shape[1], *[w.shape[0] for w in graph.embedding])
    return detectors.DsmDetector(sizes, seed=seed)


def ensemble(kind, data, train, val, device):
    own = HYPER[kind]
    nets = []
    for seed in HYPER["seeds"]:
        net = make(kind, seed)
        record = detectors.fit(
            net,
            stack(data, train),
            stack(data, val),
            lr=own["lr"],
            weight_decay=own["weight_decay"],
            batch_size=own["batch_size"],
            max_epochs=HYPER["max_epochs"],
            patience=HYPER["patience"],
            seed=seed,
            device=device,
        )
        nets.append((net, record))
    return nets


def bin_scores(nets, data, shots, device):
    """`{shot: score per 10 ms bin}`: the ensemble's mean probability, interpolated."""
    out = {}
    for s in shots:
        d = data[s]
        ok = d["model_ok"]
        prob = np.full(len(ok), np.nan)
        if ok.any():
            rows = tuple(a[ok] for a in d["inputs"])
            prob[ok] = np.mean(
                [detectors.predict(n, rows, device=device) for n, _ in nets], axis=0
            )
        out[s] = detectors.rows_to_bins(d["t_ms"], d["model_ok"], prob, d["centres"])
    return out


def pick_threshold(data, scores, shots):
    stats = [
        scoring.shot_stats(
            s,
            data[s]["y_bins"],
            data[s]["valid_bins"],
            scores[s],
            THRESHOLD_EDGES,
            None,
        )
        for s in shots
    ]
    return scoring.best_threshold(stats, THRESHOLD_EDGES)


def truth(data, shots):
    return (
        {s: data[s]["y_bins"] for s in shots},
        {s: data[s]["valid_bins"] for s in shots},
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--final", action="store_true", help="also score the 50 test shots")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument(
        "--folds", type=int, default=FOLDS, help="run only the first k folds"
    )
    args = ap.parse_args(argv)
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.28)
    name, slug = MODELS[args.model]
    cohort = pd.read_csv(CATALOG / "cohort.csv")
    dev_all = sorted(int(s) for s in cohort[cohort.split != "test"].shot)
    test_all = sorted(int(s) for s in cohort[cohort.split == "test"].shot)
    data, missing = load(args.model, dev_all + (test_all if args.final else []))
    dev = [s for s in dev_all if s in data]
    test = [s for s in test_all if s in data]
    meta = {
        "model": name,
        "slug": slug,
        "variant": "retrained from fresh weights, detection objective (horizon 0)",
        "git_sha": git_sha(),
        "labels": str(LABELS.relative_to(REPO)),
        "labels_sha256": hashlib.sha256(LABELS.read_bytes()).hexdigest(),
        "bin_ms": scoring.BIN_MS,
        "shots_without_inputs": missing,
        "dev_shots": dev,
        "n_dev": len(dev),
        "hyper": HYPER,
        "hyperparameter_provenance": (
            "fixed from earlier development-only search; reused without selecting on "
            "new held-fold or blind-test results"
        ),
        "device": device,
    }
    folds, shared_splits = scoring.shared_cv(dev_all)
    meta["cv_cohort_shots"] = dev_all
    meta["threshold_is"] = "F1-maximising on shared-fold inner validation shots only"
    oof, thresholds, fold_info = {}, {}, []
    for split in shared_splits[: min(args.folds, FOLDS)]:
        k = split["fold"]
        began = time.time()
        held = [s for s in split["held"] if s in data]
        train = [s for s in split["train"] if s in data]
        val = [s for s in split["validation"] if s in data]
        nets = ensemble(args.model, data, train, val, device)
        val_scores = bin_scores(nets, data, val, device)
        thr = pick_threshold(data, val_scores, val)
        validation_file = RESULTS / f"validation_{name}_fold{k}.npz"
        RESULTS.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            validation_file,
            **{f"s{s}": val_scores[s].astype(np.float32) for s in val},
        )
        validation_stats = [
            scoring.shot_stats(
                s,
                data[s]["y_bins"],
                data[s]["valid_bins"],
                val_scores[s],
                THRESHOLD_EDGES,
            )
            for s in val
        ]
        oof.update(bin_scores(nets, data, held, device))
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
                "threshold_status": (
                    "inner-validation F1 optimum"
                    if any(s.n_pos for s in validation_stats)
                    else "no positive validation bins; deterministic highest-edge "
                    "fallback, not an estimable F1 optimum"
                ),
                "members": [record for _, record in nets],
                "seconds": round(time.time() - began, 1),
            }
        )
        print(
            f"{name} fold {k}: threshold {thr:.3f}, "
            f"val loss {[round(r['best_val_loss'], 4) for _, r in nets]}, "
            f"{fold_info[-1]['seconds']} s",
            flush=True,
        )
    scored = [s for s in dev if s in oof]
    y, valid = truth(data, scored)
    res = scoring.evaluate(scored, y, valid, oof, thresholds, n=args.bootstrap, seed=0)
    meta["cuda_peak_allocated_gb"] = (
        torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None
    )
    write(
        f"tm_prior_retrained_{name}_cv",
        {
            **meta,
            "split": "dev, shot-grouped 5-fold out-of-fold",
            "folds": {str(s): f for s, f in folds.items()},
            "fold_info": fold_info,
            "scored_shots": scored,
            "metrics": res,
        },
    )
    np.savez_compressed(
        RESULTS / f"oof_tm_prior_retrained_{name}.npz",
        **{f"s{s}": oof[s].astype(np.float32) for s in scored},
    )
    print(
        name,
        "cv",
        f"AUROC {res['auroc']['value']:.3f} AUPRC {res['auprc']['value']:.3f}"
        f" F1 {res['f1']['value']:.3f}",
        flush=True,
    )
    if args.final:
        train_all, val_all = scoring.inner_split(dev_all, 999, HYPER["val_fraction"])
        train = [s for s in train_all if s in data]
        val = [s for s in val_all if s in data]
        nets = ensemble(args.model, data, train, val, device)
        thr = pick_threshold(data, bin_scores(nets, data, val, device), val)
        scores = bin_scores(nets, data, test, device)
        y, valid = truth(data, test)
        res = scoring.evaluate(test, y, valid, scores, thr, n=args.bootstrap, seed=0)
        write(
            f"tm_prior_retrained_{name}_test",
            {
                **meta,
                "split": "test (50 shots), nets trained on all dev shots",
                "test_shots": test,
                "threshold": thr,
                "members": [record for _, record in nets],
                "metrics": res,
            },
        )
        print(
            name,
            "test",
            f"AUROC {res['auroc']['value']:.3f} AUPRC {res['auprc']['value']:.3f}"
            f" F1 {res['f1']['value']:.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
