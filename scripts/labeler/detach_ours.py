#!/usr/bin/env python
"""detach-ours: can the detachment label be read from cheap 0-D signals alone?

    python scripts/labeler/detach_ours.py prep            # environment labelmaker
    CUDA_VISIBLE_DEVICES=1 $LABELER_ROOT/envs/phase3/bin/python \\
        scripts/labeler/detach_ours.py train               # torch, one GPU
    python scripts/labeler/detach_ours.py refresh          # unchanged arrays only

A small 1-D CNN reads a 2 s window (41 bins of 50 ms) of plasma current, line
density, divertor D-alpha, the ELM share and the EFIT scalars, and predicts the
label at the centre bin: attached, detached or marfe. Heating power is excluded
because it is the denominator of the Prad vote. The model also excludes Langmuir
probes, bolometry and camera imagery; some remaining auxiliary inputs participate
in the label's validity gates. Scores measure exploratory agreement with the
constructed labels, with no independent physical benchmark. It is trained on bins
where the label is a certain state. Only shots with every measured input present
and windows with every sample finite are retained. No missingness channels or
imputed values are inputs.

Validation is by shot: 5-fold CV over the shots that are not in the cohort's test
split, then a model trained on all of them scored on the test shots; both with
1000-replicate shot-bootstrap 95% intervals, against the majority-class predictor.
Nothing is tuned on the test shots (the epoch count and architecture are fixed).

Writes `docs/labeler/results/detachment_ours.json`; the dataset and the predictions
are under `$LABELER_ROOT/round4/detach/ours/`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from detach_json import dumps

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
    "ne_rel",
    "dalpha_rel",
    "elm_share",
    "betan",
    "wmhd_mj",
    "q95",
    "kappa",
    "bcentr",
)
EFIT_NODES = ("betan", "wmhd", "q95", "kappa", "bcentr")
CLASS_NAMES = tuple(core.STATE_NAMES[k + 1] for k in range(3))


def root() -> Path:
    return Path(os.environ["LABELER_ROOT"]) / "round4" / "detach"


def dataset_fingerprint(data) -> str:
    """Hash array contents, so NPZ archive timestamps cannot change identity."""
    digest = hashlib.sha256()
    for name in sorted(data.files):
        array = np.ascontiguousarray(data[name])
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def label_source(data) -> dict:
    """Verify prepared targets and splits against the current label table."""
    path = root() / "labels_bins.csv.gz"
    labels = pd.read_csv(path).set_index(["shot", "start_ms"])
    keys = list(zip(data["shot"], data["start_ms"], strict=True))
    rows = labels.loc[keys]
    expected = np.where(
        rows.state_lm.isin((1, 2, 3)), rows.state_lm.to_numpy(int) - 1, -1
    )
    if not np.array_equal(data["y"], expected) or not np.array_equal(
        data["split"], rows.split.to_numpy()
    ):
        raise ValueError("Rerun prep: prepared targets or splits differ from labels")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


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
            x = np.nanmedian([r / np.nanmedian(r) for r in live], axis=0)
            dalpha = core.bin_median(t, x, edges)[0]
            with np.errstate(all="ignore"):
                dalpha = dalpha / np.nanmedian(dalpha)
    columns = {
        "ip_ma": bins["aux_ip_a"] / 1e6,
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
    windows, target, shots, starts, split = [], [], [], [], []
    eligibility = []
    for shot, rows in labels.groupby("shot"):
        x = shot_channels(int(shot))
        if x is None:
            eligibility.append({"shot": int(shot), "reason": "no_signal_grid"})
            continue
        missing_channels = np.asarray(CHANNELS)[~np.isfinite(x).any(axis=0)].tolist()
        if missing_channels:
            eligibility.append(
                {
                    "shot": int(shot),
                    "reason": "incomplete_inputs",
                    "missing_channels": missing_channels,
                }
            )
            continue
        with np.load(root() / "bins" / f"{int(shot)}.npz") as npz:
            grid = npz["start_ms"]
        index = {float(s): i for i, s in enumerate(grid)}
        kept = 0
        for row in rows.itertuples():
            i = index.get(float(row.start_ms))
            if i is None or i < HALF or i + HALF >= len(x):
                continue
            window = x[i - HALF : i + HALF + 1]
            if not np.isfinite(window).all():
                continue
            windows.append(window)
            target.append(int(row.state_lm) - 1 if row.state_lm in (1, 2, 3) else -1)
            shots.append(int(shot))
            starts.append(float(row.start_ms))
            split.append(row.split)
            kept += 1
        eligibility.append(
            {
                "shot": int(shot),
                "split": str(rows.split.iloc[0]),
                "reason": "complete_inputs" if kept else "no_complete_window",
                "assessed_bins": len(rows),
                "finite_windows": kept,
            }
        )
        print("prep", shot, kept, "complete windows of", len(rows), flush=True)
    out = root() / "ours"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "dataset.npz",
        x=np.stack(windows).astype(np.float32)
        if windows
        else np.empty((0, 2 * HALF + 1, len(CHANNELS)), dtype=np.float32),
        y=np.asarray(target, dtype=np.int64),
        shot=np.asarray(shots),
        start_ms=np.asarray(starts),
        split=np.asarray(split),
        channels=np.asarray(CHANNELS),
    )
    (out / "eligibility.json").write_text(
        dumps(
            {
                "policy": "all measured channels present; all window samples finite; "
                "no imputation or missingness channels",
                "shots": eligibility,
            },
            indent=1,
        )
    )
    print("dataset", len(target), "windows", len(set(shots)), "shots")


def metrics(
    y: np.ndarray, pred: np.ndarray, reference_classes: tuple[int, ...] | None = None
) -> dict:
    """F1 requires reference support; macro averages a fixed reference class set."""
    labels = (0, 1, 2)
    if reference_classes is None:
        reference_classes = tuple(k for k in labels if np.any(y == k))
    f1 = {}
    for k in labels:
        tp = np.sum((y == k) & (pred == k))
        fp = np.sum((y != k) & (pred == k))
        fn = np.sum((y == k) & (pred != k))
        f1[k] = 2 * tp / (2 * tp + fp + fn) if np.any(y == k) else float("nan")
    po = float(np.mean(y == pred)) if len(y) else float("nan")
    pe = sum(np.mean(y == k) * np.mean(pred == k) for k in labels) if len(y) else 1.0
    return {
        "accuracy": po,
        "kappa": float((po - pe) / (1 - pe))
        if 1 - pe > np.finfo(float).eps
        else float("nan"),
        "macro_f1": float(np.mean([f1[k] for k in reference_classes]))
        if reference_classes and all(np.isfinite(f1[k]) for k in reference_classes)
        else float("nan"),
        "f1_attached": f1[0],
        "f1_detached": f1[1],
        "f1_marfe": f1[2],
    }


def class_support(y, shot) -> dict:
    """Reference bin counts and independent shot support, including absent classes."""
    return {
        name: {
            "n_bins": int(np.sum(y == k)),
            "n_shots": len(np.unique(shot[y == k])),
            "shot_ids": [int(s) for s in np.unique(shot[y == k])],
        }
        for k, name in enumerate(CLASS_NAMES)
    }


def confusion_matrix(y, pred) -> dict:
    return {
        "labels": list(CLASS_NAMES),
        "rows": "reference",
        "columns": "prediction",
        "counts": [
            [int(np.sum((y == k) & (pred == j))) for j in range(3)] for k in range(3)
        ],
    }


def fold_record(k, held, train_mask, test_mask, y, shot, prob, majority_pred) -> dict:
    """Expose the supervised population and both predictors on each held fold."""
    certain = test_mask & (y >= 0)
    scored = certain & np.isfinite(prob).all(axis=1)
    pred = prob[scored].argmax(axis=1)
    training_labels = y[train_mask]
    majority = np.bincount(training_labels).argmax() if len(training_labels) else -1
    return {
        "fold": k,
        "training_shot_ids": [int(s) for s in np.unique(shot[train_mask])],
        "heldout_shot_ids": sorted(int(s) for s in held),
        "training_certain_bins": int(train_mask.sum()),
        "predicted_bins": int(np.isfinite(prob[test_mask]).all(axis=1).sum()),
        "heldout_certain_bins": int(certain.sum()),
        "scored_certain_bins": int(scored.sum()),
        "training_class_support": class_support(y[train_mask], shot[train_mask]),
        "heldout_class_support": class_support(y[certain], shot[certain]),
        "majority_class": CLASS_NAMES[majority] if majority >= 0 else None,
        "model_metrics": metrics(y[scored], pred),
        "majority_metrics": metrics(y[scored], majority_pred[scored]),
        "model_confusion_matrix": confusion_matrix(y[scored], pred),
        "majority_confusion_matrix": confusion_matrix(y[scored], majority_pred[scored]),
    }


def marfe_transfer(y, shot, split, records) -> dict:
    training = (split != "test") & (y >= 0)
    support = class_support(y[training], shot[training])
    no_training_support = [
        record["fold"]
        for record in records
        if record["heldout_class_support"]["marfe"]["n_bins"]
        and not record["training_class_support"]["marfe"]["n_bins"]
    ]
    return {
        "status": "unsupported",
        "training_class_support": support["marfe"],
        "heldout_marfe_folds_without_training_support": no_training_support,
        "reason": "MARFE candidates lack an independent benchmark; single-shot "
        "support cannot establish transfer to a different shot. Fold support "
        "records show when held-out MARFE has no supervised training examples.",
    }


def with_ci(y, pred, shot, rng) -> dict:
    reference_classes = tuple(k for k in range(3) if np.any(y == k))
    point = metrics(y, pred, reference_classes)
    by_shot = {s: np.flatnonzero(shot == s) for s in np.unique(shot)}
    keys = list(by_shot)
    draws = {k: [] for k in point}
    for _ in range(BOOTSTRAPS if keys else 0):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([by_shot[keys[j]] for j in pick])
        for k, v in metrics(y[idx], pred[idx], reference_classes).items():
            draws[k].append(v)
    result = {
        "n_bins": len(y),
        "n_shots": len(keys),
        "shot_ids": [int(s) for s in keys],
        "reference_class_ids": list(reference_classes),
        "reference_classes": [CLASS_NAMES[k] for k in reference_classes],
        "class_support": class_support(y, shot),
        "confusion_matrix": confusion_matrix(y, pred),
        "bootstrap_policy": "reference classes frozen before shot resampling; "
        "per-class F1 undefined without reference support; macro-F1 undefined "
        "unless every frozen reference class has support; intervals use only "
        "defined replicates, with counts reported for each metric",
    }
    for k, value in point.items():
        finite = np.asarray(draws[k])[np.isfinite(draws[k])]
        result[k] = {
            "value": value,
            "ci95": [float(v) for v in np.percentile(finite, [2.5, 97.5])]
            if len(finite)
            else [float("nan"), float("nan")],
            "valid_replicates": len(finite),
            "replicates": BOOTSTRAPS if keys else 0,
        }
    return result


def baseline_strata(data, pred, rng):
    """Bins where the other two indicators agree, by split and camera source."""
    import detach_benchmark as benchmark

    labels = pd.read_csv(root() / "labels_bins.csv.gz").set_index(["shot", "start_ms"])
    keys = list(zip(data["shot"], data["start_ms"], strict=True))
    rows = labels.loc[keys].reset_index()
    reference = benchmark.loo_state(rows, "tangtv", benchmark.load_model(), 0.7)
    out = {}
    split = rows.split.to_numpy()
    sources = rows.tangtv_source.to_numpy()
    for population in ("cv", "test"):
        base = split != "test" if population == "cv" else split == "test"
        for source in ("all", "inversion", "surrogate"):
            selection = base & (
                np.ones(len(rows), bool) if source == "all" else sources == source
            )
            for name, truth in (
                ("combined", rows.state_lm.to_numpy()),
                ("loo_tangtv", reference),
            ):
                keep = selection & np.isin(truth, (1, 2, 3)) & (pred >= 0)
                entry = with_ci(truth[keep] - 1, pred[keep], data["shot"][keep], rng)
                entry["n_bins"] = int(keep.sum())
                entry["n_shots"] = len(np.unique(data["shot"][keep]))
                out[f"{population}_{source}_{name}"] = entry
    out["reference_note"] = (
        "loo_tangtv = bins where the other two indicators agree: the compatible "
        "Afrac/Prad proxy pair, with posterior gates. This selects agreement and "
        "is not independently reviewed truth."
    )
    return out


def refresh(model: str) -> None:
    """Rescore saved predictions only when all prepared arrays are unchanged."""
    out = root() / model
    result_path = RESULT.with_name(f"detachment_{model}.json")
    result = json.loads(result_path.read_text())
    with np.load(out / "dataset.npz") as data, np.load(out / "predictions.npz") as p:
        fingerprint = dataset_fingerprint(data)
        previous = result.get("dataset_fingerprint")
        if previous is None:
            raise ValueError("Rerun train: saved result lacks an input fingerprint")
        if previous != fingerprint:
            raise ValueError("Rerun train: prepared input arrays or targets changed")
        for name in ("y", "shot", "start_ms", "split"):
            if not np.array_equal(data[name], p[name]):
                raise ValueError(f"Rerun train: prediction {name} differs from dataset")
        source = label_source(data)
        y, shot, split = p["y"], p["shot"], p["split"]
        pred, majority, prob = p["pred"], p["majority_pred"], p["prob"]
        cv = (split != "test") & (y >= 0) & (pred >= 0)
        boot = np.random.default_rng(1)
        result["cv_shots"] = with_ci(y[cv], pred[cv], shot[cv], boot)
        result["cv_majority"] = metrics(y[cv], majority[cv])
        result["cv_majority_ci"] = with_ci(
            y[cv], majority[cv], shot[cv], np.random.default_rng(1)
        )
        records = []
        for old in result["fold_records"]:
            held = old["heldout_shot_ids"]
            held_mask = np.isin(shot, held)
            train_mask = (split != "test") & (y >= 0) & ~held_mask
            records.append(
                fold_record(
                    old["fold"], held, train_mask, held_mask, y, shot, prob, majority
                )
            )
        result["fold_records"] = records
        result["marfe_transfer"] = marfe_transfer(y, shot, split, records)
        result["stratified"] = baseline_strata(data, pred, boot)
        final = (split == "test") & (y >= 0) & (pred >= 0)
        if final.any():
            result["test_shots"] = with_ci(
                y[final], pred[final], shot[final], np.random.default_rng(1)
            )
            result["test_majority"] = metrics(y[final], majority[final])
            result["test_majority_ci"] = with_ci(
                y[final], majority[final], shot[final], np.random.default_rng(1)
            )
        result["dataset_fingerprint"] = fingerprint
        result["label_source"] = source
        result["bootstrap_comparison"] = (
            "model and majority use identical shot resamples with seed 1 "
            "within each population"
        )
        result["prediction_refresh"] = (
            "saved predictions rescored after verifying prepared array contents, "
            "targets and splits; no new model fit"
        )
    result_path.write_text(dumps(result, indent=1))
    print(dumps(result["cv_shots"], indent=1))


def train() -> None:
    import torch
    from torch import nn

    data = np.load(root() / "ours" / "dataset.npz")
    x, y, shot, split = data["x"], data["y"], data["shot"], data["split"]
    if tuple(data["channels"].tolist()) != CHANNELS:
        raise ValueError("Rerun prep: detach-ours input channel set has changed")
    source = label_source(data)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(SEED)

    def standardise(train_x, *others):
        mean = train_x.reshape(-1, train_x.shape[-1]).mean(axis=0)
        std = train_x.reshape(-1, train_x.shape[-1]).std(axis=0) + 1e-6
        return [(a - mean) / std for a in (train_x, *others)]

    def net(channels: int) -> nn.Module:
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.body = nn.Sequential(
                    nn.Conv1d(channels, 32, 5, padding=2),
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
        if not len(train_idx):
            return np.full((len(test_idx), 3), np.nan)
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
    if not np.isfinite(x).all():
        raise ValueError("Rerun prep: detach-ours requires complete finite inputs")
    majority_pred = np.full(len(y), -1, int)
    prob = np.full((len(y), 3), np.nan)
    fold_records = []
    for k, held in enumerate(folds):
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
    fit_labels = y[(split != "test") & (y >= 0)]
    major = np.bincount(fit_labels).argmax() if len(fit_labels) else -1
    if len(final):
        prob[final] = fit_predict(
            np.flatnonzero((split != "test") & (y >= 0)), final, SEED + FOLDS
        )
        majority_pred[final] = major
    scored = np.isfinite(prob).all(axis=1)
    pred = np.full(len(y), -1, int)
    pred[scored] = prob[scored].argmax(axis=1)
    cv = np.flatnonzero((split != "test") & (y >= 0) & scored)
    boot = np.random.default_rng(1)
    result = {
        "inputs": list(CHANNELS),
        "window_bins": 2 * HALF + 1,
        "bin_ms": core.BIN_MS,
        "epochs": EPOCHS,
        "folds": FOLDS,
        "fold_records": fold_records,
        "n_windows": len(y),
        "n_shots": len(np.unique(shot)),
        "class_counts": {
            core.STATE_NAMES[k + 1]: int(np.sum(y == k)) for k in range(3)
        },
        "input_policy": "complete measured input windows; no imputation or "
        "missingness channels",
        "fit_excluded_split": "test",
        "normalization": "mean and standard deviation fitted on training "
        "windows within each shot-held-out fold",
        "unscored_windows": int((~scored).sum()),
        "eligibility": str(root() / "ours" / "eligibility.json"),
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
    result["n_cv_windows"] = len(cv)
    result["n_cv_shots"] = len(np.unique(shot[cv]))
    result["stratified"] = baseline_strata(data, pred, boot)
    final = final[(y[final] >= 0) & scored[final]]
    if len(final):
        result["test_shots"] = with_ci(
            y[final], pred[final], shot[final], np.random.default_rng(1)
        )
        result["test_majority"] = metrics(y[final], majority_pred[final])
        result["test_majority_ci"] = with_ci(
            y[final], majority_pred[final], shot[final], np.random.default_rng(1)
        )
        result["n_test_shots"] = len(np.unique(shot[final]))
    out = root() / "ours"
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
    {"prep": prep, "train": train, "refresh": lambda: refresh("ours")}[args.step]()


if __name__ == "__main__":
    main()
