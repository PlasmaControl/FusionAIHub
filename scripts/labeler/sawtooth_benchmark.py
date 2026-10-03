"""Shot-grouped saw-hl3 / saw-ours training, calibration, inference and scoring.

Only fixed cohort train shots enter the three CV folds. Fixed val shots select
checkpoints and crash thresholds; fixed test shots are never loaded here. All
expert-reviewed shots are excluded from training and checkpoint/threshold choice.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sawtooth_physics import REPO, REVIEW, SEED, WORK, save_json
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from torch.nn import functional as F

from labeler.sawtooth.metrics import aggregate, event_cells, score_histogram, spans_at
from labeler.sawtooth.models import HL3, PhasePicker, soft_crash_target

MODELS = ("saw-hl3", "saw-ours")
LEGACY = {
    "source": "OuYang et al. PPCF 67 (2025) 105004, doi:10.1088/1361-6587/ae0786; local digest",
    "real_time": {
        "accuracy_stated": 0.922,
        "accuracy_abstract": 0.925,
        "correct": 11356,
        "windows": 13595,
        "accuracy_from_counts": 11356 / 13595,
        "class_0_recall": 0.961,
        "class_1_precision": 0.808,
        "class_1_recall": 0.908,
        "class_2_precision": 0.905,
        "class_2_recall": 0.755,
    },
    "offline": {
        "accuracy_stated": 0.956,
        "correct": 13509,
        "windows": 14900,
        "accuracy_from_counts": 13509 / 14900,
        "class_0_precision": 0.985,
        "class_0_recall": 0.873,
        "class_1_recall": 0.976,
        "class_2_precision": 0.941,
        "class_2_recall": 0.751,
    },
    "limitation": "Published three-regime window classification, no crash score or CIs; counts disagree with stated accuracy; split grouping unspecified.",
}


def manifest(work):
    cohort = pd.read_csv(REPO / "data/events/catalog/cohort.csv")
    reviewed = {int(s) for s in pd.read_csv(REVIEW).shot}
    usable = {int(p.stem) for p in (work / "signals").glob("*.npz")}
    train = sorted(set(cohort[cohort.split == "train"].shot) & usable - reviewed)
    validation = sorted(set(cohort[cohort.split == "val"].shot) & usable - reviewed)
    shuffled = np.random.default_rng(SEED).permutation(train)
    folds = {int(s): i % 3 for i, s in enumerate(shuffled)}
    data = {
        "seed": SEED,
        "folds": folds,
        "training_cohort": train,
        "validation_shots": validation,
        "expert_shots": sorted(reviewed & usable),
        "blind_test_excluded": sorted(cohort[cohort.split == "test"].shot),
        "unusable_train": sorted(set(cohort[cohort.split == "train"].shot) - usable),
        "unusable_val": sorted(set(cohort[cohort.split == "val"].shot) - usable),
        "all_reviewed_excluded": sorted(reviewed),
    }
    save_json(work / "split_manifest.json", data)
    save_json(REPO / "outputs/labeler/sawtooth/split_manifest.json", data)
    return data


def record(work, shot):
    return json.loads((work / "shots" / f"{shot}.json").read_text())


def period_boundary(work, shots):
    periods = [
        r["attrs"]["period_ms"]
        for shot in shots
        for r in record(work, shot)["intervals"]
    ]
    if not periods:
        raise ValueError("training fold has no train periods")
    return float(np.median(periods))


def targets(t, rec, boundary):
    spans = [(r["start_s"], r["end_s"]) for r in rec["intervals"]]
    presence = spans_at(t, spans).astype(np.float32)
    picks = soft_crash_target(t, [r["time_s"] for r in rec["crashes"]])
    classes = np.zeros(len(t), dtype=np.int64)
    for r in rec["intervals"]:
        classes[(t >= r["start_s"]) & (t < r["end_s"])] = 1 + int(
            r["attrs"]["period_ms"] > boundary
        )
    return picks, presence, classes


def windows(work, shots, model, boundary, *, per_shot=64):
    """Natural random windows + pick-centered windows for the dense picker."""
    width = 200 if model == "saw-hl3" else 1000
    xs, ys, owners = [], [], []
    for shot in shots:
        data = np.load(work / "signals" / f"{shot}.npz")
        t = data["t"]
        x = data["baseline" if model == "saw-hl3" else "y"].astype(np.float32)
        if len(t) < width:
            continue
        rec = record(work, shot)
        pick, present, classes = targets(t, rec, boundary)
        rng = np.random.default_rng(SEED + shot)
        starts = rng.integers(0, len(t) - width + 1, size=per_shot)
        if model == "saw-ours":
            centers = np.array(
                [np.searchsorted(t, r["time_s"]) - width // 2 for r in rec["crashes"]]
            )
            centers = np.clip(centers, 0, len(t) - width).astype(int)
            if len(centers):
                starts = np.r_[
                    starts[: per_shot // 2],
                    rng.choice(
                        centers,
                        size=per_shot // 2,
                        replace=len(centers) < per_shot // 2,
                    ),
                ]
        for start in starts:
            window = x[:, start : start + width]
            # Missing input channels remain zero after normalization; reject only
            # windows with inadequate ECE core coverage, not optional auxiliaries.
            core = window[:2] if model == "saw-hl3" else window[20:36]
            if np.isfinite(core).mean() < 0.8:
                continue
            xs.append(window.astype(np.float16))
            ys.append(
                int(classes[start + width // 2])
                if model == "saw-hl3"
                else np.stack(
                    [pick[start : start + width], present[start : start + width]]
                ).astype(np.float16)
            )
            owners.append(shot)
    return np.stack(xs), np.array(ys), np.array(owners)


def normalizer(x):
    value = x.astype(np.float32)
    finite = np.isfinite(value)
    count = finite.sum(axis=(0, 2))
    mean = np.divide(
        np.where(finite, value, 0).sum(axis=(0, 2)),
        count,
        out=np.zeros(x.shape[1]),
        where=count > 0,
    )
    squared = np.where(finite, (value - mean[None, :, None]) ** 2, 0).sum(axis=(0, 2))
    std = np.sqrt(np.divide(squared, count, out=np.ones(len(mean)), where=count > 0))
    return mean.astype(np.float32), np.maximum(std, 0.001).astype(np.float32)


def normalize(x, mean, std):
    return np.clip(
        np.nan_to_num(
            (x.astype(np.float32) - mean[None, :, None]) / std[None, :, None],
            nan=0,
            posinf=0,
            neginf=0,
        ),
        -10,
        10,
    )


def loss(model, logits, y, weights=None):
    if model == "saw-hl3":
        return F.cross_entropy(logits, y.long(), weight=weights)
    logp = F.log_softmax(logits[:, :2], dim=1)
    pick, present = y[:, 0], y[:, 1]
    point = -(20 * pick * logp[:, 1] + (1 - pick) * logp[:, 0]).mean()
    interval = F.binary_cross_entropy_with_logits(logits[:, 2], present)
    return point + interval


def infer(model, name, signal, mean, std, *, batch=128):
    t = signal["t"]
    values = signal["baseline" if name == "saw-hl3" else "y"]
    model.eval()
    with torch.no_grad():
        if name == "saw-hl3":
            starts = np.arange(0, len(t) - 200 + 1, 20)
            chunks = []
            for offset in range(0, len(starts), batch):
                selected = starts[offset : offset + batch]
                x = values[:, selected[:, None] + np.arange(200)].transpose(1, 0, 2)
                output = (
                    model(torch.from_numpy(normalize(x, mean, std)))
                    .softmax(dim=1)
                    .numpy()
                )
                chunks.append(output)
            probabilities = np.concatenate(chunks)
            centers = t[starts + 100]
            presence = np.interp(
                t,
                centers,
                1 - probabilities[:, 0],
                left=1 - probabilities[0, 0],
                right=1 - probabilities[-1, 0],
            )
            return {
                "t": t,
                "presence": presence.astype(np.float32),
                "class_t": centers,
                "class_prob": probabilities.astype(np.float32),
            }
        # Overlap-and-crop removes artificial seams; no shots are concatenated.
        width, margin, hop = 1000, 100, 800
        starts = np.arange(0, len(t), hop)
        probability = np.zeros((2, len(t)), dtype=np.float32)
        count = np.zeros(len(t), dtype=np.float32)
        for offset in range(0, len(starts), 32):
            selected = starts[offset : offset + 32]
            chunks = []
            for start in selected:
                index = np.clip(
                    np.arange(start - margin, start - margin + width), 0, len(t) - 1
                )
                chunks.append(values[:, index])
            logits = model(torch.from_numpy(normalize(np.stack(chunks), mean, std)))
            output = torch.stack(
                [logits[:, :2].softmax(dim=1)[:, 1], logits[:, 2].sigmoid()], dim=1
            ).numpy()
            for start, p in zip(selected, output, strict=True):
                end = min(len(t), start + hop)
                probability[:, start:end] += p[:, margin : margin + end - start]
                count[start:end] += 1
        probability /= np.maximum(count, 1)
        return {"t": t, "crash": probability[0], "presence": probability[1]}


def picks(name, prediction, signal, threshold, z=4):
    t = prediction["t"]
    if name == "saw-ours":
        peaks, _ = find_peaks(prediction["crash"], height=threshold, distance=50)
    else:
        # The published network has no timing head. Gate a *single-channel*
        # derivative picker with its predicted regime probability. No inversion,
        # q, coincidence or physics labels enter this inference step.
        core = np.nan_to_num(signal["baseline"][0].astype(float), nan=0)
        derivative = -gaussian_filter1d(core, 2.5, order=1)
        scale = 1.4826 * np.median(np.abs(derivative - np.median(derivative)))
        peaks, _ = find_peaks(derivative, height=z * max(scale, 1e-8), distance=50)
        peaks = peaks[prediction["presence"][peaks] >= threshold]
    return t[peaks]


def calibrate(name, predictions, work):
    options = []
    for threshold in (0.3, 0.5, 0.7):
        for z in (2, 4, 6) if name == "saw-hl3" else (0,):
            cells = np.zeros(3)
            for shot, prediction in predictions.items():
                signal = np.load(work / "signals" / f"{shot}.npz")
                estimate = picks(name, prediction, signal, threshold, z)
                reference = [r["time_s"] for r in record(work, shot)["crashes"]]
                cells += event_cells(reference, estimate, 2)
            tp, fp, fn = cells
            f1 = 2 * tp / max(1, 2 * tp + fp + fn)
            options.append(
                {
                    "threshold": threshold,
                    "z": z,
                    "f1": float(f1),
                    "cells": cells.astype(int).tolist(),
                }
            )
    selected = max(options, key=lambda r: (r["f1"], r["threshold"], r["z"]))
    return selected, options


def train(args):
    torch.set_num_threads(args.threads)
    split = manifest(args.work)
    for name in args.models:
        for fold in args.folds:
            out = args.work / "models" / name / f"fold_{fold}"
            out.mkdir(parents=True, exist_ok=True)
            if (out / "complete.json").exists():
                print(f"resume: {name} fold {fold} complete", flush=True)
                continue
            seed = SEED + fold
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            training = [
                s for s in split["training_cohort"] if split["folds"][s] != fold
            ]
            heldout = [s for s in split["training_cohort"] if split["folds"][s] == fold]
            validation = split["validation_shots"]
            boundary = period_boundary(args.work, training)
            x, y, owners = windows(
                args.work, training, name, boundary, per_shot=args.windows_per_shot
            )
            vx, vy, _ = windows(args.work, validation, name, boundary, per_shot=32)
            mean, std = normalizer(x)
            class_weights = None
            if name == "saw-hl3":
                counts = np.bincount(y, minlength=3)
                class_weights = torch.tensor(
                    len(y) / (3 * np.maximum(counts, 1)), dtype=torch.float32
                )
            model = HL3() if name == "saw-hl3" else PhasePicker()
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=0.001,
                weight_decay=0.003 if name == "saw-hl3" else 0.0001,
            )
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
            history, best, best_epoch, stale = [], float("inf"), -1, 0
            batch = 128 if name == "saw-hl3" else 32
            begun = time.monotonic()
            for epoch in range(args.epochs):
                model.train()
                losses = []
                for _ in range(args.steps):
                    selected = rng.integers(0, len(x), size=batch)
                    xx = normalize(x[selected], mean, std)
                    yy = y[selected].astype(np.float32)
                    if name == "saw-hl3":
                        # The paper's +/-0.2 temporal shift is ambiguous; use
                        # +/-20% of the 20 ms input and shift the center label too.
                        # Regime windows near boundaries are not rolled to avoid
                        # invalid class supervision. Amplitude/noise augment only.
                        xx *= rng.uniform(0.8, 1.2, size=(batch, 1, 1)).astype(
                            np.float32
                        )
                        xx += rng.normal(0, 0.1, size=xx.shape).astype(np.float32)
                    logits = model(torch.from_numpy(xx))
                    value = loss(name, logits, torch.from_numpy(yy), class_weights)
                    optimizer.zero_grad()
                    value.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    losses.append(value.item())
                model.eval()
                vloss = []
                with torch.no_grad():
                    for start in range(0, len(vx), batch):
                        logits = model(
                            torch.from_numpy(
                                normalize(vx[start : start + batch], mean, std)
                            )
                        )
                        vloss.append(
                            loss(
                                name,
                                logits,
                                torch.from_numpy(
                                    vy[start : start + batch].astype(np.float32)
                                ),
                                class_weights,
                            ).item()
                        )
                validation_loss = float(np.mean(vloss))
                history.append(
                    {
                        "epoch": epoch + 1,
                        "train_loss": float(np.mean(losses)),
                        "val_loss": validation_loss,
                    }
                )
                print(
                    f"{name} fold {fold} epoch {epoch + 1}: train {np.mean(losses):.4f} val {validation_loss:.4f}",
                    flush=True,
                )
                if validation_loss < best:
                    best, best_epoch, stale = validation_loss, epoch + 1, 0
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    stale += 1
                scheduler.step()
                if stale >= 5:
                    break
            model.load_state_dict(best_state)
            torch.save(
                {
                    "state_dict": best_state,
                    "mean": mean,
                    "std": std,
                    "boundary_ms": boundary,
                    "training_shots": training,
                    "validation_shots": validation,
                    "model": name,
                },
                out / "checkpoint.pt",
            )
            del x, y, vx, vy
            validation_predictions = {
                s: infer(
                    model, name, np.load(args.work / "signals" / f"{s}.npz"), mean, std
                )
                for s in validation
            }
            selected, options = calibrate(name, validation_predictions, args.work)
            pred_dir = args.work / "predictions" / name / f"fold_{fold}"
            pred_dir.mkdir(parents=True, exist_ok=True)
            for shot in heldout + split["expert_shots"]:
                signal = np.load(args.work / "signals" / f"{shot}.npz")
                prediction = infer(model, name, signal, mean, std)
                prediction["picks"] = picks(
                    name, prediction, signal, selected["threshold"], selected["z"]
                )
                np.savez_compressed(pred_dir / f"{shot}.npz", **prediction)
            summary = {
                "model": name,
                "fold": fold,
                "seed": seed,
                "training_shots": training,
                "heldout_shots": heldout,
                "validation_shots": validation,
                "expert_shots": split["expert_shots"],
                "training_windows": len(owners),
                "period_boundary_ms": boundary,
                "history": history,
                "best_epoch": best_epoch,
                "selected_crash_threshold": selected,
                "threshold_candidates": options,
                "presence_threshold": 0.5,
                "batch_size": batch,
                "max_epochs": args.epochs,
                "steps_per_epoch": args.steps,
                "cpu_threads": args.threads,
                "device": "cpu",
                "seconds": round(time.monotonic() - begun, 1),
                "parameters": sum(p.numel() for p in model.parameters()),
            }
            save_json(out / "complete.json", summary)
            save_json(
                REPO / f"outputs/labeler/sawtooth/{name}_fold_{fold}.json", summary
            )


def evaluate(args):
    split = manifest(args.work)
    output = {
        "legacy": LEGACY,
        "Tokamak-SI": {},
        "protocol": {
            "primary": "three-fold out-of-fold predictions on fixed train cohort; fixed val for checkpoint/crash threshold",
            "bin_ms": 2,
            "interval_threshold": 0.5,
            "crash_tolerances_ms": [1, 2],
            "test_split": "excluded, never loaded",
            "expert": "all reviewed shots excluded from every fold and validation",
            "bootstrap": "1000 shot resamples, seed 20261003; probabilities quantized to 512 bins for AUROC/AP",
            "gaussian_target_sigma_ms": 0.5,
        },
    }
    for name in args.models:
        rows = {1: [], 2: []}
        class_cells = np.zeros((3, 3), dtype=int)
        for shot in split["training_cohort"]:
            fold = split["folds"][shot]
            path = args.work / "predictions" / name / f"fold_{fold}" / f"{shot}.npz"
            if not path.exists():
                raise FileNotFoundError(path)
            pred = np.load(path)
            rec = record(args.work, shot)
            t = pred["t"][10::20]
            truth = spans_at(t, [(r["start_s"], r["end_s"]) for r in rec["intervals"]])
            hist = score_histogram(truth, pred["presence"][10::20])
            ref = [r["time_s"] for r in rec["crashes"]]
            for tolerance in (1, 2):
                rows[tolerance].append(
                    {
                        "shot": shot,
                        "cells": event_cells(ref, pred["picks"], tolerance),
                        "histogram": hist,
                    }
                )
            if name == "saw-hl3":
                summary = json.loads(
                    (
                        args.work / "models" / name / f"fold_{fold}" / "complete.json"
                    ).read_text()
                )
                _, _, classes = targets(
                    pred["class_t"], rec, summary["period_boundary_ms"]
                )
                np.add.at(class_cells, (classes, pred["class_prob"].argmax(axis=1)), 1)
        score = {
            f"crash_tolerance_{tolerance}ms": aggregate(rows[tolerance])
            for tolerance in (1, 2)
        }
        if name == "saw-hl3":
            score["three_class_confusion"] = class_cells.tolist()
            score["three_class_window_accuracy"] = float(
                class_cells.trace() / class_cells.sum()
            )
        review = pd.read_csv(REVIEW)
        expert = []
        for shot in split["expert_shots"]:
            predictions = [
                np.load(
                    args.work / "predictions" / name / f"fold_{fold}" / f"{shot}.npz"
                )
                for fold in range(3)
            ]
            t = predictions[0]["t"]
            present = np.mean([p["presence"] for p in predictions], axis=0)
            recs = review[review.shot == shot]
            positive = [
                (r.t_start / 1000, r.t_end / 1000)
                for r in recs.itertuples()
                if r.category == 1
            ]
            known = [
                (r.t_start / 1000, r.t_end / 1000)
                for r in recs.itertuples()
                if r.category in (0, 1)
            ]
            bins = t[10::20]
            valid = spans_at(bins, known)
            hist = score_histogram(
                spans_at(bins, positive)[valid], present[10::20][valid]
            )
            # Ensemble timing scores are only support checks because review has
            # no point annotations. Do not turn span support into crash precision.
            summaries = [
                json.loads(
                    (
                        args.work / "models" / name / f"fold_{f}" / "complete.json"
                    ).read_text()
                )
                for f in range(3)
            ]
            threshold = float(
                np.mean([s["selected_crash_threshold"]["threshold"] for s in summaries])
            )
            z = float(np.mean([s["selected_crash_threshold"]["z"] for s in summaries]))
            ensemble = {"t": t, "presence": present}
            if name == "saw-ours":
                ensemble["crash"] = np.mean([p["crash"] for p in predictions], axis=0)
            estimate = picks(
                name,
                ensemble,
                np.load(args.work / "signals" / f"{shot}.npz"),
                threshold,
                z,
            )
            assessed = spans_at(estimate, known)
            supported = spans_at(estimate, positive)
            expert.append(
                {
                    "shot": shot,
                    "cells": np.array(
                        [supported.sum(), (assessed & ~supported).sum(), 0]
                    ),
                    "histogram": hist,
                    "assessed_picks": int(assessed.sum()),
                    "supported_picks": int(supported.sum()),
                }
            )
        independent = aggregate(expert)
        independent["crash"] = {
            "recall": None,
            "precision": None,
            "f1": None,
            "span_supported_pick_fraction": sum(r["supported_picks"] for r in expert)
            / max(1, sum(r["assessed_picks"] for r in expert)),
            "assessed_picks": sum(r["assessed_picks"] for r in expert),
            "supported_picks": sum(r["supported_picks"] for r in expert),
        }
        independent["ci95"] = {
            k: v for k, v in independent["ci95"].items() if not k.startswith("crash_")
        }
        independent["note"] = (
            "Three-fold ensemble against reviewed spans; point crash truth is unavailable. Three shots give fragile bootstrap intervals."
        )
        score["expert"] = independent
        output["Tokamak-SI"][name] = score
    save_json(REPO / "outputs/labeler/sawtooth/benchmark.json", output)
    save_json(args.work / "benchmark.json", output)
    print(
        json.dumps(
            {
                name: {
                    "crash": score["crash_tolerance_2ms"]["crash"],
                    "presence": score["crash_tolerance_2ms"]["presence"],
                    "expert_presence": score["expert"]["presence"],
                }
                for name, score in output["Tokamak-SI"].items()
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["train", "evaluate"])
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument(
        "--folds", nargs="+", type=int, choices=(0, 1, 2), default=[0, 1, 2]
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--windows-per-shot", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.threads <= 8:
        parser.error("threads must be 1..8")
    (train if args.stage == "train" else evaluate)(args)


if __name__ == "__main__":
    main()
