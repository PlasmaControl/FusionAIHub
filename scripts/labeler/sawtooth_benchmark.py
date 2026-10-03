"""Shot-grouped saw-hl3 / saw-ours training, calibration, inference and scoring.

Only fixed cohort train shots enter three outer folds. Within each outer training
fold, a deterministic 20% inner selection split chooses early stopping and all
operating thresholds. Fixed val/test and expert shots never enter selection.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sawtooth_physics import OUTPUT, REPO, REVIEW, SEED, WORK, save_json
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from scipy.signal import find_peaks
from torch.nn import functional as F

from labeler.sawtooth.metrics import (
    aggregate,
    bin_times,
    binary_cells,
    bootstrap_classification,
    classification_metrics,
    event_cells,
    paired_bootstrap,
    point_metrics,
    presence_from_cells,
    score_histogram,
    spans_at,
)
from labeler.sawtooth.models import HL3, PhasePicker, soft_crash_target

MODELS = ("saw-hl3", "saw-ours")
BASELINES = ("saw-derivative", "saw-always-present")
WORK = WORK.parent / "fix3"
OUTPUT = OUTPUT.parent / "fix3"
INPUT_POLICY = (
    "observable-only inputs; loss-only assessment; balanced crash-centred; "
    "axis-selected single-channel timing with complete filter support; "
    "geometry outer rho .4-.65"
)
THRESHOLDS = np.r_[np.arange(1, 20) / 20, 0.975, 0.99, 0.995, 0.999]
DERIVATIVE_Z = (*range(2, 21), 25, 30, 40, 50, 75, 100, 150, 200, 300, 500, 1000)
LEGACY = {
    "source": (
        "OuYang et al. PPCF 67 (2025) 105004, "
        "doi:10.1088/1361-6587/ae0786; local digest"
    ),
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
    "limitation": (
        "Published three-regime window classification, no crash score or CIs; "
        "counts disagree with stated accuracy; split grouping unspecified."
    ),
}


def usable_signal_shots(work):
    """A failed refresh cannot re-admit its earlier cached signal/masks."""
    usable = set()
    for path in (work / "signals").glob("*.npz"):
        source = work / "shots" / f"{path.stem}.json"
        if source.exists() and not json.loads(source.read_text()).get("error"):
            usable.add(int(path.stem))
    return usable


def manifest(work):
    cohort = pd.read_csv(REPO / "data/events/catalog/cohort.csv")
    reviewed = {int(s) for s in pd.read_csv(REVIEW).shot}
    usable = usable_signal_shots(work)
    requested_train = set(cohort[cohort.split == "train"].shot) - reviewed
    requested_val = set(cohort[cohort.split == "val"].shot)
    train = sorted(requested_train & usable)
    shuffled = np.random.default_rng(SEED).permutation(train)
    folds = {int(s): i % 3 for i, s in enumerate(shuffled)}
    data = {
        "seed": SEED,
        "folds": folds,
        "training_cohort": train,
        "requested_training_cohort": sorted(requested_train),
        "fixed_validation_excluded": sorted(requested_val),
        "fixed_validation_nonexpert": sorted(requested_val - reviewed),
        "fixed_validation_supported": sorted(requested_val & usable - reviewed),
        "fixed_validation_available": sorted(requested_val & usable),
        "unavailable_training": [
            unavailable_record(work, shot) for shot in sorted(requested_train - usable)
        ],
        "unavailable_validation": [
            unavailable_record(work, shot) for shot in sorted(requested_val - usable)
        ],
        "inner_selection_fraction": 0.2,
        "expert_shots": sorted(reviewed & usable),
        "blind_test_excluded": sorted(cohort[cohort.split == "test"].shot),
        "unusable_train": sorted(set(cohort[cohort.split == "train"].shot) - usable),
        "unusable_val": sorted(set(cohort[cohort.split == "val"].shot) - usable),
        "all_reviewed_excluded": sorted(reviewed),
    }
    save_json(work / "split_manifest.json", data)
    save_json(OUTPUT / "split_manifest.json", data)
    return data


def record(work, shot):
    return json.loads((work / "shots" / f"{shot}.json").read_text())


def unavailable_record(work, shot):
    """Retain requested IDs without assigning outcomes to missing inputs."""
    path = work / "shots" / f"{shot}.json"
    source = json.loads(path.read_text()) if path.exists() else {}
    return {
        "shot": int(shot),
        "status": "unavailable",
        "outcome_status": "unknown",
        "exclusion_reason": source.get("error", "signal input unavailable"),
        "error_kind": source.get("error_kind", "signal_unavailable"),
        "source_record": str(path) if path.exists() else None,
        "observable_bins": 0,
        "assessed_bins": 0,
        "unassessed_bins": 0,
        "uncertain_bins": 0,
        "observable_picks": None,
        "assessed_picks": None,
        "excluded_picks": None,
        "presence": None,
        "presence_cells": None,
        "crash_tolerance_1ms": None,
        "crash_tolerance_2ms": None,
    }


def prediction_population(split, work, shots):
    """Keep the requested fixed-validation ledger separate from inference inputs."""
    requested = guard_prediction_shots(split, shots)
    available = set(split["fixed_validation_available"])
    supported = [shot for shot in requested if shot in available]
    unavailable = [
        unavailable_record(work, shot) for shot in requested if shot not in available
    ]
    return requested, supported, unavailable


def inner_split(outer_training, *, seed):
    """Keep whole shots together; fixed val/test never enter this split."""
    if len(outer_training) < 5:
        raise ValueError("at least five outer training shots are required")
    shuffled = np.random.default_rng(seed).permutation(sorted(outer_training))
    count = max(1, round(len(shuffled) * 0.2))
    return sorted(map(int, shuffled[count:])), sorted(map(int, shuffled[:count]))


def guard_partition(split, fitting, selection, heldout):
    """Fail before fitting/tuning if any fixed-val, test or expert shot enters."""
    allowed = set(split["training_cohort"])
    forbidden = set().union(
        split["fixed_validation_excluded"],
        split["blind_test_excluded"],
        split["all_reviewed_excluded"],
    )
    groups = [set(shots) for shots in (fitting, selection, heldout)]
    if any(group - allowed or group & forbidden for group in groups):
        raise ValueError("fitting, selection and heldout must use fixed train shots")
    if any(a & b for i, a in enumerate(groups) for b in groups[i + 1 :]):
        raise ValueError("fitting, selection and heldout shot groups overlap")


def guard_prediction_shots(split, shots):
    """Queue inference uses fixed validation shots and cannot load blind test."""
    if set(shots) - set(split["fixed_validation_excluded"]):
        raise ValueError("queue predictions require fixed validation shots")
    return sorted(set(shots))


def present_rows(rec, key):
    return [
        row
        for row in rec[key]
        if row.get("attrs", {}).get("state", row.get("state", "present")) == "present"
    ]


def masks_at(signal, times):
    """Sample explicit masks on the nearest native bin; out of range is unknown."""
    t = signal["t"]
    if "observable" not in signal or "assessed" not in signal:
        raise ValueError("signals require explicit observable and assessed masks")
    index = np.searchsorted(t, times)
    right = np.clip(index, 0, len(t) - 1)
    left = np.clip(index - 1, 0, len(t) - 1)
    nearest = np.where(np.abs(t[left] - times) <= np.abs(t[right] - times), left, right)
    inside = (np.asarray(times) >= t[0]) & (np.asarray(times) <= t[-1])
    observable = signal["observable"][nearest].astype(bool) & inside
    assessed = signal["assessed"][nearest].astype(bool) & observable
    return observable, assessed


def assessed_points(signal, times):
    """No point truth/prediction from an unknown or uncertain bin is scored."""
    times = np.asarray(times, dtype=float)
    return times[masks_at(signal, times)[1]]


def input_values(signal, name):
    """One observability-only input policy shared by fitting and inference."""
    values = signal["baseline" if name == "saw-hl3" else "y"].astype(np.float32).copy()
    observable = np.asarray(signal["observable"], dtype=bool)
    if observable.shape != (len(signal["t"]),):
        raise ValueError("observability must share the native input time grid")
    values[:, ~observable] = np.nan
    return values


def period_boundary(work, shots):
    periods = [
        r["attrs"]["period_ms"]
        for shot in shots
        for r in present_rows(record(work, shot), "intervals")
    ]
    if not periods:
        raise ValueError("training fold has no train periods")
    return float(np.median(periods))


def targets(t, rec, boundary):
    """Build point truth only from centers eligible on the supplied native grid."""
    intervals = present_rows(rec, "intervals")
    spans = [(r["start_s"], r["end_s"]) for r in intervals]
    presence = spans_at(t, spans).astype(np.float32)
    signal = {
        "t": t,
        "observable": spans_at(t, rec["observable_spans"]),
        "assessed": spans_at(t, rec["assessed_spans"]),
    }
    centers = assessed_points(
        signal, [r["time_s"] for r in present_rows(rec, "crashes")]
    )
    picks = soft_crash_target(t, centers)
    classes = np.zeros(len(t), dtype=np.int64)
    for r in intervals:
        classes[(t >= r["start_s"]) & (t < r["end_s"])] = 1 + int(
            r["attrs"]["period_ms"] > boundary
        )
    return picks, presence, classes


def sample_window_starts(t, classes, crashes, width, rng, *, count):
    """Half random, half crash-centred, balanced over available period classes."""
    random = rng.integers(0, len(t) - width + 1, size=count)
    centers = np.clip(
        np.searchsorted(t, crashes) - width // 2, 0, len(t) - width
    ).astype(int)
    groups = [centers[classes[centers + width // 2] == label] for label in (1, 2)]
    groups = [group for group in groups if len(group)]
    if not groups:
        return random
    positive_count = count // 2
    positive = [
        rng.choice(
            group,
            size=positive_count // len(groups) + int(i < positive_count % len(groups)),
            replace=True,
        )
        for i, group in enumerate(groups)
    ]
    return np.r_[random[: count - positive_count], *positive]


def windows(work, shots, model, boundary, *, per_shot=64, balanced=True):
    """Same random/crash-centred sampling and observable inputs for both models."""
    width = 200 if model == "saw-hl3" else 1000
    xs, ys, owners = [], [], []
    for shot in shots:
        data = np.load(work / "signals" / f"{shot}.npz")
        t = data["t"]
        x = input_values(data, model)
        if len(t) < width:
            continue
        rec = record(work, shot)
        pick, present, classes = targets(t, rec, boundary)
        observable, assessed = masks_at(data, t)
        rng = np.random.default_rng(SEED + shot)
        centers = assessed_points(
            data, [r["time_s"] for r in present_rows(rec, "crashes")]
        )
        starts = (
            sample_window_starts(t, classes, centers, width, rng, count=per_shot)
            if balanced
            else rng.integers(0, len(t) - width + 1, size=per_shot)
        )
        for start in starts:
            window = x[:, start : start + width].copy()
            mask = assessed[start : start + width]
            if observable[start : start + width].mean() < 0.8 or not mask.any():
                continue
            if model == "saw-hl3" and not mask[width // 2]:
                continue
            xs.append(window.astype(np.float16))
            ys.append(
                int(classes[start + width // 2])
                if model == "saw-hl3"
                else np.stack(
                    [pick[start : start + width], present[start : start + width], mask]
                ).astype(np.float16)
            )
            owners.append(shot)
    if not xs:
        raise ValueError(f"no assessed windows for {model} on {shots}")
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


def window_classification(model, x, y, mean, std, *, batch=128):
    """Diagnose the classifier on fitting/selection windows with dropout off."""
    device = next(model.parameters()).device
    model.eval()
    cells = np.zeros((3, 3), dtype=int)
    with torch.no_grad():
        for start in range(0, len(x), batch):
            logits = model(
                torch.from_numpy(normalize(x[start : start + batch], mean, std)).to(
                    device
                )
            )
            estimate = logits.argmax(dim=1).cpu().numpy()
            np.add.at(cells, (y[start : start + batch], estimate), 1)
    return classification_metrics(cells)


def prediction_classification(predictions, work, boundary):
    """Natural overlapping windows from a supplied fitting/selection shot set."""
    cells = np.zeros((3, 3), dtype=int)
    for shot, pred in predictions.items():
        _, _, truth = targets(pred["class_t"], record(work, shot), boundary)
        valid = masks_at(pred, pred["class_t"])[1]
        estimate = pred["class_prob"].argmax(axis=1)
        np.add.at(cells, (truth[valid], estimate[valid]), 1)
    return classification_metrics(cells)


def loss(model, logits, y, weights=None):
    if model == "saw-hl3":
        return F.cross_entropy(logits, y.long(), weight=weights)
    logp = F.log_softmax(logits[:, :2], dim=1)
    pick, present, mask = y[:, 0], y[:, 1], y[:, 2]
    denominator = mask.sum().clamp_min(1)
    point = -(mask * (20 * pick * logp[:, 1] + (1 - pick) * logp[:, 0])).sum()
    point /= denominator
    interval = (
        F.binary_cross_entropy_with_logits(logits[:, 2], present, reduction="none")
        * mask
    ).sum() / denominator
    return point + interval


def infer(model, name, signal, mean, std, *, batch=128):
    t = signal["t"]
    values = input_values(signal, name)
    observable, assessed = masks_at(signal, t)
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        if name == "saw-hl3":
            starts = np.arange(0, len(t) - 200 + 1, 20)
            chunks = []
            for offset in range(0, len(starts), batch):
                selected = starts[offset : offset + batch]
                x = values[:, selected[:, None] + np.arange(200)].transpose(1, 0, 2)
                output = (
                    model(torch.from_numpy(normalize(x, mean, std)).to(device))
                    .softmax(dim=1)
                    .cpu()
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
                "observable": observable,
                "assessed": assessed,
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
            logits = model(
                torch.from_numpy(normalize(np.stack(chunks), mean, std)).to(device)
            )
            output = (
                torch.stack(
                    [logits[:, :2].softmax(dim=1)[:, 1], logits[:, 2].sigmoid()], dim=1
                )
                .cpu()
                .numpy()
            )
            for start, p in zip(selected, output, strict=True):
                end = min(len(t), start + hop)
                probability[:, start:end] += p[:, margin : margin + end - start]
                count[start:end] += 1
        probability /= np.maximum(count, 1)
        return {
            "t": t,
            "crash": probability[0],
            "presence": probability[1],
            "observable": observable,
            "assessed": assessed,
        }


def ensemble_predictions(predictions):
    """Average frozen fold probabilities on identical native time/support grids."""
    first = predictions[0]
    grids = ("t", "observable", "assessed")
    if "class_t" in first:
        grids += ("class_t",)
    for prediction in predictions[1:]:
        if any(not np.array_equal(first[key], prediction[key]) for key in grids):
            raise ValueError(
                "fold predictions do not share the same time/support grids"
            )
    result = {key: first[key].copy() for key in grids}
    for key in ("presence", "crash", "class_prob"):
        if key in first:
            result[key] = np.mean([p[key] for p in predictions], axis=0).astype(
                np.float32
            )
    return result


def derivative_candidates(signal):
    """Single-channel derivative peaks and robust z values, reusable in tuning."""
    if "central_channel" not in signal or "y" not in signal:
        raise ValueError("derivative picking requires y and central_channel")
    center = int(np.asarray(signal["central_channel"]).item())
    y = np.asarray(signal["y"])
    if y.ndim != 2 or not 0 <= center < y.shape[0]:
        raise ValueError("central_channel must index a single ECE channel")
    observable = np.asarray(signal["observable"], bool)
    finite = observable & np.isfinite(y[center])
    if finite.sum() < 2:
        return np.array([], dtype=int), np.array([], dtype=float)
    # Interpolation prevents missing-bin boundaries from becoming derivative
    # edges. Only native observable finite peaks can enter the baseline.
    core = np.interp(signal["t"], signal["t"][finite], y[center, finite])
    derivative = -gaussian_filter1d(core, 2.5, order=1)
    complete = (
        maximum_filter1d((~finite).astype(np.uint8), 21, mode="constant", cval=1) == 0
    )
    values = derivative[complete]
    if not len(values):
        return np.array([], dtype=int), np.array([], dtype=float)
    scale = 1.4826 * np.median(np.abs(values - np.median(values)))
    peaks, _ = find_peaks(derivative, distance=50)
    peaks = peaks[complete[peaks]]
    return peaks, derivative[peaks] / max(scale, 1e-8)


def observable_points(signal, times):
    times = np.asarray(times, dtype=float)
    return times[masks_at(signal, times)[0]]


def pick_counts(signal, times):
    observable, assessed = masks_at(signal, times)
    return {
        "observable_picks": int(observable.sum()),
        "assessed_picks": int(assessed.sum()),
        "excluded_picks": int((observable & ~assessed).sum()),
    }


def derivative_prediction(signal, z, *, derivative=None):
    """Any single-channel >=z-sigma edge within +/-125 ms predicts presence."""
    peaks, amplitudes = (
        derivative_candidates(signal) if derivative is None else derivative
    )
    t = signal["t"]
    times = observable_points(signal, t[peaks[amplitudes >= z]])
    present = np.zeros(len(t), dtype=np.float32)
    if len(times):
        right = np.searchsorted(times, t)
        distance = np.minimum(
            np.abs(t - times[np.clip(right, 0, len(times) - 1)]),
            np.abs(t - times[np.clip(right - 1, 0, len(times) - 1)]),
        )
        present = (distance <= 0.125).astype(np.float32)
    return {
        "t": t,
        "presence": present,
        "picks": times,
        "observable": signal["observable"],
        "assessed": signal["assessed"],
    }


def load_signal(work, shot):
    with np.load(work / "signals" / f"{shot}.npz") as data:
        return {key: data[key] for key in data.files}


def derivative_classes(signal, times, z, boundary_ms, *, derivative=None):
    """Nearest edge's following gap, or preceding gap at an observable-run end.

    Presence without two supported edges is an abstention (-1), retained as an
    error in three-class metrics. A period never crosses unobservable support.
    """
    prediction = derivative_prediction(signal, z, derivative=derivative)
    times = np.asarray(times)
    presence = np.interp(times, signal["t"], prediction["presence"]) >= 0.5
    result = np.where(presence, -1, 0)
    edges = prediction["picks"]
    observable = np.asarray(signal["observable"], bool).copy()
    if "y" in signal and "central_channel" in signal:
        center = int(np.asarray(signal["central_channel"]).item())
        observable &= np.isfinite(signal["y"][center])
    boundaries = np.diff(np.r_[False, observable, False].astype(np.int8))
    dt = float(np.median(np.diff(signal["t"])))
    for start, stop in zip(
        np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1), strict=True
    ):
        lo, hi = signal["t"][start], signal["t"][stop - 1] + dt
        supported = edges[(edges >= lo) & (edges < hi)]
        selected = np.flatnonzero((times >= lo) & (times < hi) & presence)
        if len(supported) < 2 or not len(selected):
            continue
        right = np.searchsorted(supported, times[selected])
        before = np.clip(right - 1, 0, len(supported) - 1)
        after = np.clip(right, 0, len(supported) - 1)
        nearest = np.where(
            abs(times[selected] - supported[before])
            <= abs(times[selected] - supported[after]),
            before,
            after,
        )
        local = abs(times[selected] - supported[nearest]) <= 0.125
        gap = np.diff(supported)[np.minimum(nearest[local], len(supported) - 2)]
        result[selected[local]] = 1 + (gap * 1000 > boundary_ms)
    return result


def calibrate_derivative(work, selection_shots, *, split):
    """Select separate crash and presence z using only inner selection shots."""
    guard_partition(split, [], selection_shots, [])
    signals = {shot: load_signal(work, shot) for shot in selection_shots}
    derivatives = {
        shot: derivative_candidates(signal) for shot, signal in signals.items()
    }
    references, presence_targets = {}, {}
    for shot, signal in signals.items():
        rec = record(work, shot)
        references[shot] = assessed_points(
            signal, [row["time_s"] for row in present_rows(rec, "crashes")]
        )
        bins = bin_times((signal["t"][0], signal["t"][-1]))
        valid = masks_at(signal, bins)[1]
        truth = spans_at(
            bins,
            [(row["start_s"], row["end_s"]) for row in present_rows(rec, "intervals")],
        )
        presence_targets[shot] = bins[valid], truth[valid]
    options = []
    for z in DERIVATIVE_Z:
        cells, presence_cells = np.zeros(3), np.zeros(4)
        for shot, signal in signals.items():
            prediction = derivative_prediction(signal, z, derivative=derivatives[shot])
            cells += event_cells(
                references[shot], assessed_points(signal, prediction["picks"]), 2
            )
            bins, truth = presence_targets[shot]
            probability = np.interp(bins, prediction["t"], prediction["presence"])
            presence_cells += binary_cells(truth, probability >= 0.5)
        options.append(
            {
                "z": z,
                "cells": cells.astype(int).tolist(),
                **point_metrics(cells),
                "presence_cells": presence_cells.astype(int).tolist(),
                "presence": point_metrics(presence_cells[:3]),
            }
        )
    selected = max(options, key=lambda r: (-1 if r["f1"] is None else r["f1"], r["z"]))
    presence_selected = max(
        options,
        key=lambda r: (
            -1 if r["presence"]["f1"] is None else r["presence"]["f1"],
            r["z"],
        ),
    )
    return {
        **selected,
        "crash_z": selected["z"],
        "presence_z": presence_selected["z"],
        "presence_at_crash_z": selected["presence"],
        "presence_cells_at_crash_z": selected["presence_cells"],
        "presence": presence_selected["presence"],
        "presence_cells": presence_selected["presence_cells"],
        "selected_presence": presence_selected["presence"],
        "selected_presence_cells": presence_selected["presence_cells"],
        "selection_shots": selection_shots,
        "selection": (
            "independent inner selection crash F1 at +/-2 ms and presence F1; "
            "largest z breaks ties within each task"
        ),
        "presence_rule": "any >=inner-selected presence_z sigma edge within +/-125 ms",
        "input": "one raw ECE channel nearest the nominal EFIT magnetic axis",
        "options": options,
    }


def picks(name, prediction, signal, threshold, z=4, *, derivative=None):
    t = prediction["t"]
    if name == "saw-ours":
        peaks, _ = find_peaks(prediction["crash"], height=threshold, distance=50)
    else:
        # The published network has no timing head. Gate a *single-channel*
        # derivative picker with its predicted regime probability. No inversion,
        # q, coincidence or physics labels enter this inference step.
        peaks, amplitudes = (
            derivative_candidates(signal) if derivative is None else derivative
        )
        peaks = peaks[amplitudes >= z]
        peaks = peaks[prediction["presence"][peaks] >= threshold]
    return observable_points(signal, t[peaks])


def calibrate(name, predictions, work, *, selection_shots, split):
    """Select only on inner held-out training shots, never fixed val/expert."""
    if set(predictions) != set(selection_shots):
        raise ValueError("calibration predictions must be the inner selection shots")
    guard_partition(split, [], selection_shots, [])
    references, truth_by_shot = {}, {}
    signals = {}
    for shot, prediction in predictions.items():
        signals[shot] = load_signal(work, shot)
        rec = record(work, shot)
        references[shot] = assessed_points(
            signals[shot], [r["time_s"] for r in present_rows(rec, "crashes")]
        )
        t = bin_times((prediction["t"][0], prediction["t"][-1]))
        valid = masks_at(signals[shot], t)[1]
        truth = spans_at(
            t, [(r["start_s"], r["end_s"]) for r in present_rows(rec, "intervals")]
        )[valid]
        probability = np.interp(t, prediction["t"], prediction["presence"])[valid]
        truth_by_shot[shot] = truth, probability
    options = []
    derivatives = (
        {shot: derivative_candidates(signal) for shot, signal in signals.items()}
        if name == "saw-hl3"
        else {}
    )
    for threshold in THRESHOLDS:
        for z in DERIVATIVE_Z if name == "saw-hl3" else (0,):
            cells = np.zeros(3)
            for shot, prediction in predictions.items():
                estimate = picks(
                    name,
                    prediction,
                    signals[shot],
                    threshold,
                    z,
                    derivative=derivatives.get(shot),
                )
                cells += event_cells(
                    references[shot], assessed_points(signals[shot], estimate), 2
                )
            scores = point_metrics(cells)
            options.append(
                {
                    "threshold": float(threshold),
                    "z": z,
                    **scores,
                    "cells": cells.astype(int).tolist(),
                }
            )
    selected = max(
        options,
        key=lambda r: (-1 if r["f1"] is None else r["f1"], r["threshold"], r["z"]),
    )
    presence_options = []
    for threshold in THRESHOLDS:
        cells = sum(
            (
                binary_cells(truth, probability >= threshold)
                for truth, probability in truth_by_shot.values()
            ),
            start=np.zeros(4),
        )
        scores = point_metrics(cells[:3])
        presence_options.append(
            {
                "threshold": float(threshold),
                "cells": cells.astype(int).tolist(),
                **scores,
            }
        )
    best_presence = max(
        presence_options,
        key=lambda r: (-1 if r["f1"] is None else r["f1"], r["threshold"]),
    )
    selected["presence_threshold"] = best_presence["threshold"]
    selected["presence_options"] = presence_options
    return selected, options


def fit_configuration(
    args, name, fold, configuration, x, y, vx, vy, mean, std, class_weights, *, device
):
    """Early stopping uses only natural inner-selection windows for HL-3."""
    seed = SEED + fold
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = (
        HL3(dropout=configuration["dropout"]) if name == "saw-hl3" else PhasePicker()
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=configuration["learning_rate"],
        weight_decay=configuration["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
    history, best, best_epoch, stale = [], float("inf"), -1, 0
    batch = 128 if name == "saw-hl3" else 32
    epoch = 0
    best_state = None
    while stale < args.patience and epoch < args.max_epochs:
        epoch += 1
        model.train()
        losses = []
        fitting_cells = np.zeros((3, 3), dtype=int)
        for _ in range(args.steps):
            selected = rng.integers(0, len(x), size=batch)
            xx = normalize(x[selected], mean, std)
            yy = y[selected].astype(np.float32)
            if name == "saw-hl3":
                # Temporal shifts can change the regime near a boundary.
                xx *= rng.uniform(0.8, 1.2, size=(batch, 1, 1)).astype(np.float32)
                xx += rng.normal(0, 0.1, size=xx.shape).astype(np.float32)
            logits = model(torch.from_numpy(xx).to(device))
            value = loss(name, logits, torch.from_numpy(yy).to(device), class_weights)
            optimizer.zero_grad()
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(value.item())
            if name == "saw-hl3":
                np.add.at(
                    fitting_cells,
                    (y[selected], logits.detach().argmax(dim=1).cpu().numpy()),
                    1,
                )
        model.eval()
        vloss, vweights = [], []
        selection_cells = np.zeros((3, 3), dtype=int)
        with torch.no_grad():
            for start in range(0, len(vx), batch):
                xx = normalize(vx[start : start + batch], mean, std)
                yy = vy[start : start + batch].astype(np.float32)
                logits = model(torch.from_numpy(xx).to(device))
                # Select against the natural class distribution, without the
                # fitting weights, on the inner split only.
                vloss.append(loss(name, logits, torch.from_numpy(yy).to(device)).item())
                vweights.append(
                    float(len(yy)) if name == "saw-hl3" else float(yy[:, 2].sum())
                )
                if name == "saw-hl3":
                    np.add.at(
                        selection_cells,
                        (yy.astype(int), logits.argmax(dim=1).cpu().numpy()),
                        1,
                    )
        validation_loss = float(np.average(vloss, weights=vweights))
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_loss": validation_loss,
                "augmented_fit_classification": classification_metrics(fitting_cells)
                if name == "saw-hl3"
                else None,
                "natural_selection_classification": classification_metrics(
                    selection_cells
                )
                if name == "saw-hl3"
                else None,
            }
        )
        print(
            f"{name} fold {fold} {configuration} epoch {epoch}: "
            f"train {np.mean(losses):.4f} val {validation_loss:.4f}",
            flush=True,
        )
        if not np.isfinite(validation_loss):
            raise ValueError("nonfinite inner selection loss")
        if validation_loss < best - args.min_delta:
            best, best_epoch, stale = validation_loss, epoch, 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
        scheduler.step()
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    return best_state, {
        "configuration": configuration,
        "history": history,
        "best_epoch": best_epoch,
        "best_inner_loss": best,
        "stale_epochs": stale,
        "epochs_completed": epoch,
        "stopping_reason": "inner selection loss patience exhausted"
        if stale >= args.patience
        else "maximum epoch budget reached",
    }


def train(args):
    torch.set_num_threads(args.threads)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU training requires the phase3 CUDA Python environment")
    device = torch.device("cuda")
    # The shared head-node GPU budget is 10 GB; reserve at most 28% of a V100S.
    torch.cuda.set_per_process_memory_fraction(0.28)
    split = manifest(args.work)
    for name in args.models:
        for fold in args.folds:
            out = args.work / "models" / name / f"fold_{fold}"
            out.mkdir(parents=True, exist_ok=True)
            if (out / "complete.json").exists():
                previous = json.loads((out / "complete.json").read_text())
                if previous.get("input_policy") != INPUT_POLICY:
                    raise ValueError(
                        "completed fold uses obsolete input/sampling policy"
                    )
                print(f"resume: {name} fold {fold} complete", flush=True)
                continue
            seed = SEED + fold
            torch.manual_seed(seed)
            torch.cuda.reset_peak_memory_stats()
            outer_training = [
                s for s in split["training_cohort"] if split["folds"][s] != fold
            ]
            heldout = [s for s in split["training_cohort"] if split["folds"][s] == fold]
            training, validation = inner_split(outer_training, seed=seed)
            guard_partition(split, training, validation, heldout)
            boundary = period_boundary(args.work, training)
            x, y, owners = windows(
                args.work, training, name, boundary, per_shot=args.windows_per_shot
            )
            vx, vy, _ = windows(args.work, validation, name, boundary, per_shot=32)
            selection_windows = len(vx)
            mean, std = normalizer(x)
            class_weights = None
            if name == "saw-hl3":
                counts = np.bincount(y, minlength=3)
                _, natural_y, _ = windows(
                    args.work, training, name, boundary, per_shot=32, balanced=False
                )
                natural_counts = np.bincount(natural_y, minlength=3)
                majority_class = int(natural_counts.argmax())
                class_weights = torch.tensor(
                    len(y) / (3 * np.maximum(counts, 1)),
                    dtype=torch.float32,
                    device=device,
                )
            begun = time.monotonic()
            batch = 128 if name == "saw-hl3" else 32
            if name == "saw-hl3":
                sx, sy, _ = windows(
                    args.work,
                    validation,
                    name,
                    boundary,
                    per_shot=64,
                    balanced=False,
                )
                configurations = [
                    {"learning_rate": lr, "weight_decay": wd, "dropout": dropout}
                    for lr in (0.001, 0.0003)
                    for wd, dropout in ((0.003, 0.5), (0.0001, 0.2), (0.0, 0.0))
                ]
            else:
                sx, sy = vx, vy
                configurations = [
                    {"learning_rate": 0.001, "weight_decay": 0.0001, "dropout": None}
                ]
            candidates, best_state, best_candidate = [], None, None
            for configuration in configurations:
                state, candidate = fit_configuration(
                    args,
                    name,
                    fold,
                    configuration,
                    x,
                    y,
                    sx,
                    sy,
                    mean,
                    std,
                    class_weights,
                    device=device,
                )
                candidates.append(candidate)
                if best_candidate is None or (
                    candidate["best_inner_loss"] < best_candidate["best_inner_loss"]
                ):
                    best_state, best_candidate = state, candidate
                del state
            configuration = best_candidate["configuration"]
            model = (
                HL3(dropout=configuration["dropout"])
                if name == "saw-hl3"
                else PhasePicker()
            ).to(device)
            model.load_state_dict(best_state)
            history = best_candidate["history"]
            best_epoch = best_candidate["best_epoch"]
            stale = best_candidate["stale_epochs"]
            epoch = best_candidate["epochs_completed"]
            fit_classification = (
                window_classification(model, x, y, mean, std)
                if name == "saw-hl3"
                else None
            )
            balanced_selection_classification = (
                window_classification(model, vx, vy, mean, std)
                if name == "saw-hl3"
                else None
            )
            torch.save(
                {
                    "state_dict": best_state,
                    "mean": mean,
                    "std": std,
                    "boundary_ms": boundary,
                    "training_shots": training,
                    "validation_shots": validation,
                    "selection_shots": validation,
                    "majority_class": majority_class if name == "saw-hl3" else None,
                    "model": name,
                    "input_policy": INPUT_POLICY,
                    "configuration": configuration,
                },
                out / "checkpoint.pt",
            )
            del x, y, vx, vy, sx, sy
            validation_predictions = {
                s: infer(
                    model, name, np.load(args.work / "signals" / f"{s}.npz"), mean, std
                )
                for s in validation
            }
            selected, options = calibrate(
                name,
                validation_predictions,
                args.work,
                selection_shots=validation,
                split=split,
            )
            pred_dir = args.work / "predictions" / name / f"fold_{fold}"
            pred_dir.mkdir(parents=True, exist_ok=True)
            for shot in (
                heldout + split["expert_shots"] + split["fixed_validation_supported"]
            ):
                signal = np.load(args.work / "signals" / f"{shot}.npz")
                prediction = infer(model, name, signal, mean, std)
                prediction["picks"] = picks(
                    name, prediction, signal, selected["threshold"], selected["z"]
                )
                np.savez_compressed(pred_dir / f"{shot}.npz", **prediction)
            summary = {
                "model": name,
                "input_policy": INPUT_POLICY,
                "fold": fold,
                "seed": seed,
                "label_work_root": str(args.work),
                "label_freeze_json": str(args.work / "freeze.json"),
                "training_shots": training,
                "heldout_shots": heldout,
                "validation_shots": validation,
                "selection_shots": validation,
                "outer_training_shots": outer_training,
                "expert_shots": split["expert_shots"],
                "fixed_validation_nonexpert": split["fixed_validation_nonexpert"],
                "fixed_validation_supported": split["fixed_validation_supported"],
                "training_windows": len(owners),
                "selection_windows": selection_windows,
                "training_class_counts": counts.tolist() if name == "saw-hl3" else None,
                "natural_training_class_counts": (
                    natural_counts.tolist() if name == "saw-hl3" else None
                ),
                "majority_class": majority_class if name == "saw-hl3" else None,
                "fit_classification": fit_classification,
                "balanced_selection_classification": balanced_selection_classification,
                "natural_selection_classification": (
                    prediction_classification(
                        validation_predictions, args.work, boundary
                    )
                    if name == "saw-hl3"
                    else None
                ),
                "diagnosis_scope": "fitting and inner-selection fixed-train shots only",
                "period_boundary_ms": boundary,
                "history": history,
                "best_epoch": best_epoch,
                "hyperparameter_candidates": candidates,
                "selected_configuration": configuration,
                "hyperparameter_selection": (
                    "minimum unweighted cross entropy on natural random windows "
                    "from inner selection shots only"
                    if name == "saw-hl3"
                    else "fixed optimizer; inner selection masked loss stopping"
                ),
                "derivative_baseline": calibrate_derivative(
                    args.work, validation, split=split
                ),
                "selected_crash_threshold": selected,
                "threshold_candidates": options,
                "presence_threshold": selected["presence_threshold"],
                "batch_size": batch,
                "stopping_reason": best_candidate["stopping_reason"],
                "maximum_epochs": args.max_epochs,
                "early_stopping_patience": args.patience,
                "early_stopping_min_delta": args.min_delta,
                "stale_epochs": stale,
                "epochs_completed": epoch,
                "steps_per_epoch": args.steps,
                "cpu_threads": args.threads,
                "device": str(device),
                "device_name": torch.cuda.get_device_name(),
                "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
                "threshold_grid": THRESHOLDS.tolist(),
                "derivative_z_grid": list(DERIVATIVE_Z) if name == "saw-hl3" else [],
                "selected_derivative_z_at_grid_edge": (
                    selected["z"] in (min(DERIVATIVE_Z), max(DERIVATIVE_Z))
                    if name == "saw-hl3"
                    else None
                ),
                "window_sampling": (
                    "half natural random, half crash-centred; crash-centred samples "
                    "balance the available small/large period classes per shot; "
                    "identical policy for both models"
                ),
                "input_channels": (
                    [
                        "ECE core mean: nominal EFIT magnetic-axis channel band",
                        "ECE outer mean: nominal LFS rho .4-.65 channel band",
                        "Mirnov mean channels 0-1",
                        "Ip",
                    ]
                    if name == "saw-hl3"
                    else [f"ECE channel {i}" for i in range(48)]
                ),
                "unknown_truth": (
                    "observability alone masks inputs; assessed supervision/scoring "
                    "excludes uncertain bins while keeping observable measurements"
                ),
                "validation_status": "unvalidated algorithmic supervision",
                "seconds": round(time.monotonic() - begun, 1),
                "parameters": sum(p.numel() for p in model.parameters()),
            }
            save_json(out / "complete.json", summary)
            save_json(OUTPUT / f"{name}_fold_{fold}.json", summary)
            del model, best_state, validation_predictions
            torch.cuda.empty_cache()


def baseline_prediction(name, signal, z, presence_z=None):
    if name == "saw-derivative":
        derivative = derivative_candidates(signal)
        prediction = derivative_prediction(signal, z, derivative=derivative)
        if presence_z is not None:
            prediction["presence"] = derivative_prediction(
                signal, presence_z, derivative=derivative
            )["presence"]
        return prediction
    return {
        **{key: signal[key] for key in ("t", "observable", "assessed")},
        "presence": np.ones(len(signal["t"]), dtype=np.float32),
        "picks": np.array([], dtype=float),
    }


def score_prediction(shot, name, pred, signal, rec, threshold):
    """Score conditional teacher agreement, retaining all observable picks."""
    t = bin_times((pred["t"][0], pred["t"][-1]))
    observable, assessed = masks_at(signal, t)
    truth = spans_at(
        t, [(r["start_s"], r["end_s"]) for r in present_rows(rec, "intervals")]
    )
    probability = np.interp(t, pred["t"], pred["presence"])
    hist = score_histogram(truth[assessed], probability[assessed])
    presence_cells = binary_cells(truth[assessed], probability[assessed] >= threshold)
    ref = assessed_points(signal, [r["time_s"] for r in present_rows(rec, "crashes")])
    estimated = assessed_points(signal, pred["picks"])
    result = {
        "shot": shot,
        "status": "supported",
        "outcome_status": "assessed" if assessed.any() else "unknown",
        "assessed_bins": int(assessed.sum()),
        "observable_bins": int(observable.sum()),
        "unassessed_bins": int((~observable).sum()),
        "uncertain_bins": int((observable & ~assessed).sum()),
        "presence_threshold": threshold,
        "presence_cells": presence_cells.astype(int).tolist(),
        "presence": presence_from_cells(hist, presence_cells, threshold),
        **pick_counts(signal, pred["picks"]),
    }
    rows = {}
    for tolerance in (1, 2):
        cells = (
            event_cells(ref, estimated, tolerance)
            if name != "saw-always-present"
            else None
        )
        rows[tolerance] = {
            "shot": shot,
            "cells": cells,
            "histogram": hist,
            "presence_cells": presence_cells,
        }
        result[f"crash_tolerance_{tolerance}ms"] = (
            point_metrics(cells) if cells is not None else None
        )
    return rows, result


def assessment_totals(by_shot):
    totals = {
        key: sum(row[key] for row in by_shot if row[key] is not None)
        for key in (
            "assessed_bins",
            "observable_bins",
            "unassessed_bins",
            "uncertain_bins",
            "observable_picks",
            "assessed_picks",
            "excluded_picks",
        )
    }
    totals.update(
        assessed_shots=sum(row["assessed_bins"] > 0 for row in by_shot),
        observable_shots=sum(row["observable_bins"] > 0 for row in by_shot),
        both_class_shots=sum(
            row["presence"] is not None
            and row["presence"]["positive_bins"] > 0
            and row["presence"]["negative_bins"] > 0
            for row in by_shot
        ),
    )
    return totals


def coverage(by_shot):
    """Coverage counts are independent of the conditional outcome metrics."""
    supported = [row for row in by_shot if row.get("status") != "unavailable"]
    scored = [row for row in supported if row["assessed_bins"] > 0]
    unavailable = [row for row in by_shot if row.get("status") == "unavailable"]
    return {
        "requested_shots": len(by_shot),
        "supported_shots": len(supported),
        "scored_shots": len(scored),
        "observable_shots": sum(row["observable_bins"] > 0 for row in supported),
        "requested_shot_ids": sorted(row["shot"] for row in by_shot),
        "supported_shot_ids": sorted(row["shot"] for row in supported),
        "scored_shot_ids": sorted(row["shot"] for row in scored),
        "unavailable": unavailable,
        "score_support": "conditional outcomes only for assessed, supported inputs",
    }


def frozen_operating_points(name, summaries):
    """Frozen fold means, with no fixed-validation or expert label access."""
    if name in BASELINES:
        return (
            0.5,
            0.0,
            float(
                np.mean([summary["derivative_baseline"]["z"] for summary in summaries])
            ),
        )
    return (
        float(np.mean([summary["presence_threshold"] for summary in summaries])),
        float(
            np.mean(
                [
                    summary["selected_crash_threshold"]["threshold"]
                    for summary in summaries
                ]
            )
        ),
        float(
            np.mean([summary["selected_crash_threshold"]["z"] for summary in summaries])
        ),
    )


def heldout_prediction(work, name, shot, summaries):
    """Three frozen folds, plus matching baseline inputs on the same shot."""
    signal = load_signal(work, shot)
    presence_threshold, threshold, z = frozen_operating_points(name, summaries)
    if name in BASELINES:
        presence_z = float(
            np.mean(
                [summary["derivative_baseline"]["presence_z"] for summary in summaries]
            )
        )
        prediction = baseline_prediction(name, signal, z, presence_z)
    else:
        predictions = []
        for fold in range(3):
            path = work / "predictions" / name / f"fold_{fold}" / f"{shot}.npz"
            with np.load(path) as data:
                predictions.append({key: data[key] for key in data.files})
        prediction = ensemble_predictions(predictions)
        prediction["picks"] = picks(name, prediction, signal, threshold, z)
    return prediction, signal, presence_threshold


def evaluate(args):
    split = manifest(args.work)
    output = {
        "legacy": LEGACY,
        "Tokamak-SI": {},
        "protocol": {
            "primary": ("conditional agreement with the physics rule on assessed bins"),
            "split_design": (
                "three-fold out-of-fold on fixed train; 20% whole-shot inner "
                "selection from each outer training fold; fixed-validation "
                "nonexpert shots form a second held-out set"
            ),
            "bin_ms": 2,
            "interval_threshold": (
                "selected independently per fold on inner selection shots, "
                "grid 0.05..0.95 step 0.05 plus 0.975, 0.99, 0.995, 0.999"
            ),
            "crash_tolerances_ms": [1, 2],
            "fixed_validation_split": (
                f"{len(split['fixed_validation_nonexpert'])} requested nonexpert "
                f"fixed-validation shots; {len(split['fixed_validation_supported'])} "
                "supported inputs; frozen fold ensemble, excluded from fitting "
                "and selection; missing inputs retain unknown outcomes"
            ),
            "test_split": "excluded, never loaded",
            "expert": (
                "reviewed shots excluded from fitting and selection; "
                "exploratory per-shot results without bootstrap; rule revisions "
                "already used their reviewed scores, so these are not untouched"
            ),
            "bootstrap": (
                "1000 whole-shot resamples for OOF and supported fixed-validation "
                "shots, seed 20261003; paired identical draws versus derivative-only; "
                "512 probability bins for AUROC/AP; unavailable inputs are excluded"
            ),
            "gaussian_target_sigma_ms": 0.5,
            "assessment": (
                "OOF pseudolabel scores intersect observable & assessed. "
                "Exploratory review scores intersect anchored reviewed spans & "
                "observable; spans were drawn while reviewing old rule suggestions. "
                "conditional algorithm-assessed scores also reported."
            ),
            "input_parity": (
                "saw-hl3: nominal EFIT-axis core and LFS rho .4-.65 ECE averages, "
                "Mirnov mean 0-1 and Ip; "
                "saw-ours: all 48 ECE channels. Adapted HL-3 ECE inputs replace "
                "published SXR; architecture and input access differ. "
                "Derivative timing: one axis-selected ECE channel, shared by "
                "derivative-only and derivative picker gated by HL-3."
            ),
            "trivial_baselines": {
                "saw-derivative": (
                    "single axis-selected ECE channel; z selected on inner "
                    "crash F1; separate presence_z selected on inner presence F1; "
                    "any >=presence_z sigma edge within +/-125 ms gives presence"
                ),
                "saw-always-present": "presence=1 everywhere; crash score undefined",
            },
            "paired_comparison": (
                "model minus derivative-only; 1000 identical whole-shot "
                "bootstrap draws, seed 20261003, for each held-out set"
            ),
            "inputs": (
                "one observability-only masking policy in fitting and inference; "
                "assessment masks affect supervision and scoring only"
            ),
            "sampling": (
                "both models: half natural random windows, half crash-centred "
                "windows balanced over available period classes per fitting shot"
            ),
            "validation_status": (
                "unvalidated algorithmic labels; reviewed spans are exploratory, "
                "anchored to prior rule suggestions, and have no point crash truth"
            ),
        },
    }
    review = pd.read_csv(REVIEW)
    paired_rows = {"out_of_fold": {}, "fixed_validation": {}}
    derivative_classification = None
    for name in list(args.models) + list(BASELINES):
        summary_model = name if name in MODELS else args.models[0]
        summaries = [
            json.loads(
                (
                    args.work
                    / "models"
                    / summary_model
                    / f"fold_{fold}"
                    / "complete.json"
                ).read_text()
            )
            for fold in range(3)
        ]
        for summary in summaries:
            if summary.get("input_policy") != INPUT_POLICY:
                raise ValueError("evaluation requires the current input/timing policy")
            guard_partition(
                split,
                summary["training_shots"],
                summary["selection_shots"],
                summary["heldout_shots"],
            )
        rows = {1: [], 2: []}
        shot_class_cells, derivative_class_cells, by_shot = [], [], []
        for shot in split["training_cohort"]:
            fold = split["folds"][shot]
            summary = summaries[fold]
            signal = load_signal(args.work, shot)
            if name in BASELINES:
                pred = baseline_prediction(
                    name,
                    signal,
                    summary["derivative_baseline"]["z"],
                    summary["derivative_baseline"]["presence_z"],
                )
                presence_threshold = 0.5
            else:
                path = args.work / "predictions" / name / f"fold_{fold}" / f"{shot}.npz"
                with np.load(path) as data:
                    pred = {key: data[key] for key in data.files}
                presence_threshold = summary["presence_threshold"]
            shot_rows, shot_result = score_prediction(
                shot, name, pred, signal, record(args.work, shot), presence_threshold
            )
            shot_result["fold"] = fold
            for tolerance in (1, 2):
                rows[tolerance].append(shot_rows[tolerance])
            by_shot.append(shot_result)
            rec = record(args.work, shot)
            if name == "saw-hl3":
                _, _, classes = targets(
                    pred["class_t"], rec, summary["period_boundary_ms"]
                )
                valid = masks_at(signal, pred["class_t"])[1]
                cells = np.zeros((3, 3), dtype=int)
                np.add.at(
                    cells, (classes[valid], pred["class_prob"].argmax(axis=1)[valid]), 1
                )
                majority_cells = np.zeros((3, 3), dtype=int)
                majority_cells[:, summary["majority_class"]] = cells.sum(axis=1)
                shot_class_cells.append(
                    {"shot": shot, "cells": cells, "majority_cells": majority_cells}
                )
                shot_result["three_class"] = classification_metrics(cells)
                derivative_estimate = derivative_classes(
                    signal,
                    pred["class_t"],
                    summary["derivative_baseline"]["presence_z"],
                    summary["period_boundary_ms"],
                )
                derivative_cells = np.zeros((3, 3), dtype=int)
                classified = valid & (derivative_estimate >= 0)
                np.add.at(
                    derivative_cells,
                    (classes[classified], derivative_estimate[classified]),
                    1,
                )
                unclassified = np.bincount(
                    classes[valid & (derivative_estimate < 0)], minlength=3
                )
                derivative_class_cells.append(
                    {
                        "shot": shot,
                        "cells": derivative_cells,
                        "unclassified": unclassified,
                        "majority_cells": majority_cells,
                    }
                )
                shot_result["three_class_derivative_baseline"] = classification_metrics(
                    derivative_cells, unclassified=unclassified
                )
        score = {
            f"crash_tolerance_{tolerance}ms": aggregate(rows[tolerance])
            for tolerance in (1, 2)
        }
        by_shot.extend(split["unavailable_training"])
        by_shot.sort(key=lambda row: row["shot"])
        score["by_shot"] = by_shot
        score["assessment_totals"] = assessment_totals(by_shot)
        score["coverage"] = coverage(by_shot)
        score["score_label"] = output["protocol"]["primary"]
        score["crash_metric_label"] = (
            "derivative picker gated by HL-3"
            if name == "saw-hl3"
            else "single-channel derivative picker"
            if name == "saw-derivative"
            else "no crash prediction"
            if name == "saw-always-present"
            else "PhaseNet-style crash picker"
        )
        if name == "saw-derivative":
            score["operating_points_by_fold"] = [
                {
                    "fold": fold,
                    "crash_z": summary["derivative_baseline"]["z"],
                    "presence_z": summary["derivative_baseline"]["presence_z"],
                    "selection_shots": summary["selection_shots"],
                }
                for fold, summary in enumerate(summaries)
            ]
        paired_rows["out_of_fold"][name] = rows
        if name == "saw-hl3":
            classification = bootstrap_classification(shot_class_cells)
            derivative_classification = bootstrap_classification(derivative_class_cells)
            derivative_classification["policy"] = (
                "identical assessed HL3 window centers; presence any inner-selected "
                "presence_z edge within +/-125ms; nearest edge following gap, preceding gap "
                "at observable-run end, versus fitting-shot period boundary; "
                "period-undefined abstentions count as errors on original support"
            )
            score["three_class"] = classification
            score["three_class_derivative_baseline"] = derivative_classification
            score["three_class_confusion"] = classification["confusion"]
            score["three_class_per_class_recall"] = classification["per_class_recall"]
            score["three_class_recall_ci95"] = classification["ci95"][
                "per_class_recall"
            ]
            score["three_class_window_accuracy"] = classification["window_accuracy"]
            score["three_class_accuracy_ci95"] = classification["ci95"][
                "window_accuracy"
            ]
            score["three_class_macro_f1"] = classification["macro_f1"]
            score["three_class_macro_f1_ci95"] = classification["ci95"]["macro_f1"]
            majority = classification["majority_baseline"]
            majority["accuracy"] = majority["window_accuracy"]
            majority["accuracy_ci95"] = majority["ci95"]["window_accuracy"]
            majority["class_by_fold"] = [s["majority_class"] for s in summaries]
            score["three_class_majority_baseline"] = majority
            observed_class = int(np.argmax(classification["class_support"]))
            observed_rows = []
            for row in shot_class_cells:
                cells = np.zeros((3, 3), dtype=int)
                cells[:, observed_class] = row["cells"].sum(axis=1)
                observed_rows.append({**row, "majority_cells": cells})
            observed_majority = bootstrap_classification(observed_rows)[
                "majority_baseline"
            ]
            observed_majority["class"] = observed_class
            observed_majority["accuracy"] = observed_majority["window_accuracy"]
            observed_majority["accuracy_ci95"] = observed_majority["ci95"][
                "window_accuracy"
            ]
            observed_majority["selection"] = (
                "argmax of pooled OOF true class support; descriptive class imbalance, "
                "not a fitted or deployable baseline; no val/test or expert scores used"
            )
            observed_majority["bootstrap_class_policy"] = (
                "pooled OOF class chosen once and fixed through all 1000 shot resamples"
            )
            score["three_class_observed_majority_baseline"] = observed_majority
            score["three_class_window_protocol"] = {
                "classes": [
                    "no sawtooth",
                    "small-period sawtooth",
                    "large-period sawtooth",
                ],
                "class_boundaries_ms_by_fold": [
                    s["period_boundary_ms"] for s in summaries
                ],
                "window_ms": 20,
                "hop_ms": 2,
                "scoring_support": "observable & algorithm-assessed window centers",
                "class_selection": "period median on fitting shots, per fold",
            }
        elif derivative_classification is not None and name in BASELINES:
            if name == "saw-derivative":
                score["three_class"] = derivative_classification
            else:
                score["three_class"] = {
                    "windows": derivative_classification["windows"],
                    "class_support": derivative_classification["class_support"],
                    "window_accuracy": None,
                    "macro_f1": None,
                    "per_class_recall": [None, None, None],
                    "reason": "always-present provides no period class; not applicable",
                }
        fixed_rows = {1: [], 2: []}
        fixed_by_shot = []
        for shot in split["fixed_validation_nonexpert"]:
            if shot not in split["fixed_validation_supported"]:
                fixed_by_shot.append(unavailable_record(args.work, shot))
                continue
            prediction, signal, presence_threshold = heldout_prediction(
                args.work, name, shot, summaries
            )
            shot_rows, shot_result = score_prediction(
                shot,
                name,
                prediction,
                signal,
                record(args.work, shot),
                presence_threshold,
            )
            fixed_by_shot.append(shot_result)
            for tolerance in (1, 2):
                fixed_rows[tolerance].append(shot_rows[tolerance])
        score["fixed_validation"] = {
            **{
                f"crash_tolerance_{tolerance}ms": aggregate(fixed_rows[tolerance])
                for tolerance in (1, 2)
            },
            "by_shot": fixed_by_shot,
            "assessment_totals": assessment_totals(fixed_by_shot),
            "coverage": coverage(fixed_by_shot),
            "score_label": output["protocol"]["primary"],
            "protocol": (
                "second held-out set: nonexpert fixed validation; three frozen "
                "fold ensembles; means of inner-selected operating thresholds; "
                "each task's derivative z is its mean of inner selections, never retuned"
            ),
        }
        if name == "saw-derivative":
            score["fixed_validation"]["derivative_operating_points"] = {
                key: float(
                    np.mean([row[key] for row in score["operating_points_by_fold"]])
                )
                for key in ("crash_z", "presence_z")
            }
        paired_rows["fixed_validation"][name] = fixed_rows
        expert, expert_hist, expert_cells = [], [], []
        conditional_hist, conditional_cells = [], []
        for shot in split["expert_shots"]:
            ensemble, signal, presence_threshold = heldout_prediction(
                args.work, name, shot, summaries
            )
            t = ensemble["t"]
            present = ensemble["presence"]
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
            bins = bin_times(
                (
                    max(t[0], recs.t_start.min() / 1000),
                    min(t[-1], recs.t_end.max() / 1000),
                )
            )
            observable, assessed = masks_at(signal, bins)
            valid = spans_at(bins, known) & observable
            conditional_valid = spans_at(bins, known) & assessed
            truth = spans_at(bins, positive)[valid]
            probability = np.interp(bins, t, present)[valid]
            hist = score_histogram(truth, probability)
            cells = binary_cells(truth, probability >= presence_threshold)
            expert_hist.append(hist)
            expert_cells.append(cells)
            conditional_truth = spans_at(bins, positive)[conditional_valid]
            conditional_probability = np.interp(bins, t, present)[conditional_valid]
            conditional_histogram = score_histogram(
                conditional_truth, conditional_probability
            )
            conditional_confusion = binary_cells(
                conditional_truth, conditional_probability >= presence_threshold
            )
            conditional_hist.append(conditional_histogram)
            conditional_cells.append(conditional_confusion)
            estimate = ensemble["picks"]
            known_picks = spans_at(estimate, known)
            supported = spans_at(estimate, positive) & known_picks
            # Span review provides no point timing annotations.
            metrics = presence_from_cells(hist, cells, presence_threshold)
            expert.append(
                {
                    "shot": shot,
                    "presence": metrics,
                    "presence_cells": cells.astype(int).tolist(),
                    "assessed_bins": int(valid.sum()),
                    "conditional_assessed_presence": presence_from_cells(
                        conditional_histogram,
                        conditional_confusion,
                        presence_threshold,
                    ),
                    "conditional_assessed_presence_cells": conditional_confusion.astype(
                        int
                    ).tolist(),
                    "algorithm_assessed_bins": int(conditional_valid.sum()),
                    "observable_reviewed_bins": int(
                        (spans_at(bins, known) & observable).sum()
                    ),
                    "excluded_reviewed_bins": int(
                        (spans_at(bins, known) & ~assessed).sum()
                    ),
                    "uncertain_positive_bins": int(
                        (spans_at(bins, positive) & observable & ~assessed).sum()
                    ),
                    "uncertain_negative_bins": int(
                        (
                            spans_at(bins, known)
                            & ~spans_at(bins, positive)
                            & observable
                            & ~assessed
                        ).sum()
                    ),
                    "unobservable_positive_bins": int(
                        (spans_at(bins, positive) & ~observable).sum()
                    ),
                    "algorithm_assessed_positive_bins": int(conditional_truth.sum()),
                    "expert_positive_observable_bins": int(
                        (spans_at(bins, positive) & observable).sum()
                    ),
                    "assessment_mask": "known expert spans & observable",
                    "conditional_assessment_mask": (
                        "known expert spans & observable & assessed"
                    ),
                    "observable_picks": len(estimate),
                    "assessed_picks": int(known_picks.sum()),
                    "excluded_picks": int((~known_picks).sum()),
                    "physics_rule_assessed_picks": pick_counts(signal, estimate)[
                        "assessed_picks"
                    ],
                    "physics_rule_excluded_picks": pick_counts(signal, estimate)[
                        "excluded_picks"
                    ],
                    "supported_picks": int(supported.sum()),
                    "span_supported_pick_fraction": float(
                        supported.sum() / known_picks.sum()
                    )
                    if known_picks.any()
                    else None,
                    "crash_precision": None,
                    "crash_recall": None,
                }
            )
        total_hist = np.stack(expert_hist).sum(axis=0)
        total_cells = np.stack(expert_cells).sum(axis=0)
        pooled = presence_from_cells(total_hist, total_cells, presence_threshold)
        supported = sum(r["supported_picks"] for r in expert)
        assessed = sum(r["assessed_picks"] for r in expert)
        score["expert"] = {
            "shots": len(expert),
            "by_shot": expert,
            "presence": pooled,
            "conditional_assessed_presence": presence_from_cells(
                np.stack(conditional_hist).sum(axis=0),
                np.stack(conditional_cells).sum(axis=0),
                presence_threshold,
            ),
            "presence_threshold": presence_threshold,
            "pick_totals": {
                key: sum(row[key] for row in expert)
                for key in (
                    "observable_picks",
                    "assessed_picks",
                    "excluded_picks",
                    "physics_rule_assessed_picks",
                    "physics_rule_excluded_picks",
                )
            },
            "crash": {
                "recall": None,
                "precision": None,
                "f1": None,
                "span_supported_pick_fraction": supported / assessed
                if assessed
                else None,
                "assessed_picks": assessed,
                "supported_picks": supported,
            },
            "note": (
                "Three-fold ensemble; thresholds are means of inner selections. "
                "Per-shot scores without bootstrap because n=3. "
                "Exploratory spans were anchored to prior rule suggestions; "
                "rule revisions already used their scores, so not untouched. "
                "no independent point crash times or spatial validation."
            ),
        }
        if name == "saw-hl3":
            score["expert"]["inverted_auroc_shots"] = [
                row["shot"]
                for row in expert
                if row["presence"]["auroc"] is not None
                and row["presence"]["auroc"] < 0.5
            ]
            score["expert"]["auc_diagnosis"] = (
                "AUROC below 0.5 persists on listed exploratory expert shots; "
                "no expert-based inversion, retuning or threshold selection performed"
                if score["expert"]["inverted_auroc_shots"]
                else "No exploratory expert shot with defined AUROC below 0.5; "
                "all operating points were frozen on inner selection shots"
            )
        if name == "saw-always-present":
            score["expert"]["crash"] = None
        output["Tokamak-SI"][name] = score
    for group, group_rows in paired_rows.items():
        for name in list(args.models) + ["saw-always-present"]:
            comparison = {
                f"crash_tolerance_{tolerance}ms": paired_bootstrap(
                    group_rows[name][tolerance],
                    group_rows["saw-derivative"][tolerance],
                )
                for tolerance in (1, 2)
            }
            target = output["Tokamak-SI"][name]
            if group == "fixed_validation":
                target = target["fixed_validation"]
            target["paired_vs_derivative"] = comparison
    save_json(args.work / "benchmark.json", output)
    summary = dict(output)
    summary["Tokamak-SI"] = {}
    for name, result in output["Tokamak-SI"].items():
        summary["Tokamak-SI"][name] = {
            **{key: value for key, value in result.items() if key != "by_shot"},
            "by_shot_count": len(result["by_shot"]),
            "out_of_fold_shots": result["coverage"]["supported_shot_ids"],
            "by_shot_details": str(args.work / "benchmark.json"),
        }
    save_json(OUTPUT / "benchmark.json", summary)
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


def predict(args):
    """Export fixed-validation ensembles without fitting or threshold selection."""
    torch.set_num_threads(args.threads)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU inference requires the phase3 CUDA Python environment")
    torch.cuda.set_per_process_memory_fraction(0.28)
    split = manifest(args.work)
    requested, shots, unavailable = prediction_population(
        split, args.work, args.prediction_shots or split["fixed_validation_excluded"]
    )
    if not requested:
        raise ValueError("predict requires at least one --prediction-shots entry")
    device = torch.device("cuda")
    for name in args.models:
        summaries, predictions = [], {shot: [] for shot in shots}
        for fold in range(3):
            directory = args.work / "models" / name / f"fold_{fold}"
            summary = json.loads((directory / "complete.json").read_text())
            guard_partition(
                split,
                summary["training_shots"],
                summary["selection_shots"],
                summary["heldout_shots"],
            )
            if summary.get("input_policy") != INPUT_POLICY:
                raise ValueError("checkpoint uses obsolete input/sampling policy")
            checkpoint = torch.load(
                directory / "checkpoint.pt", map_location=device, weights_only=False
            )
            model = (
                HL3(dropout=checkpoint["configuration"]["dropout"])
                if name == "saw-hl3"
                else PhasePicker()
            ).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            for shot in shots:
                with np.load(args.work / "signals" / f"{shot}.npz") as signal:
                    predictions[shot].append(
                        infer(
                            model, name, signal, checkpoint["mean"], checkpoint["std"]
                        )
                    )
            summaries.append(summary)
            del model, checkpoint
            torch.cuda.empty_cache()
        threshold = float(
            np.mean([s["selected_crash_threshold"]["threshold"] for s in summaries])
        )
        z = float(np.mean([s["selected_crash_threshold"]["z"] for s in summaries]))
        presence_threshold = float(
            np.mean([s["presence_threshold"] for s in summaries])
        )
        destination = args.work / "predictions" / "ensemble" / name
        destination.mkdir(parents=True, exist_ok=True)
        files = []
        for shot in shots:
            prediction = ensemble_predictions(predictions[shot])
            with np.load(args.work / "signals" / f"{shot}.npz") as signal:
                prediction["picks"] = picks(name, prediction, signal, threshold, z)
            path = destination / f"{shot}.npz"
            np.savez_compressed(path, **prediction)
            files.append(
                {"shot": shot, "path": str(path), "picks": len(prediction["picks"])}
            )
        provenance = {
            "model": name,
            "shots": requested,
            "requested_shot_ids": requested,
            "supported_shot_ids": shots,
            "requested_shots": len(requested),
            "supported_shots": len(shots),
            "scored_shots": None,
            "scoring_status": "inference export; outcomes scored in benchmark.json",
            "unavailable": unavailable,
            "split": "fixed validation; excluded from fitting and selection",
            "files": files,
            "crash_threshold": threshold,
            "derivative_z": z,
            "presence_threshold": presence_threshold,
            "threshold_provenance": "means of frozen inner-selection thresholds",
            "ensemble": "mean of three frozen fold probabilities",
            "pick_support": "observable bins, including algorithm-uncertain bins",
            "validation_status": "unvalidated algorithmic predictions for blind review",
            "input_policy": INPUT_POLICY,
        }
        save_json(destination / "manifest.json", provenance)
        save_json(OUTPUT / f"{name}_queue_predictions.json", provenance)
        print(
            f"{name}: {len(shots)} supported / {len(requested)} requested "
            "fixed-validation predictions",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["train", "evaluate", "predict"])
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument(
        "--folds", nargs="+", type=int, choices=(0, 1, 2), default=[0, 1, 2]
    )
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--windows-per-shot", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--prediction-shots", nargs="+", type=int, default=[])
    args = parser.parse_args()
    if not 1 <= args.threads <= 8:
        parser.error("threads must be 1..8")
    if args.patience < 1 or args.min_delta <= 0 or args.max_epochs < 1:
        parser.error("patience, min-delta and max-epochs must be positive")
    {"train": train, "evaluate": evaluate, "predict": predict}[args.stage](args)


if __name__ == "__main__":
    main()
