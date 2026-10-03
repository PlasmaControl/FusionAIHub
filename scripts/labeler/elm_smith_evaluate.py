#!/usr/bin/env python
"""Freeze review-trained elm-ours, then independently evaluate Smith ELM windows.

Stages: prepare (CPU ELM-O and identical inputs), freeze (GPU; no fitting),
train (GPU Smith-only shot CV), evaluate (CPU one compact JSON). Large arrays and
checkpoints stay in LABELER_ROOT/round4/elm/smith; the small record is committed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from labeler.config import Paths, git_sha, sha256_of
from labeler.elm import inputs, labels, net, onset, prepare, score, smith, train

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/labeler/elm/smith/evaluation.json"
SMITH_DIR = Path("/projects/EKOLEMEN/dsmith/data")
SEED = 20261003


def elmo_module():
    spec = importlib.util.spec_from_file_location(
        "elmo_benchmark", REPO / "scripts/labeler/elmo_benchmark.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def work_dir(paths):
    return paths.root / "round4/elm/smith"


def windows(paths):
    d = pd.read_csv(
        paths.root / "benchmarks/elm/elmo/smith_windows.csv", dtype={"group": str}
    )
    if not d.status.eq("ok").all():
        raise ValueError("Smith index has unevaluable windows")
    d["dt_ms"] = (d.label_t1_ms - d.label_t0_ms) / (d.label_stop - d.label_start)
    d["end_ms"] = d.t0_ms + d.n * d.dt_ms
    return d


def audit(paths, d):
    review = labels.review_table(prepare.review_csv(paths))
    review_shots = sorted(review.shot.unique().tolist())
    shots = sorted(d.shot.unique().tolist())
    cohort = pd.read_csv(paths.catalog / "cohort.csv")
    smith.check_disjoint(shots, review_shots, cohort)
    wanted = set(shots) | set(review_shots)
    dates = {}
    # Extract only identity/date fields; operator text is neither retained nor printed.
    # Run identifiers can append A/B to a day; compare the calendar date.
    pattern = re.compile(r'^\{"shot": (\d+), "run": "(\d{8})[^\"]*"')
    with paths.logs_jsonl.open() as handle:
        for line in handle:
            match = pattern.match(line)
            if match and int(match[1]) in wanted:
                dates[int(match[1])] = match[2]
    for row in cohort.itertuples():
        if row.shot in wanted and pd.notna(row.run_id):
            dates.setdefault(int(row.shot), str(int(row.run_id)))
    ours_days = {dates[s] for s in review_shots if s in dates}
    smith_days = {dates[s] for s in shots if s in dates}
    return {
        "smith_windows": len(d),
        "smith_shots": shots,
        "review_shots": review_shots,
        "shot_overlap": sorted(set(shots) & set(review_shots)),
        "cohort_test_overlap": sorted(
            set(shots) & set(cohort[cohort.split == "test"].shot)
        ),
        "run_day_source": str(paths.logs_jsonl),
        "smith_run_days": sorted(smith_days),
        "review_run_days": sorted(ours_days),
        "run_day_overlap": sorted(smith_days & ours_days),
        "shots_missing_run_day": sorted(wanted - set(dates)),
        "smith_shot_run_days": {str(s): dates.get(s) for s in shots},
        "run_day_freshness": "Independent of review training; Smith previously evaluated ELM-O.",
    }


def prepare_shot(job):
    shot, rows, source, output = job
    rows = pd.DataFrame(rows)
    data = dict(np.load(source))
    record_audit = {}
    for time_key, signal_key in (
        ("t_fs_ms", "filterscopes"),
        ("t_int_ms", "interferometer"),
    ):
        time, values, diagnostic_audit = smith.first_monotone_segment(
            data[time_key], data[signal_key]
        )
        data[time_key], data[signal_key] = time, values
        record_audit[signal_key] = diagnostic_audit
    if any(row["time_resets"] for row in record_audit.values()):
        if not all(
            np.isfinite(data[key]).all() for key in ("filterscopes", "interferometer")
        ):
            raise ValueError(f"Nonfinite channels in retained acquisition: {shot}")
        derivative = output / "records" / f"{shot}.npz"
        np.savez_compressed(derivative, **data)
        refetch = output / "refetch" / f"{shot}.npz"
        refetched = dict(np.load(refetch)) if refetch.exists() else None
        (output / "records" / f"{shot}.audit.json").write_text(
            json.dumps(
                {
                    "source": str(source),
                    "source_sha256": sha256_of(source),
                    "derivative": str(derivative),
                    "derivative_sha256": sha256_of(derivative),
                    "retained_channels_finite": True,
                    "refetch": str(refetch) if refetch.exists() else None,
                    "refetch_sha256": sha256_of(refetch) if refetch.exists() else None,
                    "original_refetch_samples_identical": all(
                        np.array_equal(np.load(source)[key], refetched[key])
                        for key in (
                            "t_fs_ms",
                            "filterscopes",
                            "t_int_ms",
                            "interferometer",
                        )
                    )
                    if refetched is not None
                    else None,
                    "diagnostics": record_audit,
                },
                indent=1,
            )
            + "\n"
        )
    x = inputs.channels(
        data["t_fs_ms"], data["filterscopes"], data["t_int_ms"], data["interferometer"]
    )
    np.save(output / "inputs" / f"{shot}.npy", x)
    n_ms = x.shape[1] // inputs.CELLS_PER_MS
    state, target, mask = smith.targets(rows, n_ms)
    valid = x[inputs.VALID].reshape(-1, inputs.CELLS_PER_MS).all(axis=1)
    mask &= valid
    state[~mask] = -1
    target[~mask] = 0
    elmo = elmo_module()
    calls = np.zeros(n_ms, dtype=bool)
    detections, window_counts = [], []
    handles = {}
    try:
        for row in rows.itertuples():
            if row.file not in handles:
                handles[row.file] = h5py.File(SMITH_DIR / row.file, "r")
            group = handles[row.file][row.group]
            bes = elmo.bes_mean(group["signals"][()])
            density = elmo.source_slice(
                data["t_int_ms"], data["interferometer"], row.t0_ms, row.end_ms
            )
            fs = elmo.source_slice(
                data["t_fs_ms"], data["filterscopes"], row.t0_ms, row.end_ms
            )
            if density is None or fs is None:
                raise ValueError(f"Smith {shot}/{row.group} incomplete cached input")
            trace = elmo.Trace(
                row.t0_ms, row.dt_ms, bes, density, fs, round(elmo.CHUNK_MS / row.dt_ms)
            )
            first, last = elmo.detect(trace)
            window_counts.append(
                elmo.count(first, last, row.label_start, row.label_stop)
            )
            edge = inputs.GRID0_MS + np.arange(n_ms)
            for a, b in zip(first, last):
                t0, t1 = row.t0_ms + a * row.dt_ms, row.t0_ms + b * row.dt_ms
                calls |= (edge < t1) & (edge + 1 > t0)
                detections.append(t0)
    finally:
        for handle in handles.values():
            handle.close()
    truth = rows.label_t0_ms.to_numpy(float)
    np.savez_compressed(
        output / "targets" / f"{shot}.npz",
        state=state,
        target=target,
        mask=mask,
        truth=truth,
        elmo_call=calls,
        elmo_events=np.array(smith.deduplicate_events(detections)),
        elmo_overlap_counts=np.sum(window_counts, axis=0),
        input_valid=valid,
        window_t0=rows.t0_ms.to_numpy(float),
        window_t1=rows.end_ms.to_numpy(float),
    )
    return shot, int(mask.sum()), int((state == 1).sum()), len(truth)


def run_prepare(paths, args):
    output = work_dir(paths)
    for sub in ("inputs", "targets", "frozen", "cv/pred", "records"):
        (output / sub).mkdir(parents=True, exist_ok=True)
    d = windows(paths)
    protocol = {
        "git": git_sha(),
        "created": datetime.now(UTC).isoformat(),
        "index": str(paths.root / "benchmarks/elm/elmo/smith_windows.csv"),
        "index_sha256": sha256_of(paths.root / "benchmarks/elm/elmo/smith_windows.csv"),
        "independence": audit(paths, d),
        "frozen_source": str(paths.root / "round4/elm/cv" / args.run),
        "input_recipe": "Identical FS02-FS04/DENV2F-3F preprocessing; no BES for our models.",
        "occupancy": "Deduplicated union of 1ms cells wholly inside at least one individual Smith window; any hand region overlap is positive. Cells covered only by stitching across neighboring window edges are excluded conservatively. Not the review's 50ms crowd occupancy target.",
        "event_truth": "Hand-labelled region start, not D-alpha maximum.",
        "event_matching": "Maximum-cardinality one-to-one chronological matching at +/-2 and +/-5 ms; ties minimize absolute timing error; predictions counted only in known Smith time.",
        "frozen_ensemble": "Original cv2 five checkpoints: mean probability / each existing fold threshold, call >=1. No fitting, selection, threshold adjustment or Smith-informed changes. Auxiliary onset head normalized by its frozen fold thresholds likewise.",
        "bootstrap": "1000 physical-shot resamples, seed 20261003; identical draws across methods; valid/undefined counts recorded; <5 positive-bearing shots descriptive.",
        "onset_recipe": "Same U-Net from scratch, onset-only masked BCE on Smith hand starts; sigma=1 ms; 5 shot folds, 21 inner-validation shots; select trailing-3 mean inner-val event F1 at +/-2ms after warmup; threshold on inner validation only. No review/cohort test shots; no tuning on outer folds.",
        "limitations": "Smith windows are selected around known ELMs, not a continuous-discharge negative population. Event FPs within them depend on completeness of single-region hand annotations. Smith CV is developmental, frozen review-to-Smith transfer is independent.",
        "code_sha256": {
            str(p.relative_to(REPO)): sha256_of(p)
            for p in (Path(__file__), REPO / "src/labeler/elm/smith.py")
        },
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=1) + "\n")
    jobs = [
        (
            int(s),
            group.to_dict("records"),
            prepare.signals_dir(paths) / f"{s}.npz",
            output,
        )
        for s, group in d.groupby("shot")
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(prepare_shot, jobs):
            print("prepared", *result, flush=True)


def run_freeze(paths, args):
    directory = work_dir(paths)
    record = json.loads(
        (paths.root / "round4/elm/cv" / args.run / "run.json").read_text()
    )
    models, thresholds, onset_thresholds, sources = [], [], [], []
    for row in sorted(record["fold_records"], key=lambda r: r["fold"]):
        source = (
            paths.root / "round4/elm/cv" / args.run / f"fold{row['fold']}" / "model.pt"
        )
        model = net.ElmUNet(dropout=record["config"]["dropout"]).to(args.device)
        model.load_state_dict(
            torch.load(source, map_location=args.device, weights_only=True)
        )
        model.eval()
        models.append(model)
        thresholds.append(row["threshold"])
        onset_thresholds.append(row["onset_threshold"])
        sources.append(
            {
                "fold": row["fold"],
                "checkpoint": str(source),
                "sha256": sha256_of(source),
                "threshold": row["threshold"],
                "onset_threshold": row["onset_threshold"],
            }
        )
    for shot in sorted(windows(paths).shot.unique()):
        x = np.load(directory / "inputs" / f"{shot}.npy")
        traces = np.stack([train.predict(model, x, args.device) for model in models])
        ev = np.mean(traces[:, 0] / np.array(thresholds)[:, None], axis=0)
        on = np.mean(traces[:, 1] / np.array(onset_thresholds)[:, None], axis=0)
        np.savez_compressed(directory / "frozen" / f"{shot}.npz", event=ev, onset=on)
        print("frozen", shot, flush=True)
    (directory / "frozen/sources.json").write_text(
        json.dumps(
            {
                "git": git_sha(),
                "source_run": args.run,
                "models": sources,
                "occupancy_threshold": 1.0,
                "auxiliary_onset_threshold": 1.0,
                "max_gpu_allocated_bytes": torch.cuda.max_memory_allocated()
                if args.device.startswith("cuda")
                else 0,
            },
            indent=1,
        )
        + "\n"
    )


def data_of(paths):
    directory = work_dir(paths)
    data = {}
    for shot, rows in windows(paths).groupby("shot"):
        x = np.load(directory / "inputs" / f"{shot}.npy")
        z = dict(np.load(directory / "targets" / f"{shot}.npz"))
        data[int(shot)] = (x, z, rows)
    return data


def found_onsets(p, mask, threshold):
    cell = onset.peaks(p, threshold)
    return inputs.GRID0_MS + cell[mask[cell]] + 0.5


def validation(model, data, val, device):
    probabilities = {s: train.predict(model, data[s][0], device)[1] for s in val}
    best = (-1.0, 0.95)
    for threshold in np.arange(0.05, 0.96, 0.05):
        counts = np.zeros(3)
        for s in val:
            z = data[s][1]
            result = smith.event_counts(
                found_onsets(probabilities[s], z["mask"], threshold), z["truth"], 2.0
            )
            counts += result
        tp, fp, fn = counts
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        best = max(best, (float(f1), float(threshold)))
    return best


def training_batch(rng, data, shots, cfg, device):
    arrays, targets, masks = [], [], []
    for shot in rng.choice(shots, cfg.batch):
        x, z, rows = data[int(shot)]
        window = rows.iloc[int(rng.integers(len(rows)))]
        centre = int(window.label_t0_ms - inputs.GRID0_MS)
        offset = int(rng.integers(cfg.crop_ms // 4, 3 * cfg.crop_ms // 4))
        a = int(np.clip(centre - offset, 0, max(0, len(z["mask"]) - cfg.crop_ms)))
        b = min(a + cfg.crop_ms, len(z["mask"]))
        crop = np.zeros(
            (inputs.N_CHANNELS, cfg.crop_ms * inputs.CELLS_PER_MS), np.float32
        )
        target = np.zeros(cfg.crop_ms, np.float32)
        mask = np.zeros(cfg.crop_ms, bool)
        crop[:, : (b - a) * inputs.CELLS_PER_MS] = x[
            :, a * inputs.CELLS_PER_MS : b * inputs.CELLS_PER_MS
        ]
        target[: b - a], mask[: b - a] = z["target"][a:b], z["mask"][a:b]
        train.augment(rng, crop, cfg)
        arrays.append(crop)
        targets.append(target)
        masks.append(mask)
    return tuple(
        torch.from_numpy(np.stack(v)).to(device) for v in (arrays, targets, masks)
    )


def run_train(paths, args):
    directory = work_dir(paths)
    data = data_of(paths)
    rng = np.random.default_rng(SEED)
    shots = np.array(sorted(data))
    rng.shuffle(shots)
    folds = [sorted(shots[k::5].astype(int).tolist()) for k in range(5)]
    cfg = train.Config(epochs=args.epochs, iters=args.iters, batch=16, crop_ms=1024)
    record = {
        "git": git_sha(),
        "folds": folds,
        "fold_records": [],
        "seed": SEED,
        "model": "elm-ours-onset",
        "parameters": net.n_parameters(net.ElmUNet()),
        "epochs": cfg.epochs,
        "iters": cfg.iters,
        "batch": cfg.batch,
        "crop_ms": cfg.crop_ms,
        "target_sigma_ms": smith.SIGMA_MS,
        "inner_validation_shots": 21,
    }
    for k, test in enumerate(folds):
        others = sorted(set(data) - set(test))
        fit, val = train.split_inner(others, 21, SEED + k)
        torch.manual_seed(SEED + k)
        rng = np.random.default_rng(SEED + k)
        model = net.ElmUNet(dropout=cfg.dropout).to(args.device)
        opt = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=cfg.lr, total_steps=cfg.epochs * cfg.iters, pct_start=0.15
        )
        history, best_state, best_row, best = [], None, None, -1.0
        for epoch in range(cfg.epochs):
            model.train()
            losses = []
            for _ in range(cfg.iters):
                x, target, mask = training_batch(rng, data, fit, cfg, args.device)
                logits = model(x)[:, 1]
                weights = 1 + train.ONSET_POS_WEIGHT * target
                loss = F.binary_cross_entropy_with_logits(
                    logits, target, weight=weights, reduction="none"
                )
                loss = (loss * mask).sum() / mask.sum().clamp(min=1)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
                scheduler.step()
                losses.append(float(loss.detach()))
            f1, threshold = validation(model, data, val, args.device)
            smoothed = float(np.mean([r["val_f1_2ms"] for r in history[-2:]] + [f1]))
            row = {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "val_f1_2ms": f1,
                "threshold": threshold,
                "selection_trailing3_f1": smoothed,
            }
            history.append(row)
            if epoch >= max(3, int(np.ceil(cfg.epochs * 0.15))) and smoothed > best:
                best = smoothed
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                best_row = row
            print("onset", k, json.dumps(row), flush=True)
        if best_state is None:
            raise ValueError("No valid post-warmup onset checkpoint")
        model.load_state_dict(best_state)
        fold_dir = directory / "cv" / f"fold{k}"
        fold_dir.mkdir(exist_ok=True)
        torch.save(best_state, fold_dir / "model.pt")
        info = {
            "fold": k,
            "train": fit,
            "inner_val": val,
            "test": test,
            "best": best_row,
            "threshold": best_row["threshold"],
            "history": history,
            "checkpoint_sha256": sha256_of(fold_dir / "model.pt"),
        }
        (fold_dir / "fold.json").write_text(json.dumps(info, indent=1) + "\n")
        for shot in test:
            p = train.predict(model, data[shot][0], args.device)[1]
            np.savez_compressed(directory / "cv/pred" / f"{shot}.npz", onset=p)
        record["fold_records"].append(info)
        record["max_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
        (directory / "cv/run.json").write_text(json.dumps(record, indent=1) + "\n")


def run_evaluate(paths, args):
    directory = work_dir(paths)
    protocol = json.loads((directory / "protocol.json").read_text())
    protocol["independence"] = audit(paths, windows(paths))
    protocol["code_sha256"] = {
        str(p.relative_to(REPO)): sha256_of(p)
        for p in (Path(__file__), REPO / "src/labeler/elm/smith.py")
    }
    protocol["input_source_repairs"] = [
        json.loads(path.read_text())
        for path in sorted((directory / "records").glob("*.audit.json"))
    ]
    protocol["occupancy"] = (
        "Deduplicated union of 1ms cells wholly inside at least one individual Smith window; any hand region overlap is positive. Cells covered only by stitching across neighboring window edges are excluded conservatively. Not the review's 50ms crowd occupancy target."
    )
    protocol["onset_postprocessing"] = (
        "Local maxima at 1 ms output resolution, minimum separation 10 ms, scored in whole Smith-window cells; thresholds frozen for elm-ours and chosen only on inner validation for elm-ours-onset."
    )
    frozen = json.loads((directory / "frozen/sources.json").read_text())
    cv = json.loads((directory / "cv/run.json").read_text())
    thresholds = {s: r["threshold"] for r in cv["fold_records"] for s in r["test"]}
    shots = sorted(windows(paths).shot.unique().tolist())
    window_table = windows(paths)
    bins = {m: [] for m in ("elm-ours", "elm-ours-onset", "elm-elmo")}
    events = {m: {str(tol): [] for tol in (2, 5)} for m in bins}
    overlap = np.zeros(3, dtype=int)
    overlap_parts = []
    old_sweep_path = paths.root / "benchmarks/elm/elmo/smith_sweep.npz"
    with np.load(old_sweep_path) as legacy_sweep:
        i = int(np.flatnonzero(np.isclose(legacy_sweep["thresholds"], 1.0))[0])
        j = int(np.flatnonzero(np.isclose(legacy_sweep["etas"], 0.997))[0])
        old_per_window = legacy_sweep["counts"][:, :, i, j]
    old_overlap_parts = []
    raw_counts = {"bins": 0, "positive_bins": 0, "hand_onsets": 0}
    coverage_audit = {
        "all_windows": len(window_table),
        "fully_input_covered_windows": 0,
        "unsupported_windows": [],
        "hand_onset_cells_outside_whole_scored_window": 0,
        "hand_regions_without_scored_positive_cell": 0,
        "cells_excluded_vs_merged_window_union": 0,
        "shots_with_cells_excluded_vs_merged_window_union": [],
        "definition": "Every 1ms input cell touching the complete Smith window has both FS and interferometer samples; fully covered comparisons use the same windows and all methods. Nine edge onsets can lack a whole scored-window cell, but every truth remains in event denominators and may match an adjacent supported peak within tolerance.",
    }
    per_shot = []
    for shot in shots:
        z = dict(np.load(directory / "targets" / f"{shot}.npz"))
        p = dict(np.load(directory / "frozen" / f"{shot}.npz"))
        on = np.load(directory / "cv/pred" / f"{shot}.npz")["onset"]
        mask = z["mask"]
        rows_of_shot = window_table[window_table.shot == shot]
        starts, stops = labels.merge_intervals(
            rows_of_shot.t0_ms.to_numpy(float), rows_of_shot.end_ms.to_numpy(float)
        )
        edge = inputs.GRID0_MS + np.arange(len(mask))
        merged_known = np.zeros(len(mask), dtype=bool)
        for a, b in zip(starts, stops):
            merged_known |= (edge >= a) & (edge + 1 <= b)
        omitted = int((merged_known & z["input_valid"] & ~mask).sum())
        coverage_audit["cells_excluded_vs_merged_window_union"] += omitted
        if omitted:
            coverage_audit["shots_with_cells_excluded_vs_merged_window_union"].append(
                {"shot": shot, "cells": omitted}
            )
        for row in window_table[window_table.shot == shot].itertuples():
            a = int(np.floor(row.t0_ms - inputs.GRID0_MS))
            b = int(np.ceil(row.end_ms - inputs.GRID0_MS))
            supported = (
                a >= 0 and b <= len(z["input_valid"]) and z["input_valid"][a:b].all()
            )
            if supported:
                coverage_audit["fully_input_covered_windows"] += 1
            else:
                coverage_audit["unsupported_windows"].append(
                    {"shot": shot, "file": row.file, "group": row.group}
                )
            cell = int(np.floor(row.label_t0_ms - inputs.GRID0_MS))
            coverage_audit["hand_onset_cells_outside_whole_scored_window"] += int(
                not (0 <= cell < len(mask) and mask[cell])
            )
            edge = inputs.GRID0_MS + np.arange(len(mask))
            has_positive = (
                (edge < row.label_t1_ms) & (edge + 1 > row.label_t0_ms) & mask
            ).any()
            coverage_audit["hand_regions_without_scored_positive_cell"] += int(
                not has_positive
            )
        continuous = {
            "elm-ours": p["event"],
            "elm-ours-onset": on,
            "elm-elmo": z["elmo_call"].astype(float),
        }
        limits = {"elm-ours": 1.0, "elm-ours-onset": thresholds[shot], "elm-elmo": 0.5}
        found = {
            "elm-ours": found_onsets(p["onset"], mask, 1.0),
            "elm-ours-onset": found_onsets(on, mask, thresholds[shot]),
            "elm-elmo": z["elmo_events"],
        }
        raw_counts["bins"] += int(mask.sum())
        raw_counts["positive_bins"] += int((z["state"][mask] == 1).sum())
        raw_counts["hand_onsets"] += len(z["truth"])
        overlap += z["elmo_overlap_counts"]
        overlap_parts.append(
            dict(zip(("tp", "fp", "fn"), z["elmo_overlap_counts"].tolist()))
            | {"errors_ms": []}
        )
        old_count = old_per_window[window_table.shot.to_numpy() == shot].sum(axis=0)
        old_overlap_parts.append(
            dict(zip(("tp", "fp", "fn"), old_count.tolist())) | {"errors_ms": []}
        )
        detail = {
            "shot": shot,
            "onset_fold": next(
                r["fold"] for r in cv["fold_records"] if shot in r["test"]
            ),
            "truth_onsets": len(z["truth"]),
            "bins": int(mask.sum()),
            "methods": {},
        }
        for method, parts in bins.items():
            value = continuous[method][mask]
            truth = z["state"][mask]
            parts.append(
                score.ShotScore(
                    shot,
                    truth,
                    np.full(len(truth), "non_crowd"),
                    value >= limits[method],
                    value,
                )
            )
            detail["methods"][method] = {"detected_onsets": len(found[method])}
            for tol in (2, 5):
                result = smith.match_events(found[method], z["truth"], tol)
                events[method][str(tol)].append(result)
                detail["methods"][method][str(tol)] = {
                    k: result[k] for k in ("tp", "fp", "fn")
                }
        per_shot.append(detail)
    boot = score.draws(len(shots), seed=SEED)
    result = {
        "git": git_sha(),
        "created": datetime.now(UTC).isoformat(),
        "protocol": protocol,
        "frozen_model": frozen,
        "onset_training": {k: v for k, v in cv.items() if k != "fold_records"},
        "onset_folds": [
            {k: v for k, v in row.items() if k != "history"}
            for row in cv["fold_records"]
        ],
        "counts": raw_counts,
        "coverage_audit": coverage_audit,
        "common_covered_comparison": {
            "windows": coverage_audit["fully_input_covered_windows"],
            "identical_to_all_windows": not coverage_audit["unsupported_windows"],
            "metrics_alias": "methods"
            if not coverage_audit["unsupported_windows"]
            else None,
            "note": "After the independently audited source repair, every window is input covered, so common-covered and all-window results coincide; the record stores one copy.",
        },
        "shots": len(shots),
        "methods": {},
        "per_shot": per_shot,
        "requested_elmo_reference": {
            "precision": 0.997,
            "recall": 0.980,
            "source": "Prior local ELM-O reimplementation on these 2316 windows, reproduced from the original saved sweep below. Overlap-region precision/recall, not onset timing. Correcting the source time resets improves recall without changing settings. The paper digest Table II fixed-setting score is separately retained.",
            "verified_digest_fixed_setting": {
                "precision": 0.995,
                "recall": 0.976,
                "events": 972,
            },
        },
        "reimplemented_elmo_overlap": smith.event_summary(overlap_parts, boot),
        "original_cached_elmo_overlap": smith.event_summary(old_overlap_parts, boot)
        | {"source": str(old_sweep_path), "source_sha256": sha256_of(old_sweep_path)},
        "code_sha256": {
            str(p.relative_to(REPO)): sha256_of(p)
            for p in (Path(__file__), REPO / "src/labeler/elm/smith.py")
        },
    }
    for name in ("reimplemented_elmo_overlap", "original_cached_elmo_overlap"):
        result[name].pop("timing_error_ms")
        result[name]["bootstrap"] = {
            k: v
            for k, v in result[name]["bootstrap"].items()
            if not k.startswith("timing_")
        }
    for method, parts in bins.items():
        occupancy = score.summarise(parts, boot)
        metrics = {
            "precision",
            "recall",
            "f1",
            "false_alarm_bin_rate",
            "auroc",
            "auprc",
            "prevalence",
        }
        for key in ("point", "ci95", "bootstrap_draw_counts"):
            if key in occupancy:
                occupancy[key] = {
                    k: v for k, v in occupancy[key].items() if k in metrics
                }
        occupancy["counts"] = {
            k: occupancy["counts"][k] for k in ("tp", "fp", "fn", "tn")
        }
        result["methods"][method] = {
            "occupancy_1ms": occupancy,
            "events": {
                tol: smith.event_summary(parts, boot)
                for tol, parts in events[method].items()
            },
        }
    result["methods"]["elm-elmo"]["occupancy_1ms"]["score_source"] = (
        "Binary fixed-setting overlap call; AUROC/AP are binary-score metrics, no eta sweep."
    )
    result["methods"]["elm-ours"]["events_source"] = (
        "Frozen auxiliary onset head; separate from delivered occupancy output, no Smith threshold tuning."
    )
    result["onset_output_delivered"] = False
    result["onset_output_scope"] = "Physical-onset output in the reviewed event catalog"
    result["experimental_smith_cv_onset_traces"] = {
        "model": "elm-ours-onset",
        "delivered": True,
        "path": str(directory / "cv/pred"),
        "scope": "1ms benchmark traces on Smith shots from held-out-shot folds; selected-window evaluation only",
    }
    frozen_f1 = result["methods"]["elm-ours"]["events"]["2"]["point"]["f1"]
    smith_f1 = result["methods"]["elm-ours-onset"]["events"]["2"]["point"]["f1"]
    result["onset_delivery_reason"] = (
        f"Frozen auxiliary onset F1 at +/-2ms is {frozen_f1:.3f}; Smith-only CV onset F1 is {smith_f1:.3f}. The experimental CV trace is delivered, but physical-onset catalog output remains withheld because selected Smith windows do not validate continuous-discharge false alarms or transfer of the Smith-trained head to the review domain."
    )
    # Undefined metrics stay explicit nulls alongside valid/undefined draw counts.
    result = json.loads(json.dumps(result), parse_constant=lambda _: None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    for method, row in result["methods"].items():
        print(method, "occupancy", row["occupancy_1ms"]["point"], flush=True)
        print(
            method,
            "events",
            {tol: value["point"] for tol, value in row["events"].items()},
            flush=True,
        )
    print(
        "counts",
        raw_counts,
        "run_day_overlap",
        protocol["independence"]["run_day_overlap"],
        "missing_dates",
        len(protocol["independence"]["shots_missing_run_day"]),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("stage", choices=("prepare", "freeze", "train", "evaluate"))
    parser.add_argument("--run", default="cv2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    torch.set_num_threads(4)
    paths = Paths.from_env()
    {
        "prepare": run_prepare,
        "freeze": run_freeze,
        "train": run_train,
        "evaluate": run_evaluate,
    }[args.stage](paths, args)


if __name__ == "__main__":
    main()
