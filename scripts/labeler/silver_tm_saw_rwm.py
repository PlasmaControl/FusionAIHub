#!/usr/bin/env python
"""Silver-list tables of three detectors: tm-ours, saw-ours v2 and rwm-ours.

The owner's ask of 2026-10-15: run each event set's best model on the 450 silver
shots (`<events>/silver/silver_shots.csv`), so the review page shows its output
as a layer. One extend table per model, `<event>/extend_<producer>/silver_shots.csv`
with its `.meta.json`. Every stage runs as a SLURM job; nothing fetches.

    stage        runs on                                  writes under RUN (below)
    tm-features  SLURM CPU, silver_tm_saw_rwm.sbatch      tm/magfeatures/<shot>.npz
                 tm_magfeatures.py's features for the silver corpus shots round four
                 never featurised, over each shot's own Ip window (rule 4,
                 `frames.targets.window_for`): the catalog has no window for them
    tm-infer     SLURM GPU, silver_tm_infer.sbatch        tm/pred/<shot>.npz, tm/infer.json
                 the five tm-ours folds (three seeds each), averaged, on every silver
                 shot with features but no out-of-fold row: one batched GPU pass over
                 all of them (and over the out-of-fold shots, whose stored scores it
                 must reproduce), checked against tm_ours.predict on the CPU
    saw-frames   SLURM CPU, silver_tm_saw_rwm.sbatch      saw/frames/<shot>.npz
                 saw-ours v2 (fix6 variant c): the stored fold predictions ANDed with
                 the derivative edge score at each fold's z (the combined presence),
                 averaged over each 10 ms frame's observable 2 ms bins
    rwm-score    SLURM CPU, silver_tm_saw_rwm.sbatch      rwm/scores.parquet, rwm/model.json
                 the final rwm-ours forest, fitted on Hanson's 33 shots, scored on
                 every silver shot whose inputs load
    table        SLURM CPU, silver_tm_saw_rwm.sbatch      the three tables
                 thresholds the saved scores; a table already there is copied to
                 BACKUP first; with --limit (the pilot) it writes under
                 RUN/pilot_tables instead

    RUN    = $LABELER_ROOT/runs/silver_infer/tm_saw_rwm
    BACKUP = $LABELER_ROOT/backups/silver_infer_2026-10-15

`--limit N` (pilots) takes the first N shots of the stage's sorted plan, `--sample N`
N shots spread evenly over it; either writes the single-file outputs with a `_pilot`
suffix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for entry in (REPO / "src", Path(__file__).resolve().parent):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from labeler.config import Paths

PATHS = Paths.from_env()
ROOT = PATHS.root
RUN = ROOT / "runs" / "silver_infer" / "tm_saw_rwm"
BACKUP = ROOT / "backups" / "silver_infer_2026-10-15"
TM = ROOT / "round4" / "tm"
TM_CV = TM / "results" / "tm_ours_magnetics_cv.json"
TM_OOF = TM / "results" / "oof_tm_ours_magnetics.npz"
RWM_OOF = ROOT / "round5" / "rwm" / "classifier_rwm-ours.parquet"
RWM_RECORD = REPO / "docs" / "labeler" / "results" / "rwm_classifier.json"
SAW_RECORD = REPO / "outputs" / "labeler" / "sawtooth" / "fix6" / "benchmark.json"
GRID_MS = 10
#: The 2 ms scoring bins (sawtooth.metrics.bin_times) inside one 10 ms frame.
SAW_BIN_OFFSETS_MS = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
#: A frame is assessed when at least this many of its five 2 ms bins are observable.
SAW_MIN_OBSERVABLE = 3
TABLES = {
    "tm": ("neoclassical_tearing_mode", "extend_tm_ours"),
    "saw": ("sawtooth_oscillation", "extend_saw_ours"),
    "rwm": ("resistive_wall_mode", "extend_rwm_ours"),
}
LONGEST_WINDOW_MS = 20_000


def silver_shots() -> list[int]:
    frame = pd.read_csv(PATHS.label_tables / "silver" / "silver_shots.csv")
    return sorted(int(s) for s in frame.shot)


def silver_pools() -> dict[int, str]:
    frame = pd.read_csv(PATHS.label_tables / "silver" / "silver_shots.csv")
    return {int(s): str(p) for s, p in zip(frame.shot, frame.pool, strict=True)}


def in_corpus(shot: int) -> bool:
    return (PATHS.corpus / f"{int(shot)}_processed.h5").is_file()


def has_mirnov(shot: int) -> bool:
    """The shot has a corpus file (whatever it holds: a missing group is reported per
    shot) or a real `mirnov` record in the raw cache (`labeler.events.raw`, filled on
    the login node by `python -m labeler.events.raw --groups mirnov`)."""
    from labeler.events import raw

    return in_corpus(shot) or raw.record_tier(shot, "mirnov", paths=PATHS) is not None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pool_map(function, jobs, workers):
    """`function` over `jobs` in `workers` forked processes, in completion order."""
    if workers <= 1:
        yield from map(function, jobs)
        return
    with get_context("fork").Pool(workers) as pool:
        yield from pool.imap_unordered(function, jobs)


def suffix(args) -> str:
    return "_pilot" if args.limit or args.sample else ""


def chosen(plan: list, args) -> list:
    """The pilot's shots: the plan's first `--limit`, or `--sample` spread evenly
    over it; the whole plan otherwise."""
    if args.limit:
        return plan[: args.limit]
    if args.sample:
        index = np.unique(
            np.linspace(0, len(plan) - 1, args.sample).round().astype(int)
        )
        return [plan[i] for i in index]
    return plan


# ------------------------------------------------------------------ tm-ours


def tm_oof_shots() -> set[int]:
    with np.load(TM_OOF) as z:
        return {int(k[1:]) for k in z.files}


def tm_feature_plan() -> list[int]:
    """Silver shots with a Mirnov record (`has_mirnov`: the corpus or the raw cache) and
    no features yet, round four's or this stage's."""
    have = {
        int(p.stem)
        for folder in (TM / "magfeatures", RUN / "tm" / "magfeatures")
        for p in folder.glob("*.npz")
        if p.stem.isdigit()
    }
    return [s for s in silver_shots() if s not in have and has_mirnov(s)]


def tm_feature_one(shot: int):
    from tm_magfeatures import MARGIN_MS

    from labeler.events.panels._shared import finite
    from labeler.events.raw import raw_signal
    from labeler.events.verify import NoDataError
    from labeler.frames.targets import window_for
    from labeler.tearing import magfeatures

    out = RUN / "tm" / "magfeatures"
    started = time.monotonic()
    try:
        (w0, w1), origin = window_for(shot, PATHS, None, catalog={})
        # the corpus, then the raw cache; never fetched (LABELER_NO_FETCH in the jobs)
        array = raw_signal(
            shot,
            "mirnov",
            channels=list(magfeatures.PROBE_ROWS),
            t_range=(w0 - MARGIN_MS, w1 + MARGIN_MS),
            paths=PATHS,
        )
        values = np.stack([finite(y) for y in array.y])
        centres, feats = magfeatures.shot_features(array.x, values, (w0, w1))
    except (NoDataError, KeyError, OSError, ValueError) as exc:
        (out / f"{shot}.missing.json").write_text(
            json.dumps({"shot": shot, "reason": f"{type(exc).__name__}: {exc}"[:300]})
        )
        return shot, "missing", time.monotonic() - started
    (out / f"{shot}.missing.json").unlink(missing_ok=True)
    tmp = out / f"{shot}.tmp.npz"
    np.savez(
        tmp,
        centres_ms=centres,
        features=feats.astype(np.float32),
        names=np.array(magfeatures.FEATURE_NAMES),
        window_ms=np.array([w0, w1], dtype=float),
        window_from=np.array(origin),
    )
    tmp.rename(out / f"{shot}.npz")
    return shot, "ok", time.monotonic() - started


def tm_features(args) -> int:
    plan = tm_feature_plan()
    shots = chosen(plan, args)
    (RUN / "tm" / "magfeatures").mkdir(parents=True, exist_ok=True)
    print(
        f"tm-features: {len(shots)} of {len(plan)} planned shots, {args.workers} workers",
        flush=True,
    )
    tally = {}
    for k, (shot, status, secs) in enumerate(
        pool_map(tm_feature_one, shots, args.workers), 1
    ):
        tally[status] = tally.get(status, 0) + 1
        print(f"{k}/{len(shots)} shot {shot} {status} {secs:.1f}s", flush=True)
    print(json.dumps(tally))
    return 0


def tm_infer_plan() -> list[tuple[int, Path]]:
    """Silver shots with features (round four's, or this run's) and no OOF row."""
    oof = tm_oof_shots()
    plan = []
    for shot in silver_shots():
        if shot in oof:
            continue
        for path in (
            TM / "magfeatures" / f"{shot}.npz",
            RUN / "tm" / "magfeatures" / f"{shot}.npz",
        ):
            if path.is_file():
                plan.append((shot, path))
                break
    return plan


#: The CPU-chain check (`tm_ours.predict` on the CPU, one shot at a time) covers this
#: many inferred shots, spread over the plan; the batched GPU pass must match it.
TM_CPU_CHECK_SHOTS = 5
TM_CPU_CHECK_TOL = 1e-4
#: The stored out-of-fold scores came from the same checkpoints (both written
#: 2026-10-04 01:00); recomputing them is the check of the whole batched path. The
#: CV ran on a GPU, possibly with TF32 convolutions (PyTorch's default on Ampere),
#: so its scores carry ~1e-3 of rounding; a wrong fold, scaler or mask is far larger.
TM_OOF_CHECK_TOL = 1e-2


def tm_load_one(job):
    """`(shot, path, centres_ms, features)`: one shot's magnetics features, as
    `tm_ours.load` reads them (float64)."""
    shot, path = job
    with np.load(path) as z:
        return (
            shot,
            path,
            z["centres_ms"].astype(float),
            z["features"].astype(np.float64),
        )


def tm_networks(device):
    """`[(fold, threshold, [(net, mean, std, val_loss)])]`: the five tm-ours folds of
    the CV record, three seeds each, read from their checkpoints on `device`."""
    import tm_ours
    import torch

    cv = json.loads(TM_CV.read_text())
    folds = []
    for info in cv["fold_info"]:
        checkpoint = torch.load(
            info["checkpoint"], map_location=device, weights_only=False
        )
        if abs(checkpoint["threshold"] - info["threshold"]) > 1e-9:
            raise ValueError(
                f"fold {info['fold']}: checkpoint threshold differs from the record"
            )
        hyper = checkpoint["hyper"]
        nets = []
        for member in checkpoint["members"]:
            net = tm_ours.model_class()(
                len(member["mean"]),
                hyper["hidden"],
                hyper["kernel"],
                hyper["dilations"],
                hyper["dropout"],
            ).to(device)
            net.load_state_dict(member["state_dict"])
            net.eval()
            nets.append(
                (
                    net,
                    np.asarray(member["mean"], dtype=float),
                    np.asarray(member["std"], dtype=float),
                    float(member["val_loss"]),
                )
            )
        folds.append((int(info["fold"]), float(info["threshold"]), nets))
    return folds


def tm_masked_forward(net, x, valid):
    """`net`'s logits `(shots, bins)` on a zero-padded batch `x`, equal to running each
    shot alone (`tm_ours.predict`): the dilated convolutions pad with zeros, so each
    one's input is zeroed past every shot's last bin (`valid`, `(shots, 1, bins)`);
    the 1x1 head mixes no bins."""
    from torch import nn

    h = x
    for layer in net.body:
        if isinstance(layer, nn.Conv1d):
            h = h * valid
        h = layer(h)
    return net.head(h).squeeze(1)


def tm_batched(folds, loaded, device):
    """`(per_fold, lengths, seconds)`: the sigmoid probability of every fold (the mean
    of its seeds) for every loaded shot, `(folds, shots, bins)` on `device`, from one
    padded batch. As `tm_ours.batch`: each seed standardises with its own mean and
    std, and a bin whose features are not all finite enters as zeros."""
    import torch

    seconds = {}
    started = time.perf_counter()
    lengths = np.array([len(centres) for _, _, centres, _ in loaded])
    n_shots, n_bins = len(loaded), int(lengths.max())
    n_feat = loaded[0][3].shape[1]
    host = np.zeros((n_shots, n_feat, n_bins))
    row_ok = np.zeros((n_shots, n_bins), dtype=bool)
    for i, (_, _, _, features) in enumerate(loaded):
        ok = np.isfinite(features).all(axis=1)
        host[i, :, : len(features)] = np.where(ok[:, None], features, 0.0).T
        row_ok[i, : len(features)] = ok
    inside = np.arange(n_bins)[None, :] < lengths[:, None]
    x = torch.as_tensor(host, device=device)
    ok_t = torch.as_tensor(row_ok, device=device)[:, None, :]
    valid = torch.as_tensor(inside, dtype=torch.float32, device=device)[:, None, :]
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds["pack_h2d"] = time.perf_counter() - started
    started = time.perf_counter()
    per_fold = torch.empty((len(folds), n_shots, n_bins), device=device)
    with torch.no_grad():
        for k, (_, _, nets) in enumerate(folds):
            total = torch.zeros((n_shots, n_bins), device=device)
            for net, mean, std, _ in nets:
                m = torch.as_tensor(mean, device=device)[None, :, None]
                s = torch.as_tensor(std, device=device)[None, :, None]
                z = torch.where(ok_t, (x - m) / s, 0.0).to(torch.float32)
                total += torch.sigmoid(tm_masked_forward(net, z, valid))
            per_fold[k] = total / len(nets)
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds["gpu_compute"] = time.perf_counter() - started
    return per_fold, lengths, seconds


def stamp() -> str:
    """Local wall time as nvidia-smi's timestamp column writes it."""
    return datetime.now(UTC).astimezone().strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]


def tm_infer(args) -> int:
    """One batched pass of the five folds x three seeds over every silver shot with
    features: the shots with no out-of-fold row are inferred and written; the
    out-of-fold shots recompute their held-out fold's stored score as the check of
    the whole path; a few shots are rerun through `tm_ours.predict` on the CPU."""
    from concurrent.futures import ThreadPoolExecutor

    import tm_ours
    import torch

    times = {}
    started = time.perf_counter()
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    print(
        f"torch {torch.__version__} cuda.is_available() {cuda} device {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if cuda else ""),
        flush=True,
    )
    if not cuda and not args.allow_cpu:
        raise SystemExit("no CUDA device: tm-infer runs as a SLURM GPU job")
    # full float32 convolutions, so the batched pass can match the CPU chain
    torch.backends.cudnn.allow_tf32 = False
    if cuda:
        torch.zeros(1, device=device)
        torch.cuda.synchronize()
    times["torch_cuda_init"] = time.perf_counter() - started

    started = time.perf_counter()
    plan = tm_infer_plan()
    targets = chosen(plan, args)
    with np.load(TM_OOF) as z:
        oof_keys = set(z.files)
    oof_plan = [
        (s, TM / "magfeatures" / f"{s}.npz")
        for s in silver_shots()
        if f"s{s}" in oof_keys
    ]
    checks = chosen(oof_plan, args)
    if not targets:
        raise SystemExit("no silver shot has features and no out-of-fold row")
    times["plan"] = time.perf_counter() - started
    print(
        f"tm-infer: {len(targets)} of {len(plan)} planned shots inferred; "
        f"{len(checks)} of {len(oof_plan)} out-of-fold silver shots recomputed as the "
        f"check; {args.workers} loader threads",
        flush=True,
    )

    started = time.perf_counter()
    with ThreadPoolExecutor(max(1, args.workers)) as pool:
        loaded = list(pool.map(tm_load_one, targets + checks))
    times["load_features"] = time.perf_counter() - started

    started = time.perf_counter()
    folds = tm_networks(device)
    if cuda:
        torch.cuda.synchronize()
    times["load_checkpoints"] = time.perf_counter() - started

    print(f"shot loop (batched pass) starts {stamp()}", flush=True)
    per_fold, lengths, seconds = tm_batched(folds, loaded, device)
    print(f"shot loop (batched pass) ends {stamp()}", flush=True)
    times.update(seconds)

    started = time.perf_counter()
    per_fold = per_fold.cpu().numpy()
    times["d2h"] = time.perf_counter() - started

    started = time.perf_counter()
    out = RUN / "tm" / "pred"
    out.mkdir(parents=True, exist_ok=True)
    for i, (shot, path, centres, features) in enumerate(loaded[: len(targets)]):
        n = lengths[i]
        p_folds = per_fold[:, i, :n]
        np.savez(
            out / f"{shot}.npz",
            centres_ms=centres,
            p=p_folds.mean(axis=0).astype(np.float32),
            p_folds=p_folds.astype(np.float32),
            finite=np.isfinite(features).all(axis=1),
            features=np.array(str(path)),
        )
    times["write"] = time.perf_counter() - started

    # the check: each out-of-fold shot's held-out fold against the stored score
    started = time.perf_counter()
    cv = json.loads(TM_CV.read_text())
    held = {int(s): int(info["fold"]) for info in cv["fold_info"] for s in info["held"]}
    thresholds = {fold: threshold for fold, threshold, _ in folds}
    order = {fold: k for k, (fold, _, _) in enumerate(folds)}
    worst, flips, n_bins = 0.0, 0, 0
    with np.load(TM_OOF) as stored:
        for j, (shot, _, _, _) in enumerate(loaded[len(targets) :], start=len(targets)):
            fold = held[shot]
            mine = per_fold[order[fold], j, : lengths[j]]
            ref = stored[f"s{shot}"].astype(float)
            if len(ref) != len(mine):
                raise ValueError(
                    f"shot {shot}: stored OOF has {len(ref)} bins, not {len(mine)}"
                )
            worst = max(worst, float(np.max(np.abs(mine - ref))) if len(ref) else 0.0)
            flips += int(
                ((mine >= thresholds[fold]) != (ref >= thresholds[fold])).sum()
            )
            n_bins += len(ref)
    times["check_oof"] = time.perf_counter() - started
    oof_check = {
        "shots": len(checks),
        "bins": n_bins,
        "max_abs_diff": worst,
        "calls_differing": flips,
        "tolerance": TM_OOF_CHECK_TOL,
    }
    print(
        f"OOF check: {len(checks)} shots, {n_bins} bins, max |GPU batched - stored OOF| "
        f"{worst:.2e}, {flips} calls differ at the fold thresholds",
        flush=True,
    )

    # the CPU chain: tm_ours.predict, one shot at a time, on the CPU
    started = time.perf_counter()
    cpu_folds = tm_networks(torch.device("cpu"))
    picks = np.unique(
        np.linspace(0, len(targets) - 1, min(TM_CPU_CHECK_SHOTS, len(targets)))
        .round()
        .astype(int)
    ).tolist()
    cpu_worst = 0.0
    for i in picks:
        shot, _, centres, features = loaded[i]
        n = len(centres)
        data = {shot: (centres, features, np.zeros(n), np.zeros(n, dtype=bool))}
        for k, (_, _, nets) in enumerate(cpu_folds):
            ref = tm_ours.predict(nets, data, [shot], "cpu")[shot]
            cpu_worst = max(cpu_worst, float(np.max(np.abs(per_fold[k, i, :n] - ref))))
    times["check_cpu_chain"] = time.perf_counter() - started
    cpu_check = {
        "shots": [int(loaded[i][0]) for i in picks],
        "folds": len(cpu_folds),
        "max_abs_diff": cpu_worst,
        "tolerance": TM_CPU_CHECK_TOL,
    }
    print(
        f"CPU-chain check: {len(picks)} shots x {len(cpu_folds)} folds, max |GPU batched "
        f"- tm_ours.predict on CPU| {cpu_worst:.2e}",
        flush=True,
    )

    peak = torch.cuda.max_memory_allocated() / 2**20 if cuda else 0.0
    summary = {
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "device": torch.cuda.get_device_name(0) if cuda else "cpu",
        "torch": torch.__version__,
        "tf32": False,
        "shots_inferred": [int(s) for s, _ in targets],
        "batch": {
            "shots": len(loaded),
            "bins_padded": int(lengths.max()),
            "bins_real": int(lengths.sum()),
            "folds": len(folds),
            "seeds_per_fold": len(folds[0][2]),
        },
        "seconds": {k: round(v, 4) for k, v in times.items()},
        "peak_cuda_mib": round(peak, 1),
        "check_oof": oof_check,
        "check_cpu_chain": cpu_check,
    }
    (RUN / "tm" / f"infer{suffix(args)}.json").write_text(
        json.dumps(summary, indent=1) + "\n"
    )
    print(
        "stage seconds: " + ", ".join(f"{k} {v:.3f}" for k, v in times.items()),
        flush=True,
    )
    print(f"inferred {len(targets)} shots; peak CUDA memory {peak:.1f} MiB", flush=True)
    if cpu_worst > TM_CPU_CHECK_TOL or worst > TM_OOF_CHECK_TOL:
        raise SystemExit(
            "the batched GPU pass does not match the CPU chain or the stored OOF"
        )
    return 0


# ------------------------------------------------------------------ saw-ours v2


def saw_plan():
    """`[(shot, mode, folds, zs)]`: the OOF fold of each cohort shot, all three folds
    for the fixed-validation and expert shots; and the operating points."""
    import sawtooth_fix6 as fix6

    split = fix6.split_manifest()
    winner = json.loads((fix6.WORK / "selection.json").read_text())["winner"]
    if winner != "c":
        raise ValueError(f"fix6's selection names {winner!r}, not variant c")
    ops = [
        json.loads((fix6.WORK / f"calibration/fold_{f}/calibration.json").read_text())[
            winner
        ]["combined"]["presence"]
        for f in range(3)
    ]
    silver = set(silver_shots())
    cohort = {int(s) for s in split["training_cohort"]}
    others = {int(s) for s in split["fixed_validation_supported"]} | {
        int(s) for s in split["expert_shots"]
    }
    if cohort & others:
        raise ValueError("a cohort shot is also a fixed-validation or expert shot")
    plan = []
    for shot in sorted(silver & (cohort | others)):
        if shot in cohort:
            fold = int(split["folds"][shot])
            plan.append((shot, "oof", [fold], [float(ops[fold]["z"])]))
        else:
            plan.append((shot, "ensemble", [0, 1, 2], [float(o["z"]) for o in ops]))
    return plan, ops


def saw_one(job):
    import sawtooth_benchmark as sb
    import sawtooth_fix6 as fix6

    shot, mode, folds, zs = job
    started = time.monotonic()
    out = RUN / "saw" / "frames" / f"{shot}.npz"
    try:
        signal = sb.load_signal(
            fix6.WORK,
            shot,
            keys=("t", "y", "central_channel", "observable", "assessed"),
        )
        t = np.asarray(signal["t"], float)
        _, _, edge = fix6.derivative_parts(signal)
        presences = []
        for fold, z in zip(folds, zs, strict=True):
            raw = fix6.load_raw("c", fold, shot)
            if len(raw["t"]) != len(t) or not np.allclose(raw["t"], t):
                raise ValueError(f"fold {fold}'s prediction is off the signal grid")
            presences.append(fix6.presence_score(raw, edge, z))
        presence = np.mean(presences, axis=0)
        t_ms = t * 1000.0
        lo = GRID_MS * int(np.ceil(t_ms[0] / GRID_MS - 1e-6))
        hi = GRID_MS * int(np.floor(t_ms[-1] / GRID_MS + 1e-6))
        frames = np.arange(lo, hi, GRID_MS, dtype=np.int64)
        centres = (frames[:, None] + SAW_BIN_OFFSETS_MS[None, :]).ravel() / 1000.0
        observable, _ = sb.masks_at(signal, centres)
        observable = observable.reshape(len(frames), -1)
        p = np.interp(centres, t, presence).reshape(len(frames), -1)
        n_obs = observable.sum(axis=1)
        with np.errstate(invalid="ignore"):
            mean = np.where(n_obs > 0, (p * observable).sum(axis=1) / n_obs, np.nan)
    except (OSError, KeyError, ValueError) as exc:
        return (
            shot,
            f"failed: {type(exc).__name__}: {exc}"[:300],
            time.monotonic() - started,
        )
    np.savez(
        out,
        frame_ms=frames,
        n_observable=n_obs.astype(np.int8),
        presence=mean.astype(np.float32),
        mode=np.array(mode),
        folds=np.array(folds),
        z=np.array(zs),
    )
    return shot, "ok", time.monotonic() - started


def saw_frames(args) -> int:
    plan, ops = saw_plan()
    jobs = chosen(plan, args)
    (RUN / "saw" / "frames").mkdir(parents=True, exist_ok=True)
    print(
        f"saw-frames: {len(jobs)} of {len(plan)} planned shots "
        f"({sum(j[1] == 'oof' for j in jobs)} out-of-fold), {args.workers} workers; "
        f"operating points {[(o['threshold'], o['z']) for o in ops]}",
        flush=True,
    )
    tally, failed = {}, {}
    for k, (shot, status, secs) in enumerate(pool_map(saw_one, jobs, args.workers), 1):
        key = "ok" if status == "ok" else "failed"
        tally[key] = tally.get(key, 0) + 1
        if key == "failed":
            failed[str(shot)] = status
        if key != "ok" or k % 25 == 0 or k == len(jobs):
            print(f"{k}/{len(jobs)} shot {shot} {status} {secs:.1f}s", flush=True)
    (RUN / "saw" / f"failed{suffix(args)}.json").write_text(
        json.dumps(failed, indent=1)
    )
    print(json.dumps(tally))
    return 0


# ------------------------------------------------------------------ rwm-ours


def rwm_features_one(shot: int):
    from labeler.events.verify import NoDataError
    from labeler.rwm import data, features

    started = time.monotonic()
    try:
        signals = data.load_signals(shot, PATHS)
    except NoDataError as exc:
        return shot, None, f"NoDataError: {exc}"[:300], time.monotonic() - started
    try:
        table = features.slice_table(signals)
        if table.empty:
            return (
                shot,
                None,
                "no high-current window (|Ip| at least half its peak)",
                (time.monotonic() - started),
            )
        columns = features.bp_columns(signals, table.t_ms.to_numpy(float))
    except (KeyError, OSError, ValueError, IndexError) as exc:
        return (
            shot,
            None,
            f"{type(exc).__name__}: {exc}"[:300],
            time.monotonic() - started,
        )
    for name, values in columns.items():
        table[name] = values
    table.insert(0, "shot", int(shot))
    return shot, table, None, time.monotonic() - started


def rwm_score(args) -> int:
    import rwm_classify

    from labeler.events.review import labels as review_labels
    from labeler.events.review.silver import event_dir
    from labeler.rwm import classify
    from labeler.rwm.evaluate import Imputer
    from labeler.rwm.forest import BalancedForest

    started = time.monotonic()
    columns = list(rwm_classify.MODELS["rwm-ours"])
    table_path = review_labels.source_path(event_dir(PATHS, rwm_classify.EVENT))
    frame, audit = rwm_classify.rows(PATHS, table_path)
    oof = pd.read_parquet(RWM_OOF)
    same_rows = (
        len(oof) == len(frame)
        and np.array_equal(oof.shot.to_numpy(), frame.shot.to_numpy())
        and np.allclose(oof.t_ms.to_numpy(float), frame.t_ms.to_numpy(float))
        and np.array_equal(oof.y.to_numpy(), frame.y.to_numpy())
    )
    rows = frame[frame.y >= 0]
    imputer = Imputer().fit(rows, columns)
    model = BalancedForest(seed=rwm_classify.SEED, **classify.FOREST).fit(
        imputer.transform(rows), rows.y.to_numpy()
    )
    scored = oof[oof.y >= 0]
    cut = classify.f1_cut(scored.score.to_numpy(float), scored.y.to_numpy())
    fit_s = time.monotonic() - started
    print(
        f"rwm-score: forest fitted on {len(rows)} bins ({int(rows.y.sum())} present) of "
        f"{rows.shot.nunique()} shots in {fit_s:.1f} s; pooled OOF F1 cut {cut:.6f}; "
        f"labels and rows as recorded: {same_rows}",
        flush=True,
    )
    hanson = {int(s) for s in frame.shot.unique()}
    plan = silver_shots()
    shots = chosen(plan, args)
    pieces, skipped, seconds = [], {}, {}
    for shot in [s for s in shots if s in hanson]:
        part = oof[oof.shot == shot][["shot", "t_ms", "score", "cut"]].copy()
        part["source"] = "oof"
        part["n_finite"] = np.nan
        pieces.append(part)
    todo = [s for s in shots if s not in hanson]
    print(
        f"{len(shots) - len(todo)} Hanson shots take their OOF scores; "
        f"{len(todo)} shots to score, {args.workers} workers",
        flush=True,
    )
    finite_counts = {name: 0 for name in columns}
    n_rows = 0
    for k, (shot, table, reason, secs) in enumerate(
        pool_map(rwm_features_one, todo, args.workers), 1
    ):
        seconds[shot] = round(secs, 2)
        if table is None:
            skipped[str(shot)] = reason
            print(
                f"{k}/{len(todo)} shot {shot} skipped {secs:.1f}s: {reason}", flush=True
            )
            continue
        x = table[columns].to_numpy(float)
        finite = np.isfinite(x)
        for i, name in enumerate(columns):
            finite_counts[name] += int(finite[:, i].sum())
        n_rows += len(table)
        part = table[["shot", "t_ms"]].copy()
        part["score"] = model.predict_proba(imputer.transform(table))
        part["cut"] = cut
        part["source"] = "final"
        part["n_finite"] = finite.sum(axis=1)
        pieces.append(part)
        if k % 25 == 0 or k == len(todo):
            print(
                f"{k}/{len(todo)} shot {shot} {len(table)} slices {secs:.1f}s",
                flush=True,
            )
    out = RUN / "rwm"
    out.mkdir(parents=True, exist_ok=True)
    scores = pd.concat(pieces, ignore_index=True).sort_values(["shot", "t_ms"])
    scores.to_parquet(out / f"scores{suffix(args)}.parquet", index=False)
    record = json.loads(RWM_RECORD.read_text())["models"]["rwm-ours"]
    model_json = {
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "columns": columns,
        "forest": {
            **classify.FOREST,
            "max_features": "sqrt",
            "seed": rwm_classify.SEED,
        },
        "training": {
            "shots": sorted(hanson),
            "bins": len(rows),
            "present_bins": int(rows.y.sum()),
            "labels_table": str(table_path),
            "labels_table_sha256": sha256(Path(table_path)),
            "rows_and_labels_as_recorded": bool(same_rows),
            "audit": audit,
        },
        "imputer_medians": dict(zip(columns, map(float, imputer.fill_), strict=True)),
        "cut": cut,
        "cut_rule": "classify.f1_cut over the pooled out-of-fold scores of the record "
        f"({RWM_OOF.name}, {len(scored)} scored bins)",
        "record_cuts_by_day": record.get("cuts"),
        "record_f1": record["metrics"]["f1"],
        "silver_rows": n_rows,
        "silver_finite_fraction": {
            name: round(finite_counts[name] / n_rows, 4) if n_rows else None
            for name in columns
        },
        "shots_scored": sorted(int(s) for s in scores.shot.unique()),
        "skipped": skipped,
        "seconds_per_shot": {str(k): v for k, v in sorted(seconds.items())},
        "fit_seconds": round(fit_s, 1),
    }
    (out / f"model{suffix(args)}.json").write_text(
        json.dumps(model_json, indent=1) + "\n"
    )
    print(
        f"wrote {len(scores)} rows for {scores.shot.nunique()} shots; skipped "
        f"{len(skipped)}; {time.monotonic() - started:.1f} s",
        flush=True,
    )
    return 0


# ------------------------------------------------------------------ tables


def frame_rows(shot, starts, p, assessed, threshold):
    """Interval rows of consecutive 10 ms frames sharing one category: 1 where the
    score reaches `threshold`, 0 below it, 3 (not observable) where not `assessed`;
    `confidence` is the mean probability of the row's own class (blank for 3)."""
    starts = np.asarray(starts, dtype=float)
    p = np.asarray(p, dtype=float)
    category = np.where(assessed, (p >= threshold).astype(int), 3)
    out = []
    i, n = 0, len(starts)
    while i < n:
        j = i + 1
        while (
            j < n
            and category[j] == category[i]
            and abs(starts[j] - starts[j - 1] - GRID_MS) < 1e-6
        ):
            j += 1
        c = int(category[i])
        if c == 3:
            confidence = np.nan
        else:
            part = p[i:j]
            confidence = round(float(np.mean(part if c == 1 else 1.0 - part)), 4)
        out.append(
            (int(shot), c, float(starts[i]), float(starts[j - 1] + GRID_MS), confidence)
        )
        i = j
    return out


def check_windows(rows: list, skipped: dict, shots_labelled: list) -> list:
    """Drop (and name) any shot whose rows span more than the page's longest window."""
    frame = pd.DataFrame(
        rows, columns=["shot", "category", "t_start", "t_end", "confidence"]
    )
    extent = frame.groupby("shot").agg(lo=("t_start", "min"), hi=("t_end", "max"))
    long = extent[(extent.hi - extent.lo) > LONGEST_WINDOW_MS]
    for shot, row in long.iterrows():
        skipped[str(int(shot))] = (
            f"window {row.lo:.0f}-{row.hi:.0f} ms is longer than the page's "
            f"{LONGEST_WINDOW_MS} ms"
        )
        shots_labelled.remove(int(shot))
    dropped = {int(s) for s in long.index}
    return [r for r in rows if r[0] not in dropped]


def write_table(key: str, rows: list, meta: dict, root: Path | None = None) -> Path:
    """The table under `root` (the label tables; a pilot's scratch root), the page's
    own reader taking it back; a label table already there is backed up first."""
    from labeler.events.interval_tables import write_interval_table
    from labeler.events.review import labels as review_labels

    root = PATHS.label_tables if root is None else root
    event, producer_dir = TABLES[key]
    path = root / event / producer_dir / "silver_shots.csv"
    for existing in (path, path.with_suffix(".meta.json")):
        if existing.is_file() and root == PATHS.label_tables:
            target = BACKUP / existing.relative_to(PATHS.label_tables)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(existing, target)
            print(f"backed up {existing} -> {target}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        rows, columns=["shot", "category", "t_start", "t_end", "confidence"]
    )
    write_interval_table(frame, path, meta)
    # the review page's own reader must take the whole table, or every shot is lost
    parsed = review_labels.read_labels(path)
    print(
        f"wrote {path}: {len(frame)} rows, {frame.shot.nunique()} shots "
        f"(read back {len(parsed)})"
    )
    return path


def base_meta(event: str, producer: str, jobs: dict) -> dict:
    from labeler.config import git_sha

    return {
        "category": event,
        "producer": producer,
        "status": "experimental",
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "made_by": "python scripts/labeler/silver_tm_saw_rwm.py table",
        "made_in_job": os.environ.get("SLURM_JOB_ID"),
        "git_sha": git_sha(full=True),
        "code_sha256": sha256(Path(__file__)),
        "jobs": jobs,
        "shot_list": "silver: <events>/silver/silver_shots.csv",
        "shot_list_sha256": sha256(PATHS.label_tables / "silver" / "silver_shots.csv"),
        "grid": "10 ms frames [10k, 10k+10) from t = 0 (events.grid); runs of one "
        "category merged into one row",
    }


def tm_table(jobs: dict, shots: list[int], root: Path | None = None) -> None:
    cv = json.loads(TM_CV.read_text())
    held = {
        int(s): (int(info["fold"]), float(info["threshold"]))
        for info in cv["fold_info"]
        for s in info["held"]
    }
    mean_threshold = float(np.mean([info["threshold"] for info in cv["fold_info"]]))
    pools = silver_pools()
    rows, scored_by, skipped, labelled = [], {}, {}, []
    with np.load(TM_OOF) as oof:
        oof_keys = set(oof.files)
        for shot in shots:
            if f"s{shot}" in oof_keys:
                fold, threshold = held[shot]
                with np.load(TM / "magfeatures" / f"{shot}.npz") as z:
                    centres = z["centres_ms"].astype(float)
                    finite = np.isfinite(z["features"]).all(axis=1)
                p = oof[f"s{shot}"].astype(float)
                how = f"oof fold {fold} (threshold {threshold})"
            else:
                pred = RUN / "tm" / "pred" / f"{shot}.npz"
                if not pred.is_file():
                    missing = RUN / "tm" / "magfeatures" / f"{shot}.missing.json"
                    if missing.is_file():
                        skipped[str(shot)] = (
                            "no features: " + json.loads(missing.read_text())["reason"]
                        )
                    elif not has_mirnov(shot):
                        skipped[str(shot)] = (
                            f"no Mirnov on disk ({pools[shot]} shot outside the corpus, "
                            "none in the raw cache): fetch it with python -m "
                            "labeler.events.raw --event alfven_eigenmode --groups mirnov "
                            "--shots <shot>"
                        )
                    else:
                        skipped[str(shot)] = "features present but not inferred"
                    continue
                with np.load(pred) as z:
                    centres = z["centres_ms"].astype(float)
                    finite = z["finite"].astype(bool)
                    p = z["p"].astype(float)
                fold, threshold = None, mean_threshold
                how = "ensemble"
            if len(p) != len(centres):
                raise ValueError(f"shot {shot}: scores and feature bins differ")
            if len(centres) == 0:
                skipped[str(shot)] = "no whole 10 ms bin in the window"
                continue
            if not np.allclose(np.diff(centres), GRID_MS) or not np.allclose(
                (centres - GRID_MS / 2) % GRID_MS, 0
            ):
                raise ValueError(f"shot {shot}: feature bins are off the 10 ms grid")
            rows += frame_rows(shot, centres - GRID_MS / 2, p, finite, threshold)
            scored_by[str(shot)] = how
            labelled.append(shot)
    rows = check_windows(rows, skipped, labelled)
    checkpoints = [
        {
            "fold": int(info["fold"]),
            "path": info["checkpoint"],
            "sha256": sha256(Path(info["checkpoint"])),
            "threshold": float(info["threshold"]),
        }
        for info in cv["fold_info"]
    ]
    n_oof = sum(v.startswith("oof") for v in scored_by.values())
    f1 = cv["metrics"]["f1"]
    infer_record = RUN / "tm" / "infer.json"
    meta = {
        **base_meta("neoclassical_tearing_mode", "tm-ours", jobs),
        "inference": (
            json.loads(infer_record.read_text()) if infer_record.is_file() else None
        ),
        "model": "tm-ours 2026_10_04 (best_models.yaml): three dilated 1-D convolutions "
        "(hidden 32) over 33 magnetics spectrogram features per 10 ms bin of six midplane "
        "Mirnov probes, three seeds per fold, five shot-grouped folds of the 450-shot "
        "cohort; record round4/tm/results/tm_ours_magnetics_cv.json (git "
        f"{cv['git_sha']}), bin F1 {f1['value']:.3f} [{f1['lo']:.3f}, {f1['hi']:.3f}]",
        "record": str(TM_CV),
        "record_sha256": sha256(TM_CV),
        "oof_scores": str(TM_OOF),
        "oof_scores_sha256": sha256(TM_OOF),
        "checkpoints": checkpoints,
        "threshold": {
            "oof": "each cohort shot's held-out fold threshold (F1-best on that fold's "
            "inner validation shots)",
            "ensemble": mean_threshold,
        },
        "threshold_rule": "out-of-fold shots: their own fold's threshold; the others: "
        "the mean of the five fold thresholds over the five folds' mean probability "
        "(unevaluated)",
        "score": "probability per 10 ms bin; out-of-fold for the cohort shots (the one "
        "fold that never trained on the shot), else the mean of 5 folds x 3 seeds",
        "spans": "1 present where the bin's score reaches the threshold, 0 absent below "
        "it, 3 not observable where any of the 33 features is not finite (the network "
        "saw zeros there); a bin centred at c is the frame [c - 5, c + 5)",
        "window": "the bins wholly inside the shot's window: the catalog's rule-4 window "
        "for cohort and population shots (round4/tm/magfeatures), the shot's own Ip "
        "window (frames.targets.window_for, catalog={}) for the corpus shots round four "
        "never featurised (features under runs/silver_infer/tm_saw_rwm/tm/magfeatures)",
        "confidence": "mean probability of the row's class over its bins (1 - p for "
        "absent rows); blank for not observable",
        "shots_requested": shots,
        "shots_labelled": sorted(labelled),
        "shots_skipped": dict(sorted(skipped.items())),
        "scored_by": scored_by,
        "counts": {
            "oof": n_oof,
            "ensemble": len(scored_by) - n_oof,
            "skipped": len(skipped),
        },
        "caveats": [
            (
                "the five-fold mean and the mean threshold are unevaluated; only the "
                "out-of-fold rows carry the record's score"
            ),
            (
                "labels are interval tearing modes (extend_tm_interval) drawn from the "
                "n = 1 and n = 2 RMS: the model reproduces that rule's calls from the probes"
            ),
            (
                "folded into mhd_n1 on the silver page: only its present (1) rows show "
                "there; absent rows fall through to lower layers"
            ),
            "unreviewed model output; the owner's saved label wins over it",
        ],
    }
    write_table("tm", rows, meta, root)


def saw_table(jobs: dict, shots: list[int], root: Path | None = None) -> None:
    import sawtooth_fix6 as fix6

    plan, ops = saw_plan()
    planned = {shot for shot, *_ in plan}
    mean_threshold = float(np.mean([o["threshold"] for o in ops]))
    signals = fix6.FIX5_WORK.parent.parent / "saw" / "fix5" / "signals"
    failed = {}
    failed_path = RUN / "saw" / "failed.json"
    if failed_path.is_file():
        failed = json.loads(failed_path.read_text())
    pools = silver_pools()
    rows, scored_by, skipped, labelled = [], {}, {}, []
    for shot in shots:
        path = RUN / "saw" / "frames" / f"{shot}.npz"
        if shot not in planned:
            if not in_corpus(shot):
                skipped[str(shot)] = (
                    f"{pools[shot]} shot outside the corpus: no corpus ECE for "
                    "saw-ours v2's inputs"
                )
            else:
                skipped[str(shot)] = (
                    "no saw-ours v2 inputs on disk: outside the fix5 signal set "
                    f"({signals}); building them needs the sawtooth signal stage and a "
                    "GPU inference pass"
                )
            continue
        if not path.is_file():
            skipped[str(shot)] = failed.get(str(shot), "frames not computed")
            continue
        with np.load(path) as z:
            frames = z["frame_ms"].astype(float)
            n_obs = z["n_observable"].astype(int)
            presence = z["presence"].astype(float)
            mode = str(z["mode"])
            folds = [int(f) for f in z["folds"]]
        threshold = (
            float(ops[folds[0]]["threshold"]) if mode == "oof" else mean_threshold
        )
        if not len(frames):
            skipped[str(shot)] = "no whole 10 ms frame in the signal"
            continue
        assessed = n_obs >= SAW_MIN_OBSERVABLE
        rows += frame_rows(shot, frames, np.nan_to_num(presence), assessed, threshold)
        scored_by[str(shot)] = (
            f"oof fold {folds[0]} (threshold {threshold}, z {ops[folds[0]]['z']})"
            if mode == "oof"
            else "ensemble"
        )
        labelled.append(shot)
    rows = check_windows(rows, skipped, labelled)
    n_oof = sum(v.startswith("oof") for v in scored_by.values())
    meta = {
        **base_meta("sawtooth_oscillation", "saw-ours", jobs),
        "model": "saw-ours v2 2026_10_14 (fix6 variant c, the paper's saw-ours; "
        "best_models.yaml still names saw-hl3-full): the network's presence ANDed with "
        "the ECE derivative edge score at the fold's z (combine.combined_presence); "
        "three folds; no-q F1 0.958 [0.941, 0.970]; record outputs/labeler/sawtooth/fix6 "
        "(key saw-ours)",
        "why_this_model": "the round-five ledger (2026-10-13, 12:02) says the paper uses "
        "saw-ours v2, and its stored fold predictions and inputs are on disk for the "
        "322 silver shots it can cover, so it applies without retraining or GPU work",
        "record": str(SAW_RECORD),
        "record_sha256": sha256(SAW_RECORD),
        "predictions": str(fix6.WORK / "predictions" / "c"),
        "operating_points": [
            {"fold": f, "threshold": float(o["threshold"]), "z": float(o["z"])}
            for f, o in enumerate(ops)
        ],
        "threshold": {
            "oof": "each cohort shot's own fold threshold",
            "ensemble": mean_threshold,
        },
        "threshold_rule": "cohort shots: their held-out fold's combined-presence "
        "threshold and z; fixed-validation and expert shots: the mean of the three "
        "folds' combined presence at the mean threshold (fix6's held-out rule)",
        "score": "combined presence per native sample (10 kHz), interpolated at the "
        "benchmark's 2 ms bin centres and averaged over each frame's observable bins",
        "spans": f"a frame with at least {SAW_MIN_OBSERVABLE} of its five 2 ms bins "
        "observable (the signal's ECE observable mask) is 1 where its mean combined "
        "presence reaches the threshold and 0 below it; the other frames are 3 (not "
        "observable). The teacher's assessed mask is not used",
        "window": "the frames wholly inside the shot's fix5 signal span",
        "confidence": "mean presence of the row's class over its frames (1 - p for "
        "absent rows); blank for not observable",
        "shots_requested": shots,
        "shots_labelled": sorted(labelled),
        "shots_skipped": dict(sorted(skipped.items())),
        "scored_by": scored_by,
        "counts": {
            "oof": n_oof,
            "ensemble": len(scored_by) - n_oof,
            "skipped": len(skipped),
        },
        "caveats": [
            (
                "the edge-score AND reads the same ECE edges the rule's labels came from "
                "(the fix6 report's circularity caveat); on the three expert-reviewed "
                "shots fix6 scored presence F1 0.486"
            ),
            (
                "the three-fold mean on fixed-validation and expert shots is fix6's "
                "held-out rule; the out-of-fold rows carry the record's score"
            ),
            (
                "folded into mhd_n1 on the silver page: only its present (1) rows show "
                "there; absent rows fall through to lower layers"
            ),
            "unreviewed model output; the owner's saved label wins over it",
        ],
    }
    write_table("saw", rows, meta, root)


def rwm_table(jobs: dict, shots: list[int], root: Path | None = None) -> None:
    scores = pd.read_parquet(RUN / "rwm" / "scores.parquet")
    model = json.loads((RUN / "rwm" / "model.json").read_text())
    rows, scored_by, labelled = [], {}, []
    skipped = {k: v for k, v in model["skipped"].items() if int(k) in set(shots)}
    for shot in shots:
        part = scores[scores.shot == shot].sort_values("t_ms")
        if part.empty:
            skipped.setdefault(str(shot), "not scored")
            continue
        t = part.t_ms.to_numpy(float)
        if not np.allclose(t % GRID_MS, 0) or not np.allclose(np.diff(t), GRID_MS):
            raise ValueError(f"shot {shot}: RWM slices are off the 10 ms grid")
        cut = part.cut.to_numpy(float)
        if not np.allclose(cut, cut[0]):
            raise ValueError(f"shot {shot}: more than one cut")
        starts = t - GRID_MS  # the slice at t is the frame [t - 10, t)
        keep = starts >= 0
        rows += frame_rows(
            shot,
            starts[keep],
            part.score.to_numpy(float)[keep],
            np.ones(int(keep.sum()), bool),
            float(cut[0]),
        )
        source = str(part.source.iloc[0])
        scored_by[str(shot)] = (
            f"oof (its run day's cut {cut[0]})" if source == "oof" else "final forest"
        )
        labelled.append(shot)
    rows = check_windows(rows, skipped, labelled)
    record = json.loads(RWM_RECORD.read_text())
    ours = record["models"]["rwm-ours"]
    meta = {
        **base_meta("resistive_wall_mode", "rwm-ours", jobs),
        "model": "rwm-ours 2026_10_14 (best_models.yaml): balanced random forest "
        "(300 trees, depth 8, leaves of 5, seed 0) on the v5 inputs at t less lock_v "
        f"({len(model['columns'])} features), a presence classifier per 10 ms bin; "
        "record docs/labeler/results/rwm_classifier.json (read, never written), bin F1 "
        f"{ours['metrics']['f1']['estimate']:.3f}",
        "record": str(RWM_RECORD),
        "record_sha256": sha256(RWM_RECORD),
        "final_model": "no deployable model existed: one forest fitted in the rwm-score "
        "job on all of Hanson's scored bins with the record's hyperparameters and "
        "training-median imputation (rwm.classify.fit's recipe)",
        "training": model["training"],
        "imputer_medians": model["imputer_medians"],
        "cut": model["cut"],
        "threshold": model["cut"],
        "threshold_rule": model["cut_rule"] + "; Hanson's own shots keep their "
        "out-of-fold score and their run day's cut",
        "score": "forest positive-class probability for the slice at t",
        "spans": "the slice at t is the frame [t - 10, t); 1 present where the score "
        "reaches the cut, 0 absent below it; missing inputs take the training medians, "
        "so no frame is not observable",
        "window": "the slices of the high-current window (|Ip| at least half its peak, "
        "features.slice_table); outside it nothing is said",
        "confidence": "mean probability of the row's class over its frames (1 - p for "
        "absent rows)",
        "input_finite_fraction": model["silver_finite_fraction"],
        "shots_requested": shots,
        "shots_labelled": sorted(labelled),
        "shots_skipped": dict(sorted(skipped.items())),
        "scored_by": scored_by,
        "counts": {
            "final": sum(v == "final forest" for v in scored_by.values()),
            "oof": sum(v.startswith("oof") for v in scored_by.values()),
            "skipped": len(skipped),
        },
        "caveats": [
            (
                "the final forest and its pooled cut are unevaluated; the record's F1 is "
                "the leave-one-run-day-out score on Hanson's 33 shots"
            ),
            (
                "rwm-ours reads the labels' own sensor (the n = 1 delta-Bp), so its score "
                "is partly agreement with the delta-Bp rule (the record's caveat)"
            ),
            (
                "negatives were assumed on Hanson's shots; present bins are 0.66 % of the "
                "training bins, so a cut chosen there may over- or under-call on other "
                "eras"
            ),
            (
                "inputs the raw cache lacks (input_finite_fraction) take the training "
                "medians; no profile was fetched"
            ),
            "unreviewed model output; the owner's saved label wins over it",
        ],
    }
    write_table("rwm", rows, meta, root)


def tables(args) -> int:
    """The three tables; a pilot (`--limit N`, the first N silver shots, or `--sample
    N`, N spread over the list) writes under RUN/pilot_tables (or `--out-root`), never
    the label tables."""
    if args.jobs_file:
        jobs = json.loads(Path(args.jobs_file).read_text())
    else:
        jobs = json.loads(args.jobs) if args.jobs else {}
    shots = chosen(silver_shots(), args)
    if args.out_root:
        root = Path(args.out_root)
    else:
        pilot = args.limit or args.sample
        root = RUN / "pilot_tables" if pilot else PATHS.label_tables
    print(f"tables {args.which} for {len(shots)} silver shots under {root}", flush=True)
    for key in args.which:
        started = time.monotonic()
        {"tm": tm_table, "saw": saw_table, "rwm": rwm_table}[key](
            jobs.get(key, {}), shots, root
        )
        print(f"{key} table in {time.monotonic() - started:.1f} s", flush=True)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="stage", required=True)
    for name in ("tm-features", "tm-infer", "saw-frames", "rwm-score"):
        stage = sub.add_parser(name)
        stage.add_argument("--limit", type=int, help="pilot: the first N planned shots")
        stage.add_argument(
            "--sample",
            type=int,
            help="pilot: N planned shots spread evenly over the plan",
        )
        stage.add_argument("--workers", type=int, default=1)
        if name == "tm-infer":
            stage.add_argument(
                "--allow-cpu",
                action="store_true",
                help="run without CUDA (never on the login node)",
            )
    table = sub.add_parser("table")
    table.add_argument(
        "--which", nargs="+", choices=("tm", "saw", "rwm"), default=["tm", "saw", "rwm"]
    )
    table.add_argument("--jobs", help='JSON: {"tm": {"tm-features": id, ...}, ...}')
    table.add_argument("--jobs-file", help="the --jobs JSON, read from a file")
    table.add_argument(
        "--limit",
        type=int,
        help="pilot: the first N silver shots, written under RUN/pilot_tables",
    )
    table.add_argument(
        "--sample",
        type=int,
        help="pilot: N silver shots spread evenly over the list, as --limit",
    )
    table.add_argument("--out-root", help="write the tables under this root instead")
    args = parser.parse_args(argv)
    if args.stage == "table":
        return tables(args)
    if args.workers > 16:
        raise SystemExit("at most 16 workers")
    return {
        "tm-features": tm_features,
        "tm-infer": tm_infer,
        "saw-frames": saw_frames,
        "rwm-score": rwm_score,
    }[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
