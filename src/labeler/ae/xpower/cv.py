"""Five-fold cross-validation, and the MHD-aware choice (the ledger's Deviation 11).

    python -m labeler.ae.xpower.cv --folds                       # once, first
    python -m labeler.ae.xpower.cv --candidate NAME --fold K     # 3 x 5 tasks
    python -m labeler.ae.xpower.cv --choose                      # after all 15

all for `--version` (default v2) in `--models` (default
`$LABELER_ROOT/models/ae_xpower/<version>`), under `cv/`.

**Labels.** Only the version's snapshot (`labeler.ae.xpower.read_snapshot`),
never the owner's live table; every product records its sha256.

**Folds.** `data.make_split` as v1 does it (test = SELDNet's `valid`); the rest,
train and validation alike, is the pool (120 shots on the real data). The pool,
sorted, is permuted with seed 20260923 and dealt in turn into five folds, so
they differ in size by at most one shot (24 each on the real data). `--folds`
writes `cv/folds.csv` (shot, split, fold; the test shots have no fold) and
`cv/folds.json` before any training; a fold task refuses unless the file holds
exactly the folds the snapshot gives.

**A fold task.** Fold K is predicted; fold (K + 1) mod 5 stops the training
(`train.fit`: validation F1 at 0.5, patience 10, up to 60 epochs, v1's
`TrainConfig`); the other three train. Fold K's shots never train or stop the
model that predicts them. It writes `cv/<candidate>/fold<K>.npz`, per fold-K
shot the 0-2 s frames `evaluate.shot_frames` gives (`p<shot>` P(AE), `o<shot>`
the owner's states, `m<shot>` MHD, `s<shot>` scored), then `fold<K>.json`, the
record that marks the task done.

**The choice.** The out-of-fold frames of all five folds, pooled per candidate
(every pool shot once), at thresholds 0.10 to 0.90 in steps of 0.05, as
(candidate, threshold) rows of pooled F1, precision, recall and MHD
false-positive rate (`evaluate.cells`, `evaluate.fp_rate`):

1. if any row has MHD FP <= 0.05, the one with the highest F1 among those;
2. otherwise, if any has F1 >= 0.90, the one with the lowest MHD FP among those;
3. otherwise, the highest F1.

Ties go to the lower MHD weight, then to the threshold nearest 0.5, then (0.45
against 0.55, which the rule leaves open) to the lower threshold, as
`train.pick_threshold` orders them. A row with an undefined F1 never counts; one
with an undefined MHD FP (no MHD frame) is never in branch 1 or 2. `--choose`
refuses while any of the 15 records is missing, naming them, and writes
`cv/choice.json` (the choice, its branch, the whole table and each fold's best
epoch; the final model trains for the median of the chosen candidate's five)
and `cv/frontier.md`. Run again, it checks the saved choice and changes nothing.

**Pilots.** `--pilot N` (5-20 shots): each fold's first N // 5 shots, 2 epochs,
into `runs/ae_xpower/pilot/<version>`, where a record can be replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import resource
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from ...config import Paths, atomic_path, git_sha
from ...scoring import stats
from . import (
    CV_VERSIONS,
    EVENT,
    LABEL_SNAPSHOTS,
    evaluate,
    model_dir,
    pilot_area,
    read_snapshot,
    tokeye_masks,
    train,
)
from .data import SEED, load_shot, make_split, seldnet_split, store_rows

N_FOLDS = 5
THRESHOLDS = train.THRESHOLDS
MHD_FP_MAX = 0.05
F1_MIN = 0.90
BRANCHES = {
    1: "some (candidate, threshold) has MHD FP <= 0.05: the highest F1 among those",
    2: ("none has MHD FP <= 0.05, some has F1 >= 0.90: the lowest MHD FP among those"),
    3: "none has MHD FP <= 0.05 or F1 >= 0.90: the highest F1",
}
TIES = (
    "ties: the lower MHD weight, then the threshold nearest 0.5, then the lower "
    "threshold"
)
PILOT_EPOCHS = 2
#: The fields a saved choice must repeat when `--choose` runs again.
DECISION = ("candidate", "threshold", "branch", "table", "final_epochs", "sources")


def cv_dir(models: Path) -> Path:
    return models / "cv"


def stop_fold(k: int) -> int:
    """The fold that early-stops fold `k`'s model: the next one, cyclically."""
    return (k + 1) % N_FOLDS


def make_folds(pool, seed: int = SEED, n: int = N_FOLDS) -> dict[int, int]:
    """Fold per pool shot: the sorted pool permuted by `seed`, dealt in turn."""
    order = np.random.default_rng(seed).permutation(sorted(int(s) for s in pool))
    return {int(s): i % n for i, s in enumerate(order)}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def check_version(models: Path, version: str) -> None:
    """A cross-validated version, named by its models directory."""
    if version not in CV_VERSIONS or version not in LABEL_SNAPSHOTS:
        raise ValueError(f"version {version} is not chosen by cross-validation")
    if models.name != version:
        raise ValueError(
            f"{models}: --version {version} differs from the models directory's "
            f"name {models.name}"
        )


def _folds_text(split: dict[int, str], folds: dict[int, int]) -> str:
    lines = ["shot,split,fold"]
    for shot, which in sorted(split.items()):
        lines.append(f"{shot},{which},{folds.get(shot, '')}")
    return "\n".join(lines) + "\n"


def _expected(paths: Paths, version: str):
    """The snapshot's sha256 and labels, the split, the folds and folds.csv."""
    data, saved = read_snapshot(paths, version)
    split = make_split(saved, seldnet_split(tokeye_masks(paths)))
    folds = make_folds(s for s, v in split.items() if v != "test")
    return _sha(data), saved, split, folds, _folds_text(split, folds)


def write_folds(paths: Paths, models: Path, version: str) -> dict:
    """`cv/folds.csv` and `cv/folds.json`, once; the same folds again is a no-op."""
    check_version(models, version)
    digest, _, split, folds, text = _expected(paths, version)
    out = cv_dir(models)
    file = out / "folds.csv"
    counts = {
        "test": sum(v == "test" for v in split.values()),
        "pool": len(folds),
        "folds": np.bincount(list(folds.values()), minlength=N_FOLDS).tolist(),
    }
    if file.exists():
        if file.read_text() != text:
            raise FileExistsError(
                f"{file}: holds other folds than the snapshot gives; folds are "
                "fixed before training"
            )
        return counts
    out.mkdir(parents=True, exist_ok=True)
    record = {
        "version": version,
        "labels_sha256": digest,
        "seed": SEED,
        "n_folds": N_FOLDS,
        "stop_fold": "(K + 1) mod 5",
        "counts": counts,
        "folds_sha256": _sha(text.encode()),
        "git_sha": git_sha(),
        "made_at": _now(),
    }
    with atomic_path(out / "folds.json") as tmp:
        tmp.write_text(json.dumps(record, indent=1) + "\n")
    with atomic_path(file) as tmp:
        tmp.write_text(text)
    return counts


def _checked_folds(paths: Paths, models: Path, version: str):
    """The snapshot and folds, after checking `cv/folds.csv` against them."""
    digest, saved, _, folds, text = _expected(paths, version)
    file = cv_dir(models) / "folds.csv"
    if not file.is_file():
        raise FileNotFoundError(f"{file}: no folds; run --folds first")
    data = file.read_bytes()
    if data != text.encode():
        raise ValueError(f"{file}: differs from the folds the snapshot gives")
    record = json.loads((cv_dir(models) / "folds.json").read_text())
    if record.get("labels_sha256") != digest:
        raise ValueError(f"{cv_dir(models) / 'folds.json'}: another label snapshot")
    return digest, saved, folds, _sha(data)


def _by_fold(folds: dict[int, int], pilot: int) -> dict[int, list[int]]:
    by = {k: sorted(s for s, f in folds.items() if f == k) for k in range(N_FOLDS)}
    if pilot:
        by = {k: shots[: pilot // N_FOLDS] for k, shots in by.items()}
    return by


def _spec(version: str, candidate: str) -> dict:
    names = train.candidates(version)
    if candidate not in names:
        raise ValueError(
            f"--candidate {candidate} is not one of version {version}'s: "
            + ", ".join(names)
        )
    return names[candidate]


def _record_paths(models: Path, candidate: str, k: int) -> tuple[Path, Path]:
    out = cv_dir(models) / candidate
    return out / f"fold{k}.json", out / f"fold{k}.npz"


def run_fold(
    paths: Paths,
    models: Path,
    *,
    candidate: str,
    fold: int,
    version: str,
    pilot: int = 0,
    epochs: int = train.TrainConfig.epochs,
    log=print,
) -> dict:
    """Train on three folds, stop on fold `fold` + 1, predict fold `fold`."""
    check_version(models, version)
    spec = _spec(version, candidate)
    if not 0 <= fold < N_FOLDS:
        raise ValueError(f"--fold must be 0 to {N_FOLDS - 1}")
    in_runs = pilot_area(models, paths.runs)
    if pilot and not in_runs:
        raise ValueError(f"{models}: a pilot writes under {paths.runs}")
    record_file, npz_file = _record_paths(models, candidate, fold)
    if record_file.exists() and not in_runs:
        raise FileExistsError(f"{record_file}: a fold is trained once")
    digest, saved, folds, folds_sha = _checked_folds(paths, models, version)
    by = _by_fold(folds, pilot)
    stop = stop_fold(fold)
    train_folds = [f for f in range(N_FOLDS) if f not in (fold, stop)]
    train_shots = [s for f in train_folds for s in by[f]]

    def shot(s):
        rows = store_rows(paths.spectrogram_file(EVENT, s))
        return load_shot(s, saved[s], rows, tokeye_masks(paths), band=spec["band"])

    config = train.TrainConfig(
        epochs=PILOT_EPOCHS if pilot else epochs, mhd_weight=spec["mhd_weight"]
    )
    model, history, _ = train.fit(
        [shot(s) for s in train_shots], [shot(s) for s in by[stop]], config, log
    )
    blob = {"band_khz": list(spec["band"]), "threshold": 0.5, "candidate": candidate}
    arrays = {}
    for s in by[fold]:
        frames = evaluate.shot_frames(
            s, paths=paths, label=saved[s], model=model, blob=blob
        )
        arrays[f"p{s}"] = np.asarray(frames.prob, dtype=np.float32)
        arrays[f"o{s}"] = np.asarray(frames.owner, dtype=np.int8)
        arrays[f"m{s}"] = np.asarray(frames.mhd, dtype=bool)
        arrays[f"s{s}"] = np.asarray(frames.scored, dtype=bool)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    record_file.parent.mkdir(parents=True, exist_ok=True)
    record_file.unlink(missing_ok=True)  # a pilot's rerun is not done until written
    with atomic_path(npz_file) as tmp:
        tmp.write_bytes(buffer.getvalue())
    record = {
        "version": version,
        "candidate": candidate,
        "band_khz": list(spec["band"]),
        "mhd_weight": spec["mhd_weight"],
        "fold": fold,
        "stop_fold": stop,
        "train_folds": train_folds,
        "shots": by[fold],
        "stop_shots": by[stop],
        "train_shots": train_shots,
        "best_epoch": train.best_epoch(history),
        "epochs_run": len(history),
        "history": history,
        "config": asdict(config),
        "pilot": pilot,
        "labels_sha256": digest,
        "folds_sha256": folds_sha,
        "npz_sha256": _sha(buffer.getvalue()),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "git_sha": git_sha(),
        "made_at": _now(),
    }
    with atomic_path(record_file) as tmp:
        tmp.write_text(json.dumps(record, indent=1) + "\n")
    return record


def _number(value) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def choose_row(rows: list[dict]) -> tuple[dict, int]:
    """Deviation 11's rule over (candidate, threshold) rows with `mhd_weight`,
    `threshold`, `f1` and `fp_rate_mhd` (None when undefined): the row, and
    which branch chose it."""
    scored = [r for r in rows if r["f1"] is not None]
    if not scored:
        raise ValueError("no (candidate, threshold) has a defined F1")
    mhd = [r for r in scored if r["fp_rate_mhd"] is not None]
    first = [r for r in mhd if r["fp_rate_mhd"] <= MHD_FP_MAX]
    second = [r for r in mhd if r["f1"] >= F1_MIN]
    if first:
        branch, pool, primary = 1, first, lambda r: -r["f1"]
    elif second:
        branch, pool, primary = 2, second, lambda r: r["fp_rate_mhd"]
    else:
        branch, pool, primary = 3, scored, lambda r: -r["f1"]

    def key(r):
        cents = round(r["threshold"] * 100)  # hundredths: 0.45 and 0.55 tie
        return primary(r), r["mhd_weight"], abs(cents - 50), cents

    return min(pool, key=key), branch


def _load_fold(models: Path, candidate: str, k: int, expect: dict) -> tuple:
    """A fold's record and frames, after checking the record against `expect`."""
    record_file, npz_file = _record_paths(models, candidate, k)
    record = json.loads(record_file.read_text())
    data = npz_file.read_bytes()
    wanted = {**expect, "candidate": candidate, "fold": k, "npz_sha256": _sha(data)}
    for key, value in wanted.items():
        if record.get(key) != value:
            raise ValueError(f"{record_file}: {key} differs ({record.get(key)!r})")
    frames = []
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        if set(z.files) != {f"{c}{s}" for s in record["shots"] for c in "poms"}:
            raise ValueError(f"{npz_file}: its shots differ from {record_file}")
        for s in record["shots"]:
            frames.append(
                evaluate.ShotFrames(
                    s, z[f"o{s}"], z[f"m{s}"], z[f"s{s}"], {}, z[f"p{s}"]
                )
            )
    return record, frames


def _rows(candidate: str, weight: float, frames) -> list[dict]:
    rows = []
    for threshold in THRESHOLDS:
        for f in frames:
            f.said["ae_xpower"] = f.prob >= threshold
        cells = evaluate.cells(frames, "ae_xpower").sum(axis=0)
        mhd = evaluate.cells(frames, "ae_xpower", evaluate.mhd_absent).sum(axis=0)
        other = evaluate.cells(frames, "ae_xpower", evaluate.other_absent).sum(axis=0)
        rows.append(
            {
                "candidate": candidate,
                "mhd_weight": weight,
                "threshold": float(threshold),
                "f1": _number(stats.f1(cells)),
                "precision": _number(stats.precision(cells)),
                "recall": _number(stats.recall(cells)),
                "fp_rate_mhd": _number(evaluate.fp_rate(mhd)),
                "fp_rate_other": _number(evaluate.fp_rate(other)),
                "cells": [int(c) for c in cells],
                "mhd_cells": [int(c) for c in mhd],
            }
        )
    return rows


def run_choose(paths: Paths, models: Path, version: str) -> dict:
    """Pool the 15 fold records, apply the rule, write choice.json and frontier.md."""
    check_version(models, version)
    digest, _, folds, folds_sha = _checked_folds(paths, models, version)
    names = train.candidates(version)
    missing = [
        f"{name} fold {k}"
        for name in names
        for k in range(N_FOLDS)
        if not all(p.is_file() for p in _record_paths(models, name, k))
    ]
    total = len(names) * N_FOLDS
    if missing:
        raise FileNotFoundError(
            f"{cv_dir(models)}: {len(missing)} of {total} fold records missing: "
            + ", ".join(missing)
        )
    pilot = json.loads(_record_paths(models, next(iter(names)), 0)[0].read_text())
    pilot = pilot.get("pilot", 0)
    if pilot and not pilot_area(models, paths.runs):
        raise ValueError(f"{models}: pilot fold records outside {paths.runs}")
    by = _by_fold(folds, pilot)
    table, fold_info, best, sources, counted = [], {}, {}, {}, None
    for name, spec in names.items():
        frames, fold_info[name], best[name] = [], {}, []
        for k in range(N_FOLDS):
            expect = {
                "version": version,
                "labels_sha256": digest,
                "folds_sha256": folds_sha,
                "pilot": pilot,
                "shots": by[k],
                "stop_fold": stop_fold(k),
                "mhd_weight": spec["mhd_weight"],
                "band_khz": list(spec["band"]),
            }
            record, fold_frames = _load_fold(models, name, k, expect)
            frames += fold_frames
            best[name].append(record["best_epoch"])
            fold_info[name][str(k)] = {
                "best_epoch": record["best_epoch"],
                "epochs_run": record["epochs_run"],
                "stop_fold": record["stop_fold"],
            }
            record_file = _record_paths(models, name, k)[0]
            sources[f"{name}/fold{k}.json"] = _sha(record_file.read_bytes())
        if sorted(f.shot for f in frames) != sorted(s for v in by.values() for s in v):
            raise ValueError(f"{cv_dir(models) / name}: folds miss or repeat a shot")
        table += _rows(name, spec["mhd_weight"], frames)
        counts = {
            "shots": len(frames),
            "scored": int(sum(f.scored.sum() for f in frames)),
            "present": int(sum((f.scored & (f.owner == 1)).sum() for f in frames)),
            "mhd_absent": int(
                sum((f.scored & evaluate.mhd_absent(f)).sum() for f in frames)
            ),
        }
        if counted not in (None, counts):
            raise ValueError(f"{cv_dir(models) / name}: scores other frames")
        counted = counts
    row, branch = choose_row(table)
    epochs = sorted(best[row["candidate"]])
    final = int(epochs[len(epochs) // 2])
    if final < 1:
        raise ValueError(f"{row['candidate']}: no fold kept an epoch; median 0")
    choice = {
        "version": version,
        "candidate": row["candidate"],
        "threshold": row["threshold"],
        "branch": branch,
        "branch_text": BRANCHES[branch],
        "ties": TIES,
        "rule": "the ledger's Deviation 11",
        "chosen_row": row,
        "final_epochs": final,
        "best_epochs": best,
        "folds": fold_info,
        "frames": counted,
        "table": table,
        "labels_sha256": digest,
        "folds_sha256": folds_sha,
        "sources": sources,
        "pilot": pilot,
        "git_sha": git_sha(),
        "made_at": _now(),
    }
    file = cv_dir(models) / "choice.json"
    if file.exists() and not pilot_area(models, paths.runs):
        saved = json.loads(file.read_text())
        changed = [k for k in DECISION if saved.get(k) != choice[k]]
        if changed:
            raise FileExistsError(
                f"{file}: a saved choice differs in {', '.join(changed)}; "
                "a new choice is a new version"
            )
        return saved
    with atomic_path(cv_dir(models) / "frontier.md") as tmp:
        tmp.write_text(frontier_md(choice))
    with atomic_path(file) as tmp:
        tmp.write_text(json.dumps(choice, indent=1) + "\n")
    return choice


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def frontier_md(choice: dict) -> str:
    """The pooled out-of-fold table, the rule and what it chose."""
    n = choice["frames"]
    epochs = sorted(choice["best_epochs"][choice["candidate"]])
    lines = [
        f"# AE {choice['version']}: the cross-validated choice",
        "",
        (
            f"Out-of-fold frames of {n['shots']} shots in {N_FOLDS} folds "
            f"(each predicted by a model that neither trained nor stopped on it): "
            f"{n['scored']} frames of 0-2 s, {n['present']} present, "
            f"{n['mhd_absent']} MHD frames the owner called absent. Labels "
            f"sha256 {choice['labels_sha256'][:12]}..."
        ),
        "",
        "The rule (the ledger's Deviation 11), over every (candidate, threshold):",
        "",
        *(f"{k}. {text}." for k, text in BRANCHES.items()),
        "",
        f"{TIES[0].upper()}{TIES[1:]}.",
        "",
        (
            f"**Chosen: {choice['candidate']} at {choice['threshold']:.2f}, by "
            f"branch {choice['branch']}**; the final model trains on all "
            f"{n['shots']} shots for {choice['final_epochs']} epochs, the median "
            f"of its folds' best epochs {epochs}."
        ),
        "",
        "| candidate | fold best epochs |",
        "|---|---|",
        *(f"| {c} | {e} |" for c, e in choice["best_epochs"].items()),
        "",
        (
            "| candidate | threshold | F1 | precision | recall | MHD FP | "
            "other-absent FP |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for r in choice["table"]:
        mark = (
            " (chosen)"
            if (r["candidate"], r["threshold"])
            == (choice["candidate"], choice["threshold"])
            else ""
        )
        lines.append(
            f"| {r['candidate']}{mark} | {r['threshold']:.2f} | {_fmt(r['f1'])} | "
            f"{_fmt(r['precision'])} | {_fmt(r['recall'])} | "
            f"{_fmt(r['fp_rate_mhd'])} | {_fmt(r['fp_rate_other'])} |"
        )
    return "\n".join([*lines, ""])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--version", default="v2")
    p.add_argument("--models", type=Path, help="default models/ae_xpower/<version>")
    p.add_argument("--folds", action="store_true", help="write cv/folds.csv")
    p.add_argument("--choose", action="store_true", help="after all 15 fold tasks")
    p.add_argument("--candidate", help="one of the version's candidates")
    p.add_argument("--fold", type=int, help="0-4: the fold this task predicts")
    p.add_argument("--epochs", type=int, default=train.TrainConfig.epochs)
    p.add_argument(
        "--pilot",
        type=int,
        default=0,
        help="5-20 shots (N // 5 a fold), 2 epochs, to runs/ae_xpower/pilot/<version>",
    )
    args = p.parse_args(argv)
    task = args.candidate is not None or args.fold is not None
    if args.folds + args.choose + task != 1:
        p.error("give one of --folds, --choose, or --candidate with --fold")
    if task and (args.candidate is None or args.fold is None):
        p.error("a fold task needs both --candidate and --fold")
    if args.pilot and not 5 <= args.pilot <= 20:
        p.error("a pilot is 5 to 20 shots")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    paths = Paths.from_env()
    pilot_models = paths.runs / "ae_xpower" / "pilot" / args.version
    models = args.models or (
        pilot_models if args.pilot else model_dir(paths, args.version)
    )
    try:
        if args.folds:
            print(json.dumps(write_folds(paths, models, args.version)))
        elif args.choose:
            choice = run_choose(paths, models, args.version)
            print(
                f"chose {choice['candidate']} at {choice['threshold']} by branch "
                f"{choice['branch']}; final epochs {choice['final_epochs']}"
            )
        else:
            record = run_fold(
                paths,
                models,
                candidate=args.candidate,
                fold=args.fold,
                version=args.version,
                pilot=args.pilot,
                epochs=args.epochs,
                log=lambda m: print(m, flush=True),
            )
            print(
                f"fold {args.fold} of {args.candidate}: best epoch "
                f"{record['best_epoch']}"
            )
    except (OSError, ValueError, KeyError) as error:
        p.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
