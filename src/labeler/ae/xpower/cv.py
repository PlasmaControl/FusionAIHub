"""Five-fold cross-validation, and the MHD-aware choice (the ledger's Deviation 11).

    python -m labeler.ae.xpower.cv --folds                       # once, first
    python -m labeler.ae.xpower.cv --candidate NAME --fold K     # 3 x 5 tasks
    python -m labeler.ae.xpower.cv --choose                      # after all 15
    python -m labeler.ae.xpower.cv --candidate NAME --fold K --seed S  # seed study
    python -m labeler.ae.xpower.cv --candidate NAME --fold K --ablate  # ablation

all for `--version` (default v2) in `--models` (default
`$LABELER_ROOT/models/ae_xpower/<version>`), under `cv/`.

**Labels.** Only the version's snapshot (`labeler.ae.xpower.read_snapshot`),
never the owner's live table; every product records its sha256.

**Folds.** `data.make_split` as v1 does it (test = SELDNet's `valid`); the rest,
train and validation alike, is the pool (120 shots on the real data). The pool,
sorted, is permuted with seed 20260923 and dealt in turn into five folds, so
they differ in size by at most one shot (24 each on the real data). `--folds`
writes `cv/folds.csv` (shot, split, fold; the test shots have no fold) and
`cv/folds.json` before any training. Every later step (the fold tasks, the
choice, `train --from-cv` and `evaluate --test`) refuses unless the file holds
exactly the folds the snapshot and TokEye's masks (`xpower.tokeye_masks`, a
whole-window version's `ae/masks-full`) give then (`checked_folds`), and the
choice, the final model and `chosen.json` name the file's sha256.

**A fold task.** Fold K is predicted; fold (K + 1) mod 5 stops the training
(`train.fit`: validation F1 at 0.5, patience 10, up to 60 epochs: v1's
`TrainConfig` with the candidate's MHD weight, `fold_config`; there is no
`--epochs`); the other three train. Fold K's shots never train or stop the
model that predicts them. It writes `cv/<candidate>/fold<K>.npz`, per fold-K
shot the 0-2 s frames the test scores (`evaluate.shot_frames` with the shot's
source-table label, so only inside that table's window; a fold shot the table
lacks is refused before training): `p<shot>` P(AE), `o<shot>` the owner's
states, `m<shot>` MHD, `s<shot>` scored. Then `fold<K>.json`, the record that
marks the task done, with the source table's sha256 and the frames its window
dropped. A whole-window version's (v3's) frames are the owner's whole window
instead (`evaluate.whole_frames`, from TokEye's `ae/masks-full`), which no source
table filters: nothing is refused for lacking one, and the record's
`source_sha256` is null and `source_dropped` 0.

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
refuses while any of the 15 records is missing, or one differs from what its
task gives (its shots, stop fold, snapshot, folds, source table and
`TrainConfig`), naming them. It writes `cv/choice.json` (the choice, its branch,
the whole table, the frames and how many the source window dropped, and each
fold's best epoch; the final model trains for the median of the chosen
candidate's five) and `cv/frontier.md`. Run again, it checks the saved choice
and changes nothing.

**Pilots.** `--pilot N` (5-20 shots): each fold's first N // 5 shots, 2 epochs
(`PILOT_EPOCHS`), into `runs/ae_xpower/pilot/<version>`, where a record can be
replaced.

**The seed study** (post hoc: no decision depends on it). `--seed S`, a
`TrainConfig.seed` other than the default 20260923 (`STUDY_SEEDS`: 20260924-26),
trains the chosen candidate (`chosen.json`'s) in its fold, as `cv/` did, with
only the seed changed. It reads the version's own `cv/folds.csv` (checked as
every step checks it, and its sha256 against `chosen.json`'s `folds_sha256`)
and writes only under `runs/ae_xpower/seeds/<version>/seed<S>/` (`seed_dir`; a
pilot's under its `pilot/`): a seed aimed at any other directory, the version's
`cv/` or the pilot's `runs/ae_xpower/pilot/<version>` among them, is refused, and
so is a record already there, which a seeded run never replaces. Each record
carries its `seed`. `seed_study` pools each seed's five folds, as the choice
pools them, into one row at the chosen threshold, for `posthoc`.

**The ablation** (post hoc, cross-validation only: no decision depends on it,
and no test shot is read). `--ablate` trains `--candidate`, one of
`ABLATIONS[version]` and never one of the version's own, in its fold, as `cv/`
trains the version's own: the version's `cv/folds.csv` (checked as every step
checks it, and its sha256 against `chosen.json`'s and `cv/choice.json`'s), the
same stop fold, epoch cap and `TrainConfig`, and the version's data (its label
snapshot, TokEye record and frames); only the band and MHD weight are the
candidate's. It writes only under
`runs/ae_xpower/ablation/<version>/<candidate>/` (`ablation_dir`; a pilot's
under its `pilot/`), refuses any other directory (the version's `cv/` among
them) and a record already there, and marks each record `ablation`.
`candidate_frames` reads one back as `--choose` reads a record;
`labeler.ae.xpower.diagnosis --ablation` puts them beside the versions' own.

**Inputs.** A whole-window version's fold records and choice name
`ae/masks-full` and `ae/dataset-full` by their manifests' sha256 (`inputs`,
`labeler.ae.full.inputs_identity`), and every read of a record refuses one
naming others than the manifests list now; a record naming none (v3's own,
made before the manifests) stands on the version's `inputs.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import resource
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from ...config import Paths, atomic_path, git_sha
from ...scoring import stats
from ..full import check_inputs, inputs_identity
from . import (
    CV_VERSIONS,
    EVENT,
    LABEL_SNAPSHOTS,
    WHOLE_WINDOW_VERSIONS,
    evaluate,
    event_dir,
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
#: The seed study's training seeds, beside the default (`data.SEED`).
STUDY_SEEDS = (20260924, 20260925, 20260926)
#: The ablation's candidates, per version: outside the version's own, each the
#: other version's on its band. On v3's whole-window folds, v2's band 80-250 kHz
#: at v2's two lower weights; on v2's 0-2 s folds, v3's 0-250 kHz at weight 3.
ABLATIONS = {
    "v2": {"band0-mhd3": train.candidates("v3")["band0-mhd3"]},
    "v3": {n: train.candidates("v2")[n] for n in ("band80-mhd3", "band80-mhd10")},
}
#: The fields a saved choice must repeat when `--choose` runs again.
DECISION = ("candidate", "threshold", "branch", "table", "final_epochs", "sources")


def cv_dir(models: Path) -> Path:
    return models / "cv"


def seed_dir(paths: Paths, version: str, seed: int, pilot: int = 0) -> Path:
    """The only models directory a fold trained with a non-default `seed` writes
    to: `runs/ae_xpower/seeds/<version>/seed<S>`, a pilot's under its `pilot/`."""
    out = paths.runs / "ae_xpower" / "seeds" / version / f"seed{seed}"
    return out / "pilot" if pilot else out


def ablation_dir(paths: Paths, version: str, candidate: str, pilot: int = 0) -> Path:
    """The only models directory an ablation fold writes to:
    `runs/ae_xpower/ablation/<version>/<candidate>`, a pilot's under its `pilot/`;
    its records go in its `cv/<candidate>/`."""
    out = paths.runs / "ae_xpower" / "ablation" / version / candidate
    return out / "pilot" if pilot else out


def ablation_spec(version: str, candidate: str) -> dict:
    """The ablation candidate `candidate` of `version`, {band, mhd_weight};
    refused for one of the version's own candidates or one not in ABLATIONS."""
    if candidate in train.candidates(version):
        raise ValueError(
            f"{candidate} is one of version {version}'s own candidates; its folds "
            "are in the version's cv/"
        )
    names = ABLATIONS.get(version, {})
    if candidate not in names:
        raise ValueError(
            f"candidate {candidate} is not one of version {version}'s ablations: "
            + (", ".join(names) or "none")
        )
    return names[candidate]


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
    split = make_split(saved, seldnet_split(tokeye_masks(paths, version)))
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


class Folds(NamedTuple):
    """`cv/folds.csv`, checked against the snapshot and TokEye's masks as they are
    now: the snapshot's sha256 and labels, the split (test = SELDNet's `valid`),
    each pool shot's fold, and the file's sha256, which every later product names.
    """

    labels_sha256: str
    saved: dict
    split: dict[int, str]
    folds: dict[int, int]
    sha256: str


def checked_folds(paths: Paths, models: Path, version: str) -> Folds:
    """The folds, after checking `cv/folds.csv` byte for byte against the ones the
    snapshot and TokEye's masks give now; every step after `--folds` reads them
    through here (the fold tasks, the choice, the final model and its test)."""
    digest, saved, split, folds, text = _expected(paths, version)
    file = cv_dir(models) / "folds.csv"
    if not file.is_file():
        raise FileNotFoundError(f"{file}: no folds; run --folds first")
    data = file.read_bytes()
    if data != text.encode():
        raise ValueError(
            f"{file}: differs from the folds the snapshot and TokEye's masks give "
            "now; folds are fixed before training"
        )
    record = json.loads((cv_dir(models) / "folds.json").read_text())
    if record.get("labels_sha256") != digest:
        raise ValueError(f"{cv_dir(models) / 'folds.json'}: another label snapshot")
    return Folds(digest, saved, split, folds, _sha(data))


def _by_fold(folds: dict[int, int], pilot: int) -> dict[int, list[int]]:
    by = {k: sorted(s for s, f in folds.items() if f == k) for k in range(N_FOLDS)}
    if pilot:
        by = {k: shots[: pilot // N_FOLDS] for k, shots in by.items()}
    return by


def fold_config(spec: dict, pilot: int) -> train.TrainConfig:
    """A fold task's training: v1's `TrainConfig` with the candidate's MHD weight
    (`train.cv_config`, which the final model's is too); a pilot's, for
    `PILOT_EPOCHS`. `--choose` refuses a record trained otherwise."""
    return train.cv_config(spec, PILOT_EPOCHS if pilot else None)


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
    seed: int = SEED,
    ablate: bool = False,
    log=print,
) -> dict:
    """Train on three folds, stop on fold `fold` + 1, predict fold `fold`; with
    another `seed`, the seed study's fold, into `seed_dir` only, never replaced;
    with `ablate`, an ablation candidate's fold, into `ablation_dir` only, never
    replaced."""
    seeded = seed != SEED
    if seeded and ablate:
        raise ValueError("an ablation trains with the default seed only")
    own, what = None, ""  # a seeded or ablation fold's only directory
    if seeded:
        own, what = seed_dir(paths, version, seed, pilot), f"trained with seed {seed}"
    elif ablate:
        own, what = ablation_dir(paths, version, candidate, pilot), "of the ablation"
    if own is not None and models.resolve() != own.resolve():
        raise ValueError(f"{models}: a fold {what} writes only under {own}")
    # The version's own cv/ folds, which a seeded or ablation fold only reads.
    folds_from = models if own is None else model_dir(paths, version)
    check_version(folds_from, version)
    if ablate:
        spec = ablation_spec(version, candidate)
    else:
        spec = train.candidate_spec(version, candidate)
    if not 0 <= fold < N_FOLDS:
        raise ValueError(f"--fold must be 0 to {N_FOLDS - 1}")
    in_runs = pilot_area(models, paths.runs)
    if pilot and not in_runs:
        raise ValueError(f"{models}: a pilot writes under {paths.runs}")
    record_file, npz_file = _record_paths(models, candidate, fold)
    if own is not None and (record_file.exists() or npz_file.exists()):
        raise FileExistsError(
            f"{record_file}: a fold {what} is trained once; nothing is replaced"
        )
    if record_file.exists() and not in_runs:
        raise FileExistsError(f"{record_file}: a fold is trained once")
    digest, saved, _, folds, folds_sha = checked_folds(paths, folds_from, version)
    if seeded:
        _chosen_for_seeds(folds_from, candidate, folds_sha)
    if ablate:
        _chosen_folds(folds_from, folds_sha)
    by = _by_fold(folds, pilot)
    whole = version in WHOLE_WINDOW_VERSIONS
    inputs = inputs_identity(paths, version)  # {} unless whole-window
    if whole:  # the owner's whole windows, which no source table filters
        source, source_sha = None, evaluate.source_sha(paths, version)
    else:
        source = evaluate.source_table(paths)  # the test's source-window filter
        source_sha = source.sha256
        lacking = [s for s in by[fold] if s not in source.labels]
        if lacking:
            raise ValueError(
                f"{source.path or event_dir(paths)}: no source-table label for fold "
                f"{fold}'s shots {', '.join(map(str, lacking))}; the test scores only "
                "frames inside that table's windows"
            )
    stop = stop_fold(fold)
    train_folds = [f for f in range(N_FOLDS) if f not in (fold, stop)]
    train_shots = [s for f in train_folds for s in by[f]]

    def shot(s):
        rows = store_rows(paths.spectrogram_file(EVENT, s))
        masks = tokeye_masks(paths, version)
        return load_shot(s, saved[s], rows, masks, band=spec["band"])

    config = replace(fold_config(spec, pilot), seed=seed)
    model, history, _ = train.fit(
        [shot(s) for s in train_shots], [shot(s) for s in by[stop]], config, log
    )
    blob = {"band_khz": list(spec["band"]), "threshold": 0.5, "candidate": candidate}
    arrays, dropped = {}, 0
    for s in by[fold]:
        if whole:
            frames = evaluate.whole_frames(
                s, paths=paths, label=saved[s], model=model, blob=blob, version=version
            )
        else:
            frames = evaluate.shot_frames(
                s,
                paths=paths,
                label=saved[s],
                model=model,
                blob=blob,
                source=source.labels[s],
            )
        dropped += frames.source_dropped
        arrays[f"p{s}"] = np.asarray(frames.prob, dtype=np.float32)
        arrays[f"o{s}"] = np.asarray(frames.owner, dtype=np.int8)
        arrays[f"m{s}"] = np.asarray(frames.mhd, dtype=bool)
        arrays[f"s{s}"] = np.asarray(frames.scored, dtype=bool)
    if inputs_identity(paths, version) != inputs:
        raise ValueError(
            f"{record_file}: ae/masks-full or ae/dataset-full changed during the fold"
        )
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
        "seed": seed,
        "pilot": pilot,
        "labels_sha256": digest,
        "folds_sha256": folds_sha,
        "source_sha256": source_sha,
        "source_dropped": dropped,
        "npz_sha256": _sha(buffer.getvalue()),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "git_sha": git_sha(),
        "made_at": _now(),
    }
    if whole:
        record["inputs"] = inputs
    if ablate:
        record["ablation"] = True
    with atomic_path(record_file) as tmp:
        tmp.write_text(json.dumps(record, indent=1) + "\n")
    return record


def _chosen_for_seeds(models: Path, candidate: str, folds_sha: str) -> dict:
    """`chosen.json`, when the seed study may train `candidate` on these folds:
    the chosen candidate, on the folds (by sha256) it was chosen from."""
    file = models / "chosen.json"
    chosen = json.loads(file.read_text())
    if chosen.get("folds_sha256") != folds_sha:
        raise ValueError(
            f"{cv_dir(models) / 'folds.csv'}: sha256 {folds_sha} differs from "
            f"{file}'s folds_sha256 {chosen.get('folds_sha256')}"
        )
    if candidate != chosen.get("candidate"):
        raise ValueError(
            f"{file}: the seed study trains the chosen candidate "
            f"{chosen.get('candidate')}, not {candidate}"
        )
    return chosen


def _chosen_folds(models: Path, folds_sha: str) -> None:
    """Refuse folds (by sha256) other than the ones `chosen.json` and
    `cv/choice.json` were made from."""
    for file in (models / "chosen.json", cv_dir(models) / "choice.json"):
        named = json.loads(file.read_text()).get("folds_sha256")
        if named != folds_sha:
            raise ValueError(
                f"{cv_dir(models) / 'folds.csv'}: sha256 {folds_sha} differs from "
                f"{file}'s folds_sha256 {named}"
            )


def _study_inputs(paths: Paths, version: str) -> tuple:
    """The version's models directory, `chosen.json`, and its checked folds."""
    models = model_dir(paths, version)
    check_version(models, version)
    folds = checked_folds(paths, models, version)
    chosen = json.loads((models / "chosen.json").read_text())
    _chosen_for_seeds(models, chosen["candidate"], folds.sha256)
    return models, chosen, folds


def _lacking(models: Path, name: str) -> list[int]:
    return [
        k
        for k in range(N_FOLDS)
        if not all(p.is_file() for p in _record_paths(models, name, k))
    ]


def oof_frames(paths: Paths, version: str, seed: int = SEED) -> list:
    """The chosen candidate's out-of-fold frames of all five folds trained with
    `seed` (the default: `cv/`'s; another: `seed_dir`'s), each fold record
    checked as `--choose` checks it, with the seed in its `TrainConfig`."""
    models, chosen, folds = _study_inputs(paths, version)
    name = chosen["candidate"]
    where = models if seed == SEED else seed_dir(paths, version, seed)
    lacking = _lacking(where, name)
    if lacking:
        raise FileNotFoundError(
            f"{cv_dir(where) / name}: seed {seed}'s folds {lacking} are missing"
        )
    spec = train.candidate_spec(version, name)
    return _fold_frames(paths, version, folds, where, name, spec, seed=seed)


def _fold_frames(
    paths: Paths,
    version: str,
    folds: Folds,
    where: Path,
    name: str,
    spec: dict,
    *,
    seed: int = SEED,
    extra: dict | None = None,
) -> list:
    """Candidate `name`'s out-of-fold frames of all five folds under `where`,
    each record checked as `--choose` checks it (and against `extra`), its
    inputs among them (`check_inputs`)."""
    config = asdict(replace(fold_config(spec, 0), seed=seed))
    config = json.loads(json.dumps(config))  # as a record holds it
    source_sha = evaluate.source_sha(paths, version)
    identity = inputs_identity(paths, version)
    by = _by_fold(folds.folds, 0)
    frames = []
    for k in range(N_FOLDS):
        expect = {
            "version": version,
            "labels_sha256": folds.labels_sha256,
            "folds_sha256": folds.sha256,
            "source_sha256": source_sha,
            "pilot": 0,
            "shots": by[k],
            "stop_fold": stop_fold(k),
            "mhd_weight": spec["mhd_weight"],
            "band_khz": list(spec["band"]),
            "config": config,
            **(extra or {}),
        }
        record, fold_frames = _load_fold(where, name, k, expect)
        file = _record_paths(where, name, k)[0]
        check_inputs(paths, version, record.get("inputs"), file, identity=identity)
        frames += fold_frames
    return frames


def candidate_frames(paths: Paths, version: str, candidate: str) -> list:
    """A candidate's out-of-fold frames of all five folds on the version's folds
    (checked, and by sha256 against `chosen.json`'s and the choice's), each
    record checked as `--choose` checks it: one of the version's own from its
    `cv/`, an ablation candidate's (`ABLATIONS`) from `ablation_dir`, as an
    ablation's."""
    models = model_dir(paths, version)
    check_version(models, version)
    folds = checked_folds(paths, models, version)
    _chosen_folds(models, folds.sha256)
    if candidate in train.candidates(version):
        spec, where, extra = train.candidate_spec(version, candidate), models, None
    else:
        spec = ablation_spec(version, candidate)
        where, extra = ablation_dir(paths, version, candidate), {"ablation": True}
    lacking = _lacking(where, candidate)
    if lacking:
        raise FileNotFoundError(f"{cv_dir(where) / candidate}: folds {lacking} missing")
    return _fold_frames(paths, version, folds, where, candidate, spec, extra=extra)


def seed_study(paths: Paths, version: str, seeds=STUDY_SEEDS) -> dict:
    """Per training seed, the chosen candidate's out-of-fold frames (`oof_frames`)
    pooled as the choice pools them, at the chosen threshold: F1, precision,
    recall and MHD FP, and whether F1 meets branch 2's 0.90. A study seed with no
    fold record is "not run", one with some but not all "incomplete", and
    neither is a row. With two rows or more, each metric's range and sample
    standard deviation (ddof 1)."""
    models, chosen, folds = _study_inputs(paths, version)
    name, threshold = chosen["candidate"], float(chosen["threshold"])
    weight = train.candidate_spec(version, name)["mhd_weight"]
    rows, not_run, incomplete = [], [], {}
    for seed in (SEED, *seeds):
        where = models if seed == SEED else seed_dir(paths, version, seed)
        lacking = _lacking(where, name)
        if lacking and seed != SEED:  # cv/'s own folds must all be there
            if len(lacking) == N_FOLDS:
                not_run.append(seed)
            else:
                incomplete[str(seed)] = lacking
            continue
        frames = oof_frames(paths, version, seed)
        (row,) = _rows(name, weight, frames, (threshold,))
        rows.append(
            {
                "seed": seed,
                "records": str(cv_dir(where) / name),
                **{k: row[k] for k in ("f1", "precision", "recall", "fp_rate_mhd")},
                "meets_f1_min": row["f1"] is not None and row["f1"] >= F1_MIN,
                "cells": row["cells"],
                "mhd_cells": row["mhd_cells"],
            }
        )
    spread = {}
    for key in ("f1", "precision", "recall", "fp_rate_mhd"):
        values = [r[key] for r in rows if r[key] is not None]
        if len(values) > 1:
            spread[key] = {
                "range": float(max(values) - min(values)),
                "std": float(np.std(values, ddof=1)),
            }
    return {
        "version": version,
        "candidate": name,
        "threshold": threshold,
        "f1_min": F1_MIN,
        "default_seed": SEED,
        "seeds": list(seeds),
        "rows": rows,
        "not_run": not_run,
        "incomplete": incomplete,
        "spread": spread,
        "labels_sha256": folds.labels_sha256,
        "folds_sha256": folds.sha256,
    }


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


def _rows(candidate: str, weight: float, frames, thresholds=THRESHOLDS) -> list[dict]:
    rows = []
    for threshold in thresholds:
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
    digest, _, _, folds, folds_sha = checked_folds(paths, models, version)
    source_sha = evaluate.source_sha(paths, version)
    identity = inputs_identity(paths, version)  # {} unless whole-window
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
        frames, records, fold_info[name], best[name] = [], [], {}, []
        # As a record holds it: v1's TrainConfig, the weight, a pilot's epochs.
        config = json.loads(json.dumps(asdict(fold_config(spec, pilot))))
        for k in range(N_FOLDS):
            expect = {
                "version": version,
                "labels_sha256": digest,
                "folds_sha256": folds_sha,
                "source_sha256": source_sha,
                "pilot": pilot,
                "shots": by[k],
                "stop_fold": stop_fold(k),
                "mhd_weight": spec["mhd_weight"],
                "band_khz": list(spec["band"]),
                "config": config,
            }
            record, fold_frames = _load_fold(models, name, k, expect)
            record_file = _record_paths(models, name, k)[0]
            check_inputs(
                paths, version, record.get("inputs"), record_file, identity=identity
            )
            frames += fold_frames
            records.append(record)
            best[name].append(record["best_epoch"])
            fold_info[name][str(k)] = {
                "best_epoch": record["best_epoch"],
                "epochs_run": record["epochs_run"],
                "stop_fold": record["stop_fold"],
            }
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
            # Frames the test's source-window filter dropped (outside a window).
            "source_dropped": sum(int(r["source_dropped"]) for r in records),
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
        "source_sha256": source_sha,
        "sources": sources,
        "pilot": pilot,
        "git_sha": git_sha(),
        "made_at": _now(),
    }
    if version in WHOLE_WINDOW_VERSIONS:
        choice["inputs"] = identity
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
    if choice["version"] in WHOLE_WINDOW_VERSIONS:
        window = "the owner's whole windows"
        dropped = "none dropped, as no source table filters a whole window"
    else:
        window = "0-2 s"
        dropped = (
            f"{n['source_dropped']} frames outside the source table's window "
            "dropped, as the test drops them"
        )
    lines = [
        f"# AE {choice['version']}: the cross-validated choice",
        "",
        (
            f"Out-of-fold frames of {n['shots']} shots in {N_FOLDS} folds "
            f"(each predicted by a model that neither trained nor stopped on it): "
            f"{n['scored']} frames of {window}, {n['present']} present, "
            f"{n['mhd_absent']} MHD frames the owner called absent; {dropped}. "
            f"Labels sha256 {choice['labels_sha256'][:12]}..."
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
            f"{n['shots']} shots for {train.epochs_text(choice['final_epochs'])}, "
            f"the median of its folds' best epochs {epochs}."
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
    p.add_argument(
        "--pilot",
        type=int,
        default=0,
        help="5-20 shots (N // 5 a fold), 2 epochs, to runs/ae_xpower/pilot/<version>",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=(
            f"a fold task's TrainConfig seed (default {SEED}); another is the seed "
            "study's, into runs/ae_xpower/seeds/<version>/seed<S>"
        ),
    )
    p.add_argument(
        "--ablate",
        action="store_true",
        help=(
            "a fold task of an ablation candidate (cv.ABLATIONS) on the version's "
            "folds, into runs/ae_xpower/ablation/<version>/<candidate>"
        ),
    )
    args = p.parse_args(argv)
    task = args.candidate is not None or args.fold is not None
    if args.folds + args.choose + task != 1:
        p.error("give one of --folds, --choose, or --candidate with --fold")
    if task and (args.candidate is None or args.fold is None):
        p.error("a fold task needs both --candidate and --fold")
    if args.pilot and not 5 <= args.pilot <= 20:
        p.error("a pilot is 5 to 20 shots")
    if args.seed != SEED and not task:
        p.error("--seed is for a fold task")
    if args.ablate and not task:
        p.error("--ablate is for a fold task")
    if args.ablate and args.seed != SEED:
        p.error("an ablation trains with the default seed only")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    paths = Paths.from_env()
    pilot_models = paths.runs / "ae_xpower" / "pilot" / args.version
    models = args.models or (
        seed_dir(paths, args.version, args.seed, args.pilot)
        if args.seed != SEED
        else ablation_dir(paths, args.version, args.candidate, args.pilot)
        if args.ablate
        else pilot_models
        if args.pilot
        else model_dir(paths, args.version)
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
                seed=args.seed,
                ablate=args.ablate,
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
