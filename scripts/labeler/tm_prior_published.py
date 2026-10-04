#!/usr/bin/env python
"""The published tearing models as they are: against their own labels, and against ours.

Reads the per-shot inputs and published outputs `tm_detector_inputs.py` wrote
(``$TM_ROOT/detector_inputs``). Published weights remain fixed; every main F1 row
uses a threshold chosen on the same cohort-wide CV inner validation shots as the
learned rows. Fixed published thresholds are explicitly labelled extra results.

* ``--setting legacy``: each model against the label it was built for, on shots it was
  not trained on where that can be told.

  - `tm-onsetcnn` against Seo's archived ``tm_label`` (the label of its training store;
    its training shots are not on disk, so no row can be called held out, and the
    numbers are an upper bound), on the shots the archive's rows can be placed in;
    AUROC, AUPRC and F1 at the published 0.5.
  - `tm-dsm` against an onset within 250 ms / 500 ms / 1 s on the rows before the
    survival label's first onset (``raw/tm_labels.h5``), on shots outside the model's
    ``training_shots.txt`` that the survival labels cover; F1 at the model's own
    default survival alarm level, 0.7, equivalent to risk 0.3.

* ``--setting tokamak-si``: the unretrained published outputs against the whole-interval
  labels at 10 ms bins. The CNN's score at ``t`` is read as the mode's presence at
  ``t + 25 ms``; the DSM's risk of an onset within a horizon is the score as it is,
  at a fold-tuned risk threshold. ``--split dev`` comprises 450 development shots.

Every number is written with its shots, bins and threshold to
``$LABELER_ROOT/round4/tm/results/``, with 95 % shot-bootstrap intervals.

    PYTHONPATH=$PWD/src LABELER_ROOT=<root> pixi run --frozen --no-install \\
        -e labelmaker python scripts/labeler/tm_prior_published.py --setting tokamak-si
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for entry in (REPO / "src", Path(__file__).resolve().parent):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from tm_cv_plan import load_plan

from labeler.tearing import detectors, scoring

TM = (
    Path(os.environ.get("LABELER_ROOT", "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker"))
    / "round4/tm"
)
INPUTS = Path(os.environ.get("TM_ROOT", TM)) / "detector_inputs"
RESULTS = Path(os.environ.get("TM_ROOT", TM)) / "results"
CATALOG = REPO / "data/events/catalog"
LABELS = (
    REPO / "data/events/neoclassical_tearing_mode/extend_tm_interval/tm_interval.csv"
)
SURVIVAL = REPO / "data/events/neoclassical_tearing_mode/raw/tm_labels.h5"
MAIN = Path("/scratch/gpfs/nc1514/FusionAIHub")
CNN = "d3d_tearing_onset_cnn1d"
DSM = "d3d_tearing_time_to_event_dsm"
HORIZON_NAMES = ("250ms", "500ms", "1s")

#: Model name: slug, output column (None: one value per row), shift (ms), threshold.
SI_MODELS = {
    "tm-onsetcnn": (CNN, None, detectors.CNN_SHIFT_MS, detectors.CNN_THRESHOLD),
    **{
        f"tm-dsm-{h}": (DSM, i, 0.0, detectors.DSM_THRESHOLD)
        for i, h in enumerate(HORIZON_NAMES)
    },
}
THRESHOLD_SOURCE = (
    "published: the CNN card's F1 convention (tm_prob >= 0.5) and the DSM card's "
    "upstream metrics_helpers.py:85 default survival <= 0.7, equivalent to "
    "risk >= 0.3; "
    "the previous benchmark incorrectly applied 0.7 to risk"
)
CV_THRESHOLD_SOURCE = (
    "F1-maximising thresholds on shared cohort-450 fold inner validation shots only; "
    "cohort split=test excluded; same fold and inner shot roles for every model"
)


def development_shots():
    cohort = pd.read_csv(CATALOG / "cohort.csv")
    return sorted(int(s) for s in cohort[cohort.split != "test"].shot)


def tune(y, valid, score):
    _, splits = load_plan(development_shots())
    return scoring.cv_thresholds(
        development_shots(),
        y,
        valid,
        score,
        edges=np.linspace(0.0, 1.0, 1001),
        splits=splits,
    )


def git_sha() -> str:
    try:
        from labeler.config import git_sha as sha

        return sha()
    except Exception:  # noqa: BLE001
        return "unknown"


def load(slug: str, shot: int):
    path = INPUTS / slug / f"{shot}.npz"
    if not path.is_file():
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def write(name: str, record: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(
        json.dumps(record, indent=1, default=float) + "\n", encoding="utf-8"
    )
    print(path)
    return path


def dsm_training_shots() -> frozenset[int]:
    """The shots the published DSM was trained on (its listed `training_shots.txt`)."""
    from labeler.models import registry

    return registry.load_adapter(DSM).training_shots


def si_bins(slug, column, shift, shots, cohort, by_shot, table):
    """`(score, y, valid, used, missing)` per shot on the 10 ms bins of its window."""
    windows = cohort.set_index("shot")[["window_start_ms", "window_end_ms"]]
    score, y, valid, used, missing = {}, {}, {}, [], []
    for shot in shots:
        z = load(slug, shot)
        if z is None:
            missing.append(shot)
            continue
        win = windows.loc[shot]
        centres = scoring.bin_centres(
            (float(win.window_start_ms), float(win.window_end_ms))
        )
        y[shot], valid[shot] = scoring.label_bins(
            by_shot.get(shot, table.iloc[:0]), centres
        )
        raw = z["tm_prob"] if column is None else z["risk"][:, column]
        score[shot] = detectors.rows_to_bins(
            z["t_s"] * 1000.0, z["valid"], raw, centres, shift_ms=shift
        )
        used.append(shot)
    return score, y, valid, used, missing


def run_tokamak_si(args) -> None:
    cohort = pd.read_csv(CATALOG / "cohort.csv")
    dev = sorted(int(s) for s in cohort[cohort.split != "test"].shot)
    if args.split != "dev":
        raise ValueError("round-one regeneration scores development only")
    shots = dev
    table = pd.read_csv(LABELS)
    by_shot = {s: g for s, g in table.groupby("shot")}
    for name, (slug, column, shift, threshold) in SI_MODELS.items():
        score, y, valid, used, missing = si_bins(
            slug, column, shift, shots, cohort, by_shot, table
        )
        tuned, fold_info = tune(y, valid, score)
        res = {
            label: scoring.evaluate(
                used, y, valid, score, thr, n=args.bootstrap, seed=0
            )
            for label, thr in (("fixed_published", threshold), ("cv_tuned", tuned))
        }
        # the DSM's training shots are listed; scored apart, the rest are held out
        trained = dsm_training_shots() if slug == DSM else frozenset()
        outside = [s for s in used if s not in trained]
        held_out = (
            {
                label: scoring.evaluate(
                    outside, y, valid, score, thr, n=args.bootstrap, seed=0
                )
                for label, thr in (("fixed_published", threshold), ("cv_tuned", tuned))
            }
            if trained
            else None
        )
        write(
            f"tm_prior_published_{name}_tokamak-si_{args.split}",
            {
                "model": name,
                "slug": slug,
                "setting": "tokamak-si, unretrained published outputs",
                "split": args.split,
                "git_sha": git_sha(),
                "labels": str(LABELS.relative_to(REPO)),
                "labels_sha256": hashlib.sha256(LABELS.read_bytes()).hexdigest(),
                "score": (
                    "tm_prob"
                    if column is None
                    else f"risk, horizon {HORIZON_NAMES[column]}"
                ),
                "score_shift_ms": shift,
                "threshold": None,
                "threshold_is": CV_THRESHOLD_SOURCE,
                "fold_info": fold_info,
                "folds": scoring.shared_cv(dev)[0],
                "cv_cohort_shots": dev,
                "fixed_published_threshold": threshold,
                "fixed_published_threshold_is": THRESHOLD_SOURCE,
                "shots": used,
                "shots_without_inputs": missing,
                "metrics": res["cv_tuned"],
                "metrics_fixed_published": res["fixed_published"],
                "shots_in_training": [s for s in used if s in trained],
                "metrics_outside_training": held_out,
            },
        )
        print(
            name,
            args.split,
            f"{len(used)} shots, AUROC {res['cv_tuned']['auroc']['value']:.3f},"
            f" F1 {res['cv_tuned']['f1']['value']:.3f}"
            f" (fixed {res['fixed_published']['f1']['value']:.3f})",
        )


def seo_rows(shot, paths):
    """`(index, truth, why)` of Seo's archived tearing label on the model's rows."""
    from labeler.validate import archived_truth

    truth = archived_truth(shot, paths)
    if not truth.get("available"):
        return None, None, truth.get("reason", "unavailable")
    return truth, truth["tm_label"].astype(int), None


def run_legacy_cnn(args) -> None:
    from labeler.config import Paths

    # The scores and archive row indices must use the same exported input root.
    paths = replace(Paths.from_env(), root=TM / "lroot")
    table = pd.read_csv(CATALOG / f"{args.set}.csv")
    blind = set(pd.read_csv(CATALOG / "cohort.csv").query("split == 'test'").shot)
    shots = sorted(
        {
            int(s)
            for s in list(table.shot) + development_shots()
            if s not in blind and (INPUTS / CNN / f"{int(s)}.npz").is_file()
        }
    )
    score, y, valid, used, skipped = {}, {}, {}, [], {}
    for shot in shots:
        truth, label, why = seo_rows(shot, paths)
        if truth is None:
            skipped[shot] = why
            continue
        z = load(CNN, shot)
        index = truth["index"]
        if not np.allclose(z["t_s"][index], truth["t"]):
            skipped[shot] = "the archive's rows are not on the model's time grid"
            continue
        ok = z["valid"][index] & np.isfinite(z["tm_prob"][index])
        score[shot], y[shot], valid[shot] = z["tm_prob"][index], label, ok
        used.append(shot)
    thresholds, fold_info = tune(y, valid, score)
    selected = (
        [s for s in used if s in set(development_shots())]
        if args.set == "cohort"
        else [s for s in used if s in set(table.shot)]
    )
    threshold, population_validation = None, []
    if args.set != "cohort":
        _, splits = load_plan(development_shots())
        val = splits[0]["validation"]
        val = [s for s in val if s in score and valid[s].any()]
        population_validation = val
        edges = np.linspace(0.0, 1.0, 1001)
        threshold = scoring.best_threshold(
            [scoring.shot_stats(s, y[s], valid[s], score[s], edges) for s in val], edges
        )
        thresholds = {s: threshold for s in selected}
    res = scoring.evaluate(
        selected,
        y,
        valid,
        score,
        thresholds,
        n=args.bootstrap,
        seed=0,
        tious=(),
        bin_ms=25.0,
    )
    fixed = scoring.evaluate(
        selected,
        y,
        valid,
        score,
        detectors.CNN_THRESHOLD,
        n=args.bootstrap,
        seed=0,
        tious=(),
        bin_ms=25.0,
    )
    write(
        f"tm_prior_published_tm-onsetcnn_legacy_{args.set}",
        {
            "model": "tm-onsetcnn",
            "slug": CNN,
            "setting": "legacy, published outputs against Seo's archived tm_label",
            "shot_set": args.set,
            "git_sha": git_sha(),
            "score": "tm_prob at the archive's rows (the label's own time grid, 25 ms)",
            "threshold": threshold,
            "threshold_is": (
                CV_THRESHOLD_SOURCE
                if args.set == "cohort"
                else "F1-maximising on available shared-cohort development inner "
                "validation shots (seed 999), then fixed for population supplement"
            ),
            "population_threshold_validation_shots": population_validation,
            "fold_info": fold_info if args.set == "cohort" else [],
            "folds": scoring.shared_cv(development_shots())[0],
            "cv_cohort_shots": development_shots(),
            "fixed_published_threshold": detectors.CNN_THRESHOLD,
            "fixed_published_threshold_is": THRESHOLD_SOURCE,
            "held_out": (
                "not established: the CNN's training shots are not on disk and the "
                "archive is its training store, so these are an upper bound"
            ),
            "shots": selected,
            "blind_test_excluded": True,
            "segmental_f1": None,
            "segmental_reason": "the legacy target marks growth-phase rows, not spans",
            "skipped": {str(k): v for k, v in skipped.items()},
            "metrics": res,
            "metrics_fixed_published": fixed,
        },
    )
    print("tm-onsetcnn legacy", args.set, len(selected), "shots", res["auroc"]["value"])


def survival_onsets(shots) -> dict[int, float]:
    """Seconds of each shot's first positive survival label (NaN: none), those held."""
    import h5py

    path = (
        SURVIVAL
        if SURVIVAL.is_file()
        else (MAIN / "data/events/neoclassical_tearing_mode/raw/tm_labels.h5")
    )
    out = {}
    with h5py.File(path, "r") as f:
        for shot in shots:
            if str(shot) not in f:
                continue
            on = np.flatnonzero(f[str(shot)]["label"][:] > 0)
            out[shot] = (
                float(f[str(shot)]["time"][:][on[0]]) / 1000.0 if on.size else np.nan
            )
    return out


def run_legacy_dsm(args) -> None:
    from labeler.models import registry

    training = registry.load_adapter(DSM).training_shots
    pool = development_shots()
    blind = set(pd.read_csv(CATALOG / "cohort.csv").query("split == 'test'").shot)
    pool = [s for s in pool if s not in blind]
    have = [s for s in pool if (INPUTS / DSM / f"{s}.npz").is_file()]
    onsets = survival_onsets(have)
    held = [s for s in have if s in onsets and s not in training]
    for i, horizon in enumerate(HORIZON_NAMES):
        score, y, valid = {}, {}, {}
        for shot in held:
            z = load(DSM, shot)
            truth, keep = detectors.onset_within(
                z["t_s"], onsets[shot], detectors.DSM_HORIZONS_S[i]
            )
            score[shot] = z["risk"][:, i]
            y[shot] = truth
            valid[shot] = keep & z["valid"] & np.isfinite(z["risk"][:, i])
        thresholds, fold_info = tune(y, valid, score)
        res = scoring.evaluate(
            held,
            y,
            valid,
            score,
            thresholds,
            n=args.bootstrap,
            seed=0,
            tious=(),
            bin_ms=25.0,
        )
        fixed = scoring.evaluate(
            held,
            y,
            valid,
            score,
            detectors.DSM_THRESHOLD,
            n=args.bootstrap,
            seed=0,
            tious=(),
            bin_ms=25.0,
        )
        write(
            f"tm_prior_published_tm-dsm-{horizon}_legacy",
            {
                "model": f"tm-dsm-{horizon}",
                "slug": DSM,
                "setting": (
                    "legacy, published risk against an onset within the horizon on "
                    "rows before the survival label's first onset"
                ),
                "git_sha": git_sha(),
                "score": f"risk, horizon {horizon}",
                "threshold": None,
                "threshold_is": CV_THRESHOLD_SOURCE,
                "fold_info": fold_info,
                "folds": scoring.shared_cv(development_shots())[0],
                "cv_cohort_shots": development_shots(),
                "fixed_published_threshold": detectors.DSM_THRESHOLD,
                "fixed_published_threshold_is": THRESHOLD_SOURCE,
                "held_out": (
                    "shots outside the model's training_shots.txt that tm_labels.h5 "
                    "covers"
                ),
                "shots": held,
                "blind_test_excluded": True,
                "segmental_f1": None,
                "segmental_reason": "the legacy target is a forecast, not a TM span",
                "n_shots_with_inputs": len(have),
                "n_in_training_dropped": int(sum(s in training for s in have)),
                "shots_with_an_onset": [s for s in held if np.isfinite(onsets[s])],
                "metrics": res,
                "metrics_fixed_published": fixed,
            },
        )
        print(f"tm-dsm-{horizon} legacy", len(held), "shots", res["auroc"]["value"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--setting", choices=("legacy", "tokamak-si"), required=True)
    ap.add_argument("--split", choices=("dev", "test"), default="dev")
    ap.add_argument(
        "--set",
        choices=("cohort", "population"),
        default="cohort",
        help="legacy `tm-onsetcnn`: the shots (the labeler root must hold them)",
    )
    ap.add_argument(
        "--only", choices=("cnn", "dsm"), default=None, help="legacy: one model"
    )
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args(argv)
    if args.setting == "tokamak-si":
        run_tokamak_si(args)
        return 0
    if args.only in (None, "cnn"):
        run_legacy_cnn(args)
    if args.only in (None, "dsm"):
        run_legacy_dsm(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
