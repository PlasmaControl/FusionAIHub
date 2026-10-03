#!/usr/bin/env python
"""Predeclare positive-bearing inner validation before any TM detector fit.

Outer folds remain seed-0 round-robin assignments on the 450 development shots.
Inner roles depend only on interval presence and positive target/input support,
never scores, fitting losses or held-fold performance. The fixed design uses 10%
per presence stratum, with within-stratum swaps to retain observable positives for
all published and retrained rows. Only outer-training pools determine inner roles.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from labeler.config import Paths
from labeler.tearing import detectors, scoring

REPO = Path(__file__).resolve().parents[2]
TM = (
    Path(os.environ.get("LABELER_ROOT", "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker"))
    / "round4/tm"
)
LABELS = (
    REPO / "data/events/neoclassical_tearing_mode/extend_tm_interval/tm_interval.csv"
)
PLAN = TM / "results/inner_splits_fix2.json"


def load_plan(shots):
    record = json.loads(PLAN.read_text())
    assert sorted(shots) == record["development_shots"]
    assert record["labels_sha256"] == hashlib.sha256(LABELS.read_bytes()).hexdigest()
    return {int(s): f for s, f in record["folds"].items()}, record["splits"]


def main():
    import tm_ours
    import tm_prior_published as published
    import tm_prior_retrain as retrained

    cohort = pd.read_csv(REPO / "data/events/catalog/cohort.csv")
    shots = sorted(int(s) for s in cohort.query("split != 'test'").shot)
    table = pd.read_csv(LABELS)
    assert set(table.shot) <= set(shots)
    positive = set(table.query("category == 1 and t_end > t_start").shot)
    training = published.dsm_training_shots()
    groups = {}
    ours, _ = tm_ours.load(shots, False)
    groups["tm-ours"] = [s for s, d in ours.items() if np.any(d[2] & d[3])]
    for kind, model in (("cnn", "tm-onsetcnn"), ("dsm", "tm-dsm")):
        data, _ = retrained.load(kind, shots)
        groups[f"{model}-retrained"] = [
            s for s, d in data.items() if np.any((d["y"] > 0) & d["train_ok"])
        ]
        # A constant fake row score checks time-grid support without reading any
        # predicted probability. Published CNN rows have the published 25ms shift.
        shift = detectors.CNN_SHIFT_MS if kind == "cnn" else 0.0
        groups[f"{model}-published-interval"] = []
        for s, d in data.items():
            bins = detectors.rows_to_bins(
                d["t_ms"],
                d["model_ok"],
                np.zeros(len(d["t_ms"])),
                d["centres"],
                shift_ms=shift,
            )
            if np.any(d["y_bins"] & d["valid_bins"] & np.isfinite(bins)):
                groups[f"{model}-published-interval"].append(s)
        if kind == "dsm":
            groups["tm-dsm-published-outside-training"] = [
                s for s in groups["tm-dsm-published-interval"] if s not in training
            ]
    paths = replace(Paths.from_env(), root=TM / "lroot")
    legacy_cnn = []
    for s in shots:
        path = published.INPUTS / published.CNN / f"{s}.npz"
        if not path.is_file():
            continue
        truth, label, _ = published.seo_rows(s, paths)
        if truth is None:
            continue
        with np.load(path) as z:
            if np.any(label & z["valid"][truth["index"]]):
                legacy_cnn.append(s)
    groups["tm-onsetcnn-published-legacy"] = legacy_cnn
    have = [
        s for s in shots if (published.INPUTS / published.DSM / f"{s}.npz").is_file()
    ]
    onsets = published.survival_onsets(have)
    for i, horizon in enumerate(published.HORIZON_NAMES):
        supported = []
        for s in have:
            if s not in onsets or s in training:
                continue
            with np.load(published.INPUTS / published.DSM / f"{s}.npz") as z:
                y, valid = detectors.onset_within(
                    z["t_s"], onsets[s], detectors.DSM_HORIZONS_S[i]
                )
                if np.any(y & valid & z["valid"]):
                    supported.append(s)
        groups[f"tm-dsm-{horizon}-published-legacy"] = supported
    folds, splits = scoring.shared_cv(
        shots, positive_shots=positive, support_groups=groups
    )
    record = {
        "made_by": "scripts/labeler/tm_cv_plan.py",
        "labels_sha256": hashlib.sha256(LABELS.read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "policy": __doc__,
        "development_shots": shots,
        "positive_interval_shots": sorted(int(s) for s in positive),
        "positive_input_support": groups,
        "folds": folds,
        "splits": splits,
    }
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    PLAN.write_text(json.dumps(record, indent=2) + "\n")
    print(PLAN)
    for split in splits:
        print(
            split["fold"],
            "validation support",
            {
                name: len(set(split["validation"]) & set(group))
                for name, group in groups.items()
            },
        )


if __name__ == "__main__":
    main()
