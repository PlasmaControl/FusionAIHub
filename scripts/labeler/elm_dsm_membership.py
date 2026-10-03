#!/usr/bin/env python
"""Extract DSM source-exposure roles as physical-shot membership, offline.

The normalization population is checked against compiled_model10.pkl rather
than assumed from train/test membership. No source artifact is changed.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from labeler.config import Paths, sha256_of

SOURCE = Path("/projects/EKOLEMEN/wpqh_elm_hiro")
OUT = Path("src/labeler/models/d3d_elm_time_to_event_dsm/training_membership.json")


def _physical(phases) -> list[int]:
    # These original IDs are <physical shot>_<phase> strings, not the older
    # int(string) rendering. Decode before grouping phases of the same shot.
    shots = {int(str(p).split("_")[0]) for p in np.asarray(phases).ravel()}
    assert all(100_000 <= s < 1_000_000 for s in shots)
    return sorted(shots)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    split = SOURCE / "data/train_test_split_model10.pkl"
    compiled = SOURCE / "data/compiled_model10.pkl"
    notebook = SOURCE / "hiro_scripts/data_processing.ipynb"
    with split.open("rb") as fh:
        data = pickle.load(fh)
    phases = {
        "train": np.unique(data["train_final_shots_list"]),
        "test": np.unique(data["test_final_shots_list"]),
    }
    shots = {k: _physical(v) for k, v in phases.items()}
    del data
    with compiled.open("rb") as fh:
        data = pickle.load(fh)
    normalization = _physical(data["final_shots_list"])
    del data
    assert set(normalization) == set(shots["train"]) | set(shots["test"])
    cohort = pd.read_csv(Paths.from_env().catalog / "cohort.csv")
    blind = sorted(set(normalization) & set(cohort.loc[cohort.split == "test", "shot"]))
    result = {
        "producer": "scripts/labeler/elm_dsm_membership.py",
        "source_sha256": sha256_of(Path(__file__)),
        "encoding": (
            "six-digit physical shot IDs; upstream phase identifiers decoded "
            "before grouping"
        ),
        "split_source": {"path": str(split), "sha256": sha256_of(split)},
        "normalization_source": {
            "path": str(compiled),
            "sha256": sha256_of(compiled),
            "notebook": str(notebook),
            "notebook_sha256": sha256_of(notebook),
            "operation": (
                "model10_norms mean/std computed over final_x before phase "
                "train/test split"
            ),
        },
        "roles": {
            "training": "rows used to fit survival weights",
            "early_stopping": (
                "upstream test rows used as val_data and for checkpoint/input-set "
                "selection"
            ),
            "normalization": (
                "feature mean/std exposure before split; applies to all DSM variants"
            ),
        },
        "training_shots": shots["train"],
        "early_stopping_shots": shots["test"],
        "normalization_shots": normalization,
        "physical_shots_in_both_split_sides": sorted(
            set(shots["train"]) & set(shots["test"])
        ),
        "phase_records": {k: len(v) for k, v in phases.items()},
        "physical_shot_counts": {
            **{k: len(v) for k, v in shots.items()},
            "normalization": len(normalization),
            "all_exposed": len(normalization),
        },
        "blind_cohort_normalization_shots": blind,
        "validation_membership_policy": (
            "adapter.training_shots is the union of training, early-stopping and "
            "normalization exposure so none is classified as held_out; in_training "
            "means source-exposed for this adapter"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "record": str(args.out),
                **result["physical_shot_counts"],
                "blind_cohort_shots": blind,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
