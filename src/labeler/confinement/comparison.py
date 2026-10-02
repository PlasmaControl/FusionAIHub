"""Exploratory matched-bin comparison with the frozen historical RowsCNN.

Exclude every historical training and validation shot before predicting. The
common eligible cohort is small; different diagnostic inputs/context and prior
exposure of the historical test set prevent an independent superiority claim.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Paths
from .labels import digest


def eligible_legacy_shots(rows: pd.DataFrame, training: dict) -> set[int]:
    """New test shots unseen by both old fitting and old threshold selection."""
    used = training["shots"]
    forbidden = set(used["train"]) | set(used["val"])
    return set(rows.loc[rows.split == "test", "shot"].astype(int)) - forbidden


def bin_probabilities(probability, first: int) -> dict[int, float]:
    """Accept only five observed historical 10 ms frames per whole 50 ms bin."""
    times = (first + np.arange(len(probability))) * 10
    result = {}
    for start in np.unique(times // 50 * 50):
        ix = np.flatnonzero((times >= start) & (times < start + 50))
        values = np.asarray(probability)[ix]
        if (
            np.array_equal(times[ix], np.arange(start, start + 50, 10))
            and np.isfinite(values).all()
        ):
            if not np.allclose(values, values[0], rtol=1e-7, atol=1e-9):
                raise ValueError("Historical frame probabilities are not bin-pooled")
            result[int(start)] = float(values[0])
    return result


def compare(run_dir: Path, *, repetitions=2000) -> dict:
    # Heavy ML imports are needed only when running the real comparison.
    import torch

    from ..frames import SPECS
    from ..frames.apply import frame_probs
    from ..frames.train import load
    from .train import _bootstrap, binary_metrics

    paths = Paths.from_env()
    run_dir = Path(run_dir).resolve()
    out = run_dir / "legacy_comparison"
    for protected in (paths.corpus, paths.raw_cache, paths.label_tables):
        protected = protected.resolve()
        if out == protected or protected in out.parents:
            raise ValueError("Comparison output must be outside source data stores")
    legacy = paths.root / "models/hmode_frames/v2"
    features = paths.root / "frames/features/v2/hmode_frames"
    training = json.loads((legacy / "training.json").read_text())
    old_evaluation = json.loads((legacy / "evaluation.json").read_text())
    new_training = json.loads((run_dir / "training.json").read_text())
    evaluation = json.loads((run_dir / "evaluation.json").read_text())
    if evaluation["freeze_sha256"] != digest(run_dir / "training.json"):
        raise ValueError("Training freeze differs from scored evaluation")
    if evaluation["predictions_sha256"] != digest(run_dir / "predictions.csv"):
        raise ValueError("New predictions differ from scored evaluation")
    rows = pd.read_csv(run_dir / "predictions.csv")
    model_bytes = (legacy / "model.pt").read_bytes()
    threshold_path = legacy / "threshold.json"
    threshold_bytes = threshold_path.read_bytes() if threshold_path.exists() else None
    model, blob = load(io.BytesIO(model_bytes), threshold_json=threshold_bytes)
    torch.set_num_threads(4)
    full_rows = rows.loc[rows.full_probability_H.notna()]
    allowed = eligible_legacy_shots(full_rows, training)
    scored = []
    exclusions = {}
    feature_hashes = {}
    for shot in sorted(allowed):
        path = features / f"{shot}.npz"
        if not path.is_file():
            exclusions[str(shot)] = "No historical feature file"
            continue
        feature_hashes[str(path)] = digest(path)
        with np.load(path, allow_pickle=False) as z:
            p = frame_probs(
                model, SPECS["hmode_frames"], z["x"], z["observed"], int(z["first"])
            )
            old = bin_probabilities(p, int(z["first"]))
        common = rows.loc[(rows.shot == shot) & rows.full_probability_H.notna()].copy()
        common["legacy_probability_H"] = common.t_start.map(old)
        common = common.loc[common.legacy_probability_H.notna()]
        if common.empty:
            exclusions[str(shot)] = "No whole observed common bins"
        else:
            scored.append(common)
    if not scored:
        raise ValueError("No common eligible held-out bins")
    common = pd.concat(scored, ignore_index=True)
    thresholds = {
        "full": new_training["models"]["full"]["threshold"],
        "legacy": blob["threshold"],
        "all_H": 0.5,
    }
    probabilities = {
        "full": common.full_probability_H.to_numpy(),
        "legacy": common.legacy_probability_H.to_numpy(),
        "all_H": np.ones(len(common)),
    }
    out.mkdir(parents=True, exist_ok=True)
    target = out / "matched_predictions.csv"
    common.to_csv(target, index=False)
    result = {
        "scope": "Exploratory common-cohort comparison; imported labels are dependent supervision. Historical test was previously exposed. Inputs and temporal context differ. No retuning or independent superiority claim.",
        "excluded_old_train_and_validation_shots": sorted(
            set(training["shots"]["train"]) | set(training["shots"]["val"])
        ),
        "eligible_new_test_shots_before_feature_intersection": sorted(allowed),
        "cohort_flow": {
            "new_diagnostic_eligible_shots": int(full_rows.shot.nunique()),
            "new_diagnostic_eligible_bins": len(full_rows),
            "unseen_old_training_and_validation_shots": len(allowed),
            "unseen_old_training_and_validation_bins": int(
                full_rows.shot.isin(allowed).sum()
            ),
            "matched_observable_shots": int(common.shot.nunique()),
            "matched_observable_bins": len(common),
        },
        "feature_exclusions": exclusions,
        "shots": sorted(common.shot.unique().tolist()),
        "matched_shots_previously_in_historical_test": sorted(
            set(common.shot) & set(old_evaluation["test"]["shots"]["hmode_frames"])
        ),
        "bins": len(common),
        "support": {
            cls: {
                "bins": int((common.label == i).sum()),
                "shots": int(common.loc[common.label == i, "shot"].nunique()),
            }
            for i, cls in enumerate(("L", "H"))
        },
        "thresholds": thresholds,
        "metrics": {
            name: binary_metrics(common.label, p, thresholds[name])
            for name, p in probabilities.items()
        },
        "paired_shot_bootstrap": _bootstrap(
            common, probabilities, thresholds, repetitions
        ),
        "bootstrap_interpretation": "Conditional resampling of observed shots only. One L bin on one shot cannot establish L generalization; degenerate perfect-score intervals do not quantify unseen L-shot uncertainty.",
        "inputs_sha256": {
            str(p): digest(p)
            for p in (
                run_dir / "training.json",
                run_dir / "evaluation.json",
                run_dir / "predictions.csv",
                legacy / "training.json",
                legacy / "model.pt",
                legacy / "evaluation.json",
                *([threshold_path] if threshold_bytes else []),
            )
        }
        | feature_hashes,
        "matched_predictions_sha256": digest(target),
        "code_sha256": digest(Path(__file__)),
    }
    (out / "comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=Paths.from_env().root / "confinement/v1"
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args(argv)
    print(json.dumps(compare(args.run_dir, repetitions=args.bootstrap), indent=2))


if __name__ == "__main__":
    main()
