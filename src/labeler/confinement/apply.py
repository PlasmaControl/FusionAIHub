"""Apply a frozen confinement model with the training extractor, read-only."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Paths
from .data import extract_shot, physical_eligibility
from .labels import digest


def predict_features(frame, bundle):
    """Common H-mode schema: 0=L, 1=H, 3=not observable; no fake negatives."""
    if bundle["task"] != "binary":
        raise ValueError("Common H/L export requires a binary model")
    out = frame[["shot", "t_start", "t_end"]].copy()
    out["category"] = 3
    out["confidence"] = np.nan
    out["probability_H"] = np.nan
    eligible = physical_eligibility(frame, bundle["features"])
    if eligible.any():
        probability = bundle["model"].predict_proba(
            frame.loc[eligible, bundle["features"]]
        )[:, 1]
        prediction = probability >= bundle["threshold"]
        out.loc[eligible, "category"] = prediction.astype(int)
        out.loc[eligible, "probability_H"] = probability
        out.loc[eligible, "confidence"] = np.where(
            prediction, probability, 1 - probability
        )
    return out[["shot", "category", "t_start", "t_end", "confidence", "probability_H"]]


def apply(shot, start_ms, stop_ms, model_dir, out, *, variant="full", paths=None):
    paths = paths or Paths.from_env()
    out, model_dir = Path(out).resolve(), Path(model_dir).resolve()
    if start_ms < 0 or stop_ms <= start_ms:
        raise ValueError("Require a positive half-open time window in milliseconds")
    for root in (
        paths.corpus,
        paths.raw_cache,
        paths.features,
        paths.label_tables / "confinement/raw",
    ):
        if out == Path(root).resolve() or Path(root).resolve() in out.parents:
            raise ValueError("Outputs must be outside raw diagnostic stores")
    model_path = model_dir / "model" / f"{variant}.pkl"
    frozen = json.loads((model_dir / "training.json").read_text())
    if digest(model_path) != frozen["models"][variant]["model_sha256"]:
        raise ValueError("Model differs from frozen training artifact")
    with model_path.open("rb") as f:
        bundle = pickle.load(f)
    # Only whole 50 ms bins are assessed, using the same origin as training.
    starts = np.arange(np.ceil(start_ms / 50) * 50, np.floor(stop_ms / 50) * 50, 50.0)
    targets = pd.DataFrame(
        {
            "shot": np.full(len(starts), shot, dtype=int),
            "t_start": starts,
            "t_end": starts + 50,
        }
    )
    features, audit = extract_shot(
        shot,
        targets,
        corpus=paths.corpus,
        cache=paths.raw_cache,
        equilibrium=paths.features,
        clips=paths.label_tables / "confinement/raw/confinement_data",
    )
    prediction = predict_features(features, bundle)
    out.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(out / "predictions.csv", index=False)
    common = prediction.drop(columns="probability_H")
    common.to_csv(out / "high_confinement_mode.csv", index=False)
    low = common.copy()
    known = low.category.isin([0, 1])
    low.loc[known, "category"] = 1 - low.loc[known, "category"]
    low.to_csv(out / "low_confinement_mode.csv", index=False)
    (out / "inference.json").write_text(
        json.dumps(
            {
                "shot": shot,
                "variant": variant,
                "threshold": bundle["threshold"],
                "model_sha256": digest(model_path),
                "training_sha256": digest(model_dir / "training.json"),
                "extractor_sha256": digest(Path(__file__).with_name("data.py")),
                "schema": "Common milliseconds, 0 absent/1 present/3 not_observable; unavailable bins have NaN confidence; no probability calibration claim",
                "audit": audit,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    return prediction


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", type=int, required=True)
    parser.add_argument("--start-ms", type=float, required=True)
    parser.add_argument("--stop-ms", type=float, required=True)
    parser.add_argument(
        "--model-dir", type=Path, default=Paths.from_env().root / "confinement/v1"
    )
    parser.add_argument(
        "--variant",
        choices=("full", "no_bes", "dalpha_nbi", "bes_only"),
        default="full",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    prediction = apply(
        args.shot,
        args.start_ms,
        args.stop_ms,
        args.model_dir,
        args.out,
        variant=args.variant,
    )
    print(prediction.category.value_counts().to_json())


if __name__ == "__main__":
    main()
