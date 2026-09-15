"""Summarise what `d3d_ae_activity_seldnet` produced over a set of corpus shots.

Reads the feature and label files a `features` + `infer` run already wrote and
answers the three questions the model card and the task report need: how many
shots the corpus fills `co2` on, how many rows came out valid, and what the
labels look like per shot. It computes nothing new - re-running it after a
rerun of `infer` is cheap and always agrees with the stored labels.

    cd /scratch/gpfs/nc1514/FusionAIHub && \
    pixi run -q -e labelmaker python outputs/labelmaker/ae/scripts/corpus_coverage.py \
      --sample 24 --seed 0 --out outputs/labelmaker/ae/corpus_coverage.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from labelmaker.catalog import corpus_shots, sample_shots
from labelmaker.config import Paths
from labelmaker.models.d3d_ae_activity_seldnet import spec as ae


def _shot_row(paths: Paths, shot: int) -> dict:
    row: dict = {"shot": shot, "co2": False, "n_valid": 0}
    features = paths.features_file(shot)
    if features.exists():
        with h5py.File(features, "r") as f:
            if "co2" in f:
                row["co2"] = True
                row["co2_shape"] = list(f["co2"]["ydata"].shape)
                row["sample_rate_hz"] = str(f["co2"].attrs.get("sample_rate_hz", ""))
            row["missing"] = json.loads(f.attrs.get("missing", "{}"))
    labels = paths.labels_file(shot)
    if not labels.exists():
        return row
    with h5py.File(labels, "r") as f:
        if ae.SLUG not in f:
            return row
        g = f[ae.SLUG]
        active = np.asarray(g["ae_active"]["ydata"][0], dtype=np.float64)
        freq = np.asarray(g["ae_frequency"]["ydata"][0], dtype=np.float64)
        valid = np.asarray(g["ae_active_valid"]["ydata"][0]).astype(bool)
    finite = np.isfinite(active)
    row.update(
        n_rows=int(active.size),
        n_valid=int(valid.sum()),
        n_finite=int(finite.sum()),
        mean_active=float(active[finite].mean()) if finite.any() else None,
        max_active=float(active[finite].max()) if finite.any() else None,
        frac_rows_above_half=float((active[finite] >= 0.5).mean()) if finite.any() else None,
        n_frequency=int(np.isfinite(freq).sum()),
        median_frequency_khz=(
            float(np.median(freq[np.isfinite(freq)])) if np.isfinite(freq).any() else None
        ),
        # Inference notches nothing (see the card); recorded so the number is
        # in the artifact rather than only in the prose.
        notched_bins=0,
    )
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shots", nargs="+", type=int, default=None)
    ap.add_argument("--sample", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    paths = Paths.from_env()
    shots = args.shots or sample_shots(corpus_shots(paths), args.sample, args.seed)
    rows = [_shot_row(paths, int(s)) for s in shots]
    with_co2 = [r for r in rows if r["co2"]]
    report = {
        "slug": ae.SLUG,
        "checkpoint": ae.ARTIFACTS[0],
        "selection": (
            f"sample_shots(corpus_shots, {args.sample}, seed={args.seed})"
            if args.shots is None else "explicit --shots"
        ),
        "n_shots": len(rows),
        "n_with_co2": len(with_co2),
        "n_labelled": sum(1 for r in rows if r.get("n_rows")),
        "n_valid_rows": sum(r["n_valid"] for r in rows),
        "n_rows_total": sum(r.get("n_rows", 0) for r in rows),
        "notch_rule": ae.NOTCH_RULE,
        "notched_bins_at_inference": 0,
        "shots": rows,
    }
    print(json.dumps(report, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
