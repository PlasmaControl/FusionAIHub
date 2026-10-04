"""Read-only ae-ours inference for Figure 1; writes only an isolated CSV/JSON.

Run through the frozen labelmaker pixi environment with this worktree's
src on PYTHONPATH. Calls the model adapter on CPU: transform, recurrent context,
causal aggregation and validity rules. No training or network access.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from labeler.config import Paths, sha256_of
from labeler.events.verify import NoDataError, corpus_signal
from labeler.features.store import FeatureArray
from labeler.models.d3d_ae_activity_seldnet import spec
from labeler.paper.figure_sources import AE_THRESHOLD


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, nargs="+", default=[201978])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    paths = Paths.from_env()
    checkpoint = spec.UPSTREAM / spec.ARTIFACTS[0]
    expected = "f2d5314a7673a53c49e287cdbb2d5e7232bbf192b8ba830d3215c8c9124079ab"
    if sha256_of(checkpoint) != expected:
        raise ValueError("ae-ours checkpoint differs from the model card")
    predict = spec.load(checkpoint.parent)
    grid = np.arange(240) * spec.DT_S
    rows, records = [], []
    for shot in args.shots:
        try:
            co2 = corpus_signal(shot, "co2", corpus=paths.corpus)
        except NoDataError as exc:
            records.append({"shot": shot, "status": "unavailable", "reason": str(exc)})
            continue
        x = co2.x / 1000
        fs = (len(x) - 1) / (x[-1] - x[0])
        if not np.isclose(fs, 500000, rtol=0.001):
            raise ValueError(f"{shot}: model card requires 500 kHz CO2; got {fs}")
        built = spec.INPUT_SPEC.build(
            {"co2": FeatureArray(x=x, y=co2.y, attrs={"resolver": "corpus"})}, grid
        )
        active = predict(built)[0, :, 0]
        valid = built.valid & np.isfinite(active)
        states = np.where(valid, (active >= AE_THRESHOLD).astype(int), 3)
        shot_rows = []
        for t, state in zip(grid * 1000, states, strict=True):
            a, b = float(t - 25), float(t)
            if shot_rows and shot_rows[-1][1] == state:
                shot_rows[-1][3] = b
            else:
                shot_rows.append([shot, int(state), a, b, ""])
        rows.extend(shot_rows)
        records.append(
            {
                "shot": shot,
                "status": "inferred",
                "device": "cpu",
                "inference": "spec.load / INPUT_SPEC.build / predict",
                "input": str(paths.corpus_file(shot)),
                "input_sha256": sha256_of(paths.corpus_file(shot)),
                "activity": active.tolist(),
                "valid": valid.tolist(),
                "grid_ms": (grid * 1000).tolist(),
                "sample_rate_hz": fs,
            }
        )
        print(f"{shot}: inferred ae-ours, {valid.sum()} valid bins", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["shot", "category", "t_start", "t_end", "confidence"])
        writer.writerows(rows)
    args.out.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "method": "ae-ours, CO2 activity inference",
                "threshold": AE_THRESHOLD,
                "checkpoint": str(checkpoint),
                "sha256": expected,
                "window_frames": spec.WINDOW_FRAMES,
                "context_frames": spec.CONTEXT_FRAMES,
                "causal_bins": "(t-25ms,t]",
                "records": records,
            },
            indent=1,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
