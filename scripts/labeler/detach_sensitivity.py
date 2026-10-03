#!/usr/bin/env python
"""Bin-width sensitivity of the detachment label: 20, 50 and 100 ms.

    for w in 20 50 100; do
      python scripts/labeler/detach_bins.py --shots-file shots.txt --redo \\
          --width-ms $w --out-dir $LABELER_ROOT/round4/detach/bins_w$w
    done
    python scripts/labeler/detach_sensitivity.py --shots-file shots.txt

Writes `docs/labeler/results/detachment_bin_sensitivity.json`. For every width, the
SAME compatibility rule labels the bins of the listed shots, with the fitted
diagnostic model retained only for auxiliary fields (nothing is refitted per
width), and the script reports:

* the share of bins assessed (at least two indicators valid) and the share of the
  assessed bins that are uncertain;
* pairwise agreement and kappa of the indicators' votes;
* `flicker_per_s`: transitions per second between certain states (a finer grid that
  only resolves noise flickers more);
* `vs_50ms`: the agreement and kappa of the certain states with the 50 ms labels,
  each bin read at the 50 ms bin that holds its centre.

The selected non-test shots include the owner's plasma_tv inversions and original
development subset; the cohort's test split is excluded.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import detach_label as dl
import numpy as np
import pandas as pd
from detach_json import dumps

from labeler.events.detachment import core, label_model

REPO = Path(__file__).resolve().parents[2]
RESULT = REPO / "docs" / "labeler" / "results" / "detachment_bin_sensitivity.json"
WIDTHS = (20, 50, 100)


def root() -> Path:
    return Path(os.environ["LABELER_ROOT"]) / "round4" / "detach"


def load_model() -> label_model.LabelModel:
    path = dl.OUT / "records" / "label_model.json"
    record = json.loads(path.read_text())["model"]
    model = label_model.LabelModel(
        names=tuple(record["names"]), corr=tuple(tuple(p) for p in record["corr"])
    )
    model.theta = np.asarray(record["theta"], dtype=float)
    return model


def load_width(width: int, shots: set[int]) -> pd.DataFrame:
    directory = root() / f"bins_w{width}"
    frames = []
    for shot in sorted(shots):
        path = directory / f"{shot}.npz"
        if not path.is_file():
            continue
        frames.append(dl.load_one(path))
    if not frames:
        raise SystemExit(f"no bins under {directory}")
    return pd.concat(frames, ignore_index=True)


def flicker(frame: pd.DataFrame, width: int) -> float:
    changes, seconds = 0, 0.0
    for idx in frame.groupby("shot").indices.values():
        state = frame.state_lm.to_numpy()[idx]
        start = frame.start_ms.to_numpy()[idx]
        certain = np.isin(state, (1, 2, 3))
        pair = certain[1:] & certain[:-1] & (np.diff(start) == width)
        changes += int(np.sum(pair & (state[1:] != state[:-1])))
        seconds += certain.sum() * width / 1000.0
    return changes / seconds if seconds else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shots-file", required=True)
    parser.add_argument("--out", default=str(RESULT))
    args = parser.parse_args()
    shots = {int(s) for s in Path(args.shots_file).read_text().split()}
    split = dl.cohort_split(shots)
    excluded = sorted(s for s in shots if split[s] == "test")
    shots -= set(excluded)
    model = load_model()
    labelled = {}
    result = {
        "shots_requested": len(shots),
        "shots": sorted(shots),
        "excluded_test": excluded,
        "model_record": "label_model.json",
    }
    for width in WIDTHS:
        frame = load_width(width, shots)
        votes, valid = dl.matrices(frame)
        out, _, _ = dl.label_frame(frame, model, dl.POSTERIOR_THRESHOLD, float(width))
        assessed = out.assessed.to_numpy()
        certain = out.state_lm.isin((1, 2, 3)).to_numpy()
        labelled[width] = out
        result[f"{width}ms"] = {
            "n_shots": int(frame.shot.nunique()),
            "n_bins": len(frame),
            "n_assessed_bins": int(assessed.sum()),
            "n_assessed_shots": int(frame.loc[assessed, "shot"].nunique()),
            "n_certain_bins": int(certain.sum()),
            "n_certain_shots": int(frame.loc[certain, "shot"].nunique()),
            "assessed_share": float(assessed.mean()),
            "uncertain_share_of_assessed": float(
                np.mean(out.state_lm.to_numpy()[assessed] == core.UNCERTAIN)
            ),
            "flicker_per_s": flicker(out, width),
            "pairwise": {
                k: {f: v[f] for f in ("both_vote_bins", "agreement", "kappa")}
                for k, v in dl.pairwise_agreement(votes, valid).items()
            },
        }
    reference = labelled[50]
    ref_state = {
        (int(s), float(t)): int(v)
        for s, t, v in zip(
            reference.shot, reference.start_ms, reference.state_lm, strict=True
        )
    }
    for width in WIDTHS:
        if width == 50:
            continue
        out = labelled[width]
        centre = out.start_ms.to_numpy() + width / 2.0
        bin50 = np.floor(centre / 50.0) * 50.0
        theirs = np.array(
            [
                ref_state.get((int(s), float(t)), 0)
                for s, t in zip(out.shot, bin50, strict=True)
            ]
        )
        mine = out.state_lm.to_numpy()
        both = np.isin(mine, (1, 2, 3)) & np.isin(theirs, (1, 2, 3))
        result[f"{width}ms"]["vs_50ms"] = {
            "bins_certain_in_both": int(both.sum()),
            "agreement": float(np.mean(mine[both] == theirs[both])),
            "kappa": dl.cohen_kappa(mine[both], theirs[both]),
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(dumps(result, indent=1))
    for width in WIDTHS:
        r = result[f"{width}ms"]
        print(
            f"{width:4d} ms: assessed {r['assessed_share']:.2f} uncertain "
            f"{r['uncertain_share_of_assessed']:.2f} flicker {r['flicker_per_s']:.2f}/s"
            + (f" vs50 kappa {r['vs_50ms']['kappa']:.2f}" if "vs_50ms" in r else "")
        )


if __name__ == "__main__":
    main()
