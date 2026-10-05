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

* the certain (TangTV + Afrac agreement) and TangTV-only (silver) bins by state, with
  their shots, first (the counts a reader needs before any rate), then the share of
  bins assessed (at least two indicators valid or a TangTV vote) and the share of the
  assessed bins that are uncertain, and the tier counts;
* the duration constants at that width (`afrac_reference_min_bins`,
  `prad_baseline_min_bins`, `marfe_min_bins`, `shot_eligibility`: each is a fixed
  duration in ms, so the count scales with the width; the shot eligibility is
  applied, so a shot with too little valid time at that width is dropped and
  listed in `shots_ineligible`) and the Afrac valid time, in seconds and shots, and
  the time lost to `short_reference` and to every other reason (`afrac_reason_seconds`):
  the width changes what Afrac can see only through the grid, not through the length
  of the reference;
* pairwise agreement and kappa of the indicators' votes;
* `flicker_per_s`: transitions per second between labelled states (certain or
  TangTV only; a finer grid that only resolves noise flickers more);
* `vs_50ms`: the agreement and kappa of the labelled states with the 50 ms labels,
  each bin read at the 50 ms bin that holds its centre.

`w50_matches_exported_bins` says the 50 ms grid of this run reproduces the exported
bins' valid flags and votes. The selected non-test shots include the owner's plasma_tv inversions and original
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
from labeler.events.detachment import thresholds as th

REPO = Path(__file__).resolve().parents[2]
RESULT = REPO / "docs" / "labeler" / "results" / "detachment_bin_sensitivity.json"
WIDTHS = (20, 50, 100)


def by_state(mask, state, frame) -> dict:
    """Bins and shots of the masked bins in each of the attached/detached states."""
    return {
        core.STATE_NAMES[c]: {
            "bins": int((mask & (state == c)).sum()),
            "shots": int(frame.shot[mask & (state == c)].nunique()),
        }
        for c in (core.ATTACHED, core.DETACHED)
    }


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


def load_exported(shots: set[int]) -> pd.DataFrame:
    """The 50 ms bins the exported labels were made from, for the same shots."""
    frames = [
        dl.load_one(root() / "bins" / f"{s}.npz")
        for s in sorted(shots)
        if (root() / "bins" / f"{s}.npz").is_file()
    ]
    return pd.concat(frames, ignore_index=True)


def flicker(frame: pd.DataFrame, width: int) -> float:
    changes, seconds = 0, 0.0
    for idx in frame.groupby("shot").indices.values():
        state = frame.state_lm.to_numpy()[idx]
        start = frame.start_ms.to_numpy()[idx]
        certain = np.isin(state, (1, 2))
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
        # shot eligibility is the same duration at every width (`MIN_VALID_MS` of
        # valid bins from two indicators and of assessed bins), counted in bins
        eligible = dl.eligible_shots(frame, float(width))
        loaded = {int(x) for x in frame.shot.unique()}
        frame = frame[frame.shot.isin(eligible)].reset_index(drop=True)
        votes, valid = dl.matrices(frame)
        out, _, _ = dl.label_frame(frame, model, dl.POSTERIOR_THRESHOLD, float(width))
        assessed = out.assessed.to_numpy()
        tier = out.tier.to_numpy()
        state = out.state_lm.to_numpy()
        certain = (tier == "certain") & np.isin(state, (1, 2))
        silver = (tier == "tangtv_only") & np.isin(state, (1, 2))
        labelled[width] = out

        afrac_valid = frame.afrac_valid.to_numpy(bool)
        short = frame.afrac_reason.to_numpy() == "short_reference"
        seconds = width / 1000.0
        result[f"{width}ms"] = {
            "bin_ms": width,
            "reference_min_bins": {
                "afrac": th.min_bins(th.AFRAC_REFERENCE_MIN_MS, width),
                "prad_baseline": th.min_bins(th.PRAD_BASELINE_MIN_MS, width),
                "marfe": th.min_bins(th.MARFE_MIN_MS, width),
                "shot_eligibility": th.min_bins(dl.MIN_VALID_MS, width),
            },
            "shots_loaded": len(loaded),
            "shots_ineligible": sorted(loaded - eligible),
            "afrac_valid_seconds": float(afrac_valid.sum() * seconds),
            "afrac_valid_shots": int(frame.loc[afrac_valid, "shot"].nunique()),
            "afrac_short_reference_seconds": float(short.sum() * seconds),
            "afrac_reason_seconds": {
                str(k): float(v * seconds)
                for k, v in frame.afrac_reason.value_counts().items()
                if str(k)
            },
            "tangtv_valid_seconds": float(
                frame.tangtv_valid.to_numpy(bool).sum() * seconds
            ),
            "certain_seconds": float(certain.sum() * seconds),
            "tangtv_only_seconds": float(silver.sum() * seconds),
            "n_shots": int(frame.shot.nunique()),
            "n_bins": len(frame),
            "certain_by_state": by_state(certain, state, frame),
            "tangtv_only_by_state": by_state(silver, state, frame),
            "n_certain_bins": int(certain.sum()),
            "n_certain_shots": int(frame.loc[certain, "shot"].nunique()),
            "n_tangtv_only_bins": int(silver.sum()),
            "n_tangtv_only_shots": int(frame.loc[silver, "shot"].nunique()),
            "tier_counts": {
                str(k): int(v)
                for k, v in pd.Series(tier[assessed]).value_counts().items()
            },
            "n_assessed_bins": int(assessed.sum()),
            "n_assessed_shots": int(frame.loc[assessed, "shot"].nunique()),
            "assessed_share": float(assessed.mean()),
            "uncertain_share_of_assessed": float(
                np.mean(out.state_lm.to_numpy()[assessed] == core.UNCERTAIN)
            ),
            "flicker_per_s": flicker(out, width),
            "pairwise_by_tangtv_tier": {
                str(tier): {
                    k: {f: v[f] for f in ("both_vote_bins", "agreement", "kappa")}
                    for k, v in dl.pairwise_agreement(
                        votes[frame.tangtv_tier.eq(tier)],
                        valid[frame.tangtv_tier.eq(tier)],
                    ).items()
                }
                for tier in sorted(frame.tangtv_tier.unique())
            },
        }
    exported = load_exported(shots)
    fifty = load_width(50, shots)
    columns = [
        f"{n}_{k}" for n in ("afrac", "prad", "tangtv") for k in ("valid", "vote")
    ]
    result["w50_matches_exported_bins"] = bool(
        len(exported) == len(fifty)
        and np.array_equal(
            exported[["shot", "start_ms"]].to_numpy(), fifty[["shot", "start_ms"]]
        )
        and all(
            np.array_equal(exported[c].to_numpy(), fifty[c].to_numpy()) for c in columns
        )
    )
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
        both = np.isin(mine, (1, 2)) & np.isin(theirs, (1, 2))
        result[f"{width}ms"]["vs_50ms"] = {
            "tangtv_tier": "upper_shelf",
            "bins_labelled_in_both": int(both.sum()),
            "agreement": float(np.mean(mine[both] == theirs[both])),
            "kappa": dl.cohen_kappa(mine[both], theirs[both]),
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(dumps(result, indent=1))
    for width in WIDTHS:
        r = result[f"{width}ms"]
        print(
            f"{width:4d} ms: certain {r['certain_by_state']['attached']['bins']}/"
            f"{r['certain_by_state']['detached']['bins']} assessed "
            f"{r['assessed_share']:.2f} uncertain "
            f"{r['uncertain_share_of_assessed']:.2f} flicker {r['flicker_per_s']:.2f}/s"
            + (f" vs50 kappa {r['vs_50ms']['kappa']:.2f}" if "vs_50ms" in r else "")
        )


if __name__ == "__main__":
    main()
