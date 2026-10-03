#!/usr/bin/env python
"""How the n = 1 rule's numbers move its agreement with the survival labels.

Re-runs the n = 1 rule on the cohort's training and validation shots (never the test
shots) with one number changed at a time (onset level, hold, duty, merge gap, release
fraction) and compares each result with the survival labels' onsets (`tm_agreement.py`'s
match, a 100 ms tolerance). The rule is frozen at the published numbers; this table is
the evidence for keeping them, not a search for better ones.

    PYTHONPATH=$PWD/src pixi run --frozen --no-install -e labelmaker python \\
        scripts/labeler/tm_sensitivity.py
"""

from __future__ import annotations

import argparse
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

from tm_agreement import survival_references

from labeler.config import git_sha
from labeler.tearing import agreement, rule

OUT_ROOT = (
    Path(os.environ.get("LABELER_ROOT", "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker"))
    / "round4/tm"
)
CATALOG = REPO / "data/events/catalog"

#: `(name, changes to N1_RULE)`; the first is the frozen rule.
VARIANTS = (
    ("frozen: 12 G, 50 ms, duty 0.5, release 10 %", {}),
    ("onset 8 G", {"onset_g": 8.0}),
    ("onset 10 G", {"onset_g": 10.0}),
    ("onset 15 G", {"onset_g": 15.0}),
    ("hold 20 ms", {"hold_ms": 20.0}),
    ("hold 30 ms", {"hold_ms": 30.0}),
    ("hold 100 ms", {"hold_ms": 100.0}),
    ("no duty test", {"min_duty": 0.0}),
    ("duty 0.8", {"min_duty": 0.8}),
    ("merge gap 20 ms", {"merge_gap_ms": 20.0}),
    ("merge gap 100 ms", {"merge_gap_ms": 100.0}),
    ("release 5 %", {"release_fraction": 0.05}),
    ("release 20 %", {"release_fraction": 0.20}),
)


def intervals_with(rule_, shots, windows, starts, signals: Path) -> pd.DataFrame:
    rows = []
    for shot in shots:
        with np.load(signals / f"{shot}.npz") as npz:
            t_ms, n1 = npz["t_ms"], npz["n1rms"]
        w0, w1 = windows[shot]
        start = min(max(float(starts[str(shot)]["start_ms"]), w0), w1)
        for item in rule.tearing_intervals(t_ms, n1, None, (start, w1), (rule_,)):
            rows.append((shot, 1, item.start_ms, item.end_ms))
    return pd.DataFrame(rows, columns=["shot", "n", "t_start", "t_end"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="_dev")
    args = ap.parse_args(argv)

    cohort = pd.read_csv(CATALOG / "cohort.csv")
    cohort = cohort[cohort.split != "test"]
    signals = OUT_ROOT / "signals"
    shots = [s for s in cohort.shot if (signals / f"{s}.npz").is_file()]
    windows = {
        int(r.shot): (float(r.window_start_ms), float(r.window_end_ms))
        for r in cohort.itertuples(index=False)
    }
    starts = json.loads(
        (OUT_ROOT / "labels/plasma_start_cohort.json").read_text(encoding="utf-8")
    )
    refs, absent = survival_references(shots)
    out = []
    for name, changes in VARIANTS:
        rule_ = replace(rule.N1_RULE, **changes)
        found = intervals_with(rule_, shots, windows, starts, signals)
        onsets, compared = agreement.compare_onsets(refs, found, ns=(1,))
        summary = agreement.summarize(onsets, compared, refs)
        dur = (found.t_end - found.t_start).to_numpy(float)
        out.append(
            {
                "variant": name,
                "changes": changes,
                "intervals": len(found),
                "shots_with_an_interval": int(found.shot.nunique()),
                "median_duration_ms": float(np.median(dur)) if dur.size else None,
                "reference_onsets": summary["reference_onsets"],
                "matched_fraction": summary["matched_fraction"],
                "within_50ms_of_matched": summary.get("error_ms", {})
                .get("within_ms", {})
                .get("50"),
                "intervals_without_an_onset_fraction": summary.get(
                    "intervals_without_an_onset_fraction"
                ),
            }
        )
        print(
            f"{name:44s} intervals {len(found):4d}  matched "
            f"{out[-1]['matched_fraction']:.2f}  without onset "
            f"{out[-1]['intervals_without_an_onset_fraction']:.2f}"
        )
    record = {
        "reference": "survival",
        "shots": "cohort split != test",
        "n_shots": len(shots),
        "not_in_reference": len(absent),
        "git_sha": git_sha(),
        "variants": out,
    }
    path = OUT_ROOT / "agreement" / f"sensitivity_survival{args.tag}.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
