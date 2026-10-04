"""Absent-class composition before and after the tested-absence policy.

Before: the previous round called time absent when EFIT01 q_min stayed >= 1.5
(or an ECE test passed). After: only the ECE quiet-core test makes time absent;
q-prior-only time is the separate state ``absent_q_prior``.

Reads the previous round's committed ``data_summary.json`` and its per-shot
records (for the high-q share of its absent samples) and this round's
``data_summary.json``. Writes ``absent_composition.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sawtooth_physics import FS, OUTPUT, REPO, WORK, save_json

PREVIOUS_OUTPUT = OUTPUT.parent / "fix3"
PREVIOUS_WORK = WORK.parent / "fix3"


def before_group(records):
    """The previous round's absent class and how much of it was high q alone."""
    used = [r for r in records if "error" not in r]
    absent_s = sum(r["state_seconds"]["absent"] for r in used)
    high_q_s = (
        sum(
            r["absence_diagnostics"]["reason_samples"].get(
                "sustained_high_q_absence", 0
            )
            for r in used
        )
        / FS
    )
    shots_with_absent = [r for r in used if r["state_seconds"]["absent"] > 0]
    all_high_q = [
        r
        for r in shots_with_absent
        if r["absence_diagnostics"]["reason_samples"].get("sustained_high_q_absence", 0)
        / FS
        >= r["state_seconds"]["absent"] - 1 / FS
    ]
    return {
        "absent_s": absent_s,
        "sustained_high_q_absence_s": high_q_s,
        "high_q_fraction_of_absent": high_q_s / absent_s if absent_s else None,
        "shots_with_absent_time": len(shots_with_absent),
        "shots_whose_absent_time_is_all_high_q": len(all_high_q),
    }


def after_group(summary):
    composition = summary["absent_composition"]
    seconds = summary["state_seconds"]
    total = seconds["absent"] + seconds["absent_q_prior"]
    return {
        "tested_absent_s": seconds["absent"],
        "q_prior_only_s": seconds["absent_q_prior"],
        "tested_absent_with_high_q_s": composition["tested_absence_with_high_q_s"],
        "tested_absent_without_high_q_s": composition[
            "tested_absence_without_high_q_s"
        ],
        "tested_fraction_of_former_absent_class": (
            seconds["absent"] / total if total else None
        ),
        "q_prior_only_fraction_of_former_absent_class": (
            seconds["absent_q_prior"] / total if total else None
        ),
        "shots_with_tested_absence": composition["shots_with_tested_absence"],
        "shots_with_q_prior_only": composition["shots_with_q_prior_only"],
        "shots": composition["shots"],
        "tested_absent_fraction_of_observable": seconds["absent"]
        / summary["observable_seconds"],
        "q_prior_only_fraction_of_observable": seconds["absent_q_prior"]
        / summary["observable_seconds"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--previous-work", type=Path, default=PREVIOUS_WORK)
    parser.add_argument("--previous-output", type=Path, default=PREVIOUS_OUTPUT)
    args = parser.parse_args()
    cohort = pd.read_csv(REPO / "data/events/catalog/cohort.csv")
    previous = json.loads((args.previous_output / "data_summary.json").read_text())
    current = json.loads((OUTPUT / "data_summary.json").read_text())
    previous_records = {
        int(path.stem): json.loads(path.read_text())
        for path in sorted((args.previous_work / "shots").glob("*.json"))
    }
    out = {
        "definition": (
            "previous absent class = ECE test or sustained EFIT01 q_min >= 1.5; "
            "current absent class = ECE quiet-core test only"
        ),
        "previous_source": str(args.previous_output / "data_summary.json"),
        "sample_period_s": 1 / FS,
        "before": {},
        "after": {},
    }
    for split in ("train", "val", "test"):
        shots = cohort.loc[cohort.split == split, "shot"].astype(int)
        out["before"][split] = {
            **before_group(
                [previous_records[s] for s in shots if s in previous_records]
            ),
            "state_seconds_from_summary": previous["splits"][split]["state_seconds"],
        }
        out["after"][split] = after_group(current["splits"][split])
    out["before"]["population"] = {
        **before_group(list(previous_records.values())),
        "state_seconds_from_summary": previous["population"]["state_seconds"],
    }
    out["after"]["population"] = after_group(current["population"])
    out["guards"] = {
        split: current["splits"][split]["guards"] for split in ("train", "val", "test")
    }
    out["guards"]["population"] = current["population"]["guards"]
    save_json(OUTPUT / "absent_composition.json", out)
    print(
        json.dumps({"before": out["before"]["train"], "after": out["after"]["train"]})
    )


if __name__ == "__main__":
    main()
