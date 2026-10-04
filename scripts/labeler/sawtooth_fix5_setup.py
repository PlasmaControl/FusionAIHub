"""Create the fix-round-5 work directory: frozen rule and read-only inputs.

The rule is the packaged ``src/labeler/sawtooth/freeze.json``. Its isolated-edge
veto length must equal the value chosen by the TRAIN-only derivation record
``edge_context_derivation.json`` (``sawtooth_edge_context.py``); this script
refuses to build the freeze when the two disagree, so the packaged rule, the
derivation record and the freeze record state one provenance. The previous
freeze's per-shot TRAIN audit, measured under the superseded absence policy,
is dropped. The EFIT/ECE geometry bundles are unchanged and are linked, not
copied; the fix-round-2 prior-input snapshot is copied from the previous round.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from sawtooth_physics import OUTPUT, PRIOR_INPUTS, REPO, WORK, save_json

PREVIOUS = REPO / "outputs/labeler/sawtooth/fix4/freeze.json"
PREVIOUS_WORK = WORK.parent / "fix4"
PACKAGED = REPO / "src/labeler/sawtooth/freeze.json"
DERIVATION = OUTPUT / "edge_context_derivation.json"
NEW_GUARDS = {
    "ece_validity_smooth_ms": "conservative prior; smooths single-sample scatter",
    "ece_validity_sustain_ms": "review value; shorter than the shortest period",
    "ece_step_ratio": "review value: adjacent-channel step above 2",
    "ece_step_max_rho": "conservative prior; outer pedestal steps are legitimate",
    "ece_axis_to_max": "review value: axis channel below 0.6 of the profile maximum",
    "ece_axis_max_distance_m": "conservative prior; 0.1 m from the EFIT axis",
    "quiet_core_max_rho": "review value: nominal geometric rho below 0.5",
}


def edge_context_guard(derivation, rule):
    """Provenance of the isolated-edge veto, read from the derivation record."""
    if derivation["chosen_ms"] != rule["isolated_edge_context_ms"]:
        raise ValueError(
            "packaged isolated_edge_context_ms "
            f"{rule['isolated_edge_context_ms']} differs from the TRAIN-only "
            f"derivation {derivation['chosen_ms']}; update freeze.json"
        )
    return {
        "value": derivation["chosen_ms"],
        "shots": derivation["derivation_shots"],
        "origin": derivation["criterion"],
        "evidence": (
            "edge_context_derivation.json: TRAIN shots only, no val or test shot "
            "read. The earlier choice of 5.15 ms came from an exploration of "
            f"{len(derivation['previous_exploration']['shots'])} shots of which "
            f"{derivation['previous_exploration']['split_counts'].get('val', 0)} "
            "were val (blind-queue) shots, so it is not carried forward"
        ),
        "previous_exploration": derivation["previous_exploration"],
        "candidate_rows": derivation["rows"],
    }


def build():
    previous = json.loads(PREVIOUS.read_text())
    packaged = json.loads(PACKAGED.read_text())
    derivation = json.loads(DERIVATION.read_text())
    frozen = {k: v for k, v in previous.items() if k != "train_audit"}
    frozen["rule"] = packaged["rule"]
    for key, guard in frozen["guards"].items():
        value = packaged["rule"].get(key)
        if value is not None and guard["value"] != value:
            guard["value_in_previous_freeze"] = guard["value"]
            guard["value"] = value
    for key, origin in NEW_GUARDS.items():
        frozen["guards"][key] = {
            "value": packaged["rule"][key],
            "shots": [],
            "origin": origin,
            "evidence": "fix_round_4; no cohort shot tunes it",
        }
    frozen["guards"]["isolated_edge_context_ms"] = edge_context_guard(
        derivation, packaged["rule"]
    )
    frozen["train_audit_note"] = (
        "the per-shot TRAIN audit of the previous freeze measured the superseded "
        "absence policy and is not carried forward"
    )
    frozen["fix_round_4"] = packaged["fix_round_4"]
    frozen["fix_round_5"] = packaged["fix_round_5"]
    frozen["source"] = "outputs/labeler/sawtooth/fix5/freeze.json"
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument(
        "--derivation-run",
        action="store_true",
        help="work directory of the TRAIN-only edge-context derivation: the "
        "freeze is written there only, nothing under outputs/",
    )
    args = parser.parse_args()
    if args.derivation_run:
        # The derivation precedes the chosen value, so its freeze is the
        # packaged rule as it stands; only the work directory is prepared.
        frozen = {"rule": json.loads(PACKAGED.read_text())["rule"]}
    else:
        frozen = build()
        save_json(OUTPUT / "freeze.json", frozen)
    save_json(args.work / "freeze.json", frozen)
    link = args.work / "geometry"
    if not link.exists():
        link.symlink_to(args.work.parent / "fix3/geometry")
    inputs = args.work / "labels" / PRIOR_INPUTS
    if not inputs.exists():
        inputs.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PREVIOUS_WORK / "labels" / PRIOR_INPUTS, inputs)
    if not args.derivation_run:
        committed = OUTPUT / PRIOR_INPUTS
        if not committed.exists():
            committed.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(inputs, committed)
    old_rule = args.work / "old_rule"
    if not old_rule.exists():
        old_rule.symlink_to(args.work.parent / "fix3/old_rule")
    print(json.dumps({"rule": frozen["rule"], "geometry": str(link.resolve())}))


if __name__ == "__main__":
    main()
