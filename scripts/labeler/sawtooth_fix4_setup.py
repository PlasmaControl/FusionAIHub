"""Create the fix-round-4 work directory: frozen rule and read-only inputs.

The rule is the packaged ``src/labeler/sawtooth/freeze.json`` (new ECE-validity
and quiet-core settings on top of the previous freeze). The previous freeze's
per-shot TRAIN audit, measured under the superseded absence policy, is dropped.
The EFIT/ECE geometry bundles are unchanged and are linked, not copied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sawtooth_physics import OUTPUT, REPO, WORK, save_json

PREVIOUS = REPO / "outputs/labeler/sawtooth/fix3/freeze.json"
PACKAGED = REPO / "src/labeler/sawtooth/freeze.json"
NEW_GUARDS = {
    "ece_validity_smooth_ms": "conservative prior; smooths single-sample scatter",
    "ece_validity_sustain_ms": "review value; shorter than the shortest period",
    "ece_step_ratio": "review value: adjacent-channel step above 2",
    "ece_step_max_rho": "conservative prior; outer pedestal steps are legitimate",
    "ece_axis_to_max": "review value: axis channel below 0.6 of the profile maximum",
    "ece_axis_max_distance_m": "conservative prior; 0.1 m from the EFIT axis",
    "quiet_core_max_rho": "review value: nominal geometric rho below 0.5",
    "isolated_edge_context_ms": "frame holdoff; periodic-only quiet-core test",
}


def build():
    previous = json.loads(PREVIOUS.read_text())
    packaged = json.loads(PACKAGED.read_text())
    frozen = {k: v for k, v in previous.items() if k != "train_audit"}
    frozen["rule"] = packaged["rule"]
    for key, origin in NEW_GUARDS.items():
        frozen["guards"][key] = {
            "value": packaged["rule"][key],
            "shots": [],
            "origin": origin,
            "evidence": "fix_round_4; no cohort shot tunes it",
        }
    frozen["train_audit_note"] = (
        "the per-shot TRAIN audit of the previous freeze measured the superseded "
        "absence policy and is not carried forward"
    )
    frozen["fix_round_4"] = packaged["fix_round_4"]
    frozen["source"] = "outputs/labeler/sawtooth/fix4/freeze.json"
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=WORK)
    args = parser.parse_args()
    frozen = build()
    save_json(args.work / "freeze.json", frozen)
    save_json(OUTPUT / "freeze.json", frozen)
    link = args.work / "geometry"
    if not link.exists():
        link.symlink_to(args.work.parent / "fix3/geometry")
    print(json.dumps({"rule": frozen["rule"], "geometry": str(link.resolve())}))


if __name__ == "__main__":
    main()
