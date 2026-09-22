#!/usr/bin/env python
"""One design request per shot of a list, worded from the shot's own record and theme.

The assistant is prompt-driven; a batch over a shot list therefore needs a prompt per shot.
Each prompt names the shot as the intended reference (batch_design.py also passes it as
`--ref-shot`), quotes the mini-proposal title the database holds for it, and asks for one
bounded actuator study in the theme's own terms over the 1-5 s window. The wording asks for
a candidate to simulate, never for a claimed outcome -- the harness's own system prompts
forbid invented results and this must not pull the other way.

    SHOT_DESIGN_DATA_ROOT=<batch root> python scripts/shot_design/batch_prompts.py \
        --list stellar_1k --out <root>/prompts/prompts.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]

# What to study, per §5.7 theme. Each names the physics topic in the retrieval vocabulary the
# assistant's planning prompt already knows (tearing/NTM/ECCD, Alfven/AE/TAE, ELM/RMP, ...)
# and the actuator family a bounded change would most plausibly touch.
THEME_ASK = {
    "tearing_mhd": (
        "Study tearing-mode / NTM behaviour: propose a bounded change to the ECH/ECCD "
        "deposition or the NBI heating that could alter the n=1 or n=2 mode drive."
    ),
    "rmp_elm": (
        "Study ELM control with resonant magnetic perturbations: propose a bounded change "
        "to the RMP or I-coil currents intended to move the ELM response."
    ),
    "fast_ion_ae": (
        "Study Alfven-eigenmode (AE/TAE) activity: propose a bounded change to the neutral "
        "beam power or torque that would alter the fast-ion drive."
    ),
    "qh_mode": (
        "Study quiescent H-mode access: propose a bounded change to the injected torque or "
        "beam mix that would test the QH-mode / EHO window."
    ),
    "pedestal": (
        "Study pedestal structure: propose a bounded change to the gas fuelling or the "
        "heating power that would move the pedestal density or temperature."
    ),
    "detachment_divertor": (
        "Study divertor detachment: propose a bounded change to the gas fuelling that "
        "would move the divertor toward or away from detachment."
    ),
    "hybrid_high_beta": (
        "Study the high-beta hybrid scenario: propose a bounded change to the NBI or ECH "
        "power that would test the beta limit or the q-profile."
    ),
    "current_drive": (
        "Study non-inductive current drive: propose a bounded change to the ECCD or beam "
        "power that would alter the driven current fraction."
    ),
    "neg_tri": (
        "Study the negative-triangularity scenario: propose a bounded change to the "
        "heating power or fuelling that would test its confinement or edge behaviour."
    ),
    "transport": (
        "Study core transport: propose a bounded change to the heating mix (NBI versus "
        "ECH) that would alter the core temperature or rotation profile."
    ),
    "lh_threshold_isotope": (
        "Study the L-H transition threshold: propose a bounded change to the heating power "
        "ramp that would probe the power threshold."
    ),
    "disruption_runaway": (
        "Study disruption avoidance: propose a bounded, conservative change to the fuelling "
        "or heating that would keep the plasma further from its operational limits."
    ),
    "control": (
        "Study plasma control: propose a bounded change to one actuator waveform that "
        "would test the controller's response."
    ),
    "startup_checkout": (
        "Study the startup / checkout scenario: propose a bounded change to the heating or "
        "fuelling during the flat-top that would give a cleaner reference."
    ),
}
DEFAULT_ASK = (
    "Propose one bounded actuator study on this scenario: change one heating, fuelling or "
    "coil waveform by a modest factor and say what to compare against the reference."
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--list", required=True, help="configs/shot_design/shot_lists/<name>.yaml"
    )
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--db", type=Path, default=None, help="db dir (default <data_root>/db)"
    )
    args = ap.parse_args()

    root = os.environ.get("SHOT_DESIGN_DATA_ROOT")
    if not root:
        print("set SHOT_DESIGN_DATA_ROOT to the batch root", file=sys.stderr)
        return 2
    db_dir = args.db or Path(root) / "db"
    shots = pd.read_parquet(db_dir / "shots.parquet").set_index("shot")
    doc = yaml.safe_load(
        (REPO / "configs/shot_design/shot_lists" / f"{args.list}.yaml").read_text()
    )
    rows = []
    missing = 0
    for entry in doc["shots"]:
        shot = int(entry["shot"])
        theme = entry.get("theme") or "none"
        if shot not in shots.index:
            missing += 1
            continue
        rec = shots.loc[shot]
        title = str(rec.get("mp_title") or "").strip()
        regime = str(rec.get("regime") or "").strip()
        about = f", run under the mini-proposal '{title[:120]}'" if title else ""
        regime_s = f" (regime: {regime})" if regime and regime != "unknown" else ""
        ask = THEME_ASK.get(theme, DEFAULT_ASK)
        prompt = (
            f"Use DIII-D shot {shot} as the reference{about}{regime_s}. {ask} "
            f"Keep the window from 1 to 5 seconds and keep the change within 0.7x to 1.3x of "
            f"the measured waveform. This is a candidate for IGNITE simulation and expert "
            f"review, not a claimed result."
        )
        rows.append(
            {
                "shot": shot,
                "theme": theme,
                "year": entry.get("year"),
                "run_id": entry.get("run_id"),
                "mpid": entry.get("mpid"),
                "mp_title": title,
                "regime": regime,
                "has_frame_codes": bool(rec.get("has_frame_codes", False)),
                "prompt": prompt,
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(
        f"{len(rows)} prompts -> {args.out}; {missing} list shots not in the database"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
