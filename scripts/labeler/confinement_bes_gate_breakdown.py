#!/usr/bin/env python
"""Why the beam gate drops windows, by regime, for the BES confinement ablation.

For each class the share of its windows that pass the paper's gate (150L at or above
700 kW, 150R at or below 200 kW) and, of those that fail, where the 150L beam stood
(no beam power record, under 50 kW, 50 to 700 kW) and whether 150R was over 200 kW. The
reasons overlap, so the shares need not add to one. Two window sets: the 117 corpus
shots at 500 kHz (the first retrain's) and every fetched shot at 1 MHz.

    python scripts/labeler/confinement_bes_gate_breakdown.py

Writes ``outputs/labeler/confinement/bes/gate_breakdown.json``. The cohort's blind test
shots are not in the window tables.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from labeler.confinement import bes_protocol as bp
from labeler.confinement import bes_windows as bw

LABELER = Path(
    os.environ.get("LABELER_ROOT", "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")
)
WORK = LABELER / "round4/conf"
DEFAULT_OUT = REPO / "outputs/labeler/confinement/bes/gate_breakdown.json"
BEAM_OFF_W = 50e3


def breakdown(table) -> dict:
    left, right = table.p15L_min, table.p15R_max
    out = {}
    for code, name in enumerate(bp.CLASSES):
        sel = table.label == code
        n = int(sel.sum())
        if not n:
            continue
        lo, ro = left[sel], right[sel]
        passed = (lo >= bp.GATE_LEFT_W) & (ro <= bp.GATE_RIGHT_W)
        out[name] = {
            "windows": n,
            "shots": int(table.shot[sel].nunique()),
            "passes_gate": float(passed.mean()),
            "no_beam_record": float(lo.isna().mean()),
            "left_below_50kW": float((lo < BEAM_OFF_W).mean()),
            "left_50_to_700kW": float(
                ((lo >= BEAM_OFF_W) & (lo < bp.GATE_LEFT_W)).mean()
            ),
            "right_above_200kW": float((ro > bp.GATE_RIGHT_W).mean()),
            "left_at_least_400kW_and_right_ok": float(
                ((lo >= bp.GATE_LEFT_WP_TRAIN_W) & (ro <= bp.GATE_RIGHT_W)).mean()
            ),
        }
    return out


def git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    corpus = {
        int(f.stem) for f in (WORK / "bes500k").glob("*.npz") if "tmp" not in f.name
    }
    record = {
        "git": git_sha(),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    for key, data in (("corpus_500k", "500k"), ("all_shots_1m", "1m")):
        table, _, _ = bw.open_dataset(WORK / "datasets" / data)
        if key == "corpus_500k":
            table = table[table.shot.isin(corpus)]
        record[key] = {
            "shots": int(table.shot.nunique()),
            "windows": len(table),
            "classes": breakdown(table),
        }
        print(key, record[key]["shots"], "shots", record[key]["windows"], "windows")
        for name, row in record[key]["classes"].items():
            print(f"  {name}: {100 * row['passes_gate']:.0f} % pass")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
