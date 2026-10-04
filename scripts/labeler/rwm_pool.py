#!/usr/bin/env python
"""List the candidate shots of the RWM baseline: Hanson's 33 and their run-day pool.

Reads the operator logbook's run record for the two campaigns Jeremy Hanson's onsets
come from and writes `shots_pool.csv` (`shot, role, run, run_title, mpid`) under
``$LABELER_ROOT/round4/rwm/``. The rules are in `labeler.rwm.shots`. `rwm_fetch.py` then
fetches the inputs for these shots and `rwm_build.py` picks the matched comparison set.

    python scripts/labeler/rwm_pool.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from labeler.config import Paths
from labeler.rwm import shots


def main() -> None:
    paths = Paths.from_env()
    onsets = pd.concat(
        pd.read_csv(path)
        for path in sorted(
            (paths.label_tables / "resistive_wall_mode" / "raw").glob("*.csv")
        )
    )
    runs = shots.read_runs(paths.logs_jsonl)
    pool = shots.choose(runs, onsets.SHOT.unique())
    out = paths.root / "round4" / "rwm"
    out.mkdir(parents=True, exist_ok=True)
    pool.to_csv(out / "shots_pool.csv", index=False)
    print(pool.role.value_counts().to_string())
    print(f"{len(pool)} shots -> {out / 'shots_pool.csv'}")


if __name__ == "__main__":
    main()
