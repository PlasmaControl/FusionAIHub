"""Collect every finished checkpoint scorecard into one table.

Reads ``runs/<run>/scorecard_<step>/scorecard.json`` (written by
``scripts/slurm_stellarai/ckpt_scorecard.sh`` via ``analysis/checkpoint_scorecard.py``),
prints them sorted by run and step, and writes ``runs/scorecard_all.json``.

Usage::

    python analysis/scorecard_table.py --runs_dir <runs>
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--runs_dir", required=True)
    a = p.parse_args()
    rows = []
    for path in glob.glob(f"{a.runs_dir}/*/scorecard_*/scorecard.json"):
        run = Path(path).parents[1].name
        step = int(re.search(r"scorecard_(\d+)", path).group(1))
        for row in json.load(open(path)):
            rows.append({"run": run, "step": step, **row})
    rows.sort(key=lambda r: (r["run"], r["step"]))
    json.dump(rows, open(Path(a.runs_dir) / "scorecard_all.json", "w"), indent=1)
    print(f"{'run':16s} {'step':>6s} {'skill':>7s} {'halluc':>7s} {'missed':>7s} | "
          f"{'NBI off dTi':>11s} {'Ti down':>7s} | {'ECH off dTe':>11s} {'Te down':>7s}")
    for r in rows:
        print(f"{r['run']:16s} {r['step']:6d} {r['profile_skill']:+7.3f} {r['hallucinated']:7.3f} "
              f"{r['missed']:7.3f} | {r['nbi_dTi_keV']:+7.2f} keV {r['nbi_frac_down']:7.0%} | "
              f"{r['ech_dTe_keV']:+7.2f} keV {r['ech_frac_down']:7.0%}")


if __name__ == "__main__":
    main()
