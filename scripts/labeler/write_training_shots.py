"""Write the survival model's training-shot list into its model folder.

The shipped tearing-survival checkpoint (`rt_fixed_rot.pkl`) was fitted on
`/projects/EKOLEMEN/survival_tm_2/data/rt_filtered_shots_pcb_rot.pkl`, a
per-ROW array of shot numbers - one entry per training row, 914,898 of them,
8,923 distinct shots. Nothing in the artifact records which shots those were,
so every number labeler measures on its own pool is silently part in-sample
until this list is on disk beside the model: 214 of the 500 pool shots are
training shots.

The list is committed rather than read from `/projects` at import time because
`spec.py` must load it wherever labeler runs, including where that tree is
not mounted, and because a committed file is what makes the split reproducible.
It is one integer per line, sorted ascending, unique.

    pixi run -e labelmaker python scripts/labeler/write_training_shots.py

Prints the source pickle's sha256 and the line count; both belong in the card's
`upstream.notes`. The retrained variant shares the file: it continued this fit
on the same rows.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from labeler.config import atomic_path, sha256_of

SOURCE = Path("/projects/EKOLEMEN/survival_tm_2/data/rt_filtered_shots_pcb_rot.pkl")
DESTINATION = (Path(__file__).resolve().parents[2] / "src/labeler/models"
               / "d3d_tearing_time_to_event_dsm" / "training_shots.txt")


def main() -> int:
    """Read the pickled per-row shot array, write the sorted unique shots."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=Path, default=SOURCE,
                    help="pickled per-row array of training shot numbers")
    ap.add_argument("--destination", type=Path, default=DESTINATION,
                    help="text file to write, one shot per line")
    args = ap.parse_args()

    with args.source.open("rb") as fh:
        rows = np.asarray(pickle.load(fh)).ravel()
    # The upstream array is a `<U6` string array, one entry per training row.
    shots = np.unique(rows.astype(np.int64))
    if shots.min() < 100000 or shots.max() > 999999:
        raise ValueError(f"shot numbers out of range: {shots.min()}-{shots.max()}")
    with atomic_path(args.destination) as tmp:
        tmp.write_text("\n".join(str(int(shot)) for shot in shots) + "\n")
    print(f"source {args.source}")
    print(f"sha256 {sha256_of(args.source)}")
    print(f"rows {rows.size} shots {shots.size} range {shots.min()}-{shots.max()}")
    print(f"wrote {args.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
