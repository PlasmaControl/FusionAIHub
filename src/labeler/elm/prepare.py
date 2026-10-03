"""Write each reviewed shot's input channels to `<out>/<shot>.npy`.

    python -m labeler.elm.prepare [--shots S ...] [--workers N]

Reads the fetched records of `scripts/labeler/elmo_fetch.py`
(`$LABELER_ROOT/benchmarks/elm/elmo/signals/<shot>.npz`) and writes the
`(inputs.N_CHANNELS, n)` float32 array of `inputs.channels` for each shot of the
review table, to `$LABELER_ROOT/round4/elm/inputs`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..config import Paths
from . import inputs, labels


def signals_dir(paths: Paths) -> Path:
    return paths.root / "benchmarks" / "elm" / "elmo" / "signals"


def work_dir(paths: Paths) -> Path:
    return paths.root / "round4" / "elm"


def inputs_dir(paths: Paths) -> Path:
    return work_dir(paths) / "inputs"


def review_csv(paths: Paths) -> Path:
    return paths.label_tables / "edge_localized_mode" / "review" / "labels.csv"


def _one(job: tuple[int, Path, Path]) -> tuple[int, int]:
    shot, source, target = job
    x = inputs.read_channels(source)
    np.save(target, x)
    return shot, x.shape[1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shots", type=int, nargs="+")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    paths = Paths.from_env()
    table = labels.review_table(review_csv(paths))
    shots = sorted(int(s) for s in table.shot.unique())
    if args.shots:
        shots = [s for s in shots if s in set(args.shots)]
    out = inputs_dir(paths)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(s, signals_dir(paths) / f"{s}.npz", out / f"{s}.npy") for s in shots]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for shot, n in pool.map(_one, jobs):
            print(f"{shot} {n} cells", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
