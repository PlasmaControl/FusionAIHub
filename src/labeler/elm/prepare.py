"""Write each reviewed shot's input channels to `<out>/<shot>.npy`.

    python -m labeler.elm.prepare [--shots S ...] [--workers N]

Reads the fetched records of `scripts/labeler/elmo_fetch.py`
(`$LABELER_ROOT/benchmarks/elm/elmo/signals/<shot>.npz`) and writes the
`(inputs.N_CHANNELS, n)` float32 array of `inputs.channels` for each shot of the
review table, to `$LABELER_ROOT/round4/elm/inputs`.

**Pinned labels.** `review_csv` is the table every ELM benchmark, training and
evaluation script reads. By default it is the pinned copy `REVIEW_PIN`: the 119
reviewed shots as they stood before the 10 ms regrid of 2026-10-10. Those bytes are
what the `cv2` run trained on (its `run.json` records sha256 6ae3497e...) and what the
README's Models lines were scored on (12,409 bins), and every shot of the pin has a
prepared input, so an unfiltered `train.load` runs. The table of 2026-10-17 (the regridded
set, 12,481 bins, plus 189093, which has no input) is a second pin, read by
`labeler.elm.sources` only; Ricardo Shousha's 81 shots of 2026-10-23 are a third, read
by `scripts/labeler/elm_heldout_evaluate.py`. A pin is a folder of `labels.csv`, a
`SHA256SUMS` and, where the review kept one, `history.jsonl`, written once and never
edited. A later review changes the live table, never a pin.
"""

from __future__ import annotations

import argparse
import json
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


#: The pinned review labels the ELM scripts read unless they ask for another.
REVIEW_PIN = "2026-10-10"


def pins_dir(paths: Paths) -> Path:
    return paths.root / "benchmarks" / "elm" / "review_pins"


def live_review_csv(paths: Paths) -> Path:
    """The owner's live table: it changes with every save, so only a pin is evidence."""
    return paths.label_tables / "edge_localized_mode" / "review" / "labels.csv"


def review_csv(paths: Paths, pin: str | None = REVIEW_PIN) -> Path:
    """`labels.csv` of the pinned review `pin` (`REVIEW_PIN` unless named); `pin=None`
    is the live table. A missing pin is an error, never a fall back to the live table:
    that would put every reviewed shot since the pin into a benchmark."""
    if pin is None:
        return live_review_csv(paths)
    path = pins_dir(paths) / pin / "labels.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"no pinned ELM review labels at {path}: copy the table, its history.jsonl "
            "and a SHA256SUMS there once (docs/labeler/elm_heldout.md)"
        )
    return path


def _one(job: tuple[int, Path, Path]) -> tuple[int, int]:
    shot, source, target = job
    x = inputs.read_channels(source)
    np.save(target, x)
    target.with_suffix(".metadata.json").write_text(
        json.dumps(inputs.read_metadata(source), indent=2) + "\n"
    )
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
