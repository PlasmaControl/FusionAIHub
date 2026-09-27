"""How close a page's suggestions came to what the reviewer saved.

The reference is each saved label (`review/labels.csv`), the estimate the row
the page opened the shot on (`labels.read_source`), both on `scoring.frames`'
10 ms frames over the saved window. Frames the reviewer called uncertain or not
observable are left out; the counts are pooled over the shots saved from the
table.

The saves started from these suggestions, so this is an upper bound on what a
blind reader would measure: a reviewer who confirms without looking agrees
perfectly. It answers whether the method's output is close to what the reviewer
settles on, the question for extending it to more shots. It is not the paper's
accuracy (v1 Part 9 scores methods on the blind test split).

    pixi run -e labelmaker python -m labeler.events.review.agreement \\
        --event high_confinement_mode

prints the counts, precision and recall, the shots saved unchanged, and
`ready`: at least `MIN_SHOTS` shots and precision and recall at least
`MIN_SCORE`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ...config import Paths
from ...scoring.frames import Assessment, frame_counts
from . import labels

MIN_SHOTS = 50
MIN_SCORE = 0.75
CELLS = ("tp", "fp", "fn", "tn", "excluded")


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def agreement(event_dir: Path) -> dict:
    saved, source = labels.read_saved(event_dir), labels.read_source(event_dir)
    shots = sorted(set(saved) & set(source))
    totals = dict.fromkeys(CELLS, 0)
    for shot in shots:
        counts = frame_counts(
            Assessment.from_label(saved[shot]), Assessment.from_label(source[shot])
        )
        for cell in CELLS:
            totals[cell] += getattr(counts, cell)
    precision = _ratio(totals["tp"], totals["tp"] + totals["fp"])
    recall = _ratio(totals["tp"], totals["tp"] + totals["fn"])
    table = labels.source_path(event_dir)
    return {
        "event": Path(event_dir).name,
        "table": None if table is None else str(table),
        "shots": len(shots),
        "unchanged": sum(saved[shot] == source[shot] for shot in shots),
        **totals,
        "precision": precision,
        "recall": recall,
        "ready": len(shots) >= MIN_SHOTS
        and None not in (precision, recall)
        and min(precision, recall) >= MIN_SCORE,
        "min_shots": MIN_SHOTS,
        "min_score": MIN_SCORE,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--event", required=True)
    args = parser.parse_args(argv)
    event_dir = Paths.from_env().label_tables / args.event
    print(json.dumps(agreement(event_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
