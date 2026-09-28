"""How close a page's suggestions came to what the reviewer saved.

The reference is each saved label (`review/labels.csv`), the estimate the row
the page opened the shot on (`labels.read_source`), both on `scoring.frames`'
10 ms frames over the saved window. Frames the reviewer called uncertain or not
observable are left out; the counts are pooled over the shots saved from the
table.

A save counts only when the page opened it on the table the pointer names now:
the shot's last save in `review/history.jsonl` records that table's name as its
`source`. A save opened on another table, before the pointer moved to this one,
compares nothing this method drew; it is left out, and `excluded_saves` counts
such saves by the table they were opened on.

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
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from ...config import Paths
from ...scoring.frames import Assessment, frame_counts
from . import labels

MIN_SHOTS = 50
MIN_SCORE = 0.75
CELLS = ("tp", "fp", "fn", "tn", "excluded")


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def pooled(pairs: Iterable[tuple[Assessment, Assessment]]) -> dict:
    """`frame_counts` of each `(reference, estimate)` summed, with frame precision
    and recall (None when nothing was counted)."""
    totals = dict.fromkeys(CELLS, 0)
    for reference, estimate in pairs:
        counts = frame_counts(reference, estimate)
        for cell in CELLS:
            totals[cell] += getattr(counts, cell)
    return {
        **totals,
        "precision": _ratio(totals["tp"], totals["tp"] + totals["fp"]),
        "recall": _ratio(totals["tp"], totals["tp"] + totals["fn"]),
    }


def opened_on(event_dir: Path) -> dict[int, str | None]:
    """The table each saved shot's last save was opened on, by name."""
    return {int(e["shot"]): e.get("source") for e in labels.read_history(event_dir)}


def agreement(event_dir: Path) -> dict:
    saved, source = labels.read_saved(event_dir), labels.read_source(event_dir)
    table = labels.source_path(event_dir)
    name = None if table is None else table.name
    opened = opened_on(event_dir)
    elsewhere = {shot: opened.get(shot) for shot in saved if opened.get(shot) != name}
    shots = sorted(set(saved) & set(source) - set(elsewhere))
    scores = pooled(
        (Assessment.from_label(saved[shot]), Assessment.from_label(source[shot]))
        for shot in shots
    )
    precision, recall = scores["precision"], scores["recall"]
    excluded = Counter(str(table) for table in elsewhere.values())
    return {
        "event": Path(event_dir).name,
        "table": None if table is None else str(table),
        "shots": len(shots),
        "unchanged": sum(saved[shot] == source[shot] for shot in shots),
        **scores,
        "excluded_saves": dict(sorted(excluded.items())),
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
