"""The round-two editors' rosters: the frozen cohort's review queue.

The ELM, H-mode, sawtooth and tearing-mode editors review the cohort's non-blind
shots in queue order, so any prefix of what has been reviewed is a random
subsample of every group (`catalog.cohort`). The blind shots are left to v1's
blind review and never get a suggestion (`spans`).

    pixi run -e labelmaker python -m labeler.events.review.cohort_rosters \\
        --event high_confinement_mode --point

writes `data/events/<event>/shots.csv`: the queue's shots in order, each keeping
the row the roster already had for it, then every other row the roster had, as
it was (the v1 gold sawtooth shots), less the `EXAMPLE` placeholder rows. A new
row is unverified and not held out. Run again, it writes the same file.
`--point` also opens the event's page on its `spans` suggestion table
(`labels.write_pointer`), which must exist by then.

`rosters.record_review`, which the verification notebooks call, sorts a roster
by shot; the review page never calls it.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable

import pandas as pd

from ...config import Paths
from .. import rosters, spans, suggestions
from . import labels

EVENTS = tuple(sorted(spans.METHODS))
EXAMPLE = "EXAMPLE"


def cohort_roster(queue_shots: Iterable[int], existing: pd.DataFrame) -> pd.DataFrame:
    """The queue's rows in order, then the other existing rows, less examples."""
    existing = existing[~existing.notes.str.startswith(EXAMPLE)]
    by_shot = {int(row.shot): list(row) for row in existing.itertuples(index=False)}
    queue_shots = [int(shot) for shot in queue_shots]
    new = ["unverified", "false", "", "", ""]
    rows = [by_shot.get(shot, [shot, *new]) for shot in queue_shots]
    inside = set(queue_shots)
    rows += [row for shot, row in by_shot.items() if shot not in inside]
    frame = pd.DataFrame(rows, columns=list(rosters.ROSTER_COLUMNS))
    return rosters.validate_roster(frame)


def build(
    event: str, paths: Paths, *, point: bool = False, version: str = spans.VERSION,
) -> dict:
    """Write the event's roster and, with `point`, its page's source pointer."""
    method = spans.METHODS[event]
    table = suggestions.table_path(paths, event, method.name, version)
    if point and not table.is_file():
        raise FileNotFoundError(
            f"no suggestion table at {table}; run `python -m labeler.events.spans "
            f"--event {event} --version {version}` first"
        )
    path = rosters.roster_path(event, root=paths.label_tables)
    existing = (
        rosters.read_roster(path)
        if path.is_file()
        else pd.DataFrame(columns=list(rosters.ROSTER_COLUMNS))
    )
    queue = spans.targets(event, paths, "cohort")
    roster = cohort_roster(queue.shot, existing)
    rosters.write_roster(roster, path, keep_order=True)
    summary = {
        "roster": str(path),
        "shots": len(roster),
        "queue": len(queue),
        "kept": len(roster) - len(queue),
    }
    if point:
        summary["pointer"] = labels.write_pointer(
            path.parent, table, method=method.name, version=version
        )
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--event", required=True, choices=EVENTS)
    parser.add_argument("--version", type=spans._version, default=spans.VERSION)
    parser.add_argument(
        "--point", action="store_true", help="open the page on the spans table"
    )
    args = parser.parse_args(argv)
    try:
        summary = build(args.event, Paths.from_env(), point=args.point,
                        version=args.version)
    except FileNotFoundError as error:
        parser.error(str(error))
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
