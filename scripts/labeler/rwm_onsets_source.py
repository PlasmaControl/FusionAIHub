"""Draft RWM's review source, `rwm_onsets` (round five), and point the page at it.

    pixi run --frozen -e labeler python scripts/labeler/rwm_onsets_source.py \\
        --version v4 [--point] [--workers 4]

The shots are the RWM roster (the cohort's review queue and Hanson's curated
shots) and the silver and gold lists, less the blind ones. A population shot
keeps the catalog's rule-4 window; Hanson's shots (`rwm.database_windows`) and
the silver and gold shots outside the population are windowed by their own Ip
(`frames.targets.window_for`). A blind shot among them, or already in the
table, is refused before anything is written. `spans.run` drafts them with
`spans.METHODS["resistive_wall_mode"]` into that method's `--version` table; a
table drafted under other rules is refused (`spans.check_rule`), and a shot
already in it is kept as drafted. `--point` then moves the page's
`review/source.json` to the table (`labels.write_pointer`).

It writes `outputs/labeler/rwm/onsets_source_<version>.json`: per curated
onset, the present row the table gives it (the row holding the onset; onsets
whose spans touch share one), why its end came (`rwm.onset_end`), where an
uncertain stretch after it ends and, for n = 2, the amplitude its end was
measured on (`rwm.end_rms`), with the counts and the median of each onset's own
duration (onset to its end). `rise_sensitivity` recomputes those durations from
the same signals with other rise windows (`rwm.RISE_MS` is v2's to v4's). It
copies the table meta's `screen_gate` (`rwm.screen_gate_summary`, which
`spans.run` writes, so a rerun of the table reproduces it): the screen
candidates the n = 1 Mirnov tracks dropped, those kept after the check and
those not assessed, and the MHD cache files read, with one sha256 over them.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from labeler.config import Paths, git_sha
from labeler.events import rwm, spans, suggestions
from labeler.events.catalog.cohort import read_cohort
from labeler.events.review import labels
from labeler.events.review.gold import read_gold
from labeler.events.review.silver import read_silver
from labeler.frames.targets import window_for

EVENT = "resistive_wall_mode"
OUT = Path(__file__).resolve().parents[2] / "outputs/labeler/rwm"
#: N1RMS rise windows (ms after the onset) the sensitivity recomputes; None is
#: no bound, with v2's no-response rule (v1 drafted those onsets to their other ends).
RISE_WINDOWS = (10.0, rwm.RISE_MS, 50.0, None)


def shot_targets(paths: Paths) -> pd.DataFrame:
    """The roster's, silver's and gold's shots, each with its window."""
    cohort = read_cohort(spans.cohort_path(paths))
    blind = set(cohort.query("blind").shot.astype(int))
    wanted = (
        set(spans.queue(paths).shot)
        | set(rwm.onset_table(paths).shot)
        | set(read_silver(paths.label_tables).shot)
        | set(read_gold(paths.label_tables).shot)
    ) - blind
    frame = spans.targets(EVENT, paths, "silver")
    rows = []
    for shot in sorted(wanted - set(frame.shot)):
        try:
            (lo, hi), _ = window_for(int(shot), paths, None, catalog={})
        except ValueError:
            lo = hi = np.nan
        rows.append([int(shot), lo, hi])
    extra = pd.DataFrame(rows, columns=["shot", "window_start_ms", "window_end_ms"])
    frame = pd.concat([frame, extra], ignore_index=True)
    return frame[frame.shot.isin(wanted)].reset_index(drop=True)


def onset_summary(table: Path) -> dict:
    """Each curated onset's present row in `table`, its end and why it ended."""
    frame = pd.read_csv(table, keep_default_na=False)
    meta = json.loads(table.with_suffix(".meta.json").read_text())
    onsets, durations, ends, stretches = [], [], Counter(), []
    for shot, info in meta["per_shot"].items():
        rows = frame[(frame.shot == int(shot)) & (frame.category == 1)]
        for point in info.get("onsets", []):
            t = float(point["t_ms"])
            held = rows[(rows.t_start <= t) & (rows.t_end > t)]
            span = (
                None
                if held.empty
                else [int(held.t_start.iloc[0]), int(held.t_end.iloc[0])]
            )
            # The onset's own span: two onsets whose spans touch share a row.
            if span is not None:
                durations.append(min(float(point.get("end_ms", span[1])), span[1]) - t)
            ends[point.get("end", "screen only")] += 1
            if "uncertain_to_ms" in point:
                stretches.append(
                    float(point["uncertain_to_ms"]) - float(point["end_ms"])
                )
            onsets.append(
                {
                    "shot": int(shot),
                    "t_ms": t,
                    "ntor": point.get("ntor"),
                    "present_row": span,
                    "end_ms": point.get("end_ms"),
                    "end": point.get("end"),
                    "uncertain_to_ms": point.get("uncertain_to_ms"),
                    "end_rms": point.get("end_rms"),
                    "n2_end_on_n1rms": point.get("n2_end_on_n1rms", False),
                }
            )
    n2 = [o for o in onsets if o["ntor"] == 2]
    return {
        "onsets": len(onsets),
        "n2_onsets": len(n2),
        "n2_end_rms": dict(Counter(o["end_rms"] for o in n2)),
        "n2_end_on_n1rms": sum(o["n2_end_on_n1rms"] for o in n2),
        "with_present_span": len(durations),
        "end_counts": dict(ends.most_common()),
        "median_ms": float(np.median(durations)) if durations else None,
        "quartiles_ms": (
            [float(q) for q in np.percentile(durations, [25, 75])]
            if durations
            else None
        ),
        "max_ms": float(max(durations)) if durations else None,
        "with_uncertain_stretch": len(stretches),
        "uncertain_stretch_median_ms": (
            float(np.median(stretches)) if stretches else None
        ),
        "per_onset": onsets,
    }


def rise_sensitivity(paths: Paths) -> dict:
    """Each curated onset's present duration under each of `RISE_WINDOWS`, from
    the signals `rwm.detect_onsets` reads."""
    durations = {window: [] for window in RISE_WINDOWS}
    for shot, frame in rwm.onset_table(paths).groupby("shot"):
        points = [
            {k: v for k, v in row.items() if pd.notna(v)}
            for row in frame.to_dict("records")
        ]
        signals = rwm.end_signals(int(shot), paths, points)
        for point in points:
            for window in RISE_WINDOWS:
                end = rwm.point_end(point, signals, rise_ms=window)[0]
                durations[window].append(end - float(point["t_ms"]))
    return {
        "none" if window is None else f"{window:g}": {
            "median_ms": float(np.median(values)),
            "at_min": int(sum(v <= rwm.MIN_PRESENT_MS + 1e-9 for v in values)),
            "onsets": len(values),
        }
        for window, values in durations.items()
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--version", type=spans._version, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--point", action="store_true", help="move review/source.json to the table"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    paths = Paths.from_env()
    method = spans.METHODS[EVENT]
    table = suggestions.table_path(paths, EVENT, method.name, args.version)
    spans.check_rule(method, table)
    targets = shot_targets(paths)
    blind = set(read_cohort(spans.cohort_path(paths)).query("blind").shot.astype(int))
    have = set(pd.read_csv(table).shot) if table.is_file() else set()
    leaked = sorted(blind & (set(targets.shot) | have))
    if leaked:  # before spans.run writes anything
        raise SystemExit(f"blind cohort shots among the targets or in {table}: {leaked}")
    summary = spans.run(
        method,
        targets,
        paths,
        windows="rwm roster, silver and gold",
        version=args.version,
        workers=args.workers,
    )
    record = {
        "table": summary["table"],
        "method": method.name,
        "version": args.version,
        "git_sha": git_sha(),
        "shots": summary["shots"],
        "rows": summary["rows"],
        "skipped": len(summary["skipped"]),
        "categories": dict(Counter(int(c) for c in pd.read_csv(table).category)),
        "blind_shots_in_table": len(blind & set(pd.read_csv(table).shot)),
        "screen_gate": summary["screen_gate"],
        **onset_summary(table),
        "rise_sensitivity": rise_sensitivity(paths),
    }
    if args.point:
        record["pointer"] = labels.write_pointer(
            paths.label_tables / EVENT, table, method=method.name, version=args.version
        )
    (OUT / f"onsets_source_{args.version}.json").write_text(
        json.dumps(record, indent=1) + "\n"
    )
    shown = {k: v for k, v in record.items() if k != "per_onset"}
    shown["screen_gate"] = {
        k: v for k, v in record["screen_gate"].items() if k != "mhd_source"
    }
    print(json.dumps(shown, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
