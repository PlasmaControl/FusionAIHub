"""Draft RWM's review source, `rwm_onsets` (round five), and point the page at it.

    pixi run --frozen -e labeler python scripts/labeler/rwm_onsets_source.py \\
        --version v5 [--point] [--workers 4]

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
whose spans touch share one), why its end came (v5: `rwm.bp_end` on the n = 1
delta-Bp, `rwm.onset_end` on N2RMS for n = 2), where an uncertain stretch after
it ends, the delta-Bp rise and peak and, for n = 2, the amplitude its end was
measured on (`rwm.end_rms`), with the counts and the median of each onset's own
duration (onset to its end). `rise_sensitivity` recomputes those durations from
the same signals with other rise windows (`rwm.BP_RISE_MS` for n = 1,
`rwm.RISE_MS` for n = 2). It copies the table meta's `screen_gate`
(`rwm.screen_gate_summary`, which `spans.run` writes, so a rerun of the table
reproduces it): the delta-Bp runs that passed, failed and could not test the
start condition, the screen candidates the n = 1 Mirnov tracks dropped, those
kept after the check and those not assessed, and the MHD cache files read,
with one sha256 over them. `comparison` runs the same screen on round four's
comparison shots (`round4/rwm/shots.csv`, the roles other than hanson), which
round four assumed RWM-free.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from multiprocessing import Pool
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
#: Rise windows (ms after the onset) the sensitivity recomputes, delta-Bp for
#: n = 1 and N2RMS for n = 2; None is no bound, with v2's no-response rule (v1
#: drafted those onsets to their other ends).
RISE_WINDOWS = (10.0, rwm.BP_RISE_MS, 50.0, None)


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
                    "bp_rise_ms": point.get("bp_rise_ms"),
                    "bp_peak_g": point.get("bp_peak_g"),
                }
            )
    n2 = [o for o in onsets if o["ntor"] == 2]
    rises = [o["bp_rise_ms"] for o in onsets if o["bp_rise_ms"] is not None]
    return {
        "onsets": len(onsets),
        "n1_with_bp_response": len(rises),
        "bp_rise_ms_quartiles": (
            [float(q) for q in np.percentile(rises, [25, 50, 75])] if rises else None
        ),
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
    the signals `rwm.detect_onsets` reads: the delta-Bp series for n = 1, N2RMS
    for n = 2."""
    durations = {window: [] for window in RISE_WINDOWS}
    for shot, frame in rwm.onset_table(paths).groupby("shot"):
        points = [
            {k: v for k, v in row.items() if pd.notna(v)}
            for row in frame.to_dict("records")
        ]
        signals = rwm.end_signals(int(shot), paths, points)
        others = {k: signals[k] for k in ("betan", "ip") if k in signals}
        series = rwm._screen(int(shot), paths)[1]
        for point in points:
            onset = float(point["t_ms"])
            for window in RISE_WINDOWS:
                if int(point.get("ntor", 1)) == 2:
                    end = rwm.point_end(point, signals, rise_ms=window)[0]
                else:
                    rise = np.inf if window is None else window
                    end = rwm.bp_end(onset, series=series, **others, rise_ms=rise)[0]
                durations[window].append(end - onset)
    return {
        "none" if window is None else f"{window:g}": {
            "median_ms": float(np.median(values)),
            "at_min": int(sum(v <= rwm.MIN_PRESENT_MS + 1e-9 for v in values)),
            "onsets": len(values),
        }
        for window, values in durations.items()
    }


def _screen_shot(shot: int) -> dict:
    """One comparison shot's v5 screen: what ran and what it kept."""
    paths = Paths.from_env()
    found = rwm.detect_onsets(shot, paths)
    info = found.info
    gate = info.get("screen_gate", {}).get("candidates", [])
    return {
        "shot": shot,
        "not_run": info.get("not_run"),
        "measured_s": round(sum(b - a for a, b in found.measured) / 1000, 3),
        "kept": [[round(a, 1), round(b, 1)] for a, b, _ in found.spans],
        "start_failed": info.get("start", {}).get("failed", 0),
        "start_untested": len(info.get("start", {}).get("untested", ())),
        "gate": dict(Counter(c[-1] for c in gate)),
    }


def comparison(paths: Paths, workers: int) -> dict:
    """The v5 screen on round four's comparison shots, per role: shots screened,
    candidates kept (uncertain) after the start condition and the Mirnov check,
    and the shots holding them."""
    path = paths.root / "round4" / "rwm" / "shots.csv"
    if not path.is_file():
        return {"skipped": f"no {path}"}
    shots = pd.read_csv(path)
    shots = shots[shots.role != "hanson"]
    with Pool(workers) as pool:
        got = pool.map(_screen_shot, sorted(shots.shot.astype(int)), chunksize=2)
    role = dict(zip(shots.shot.astype(int), shots.role, strict=True))
    out = {}
    for name in sorted(set(role.values())):
        rows = [g for g in got if role[g["shot"]] == name]
        ran = [g for g in rows if g["not_run"] is None]
        kept = [g for g in ran if g["kept"]]
        out[name] = {
            "shots": len(rows),
            "screened": len(ran),
            "measured_s": round(sum(g["measured_s"] for g in ran), 1),
            "start_failed": sum(g["start_failed"] for g in ran),
            "start_untested": sum(g["start_untested"] for g in ran),
            "gate": dict(sum((Counter(g["gate"]) for g in ran), Counter())),
            "kept_candidates": sum(len(g["kept"]) for g in ran),
            "kept_s": round(sum(b - a for g in ran for a, b in g["kept"]) / 1000, 3),
            "shots_with_kept": len(kept),
            "kept_per_shot": {str(g["shot"]): g["kept"] for g in kept},
        }
    return out


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
        raise SystemExit(
            f"blind cohort shots among the targets or in {table}: {leaked}"
        )
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
        "comparison": comparison(paths, args.workers),
    }
    if args.point:
        record["pointer"] = labels.write_pointer(
            paths.label_tables / EVENT, table, method=method.name, version=args.version
        )
    (OUT / f"onsets_source_{args.version}.json").write_text(
        json.dumps(record, indent=1) + "\n"
    )
    shown = {k: v for k, v in record.items() if k != "per_onset"}
    shown["comparison"] = {
        k: {n: x for n, x in v.items() if n != "kept_per_shot"}
        for k, v in record["comparison"].items()
        if isinstance(v, dict)
    }
    shown["screen_gate"] = {
        k: v for k, v in record["screen_gate"].items() if k != "mhd_source"
    }
    print(json.dumps(shown, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
