#!/usr/bin/env python
"""Score the benchmark methods on spans whose reviewed boundaries left the clock.

The D-alpha clock seeded the review: many reviewed crowd starts and ends sit within
1 ms of a clock boundary. The reviewer's own contribution is clearest where both
boundaries of a span lie more than 1 ms from the clock's. This script scores
`elm-ours`, `elm-clock` and (on the BES subset) ELM-O on exactly those spans' bins.

A span is `moved` when no boundary of it lies within 1 ms of a clock boundary of the
same role: a present span's start against the clock's span starts and its end against
the clock's ends; an absent span's start against the clock's ends and its end against
the clock's starts (an absent span is the gap between clock spans). Two strata, both
on the primary interior bins of each set, saved predictions and thresholds:

- `moved_spans`: bins of moved present spans and of moved absent spans.
- `moved_present_all_absent`: bins of moved present spans and every absent bin.

First run `elm_ours_evaluate.py --run cv2`. Output
`outputs/labeler/elm/ours/moved_boundary_stratum.json`.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from labeler.config import Paths, git_sha, sha256_of
from labeler.elm import compare, methods, score, swap, train

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/labeler/elm/ours/moved_boundary_stratum.json"
NEAR_MS = 1.0
METRICS = ("auroc", "auprc", "f1", "precision", "recall", "false_alarm_bin_rate")
STRATA = {
    "moved_spans": "bins of moved present and moved absent spans",
    "moved_present_all_absent": "bins of moved present spans and every absent bin",
}


def moved_flags(spans, clock) -> np.ndarray:
    """Per review span: True when no boundary lies within `NEAR_MS` of the clock's."""
    starts = clock.t_start_ms.to_numpy(float)
    ends = clock.t_end_ms.to_numpy(float)

    def near(value, edges):
        return bool(edges.size and np.abs(edges - value).min() <= NEAR_MS)

    flags = []
    for row in spans.itertuples():
        if row.kind == "absent":
            seeded = near(row.t_start, ends) or near(row.t_end, starts)
        else:
            seeded = near(row.t_start, starts) or near(row.t_end, ends)
        flags.append(not seeded)
    return np.asarray(flags, dtype=bool)


def stratum_keep(strata, bins, spans, moved):
    """Boolean mask of the bins in each stratum."""
    span_moved = moved[bins.span]
    present = bins.kind != "absent"
    return {
        "moved_spans": span_moved,
        "moved_present_all_absent": (present & span_moved) | ~present,
    }[strata]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="cv2")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    paths = Paths.from_env()
    data = train.load(paths)
    sets = compare.load_sets(paths, data)
    elmo, clock = compare.load_detected(paths)
    sweep = compare.load_elmo_sweep(paths)
    oof = methods.Oof(paths.root / "round4/elm/cv" / args.run)
    empty = methods.span_frame([], [])
    record = {
        "git": git_sha(full=True),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "script_sha256": sha256_of(Path(__file__)),
        "run": args.run,
        "clock_source": str(paths.root / compare.CLOCK_CSV),
        "near_ms": NEAR_MS,
        "definition": " ".join(" ".join(__doc__.split("\n\n")[1:3]).split()),
        "strata": STRATA,
        "cohort_test_shots_used": 0,
        "sets": {},
    }
    for panel, sdef in sets.items():
        moved = {s: moved_flags(data[s].spans, clock.get(s, empty)) for s in sdef.shots}
        body = {"n_shots_in_set": len(sdef.shots), "strata": {}}
        for name, definition in STRATA.items():
            keep = {
                s: stratum_keep(name, sdef.bins[s], data[s].spans, moved[s])
                for s in sdef.shots
            }
            parts = {"elm-ours": [], "elm-clock": []}
            if sdef.has_elmo:
                parts["elm-elmo"] = []
            for s in sdef.shots:
                bins = methods.restrict_bins(sdef.bins[s], keep[s])
                cover = sdef.cover[s]
                spans = data[s].spans
                parts["elm-ours"].append(
                    methods.trace_part(
                        spans, s, bins, cover, oof.trace(s)[0], oof.threshold[s]
                    )
                )
                parts["elm-clock"].append(
                    methods.span_part(spans, s, bins, cover, clock.get(s, empty))
                )
                if sdef.has_elmo:
                    part = methods.span_part(spans, s, bins, cover, elmo.get(s, empty))
                    part.score = swap.sweep_bin_scores(sweep[sweep.shot == s], bins)
                    parts["elm-elmo"].append(part)
            summary = methods.summarise_methods(
                parts, score.draws(len(sdef.shots)), "elm-ours"
            )
            n_present = sum(
                int(((sdef.bins[s].kind != "absent") & keep[s]).sum())
                for s in sdef.shots
            )
            n_bins = sum(int(keep[s].sum()) for s in sdef.shots)
            span_ids = {
                kind: sum(
                    len(
                        np.unique(
                            sdef.bins[s].span[keep[s] & is_kind(sdef.bins[s], kind)]
                        )
                    )
                    for s in sdef.shots
                )
                for kind in ("present", "absent")
            }
            shots_with = [s for s in sdef.shots if keep[s].any()]
            positive_shots = [
                s
                for s in sdef.shots
                if ((sdef.bins[s].kind != "absent") & keep[s]).any()
            ]
            body["strata"][name] = {
                "definition": definition,
                "bins": n_bins,
                "positive_bins": n_present,
                "present_spans": span_ids["present"],
                "absent_spans": span_ids["absent"],
                "shots_with_bins": len(shots_with),
                "shots_with_positive_bins": len(positive_shots),
                "methods": {
                    n: {
                        "point": {m: v["point"].get(m) for m in METRICS},
                        "ci95": {m: v["ci95"].get(m) for m in METRICS},
                    }
                    for n, v in summary["methods"].items()
                    if n in parts
                },
                "paired_ours_minus": {
                    k: {"value": v["value"], "ci95": v["ci95"]}
                    for k, v in summary["paired"].items()
                    if k.rsplit(": ", 1)[-1] in METRICS
                },
            }
        record["sets"][panel] = body
    # context: present spans overall, to say how many are moved
    for panel, sdef in sets.items():
        total = moved_total = 0
        for s in sdef.shots:
            bins = sdef.bins[s]
            flags = moved_flags(data[s].spans, clock.get(s, empty))
            present_idx = np.unique(bins.span[bins.kind != "absent"])
            total += len(present_idx)
            moved_total += int(flags[present_idx].sum())
        record["sets"][panel]["present_spans_with_bins"] = total
        record["sets"][panel]["present_spans_moved"] = moved_total
    args.out.write_text(json.dumps(record, indent=1) + "\n")
    for panel, body in record["sets"].items():
        for name, st in body["strata"].items():
            print(
                panel,
                name,
                {k: st[k] for k in ("bins", "positive_bins", "present_spans")},
                {n: m["point"]["f1"] for n, m in st["methods"].items()},
            )
    return 0


def is_kind(bins, kind):
    return (bins.kind != "absent") if kind == "present" else (bins.kind == "absent")


if __name__ == "__main__":
    raise SystemExit(main())
