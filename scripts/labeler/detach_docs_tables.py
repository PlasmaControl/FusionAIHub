#!/usr/bin/env python
"""Print current exploratory coverage/agreement tables from source JSON records.

No F1-against-consensus indicator table is emitted: no independent benchmark is
available. Geometry tiers remain explicit in every indicator agreement table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from detach_protocol import cnn_tables, metric, plain_summary, table

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "docs/labeler/results"


def load(name):
    path = RESULTS / f"detachment_{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def agreement_table(bench):
    rows = []
    for tier, pairs in bench["pairwise_agreement"]["by_tangtv_tier"].items():
        for pair, e in pairs.items():
            if e["both_vote_bins"]:
                rows.append(
                    [
                        tier,
                        pair.replace("__", " / "),
                        f"{e['both_vote_bins']} / {e['both_vote_shots']}",
                        metric(e["agreement"]),
                        metric(e["kappa"]),
                        metric(e["binary_kappa"]),
                    ]
                )
    return table(
        [
            "TangTV tier",
            "Pair",
            "Bins / shots",
            "Agreement [95% shot CI]",
            "Three-state κ [95% shot CI]",
            "Binary κ [95% shot CI]",
        ],
        rows,
    )


def composition_table(rec):
    return table(
        ["State", "Bins", "Shots", "Shot IDs", "Single-bin intervals"],
        [
            [
                name,
                e["bins"],
                e["shots"],
                ", ".join(map(str, e["shot_ids"])) or "—",
                e["single_50ms_intervals"],
            ]
            for name, e in rec["composition"]["by_state"].items()
            if name != "uncertain"
        ],
    )


def sensitivity_table(rec):
    rows = []
    for family in ("prad", "greenwald"):
        for e in rec["threshold_sensitivity"][family]:
            setting = (
                f"{e['attached_max']:.2f} / {e['detached_min']:.2f}"
                if family == "prad"
                else f"fG≥{e['cue_min']:.2f}"
            )
            rows.append(
                [
                    family,
                    f"{e['shift']:+.2f}",
                    setting,
                    *[
                        e["by_state"][s]["bins"]
                        for s in ("attached", "detached", "marfe")
                    ],
                    f"{e['certain']['bins']} / {e['certain']['shots']}",
                ]
            )
    return table(
        [
            "Family",
            "Shift",
            "Cutoffs / cue",
            "Attached",
            "Detached",
            "MARFE",
            "Certain bins / shots",
        ],
        rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    rec, bench = load("round3"), load("benchmark")
    baselines = {name: load(name) for name in ("ours", "victor")}
    parts = []
    if rec and bench:
        parts += [
            plain_summary(rec, bench),
            "<!-- COMPOSITION -->\n\n" + composition_table(rec),
            "<!-- INDICATOR AGREEMENT -->\n\n" + agreement_table(bench),
            "<!-- THRESHOLD SENSITIVITY -->\n\n" + sensitivity_table(rec),
        ]
    if all(baselines.values()):
        parts.append("<!-- CNN DIAGNOSTICS -->\n\n" + cnn_tables(baselines))
    text = "\n\n".join(parts) + "\n"
    if args.out:
        args.out.write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
