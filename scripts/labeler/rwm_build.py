#!/usr/bin/env python
"""Choose the RWM comparison shots, build the slice table and write the label windows.

Reads the candidate pool (`labeler.rwm.shots.choose`, written by
``scripts/labeler/rwm_pool.py``) and the raw cache that ``rwm_fetch.py`` filled, and:

1. measures every pool shot's flat-top beta_N and beta_N / l_i (95th percentile);
2. matches each of Jeremy Hanson's 33 shots to ``--per-hanson`` comparison shots of the
   same campaign, nearest in those two numbers, without reusing a shot (greedy, the
   Hanson shots with the highest beta_N first so the scarce high-beta shots are not
   used up), and records how well the matching balanced the two sets;
3. stacks the causal slice features of the chosen shots (`labeler.rwm.data`) with the
   `rwm_candidates` screen's call, into ``$LABELER_ROOT/round4/rwm/slices.parquet``;
4. writes the label windows: the growth window before every listed onset (category 1)
   and the flat-top outside the precursor and aftermath of a listed onset (category 0)
   on the Hanson shots, and the screen's uncertain spans (category 2) on the comparison
   shots, which are unlabelled.

    python scripts/labeler/rwm_build.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from labeler.config import Paths
from labeler.events import rwm
from labeler.events.interval_tables import validate_intervals
from labeler.rwm import data, features, labels

LABEL_DIR = REPO / "data" / "events" / "resistive_wall_mode" / "extend_rwm_growth"
STEM = "rwm_windows"
CAMPAIGN_SPLIT = 170000  # shots below are the 2014 campaign, above the 2018 one


def campaign(shot: int) -> int:
    return 2014 if shot < CAMPAIGN_SPLIT else 2018


def measure(shot: int, paths: Paths) -> dict:
    """Flat-top beta_N statistics of one shot, or the reason it cannot be used."""
    try:
        signals = data.load_signals(shot, paths)
    except Exception as error:  # noqa: BLE001 - recorded in the table
        return {"usable": False, "reason": re.sub(r"^shot \d+: ", "", str(error))[:80]}
    window = features.flattop_window(*signals["ip"], 0.5)
    t, betan = signals["betan"]
    _, li = signals["li"]
    if window is None:
        return {"usable": False, "reason": "no flat-top"}
    inside = (t >= window[0]) & (t <= window[1]) & np.isfinite(betan) & (li > 0)
    if inside.sum() < 5:
        return {"usable": False, "reason": "fewer than 5 EFIT slices in the flat-top"}
    return {
        "usable": True,
        "reason": "",
        "flattop_start_ms": window[0],
        "flattop_end_ms": window[1],
        "betan_p95": float(np.percentile(betan[inside], 95)),
        "betan_over_li_p95": float(np.percentile(betan[inside] / li[inside], 95)),
    }


def match(stats: pd.DataFrame, per_hanson: int) -> pd.DataFrame:
    """The chosen comparison shots, each tagged with the Hanson shot it was matched to.

    Within a campaign every comparison shot is described by its standardised flat-top
    beta_N and beta_N / l_i (95th percentiles, the standard deviation taken over the
    campaign's pool). The Hanson shots, highest beta_N first, each take the
    `per_hanson` nearest comparison shots not yet taken.
    """
    chosen = []
    for year, group in stats[stats.usable].groupby("campaign"):
        hanson = group[group.role == "hanson"].sort_values("betan_p95", ascending=False)
        pool = group[group.role != "hanson"].copy()
        columns = ["betan_p95", "betan_over_li_p95"]
        scale = group[columns].std().replace(0, 1.0)
        taken: set[int] = set()
        for row in hanson.itertuples():
            free = pool[~pool.shot.isin(taken)]
            distance = (
                (
                    (free[columns] - np.array([row.betan_p95, row.betan_over_li_p95]))
                    / scale
                )
                ** 2
            ).sum(axis=1)
            for shot in free.loc[distance.nsmallest(per_hanson).index, "shot"]:
                taken.add(int(shot))
                chosen.append({"shot": int(shot), "matched_to": int(row.shot)})
    return pd.DataFrame(chosen, columns=["shot", "matched_to"])


def balance(stats: pd.DataFrame, chosen_shots) -> dict:
    """Mean flat-top beta_N of the Hanson, matched and unmatched pool shots, by campaign."""
    out = {}
    for year, group in stats[stats.usable].groupby("campaign"):
        hanson = group[group.role == "hanson"]
        rest = group[group.role != "hanson"]
        picked = rest[rest.shot.isin(chosen_shots)]
        left = rest[~rest.shot.isin(chosen_shots)]
        out[str(year)] = {
            name: {
                "shots": len(frame),
                "betan_p95_mean": float(frame.betan_p95.mean()) if len(frame) else None,
                "betan_over_li_p95_mean": (
                    float(frame.betan_over_li_p95.mean()) if len(frame) else None
                ),
            }
            for name, frame in (
                ("hanson", hanson),
                ("matched", picked),
                ("pool_not_chosen", left),
            )
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pool", type=Path)
    parser.add_argument("--per-hanson", type=int, default=4)
    args = parser.parse_args()
    paths = Paths.from_env()
    out_dir = paths.root / "round4" / "rwm"
    pool = pd.read_csv(args.pool or out_dir / "shots_pool.csv")
    table = rwm.onset_table(paths)
    onsets: dict[int, list] = {}
    other: dict[int, list] = {}
    for row in table.itertuples():
        (onsets if row.ntor == 1 else other).setdefault(int(row.shot), []).append(
            float(row.t_ms)
        )
    onsets = {s: labels.merge_close(v) for s, v in onsets.items()}
    other = {s: labels.merge_close(v) for s, v in other.items()}
    hanson_shots = sorted(table.shot.unique())

    stats = pool.copy()
    stats["campaign"] = stats.shot.map(campaign)
    measured = pd.DataFrame([measure(int(s), paths) for s in stats.shot])
    stats = pd.concat([stats.reset_index(drop=True), measured], axis=1)
    missing_hanson = sorted(set(hanson_shots) - set(stats.shot[stats.usable]))

    matched = match(stats, args.per_hanson)
    stats = stats.merge(matched, on="shot", how="left")
    stats["selected"] = (stats.role == "hanson") | stats.matched_to.notna()
    stats["matched_to"] = stats.matched_to.astype("Int64")
    selected = stats[stats.usable & stats.selected].copy()
    selected["role_kind"] = np.where(selected.role == "hanson", "hanson", "comparison")

    # The slice table, with the candidate screen's call on every row.
    pieces, spans_by_shot = [], {}
    for row in selected.itertuples():
        shot = int(row.shot)
        examined = row.role == "hanson"
        frame = data.shot_table(
            shot,
            paths,
            onsets.get(shot, []),
            other_onsets_ms=other.get(shot, []),
            examined=examined,
        )
        found = rwm.detect(shot, paths)
        spans = [(a, b) for a, b, _ in found.spans]
        spans_by_shot[shot] = spans
        call = np.zeros(len(frame))
        for a, b in spans:
            call[(frame.t_ms >= a) & (frame.t_ms <= b)] = 1.0
        frame["rule_candidate"] = call
        frame["role"] = row.role_kind
        frame["campaign"] = int(row.campaign)
        pieces.append(frame)
    slices = pd.concat(pieces, ignore_index=True)
    slices.to_parquet(out_dir / "slices.parquet", index=False)

    # The label windows.
    rows = []
    for row in selected.itertuples():
        shot = int(row.shot)
        if row.role == "hanson":
            every = sorted(onsets.get(shot, []) + other.get(shot, []))
            rows += labels.window_rows(
                shot,
                every,
                (row.flattop_start_ms, row.flattop_end_ms),
                examined=True,
            )
        else:
            rows += [
                (shot, labels.UNCERTAIN, float(a), float(b))
                for a, b in spans_by_shot.get(shot, [])
            ]
    windows = pd.DataFrame(rows, columns=["shot", "category", "t_start", "t_end"])
    windows["confidence"] = ""
    windows = windows.sort_values(["shot", "t_start", "t_end"], ignore_index=True)
    validate_intervals(windows)
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    windows.to_csv(LABEL_DIR / f"{STEM}.csv", index=False)

    keep = [
        "shot",
        "role",
        "campaign",
        "matched_to",
        "run",
        "run_title",
        "mpid",
        "usable",
        "reason",
        "selected",
        "flattop_start_ms",
        "flattop_end_ms",
        "betan_p95",
        "betan_over_li_p95",
    ]
    stats[keep].to_csv(LABEL_DIR / "shots.csv", index=False)
    stats[keep].to_csv(out_dir / "shots.csv", index=False)

    hanson_rows = windows[windows.shot.isin(hanson_shots)]
    summary = {
        "script": "scripts/labeler/rwm_build.py",
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "pool": {
            "shots": len(stats),
            "usable": int(stats.usable.sum()),
            "unusable": {
                reason: int(n)
                for reason, n in stats[~stats.usable].reason.value_counts().items()
            },
        },
        "hanson": {
            "shots": len(hanson_shots),
            "shots_without_inputs": missing_hanson,
            "listed_onsets": len(table),
            "n1_events": int(sum(len(v) for v in onsets.values())),
            "n2_events": int(sum(len(v) for v in other.values())),
        },
        "comparison": {
            "per_hanson": args.per_hanson,
            "chosen": int(len(selected) - (selected.role == "hanson").sum()),
            "by_campaign": {
                str(k): int(v)
                for k, v in selected[selected.role != "hanson"]
                .campaign.value_counts()
                .items()
            },
            "balance": balance(stats, set(matched.shot)),
        },
        "slices": {
            "rows": len(slices),
            "labels": {
                str(k): int(v)
                for k, v in slices[slices.role == "hanson"].label.value_counts().items()
            },
            "unlabelled_comparison_rows": int((slices.role == "comparison").sum()),
        },
        "windows": {
            "csv": str((LABEL_DIR / f"{STEM}.csv").relative_to(REPO)),
            "rows": len(windows),
            "present_growth_windows": int(
                (hanson_rows.category == labels.PRESENT).sum()
            ),
            "absent_spans": int((hanson_rows.category == labels.ABSENT).sum()),
            "candidate_spans_on_comparison_shots": int(
                (windows.category == labels.UNCERTAIN).sum()
            ),
            "growth_ms": labels.GROWTH_MS,
            "horizon_ms": labels.HORIZON_MS,
            "post_ms": labels.POST_MS,
        },
    }
    meta = {
        "categories": {"0": "absent", "1": "present", "2": "uncertain"},
        "category": "resistive_wall_mode",
        "columns": ["shot", "category", "t_start", "t_end", "confidence"],
        "coverage": (
            "Hanson shots only for categories 0 and 1; category 2 is the rwm_candidates "
            "screen on the matched comparison shots, which are unlabelled and not "
            "negatives. A shot absent from the table was not examined."
        ),
        "made_at": summary["made_at"],
        "made_by": "scripts/labeler/rwm_build.py",
        "n_rows": summary["windows"]["rows"],
        "n_shots": int(windows.shot.nunique()),
        "rules": {
            "present": (
                f"the {labels.GROWTH_MS:g} ms before each listed onset, of either mode "
                "number (a growth window; see outputs/labeler/rwm/growth.json)"
            ),
            "absent": (
                f"the high-current window (Ip at least half its peak) outside "
                f"[onset - {labels.HORIZON_MS:g}, onset + {labels.POST_MS:g}] ms of every "
                "listed onset; the precursor before a growth window and the aftermath "
                "carry no row (not assessed)"
            ),
            "uncertain": "rwm_candidates screen spans, comparison shots only",
        },
        "source": "data/events/resistive_wall_mode/raw/rwm_onsets_{2017,2024}.csv",
        "summary": "outputs/labeler/rwm/shots.json",
    }
    (LABEL_DIR / f"{STEM}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    target = REPO / "outputs" / "labeler" / "rwm" / "shots.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
