#!/usr/bin/env python
"""Choose the RWM comparison shots, build the slice table and write the label windows.

Reads the candidate pool (`labeler.rwm.shots.choose`, written by
``scripts/labeler/rwm_pool.py``) and the raw cache that ``rwm_fetch.py`` filled, and:

1. measures every pool shot's flat-top beta_N and beta_N / l_i (95th percentile);
2. matches each of Jeremy Hanson's 33 shots to ``--per-hanson`` comparison shots of the
   same campaign, nearest in those two numbers, without reusing a shot (greedy, the
   Hanson shots with the highest beta_N first so the scarce high-beta shots are not
   used up), and records how well the matching balanced the two sets;
3. stacks trailing features on offline inputs (`labeler.rwm.data`) with the
   `rwm_candidates` screen's call, into ``$LABELER_ROOT/round4/rwm/slices.parquet``;
4. writes uncertain windows before listed onsets (category 2), since the sources
   do not define ONSET_TIME as a detection/threshold time,
   and assumed-absent time before the first precursor (category 0) on Hanson shots.
   Explicit unassessed intervals (category 4) preserve every hole and all post-onset
   physical state; comparison screen spans are uncertain (category 2), with remaining
   comparison time explicitly unassessed.

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
from labeler.events import raw, rwm
from labeler.events.interval_tables import validate_intervals
from labeler.rwm import data, features, labels, shots

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


def cohort_overlap(shots_used, paths: Paths) -> dict:
    """How many of the shots used are in the frozen cohort, by its split.

    The blind test split must never train or tune anything; none of these shots is in it
    if the counts are zero.
    """
    path = paths.label_tables / "catalog" / "cohort.csv"
    if not path.is_file():
        return {"checked": False}
    cohort = pd.read_csv(path)
    hit = cohort[cohort.shot.isin({int(s) for s in shots_used})]
    return {
        "checked": True,
        "cohort_shots": len(cohort),
        "in_cohort": len(hit),
        "by_split": {str(k): int(v) for k, v in hit.split.value_counts().items()},
    }


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

    matched = shots.match_comparison(stats, args.per_hanson)
    stats = stats.merge(matched, on="shot", how="left")
    stats["selected"] = (stats.role == "hanson") | stats.matched_to.notna()
    stats["matched_to"] = stats.matched_to.astype("Int64")
    selected = stats[stats.usable & stats.selected].copy()
    selected["role_kind"] = np.where(selected.role == "hanson", "hanson", "comparison")

    # The slice table, with the candidate screen's call on every row.
    pieces, spans_by_shot = [], {}
    for row in selected.itertuples():
        shot = int(row.shot)
        hanson = row.role == "hanson"
        frame = data.shot_table(
            shot,
            paths,
            onsets.get(shot, []),
            other_onsets_ms=other.get(shot, []),
            hanson=hanson,
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
                assumed_absent=True,
            )
        else:
            lo, hi = row.flattop_start_ms, row.flattop_end_ms
            spans = [
                (max(lo, a), min(hi, b))
                for a, b in spans_by_shot.get(shot, [])
                if max(lo, a) < min(hi, b)
            ]
            rows += [(shot, labels.UNCERTAIN, float(a), float(b)) for a, b in spans]
            rows += [
                (shot, labels.UNASSESSED, float(a), float(b))
                for a, b in labels._subtract((lo, hi), spans)
            ]
    windows = pd.DataFrame(rows, columns=["shot", "category", "t_start", "t_end"])
    windows[["t_start", "t_end"]] = windows[["t_start", "t_end"]].round(3)
    windows["confidence"] = ""
    tiers = windows.category.map(
        {
            labels.ABSENT: "assumed_absent",
            labels.UNCERTAIN: "unlabelled_screen",
            labels.UNASSESSED: "unassessed",
        }
    )
    tiers.loc[
        windows.shot.isin(hanson_shots) & (windows.category == labels.UNCERTAIN)
    ] = "onset_window_uncertain"
    windows["attrs"] = tiers.map(
        lambda tier: json.dumps({"evidence_tier": tier, "coverage_verified": False})
    )
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
    stats[keep].to_csv(LABEL_DIR / f"{STEM}.shots.csv", index=False)
    stats[keep].to_csv(out_dir / "shots.csv", index=False)

    hanson_rows = windows[windows.shot.isin(hanson_shots)]
    probe_path = REPO / "outputs" / "labeler" / "rwm" / "sensor_probe.json"
    probe = json.loads(probe_path.read_text()) if probe_path.is_file() else {}
    onset_provenance = {
        "field": "ONSET_TIME",
        "detection_or_threshold_time_verified": False,
        "finding": (
            "Hanson source CSVs and category README name onset times but do not "
            "define growth-start versus detection/threshold timing. Piccione "
            "digests describe NSTX threshold t_RWM and cannot establish the "
            "meaning of Hanson's DIII-D ONSET_TIME."
        ),
        "sources_searched": [
            "data/events/resistive_wall_mode/raw/rwm_onsets_2017.csv",
            "data/events/resistive_wall_mode/raw/rwm_onsets_2024.csv",
            "main:data/events/resistive_wall_mode/README.md",
            "main:.tmp/label_papers/Piccione_2022_Nucl._Fusion_62_036002.md",
            "main:.tmp/label_papers/outside/Piccione_tsdw2021_RWM_poster.md",
        ],
        "physical_window_category": labels.UNCERTAIN,
        "forecast_target": "separate 100 ms point-time forecast target; unchanged",
        "source_audit": "outputs/labeler/rwm/sensor_probe.json#/source_search",
    }
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
            "n1_shots": len(onsets),
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
            "balance": shots.balance(stats, set(matched.shot)),
        },
        "cohort_overlap": cohort_overlap(selected.shot, paths),
        "feature_coverage": {
            f"{role}_{year}": {
                name: {
                    "shots_with_any": int(
                        group.groupby("shot")[name]
                        .apply(lambda v: v.notna().any())
                        .sum()
                    ),
                    "slice_fraction": float(group[name].notna().mean()),
                }
                for name in features.FEATURES
            }
            for (role, year), group in slices.groupby(["role", "campaign"])
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
            "conventional_weak_windows": int(
                (hanson_rows.category == labels.PRESENT).sum()
            ),
            "uncertain_onset_windows": int(
                (hanson_rows.category == labels.UNCERTAIN).sum()
            ),
            "assumed_absent_spans": int((hanson_rows.category == labels.ABSENT).sum()),
            "verified_absent_spans": 0,
            "hanson_unassessed_spans": int(
                (hanson_rows.category == labels.UNASSESSED).sum()
            ),
            "comparison_unassessed_spans": int(
                (
                    (windows.category == labels.UNASSESSED)
                    & ~windows.shot.isin(hanson_shots)
                ).sum()
            ),
            "candidate_spans_on_comparison_shots": int(
                (
                    (windows.category == labels.UNCERTAIN)
                    & ~windows.shot.isin(hanson_shots)
                ).sum()
            ),
            "growth_ms": labels.GROWTH_MS,
            "horizon_ms": labels.HORIZON_MS,
            "post_ms": labels.POST_MS,
        },
        "evidence": {
            "verified": "Hanson listed onset points only; no verified negative coverage",
            "windows": "uncertain 20 ms pre-onset convention; direction and extent unverified",
            "onset_time_provenance": onset_provenance,
            "negative_assumption": "onset listing is complete only before first onset precursor for physical category 0; not termination evidence",
            "unassessed": "explicit category 4 for precursors and post-onset physical state through end of analysis; comparison time outside screen spans",
            "primary_scoring": "before last n=1 onset only; n=2-only shots excluded",
            "comparison": "unlabelled, never primary supervised negatives",
        },
        "input_audit": {
            "dusbradial_zero_2014_shots": int(
                slices[slices.campaign == 2014]
                .groupby("shot")
                .lock_v.apply(lambda v: v.notna().any() and v.dropna().eq(0).all())
                .sum()
            ),
            "dusbradial_corrupted_shot_range": [176030, 176912],
            "dusbradial_by_role_campaign": {
                f"{role}_{year}": {
                    "shots": int(group.shot.nunique()),
                    "zero_only_shots": int(
                        group.groupby("shot")
                        .lock_v.apply(
                            lambda v: v.notna().any() and v.dropna().eq(0).all()
                        )
                        .sum()
                    ),
                    "nonzero_shots": int(
                        group.groupby("shot")
                        .lock_v.apply(
                            lambda v: v.notna().any() and v.dropna().gt(0).any()
                        )
                        .sum()
                    ),
                    "missing_shots": int(
                        group.groupby("shot")
                        .lock_v.apply(lambda v: v.isna().all())
                        .sum()
                    ),
                }
                for (role, year), group in slices.groupby(["role", "campaign"])
            },
            "dusbradial_2014_nonzero_shots": [
                int(s)
                for s, v in slices[slices.campaign == 2014].groupby("shot").lock_v
                if v.dropna().gt(0).any()
            ],
            "corruption_source": "src/labeler/features/namespace.py (Fu et al. 2020 note)",
            "dusbradial_in_primary_model": False,
            "fetch_specs": sorted(raw.FETCH_SPECS),
            "low_frequency_n1_rwm_sensor_specs": [],
            "low_frequency_sensor_fetch_attempted": bool(
                probe.get("fetch_attempted", False)
            ),
            "low_frequency_sensor_status": probe.get(
                "status", "sensor candidate source audit has not been run"
            ),
            "low_frequency_sensor_probe": "outputs/labeler/rwm/sensor_probe.json",
            "low_frequency_sensor_candidate_specs": probe.get("candidate_specs", []),
        },
    }
    meta = {
        "categories": {
            "0": "absent",
            "1": "present",
            "2": "uncertain",
            "4": "unassessed",
        },
        "category": "resistive_wall_mode",
        "columns": [
            "shot",
            "category",
            "t_start",
            "t_end",
            "confidence",
            "attrs",
        ],
        "coverage": (
            "Hanson category 0 is assumed absence before the first precursor; "
            "Hanson category 2 is the uncertain 20 ms pre-onset convention. "
            "Comparison category 2 is the rwm_candidates screen, unlabelled and "
            "not negative. Category 1 has no duration rows here: only the raw "
            "onset points are verified. Reviewed negative coverage is unknown."
        ),
        "made_at": summary["made_at"],
        "made_by": "scripts/labeler/rwm_build.py",
        "n_rows": summary["windows"]["rows"],
        "n_shots": int(windows.shot.nunique()),
        "rules": {
            "present": (
                "no present duration inferred; verified point onsets remain in "
                "the curated format source"
            ),
            "absent": (
                f"high-current time only before the first listed onset minus "
                f"{labels.HORIZON_MS:g} ms, assuming complete onset listing. "
                "No post-onset physical absence is inferred without mode termination evidence"
            ),
            "uncertain": (
                f"Hanson: [{labels.GROWTH_MS:g} ms before ONSET_TIME, ONSET_TIME), "
                "of either mode number, as a convention with unverified direction "
                "and extent. Comparison: rwm_candidates screen spans."
            ),
            "unassessed": "explicit precursor holes and physical post-onset time to analysis end, apart from later uncertain onset windows; comparison time outside screen spans",
        },
        "onset_time_provenance": onset_provenance,
        "evidence_tiers": {
            "onset_window_uncertain": "20 ms pre-onset convention; ONSET_TIME meaning and physical extent unverified",
            "assumed_absent": "Hanson time before first precursor; completeness assumption",
            "unlabelled_screen": "screen candidates on comparison shots; not negatives",
            "unassessed": "no assessed physical state or termination evidence",
        },
        "reader_contract": "review reader preserves all explicit rows and attrs; tier-less saves of tiered sources are rejected",
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
