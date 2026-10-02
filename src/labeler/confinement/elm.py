"""Confinement–ELM consistency, with reviewed occupancy and legacy onsets apart."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Paths
from ..events.source_formatters import read_elm
from .labels import MODES, digest


def joint_intervals(confinement, elm, guard_ms=0.0):
    """Partition confinement coverage by ELM assessment state on the same clock.

    Missing review time stays -1; contradictory review states become uncertain
    (2). Unresolved confinement subtypes are excluded, not forced into QH/H.
    A guard trims each confinement span; it never expands annotation coverage.
    """
    if guard_ms < 0 or not np.isfinite(guard_ms):
        raise ValueError("Guard must be finite and nonnegative")
    if not elm.category.isin([0, 1, 2, 3]).all():
        raise ValueError("ELM review states must be 0,1,2,3")
    if not np.isfinite(elm[["t_start", "t_end"]].to_numpy()).all():
        raise ValueError("ELM review bounds must be finite")
    result = []
    known = confinement.loc[(confinement.label >= 0) & confinement.regimes.isin(MODES)]
    for shot, group in known.groupby("shot", sort=True):
        events = defaultdict(list)
        i = 0
        for row in group.itertuples(index=False):
            start, end = row.t_start + guard_ms, row.t_end - guard_ms
            if start < end:
                events[start].append((i, "confinement", row.regimes))
                events[end].append((i, None, None))
                i += 1
        for row in elm.loc[elm.shot == shot].itertuples(index=False):
            if row.t_end <= row.t_start:
                continue
            events[row.t_start].append((i, "elm", int(row.category)))
            events[row.t_end].append((i, None, None))
            i += 1
        axis, active = sorted(events), {}
        for start, end in pairwise(axis):
            for key, kind, value in events[start]:
                if kind is None:
                    active.pop(key, None)
                else:
                    active[key] = (kind, value)
            regimes = {
                value for kind, value in active.values() if kind == "confinement"
            }
            if len(regimes) != 1:
                continue
            states = {value for kind, value in active.values() if kind == "elm"}
            state = -1 if not states else next(iter(states)) if len(states) == 1 else 2
            result.append(
                (
                    int(shot),
                    start,
                    end,
                    next(iter(regimes)),
                    state,
                    end - start,
                    len(states) > 1,
                )
            )
    return pd.DataFrame(
        result,
        columns=[
            "shot",
            "t_start",
            "t_end",
            "regime",
            "elm_state",
            "duration_ms",
            "review_conflict",
        ],
    )


def summarize(joint):
    """Reviewed ELMing occupancy divided only by assessed 0/1 exposure."""
    rows = []
    for mode in MODES:
        group = joint.loc[joint.regime == mode]
        by_state = group.groupby("elm_state").duration_ms.sum()
        absent, present = float(by_state.get(0, 0)), float(by_state.get(1, 0))
        assessed = group.loc[group.elm_state.isin([0, 1])]
        rows.append(
            {
                "regime": mode,
                "confinement_shots": int(group.shot.nunique()),
                "assessed_shots": int(assessed.shot.nunique()),
                "absent_ms": absent,
                "present_ms": present,
                "assessed_ms": absent + present,
                "unknown_ms": float(by_state.get(-1, 0)),
                "uncertain_ms": float(by_state.get(2, 0)),
                "not_observable_ms": float(by_state.get(3, 0)),
                "elming_fraction": present / (absent + present)
                if absent + present
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def onset_exposure(confinement, traces, guard_ms=0.0):
    """Count original positive onset samples over supplied 1 ms sample coverage.

    An archived zero means no onset in that millisecond. It is not a dense
    non-ELMing phase label. The sample cadence is part of the archive contract.
    """
    known = confinement.loc[(confinement.label >= 0) & confinement.regimes.isin(MODES)]
    result = []
    for shot, times, labels in traces:
        group = known.loc[known.shot == shot]
        for row in group.itertuples(index=False):
            start, stop = row.t_start + guard_ms, row.t_end - guard_ms
            if stop <= start:
                continue
            lo, hi = np.searchsorted(times, [start - 1.0, stop], side="right")
            t, values = times[lo:hi], labels[lo:hi]
            exposure = np.maximum(
                0, np.minimum(t + 1.0, stop) - np.maximum(t, start)
            ).sum()
            # Half-open time ownership; no double-counting at a regime boundary.
            count = ((values == 1) & (t >= start) & (t < stop)).sum()
            if exposure > 0:
                result.append((int(shot), row.regimes, float(exposure), int(count)))
    raw = pd.DataFrame(result, columns=["shot", "regime", "exposure_ms", "onsets"])
    if not len(raw):
        return raw
    return raw.groupby(["shot", "regime"], as_index=False)[
        ["exposure_ms", "onsets"]
    ].sum()


def _onset_summary(per_shot, replicates=2000, seed=20261001):
    rng = np.random.default_rng(seed)
    records = []
    for mode in MODES:
        group = per_shot.loc[per_shot.regime == mode]
        time, events = group.exposure_ms.to_numpy() / 1000.0, group.onsets.to_numpy()
        rate = float(events.sum() / time.sum()) if time.sum() else None
        ci = None
        if len(time) >= 2 and events.sum() > 0:
            indices = rng.integers(0, len(time), size=(replicates, len(time)))
            sampled = events[indices].sum(axis=1) / time[indices].sum(axis=1)
            ci = np.percentile(sampled, [2.5, 97.5]).tolist()
        records.append(
            {
                "regime": mode,
                "shots": len(group),
                "exposure_s": float(time.sum()),
                "onsets": int(events.sum()),
                "onset_positive_shots": int((events > 0).sum()),
                "onsets_per_second": rate,
                "rate_ci95_shot_bootstrap": ci,
            }
        )
    return records


def build(run_dir: Path, elm_review: Path, *, legacy_paths=()):
    run_dir, elm_review = Path(run_dir), Path(elm_review)
    payload = elm_review.read_bytes()
    elm = pd.read_csv(io.BytesIO(payload))
    merged = pd.read_csv(run_dir / "merged_intervals.csv")
    # Guards apply to regime boundaries, not provenance changes inside one
    # same-regime phase. Collapse adjacent same-regime spans before trimming.
    phases = []
    for row in merged.itertuples(index=False):
        if (
            phases
            and phases[-1][0] == row.shot
            and phases[-1][2] == row.t_start
            and phases[-1][3:] == [row.regimes, row.label]
        ):
            phases[-1][2] = row.t_end
        else:
            phases.append([row.shot, row.t_start, row.t_end, row.regimes, row.label])
    confinement = pd.DataFrame(
        phases, columns=["shot", "t_start", "t_end", "regimes", "label"]
    )
    out = run_dir / "elm_study"
    out.mkdir(exist_ok=True)
    (out / "elm_review_snapshot.csv").write_bytes(payload)
    all_joint, summaries = [], {}
    for guard in (0.0, 50.0, 100.0):
        joint = joint_intervals(confinement, elm, guard)
        joint["guard_ms"] = guard
        all_joint.append(joint)
        table = summarize(joint)
        summaries[str(int(guard))] = json.loads(table.to_json(orient="records"))
    joint = pd.concat(all_joint, ignore_index=True)
    joint.to_csv(out / "reviewed_joint_intervals.csv", index=False)
    candidates = joint.loc[(joint.regime == "QH") & (joint.elm_state == 1)]
    candidates.to_csv(out / "qh_elm_candidates.csv", index=False)
    legacy_results = {}
    if legacy_paths:
        traces = list(read_elm(*legacy_paths))
        events = []
        for shot, times, labels in traces:
            positive = times[labels == 1]
            for phase in confinement.loc[
                (confinement.shot == shot)
                & (confinement.regimes == "QH")
                & (confinement.label >= 0)
            ].itertuples(index=False):
                for time in positive[
                    (positive >= phase.t_start) & (positive < phase.t_end)
                ]:
                    distance = min(time - phase.t_start, phase.t_end - time)
                    events.append(
                        (
                            int(shot),
                            float(time),
                            phase.t_start,
                            phase.t_end,
                            float(distance),
                            phase.t_start + 100 <= time < phase.t_end - 100,
                        )
                    )
        candidates = pd.DataFrame(
            events,
            columns=[
                "shot",
                "onset_ms",
                "qh_start_ms",
                "qh_end_ms",
                "boundary_distance_ms",
                "inside_100ms_guard",
            ],
        )
        candidates.to_csv(out / "legacy_qh_onset_candidates.csv", index=False)
        for guard in (0.0, 50.0, 100.0):
            table = onset_exposure(confinement, traces, guard)
            table.to_csv(out / f"legacy_onsets_guard{int(guard)}.csv", index=False)
            legacy_results[str(int(guard))] = _onset_summary(table)
    shared = set(map(int, elm.shot)) & set(map(int, merged.shot))
    summary = {
        "study": "Exploratory consistency of confinement regimes and ELM labels",
        "confinement_sha256": digest(run_dir / "merged_intervals.csv"),
        "reviewed_elm_path": str(elm_review.resolve()),
        "reviewed_elm_sha256": hashlib.sha256(payload).hexdigest(),
        "reviewed_elm_shots": int(elm.shot.nunique()),
        "common_reviewed_shots": sorted(shared),
        "guards_ms": [0, 50, 100],
        "reviewed_occupancy": summaries,
        "legacy_onset_rates": legacy_results,
        "legacy_sources": [
            {"path": str(p.resolve()), "sha256": digest(p)} for p in legacy_paths
        ],
        "units": "Reviewed outcome: fraction of assessed ELMing phase time. Legacy outcome: positive onset samples per supplied sample second.",
        "limitations": [
            "Reviewed ELM overlap is sparse (see common_reviewed_shots); no population association is established by this source-conditioned study.",
            "An unreviewed/uncertain/unobservable ELM interval is not evidence of ELM absence.",
            "Assisted review and dependent legacy annotations are not independent gold validation.",
            "QH/ELM overlap flags a candidate inconsistency; the pipeline never relabels QH or ELM from the other's definition.",
            "Legacy onset zeros cannot substitute for revised dense ELMing phase absence.",
            "Zero observed onsets and a degenerate nonparametric interval do not establish a true zero event rate.",
        ],
    }
    summary["output_hashes"] = {p.name: digest(p) for p in sorted(out.glob("*.csv"))}
    (out / "study.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    return summary


def main(argv=None):
    paths = Paths.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=paths.root / "confinement/v1")
    parser.add_argument(
        "--elm-review",
        type=Path,
        default=paths.label_tables / "edge_localized_mode/review/labels.csv",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Separate exploratory original-onset study",
    )
    args = parser.parse_args(argv)
    legacy = (
        [
            paths.label_tables / "edge_localized_mode/raw" / name
            for name in ("elm_labels_dict.pkl", "elm_labels_dict_wpqh.pkl")
        ]
        if args.legacy
        else []
    )
    result = build(args.run_dir, args.elm_review, legacy_paths=legacy)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "common_reviewed_shots",
                    "reviewed_occupancy",
                    "legacy_onset_rates",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
