#!/usr/bin/env python
"""Audit current nonblind TM labels, coherent criterion rates and screening coverage.

No historical label or blind signal artifacts are opened. Sample rates use the
uniform RMS grid and its median time step; interval durations use unioned spans.
Uncertainty takes priority over presence; missing acquisition is unobservable.
Frequency and Mirnov criterion rates condition on measured respective inputs;
combined span/seed support rates condition on all valid RMS samples.

Criterion pass rates are split by toroidal number: the present rows of n are the time
inside n's own intervals (not the other n's), and the absent rows are the time the
finished labels call absent. Absent rates are therefore post-labelling: time that passed
a weak screen was moved out of absent by construction, so a low absent pass rate is
partly a consequence of the labelling, not an independent measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tm_label import line_evidence

from labeler.config import git_sha
from labeler.events.interval_tables import parse_attrs
from labeler.tearing import rule, scoring

REPO = Path(__file__).resolve().parents[2]
OUT = Path(os.environ["LABELER_ROOT"]) / "round4/tm"


def longest(t, mask):
    dt = float(np.median(np.diff(t)))
    return max((b - a for a, b in zip(*rule._runs(mask), strict=True)), default=0) * dt


def uncertain_reasons(table):
    """Number and seconds of uncertain rows (category 2) by their `reason` attribute."""
    rows = table[table.category.eq(2) & (table.t_end > table.t_start)]
    reasons = rows["attrs"].map(lambda a: parse_attrs(a).get("reason", "ramp_up"))
    seconds = (rows.t_end - rows.t_start) / 1000.0
    return {
        reason: {
            "rows": int((reasons == reason).sum()),
            "seconds": float(seconds[reasons == reason].sum()),
        }
        for reason in sorted(set(reasons))
    }


def audit_set(name, intervals, table, shots):
    details = []
    totals = {str(n): {state: {} for state in ("absent", "present")} for n in (1, 2)}
    coverage = {
        "requested_shots": len(shots),
        "labelled_shots": int(table.shot.nunique()),
        "mirnov_feature_shots": sum(
            (OUT / "magfeatures" / f"{int(s)}.npz").is_file() for s in shots.shot
        ),
        "frequency_record_shots": sum(
            (OUT / "signals_freq" / f"{int(s)}.npz").is_file() for s in shots.shot
        ),
        "window_seconds": 0.0,
        "absent_seconds": 0.0,
        "present_seconds": 0.0,
        "uncertain_seconds": 0.0,
        "not_observable_seconds": 0.0,
        "screened_seconds_by_n": {},
        "unscreened_nonquiet_seconds_by_n": {},
    }
    for n in (1, 2):
        coverage["screened_seconds_by_n"][str(n)] = 0.0
        coverage["unscreened_nonquiet_seconds_by_n"][str(n)] = 0.0
    for row in shots.itertuples(index=False):
        shot = int(row.shot)
        path = OUT / "signals" / f"{shot}.npz"
        if not path.is_file():
            continue
        with np.load(path) as z:
            rt = z["t_ms"]
            t, n1, dt = rule.uniform(rt, z["n1rms"])
            n2 = rule.uniform(rt, z["n2rms"])[1]
        window = (t >= row.window_start_ms) & (t <= row.window_end_ms)
        rows = table[table.shot.eq(shot)]
        state = np.zeros(t.shape, int)
        # Extension spans can overlap across n; category priority defines time.
        for category in (1, 2, 3):
            for r in rows[
                rows.category.eq(category) & (rows.t_end > rows.t_start)
            ].itertuples():
                state[(t >= r.t_start) & (t <= r.t_end)] = category
        busy = []
        for key, category in (
            ("not_observable", 3),
            ("uncertain", 2),
            ("present", 1),
            ("absent", 0),
        ):
            spans = rule._union(
                [
                    (float(r.t_start), float(r.t_end))
                    for r in rows[rows.category.eq(category)].itertuples()
                    if r.t_end > r.t_start
                ]
            )
            clean = rule._minus(spans, busy)
            coverage[f"{key}_seconds"] += sum(b - a for a, b in clean) / 1000
            busy = rule._union(busy + spans)
        coverage["window_seconds"] += (
            float(row.window_end_ms - row.window_start_ms) / 1000
        )
        coherent, weak, _, seed, _, screened = line_evidence(
            shot, t, OUT / "signals_freq", OUT / "magfeatures"
        )
        frequency = {1: np.full(t.shape, np.nan), 2: np.full(t.shape, np.nan)}
        fp = OUT / "signals_freq" / f"{shot}.npz"
        if fp.is_file():
            with np.load(fp) as z:
                for n in (1, 2):
                    frequency[n] = scoring.align_scores(z["t_ms"], z[f"n{n}freq"], t)
        features = {}
        mp = OUT / "magfeatures" / f"{shot}.npz"
        if mp.is_file():
            with np.load(mp) as z:
                centres, values, names = (
                    z["centres_ms"],
                    z["features"],
                    list(z["names"]),
                )
            idx = np.searchsorted(centres + 5, t)
            inside = (idx < len(centres)) & (t >= centres[0] - 5)
            for feature in ("fit1", "fit2", "line_prominence_db", "line_khz"):
                features[feature] = np.full(t.shape, np.nan)
                features[feature][inside] = values[idx[inside], names.index(feature)]
        for n, y, mr in ((1, n1, rule.N1_RULE), (2, n2, rule.N2_RULE)):
            valid = np.isfinite(y) & window
            coverage["screened_seconds_by_n"][str(n)] += float(
                (valid & screened[n]).sum() * dt / 1000
            )
            coverage["unscreened_nonquiet_seconds_by_n"][str(n)] += float(
                (valid & ~screened[n] & (y > mr.weak_g)).sum() * dt / 1000
            )
            masks = {
                "rms_above_seed": y > mr.onset_g,
                "median5ms_above_seed": rule.smoothed(y, dt, 5) > mr.onset_g,
                "rms_above_weak": y > mr.weak_g,
                "frequency_in_range_legacy": (frequency[n] >= 1) & (frequency[n] <= 30),
                "frequency_coherent50ms": rule.coherent_frequency(frequency[n], dt),
                "mirnov_phase_fit": features.get(f"fit{n}", np.zeros(t.shape)) >= 0.9,
                "mirnov_prominence": features.get(
                    "line_prominence_db", np.zeros(t.shape)
                )
                >= 10,
                "span_coherent_support": coherent[n],
                "seed_coherent_support": seed[n],
                "weak_mirnov_support": weak[n],
                "screening_available": screened[n],
            }
            own = np.zeros(t.shape, bool)
            for item in intervals[
                intervals.shot.eq(shot) & intervals.n.eq(n)
            ].itertuples():
                own |= (t >= item.t_start - 1e-6) & (t <= item.t_end + 1e-6)
            for category, title in ((0, "absent"), (1, "present")):
                chosen = valid & (state == category)
                if category == 1:
                    chosen &= own
                for criterion, mask in masks.items():
                    measured = chosen.copy()
                    if criterion.startswith("frequency_"):
                        measured &= np.isfinite(frequency[n])
                    elif criterion == "mirnov_phase_fit":
                        measured &= np.isfinite(
                            features.get(f"fit{n}", np.full(t.shape, np.nan))
                        )
                    elif criterion in ("mirnov_prominence", "weak_mirnov_support"):
                        measured &= np.isfinite(
                            features.get("line_prominence_db", np.full(t.shape, np.nan))
                        )
                    counts = totals[str(n)][title].setdefault(
                        criterion, {"total_ms": 0.0, "pass_ms": 0.0}
                    )
                    counts["total_ms"] += float(measured.sum() * dt)
                    counts["pass_ms"] += float((measured & mask).sum() * dt)
            for item in intervals[
                intervals.shot.eq(shot) & intervals.n.eq(n)
            ].itertuples():
                inside = (t >= item.t_start - 1e-6) & (t <= item.t_end + 1e-6)
                raw = longest(t, inside & (y > mr.onset_g))
                median = longest(t, inside & (rule.smoothed(y, dt, 5) > mr.onset_g))
                combined = (
                    inside
                    & (y > mr.onset_g)
                    & (rule.smoothed(y, dt, 5) > mr.onset_g)
                    & seed[n]
                )
                if n == 2:
                    combined &= y > mr.harmonic_ratio * n1
                supported = longest(t, combined)
                details.append(
                    {
                        "shot": shot,
                        "n": n,
                        "t_start": item.t_start,
                        "t_end": item.t_end,
                        "raw_longest_seed_ms": raw,
                        "median5ms_longest_seed_ms": median,
                        "supported_longest_seed_ms": supported,
                        "span_support_fraction": float(coherent[n][inside].mean()),
                        "raw_seed_failure": raw < 50 - 1e-6,
                        "median_seed_failure": median < 50 - 1e-6,
                        "supported_seed_failure": supported < 50 - 1e-6,
                    }
                )
    for modes in totals.values():
        for state in modes.values():
            for value in state.values():
                value["pass_rate"] = (
                    value["pass_ms"] / value["total_ms"] if value["total_ms"] else None
                )
    unknown = ~intervals.locked_known.astype(bool)
    abrupt = intervals.abrupt_collapse_ms.notna()
    audit = {
        "after_seed_audit": {
            "n_intervals_audited": len(details),
            "raw_failure_count": sum(r["raw_seed_failure"] for r in details),
            "median_failure_count": sum(r["median_seed_failure"] for r in details),
            "supported_failure_count": sum(
                r["supported_seed_failure"] for r in details
            ),
            "details": details,
        },
        "locking_audit": {
            "n_confirmed_locked": int(intervals.locked.sum()),
            "n_locked_candidates": int(intervals.locked_candidate.sum()),
            "n_lock_confirmation_unknown": int(unknown.sum()),
            "n_abrupt_collapses": int(abrupt.sum()),
            "n_abrupt_incorrect_decay": int(
                (abrupt & intervals.ended.eq("decay")).sum()
            ),
            "intervals_by_end": intervals.ended.value_counts().to_dict(),
            "uncertain_rows_by_reason": uncertain_reasons(table),
        },
        "screening_coverage": coverage,
    }
    return audit, totals


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--out", type=Path, default=OUT / "labels/audit_fix4_current.json"
    )
    parser.add_argument(
        "--criterion-out", type=Path, default=OUT / "labels/criterion_support_fix4.json"
    )
    parser.add_argument(
        "--sets",
        nargs="+",
        choices=("cohort", "population"),
        default=["cohort", "population"],
    )
    parser.add_argument("--fetch-log-dir", type=Path)
    args = parser.parse_args(argv)
    cohort = pd.read_csv(REPO / "data/events/catalog/cohort.csv")
    blind = set(cohort.loc[cohort.split.eq("test"), "shot"])
    audit = {
        "made_by": "scripts/labeler/tm_audit_rule.py",
        "made_at": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
        "policy": __doc__,
        "excluded_blind_shots": len(blind),
        "source_sha256": {
            str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                REPO / "src/labeler/tearing/rule.py",
                REPO / "scripts/labeler/tm_label.py",
            )
        },
        "label_sha256": {},
    }
    criteria = {
        "made_by": audit["made_by"],
        "git_sha": audit["git_sha"],
        "policy": __doc__,
        "source_sha256": audit["source_sha256"],
    }
    if args.fetch_log_dir is not None:
        events = []
        for path in sorted(args.fetch_log_dir.glob("lock-*.log")):
            for line in path.read_text().splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "shot" in event and event["shot"] not in blind:
                    events.append(event)
        requested = pd.read_csv(args.fetch_log_dir / "lock-fetch-population.csv")
        requested = requested[~requested.shot.isin(blind)]
        success = {e["shot"] for e in events if e.get("status") == "fetched"}
        failure = {
            e["shot"] for e in events if e.get("status") in ("failed", "missing")
        }
        unavailable = [
            int(s)
            for s in requested.shot
            if not (OUT / "signals_lock" / f"{int(s)}.npz").is_file()
        ]
        final_modes = pd.read_csv(OUT / "labels/tm_intervals_full_population.csv")
        final_modes = final_modes[~final_modes.shot.isin(blind)]
        n1_shots = set(final_modes.loc[final_modes.n.eq(1), "shot"].astype(int))
        n1_missing = {
            s for s in n1_shots if not (OUT / "signals_lock" / f"{s}.npz").is_file()
        }
        audit["fetch_status"] = {
            "source": str(OUT / "signals_lock/source.json"),
            "log_dir": str(args.fetch_log_dir),
            "requested_nonblind_mode_candidate_shots": len(requested),
            "successful_unique_shots_in_logs": len(success),
            "failed_unique_shots_in_logs": len(failure),
            "available_requested_lock_records": len(requested) - len(unavailable),
            "unavailable_requested_lock_shots": unavailable,
            "final_population_n1_mode_shots": len(n1_shots),
            "final_population_n1_missing_lock_record_shots": sorted(n1_missing),
            "final_population_n1_unattempted_shots": sorted(
                n1_missing - success - failure
            ),
            "auth_stop_marker_present": (OUT / "fetch_auth_stop.json").exists(),
            "auth_error_events": sum(
                e.get("stopped") == "authentication" for e in events
            ),
            "failure_events": [
                e for e in events if e.get("status") in ("failed", "missing")
            ],
        }
    for name in args.sets:
        shots = pd.read_csv(REPO / f"data/events/catalog/{name}.csv")
        shots = shots[~shots.shot.isin(blind)]
        ip = OUT / f"labels/tm_intervals_full_{name}.csv"
        tp = (
            REPO
            / "data/events/neoclassical_tearing_mode/extend_tm_interval/tm_interval.csv"
            if name == "cohort"
            else OUT / "labels/tm_interval_population.csv"
        )
        intervals, table = pd.read_csv(ip), pd.read_csv(tp)
        intervals, table = (
            intervals[~intervals.shot.isin(blind)],
            table[~table.shot.isin(blind)],
        )
        audit["label_sha256"][name] = {
            "intervals": hashlib.sha256(ip.read_bytes()).hexdigest(),
            "table": hashlib.sha256(tp.read_bytes()).hexdigest(),
        }
        audit[name], criteria[name] = audit_set(name, intervals, table, shots)
    criteria["label_sha256"] = audit["label_sha256"]
    args.out.write_text(json.dumps(audit, indent=2, default=lambda v: v.item()) + "\n")
    args.criterion_out.write_text(json.dumps(criteria, indent=2) + "\n")
    print(
        json.dumps(
            {
                name: {k: v for k, v in audit[name].items() if k != "after_seed_audit"}
                for name in args.sets
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
