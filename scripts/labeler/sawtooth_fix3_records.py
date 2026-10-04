"""Audit regenerated labels, geometry coverage and deterministic label hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sawtooth_physics import (
    FS,
    OUTPUT,
    PRIOR_INPUTS,
    READER_POLICY,
    REPO,
    WORK,
    save_json,
)

from labeler.sawtooth.physics import DEFAULT_RULE
from labeler.sawtooth.preprocessing import STATES

OBSERVED_STATES = tuple(state for state in STATES if state != "unassessed")


def distribution(values, *, distance=False):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {
        "n": len(values),
        "quantile_levels": [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1],
        "quantiles": (
            np.quantile(values, [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]).tolist()
            if len(values)
            else None
        ),
        "mean": float(values.mean()) if len(values) else None,
        "within_0p15_m_fraction": (
            float(np.mean(np.abs(values) <= 0.15)) if len(values) else None
        )
        if distance
        else None,
    }


def state_totals(records):
    seconds = {
        state: sum(r.get("state_seconds", {}).get(state, 0) for r in records)
        for state in STATES
    }
    observable = sum(seconds[s] for s in OBSERVED_STATES)
    return {
        "state_seconds": seconds,
        "observable_seconds": observable,
        "fractions_of_observable": {
            state: seconds[state] / observable if observable else None
            for state in OBSERVED_STATES
        },
        "assessed_fraction_of_observable": (
            (seconds["present"] + seconds["absent"]) / observable
            if observable
            else None
        ),
        "q_prior_only_fraction_of_absent_class": (
            seconds["absent_q_prior"] / (seconds["absent"] + seconds["absent_q_prior"])
            if seconds["absent"] + seconds["absent_q_prior"]
            else None
        ),
        "processed_count": sum("error" not in r for r in records),
        "excluded_records": sum("error" in r for r in records),
        "exclusion_reason_counts": dict(
            Counter(r.get("error_kind", "unspecified") for r in records if "error" in r)
        ),
        "exclusion_messages": dict(
            Counter(
                str(r["error"]).split(" (")[0][:80] for r in records if "error" in r
            )
        ),
        "diagnostic_crashes": sum(len(r["crashes"]) for r in records),
        "candidate_state_counts": dict(
            Counter(p["attrs"]["state"] for r in records for p in r["crashes"])
        ),
    }


GUARD_KEYS = (
    "core_observable_samples",
    "removed_by_reference_density_guard",
    "removed_by_density_guard",
    "removed_by_ece_validity",
    "observable_samples_after_guards",
)


def guard_totals(records):
    """Observable time the cutoff guards remove, in seconds and as fractions."""
    used = [r for r in records if "error" not in r]
    accounts = [r["absence_diagnostics"].get("guard_accounting", {}) for r in used]
    totals = {key: sum(a.get(key, 0) for a in accounts) for key in GUARD_KEYS}
    core = totals["core_observable_samples"]
    fractions = [
        a["removed_by_ece_validity"] / a["core_observable_samples"]
        for a in accounts
        if a.get("core_observable_samples")
    ]
    return {
        "unit": "seconds of core-observable time",
        "seconds": {key: value / FS for key, value in totals.items()},
        "fraction_of_core_observable": {
            key: (value / core if core else None) for key, value in totals.items()
        },
        "shots": len(fractions),
        "ece_validity_shot_fraction_quantiles_0_25_50_75_100": (
            np.quantile(fractions, [0, 0.25, 0.5, 0.75, 1]).tolist()
            if fractions
            else None
        ),
        "shots_with_ece_validity_above_20_percent": int(
            np.sum(np.asarray(fractions) > 0.2)
        ),
        "shots_with_ece_validity_above_90_percent": int(
            np.sum(np.asarray(fractions) > 0.9)
        ),
    }


def absent_composition(records):
    """What the absent class and the q-prior-only time consist of, in seconds."""
    used = [r for r in records if "error" not in r]
    reasons = [r["absence_diagnostics"]["reason_samples"] for r in used]

    def seconds(key):
        return sum(x.get(key, 0) for x in reasons) / FS

    states = [r["state_seconds"] for r in used]
    return {
        "tested_absence_s": sum(x["absent"] for x in states),
        "tested_absence_with_high_q_s": seconds("tested_absence_with_high_q"),
        "tested_absence_without_high_q_s": seconds("tested_absence_without_high_q"),
        "q_prior_only_s": sum(x["absent_q_prior"] for x in states),
        "sustained_high_q_s": seconds("sustained_high_q"),
        "shots_with_tested_absence": sum(x["absent"] > 0 for x in states),
        "shots_with_q_prior_only": sum(x["absent_q_prior"] > 0 for x in states),
        "shots": len(used),
    }


def records(args):
    from dataclasses import asdict

    cohort = pd.read_csv(REPO / "data/events/catalog/cohort.csv")
    all_records = [
        json.loads(p.read_text()) for p in sorted((args.work / "shots").glob("*.json"))
    ]
    by_shot = {r["shot"]: r for r in all_records}
    assert set(map(int, cohort.shot)) <= set(by_shot)
    for record in all_records:
        assert record["rule"] == asdict(DEFAULT_RULE), record["shot"]
        assert record["reader_policy"] == READER_POLICY, record["shot"]
        if "error" in record:
            continue
        info = record["core_geometry"]
        assert max(info["core_channels"]) < 40
        assert all(channel < 40 for channel in info["outer_channels"])
        assert (
            record["absence_diagnostics"]["reason_samples"]["tested_absence"]
            is not None
        )
        for point in record["crashes"]:
            attrs = point["attrs"]
            assert attrs["drop_stop"] <= 40 and attrs["rise_stop"] <= 40
            assert attrs["inversion_channel"] <= 39
        for phase in record["absence_diagnostics"]["core_relaxation_test"][
            "phase_spans"
        ]:
            assert phase["minimum_gap_ms"] >= 10 - 1e-6
            assert phase["null_p_value"] <= DEFAULT_RULE.periodicity_null_alpha
    splits = {}
    for split, rows in cohort.groupby("split"):
        selected = [by_shot[int(shot)] for shot in rows.shot]
        splits[split] = {
            "shots": rows.shot.astype(int).tolist(),
            "shots_requested": len(rows),
            **state_totals(selected),
            "absent_composition": absent_composition(selected),
            "guards": guard_totals(selected),
        }
    save_json(
        OUTPUT / "data_summary.json",
        {
            "source_records": str(args.work / "shots"),
            "splits": splits,
            "population": {
                **state_totals(all_records),
                "absent_composition": absent_composition(all_records),
                "guards": guard_totals(all_records),
            },
            "claim": "physics-rule labels validated only by reported checks",
            "prior_unverified_candidates_preserved": sum(
                r.get("prior_unverified_candidates_preserved", 0) for r in all_records
            ),
        },
    )
    comparisons, inversion_rho, outer_minus_inversion_rho = [], [], []
    axis_offsets = []  # signed inversion R minus axis R at paired points, metres
    train = set(cohort.loc[cohort.split == "train", "shot"])
    shot_table = []
    for record in all_records:
        geometry = record.get("radius_geometry", {})
        deltas = [
            p["attrs"]["q1_radius_difference_m"]
            for p in record["crashes"]
            if p["attrs"].get("q1_radius_difference_m") is not None
        ]
        comparisons.extend(deltas)
        axis_offsets.extend(
            p["attrs"]["inversion_R_m"] - p["attrs"]["axis_R_m"]
            for p in record["crashes"]
            if p["attrs"].get("q1_radius_difference_m") is not None
            and p["attrs"].get("axis_R_m") is not None
        )
        if record["shot"] in train and "error" not in record:
            outer_rho = record["core_geometry"].get("outer_nominal_rho_median")
            for point in record["crashes"]:
                attrs = point["attrs"]
                rho = attrs.get("inversion_nominal_rho")
                if attrs["state"] != "present" or rho is None:
                    continue
                inversion_rho.append(rho)
                if outer_rho is not None:
                    outer_minus_inversion_rho.append(outer_rho - rho)
        shot_table.append(
            {
                "shot": record["shot"],
                "efit_available": bool(geometry.get("field_product_source")),
                "qmin_available": record.get("qmin_available", False),
                "q1_checked": geometry.get("q1_checked", False),
                "q1_status": geometry.get("q1_status", geometry.get("status")),
                "paired_crashes": len(deltas),
                "delta_R_m": distribution(deltas, distance=True),
                "axis_channel": record.get("core_geometry", {}).get("central_channel"),
                "nominal_outer_rho": record.get("core_geometry", {}).get(
                    "outer_nominal_rho_median"
                ),
                "sensor_support_exclusion": record.get("error"),
            }
        )
    full = {
        "claim": (
            "nominal vacuum resonance versus magnetics-only EFIT01; no radial truth"
        ),
        "difference": "inversion major R minus same-side q=1 major R, metres",
        "distribution_all_diagnostic_points": distribution(comparisons, distance=True),
        "paired_point_side": {
            "definition": (
                "same-side q=1 comparison; low-field side (LFS) means the "
                "inversion R is at or beyond the EFIT magnetic axis R"
            ),
            "paired_points_with_axis_R": len(axis_offsets),
            "low_field_side_points": int(np.sum(np.asarray(axis_offsets) >= 0)),
            "high_field_side_points": int(np.sum(np.asarray(axis_offsets) < 0)),
            "low_field_side_fraction": (
                float(np.mean(np.asarray(axis_offsets) >= 0)) if axis_offsets else None
            ),
            "absolute_inversion_R_minus_axis_R_m": distribution(
                np.abs(axis_offsets)
            ),
        },
        "train_present_inversion_nominal_rho": distribution(inversion_rho),
        "train_present_outer_minus_inversion_nominal_rho": distribution(
            outer_minus_inversion_rho
        ),
        "train_present_outer_beyond_inversion_fraction": (
            float(np.mean(np.asarray(outer_minus_inversion_rho) > 0))
            if outer_minus_inversion_rho
            else None
        ),
        "efit_shots": sum(r["efit_available"] for r in shot_table),
        "q1_checked_shots": sum(r["q1_checked"] for r in shot_table),
        "paired_shots": sum(r["paired_crashes"] > 0 for r in shot_table),
        "by_shot": shot_table,
    }
    save_json(args.work / "q1_radius_audit.json", full)
    small = {key: value for key, value in full.items() if key != "by_shot"}
    small["by_shot_source"] = str(args.work / "q1_radius_audit.json")
    small["cohort_by_shot"] = [r for r in shot_table if r["shot"] in set(cohort.shot)]
    save_json(OUTPUT / "q1_radius_audit.json", small)
    # Bias justification uses the previous rule's TRAIN candidates only. They are
    # read from the snapshot beside the labels, never from the superseded round.
    # The reviewed/test shots never select the numerical thresholds here.
    snapshot_path = args.work / "labels" / PRIOR_INPUTS
    snapshot = json.loads(snapshot_path.read_text())
    prior_q = [
        q
        for shot, values in snapshot["train_qmin_conflict_candidates"].items()
        if int(shot) in train
        for q in values
    ]
    shot186532 = by_shot.get(186532, {})
    save_json(
        OUTPUT / "qmin_bias_audit.json",
        {
            "source": f"labels/{PRIOR_INPUTS}",
            "source_sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
            "selection": "fixed TRAIN only; sole prior uncertainty qmin_conflict",
            "prior_train_candidate_qmin": distribution(prior_q),
            "efit01_conflict_threshold": 1 + DEFAULT_RULE.qmin_margin,
            "absence_threshold": DEFAULT_RULE.qmin_absence,
            "sustain_ms": DEFAULT_RULE.qmin_sustain_ms,
            "justification": (
                "Magnetics-only EFIT01 does not constrain central current like MSE. "
                "The earlier 1.05 conflict systematically flags ECE-supported trains "
                "near 1.1-1.3, and 5% of the prior TRAIN candidates sit above "
                f"{np.quantile(prior_q, 0.95):.2f}. "
                "A prescribed 1.4 conflict threshold leaves this bias band unresolved "
                "by the equilibrium. High q is no longer sufficient for absence: "
                "sustained q_min >= 1.5 supplies only the q-prior state "
                "absent_q_prior, which is uncertain on export and supplies no "
                "benchmark negative. Thresholds are not fitted to expert or test "
                "results."
            ),
            "shot_186532": {
                "state_seconds": shot186532.get("state_seconds"),
                "reasons": shot186532.get("absence_diagnostics", {}).get(
                    "reason_samples"
                ),
            },
        },
    )


def manifest(args):
    labels = args.work / "labels"
    # Population shards are the complete nonduplicated export. Retain the
    # cohort bundle separately, rather than duplicate500shots in this path.
    if any(labels.glob("population-*.csv")):
        for path in labels.glob("cohort-*.csv"):
            destination = args.work / "cohort_labels" / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            path.replace(destination)
    lines, files = [], []
    # The CSV shards and the frozen prior-round inputs the rule reads are hashed
    # together, so deleting or regenerating an earlier round cannot change them.
    hashed = sorted(labels.glob("*.csv")) + sorted(
        (labels / PRIOR_INPUTS).parent.glob("*.json")
    )
    for path in hashed:
        sha = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                sha.update(chunk)
        digest = sha.hexdigest()
        name = str(path.relative_to(labels))
        lines.append(f"{digest}  {name}\n")
        files.append({"file": name, "sha256": digest, "bytes": path.stat().st_size})
    sums = labels / "SHA256SUMS"
    sums.write_text("".join(lines))
    record = {
        "labels_path": str(labels),
        "manifest": str(sums),
        "manifest_sha256": hashlib.sha256(sums.read_bytes()).hexdigest(),
        "csv_shards": sum(f["file"].endswith(".csv") for f in files),
        "prior_input_files": [f["file"] for f in files if f["file"].endswith(".json")],
        "freeze_json_sha256": hashlib.sha256(
            (args.work / "freeze.json").read_bytes()
        ).hexdigest(),
        "files": files,
    }
    save_json(OUTPUT / "label_manifest.json", record)
    print(json.dumps(record))


def verification(args):
    names = ("covering_tests", "ruff", "format", "diff_check")
    logs = {name: args.work / f"{name}.log" for name in names}
    texts = {name: path.read_text() for name, path in logs.items()}
    match = re.search(r"(\d+) passed", texts["covering_tests"])
    assert match and "failed" not in texts["covering_tests"].lower()
    assert "All checks passed!" in texts["ruff"]
    assert "already formatted" in texts["format"]
    assert not texts["diff_check"].strip()
    code_files = [
        *sorted((REPO / "src/labeler/sawtooth").glob("*.py")),
        *sorted((REPO / "scripts/labeler").glob("sawtooth_*.py")),
        *sorted((REPO / "tests/labeler").glob("test_sawtooth*.py")),
    ]
    result = {
        "source": str(Path(__file__).relative_to(REPO)),
        "covering_tests_passed": int(match[1]),
        "covering_test_files": [
            "test_sawtooth_physics.py",
            "test_sawtooth_fix.py",
            "test_sawtooth_fix3.py",
            "test_sawtooth_phase_null.py",
            "test_sawtooth_preprocessing.py",
            "test_sawtooth_geometry.py",
            "test_sawtooth_shot_geometry.py",
            "test_sawtooth_benchmark.py",
            "test_sawtooth_classification.py",
            "test_sawtooth_masked_metrics.py",
        ],
        "ruff": "all changed Python files passed",
        "format": "all new Python files passed",
        "diff_check": "passed",
        "logs": {name: str(path) for name, path in logs.items()},
        "output_tails": {name: text.splitlines()[-8:] for name, text in texts.items()},
        "code_sha256": {
            str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in code_files
        },
    }
    save_json(OUTPUT / "verification.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "code_sha256"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("records", "manifest", "verification"))
    parser.add_argument("--work", type=Path, default=WORK)
    args = parser.parse_args()
    {"records": records, "manifest": manifest, "verification": verification}[
        args.stage
    ](args)


if __name__ == "__main__":
    main()
