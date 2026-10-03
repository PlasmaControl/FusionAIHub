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
from sawtooth_physics import OUTPUT, READER_POLICY, REPO, WORK, save_json

from labeler.sawtooth.physics import DEFAULT_RULE


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
        for state in ("present", "absent", "uncertain", "unassessed")
    }
    observable = sum(seconds[s] for s in ("present", "absent", "uncertain"))
    return {
        "state_seconds": seconds,
        "observable_seconds": observable,
        "fractions_of_observable": {
            state: seconds[state] / observable if observable else None
            for state in ("present", "absent", "uncertain")
        },
        "assessed_fraction_of_observable": (
            (seconds["present"] + seconds["absent"]) / observable
            if observable
            else None
        ),
        "processed_count": sum("error" not in r for r in records),
        "excluded_records": sum("error" in r for r in records),
        "exclusion_reason_counts": dict(
            Counter(r.get("error_kind", "unspecified") for r in records if "error" in r)
        ),
        "diagnostic_crashes": sum(len(r["crashes"]) for r in records),
        "candidate_state_counts": dict(
            Counter(p["attrs"]["state"] for r in records for p in r["crashes"])
        ),
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
        }
    save_json(
        OUTPUT / "data_summary.json",
        {
            "source_records": str(args.work / "shots"),
            "splits": splits,
            "population": state_totals(all_records),
            "claim": "physics-rule labels validated only by reported checks",
            "prior_unverified_candidates_preserved": sum(
                r.get("prior_unverified_candidates_preserved", 0) for r in all_records
            ),
        },
    )
    comparisons, inversion_rho, outer_minus_inversion_rho = [], [], []
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
    # Bias justification uses the previous rule's TRAIN candidates only. The
    # reviewed/test shots never select the numerical thresholds in this audit.
    prior_q = []
    for shot in train:
        path = args.work.parent / "fix2/shots" / f"{shot}.json"
        if not path.exists():
            continue
        old = json.loads(path.read_text())
        prior_q.extend(
            p["attrs"]["qmin"]
            for p in old["crashes"]
            if p["attrs"].get("uncertainty_reasons") == ["qmin_conflict"]
            and p["attrs"].get("qmin") is not None
        )
    shot186532 = by_shot.get(186532, {})
    save_json(
        OUTPUT / "qmin_bias_audit.json",
        {
            "source": str(args.work.parent / "fix2/shots"),
            "selection": "fixed TRAIN only; sole prior uncertainty qmin_conflict",
            "prior_train_candidate_qmin": distribution(prior_q),
            "efit01_conflict_threshold": 1 + DEFAULT_RULE.qmin_margin,
            "absence_threshold": DEFAULT_RULE.qmin_absence,
            "sustain_ms": DEFAULT_RULE.qmin_sustain_ms,
            "justification": (
                "Magnetics-only EFIT01 does not constrain central current like MSE. "
                "Prior 1.05 conflict systematically flags ECE-supported trains near "
                "1.1-1.3. A prescribed 1.4 conflict and sustained1.5 absence guard "
                "leave this bias band unresolved by equilibrium, while requiring "
                "clear high-q evidence; thresholds are not fitted to "
                "expert/test results."
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
    for path in sorted(labels.glob("*.csv")):
        sha = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                sha.update(chunk)
        digest = sha.hexdigest()
        lines.append(f"{digest}  {path.name}\n")
        files.append(
            {"file": path.name, "sha256": digest, "bytes": path.stat().st_size}
        )
    sums = labels / "SHA256SUMS"
    sums.write_text("".join(lines))
    record = {
        "labels_path": str(labels),
        "manifest": str(sums),
        "manifest_sha256": hashlib.sha256(sums.read_bytes()).hexdigest(),
        "csv_shards": len(files),
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
