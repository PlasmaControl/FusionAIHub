"""Audit frozen rules, four-state coverage, split isolation and GPU stopping."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from itertools import pairwise

import numpy as np
import pandas as pd
from sawtooth_physics import (
    OUTPUT,
    REPO,
    REVIEW,
    WORK,
    export_rows,
    records_at,
    save_json,
)

from labeler.config import Paths


def merged_support(states, allowed):
    support = []
    for span in states:
        if span["state"] not in allowed:
            continue
        if support and support[-1][1] == span["start_s"]:
            support[-1][1] = span["end_s"]
        else:
            support.append([span["start_s"], span["end_s"]])
    return support


def population_audit(frozen):
    """Check every durable population record, including failed input reads."""
    shots = sorted(
        int(path.name.split("_")[0])
        for path in Paths.from_env().corpus.glob("*_processed.h5")
    )
    counters = Counter()
    policies, q_sources = Counter(), Counter()
    export_counts = Counter()
    cohort = set(pd.read_csv(REPO / "data/events/catalog/cohort.csv").shot)
    cohort |= set(pd.read_csv(REVIEW).shot)
    for shot in shots:
        record_path = WORK / "shots" / f"{shot}.json"
        assert record_path.exists(), f"missing population record {shot}"
        row = json.loads(record_path.read_text())
        counters["records"] += 1
        assert row["rule"] == frozen["rule"], f"rule mismatch {shot}"
        if "error" in row:
            counters["failed_reads_excluded"] += 1
            assert not row["crashes"] and not row["intervals"]
            continue
        counters["usable_records"] += 1
        for _, _, crowd, _, attrs in export_rows(row):
            kind = "spans" if crowd else "points"
            export_counts[("population", shot, kind, attrs["state"])] += 1
            if shot in cohort:
                export_counts[("cohort", shot, kind, attrs["state"])] += 1
        policies[row.get("assessment_policy", "missing")] += 1
        q_sources[row.get("q_source", "missing")] += 1
        assert row.get("assessment_policy") == (
            "finite_core_and_four_profile_channels_uncertain_filter_support"
        ), f"missing mask repair {shot}"
        states = row["states"]
        assert states and states[0]["start_s"] == row["window_s"][0]
        assert states[-1]["end_s"] > row["window_s"][1]
        assert all(span["start_s"] < span["end_s"] for span in states)
        assert all(
            left["end_s"] == right["start_s"] for left, right in pairwise(states)
        ), f"nonpartitioned states {shot}"
        assert (
            merged_support(states, {"present", "absent", "uncertain"})
            == (row["observable_spans"])
        ), f"observable support mismatch {shot}"
        assert (
            merged_support(states, {"present", "absent"}) == (row["assessed_spans"])
        ), f"assessment support mismatch {shot}"
        if row.get("mask_repair_native_error"):
            counters["repair_native_clock_errors_unassessed"] += 1
            assert not row["observable_spans"] and not row["assessed_spans"]
        for crash in row["crashes"]:
            attrs = crash["attrs"]
            counters[f"{attrs['state']}_crashes"] += 1
            assert any(
                lo <= crash["time_s"] < hi for lo, hi in row["observable_spans"]
            ), f"unobservable crash {shot}"
            if attrs["state"] == "uncertain":
                assert not any(
                    lo <= crash["time_s"] < hi for lo, hi in row["assessed_spans"]
                ), f"uncertain crash used as absence {shot}"
            if attrs.get("qmin") is not None and attrs["qmin"] > (
                1 + frozen["rule"]["qmin_margin"]
            ):
                counters["q_conflicted_crashes"] += 1
                assert attrs["state"] == "uncertain", f"q veto regression {shot}"
        for span in row["intervals"] + row["uncertain_intervals"]:
            assert any(
                lo <= span["start_s"] and span["end_s"] <= hi
                for lo, hi in row["observable_spans"]
            ), f"span crosses unobservability {shot}"
            attrs = span["attrs"]
            if attrs.get("period_ms") is not None:
                assert (
                    attrs["inversion_channel_spread"]
                    <= (frozen["rule"]["inversion_spread_channels"])
                )
                assert (
                    frozen["rule"]["minimum_period_ms"]
                    <= attrs["period_ms"]
                    <= (frozen["rule"]["maximum_period_ms"])
                )
        for state in ("present", "absent", "uncertain", "unassessed"):
            counters[f"{state}_seconds"] += row["state_seconds"][state]
    summary = {
        "requested_shots": len(shots),
        "counts": dict(counters),
        "assessment_policies": dict(policies),
        "q_sources": dict(q_sources),
        "all_rules_identical": True,
        "q_conflicted_absent": 0,
        "spans_crossing_unobservability": 0,
        "source_records": str(WORK / "shots"),
    }
    return summary, export_counts


def csv_audit(expected):
    destination = REPO / "data/events/sawtooth_oscillation/extend_saw_physics"
    counts = Counter()
    by_shot = Counter()
    files = []
    for path in sorted(destination.glob("*.csv")):
        counterpart = WORK / "labels" / path.name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert counterpart.exists()
        assert hashlib.sha256(counterpart.read_bytes()).hexdigest() == digest
        with path.open() as handle:
            for row in csv.DictReader(handle):
                attrs = json.loads(row["attrs"])
                state = attrs["state"]
                assert (
                    int(row["category"])
                    == {
                        "absent": 0,
                        "present": 1,
                        "uncertain": 2,
                        "unassessed": 3,
                    }[state]
                )
                kind = "spans" if row["crowd"] == "True" else "points"
                scope = path.name.split("-")[0]
                counts[f"{scope}_{kind}_{state}"] += 1
                by_shot[(scope, int(row["shot"]), kind, state)] += 1
                if kind == "spans":
                    assert float(row["t_start"]) < float(row["t_end"])
                else:
                    assert row["t_start"] == row["t_end"]
        files.append({"name": path.name, "sha256": digest})
    assert {key[0] for key in by_shot} == {"cohort", "population"}
    assert by_shot == expected, "CSV rows or shots differ from repaired records"
    return {
        "row_counts": dict(counts),
        "shot_counts": {
            scope: len({key[1] for key in by_shot if key[0] == scope})
            for scope in ("cohort", "population")
        },
        "rows_and_shots_match_records": True,
        "copies_identical": True,
        "files": files,
    }


def main():
    frozen = json.loads((WORK / "freeze.json").read_text())
    cohort = pd.read_csv(REPO / "data/events/catalog/cohort.csv")
    experts = set(pd.read_csv(REVIEW).shot)
    train = set(cohort[cohort.split == "train"].shot)
    dev = set(frozen["development_shots"])
    assert dev <= train and not dev & experts
    records = records_at(WORK, sorted(cohort.shot))
    assert len(records) == len(cohort)
    valid = [r for r in records if "error" not in r]
    assert all(r["rule"] == frozen["rule"] for r in valid)
    spans = [
        row
        for r in valid
        for row in r["intervals"] + r["uncertain_intervals"]
        if row["attrs"].get("period_ms") is not None
    ]
    spread = [r["attrs"]["inversion_channel_spread"] for r in spans]
    periods = [r["attrs"]["period_ms"] for r in spans]
    assert all(s <= frozen["rule"]["inversion_spread_channels"] for s in spread)
    assert all(
        frozen["rule"]["minimum_period_ms"] <= p <= frozen["rule"]["maximum_period_ms"]
        for p in periods
    )
    coverage = {}
    for state in ("present", "absent", "uncertain", "unassessed"):
        coverage[state] = sum(r["state_seconds"][state] for r in valid)
    q_conflicted = [
        r
        for rec in valid
        for r in rec["crashes"]
        if r["attrs"].get("qmin") is not None
        and r["attrs"]["qmin"] > 1 + frozen["rule"]["qmin_margin"]
    ]
    assert all(r["attrs"]["state"] == "uncertain" for r in q_conflicted)
    missing_crossings = 0
    for rec in valid:
        for span in rec["intervals"] + rec["uncertain_intervals"]:
            a, b = span["start_s"], span["end_s"]
            if not any(lo <= a and b <= hi for lo, hi in rec["observable_spans"]):
                missing_crossings += 1
    assert missing_crossings == 0
    amplitude = [
        r["attrs"]["central_relative_drop"] for rec in valid for r in rec["crashes"]
    ]
    assert all(a >= frozen["rule"]["central_relative_drop"] for a in amplitude)
    models = []
    for name in ("saw-hl3", "saw-ours"):
        for fold in range(3):
            row = json.loads((OUTPUT / f"{name}_fold_{fold}.json").read_text())
            fit, select, test = (
                set(row[key])
                for key in ("training_shots", "selection_shots", "heldout_shots")
            )
            assert fit | select | test == train
            assert not fit & select and not fit & test and not select & test
            assert not (fit | select | test) & experts
            assert row["device"] == "cuda"
            assert row["stale_epochs"] >= row["early_stopping_patience"]
            assert np.allclose(row["threshold_grid"], np.arange(0.05, 1, 0.05))
            if name == "saw-hl3":
                assert row["derivative_z_grid"] == list(range(2, 11))
            models.append(
                {
                    k: row[k]
                    for k in (
                        "model",
                        "fold",
                        "best_epoch",
                        "epochs_completed",
                        "stale_epochs",
                        "device",
                        "peak_cuda_memory_bytes",
                        "selected_crash_threshold",
                        "presence_threshold",
                    )
                }
            )
    files = [path for path in OUTPUT.glob("*.json") if path.name != "audit.json"]
    size = sum(p.stat().st_size for p in files)
    tests = (WORK / "tests.log").read_text()
    count = int(re.search(r"(\d+) passed", tests).group(1))
    lint = (WORK / "lint.log").read_text()
    assert "All checks passed!" in lint
    population, expected_export = population_audit(frozen)
    result = {
        "development_shots": sorted(dev),
        "development_split": "train",
        "cohort_shots": len(records),
        "usable_cohort_shots": len(valid),
        "state_seconds": coverage,
        "state_fraction": {k: v / sum(coverage.values()) for k, v in coverage.items()},
        "train_spans": len(spans),
        "inversion_spread_quantiles": np.quantile(spread, [0, 0.5, 0.75, 1]).tolist()
        if spread
        else [],
        "period_ms_quantiles": np.quantile(periods, [0, 0.25, 0.5, 0.75, 1]).tolist()
        if periods
        else [],
        "central_amplitude_quantiles": np.quantile(
            amplitude, [0, 0.25, 0.5, 0.75, 1]
        ).tolist()
        if amplitude
        else [],
        "q_conflicted_crashes": len(q_conflicted),
        "q_conflicted_absent": 0,
        "spans_crossing_unobservability": missing_crossings,
        "population": population,
        "csv_export": csv_audit(expected_export),
        "models": models,
        "covering_tests_passed": count,
        "lint_tail": lint.strip().splitlines()[-1],
        "summary_json_bytes_before_audit": size,
        "label_csvs": {
            str(directory): [
                {"name": p.name, "bytes": p.stat().st_size}
                for p in sorted(directory.glob("*.csv"))
            ]
            for directory in (
                REPO / "data/events/sawtooth_oscillation/extend_saw_physics",
                WORK / "labels",
            )
        },
    }
    save_json(OUTPUT / "audit.json", result)
    assert size + (OUTPUT / "audit.json").stat().st_size < 2_000_000
    print(
        {k: v for k, v in result.items() if k not in ("models", "label_csvs")},
        flush=True,
    )


if __name__ == "__main__":
    main()
