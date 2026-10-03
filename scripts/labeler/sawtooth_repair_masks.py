"""Apply frozen assessment support to existing sawtooth records, without refitting.

The cohort uses cached antialiased ECE. Population passes read only native ECE
from the corpus and may run repeatedly while label generation finishes. Completed
repairs are skipped; every write is atomic and originals are retained under WORK.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import maximum_filter1d

from labeler.config import Paths
from labeler.sawtooth.physics import Rule
from labeler.sawtooth.preprocessing import mask_spans, sample_native, state_spans

REPO = Path(__file__).resolve().parents[2]
WORK = Paths.from_env().root / "round4/saw/fix"
OUTPUT = REPO / "outputs/labeler/sawtooth/fix"
POLICY = "finite_core_and_four_profile_channels_uncertain_filter_support"


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".mask-partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return list(zip(edges[::2], edges[1::2], strict=True))


def nearest_index(t, time_s):
    right = min(len(t) - 1, int(np.searchsorted(t, time_s)))
    left = max(0, right - 1)
    return left if abs(t[left] - time_s) < abs(t[right] - time_s) else right


def support_from_spans(t, spans):
    support = np.zeros(len(t), dtype=bool)
    for start, end in spans:
        support[(t >= start) & (t < end)] = True
    return support


def clipped_interval(t, run, start, end, padding):
    dt = float(np.median(np.diff(t)))
    stop = float(t[run[1]]) if run[1] < len(t) else float(t[-1] + dt)
    return (
        max(float(t[run[0]]), start - padding),
        min(stop, end + padding),
    )


def profile_support(t, y, rule):
    """Four finite profile channels throughout the Gaussian filter support."""
    dt = float(np.median(np.diff(t)))
    radius = int(np.ceil(3.5 * rule.sigma_ms / 1000 / dt))
    bad = np.isfinite(y).sum(axis=0) < 4
    return maximum_filter1d(bad.astype(np.uint8), 2 * radius + 1) == 0


def subset_sufficient(t, core, old_observable, rule):
    """A core subset is an exact lower-bound certificate for the full profile."""
    return not (old_observable & ~profile_support(t, core, rule)).any()


def repair_record(record, t, y, old_observable, rule):
    """Rebuild train support while retaining every accepted crash's amplitude."""
    before = copy.deepcopy(record)
    dt = float(np.median(np.diff(t)))
    observable = old_observable & profile_support(t, y, rule)
    support_runs = runs(observable)
    run_id = np.full(len(t), -1, dtype=np.int32)
    for index, (lo, hi) in enumerate(support_runs):
        run_id[lo:hi] = index
    padding = rule.frame_ms / 2000
    retained = []
    present, uncertain = [], []
    removed_unobservable, removed_fragments = 0, 0
    widened, split_trains = 0, 0
    points = before.get("crashes", [])
    assigned = set()
    trains = [
        (kind, index, span)
        for kind in ("intervals", "uncertain_intervals")
        for index, span in enumerate(before.get(kind, []))
        if span.get("attrs", {}).get("period_ms") is not None
    ]
    for kind, index, span in trains:
        members = [
            (i, point)
            for i, point in enumerate(points)
            if i not in assigned
            and span["start_s"] - dt / 2 <= point["time_s"] <= span["end_s"] + dt / 2
        ]
        assigned.update(i for i, _ in members)
        segments = {}
        for _, point in members:
            support = int(run_id[nearest_index(t, point["time_s"])])
            if support < 0:
                removed_unobservable += 1
                continue
            segments.setdefault(support, []).append(point)
        split_trains += int(len(segments) > 1)
        for support, segment in segments.items():
            segment.sort(key=lambda point: point["time_s"])
            original_id = span.get("attrs", {}).get(
                "original_train_id", f"{record['shot']}:{kind}:{index}"
            )
            if len(segment) < rule.minimum_train:
                removed_fragments += len(segment)
                for point in segment:
                    attrs = copy.deepcopy(point["attrs"])
                    attrs.update(
                        state="uncertain",
                        period_ms=None,
                        original_train_id=original_id,
                        support_padding_ms=rule.frame_ms / 2,
                    )
                    attrs["uncertainty_reasons"] = sorted(
                        set(
                            attrs.get("uncertainty_reasons", [])
                            + ["observability_gap_fragment", "isolated_candidate"]
                        )
                    )
                    start, end = clipped_interval(
                        t,
                        support_runs[support],
                        point["time_s"],
                        point["time_s"],
                        padding,
                    )
                    uncertain.append({"start_s": start, "end_s": end, "attrs": attrs})
                continue
            state = "uncertain" if kind == "uncertain_intervals" else "present"
            attrs = copy.deepcopy(span["attrs"])
            attrs.update(
                state=state, crashes=len(segment), original_train_id=original_id
            )
            attrs["period_ms"] = float(
                np.median(np.diff([point["time_s"] for point in segment])) * 1000
            )
            attrs["inversion_channel_spread"] = float(
                np.ptp([point["attrs"]["inversion_channel"] for point in segment])
            )
            start, end = segment[0]["time_s"], segment[-1]["time_s"]
            if state == "uncertain":
                start, end = clipped_interval(
                    t, support_runs[support], start, end, padding
                )
                attrs["support_padding_ms"] = rule.frame_ms / 2
                widened += int(start != span["start_s"] or end != span["end_s"])
            target = uncertain if state == "uncertain" else present
            target.append({"start_s": start, "end_s": end, "attrs": attrs})
            for point in segment:
                point = copy.deepcopy(point)
                point["attrs"].update(
                    state=state,
                    period_ms=attrs["period_ms"],
                    original_train_id=original_id,
                )
                retained.append(point)
    # Isolated candidates keep uncertainty only over their actual observable support.
    for span in before.get("uncertain_intervals", []):
        if span.get("attrs", {}).get("period_ms") is not None:
            continue
        for lo, hi in support_runs:
            start = max(span["start_s"], float(t[lo]))
            stop = float(t[hi]) if hi < len(t) else float(t[-1] + dt)
            end = min(span["end_s"], stop)
            if start < end:
                updated = copy.deepcopy(span)
                updated.update(start_s=start, end_s=end)
                uncertain.append(updated)
    # A legacy orphan point must abstain, rather than become negative truth.
    for i, point in enumerate(points):
        if i in assigned:
            continue
        support = int(run_id[nearest_index(t, point["time_s"])])
        if support < 0:
            removed_unobservable += 1
            continue
        attrs = copy.deepcopy(point["attrs"])
        attrs.update(
            state="uncertain", period_ms=None, support_padding_ms=rule.frame_ms / 2
        )
        attrs["uncertainty_reasons"] = sorted(
            set(attrs.get("uncertainty_reasons", []) + ["isolated_candidate"])
        )
        start, end = clipped_interval(
            t, support_runs[support], point["time_s"], point["time_s"], padding
        )
        uncertain.append({"start_s": start, "end_s": end, "attrs": attrs})
        removed_fragments += 1
    record["crashes"] = sorted(retained, key=lambda point: point["time_s"])
    record["intervals"] = sorted(present, key=lambda span: span["start_s"])
    record["uncertain_intervals"] = sorted(uncertain, key=lambda span: span["start_s"])
    states, assessed = state_spans(
        t,
        observable,
        [(span["start_s"], span["end_s"]) for span in present],
        [(span["start_s"], span["end_s"]) for span in uncertain],
    )
    record.update(
        states=states,
        observable_spans=mask_spans(t, observable),
        assessed_spans=mask_spans(t, assessed),
        assessment_policy=POLICY,
    )
    record["state_seconds"] = {
        state: sum(
            span["end_s"] - span["start_s"] for span in states if span["state"] == state
        )
        for state in ("present", "absent", "uncertain", "unassessed")
    }
    record.setdefault(
        "q_source", "EFIT01" if record.get("qmin_available") else "unavailable"
    )
    return (
        observable,
        assessed,
        {
            "shot": record["shot"],
            "changed_observable_samples": int((old_observable != observable).sum()),
            "changed_observable_seconds": float(
                (old_observable != observable).sum() * dt
            ),
            "widened_uncertain_trains": widened,
            "split_trains": split_trains,
            "removed_unobservable_crashes": removed_unobservable,
            "removed_fragment_crashes": removed_fragments,
            "crashes_before": len(points),
            "crashes_after": len(retained),
            "present_intervals_before": len(before.get("intervals", [])),
            "present_intervals_after": len(present),
            "uncertain_intervals_before": len(before.get("uncertain_intervals", [])),
            "uncertain_intervals_after": len(uncertain),
        },
    )


def repair_shot(job):
    shot, work, population = job
    work = Path(work)
    path = work / "shots" / f"{shot}.json"
    record = json.loads(path.read_text())
    if record.get("assessment_policy") == POLICY:
        previous = work / "mask_repair_audit" / f"{shot}.json"
        audit = json.loads(previous.read_text()) if previous.exists() else {}
        return dict(audit, shot=shot, skipped="already_repaired")
    if "error" in record:
        return {"shot": shot, "skipped": "original_error", "error": record["error"]}
    # A stricter profile or clock guard cannot add assessed truth to the exact
    # existing all-unassessed partition, so it needs no further waveform read.
    if (
        not record.get("observable_spans")
        and not record.get("assessed_spans")
        and record.get("states")
        and all(span["state"] == "unassessed" for span in record["states"])
        and not any(
            record.get(key) for key in ("crashes", "intervals", "uncertain_intervals")
        )
        and not (work / "signals" / f"{shot}.npz").exists()
    ):
        backup = work / "mask_repair_before" / f"{shot}.json"
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            shutil.copyfile(path, backup)
        record["assessment_policy"] = POLICY
        record.setdefault(
            "q_source", "EFIT01" if record.get("qmin_available") else "unavailable"
        )
        changes = {
            "shot": shot,
            "changed_observable_samples": 0,
            "changed_observable_seconds": 0.0,
            "widened_uncertain_trains": 0,
            "split_trains": 0,
            "removed_unobservable_crashes": 0,
            "removed_fragment_crashes": 0,
            "crashes_before": 0,
            "crashes_after": 0,
            "present_intervals_before": 0,
            "present_intervals_after": 0,
            "uncertain_intervals_before": 0,
            "uncertain_intervals_after": 0,
            "population": population,
            "cached_signal": False,
            "native_error": None,
            "state_seconds": record["state_seconds"],
            "profile_support_read": "none; exact all-unassessed partition preserved",
        }
        save_json(path, record)
        save_json(work / "mask_repair_audit" / f"{shot}.json", changes)
        return changes
    rule = Rule(**json.loads((work / "freeze.json").read_text())["rule"])
    signal = work / "signals" / f"{shot}.npz"
    cached = None
    read_scope = "cached_full_profile"
    if signal.exists():
        with np.load(signal) as data:
            cached = {key: data[key] for key in data.files}
        t, y, old_observable = cached["t"], cached["y"], cached["observable"]
    else:
        try:
            with h5py.File(
                Paths.from_env().corpus_file(shot), "r", locking=False
            ) as file:
                t, y = sample_native(file["ece"], rows=slice(20, 36))
                y[(y < 0) | (y > 100)] = np.nan
                lo, hi = record["window_s"]
                keep = (t >= lo) & (t <= hi)
                t, y = t[keep], y[:, keep]
                old_observable = support_from_spans(
                    t, record.get("observable_spans", [])
                )
                if subset_sufficient(t, y, old_observable, rule):
                    read_scope = "core16_exact_lower_bound"
                else:
                    t, y = sample_native(file["ece"])
                    y[(y < 0) | (y > 100)] = np.nan
                    keep = (t >= lo) & (t <= hi)
                    t, y = t[keep], y[:, keep]
                    old_observable = support_from_spans(
                        t, record.get("observable_spans", [])
                    )
                    read_scope = "full48_after_insufficient_core_lower_bound"
        except (OSError, KeyError, ValueError) as error:
            # A failed native clock cannot contribute assessed absence.
            lo, hi = record["window_s"]
            t = np.arange(lo, hi + 0.000025, 0.0001)
            y = np.full((4, len(t)), np.nan, dtype=np.float32)
            old_observable = support_from_spans(t, record.get("observable_spans", []))
            record["mask_repair_native_error"] = f"{type(error).__name__}: {error}"
            read_scope = "failed_native_clock_all_unassessed"
    backup = work / "mask_repair_before" / f"{shot}.json"
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        shutil.copyfile(path, backup)
    observable, assessed, changes = repair_record(record, t, y, old_observable, rule)
    if cached is not None:
        cached.update(observable=observable, assessed=assessed)
        temporary = signal.with_suffix(".mask-partial")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **cached)
        temporary.replace(signal)
    save_json(path, record)
    changes.update(
        population=population,
        cached_signal=cached is not None,
        native_error=record.get("mask_repair_native_error"),
        state_seconds=record["state_seconds"],
        profile_support_read=read_scope,
    )
    save_json(work / "mask_repair_audit" / f"{shot}.json", changes)
    return changes


def cohort_shots():
    with (REPO / "data/events/catalog/cohort.csv").open() as handle:
        shots = {int(row["shot"]) for row in csv.DictReader(handle)}
    review = Paths.from_env().label_tables / "sawtooth_oscillation/review/labels.csv"
    if review.exists():
        with review.open() as handle:
            shots.update(int(row["shot"]) for row in csv.DictReader(handle))
    return shots


def audit_shot(job):
    shot, work = job
    work = Path(work)
    record = json.loads((work / "shots" / f"{shot}.json").read_text())
    if "error" in record:
        return {"shot": shot, "original_error": record["error"]}
    with np.load(work / "signals" / f"{shot}.npz") as data:
        t, y = data["t"], data["y"]
        observable, assessed = data["observable"], data["assessed"]
    cells = np.zeros(len(t), dtype=np.uint8)
    states = np.full(len(t), "", dtype="U10")
    for span in record["states"]:
        support = (t >= span["start_s"]) & (t < span["end_s"])
        cells[support] += 1
        states[support] = span["state"]
    violations = {
        "state_partition_samples": int((cells != 1).sum()),
        "observable_state_samples": int((observable != (states != "unassessed")).sum()),
        "assessed_state_samples": int(
            (assessed != ((states == "present") | (states == "absent"))).sum()
        ),
        "observable_under_four_finite_channels": int(
            (observable & (np.isfinite(y).sum(axis=0) < 4)).sum()
        ),
        "unobservable_crashes": 0,
        "assessed_uncertain_crashes": 0,
        "train_crosses_observability_gap": 0,
    }
    for point in record["crashes"]:
        index = nearest_index(t, point["time_s"])
        violations["unobservable_crashes"] += int(not observable[index])
        violations["assessed_uncertain_crashes"] += int(
            point["attrs"].get("state") == "uncertain" and assessed[index]
        )
    for kind in ("intervals", "uncertain_intervals"):
        for span in record[kind]:
            if span.get("attrs", {}).get("period_ms") is None:
                continue
            support = (t >= span["start_s"]) & (t < span["end_s"])
            violations["train_crosses_observability_gap"] += int(
                (~observable[support]).any()
            )
    return {"shot": shot, "samples": len(t), "violations": violations}


def audit_cohort(work, workers):
    jobs = [(shot, str(work)) for shot in sorted(cohort_shots())]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(audit_shot, jobs, chunksize=1))
    valid = [row for row in rows if "violations" in row]
    summary = {
        "assessment_policy": POLICY,
        "requested_shots": len(jobs),
        "audited_shots": len(valid),
        "original_errors": {
            row["shot"]: row["original_error"]
            for row in rows
            if "original_error" in row
        },
        "violations": {
            key: sum(row["violations"][key] for row in valid)
            for key in valid[0]["violations"]
        },
        "by_shot": rows,
    }
    save_json(work / "mask_audit_cohort.json", summary)
    small = {key: value for key, value in summary.items() if key != "by_shot"}
    small["source_details"] = str(work / "mask_audit_cohort.json")
    save_json(OUTPUT / "mask_audit_cohort.json", small)
    print(json.dumps(small), flush=True)


def audit_optimization_shot(job):
    shot, work = job
    with np.load(Path(work) / "signals" / f"{shot}.npz") as data:
        t, y, old = data["t"], data["y"], data["observable"]
    rule = Rule(**json.loads((Path(work) / "freeze.json").read_text())["rule"])
    full = old & profile_support(t, y, rule)
    core = y[20:36]
    sufficient = subset_sufficient(t, core, old, rule)
    optimized = old & profile_support(t, core if sufficient else y, rule)
    return {
        "shot": shot,
        "core16_sufficient": sufficient,
        "samples": len(t),
        "observable_samples": int(old.sum()),
        "mismatching_mask_samples": int((full != optimized).sum()),
    }


def audit_optimization(work, workers):
    jobs = [(shot, str(work)) for shot in sorted(cohort_shots())]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(audit_optimization_shot, jobs, chunksize=1))
    rule = Rule(**json.loads((Path(work) / "freeze.json").read_text())["rule"])
    synthetic_t = np.arange(400) * 0.0001
    synthetic_y = np.full((48, len(synthetic_t)), np.nan)
    synthetic_y[:4] = 1.0
    synthetic_y[20:22] = 1.0
    synthetic_old = np.ones(len(synthetic_t), dtype=bool)
    fallback_verified = not subset_sufficient(
        synthetic_t, synthetic_y[20:36], synthetic_old, rule
    ) and bool(profile_support(synthetic_t, synthetic_y, rule).all())
    synthetic_y[20:36] = 1.0
    synthetic_y[23:36, 201] = np.nan
    synthetic_old[:] = False
    synthetic_old[200] = True
    halo_verified = not subset_sufficient(
        synthetic_t, synthetic_y[20:36], synthetic_old, rule
    )
    synthetic_old[:] = False
    empty_verified = subset_sufficient(
        synthetic_t, synthetic_y[20:36], synthetic_old, rule
    )
    summary = {
        "scope": "all cached cohort and expert shots",
        "audited_shots": len(rows),
        "core16_sufficient_shots": sum(row["core16_sufficient"] for row in rows),
        "full48_fallback_shots": [
            row["shot"] for row in rows if not row["core16_sufficient"]
        ],
        "mismatching_mask_samples": sum(
            row["mismatching_mask_samples"] for row in rows
        ),
        "proof": (
            "Per-channel filtering is independent. Core16 finite counts "
            "lower-bound full48 counts. If core16 has four finite channels "
            "throughout every Gaussian support overlapping old observability, "
            "full48 has four too; both final masks equal old observability. "
            "Otherwise reread full48."
        ),
        "empty_support_fastpath": (
            "Preserve exact existing all-unassessed partition only when "
            "observable/assessed support and all event lists are empty, "
            "with no cached signal to rewrite."
        ),
        "refit": False,
        "criteria_changed": False,
        "synthetic_checks": {
            "two_core_channels_with_four_outer_requires_full48": fallback_verified,
            "bad_core_neighbor_inside_filter_halo_requires_full48": halo_verified,
            "empty_observable_support_certificate": empty_verified,
        },
        "by_shot": rows,
    }
    save_json(work / "mask_optimization.json", summary)
    small = {key: value for key, value in summary.items() if key != "by_shot"}
    small["source_details"] = str(work / "mask_optimization.json")
    save_json(OUTPUT / "mask_optimization.json", small)
    print(json.dumps(small), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--population", action="store_true")
    parser.add_argument("--existing-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--audit-optimization", action="store_true")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or not 0 <= args.shard < args.shards:
        parser.error("workers must be 1..4 and shard must be within shards")
    if args.audit_only:
        audit_cohort(args.work, args.workers)
        return
    if args.audit_optimization:
        audit_optimization(args.work, args.workers)
        return
    cohort = cohort_shots()
    if args.population:
        available = (
            {int(path.stem) for path in (args.work / "shots").glob("*.json")}
            if args.existing_only
            else {
                int(path.name.split("_")[0])
                for path in Paths.from_env().corpus.glob("*_processed.h5")
            }
        )
        shots = sorted(available - cohort)
    else:
        shots = sorted(cohort)
    shots = [shot for shot in shots if shot % args.shards == args.shard]
    jobs = [
        (shot, str(args.work), args.population)
        for shot in shots
        if (args.work / "shots" / f"{shot}.json").exists()
    ]
    started = time.monotonic()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(repair_shot, jobs, chunksize=1)):
            rows.append(row)
            if i % 50 == 0:
                print(
                    f"{i + 1}/{len(jobs)} shot {row['shot']} "
                    f"{row.get('skipped', 'repaired')}",
                    flush=True,
                )
    changed = [row for row in rows if "skipped" not in row]
    audits = [row for row in rows if "changed_observable_samples" in row]
    summary = {
        "assessment_policy": POLICY,
        "scope": "population" if args.population else "cohort_plus_review",
        "requested_count": len(shots),
        "existing_record_count": len(jobs),
        "repaired_count": len(changed),
        "repaired_total_count": len(audits),
        "skip_counts": dict(
            Counter(row["skipped"] for row in rows if "skipped" in row)
        ),
        "rule_source": str(args.work / "freeze.json"),
        "original_records": str(args.work / "mask_repair_before"),
        "refit": False,
        "amplitudes_modified": False,
        "totals": {
            key: sum(row[key] for row in audits)
            for key in (
                "changed_observable_samples",
                "changed_observable_seconds",
                "widened_uncertain_trains",
                "split_trains",
                "removed_unobservable_crashes",
                "removed_fragment_crashes",
                "crashes_before",
                "crashes_after",
                "present_intervals_before",
                "present_intervals_after",
                "uncertain_intervals_before",
                "uncertain_intervals_after",
            )
        },
        "by_shot": rows,
        "elapsed_s": round(time.monotonic() - started, 3),
    }
    scope = "population" if args.population else "cohort"
    if args.shards > 1:
        scope += f"_shard_{args.shard}"
    small = {key: value for key, value in summary.items() if key != "by_shot"}
    small["source_details"] = str(args.work / f"mask_repair_{scope}.json")
    save_json(OUTPUT / f"mask_repair_{scope}.json", small)
    save_json(args.work / f"mask_repair_{scope}.json", summary)
    print(
        json.dumps({key: value for key, value in summary.items() if key != "by_shot"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
