"""Freeze a blind validation sample and audit legacy crash timing locally.

Run queue-base before model predictions exist, queue after frozen predictions,
reader-audit on the retained legacy cache, and figures after signal regeneration.
Annotators receive only annotation_pack; selection_audit is never distributed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sawtooth_physics import REPO, REVIEW, SEED, records_at, save_json

from labeler.config import Paths
from labeler.events import heuristics
from labeler.sawtooth.metrics import event_cells, point_metrics, spans_at

WORK4 = Paths.from_env().root / "round4/saw/fix4"
PREVIOUS = WORK4.parent / "fix2"
OUTPUT = REPO / "outputs/labeler/sawtooth/fix4"
LEGACY_SHOTS = [190602, 190604, 192090, 201948, 192154, 191384, 203349]
WINDOW_SECONDS = 0.6
MINIMUM_WINDOW_SECONDS = 0.15


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(UTC).isoformat()


def points(record):
    return np.asarray([row["time_s"] for row in record["crashes"]], dtype=float)


def finite_summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0, "median": None, "p25": None, "p75": None}
    return {
        "n": len(values),
        "median": float(np.median(values)),
        "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)),
    }


def mean_trace(values):
    finite = np.isfinite(values)
    counts = finite.sum(axis=0)
    return np.divide(
        np.where(finite, values, 0).sum(axis=0),
        counts,
        out=np.full(values.shape[1], np.nan),
        where=counts > 0,
    )


def median_rows(values):
    valid = np.isfinite(values).any(axis=1)
    result = np.full(values.shape[0], np.nan)
    result[valid] = np.nanmedian(values[valid], axis=1)
    return result


def nearest_offsets(reference, other):
    """Signed other-reference offsets; nearest neighbours may be reused."""
    reference, other = np.asarray(reference), np.sort(other)
    if not len(other):
        return np.full(len(reference), np.nan)
    at = np.searchsorted(other, reference)
    left, right = np.clip(at - 1, 0, len(other) - 1), np.clip(at, 0, len(other) - 1)
    take = np.where(
        np.abs(other[left] - reference) <= np.abs(other[right] - reference),
        left,
        right,
    )
    return (other[take] - reference) * 1000


def tile_key(row):
    return row["shot"], row["start_s"], row["end_s"]


def observable_tiles(record):
    """Nonoverlapping 150–600ms tiles independent of any event output."""
    tiles = []
    for start, end in record["observable_spans"]:
        count = int(np.floor((end - start + 1e-9) / WINDOW_SECONDS))
        for index in range(count):
            lo = float(start + index * WINDOW_SECONDS)
            tiles.append(
                {"shot": int(record["shot"]), "start_s": lo, "end_s": lo + 0.6}
            )
        residual_start = float(start + count * WINDOW_SECONDS)
        if end - residual_start >= MINIMUM_WINDOW_SECONDS - 1e-9:
            tiles.append(
                {
                    "shot": int(record["shot"]),
                    "start_s": residual_start,
                    "end_s": float(end),
                }
            )
    return tiles


def draw_balanced(pool, count, rng):
    """Random shot order each round, then a random remaining tile of that shot."""
    remaining = list(pool)
    selected = []
    while remaining and len(selected) < count:
        shots = np.unique([row["shot"] for row in remaining])
        for shot in rng.permutation(shots):
            rows = [row for row in remaining if row["shot"] == shot]
            row = rows[int(rng.integers(len(rows)))]
            selected.append(row)
            remaining.remove(row)
            if len(selected) == count:
                break
    return selected


def queue_base(args):
    """Freeze probability sample and candidate-free draws before model access."""
    audit = args.work / "selection_audit"
    audit.mkdir(parents=True, exist_ok=True, mode=0o700)
    audit.chmod(0o700)
    destination = audit / "base_selection.json"
    if destination.exists():
        raise ValueError("base already frozen; do not resample after predictions")
    if any((args.work / "predictions").rglob("*.npz")):
        raise ValueError("freeze the primary sample before any model predictions")
    cohort = pd.read_csv(REPO / "data/events/catalog/cohort.csv")
    experts = set(pd.read_csv(REVIEW).shot)
    eligible = sorted(set(cohort.loc[cohort.split == "val", "shot"]) - experts)
    records = {r["shot"]: r for r in records_at(args.work, eligible)}
    if len(eligible) != 47 or set(records) != set(eligible):
        raise ValueError("requires all 47 nonexpert fixed-validation records")
    rng = np.random.default_rng(SEED)
    rows, frame, exclusions, shot_frame_counts = [], [], [], {}
    for shot in eligible:
        rec = records[shot]
        if "error" in rec:
            exclusions.append(
                {
                    "shot": shot,
                    "reason": "reader_or_sensor_support_error",
                    "error": rec["error"],
                }
            )
            continue
        tiles = observable_tiles(rec)
        if not tiles:
            exclusions.append(
                {
                    "shot": shot,
                    "reason": "no_contiguous_observable_150ms",
                    "observable_seconds": sum(
                        b - a for a, b in rec["observable_spans"]
                    ),
                }
            )
            continue
        shot_frame_counts[shot] = len(tiles)
        frame.extend(tiles)
        for index in rng.choice(len(tiles), min(3, len(tiles)), replace=False):
            rows.append(
                {
                    **tiles[int(index)],
                    "stratum": "random_observable",
                    "eligible_tiles_in_shot": len(tiles),
                }
            )
    if not frame:
        raise ValueError("no observable annotation windows of at least 150ms")
    initial_count = len(rows)
    used = {tile_key(row) for row in rows}
    remaining = [row for row in frame if tile_key(row) not in used]
    topup_count = min(max(0, 141 - len(rows)), len(remaining))
    for index in rng.choice(len(remaining), topup_count, replace=False):
        row = remaining[int(index)]
        rows.append(
            {
                **row,
                "stratum": "random_observable",
                "eligible_tiles_in_shot": shot_frame_counts[row["shot"]],
            }
        )
    for row in rows:
        n = row["eligible_tiles_in_shot"]
        initial_probability = min(3, n) / n
        row["inclusion_probability"] = initial_probability + (
            (1 - initial_probability) * topup_count / len(remaining) if remaining else 0
        )
    primary_count = len(rows)
    # The full random sample is complete before consulting detector outputs.
    random_hash = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    zero_shots = [shot for shot in shot_frame_counts if not records[shot]["crashes"]]
    used = {tile_key(row) for row in rows}
    zero_pool = [
        row for row in frame if row["shot"] in zero_shots and tile_key(row) not in used
    ]
    zero_rows = draw_balanced(zero_pool, 15, rng)
    rows.extend({**row, "stratum": "zero_candidate_shot"} for row in zero_rows)
    missing = 15 - len(zero_rows)
    if missing:
        used = {tile_key(row) for row in rows}
        replacements = draw_balanced(
            [row for row in frame if tile_key(row) not in used], missing, rng
        )
        rows.extend(
            {**row, "stratum": "random_observable_reserve"} for row in replacements
        )
    save_json(
        destination,
        {
            "frozen_utc": now(),
            "seed": SEED,
            "eligible_shots": eligible,
            "observable_shots": sorted(shot_frame_counts),
            "shot_exclusions": exclusions,
            "random_primary_windows": primary_count,
            "random_initial_windows": initial_count,
            "random_topup_windows": topup_count,
            "random_topup_remaining_pool": len(remaining),
            "shot_frame_counts": shot_frame_counts,
            "expert_excluded_shots": sorted(int(shot) for shot in experts),
            "observable_frame_tiles": len(frame),
            "observable_frame_seconds": sum(
                row["end_s"] - row["start_s"] for row in frame
            ),
            "evaluation_frame_interior_seconds": sum(
                row["end_s"] - row["start_s"] - 0.004 for row in frame
            ),
            "total_observable_seconds": sum(
                sum(b - a for a, b in r["observable_spans"])
                for r in records.values()
                if "error" not in r
            ),
            "random_selection_sha256": random_hash,
            "random_selection_completed_before_candidate_access": True,
            "model_predictions_read": False,
            "zero_candidate_shots": zero_shots,
            "zero_candidate_definition": (
                "No saved accepted or uncertain physics crash points; raw "
                "filter/noise candidates do not define the sampling frame"
            ),
            "zero_candidate_pool_tiles": len(zero_pool),
            "rows": rows,
            "shot_record_sha256": {
                str(shot): digest(args.work / "shots" / f"{shot}.json")
                for shot in eligible
            },
            "sampling_design": (
                "Draw min(3,Nshot) tiles uniformly per shot, then uniformly "
                "sample remaining tiles to reach 141 where available. Tile "
                "widths 150–600ms; gaps never bridged. Marginal inclusion "
                "probability pi=p0+(1-p0)*K/R, p0=min(3,Nshot)/Nshot with "
                "deterministic remaining-frame count R and topup count K. "
                "Zero-candidate extras use observable support before models. "
                "Residual contiguous supports shorter than 150ms are excluded."
            ),
        },
    )
    prereg = {
        "frozen_utc": now(),
        "annotation_status": "pending owner; no truth fabricated",
        "base_selection_sha256": digest(destination),
        "primary_sample": {
            "windows": primary_count,
            "shots": sorted(shot_frame_counts),
            "eligible_shots": 47,
            "shot_exclusions": exclusions,
            "window_duration_s": [MINIMUM_WINDOW_SECONDS, WINDOW_SECONDS],
        },
        "secondary_samples": (
            "Zero-candidate, model-negative, uncertain and disagreement "
            "supplements: report separately; do not pool as prevalence estimates"
        ),
        "truth": (
            "Blind crash times, present/absent intervals and explicit ambiguity "
            "masks from ECE; ambiguity excluded identically for all methods"
        ),
        "crash_matching": (
            "One-to-one labeler.sawtooth.metrics.event_cells matching within +/-2ms"
        ),
        "presence": "2ms bins on annotated observable nonambiguous support",
        "window_boundary_policy": (
            "Exclude crash truth/picks within 2ms of window edges for every "
            "method; annotator may extend ambiguity where context is insufficient"
        ),
        "primary_estimand": (
            "Crash precision/recall on eligible observable tile interiors from "
            "inverse marginal "
            "inclusion weighted event counts (1/pi) on random tiles, and "
            "presence precision/recall from identically weighted 2ms bin "
            "counts. Report unweighted sensitivity too."
        ),
        "methods": [
            "physics rule",
            "legacy ECE rule",
            "saw-ours",
            "derivative picker gated by HL-3",
            "derivative-only",
            "always-present presence baseline",
        ],
        "method_applicability": {
            "always-present presence baseline": (
                "Presence only; no crash picker, so crash metrics are not applicable"
            )
        },
        "retuning": "No rule, checkpoint, threshold or picker changes after annotation",
        "intervals": "1000 whole-shot bootstrap replicates, seed 20261003",
        "event_recall_precision_target": {
            "nominal_95percent_half_width": 0.1,
            "worst_case_recall": 0.5,
            "normal_approximation_required_independent_positive_events": 97,
            "wilson_half_width_at_n97_p_half": float(
                1.959963984540054 / (2 * np.sqrt(97 + 1.959963984540054**2))
            ),
            "caveat": (
                "200 windows do not guarantee 97 independent positive events. "
                "Repeated crashes within shot/train are correlated, so Wilson "
                "event intervals are descriptive and may understate uncertainty; "
                "shot bootstrap is primary. Report positive events, positive "
                "trains and positive shots and achieved CI widths; request more "
                "random windows if positive support is insufficient."
            ),
        },
        "source_prediction_hashes": "Freeze in completed_selection before annotation",
    }
    save_json(audit / "preregistered_evaluation.json", prereg)
    save_json(args.output / "blind_queue_preregistration.json", prereg)
    print(
        {
            "base_windows": len(rows),
            "random_windows": primary_count,
            "observable_shots": len(shot_frame_counts),
            "exclusions": exclusions,
            "zero_shots": zero_shots,
        }
    )


def queue(args):
    """Enrich the frozen sample without exposing selections or predictions."""
    import h5py

    from labeler.sawtooth.preprocessing import sample_native

    audit = args.work / "selection_audit"
    if (audit / "completed_selection.json").exists():
        raise ValueError("completed queue is frozen; do not overwrite annotations")
    base_path = audit / "base_selection.json"
    base = json.loads(base_path.read_text())
    rows = list(base["rows"])
    eligible = base["observable_shots"]
    records = {r["shot"]: r for r in records_at(args.work, eligible)}
    for shot in eligible:
        if base["shot_record_sha256"][str(shot)] != digest(
            args.work / "shots" / f"{shot}.json"
        ):
            raise ValueError("input records changed after base selection")
    predictions, hashes, thresholds = {}, {}, {}
    benchmark_path = args.output / "benchmark.json"
    if not benchmark_path.exists():
        raise ValueError(
            "finish frozen benchmark operating points before queue completion"
        )
    hashes[str(benchmark_path)] = digest(benchmark_path)
    hashes[str(args.work / "freeze.json")] = digest(args.work / "freeze.json")
    for name in ("saw-ours", "saw-hl3"):
        directory = args.work / "predictions/ensemble" / name
        manifest = json.loads((directory / "manifest.json").read_text())
        thresholds[name] = manifest["presence_threshold"]
        hashes[str(directory / "manifest.json")] = digest(directory / "manifest.json")
        predictions[name] = {}
        for shot in eligible:
            path = directory / f"{shot}.npz"
            with np.load(path) as stored:
                predictions[name][shot] = {key: stored[key] for key in stored.files}
            hashes[str(path)] = digest(path)
    used = {tile_key(row) for row in rows}
    frame = [
        tile
        for shot in eligible
        for tile in observable_tiles(records[shot])
        if tile_key(tile) not in used
    ]
    stats = {}
    for tile in frame:
        shot, lo, hi = tile_key(tile)
        flags, pick_counts = [], []
        for name in ("saw-ours", "saw-hl3"):
            pred = predictions[name][shot]
            keep = (pred["t"] >= lo) & (pred["t"] < hi)
            flags.append(float(np.mean(pred["presence"][keep] >= thresholds[name])))
            pick_counts.append(
                int(((pred["picks"] >= lo) & (pred["picks"] < hi)).sum())
            )
        uncertain = sum(
            max(0, min(hi, span["end_s"]) - max(lo, span["start_s"]))
            for span in records[shot]["states"]
            if span["state"] == "uncertain"
        ) / (hi - lo)
        teacher = points(records[shot])
        old = json.loads((args.previous / "old_rule" / f"{shot}.json").read_text())
        legacy = points(old)
        a, b = (
            legacy[(legacy >= lo) & (legacy < hi)],
            teacher[(teacher >= lo) & (teacher < hi)],
        )
        agreement = point_metrics(event_cells(a, b, 2))["f1"]
        stats[tile_key(tile)] = {
            "model_positive_fractions": flags,
            "model_picks": pick_counts,
            "uncertain_fraction": uncertain,
            "legacy_physics_f1_2ms": agreement,
        }
    rng = np.random.default_rng(SEED + 1)
    pools = {
        "model_predicted_negative": [
            row
            for row in frame
            if any(
                fraction < 0.1 and picks == 0
                for fraction, picks in zip(
                    stats[tile_key(row)]["model_positive_fractions"],
                    stats[tile_key(row)]["model_picks"],
                    strict=True,
                )
            )
        ],
        "algorithm_uncertain": [
            row for row in frame if stats[tile_key(row)]["uncertain_fraction"] >= 0.5
        ],
        "disagreement": [
            row
            for row in frame
            if abs(np.diff(stats[tile_key(row)]["model_positive_fractions"])[0]) >= 0.5
            or (
                stats[tile_key(row)]["legacy_physics_f1_2ms"] is not None
                and stats[tile_key(row)]["legacy_physics_f1_2ms"] < 0.3
            )
        ],
    }
    pool_counts = {}
    for name, target in zip(pools, (15, 15, 14), strict=True):
        used = {tile_key(row) for row in rows}
        pool = [row for row in pools[name] if tile_key(row) not in used]
        pool_counts[name] = len(pool)
        selection = draw_balanced(pool, target, rng)
        rows.extend({**row, "stratum": name} for row in selection)
    used = {tile_key(row) for row in rows}
    target_count = min(200, base["observable_frame_tiles"])
    reserve = draw_balanced(
        [row for row in frame if tile_key(row) not in used],
        target_count - len(rows),
        rng,
    )
    rows.extend({**row, "stratum": "random_observable_reserve"} for row in reserve)
    if (
        len(rows) != target_count
        or len({tile_key(row) for row in rows}) != target_count
    ):
        raise ValueError("annotation selection contains duplicates or lost support")
    rng.shuffle(rows)
    pack = args.work / "annotation_pack"
    if pack.exists() and any(pack.iterdir()):
        raise ValueError("annotation pack contains files; do not overwrite reviews")
    pack.mkdir(parents=True, exist_ok=True)
    raw_signals, raw_provenance = {}, []
    for shot in sorted({row["shot"] for row in rows}):
        source = Paths.from_env().corpus_file(shot)
        with h5py.File(source, "r", locking=False) as file:
            raw_t, raw_y = sample_native(file["ece"])
            native_shape = list(file["ece/ydata"].shape)
        raw_signals[shot] = (raw_t, raw_y)
        stat = source.stat()
        raw_provenance.append(
            {
                "shot": shot,
                "source": str(source),
                "native_shape": native_shape,
                "file_size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "input_group": "ece/xdata and ece/ydata",
                "processing": "sample_native FIR antialias to 10kHz; all 48 ECE channels",
                "channel_geometry_mask_applied": False,
            }
        )
        print({"blind_raw_signal_read": shot}, flush=True)
    manifest_rows, private_rows = [], []
    for index, row in enumerate(rows):
        shot, lo, hi = tile_key(row)
        item = f"window_{index + 1:03d}"
        with np.load(args.work / "signals" / f"{shot}.npz") as signal:
            keep = (signal["t"] >= lo) & (signal["t"] < hi)
            raw_t, raw_y = raw_signals[shot]
            raw_keep = (raw_t >= lo) & (raw_t < hi)
            if not np.array_equal(signal["t"][keep], raw_t[raw_keep]):
                raise ValueError(f"native clock mismatch in blind window {row}")
            geometry = records[shot]["core_geometry"]
            rho = np.asarray(
                [
                    np.nan if value is None else value
                    for value in geometry.get("nominal_rho_median", [None] * 48)
                ]
            )
            arrays = {
                "t_s": raw_t[raw_keep],
                "ece_kev": raw_y[:, raw_keep],
                "observable": signal["observable"][keep],
                "channel": np.arange(signal["y"].shape[0]),
                "nominal_rho_median": rho,
            }
            if not arrays["observable"].all():
                raise ValueError(f"selected tile leaves observable support: {row}")
            np.savez_compressed(pack / f"{item}.npz", **arrays)
        public = {
            "window_id": item,
            "shot": shot,
            "start_s": lo,
            "end_s": hi,
            "signal_file": f"{item}.npz",
        }
        manifest_rows.append(public)
        private_rows.append(
            {**public, **row, "selection_diagnostics": stats.get(tile_key(row))}
        )
        save_json(
            pack / f"{item}.json",
            {
                "window_id": item,
                "crash_times_s": [],
                "present_intervals_s": [],
                "absent_intervals_s": [],
                "ambiguous_intervals_s": [],
                "review_complete": False,
                "annotator_notes": "",
            },
        )
    save_json(
        pack / "manifest.json",
        {
            "windows": manifest_rows,
            "instructions": (
                "Mark all physically credible sawtooth crash times and positive "
                "and negative intervals in each ECE window. Mark ambiguity "
                "explicitly. Partition each window completely into positive, "
                "negative or ambiguous intervals. Empty lists are unreviewed until review_complete "
                "is true. Times are absolute seconds. Channel nominal rho is "
                "abs(R-axis)/(LCFS_outer_R-axis), a geometric coordinate, "
                "not calibrated flux. Only channels 0–39 have nominal radial "
                "ordering; channels 40–47 remain spatially unverified. Raw "
                "traces are preserved; do not infer redistribution from "
                "unverified adjacency, and mark ambiguity when geometry is "
                "insufficient. Review every window before returning."
            ),
            "npz_fields": list(arrays),
            "signal_units": "t_s seconds; ece_kev keV; zero-based channel indices",
            "raw_data_provenance": raw_provenance,
            "observable_eligibility": (
                "Preregistered physical sensor support; not a crash, presence, "
                "uncertainty, geometry-channel-mask or model decision"
            ),
        },
    )
    csv_path = REPO / "data/events/sawtooth_oscillation/review/crash_time_queue.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "window_id": row["window_id"],
                "shot": row["shot"],
                "window": f"{row['start_s'] * 1000:.3f}:{row['end_s'] * 1000:.3f} ms",
            }
            for row in manifest_rows
        ]
    ).to_csv(csv_path, index=False)
    # Field allowlist is deliberately independent of the saved model schemas.
    allowed_fields = {"t_s", "ece_kev", "observable", "channel", "nominal_rho_median"}
    for path in pack.glob("*.npz"):
        with np.load(path) as stored:
            if set(stored.files) != allowed_fields:
                raise ValueError(f"unexpected field in blind signal pack: {path}")
    selection_path = audit / "completed_selection.json"
    save_json(
        selection_path,
        {
            "frozen_utc": now(),
            "base_selection_sha256": digest(base_path),
            "prediction_sha256": hashes,
            "thresholds": thresholds,
            "supplement_pool_counts": pool_counts,
            "strata": dict(Counter(row["stratum"] for row in rows)),
            "rows": private_rows,
            "annotation_pack_sha256": {
                path.name: digest(path) for path in sorted(pack.iterdir())
            },
            "audit_is_not_part_of_annotation_pack": True,
        },
    )
    summary = {
        "annotation_status": "pending owner; no independent labels available",
        "windows": len(rows),
        "random_primary_windows": base["random_primary_windows"],
        "target_windows": 200,
        "available_observable_frame_tiles": base["observable_frame_tiles"],
        "eligible_shots": base["eligible_shots"],
        "shot_exclusions": base["shot_exclusions"],
        "shots": eligible,
        "shot_count": len(eligible),
        "total_window_seconds": sum(row["end_s"] - row["start_s"] for row in rows),
        "strata": dict(Counter(row["stratum"] for row in rows)),
        "annotation_pack": str(pack),
        "queue_csv": str(csv_path.relative_to(REPO)),
        "private_selection_audit": str(selection_path),
        "private_selection_sha256": digest(selection_path),
        "pack_has_no_predictions_candidates_or_states": True,
        "input": "Native ECE FIR antialias to 10kHz, all48 channels; no spatial masks",
        "preregistered_evaluation": str(audit / "preregistered_evaluation.json"),
        "source": str(Path(__file__).relative_to(REPO)),
    }
    save_json(args.output / "crash_time_queue.json", summary)
    print(summary, flush=True)


def reader_audit(args):
    """Check native reader, cached old picks and picker time-index arithmetic."""
    import h5py
    from sawtooth_clock_audit import VirtualZeroChannel

    from labeler.events.pipeline import _read_group
    from labeler.sawtooth.preprocessing import sample_native

    rows = []
    for shot in LEGACY_SHOTS:
        old = json.loads((args.previous / "old_rule" / f"{shot}.json").read_text())
        lo, hi = old["window_s"]
        path = Paths.from_env().corpus_file(shot)
        t, y = _read_group(path, "ece")
        keep = (t >= lo) & (t <= hi)
        clipped_t, clipped_y = t[keep], y[:, keep]
        recomputed = heuristics.sawtooth_events(
            clipped_y, clipped_t, shot=shot, t_cov=(lo, hi)
        )
        clipped_times = np.asarray([event.t0_s for event in recomputed])
        cached_times = points(old)
        equal = np.array_equal(cached_times, clipped_times)
        full = heuristics.sawtooth_events(y, t, shot=shot, t_cov=(t[0], t[-1]))
        full_times = np.asarray(
            [event.t0_s for event in full if lo <= event.t0_s <= hi]
        )
        env, te = heuristics.envelope(clipped_y, clipped_t)
        drops = heuristics._bin_drops(env)
        candidates = np.flatnonzero((drops > heuristics.DROP_FRAC).sum(axis=0) >= 2)
        retained = np.asarray(
            heuristics._crash_bins(
                drops,
                env_ms=1.0,
                drop_frac=heuristics.DROP_FRAC,
                min_channels=2,
                min_interval_ms=10.0,
            ),
            dtype=int,
        )
        candidate_times, retained_times = te[candidates + 1], te[retained + 1]
        with h5py.File(path, "r", locking=False) as file:
            native_t, _ = sample_native(
                {
                    "xdata": file["ece/xdata"],
                    "ydata": VirtualZeroChannel(len(t)),
                }
            )
        with np.load(args.previous / "signals" / f"{shot}.npz") as signal:
            cache_t = signal["t"]
        native_selected = native_t[(native_t >= lo) & (native_t <= hi)]
        production = {"file": str(Paths.from_env().events_file(shot))}
        try:
            table = pd.read_parquet(Paths.from_env().events_file(shot))
            selected = table.loc[table.source == "ece_sawtooth"]
            attrs = [
                json.loads(value) if isinstance(value, str) else value
                for value in selected["attrs"].tolist()
            ]
            window = selected.loc[(selected.t0_s >= lo) & (selected.t0_s <= hi)]
            stored_times = window.t0_s.to_numpy()
            production.update(
                rows=len(selected),
                columns=list(selected.columns),
                attrs_samples=attrs[:3],
                detector_counts=dict(
                    Counter(value.get("detector", "unspecified") for value in attrs)
                ),
                diagnostic_counts=dict(Counter(selected.diag.tolist())),
                t0_s=selected.t0_s.tolist(),
                comparison_window_s=[lo, hi],
                stored_points_in_comparison_window=len(window),
                diagnostic_counts_in_comparison_window=dict(
                    Counter(window.diag.tolist())
                ),
                stored_vs_full_legacy_cells_2ms=event_cells(
                    full_times, stored_times, 2
                ).tolist(),
                stored_vs_full_legacy_cells_2ms_by_diagnostic={
                    diagnostic: event_cells(
                        full_times,
                        window.loc[window.diag == diagnostic, "t0_s"].to_numpy(),
                        2,
                    ).tolist()
                    for diagnostic in sorted(set(window.diag))
                },
            )
        except (OSError, ValueError, KeyError, AttributeError) as error:
            production["error"] = f"{type(error).__name__}: {error}"
        rows.append(
            {
                "shot": shot,
                "corpus": str(path),
                "native_shape": list(y.shape),
                "native_clock_start_s": float(t[0]),
                "native_clock_median_dt_s": float(np.median(np.diff(t))),
                "window_s": [lo, hi],
                "cached_legacy_equal_native_clipped_rerun": equal,
                "cached_legacy_count": len(cached_times),
                "rerun_legacy_count": len(clipped_times),
                "whole_record_old_count_in_window": len(full_times),
                "whole_record_vs_clipped_cells_2ms": event_cells(
                    clipped_times, full_times, 2
                ).tolist(),
                "whole_record_vs_clipped_signed_offset_ms": finite_summary(
                    nearest_offsets(clipped_times, full_times)
                ),
                "cached_clock_equals_exact_native_sample_clock": bool(
                    np.array_equal(cache_t, native_selected)
                ),
                "picker_clock_contract": {
                    "envelope_first_center_s": float(te[0]),
                    "expected_first_center_s": float(clipped_t[0] + 0.0005),
                    "bin_drop_i_event_time": "te[i+1]; first bin after negative step",
                    "time_mapping_error_ms": float(
                        abs(te[0] - clipped_t[0] - 0.0005) * 1000
                    ),
                    "raw_candidate_count": len(candidate_times),
                    "greedy_retained_candidate_count": len(retained_times),
                    "inversion_accepted_count": len(clipped_times),
                    "expected_quantization_lag_ms": [0, 1],
                },
                "production_event_store": production,
            }
        )
        # Retained envelope data supports later timing diagnosis, never annotation.
        destination = args.work / "legacy_audit" / f"{shot}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            t=te,
            y=env,
            raw_candidates=candidate_times,
            retained_candidates=retained_times,
            legacy_picks=clipped_times,
        )
        save_json(args.work / "legacy_audit/reader_audit.json", {"by_shot": rows})
        print(
            {
                "shot": shot,
                "legacy_cache_exact": equal,
                "clock_exact": rows[-1][
                    "cached_clock_equals_exact_native_sample_clock"
                ],
            },
            flush=True,
        )
    summary = {
        "by_shot": rows,
        "legacy_method": "sawtooth_events ECE rule; retained unchanged defaults",
        "current_pipeline_method": (
            "sawtooth_block now uses sawtooth_crashes and sawtooth_events_v3 "
            "over ECE/SXR since 2026-09-30; the retained legacy cache audits "
            "the preceding omnimode-style ECE rule, not the current pipeline"
        ),
        "cache_reader_conclusion": (
            "No cached reader or clock defect found"
            if all(
                row["cached_legacy_equal_native_clipped_rerun"]
                and row["cached_clock_equals_exact_native_sample_clock"]
                for row in rows
            )
            else "Reader/clock mismatch requires investigation before reuse"
        ),
        "picker_time_contract": (
            "Original envelope time is first post-step 1ms bin center; "
            "this can explain up to 1ms lag, not a fixed +/-10ms offset. "
            "Greedy candidate spacing is 10ms, applied before inversion."
        ),
    }
    save_json(args.output / "legacy_reader_audit.json", summary)


def trace_channels(record, selected):
    core = int(selected["attrs"]["central_channel"])
    geometry = record["core_geometry"]
    outer = list(geometry["outer_channels"])
    return core, outer


def channel_label(record, channels, kind):
    geometry = record["core_geometry"]
    rho = geometry.get("nominal_rho_median", [None] * 48)
    values = [rho[channel] for channel in channels if rho[channel] is not None]
    if values:
        lo, hi = min(values), max(values)
        text = f"{lo:.2f}" if hi - lo < 0.02 else f"{lo:.2f}–{hi:.2f}"
        return f"{kind}: nominal ρ={text}"
    return f"{kind}: ch " + ",".join(map(str, channels))


def style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
        }
    )
    return plt


def save_plot(fig, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination.with_suffix(".pdf"))
    fig.savefig(destination.with_suffix(".png"), dpi=150)
    return {
        "pdf": str(destination.with_suffix(".pdf")),
        "png": str(destination.with_suffix(".png")),
        "png_inspected": False,
    }


def paper_example(args):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    plt = style()
    shot = args.example_shot
    cohort = pd.read_csv(REPO / "data/events/catalog/cohort.csv").set_index("shot")
    if cohort.loc[shot, "split"] != "train" or shot in set(pd.read_csv(REVIEW).shot):
        raise ValueError("paper example requires a nonexpert train shot")
    record = records_at(args.work, [shot])[0]
    with np.load(args.work / "signals" / f"{shot}.npz") as signal:
        t, y = signal["t"], signal["y"]
    candidates = [
        row for row in record["crashes"] if row["attrs"]["state"] == "present"
    ]
    if not candidates:
        raise ValueError("paper example needs a present train")
    selected = candidates[len(candidates) // 2]
    lo, hi = selected["time_s"] - 0.15, selected["time_s"] + 0.15
    core, outer = trace_channels(record, selected)
    labels = [
        channel_label(record, [core], "Core"),
        channel_label(record, outer, "Outer LFS"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(3.25, 2.8), sharex=True)
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.15, top=0.74, hspace=0.13)
    colors = {
        "present": "#009E73",
        "absent": "#FFFFFF",
        "absent_q_prior": "#CC79A7",
        "uncertain": "#E69F00",
        "unassessed": "#999999",
    }
    state_names = {
        "absent_q_prior": "Q-prior only",
        "absent": "Absent (ECE-tested)",
    }
    shown_states = set()
    near = (t >= lo) & (t <= hi)
    traces = [y[core], mean_trace(y[outer])]
    handles = []
    for axis, trace, label, color in zip(
        axes, traces, labels, ("#0072B2", "#D55E00"), strict=True
    ):
        for span in record["states"]:
            a, b = max(lo, span["start_s"]), min(hi, span["end_s"])
            if b > a:
                shown_states.add(span["state"])
                axis.axvspan(
                    a * 1000, b * 1000, color=colors[span["state"]], alpha=0.18, lw=0
                )
        (line,) = axis.plot(
            t[near] * 1000, trace[near], color=color, lw=0.7, label=label
        )
        handles.append(line)
        for crash in record["crashes"]:
            if lo <= crash["time_s"] <= hi:
                axis.axvline(crash["time_s"] * 1000, color="0.15", ls=":", lw=0.6)
        axis.set_ylabel("Te (keV)")
        axis.grid(axis="y", color="0.9", lw=0.3)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].set(xlabel="Time (ms)", xlim=(lo * 1000, hi * 1000))
    axes[-1].xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    state_handles = [
        Patch(
            facecolor=colors[state],
            edgecolor="0.6",
            alpha=0.4,
            label=f"{state_names.get(state, state.capitalize())} (algorithmic state)",
        )
        for state in colors
        if state in shown_states
    ]
    fig.legend(
        handles=handles
        + [Line2D([], [], color="0.15", ls=":", lw=0.6, label="Crash candidate")]
        + state_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=1,
        frameon=False,
        labelspacing=0.2,
        handlelength=1.0,
    )
    paths = save_plot(fig, args.work / "figures/sawtooth_example")
    plt.close(fig)
    result = {
        "shot": shot,
        "split": "train",
        "window_s": [lo, hi],
        "core_channel": core,
        "outer_channels": outer,
        "trace_labels": labels,
        "shown_states": sorted(shown_states),
        "geometry": record["core_geometry"],
        **paths,
        "width_inches": 3.25,
        "minimum_font_points": 7,
        "png_dpi": 150,
        "generated_utc": now(),
        "source_sha256": {
            str(path): digest(path)
            for path in (
                args.work / "shots" / f"{shot}.json",
                args.work / "signals" / f"{shot}.npz",
            )
        },
        "selection": (
            "192148 was in the original random train gallery; selected "
            "for an illustrative train, not for same-shot RF metadata"
        ),
        "required_caption": (
            "Nominal rho is abs(R-axis)/(LCFS_outer_R-axis), not calibrated "
            "flux. The outer channels lie on the low-field side. Markers "
            "are unvalidated physics-rule candidates; shaded spans show "
            "the algorithmic five-state assessment."
        ),
    }
    save_json(args.output / "paper_example.json", result)


def legacy_profile_diagnosis(profile, geometry):
    """Explain the existing inversion gate, without changing its answer."""
    if heuristics.inversion_block(profile) is not None:
        return {"reason": "accepted", "dominant_drop": None}
    if np.isfinite(profile).sum() < 6:
        return {"reason": "insufficient_finite_channels", "dominant_drop": None}
    smoothed = heuristics.medfilt(np.where(np.isfinite(profile), profile, 0), 3)
    contrast = float(np.max(np.abs(smoothed)))
    if contrast <= 0 or not (smoothed < -0.25 * contrast).any():
        return {"reason": "no_significant_negative_block", "dominant_drop": None}
    peak = int(np.argmin(smoothed))
    block = next(
        (
            pair
            for pair in heuristics._runs(smoothed < -0.25 * contrast)
            if pair[0] <= peak < pair[1]
        ),
        None,
    )
    if block is None:
        return {"reason": "no_dominant_drop_block", "dominant_drop": None}
    excluded = set(
        geometry["harmonic_overlap_excluded_channels"]
        + geometry["terminal_channels_excluded"]
    )
    details = {
        "dominant_drop": list(block),
        "spatially_excluded_drop_channels": [c for c in range(*block) if c in excluded],
    }
    if block[1] - block[0] < 3:
        return {"reason": "dominant_drop_block_shorter_than_3ch", **details}
    if block[0] == 0 or block[1] == len(smoothed):
        return {"reason": "dominant_drop_touches_array_end", **details}
    gains = [
        pair
        for pair in heuristics._runs(smoothed > 0.3 * contrast)
        if pair[1] - pair[0] >= 3
    ]
    return {
        "reason": "no_adjacent_3ch_gain_block" if not gains else "gain_block_too_far",
        "gain_blocks": [list(pair) for pair in gains],
        **details,
    }


def prior_spatial_diagnostic(args, record, plt):
    """Show the original timing example while retaining its geometry limitation."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    shot = 190602
    prior = json.loads((args.previous / "shots" / f"{shot}.json").read_text())
    old = json.loads((args.previous / "old_rule" / f"{shot}.json").read_text())
    selected = prior["crashes"][len(prior["crashes"]) // 2]
    center = selected["time_s"]
    lo, hi = center - 0.075, center + 0.075
    core = int(record["core_geometry"]["central_channel"])
    attrs = selected["attrs"]
    auxiliary = list(range(attrs["rise_start"], attrs["rise_stop"]))
    # The corrected model cache masks these auxiliary channels; use the exact
    # native legacy envelope to expose the signal being diagnosed.
    with np.load(args.work / "legacy_audit" / f"{shot}.npz") as signal:
        t, y = signal["t"], signal["y"]
    near = (t >= lo) & (t <= hi)
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.1), sharex=True)
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.18, top=0.76, hspace=0.16)
    for axis, trace, color in zip(
        axes, (y[core], mean_trace(y[auxiliary])), ("#0072B2", "#D55E00"), strict=True
    ):
        axis.plot(t[near] * 1000, trace[near], color=color, lw=0.7)
        for crashes, markercolor, linestyle in (
            (prior["crashes"], "#009E73", ":"),
            (old["crashes"], "#CC79A7", "--"),
        ):
            for crash in crashes:
                if lo <= crash["time_s"] <= hi:
                    axis.axvline(
                        crash["time_s"] * 1000, color=markercolor, ls=linestyle, lw=0.7
                    )
        axis.set_ylabel("Te (keV)")
        axis.grid(axis="y", color="0.9", lw=0.3)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].set(xlabel="Time (ms)", xlim=(lo * 1000, hi * 1000))
    axes[-1].xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    axes[0].text(0.01, 0.97, f"DIII-D {shot}", transform=axes[0].transAxes, va="top")
    labels = [
        channel_label(record, [core], "Core"),
        channel_label(record, auxiliary, "Aux HFS") + " (excluded)",
    ]
    fig.legend(
        handles=[
            Line2D([], [], color="#0072B2", lw=0.7, label=labels[0]),
            Line2D([], [], color="#D55E00", lw=0.7, label=labels[1]),
            Line2D(
                [], [], color="#009E73", ls=":", lw=0.7, label="Prior unverified pick"
            ),
            Line2D([], [], color="#CC79A7", ls="--", lw=0.7, label="Legacy ECE pick"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
        ncol=1,
        handlelength=1.0,
        labelspacing=0.15,
    )
    fig.text(0.17, 0.025, "Profile unobservable: no LFS outer ECE", fontsize=7)
    paths = save_plot(fig, args.work / "figures/legacy_prior_geometry_190602")
    plt.close(fig)
    return {
        "shot": shot,
        "window_s": [lo, hi],
        "core_channel": core,
        "auxiliary_channels": auxiliary,
        "trace_labels": labels,
        "current_physics_picks": len(record["crashes"]),
        "current_observable_seconds": sum(b - a for a, b in record["observable_spans"]),
        "current_geometry": record["core_geometry"],
        "input": "Unmodified native ECE, 1ms bin-mean envelope used by the old rule",
        **paths,
        "required_caption": (
            "The original timing example uses prior spatially unverified "
            "candidates, not the corrected physics labels. Nominal RF/EFIT "
            "geometry places only ECE channels 0–2 outside the third-harmonic "
            "overlap guard and provides no LFS outer group. The auxiliary "
            "high-field-side channels are masked; the corrected rule abstains "
            "on the whole shot. Candidate lines are not independent truth."
        ),
    }


def legacy_comparison(args):
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    plt = style()
    cohort = pd.read_csv(REPO / "data/events/catalog/cohort.csv").set_index("shot")
    diagnostic_shots = args.diagnostic_shots or LEGACY_SHOTS
    records = {r["shot"]: r for r in records_at(args.work, diagnostic_shots)}
    rows = []
    for shot in diagnostic_shots:
        record = records[shot]
        old = json.loads((args.previous / "old_rule" / f"{shot}.json").read_text())
        a, b = points(old), points(record)
        observed_b = b[spans_at(b, record["observable_spans"])]
        present = [
            (span["start_s"], span["end_s"])
            for span in record["states"]
            if span["state"] == "present"
        ]
        scopes = {}
        for name, support in (
            ("observable", record["observable_spans"]),
            ("physics_present", present),
        ):
            pa, pb = a[spans_at(a, support)], b[spans_at(b, support)]
            offsets = nearest_offsets(pb, pa)
            scopes[name] = {
                "legacy_count": len(pa),
                "physics_count": len(pb),
                "signed_nearest_offset_ms_old_minus_physics": finite_summary(offsets),
                "signed_nearest_offsets_within_15ms": finite_summary(
                    offsets[np.abs(offsets) <= 15]
                ),
                "nearest_legacy_anywhere_in_observable_support_offset_ms": finite_summary(
                    nearest_offsets(pb, a[spans_at(a, record["observable_spans"])])
                ),
                "agreement": {
                    str(tolerance): {
                        "cells": event_cells(pa, pb, tolerance).tolist(),
                        "metrics": point_metrics(event_cells(pa, pb, tolerance)),
                    }
                    for tolerance in (2, 10, 12)
                },
            }
            scopes[name]["nearest_offset_histogram"] = {
                "edges_ms": np.arange(-25, 26, 1).tolist(),
                "counts": np.histogram(
                    offsets[np.isfinite(offsets)], bins=np.arange(-25, 26, 1)
                )[0].tolist(),
            }
        prior = json.loads((args.previous / "shots" / f"{shot}.json").read_text())
        prior_support = [
            (span["start_s"], span["end_s"])
            for span in prior["states"]
            if span["state"] == "present"
        ]
        prior_times = points(prior)
        prior_present = prior_times[spans_at(prior_times, prior_support)]
        old_prior_present = a[spans_at(a, prior_support)]
        prior_offsets = nearest_offsets(prior_present, a)
        prior_comparison = {
            "source": str(args.previous / "shots" / f"{shot}.json"),
            "prior_present_candidate_count": len(prior_present),
            "legacy_count_inside_prior_present_support": len(old_prior_present),
            "legacy_minus_prior_candidate_ms": finite_summary(prior_offsets),
            "offset_support": "Prior-present physics points; nearest legacy anywhere in native window",
            "legacy_minus_prior_candidate_ms_within_15ms": finite_summary(
                prior_offsets[np.abs(prior_offsets) <= 15]
            ),
            "prior_geometry_status": prior["core_geometry"],
            "claim": "Prior spatially unverified algorithmic candidates; not physical truth",
        }
        with np.load(args.work / "legacy_audit" / f"{shot}.npz") as stored:
            env, te = stored["y"], stored["t"]
            raw, retained = stored["raw_candidates"], stored["retained_candidates"]
        gates = Counter()
        for time in observed_b:
            if not len(raw) or np.min(np.abs(raw - time)) > 0.002:
                gates["not_legacy_raw_candidate_within_2ms"] += 1
            elif not len(retained) or np.min(np.abs(retained - time)) > 0.002:
                gates["legacy_raw_candidate_suppressed_by_10ms_greedy_spacing"] += 1
            elif not len(a) or np.min(np.abs(a - time)) > 0.002:
                gates["legacy_retained_candidate_rejected_by_inversion_test"] += 1
            else:
                gates["legacy_accepted_within_2ms"] += 1
        ablation = {
            "performed": False,
            "reason": "exploratory ablation uses train only",
        }
        if cohort.loc[shot, "split"] == "train":
            from sawtooth_benchmark import derivative_candidates

            with np.load(args.work / "signals" / f"{shot}.npz") as signal:
                center = int(signal["central_channel"])
                probe = {
                    "t": signal["t"],
                    "y": signal["y"],
                    "central_channel": signal["central_channel"],
                    "observable": np.isfinite(signal["y"][center]),
                }
                peaks, z = derivative_candidates(probe)
                derivative_times = signal["t"][peaks[z >= 20]]
            accepted_without_holdoff = []
            for time in raw:
                index = int(np.argmin(abs(te - time)))
                profile = heuristics._crash_step(
                    env, index, env_ms=1, gap_ms=2, span_ms=8
                )
                if heuristics.inversion_block(profile) is not None:
                    accepted_without_holdoff.append(float(time))
            no_holdoff = np.asarray(accepted_without_holdoff)
            short_profile_times = []
            for time in retained:
                index = int(np.argmin(abs(te - time)))
                profile = heuristics._crash_step(
                    env, index, env_ms=1, gap_ms=1, span_ms=2
                )
                if heuristics.inversion_block(profile) is not None:
                    short_profile_times.append(float(time))
            short_profile = np.asarray(short_profile_times)
            derivative_gates = Counter()
            profile_failures = Counter()
            blocking_offsets = []
            examples = []
            for time in derivative_times:
                offset_raw = nearest_offsets([time], raw)[0]
                offset_kept = nearest_offsets([time], retained)[0]
                offset_old = nearest_offsets([time], a)[0]
                raw_time = time + offset_raw / 1000
                profile_index = int(np.argmin(abs(te - raw_time)))
                profile = heuristics._crash_step(
                    env, profile_index, env_ms=1, gap_ms=2, span_ms=8
                )
                diagnosis = legacy_profile_diagnosis(profile, record["core_geometry"])
                profile_failures[diagnosis["reason"]] += 1
                if not np.isfinite(offset_raw) or abs(offset_raw) > 2:
                    reason = "no_raw_multichannel_2percent_edge"
                elif not np.isfinite(offset_kept) or abs(offset_kept) > 2:
                    reason = "raw_edge_suppressed_by_10ms_holdoff"
                    raw_index = int(np.argmin(abs(te - (time + offset_raw / 1000))))
                    retained_indices = np.searchsorted(te, retained)
                    if not np.any(abs(retained_indices - raw_index) < 10):
                        raise ValueError(
                            "suppression explanation lacks blocking candidate"
                        )
                    blocking_offsets.append(float(offset_kept))
                elif not np.isfinite(offset_old) or abs(offset_old) > 2:
                    reason = "retained_edge_failed_2to8ms_inversion_profile"
                else:
                    reason = "accepted_legacy_edge"
                derivative_gates[reason] += 1
                if len(examples) < 12 and reason != "accepted_legacy_edge":
                    examples.append(
                        {
                            "derivative_peak_s": float(time),
                            "nearest_raw_offset_ms": float(offset_raw),
                            "nearest_retained_offset_ms": float(offset_kept),
                            "nearest_accepted_offset_ms": float(offset_old),
                            "reason": reason,
                            "profile_diagnosis": diagnosis,
                        }
                    )
            ablation = {
                "performed": True,
                "split": "train",
                "derivative_source": "sawtooth_benchmark.derivative_candidates",
                "derivative_z_fixed": 20,
                "derivative_support": (
                    "Finite central-channel ECE only, independent of the new "
                    "rule's multichannel/profile observability mask"
                ),
                "derivative_count": len(derivative_times),
                "legacy_count": len(a),
                "physics_matches_derivative_2ms": int(
                    event_cells(derivative_times, observed_b, 2)[0]
                ),
                "physics_minus_derivative_nearest_offset_ms": finite_summary(
                    nearest_offsets(derivative_times, observed_b)
                ),
                "no_holdoff_inversion_accepted_count": len(no_holdoff),
                "original_matches_derivative_2ms": int(
                    event_cells(derivative_times, a, 2)[0]
                ),
                "no_holdoff_matches_derivative_2ms": int(
                    event_cells(derivative_times, no_holdoff, 2)[0]
                ),
                "short_1to2ms_profile_same_holdoff_count": len(short_profile),
                "short_1to2ms_profile_same_holdoff_matches_derivative_2ms": int(
                    event_cells(derivative_times, short_profile, 2)[0]
                ),
                "additional_derivative_edges_recovered_without_holdoff": int(
                    event_cells(derivative_times, no_holdoff, 2)[0]
                    - event_cells(derivative_times, a, 2)[0]
                ),
                "old_minus_derivative_nearest_offset_ms": finite_summary(
                    nearest_offsets(derivative_times, a)
                ),
                "blocking_candidate_minus_derivative_offset_ms": finite_summary(
                    blocking_offsets
                ),
                "derivative_gate_diagnosis": dict(derivative_gates),
                "inversion_profile_failure_at_raw_edge": dict(profile_failures),
                "examples": examples,
                "interpretation": (
                    "A fixed 20-sigma central derivative is an independent "
                    "diagnostic edge proxy, not crash truth. Disable only the "
                    "legacy 10ms candidate holdoff and keep its 2to8ms inversion "
                    "test. A separate profile-duration ablation uses the same "
                    "retained candidates and thresholds, only 1to2ms rather "
                    "than 2to8ms windows. Raw dense picks can duplicate a single edge; compare "
                    "one-to-one matches, not counts. No rule changes or tuning."
                ),
            }
        # Direct native 1ms envelope profiles at true-edge and old nearby picks.
        example_profiles = []
        for time in observed_b[:: max(1, len(observed_b) // 12)]:
            index = int(np.argmin(abs(te - time)))
            steps = heuristics._crash_step(
                env, index, env_ms=1.0, gap_ms=2.0, span_ms=8.0
            )
            example_profiles.append(
                {
                    "physics_time_s": float(time),
                    "legacy_inversion_block_at_physics_time": heuristics.inversion_block(
                        steps
                    ),
                    "old_minus_physics_nearest_ms": float(nearest_offsets([time], a)[0])
                    if len(a)
                    else None,
                }
            )
        selected = next(
            (
                row
                for row in record["crashes"]
                if row["attrs"].get("state") == "present"
            ),
            record["crashes"][0] if record["crashes"] else None,
        )
        trace_evidence = {}
        if "core_geometry" in record:
            if selected is not None:
                core, outer = trace_channels(record, selected)
            else:
                core = int(record["core_geometry"]["central_channel"])
                outer = record["core_geometry"]["outer_channels"]
            with np.load(args.work / "signals" / f"{shot}.npz") as signal:
                t, y = signal["t"], signal["y"]
            trace_sets = [("physics", observed_b), ("legacy", a)]
            if ablation["performed"]:
                trace_sets.append(("central_derivative", derivative_times))
            for name, times in trace_sets:
                core_drop, outer_rise = [], []
                for time in times:
                    before = (t >= time - 0.002) & (t < time - 0.0005)
                    after = (t >= time + 0.0005) & (t < time + 0.002)
                    if not before.any() or not after.any():
                        continue
                    pre, post = (
                        median_rows(y[:, before]),
                        median_rows(y[:, after]),
                    )
                    if np.isfinite(pre[core]) and pre[core] > 0:
                        core_drop.append((pre[core] - post[core]) / pre[core])
                    if outer:
                        pre_outer, post_outer = (
                            np.nanmean(pre[outer]),
                            np.nanmean(post[outer]),
                        )
                        if np.isfinite(pre_outer) and pre_outer > 0:
                            outer_rise.append((post_outer - pre_outer) / pre_outer)
                trace_evidence[name] = {
                    "core_channel": core,
                    "outer_channels": outer,
                    "core_fractional_drop": finite_summary(core_drop),
                    "outer_fractional_rise": finite_summary(outer_rise),
                    "physical_validation": False,
                }
        rows.append(
            {
                "shot": shot,
                "scopes": scopes,
                "legacy_gate_diagnosis": dict(gates),
                "example_profiles": example_profiles,
                "trace_evidence": trace_evidence,
                "train_only_holdoff_ablation": ablation,
                "split": cohort.loc[shot, "split"],
                "new_rule_observable_seconds": sum(
                    hi - lo for lo, hi in record["observable_spans"]
                ),
                "geometry_observability": record["core_geometry"]["outer_status"],
                "prior_unverified_candidate_comparison": prior_comparison,
            }
        )
        if shot == 192090:
            store = pd.read_parquet(Paths.from_env().events_file(shot))
            current = store.loc[store.source == "ece_sawtooth"]
            table = []
            for diagnostic in ("all", "ece", "sxr"):
                observed = (
                    current
                    if diagnostic == "all"
                    else current.loc[current.diag == diagnostic]
                )
                times = observed.t0_s.to_numpy()
                scope = scopes["physics_present"]
                p = b[spans_at(b, present)]
                q = times[spans_at(times, present)]
                table.append(
                    {
                        "diagnostic": diagnostic,
                        "current_catalog_count": len(times),
                        "physics_present_catalog_count": len(q),
                        "physics_present_physics_count": scope["physics_count"],
                        "present_cells_2ms": event_cells(p, q, 2).tolist(),
                        "catalog_minus_physics_nearest_ms": finite_summary(
                            nearest_offsets(p, q)
                        ),
                    }
                )
            rows[-1]["current_production_v3_comparison"] = {
                "source": str(Paths.from_env().events_file(shot)),
                "by_diagnostic": table,
                "interpretation": (
                    "This store has explicit detector v3 (ECE and SXR). Its "
                    "point times are compared directly, separately from the "
                    "legacy v2 cache. No catalog defect follows from legacy "
                    "disagreement alone; both are unvalidated detectors."
                ),
            }
    plots = []
    for shot in (192090, 191384, 201948, 203349):
        if shot not in records:
            continue
        record = records[shot]
        old = json.loads((args.previous / "old_rule" / f"{shot}.json").read_text())
        selected_rows = [
            row for row in record["crashes"] if row["attrs"]["state"] == "present"
        ] or record["crashes"]
        if not selected_rows:
            continue
        selected = selected_rows[len(selected_rows) // 2]
        center = selected["time_s"]
        lo, hi = center - 0.075, center + 0.075
        core, outer = trace_channels(record, selected)
        with np.load(args.work / "signals" / f"{shot}.npz") as signal:
            t, y = signal["t"], signal["y"]
        near = (t >= lo) & (t <= hi)
        fig, axes = plt.subplots(2, 1, figsize=(3.5, 2.7), sharex=True)
        fig.subplots_adjust(left=0.17, right=0.98, bottom=0.16, top=0.77, hspace=0.14)
        for axis, trace, color in zip(
            axes,
            (y[core], mean_trace(y[outer])),
            ("#0072B2", "#D55E00"),
            strict=True,
        ):
            axis.plot(t[near] * 1000, trace[near], color=color, lw=0.7)
            for crashes, markercolor, linestyle in (
                (record["crashes"], "#009E73", ":"),
                (old["crashes"], "#CC79A7", "--"),
            ):
                for crash in crashes:
                    if lo <= crash["time_s"] <= hi:
                        axis.axvline(
                            crash["time_s"] * 1000,
                            color=markercolor,
                            ls=linestyle,
                            lw=0.8,
                        )
            axis.set_ylabel("Te (keV)")
            axis.grid(axis="y", color="0.9", lw=0.3)
            axis.spines[["top", "right"]].set_visible(False)
        axes[-1].set(xlabel="Time (ms)", xlim=(lo * 1000, hi * 1000))
        axes[-1].xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        axes[0].text(
            0.01, 0.97, f"DIII-D {shot}", transform=axes[0].transAxes, va="top"
        )
        labels = [
            channel_label(record, [core], "Core"),
            channel_label(record, outer, "Outer LFS"),
        ]
        handles = [
            Line2D([], [], color="#0072B2", lw=0.7, label=labels[0]),
            Line2D([], [], color="#D55E00", lw=0.7, label=labels[1]),
        ]
        for crashes, color, line_style, label in (
            (record["crashes"], "#009E73", ":", "Physics pick"),
            (old["crashes"], "#CC79A7", "--", "Legacy ECE pick"),
        ):
            if any(lo <= crash["time_s"] <= hi for crash in crashes):
                handles.append(
                    Line2D([], [], color=color, ls=line_style, lw=0.8, label=label)
                )
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=2,
            frameon=False,
            handlelength=1.0,
            labelspacing=0.25,
            columnspacing=0.8,
        )
        paths = save_plot(fig, args.work / f"figures/legacy_disagreement_{shot}")
        plt.close(fig)
        plots.append(
            {
                "shot": shot,
                "window_s": [lo, hi],
                "trace_labels": labels,
                "core_channel": core,
                "outer_channels": outer,
                **paths,
            }
        )
    if 190602 in records:
        plots.append(prior_spatial_diagnostic(args, records[190602], plt))
    summary = {
        "by_shot": rows,
        "figures": plots,
        "generated_utc": now(),
        "source_sha256": {
            str(path): digest(path)
            for shot in diagnostic_shots
            for path in (
                args.work / "shots" / f"{shot}.json",
                args.work / "signals" / f"{shot}.npz",
                args.previous / "old_rule" / f"{shot}.json",
                args.work / "legacy_audit" / f"{shot}.npz",
            )
        },
        "offset_sign": "old time minus physics time; negative means old pick precedes",
        "nearest_offset_warning": (
            "Nearest neighbours can be reused and are descriptive. One-to-one "
            "event cells are reported separately; neither rule is ground truth."
        ),
        "legacy_reader_audit": str(args.output / "legacy_reader_audit.json"),
        "interpretation": (
            "Use the exact-cache native reader audit to exclude a clock shift. "
            "Raw-candidate gate counts localize mismatch to missing edges, "
            "10ms greedy suppression, or the inversion filter; profile windows "
            "use +/-2 to8ms and can reject faster true redistribution. "
            "Legacy picks and current catalog detector are different methods."
        ),
        "validation_status": "algorithm disagreement; no blind crash truth",
        "required_caption": (
            "Native-antialiased ECE at a central and outer low-field-side "
            "channel group; geometry is nominal geometric rho. Dotted green "
            "lines are new physics picks, dashed purple are the retained "
            "omnimode-style legacy ECE rule. Neither set is independent truth."
        ),
    }
    table_rows = []
    for row in rows:
        fresh = row["scopes"]["physics_present"]
        prior = row["prior_unverified_candidate_comparison"]
        ablation = row["train_only_holdoff_ablation"]
        offset = fresh["signed_nearest_offsets_within_15ms"]
        table_rows.append(
            {
                "shot": row["shot"],
                "split": row["split"],
                "prior_unverified_present_points": prior[
                    "prior_present_candidate_count"
                ],
                "legacy_minus_prior_ms": prior["legacy_minus_prior_candidate_ms"][
                    "median"
                ],
                "fresh_observable_seconds": row["new_rule_observable_seconds"],
                "fresh_present_points": fresh["physics_count"],
                "fresh_old_minus_physics_ms_within_15ms": offset["median"],
                "fresh_offset_pairs_within_15ms": offset["n"],
                "fresh_present_2ms_matches": fresh["agreement"]["2"]["cells"][0],
                "fixed_derivative_edge_proxy_n": ablation.get("derivative_count"),
                "legacy_matches_derivative_2ms": ablation.get(
                    "original_matches_derivative_2ms"
                ),
                "no_holdoff_matches_derivative_2ms": ablation.get(
                    "no_holdoff_matches_derivative_2ms"
                ),
                "old_minus_derivative_ms": ablation.get(
                    "old_minus_derivative_nearest_offset_ms", {}
                ).get("median"),
                "inversion_failures": ablation.get(
                    "inversion_profile_failure_at_raw_edge"
                ),
            }
        )
    summary["timing_table"] = table_rows
    pd.DataFrame(table_rows).to_csv(
        args.output / "legacy_timing_by_shot.csv", index=False
    )
    save_json(args.output / "legacy_disagreement.json", summary)
    print({row["shot"]: row["legacy_gate_diagnosis"] for row in rows}, flush=True)


def figures(args):
    paper_example(args)
    legacy_comparison(args)


def confirm_inspection(args):
    """Record viewer inspection only after every generated PNG was viewed."""
    ledger = []
    for name in ("paper_example", "legacy_disagreement"):
        path = args.output / f"{name}.json"
        record = json.loads(path.read_text())
        if any(
            digest(source) != value for source, value in record["source_sha256"].items()
        ):
            raise ValueError("figure inputs changed; regenerate and inspect again")
        figures = record.get("figures", [record])
        for figure in figures:
            figure["png_inspected"] = True
            figure["png_sha256"] = digest(figure["png"])
            ledger.append(
                {
                    "group": name,
                    "png": figure["png"],
                    "sha256": figure["png_sha256"],
                    "source_record": str(path),
                }
            )
        record["png_inspection"] = {
            "complete": True,
            "method": "Every individual PNG viewed with the agent image viewer",
            "frozen_utc": now(),
        }
        save_json(path, record)
    gallery_path = args.output / "gallery.json"
    if gallery_path.exists():
        gallery = json.loads(gallery_path.read_text())
        inspection = gallery.get("png_inspection", {})
        if isinstance(inspection, dict) and inspection.get("complete"):
            for png, recorded_hash in inspection["sha256"].items():
                if digest(png) != recorded_hash:
                    raise ValueError("gallery changed after visual inspection")
                ledger.append(
                    {
                        "group": "gallery",
                        "png": png,
                        "sha256": recorded_hash,
                        "source_record": str(gallery_path),
                    }
                )
            save_json(
                args.output / "figure_inspection.json",
                {
                    "complete": True,
                    "frozen_utc": now(),
                    "png_count": len(ledger),
                    "method": "Every individual PNG viewed with the agent image viewer",
                    "figures": ledger,
                },
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "queue-base",
            "queue",
            "reader-audit",
            "paper",
            "legacy",
            "figures",
            "confirm-inspection",
        ],
    )
    parser.add_argument("--work", type=Path, default=WORK4)
    parser.add_argument("--previous", type=Path, default=PREVIOUS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--example-shot", type=int, default=192148)
    parser.add_argument("--diagnostic-shots", nargs="+", type=int, default=[])
    args = parser.parse_args()
    {
        "queue-base": queue_base,
        "queue": queue,
        "reader-audit": reader_audit,
        "paper": paper_example,
        "legacy": legacy_comparison,
        "figures": figures,
        "confirm-inspection": confirm_inspection,
    }[args.stage](args)


if __name__ == "__main__":
    main()
