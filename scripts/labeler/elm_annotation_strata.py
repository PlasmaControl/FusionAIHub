#!/usr/bin/env python
"""Export annotation strata, coverage corrections and checkpoint-selection audit.

First run ``elm_ours_evaluate.py --run cv2``. This script exports its reproducible
stratified metrics and audits saved fold histories without changing checkpoints,
thresholds or training. Large predictions/checkpoints remain under LABELER_ROOT.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from labeler.config import Paths, git_sha, sha256_of
from labeler.elm import compare, dsm, labels, methods, score, swap, train

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/labeler/elm/ours"
RUNS = ("cv2", "cv2_seed20261004", "cv2_seed20261005", "cv2_seed20261006")


def historical_counts(found, cover, review):
    """Reproduce the former unclipped touches with the original eligibility rule."""
    c0, c1 = labels.merge_intervals(cover.t_start_ms, cover.t_end_ms)
    eligible = []
    for row in review.itertuples():
        analysed = np.clip(
            np.minimum(c1, row.t_end) - np.maximum(c0, row.t_start), 0, None
        ).sum()
        eligible.append(
            row.kind in labels.SCORED_KINDS
            and row.t_end > row.t_start
            and analysed >= 0.5 * (row.t_end - row.t_start)
        )
    selected = review.loc[eligible]
    if selected.empty:
        return dict.fromkeys(methods.score.SPAN_KEYS, 0)
    # Once eligibility is fixed, full-span coverage reproduces the old touch test.
    full = methods.cover_frame([selected.t_start.min()], [selected.t_end.max()])
    return methods.span_counts(found, full, selected)


def coverage_audit(paths, run):
    data = train.load(paths)
    sets = compare.load_sets(paths, data)
    elmo, clock = compare.load_detected(paths)
    oof = methods.Oof(paths.root / "round4/elm/cv" / run)
    detected = {}
    for shot in data:
        starts, stops = labels.runs_of(oof.trace(shot)[0], oof.threshold[shot])
        detected[shot] = {
            "elm-ours": methods.span_frame(starts, stops),
            "elm-clock": clock.get(shot, methods.span_frame([], [])),
            "elm-elmo": elmo.get(shot, methods.span_frame([], [])),
        }
    output = {}
    for panel, sdef in sets.items():
        for common in (False, True):
            changes = []
            for shot in sdef.shots:
                cover = sdef.cover[shot]
                if common:
                    cache = paths.root / "round4/elm/dsm" / f"{shot}.npz"
                    with np.load(cache) as saved:
                        cover = dsm.usable_cover(saved["usable"], cover)
                for method, found in detected[shot].items():
                    if method == "elm-elmo" and not sdef.has_elmo:
                        continue
                    before = historical_counts(found, cover, data[shot].spans)
                    after = methods.span_counts(found, cover, data[shot].spans)
                    changed = {key for key in before if before[key] != after[key]}
                    if changed:
                        changes.append(
                            {
                                "shot": shot,
                                "method": method,
                                "before": {key: before[key] for key in sorted(changed)},
                                "after": {key: after[key] for key in sorted(changed)},
                            }
                        )
            output[("common_" if common else "") + panel] = {
                "changes": changes,
                "affected_shots": sorted({row["shot"] for row in changes}),
            }
    bin_coverage = {}
    for panel, sdef in sets.items():
        bin_coverage[panel] = {
            str(shot): len(sdef.bins[shot].truth) for shot in sdef.shots
        }
    return output, bin_coverage


def checkpoint_audit(paths, runs):
    records, thresholds, selected_epochs = [], [], []
    training_seconds = 0.0
    for run in runs:
        directory = paths.root / "round4/elm/cv" / run
        source = json.loads((directory / "run.json").read_text())
        warmup = source["config"]["epochs"] * 0.15
        for fold in range(len(source["folds"])):
            path = directory / f"fold{fold}/fold.json"
            record = json.loads(path.read_text())
            history = record["history"]
            ap = np.array([row["val_auprc"] for row in history])
            selected = record["best"]["epoch"]
            # A numeric sensitivity audit, not a retrospectively selected model.
            candidates = []
            for index in range(2, len(history)):
                window = history[index - 2 : index + 1]
                if window[0]["epoch"] + 1 <= warmup:
                    continue
                values = ap[index - 2 : index + 1]
                if np.isfinite(values).all():
                    candidates.append((float(values.mean()), index))
            smooth, index = max(candidates, key=lambda value: value[0])
            neighbours = ap[max(0, selected - 1) : selected + 2]
            surrounding = np.delete(neighbours, selected - max(0, selected - 1))
            training_seconds += sum(row["seconds"] for row in history)
            thresholds.append(record["threshold"])
            selected_epochs.append(selected)
            records.append(
                {
                    "run": run,
                    "fold": fold,
                    "source": str(path),
                    "source_sha256": sha256_of(path),
                    "selected_epoch": selected,
                    "selected_in_onecycle_warmup": selected + 1 <= warmup,
                    "selected_auprc": float(ap[selected]),
                    "neighbour_mean_auprc": float(surrounding.mean()),
                    "selected_minus_neighbour_mean": float(
                        ap[selected] - surrounding.mean()
                    ),
                    "post_warmup_trailing3_candidate_epoch": history[index]["epoch"],
                    "post_warmup_trailing3_mean_auprc": smooth,
                    "candidate_epoch_auprc": float(ap[index]),
                    "candidate_epoch_threshold": history[index]["val_threshold"],
                    "recorded_threshold": record["threshold"],
                    "best_state_only_saved": (
                        sorted(p.name for p in path.parent.glob("*.pt")) == ["model.pt"]
                    ),
                }
            )
    return {
        "protocol": "Original selection maximizes single-epoch inner-validation "
        "AUPRC. Audit compares the maximum trailing three-epoch mean after all "
        "three epochs complete the 15% OneCycle warmup. Zero-based epochs; no "
        "held-out scores are used to choose epochs or thresholds.",
        "folds": records,
        "threshold_min": min(thresholds),
        "threshold_max": max(thresholds),
        "selected_epochs": selected_epochs,
        "selected_during_warmup": sum(
            row["selected_in_onecycle_warmup"] for row in records
        ),
        "smoothed_candidate_differs": sum(
            row["selected_epoch"] != row["post_warmup_trailing3_candidate_epoch"]
            for row in records
        ),
        "observed_training_seconds_all_folds": training_seconds,
        "estimated_serial_retrain_hours": training_seconds / 3600,
        "smoothed_checkpoint_rescore_available": False,
        "limitation": "Only the selected best state was saved for each fold. A "
        "smoothed-checkpoint comparison needs all 20 folds rerun on the fixed "
        "partitions; the estimate excludes queue time and artifact serialization. "
        "This audit changes no reported operating point.",
    }


def reference_sensitivities(paths, run):
    """Fixed OOF predictions under review merging, plus clock boundary identity."""
    data = train.load(paths)
    sets = compare.load_sets(paths, data)
    elmo, clock = compare.load_detected(paths)
    sweep = compare.load_elmo_sweep(paths)
    oof = methods.Oof(paths.root / "round4/elm/cv" / run)
    crowd_records = []
    for shot, datum in data.items():
        found = clock.get(shot, methods.span_frame([], []))
        for span in datum.spans[datum.spans.kind == "crowd"].itertuples():
            starts = np.abs(found.t_start_ms.to_numpy(float) - span.t_start)
            ends = np.abs(found.t_end_ms.to_numpy(float) - span.t_end)
            near_start = bool(np.any(starts <= 1.0))
            near_end = bool(np.any(ends <= 1.0))
            crowd_records.append(
                {
                    "shot": shot,
                    "t_start_ms": float(span.t_start),
                    "t_end_ms": float(span.t_end),
                    "start_within_1ms": near_start,
                    "end_within_1ms": near_end,
                    "both_within_1ms": near_start and near_end,
                    "same_clock_span_both_within_1ms": bool(
                        np.any((starts <= 1.0) & (ends <= 1.0))
                    ),
                }
            )
    n_spans = len(crowd_records)
    identity = {
        "definition": "Category-1 crowd review spans, inclusive 1 ms proximity "
        "to original elm-clock present-span starts/ends; all 119 review shots. "
        "The clock seeded the review, so this is dependence, not validation.",
        "clock_source": str(paths.root / compare.CLOCK_CSV),
        "clock_source_sha256": sha256_of(paths.root / compare.CLOCK_CSV),
        "crowd_spans": n_spans,
        "counts": {
            key: sum(row[key] for row in crowd_records)
            for key in (
                "start_within_1ms",
                "end_within_1ms",
                "both_within_1ms",
                "same_clock_span_both_within_1ms",
            )
        },
        "per_span": crowd_records,
    }
    identity["shares"] = {
        key: value / n_spans for key, value in identity["counts"].items()
    }
    sensitivity = {
        "definition": "Fixed interior-bin universe, OOF predictions and "
        "review-tuned thresholds. Merge non-crowd review spans across fully "
        "reviewed absent gaps <=tau (0, 100, 200, 300 ms); crowd, uncertain, "
        "not-observable and unlabelled time are barriers. At least 25 ms merged "
        "occupancy makes an originally absent bin present. No retraining or "
        "threshold selection; event counts are not inferred from crowds.",
        "sets": {},
    }
    for panel, sdef in sets.items():
        parts = {"elm-ours": [], "elm-clock": []}
        if sdef.has_elmo:
            parts["elm-elmo"] = []
        for shot in sdef.shots:
            datum, bins, cover = data[shot], sdef.bins[shot], sdef.cover[shot]
            parts["elm-ours"].append(
                methods.trace_part(
                    datum.spans,
                    shot,
                    bins,
                    cover,
                    oof.trace(shot)[0],
                    oof.threshold[shot],
                )
            )
            parts["elm-clock"].append(
                methods.span_part(
                    datum.spans,
                    shot,
                    bins,
                    cover,
                    clock.get(shot, methods.span_frame([], [])),
                )
            )
            if sdef.has_elmo:
                part = methods.span_part(
                    datum.spans,
                    shot,
                    bins,
                    cover,
                    elmo.get(shot, methods.span_frame([], [])),
                )
                part.score = swap.sweep_bin_scores(sweep[sweep.shot == shot], bins)
                parts["elm-elmo"].append(part)
        boot = score.draws(len(sdef.shots))
        rows = {}
        for gap in (0, *swap.OCCUPANCY_GAPS_MS):
            targets = {
                shot: swap.review_merged_truth(data[shot].spans, sdef.bins[shot], gap)
                for shot in sdef.shots
            }
            rows[f"gap_{gap}ms"] = {
                "gap_ms": gap,
                "positive_bins": sum(int(t.sum()) for t in targets.values()),
                "added_positive_bins": sum(
                    int(((targets[shot] == 1) & (sdef.bins[shot].truth == 0)).sum())
                    for shot in sdef.shots
                ),
                "methods": {
                    name: score.summarise(
                        [swap.retruth(part, targets[part.shot]) for part in values],
                        boot,
                    )
                    for name, values in parts.items()
                },
            }
        sensitivity["sets"][panel] = {
            "shots": sdef.shots,
            "n_shots": len(sdef.shots),
            "bins": sum(len(sdef.bins[shot].truth) for shot in sdef.shots),
            "by_gap": rows,
        }
    return identity, sensitivity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--evaluation", type=Path, default=OUT / "evaluation.json")
    parser.add_argument("--run", default="cv2")
    parser.add_argument("--runs", nargs="+", default=RUNS)
    parser.add_argument("--out", type=Path, default=OUT / "annotation_strata.json")
    args = parser.parse_args(argv)
    evaluation = json.loads(args.evaluation.read_text())
    paths = Paths.from_env()
    clipping, bin_coverage = coverage_audit(paths, args.run)
    identity, merging = reference_sensitivities(paths, args.run)
    result = {
        "git": git_sha(),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_script": str(Path(__file__).relative_to(REPO)),
        "source_script_sha256": sha256_of(Path(__file__)),
        "source_evaluation": str(args.evaluation),
        "source_evaluation_sha256": sha256_of(args.evaluation),
        "run": args.run,
        "annotation_mode_definition": evaluation["annotation_modes"],
        "ci_method": "95% percentile bootstrap of physical shots within each "
        "annotation group, 1000 replicates, seed 20261003; fixed OOF thresholds.",
        "sets": {
            name: {
                "shots": panel["shots"],
                "n_shots": panel["n_shots"],
                "bins": panel["bins"],
                "per_kind": panel["per_kind"],
                "annotation_modes": panel["annotation_modes"],
            }
            for name, panel in evaluation["sets"].items()
        },
        "coverage_clipping_audit": clipping,
        "checkpoint_selection_audit": checkpoint_audit(paths, args.runs),
        "clock_boundary_identity": identity,
        "review_non_crowd_merge_sensitivity": merging,
    }
    for panel, values in result["sets"].items():
        for group in values["annotation_modes"].values():
            group["shots_with_scored_bins"] = [
                shot for shot in group["shots"] if bin_coverage[panel][str(shot)] > 0
            ]
            group["n_shots_with_scored_bins"] = len(group["shots_with_scored_bins"])
            group["shots_without_scored_bins"] = [
                shot for shot in group["shots"] if bin_coverage[panel][str(shot)] == 0
            ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(args.out)
    for panel, values in result["sets"].items():
        print(
            panel,
            {key: row["n_shots"] for key, row in values["annotation_modes"].items()},
        )
    print("coverage", result["coverage_clipping_audit"])
    print(
        "retrain estimate hours",
        result["checkpoint_selection_audit"]["estimated_serial_retrain_hours"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
