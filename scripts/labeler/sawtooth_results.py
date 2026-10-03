"""Render reviewable sawtooth result tables from evaluation JSON records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sawtooth_physics import REPO, WORK, records_at, save_json


def read(output, name):
    return json.loads((output / name).read_text())


def number(value, digits=3):
    return "undefined" if value is None else f"{value:.{digits}f}"


def scored(value, interval=None):
    result = number(value)
    if interval is not None:
        result += f" [{interval[0]:.3f}, {interval[1]:.3f}]"
    return result


def source(path, field=None):
    suffix = f" → `{field}`" if field else ""
    return f"Source: `{path}`{suffix}."


def expert_rows(block):
    rows = block.get("by_shot", block.get("per_shot", []))
    if isinstance(rows, dict):
        return [dict(value, shot=int(shot)) for shot, value in rows.items()]
    return rows


def presence(row):
    return row.get("presence", row.get("metrics", {}))


def render_experts(lines, validation, validation_path):
    lines += [
        "",
        "## Old rule and physics labels against expert spans",
        "",
        (
            "Scores are shown separately for every reviewed shot. Primary metrics "
            "retrieve definite-present labels on all known expert bins with valid "
            "core ECE. No bootstrap is used for these shots."
        ),
        "",
        (
            "| Shot | Assessed / observable known bins | Assessment fraction | "
            "Uncertain / observable positive bins |"
        ),
        "|---|---:|---:|---:|",
    ]
    rows = expert_rows(validation["expert"])
    for row in rows:
        lines.append(
            f"| {row['shot']} | {row['assessed_bins']} / "
            f"{row['observable_known_bins']} | {number(row['assessment_fraction'])} | "
            f"{row['expert_positive_uncertain_bins']} / "
            f"{row['expert_positive_observable_bins']} |"
        )
    lines += [
        "",
        (
            "| Rule | Shot | Observable precision | Observable recall | "
            "Observable F1 | Observable span F1 | Conditional F1 | "
            "Conditional span F1 |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        for name, key in (("Physics", "new"), ("ece_sawtooth", "old")):
            values = row[key]
            metrics = values.get("conditional_assessed_presence", {})
            span = values.get("conditional_assessed_span_matching", {})
            span_metrics = span.get("metrics", span)
            observable = presence(values)
            observable_span = values["span_matching"]["metrics"]
            lines.append(
                f"| {name} | {row['shot']} | "
                + " | ".join(
                    number(observable.get(key)) for key in ("precision", "recall", "f1")
                )
                + f" | {number(observable_span.get('f1'))} | "
                + f"{number(metrics.get('f1'))} | "
                + f"{number(span_metrics.get('f1'))} |"
            )
    lines += [
        "",
        source(
            validation_path,
            "expert.by_shot[0:3].new; expert.by_shot[0:3].old (shot-ordered)",
        ),
        "",
        (
            "Observable scores use the same expert-known core-ECE denominator for "
            "both rules and both models. Uncertain positive bins count as unresolved "
            "misses for definite-present recall/F1; they are never exported as absent. "
            "Conditional scores exclude the same uncertain support for both rules. "
            "Coverage and uncertain-positive counts expose the cost of abstention. "
            "Undefined "
            "precision means that the rule made no positive calls."
        ),
        "",
        (
            "Expert annotations provide spans rather than point crash times. "
            "Picks inside "
            "positive spans are a support check; expert crash precision, recall and F1 "
            "remain unavailable. Calibrated radial localization and independent crash "
            "annotations remain necessary before promoting these labels to "
            "ground truth."
        ),
    ]
    ious = sorted(
        {
            row[key]["span_matching"]["minimum_iou"]
            for row in rows
            for key in ("old", "new")
        }
    )
    lines += [
        "",
        "Span matching preserves original annotation identities and uses one-to-one "
        "IoU inside the stated assessment mask. Its recorded IoU threshold is "
        + ", ".join(number(value) for value in ious)
        + "; spans with no observable support are excluded.",
    ]


def render_agreement(lines, validation, validation_path):
    agreement = validation.get("legacy_agreement", validation.get("old_agreement", {}))
    if not agreement:
        return
    count = validation.get("agreement_shot_count", agreement.get("shots"))
    if isinstance(count, list):
        count = len(count)
    lines += [
        "",
        "## Agreement with the old detector",
        "",
        (
            "The read-only `heuristics.sawtooth_events` rule and the physics "
            "labels both "
            f"ran on {count} shots. Agreement is measured within their common assessed "
            "core-ECE support; it measures detector consistency rather than accuracy."
        ),
        "",
        "| Crash metric, ±2 ms | Value [95% shot-bootstrap CI] |",
        "|---|---:|",
    ]
    metrics, intervals = agreement.get("crash", {}), agreement.get("ci95", {})
    for key in ("precision", "recall", "f1"):
        lines.append(
            f"| {key} | {scored(metrics.get(key), intervals.get('crash_' + key))} |"
        )
    lines += ["", source(validation_path, "legacy_agreement")]


def render_antialias(lines, record, path):
    lines += [
        "",
        "## Native-rate antialias filtering check",
        "",
        (
            "The detector and all thresholds are identical in both preparations. Only "
            "the ECE reader changes; these train shots did not select model thresholds."
        ),
        "",
        (
            "| Shot | Stride-first candidates | Native-filter candidates | "
            "Stride-first crashes | Native-filter crashes | Median timing change (ms) |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in record["by_shot"]:
        lines.append(
            f"| {row['shot']} | {row['old_candidates']} | "
            f"{row['native_filtered_candidates']} | {row['old_crashes']} | "
            f"{row['native_filtered_crashes']} | "
            f"{number(row['median_absolute_timing_change_ms'], 6)} |"
        )
    lines += [
        "",
        (
            "Timing change is measured only for one-to-one matched points; undefined "
            "means that no pair met the recorded tolerance. This is preprocessing "
            "sensitivity rather than a ground-truth timing score."
        ),
        "",
        source(path, "by_shot"),
    ]


def render_models(lines, benchmark, benchmark_path):
    lines += [
        "",
        "## Models on out-of-fold training-cohort shots",
        "",
        (
            "Each whole shot belongs to one outer fold. Inner selection shots "
            "from that "
            "fold's training portion select the checkpoint and operating "
            "thresholds. Fixed validation, expert and blind test shots never select a "
            "threshold or checkpoint. Training uses the CUDA environment and stops on "
            "inner selection loss patience. Intervals use 1,000 shot-bootstrap "
            "replicates."
        ),
        "",
        (
            "| Model | Crash F1, ±1 ms | Crash F1, ±2 ms | Bin AUROC | Bin AUPRC | "
            "Bin F1 |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    models = benchmark.get("Tokamak-SI", {})
    for name, row in models.items():
        one, two = row["crash_tolerance_1ms"], row["crash_tolerance_2ms"]
        values = [
            scored(one["crash"].get("f1"), one.get("ci95", {}).get("crash_f1")),
            scored(two["crash"].get("f1"), two.get("ci95", {}).get("crash_f1")),
        ]
        values += [
            scored(two["presence"].get(key), two.get("ci95", {}).get("presence_" + key))
            for key in ("auroc", "auprc", "f1")
        ]
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    lines += [
        "",
        (
            "| Model | Assessed bins | Observable bins | Uncertain bins | "
            "Unassessed bins |"
        ),
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in models.items():
        totals = row["assessment_totals"]
        lines.append(
            f"| {name} | {totals['assessed_bins']} | {totals['observable_bins']} | "
            f"{totals['uncertain_bins']} | {totals['unassessed_bins']} |"
        )
    lines += [
        "",
        source(benchmark_path, "Tokamak-SI"),
        "",
        (
            "The inputs differ: `saw-hl3` receives two ECE group means, "
            "Mirnov mean and "
            "Ip; `saw-ours` receives all 48 ECE channels. The HL-3 paper used "
            "core/edge "
            "SXR, and verified DIII-D SXR spatial pairing is unavailable. "
            "This comparison "
            "therefore combines input information with architecture. The adapted HL-3 "
            "classifier gates a separate derivative picker because its "
            "published network "
            "does not output crash times."
        ),
        "",
        "## Independent model check per expert shot",
        "",
        (
            "Model expert scores use all known observable bins because independent "
            "expert annotations resolve algorithmic uncertainty. The coverage table "
            "retains the number of expert-positive bins the label rule called "
            "uncertain. "
            "No confidence interval is computed for the small expert set."
        ),
        "",
        (
            "| Model | Shot | Observable / algorithm-assessed reviewed bins | "
            "Uncertain / observable positive bins |"
        ),
        "|---|---:|---:|---:|",
    ]
    for name, row in models.items():
        for expert in expert_rows(row.get("expert", {})):
            lines.append(
                f"| {name} | {expert['shot']} | "
                f"{expert.get('observable_reviewed_bins', 'unavailable')} / "
                f"{expert.get('algorithm_assessed_bins', 'unavailable')} | "
                f"{expert.get('uncertain_positive_bins', 'unavailable')} / "
                f"{expert.get('expert_positive_observable_bins', 'unavailable')} |"
            )
    lines += [
        "",
        (
            "| Model | Shot | Observable reviewed bins | AUROC | AUPRC | "
            "Precision | Recall | F1 |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in models.items():
        for expert in expert_rows(row.get("expert", {})):
            metrics = presence(expert)
            bins = expert.get(
                "assessed_bins",
                metrics.get("positive_bins", 0) + metrics.get("negative_bins", 0),
            )
            lines.append(
                f"| {name} | {expert['shot']} | {bins} | "
                + " | ".join(
                    number(metrics.get(key))
                    for key in ("auroc", "auprc", "precision", "recall", "f1")
                )
                + " |"
            )
    lines += ["", source(benchmark_path, "Tokamak-SI.<model>.expert.by_shot")]
    for name, row in models.items():
        pooled = row.get("expert", {}).get("presence", {}).get("auroc")
        if pooled is not None and pooled < 0.5:
            lines += [
                "",
                (
                    f"{name} pooled expert AUROC is {number(pooled)}, below chance. "
                    "This independent ranking failure limits the model's physical "
                    "validation despite its out-of-fold agreement with "
                    "algorithmic labels."
                ),
                "",
                source(benchmark_path, f"Tokamak-SI.{name}.expert.presence.auroc"),
            ]
        below = [
            str(expert["shot"])
            for expert in expert_rows(row.get("expert", {}))
            if presence(expert).get("auroc") is not None
            and presence(expert)["auroc"] < 0.5
        ]
        if below:
            lines += [
                "",
                (
                    f"{name} ranks expert truth below chance on shots "
                    f"{', '.join(below)}. "
                    "This is an independent validation failure; agreement with its "
                    "training labels does not resolve it."
                ),
            ]
    legacy = benchmark.get("legacy", {})
    if legacy:
        lines += ["", "## Published HL-3 context", ""]
        for setting, row in legacy.items():
            if isinstance(row, dict) and "accuracy_stated" in row:
                lines.append(
                    f"{setting.replace('_', ' ').capitalize()} three-regime window "
                    f"accuracy: stated {number(row['accuracy_stated'])}, count-derived "
                    f"{number(row.get('accuracy_from_counts'))}."
                )
        lines += [
            "",
            (
                "OuYang et al., PPCF 67 (2025) 105004 reports HL-3 classification, a "
                "different task and population from DIII-D crash-tolerance scoring."
            ),
            "",
            source(benchmark_path, "legacy"),
        ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=REPO / "outputs/labeler/sawtooth/fix"
    )
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument(
        "--document", type=Path, default=REPO / "docs/labeler/sawtooth_results.md"
    )
    args = parser.parse_args()
    relative = (
        str(args.output.relative_to(REPO))
        if args.output.is_relative_to(REPO)
        else str(args.output)
    )
    cohort = pd.read_csv(REPO / "data/events/catalog/cohort.csv")
    records = {r["shot"]: r for r in records_at(args.work, cohort.shot)}
    splits = {}
    for split in ("train", "val", "test"):
        shots = sorted(int(s) for s in cohort.loc[cohort.split == split, "shot"])
        kept = [records[s] for s in shots if s in records]
        splits[split] = {
            "shots": shots,
            "shot_count": len(shots),
            "processed_count": len(kept),
            "crashes": sum(len(r.get("crashes", [])) for r in kept),
            "present_intervals": sum(len(r.get("intervals", [])) for r in kept),
            "uncertain_intervals": sum(
                len(r.get("uncertain_intervals", [])) for r in kept
            ),
            "positive_shots": sum(bool(r.get("intervals")) for r in kept),
            "uncertain_shots": sum(bool(r.get("uncertain_intervals")) for r in kept),
            "state_seconds": {
                state: sum(r.get("state_seconds", {}).get(state, 0) for r in kept)
                for state in ("present", "absent", "uncertain", "unassessed")
            },
        }
    save_json(args.output / "data_summary.json", {"splits": splits})
    lines = [
        "# Sawtooth evaluation records",
        "",
        (
            "These tables describe the corrected four-state physics labels and GPU "
            "models. Uncertain and unassessed support never supplies negative targets. "
            "Algorithmic agreement does not establish independent physical accuracy."
        ),
        "",
        "## Data",
        "",
        (
            "| Fixed split | Shots | Diagnostic crash points | "
            "Definite train candidates | Uncertain candidate spans | "
            "Definite train shots |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split, row in splits.items():
        lines.append(
            f"| {split} | {row['shot_count']} | {row['crashes']} | "
            f"{row['present_intervals']} | {row['uncertain_intervals']} | "
            f"{row['positive_shots']} |"
        )
    lines += ["", source(relative + "/data_summary.json", "splits")]
    lines += [
        "",
        (
            "These diagnostic counts describe train/candidate metadata. CSV state "
            "spans come from the canonical four-state support; overlapping uncertainty "
            "or unassessed support overrides a point's candidate state at export."
        ),
        "",
        "| Fixed split | Present (s) | Absent (s) | Uncertain (s) | Unassessed (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, row in splits.items():
        lines.append(
            f"| {split} | "
            + " | ".join(
                number(row["state_seconds"][state])
                for state in ("present", "absent", "uncertain", "unassessed")
            )
            + " |"
        )
    lines += [
        "",
        source(relative + "/data_summary.json", "splits.<split>.state_seconds"),
    ]
    population_path = args.output / "population_labels.json"
    if population_path.exists():
        population = read(args.output, "population_labels.json")
        lines += [
            "",
            (
                f"Population: {population['requested_count']} corpus files inspected; "
                f"{len(population['processed_shots'])} processed records, "
                f"{population['crashes']} diagnostic crash points and "
                f"{population['intervals']} "
                "definite train candidates. "
                f"{len(population['errors'])} read failures are excluded "
                "from label truth. Processed records can have no observable support; "
                "those bins are unassessed. The same frozen rule applies to "
                "cohort and population."
            ),
            "",
            source(relative + "/population_labels.json"),
        ]
    validation = read(args.output, "validation.json")
    validation_path = relative + "/validation.json"
    render_experts(lines, validation, validation_path)
    render_agreement(lines, validation, validation_path)
    impact_path = args.output / "antialias_impact.json"
    if impact_path.exists():
        render_antialias(
            lines,
            read(args.output, "antialias_impact.json"),
            relative + "/antialias_impact.json",
        )
    reference_path = args.output / "muscatello_reference.json"
    if reference_path.exists():
        reference = read(args.output, "muscatello_reference.json")
        lines += ["", "## Muscatello reference shots", ""]
        for row in reference["by_shot"]:
            if not row["corpus_present"]:
                lines += [
                    (
                        f"Shot {row['shot']}: local corpus file unavailable; "
                        "no detector "
                        "run or physical period/amplitude/timing check was performed."
                    ),
                    "",
                ]
                continue
            lines.append(
                f"Shot {row['shot']}: {row['status']}. "
                f"Detected period {number(row.get('detected_period_ms'))} ms, "
                "relative central "
                f"drop {number(row.get('detected_amplitude'))}, crash time "
                f"{number(row.get('detected_crash_time_ms'))} ms."
            )
        lines += ["", source(relative + "/muscatello_reference.json", "by_shot")]
    benchmark = read(args.output, "benchmark.json")
    render_models(lines, benchmark, relative + "/benchmark.json")
    lines += [
        "",
        "## GPU convergence and inner selection",
        "",
        (
            "| Model | Fold | Fit / selection shots | Best / completed epochs | "
            "Presence threshold | Crash threshold | Derivative z |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    stopping, derivative_bound_folds = [], []
    for path in sorted(args.output.glob("saw-*_fold_*.json")):
        fold = json.loads(path.read_text())
        selected = fold["selected_crash_threshold"]
        lines.append(
            f"| {fold['model']} | {fold['fold']} | "
            f"{len(fold['training_shots'])} / {len(fold['selection_shots'])} | "
            f"{fold['best_epoch']} / {fold['epochs_completed']} | "
            f"{number(fold['presence_threshold'])} | "
            f"{number(selected['threshold'])} | {number(selected['z'])} |"
        )
        stopping.append(
            f"{fold['model']} fold {fold['fold']}: {fold['device_name']}; "
            f"{fold['stopping_reason']}."
        )
        grid = fold.get("derivative_z_grid", [])
        if grid and selected["z"] == max(grid):
            derivative_bound_folds.append(f"{fold['model']} fold {fold['fold']}")
    lines += ["", " ".join(stopping), "", source(relative + "/saw-*_fold_*.json")]
    if derivative_bound_folds:
        lines += [
            "",
            (
                "The selected derivative threshold reaches the upper requested grid "
                f"boundary in {', '.join(derivative_bound_folds)}. "
                "The adapted baseline's "
                "crash performance remains limited despite the broader search."
            ),
        ]
    lines += [
        "",
        (
            "See [the method, calibration provenance and "
            "reproduction](sawtooth_physics.md)."
        ),
        "",
    ]
    args.document.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
