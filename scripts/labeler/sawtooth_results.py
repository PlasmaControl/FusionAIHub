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
        "## Old rule and physics labels against anchored expert spans",
        "",
        (
            "The span annotations for three expert shots were drawn while viewing "
            "the old "
            "`ece_sawtooth` suggestions and are anchored to that rule. Shot 190637's "
            "span may contain edge-originated relaxations. These checks are "
            "exploratory, unvalidated comparisons, not independent ground truth. "
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
        "## Model comparison with anchored expert spans",
        "",
        (
            "Model span scores use all known observable bins; the anchored "
            "annotations supply exploratory span targets. The coverage table "
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
                    "This anchored-span ranking result supplies no independent "
                    "physical validation despite its out-of-fold agreement with "
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
                    f"{name} ranks the anchored span targets below chance on "
                    f"{'shot' if len(below) == 1 else 'shots'} "
                    f"{', '.join(below)}. "
                    "Neither this comparison nor agreement with algorithmic "
                    "training labels establishes physical accuracy."
                ),
            ]
    legacy = benchmark.get("legacy", {})
    if legacy:
        lines += ["", "## Published HL-3 context", ""]
        lines += [
            (
                "| System / dataset | Three-class window accuracy [95% CI] | "
                "Macro-F1 [95% CI] |"
            ),
            "|---|---:|---:|",
        ]
        baseline = models.get("saw-hl3", {})
        if "three_class_window_accuracy" in baseline:
            majority = baseline["three_class_majority_baseline"]
            lines.append(
                "| Adapted saw-hl3 / unvalidated DIII-D labels | "
                + scored(
                    baseline["three_class_window_accuracy"],
                    baseline.get("three_class_accuracy_ci95"),
                )
                + " | "
                + scored(
                    baseline.get("three_class_macro_f1"),
                    baseline.get("three_class_macro_f1_ci95"),
                )
                + " |"
            )
            lines.append(
                "| Fit-chosen majority / same DIII-D windows | "
                + scored(majority["accuracy"], majority.get("accuracy_ci95"))
                + " | "
                + scored(
                    majority.get("macro_f1"),
                    majority.get("ci95", {}).get("macro_f1"),
                )
                + " |"
            )
            observed = baseline.get("three_class_observed_majority_baseline")
            if observed:
                lines.append(
                    f"| Observed majority class {observed['class']} / "
                    "same DIII-D windows | "
                    + scored(observed["accuracy"], observed.get("accuracy_ci95"))
                    + " | "
                    + scored(observed["macro_f1"], observed["ci95"]["macro_f1"])
                    + " |"
                )
        for setting, row in legacy.items():
            if isinstance(row, dict) and "accuracy_stated" in row:
                lines.append(
                    f"| OuYang {setting.replace('_', ' ')} / HL-3 | "
                    f"{number(row['accuracy_stated'])} stated; "
                    f"{number(row.get('accuracy_from_counts'))} count-derived | "
                    "not reported |"
                )
        if "three_class_window_accuracy" in baseline:
            protocol = baseline["three_class_window_protocol"]
            lines += [
                "",
                (
                    f"Classification covers {baseline['three_class']['windows']:,} "
                    f"{protocol['window_ms']} ms windows at "
                    f"{protocol['hop_ms']} ms hops, scored on "
                    f"{protocol['scoring_support']}. Per-fold fitting period "
                    f"boundaries are {protocol['class_boundaries_ms_by_fold']} ms."
                ),
                "",
                source(
                    benchmark_path, "Tokamak-SI.saw-hl3.three_class_window_protocol"
                ),
            ]
        if "three_class_confusion" in baseline:
            recalls = baseline["three_class_per_class_recall"]
            lines += [
                "",
                (
                    "Confusion rows are algorithmic truth; columns are predictions. "
                    "Classes are absent (0), short-period (1), long-period (2); the "
                    "period boundary is fitted on each training fold."
                ),
                "",
                "| Truth class | Predicted 0 | Predicted 1 | Predicted 2 | Recall |",
                "|---|---:|---:|---:|---:|",
            ]
            for index, row in enumerate(baseline["three_class_confusion"]):
                lines.append(
                    f"| {index} | {row[0]} | {row[1]} | {row[2]} | "
                    f"{number(recalls[index])} |"
                )
            lines += ["", source(benchmark_path, "Tokamak-SI.saw-hl3.three_class_*")]
            lines += [
                "",
                (
                    "The fit-chosen baseline predicts the natural fitting-window "
                    "majority selected separately in each fold. The observed "
                    "majority describes pooled out-of-fold class prevalence; its "
                    "class is fixed across bootstrap resamples. Neither baseline "
                    "selects model weights or thresholds."
                ),
            ]
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
        "--output", type=Path, default=REPO / "outputs/labeler/sawtooth/fix2"
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
            "These tables describe unvalidated four-state research labels and GPU "
            "models. Uncertain and unassessed support never supplies negative targets. "
            "Blind expert crash times are unavailable. No model is recommended "
            "as latest or stable; algorithmic agreement does not establish "
            "independent physical accuracy."
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
    audit_path = args.output / "state_transition_audit.json"
    if audit_path.exists():
        audit = read(args.output, "state_transition_audit.json")
        lines += [
            "",
            "## State policy before and after",
            "",
            (
                "Absence requires candidate-free support within ±1.5 maximum periods "
                "and a recorded core-ECE relaxation test with no periodic pattern. "
                "Stable significant negative core edges protect their entire phase "
                "without the positive train's period bounds or crossing an "
                "observability gap. "
                "Undetected ambiguous support stays uncertain. This is a conservative "
                "research negative-label policy, not expert-validated absence."
            ),
            "",
            (
                "| Scope / run | Present (s) | Absent (s) | Uncertain (s) | "
                "Unassessed (s) | Absent in <300 ms state holes (s) | "
                "Absent between candidate spans (s) |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for scope, comparison in audit["scopes"].items():
            for phase in ("before", "after"):
                row = comparison[phase]
                lines.append(
                    f"| {scope} / {phase} | "
                    + " | ".join(
                        number(row["state_seconds"][s])
                        for s in ("present", "absent", "uncertain", "unassessed")
                    )
                    + f" | {number(row['absent_in_short_holes_seconds'])} |"
                    + f" {number(row['absent_in_candidate_support_holes_seconds'])} |"
                )
        lines += [
            "",
            (
                "Candidate holes are gaps <300 ms between merged detected present/"
                "uncertain candidate spans. Canonical state holes also include tested "
                "quiet spans flanked by default uncertainty from context/noise limits; "
                "both definitions are reported explicitly."
            ),
            "",
            source(relative + "/state_transition_audit.json", "scopes"),
            source(relative + "/cohort_phase_refinement.json", "changed_shots"),
            source(relative + "/population_phase_refinement.json", "changed_shots"),
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
                "cohort and population, with different evidence availability."
            ),
            "",
            source(relative + "/population_labels.json"),
        ]
        prior_population = read(
            args.output.parent / "fix", "population_labels.json"
        )
        newly_excluded = sorted(
            set(prior_population["processed_shots"])
            - set(population["processed_shots"])
        )
        lines += [
            "",
            (
                f"Previously processed shots now excluded: {newly_excluded}; "
                "their screened ECE lacks enough coherent physical core channels. "
                "These failures supply no training or scoring truth."
            ),
            "",
            source(relative + "/population_labels.json", "errors"),
        ]
        current_cohort = read(args.output, "cohort_labels.json")

        def efit_count(summary):
            return summary.get("q_sources", {}).get("EFIT01", 0)

        lines += [
            "",
            (
                f"EFIT01 availability: {efit_count(population):,}/"
                f"{len(population['processed_shots']):,} population shots versus "
                f"{efit_count(current_cohort):,}/"
                f"{len(current_cohort['processed_shots']):,} cohort shots. The prior "
                "run had 1,213/13,650 versus 452/500; equality of the rule does not "
                "give equality of equilibrium evidence."
            ),
            "",
            source(relative + "/population_labels.json", "q_sources"),
            source(relative + "/cohort_labels.json", "q_sources"),
            source("outputs/labeler/sawtooth/fix/population_labels.json", "q_sources"),
            source("outputs/labeler/sawtooth/fix/cohort_labels.json", "q_sources"),
        ]
        lines += [
            "",
            "## Nominal geometry coverage and q=1 comparison",
            "",
            (
                "Same-shot RF metadata and field/axis support determine nominal R. "
                "Other shots use a per-shot hottest physical channel proxy; their "
                "radii are null. These comparisons remain physically unvalidated."
            ),
            "",
            (
                "| Scope | Mapped shots | Paired inversion / q=1 crashes | "
                "Median R difference (m) | Maximum absolute difference (m) |"
            ),
            "|---|---:|---:|---:|---:|",
        ]
        for scope, summary in (("cohort", current_cohort), ("population", population)):
            mapped = sum(
                n
                for key, n in summary.get("radius_geometry_counts", {}).items()
                if key.startswith("nominal_second_harmonic_R")
            )
            comparison = summary.get("q1_major_radius_comparison", {})
            lines.append(
                f"| {scope} | {mapped} | {comparison.get('comparable_crashes', 0)} | "
                f"{number(comparison.get('median_difference_m'))} | "
                f"{number(comparison.get('maximum_absolute_difference_m'))} |"
            )
        lines += [
            "",
            source(
                relative + "/population_labels.json",
                "radius_geometry_counts; q1_major_radius_comparison",
            ),
            source(
                relative + "/cohort_labels.json",
                "radius_geometry_counts; q1_major_radius_comparison",
            ),
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
        "## Pending blind annotation and paper example",
        "",
        (
            "The 15-shot nonexpert validation queue is in "
            "`data/events/sawtooth_oscillation/review/crash_time_queue.csv`; its "
            "regime stratification and shot/window list are recorded in "
            f"`{relative}/crash_time_queue.json`. The owner is away; blind crash "
            "accuracy remains unvalidated. The README contains the blind protocol."
        ),
        "",
        source(relative + "/crash_time_queue.json"),
        (
            "The 3.25-inch example's trace channels, 300 ms window, vector PDF and "
            f"150-dpi PNG paths are recorded in `{relative}/paper_example.json`. "
            "Markers are algorithmic candidates, not expert crash truth."
        ),
        "",
        source(relative + "/paper_example.json"),
        "",
        "## History appendix",
        "",
        (
            "The first correction used observable gaps as absence, an auxiliary "
            "noncorroboration gate, fixed ECE proxies and different training/inference "
            "masking. Its results remain archived under "
            "`outputs/labeler/sawtooth/fix/` "
            "and are superseded here. The three span comparisons were previously "
            "described as independent; the old-suggestion anchoring makes that "
            "description incorrect. No blind crash validation has been completed."
        ),
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
