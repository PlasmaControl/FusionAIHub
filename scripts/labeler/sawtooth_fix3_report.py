"""Render one current-state, source-backed sawtooth report and repository docs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from sawtooth_physics import OUTPUT, REPO, WORK

NAMES = {
    "saw-derivative": "Single-channel derivative / ±125 ms presence",
    "saw-always-present": "Always present",
    "saw-hl3": "Adapted HL-3 / derivative picker gated by HL-3",
    "saw-ours": "PhaseNet-style picker",
}


def number(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def interval(value, ci):
    return number(value) + (f" [{number(ci[0])}, {number(ci[1])}]" if ci else "")


def source(name, key=""):
    suffix = f" → `{key}`" if key else ""
    return f"Source: `outputs/labeler/sawtooth/fix3/{name}`{suffix}.\n"


def timing_table(legacy):
    lines = [
        (
            "| Shot / split | Prior offset ms | Fresh offset ms (pairs) | "
            "Fresh present picks | Observable s | Derivative matches: "
            "legacy / no holdoff |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in legacy["timing_table"]:
        pairs = row["fresh_offset_pairs_within_15ms"]
        fresh = number(row["fresh_old_minus_physics_ms_within_15ms"], 2)
        lines.append(
            f"| {row['shot']} / {row['split']} | "
            f"{number(row['legacy_minus_prior_ms'], 2)} | {fresh} ({pairs}) | "
            f"{row['fresh_present_points']} | "
            f"{number(row['fresh_observable_seconds'])} | "
            f"{number(row['legacy_matches_derivative_2ms'], 0)} / "
            f"{number(row['no_holdoff_matches_derivative_2ms'], 0)} |"
        )
    return "\n".join(lines) + "\n"


def reference_table(reference):
    lines = [
        "| Shot | Window s | Period ms | Relative amplitude | Status |",
        "|---|---|---:|---:|---|",
    ]
    for row in reference["by_shot"]:
        if not row.get("windows"):
            lines.append(f"| {row['shot']} | — | — | — | {row['status']} |")
        for window in row.get("windows", []):
            left, right = window["window_s"]
            lines.append(
                f"| {row['shot']} | {left}–{right} | "
                f"{number(window['median_period_ms'], 2)} | "
                f"{number(window['median_amplitude'])} | {row['status']} |"
            )
    return "\n".join(lines) + "\n"


def model_table(models, *, fixed=False):
    lines = [
        (
            "| Method | Crash F1 ±2 ms [95% CI] | Presence F1 [95% CI] | "
            "AUROC | AUPRC | Assessed / excluded / observable picks |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, label in NAMES.items():
        result = models[name]["fixed_validation"] if fixed else models[name]
        stats = result["crash_tolerance_2ms"]
        crash, presence, ci = stats["crash"], stats["presence"], stats["ci95"]
        counts = result["assessment_totals"]
        crash_f1 = interval(crash["f1"], ci.get("crash_f1")) if crash else "—"
        lines.append(
            f"| {label} | {crash_f1} | "
            f"{interval(presence['f1'], ci.get('presence_f1'))} | "
            f"{number(presence['auroc'])} | {number(presence['auprc'])} | "
            f"{counts['assessed_picks']} / {counts['excluded_picks']} / "
            f"{counts['observable_picks']} |"
        )
    return "\n".join(lines) + "\n"


def report(args):
    def read(name):
        return json.loads((args.output / name).read_text())

    data = read("data_summary.json")
    cohort, population = read("cohort_labels.json"), read("population_labels.json")
    bench, validation = read("benchmark.json"), read("validation.json")
    q1 = read("q1_radius_audit.json")
    frequency = read("fetched_frequency_audit.json")
    bias, null = read("qmin_bias_audit.json"), read("phase_null_audit.json")
    legacy, queue = read("legacy_disagreement.json"), read("crash_time_queue.json")
    reference, manifest = read("muscatello_reference.json"), read("label_manifest.json")
    verification = read("verification.json")
    models = bench["Tokamak-SI"]
    oof_coverage = models["saw-ours"]["coverage"]
    val_coverage = models["saw-ours"]["fixed_validation"]["coverage"]
    counts = models["saw-ours"]["assessment_totals"]
    percentage = 100 * counts["assessed_bins"] / counts["observable_bins"]
    lines = [
        "# Sawtooth physics-rule labels: current state\n",
        (
            "These are **physics-rule labels validated only by the checks described "
            "here**. There are no blind expert crash times. Neither physical label "
            "accuracy nor improvement over the production catalog is established. "
            "The reviewed spans are anchored to old suggestions and were consulted "
            "in previous rule revisions; they are exploratory, not untouched "
            "validation. No model is recommended as latest or stable.\n"
        ),
        (
            f"The cohort has {len(cohort['processed_shots'])}/{cohort['requested_count']} "
            f"successful records and {cohort['crashes']} diagnostic crash candidates. "
            f"The population has {len(population['processed_shots'])}/"
            f"{population['requested_count']} successful records and "
            f"{population['crashes']} diagnostic points. "
            f"OOF model scores assess {counts['assessed_bins']:,}/"
            f"{counts['observable_bins']:,} observable bins ({percentage:.1f}%).\n"
        ),
        source("cohort_labels.json"),
        source("population_labels.json"),
        source("benchmark.json", "Tokamak-SI.saw-ours.assessment_totals"),
        (
            "Conditional OOF crash F1 at ±2 ms: derivative "
            f"{number(models['saw-derivative']['crash_tolerance_2ms']['crash']['f1'])}, "
            "HL-3-gated derivative "
            f"{number(models['saw-hl3']['crash_tolerance_2ms']['crash']['f1'])}, "
            "saw-ours "
            f"{number(models['saw-ours']['crash_tolerance_2ms']['crash']['f1'])}. "
            "The derivative baseline has the highest point estimate; the "
            "HL-3 paired crash-F1 interval includes zero. "
            "HL-3's expert-shot ranking remains inverted on shot 190637; "
            "independent physical validation remains pending.\n"
        ),
        source("benchmark.json", "Tokamak-SI.*"),
        "### Geometry and equilibrium\n",
        (
            "Channels 0–39 use the archived fixed RF grid; same-shot setup takes "
            "precedence, including the documented exceptional archived shot. "
            "Channels 40–47 are excluded from core, outer, coincidence, "
            "redistribution and inversion evidence. Nominal vacuum resonance is "
            "R=2×27.992 GHz/T×|F_boundary|/f. The EFIT magnetic axis selects the core. "
            "Every time sample with R₂<(2/3)R_LCFS,out is excluded; the shot core "
            "screen also rejects channels in that overlap region. Missing radial "
            "metadata does not establish definite-positive spatial evidence. "
            "Previously terminal-dependent candidates are retained as uncertainty.\n"
        ),
        (
            f"The archived grid audit covers "
            f"{frequency['archived_grid_proof']['shots_audited']} shots with "
            f"{len(frequency['archived_grid_proof']['exceptions'])} documented "
            "setup exception. The fetched cohort setup check matches "
            f"{frequency['matches']}/{frequency['cached_setup_records']} "
            "first-40 grids. The fixed grid is transferred to other population "
            "shots; an unaudited historical setup change cannot be excluded.\n"
        ),
        source("fetched_frequency_audit.json"),
        (
            "The adapted HL-3 outer input uses low-field-side nominal geometric "
            "ρ=0.4–0.65, beyond the typical inversion region, rather than adjacent "
            "array rows. Geometric ρ=|R−R_axis|/(R_LCFS,out−R_axis) is **not** "
            "normalized flux. Vacuum mapping omits relativistic and optical-depth "
            "corrections. Missing ECEZH uses the published first-40 midplane "
            "assumption explicitly. Full EFIT profiles are needed for q=1; "
            "minimal field/axis/boundary metadata cannot supply that comparison.\n"
        ),
        source("geometry_metadata_audit.json"),
        (
            f"The q=1 audit contains {q1['efit_shots']} EFIT-supported shots, "
            f"{q1['q1_checked_shots']} with a profile intersection check, and "
            f"{q1['paired_shots']} with paired diagnostic inversion points. "
            f"All-point ΔR distribution: `{q1['distribution_all_diagnostic_points']}`. "
            "The full per-shot ledger includes checks with no axis-connected "
            "surface, checks with no candidate, and missing-profile cases. "
            "Paired nominal differences greater than 0.15 m flag uncertainty; "
            "a missing EFIT01 surface near q≈1 cannot distinguish reconstruction "
            "bias from the observed ECE train. It is recorded as incomparable.\n"
        ),
        source("q1_radius_audit.json"),
        (
            f"TRAIN definite-present nominal inversion ρ distribution: "
            f"`{q1['train_present_inversion_nominal_rho']}`. The shot-median "
            "outer input lies beyond the candidate inversion at "
            f"{100 * q1['train_present_outer_beyond_inversion_fraction']:.1f}% "
            "of comparable TRAIN definite-present points; the full ledger "
            "retains the distribution rather than assuming this holds for "
            "every event.\n"
        ),
        (
            "EFIT01 conflict is q_min>1.4. Sustained q_min≥1.5 for at least 50 ms "
            "supplies absence evidence. A magnetics-only reconstruction is not an "
            "MSE-constrained central-current measurement: the review-prescribed "
            "1.3–1.5 band replaces the unsupported 1.05 cutoff, rather than "
            "estimating a calibrated q correction. Only prior TRAIN candidates "
            "enter the bias audit. MSE-constrained q retains the stricter conflict "
            "test. Conflicting inversion-qualified trains remain uncertain over "
            "their full context; isolated POSR edges protect finite edge support "
            "without vetoing an entire high-q phase.\n"
        ),
        (
            f"Prior train q-conflict distribution: "
            f"`{bias['prior_train_candidate_qmin']}`. "
            f"Shot 186532 current state seconds: `{bias['shot_186532']['state_seconds']}`.\n"
        ),
        source("qmin_bias_audit.json"),
        source("freeze.json"),
        (
            "This TRAIN subset measures sensitivity to the previous cutoff, "
            "not the true EFIT bias. The 1.4/1.5 guards are prescribed "
            "conservative tolerances, with no calibrated q correction. "
            "Muscatello's radial reference uses MSE-constrained EFIT; the "
            "present nominal comparison uses EFIT01. See "
            "[Muscatello et al. (2012)]"
            "(https://doi.org/10.1088/0741-3335/54/2/025006) and the locally "
            "archived `Muscatello_ST.md` digest.\n"
        ),
        "### Relaxation phases and support\n",
        (
            "Phase edges require both a ≥2% fractional drop and POSR≥6. The phase "
            "period floor is 10 ms, above the old 5.15 ms picker holdoff artifact "
            "and conservatively below Muscatello's DIII-D reference periods. "
            "Positive trains retain the frozen 20–250 ms bounds. Phase expansion "
            "requires at least six edges and shuffled-time p≤0.05. Each null "
            "preserves count, span and picker holdoff, repeats the same grouping, "
            "and compares the minimum gap CV over all groups. It accounts for "
            "search and multiplicity within each observable run. Independent "
            "generated noise and regular-train checks test this calibration, not "
            "physical label validity.\n"
        ),
        (
            "In 1,000 independent shuffled trials, the full search accepts "
            f"{null['full_group_search_null']['rejected']}/"
            f"{null['full_group_search_null']['draws']} noise sequences "
            f"({100 * null['full_group_search_null']['rate']:.1f}%; 95% CI "
            f"{100 * null['full_group_search_null']['ci95'][0]:.2f}–"
            f"{100 * null['full_group_search_null']['ci95'][1]:.2f}%). "
            f"It accepts {null['periodic_contrast']['rejected']}/"
            f"{null['periodic_contrast']['draws']} jittered regular trains "
            "and rejects a regular 6 ms sequence. Qualification by POSR is "
            "conditioned on in this timing-null audit, rather than simulated. "
            "There is no global familywise guarantee across shots/runs.\n"
        ),
        source("phase_null_audit.json"),
        "| Split | Shots | Present s | Absent s | Uncertain s | Unassessed s | Assessed / observable |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "val", "test"):
        row = data["splits"][split]
        seconds = row["state_seconds"]
        lines.append(
            f"| {split} | {row['shots_requested']} | "
            + " | ".join(
                number(seconds[s])
                for s in ("present", "absent", "uncertain", "unassessed")
            )
            + f" | {100 * row['assessed_fraction_of_observable']:.1f}% |"
        )
    lines += [
        "",
        source("data_summary.json", "splits"),
        "### Conditional agreement with the physics rule on assessed bins\n",
        (
            f"OOF shots: {oof_coverage['requested_shots']} requested, "
            f"{oof_coverage['supported_shots']} with usable physical inputs, "
            f"{oof_coverage['scored_shots']} with assessed outcomes. Unsupported "
            "shots remain explicit unknown entries and supply no invented "
            "predictions or negatives.\n"
        ),
        (
            "OOF: fixed TRAIN shots, three whole-shot folds; inner shots select "
            "weights, LR/regularisation and operating points. Fixed val/test "
            "shots are excluded from all selection. Assessment masks affect "
            "loss and scoring, never the input definition. Uncertain and "
            "unassessed bins supply no negative truth. Excluded picks have "
            "unknown outcomes, not established false positives. CIs and "
            "paired comparisons use 1,000 whole-shot bootstrap draws.\n"
        ),
        (
            "The trivial picker differentiates only the EFIT-axis-selected "
            "single ECE channel. Inner shots independently select z for crash "
            "F1 and for the presence rule: any edge ≥zσ within ±125 ms. "
            "Thresholds, fitting/inner/scoring shot IDs and support counts "
            "are recorded for every fold. Always present supplies no crash time.\n"
        ),
        model_table(models),
        source("benchmark.json", "Tokamak-SI.*.crash_tolerance_2ms; assessment_totals"),
        (
            f"Second held-out set: {val_coverage['requested_shots']} nonexpert "
            f"fixed-validation shots requested; {val_coverage['supported_shots']} "
            f"have usable physical inputs and {val_coverage['scored_shots']} "
            "have assessed outcomes. Frozen "
            "three-fold ensembles and means of inner-selected thresholds. "
            "These still measure conditional teacher agreement.\n"
        ),
        model_table(models, fixed=True),
        source("benchmark.json", "Tokamak-SI.*.fixed_validation"),
        "| Held-out set | Method minus derivative-only | Δ crash F1 [95% CI] | Δ presence F1 [95% CI] |",
        "|---|---|---:|---:|",
    ]
    for fixed, label in ((False, "OOF"), (True, "Fixed validation")):
        for name in ("saw-derivative", "saw-hl3", "saw-ours", "saw-always-present"):
            row = models[name]["fixed_validation"] if fixed else models[name]
            if name == "saw-derivative":
                lines.append(f"| {label} | {NAMES[name]} (reference) | 0 | 0 |")
                continue
            paired = row["paired_vs_derivative"]["crash_tolerance_2ms"]
            lines.append(
                f"| {label} | {NAMES[name]} | "
                f"{interval(paired['difference']['crash_f1'], paired['ci95']['crash_f1'])} | "
                f"{interval(paired['difference']['presence_f1'], paired['ci95']['presence_f1'])} |"
            )
    lines += [
        "",
        source("benchmark.json", "paired_vs_derivative"),
        (
            "HL-3 is an adapted external architecture, replacing the paper's "
            "SXR pair with geometry-selected ECE plus Mirnov and Ip. It has "
            "no crash head: its timing score is the **derivative picker gated "
            "by HL-3**. LR, weight decay and dropout are selected on inner "
            "shots. The U-Net probability grid extends through 0.999; fold "
            "records retain selected operating points and boundary flags.\n"
        ),
        source("saw-hl3_fold_0.json"),
        source("saw-hl3_fold_1.json"),
        source("saw-hl3_fold_2.json"),
        source("saw-ours_fold_0.json"),
        source("saw-ours_fold_1.json"),
        source("saw-ours_fold_2.json"),
        "| Three-regime window classifier | Accuracy | Macro-F1 |",
        "|---|---:|---:|",
    ]
    classification = models["saw-hl3"]["three_class"]
    rows = (
        ("Adapted HL-3", classification),
        (
            "Single-channel derivative with period",
            models["saw-derivative"]["three_class"],
        ),
        ("Fitting-chosen majority", classification["majority_baseline"]),
    )
    for name, row in rows:
        lines.append(
            f"| {name} | {number(row['window_accuracy'])} | {number(row['macro_f1'])} |"
        )
    lines += [
        "| Always present (period class undefined) | — | — |",
        "",
        (
            "Derivative period abstentions count as errors on the original "
            "class support. Always present has no period class or crash time.\n"
        ),
        source("benchmark.json", "Tokamak-SI.*.three_class"),
        (
            "Published HL-3 context, different task/population: real-time "
            f"accuracy stated {number(bench['legacy']['real_time']['accuracy_stated'])}, "
            "count-derived "
            f"{number(bench['legacy']['real_time']['accuracy_from_counts'])}; "
            "offline accuracy stated "
            f"{number(bench['legacy']['offline']['accuracy_stated'])}, count-derived "
            f"{number(bench['legacy']['offline']['accuracy_from_counts'])}. "
            "These three-regime classification scores are not DIII-D crash "
            "scores. The stated and count-derived accuracies differ in the "
            "source.\n"
        ),
        source("benchmark.json", "legacy"),
        "### Exploratory reviewed spans and old-rule disagreement\n",
        (
            "| Method | Pooled reviewed-span AUROC | Per-shot reviewed-span F1 | "
            "Known / excluded / observable picks |"
        ),
        "|---|---:|---|---|",
    ]
    for name, label in NAMES.items():
        row = models[name]["expert"]
        cells = "; ".join(
            f"{r['shot']}: {number(r['presence']['f1'])}" for r in row["by_shot"]
        )
        picks = row["pick_totals"]
        lines.append(
            f"| {label} | {number(row['presence']['auroc'])} | {cells} | "
            f"{picks['assessed_picks']} / {picks['excluded_picks']} / "
            f"{picks['observable_picks']} |"
        )
    lines += [
        "",
        source("benchmark.json", "Tokamak-SI.*.expert"),
        (
            "Reviewed-score support is the intersection of known reviewed spans "
            "and observable inputs. Picks outside it have unknown outcomes. "
            "The JSON separately records picks excluded by physics-rule "
            "assessment for the conditional agreement diagnostic. Model "
            "predictions on reviewed spans are scored independently of the "
            "physics rule's assessment mask.\n"
        ),
        (
            f"HL-3 ranking check: {models['saw-hl3']['expert']['auc_diagnosis']}. "
            "No expert-based inversion or retuning was performed.\n"
        ),
        "| Reviewed shot | HL-3 AUROC | HL-3 assessed diagnostic AUROC | Saw-ours AUROC |",
        "|---|---:|---:|---:|",
    ]
    ours_by_shot = {row["shot"]: row for row in models["saw-ours"]["expert"]["by_shot"]}
    for row in models["saw-hl3"]["expert"]["by_shot"]:
        lines.append(
            f"| {row['shot']} | {number(row['presence']['auroc'])} | "
            f"{number(row['conditional_assessed_presence']['auroc'])} | "
            f"{number(ours_by_shot[row['shot']]['presence']['auroc'])} |"
        )
    lines += [
        "",
        source("benchmark.json", "Tokamak-SI.*.expert.by_shot"),
        (
            "HL-3's residual inversion is concentrated in shot 190637 and "
            "persists on assessed support. The physics rule calls no "
            "definite-present phase there while the reviewed spans contain "
            "substantial positive support. This is a teacher/review "
            "disagreement; it does not establish which labels are physically "
            "correct. The model is not validated for use.\n"
        ),
        (
            f"Source: `{args.work}/shots/190637.json` → `state_seconds`; "
            "`outputs/labeler/sawtooth/fix3/validation.json` → `expert.by_shot`.\n"
        ),
        (
            "| Reviewed shot | New recall | Legacy recall | New F1 | "
            "Uncertain / observable positive bins |"
        ),
        "|---|---:|---:|---:|---:|",
    ]
    for row in validation["expert"]["by_shot"]:
        lines.append(
            f"| {row['shot']} | {number(row['new']['presence']['recall'])} | "
            f"{number(row['old']['presence']['recall'])} | "
            f"{number(row['new']['presence']['f1'])} | "
            f"{row['expert_positive_uncertain_bins']} / "
            f"{row['expert_positive_observable_bins']} |"
        )
    lines += [
        "",
        "Span-supported picks are not crash precision/recall.\n",
        source("validation.json", "expert; legacy_agreement"),
        (
            "Native clipped legacy replay reproduces every retained pick "
            "array on the seven requested shots; clocks are exact. This "
            "excludes a reader timestamp bug. The ledger separates missing "
            "raw edges, greedy 10 ms suppression and inversion-window "
            "rejection; only training shots enter the diagnostic holdoff "
            "ablation. Current catalog v3 and the retained legacy v2 are "
            "distinct detectors, so a catalog-wide defect cannot be inferred "
            "from the legacy cache alone.\n"
        ),
        source("legacy_reader_audit.json"),
        "Per-shot timing and gate diagnosis (old minus physics, ms):\n",
        timing_table(legacy),
        (
            "Prior offsets refer to the superseded rule, whose spatial "
            "selection included unverified channels. Fresh offsets retain "
            "pairs within 15 ms; nearest neighbours are descriptive and can "
            "be reused. One-to-one cells are separate. Shots 190602/190604 "
            "have too few uncontaminated channels at their field for a "
            "verified redistribution profile and correctly abstain. "
            "The train-only ablation changes holdoff solely for diagnosis. "
            "Its full ledger gives the inversion rejection reasons and "
            "core/outer changes at both rules' picks. Shortening the legacy "
            "profile windows alone does not recover the matches: its block "
            "interiority, gain adjacency and amplitude conditions reject "
            "many central derivative edges.\n"
        ),
        source("legacy_disagreement.json", "by_shot; figures"),
    ]
    current = next(r for r in legacy["by_shot"] if r["shot"] == 192090)
    rows = current["current_production_v3_comparison"]["by_diagnostic"]
    lines += [
        (
            "| Shot 192090 current catalog | Total picks | In physics-present "
            "support | TP / FP / FN ±2 ms | Median offset ms |"
        ),
        "|---|---:|---:|---|---:|",
    ]
    for row in rows:
        cells = " / ".join(number(v, 0) for v in row["present_cells_2ms"])
        lines.append(
            f"| {row['diagnostic']} | {row['current_catalog_count']} | "
            f"{row['physics_present_catalog_count']} | {cells} | "
            f"{number(row['catalog_minus_physics_nearest_ms']['median'], 3)} |"
        )
    lines += [
        "",
        (
            "These TP/FP/FN cells name algorithm agreement against the physics "
            "rule. They do not establish physical errors. The current detector "
            "does not reproduce the retained legacy's systematic 10 ms offset "
            "on this available shot. The other six current event stores are "
            "unavailable locally.\n"
        ),
        source("legacy_disagreement.json", "by_shot.current_production_v3_comparison"),
        "### Muscatello references and blind queue\n",
        reference_table(reference),
        (
            "Compare only the published windows and amplitude definition; "
            "the reference measurements are physical sanity checks and "
            "never model/rule threshold selection.\n"
        ),
        source("muscatello_reference.json"),
        (
            f"Blind queue: {queue['windows']} windows on {queue['shot_count']} "
            f"shots, including {queue['random_primary_windows']} primary "
            f"random windows. Stratum counts: `{queue['strata']}`. "
            "Random observable windows are frozen "
            "before prediction access. Candidate-free, model-negative, "
            "uncertain and disagreement supplements are separate strata. "
            "Only sensor inputs and blank targets enter the annotation pack; "
            "private selection and all predictions remain hidden. About "
            "97 independent positive events give a worst-case 95% recall "
            "half-width of 0.1; clustered events do not supply that effective "
            "sample size automatically. Primary sampling weights and "
            "whole-shot CIs are preregistered. Annotation remains pending.\n"
        ),
        source("crash_time_queue.json"),
        "### Artifacts, reproduction and verification\n",
        (
            f"Complete labels: `$LABELER_ROOT/round4/saw/fix3/labels/`. "
            f"Manifest `SHA256SUMS` sha256: `{manifest['manifest_sha256']}`. "
            "Population shards export current states without duplicate cohort "
            "rows; failed-shot records remain in the ledger. The cohort bundle "
            "is separate. Production stores were read only.\n"
        ),
        source("label_manifest.json"),
        (
            "Large signals, EFIT metadata, models, predictions, annotation "
            "pack, PDFs and 150-dpi PNGs are under the same fix3 directory. "
            "Paper example, three old-rule figures and the 12-shot gallery "
            "were inspected; figure ledgers retain paths and hashes.\n"
        ),
        source("paper_example.json"),
        source("gallery.json"),
        source("figure_inspection.json"),
        (
            f"Covering tests: {verification['covering_tests_passed']} passed. "
            "Ruff passes on all changed Python files; formatting passes on "
            "all new Python files; git diff --check passes. Long jobs used "
            "timeouts and logs; temporary storage was swept afterward. "
            "GPU training used CUDA_VISIBLE_DEVICES=1 within the assigned "
            "memory budget.\n"
        ),
        source("verification.json"),
        "```text\n"
        + "\n".join(
            line
            for name in ("covering_tests", "ruff", "format")
            for line in verification["output_tails"][name]
        )
        + "\n```\n",
        (
            "Reproduce with the prescribed pixi environment and TMPDIR. "
            "Entrypoints: sawtooth_geometry_fix3.py fetch/audit/reference; "
            "sawtooth_physics.py labels; sawtooth_benchmark.py train/predict/"
            "evaluate; sawtooth_fix_validation.py validate; "
            "sawtooth_fix3_artifacts.py queue-base/queue/figures; "
            "sawtooth_fix3_records.py records/manifest; "
            "sawtooth_phase_null_audit.py; sawtooth_fix3_report.py. "
            "Run queue-base before model prediction access. Full command "
            "logs remain under the large output directory.\n"
        ),
        "### Concerns and next work\n",
        (
            "Blind physical accuracy remains unmeasured. Reviewed-span "
            "failures and any inverted HL-3 ranking remain failures, not "
            "reasons to retune on those shots. Nominal geometry is not a "
            "calibrated flux measurement; missing field/profile data limits "
            "population evidence. Algorithm-assessed benchmarks remain "
            "conditional and can be solved by derivative rules. Next: obtain "
            "the queued blind crash/span/ambiguity annotations, lock them, "
            "and evaluate all frozen methods over independently observable "
            "support without retuning.\n"
        ),
        "### Appendix: superseded history\n",
        (
            "Earlier rounds used a hottest-channel proxy, sparse same-shot "
            "RF localization, a q_min>1.05 conflict and unbounded fractional "
            "edge phases. Their results are superseded. The first-round "
            "`outputs/labeler/sawtooth/fix` records and the two earlier plan "
            "documents were removed; `outputs/labeler/sawtooth/fix2` remains "
            "as the immediate predecessor record, and the earlier rounds' "
            "narrative is in the stream report appendix.\n"
        ),
    ]
    body = "\n".join(lines[1:])
    # Repository doc: subsections are level two under the title.
    text = lines[0] + "\n" + re.sub(r"(?m)^### ", "## ", body)
    (REPO / "docs/labeler/sawtooth_results.md").write_text(text)
    (REPO / "docs/labeler/sawtooth_physics.md").write_text(
        "# Sawtooth physics-rule method\n\n"
        "The current method, thresholds, validations, results, limitations and "
        "reproduction commands are in [the current-state report]"
        "(sawtooth_results.md). Labels are physics-rule labels validated only "
        "by the checks reported there; blind expert crash annotations are "
        "pending. Production labels are not replaced.\n"
    )
    if args.report:
        # Appended as the last section; an earlier copy of this section is
        # replaced so reruns stay idempotent.
        marker = "\n## Fix round 3\n"
        existing = args.report.read_text()
        if marker in existing:
            existing = existing[: existing.index(marker)]
        status = (
            f"Status: DONE_WITH_CONCERNS. Commit range: `{args.report_commit_range}`.\n"
        )
        args.report.write_text(
            existing.rstrip("\n") + "\n" + marker + "\n" + status + "\n" + body
        )
    update_readme(manifest)


def update_readme(manifest):
    path = REPO / "data/events/sawtooth_oscillation/README.md"
    text = path.read_text()
    inputs = text.index("**saw-hl3**:", text.index("## Inputs"))
    method = text.index("## Method\n", inputs)
    # Keep the production ece_sawtooth description; replace everything from
    # the physics-rule paragraphs (first run) or the previous rewrite onward.
    tails = [
        text.find(marker, method)
        for marker in (
            "`labeler.sawtooth.physics.detect`",
            "## Physics-rule labels and validation",
        )
    ]
    tail = min(index for index in tails if index >= 0)
    last = text.index("## Alias", tail)
    inputs_content = """**saw-hl3**:
- EFIT-axis core ECE and low-field-side outer ECE at nominal geometric ρ=0.4–0.65
- Mirnov 0–1 mean and Ip in MA; missing values use fitting-shot means

**saw-ours**:
- The physical first-40-channel ECE array, 100 ms context at 10 kHz; unverified
  channels 40–47 and third-harmonic overlap samples are masked

"""
    content = f"""## Physics-rule labels and validation

These are **physics-rule labels validated only by the checks described** in the
[current-state report](../../../docs/labeler/sawtooth_results.md). The rule uses
Gude-style POSR, multichannel coincidence, core loss / outer gain, central drop,
stable trains and nominal EFIT localization. It screens harmonic overlap and
uses EFIT01 bias-aware q-min conflicts and sustained high-q absence evidence.
POSR-qualified phase edges require ≥10 ms periods and a shuffled-time null that
repeats the same group search. Geometry is nominal, with no flux calibration.
Present/absent/uncertain/unassessed states remain distinct. Production labels
are not replaced. Old `ece_sawtooth` disagreement and its reader audit are in
the report; the retained legacy rule and current catalog detector differ.
Valid core ECE defines observability: missing ECE, low temperature and detected
cutoff yield `unassessed`, and trains split at observability gaps. Native-rate
antialiasing precedes decimation to 10 kHz. Where local neutron-rate and Mirnov
data exist, their drop/burst flags give optional corroboration; no SXR
corroboration is claimed without verified core/edge spatial pairing. Absence
requires complete candidate-free context with a noise-resolved core relaxation
test, or sustained EFIT01 q-min ≥ 1.5; ambiguous observable support remains
uncertain. The untracked exports in `extend_saw_physics/` hold four-state spans
and crash points; they are additive research labels.

Complete population label shards are at
`$LABELER_ROOT/round4/saw/fix3/labels/`. Verify with `sha256sum -c SHA256SUMS`
from that directory. The `SHA256SUMS` file has sha256
`{manifest["manifest_sha256"]}`; individual CSV hashes are in
`outputs/labeler/sawtooth/fix3/label_manifest.json`. Cohort shards are a separate
bundle at `$LABELER_ROOT/round4/saw/fix3/cohort_labels/`; `extend_saw_physics/`
is the untracked integration copy. Git does not carry the large label store.

Both learned models use three whole-shot TRAIN folds with inner-shot selection
of checkpoint, hyperparameters and thresholds; CUDA training stops on
inner-selection loss patience. The trivial derivative and always-present
baselines use the same folds. The three reviewed shots and the blind test split
are excluded from training and tuning. `saw-hl3` receives adapted inputs
(EFIT-axis core ECE, low-field-side outer ECE, Mirnov, Ip) while `saw-ours`
receives the first 40 ECE channels, so comparisons include input information as
well as architecture. The second held-out set is the 47 nonexpert
fixed-validation shots. Headline scores are **conditional agreement with the
physics rule on assessed bins** and include excluded-pick counts and paired
shot-bootstrap comparisons. HL-3 crash timing is **derivative picker gated by
HL-3**, an adapted baseline, rather than a learned crash head. Reviewed spans
were anchored to old suggestions and used in previous rule revisions; they
are exploratory and provide no independent crash-time precision/recall; the
190637 span may include edge-originated relaxations.

## Blind crash-time annotation queue

The prediction-free input pack is
`$LABELER_ROOT/round4/saw/fix3/annotation_pack/`. It contains sensor windows,
observable masks, nominal geometry and blank annotation targets. Random windows
are frozen before prediction access. Candidate-free, model-negative, uncertain
and disagreement cases supplement the primary probability sample. Keep the
private `selection_audit/` directory and every detector/model prediction hidden.
Mark crash times, timing tolerances, positive/negative observable spans and
ambiguity masks; lock annotations before revealing picks. Use preregistered
sampling weights and whole-shot bootstrap intervals. Approximately 97
independent positive events give a worst-case 95% recall half-width of 0.1;
shot clustering reduces the effective count. The owner is away: annotation is
pending and physical accuracy remains unvalidated. No model is recommended.

"""
    text = text[:inputs] + inputs_content + text[method:tail] + content + text[last:]
    text = text.replace(
        "q conflicts abstain)", "nominal geometry and bias-aware q evidence)"
    )
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--report-commit-range", default="ad0ca40f..r4-saw")
    report(parser.parse_args())


if __name__ == "__main__":
    main()
