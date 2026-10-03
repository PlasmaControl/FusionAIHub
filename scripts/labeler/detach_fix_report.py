#!/usr/bin/env python
"""Append a current-state-first fix-round-two report from regenerated records."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from detach_protocol import metric

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ["LABELER_ROOT"]) / "round4/detach"
RESULTS = REPO / "docs/labeler/results"
DEFAULT_REPORT = (
    Path(os.environ["LABELER_ROOT"]) / "scratch/claude-89242e53/r4/reports/detach.md"
)


def load(name):
    return json.loads((RESULTS / f"detachment_{name}.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--start-commit", default="a5d18f9c")
    args = parser.parse_args()
    round2, fix, reference, validation = map(
        load, ("round2", "fix", "reference", "validation")
    )
    before, after = round2["before"], round2["after"]
    pop, certain = after["assessed"], after["certain"]
    states = after["states"]
    cohort_bins = sum(certain["by_split"][s]["bins"] for s in ("train", "val", "test"))
    bench = load("benchmark")
    model = json.loads(
        (
            REPO / "data/events/detachment/extend_detach_vote/records/label_model.json"
        ).read_text()
    )
    end = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True
    ).strip()
    lines = [
        "",
        "## Fix round 2",
        "",
        (
            f"Current state — **DONE_WITH_CONCERNS**. The compatibility rule "
            f"exports {pop['bins']:,} assessed bins/{pop['shots']} shots: "
            f"{states.get('1', 0)} attached, {states.get('2', 0)} detached, "
            f"{states.get('3', 0)} MARFE and {states.get('4', 0):,} uncertain. "
            f"There are {certain['bins']} certain bins/{certain['shots']} shots, "
            f"{certain['seconds']:.2f} s, with "
            f"{certain['single_50ms_intervals']}/{certain['intervals']} "
            "certain intervals lasting one 50 ms bin. The fixed cohort now has "
            f"{cohort_bins} certain bins on validation shot 189061; train/test "
            "remain zero. Paper use is exploratory agreement and coverage."
        ),
        "",
        (
            f"Commits: `{args.start_commit}..{end}` on `r4-detach`; all new commit "
            "subjects start `labeler:` with the requested Codex trailer. No push, "
            "merge, manuscript edit or production-store mutation."
        ),
        "",
        (
            "Sources for all current numbers: worktree "
            "`docs/labeler/results/detachment_round2.json` (before/after populations), "
            "`detachment_fix.json` (physics/quality invariants), "
            "`detachment_reference.json` (published points and manual DZ), "
            "`detachment_benchmark.json`, `detachment_{ours,victor}.json`, "
            "`detachment_bin_sensitivity.json`, `detachment_physics.json`, "
            "`figure2_detach.json` and `detachment_validation.json`. Each is produced "
            "by a committed script; exact shot lists, masks, thresholds and counts "
            "remain in the records."
        ),
        "",
        "### Physics, masks and geometry",
        "",
        (
            "Prad/Jsat now use their own ±2 ms D-alpha ELM masks. TangTV accepts "
            "ELM-integrated 30 Hz frames as Chen 2026 did; unknown filterscope "
            "coverage "
            "still abstains. FS01–FS04 supplements were attempted where needed; "
            "unknown coverage is elm_unknown and aux_elm_share remains NaN. "
            "The previous camera-wide −17/+50 ms mask no longer reaches Prad/Jsat."
        ),
        "",
        "Chen H-mode campaign 189057–189101, full discharge-bin population:",
        "",
        "| Indicator | Before valid bins / shots | After valid bins / shots |",
        "|---|---:|---:|",
    ]
    for name in ("tangtv", "prad", "afrac"):
        a, b = [r["chen_hmode"]["indicators"][name] for r in (before, after)]
        lines.append(
            f"| {name} | {a['valid_bins']} / {a['shots']} | "
            f"{b['valid_bins']} / {b['shots']} |"
        )
    lines += [
        "",
        "| ELM coverage population | Before NaN bins | After NaN bins |",
        "|---|---:|---:|",
    ]
    for name in ("discharge", "assessed", "certain"):
        lines.append(
            f"| {name} | {before['elm_coverage'][name]['nan']} | "
            f"{after['elm_coverage'][name]['nan']} |"
        )
    lower = fix["lower_shelf"]
    lines += [
        "",
        (
            "The owner's "
            "make_labels_2026.make_labels_for_file(r_max=SHELF_WALL_R=1.37) "
            "lower-shelf extraction is retained. Its primary bins remain uncertain "
            "in lower_shelf_window pending owner sign-off, with separate provisional "
            f"states: {lower['exported_tier']['bins']} assessed bins/"
            f"{lower['exported_tier']['shots']} shots, {lower['valid']['bins']} "
            "valid DZ measurements. The appendix selects an upper-shelf shot."
        ),
        "",
        (
            "Only positioned SOL-side current may vote: probe ΔR≥5 mm outboard of "
            "strike, distance≤2 cm and ψN≥1.01, on a close multi-slice EFIT map. "
            "Selected probe/R/Z/ψN/distance/strike/source are exported per bin. "
            "Unpositioned raw sweeps abstain. The inactive TangTV-attached Eldon fit "
            "was removed; current remains a self-referenced local proxy. "
            "Strict SOL/reference gates yield zero valid current votes; current "
            "certainty therefore uses Prad and TangTV only."
        ),
        "",
        (
            "MARFE spatial checks accept explicit EFIT01 fallback maps when EFIT02 "
            "is absent/sparse. fG uses per-shot confirmed BCI DENV2 units and "
            "nG=|Ip(MA)|/(πa²), with an elliptical V2 chord at R=1.94 m using EFIT R0 "
            "(not the unrelated ROUT node). Older V-unit records supply no cue. "
            "Triangularity/chord approximation remains a limitation."
        ),
    ]
    for row in reference["rows"]:
        if int(row["shot"]) == 199166:
            lines += [
                "",
                "Published 199166 at 3.705 s, exact containing bin: "
                + json.dumps(row, ensure_ascii=False),
            ]
    lines += [
        "",
        "### Label and benchmarks",
        "",
        (
            "Primary labels use the compatible-vote rule with upper-shelf TangTV "
            "required; no fitted posterior decides them. The historical state_lm "
            "column aliases state_rule. Snorkel remains state_model_diagnostic, with "
            "uncalibrated parameters. Rule/model agreement is "
            f"{model['rule_vs_label_model']['agreement']:.3%} on "
            f"{model['rule_vs_label_model']['bins']} assessed bins."
        ),
        "",
    ]
    p = fix["prad_development_validation"]
    lines += [
        (
            "Prad fixed local-threshold check uses upper-shelf development "
            f"inversions: {p['voting_reference']['bins']} cast pairs/"
            f"{p['voting_reference']['shots']} shots, binary κ "
            f"{metric({'value': p['binary_kappa'], 'ci95': p['kappa_ci95']})}. "
            f"Cohort-train has {p['cohort_train_valid_reference']['bins']} usable "
            "upper-shelf reference bins. Its "
            f"{p['provisional_lower_shelf_cohort_train_reference']['bins']} "
            "lower-shelf reference bins are separately provisional, pending "
            "owner sign-off. This is agreement, not physical validation."
        ),
        "",
        "| Unselected eligible-shot pair | Voting bins / shots | κ [95% CI] |",
        "|---|---:|---|",
    ]
    for pair, e in bench["pairwise_agreement"]["all_eligible_bins"].items():
        lines.append(
            f"| {pair} | {e['both_vote_bins']} / {e['both_vote_shots']} | "
            f"{metric(e['kappa'])} |"
        )
    lines += [
        "",
        (
            "LOO explicitly means bins where the other two indicators agree; "
            "selection favors agreement. The pairwise matrix is presented alongside "
            "it. Undefined κ is null and every metric records valid bootstrap "
            "replicate counts. CIs resample shots 1000 times. Ours has no missingness "
            "channels or imputation and only complete finite windows. The README "
            "Models lines report LOO κ and Victor's pre-fix 0/28 test-LOO case."
        ),
        "",
        "| Model | Combined CV κ | LOO all-source κ | LOO bins / shots |",
        "|---|---|---|---:|",
    ]
    for name in ("ours", "victor"):
        r = load(name)["stratified"]
        e = r["cv_all_loo_tangtv"]
        lines.append(
            f"| detach-{name} | {metric(r['cv_all_combined'].get('kappa'))} | "
            f"{metric(e.get('kappa'))} | {e['n_bins']} / {e['n_shots']} |"
        )
    lines += [
        "",
        (
            "Figure-2 F1 input is docs/labeler/figure2_detach.json, matching the "
            "fig_benchmarks row/value/CI source format. Local single settings and "
            "held-shot Victor are scored against the primary label. Published "
            "settings are unavailable; Chen SSA extraction is partial, not a "
            "reproduced published classifier. Coverage is separately tabulated "
            "with explicit measurement/valid-measurement/vote/consensus populations."
        ),
        "",
        "### Independent reference and coverage limits",
        "",
        (
            f"The published reference has {reference['n_primary_reference_points']} "
            f"explicit timed statements on {reference['n_primary_reference_shots']} "
            "shots, including MARFE; ambiguous transition statements are excluded. "
            "Scoring retains abstentions and groups results/bootstrap by shot. "
            "The papers overlap threshold motivation, so this is a small external "
            "sanity check, not held-out expert population accuracy."
        ),
        (
            "The primary consensus abstains on all 3/3 published points: zero "
            "reference coverage, 0/3 strict agreement, and undefined accuracy "
            "among cast votes. This check supplies no positive state validation."
        ),
    ]
    for method, score in reference["scores"].items():
        lines += [
            "",
            f"{method}: " + json.dumps(score["overall"]),
            "Per-shot: " + json.dumps(score["by_shot"]),
        ]
    manual = reference["manual_front_check"]
    lines += [
        "",
        (
            f"Manual front DZ check: {manual['n_annotated_bins']} annotated bins, "
            f"{manual['n_paired_valid_bins']} valid pairs/{manual['n_paired_shots']} "
            f"shot(s), MAE={manual['mae_dz']}, RMSE={manual['rmse_dz']}. "
            "These owner front points are not independent state truth."
        ),
        "",
        (
            f"The reviewed zero-certain-cohort count is superseded by {cohort_bins} "
            "certain validation bins after restoring ELM coverage on 189061. "
            "No train/test bin is certain. TangTV inversions exist on "
            f"{after['inversion_shots']} shots; the surrogate "
            f"contributes {after['surrogate_valid_bins']} valid bins. The reviewed "
            "pre-fix export's 56/113 single-bin intervals and 17.2 s are explicitly "
            "historical; current fragmentation/duration are in the opening summary."
        ),
        "",
        "Corpus survey: " + json.dumps(after["corpus_survey"]),
        "",
        "### Artifacts and verification",
        "",
        (
            "Regenerated bins/labels/benchmarks/CNN datasets and predictions/figures "
            "are under $LABELER_ROOT/round4/detach. Figure2 is 3.25 inches, ≥7 pt. "
            "Appendix views and timeline require full text width 6.75 inches, ≥7 pt; "
            "all PNGs inspected. Timeline plots voted f_div and DZ with thresholds; "
            "bolo chord profiles are restored without inventing spatial rays. "
            "Figure metadata names exact shot/times and source hashes. PDFs plus "
            "150-dpi PNGs: figure/fig_detachment_{views,timeline,figure2}.{pdf,png}."
        ),
        "",
        (
            "Changed library files: detachment/{core,signals,afrac,prad,tangtv,"
            "thresholds,label_model}.py. Orchestration/evaluation scripts: "
            "detach_{bins,label,benchmark,ours,victor,paper_panel,figure,reference,"
            "fetch_physics,physics_audit,fix_records,round2_records,protocol,"
            "validate_outputs,fix_report,docs_tables}.py. Covering tests updated "
            "in test_detachment_{core,indicators,label_model,surrogate}.py."
        ),
        "",
        "Validation: " + json.dumps(validation),
        "",
        "Test/lint logs: "
        + validation["covering_test_log"]
        + ", "
        + validation["lint_log"]
        + ". Long jobs used timeout and logs; "
        "tmpsweep ran after jobs. Only covering tests ran. Required CPU commands "
        "used pixi --frozen --no-install; CUDA jobs used assigned GPU 1.",
        "",
        (
            "UI paths and core schemas are stable. HANDOFF.md lists additive "
            "ELM/tier/probe fields and the semantic state_lm alias change. "
            "Interval confidence is null for the rule; model probabilities are "
            "diagnostic. detach-ui should rebuild from the regenerated outputs."
        ),
        "",
        "### Concerns and next steps",
        "",
        (
            "Owner sign-off is still needed for lower_shelf_window. The independent "
            "reference is only three points/two shots and the manual DZ check one "
            "shot. No expert reference, attached-current calibration, gold-test "
            "accuracy or surrogate transfer coverage is available. Missing/"
            "unit-ambiguous data abstains. fG chord/EFIT uncertainty margins are "
            "operational approximations. Published MARFE onset discrepancies are "
            "shown with failed gates rather than tuned away. Next: expert review "
            "of source-localized state examples and attached-current calibration."
        ),
    ]
    with args.report.open("a") as stream:
        stream.write("\n".join(lines) + "\n")
    print("appended Fix round 2 to", args.report)


if __name__ == "__main__":
    main()
