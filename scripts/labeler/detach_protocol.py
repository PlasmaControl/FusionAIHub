#!/usr/bin/env python
"""Write current protocol, README and UI handoff summaries from source records."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "docs/labeler/results"
ROOT = Path(os.environ["LABELER_ROOT"]) / "round4/detach"


def load(name):
    return json.loads((RESULTS / f"detachment_{name}.json").read_text())


def metric(entry):
    if not entry or entry.get("value") is None:
        return "N/A"
    value = entry["value"]
    if not math.isfinite(value):
        return "N/A"
    lo, hi = entry.get("ci95", (None, None))
    if lo is None or hi is None:
        return f"{value:.3f}"
    return f"{value:.3f} [{lo:.3f}, {hi:.3f}]"


def replace_block(path, name, text):
    start, end = f"<!-- {name} -->", f"<!-- /{name} -->"
    doc = path.read_text()
    a = doc.index(start)
    b = doc.index(end, a) + len(end)
    path.write_text(doc[:a] + start + "\n\n" + text + "\n\n" + end + doc[b:])


def main():
    fix, bench, round2, reference, width = map(
        load, ("fix", "benchmark", "round2", "reference", "bin_sensitivity")
    )
    records = REPO / "data/events/detachment/extend_detach_vote/records"
    model = json.loads((records / "label_model.json").read_text())
    coverage = json.loads((records / "coverage.json").read_text())
    baselines = {name: load(name) for name in ("ours", "victor")}
    before, after = round2["before"], round2["after"]
    pop, certain = after["assessed"], after["certain"]
    states = after["states"]
    survey = after["corpus_survey"]
    state_text = ", ".join(
        f"{states.get(str(k), 0):,} {name}"
        for k, name in (
            (1, "attached"),
            (2, "detached"),
            (3, "MARFE"),
            (4, "uncertain"),
        )
    )
    cohort_bins = sum(certain["by_split"][s]["bins"] for s in ("train", "val", "test"))
    coverage_honesty = (
        f"The fixed 500-shot cohort contains **{cohort_bins} certain bins** "
        f"on {certain['by_split']['val']['shots']} validation shot(s); "
        f"train has {certain['by_split']['train']['bins']} and test has "
        f"{certain['by_split']['test']['bins']}. Restored ELM coverage on 189061 "
        "supersedes the reviewed export's zero-cohort count. "
        "True TangTV inversions exist on only "
        f"{after['inversion_shots']} shots; the "
        f"surrogate supplies {after['surrogate_valid_bins']} valid bins. "
        f"Current certainty totals {certain['seconds']:.2f} s; "
        f"{certain['single_50ms_intervals']}/{certain['intervals']} certain intervals "
        "are single 50 ms bins. The reviewed pre-fix export had "
        f"{before['certain']['single_50ms_intervals']}/"
        f"{before['certain']['intervals']} "
        f"single-bin intervals and {before['certain']['seconds']:.1f} s total; "
        "those historical counts are superseded. "
        "[Before/after population record](results/detachment_round2.json)."
    )
    lines = [
        (
            "**Exploratory agreement and coverage.** "
            f"The primary rule exports {pop['bins']:,} assessed bins/{pop['shots']} "
            f"shots: "
            f"{state_text}. Its {certain['bins']} certain bins cover "
            f"{certain['shots']} shots. "
            f"The full discharge population is {after['population']['bins']:,} bins/"
            f"{after['population']['shots']} shots."
        ),
        "",
        coverage_honesty,
        "",
        (
            f"The corpus survey has {survey['shots']:,} shots with usable TangTV "
            f"{survey['tangtv']:,}, bolometer {survey['bolo']:,}, Langmuir "
            f"{survey['langmuir']:,}, IRTV {survey['irtv']:,} (time length >1; "
            "group presence alone is insufficient). Candidate selection is the fixed "
            f"{fix['candidate_selection']['count']}-shot union described in the fix "
            f"audit."
        ),
        "",
        (
            "| Indicator, full discharge population | Valid measurement bins / shots | "
            "Cast vote bins |"
        ),
        "|---|---:|---:|",
    ]
    for name, label in (
        ("afrac", "SOL Jsat local proxy"),
        ("prad", "Local f_div"),
        ("tangtv", "Inversion DZ"),
    ):
        c = coverage[name]
        lines.append(
            f"| {label} | {c['valid_bins']:,} / {c['shots_with_valid_bins']} | "
            f"{sum(c['votes'].values()):,} |"
        )
    lines += [
        "",
        (
            "Strict SOL selection leaves no valid Jsat votes: positioned current "
            "is sparse and lacks the required 3 s normalization reference. "
            "Current certain bins therefore depend on Prad and TangTV. "
            "The model has no three-indicator anchors; its dependence parameters "
            "are unidentifiable from this population."
        ),
    ]
    lines += ["", "| Exported tier | Bins |", "|---|---:|"]
    lines += [f"| `{tier}` | {n:,} |" for tier, n in after["tiers"].items()]
    lower = fix["lower_shelf"]
    lines += [
        "",
        (
            f"Lower shelf is reported separately: {lower['valid']['bins']} valid DZ "
            f"measurements "
            f"on {lower['valid']['shots']} shots; {lower['exported_tier']['bins']} "
            f"assessed "
            "bins have `tier=lower_shelf_window`, all uncertain in the primary export. "
            f"Provisional state counts are {lower['provisional_states']}. Pending "
            f"owner sign-off."
        ),
        "",
        "| ELM coverage population | Finite bins | NaN bins |",
        "|---|---:|---:|",
    ]
    for name, e in after["elm_coverage"].items():
        lines.append(f"| {name} | {e['finite']} | {e['nan']} |")
    lines += [
        "",
        (
            "Unknown masks are counted explicitly. All primary certainty, conflict, "
            "geometry "
            "and MARFE audit violations are zero; integrated TangTV frames are allowed "
            "to "
            "overlap ELMs. The masks for Prad and Jsat are ±2 ms."
        ),
        "",
        (
            "Chen H-mode campaign coverage below uses all discharge bins on the "
            "specified "
            "inversion shots 189057–189101, before export eligibility: "
            f"{after['chen_hmode']['bins']} bins/{after['chen_hmode']['shots']} "
            "shots. It is a "
            "measurement "
            "coverage comparison; the exact per-shot counts are in the round-two "
            "record."
        ),
        "",
        (
            "| Indicator | Before valid bins / contributing shots | After valid bins / "
            "contributing shots |"
        ),
        "|---|---:|---:|",
    ]
    for name in ("tangtv", "prad", "afrac"):
        a, b = [r["chen_hmode"]["indicators"][name] for r in (before, after)]
        lines.append(
            f"| {name} | {a['valid_bins']} / {a['shots']} | {b['valid_bins']} / "
            f"{b['shots']} |"
        )
    lines += [
        "",
        (
            "| Pairwise vote agreement, eligible-shot population | Both vote bins / "
            "shots | κ [95% shot CI] | Valid κ replicates / 1000 |"
        ),
        "|---|---:|---|---:|",
    ]
    for pair, e in bench["pairwise_agreement"]["all_eligible_bins"].items():
        lines.append(
            f"| {pair.replace('__', ' / ')} | {e['both_vote_bins']} / "
            f"{e['both_vote_shots']} | {metric(e['kappa'])} | "
            f"{e['kappa'].get('valid_replicates', 0)} |"
        )
    pairs = bench["pairwise_agreement"]["all_eligible_bins"]
    names = ("afrac", "prad", "tangtv")
    lines += [
        "",
        "| κ matrix [95% shot CI] | Jsat | Prad | TangTV |",
        "|---|---|---|---|",
    ]
    for i, name in enumerate(names):
        cells = []
        for j, other in enumerate(names):
            key = "__".join((names[min(i, j)], names[max(i, j)]))
            cells.append("—" if i == j else metric(pairs[key]["kappa"]))
        lines.append("| " + name + " | " + " | ".join(cells) + " |")
    lines += [
        "",
        (
            "This is the unselected pairwise matrix. The following LOO reference is "
            "**bins where the other two indicators agree**. It selects compatible "
            "valid "
            "other votes and favors agreement; it is not expert truth. TangTV-withheld "
            "bins use the weak Jsat/Prad pair. All CIs resample shots 1000 times. "
            "Undefined κ is null, with valid replicate counts retained."
        ),
        "",
        (
            "| Indicator | Reference bins | Cast comparison bins / shots | Binary κ "
            "[95% CI] |"
        ),
        "|---|---:|---:|---|",
    ]
    for name, entries in bench["indicators"].items():
        e = entries.get("loo", {}).get("all")
        if e:
            lines.append(
                f"| {name} | {e['reference_certain_bins']} | "
                f"{int(e['n_bins']['value'])} / {e['n_shots']} | "
                f"{metric(e['binary_kappa'])} |"
            )
        else:
            lines.append(f"| {name} | 0 | 0 / 0 | N/A |")
    match = model["rule_vs_label_model"]
    lines += [
        "",
        (
            f"The fitted Snorkel diagnostic and primary rule agree on "
            f"{match['agreement']:.3%} "
            f"of {match['bins']} assessed bins. The model uses "
            f"{model['fit']['anchor_bins']} "
            f"inversion-only anchors/{model['fit']['anchor_shots']} shots, with "
            "uncalibrated weights/posteriors. This diagnostic does not confer extra "
            "certainty. The compatibility rule is the primary labeler. "
            "[Model "
            "record](../../data/events/detachment/extend_detach_vote/records/"
            "label_model.json)."
        ),
    ]
    p = fix["prad_development_validation"]
    lines += [
        "",
        (
            f"Prad local-threshold development check: {p['voting_reference']['bins']} "
            f"cast pairs/{p['voting_reference']['shots']} upper-shelf development "
            "shots, binary κ "
            f"{metric({'value': p['binary_kappa'], 'ci95': p['kappa_ci95']})}; "
            f"cohort-train has {p['cohort_train_valid_reference']['bins']} usable "
            f"reference "
            "bins. Thresholds were fixed before the check. This is not independent "
            "validation."
        ),
        (
            "The lower-shelf check is provisional: "
            f"{p['provisional_lower_shelf_valid_reference']['bins']} valid "
            "reference bins, including "
            f"{p['provisional_lower_shelf_cohort_train_reference']['bins']} "
            "cohort-train bins, are reported separately pending owner sign-off."
        ),
        "",
        (
            "| CNN, shot-held-out reference | Scored bins / shots | Accuracy [95% CI] "
            "| κ [95% CI] |"
        ),
        "|---|---:|---|---|",
    ]
    for name, r in baselines.items():
        for key, label in (
            ("cv_all_combined", "primary combined label"),
            ("cv_inversion_loo_tangtv", "other two agree, inversion shots"),
            ("cv_all_loo_tangtv", "other two agree, all sources"),
            ("test_all_loo_tangtv", "other two agree, fixed test"),
        ):
            e = r["stratified"][key]
            lines.append(
                f"| detach-{name}: {label} | {e['n_bins']} / {e['n_shots']} | "
                f"{metric(e.get('accuracy'))} | {metric(e.get('kappa'))} |"
            )
    lines += [
        "",
        (
            "Ours uses complete finite input windows and no missingness channels or "
            "imputation. CNN epochs and architecture are fixed, folds group by shot, "
            "and normalization uses training-fold data. These weak-reference scores "
            "cannot establish gold-test accuracy. The reviewed Victor test-LOO result "
            "was 0/28 correct bins on one shot; the current test row above supersedes "
            "it after regeneration."
        ),
        "",
        (
            f"The independently sourced state reference has "
            f"{reference['n_primary_reference_points']} "
            f"points on {reference['n_primary_reference_shots']} shots, selected from "
            f"explicit "
            "published statements before scoring. It includes the published MARFE on "
            "199166 at 3.705 s. Per-shot labels, indicator abstentions, agreement and "
            "coverage are in [the reference "
            "record](results/detachment_reference.json). "
            "These few statements are not an expert-reviewed population reference. "
            "The owner's manual fronts supply a separate DZ check, not state truth."
        ),
        (
            "**The primary consensus abstains on all 3/3 published points: zero "
            "reference coverage, 0/3 strict agreement, and undefined accuracy among "
            "cast votes.** This external check provides no positive state validation."
        ),
        "",
    ]
    for row in reference["rows"]:
        if int(row["shot"]) == 199166:
            pred = row["prediction"]
            gates = row["gate_diagnostics"]
            lines.append(
                f"199166 at {row['reference_time_ms']} ms: primary "
                "**uncertain, candidate_marfe** (scored as abstention); "
                f"TangTV {pred['tangtv']['vote_name']}, Jsat "
                f"{pred['afrac']['vote_name']}, "
                f"Prad {pred['prad']['vote_name']}. DZ="
                f"{pred['tangtv']['value']:.3f}; fG="
                f"{gates['aux_greenwald_fraction']:.3f} fails the unchanged 0.8 cue, "
                "with no H–L back-transition. Spatial evidence also fails at the "
                "onset bin; it passes at 3750/3800 ms but fG remains below 0.8. "
                "The published MARFE is missed despite available EFIT01 maps and "
                "confirmed density units. The threshold was not retuned to this "
                "reference. All gates/provenance are in the bin and reference "
                "records; nearby witness bins are in "
                "[the physics record](results/detachment_physics.json)."
            )
    lines += [
        "",
        (
            "| Published reference shot / method | Reference points | Cast votes | "
            "Correct votes |"
        ),
        "|---|---:|---:|---:|",
    ]
    for method, scores in reference["scores"].items():
        for shot, score in scores["by_shot"].items():
            correct = sum(
                score["confusion_truth_by_prediction"][k][k] for k in range(3)
            )
            lines.append(
                f"| {shot} / {method} | {score['n_reference_points']} | "
                f"{score['n_cast_votes']} | {correct} |"
            )
    manual = reference["manual_front_check"]
    lines += [
        "",
        (
            f"Manual front check: {manual['n_paired_valid_bins']} paired valid bins/"
            f"{manual['n_paired_shots']} shot(s); DZ MAE {manual.get('mae_dz')} "
            f"on {manual['n_annotated_bins']} annotated bins. Its manual points reuse "
            f"the "
            "same imaging modality and do not independently validate detachment states."
        ),
        "",
        (
            "| Sensitivity width (ms), non-test shots | Assessed bins / shots | "
            "Certain bins / shots |"
        ),
        "|---|---:|---:|",
    ]
    for w in (20, 50, 100):
        e = width[f"{w}ms"]
        lines.append(
            f"| {w} | {e['n_assessed_bins']} / {e['n_assessed_shots']} | "
            f"{e['n_certain_bins']} / {e['n_certain_shots']} |"
        )
    lines += [
        "",
        (
            "The sensitivity is descriptive; no fixed test shot selects bin width. "
            "[Benchmark](results/detachment_benchmark.json), "
            "[fix audit](results/detachment_fix.json), "
            "[width record](results/detachment_bin_sensitivity.json), "
            "[F1 input](figure2_detach.json), "
            "[separate coverage populations](results/detachment_figure2.json)."
        ),
    ]
    replace_block(REPO / "docs/labeler/detachment.md", "RESULTS", "\n".join(lines))
    models = [
        (
            f"- detach_vote | 2026_10_03: compatibility rule; {pop['bins']} assessed "
            f"bins/{pop['shots']} shots, {certain['bins']} certain/{certain['shots']} "
            f"shots, "
            f"{states.get('3', 0)} MARFE; Snorkel is only a diagnostic."
        ),
        "- detach_rule | 2026_10_03: alias of the primary compatibility rule.",
    ]
    for name, r in baselines.items():
        e = r["stratified"]["cv_all_combined"]
        loo = r["stratified"]["cv_all_loo_tangtv"]
        inversion = r["stratified"]["cv_inversion_loo_tangtv"]
        test = r["stratified"]["test_all_loo_tangtv"]
        models.append(
            f"- detach-{name} | 2026_10_03: combined-label CV κ "
            f"{metric(e.get('kappa'))} "
            f"({e['n_bins']} bins/{e['n_shots']} shots); **LOO κ "
            f"{metric(loo.get('kappa'))}** "
            f"({loo['n_bins']} bins/{loo['n_shots']} shots), inversion-only "
            f"{metric(inversion.get('kappa'))}; test-LOO accuracy "
            f"{metric(test.get('accuracy'))} "
            f"({test['n_bins']} bins/{test['n_shots']} shots). Reference: bins where "
            "the other two indicators agree. "
            + (
                "Reviewed pre-fix Victor test-LOO was 0/28 on one shot."
                if name == "victor"
                else "Complete finite inputs only; no missingness channels."
            )
        )
    models += [
        "",
        coverage_honesty.replace(
            "(results/detachment_round2.json)",
            "(../../../docs/labeler/results/detachment_round2.json)",
        ),
        "",
        (
            "Strict SOL/reference gates yield zero valid Jsat votes, so certainty "
            "currently uses Prad and TangTV. The primary consensus abstains on "
            "all 3 published reference points; independent state validation remains "
            "unavailable."
        ),
        "",
        (
            f"Corpus usable records out of {survey['shots']:,}: TangTV "
            f"{survey['tangtv']:,}, "
            f"bolo {survey['bolo']:,}, Langmuir {survey['langmuir']:,}, IRTV "
            f"{survey['irtv']:,}. "
            "Source: detachment_round2.json. Scores are exploratory agreement and "
            "coverage; "
            "no primary gold-test accuracy is available."
        ),
    ]
    replace_block(
        REPO / "data/events/detachment/README.md", "MODELS", "\n".join(models)
    )
    handoff = [
        "# Detachment data interface for detach-ui",
        "",
        (
            f"Fix round 2 complete: {pop['bins']} assessed bins/{pop['shots']} shots; "
            f"{state_text}. "
            f"{certain['bins']} certain bins/{certain['shots']} shots, "
            f"{certain['seconds']:.2f} s. "
            "Earlier summaries are superseded. Source records: this worktree's "
            "docs/labeler/results/"
            "detachment_{round2,fix,reference,benchmark,figure2}.json and "
            "extend_detach_vote/records/."
        ),
        "",
        (
            "Core state codes and paths are stable: absent=0 internally, attached=1, "
            "detached=2, "
            "MARFE=3, uncertain=4. Missing rows mean unassessed. Primary state is the "
            "compatible "
            "vote rule with upper-shelf TangTV required and known ELM coverage. "
            "Confidence is null. "
            "`state_lm` is now the legacy alias of `state_rule`; use "
            "`state_model_diagnostic` for "
            "the fitted model. Lower shelf remains state=4, tier=lower_shelf_window "
            "pending owner "
            "sign-off, with separate `state_lower_shelf_window` suggestions. Temporal "
            "imputation "
            "remains a separate suggestion and cannot promote certainty."
        ),
        "",
        (
            "- Worktree intervals: "
            "data/events/detachment/extend_detach_vote/detach_shots.csv; "
            "shot,category,t_start,t_end,confidence,attrs. attrs contains tier. Sparse "
            "per-shot "
            "grids remain extend_detach_vote/detach_shots/<shot>.npz."
        ),
        (
            "- Full bins: round4/detach/labels_bins.csv.gz; canonical states/tier and "
            "all input "
            "provenance. labels_rule.csv remains stable; labels_label_model.csv is "
            "diagnostic."
        ),
        (
            "- indicators/<shot>.csv: existing traces remain; additive prad_fraction, "
            "tangtv_dz, "
            "elm_known, elm_share, state_lower_shelf_window and selected-probe "
            "provenance fields. "
            "Display current as uncalibrated Jsat ratio (local proxy)."
        ),
        (
            "- bins/<shot>.npz: existing arrays retained with tangtv_tier, explicit "
            "aux_elm_share "
            "(NaN unknown), fG and positioned selected-probe provenance. Inspect keys "
            "beginning "
            "aux_jsat_ or afrac_probe for exact selection details. Unknown "
            "ELM vetoes "
            "certainty; Prad/Jsat use ±2 ms masks; TangTV accepts integrated camera "
            "frames."
        ),
        (
            "- processed_probes/<shot>.npz: positioned processed Jsat in amps/cm² and "
            "metres. "
            "Only probes outboard of strike with uncertainty margin and psiN>1 margin "
            "may vote."
        ),
        (
            "- geometry02/<shot>.npz and cache/<shot>.npz: scalar "
            "geometry/current/heating "
            "times in ms. EFIT01 fallback is named; maps in efit/<shot>.npz contain "
            "source, "
            "gtime_ms,r,z,psirz(time,z,r),ssimag,ssibry,boundary and wall LIM."
        ),
        "",
        (
            "The read-only corpus "
            "/scratch/gpfs/EKOLEMEN/foundation_model/<shot>_processed.h5 "
            "has tangtv xdata seconds and ydata[channel,time,240,720]; channel 2 is "
            "lower "
            "perpendicular. Check xdata length>1 and liveness. Inversions/<shot>.npz "
            "stores "
            "emissivity(time,Z,R),times_ms,radii,elevation in metres. Raw SAV "
            "VID/VID_TIMES "
            "can be read by detach_figure.read_video where present. Bolo has 48 raw "
            "chord "
            "voltages, not an image; restored chord-profile row does not invent "
            "spatial rays. "
            "IRTV HEATFLUX NODATA was only an explicit shot-level probe, not a "
            "campaign survey."
        ),
        "",
        (
            "Appendix views/timeline require full 6.75-inch text width, >=7 pt. "
            "Figure2 F1 source is docs/labeler/figure2_detach.json; separate coverage "
            "JSON "
            "records explicit populations. Rebuild the UI from these regenerated "
            "outputs."
        ),
    ]
    (ROOT / "HANDOFF.md").write_text("\n".join(handoff) + "\n")
    print("refreshed protocol, README and HANDOFF from final records")


if __name__ == "__main__":
    main()
