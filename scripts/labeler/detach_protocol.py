#!/usr/bin/env python
"""Refresh protocol/README result blocks and the external UI handoff from records."""

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
    value = entry.get("value")
    if value is None or not math.isfinite(value):
        return "N/A"
    lo, hi = entry["ci95"]
    if lo is None or hi is None:
        return f"{value:.3f}"
    return f"{value:.3f} [{lo:.3f}, {hi:.3f}]"


def replace_block(path, name, text):
    start, end = f"<!-- {name} -->", f"<!-- /{name} -->"
    doc = path.read_text()
    a = doc.index(start)
    b = doc.index(end, a) + len(end) if end in doc[a:] else a + len(start)
    path.write_text(doc[:a] + start + "\n\n" + text + "\n\n" + end + doc[b:])


def main():
    fix, bench, surrogate, width = map(
        load, ("fix", "benchmark", "tangtv_surrogate", "bin_sensitivity")
    )
    records = REPO / "data/events/detachment/extend_detach_vote/records"
    model = json.loads((records / "label_model.json").read_text())
    coverage = json.loads((records / "coverage.json").read_text())
    baselines = {name: load(name) for name in ("ours", "victor")}
    states = model["state_counts_bins"]["label_model"]
    pop, certain = fix["population"], fix["certain"]
    lines = [
        (
            f"The survey selects {fix['candidate_selection']['count']} candidates: "
            f"{fix['candidate_selection']['cohort_selected']} cohort shots with real bolo and "
            "real TangTV or Langmuir records, union all local inversions and four explicit "
            "Eldon/Victor examples. Stubs do not count. The exact union and shot IDs are "
            "in [the fix audit](results/detachment_fix.json)."
        ),
        "",
        (
            f"There are {coverage['n_bins']:,} discharge bins on "
            f"{coverage['n_shots_with_bins']} cached live shots. Eligibility requires "
            "at least 20 valid bins in each of two indicators and 20 assessed bins. "
            f"The export assesses {pop['bins']:,} bins on {pop['shots']} eligible shots: "
            f"{states['attached']} attached, {states['detached']} detached, "
            f"{states['marfe']} MARFE and {states['uncertain']:,} uncertain. "
            f"The {certain['bins']} certain bins involve {certain['shots']} shots. "
            "The rule produces the same state counts. "
            "[Coverage](../../data/events/detachment/extend_detach_vote/records/coverage.json), "
            "[model](../../data/events/detachment/extend_detach_vote/records/label_model.json)."
        ),
        "",
        "| Indicator | Valid bins / contributing shots |",
        "|---|---:|",
    ]
    for name, label in (
        ("afrac", "Uncalibrated Jsat ratio (local proxy)"),
        ("prad", "Prad,div local thresholds"),
        ("tangtv", "TangTV inversion"),
    ):
        c = coverage[name]
        lines.append(
            f"| {label} | {c['valid_bins']:,} / {c['shots_with_valid_bins']} |"
        )
    lines += [
        "",
        (
            "These indicator populations include all cached live shots, before export "
            "eligibility. No bin obtained a fitted pre-puff L/H reference; all target-current "
            "ratios therefore retain the local-proxy name. "
            f"The model uses {model['fit']['anchor_bins']} inversion-only anchor bins on "
            f"{model['fit']['anchor_shots']} development shots. TangTV still has a bound "
            "weight solution; it is not evidence of 96% physical accuracy. Mixed versus "
            f"inversion-only anchors change {fix['current_anchor_sensitivity']['changed_bins']} "
            "current states because no surrogate bin supplies an anchor. The old sensitivity "
            "is retained in the appendix and audit."
        ),
        "",
        "| Assessed tier | Bins |",
        "|---|---:|",
    ]
    lines += [f"| `{tier}` | {count:,} |" for tier, count in fix["tiers"].items()]
    lines += [
        "",
        (
            f"All certainty/conflict/MARFE/ELM invariant violations are zero. "
            f"There are {fix['audit']['temporal_imputations']} separate temporal suggestions; "
            "they do not alter observed labels."
        ),
        "",
        "| TangTV source / inversion envelope | All bins | Valid | Candidate MARFE | MARFE votes / labels |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, d in fix["marfe_by_source_envelope"].items():
        lines.append(
            f"| {key.replace('_', ' / ')} | {d['bins']:,} | {d['valid_bins']} | {d['candidate_bins']} | {d['marfe_votes']} / {d['marfe_labels']} |"
        )
    lower = fix["lower_shelf"]
    lines += [
        "",
        (
            "The upper-shelf surrogate training envelope is a deployment restriction "
            "for the surrogate only; true lower-shelf inversions can be outside that "
            "envelope while satisfying their own valid geometry. Domain rows use all "
            "discharge bins, not just eligible export bins."
        ),
        "",
        (
            f"All {lower['shots']} lower-shelf inversion shots were attempted. "
            f"They contribute {lower['bins']} lower-geometry bins; "
            f"{lower['valid']['bins']} bins on {lower['valid']['shots']} shots pass "
            "TangTV validity. Their indicator checks are against the other votes "
            "with the scored vote withheld:"
        ),
        "",
        "| Lower-shelf indicator | Cast comparison bins / shots | Binary kappa [95% shot CI] |",
        "|---|---:|---|",
    ]
    for name, e in lower["loo_other_indicators"].items():
        lines.append(
            f"| {name} | {int(e['n_bins']['value'])} / {e['n_shots']} | {metric(e['binary_kappa'])} |"
        )
    p = fix["prad_development_validation"]
    lines += [
        "",
        (
            f"The fixed radiation thresholds have {p['valid_reference']['bins']} "
            f"valid development inversion bins/{p['valid_reference']['shots']} shots; "
            f"{p['voting_reference']['bins']} bins/{p['voting_reference']['shots']} shots "
            f"cast both binary votes, kappa {p['binary_kappa']:.3f} "
            f"[{p['kappa_ci95'][0]:.3f}, {p['kappa_ci95'][1]:.3f}]. "
            "The cohort-train subset contributes zero usable pairs after the gates; "
            "the check uses the external inversion campaigns already treated as "
            "development training inputs. Val/test are excluded. Thresholds were "
            "fixed before the check, not selected from it. This is diagnostic "
            "agreement, not independently validated classification accuracy."
        ),
        "",
        (
            f"Nested surrogate LOSO uses {surrogate['n_frames']} frames on "
            f"{len(surrogate['training_shots'])} development shots. SAV ZE MAE is "
            f"{surrogate['loso_sav']['ze_mae_cm']:.3f} cm "
            f"[{surrogate['loso_sav']['ci95']['ze_mae_cm'][0]:.3f}, "
            f"{surrogate['loso_sav']['ci95']['ze_mae_cm'][1]:.3f}]. "
            f"Only {len(surrogate['loso_corpus']['shots'])} held-out shot has matched "
            f"corpus inputs ({surrogate['loso_corpus']['n_frames']} frames), with "
            f"ZE MAE {surrogate['loso_corpus']['ze_mae_cm']:.3f} cm and vote kappa "
            f"{surrogate['loso_corpus']['vote_kappa']:.3f}. A single shot cannot "
            "establish transfer accuracy; no new surrogate shot passes the camera "
            "provenance gate. [Nested fit record](results/detachment_tangtv_surrogate.json)."
        ),
        "",
        (
            "Single-indicator leave-one-out comparisons use two compatible, valid "
            "other votes, with posterior threshold 0.7 and a conflict veto. The "
            "TangTV-withheld reference is the low-confidence Jsat/Prad pair, not "
            "a primary certain state. Failure analysis uses these references. "
            "All intervals below are 1000-replicate shot bootstraps."
        ),
        "",
        "| Indicator | Reference bins | Valid reference bins | Cast comparison bins / shots | Binary kappa [95% CI] |",
        "|---|---:|---:|---:|---|",
    ]
    for name, entries in bench["indicators"].items():
        e = entries["loo"]["all"]
        lines.append(
            f"| {name} | {e['reference_certain_bins']} | {e['valid_reference_bins']} | {int(e['n_bins']['value'])} / {e['n_shots']} | {metric(e['binary_kappa'])} |"
        )
    lines += [
        "",
        (
            "[Full benchmark](results/detachment_benchmark.json) includes split and "
            "source strata, all metric denominators and source-specific failure analysis. "
            "Combined-label agreement includes the scored vote and is circular."
        ),
        "",
        "| Baseline / CV reference | Bins / shots | Accuracy [95% CI] | Kappa [95% CI] |",
        "|---|---:|---|---|",
    ]
    for name, r in baselines.items():
        for key, label in (
            ("cv_all_combined", "combined; all inversion"),
            ("cv_inversion_loo_tangtv", "LOO; inversion only"),
            ("cv_all_loo_tangtv", "LOO; all sources"),
        ):
            e = r["stratified"][key]
            lines.append(
                f"| detach-{name}: {label} | {e['n_bins']} / {e['n_shots']} | {metric(e.get('accuracy', {}))} | {metric(e.get('kappa', {}))} |"
            )
    lines += [
        "",
        (
            "The combined-label CV populations are certain labels only. "
            f"For LOO evaluation the full assessed datasets retain "
            f"{baselines['ours']['n_windows']} windows/{baselines['ours']['n_shots']} shots "
            f"and {baselines['victor']['n_frames']} frames/{baselines['victor']['n_shots']} shots. "
            "Surrogate-source combined-label and blind-test combined-label populations "
            "are both empty. There is no MARFE training/evaluation class; its score is "
            "undefined (JSON null), not zero. Fixed-epoch CNNs train only on certain "
            "labels. Normalization is learned within each fold, and majority-class "
            "predictions are selected within each training fold. "
            "[Ours](results/detachment_ours.json), [Victor-style adaptation](results/detachment_victor.json)."
        ),
        "",
        (
            "Blind-test LOO agreement is supported by only "
            f"{baselines['ours']['stratified']['test_all_loo_tangtv']['n_bins']} bins/"
            f"{baselines['ours']['stratified']['test_all_loo_tangtv']['n_shots']} shots for ours "
            f"and {baselines['victor']['stratified']['test_all_loo_tangtv']['n_bins']} bins/"
            f"{baselines['victor']['stratified']['test_all_loo_tangtv']['n_shots']} shot for Victor. "
            "These weak-reference populations do not support a gold-test accuracy claim."
        ),
        "",
        f"Bin sensitivity uses {width['shots_requested']} non-test shots, excluding "
        + ", ".join(map(str, width["excluded_test"]))
        + ". It is descriptive, "
        "not a test-driven choice of bin width.",
        "",
        "| Width (ms) | All bins | Assessed bins / shots | Certain bins / shots |",
        "|---|---:|---:|---:|",
    ]
    for w in (20, 50, 100):
        e = width[f"{w}ms"]
        lines.append(
            f"| {w} | {e['n_bins']:,} | {e['n_assessed_bins']} / {e['n_assessed_shots']} | {e['n_certain_bins']} / {e['n_certain_shots']} |"
        )
    lines += [
        "",
        (
            "[Width record](results/detachment_bin_sensitivity.json). Comparison "
            "agreement applies only to bins certain at both resolutions; it does "
            "not measure preservation of coverage or uncertain intervals."
        ),
    ]
    replace_block(REPO / "docs/labeler/detachment.md", "RESULTS", "\n".join(lines))
    models = [
        (
            f"- detach_vote | 2026_10_03: {pop['bins']} assessed bins/{pop['shots']} shots; "
            f"{certain['bins']} certain/{certain['shots']} shots; {states['marfe']} confirmed MARFE."
        ),
        "- detach_rule | 2026_10_03: the same observed states with transparent compatible-vote support.",
    ]
    for name, r in baselines.items():
        e = r["stratified"]["cv_all_combined"]
        models.append(
            f"- detach-{name} | 2026_10_03: CV {e['n_bins']} bins/{e['n_shots']} shots; "
            f"agreement accuracy {metric(e['accuracy'])}; kappa {metric(e['kappa'])}. "
            "Reference: unverified combined label, all inversion-sourced."
        )
    models += [
        "",
        (
            "These are weak-reference agreement scores. LOO and source-specific "
            "results, bootstrap intervals and exact denominators are in the protocol. "
            "No primary blind-test or MARFE accuracy is available."
        ),
    ]
    replace_block(
        REPO / "data/events/detachment/README.md", "MODELS", "\n".join(models)
    )
    handoff = ROOT / "HANDOFF.md"
    text = handoff.read_text()
    old = "Fix regeneration is in progress. The previous final counts/claims are superseded.\nUse the final coverage and detachment_fix.json after regeneration completes."
    new = (
        f"Fix round complete: {pop['bins']} assessed bins/{pop['shots']} shots, "
        f"{states['attached']} attached, {states['detached']} detached, {states['marfe']} MARFE, "
        f"{states['uncertain']} uncertain. {certain['bins']} certain bins/{certain['shots']} shots. "
        "The previous final counts/claims are superseded. Source records: this worktree's "
        "docs/labeler/results/detachment_fix.json and extend_detach_vote/records/coverage.json."
    )
    if old in text:
        handoff.write_text(text.replace(old, new))
    print("refreshed protocol, README and HANDOFF")


if __name__ == "__main__":
    main()
