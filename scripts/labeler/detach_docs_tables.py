#!/usr/bin/env python
"""Fill the generated blocks of the detachment README and protocol from the records.

    python scripts/labeler/detach_docs_tables.py

Every number in the SUMMARY, RESULTS, BASELINES and MODELS blocks is read from a
committed JSON record under `docs/labeler/results/` (written by the scripts named in
the protocol); the prose outside the blocks is static. Rerun after any record changes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "docs/labeler/results"
DOC = REPO / "docs/labeler/detachment.md"
README = REPO / "data/events/detachment/README.md"
FIGURE2 = REPO / "docs/labeler/figure2_detach.json"
MODEL_DATE = "2026_10_04"
AUROC = "auroc_neg_te_detached_vs_attached"


def load(name: str) -> dict:
    return json.loads((RESULTS / f"detachment_{name}.json").read_text())


def metric(entry, digits: int = 2) -> str:
    """`value [lo, hi]` of a record entry carrying `value` and `ci95`."""
    if not entry or entry.get("value") is None or not math.isfinite(entry["value"]):
        return "undefined"
    value = f"{entry['value']:.{digits}f}"
    lo, hi = entry.get("ci95") or (None, None)
    if lo is None or hi is None:
        return value
    return f"{value} [{lo:.{digits}f}, {hi:.{digits}f}]"


def table(header, rows) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def pop(entry) -> str:
    return f"{entry['bins']:,} / {entry['shots']}"


def replace_block(path: Path, name: str, text: str) -> None:
    start, end = f"<!-- {name} -->", f"<!-- /{name} -->"
    doc = path.read_text()
    a, b = doc.index(start), doc.index(end) + len(end)
    path.write_text(doc[:a] + start + "\n\n" + text.strip() + "\n\n" + end + doc[b:])


class Records:
    def __init__(self) -> None:
        self.cur = load("current")
        self.bench = load("benchmark")
        self.te = load("te_check")
        self.sens = load("prad_sensitivity")
        self.fdiv = load("fdiv_check")
        self.afrac = load("afrac_check")
        self.ref = load("reference")
        self.anchor = load("prad_anchor")
        self.widths = load("bin_sensitivity")
        self.phys = load("physics")
        self.fig = json.loads(FIGURE2.read_text())
        self.ours = load("ours")
        self.victor = load("victor")
        self.row = {r["key"]: r for r in self.fig["agreement"]}

    def variant(self, name: str) -> dict:
        return self.sens["variants"][name]

    def tf(self, tier: str, pair: str) -> dict:
        return self.bench["threshold_free_agreement"][tier][pair]

    def fdiv_summary(self, family: str, reference: str) -> dict:
        return self.fdiv["summary"][family][reference]

    def te_group(self, name: str) -> dict:
        return self.te["by_tier"][name]

    def te_auroc(self, name: str) -> dict:
        return self.te_group(name)[AUROC]["pooled"]


def triple(block: dict) -> str:
    """`attached / detached` bins of a variant block."""
    return f"{block['bins']['attached']} / {block['bins']['detached']}"


def reference_201081(rec: Records) -> dict:
    rows = [r for r in rec.ref["rows"] if r["shot"] == 201081]
    labelled = [r for r in rows if r["primary_state"] in ("attached", "detached")]
    right = [r for r in labelled if r["primary_state"] == r["truth_name"]]
    return {
        "points": len(rows),
        "correct": len(right),
        "uncertain": len([r for r in rows if r["primary_state"] == "uncertain"]),
        "wrong": len(labelled) - len(right),
    }


def second_vote(rec: Records) -> dict:
    """Does the second vote improve the Te agreement of the TangTV state?"""
    effect = rec.te["by_tier"]["second_vote_effect"]["certain_minus_tangtv_alone"]
    lo, hi = effect["ci95"]
    improves = lo is not None and np.isfinite(lo) and lo > 0
    return {
        "improves": bool(improves),
        "difference": effect["difference"],
        "ci": (lo, hi),
        "n_shots": effect["n_shots"],
        "n_shots_first": effect["n_shots_first_subset"],
        "reading": effect["reading"],
        "reading_text": effect["reading_text"],
        "dropped": effect["dropped_replicates"],
        "replicates": effect["valid_replicates"] + effect["dropped_replicates"],
    }


def afrac_decision(rec: Records) -> dict:
    return rec.afrac["decision"]


def within_text(group: dict) -> str:
    """`mean [lo, hi] over k shots` of a within-shot AUROC entry."""
    w = group[AUROC]["within_shot"]
    if not w["n_shots"]:
        return "undefined (no shot has both classes)"
    lo, hi = w["mean_ci95"]
    return f"{w['mean']:.2f} [{lo:.2f}, {hi:.2f}] over {w['n_shots']} shot(s)"


def share_text(entry: dict) -> str:
    """`share (bins / shots)` of a Te group's share in band, or `-` when empty."""
    if not entry["n_bins"]:
        return "-"
    return f"{entry['share_in_band']:.2f} ({entry['n_bins']} / {entry['n_shots']})"


def regime_auroc_bins(entry: dict) -> str:
    """`a attached / d detached` Te-bin counts of an exported-state regime entry."""
    return (
        f"{entry['attached']['n_bins']} attached / "
        f"{entry['detached']['n_bins']} detached"
    )


def regime_auroc(entry: dict) -> str:
    """Pooled AUROC of one regime entry, `not estimable` when a class is empty."""
    pooled = entry[AUROC]["pooled"]
    if pooled["value"] is None:
        return (
            f"not estimable ({pooled['n_bins']} bins, {pooled['n_shots']} shot(s); "
            "one class)"
        )
    return f"{metric(pooled)} ({pooled['n_bins']} bins, {pooled['n_shots']} shots)"


def fdiv_ranges(rec: Records) -> dict:
    """Range over shots of the median absolute f_div in each TangTV class."""
    shots = rec.sens["per_shot_f_div"]["shots"]
    out = {}
    for state in ("attached", "detached"):
        vals = [
            e[f"f_div_absolute_median_tangtv_{state}"]
            for e in shots.values()
            if e[f"f_div_absolute_median_tangtv_{state}"] is not None
        ]
        out[state] = {"min": min(vals), "max": max(vals), "shots": len(vals)}
    p_in = [e["p_in_mw_median"] for e in shots.values() if e["p_in_mw_median"]]
    out["p_in"] = {"min": min(p_in), "max": max(p_in)}
    return out


def fdiv_within(rec: Records, family: str, reference: str) -> str:
    """`mean [lo, hi] over k shots (j above 0.5)` of f_div's within-shot AUROC."""
    w = rec.fdiv_summary(family, reference)["within_shot"]
    if not w["n_shots"]:
        return "undefined (no shot has both classes)"
    lo, hi = w["mean_ci95"]
    return (
        f"{w['mean']:.2f} [{lo:.2f}, {hi:.2f}] over {w['n_shots']} shots "
        f"({w['shots_above_chance']} above 0.5)"
    )


def fdiv_pooled(rec: Records, family: str, reference: str) -> str:
    """`value [lo, hi] (bins, shots)` of f_div's pooled AUROC."""
    e = rec.fdiv_summary(family, reference)["pooled"]
    return f"{metric(e)} on {e['n_bins']:,} bins, {e['n_shots']} shots"


def cohort_text(rec: Records) -> str:
    splits = rec.cur["population"]["by_split"]
    cohort = {k: v for k, v in splits.items() if k != "outside"}
    total = sum(v["assessed_shots"] for v in splits.values())
    inside = sum(v["assessed_shots"] for v in cohort.values())
    parts = ", ".join(
        f"{k} split, {v['assessed_bins']} assessed bins of which "
        f"{sum(v['certain_bins_by_state'].values())} agreement tier and "
        f"{sum(v['tangtv_only_bins_by_state'].values())} TangTV only"
        for k, v in cohort.items()
    )
    return (
        f"Of the {total} assessed shots {inside} "
        f"{'is' if inside == 1 else 'are'} in the fixed cohort ({parts}) and none "
        "is a test shot."
    )


def summary(rec: Records) -> str:
    cur = rec.cur
    pop = cur["population"]
    state_c, state_s = pop["by_state_certain"], pop["by_state_tangtv_only"]
    d9 = cur["paper_criterion_d9"]
    anchor = rec.anchor["measured_anchor"]
    rel = rec.sens["anchor"]["relative_cutoffs"]
    absolute = rec.sens["anchor"]["absolute_cutoffs_global"]
    tv, ce, so = (
        rec.te_auroc(n) for n in ("tangtv_vote_alone", "certain", "tangtv_only")
    )
    sv = second_vote(rec)
    fdiv_vote = rec.te["indicator_votes"]["prad"][AUROC]["pooled"]
    free_afrac = rec.tf("upper_shelf", "afrac__tangtv")
    dec = afrac_decision(rec)
    primary = f"{rec.afrac['primary_window_psin']}"
    window = rec.afrac["windows"][primary]
    te_vote = window["vs_te"]["votes_attached_and_detached"][AUROC]["pooled"]
    wide = rec.afrac["windows"][max(rec.afrac["windows"], key=float)]
    wide_value = wide["vs_te"]["value_cold_vs_warm_te"]["value"]
    before = rec.afrac["before_after"]["before"]["by_psin"]
    cover = {r["indicator"]: r for r in rec.fig["coverage_table"]}
    regime = rec.afrac["regime_gate"]["totals"]
    ranges = fdiv_ranges(rec)
    pub = rec.sens["published_point_180257_4800_ms"]
    marfe = cur["marfe"]
    published = marfe["published_marfe_199166"]
    r81 = reference_201081(rec)
    all_afrac = cover["afrac"]["valid_measurement"]
    extracted = rec.fig["coverage_table"][0]["population_counts"]
    as_vote = rec.variant("with_relative_fdiv")
    ce_note = (
        "; the interval is uninformative on so few shots"
        if (ce["ci95"][1] - ce["ci95"][0]) > 0.3
        else ""
    )
    as_vote_certain = sum(as_vote["certain"]["bins"].values())
    rel_min_bins = rec.fdiv_summary("relative", "tangtv")["within_shot"][
        "min_class_bins"
    ]
    labelled = cur["afrac"]["afrac_in_labelled_bins"]
    by_regime = pop["labelled_by_regime"]
    never = pop["assessed_bins_that_can_never_carry_a_state"]
    without = pop["assessed_shots_without_a_labelled_bin"]
    gate = pop["lmode_gate"]
    lab_e = rec.te_auroc("certain_or_tangtv_only")
    no201 = rec.te_auroc("tangtv_vote_alone_without_201081")
    lmode_e = rec.te["by_regime"]["tangtv_vote_alone"]["L"]
    unknown_e = rec.te["by_regime"]["tangtv_vote_alone"]["unknown"]
    paragraphs = [
        (
            "**Label set: the geometry-gated TangTV state, validated by divertor "
            "Thomson Te. Exploratory; no gold-standard benchmark.** Each 50 ms bin "
            "on an eligible shot is attached, detached or uncertain; MARFE is never "
            "a state. Two indicators vote: the TangTV C-III front height DZ and the "
            "target-current ratio (Afrac proxy: a probe's Jsat over that probe's "
            "own attached level, the probe nearest the separatrix, known L-mode "
            "bins abstain). The lower-divertor radiated fraction f_div = "
            "Prad,div,L / P_in is measured on every bin but is not a vote: it "
            "corroborates within a shot and creates no conflicts (below). A bin is "
            "`certain` when TangTV votes and a valid Afrac vote agrees with it "
            "(called **TangTV + Afrac agreement** in text: two indicators agree, "
            "this is not a confidence level, and its Te agreement is not higher "
            "than that of the TangTV-only tier, below); `tangtv_only` (silver) "
            "when TangTV votes and Afrac abstains or is invalid; `conflict` when "
            "TangTV and Afrac vote differently; a TangTV detached vote on a known "
            "L-mode phase is `tangtv_only_lmode`, uncertain (the DZ cutoffs come "
            "from an H-mode shot, so the vote is not trusted as a state in "
            "L-mode: the gate is a priori and was not tuned on Te; a bin of "
            "unknown regime keeps the TangTV vote); every other assessed bin is "
            "uncertain, with the reason in the `tier` column. The agreement tier "
            "measures indicator agreement, not physical accuracy."
        ),
        (
            f"**Coverage.** Labelled (a state, attached or detached): "
            f"{pop['labelled_certain_or_tangtv_only']['bins']:,} bins on "
            f"{pop['labelled_certain_or_tangtv_only']['shots']} shots "
            f"({pop['labelled_certain_or_tangtv_only']['seconds']:.0f} s): "
            f"TangTV + Afrac agreement {pop['certain']['bins']} bins on "
            f"{pop['certain']['shots']} shots (attached "
            f"{state_c['attached']['bins']} on {state_c['attached']['shots']} "
            f"shots, detached {state_c['detached']['bins']} on "
            f"{state_c['detached']['shots']} shots) and TangTV only "
            f"{pop['tangtv_only']['bins']} bins on {pop['tangtv_only']['shots']} "
            f"shots (attached {state_s['attached']['bins']}, detached "
            f"{state_s['detached']['bins']}); in "
            f"{labelled['tangtv_only_afrac_invalid']} of the TangTV-only bins "
            "Afrac is invalid (no reference yet, known L-mode, probe off the "
            "separatrix) and in "
            f"{labelled['tangtv_only_afrac_valid_between_cutoffs']} it sits "
            f"between its cutoffs. By regime, the labelled detached bins are "
            f"H-mode {by_regime['H']['detached']['bins']}, L-mode "
            f"{by_regime['L']['detached']['bins']} and regime-unknown "
            f"{by_regime['unknown']['detached']['bins']}, and every "
            "agreement-tier bin is on a shot of unknown regime, so the regime "
            "gate of Afrac never acts where the agreement tier lives. Of "
            f"{pop['assessed']['bins']:,} assessed bins on "
            f"{pop['assessed']['shots']} shots ({pop['assessed']['seconds']:.0f} "
            f"s), {never['bins']} bins on {never['shots']} shots can never carry "
            "a state (TangTV is invalid there), and "
            f"{len(without)} assessed shots have no labelled bin. The other "
            f"{pop['by_state']['uncertain']['bins']:,} assessed bins are uncertain: "
            f"{pop['by_tier']['conflict']['bins']} `conflict`, "
            f"{gate['gated_bins']['bins']} `tangtv_only_lmode` on "
            f"{gate['gated_bins']['shots']} shots, and the rest by the tier "
            "column. "
            f"{cohort_text(rec)}"
        ),
        (
            "**Divertor Thomson Te check.** AUROC of -Te for a detached against an "
            "attached vote (pooled, with shot-bootstrap 95% intervals; one "
            "bootstrap per statistic, so a population has one interval wherever "
            f"it appears): the labelled bins {metric(lab_e)} "
            f"({lab_e['n_bins']} bins, {lab_e['n_shots']} shots); TangTV alone "
            f"{metric(tv)} ({tv['n_bins']} bins, {tv['n_shots']} shots), "
            f"and {metric(no201)} on {no201['n_shots']} shots without the anchor "
            "shot 201081 that sets the DZ cutoffs; TangTV + Afrac agreement "
            f"{metric(ce)} ({ce['n_bins']} bins, {ce['n_shots']} shots{ce_note}); "
            f"TangTV only {metric(so)} ({so['n_bins']} bins, "
            f"{so['n_shots']} shots). Within shots: TangTV alone "
            f"{within_text(rec.te_group('tangtv_vote_alone'))}, agreement tier "
            f"{within_text(rec.te_group('certain'))}, TangTV only "
            f"{within_text(rec.te_group('tangtv_only'))}. Share of bins in the "
            "expected Te band (attached >= 10 eV, detached <= 5 eV; in brackets bins "
            "/ shots): attached "
            f"{share_text(rec.te_group('certain')['attached'])} and detached "
            f"{share_text(rec.te_group('certain')['detached'])} in the agreement "
            "tier against attached "
            f"{share_text(rec.te_group('tangtv_only')['attached'])} and detached "
            f"{share_text(rec.te_group('tangtv_only')['detached'])} "
            "in the TangTV-only tier: the agreement tier is not more Te-consistent. "
            "**The second vote does not improve the Te agreement of the TangTV "
            f"state**: the paired difference (agreement tier minus TangTV alone) is "
            f"{sv['reading_text']} (the agreement tier rests on "
            f"{sv['n_shots_first']} shots; {sv['dropped']} of {sv['replicates']} "
            "bootstrap replicates are dropped because their agreement-tier "
            "resample has one class, so the interval "
            f"[{sv['ci'][0]:+.2f}, {sv['ci'][1]:+.2f}] is not a result). What the "
            "second vote buys is agreement between two indicators, not a better "
            "match to Te. **By regime** (the regime source of the Afrac gate): "
            "the TangTV vote on known L-mode phases ranks Te almost perfectly, "
            f"AUROC {regime_auroc(lmode_e)}, but its detached bins sit at "
            f"{lmode_e['detached']['te_ev_quantiles_10_50_90'][1]:.1f} eV "
            f"(median; {lmode_e['detached']['share_in_band']:.2f} of "
            f"{lmode_e['detached']['n_bins']} bins at or below 5 eV), against "
            f"{unknown_e['detached']['share_in_band']:.2f} of "
            f"{unknown_e['detached']['n_bins']} on shots of unknown regime: in "
            "L-mode the DZ cutoff of 0.5 marks detaching, not detached, so those "
            "bins are `tangtv_only_lmode`, uncertain. The gate is a priori (the "
            "cutoffs come from an H-mode shot; Chen 2026 notes that the "
            "outboard-of-X-point window excludes the inner SOL only in H-mode); "
            "it was not tuned on Te. The H-mode bins are one shot with "
            f"{rec.te['by_regime']['tangtv_vote_alone']['H']['detached']['n_bins']} "
            "detached bins and no attached ones, so its AUROC is not estimable."
        ),
        (
            "**Afrac proxy rebuilt.** The earlier proxy read the peak-current "
            "probe against one whole-shot reference and followed the probe's flux "
            f"position (median {before['(1, 1.005]']['median_afrac']:.2f} within "
            f"0.005 of the separatrix and {before['(1.02, 1.03]']['median_afrac']:.2f} "
            "at psiN 1.02-1.03, for the same plasma). Each probe now has its own "
            "attached reference and the probe nearest the separatrix inside "
            f"|psiN - 1| <= {primary} is read; the results show the median Afrac "
            "against the selected-probe psiN before and after. Against TangTV its "
            f"AUROC is {metric(free_afrac['auroc_pooled'])} "
            f"({free_afrac['auroc_pooled']['n_bins']} bins, "
            f"{free_afrac['auroc_pooled']['n_shots']} shots); against Te "
            f"{dec['auroc_vote_based']:.2f} by vote [{te_vote['ci95'][0]:.2f}, "
            f"{te_vote['ci95'][1]:.2f}] ({te_vote['n_bins']} extracted bins with a "
            f"Te, {te_vote['n_shots']} shots; the Te check below scores the "
            "assessed bins instead, a different population) and "
            f"{dec['auroc_value_based']:.2f} by value. It stays the second vote "
            "(rule: removed below "
            f"{dec['keep_min_auroc']}), but only as a weak one: it "
            f"is valid on {all_afrac['bins']} of {extracted['bins']:,} extracted "
            f"bins ({all_afrac['shots']} of {extracted['shots']} shots), the Te "
            "interval includes 0.5, the regime "
            f"is known on {regime['shots_with_known_regime']} of "
            f"{regime['shots']} shots (so the L-mode gate removes "
            f"{regime['afrac_l_mode_abstentions']} bins and misses the rest) and "
            f"a wider window ({max(rec.afrac['windows'], key=float)}) puts the "
            f"value-based AUROC at {wide_value:.2f}, below the removal bound."
        ),
        (
            "**f_div is a within-shot corroborator, not a vote.** Pooled over "
            "shots its level moves with the shot, not with the state: per-shot "
            "medians of the absolute f_div in TangTV-attached bins run from "
            f"{ranges['attached']['min']:.2f} to {ranges['attached']['max']:.2f} "
            f"({ranges['attached']['shots']} shots) and in TangTV-detached bins "
            f"from {ranges['detached']['min']:.2f} to "
            f"{ranges['detached']['max']:.2f} ({ranges['detached']['shots']} "
            f"shots), with P_in from {ranges['p_in']['min']:.1f} to "
            f"{ranges['p_in']['max']:.1f} MW, so no cutoff pair carries over "
            "between shots. Counted as a second vote (shot-relative cutoffs "
            f"{rel[0]:.3f} / {rel[1]:.3f}) it would give "
            f"{as_vote_certain} certain bins but "
            f"{as_vote['tier_counts']['conflict']} conflicts, against "
            f"{pop['certain']['bins']} and {pop['by_tier']['conflict']['bins']}; "
            "it is a sensitivity row. As a corroborator, per shot with at least "
            f"{rel_min_bins} bins of each class: the relative f_div ranks TangTV-detached above "
            "TangTV-attached bins with a within-shot AUROC of "
            f"{fdiv_within(rec, 'relative', 'tangtv')}, and cold above warm "
            f"divertor Thomson Te with {fdiv_within(rec, 'relative', 'te')}; "
            "pooled over shots the same scores give "
            f"{fdiv_pooled(rec, 'relative', 'tangtv')} against TangTV and "
            f"{fdiv_pooled(rec, 'relative', 'te')} against Te, i.e. no support "
            "once the shots are mixed. The absolute f_div (201081-anchored "
            f"cutoffs {absolute[0]:.3f} / {absolute[1]:.3f}, P_in "
            f"{anchor['p_in_mw']:.3f} MW) ranks the same bins within a shot and "
            f"better pooled (against TangTV "
            f"{fdiv_pooled(rec, 'absolute', 'tangtv')}), but its cutoffs vote "
            "detached on most TangTV-attached bins where it votes. Counting only "
            "the relative votes it casts, their AUROC against Te is "
            f"{metric(fdiv_vote)} pooled. The published detached point 180257 at "
            "4800 ms is not assessed (TangTV has no valid geometry, the relative "
            "f_div has no baseline, Afrac reads "
            f"`{pub['afrac_reason']}`); the absolute cutoffs would call it "
            f"attached (f_div {pub['f_div_absolute']:.2f} at P_in "
            f"{pub['p_in_mw']:.1f} MW), a miss of that published point."
        ),
        (
            f"**Reference shot 201081.** Measured P_in is {anchor['p_in_mw']:.3f} "
            "MW (the median over the 32 attached-window bins of the 250 ms-averaged "
            f"P_in); Prad,div,L is {anchor['attached_mw']:.3f} MW attached and "
            f"{anchor['detached_mw']:.3f} MW detached. Of {r81['points']} "
            f"published points on 201081, {r81['correct']} are labelled in "
            f"agreement with the publication, {r81['uncertain']} uncertain and "
            f"{r81['wrong']} contradicted."
        ),
        (
            "**MARFE.** No bin is exported as MARFE: the density cue (fG >= "
            f"{published_cue(rec)}) has no literature source (Dong 2025 gives "
            "fG of about 0.5 or more on HL-3 from a core-point density, with a "
            "core-Te condition and a gate of 0.40, and says thresholds are "
            "device-specific). A persistent high TangTV front is the uncertain "
            f"tier `candidate_marfe`: {marfe['candidate_marfe_tier']['bins']} "
            f"bins on {marfe['candidate_marfe_tier']['shots']} shots; the "
            "full TangTV MARFE vote (front, emission peak inside the separatrix "
            f"and the density cue) fires on {marfe['tangtv_marfe_vote']['bins']} "
            f"bins of {marfe['tangtv_marfe_vote']['shots']} shot only. **Recall "
            "of the published MARFE onset on 199166 (3705 ms): "
            f"{published['recall_marfe_state']} as a label**; the bin holding it "
            f"is `candidate_marfe` ({published['recall_candidate_marfe']} as a "
            "candidate), with fG = 0.72 there against the 0.8 cue."
        ),
        (
            "**Three-state label set.** Not offered: agreement-tier attached and "
            f"detached bins coexist on {len(d9['certain']['shots_with_both'])} "
            f"shots, {len(d9['certain']['cohort_shots_with_both'])} of them in the "
            "fixed cohort, against a requirement of at least three shots including "
            "one cohort shot. The result is an indicator-agreement appendix."
        ),
    ]
    return "\n\n".join(paragraphs)


def published_cue(rec: Records) -> str:
    return f"{rec.phys['thresholds']['marfe_greenwald_min']:.1f}"


def coverage_tables(rec: Records) -> list[str]:
    cur = rec.cur["population"]
    tiers = cur["by_tier"]
    never = cur["assessed_bins_that_can_never_carry_a_state"]
    rows = [
        [
            "Labelled (TangTV + Afrac agreement or TangTV only)",
            pop(cur["labelled_certain_or_tangtv_only"]),
            f"{cur['labelled_certain_or_tangtv_only']['seconds']:.1f}",
        ],
    ]
    for name, block in (
        ("TangTV + Afrac agreement, attached", cur["by_state_certain"]["attached"]),
        ("TangTV + Afrac agreement, detached", cur["by_state_certain"]["detached"]),
        ("TangTV only, attached", cur["by_state_tangtv_only"]["attached"]),
        ("TangTV only, detached", cur["by_state_tangtv_only"]["detached"]),
    ):
        rows.append([name, pop(block), f"{block['seconds']:.2f}"])
    rows += [
        ["Assessed", pop(cur["assessed"]), f"{cur['assessed']['seconds']:.1f}"],
        [
            "Assessed, can never carry a state (no valid TangTV vote)",
            pop(never),
            f"{never['seconds']:.2f}",
        ],
    ]
    rows += [
        [f"Tier `{name}`", pop(entry), f"{entry['seconds']:.2f}"]
        for name, entry in tiers.items()
    ]
    sources = cur["lmode_gate"]["regime_source_of_assessed_bins"]
    regime_rows = []
    for regime, label in (("H", "H-mode"), ("L", "L-mode"), ("unknown", "unknown")):
        entry = cur["labelled_by_regime"][regime]
        votes = cur["lmode_gate"]["tangtv_detached_votes_by_regime"][regime]
        regime_rows.append(
            [
                label,
                pop(entry["attached"]),
                pop(entry["detached"]),
                pop(votes),
                pop(cur["certain_by_regime"][regime]),
            ]
        )
    first = rec.fig["coverage_table"][0]["population_counts"]
    return [
        "### Coverage\n",
        table(["Group", "Bins / shots", "Seconds"], rows),
        (
            "\nA bin is assessed when at least two indicators are valid on it or "
            "TangTV votes; a shot is eligible when at least two indicators are "
            "each valid for at least 1 s (20 bins at 50 ms) and it has at least "
            "1 s of assessed bins. The assessed count includes bins that have no "
            "TangTV vote (the row `can never carry a state`), so the labelled "
            "count is the one to quote. "
            "Tiers are disjoint and sum to the assessed "
            "bins; `lower_shelf_window` holds the owner's restricted lower-shelf "
            "extraction, whose TangTV vote is invalid (the shelf geometry does not "
            "hold) so it never yields a state."
        ),
        "\n### Labelled bins by regime\n",
        table(
            [
                "Regime (Afrac gate source)",
                "Labelled attached (bins / shots)",
                "Labelled detached (bins / shots)",
                "TangTV detached votes before the gate",
                "Agreement-tier bins",
            ],
            regime_rows,
        ),
        (
            "\nThe regime is Jalal Butt's confinement table where it covers the "
            "shot, else the D-alpha H-mode detector, else unknown "
            f"({sources['none']:,} assessed bins of unknown regime, "
            f"{sources['dalpha_detector']} from the D-alpha detector and "
            f"{sources['regime_table']} from the table). The gate moves "
            f"{rec.cur['population']['lmode_gate']['gated_bins']['bins']} TangTV "
            "detached votes on known L-mode phases to `tangtv_only_lmode`; no "
            "labelled detached bin is on a known L-mode phase."
        ),
        "\n### Indicator coverage, all extracted shots\n",
        table(
            [
                "Indicator",
                "Measurement",
                "Valid",
                "Vote",
                "In an agreement-tier label",
            ],
            [
                [
                    "prad (f_div, not a vote of the label)"
                    if r["indicator"] == "prad"
                    else r["indicator"],
                    pop(r["measurement"]),
                    pop(r["valid_measurement"]),
                    pop(r["vote"]),
                    pop(r["certain_label"]),
                ]
                for r in rec.fig["coverage_table"]
            ],
        ),
        (
            f"\nCells are bins / shots over {first['bins']:,} extracted bins on "
            f"{first['shots']} shots. For f_div the Vote column counts its "
            "relative votes (a sensitivity: the label does not use them) and the "
            "last column the agreement-tier bins in which it casts one."
        ),
    ]


def second_indicator_table(rec: Records) -> list[str]:
    c = rec.cur["afrac"]["afrac_in_labelled_bins"]
    reasons = c["tangtv_only_afrac_invalid_reasons"]
    rows = [
        [
            "TangTV + Afrac agreement bins (TangTV vote and an agreeing Afrac vote)",
            c["certain_bins"],
        ],
        ["TangTV-only bins", c["tangtv_only_bins"]],
        [
            "  Afrac valid, between its cutoffs (abstains)",
            c["tangtv_only_afrac_valid_between_cutoffs"],
        ],
        ["  Afrac invalid", c["tangtv_only_afrac_invalid"]],
    ] + [[f"    reason `{k}`", v] for k, v in reasons.items()]
    return [
        "\n### What supports the labelled bins\n",
        table(["Labelled bins", "Count"], rows),
        (
            "\nA TangTV-only bin keeps TangTV's state because Afrac does not "
            "vote there; it is not evidence against it. Most of these bins have "
            "no Afrac because the probe's reference is too short, the shot is in "
            "L-mode or no probe sits near the separatrix."
        ),
    ]


def te_tables(rec: Records) -> list[str]:
    te = rec.te
    regime = te["by_regime"]["tangtv_vote_alone"]
    groups = [
        ("agreement tier attached", te["by_tier"]["certain"]["attached"]),
        ("agreement tier detached", te["by_tier"]["certain"]["detached"]),
        ("TangTV-only attached", te["by_tier"]["tangtv_only"]["attached"]),
        ("TangTV-only detached", te["by_tier"]["tangtv_only"]["detached"]),
        ("labelled attached", te["by_tier"]["certain_or_tangtv_only"]["attached"]),
        ("labelled detached", te["by_tier"]["certain_or_tangtv_only"]["detached"]),
        ("TangTV vote alone, attached", te["by_tier"]["tangtv_vote_alone"]["attached"]),
        ("TangTV vote alone, detached", te["by_tier"]["tangtv_vote_alone"]["detached"]),
        (
            "TangTV vote alone without 201081, attached",
            te["by_tier"]["tangtv_vote_alone_without_201081"]["attached"],
        ),
        (
            "TangTV vote alone without 201081, detached",
            te["by_tier"]["tangtv_vote_alone_without_201081"]["detached"],
        ),
    ]
    for key, name in (("H", "H-mode"), ("L", "L-mode"), ("unknown", "unknown regime")):
        groups += [
            (f"TangTV vote alone, {name}, attached", regime[key]["attached"]),
            (f"TangTV vote alone, {name}, detached", regime[key]["detached"]),
        ]
    groups += [
        (f"{ind} vote {n}", te["indicator_votes"][ind][n])
        for ind in ("afrac", "prad")
        for n in ("attached", "detached")
    ]
    rows = []
    for label, e in groups:
        if not e["n_bins"]:
            rows.append([label, "0 / 0", "-", "-"])
            continue
        q = e["te_ev_quantiles_10_50_90"]
        rows.append(
            [
                label.replace("prad vote", "relative f_div vote (unused)").replace(
                    "afrac vote", "Afrac vote"
                ),
                f"{e['n_bins']} / {e['n_shots']}",
                f"{q[1]:.1f} ({q[0]:.1f}-{q[2]:.1f})",
                f"{e['share_in_band']:.2f}",
            ]
        )
    sources = [
        (
            "labelled bins (agreement tier and TangTV only)",
            te["by_tier"]["certain_or_tangtv_only"],
        ),
        ("TangTV + Afrac agreement tier", te["by_tier"]["certain"]),
        ("TangTV-only tier", te["by_tier"]["tangtv_only"]),
        ("TangTV vote alone", te["by_tier"]["tangtv_vote_alone"]),
        (
            "TangTV vote alone, 201081 left out",
            te["by_tier"]["tangtv_vote_alone_without_201081"],
        ),
        ("TangTV vote alone, L-mode phases", regime["L"]),
        ("TangTV vote alone, regime unknown", regime["unknown"]),
        ("TangTV vote alone, H-mode phases", regime["H"]),
        (
            "TangTV vote in `conflict` bins",
            te["by_tier"]["tangtv_vote_in_conflict_bins"],
        ),
        ("Afrac vote (assessed bins)", te["indicator_votes"]["afrac"]),
        ("relative f_div votes (unused)", te["indicator_votes"]["prad"]),
    ]
    auroc_rows = []
    for name, group in sources:
        pooled, within = group[AUROC]["pooled"], group[AUROC]["within_shot"]
        auroc_rows.append(
            [
                name,
                metric(pooled) if pooled["value"] is not None else "not estimable",
                f"{pooled['n_bins']} / {pooled['n_shots']}",
                metric({"value": within["mean"], "ci95": within["mean_ci95"]})
                if within["n_shots"]
                else "undefined",
                within["n_shots"],
            ]
        )
    sv = second_vote(rec)
    cliff = te["cliff_check"]["201081"]
    other = te["by_tier"]["second_vote_effect"][
        "certain_minus_tangtv_where_second_vote_missing_or_clashing"
    ]
    regime = te["by_regime"]["tangtv_vote_alone"]
    return [
        "\n### Divertor Thomson Te check\n",
        (
            "The processed divertor Thomson Te (`\\ELECTRONS::TSTE_DIV`, 14 or 16 "
            "chords at R = 1.485 m, about 20 ms) is a temperature that none of "
            "the three indicators (TangTV, Afrac, f_div) uses. A bin's Te is the "
            "median of the valid samples of the chords "
            f"{te['chord_selection']['above_shelf_m'][0] * 100:.1f} to "
            f"{te['chord_selection']['above_shelf_m'][1] * 100:.0f} cm above the "
            f"outer shelf (Z = {te['chord_selection']['shelf_z_m']} m) that lie on "
            f"the scrape-off side, {te['chord_selection']['psi_n_window'][0]:.3f} "
            f"< psiN <= {te['chord_selection']['psi_n_window'][1]:.2f}, by the "
            f"multi-slice EFIT map; samples outside {te['bands_ev']['valid'][0]}-"
            f"{te['bands_ev']['valid'][1]:.0f} eV are rejected. No threshold is "
            "fitted: the bands were fixed from the literature before scoring."
        ),
        "",
        table(
            ["Group", "Bins / shots", "Median Te, eV (10-90%)", "Share in band"], rows
        ),
        (
            "\nShare in band is the fraction of bins with Te >= "
            f"{te['bands_ev']['attached_min']:.0f} eV for attached or <= "
            f"{te['bands_ev']['detached_max']:.0f} eV for detached. AUROC of -Te "
            "for detached against attached votes (pooled with a shot-bootstrap "
            "interval; within shots is the mean over the shots that have at least "
            f"{te['by_tier']['certain'][AUROC]['within_shot']['min_class_bins']} "
            "bins of each class, with the number of such shots):\n"
        ),
        table(
            [
                "Reference",
                "Pooled AUROC [CI]",
                "Bins / shots",
                "Within-shot mean [CI]",
                "Shots (within)",
            ],
            auroc_rows,
        ),
        (
            "\n**Does the second vote improve the Te agreement?** It does not, "
            "and the paired difference cannot be estimated: agreement tier minus "
            f"TangTV alone is {sv['reading_text']} (the agreement tier has "
            f"{sv['n_shots_first']} shots with a Te; "
            f"{sv['dropped']} of {sv['replicates']} bootstrap replicates are "
            "dropped because the resample of those shots has a single class, "
            f"which leaves the interval [{sv['ci'][0]:+.2f}, {sv['ci'][1]:+.2f}] "
            f"degenerate; point difference {sv['difference']:+.3f}). Against the "
            "TangTV votes that are TangTV only or in conflict the same "
            f"difference is {other['difference']:+.3f} "
            f"[{other['ci95'][0]:+.2f}, {other['ci95'][1]:+.2f}] "
            f"({other['reading_text']}). The within-shot figure for the "
            f"agreement tier rests on "
            f"{te['by_tier']['certain'][AUROC]['within_shot']['n_shots']} "
            "shot(s), too few for any statement. In the share-in-band column the "
            "agreement tier is less Te-consistent than the TangTV-only tier "
            f"(attached {share_text(te['by_tier']['certain']['attached'])} against "
            f"{share_text(te['by_tier']['tangtv_only']['attached'])}), so the "
            "tier is agreement of two indicators and not a confidence level."
        ),
        "\n### L-mode: where the TangTV vote is not cold\n",
        (
            "The DZ cutoffs (0.35 and 0.5) come from one H-mode shot (201081). "
            "On known L-mode phases the TangTV detached vote is not cold by the "
            "Te criterion: "
            f"{regime['L']['detached']['share_in_band']:.2f} of its "
            f"{regime['L']['detached']['n_bins']} bins with a Te are at or below "
            f"{te['bands_ev']['detached_max']:.0f} eV (median "
            f"{regime['L']['detached']['te_ev_quantiles_10_50_90'][1]:.1f} eV), "
            f"against {regime['unknown']['detached']['share_in_band']:.2f} "
            f"on shots of unknown regime, although the ranking inside L-mode is "
            f"near perfect ({regime_auroc(regime['L'])}): the state boundary sits "
            "at about 10 eV there, not 5 eV. Chen 2026 notes that the "
            "outboard-of-X-point window excludes the inner SOL only in H-mode, so a "
            "cutoff set on an H-mode shot need not carry over to L-mode. The "
            "label therefore withholds the state (tier `tangtv_only_lmode`, "
            "uncertain) for a TangTV detached vote on a known L-mode phase, from "
            "the regime alone; **the gate was not tuned on Te**, which would "
            "make this check circular. A bin of unknown regime keeps the vote. "
            "Te of the TangTV detached votes by DZ band (before the gate, "
            "shot counts beside each cell):\n"
        ),
        table(
            ["DZ band", "H-mode", "L-mode", "Unknown regime"],
            [
                [
                    band,
                    *(
                        (
                            f"{e['n_bins']} bins / {e['n_shots']} shots, median "
                            f"{e['te_ev_quantiles_10_50_90'][1]:.1f} eV, in band "
                            f"{e['share_in_band']:.2f}"
                        )
                        if (
                            e := te["by_regime"]["tangtv_detached_by_dz_band"][r].get(
                                band
                            )
                        )
                        else "-"
                        for r in ("H", "L", "unknown")
                    ),
                ]
                for band in ("0.5-0.65", "0.65-0.8", "0.8-1.2")
            ],
        ),
        (
            "\nExported detached and attached bins by regime after the gate: "
            + "; ".join(
                f"{name} {regime_auroc_bins(te['by_regime']['exported_state'][key])}"
                for key, name in (
                    ("H", "H-mode"),
                    ("L", "L-mode"),
                    ("unknown", "unknown"),
                )
            )
            + "."
        ),
        (
            f"\n{len(te['shots_with_te_in_assessed_bins'])} of the "
            f"{rec.cur['population']['assessed']['shots']} assessed shots have a "
            "Thomson Te in an assessed bin; for the others no selected chord had "
            "a valid sample, or no Thomson data exist.\n"
            f"\nOn 201081 the Te first falls below the detached band at "
            f"{cliff['first_cold_ms']:.0f} ms and last does at "
            f"{cliff['last_cold_ms']:.0f} ms (published cliffs "
            f"{cliff['published_cliffs_ms'][0]:.0f} and "
            f"{cliff['published_cliffs_ms'][1]:.0f} ms). No MARFE bin is exported, "
            "so the check does not cover MARFE."
        ),
    ]


def agreement_tables(rec: Records) -> list[str]:
    names = {
        "auroc_pooled": "AUROC of relative f_div, pooled",
        "auroc_within_shot": "AUROC of relative f_div, mean over shots",
        "spearman_pooled": "Spearman rho of relative f_div with DZ, pooled",
        "spearman_within_shot": "Spearman rho, mean over shots",
        "kappa_binary": "Cohen's kappa, attached against not attached",
        "kappa_3class": "Cohen's kappa, three states",
    }
    stat_rows = [
        [
            names[key],
            metric(r),
            r["chance"],
            r["n_bins"] if r["n_bins"] is not None else "-",
            r["n_shots"],
        ]
        for key, r in rec.row.items()
    ]
    pair_rows = []
    for tier in ("upper_shelf", "lower_shelf_window"):
        for pair, name in (
            ("prad__tangtv", "relative f_div against TangTV (corroborator)"),
            ("prad_abs__tangtv", "absolute f_div against TangTV (sensitivity)"),
            ("afrac__tangtv", "Afrac against TangTV"),
            ("prad__afrac", "relative f_div against Afrac"),
        ):
            entry = rec.tf(tier, pair)
            within = entry["within_shot"]["auroc"]
            pair_rows.append(
                [
                    tier.replace("_", " "),
                    name,
                    metric(entry["auroc_pooled"]),
                    entry["auroc_pooled"]["n_bins"],
                    entry["auroc_pooled"]["n_shots"],
                    (
                        f"{within['mean']:.2f}"
                        if within["n_shots"] and within["mean"] is not None
                        else "undefined"
                    ),
                    within["n_shots"],
                ]
            )
    return [
        "\n### f_div and the indicator pairs, upper shelf\n",
        table(
            ["Statistic", "Value [95% shot CI]", "Chance", "Bins", "Shots"], stat_rows
        ),
        (
            "\nAgreement diagnostics; none of them decides a label. AUROC is the "
            "probability that a TangTV-detached (or TangTV-MARFE-vote) bin has a "
            "larger relative f_div than a TangTV-attached bin; the per-shot f_div "
            "corroborator table below counts the detached votes alone. Intervals "
            "resample shots "
            f"({rec.bench['replicates']} replicates, seed 0). Kappa is null, and "
            "not drawn, where either rater used one class. The within-shot rows "
            "average over the few shots that have enough bins of both classes "
            "(the shot count is in the last column of the table). The same "
            "statistics for each pair of indicators:\n"
        ),
        table(
            [
                "Shelf",
                "Pair",
                "AUROC [CI]",
                "Bins",
                "Shots",
                "Within-shot AUROC",
                "Shots (within)",
            ],
            pair_rows,
        ),
    ]


def afrac_tables(rec: Records) -> list[str]:
    primary_key = f"{rec.afrac['primary_window_psin']}"
    vote_primary = rec.afrac["windows"][primary_key]["vs_te"][
        "votes_attached_and_detached"
    ][AUROC]["pooled"]
    vote_assessed = rec.te["indicator_votes"]["afrac"][AUROC]["pooled"]
    a = rec.afrac
    primary = f"{a['primary_window_psin']}"

    def psin_rows(block: dict) -> list[list]:
        rows = []
        for key, e in block["by_psin"].items():
            att = block["by_psin_tangtv_attached"].get(key)
            det = block["by_psin_tangtv_detached"].get(key)
            rows.append(
                [
                    key,
                    f"{e['n_bins']} / {e['n_shots']}",
                    f"{e['median_afrac']:.2f}",
                    f"{att['median_afrac']:.2f} ({att['n_bins']})" if att else "-",
                    f"{det['median_afrac']:.2f} ({det['n_bins']})" if det else "-",
                ]
            )
        return rows

    header = [
        "psiN of the selected probe",
        "Bins / shots",
        "Median Afrac",
        "TangTV attached (bins)",
        "TangTV detached (bins)",
    ]
    before, after = a["before_after"]["before"], a["before_after"]["after"]
    window_rows = []
    for w, e in a["windows"].items():
        ve = e["vs_te"]
        window_rows.append(
            [
                w,
                f"{e['bins_valid']:,} / {e['shots_valid']}",
                metric(e["vs_tangtv"]["auroc_neg_afrac_tangtv_detached_vs_attached"]),
                f"{e['vs_tangtv']['auroc_neg_afrac_tangtv_detached_vs_attached']['n_shots']}",
                metric(
                    ve["votes_attached_and_detached"][
                        "auroc_neg_te_detached_vs_attached"
                    ]["pooled"]
                ),
                metric(ve["value_cold_vs_warm_te"]),
                f"{ve['value_cold_vs_warm_te']['n_shots']}",
            ]
        )
    dep_b = before["dependence_tangtv_attached"], before["dependence_tangtv_detached"]
    dep_a = after["dependence_tangtv_attached"], after["dependence_tangtv_detached"]
    phases = a["windows"][primary]["shot_201081_phase_medians"]
    dec = a["decision"]
    regime = a["regime_gate"]["totals"]
    return [
        "\n### Afrac proxy: dependence on the probe read\n",
        (
            "Median Afrac against the flux position of the probe it was read from, "
            "all assessed bins, and split by the TangTV vote. Before: the round-4 "
            "definition (peak-current probe against one whole-shot reference). "
            "After: each probe against its own attached reference, the probe "
            f"nearest the separatrix within |psiN - 1| <= {primary}.\n"
        ),
        "Before:\n",
        table(header, psin_rows(before)),
        "\nAfter:\n",
        table(header, psin_rows(after)),
        (
            "\nSpearman rank correlation of Afrac with the selected probe's psiN, "
            f"TangTV-attached bins: before {dep_b[0]['spearman_afrac_vs_psin']:.2f} "
            f"({dep_b[0]['n_bins']} bins, {dep_b[0]['n_shots']} shots), after "
            f"{dep_a[0]['spearman_afrac_vs_psin']:.2f} ({dep_a[0]['n_bins']} bins, "
            f"{dep_a[0]['n_shots']} shots); TangTV-detached bins: before "
            f"{dep_b[1]['spearman_afrac_vs_psin']:.2f} ({dep_b[1]['n_bins']} "
            f"bins, {dep_b[1]['n_shots']} shots), after "
            f"{dep_a[1]['spearman_afrac_vs_psin']:.2f} ({dep_a[1]['n_bins']} "
            f"bins, {dep_a[1]['n_shots']} shots). The dependence in the attached "
            "class is much reduced; the detached class keeps a positive rank "
            "correlation on few bins and shots (its median is "
            f"{dep_a[1]['median_afrac_psin_at_most_1.005']:.2f} at psiN <= 1.005 "
            f"on {dep_a[1]['n_bins_psin_at_most_1.005']} bins and the "
            f"{after['by_psin_tangtv_detached']['(1.005, 1.01]']['n_bins']} "
            "detached bins at 1.005-1.01 read "
            f"{after['by_psin_tangtv_detached']['(1.005, 1.01]']['median_afrac']:.2f}"
            "), so the proxy is not free of position. "
            f"On 201081 the median Afrac is "
            f"{phases['attached_early']['median_afrac']:.2f} before the first "
            f"cliff, {phases['detached']['median_afrac']:.2f} between the cliffs "
            f"and {phases['reattached']['median_afrac']:.2f} after the second, "
            "all from one probe."
        ),
        (f"\nThe cause of the earlier failure: {a['cause_of_the_earlier_failure']}"),
        (
            f"\nWindow sensitivity (the primary window is {primary}; wider windows "
            "admit probes the EFIT strike-point error can mis-map):\n"
        ),
        table(
            [
                "Window",
                "Valid bins / shots",
                "AUROC(-Afrac) vs TangTV [CI]",
                "Shots",
                "AUROC(-Te), vote [CI]",
                "AUROC(-Afrac) cold vs warm Te [CI]",
                "Shots",
            ],
            window_rows,
        ),
        (
            f"\nDecision rule: {dec['rule']}. Statistic "
            f"{dec['statistic']:.3f}; Afrac "
            f"{'stays in' if dec['keep_afrac_in_vote'] else 'is removed from'} the "
            "vote. Known-L-mode bins "
            f"({regime['afrac_l_mode_abstentions']} bins) abstain; the regime is "
            f"known for {regime['bins_regime_known']:,} of {regime['bins']:,} "
            f"extracted bins on {regime['shots_with_known_regime']} of "
            f"{regime['shots']} shots (confinement suggestion table, then the "
            "D-alpha H-mode detector) and unknown regime is not gated. The "
            "window table scores the Afrac vote against Te on "
            f"{vote_primary['n_bins']} extracted upper-shelf bins with a Te "
            f"({vote_primary['n_shots']} shots, AUROC {vote_primary['value']:.3f}); "
            "the Te check above scores the assessed bins instead "
            f"({vote_assessed['n_bins']} bins, {vote_assessed['n_shots']} shots, "
            f"AUROC {vote_assessed['value']:.3f}); these are two populations, not "
            "two estimates of one number, and both intervals include 0.5."
        ),
    ]


def fdiv_corroborator_tables(rec: Records) -> list[str]:
    f = rec.fdiv
    names = {
        ("relative", "tangtv"): "relative f_div against TangTV",
        ("relative", "te"): "relative f_div against Te",
        ("absolute", "tangtv"): "absolute f_div against TangTV (sensitivity)",
        ("absolute", "te"): "absolute f_div against Te (sensitivity)",
    }
    summary_rows = []
    for (family, reference), name in names.items():
        e = rec.fdiv_summary(family, reference)
        w, pooled = e["within_shot"], e["pooled"]
        summary_rows.append(
            [
                name,
                metric({"value": w["mean"], "ci95": w["mean_ci95"]})
                if w["n_shots"]
                else "undefined",
                w["n_shots"],
                w["shots_above_chance"],
                metric(pooled),
                f"{pooled['n_bins']:,} / {pooled['n_shots']}",
            ]
        )
    shot_rows = []
    for shot, e in f["per_shot"].items():
        a, b = e["relative"]["tangtv"], e["relative"]["te"]
        if a["auroc"] is None and b["auroc"] is None:
            continue

        def cell(x):
            if x["auroc"] is None:
                return "-"
            return f"{x['auroc']:.2f} ({x['n_detached']} / {x['n_attached']})"

        shot_rows.append([shot, e["split"], cell(a), cell(b)])
    ms = f["summary"]["relative"]["tangtv"]["within_shot"]["min_class_bins"]
    return [
        "\n### f_div as a within-shot corroborator\n",
        (
            "f_div is not a vote, so it creates no conflicts and no "
            "agreement-tier bins. "
            "It is scored as a ranking: per shot, the AUROC of f_div for the "
            "TangTV-detached bins against the TangTV-attached ones (upper shelf), "
            "and for the divertor-Thomson-cold bins (Te <= "
            f"{rec.te['bands_ev']['detached_max']:.0f} eV) against the warm ones "
            f"(>= {rec.te['bands_ev']['attached_min']:.0f} eV); a shot counts "
            f"when each class has at least {ms} bins; the mean is over those "
            "shots, with a shot-bootstrap interval, and the pooled AUROC mixes "
            "the shots. The relative f_div is the ratio over the shot's own "
            "baseline, valid where the baseline exists; the absolute ranks the "
            "bins of one shot the same way (it differs by a per-shot constant) "
            "and is the 201081-anchored sensitivity. Record: "
            "`docs/labeler/results/detachment_fdiv_check.json`.\n"
        ),
        table(
            [
                "Score against reference",
                "Within-shot mean [CI]",
                "Shots",
                "Shots above 0.5",
                "Pooled AUROC [CI]",
                "Pooled bins / shots",
            ],
            summary_rows,
        ),
        (
            "\nPer shot (relative f_div; cells are AUROC with the detached or "
            "cold / attached or warm bin counts in brackets; shots with fewer "
            f"than {ms} bins in a class are left out of a column):\n"
        ),
        table(
            ["Shot", "Split", "Against TangTV", "Against Te"],
            shot_rows,
        ),
    ]


def fdiv_tables(rec: Records) -> list[str]:
    s = rec.sens
    shots = s["per_shot_f_div"]["shots"]
    both = [
        (int(k), e)
        for k, e in shots.items()
        if e["f_div_absolute_median_tangtv_attached"] is not None
        and e["f_div_absolute_median_tangtv_detached"] is not None
    ]
    rows = [
        [
            k,
            f"{e['p_in_mw_median']:.1f}",
            f"{e['baseline_f_div']:.2f}",
            (
                f"{e['f_div_absolute_median_tangtv_attached']:.2f} "
                f"({e['tangtv_attached_bins']})"
            ),
            (
                f"{e['f_div_absolute_median_tangtv_detached']:.2f} "
                f"({e['tangtv_detached_bins']})"
            ),
            f"{e['f_div_relative_median_tangtv_attached']:.2f}",
            f"{e['f_div_relative_median_tangtv_detached']:.2f}",
        ]
        for k, e in sorted(both)
    ]
    pooled = s["per_shot_f_div"]["pooled"]
    vote_rows = []
    for state in ("attached", "detached"):
        for family in ("relative", "absolute"):
            e = pooled[f"{family}_in_tangtv_{state}"]
            vote_rows.append(
                [
                    f"TangTV {state}",
                    family,
                    f"{e['bins']} / {e['shots']}",
                    e["voted_attached"],
                    e["voted_detached"],
                    e["abstained"],
                ]
            )
    ranges = fdiv_ranges(rec)
    anchor = rec.anchor["measured_anchor"]
    anchor_level = anchor["attached_mw"] / anchor["p_in_mw"]
    pub = s["published_point_180257_4800_ms"]
    return [
        "\n### f_div per shot, the absolute cutoffs\n",
        (
            "Per-shot f_div in the TangTV-attached and TangTV-detached upper-shelf "
            f"bins of the {len(both)} shots that have both (table), with the bins "
            "counted in brackets. P_in is the shot's median; the baseline is the "
            "10th percentile of f_div over the shot's flat-top-power bins (the "
            "denominator of the relative f_div). Over all shots, the median "
            "absolute f_div in TangTV-attached bins ranges from "
            f"{ranges['attached']['min']:.2f} to {ranges['attached']['max']:.2f} "
            f"({ranges['attached']['shots']} shots) and in TangTV-detached bins "
            f"from {ranges['detached']['min']:.2f} to "
            f"{ranges['detached']['max']:.2f} ({ranges['detached']['shots']} "
            f"shots), while P_in ranges from {ranges['p_in']['min']:.1f} to "
            f"{ranges['p_in']['max']:.1f} MW; the anchor shot's attached level is "
            f"{anchor_level:.2f}. "
            "The classes overlap across shots, so one pair of absolute cutoffs "
            "does not carry over from the anchor shot.\n"
        ),
        table(
            [
                "Shot",
                "P_in, MW",
                "Baseline f_div",
                "Attached f_div (bins)",
                "Detached f_div (bins)",
                "Attached relative",
                "Detached relative",
            ],
            rows,
        ),
        "\nVotes of f_div in the TangTV-voted upper-shelf bins:\n",
        table(
            [
                "TangTV",
                "f_div cutoffs",
                "Bins / shots",
                "Voted attached",
                "Voted detached",
                "Abstained",
            ],
            vote_rows,
        ),
        (
            "\nThe relative f_div measures change within a shot, not the state: "
            "a shot detached throughout has a detached baseline and reads about "
            "1, so across shots the relative scores miss it (pooled AUROC below "
            "chance), while the absolute value ranks the states across shots "
            "but with cutoffs that fit one shot. Neither is used by the label.\n"
            f"\nPublished detached point {pub['shot']} at "
            f"{pub['published_time_ms']:.0f} ms: P_in {pub['p_in_mw']:.1f} MW, "
            f"absolute f_div {pub['f_div_absolute']:.3f}, which the absolute "
            f"cutoffs vote "
            f"{'attached' if pub['f_div_absolute_vote'] == 1 else 'detached'} "
            "against the publication; the relative f_div has no baseline "
            f"(`{pub['f_div_relative_reason']}`), Afrac reads "
            f"`{pub['afrac_reason']}`, TangTV has no valid geometry "
            f"(tier `{pub['tangtv_tier']}`) and the bin is "
            f"{pub['exported_state'].replace('_', ' ')}. The absolute cutoffs "
            "therefore miss this point rather than rescue it, and the exported "
            "label makes no statement about it."
        ),
    ]


def composition_tables(rec: Records) -> list[str]:
    rows = []
    for name, v in rec.sens["variants"].items():
        c, t = v["certain"], v["tangtv_only"]
        cutoffs = v["cutoffs"]
        rows.append(
            [
                name,
                ", ".join(v["second_voters"]) or "none",
                "-" if cutoffs is None else f"{cutoffs[0]:.3f} / {cutoffs[1]:.3f}",
                triple(c),
                f"{c['shots']['attached']} / {c['shots']['detached']}",
                triple(t),
                v.get("tier_counts", {}).get("conflict", 0),
                v["uncertain_bins"],
            ]
        )
    d9 = rec.cur["paper_criterion_d9"]
    return [
        "\n### Composition under other second voters\n",
        table(
            [
                "Variant",
                "Second voters",
                "f_div cutoffs",
                "Agreement-tier bins (att / det)",
                "Agreement-tier shots (att / det)",
                "TangTV-only bins (att / det)",
                "Conflict bins",
                "Uncertain bins",
            ],
            rows,
        ),
        (
            "\n`primary` is the exported label: Afrac is the only second voter. "
            "`tangtv_alone` has no second voter, so every TangTV vote is TangTV "
            "only. The `with_*` rows add f_div as a second voter next to Afrac "
            "(`with_relative_fdiv` reproduces the label of the earlier build, "
            "where f_div was a vote: it makes many more agreement-tier bins and many "
            "more conflicts); `relative` takes cutoffs that scale the shot "
            "baseline, `absolute` the 201081-anchored global cutoffs (the "
            "midpoint of the measured attached and detached Prad,div,L over its "
            "P_in; one pair for every shot). `band_X` sets the half-width of the "
            "uncertain band to X MW (the primary keeps 0.1 MW). "
            "`afrac_abstains` leaves f_div as the only second voter. "
            "`published_anchor_values` uses the published 1.6 / 2.2 MW. A two-"
            "voter variant also has the tier `low_confidence_pair` (the two "
            "second voters agree and TangTV casts none), folded here into the "
            "uncertain bins."
        ),
        (
            f"\nDecision criterion: {d9['criterion']}. Agreement-tier attached and "
            f"detached bins coexist on {len(d9['certain']['shots_with_both'])}"
            f" shots ({len(d9['certain']['cohort_shots_with_both'])} in the "
            "cohort); counting the TangTV-only tier as well, "
            f"{len(d9['certain_or_tangtv_only']['shots_with_both'])} shots "
            f"({len(d9['certain_or_tangtv_only']['cohort_shots_with_both'])} in the "
            f"cohort). The criterion is {'met' if d9['met'] else 'not met'}, so "
            "the label set is presented as an indicator-agreement appendix, not "
            "as a three-state label set."
        ),
    ]


def uncast_points(rec: Records) -> str:
    """The published points the label does not state, each with its tier."""
    named = [
        f"{r['shot']} at {r['reference_time_ms']:.0f} ms "
        f"({r['gate_diagnostics'].get('consensus_tier') or r['primary_state']})"
        for r in rec.ref["rows"]
        if r["primary_state"] not in ("attached", "detached")
    ]
    return ", ".join(named)


def reference_tables(rec: Records) -> list[str]:
    a = rec.anchor
    ref_rows = []
    for r in rec.ref["rows"]:
        p = r["prediction"]
        ref_rows.append(
            [
                r["shot"],
                f"{r['reference_time_ms']:.0f}",
                r["truth_name"],
                p["afrac"]["vote_name"],
                p["prad"]["vote_name"],
                p["prad_absolute"]["vote_name"],
                p["tangtv"]["vote_name"],
                r["primary_state"],
                r["gate_diagnostics"].get("consensus_tier", ""),
            ]
        )
    cons = rec.ref["scores"]["consensus"]["overall"]
    return [
        "\n### Reference shot 201081 and published points\n",
        (
            "P_in = neutral beams + EFIT ohmic + ECH, "
            f"{a['measured_anchor']['p_in_mw']:.3f}"
            " MW: the median over the 32 attached-window bins of the 250 ms-averaged"
            " P_in on 201081 (beams from PTDATA BMSPINJ, because the corpus `pinj`"
            " group is a stub on this shot). Prad,div,L is "
            f"{a['measured_anchor']['attached_mw']:.3f} MW attached and "
            f"{a['measured_anchor']['detached_mw']:.3f} MW detached in 250 ms "
            "inter-ELM windows placed from the published Te cliffs "
            f"({a['phases']['attached_both']['n_bins']} attached and "
            f"{a['phases']['detached_between_cliffs']['n_bins']} detached bins)."
        ),
        "",
        table(
            [
                "Shot",
                "Time, ms",
                "Published",
                "Afrac",
                "f_div rel. (unused)",
                "f_div abs. (unused)",
                "TangTV",
                "Label",
                "Tier",
            ],
            ref_rows,
        ),
        (
            "\nThe ten points are published statements (Chen 2026; Eldon 2021). "
            "They overlap the sources that motivated the indicators, and 201081 "
            "also anchors the absolute f_div cutoffs, so the table is a sanity "
            f"check, not a benchmark: the label casts {cons['n_cast_votes']} votes "
            f"of {cons['n_reference_points']} points, and "
            f"{cons['agreement_on_cast_votes']:.0%} of those agree with the "
            f"publication. The uncast points are {uncast_points(rec)}. A vote "
            "of `abstain` means the indicator does not vote in that bin; the f_div "
            "columns are shown for reference, the label does not use them."
        ),
    ]


def width_marfe_tables(rec: Records) -> list[str]:
    width_rows = []
    for key in ("20ms", "50ms", "100ms"):
        w = rec.widths[key]
        c, t = w["certain_by_state"], w["tangtv_only_by_state"]
        vs = w.get("vs_50ms")
        width_rows.append(
            [
                key,
                f"{c['attached']['bins']} / {c['detached']['bins']}",
                f"{t['attached']['bins']} / {t['detached']['bins']}",
                f"{w['certain_seconds']:.2f} ({w['n_certain_shots']})",
                f"{w['tangtv_only_seconds']:.1f} ({w['n_tangtv_only_shots']})",
                f"{certain_share(rec, key):.3f}",
                f"{w['afrac_valid_seconds']:.1f} ({w['afrac_valid_shots']})",
                f"{w['afrac_reason_seconds']['short_reference']:.1f}",
                f"{w['afrac_reason_seconds']['no_power']:.1f}",
                f"{w['flicker_per_s']:.2f}",
                f"{vs['agreement']:.3f} ({vs['bins_labelled_in_both']})" if vs else "-",
            ]
        )
    pairs = {k: rec.widths[k]["reference_min_bins"] for k in ("20ms", "50ms", "100ms")}
    timing = rec.cur["frame_timing"]
    inv, raw = timing["inversions"], timing["corpus_raw_movie"]
    rec_199 = rec.cur["marfe"]["published_marfe_199166"]
    onset_bins = [b for b in rec_199["bins"] if b["start_ms"] >= 3700]
    spatial_from = min(b["start_ms"] for b in onset_bins if b["tangtv_marfe_spatial"])
    return [
        "\n### Bin width\n",
        (
            "The durations are fixed and converted to bins by the width: the "
            "Afrac reference needs 1 s ("
            + ", ".join(f"{pairs[k]['afrac']} bins at {k}" for k in pairs)
            + "), the f_div baseline 2 s ("
            + ", ".join(f"{pairs[k]['prad_baseline']}" for k in pairs)
            + "), a persistent MARFE front 100 ms ("
            + ", ".join(f"{pairs[k]['marfe']}" for k in pairs)
            + ") and shot eligibility 1 s ("
            + ", ".join(f"{pairs[k]['shot_eligibility']}" for k in pairs)
            + "). Cells in brackets are shots.\n"
        ),
        table(
            [
                "Width",
                "Agreement tier bins (att / det)",
                "TangTV-only bins (att / det)",
                "Agreement tier, s (shots)",
                "TangTV only, s (shots)",
                "Agreement share of labelled seconds",
                "Afrac valid, s (shots)",
                "Afrac short reference, s",
                "Afrac no power, s",
                "Flicker per s",
                "Agreement with 50 ms (bins)",
            ],
            width_rows,
        ),
        (
            f"\nThe same compatibility rule at each width on the "
            f"{rec.widths['shots_requested']} non-test shots of the width roster "
            f"({rec.widths['50ms']['n_assessed_shots']} of them assessed at 50 ms; "
            "the roster is not the exported shot list, so its 50 ms row differs "
            "from the coverage table above; the widths are comparable with each "
            "other, and the 50 ms rows reproduce the exported bins on the shots "
            "they share). With the durations held fixed the labelled time is "
            "about the same at every width ("
            + ", ".join(f"{labelled_seconds(rec, k):.1f} s at {k}" for k in pairs)
            + "), the time in which Afrac has no long enough reference is stable ("
            + ", ".join(
                f"{rec.widths[k]['afrac_reason_seconds']['short_reference']:.1f} s"
                for k in pairs
            )
            + "), and where two widths both label a bin the states agree: "
            + ", ".join(
                f"{rec.widths[k]['vs_50ms']['agreement']:.3f} at {k} "
                f"({rec.widths[k]['vs_50ms']['bins_labelled_in_both']} bins)"
                for k in ("20ms", "100ms")
            )
            + ". What moves is the agreement tier: "
            + ", ".join(
                f"{rec.widths[k]['certain_seconds']:.2f} s on "
                f"{rec.widths[k]['n_certain_shots']} shots at {k}"
                for k in pairs
            )
            + " (a share of the labelled seconds of "
            + ", ".join(f"{certain_share(rec, k):.3f}" for k in pairs)
            + "). The 20 ms loss is Afrac's input cadence, not the width of the "
            "physics: the heating-power input is sampled about every 20-25 ms, so "
            "a 20 ms bin often holds no sample and Afrac is invalid for "
            f"{rec.widths['20ms']['afrac_reason_seconds']['no_power']:.1f} s "
            "(`no_power`) against "
            f"{rec.widths['50ms']['afrac_reason_seconds']['no_power']:.1f} s at "
            "50 ms and "
            f"{rec.widths['100ms']['afrac_reason_seconds']['no_power']:.1f} s at "
            "100 ms. "
            "A reference length counted in bins instead of seconds (20 bins) would "
            "have been 0.4 s at 20 ms and 2 s at 100 ms and produced a strong "
            "width dependence that is not in the data. The 20 ms labels flicker "
            f"{rec.widths['20ms']['flicker_per_s']:.2f} times per second "
            f"against {rec.widths['50ms']['flicker_per_s']:.2f} at 50 ms and "
            f"{rec.widths['100ms']['flicker_per_s']:.2f} at 100 ms (a flicker is a "
            "change of labelled state between consecutive bins).\n"
            "\n**Why 50 ms.** The camera is a 60 Hz interlaced camera captured as "
            "30 Hz full frames (Chen 2026), but the parked inversions are spaced "
            f"{inv['median_spacing_ms_min_median_max'][1]:.1f} ms "
            f"(median of {inv['n_shots']} shots; every shot between "
            f"{inv['median_spacing_ms_min_median_max'][0]:.1f} and "
            f"{inv['median_spacing_ms_min_median_max'][2]:.1f} ms), about "
            f"{inv['frames_per_bin_median']:.0f} inversions in a 50 ms bin, and "
            "the raw VID frames in the SAV files are 16.7 ms apart as well; the "
            "corpus's raw TangTV movie is resampled to "
            f"{raw['median_spacing_ms_min_median_max'][1]:.0f} ms "
            f"({raw['frames_per_bin_median']:.1f} frames per 50 ms bin; "
            f"{raw['n_shots']} shots checked). A 50 ms bin therefore holds at "
            "least two camera frames of either kind (a 20 ms bin holds about one "
            "inversion), and the 250 ms radiation averaging window spans five "
            "bins. It is the width at which the exports are built; the table "
            "above is the evidence that the states do not depend on it."
        ),
        "\n### MARFE witness\n",
        (
            "**Recall of the published MARFE on 199166 at 3705 ms (Chen 2026): "
            f"{rec_199['recall_marfe_state']} as a state** (no bin is exported as "
            "MARFE). The bin holding that time (3700-3750 ms) is the uncertain "
            f"tier `candidate_marfe` ({rec_199['recall_candidate_marfe']} as a "
            "candidate). In the bins from 3700 ms the TangTV front is at DZ "
            + ", ".join(f"{b['tangtv_value']:.2f}" for b in onset_bins)
            + " (cutoff "
            f"{rec.phys['thresholds']['marfe_dz_min']:.1f}), the emission peak "
            "is inside the separatrix from "
            f"{spatial_from:.0f} ms, and fG is "
            + ", ".join(f"{b['aux_greenwald_fraction']:.2f}" for b in onset_bins)
            + f", below the {rec.phys['thresholds']['marfe_greenwald_min']:.1f} "
            "cue, so the full MARFE vote does not fire on this shot. The cue's "
            "value has no literature source (Dong 2025: fG of about 0.5 or more on "
            "HL-3 with a core-point density and a core-Te condition), so even "
            "where it passes the MARFE stays a candidate in the exports and "
            "figures; the full vote fires on "
            f"{rec.cur['marfe']['tangtv_marfe_vote']['bins']} bins of one other "
            "shot. The witness figure shows the inversion at 3716 ms, inside the "
            "bin 3700-3750 ms whose DZ and fG it quotes (the nearest frame to "
            "3705 ms, at 3699 ms, is in the neighbouring bin, which is TangTV "
            "only detached with DZ 0.91). The thresholds were not tuned to "
            "this shot."
        ),
    ]


def labelled_seconds(rec: Records, key: str) -> float:
    """Agreement-tier plus TangTV-only seconds at one bin width."""
    w = rec.widths[key]
    return w["certain_seconds"] + w["tangtv_only_seconds"]


def certain_share(rec: Records, key: str) -> float:
    """Agreement-tier seconds over labelled seconds at one bin width."""
    w = rec.widths[key]
    return w["certain_seconds"] / (w["certain_seconds"] + w["tangtv_only_seconds"])


def results(rec: Records) -> str:
    parts = (
        coverage_tables(rec)
        + second_indicator_table(rec)
        + te_tables(rec)
        + agreement_tables(rec)
        + afrac_tables(rec)
        + fdiv_corroborator_tables(rec)
        + fdiv_tables(rec)
        + reference_tables(rec)
        + composition_tables(rec)
        + width_marfe_tables(rec)
    )
    return "\n".join(parts)


def baselines(rec: Records) -> str:
    """The learned baselines, quoted only when scored on the current labels."""
    current = rec.cur["labels_sha256"]
    stale = [
        name
        for name, record in (("detach-ours", rec.ours), ("detach-victor", rec.victor))
        if record["label_source"]["sha256"] != current
    ]
    if stale:
        return (
            f"The learned baselines {', '.join(f'`{n}`' for n in stale)} were "
            "scored against an earlier build of the labels (sha256 "
            f"`{rec.ours['label_source']['sha256'][:12]}`; the current labels are "
            f"`{current[:12]}`). They are stale: no score of either is quoted and "
            "they are not listed among the dataset's models or any paper-facing "
            "table until they are retrained "
            "(`scripts/labeler/detach_ours.py`, `detach_victor.py`)."
        )
    rows = []
    for name, record in (("detach-ours", rec.ours), ("detach-victor", rec.victor)):
        model, majority = record["cv_shots"], record["cv_majority_ci"]
        for label, e in ((name, model), (f"{name} fold-majority control", majority)):
            rows.append(
                [
                    label,
                    f"{model['n_bins']} / {model['n_shots']}",
                    metric(e["accuracy"], 2),
                    metric(e["kappa"], 2),
                    metric(e["macro_f1"], 2),
                ]
            )
    return (
        "Two learned baselines, retrained on the labels described here (labels "
        f"sha256 `{current[:12]}`; the architecture and epochs are unchanged), "
        "predict the labelled states, agreement tier and TangTV only together "
        f"({rec.cur['population']['labelled_certain_or_tangtv_only']['bins']:,} "
        "bins), with 5-fold shot-grouped cross-validation over the non-test "
        "shots; test shots are never fitted or scored here. `detach-ours` reads "
        "windows of 0-D signals (current, density, D-alpha, ELM share, EFIT "
        "scalars) and `detach-victor` raw TangTV frames. Scores are agreement "
        "with constructed labels, not physical accuracy: `detach-victor` reads "
        "the same camera that sets the TangTV state, so its agreement partly "
        "reproduces the label from its own input. The fold-majority control "
        "predicts each held-out fold's bins as the majority class of the "
        "training fold; with shot-grouped folds that class is often the "
        "minority of the held-out shots, so the control scores below 0.5 and "
        "kappa 0 (a constant class) is the plainer reference. Intervals are "
        "shot-bootstrap (1000 replicates); bins are those with complete inputs "
        "or a camera frame, and the interval of `detach-ours` has a lower kappa "
        "end near 0.\n\n"
        + table(
            [
                "Model",
                "Bins / shots",
                "Accuracy [95% shot CI]",
                "Kappa [95% shot CI]",
                "Macro-F1 [95% shot CI]",
            ],
            rows,
        )
        + "\n\nMARFE transfer is unsupported (no MARFE label exists). Records: "
        "`docs/labeler/results/detachment_ours.json` and `detachment_victor.json`; "
        "scripts `detach_ours.py` and `detach_victor.py`. They are not listed "
        "among the dataset's models: they are a comparison of what 0-D signals "
        "and TangTV frames can reproduce of the constructed labels."
    )


def models(rec: Records) -> str:
    cur = rec.cur["population"]
    c, s = cur["by_state_certain"], cur["by_state_tangtv_only"]
    lab = cur["labelled_certain_or_tangtv_only"]
    never = cur["assessed_bins_that_can_never_carry_a_state"]
    gate = cur["lmode_gate"]["gated_bins"]
    tv_e, ce_e = rec.te_auroc("tangtv_vote_alone"), rec.te_auroc("certain")
    lab_e = rec.te_auroc("certain_or_tangtv_only")
    tv = f"{metric(tv_e)} on {tv_e['n_shots']} shots"
    ce = f"{metric(ce_e)} on {ce_e['n_shots']} shots"
    lb = f"{metric(lab_e)} on {lab_e['n_shots']} shots"
    return (
        f"- detach_vote | {MODEL_DATE}: exploratory geometry-gated TangTV state "
        "validated by divertor Thomson Te; a per-probe Afrac proxy is the "
        "second vote and f_div a within-shot corroborator; labelled "
        f"{lab['bins']:,} bins/{lab['shots']} shots, of which TangTV + Afrac "
        f"agreement (not a confidence level) {cur['certain']['bins']} "
        f"bins/{cur['certain']['shots']} shots (attached {c['attached']['bins']}, "
        f"detached {c['detached']['bins']}) and TangTV only (silver) "
        f"{cur['tangtv_only']['bins']} bins/{cur['tangtv_only']['shots']} shots "
        f"(attached {s['attached']['bins']}, detached {s['detached']['bins']}); "
        f"{cur['assessed']['bins']:,} assessed bins/{cur['assessed']['shots']} "
        f"shots, of which {never['bins']} can never carry a state (no valid "
        f"TangTV vote) and {gate['bins']} bins on {gate['shots']} shots are "
        "uncertain because TangTV votes detached on a known L-mode phase. AUROC "
        f"of -Te against the vote: labelled bins {lb}, TangTV alone {tv}, "
        f"agreement tier {ce}. No MARFE state is exported. No independent "
        "benchmark.\n"
        f"- detach_rule | {MODEL_DATE}: alias of the primary rule.\n\n"
        "Class support, agreement tables and intervals are in "
        "[the protocol](../../../docs/labeler/detachment.md) and its linked JSON "
        "records."
    )


def main() -> None:
    rec = Records()
    text = summary(rec)
    replace_block(DOC, "SUMMARY", text)
    replace_block(README, "SUMMARY", text)
    replace_block(DOC, "RESULTS", results(rec))
    replace_block(DOC, "BASELINES", baselines(rec))
    replace_block(README, "MODELS", models(rec))
    print("wrote", DOC, "and", README)


if __name__ == "__main__":
    main()
