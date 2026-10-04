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
        "reading": effect["reading"],
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
        f"{sum(v['certain_bins_by_state'].values())} certain and "
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
            "`certain` when TangTV votes and a valid Afrac vote agrees with it; "
            "`tangtv_only` (silver) when TangTV votes and Afrac abstains or is "
            "invalid; `conflict` when TangTV and Afrac vote differently; every "
            "other assessed bin is uncertain, with the reason in the `tier` "
            "column. Certain labels measure indicator agreement, not physical "
            "accuracy."
        ),
        (
            f"**Coverage.** {pop['assessed']['bins']:,} assessed bins on "
            f"{pop['assessed']['shots']} shots ({pop['assessed']['seconds']:.0f} "
            f"s). Certain: {pop['certain']['bins']} bins on "
            f"{pop['certain']['shots']} shots (attached "
            f"{state_c['attached']['bins']} on {state_c['attached']['shots']} "
            f"shots, detached {state_c['detached']['bins']} on "
            f"{state_c['detached']['shots']} shots). TangTV only: "
            f"{pop['tangtv_only']['bins']} bins on {pop['tangtv_only']['shots']} "
            f"shots (attached {state_s['attached']['bins']}, detached "
            f"{state_s['detached']['bins']}); in "
            f"{labelled['tangtv_only_afrac_invalid']} of them Afrac is invalid "
            f"(no reference yet, known L-mode, probe off the separatrix) and in "
            f"{labelled['tangtv_only_afrac_valid_between_cutoffs']} it sits "
            "between its cutoffs. The other "
            f"{pop['by_state']['uncertain']['bins']:,} bins are uncertain; "
            f"{pop['by_tier']['conflict']['bins']} of them are `conflict`. "
            f"{cohort_text(rec)}"
        ),
        (
            "**Divertor Thomson Te check.** AUROC of -Te for a detached against an "
            "attached vote (pooled, with shot-bootstrap 95% intervals): TangTV "
            f"alone {metric(tv)} ({tv['n_bins']} bins, {tv['n_shots']} shots), "
            f"certain tier {metric(ce)} ({ce['n_bins']} bins, {ce['n_shots']} "
            f"shots{ce_note}), TangTV-only tier {metric(so)} ({so['n_bins']} bins, "
            f"{so['n_shots']} shots). Within shots: TangTV alone "
            f"{within_text(rec.te_group('tangtv_vote_alone'))}, certain "
            f"{within_text(rec.te_group('certain'))}, TangTV only "
            f"{within_text(rec.te_group('tangtv_only'))}. **The second vote does "
            "not improve the Te agreement of the TangTV state**: certain minus "
            f"TangTV alone is {sv['difference']:+.3f} [{sv['ci'][0]:+.3f}, "
            f"{sv['ci'][1]:+.3f}] over {sv['n_shots']} shots ({sv['reading']}). "
            "What the second vote buys is agreement between two indicators, not "
            "a better match to Te."
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
            f"{te_vote['ci95'][1]:.2f}] and {dec['auroc_value_based']:.2f} by "
            f"value. It stays the second vote (rule: removed below "
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
            "**Three-state label set.** Not offered: certain attached and certain "
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
    rows = [
        ["Assessed", pop(cur["assessed"]), f"{cur['assessed']['seconds']:.1f}"],
        [
            "Certain or TangTV only",
            pop(cur["labelled_certain_or_tangtv_only"]),
            f"{cur['labelled_certain_or_tangtv_only']['seconds']:.1f}",
        ],
    ]
    for name, block in (
        ("Certain attached", cur["by_state_certain"]["attached"]),
        ("Certain detached", cur["by_state_certain"]["detached"]),
        ("TangTV only attached", cur["by_state_tangtv_only"]["attached"]),
        ("TangTV only detached", cur["by_state_tangtv_only"]["detached"]),
    ):
        rows.append([name, pop(block), f"{block['seconds']:.2f}"])
    rows += [
        [f"Tier `{name}`", pop(entry), f"{entry['seconds']:.2f}"]
        for name, entry in tiers.items()
    ]
    first = rec.fig["coverage_table"][0]["population_counts"]
    return [
        "### Coverage\n",
        table(["Group", "Bins / shots", "Seconds"], rows),
        (
            "\nA bin is assessed when at least two indicators are valid on it or "
            "TangTV votes; a shot is eligible when at least two indicators are "
            "each valid on at least 20 bins and it has at least 20 assessed bins. "
            "Tiers are disjoint and sum to the assessed "
            "bins; `lower_shelf_window` holds the owner's restricted lower-shelf "
            "extraction, whose TangTV vote is invalid (the shelf geometry does not "
            "hold) so it never yields a state."
        ),
        "\n### Indicator coverage, all extracted shots\n",
        table(
            ["Indicator", "Measurement", "Valid", "Vote", "In a certain label"],
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
            "last column the certain bins in which it casts one."
        ),
    ]


def second_indicator_table(rec: Records) -> list[str]:
    c = rec.cur["afrac"]["afrac_in_labelled_bins"]
    reasons = c["tangtv_only_afrac_invalid_reasons"]
    rows = [
        ["certain bins (TangTV vote and an agreeing Afrac vote)", c["certain_bins"]],
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
    groups = [
        ("certain attached", te["by_tier"]["certain"]["attached"]),
        ("certain detached", te["by_tier"]["certain"]["detached"]),
        ("TangTV-only attached", te["by_tier"]["tangtv_only"]["attached"]),
        ("TangTV-only detached", te["by_tier"]["tangtv_only"]["detached"]),
        ("TangTV vote alone, attached", te["by_tier"]["tangtv_vote_alone"]["attached"]),
        ("TangTV vote alone, detached", te["by_tier"]["tangtv_vote_alone"]["detached"]),
    ] + [
        (f"{ind} vote {n}", te["indicator_votes"][ind][n])
        for ind in ("afrac", "prad")
        for n in ("attached", "detached")
    ]
    rows = []
    for label, e in groups:
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
        ("TangTV vote alone", te["by_tier"]["tangtv_vote_alone"]),
        ("certain tier", te["by_tier"]["certain"]),
        ("TangTV-only tier", te["by_tier"]["tangtv_only"]),
        (
            "TangTV vote in `conflict` bins",
            te["by_tier"]["tangtv_vote_in_conflict_bins"],
        ),
        ("Afrac vote", te["indicator_votes"]["afrac"]),
        ("relative f_div votes (unused)", te["indicator_votes"]["prad"]),
    ]
    auroc_rows = []
    for name, group in sources:
        pooled, within = group[AUROC]["pooled"], group[AUROC]["within_shot"]
        auroc_rows.append(
            [
                name,
                metric(pooled),
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
            "\n**Does the second vote improve the Te agreement?** No. Certain "
            f"minus TangTV alone is {sv['difference']:+.3f} "
            f"[{sv['ci'][0]:+.3f}, {sv['ci'][1]:+.3f}] over {sv['n_shots']} shots "
            f"({sv['reading']}); against the TangTV votes that are TangTV only or "
            f"in conflict it is {other['difference']:+.3f} "
            f"[{other['ci95'][0]:+.3f}, {other['ci95'][1]:+.3f}]. The within-shot "
            "figure for the certain tier rests on "
            f"{te['by_tier']['certain'][AUROC]['within_shot']['n_shots']} shot(s), "
            "too few for any statement."
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
            "D-alpha H-mode detector) and unknown regime is not gated."
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
            "f_div is not a vote, so it creates no conflicts and no certain bins. "
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
                "Certain bins (att / det)",
                "Certain shots (att / det)",
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
            "where f_div was a vote: it makes many more certain bins and many "
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
            f"\nDecision criterion: {d9['criterion']}. Certain attached and "
            f"certain detached bins coexist on {len(d9['certain']['shots_with_both'])}"
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
                f"{c['attached']['shots']} / {c['detached']['shots']}",
                f"{t['attached']['bins']} / {t['detached']['bins']}",
                f"{w['n_assessed_bins']:,}",
                f"{w['uncertain_share_of_assessed']:.2f}",
                f"{w['flicker_per_s']:.3f}",
                f"{vs['agreement']:.3f} ({vs['bins_labelled_in_both']})" if vs else "-",
            ]
        )
    rec_199 = rec.cur["marfe"]["published_marfe_199166"]
    onset_bins = [b for b in rec_199["bins"] if b["start_ms"] >= 3700]
    spatial_from = min(b["start_ms"] for b in onset_bins if b["tangtv_marfe_spatial"])
    return [
        "\n### Bin width\n",
        table(
            [
                "Width",
                "Certain bins (att / det)",
                "Certain shots (att / det)",
                "TangTV-only bins (att / det)",
                "Assessed bins",
                "Uncertain share",
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
            "other). The certain tier depends on the width: 20 ms gives "
            f"{certain_text(rec, '20ms')}, 50 ms {certain_text(rec, '50ms')} and "
            f"100 ms {certain_text(rec, '100ms')}; the certain share of the "
            "labelled bins is "
            + ", ".join(
                f"{certain_share(rec, k):.2f} at {k}" for k in ("20ms", "50ms", "100ms")
            )
            + " and the TangTV-only tier takes over. Where "
            "two widths both label a bin the states agree: "
            + ", ".join(
                f"{rec.widths[k]['vs_50ms']['agreement']:.3f} at {k} "
                f"({rec.widths[k]['vs_50ms']['bins_labelled_in_both']} bins)"
                for k in ("20ms", "100ms")
            )
            + "."
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
            "shot. The witness figure shows the frame at 3699 ms. The thresholds "
            "were not tuned to this shot."
        ),
    ]


def certain_share(rec: Records, key: str) -> float:
    """Certain bins over certain plus TangTV-only bins at one bin width."""
    w = rec.widths[key]
    return w["n_certain_bins"] / (w["n_certain_bins"] + w["n_tangtv_only_bins"])


def certain_text(rec: Records, key: str) -> str:
    w = rec.widths[key]["certain_by_state"]
    return (
        f"{w['attached']['bins']} attached / {w['detached']['bins']} detached "
        f"certain bins ({w['attached']['shots']} / {w['detached']['shots']} shots)"
    )


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
        "predict the labelled states, certain and TangTV only together "
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
    tv_e, ce_e = rec.te_auroc("tangtv_vote_alone"), rec.te_auroc("certain")
    tv = f"{metric(tv_e)} on {tv_e['n_shots']} shots"
    ce = f"{metric(ce_e)} on {ce_e['n_shots']} shots"
    return (
        f"- detach_vote | {MODEL_DATE}: exploratory geometry-gated TangTV state "
        "validated by divertor Thomson Te; a per-probe Afrac proxy is the "
        "second vote and f_div a within-shot corroborator; "
        f"{cur['assessed']['bins']:,} "
        f"assessed bins/{cur['assessed']['shots']} shots; certain "
        f"{cur['certain']['bins']} bins/{cur['certain']['shots']} shots "
        f"(attached {c['attached']['bins']}, detached {c['detached']['bins']}); "
        f"TangTV only (silver) {cur['tangtv_only']['bins']} bins/"
        f"{cur['tangtv_only']['shots']} shots (attached "
        f"{s['attached']['bins']}, detached {s['detached']['bins']}). AUROC of "
        f"-Te against the vote: TangTV alone {tv}, certain tier {ce}. No MARFE "
        "state is exported. No independent benchmark.\n"
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
