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

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "docs/labeler/results"
DOC = REPO / "docs/labeler/detachment.md"
README = REPO / "data/events/detachment/README.md"
FIGURE2 = REPO / "docs/labeler/figure2_detach.json"
MODEL_DATE = "2026_10_03"
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

    def certain(self, name: str) -> str:
        v = self.variant(name)["certain_bins"]
        return f"{v['attached']} / {v['detached']} / {v['marfe']}"

    def tf(self, tier: str, pair: str) -> dict:
        return self.bench["threshold_free_agreement"][tier][pair]


def reference_201081(rec: Records) -> dict:
    rows = [r for r in rec.ref["rows"] if r["shot"] == 201081]
    certain = [r for r in rows if r["primary_state"] in ("attached", "detached")]
    right = [r for r in certain if r["primary_state"] == r["truth_name"]]
    uncertain = [r for r in rows if r["primary_state"] == "uncertain"]
    attached = [r for r in rows if r["truth_name"] == "attached"]
    return {
        "points": len(rows),
        "correct": len(right),
        "uncertain": len(uncertain),
        "wrong": len(certain) - len(right),
        "attached_points": len(attached),
        "afrac_detached_on_attached": sum(
            r["prediction"]["afrac"]["vote_name"] == "detached" for r in attached
        ),
    }


def summary(rec: Records) -> str:
    cur, row, te = rec.cur, rec.row, rec.te
    total = cur["population"]
    state = total["by_state"]
    d9 = cur["paper_criterion_d9"]
    anchor = rec.anchor["measured_anchor"]
    band = rec.sens["anchor"]["band_mw"]
    cliffs = te["cliff_check"]["201081"]
    r81 = reference_201081(rec)
    marfe = cur["marfe"]
    cutoffs = rec.variant("primary_absolute")["cutoffs"]
    no_afrac = rec.variant("primary_absolute_afrac_abstains")["certain_bins"]
    paragraphs = [
        (
            "**Exploratory label set with an independent temperature check; no "
            "gold-standard benchmark.** Each 50 ms bin on an eligible shot is "
            "attached, detached, MARFE or uncertain. Three indicators vote: the "
            "target-current ratio (Afrac proxy), the lower-divertor radiated "
            "fraction f_div = Prad,div,L / P_in, and the TangTV C-III front "
            "height DZ. A bin is certain attached or detached when TangTV votes "
            "and a second indicator agrees with it and none disagrees; it is "
            "certain MARFE on the TangTV MARFE vote (spatial cue, density cue, "
            "persistence). Certain labels measure indicator agreement, not "
            "physical accuracy."
        ),
        (
            f"**Coverage.** {total['assessed']['bins']:,} assessed bins on "
            f"{total['assessed']['shots']} shots ({total['assessed']['seconds']:.0f} "
            f"s); {total['certain']['bins']} certain bins on "
            f"{total['certain']['shots']} shots: attached "
            f"{state['attached']['bins']} bins on {state['attached']['shots']} "
            f"shots, detached {state['detached']['bins']} bins on "
            f"{state['detached']['shots']} shots, MARFE {state['marfe']['bins']} "
            f"bins on {state['marfe']['shots']} shot. The rest is uncertain, with "
            "the reason in the `tier` column."
        ),
        (
            "**Indicator agreement (upper shelf).** f_div ranks TangTV-detached "
            "above TangTV-attached bins with AUROC "
            f"{metric(row['auroc_pooled'])} pooled "
            f"({row['auroc_pooled']['n_bins']:,} bins, "
            f"{row['auroc_pooled']['n_shots']} shots; shot-bootstrap 95% "
            f"intervals) and {metric(row['auroc_within_shot'])} within shots. "
            f"Cohen's kappa is {metric(row['kappa_3class'])} (three states) and "
            f"{metric(row['kappa_binary'])} (attached against not attached). The "
            "Afrac proxy is near chance against TangTV: AUROC "
            f"{metric(rec.tf('upper_shelf', 'afrac__tangtv')['auroc_pooled'])}."
        ),
        (
            "**Divertor Thomson Te check.** "
            f"{te['bins_with_te']:,} of the {te['bins_assessed']:,} assessed bins "
            "have a processed divertor Thomson Te near the target. Median Te is "
            f"{te['primary_state']['attached']['te_ev_quantiles_10_50_90'][1]:.1f}"
            " eV in certain attached bins and "
            f"{te['primary_state']['detached']['te_ev_quantiles_10_50_90'][1]:.1f} "
            "eV in certain detached bins; -Te separates them with AUROC "
            f"{metric(te['primary_state'][AUROC]['pooled'])}. Against Te, the "
            "TangTV vote has AUROC "
            f"{metric(te['indicator_votes']['tangtv'][AUROC]['pooled'])}, f_div "
            f"{metric(te['indicator_votes']['prad'][AUROC]['pooled'])} and Afrac "
            f"{metric(te['indicator_votes']['afrac'][AUROC]['pooled'])}."
        ),
        (
            f"**Reference shot 201081.** Measured P_in is {anchor['p_in_mw']:.2f} "
            f"MW. The f_div cutoffs {cutoffs[0]:.3f} / {cutoffs[1]:.3f} are the "
            f"midpoint of the attached ({anchor['attached_mw']:.3f} MW) and "
            f"detached ({anchor['detached_mw']:.3f} MW) Prad,div,L levels over "
            f"that P_in, with a stated band of +-{band:.1f} MW. Of "
            f"{r81['points']} published points on 201081, {r81['correct']} are "
            f"certain and correct, {r81['uncertain']} are uncertain and "
            f"{r81['wrong']} are wrong; the Afrac proxy votes detached on "
            f"{r81['afrac_detached_on_attached']} of the {r81['attached_points']} "
            "attached points. The Te cliffs fall at "
            f"{cliffs['first_cold_ms']:.0f} and {cliffs['last_cold_ms']:.0f} ms "
            "against 2650 and 4450 ms published."
        ),
        (
            "**Composition under the cutoffs.** Certain attached / detached / "
            f"MARFE bins are {rec.certain('primary_absolute')} with the absolute "
            f"cutoffs, {rec.certain('relative')} with the per-shot relative "
            "f_div (sensitivity), and "
            f"{no_afrac['attached']} / {no_afrac['detached']} / "
            f"{no_afrac['marfe']} when the Afrac proxy does not vote. The band "
            "sweep is in the results table."
        ),
        (
            "**MARFE.** Certain on one shot (199172). "
            f"{marfe['candidate_marfe_tier']['bins']} further bins on "
            f"{marfe['candidate_marfe_tier']['shots']} shots carry a TangTV "
            "MARFE signal without the full evidence (tier `candidate_marfe`). "
            "The published MARFE on 199166 (3.705 s) is not labelled: DZ exceeds "
            "1.2 from 3.700 s, but the emission peak is outside the separatrix at "
            "the onset and fG = 0.72-0.75 is below the 0.8 cue."
        ),
        (
            "**Afrac proxy.** It reads the peak-Jsat probe on the scrape-off side "
            "of the outer strike. Its votes disagree with TangTV often enough "
            "that conflict is the largest uncertain tier "
            f"({total['by_tier']['conflict']['bins']:,} bins); without its vote "
            f"the certain composition is {no_afrac['attached']} / "
            f"{no_afrac['detached']} / {no_afrac['marfe']}."
        ),
        (
            "**Three-state label set.** Not offered: certain attached and certain "
            f"detached bins coexist on {len(d9['shots_with_both'])} shots, "
            f"{len(d9['cohort_shots_with_both'])} of them in the fixed cohort, "
            "against a requirement of at least three shots including one cohort "
            "shot. The result is an indicator-agreement appendix."
        ),
    ]
    return "\n\n".join(paragraphs)


def coverage_tables(rec: Records) -> list[str]:
    cur = rec.cur["population"]
    tiers, state = cur["by_tier"], cur["by_state"]
    rows = [
        ["Assessed", pop(cur["assessed"]), f"{cur['assessed']['seconds']:.1f}"],
        ["Certain", pop(cur["certain"]), f"{cur['certain']['seconds']:.1f}"],
    ]
    rows += [
        [
            f"Certain {'MARFE' if name == 'marfe' else name}",
            pop(state[name]),
            f"{state[name]['seconds']:.2f}",
        ]
        for name in ("attached", "detached", "marfe")
    ]
    rows += [
        [f"Tier `{name}`", pop(entry), f"{entry['seconds']:.2f}"]
        for name, entry in tiers.items()
        if name != "certain"
    ]
    first = rec.fig["coverage_table"][0]["population_counts"]
    return [
        "### Coverage\n",
        table(["Group", "Bins / shots", "Seconds"], rows),
        (
            "\nAssessed means at least two valid indicators on an eligible shot "
            "(at least 20 assessed bins and 20 valid bins per contributing "
            "indicator). Tiers are disjoint and sum to the assessed bins; the "
            "`lower_shelf_window` tier holds the owner's restricted lower-shelf "
            "extraction, which never yields a state."
        ),
        "\n### Indicator coverage, all extracted shots\n",
        table(
            ["Indicator", "Measurement", "Valid", "Vote", "In a certain label"],
            [
                [
                    r["indicator"],
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
            f"{first['shots']} shots."
        ),
    ]


def agreement_tables(rec: Records) -> list[str]:
    names = {
        "auroc_pooled": "AUROC of f_div, pooled",
        "auroc_within_shot": "AUROC of f_div, mean over shots",
        "spearman_pooled": "Spearman rho of f_div with DZ, pooled",
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
            ("prad__tangtv", "f_div against TangTV"),
            ("afrac__tangtv", "Afrac against TangTV"),
            ("prad__afrac", "f_div against Afrac"),
        ):
            entry = rec.tf(tier, pair)
            pair_rows.append(
                [
                    tier.replace("_", " "),
                    name,
                    metric(entry["auroc_pooled"]),
                    metric(entry["spearman_pooled"]),
                    entry["auroc_pooled"]["n_bins"],
                    entry["auroc_pooled"]["n_shots"],
                ]
            )
    return [
        "\n### Indicator agreement, upper shelf\n",
        table(
            ["Statistic", "Value [95% shot CI]", "Chance", "Bins", "Shots"], stat_rows
        ),
        (
            "\nAUROC is the probability that a TangTV-detached or MARFE bin has a "
            "larger f_div than a TangTV-attached bin. Intervals resample shots "
            f"({rec.bench['replicates']} replicates, seed 0). Kappa is null, and "
            "not drawn, where either rater used one class. The same statistics "
            "for each pair of indicators:\n"
        ),
        table(
            ["Shelf", "Pair", "AUROC [CI]", "Spearman rho [CI]", "Bins", "Shots"],
            pair_rows,
        ),
    ]


def te_tables(rec: Records) -> list[str]:
    te = rec.te
    rows = []
    for label, e in [
        (f"certain {n}", te["primary_state"][n]) for n in ("attached", "detached")
    ] + [
        (f"{ind} votes {n}", te["indicator_votes"][ind][n])
        for ind in ("tangtv", "prad", "afrac")
        for n in ("attached", "detached")
    ]:
        q = e["te_ev_quantiles_10_50_90"]
        rows.append(
            [
                label,
                f"{e['n_bins']} / {e['n_shots']}",
                f"{q[1]:.1f} ({q[0]:.1f}-{q[2]:.1f})",
                f"{e['share_in_band']:.2f}",
            ]
        )
    aurocs = [["certain labels", te["primary_state"][AUROC]["pooled"]]] + [
        [f"{ind} vote", te["indicator_votes"][ind][AUROC]["pooled"]]
        for ind in ("tangtv", "prad", "afrac")
    ]
    cliff = te["cliff_check"]["201081"]
    return [
        "\n### Divertor Thomson Te check\n",
        (
            "The processed divertor Thomson Te (`\\ELECTRONS::TSTE_DIV`, 14 or 16 "
            "chords at R = 1.485 m, about 20 ms) is a temperature that none of "
            "the three indicators uses. A bin's Te is the median of the valid "
            "samples of the chords "
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
            "for detached against attached bins (pooled, shot-bootstrap "
            "interval):\n"
        ),
        table(
            ["Reference", "AUROC [CI]", "Bins", "Shots"],
            [[n, metric(e), e["n_bins"], e["n_shots"]] for n, e in aurocs],
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
            f"{cliff['published_cliffs_ms'][1]:.0f} ms). No certain MARFE bin has "
            "a Thomson Te: the check does not cover MARFE."
        ),
    ]


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
                p["tangtv"]["vote_name"],
                r["primary_state"],
            ]
        )
    return [
        "\n### Reference shot 201081 and published points\n",
        (
            "P_in = neutral beams + EFIT ohmic + ECH, median "
            f"{a['p_in_mw']['median_flat_top']:.2f} MW over the flat top (beams "
            f"{a['p_in_mw']['beams_median']:.2f} MW from PTDATA BMSPINJ, because "
            "the corpus `pinj` group is a stub on this shot). Prad,div,L is "
            f"{a['measured_anchor']['attached_mw']:.3f} MW attached and "
            f"{a['measured_anchor']['detached_mw']:.3f} MW detached in 250 ms "
            "inter-ELM windows placed from the published Te cliffs "
            f"({a['phases']['attached_both']['n_bins']} attached and "
            f"{a['phases']['detached_between_cliffs']['n_bins']} detached bins)."
        ),
        "",
        table(
            ["Shot", "Time, ms", "Published", "Afrac", "f_div", "TangTV", "Label"],
            ref_rows,
        ),
        (
            "\nThe ten points are published statements (Chen 2026; Eldon 2021). "
            "They overlap the sources that motivated the indicators, and 201081 "
            "also anchors the f_div cutoffs, so the table is a sanity check, not "
            "a benchmark. A vote of `abstain` means the indicator does not vote "
            "in that bin."
        ),
    ]


def composition_tables(rec: Records) -> list[str]:
    rows = []
    for name, v in rec.sens["variants"].items():
        cert, shots = v["certain_bins"], v["certain_shots"]
        rows.append(
            [
                name,
                f"{v['cutoffs'][0]:.3f} / {v['cutoffs'][1]:.3f}",
                f"{cert['attached']} / {cert['detached']} / {cert['marfe']}",
                f"{shots['attached']} / {shots['detached']} / {shots['marfe']}",
                v["shots_with_certain_attached_and_detached"],
                v["cohort_shots_with_certain_attached_and_detached"],
            ]
        )
    d9 = rec.cur["paper_criterion_d9"]
    return [
        "\n### Composition under the f_div cutoffs\n",
        table(
            [
                "Variant",
                "Cutoffs",
                "Certain bins (att / det / MARFE)",
                "Certain shots",
                "Shots with both",
                "Cohort shots with both",
            ],
            rows,
        ),
        (
            "\n`absolute` cutoffs (primary) are the 201081 midpoint over the "
            "measured P_in; `relative` cutoffs scale a per-shot baseline (the 10th "
            "percentile of f_div over flat-top-power bins); `band_X` sets the "
            "half-width of the uncertain band to X MW (the primary variant keeps "
            "0.1 MW); `afrac_abstains` removes the Afrac vote; "
            "`published_anchor_values` uses the published 1.6 / 2.2 MW."
        ),
        (
            f"\nDecision criterion: {d9['criterion']}. Certain attached and "
            f"certain detached bins coexist on {len(d9['shots_with_both'])} shots "
            f"({len(d9['cohort_shots_with_both'])} in the cohort), so the criterion "
            f"is {'met' if d9['met'] else 'not met'} and the label set is "
            "presented as an indicator-agreement appendix, not as a three-state "
            "label set. "
            + (
                "No variant in the table meets it."
                if not any(
                    v["three_state_criterion_met"]
                    for v in rec.sens["variants"].values()
                )
                else "At least one variant in the table meets it."
            )
        ),
    ]


def width_marfe_tables(rec: Records) -> list[str]:
    width_rows = []
    for key in ("20ms", "50ms", "100ms"):
        w = rec.widths[key]
        vs = w.get("vs_50ms")
        width_rows.append(
            [
                key,
                f"{w['n_assessed_bins']:,}",
                f"{w['n_certain_bins']} / {w['n_certain_shots']}",
                f"{w['uncertain_share_of_assessed']:.2f}",
                f"{w['flicker_per_s']:.3f}",
                f"{vs['agreement']:.3f}" if vs else "-",
            ]
        )
    shot = rec.phys["shots"][0]
    bins = {int(b["start_ms"]): b for b in shot["witness_bins"]}
    onset = bins[3700]
    afr = rec.cur["afrac"]["earlier_gate_check"]
    return [
        "\n### Bin width\n",
        table(
            [
                "Width",
                "Assessed bins",
                "Certain bins / shots",
                "Uncertain share",
                "Flicker per s",
                "Agreement with 50 ms",
            ],
            width_rows,
        ),
        (
            f"\nThe same rule on the {rec.widths['shots_requested']} roster shots at "
            "each width, without the export's eligibility filter, so counts differ "
            "from the coverage table. Where both a 50 ms and a finer or coarser "
            "bin are certain, the states agree: "
            + ", ".join(
                f"{rec.widths[k]['vs_50ms']['agreement']:.3f} at {k} "
                f"({rec.widths[k]['vs_50ms']['bins_certain_in_both']} bins)"
                for k in ("20ms", "100ms")
            )
            + "."
        ),
        "\n### Afrac probe selection\n",
        (
            "Probes eligible for Afrac are on the scrape-off side (psiN > 1.000, at "
            "least 5 mm outboard of the outer strike point, psiN <= 1.05); the "
            "peak-Jsat eligible probe is read, and its position, flux, distance "
            "and eligible-probe count are exported for every bin, valid or not. "
            "Earlier exports reported no upper-shelf probe. The cause was a gate "
            "that required a probe within 2 cm of the outer strike and psiN >= "
            f"1.01 together: only {afr['both']} of {afr['bins_with_selected_probe']:,}"
            " bins with a selected probe meet both. It was not sparse EFIT flux "
            "maps."
        ),
        "\n### MARFE witness\n",
        (
            "Published MARFE on 199166 at 3705 ms (Chen 2026): the TangTV front "
            f"exceeds the 1.2 cutoff from the 3700 ms bin (DZ "
            f"{onset['tangtv_value']:.2f}) but the emission peak lies outside the "
            "separatrix (psiN 1.48 at 3565 ms, EFIT01) and fG is "
            f"{onset['aux_greenwald_fraction']:.2f} at 3700 ms and "
            f"{bins[3800]['aux_greenwald_fraction']:.2f} at 3800 ms, below the 0.8 "
            f"cue. The bin is `candidate_marfe` ({shot['marfe_votes']} MARFE "
            "votes). The thresholds were not tuned to this shot."
        ),
    ]


def results(rec: Records) -> str:
    parts = (
        coverage_tables(rec)
        + agreement_tables(rec)
        + te_tables(rec)
        + reference_tables(rec)
        + composition_tables(rec)
        + width_marfe_tables(rec)
    )
    return "\n".join(parts)


def baselines(rec: Records) -> str:
    rows = []
    for name, record in (("detach-ours", rec.ours), ("detach-victor", rec.victor)):
        model, majority = record["cv_shots"], record["cv_majority_ci"]
        bins = sum(e["n_bins"] for e in model["class_support"].values())
        shots = len({s for e in model["class_support"].values() for s in e["shot_ids"]})
        for label, e in ((name, model), (f"{name} fold majority", majority)):
            rows.append(
                [
                    label,
                    f"{bins} / {shots}",
                    metric(e["accuracy"], 3),
                    metric(e["kappa"], 3),
                    metric(e["macro_f1"], 3),
                ]
            )
    return (
        "Two learned baselines were trained to predict the labels with "
        "shot-grouped folds: `detach-ours` on windows of current, density, "
        "D-alpha, ELM share and EFIT scalars, and `detach-victor` on raw camera "
        "frames. Neither beats its fold-majority control by a margin that "
        "survives the shot bootstrap, and the fold holding the only MARFE shot "
        "has no MARFE training examples, so MARFE transfer is unsupported. They "
        "are not listed among the dataset's models and are not paper-facing "
        "results.\n\n"
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
        + "\n\nRecords: `docs/labeler/results/detachment_ours.json` and "
        "`detachment_victor.json`; scripts `detach_ours.py` and `detach_victor.py`."
    )


def models(rec: Records) -> str:
    cur = rec.cur["population"]
    state = cur["by_state"]
    te_auroc = metric(rec.te["indicator_votes"]["tangtv"][AUROC]["pooled"])
    return (
        f"- detach_vote | {MODEL_DATE}: exploratory TangTV-led compatibility rule "
        f"over three indicators; {cur['assessed']['bins']:,} assessed bins/"
        f"{cur['assessed']['shots']} shots; {cur['certain']['bins']} certain "
        f"bins/{cur['certain']['shots']} shots (attached "
        f"{state['attached']['bins']}, detached {state['detached']['bins']}, "
        f"MARFE {state['marfe']['bins']}). Agreement of the TangTV vote with "
        f"divertor Thomson Te: AUROC {te_auroc}. No independent benchmark.\n"
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
