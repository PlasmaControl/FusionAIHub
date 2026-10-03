#!/usr/bin/env python
"""Write current exploratory protocol/README summaries from source records."""

from __future__ import annotations

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "docs/labeler/results"
RECORDS = REPO / "data/events/detachment/extend_detach_vote/records"


def load(name):
    return json.loads((RESULTS / f"detachment_{name}.json").read_text())


def metric(entry):
    if not entry or entry.get("value") is None:
        return "undefined"
    value = entry["value"]
    if not math.isfinite(value):
        return "undefined"
    lo, hi = entry.get("ci95", (None, None))
    if lo is None or hi is None:
        return f"{value:.3f}"
    return f"{value:.3f} [{lo:.3f}, {hi:.3f}]"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(
        lines + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    )


def replace_block(path, name, text):
    start, end = f"<!-- {name} -->", f"<!-- /{name} -->"
    doc = path.read_text()
    a, b = doc.index(start), doc.index(end) + len(end)
    path.write_text(doc[:a] + start + "\n\n" + text + "\n\n" + end + doc[b:])


def plain_summary(rec, bench):
    current, afrac = rec["current"], rec["afrac_absence"]
    certain, states = current["certain"], rec["composition"]["by_state"]
    pair = bench["paper_agreement"]
    marfe = rec["threshold_margins"]["by_certain_state"]["marfe"]
    lines = [
        (
            "**Exploratory coverage and indicator agreement; no independent benchmark.** "
            "Certain labels use two indicators in practice: upper-shelf C-III front "
            "height and local Prad,div/P_in corroboration. They do not establish "
            "physical state accuracy."
        ),
        f"The export has {current['assessed']['bins']:,} assessed bins on "
        f"{current['assessed']['shots']} shots, including {certain['bins']} certain "
        f"bins on {certain['shots']} shots: {states['attached']['bins']} attached "
        f"on {states['attached']['shots']} shots, {states['detached']['bins']} "
        f"detached on {states['detached']['shots']} shots, and "
        f"{states['marfe']['bins']} MARFE on {states['marfe']['shots']} shot(s). "
        + rec["composition"]["transition_interpretation"],
        (
            f"Afrac is not reproduced: EFIT flux maps cover {afrac['flux_map_files']} "
            f"shots versus {afrac['processed_probe_files']} positioned-probe files; "
            f"{afrac['probe_flux_unknown_bins']:,} bins have unknown probe flux. "
            f"No upper-shelf SOL probe passes the position/flux gate "
            f"({afrac['upper_shelf_sol_probe_valid_bins']} bins). The local Jsat proxy "
            "is uncalibrated; its arbitrary 3000 ms reference minimum is absent. "
            "Its lower-shelf votes remain provisional. The Snorkel model is vestigial: "
            "no three-source upper-shelf anchors identify its accuracies, and it "
            "never decides certainty."
        ),
        (
            f"Upper-shelf Prad–TangTV agreement uses {pair['both_vote_bins']} cast "
            f"pairs on {pair['both_vote_shots']} shots: three-state κ "
            f"{metric(pair['kappa'])}; binary attached/not-attached κ "
            f"{metric(pair['binary_kappa'])}. The 95% intervals resample shots. "
            "Every upper-shelf cast radiation vote is detached, so agreement "
            "cannot establish discrimination between states. "
            "Lower-shelf measurements are reported separately as provisional."
        ),
    ]
    if marfe["bins"]:
        fg, f = marfe["greenwald_fraction"], marfe["f_div"]
        shots = ", ".join(str(s) for s in marfe["shot_ids"])
        lines.append(
            "MARFE bins are **single-shot MARFE candidates within threshold "
            f"uncertainty** on {shots}: fG={fg['min']:.3f}–{fg['max']:.3f} "
            f"against a 0.8 cue and f_div={f['min']:.3f}–{f['max']:.3f} "
            "against 0.50. Chord-based fG has about 10–20% geometric uncertainty. "
            "Published MARFE on 199166 is missed. Descriptive non-test sensitivity "
            "in the protocol shifts Prad cutoffs by ±0.05/±0.1 and the fG cue by ±0.1; "
            "no threshold is selected."
        )
    return "\n\n".join(lines)


def cnn_tables(baselines):
    rows, support, confusion = [], [], []
    for name, rec in baselines.items():
        model, majority = rec["cv_shots"], rec["cv_majority_ci"]
        bins = sum(e["n_bins"] for e in model["class_support"].values())
        shots = len({s for e in model["class_support"].values() for s in e["shot_ids"]})
        if not bins:
            continue
        for label, e in ((f"detach-{name}", model), ("Fold majority", majority)):
            rows.append(
                [
                    f"{label} ({name} population)",
                    f"{bins} / {shots}",
                    metric(e["accuracy"]),
                    metric(e["kappa"]),
                    metric(e["macro_f1"]),
                ]
            )
            matrix = e["confusion_matrix"]
            confusion.append(
                f"{label}, {name} population: rows are reference, columns prediction."
            )
            confusion.append(
                table(
                    ["Reference", *matrix["labels"]],
                    [
                        [state, *values]
                        for state, values in zip(
                            matrix["labels"], matrix["counts"], strict=True
                        )
                    ],
                )
            )
        for state, e in model["class_support"].items():
            support.append([f"detach-{name}", state, e["n_bins"], e["n_shots"]])
    return "\n\n".join(
        [
            table(
                [
                    "Held-shot diagnostic / control",
                    "Bins / shots",
                    "Accuracy [95% shot CI]",
                    "κ [95% shot CI]",
                    "Fixed-class macro-F1 [95% shot CI]",
                ],
                rows,
            ),
            "Class support in each scored population:",
            table(["CNN", "Reference class", "Bins", "Shots"], support),
            *confusion,
            (
                "These CNNs predict weak rule labels, so their scores are exploratory "
                "agreement diagnostics. Neither establishes learning beyond the "
                "adjacent fold-majority control. The fold holding the sole MARFE shot "
                "has no MARFE training examples; **MARFE transfer is unsupported**. "
                "Epochs and architecture are fixed, folds group by shot, and normalization "
                "uses training-fold data. Macro-F1 keeps the population's class set "
                "fixed across draws; draws missing a required class are excluded and "
                "counted in the JSON. Ours uses complete finite windows without heating "
                "inputs, missingness channels or imputation. The scored populations "
                "differ and cannot rank model quality."
            ),
        ]
    )


def results_text(rec, bench, reference, baselines):
    current, states = rec["current"], rec["composition"]["by_state"]
    certain, survey = current["certain"], current["corpus_survey"]
    cov = json.loads((RECORDS / "coverage.json").read_text())
    model = json.loads((RECORDS / "label_model.json").read_text())
    parts = [
        (
            f"Full discharge extraction: {current['population']['bins']:,} bins/"
            f"{current['population']['shots']} shots. Eligibility requires **20 valid "
            "bins per indicator for at least two indicators and 20 jointly assessed "
            "bins per shot**. This one-second minimum narrows the requested every-shot "
            "coverage; shorter overlaps are not exported."
        ),
        (
            f"Certain coverage: {certain['seconds']:.2f} s/{certain['intervals']} "
            f"intervals; {certain['single_50ms_intervals']} are single 50 ms bins. "
            f"Fixed-cohort certain bins: train {certain['by_split']['train']['bins']}, "
            f"validation {certain['by_split']['val']['bins']}, test "
            f"{certain['by_split']['test']['bins']}. True inversions exist on "
            f"{current['inversion_shots']} shots; surrogate valid bins: "
            f"{current['surrogate_valid_bins']}."
        ),
        table(
            ["Certain state", "Bins", "Shots", "Shot IDs", "Single-bin intervals"],
            [
                [
                    name,
                    e["bins"],
                    e["shots"],
                    ", ".join(map(str, e["shot_ids"])) or "—",
                    e["single_50ms_intervals"],
                ]
                for name, e in states.items()
                if name != "uncertain"
            ],
        ),
        "Certain composition by shot (zeros denote absent state support):",
        table(
            ["Shot", "Attached bins", "Detached bins", "MARFE bins"],
            [
                [shot, e["attached"], e["detached"], e["marfe"]]
                for shot, e in rec["composition"]["per_shot"].items()
                if sum(e[s] for s in ("attached", "detached", "marfe"))
            ],
        ),
        (
            f"Survey checks time length >1 on {survey['shots']:,} shots: usable "
            f"TangTV {survey['tangtv']:,}, bolometer {survey['bolo']:,}, Langmuir "
            f"{survey['langmuir']:,}, IRTV {survey['irtv']:,}. Group presence alone "
            "is insufficient."
        ),
        table(
            ["Indicator, full discharge", "Valid bins / shots", "Cast vote bins"],
            [
                [
                    label,
                    f"{cov[name]['valid_bins']:,} / {cov[name]['shots_with_valid_bins']}",
                    sum(cov[name]["votes"].values()),
                ]
                for name, label in (("prad", "Local f_div"), ("tangtv", "Inversion DZ"))
            ],
        ),
        table(["Primary export tier", "Bins"], list(current["tiers"].items())),
        (
            "Pairwise cast votes are compared before consensus selection and stratified "
            "by accepted `tangtv_tier`. The upper shelf is the paper population; "
            "lower-shelf primary bins remain uncertain."
        ),
    ]
    pairs = []
    for tier, entries in bench["pairwise_agreement"]["by_tangtv_tier"].items():
        for pair, e in entries.items():
            if e["both_vote_bins"]:
                pairs.append(
                    [
                        tier,
                        pair.replace("__", " / "),
                        f"{e['both_vote_bins']} / {e['both_vote_shots']}",
                        metric(e["agreement"]),
                        metric(e["kappa"]),
                        metric(e["binary_kappa"]),
                    ]
                )
    parts.append(
        table(
            [
                "TangTV tier",
                "Pair",
                "Cast pairs / shots",
                "Agreement [95% shot CI]",
                "Three-state κ [95% shot CI]",
                "Binary κ [95% shot CI]",
            ],
            pairs,
        )
    )
    conflict = bench["failure_analysis"]["prad_detached_where_tangtv_attached_by_tier"][
        "upper_shelf"
    ]
    parts.append(
        f"On the upper shelf, Prad votes detached in {conflict['prad_detached_bins']} "
        f"of {conflict['tangtv_attached_bins_with_cast_prad']} cast pairs where "
        "TangTV votes attached. There are "
        f"{conflict['tangtv_attached_bins_with_valid_prad']} such bins with valid "
        f"Prad measurements; Prad abstains in {conflict['prad_abstain_bins']}. "
        "Every upper-shelf cast Prad vote is detached. Radiation corroboration "
        "cannot establish a physical classifier."
    )
    bulk = rec["threshold_margins"]["upper_shelf_both_valid"]
    f = bulk["f_div"]
    parts.append(
        f"Across {bulk['bins']} upper-shelf bins with both measurements valid, "
        f"f_div has median {f['median']:.3f} and 5th–95th percentiles "
        f"{f['q05']:.3f}–{f['q95']:.3f}. Local cutoff placement therefore affects "
        "which imaging states receive radiation corroboration."
    )
    margins = []
    for state, e in rec["threshold_margins"]["by_certain_state"].items():
        if e["bins"]:
            f, d = e["f_div"], e["f_div_signed_distance_to_voting_threshold"]
            margins.append(
                [
                    state,
                    e["bins"],
                    f"{f['min']:.3f}–{f['max']:.3f}",
                    f"{d['q25']:.3f} / {d['median']:.3f}",
                    e["f_div_bins_within_0_05"],
                    e["f_div_bins_within_0_1"],
                ]
            )
    parts += [
        "Threshold margins in the certain set:",
        table(
            [
                "State",
                "Bins",
                "f_div range",
                "Margin Q25 / median",
                "Within .05",
                "Within .1",
            ],
            margins,
        ),
    ]
    parts.append(
        "Prad margins are signed distances from 0.36 for attached and 0.50 for "
        "detached/MARFE. Sensitivity shifts both Prad cutoffs together. Greenwald "
        "changes recompute the complete adjacent-bin spatial/cue MARFE gate and "
        "retain the H–L cue. Measurement gates stay fixed. Only non-test eligible "
        "shots enter this descriptive analysis."
    )
    sensitivity = []
    for family in ("prad", "greenwald"):
        for e in rec["threshold_sensitivity"][family]:
            setting = (
                f"{e['attached_max']:.2f} / {e['detached_min']:.2f}"
                if family == "prad"
                else f"fG≥{e['cue_min']:.2f}"
            )
            sensitivity.append(
                [
                    family,
                    f"{e['shift']:+.2f}",
                    setting,
                    *[
                        e["by_state"][s]["bins"]
                        for s in ("attached", "detached", "marfe")
                    ],
                    f"{e['certain']['bins']} / {e['certain']['shots']}",
                ]
            )
    parts += [
        table(
            [
                "Family",
                "Shift",
                "Cutoffs / cue",
                "Attached",
                "Detached",
                "MARFE",
                "Certain bins / shots",
            ],
            sensitivity,
        ),
        (
            f"The vestigial Snorkel diagnostic has {model['fit']['anchor_bins']} "
            f"three-source anchors on {model['fit']['anchor_shots']} shots. Its "
            "uncalibrated model/rule agreement supplies no physical validation."
        ),
        cnn_tables(baselines),
        (
            f"The independent published reference has {reference['n_primary_reference_points']} "
            f"points on {reference['n_primary_reference_shots']} shots. The primary rule "
            "abstains on all three points, including 199166 MARFE at 3.705 s; no positive "
            "independent state validation is available."
        ),
    ]
    for row in reference["rows"]:
        if int(row["shot"]) == 199166:
            pred, gates = row["prediction"], row["gate_diagnostics"]
            parts.append(
                f"199166/3705 ms: DZ={pred['tangtv']['value']:.3f}; "
                f"fG={gates['aux_greenwald_fraction']:.3f} fails the fixed 0.8 cue, "
                "and no H–L back-transition occurs. Spatial evidence fails at "
                "onset; nearby bins pass spatial evidence but still fail fG. "
                "The primary state is uncertain/candidate_marfe despite usable "
                "EFIT01 maps and confirmed density units. The cue was not retuned."
            )
    manual = reference["manual_front_check"]
    parts += [
        (
            f"Manual front check: {manual['n_paired_valid_bins']} valid paired bins/"
            f"{manual['n_paired_shots']} shot(s), DZ MAE {manual['mae_dz']:.3f} on "
            f"{manual['n_annotated_bins']} annotated bins. These points reuse the "
            "imaging modality and do not independently validate states."
        ),
        (
            "Sources: [current coverage/margins](results/detachment_round3.json), "
            "[tier-stratified agreement](results/detachment_benchmark.json), "
            "[published reference](results/detachment_reference.json), "
            "[ours CNN](results/detachment_ours.json) and "
            "[Victor CNN](results/detachment_victor.json). These are exploratory "
            "records; **no independent benchmark** is available."
        ),
    ]
    return "\n\n".join(parts)


def main():
    rec, bench, reference = map(load, ("round3", "benchmark", "reference"))
    baselines = {name: load(name) for name in ("ours", "victor")}
    summary = plain_summary(rec, bench)
    doc = REPO / "docs/labeler/detachment.md"
    replace_block(doc, "SUMMARY", summary)
    replace_block(doc, "RESULTS", results_text(rec, bench, reference, baselines))
    readme = REPO / "data/events/detachment/README.md"
    replace_block(readme, "SUMMARY", summary)
    current, certain = rec["current"], rec["current"]["certain"]
    models = [
        (
            f"- detach_vote | 2026_10_03: unverified two-indicator rule; "
            f"{current['assessed']['bins']} assessed bins/{current['assessed']['shots']} "
            f"shots; {certain['bins']} certain bins/{certain['shots']} shots. "
            "Exploratory coverage/agreement; no independent benchmark."
        ),
        "- detach_rule | 2026_10_03: alias of the primary compatibility rule.",
    ]
    for name, e in baselines.items():
        model, majority = e["cv_shots"], e["cv_majority_ci"]
        models.append(
            f"- detach-{name} | 2026_10_03: held-shot weak-label accuracy "
            f"{metric(model['accuracy'])}; adjacent fold-majority "
            f"{metric(majority['accuracy'])}; κ {metric(model['kappa'])}. "
            "No established learning beyond majority; MARFE transfer unsupported."
        )
    models += [
        "",
        (
            "Class support, model/control confusion matrices and intervals "
            "are in [the protocol](../../../docs/labeler/detachment.md) and "
            "linked JSON records. The primary rule abstains on all three "
            "published reference points; state validation remains unavailable."
        ),
    ]
    replace_block(readme, "MODELS", "\n".join(models))


if __name__ == "__main__":
    main()
