#!/usr/bin/env python
"""Render named RWM tables and a paper LaTeX table from the evaluation JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from labeler.rwm import alarm
from labeler.rwm.records import load_evaluation

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "labeler" / "rwm"
NAMES = (
    "rwm-brf",
    "rwm-rule-elapsed-time",
    "rwm-rule-betan",
    "rwm-rule-betan-over-li",
)
RULE_NAMES = {
    "rwm-rule-elapsed-time": "Elapsed time",
    "rwm-rule-betan": "βN",
    "rwm-rule-betan-over-li": "βN/li",
    "rwm-rule-rwm-candidates": "RWM screen",
}
LATEX_NAMES = {
    **RULE_NAMES,
    "rwm-brf": r"\texttt{rwm-brf}",
    "rwm-rule-betan": r"$\beta_N$",
    "rwm-rule-betan-over-li": r"$\beta_N/l_i$",
}


def paper_name(value, latex=False):
    names = LATEX_NAMES if latex else RULE_NAMES
    for name in sorted(names, key=len, reverse=True):
        value = value.replace(name, names[name])
    return value


def interval(metric, digits=3):
    if not metric or metric.get("estimate") is None:
        return "-"
    text = f"{metric['estimate']:.{digits}f}"
    if metric.get("low") is not None and metric.get("high") is not None:
        text += f" [{metric['low']:.{digits}f}, {metric['high']:.{digits}f}]"
    return text


def point(metric, digits=3):
    if metric is None:
        return "-"
    return f"{metric:.{digits}f}"


def split_range(values, digits=3, separator="–"):
    return f"{values['min']:.{digits}f}{separator}{values['max']:.{digits}f}"


def latex_range(values, digits=2, stacked=False):
    if stacked:
        return (
            r"\shortstack{"
            + f"${values['min']:.{digits}f}$"
            + r"\\to "
            + f"${values['max']:.{digits}f}$"
            + "}"
        )
    return f"${values['min']:.{digits}f}$ to ${values['max']:.{digits}f}$"


def forest_runs(record):
    config = record["configs"]["rwm-brf"]
    return {"0": config, **config["split_seeds"]}


def phase_headline(record, latex=False):
    summary = record["split_sensitivity"]
    phase = summary["phase_controlled_auroc"]
    time = summary["paired_phase"]["rwm-rule-elapsed-time"]
    beta = summary["paired_phase"]["rwm-rule-betan-over-li"]
    n = len(summary["seeds"])
    above = (
        "every split"
        if time["positive_estimates"] == n
        else f"{time['positive_estimates']} of {n} splits"
    )
    text = (
        f"Phase-controlled AUROC {split_range(phase, 2)} over {n} splits "
        f"(seed {phase['lowest_seed']} lowest); above the elapsed-time floor on "
        f"{above}, CI excluding zero on "
        f"{time['cis_excluding_zero']} of {n}; "
        + (
            "never distinguishable from βN/li alone"
            if beta["cis_excluding_zero"] == 0
            else f"βN/li difference CI excludes zero on {beta['cis_excluding_zero']} of {n}"
        )
        + "."
    )
    return text.replace("βN/li", r"$\beta_N/l_i$").replace("–", "--") if latex else text


def phase_split_table(record):
    summary = record["split_sensitivity"]
    phase, pairs = summary["phase_controlled_auroc"], summary["paired_phase"]
    keys = ("rwm-rule-elapsed-time", "rwm-rule-betan-over-li")
    return table(
        ["Evaluation", "Forest phase AUROC", "Forest − elapsed time", "Forest − βN/li"],
        [
            [
                f"Seed {s}",
                interval(v),
                *(interval(pairs[k]["by_seed"][s]) for k in keys),
            ]
            for s, v in phase["by_seed"].items()
        ]
        + [
            [
                "Run-record holdout",
                interval(phase["holdout"]),
                *(interval(pairs[k]["holdout"]) for k in keys),
            ]
        ],
    )


def split_scores(record):
    ranges = record["split_sensitivity"]["auroc_ranges"]
    return (
        table(
            [
                "campaign / scope",
                "primary AUROC range",
                "high-beta conditional AUROC range",
                "above-proxy conditional AUROC range",
            ],
            [
                [
                    campaign,
                    *(
                        split_range(row[key])
                        for key in (
                            "slice_auroc",
                            "high_beta_auroc",
                            "above_proxy_auroc",
                        )
                    ),
                ]
                for campaign, row in ranges.items()
            ],
        )
        + "\n\nRanges are min/max point estimates across seeds 0–4, not confidence intervals."
    )


def split_pairs(record):
    summary = record["split_sensitivity"]
    keys = ("slice_auroc", "high_beta_auroc", "above_proxy_auroc")
    borderline = summary["paired_time_by_seed"]["3"]["slice_auroc"]
    return table(
        [
            "fold seed",
            "primary AUROC difference",
            "high-beta conditional AUROC difference",
            "above-proxy conditional AUROC difference",
        ],
        [
            [seed, *(interval(row[k]) for k in keys)]
            for seed, row in summary["paired_time_by_seed"].items()
        ]
        + [
            [
                "point range",
                *(split_range(summary["paired_time_ranges"][k]) for k in keys),
            ]
        ],
    ) + (
        "\n\nForest minus elapsed time; 95% basic paired shot-bootstrap intervals. "
        "Intervals condition on fixed fitted predictions; elapsed-time ranks "
        "are fixed across splits. These three pooled strata retain "
        "discharge-phase information. The primary seed-3 lower bound is "
        f"{borderline['low']:.4f}, borderline near zero; a bootstrap-bound "
        "sign change alone would not "
        "establish robust superiority."
    )


def campaign_pairs(record):
    keys = ("slice_auroc", "high_beta_auroc", "above_proxy_auroc")
    rows = []
    for campaign, seeds in record["split_sensitivity"][
        "paired_time_by_campaign"
    ].items():
        for seed, metrics in seeds.items():
            rows.append(
                [campaign, f"seed {seed}", *(interval(metrics[k]) for k in keys)]
            )
        holdout = record["leave_one_run_record_out"]["paired_time_by_campaign"][
            campaign
        ]
        rows.append(
            [campaign, "run-record holdout", *(interval(holdout[k]) for k in keys)]
        )
    ranges = record["split_sensitivity"]["auroc_ranges"]["2014"]["high_beta_auroc"]
    reference = record["configs"]["rwm-brf"]["by_campaign"]["2014"]["metrics"][
        "high_beta_auroc"
    ]
    return table(
        [
            "campaign",
            "evaluation",
            "primary AUROC difference",
            "high-beta conditional AUROC difference",
            "above-proxy conditional AUROC difference",
        ],
        rows,
    ) + (
        "\n\nForest minus elapsed time; 95% basic paired shot-bootstrap intervals "
        "condition on fixed fitted predictions. High-beta: beta_N >= 0.8 times "
        "the shot's beta_N p95; above-proxy: beta_N/li > 4. Campaign 2014 "
        "forest is below chance on the reference split "
        f"({reference['estimate']:.3f} [{reference['low']:.2f}, "
        f"{reference['high']:.2f}]); the scalar rules are near chance. "
        f"Forest five-split range: {split_range(ranges, 3)}; below elapsed time "
        "on every split (point estimates; CI excludes zero on 2 of 5 seeds). "
        "Included 2018 run-record holdout CIs exclude zero: primary "
        + interval(
            record["leave_one_run_record_out"]["paired_time_by_campaign"]["2018"][
                "slice_auroc"
            ]
        )
        + "; high-beta "
        + interval(
            record["leave_one_run_record_out"]["paired_time_by_campaign"]["2018"][
                "high_beta_auroc"
            ]
        )
        + "."
    )


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(
        lines + ["| " + " | ".join(paper_name(c) for c in row) + " |" for row in rows]
    )


def scores(configs, prefix="slice", conditional=False):
    keys = [f"{prefix}_auroc", f"{prefix}_auprc"]
    if prefix == "slice":
        keys += ["slice_f1", "slice_tpr", "slice_fpr"]
    rows = []
    for name in NAMES:
        c = configs[name]
        values = [interval(c["metrics"][k]) for k in keys]
        if conditional:
            n = c["counts"]
            values += [
                str(n[f"{prefix}_positive_slices"]),
                str(n[f"{prefix}_negative_slices"]),
                f"{n[f'{prefix}_prevalence']:.3f}",
            ]
        rows.append([name, *values])
    return table(
        ["model", "AUROC (95% CI)", "AUPRC (95% CI)"]
        + (
            ["F1 (95% CI)", "slice TPR (95% CI)", "slice FPR (95% CI)"]
            if prefix == "slice"
            else []
        )
        + (
            ["positive slices", "assumed-negative slices", "prevalence"]
            if conditional
            else []
        ),
        rows,
    )


def alarm_tables(configs):
    counts, rates, warnings = [], [], []
    for name in NAMES:
        c = configs[name]
        n, m = c["counts"], c["metrics"]
        counts.append(
            [
                name,
                f"{n['onsets_warned']}/{n['target_onsets']}",
                f"{n['hanson_shots_with_an_unexplained_alarm']}/{n['hanson_shots']}",
                f"{n['comparison_shots_with_an_alarm']}/{n['comparison_shots']}",
            ]
        )
        rates.append(
            [
                name,
                interval(m["onset_detection_rate"]),
                interval(m["hanson_unexplained_alarm_incidence"]),
                interval(m["comparison_alarm_incidence"]),
            ]
        )
        warnings.append(
            [
                name,
                interval(m["warning_ms_median"], 0),
                interval(m["uniform_alarm_reference"]),
                interval(m["detection_minus_uniform_reference"]),
            ]
        )
    return {
        "Alarm counts — all models (n=1 targets)": table(
            [
                "model",
                "onsets warned",
                "Hanson shots: unexplained alarm",
                "unlabelled shots: any alarm",
            ],
            counts,
        ),
        "Alarm rates — all models (95% shot CIs)": table(
            [
                "model",
                "onset detection",
                "Hanson unexplained incidence",
                "alarm incidence on unlabelled shots",
            ],
            rates,
        ),
        "Warning times — all models (detected onsets only)": table(
            [
                "model",
                "median warning, ms (95% CI)",
                "uniform-alarm reference",
                "detection minus reference",
            ],
            warnings,
        ),
    }


def forest_alarm_splits(record):
    keys = (
        "onset_detection_rate",
        "detection_minus_uniform_reference",
        "warning_ms_median",
    )
    runs = forest_runs(record)
    rows = []
    for label, result in (
        *((f"seed {seed}", run) for seed, run in runs.items()),
        ("run-record holdout", record["leave_one_run_record_out"]),
    ):
        n, metrics = result["counts"], result["metrics"]
        rows.append(
            [
                label,
                "/".join(
                    str(n[f"hanson_{k}_shots"]) for k in ("detected", "early", "missed")
                ),
                "/".join(
                    str(n[f"hanson_any_alarm_{k}_shots"])
                    for k in ("detected", "early", "missed")
                ),
                f"{n['onsets_warned']}/{n['target_onsets']}",
                *(
                    interval(metrics[k], 0 if k == "warning_ms_median" else 3)
                    for k in keys
                ),
            ]
        )
    rows.append(
        [
            "five-split point range",
            "—",
            "—",
            "—",
            *(
                split_range(
                    record["split_sensitivity"]["alarm_ranges"][k],
                    0 if k == keys[2] else 3,
                )
                for k in keys
            ),
        ]
    )
    betan = record["configs"]["rwm-rule-betan"]
    betan_count = betan["counts"]
    return table(
        [
            "evaluation",
            "first alarm Detected/Early/Missed",
            "any alarm Detected/Early/Missed (sensitivity)",
            "onsets warned",
            "onset detection (95% CI)",
            "detection minus random reference (95% percentile CI)",
            "median warning, ms (95% CI)",
        ],
        rows,
    ) + (
        "\n\nNo improvement over the approximate rate-matched random-alarm "
        "reference was established: all five detection-difference intervals "
        "include zero. This does not establish equivalence. Warning medians "
        "condition on detected onsets; intervals condition on fixed fitted "
        "predictions. In the reference split, the beta_N rule warns "
        f"{betan_count['onsets_warned']}/{betan_count['target_onsets']} onsets "
        "and has detection minus reference "
        f"{interval(betan['metrics']['detection_minus_uniform_reference'])}; its low "
        "detection coverage limits that result."
    )


def onset_flag(row, actual=False):
    prefix = "onset_" if actual else ""
    if row[f"{prefix}efit_missing"]:
        return "missing EFIT"
    return "below 4" if row[f"{prefix}below_proxy"] else "at or above 4"


def onset_physics_table(record, actual=False):
    physics = record["onset_physics"]
    prefix = "onset_" if actual else ""
    rows = [
        [
            str(row["campaign"]),
            str(row["shot"]),
            point(row["onset_ms"], 1),
            *([] if actual else [point(row["sample_ms"], 1)]),
            point(row[f"{prefix}betan"], 2),
            point(row[f"{prefix}li"], 2),
            point(row[f"{prefix}betan_over_li"], 2),
            point(row[f"{prefix}elapsed_time_ms"], 0),
            onset_flag(row, actual),
            str(row["n_high_beta_pre_onset_slices"]),
        ]
        for row in physics["rows"]
    ]
    below = sum(bool(row[f"{prefix}below_proxy"]) for row in physics["rows"])
    missing = sum(row[f"{prefix}efit_missing"] for row in physics["rows"])
    scope = (
        "Offline EFIT inputs held at the actual listed n=1 onset "
        f"(last sample age <= {physics['efit_max_age_ms']:g} ms)"
        if actual
        else "First slice in the [onset - 20 ms, onset) window"
    )
    return table(
        [
            "campaign",
            "shot",
            "n=1 onset, ms",
            *([] if actual else ["window sample, ms"]),
            "beta_N",
            "li",
            "beta_N/li",
            "elapsed time, ms",
            "proxy category",
            "pre-onset high-beta slices",
        ],
        rows,
    ) + (
        f"\n\n{scope}: {below} snapshots are below beta_N/li = 4 and {missing} have "
        "missing EFIT inputs; missing inputs remain explicit rather than "
        "being classified above or below the proxy. Elapsed time is measured "
        "from the first |Ip| >= 0.5 MA crossing. The high-beta slice count "
        "uses the forecast-positive pre-onset window."
    )


def onset_campaign_summary(record):
    rows = []
    physics = record["onset_physics"]
    for campaign, summary in physics["by_campaign"].items():
        denominator = summary["onsets"]
        rows.append(
            [
                str(campaign),
                str(denominator),
                *(
                    f"{summary[k]}/{denominator} ({summary[k] / denominator:.1%})"
                    for k in (
                        "onset_below_proxy",
                        "onset_efit_missing",
                        "below_proxy",
                        "efit_missing",
                        "no_high_beta_pre_onset_slices",
                    )
                ),
            ]
        )
    return table(
        [
            "campaign",
            "n=1 onsets",
            "actual onset below beta_N/li = 4",
            "actual onset missing EFIT",
            "window sample below beta_N/li = 4",
            "window sample missing EFIT",
            "no high-beta pre-onset slices",
        ],
        rows,
    )


def paired(record, prefix):
    rows = [
        [name, interval(m[f"{prefix}_auroc"]), interval(m[f"{prefix}_auprc"])]
        for name, m in record["paired"].items()
        if not name.endswith("rwm-rule-rwm-candidates")
    ]
    return table(
        [
            "first model minus rule",
            "AUROC difference (95% basic CI)",
            "AUPRC difference (95% basic CI)",
        ],
        rows,
    )


def grouped_scores(groups, intervals=True):
    rows = []
    masks = (
        ("slice", "primary"),
        ("high_beta", "high-beta conditional"),
        ("above_proxy", "above-proxy conditional"),
    )
    formatter = interval if intervals else point
    for group, record in groups.items():
        n, m = record["counts"], record["metrics"]
        for prefix, label in masks:
            count_prefix = "" if prefix == "slice" else f"{prefix}_"
            rows.append(
                [
                    str(group),
                    label,
                    str(n["hanson_shots"]),
                    str(n[f"{count_prefix}positive_slices"]),
                    str(n[f"{count_prefix}negative_slices"]),
                    point(n[f"{count_prefix}prevalence"]),
                    formatter(m[f"{prefix}_auroc"]),
                    formatter(m[f"{prefix}_auprc"]),
                ]
            )
    suffix = " (95% shot CI)" if intervals else " (point estimate)"
    return table(
        [
            "group",
            "slice mask",
            "Hanson shots",
            "positive slices",
            "assumed-negative slices",
            "prevalence",
            f"AUROC{suffix}",
            f"AUPRC{suffix}",
        ],
        rows,
    )


def shot_categories(configs, any_alarm=False):
    rows = []
    for name in NAMES:
        n = configs[name]["counts"]
        denominator = n["hanson_target_shots"]
        prefix = "hanson_any_alarm" if any_alarm else "hanson"
        rows.append(
            [
                name,
                f"{n[prefix + '_detected_shots']}/{denominator}",
                f"{n[prefix + '_missed_shots']}/{denominator}",
                f"{n[prefix + '_early_shots']}/{denominator}",
                str(n["hanson_no_target_shots"]),
                f"{n['comparison_shots_with_an_alarm']}/{n['comparison_shots']}",
            ]
        )
    return table(
        [
            "model",
            "Detected Hanson shots",
            "Missed Hanson shots",
            "Early Hanson shots",
            "Hanson shots without n=1 targets (excluded)",
            "FP on comparison shots (alarm incidence)",
        ],
        rows,
    ) + (
        "\n\nDetected, Early and Missed are mutually exclusive on Hanson shots "
        "with an n=1 target. "
        + (
            "Sensitivity: any warning wins, otherwise any Early alarm, then Missed. "
            if any_alarm
            else "The first considered alarm decides the primary shot category. "
        )
        + "Early means unexplained and more than 400 ms before a future target. "
        "The FP column reports unlabelled-shot alarm incidence, not a verified "
        "stable-shot false-positive rate."
    )


def grouped_alarm_scores(groups, intervals=True):
    formatter = interval if intervals else point
    rows = []
    for group, result in groups.items():
        n, m = result["counts"], result["metrics"]
        rows.append(
            [
                str(group),
                formatter(m["slice_f1"]),
                f"{n['onsets_warned']}/{n['target_onsets']}",
                formatter(m["onset_detection_rate"]),
                f"{n['hanson_early_shots']}/{n['hanson_target_shots']}",
                f"{n['hanson_shots_with_an_unexplained_alarm']}/{n['hanson_shots']}",
                formatter(m["hanson_unexplained_alarm_incidence"]),
                formatter(m["warning_ms_median"], 0),
            ]
        )
    suffix = " (95% shot CI)" if intervals else " (point estimate)"
    return table(
        [
            "held-out group",
            f"primary F1{suffix}",
            "onsets warned",
            f"onset detection{suffix}",
            "Early Hanson shots",
            "Hanson shots with an unexplained alarm",
            f"Hanson unexplained incidence{suffix}",
            f"median warning, ms{suffix}",
        ],
        rows,
    )


def alarm_sensitivity(config):
    rows = []
    for label, result in (
        ("primary: end 100 ms after last n=1/n=2 onset", config),
        ("full-trace sensitivity", config["full_trace_alarm_sensitivity"]),
    ):
        n, m = result["counts"], result["metrics"]
        rows.append(
            [
                label,
                f"{n['onsets_warned']}/{n['target_onsets']}",
                interval(m["onset_detection_rate"]),
                f"{n['hanson_early_shots']}/{n['hanson_target_shots']}",
                f"{n['hanson_shots_with_an_unexplained_alarm']}/{n['hanson_shots']}",
                interval(m["hanson_unexplained_alarm_incidence"]),
                f"{n['comparison_shots_with_an_alarm']}/{n['comparison_shots']}",
                interval(m["comparison_alarm_incidence"]),
            ]
        )
    return table(
        [
            "alarm definition",
            "onsets warned",
            "onset detection (95% CI)",
            "Early Hanson shots",
            "Hanson shots with an unexplained alarm",
            "Hanson unexplained incidence (95% CI)",
            "unlabelled shots with an alarm",
            "unlabelled alarm incidence (95% CI)",
        ],
        rows,
    ) + (
        "\n\nThe primary alarm window ends 100 ms after the last n=1 or n=2 "
        "explanation onset on Hanson shots; comparison traces retain their full "
        "span. This tolerance extends beyond the primary slice mask, "
        "which ends at the last n=1 target onset. Both alarm definitions are "
        "tuned within the inner folds; unlabelled comparisons never tune alarms."
    )


def legacy_table(legacy):
    return table(
        [
            "published model",
            "slice AUROC",
            "slice TPR",
            "slice FPR",
            "detected unstable shots",
            "false-positive stable shots",
        ],
        [
            [
                legacy["model"],
                point(legacy["slice_auroc"]),
                f"{100 * legacy['slice_tpr']:.1f}%",
                f"{100 * legacy['slice_fpr']:.1f}%",
                f"{legacy['onsets_warned']}/{legacy['target_onsets']}",
                (f"{legacy['stable_shots_with_an_alarm']}/{legacy['stable_shots']}"),
            ]
        ],
    ) + (
        "\n\nPiccione et al. (2022), doi:10.1088/1741-4326/ac44af. "
        "Different machine (NSTX), expert-reviewed stable shots, different inputs "
        "and validation; these published test results are not comparable to the "
        "DIII-D benchmark. F1 and confidence intervals are not available in the "
        "source digest. "
        f"Source in evaluation.json: legacy.source = {legacy['source']}."
    )


def latex_cell(m, stacked=False, bound_digits=2, digits=3):
    if m["estimate"] is None:
        return "--"
    point = f"${m['estimate']:.{digits}f}$"
    if m["low"] is None or m["high"] is None:
        return point
    bounds = (
        r"{\scriptsize $["
        + f"{m['low']:.{bound_digits}f}, {m['high']:.{bound_digits}f}]"
        + "$}"
    )
    if stacked:
        return r"\shortstack{" + point + r"\\" + bounds + "}"
    return point + " " + bounds


def latex_incidence(numerator, denominator, metric):
    if not denominator:
        return "--"
    return (
        r"\shortstack{"
        + f"{numerator}/{denominator}"
        + r"\\"
        + latex_cell(metric, stacked=True, bound_digits=3)
        + "}"
    )


def phase_width_table(record):
    return table(
        ["Time-bin width", "Forest", "Elapsed time", "βN", "βN/li"],
        [
            [
                width + " ms" if width != "no_time_control" else "No time control",
                *(
                    point(record["configs"][name]["phase_bin_width_sensitivity"][width])
                    for name in NAMES
                ),
            ]
            for width in ("50", "100", "200", "400", "800", "no_time_control")
        ],
    )


def write_documentation(record, comparison):
    """Refresh marked numerical prose and README model lines from saved scores."""
    configs, protocol = record["configs"], record["protocol"]
    forest, elapsed = configs["rwm-brf"], configs["rwm-rule-elapsed-time"]
    change = comparison["paired_change"]["phase_controlled_auroc"]
    n = comparison["counts"]
    sensitivity = (
        "Adding the unverified comparisons as label-noisy training negatives "
        f"gives phase-controlled AUROC **{interval(comparison['phase_controlled_auroc'])}**, "
        f"a paired change of **{change['estimate']:+.3f} "
        f"[{change['low']:.3f}, {change['high']:.3f}]**. "
        f"It alarms on **{n['comparison_shots_with_an_alarm']}/"
        f"{n['comparison_shots']}** comparison shots and warns "
        f"**{n['onsets_warned']}/{n['target_onsets']}** onsets. "
        "The paired interval includes zero; whether verified stable-shot "
        "negatives would help is untested. Source: "
        "`C#/{phase_controlled_auroc,paired_change,counts}`."
    )
    phase_rows = []
    for name in NAMES:
        pair = record["paired"].get(f"rwm-brf - {name}")
        phase_rows.append(
            [
                paper_name(name),
                interval(configs[name]["phase_controlled_auroc"]),
                interval(pair["phase_controlled_auroc"]) if pair else "—",
            ]
        )
    # Warning distribution is measured from the verified external E payload.
    complete = load_evaluation(OUT / "evaluation.json", details=True)
    warnings = sorted(
        value
        for row in complete["configs"]["rwm-brf"]["per_shot"]
        for value in row["warning_ms"]
        if value is not None
    )
    distribution = {
        "n": len(warnings),
        "min_ms": min(warnings, default=None),
        "max_ms": max(warnings, default=None),
        "median": forest["metrics"]["warning_ms_median"],
        "matching_min_ms": alarm.MIN_WARNING_MS,
        "matching_max_ms": alarm.MAX_WARNING_MS,
        "source": "E#/configs/rwm-brf/per_shot/*/warning_ms (external details)",
    }
    warning_range = (
        f"{min(warnings):.0f}–{max(warnings):.0f} ms"
        if warnings
        else "unavailable (no matched warnings)"
    )
    alarm_text = (
        f"The reference split warns **{forest['counts']['onsets_warned']}/"
        f"{forest['counts']['target_onsets']}** onsets; median warning is "
        f"**{interval(forest['metrics']['warning_ms_median'], 0)} ms**. "
        f"The {len(warnings)} matched warnings have range **{warning_range}** "
        f"within the **{alarm.MIN_WARNING_MS:g}–"
        f"{alarm.MAX_WARNING_MS:g} ms** matching window. The median is close "
        "to its upper limit, consistent with early phase-driven alarms; it "
        "does not establish onset-specific anticipation. Comparison shots "
        f"alarmed: **{forest['counts']['comparison_shots_with_an_alarm']}/"
        f"{forest['counts']['comparison_shots']}**, "
        f"**{interval(forest['metrics']['comparison_alarm_incidence'])}**. "
        "Hanson shots with unexplained alarms: "
        f"**{forest['counts']['hanson_shots_with_an_unexplained_alarm']}/"
        f"{forest['counts']['hanson_shots']}**, "
        f"**{interval(forest['metrics']['hanson_unexplained_alarm_incidence'])}**. "
        "These are unlabelled-shot incidences, not verified false-positive "
        "rates. Sources: `E#/configs/rwm-brf/{counts,metrics,per_shot}`; "
        "distribution calculation: `P#/documentation/warning_distribution`."
    )
    chunks = {
        "lead": (
            "This retrospective baseline has a small, split-dependent margin "
            "over elapsed time and no established skill beyond βN/li. "
            + phase_headline(record)
            + " Reference split (seed 0): `rwm-brf` "
            f"**{interval(forest['phase_controlled_auroc'])}** in "
            f"**{protocol['phase_bin_ms']:g} ms bins**, with at least "
            f"**{protocol['phase_min_slices']} eligible slices per bin**. "
            "Elapsed time's residual value "
            f"**{interval(elapsed['phase_controlled_auroc'])}** is the phase floor. "
            f"Pooled primary AUROC **{interval(forest['metrics']['slice_auroc'])}** "
            "is phase-confounded. The task is when an RWM comes in a Hanson "
            "shot that has one; physical duration, negative coverage and "
            "online input timing remain unverified. Sources: "
            "`E#/split_sensitivity/{phase_controlled_auroc,paired_phase}`, "
            "`E#/configs/{rwm-brf,rwm-rule-elapsed-time}/phase_controlled_auroc` "
            "and `E#/configs/rwm-brf/metrics/slice_auroc`; E is defined below."
        ),
        "phase": table(
            ["Model / rule", "Phase AUROC (95% CI)", "Forest minus rule (95% CI)"],
            phase_rows,
        )
        + "\n\n"
        + phase_split_table(record)
        + "\n\n"
        + phase_width_table(record),
        "sensitivity": sensitivity,
        "alarms": alarm_text,
        "screen": (
            "The retrospective `rwm_candidates` screen serves a concurrent "
            "review role, with "
            f"{record['screen_audit']['calls_after_first_listed_onset']} post-onset "
            "Hanson slice calls and calls on "
            f"{configs['rwm-rule-rwm-candidates']['counts']['comparison_shots_with_an_alarm']}/"
            f"{configs['rwm-rule-rwm-candidates']['counts']['comparison_shots']} "
            "unlabelled comparisons, so it is excluded from forecasting tables "
            "(`E#/screen_audit` and that screen's `counts`)."
        ),
    }
    audit = record["forecast_label_audit"]
    missing, rotation = audit["efit_missing"], audit["rotation_raw"]["rot_core_khz"]
    first, first_time = forest["first_onset_only"], elapsed["first_onset_only"]
    growth_record = json.loads((OUT / "growth.json").read_text())
    growth = growth_record["betan"]
    chunks.update(
        {
            "forecast": (
                f"Primary forecast labels contain **{audit['negative_after_first']:,}/"
                f"{audit['n_negative']:,} negatives ({audit['negative_after_first'] / audit['n_negative']:.1%})** "
                "after a first n=1 onset and "
                f"**{audit['positive_before_repeats']}/{audit['n_positive']} positives "
                f"({audit['positive_before_repeats'] / audit['n_positive']:.1%})** preceding repeat onsets. "
                "These labels concern the next onset; inter-onset physical state "
                "remains category 4 (unassessed) in the interval export. Restricting "
                "the same saved reference-split scores to primary slices strictly "
                "before the first n=1 onset gives pooled forest AUROC "
                f"**{interval(first['pooled_auroc'])}** versus elapsed time "
                f"**{interval(first_time['pooled_auroc'])}**, on "
                f"{first['n_positive']} positives / {first['n_negative']:,} assumed "
                f"negatives across {first['n_shots']} shots; no refit or tuning. "
                "Sources: `E#/forecast_label_audit` and "
                "`E#/configs/{rwm-brf,rwm-rule-elapsed-time}/first_onset_only`."
            ),
            "missingness": (
                f"EFIT inputs are missing in {missing['positive_fraction']:.0%} of "
                f"positive slices versus {missing['negative_fraction']:.0%} of "
                "negatives (at least one missing βN, li, q95, qmin or W_MHD input); "
                "median imputation could act as a weak missingness signal "
                "(`E#/forecast_label_audit/efit_missing`)."
            ),
            "rotation_units": (
                "The `TROTFIT` units field says kHz, but core magnitudes "
                f"(median {rotation['median']:.0f}, maximum {rotation['max']:.0f}) "
                "match krad/s, the ZIPFIT convention; kHz would imply supersonic "
                "toroidal velocity. Legacy `rot_*_khz` names retain raw values "
                "without conversion and do not establish physical units. The "
                "forest is invariant to a positive constant unit conversion "
                "(`E#/forecast_label_audit/rotation_raw`; namespace metadata)."
            ),
            "collapse": (
                f"βN falls at least 20% by 40 ms after **{growth['dropping_onsets']}/"
                f"{growth['eligible_onsets']} eligible onsets "
                f"({growth['fraction_dropping_20pct_by_p40']:.1%})**, relative to "
                f"the value 100 ms before onset; **{growth['unknown_onsets']} of "
                f"{growth_record['onsets']['n1']} n=1 onsets** are unknown. "
                "Both reference values require finite "
                f"bracketing samples separated by at most {growth['max_bracket_gap_ms']:g} ms, "
                "with no extrapolation. This supports alignment with a plasma "
                "event while leaving mode identity and growth-start timing "
                "unresolved (`G#/betan`)."
            ),
        }
    )
    doc = REPO / "docs/labeler/rwm_baseline.md"
    body = doc.read_text()
    for name, text in chunks.items():
        pattern = rf"(<!-- rwm:{name}:start -->).*?(<!-- rwm:{name}:end -->)"
        body, count = re.subn(
            pattern,
            lambda m, text=text: m[1] + "\n" + text + "\n" + m[2],
            body,
            flags=re.DOTALL,
        )
        if count != 1:
            raise ValueError(f"expected one documentation block: {name}")
    doc.write_text(body)
    readme = REPO / "data/events/resistive_wall_mode/README.md"
    model_lines = []
    for name in NAMES:
        config = configs[name]
        model_lines.append(
            f"- {name} | 2026_10_03 | Phase-controlled AUROC: "
            f"{interval(config['phase_controlled_auroc'])} | "
            f"AUROC: {interval(config['metrics']['slice_auroc'])} (primary; phase-confounded) | "
            f"AUPRC: {interval(config['metrics']['slice_auprc'])} | "
            f"F1: {interval(config['metrics']['slice_f1'])}"
            + (" | " + phase_headline(record) if name == "rwm-brf" else "")
        )
    models = (
        "## Models\n**stable**: none\n\n**latest**: rwm-brf\n\n**all**:\n\n"
        + "\n".join(model_lines)
        + "\n\nReference split (seed 0), Hanson primary forecasts, assumed "
        f"negatives, {protocol['phase_bin_ms']:g} ms phase bins with at least "
        f"{protocol['phase_min_slices']} slices. Pooled primary AUROC is "
        "phase-confounded; elapsed time's residual AUROC is the phase floor. "
        "Intervals are exploratory shot-bootstrap intervals. "
        + sensitivity.replace(
            "Source: `C#/{phase_controlled_auroc,paired_change,counts}`.", ""
        )
        + "See [protocol and results](../../../docs/labeler/rwm_baseline.md) "
        "and [comparison sensitivity](../../../outputs/labeler/rwm/"
        "comparison_sensitivity.json).\n\n"
    )
    body, count = re.subn(
        r"## Models\n.*?(?=## Inputs)",
        lambda _: models,
        readme.read_text(),
        flags=re.DOTALL,
    )
    if count != 1:
        raise ValueError("expected one README Models section")
    readme.write_text(body)
    return {
        "doc": str(doc),
        "readme": str(readme),
        "words": len(doc.read_text().split()),
        "warning_distribution": distribution,
    }


def write_latex(record, out_dir):
    configs, legacy = record["configs"], record["legacy"]
    labels = LATEX_NAMES
    keys = (
        "slice_auroc",
        "broad_auroc",
        "slice_auprc",
        "slice_f1",
        "slice_tpr",
        "slice_fpr",
    )
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{1.5pt}",
        r"\begin{tabular}{@{}lccccccc@{}}",
        r"\toprule",
        r"\multicolumn{8}{@{}l}{Tokamak-SI (DIII-D): reference split (seed 0)} \\",
        r"\midrule",
        (
            r"Model / rule & \shortstack{Phase-controlled\\AUROC} & "
            r"\shortstack{Primary AUROC\\phase-confounded} & "
            r"\shortstack{Broad\\AUROC} & "
            r"\shortstack{Primary\\AUPRC} & \shortstack{Primary\\F1} & "
            r"\shortstack{Slice\\TPR} & \shortstack{Slice\\FPR} \\"
        ),
        r"\midrule",
    ]
    provenance = {}
    for name in NAMES:
        config = configs[name]
        cells = [
            latex_cell(config["metrics"][k], stacked=True, bound_digits=3) for k in keys
        ]
        phase = config["phase_controlled_auroc"]
        cells.insert(0, latex_cell(phase, stacked=True, bound_digits=3))
        lines.append(labels[name] + " & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace[1.5pt]")
        provenance[name] = {
            k: {
                "json_path": f"configs.{name}.metrics.{k}",
                **config["metrics"][k],
            }
            for k in keys
        }
        provenance[name]["phase_controlled_auroc"] = {
            "json_path": f"configs.{name}.phase_controlled_auroc",
            **phase,
        }
    lines += [
        r"\midrule",
        r"\multicolumn{8}{@{}l}{Legacy (Piccione et al., NSTX): not comparable} \\",
        r"\midrule",
        r"\multicolumn{8}{@{}l}{NSTX RUS forest: "
        f"AUROC ${legacy['slice_auroc']:.3f}$; "
        f"TPR ${legacy['slice_tpr']:.3f}$; FPR ${legacy['slice_fpr']:.3f}$" + r"} \\",
        r"\multicolumn{8}{@{}l}{Detected unstable shots: "
        f"{legacy['onsets_warned']}/{legacy['target_onsets']}; "
        "stable shots with false alarms: "
        f"{legacy['stable_shots_with_an_alarm']}/"
        f"{legacy['stable_shots']}" + r"} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    protocol = record["protocol"]
    n = configs["rwm-brf"]["counts"]
    caption = (
        r"Within-RWM-shot timing with assumed negatives. "
        + phase_headline(record, latex=True)
        + " "
        r"Phase control compares pairs within campaign and "
        f"{protocol['phase_bin_ms']:g} ms bins "
        f"(at least {protocol['phase_min_slices']} slices); elapsed time measures "
        r"the residual-phase floor. Primary negatives end at the last target "
        f"onset: {n['positive_slices']:,} positive / {n['negative_slices']:,} "
        f"assumed-negative slices (prevalence {n['positive_slices'] / (n['positive_slices'] + n['negative_slices']):.2%}".replace(
            "%", r"\%"
        )
        + "). Broad adds later Hanson time and "
        f"{record['forecast_label_audit']['broad_n2_only_negatives']:,} negatives "
        r"from $n=2$-only shots. Cutoffs use inner-fold primary "
        r"slices. Brackets: exploratory shot-bootstrap intervals at fixed "
        r"predictions, unadjusted for multiplicity. Legacy differs in machine, "
        r"inputs and validation."
    )
    lines += [
        r"\caption{" + caption + "}",
        r"\label{tab:rwm-baseline}",
        r"\end{table*}",
    ]
    target = out_dir / "table_rwm.tex"
    source = OUT / target.name
    tex = "\n".join(lines) + "\n"
    target.write_text(tex)
    source.write_text(tex)
    return {
        "path": str(target),
        "source_path": str(source),
        "cells": provenance,
        "caption": caption,
        "caption_words": len(caption.split()),
        "legacy": {
            "source": legacy["source"],
            "cells": {
                k: {"json_path": f"legacy.{k}", "value": legacy[k]}
                for k in (
                    "slice_auroc",
                    "slice_tpr",
                    "slice_fpr",
                    "onsets_warned",
                    "target_onsets",
                    "stable_shots_with_an_alarm",
                    "stable_shots",
                )
            },
            "slice_f1": "not available in source digest",
        },
    }


def write_supplemental_latex(record, out_dir):
    """Keep full paired, alarm and snapshot evidence readable in separate tables."""
    artifacts = {}
    n_bootstrap = record["protocol"]["bootstrap_replicates"]

    def write(name, columns, header, rows, caption, source, long=False, panel=None):
        environment = "longtable" if long else "tabular"
        lines = [r"\begingroup"] if long else [r"\begin{table*}[t]", r"\centering"]
        spacing = "2pt" if name in ("split_summary", "alarms") else "4pt"
        lines += [r"\small", r"\setlength{\tabcolsep}{" + spacing + "}"]
        if panel:
            lines += [
                r"\begin{tabular}{@{}lccc@{}}",
                r"\toprule",
                r"Evaluation & Forest phase AUROC & Forest $-$ elapsed time & Forest $-$ $\beta_N/l_i$ \\",
                r"\midrule",
                *(" & ".join(row) + r" \\" for row in panel),
                r"\bottomrule",
                r"\end{tabular}\par\vspace{5pt}",
            ]
        lines += [r"\begin{" + environment + "}{@{}" + columns + "@{}}"]
        if long:
            lines += [r"\caption{" + caption + r"} \\"]
        lines += [r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
        if long:
            lines += [
                r"\endfirsthead",
                r"\toprule",
                " & ".join(header) + r" \\",
                r"\midrule",
                r"\endhead",
                r"\midrule",
                r"\endfoot",
                r"\bottomrule",
                r"\endlastfoot",
            ]
        lines += [
            " & ".join(paper_name(cell, latex=True) for cell in row) + r" \\"
            for row in rows
        ]
        if not long:
            lines += [r"\bottomrule"]
        lines += [r"\end{" + environment + "}"]
        if not long:
            lines += [r"\caption{" + caption + "}", r"\end{table*}"]
        else:
            lines += [r"\endgroup"]
        path = out_dir / f"table_rwm_{name}.tex"
        source_path = OUT / path.name
        tex = "\n".join(lines) + "\n"
        path.write_text(tex)
        source_path.write_text(tex)
        artifacts[name] = {
            "path": str(path),
            "source_path": str(source_path),
            "caption": caption,
            "source": source,
        }

    keys = ("slice_auroc", "high_beta_auroc", "above_proxy_auroc")
    summary = record["split_sensitivity"]
    holdout = record["leave_one_run_record_out"]
    phase = summary["phase_controlled_auroc"]
    pairs = summary["paired_phase"]
    phase_panel = [
        [
            f"Seed {s}",
            latex_cell(v, bound_digits=3),
            *(
                latex_cell(pairs[k]["by_seed"][s], bound_digits=3)
                for k in ("rwm-rule-elapsed-time", "rwm-rule-betan-over-li")
            ),
        ]
        for s, v in phase["by_seed"].items()
    ] + [
        [
            "Run-record holdout",
            latex_cell(phase["holdout"], bound_digits=3),
            *(
                latex_cell(pairs[k]["holdout"], bound_digits=3)
                for k in ("rwm-rule-elapsed-time", "rwm-rule-betan-over-li")
            ),
        ]
    ]
    rows = []
    broad_ranges = {}
    for campaign, ranges in summary["auroc_ranges"].items():
        broad_values = {
            seed: (run if campaign == "pooled" else run["by_campaign"][campaign])[
                "metrics"
            ]["broad_auroc"]["estimate"]
            for seed, run in forest_runs(record).items()
        }
        broad_ranges[campaign] = {
            "by_seed": broad_values,
            "min": min(broad_values.values()),
            "max": max(broad_values.values()),
        }
        rows.append(
            [
                campaign.capitalize(),
                "Forest range",
                latex_range(ranges[keys[0]], 3),
                latex_range(broad_ranges[campaign], 3),
                *(latex_range(ranges[k], 3) for k in keys[1:]),
            ]
        )
        scores = holdout if campaign == "pooled" else holdout["by_campaign"][campaign]
        rows.append(
            [
                campaign.capitalize(),
                "Forest holdout",
                *(
                    latex_cell(scores["metrics"][k], bound_digits=3)
                    for k in (keys[0], "broad_auroc", *keys[1:])
                ),
            ]
        )
    for seed, metrics in summary["paired_time_by_seed"].items():
        rows.append(
            [
                "Pooled difference",
                f"Seed {seed}",
                latex_cell(
                    metrics[keys[0]],
                    bound_digits=3,
                ),
                (
                    latex_cell(
                        record["paired"]["rwm-brf - rwm-rule-elapsed-time"][
                            "broad_auroc"
                        ],
                        bound_digits=3,
                    )
                    if seed == "0"
                    else "--"
                ),
                *(latex_cell(metrics[k], bound_digits=3) for k in keys[1:]),
            ]
        )
    rows.append(
        [
            "Pooled difference",
            "Five-split range",
            latex_range(summary["paired_time_ranges"][keys[0]], 3),
            "--",
            *(latex_range(summary["paired_time_ranges"][k], 3) for k in keys[1:]),
        ]
    )
    rows.append(
        [
            "Pooled difference",
            "Run-record holdout",
            *(
                latex_cell(holdout["paired_time"][k], bound_digits=3)
                for k in (keys[0], "broad_auroc", *keys[1:])
            ),
        ]
    )
    write(
        "split_summary",
        "llcccc",
        [
            "Scope",
            "Evaluation",
            r"\shortstack{Primary\\AUROC}",
            r"\shortstack{Broad\\AUROC}",
            r"\shortstack{High-$\beta$ conditional\\AUROC}",
            r"\shortstack{Above-proxy conditional\\AUROC}",
        ],
        rows,
        r"Phase-controlled scores and paired phase differences (top); pooled "
        r"forest scores and forest-minus-elapsed-time differences (bottom). "
        + phase_headline(record, latex=True)
        + " "
        r"Ranges are "
        r"point estimates across seeds 0--4, not confidence intervals. Brackets: "
        r"exploratory 95\% shot-bootstrap intervals, unadjusted for multiplicity, "
        r"at fixed predictions; individual scores "
        r"use percentile intervals, paired differences use basic intervals. "
        r"High-$\beta$: $\beta_N\geq0.8$ shot p95; above-proxy: $\beta_N/l_i>4$. "
        r"Holdout retains four records across three dates; broad paired "
        r"intervals are available for seed 0 and holdout only. The primary "
        r"seed-3 lower bound rounds to $-0.000$; its unrounded value is negative.",
        {
            "split_sensitivity.phase_controlled_auroc": phase,
            "split_sensitivity.paired_phase": pairs,
            "split_sensitivity.auroc_ranges": summary["auroc_ranges"],
            "broad_ranges": broad_ranges,
            "paired.rwm-brf - rwm-rule-elapsed-time.broad_auroc": record["paired"][
                "rwm-brf - rwm-rule-elapsed-time"
            ]["broad_auroc"],
            "split_sensitivity.paired_time_by_seed": summary["paired_time_by_seed"],
            "split_sensitivity.paired_time_ranges": summary["paired_time_ranges"],
            "leave_one_run_record_out": {
                "metrics": holdout["metrics"],
                "by_campaign": {
                    c: result["metrics"] for c, result in holdout["by_campaign"].items()
                },
                "paired_time": holdout["paired_time"],
            },
        },
        panel=phase_panel,
    )
    within_pairs = [
        pair["within_shot_auroc"][mask]
        for name, pair in record["paired"].items()
        if name.endswith(("rule-betan", "rule-betan-over-li"))
        for mask in ("primary", "broad")
    ]
    below = [abs(row["estimate"]) for row in within_pairs]
    excludes_zero = sum(row["low"] > 0 or row["high"] < 0 for row in within_pairs)
    time_mean = record["configs"]["rwm-rule-elapsed-time"]["within_shot_auroc"][
        "primary"
    ]
    rows = [
        [
            name,
            *(
                f"${record['configs'][name]['within_shot_auroc'][m]['mean']:.3f}$"
                for m in ("primary", "broad")
            ),
            latex_cell(
                record["configs"][name]["phase_controlled_auroc"], bound_digits=3
            ),
        ]
        for name in NAMES
    ]
    rows += [
        [
            name,
            *(
                latex_cell(pair["within_shot_auroc"][mask], bound_digits=3)
                for mask in ("primary", "broad")
            ),
            latex_cell(pair["phase_controlled_auroc"], bound_digits=3),
        ]
        for name, pair in record["paired"].items()
    ]
    write(
        "within_shot",
        "lccc",
        [
            "Model / rule (or difference)",
            "Primary mean AUROC",
            "Broad mean AUROC",
            "Phase-controlled AUROC",
        ],
        rows,
        f"Within-shot AUROC means on the reference split; each of "
        f"{time_mean['n_shots']} two-class "
        r"Hanson shots receives equal weight. Within-shot model means are point "
        r"estimates; "
        r"differences have 95\% basic paired shot-bootstrap intervals "
        f"({n_bootstrap:,} replicates, seed 0), unadjusted for multiplicity. "
        f"Within shot, the forest is {min(below):.3f}--{max(below):.3f} below "
        r"$\beta_N$ and $\beta_N/l_i$ "
        f"on both masks ({excludes_zero} of {len(within_pairs)} unadjusted "
        r"intervals exclude zero). "
        f"Elapsed time's {time_mean['mean']:.3f} is an artefact of the primary "
        r"mask's cutoff at the "
        r"last onset. Phase control uses primary positive-negative pairs within "
        f"campaign and {record['protocol']['phase_bin_ms']:g} ms elapsed-time "
        r"bins, weighted by pair counts. "
        r"Individual phase intervals are percentile; all three basic paired "
        r"forest-minus-scalar phase intervals include zero. One-class cells "
        r"are omitted; phase resampling is stratified by campaign.",
        {
            "paired_within_shot_auroc": {
                name: pair["within_shot_auroc"]
                for name, pair in record["paired"].items()
            },
            "phase_controlled_auroc": {
                name: row["phase_controlled_auroc"]
                for name, row in record["configs"].items()
            },
            "paired_phase_controlled_auroc": {
                name: row["phase_controlled_auroc"]
                for name, row in record["paired"].items()
            },
            **{
                f"configs.{name}.within_shot_auroc": {
                    mask: {k: row[k] for k in ("mean", "median", "n_shots")}
                    for mask, row in record["configs"][name][
                        "within_shot_auroc"
                    ].items()
                }
                for name in NAMES
            },
        },
    )
    rows = []
    campaigns = record["split_sensitivity"]["paired_time_by_campaign"]
    for campaign, seeds in campaigns.items():
        for seed, metrics in seeds.items():
            rows.append(
                [
                    campaign,
                    f"Seed {seed}",
                    *(latex_cell(metrics[k], bound_digits=3) for k in keys),
                ]
            )
        holdout = record["leave_one_run_record_out"]["paired_time_by_campaign"][
            campaign
        ]
        rows.append(
            [
                campaign,
                "Run-record holdout",
                *(latex_cell(holdout[k], bound_digits=3) for k in keys),
            ]
        )
    write(
        "campaign_pairs",
        "llccc",
        [
            "Campaign",
            "Evaluation",
            r"\shortstack{Primary\\AUROC difference}",
            r"\shortstack{High-$\beta$ conditional\\AUROC difference}",
            r"\shortstack{Above-proxy conditional\\AUROC difference}",
        ],
        rows,
        r"Forest minus elapsed time by campaign. Brackets: exploratory 95\% "
        r"basic paired "
        f"shot-bootstrap intervals ({n_bootstrap:,} resamples), conditional on fixed "
        r"fitted predictions, unadjusted for multiplicity. High-$\beta$: "
        r"$\beta_N\geq0.8$ shot p95; "
        r"above-proxy: $\beta_N/l_i>4$. All five 2014 point differences are "
        r"negative in both conditional strata; all five 2018 split-seed "
        r"intervals include zero. The included 2018 run-record holdout "
        r"intervals exclude zero: primary "
        + latex_cell(
            record["leave_one_run_record_out"]["paired_time_by_campaign"]["2018"][
                "slice_auroc"
            ],
            bound_digits=3,
        )
        + r"; high-$\beta$ "
        + latex_cell(
            record["leave_one_run_record_out"]["paired_time_by_campaign"]["2018"][
                "high_beta_auroc"
            ],
            bound_digits=3,
        )
        + ".",
        {
            "split_sensitivity.paired_time_by_campaign": campaigns,
            "leave_one_run_record_out.paired_time_by_campaign": record[
                "leave_one_run_record_out"
            ]["paired_time_by_campaign"],
        },
    )
    keys = (
        "onset_detection_rate",
        "detection_minus_uniform_reference",
        "warning_ms_median",
    )
    rows = []
    runs = forest_runs(record)
    for label, result in (
        *((f"Seed {seed}", result) for seed, result in runs.items()),
        (r"\shortstack[l]{Run-record\\holdout}", record["leave_one_run_record_out"]),
    ):
        n, metrics = result["counts"], result["metrics"]
        rows.append(
            [
                label,
                "/".join(
                    str(n[f"hanson_{k}_shots"]) for k in ("detected", "early", "missed")
                ),
                "/".join(
                    str(n[f"hanson_any_alarm_{k}_shots"])
                    for k in ("detected", "early", "missed")
                ),
                f"{n['onsets_warned']}/{n['target_onsets']}",
                *(
                    latex_cell(metrics[k], stacked=True, bound_digits=3)
                    for k in keys[:2]
                ),
                latex_cell(metrics[keys[2]], stacked=True, bound_digits=0, digits=0),
                latex_incidence(
                    n["comparison_shots_with_an_alarm"],
                    n["comparison_shots"],
                    metrics["comparison_alarm_incidence"],
                ),
                latex_incidence(
                    n["hanson_shots_with_an_unexplained_alarm"],
                    n["hanson_shots"],
                    metrics["hanson_unexplained_alarm_incidence"],
                ),
            ]
        )
    ranges = record["split_sensitivity"]["alarm_ranges"]
    rows.append(
        [
            r"\shortstack[l]{Five-split\\point range}",
            "--",
            "--",
            "--",
            *(
                latex_range(ranges[k], 0 if k == keys[2] else 3, stacked=True)
                for k in keys
            ),
            "--",
            "--",
        ]
    )
    write(
        "alarms",
        "lcccccccc",
        [
            "Evaluation",
            r"\shortstack{First alarm\\D/E/M}",
            r"\shortstack{Any alarm\\D/E/M}",
            r"\shortstack{Onsets\\warned}",
            r"\shortstack{Onset\\detection}",
            r"\shortstack{Detection minus\\random\\reference}",
            r"\shortstack{Median warning\\(ms)}",
            r"\shortstack{Comparison\\shots alarmed}",
            r"\shortstack{Hanson shots\\with unexplained\\alarms}",
        ],
        rows,
        r"Forest alarms across all five splits and the run-record holdout. "
        r"D/E/M: Detected/Early/Missed among 30 target shots. The first considered "
        r"alarm decides the primary shot category; any-warning precedence is a "
        r"sensitivity. Per-onset counts retain all alarms. "
        r"Brackets: 95\% percentile shot-bootstrap intervals, including "
        r"detection-minus-reference, conditional on fixed fitted predictions "
        r"and unadjusted for multiplicity. "
        r"Between-model differences elsewhere use basic paired intervals. "
        r"No improvement "
        r"over the approximate rate-matched random reference was established: "
        r"all five difference intervals include zero; equivalence is not "
        r"established. Comparison and Hanson "
        r"incidences are unlabelled-shot incidences, not verified false-positive "
        r"rates. Warning medians are conditional on the matching window; the "
        r"reference-split median is "
        + latex_cell(
            record["configs"]["rwm-brf"]["metrics"]["warning_ms_median"],
            bound_digits=0,
            digits=0,
        )
        + " ms, near the "
        + f"{alarm.MAX_WARNING_MS:g} ms limit.",
        {
            "configs.rwm-brf": {
                s: {"metrics": r["metrics"], "counts": r["counts"]}
                for s, r in runs.items()
            },
            "split_sensitivity.alarm_ranges": ranges,
            "leave_one_run_record_out.metrics": record["leave_one_run_record_out"][
                "metrics"
            ],
        },
    )
    physics = record["onset_physics"]
    for actual in (True, False):
        prefix = "onset_" if actual else ""
        rows = []
        for row in physics["rows"]:
            rows.append(
                [
                    str(row["campaign"]),
                    str(row["shot"]),
                    point(row["onset_ms"], 1),
                    *([] if actual else [point(row["sample_ms"], 1)]),
                    *(
                        point(row[f"{prefix}{k}"], 2)
                        for k in ("betan", "li", "betan_over_li")
                    ),
                    point(row[f"{prefix}elapsed_time_ms"], 0),
                    onset_flag(row, actual),
                    str(row["n_high_beta_pre_onset_slices"]),
                ]
            )
        below = sum(bool(row[f"{prefix}below_proxy"]) for row in physics["rows"])
        missing = sum(row[f"{prefix}efit_missing"] for row in physics["rows"])
        scope = (
            "Offline EFIT inputs held at the actual listed onset "
            f"(last sample age $\\leq{physics['efit_max_age_ms']:g}$ ms)"
            if actual
            else r"First slice in the $[\mathrm{onset}-20\,\mathrm{ms},\mathrm{onset})$ window"
        )
        write(
            "onset_actual" if actual else "onset_window",
            "rrrrrrrlr" if actual else "rrrrrrrrlr",
            [
                "Campaign",
                "Shot",
                r"\shortstack{Onset\\(ms)}",
                *([] if actual else [r"\shortstack{Window sample\\(ms)}"]),
                r"$\beta_N$",
                r"$l_i$",
                r"$\beta_N/l_i$",
                r"\shortstack{Elapsed\\(ms)}",
                "Proxy category",
                r"\shortstack{High-$\beta$\\slices}",
            ],
            rows,
            scope + f": {below} snapshots below $\\beta_N/l_i=4$; {missing} missing "
            r"EFIT snapshots. Missing inputs retain their own category. "
            r"Elapsed time starts at the first $|I_p|\geq0.5$ MA crossing. "
            r"High-$\beta$ slices are counted in the 100 ms pre-onset forecast "
            r"window. The pre-onset-window snapshot is distinct from the actual "
            r"onset; no post-onset extent is inferred.",
            {"onset_physics": physics},
            long=True,
        )
    return artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT / "tables.md")
    args = parser.parse_args()
    record = load_evaluation(OUT / "evaluation.json")
    configs = record["configs"]
    c = configs["rwm-brf"]
    run_out = record["leave_one_run_record_out"]
    rotation = json.loads((OUT / "rotation_ablation.json").read_text())
    comparison = load_evaluation(OUT / "comparison_sensitivity.json")
    sections = {
        "Label-noisy comparison-negative sensitivity — one reference-split CV": table(
            ["Hanson primary score", "baseline", "sensitivity", "paired change"],
            [
                [
                    "Phase-controlled AUROC",
                    interval(c["phase_controlled_auroc"]),
                    interval(comparison["phase_controlled_auroc"]),
                    interval(comparison["paired_change"]["phase_controlled_auroc"]),
                ],
                [
                    "Pooled AUROC (phase-confounded)",
                    interval(c["metrics"]["slice_auroc"]),
                    interval(comparison["metrics"]["slice_auroc"]),
                    interval(comparison["paired_change"]["slice_auroc"]),
                ],
            ],
        )
        + (
            "\n\nSource: outputs/labeler/rwm/comparison_sensitivity.json. "
            "Comparisons enter training as label-noisy negatives. Headline "
            "scoring and cutoff tuning use primary Hanson slices; alarm tuning "
            "keeps the original Hanson trace scope. Identical outer "
            "shots, inner splits and seeds; 5×3 nested shot-grouped CV. Paired "
            "change is sensitivity minus baseline, with 95% basic shot intervals. "
            "The paired change includes zero; whether verified stable-shot "
            "negatives would help is untested. Comparison shots alarmed: "
            f"{comparison['counts']['comparison_shots_with_an_alarm']}/"
            f"{comparison['counts']['comparison_shots']}; onsets warned: "
            f"{comparison['counts']['onsets_warned']}/"
            f"{comparison['counts']['target_onsets']}. Comparisons are not verified "
            "stable shots; this sensitivity does not replace the baseline."
        ),
        "Five-split AUROC ranges — rwm-brf (seeds 0–4)": split_scores(record),
        "Five-split paired AUROC — forest minus elapsed time": split_pairs(record),
        "Campaign paired AUROC — five splits and run-record holdout": campaign_pairs(
            record
        ),
        "Piccione-style primary scores — all models (reference split, seed 0)": scores(
            configs
        ),
        "Broader Hanson-negative sensitivity — same models and predictions": scores(
            configs, "broad"
        ),
        (
            "Within-shot AUROC — primary and broad Hanson masks "
            "(reference split, seed 0)"
        ): table(
            ["model", "mask", "shots with both classes", "median", "mean"],
            [
                [
                    name,
                    mask,
                    str(row["n_shots"]),
                    point(row["median"]),
                    point(row["mean"]),
                ]
                for name in NAMES
                for config in [configs[name]]
                for mask, row in config["within_shot_auroc"].items()
            ],
        ),
        "Paired within-shot mean AUROC differences (95% basic shot CIs)": table(
            ["forest minus scalar", "primary", "broad", "two-class shots per mask"],
            [
                [
                    name,
                    *(
                        interval(pair["within_shot_auroc"][mask])
                        for mask in ("primary", "broad")
                    ),
                    "/".join(
                        str(pair["within_shot_auroc"][mask]["n_shots"])
                        for mask in ("primary", "broad")
                    ),
                ]
                for name, pair in record["paired"].items()
            ],
        ),
        "Phase-controlled primary AUROC — within campaign and 100 ms time bins": table(
            ["model / rule", "AUROC (95% percentile shot CI)"],
            [
                [name, interval(configs[name]["phase_controlled_auroc"])]
                for name in NAMES
            ],
        ),
        "Phase split sensitivity — five seeds and run-record holdout": phase_split_table(
            record
        ),
        "Phase-bin-width sensitivity — primary mask, campaign control retained": phase_width_table(
            record
        ),
        "Paired phase-controlled AUROC — forest minus scalar": table(
            ["forest minus scalar", "AUROC difference (95% basic paired shot CI)"],
            [
                [name, interval(pair["phase_controlled_auroc"])]
                for name, pair in record["paired"].items()
            ],
        ),
        "Comparison pool — reference forest alarms by run title": table(
            ["run title", "unlabelled shots", "shots with an alarm", "alarm incidence"],
            [
                [
                    title,
                    str(row["n_shots"]),
                    str(row["n_alarm_shots"]),
                    point(row["alarm_incidence"]),
                ]
                for title, row in c["comparison_by_run_title"].items()
            ],
        ),
        "Rotation sensitivity — one reference-split no-rotation CV": table(
            ["mask", "original AUROC", "no-rotation AUROC", "paired change"],
            [
                [
                    mask,
                    interval(rotation["reference_metrics"][key]),
                    interval(rotation["metrics"][key]),
                    interval(rotation["paired_change"][key]),
                ]
                for mask, key in (("primary", "slice_auroc"), ("broad", "broad_auroc"))
            ],
        )
        + (
            "\n\nSource: outputs/labeler/rwm/rotation_ablation.json. "
            "Change is no rotation minus original forest; identical outer shots, "
            "inner splits and seeds. Input-dependence sensitivity does not "
            "measure upstream timing bias."
        ),
        "Broad-mask run-record holdout — forest minus elapsed time": table(
            ["AUROC difference (95% basic paired CI)"],
            [[interval(run_out["paired_time"]["broad_auroc"])]],
        ),
        "High-beta conditional scores — all models": scores(configs, "high_beta", True),
        "Above no-wall-proxy conditional scores — all models": scores(
            configs, "above_proxy", True
        ),
        "Leave-one-run-record-out — rwm-brf (95% shot CIs)": grouped_scores(
            {"pooled four-record holdout": run_out}
        ),
        "Leave-one-run-record-out — each held-out run record": grouped_scores(
            run_out["by_run_record"], intervals=False
        ),
        "Leave-one-run-record-out — F1 and alarms (95% shot CIs)": (
            grouped_alarm_scores({"pooled four-record holdout": run_out})
        ),
        "Leave-one-run-record-out — F1 and alarms by held-out run record": (
            grouped_alarm_scores(run_out["by_run_record"], intervals=False)
        ),
        **alarm_tables(configs),
        "Forest alarm sensitivity — five splits and run-record holdout": forest_alarm_splits(
            record
        ),
        "n=1 onset physics — actual onset, by campaign": onset_physics_table(
            record, actual=True
        ),
        "n=1 onset-window physics — first pre-onset window slice, by campaign": onset_physics_table(
            record
        ),
        "n=1 onset and window coverage summary — by campaign": onset_campaign_summary(
            record
        ),
        "Piccione-style per-shot categories — primary alarm definition": (
            shot_categories(configs)
        ),
        "Any-alarm per-shot categories — labelled sensitivity": (
            shot_categories(configs, any_alarm=True)
        ),
        "Alarm definition sensitivity — rwm-brf": alarm_sensitivity(c),
        "Legacy NSTX — separately sourced published reference": legacy_table(
            record["legacy"]
        ),
    }
    for prefix, label in (
        ("slice", "primary"),
        ("broad", "broad"),
        ("high_beta", "high-beta conditional"),
        ("above_proxy", "above-proxy conditional"),
    ):
        sections[f"Paired differences — rwm-brf versus rules, {label}"] = paired(
            record, prefix
        )
    alarm_keys = (
        "onset_detection_rate",
        "hanson_unexplained_alarm_incidence",
        "comparison_alarm_incidence",
    )
    sections["Paired alarm differences — rwm-brf versus rules (95% basic CIs)"] = table(
        [
            "first model minus rule",
            "detection difference",
            "Hanson incidence difference",
            "unlabelled incidence difference",
        ],
        [
            [name, *(interval(m[k]) for k in alarm_keys)]
            for name, m in record["paired"].items()
            if not name.endswith("rwm-rule-rwm-candidates")
        ],
    ) + (
        "\n\nAlarm-rule comparisons mostly reflect tuning monotone rules on "
        "truncated Hanson traces; they do not establish forest forecasting skill."
    )
    runs = forest_runs(record)
    sections["Split sensitivity — rwm-brf (fixed hyperparameters)"] = table(
        [
            "model",
            "fold seed",
            "primary AUROC",
            "high-beta conditional AUROC",
            "detection rate",
            "unlabelled alarm incidence",
        ],
        [
            [
                "rwm-brf",
                s,
                *(
                    f"{m[k]['estimate']:.3f}"
                    for k in (
                        "slice_auroc",
                        "high_beta_auroc",
                        "onset_detection_rate",
                        "comparison_alarm_incidence",
                    )
                ),
            ]
            for s, run in runs.items()
            for m in [run["metrics"]]
        ],
    )
    for seed, run in runs.items():
        scope = "reference split, seed 0" if seed == "0" else f"seed {seed}"
        sections[f"Campaign sensitivity — rwm-brf, {scope} (95% shot CIs)"] = (
            grouped_scores(run["by_campaign"])
        )
    sections["Leave-one-run-record-out — by campaign (95% shot CIs)"] = grouped_scores(
        run_out["by_campaign"]
    )
    text = (
        "Source: outputs/labeler/rwm/evaluation.json. Brackets report 95% "
        f"shot-bootstrap intervals ({record['protocol']['bootstrap_replicates']:,} "
        "resamples), conditional on fixed "
        "fitted predictions; exploratory, unadjusted for multiple comparisons. "
        "Individual metrics and detection-minus-reference use "
        "percentile intervals; between-model differences use basic paired intervals. "
        "Within-shot means/medians weight each two-class Hanson shot equally; "
        "one-class shots are omitted from those summaries, with counts in JSON. "
        "Phase-controlled AUROC compares only positive-negative pairs within "
        "campaign x 100 ms elapsed-time bin (at least five eligible slices), "
        "weighted by pair count; elapsed time is the residual-phase floor; phase "
        "shot resampling is stratified by campaign. "
        "High-beta "
        "means beta_N >= 0.8 times the shot's whole-window beta_N p95; "
        "above-proxy means beta_N/li > 4. Negative slices are assumed negative.\n\n"
        + "\n\n".join(f"### {name}\n\n{body}" for name, body in sections.items())
        + "\n"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    out_dir = Path(os.environ["LABELER_ROOT"]) / "round4" / "rwm"
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "script": "scripts/labeler/rwm_tables.py",
        "source": "outputs/labeler/rwm/evaluation.json",
        "source_sha256": hashlib.sha256(
            (OUT / "evaluation.json").read_bytes()
        ).hexdigest(),
        "latex": write_latex(record, out_dir),
        "supplemental_latex": write_supplemental_latex(record, out_dir),
        "markdown": str(args.out),
        "sections": list(sections),
        "documentation": write_documentation(record, comparison),
        "rotation_ablation_source": {
            "path": "outputs/labeler/rwm/rotation_ablation.json",
            "sha256": hashlib.sha256(
                (OUT / "rotation_ablation.json").read_bytes()
            ).hexdigest(),
        },
        "comparison_sensitivity_source": {
            "path": "outputs/labeler/rwm/comparison_sensitivity.json",
            "sha256": hashlib.sha256(
                (OUT / "comparison_sensitivity.json").read_bytes()
            ).hexdigest(),
        },
        "growth_source": {
            "path": "outputs/labeler/rwm/growth.json",
            "sha256": hashlib.sha256((OUT / "growth.json").read_bytes()).hexdigest(),
        },
        "external_evaluation_details": record["external_details"],
        "paper_tables": ["table_rwm.tex", "table_rwm_alarms.tex"],
        "other_tables_scope": "repository-only supplements",
    }
    provenance["rendered_latex"] = {}
    for name, artifact in {
        "main": provenance["latex"],
        **provenance["supplemental_latex"],
    }.items():
        tex_path = Path(artifact["path"])
        paths = [tex_path.with_suffix(".pdf"), *out_dir.glob(tex_path.stem + "-*.png")]
        provenance["rendered_latex"][name] = [
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in paths
            if path.is_file()
        ]
    # Preserve exact covering-check output in the small provenance record for
    # report citations. These logs are supplied by the stream's required runners;
    # rendering tables does not itself run or certify the checks.
    tmp_dir = Path(os.environ["TMPDIR"])
    provenance["supplied_check_logs"] = {
        name: {
            "path": str(tmp_dir / filename),
            "output": (tmp_dir / filename).read_text().strip(),
        }
        for name, filename in (
            ("covering_tests", "fix11-covering-tests.log"),
            ("ruff_check", "fix11-ruff.log"),
            ("python_format", "fix11-format.log"),
            ("table_and_record_validation", "fix11-validation.log"),
            ("latex_compile", "fix11-latex.log"),
            ("visual_inspection", "fix11-visual.log"),
            ("baseline_saved_replay", "fix11-rescore.log"),
            ("comparison_saved_replay", "fix11-comparison-replay.log"),
            ("rotation_saved_replay", "fix11-rotation-replay.log"),
            ("growth_regeneration", "fix11-growth.log"),
            ("empty_efit_generation", "fix11-empty-efit-green.log"),
            ("final_code_review", "fix11-review.log"),
            ("saved_prediction_preservation", "fix11-predictions.log"),
            ("artifact_hashes", "fix11-artifact-hashes.log"),
        )
        if (tmp_dir / filename).is_file()
    }
    (OUT / "presentation.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"wrote {args.out} and {out_dir / 'table_rwm.tex'}")


if __name__ == "__main__":
    main()
