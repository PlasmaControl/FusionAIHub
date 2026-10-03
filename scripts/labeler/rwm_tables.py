#!/usr/bin/env python
"""Render named RWM tables and a paper LaTeX table from the fix-round JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "labeler" / "rwm"
NAMES = (
    "rwm-brf",
    "rule-elapsed-time",
    "rule-betan",
    "rule-betan-over-li",
    "rule-rwm-candidates",
)


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


def latex_range(values, digits=2):
    return f"${values['min']:.{digits}f}$ to ${values['max']:.{digits}f}$"


def forest_runs(record):
    config = record["configs"]["rwm-brf"]
    return {"0": config, **config["split_seeds"]}


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
        "high-beta AUROC is about chance or below across the five splits "
        f"({split_range(ranges, 2)}; reference-split CI "
        f"[{reference['low']:.2f}, {reference['high']:.2f}]) and below elapsed time "
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
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in rows])


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
            *(
                split_range(
                    record["split_sensitivity"]["alarm_ranges"][k],
                    0 if k == keys[2] else 3,
                )
                for k in keys
            ),
        ]
    )
    betan = record["configs"]["rule-betan"]
    betan_count = betan["counts"]
    return table(
        [
            "evaluation",
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
        if not name.endswith("rule-rwm-candidates")
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


def shot_categories(configs):
    rows = []
    for name in NAMES:
        n = configs[name]["counts"]
        denominator = n["hanson_target_shots"]
        rows.append(
            [
                name,
                f"{n['hanson_detected_shots']}/{denominator}",
                f"{n['hanson_missed_shots']}/{denominator}",
                f"{n['hanson_early_shots']}/{denominator}",
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
        "with an n=1 target: any Detected alarm takes precedence over Early, "
        "then Missed. Early means more than 400 ms before a listed onset. "
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
                (
                    f"{legacy['comparison_shots_with_an_alarm']}/"
                    f"{legacy['comparison_shots']}"
                ),
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


def latex_cell(m, stacked=False, bound_digits=2):
    if m["estimate"] is None:
        return "--"
    point = f"${m['estimate']:.3f}$"
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


def write_latex(record, out_dir):
    configs, legacy = record["configs"], record["legacy"]
    labels = {
        "rwm-brf": r"\texttt{rwm-brf}",
        "rule-elapsed-time": "Elapsed time",
        "rule-betan": r"$\beta_N$",
        "rule-betan-over-li": r"$\beta_N/l_i$",
        "rule-rwm-candidates": "RWM screen",
    }
    keys = (
        "slice_auroc",
        "slice_auprc",
        "slice_f1",
        "slice_tpr",
        "slice_fpr",
        "high_beta_auroc",
    )
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lcccccc@{}}",
        r"\toprule",
        r"\multicolumn{7}{@{}l}{Tokamak-SI (DIII-D): reference split (seed 0)} \\",
        r"\midrule",
        (
            r"Model / rule & \shortstack{Primary\\AUROC} & "
            r"\shortstack{Primary\\AUPRC} & \shortstack{Primary\\F1} & "
            r"\shortstack{Slice\\TPR} & \shortstack{Slice\\FPR} & "
            r"\shortstack{High-$\beta$ conditional\\AUROC} \\"
        ),
        r"\midrule",
    ]
    provenance = {}
    for name in NAMES:
        lines.append(
            labels[name]
            + " & "
            + " & ".join(
                latex_cell(configs[name]["metrics"][k], stacked=True) for k in keys
            )
            + r" \\"
        )
        lines.append(r"\addlinespace[1.5pt]")
        provenance[name] = {
            k: {
                "json_path": f"configs.{name}.metrics.{k}",
                **configs[name]["metrics"][k],
            }
            for k in keys
        }
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\smallskip",
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        r"Scope & Primary AUROC & High-$\beta$ conditional AUROC & Above-proxy conditional AUROC \\",
        r"\midrule",
    ]
    ranges = record["split_sensitivity"]["auroc_ranges"]
    for campaign, row in ranges.items():
        lines.append(
            f"{campaign.capitalize()} (five splits) & "
            + " & ".join(
                latex_range(row[k])
                for k in ("slice_auroc", "high_beta_auroc", "above_proxy_auroc")
            )
            + r" \\"
        )
    holdout = record["leave_one_run_record_out"]
    lines += [
        "Run-record holdout & "
        + " & ".join(
            latex_cell(holdout["metrics"][k])
            for k in ("slice_auroc", "high_beta_auroc", "above_proxy_auroc")
        )
        + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\smallskip",
        r"\begin{tabular}{@{}lc@{}}",
        r"\toprule",
        (r"\multicolumn{2}{@{}l}{Legacy (Piccione et al., NSTX)} \\"),
        r"\midrule",
        f"Slice AUROC / F1 & {legacy['slice_auroc']:.3f} / --" + r" \\",
        (
            f"Slice TPR / FPR & {100 * legacy['slice_tpr']:.1f}"
            + r"\% / "
            + f"{100 * legacy['slice_fpr']:.1f}"
            + r"\% \\"
        ),
        (
            "Detected unstable shots & "
            f"{legacy['onsets_warned']}/{legacy['target_onsets']}" + r" \\"
        ),
        (
            "False-positive stable shots & "
            f"{legacy['comparison_shots_with_an_alarm']}/"
            f"{legacy['comparison_shots']}" + r" \\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    summary = record["split_sensitivity"]
    campaign_range = ranges["2014"]["high_beta_auroc"]
    campaign_reference = configs["rwm-brf"]["by_campaign"]["2014"]["metrics"][
        "high_beta_auroc"
    ]
    n_hanson = configs["rwm-brf"]["counts"]["hanson_shots"]
    n_bootstrap = record["protocol"]["bootstrap_replicates"]
    within = configs["rule-elapsed-time"]["within_shot_auroc"]["primary"]
    campaign_pairs = summary["paired_time_by_campaign"]["2014"]
    n_exclude_zero = sum(
        not (row["high_beta_auroc"]["low"] <= 0 <= row["high_beta_auroc"]["high"])
        for row in campaign_pairs.values()
    )
    broad_time = record["paired"]["rwm-brf - rule-elapsed-time"]["broad_auroc"]
    broad_ratio = record["paired"]["rwm-brf - rule-betan-over-li"]["broad_auroc"]
    caption = (
        f"Retrospective DIII-D forecasting on {n_hanson} Hanson shots with offline ZIPFIT "
        r"and postprocessed magnetic-RMS inputs, and unverified negative coverage. "
        r"Positives precede listed $n=1$ "
        r"onsets by at most 100 ms. Primary negatives end at the last onset, so "
        r"elapsed time ranks within-shot almost perfectly (median AUROC "
        f"{within['median']:.1f}, mean {within['mean']:.2f}, "
        f"{within['n_shots']} shots). "
        r"High-$\beta$ conditions on $\beta_N\geq0.8$ "
        r"shot whole-window p95; above-proxy conditions on $\beta_N/l_i>4$. "
        r"Brackets: 95\% "
        f"shot-bootstrap intervals ({n_bootstrap:,} resamples), conditional on fixed fitted "
        r"predictions. Between-model differences use basic paired intervals. "
        r"Five-split forest-minus-elapsed-time AUROC ranges: primary "
        + latex_range(summary["paired_time_ranges"]["slice_auroc"], 3)
        + r"; high-$\beta$ "
        + latex_range(summary["paired_time_ranges"]["high_beta_auroc"])
        + r"; above-proxy "
        + latex_range(summary["paired_time_ranges"]["above_proxy_auroc"])
        + r". The primary seed-3 lower bound is borderline near zero. "
        r"Campaign 2014 high-$\beta$ AUROC is about chance or below ("
        + latex_range(campaign_range)
        + f"; reference-split CI [{campaign_reference['low']:.2f}, "
        + f"{campaign_reference['high']:.2f}]) and below elapsed time on every split "
        + f"(point estimates; CI excludes zero on {n_exclude_zero} of 5 seeds). "
        r"Under broad negatives, forest minus elapsed time is "
        + latex_cell(broad_time, bound_digits=3)
        + r" (run-record holdout "
        + latex_cell(holdout["paired_time"]["broad_auroc"], bound_digits=3)
        + r"); forest minus $\beta_N/l_i$ is "
        + latex_cell(broad_ratio, bound_digits=3)
        + r". The forest matches the best single scalar under either mask "
        r"(elapsed time on primary, $\beta_N/l_i$ on broad); no onset-specific skill. "
        r"Run-record holdout retains four records across three dates. "
        r"Legacy uses different inputs and expert-reviewed stable shots; results "
        r"are not comparable."
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
        "split_ranges": {
            "json_path": "split_sensitivity.auroc_ranges",
            "values": ranges,
        },
        "paired_time_ranges": {
            "json_path": "split_sensitivity.paired_time_ranges",
            "values": summary["paired_time_ranges"],
        },
        "within_shot_elapsed_time": {
            "json_path": "configs.rule-elapsed-time.within_shot_auroc.primary",
            "values": within,
        },
        "broad_pairs": {
            "paired.rwm-brf - rule-elapsed-time.broad_auroc": broad_time,
            "paired.rwm-brf - rule-betan-over-li.broad_auroc": broad_ratio,
            "leave_one_run_record_out.paired_time.broad_auroc": holdout["paired_time"][
                "broad_auroc"
            ],
        },
        "paired_time_by_campaign": {
            "json_path": "split_sensitivity.paired_time_by_campaign",
            "values": summary["paired_time_by_campaign"],
        },
        "run_record_cells": {
            k: {
                "json_path": f"leave_one_run_record_out.metrics.{k}",
                **holdout["metrics"][k],
            }
            for k in ("slice_auroc", "high_beta_auroc", "above_proxy_auroc")
        },
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
                    "comparison_shots_with_an_alarm",
                    "comparison_shots",
                )
            },
            "slice_f1": "not available in source digest",
        },
    }


def write_supplemental_latex(record, out_dir):
    """Keep full paired, alarm and snapshot evidence readable in separate tables."""
    artifacts = {}
    n_bootstrap = record["protocol"]["bootstrap_replicates"]

    def write(name, columns, header, rows, caption, source, long=False):
        environment = "longtable" if long else "tabular"
        lines = [r"\begingroup"] if long else [r"\begin{table*}[t]", r"\centering"]
        lines += [r"\small", r"\setlength{\tabcolsep}{4pt}"]
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
        lines += [" & ".join(row) + r" \\" for row in rows]
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
        r"Forest minus elapsed time by campaign. Brackets: 95\% basic paired "
        f"shot-bootstrap intervals ({n_bootstrap:,} resamples), conditional on fixed "
        r"fitted predictions. High-$\beta$: $\beta_N\geq0.8$ shot p95; "
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
        ("Run-record holdout", record["leave_one_run_record_out"]),
    ):
        n, metrics = result["counts"], result["metrics"]
        rows.append(
            [
                label,
                f"{n['onsets_warned']}/{n['target_onsets']}",
                *(latex_cell(metrics[k], bound_digits=3) for k in keys[:2]),
                interval(metrics[keys[2]], 0),
            ]
        )
    ranges = record["split_sensitivity"]["alarm_ranges"]
    rows.append(
        [
            "Five-split point range",
            "--",
            *(latex_range(ranges[k], 0 if k == keys[2] else 3) for k in keys),
        ]
    )
    betan = record["configs"]["rule-betan"]
    n, m = betan["counts"], betan["metrics"]
    write(
        "alarms",
        "lcccc",
        [
            "Evaluation",
            r"\shortstack{Onsets\\warned}",
            r"\shortstack{Onset\\detection}",
            r"\shortstack{Detection minus\\random reference}",
            r"\shortstack{Median warning\\(ms)}",
        ],
        rows,
        r"Forest alarms across all five splits and the run-record holdout. "
        r"Brackets: 95\% percentile shot-bootstrap intervals, including "
        r"detection-minus-reference, conditional on fixed fitted predictions. "
        r"Between-model differences elsewhere use basic paired intervals. "
        r"No improvement "
        r"over the approximate rate-matched random reference was established: "
        r"all five difference intervals include zero; equivalence is not "
        r"established. Warning medians condition on detected onsets. The "
        r"reference-split $\beta_N$ rule warns "
        f"{n['onsets_warned']}/{n['target_onsets']} onsets; detection minus "
        "reference is "
        + latex_cell(m["detection_minus_uniform_reference"], bound_digits=3)
        + ". Its low detection coverage limits this result.",
        {
            "configs.rwm-brf": {s: r["metrics"] for s, r in runs.items()},
            "split_sensitivity.alarm_ranges": ranges,
            "leave_one_run_record_out.metrics": record["leave_one_run_record_out"][
                "metrics"
            ],
            "configs.rule-betan": {"counts": n, "metrics": m},
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
    record = json.loads((OUT / "evaluation.json").read_text())
    configs = record["configs"]
    c = configs["rwm-brf"]
    run_out = record["leave_one_run_record_out"]
    sections = {
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
                for name, config in configs.items()
                for mask, row in config["within_shot_auroc"].items()
            ],
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
            if not name.endswith("rule-rwm-candidates")
        ],
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
        "fitted predictions. Individual metrics and detection-minus-reference use "
        "percentile intervals; between-model differences use basic paired intervals. "
        "Within-shot means/medians weight each two-class Hanson shot equally; "
        "one-class shots are omitted from those summaries, with counts in JSON. "
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
            ("covering_tests", "tests.log"),
            ("ruff_check", "ruff.log"),
            ("ruff_format", "format.log"),
            ("table_ruff_check", "tables-ruff.log"),
            ("table_ruff_format", "tables-format.log"),
            ("table_validation", "tables-validation.log"),
            ("latex_compile", "tables-latex.log"),
            ("visual_inspection", "tables-visual.log"),
        )
        if (tmp_dir / filename).is_file()
    }
    (OUT / "presentation.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"wrote {args.out} and {out_dir / 'table_rwm.tex'}")


if __name__ == "__main__":
    main()
