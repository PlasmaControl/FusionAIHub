#!/usr/bin/env python
"""Render named RWM tables and a paper LaTeX table from the evaluation JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from labeler.rwm import alarm, features
from labeler.rwm.metrics import CONDITIONAL_KEYS
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


def sci(metric, latex=False, plus=False):
    """Estimate and interval with true minus signs (math mode for LaTeX)."""
    estimate = f"{metric['estimate']:+.3f}" if plus else f"{metric['estimate']:.3f}"
    bounds = f"{metric['low']:.3f}, {metric['high']:.3f}"
    if latex:
        return f"${estimate}$ $[{bounds}]$"
    return f"{estimate} [{bounds}]".replace("-", "−")


def num(value, latex=False, plus=False):
    text = f"{value:+.3f}" if plus else f"{value:.3f}"
    return f"${text}$" if latex else text.replace("-", "−")


def texify(text):
    """Plain wording to LaTeX text (numbers are already in math mode)."""
    return (
        text.replace("βN/li", r"$\beta_N/l_i$")
        .replace("βN", r"$\beta_N$")
        .replace("high-β", r"high-$\beta$")
        .replace("`rwm-brf`", r"\texttt{rwm-brf}")
        .replace("–", "--")
        .replace("%", r"\%")
    )


RULES = ("rwm-rule-elapsed-time", "rwm-rule-betan", "rwm-rule-betan-over-li")


def pair_facts(pair, n):
    """Seed-level facts of one forest-minus-rule phase comparison."""
    estimates = [row["estimate"] for row in pair["by_seed"].values()]
    return {
        "min": pair["min"],
        "max": pair["max"],
        "positive": sum(value > 0 for value in estimates),
        "excluding_zero": pair["cis_excluding_zero"],
        "above_zero": sum(row["low"] > 0 for row in pair["by_seed"].values()),
        "holdout": pair["holdout"],
        "n": n,
    }


def headline_facts(record):
    """Every number behind the headline, read from the JSON; nothing is asserted.

    The wording is chosen from these values, so a rerun that changes the outcome
    changes the sentence instead of failing the build.
    """
    summary = record["split_sensitivity"]
    seeds = [str(seed) for seed in summary["seeds"]]
    n = len(seeds)
    configs, paired = record["configs"], record["paired"]
    time = pair_facts(summary["paired_phase"]["rwm-rule-elapsed-time"], n)
    beta = pair_facts(summary["paired_phase"]["rwm-rule-betan-over-li"], n)
    holdout = record["leave_one_run_record_out"]
    campaigns = holdout["paired_time_by_campaign"]
    widths = {
        name: configs[name]["phase_bin_width_sensitivity"]
        for name in ("rwm-brf", "rwm-rule-elapsed-time")
    }
    campaign_2014 = summary["paired_time_by_campaign"]["2014"]
    primary_2014 = [campaign_2014[s]["slice_auroc"]["estimate"] for s in seeds]
    within = {
        name: paired[f"rwm-brf - {name}"]["within_shot_auroc"] for name in RULES
    }
    return {
        "n": n,
        "bin_ms": record["protocol"]["phase_bin_ms"],
        "phase_range": summary["phase_controlled_auroc"],
        "lowest_seed": summary["phase_controlled_auroc"]["lowest_seed"],
        "forest_phase": configs["rwm-brf"]["phase_controlled_auroc"],
        "time_phase": configs["rwm-rule-elapsed-time"]["phase_controlled_auroc"],
        "beta_phase": configs["rwm-rule-betan-over-li"]["phase_controlled_auroc"],
        "time": time,
        "beta": beta,
        # Elapsed-time and βN/li margins above zero on most splits would be skill.
        "skill_beyond_phase": min(time["above_zero"], beta["above_zero"]) > n / 2,
        "within": {name: row["primary"] for name, row in within.items()},
        "within_broad": {name: row["broad"] for name, row in within.items()},
        "elapsed_within": configs["rwm-rule-elapsed-time"]["within_shot_auroc"][
            "primary"
        ]["mean"],
        "holdout_campaign": {c: campaigns[c]["slice_auroc"] for c in ("2014", "2018")},
        "holdout_pooled": holdout["paired_time"]["slice_auroc"],
        "margin_200": widths["rwm-brf"]["200"] - widths["rwm-rule-elapsed-time"]["200"],
        "primary_2014": (min(primary_2014), max(primary_2014)),
    }


def headline(record, latex=False):
    """The verdict in one sentence; the balanced numbers follow it."""
    f = headline_facts(record)
    text = (
        "Skill beyond elapsed time and βN/li under phase control on most splits"
        if f["skill_beyond_phase"]
        else "No demonstrated skill beyond elapsed time or βN/li under phase control"
    )
    return texify(text) if latex else text


def balanced_statement(record, latex=False):
    """The result in one clause per metric: phase, within shot and holdout.

    Every figure is read from the record, and each metric reports all of its
    comparators, so the sentence does not select the favourable one.
    """
    f = headline_facts(record)
    n, time = f["n"], f["time"]
    holdout_up = time["holdout"]["estimate"] > 0
    if time["positive"] == n and holdout_up:
        where = f"positive on {n}/{n} seeds and the holdout"
    else:
        where = (
            f"positive on {time['positive']}/{n} seeds and "
            f"{'positive' if holdout_up else 'negative'} on the holdout"
        )
    weakest = (
        "seed 0 (the forest's weakest split)"
        if f["lowest_seed"] == "0"
        else f"seed 0 (the forest's weakest split is seed {f['lowest_seed']})"
    )
    within = f["within"]
    holdouts = f["holdout_campaign"]

    def m(value, plus=False):
        return num(value, latex, plus)

    text = (
        f"{headline(record)}; `rwm-brf` is an equilibrium-scalar timing baseline "
        "and no input senses the RWM. "
        f"Phase-controlled AUROC, {weakest}: forest {sci(f['forest_phase'], latex)}, "
        f"elapsed time {m(f['time_phase']['estimate'])}, "
        f"βN/li {m(f['beta_phase']['estimate'])}. "
        f"Over {n} seeds the forest's margin over elapsed time is "
        f"{m(time['min'], True)} to {m(time['max'], True)}, {where}, with an "
        f"interval excluding zero on {time['excluding_zero']}/{n}. "
        "Within shot, forest minus elapsed time, βN and βN/li is "
        f"{sci(within['rwm-rule-elapsed-time'], latex)}, "
        f"{sci(within['rwm-rule-betan'], latex)} and "
        f"{sci(within['rwm-rule-betan-over-li'], latex)}; primary-mask "
        "within-shot ranking is dominated by phase (elapsed time "
        f"{m(f['elapsed_within'])}). "
        "Run-record holdout, forest minus elapsed time (primary AUROC): "
        f"2018 {sci(holdouts['2018'], latex, True)}, "
        f"2014 {sci(holdouts['2014'], latex, True)}, "
        f"pooled {sci(f['holdout_pooled'], latex, True)}."
    )
    return texify(text) if latex else text


def brief_statement(record):
    """The balanced result in one README clause: all metrics, no selected one."""
    f = headline_facts(record)
    time, within, holdouts = f["time"], f["within"], f["holdout_campaign"]
    head = headline(record)
    return (
        head[0].lower()
        + head[1:]
        + f"; phase-controlled margin over elapsed time "
        f"{num(time['min'], plus=True)} to {num(time['max'], plus=True)} over "
        f"{f['n']} seeds (interval excluding zero on {time['excluding_zero']}/"
        f"{f['n']}); within shot, forest minus elapsed time, βN and βN/li "
        f"{num(within['rwm-rule-elapsed-time']['estimate'], plus=True)}, "
        f"{num(within['rwm-rule-betan']['estimate'], plus=True)} and "
        f"{num(within['rwm-rule-betan-over-li']['estimate'], plus=True)}, a "
        "ranking dominated by phase (elapsed time "
        f"{num(f['elapsed_within'])}); run-record holdout 2018 "
        f"{num(holdouts['2018']['estimate'], plus=True)}, 2014 "
        f"{num(holdouts['2014']['estimate'], plus=True)}; equilibrium-scalar "
        "timing baseline, no input senses the RWM"
    )


def bin_choice(record, latex=False):
    """The post-hoc bin-width disclosure, with the 200 ms margin from the record."""
    f = headline_facts(record)
    text = (
        f"The {f['bin_ms']:g} ms bin width was chosen after an earlier run with "
        f"200 ms bins, where seed 0's forest-minus-elapsed-time margin was "
        f"{num(f['margin_200'], latex)}; it is a post-hoc choice."
    )
    return text


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
    by_seed = summary["paired_time_by_seed"]
    near_zero = min(by_seed, key=lambda s: abs(by_seed[s]["slice_auroc"]["low"]))
    borderline = by_seed[near_zero]["slice_auroc"]
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
        f"discharge-phase information. The primary seed-{near_zero} lower bound "
        f"is {borderline['low']:.4f}, the one nearest zero; a bootstrap-bound "
        "sign change alone would not "
        "establish robust superiority."
    )


def campaign_notes(record, latex=False):
    """Computed campaign wording: split-seed signs, intervals and both holdouts."""
    summary = record["split_sensitivity"]["paired_time_by_campaign"]
    holdout = record["leave_one_run_record_out"]["paired_time_by_campaign"]
    seeds = list(summary["2014"])
    n = len(seeds)

    def negative(campaign, key):
        return sum(summary[campaign][s][key]["estimate"] < 0 for s in seeds)

    def excluding(campaign, key):
        return sum(
            m[key]["low"] > 0 or m[key]["high"] < 0 for m in summary[campaign].values()
        )

    low, high = headline_facts(record)["primary_2014"]
    text = (
        f"In 2014, {negative('2014', 'high_beta_auroc')} of {n} high-β and "
        f"{negative('2014', 'above_proxy_auroc')} of {n} above-proxy point "
        "differences are negative, and the primary differences range from "
        f"{num(low, latex, True)} to {num(high, latex, True)} "
        f"({excluding('2014', 'slice_auroc')} of {n} intervals exclude zero); in "
        f"2018, {excluding('2018', 'slice_auroc')} of {n} split-seed primary "
        "intervals exclude zero. Run-record holdout, primary AUROC difference: "
        f"2014 {sci(holdout['2014']['slice_auroc'], latex, True)}, "
        f"2018 {sci(holdout['2018']['slice_auroc'], latex, True)}; high-β: "
        f"2014 {sci(holdout['2014']['high_beta_auroc'], latex, True)}, "
        f"2018 {sci(holdout['2018']['high_beta_auroc'], latex, True)}."
    )
    return texify(text) if latex else text.replace("high-β", "high-beta")


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
        "forest high-beta AUROC on the reference split is "
        f"{'below chance ' if reference['high'] < 0.5 else ''}"
        f"({reference['estimate']:.3f} [{reference['low']:.2f}, "
        f"{reference['high']:.2f}]); the scalar rules are near chance. "
        f"Forest five-split high-beta range: {split_range(ranges, 3)}. "
        + campaign_notes(record)
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


def alarm_reference_statement(record):
    """Detection minus the random-alarm reference: what the intervals allow."""
    intervals = [
        run["metrics"]["detection_minus_uniform_reference"]
        for run in forest_runs(record).values()
    ]
    intervals.append(
        record["leave_one_run_record_out"]["metrics"][
            "detection_minus_uniform_reference"
        ]
    )
    excluding = sum(m["low"] > 0 or m["high"] < 0 for m in intervals)
    if excluding == 0:
        return (
            "No improvement over the approximate rate-matched random-alarm "
            f"reference was established: all {len(intervals)} detection-difference "
            "intervals include zero. This does not establish equivalence."
        )
    return (
        f"The detection-minus-random-reference interval excludes zero on {excluding} "
        f"of {len(intervals)} evaluations (signs in the table); the reference is "
        "approximate and this does not establish onset-specific skill."
    )


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
        "\n\n"
        + alarm_reference_statement(record)
        + " Warning medians "
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


def rounding_note(summary):
    """Say when the primary lower bound nearest zero rounds to zero."""
    by_seed = summary["paired_time_by_seed"]
    seed = min(by_seed, key=lambda s: abs(by_seed[s]["slice_auroc"]["low"]))
    low = by_seed[seed]["slice_auroc"]["low"]
    if round(low, 3) != 0:
        return f"The primary seed-{seed} lower bound is ${low:.3f}$."
    return (
        f"The primary seed-{seed} lower bound rounds to ${'-' if low < 0 else ''}"
        f"0.000$; its unrounded value is {'negative' if low < 0 else 'positive'}."
    )


def stack(top, bottom=None, align=""):
    """Two-line cell (value over interval); a hidden interval keeps rows aligned."""
    position = f"[{align}]" if align else ""
    hidden = r"\phantom{{\scriptsize $[\,]$}}"
    return r"\shortstack" + position + "{" + top + r"\\" + (bottom or hidden) + "}"


def latex_parts(m, bound_digits=2, digits=3):
    """The point estimate and, when it has one, its bracketed interval."""
    if m["estimate"] is None:
        return "--", None
    point = f"${m['estimate']:.{digits}f}$"
    if m["low"] is None or m["high"] is None:
        return point, None
    bounds = (
        r"{\scriptsize $["
        + f"{m['low']:.{bound_digits}f}, {m['high']:.{bound_digits}f}]"
        + "$}"
    )
    return point, bounds


def latex_cell(m, stacked=False, bound_digits=2, digits=3):
    point, bounds = latex_parts(m, bound_digits, digits)
    if bounds is None:
        return point
    if stacked:
        return stack(point, bounds)
    return point + " " + bounds


def latex_incidence(numerator, denominator, metric):
    """Count and rate on one line, the interval beneath."""
    if not denominator:
        return "--"
    point, bounds = latex_parts(metric, bound_digits=3)
    return stack(f"{numerator}/{denominator} ({point})", bounds)


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


def finite_draw_summary(replicates, *records):
    """Smallest finite-draw share over every bootstrap interval in the records."""
    found = {"exact": [], "conditional": [], "undefined": []}

    def walk(node, key=None):
        if isinstance(node, dict):
            if "n_finite" in node and "estimate" in node:
                if node["estimate"] is None:
                    found["undefined"].append(node["n_finite"])
                    return
                kind = "conditional" if key in CONDITIONAL_KEYS else "exact"
                found[kind].append(node["n_finite"])
                return
            for name, value in node.items():
                walk(value, name)
        elif isinstance(node, list):
            for value in node:
                walk(value, key)

    for record in records:
        walk(record)
    exact = found["exact"]
    return {
        "intervals": len(exact),
        "intervals_below_all_draws": sum(n < replicates for n in exact),
        "minimum_finite_share": min(exact) / replicates,
        "conditional_intervals": len(found["conditional"]),
        "undefined_estimates": len(found["undefined"]),
    }


CANDIDATES = "rwm-rule-rwm-candidates"


def candidate_facts(record):
    """The review screen scored as a rule, read from the evaluation record."""
    config, audit = record["configs"][CANDIDATES], record["screen_audit"]
    n, m = config["counts"], config["metrics"]
    return {
        "auroc": m["slice_auroc"],
        "warned": n["onsets_warned"],
        "onsets": n["target_onsets"],
        "alarmed": n["comparison_shots_with_an_alarm"],
        "comparison": n["comparison_shots"],
        "incidence": m["comparison_alarm_incidence"]["estimate"],
        "primary_calls": audit["primary_calls"],
        "hanson_calls": audit["hanson_calls"],
        "calls_before_first_onset": audit["calls_before_first_listed_onset"],
    }


def candidate_line(record):
    """One line for the doc and tables.md: the screen as a rule baseline."""
    f = candidate_facts(record)
    return (
        "As a rule baseline the `rwm_candidates` screen scores primary AUROC "
        f"{f['auroc']['estimate']:.3f}, warns {f['warned']}/{f['onsets']} onsets "
        f"and alarms on {f['alarmed']}/{f['comparison']} ({f['incidence']:.0%}) "
        f"unlabelled comparison shots; it makes {f['primary_calls']} calls on "
        f"primary Hanson slices ({f['hanson_calls']} Hanson calls in all, "
        f"{f['calls_before_first_onset']} before the first listed onset), so its "
        "AUROC is the no-information value, not a measured skill "
        f"(`E#/configs/{CANDIDATES}`, `E#/screen_audit`)."
    )


def candidate_table(record):
    config = record["configs"][CANDIDATES]
    n, m = config["counts"], config["metrics"]
    return table(
        [
            "rule",
            "primary AUROC (95% CI)",
            "primary AUPRC (95% CI)",
            "primary F1",
            "onsets warned",
            "unlabelled shots with an alarm",
            "primary slice calls",
        ],
        [
            [
                CANDIDATES,
                interval(m["slice_auroc"]),
                interval(m["slice_auprc"]),
                point(m["slice_f1"]["estimate"]),
                f"{n['onsets_warned']}/{n['target_onsets']}",
                (
                    f"{n['comparison_shots_with_an_alarm']}/{n['comparison_shots']} "
                    f"({m['comparison_alarm_incidence']['estimate']:.0%})"
                ),
                str(record["screen_audit"]["primary_calls"]),
            ]
        ],
    ) + "\n\n" + candidate_line(record)


def absent_collapse_text(shots):
    """Collapses inside category-0 spans, flagged rather than re-categorised."""
    found = shots["assumed_absent_collapses"]
    flagged = found["shots_with_a_collapse"]
    cached = len(found["neutral_beam_power_cached_shots"])
    cached_text = "none" if cached == 0 else str(cached)
    return (
        f"**Collapses inside category 0.** A βN fall of at least "
        f"{features.COLLAPSE_DROP:.0%} within {features.COLLAPSE_WITHIN_MS:g} ms, "
        f"from βN of at least {features.COLLAPSE_MIN_BETAN:g}, starts inside the "
        f"assumed-absent span of {len(flagged)} of {found['hanson_shots_scanned']} "
        f"Hanson shots ({', '.join(str(s) for s in flagged)}; "
        f"{len(found['events'])} events, `S#/assumed_absent_collapses`). Neutral-beam "
        f"power is cached for {cached_text} of "
        "them, so the cause (a beam trip, a mode, a disruption precursor) is not "
        "established. Each such span keeps category 0 and carries "
        "`attrs.unexplained_beta_collapse`; no span was re-categorised."
    )


def abstract_text(record, shots, growth):
    """Five plain sentences: task, shots, labels, result, limitation."""
    hanson, windows = shots["hanson"], shots["windows"]
    balance = shots["comparison"]["balance"]
    count = record["configs"]["rwm-brf"]["counts"]
    f = headline_facts(record)
    within = f["within"]
    return (
        "1. **Task.** From equilibrium scalars on a 10 ms grid, say whether a "
        "resistive wall mode (RWM) onset comes within the next 100 ms in a DIII-D "
        "discharge that is known to have one.\n"
        f"2. **Shots.** {hanson['shots']} shots from Jeremy Hanson's onset lists "
        f"({balance['2014']['hanson']['shots']} in 2014, "
        f"{balance['2018']['hanson']['shots']} in 2018; {hanson['listed_onsets']} "
        f"listed onsets merge into {hanson['n1_events']} n=1 and "
        f"{hanson['n2_events']} n=2 events) and {shots['comparison']['chosen']} "
        "matched comparison shots without a listed onset.\n"
        f"3. **Labels.** A {windows['rows']}-row tiered interval table "
        "(categories 0 assumed absent, 1 minimal present, 2 uncertain or candidate, "
        "4 unassessed; no verified absence) and forecast labels of "
        f"{count['positive_slices']} positive and {count['negative_slices']:,} "
        "assumed-negative 10 ms slices.\n"
        f"4. **Result.** {headline(record)}: a balanced random forest after "
        "Piccione 2022 (`rwm-brf`) has phase-controlled AUROC "
        f"{interval(f['forest_phase'])} against "
        f"{point(f['time_phase']['estimate'])} for elapsed time and "
        f"{point(f['beta_phase']['estimate'])} for βN/li on seed 0, and within "
        "shot, forest minus elapsed time, βN and βN/li is "
        f"{sci(within['rwm-rule-elapsed-time'])}, {sci(within['rwm-rule-betan'])} "
        f"and {sci(within['rwm-rule-betan-over-li'])}.\n"
        "5. **Limitation.** No input senses the RWM, so `rwm-brf` is an "
        "equilibrium-scalar timing baseline, not an RWM predictor; the next step "
        "is a fetch of the radial-field sensor with confirmed semantics."
    )


def glossary_text(record):
    protocol, configs = record["protocol"], record["configs"]
    return "\n".join(
        [
            (
                "- **Phase-controlled AUROC:** AUROC over positive-negative slice pairs "
                "from the same campaign and the same "
                f"{protocol['phase_bin_ms']:g} ms elapsed-time bin; a bin needs both "
                f"classes and at least {protocol['phase_min_slices']} slices, bins are "
                "weighted by pair count, ties count half. It removes the ranking that "
                "discharge phase supplies on its own."
            ),
            (
                "- **Residual-phase floor:** the phase-controlled AUROC of elapsed time "
                f"itself ({interval(configs['rwm-rule-elapsed-time']['phase_controlled_auroc'])}), "
                "what time since flat-top still ranks inside one bin. A model must beat "
                "it to show more than phase."
            ),
            (
                "- **Primary and broad masks:** primary positives are slices with a "
                "merged n=1 onset in the next 100 ms; primary negatives end at the "
                "last n=1 onset (aftermath and n=2-only time excluded). The broad mask "
                "keeps the same scores and positives and adds post-last-onset and "
                "n=2-only Hanson time as negatives."
            ),
            (
                "- **Assumed negatives:** slices of a Hanson shot with no listed onset "
                "in the next 100 ms. The onset lists are unverified for completeness, "
                "so none is a confirmed absence."
            ),
            (
                "- **Categories 0/1/2/4:** interval labels of the export. 0 assumed "
                "absent (before the first onset's 100 ms precursor); 1 minimal present "
                "([o, o+10 ms)); 2 uncertain, either a Hanson pre-onset window "
                "([o−20, o) ms, tier `onset_window_uncertain`) or a comparison-shot "
                "screen candidate (tier `unlabelled_screen`); 4 unassessed. None is "
                "verified absence."
            ),
            (
                "- **High-β and above-proxy strata:** evaluation masks, βN at least 0.8 "
                "times the shot's p95 and βN/li above 4; not predictors."
            ),
        ]
    )


def physics_text(growth, shots, data_audit):
    """What each candidate input does and does not sense, from G, S and D."""
    audit = shots["input_audit"]
    by_role = audit["dusbradial_by_role_campaign"]
    zero_2014 = sum(
        by_role[f"{role}_2014"]["zero_only_shots"] for role in ("hanson", "comparison")
    )
    shots_2014 = sum(
        by_role[f"{role}_2014"]["shots"] for role in ("hanson", "comparison")
    )
    low, high = audit["dusbradial_corrupted_shot_range"]
    onset = growth["max_growth_per_s_n1"]
    rank = growth["slope_rank_test_n1"]
    pre, every = rank["pre_onset_controls"], rank["all_controls"]
    late_share = rank["controls_after_first_onset"] / every["n_controls"]
    traces = data_audit["operations_n1_probe"]["traces"]

    def probe(shot, name):
        return traces[f"{shot}_{name}"]

    def amp(shot):
        c, i, u = (probe(shot, n) for n in ("cn1bamp", "iln1bamp", "iun1bamp"))
        return c, i, u

    c1, i1, u1 = amp(156785)
    c2, i2, u2 = amp(158021)
    c3, i3, u3 = amp(176068)
    return (
        "No model input senses the RWM, and the evidence that it could is "
        "negative:\n\n"
        f"- `dusbradial` is unusable: zero on {zero_2014}/{shots_2014} selected 2014 "
        f"traces ({by_role['hanson_2014']['zero_only_shots']}/"
        f"{by_role['hanson_2014']['shots']} Hanson) and flagged corrupted for "
        f"shots {low}–{high}, which covers all 2018 Hanson shots "
        "(`S#/input_audit`).\n"
        "- N1RMS shows no detectably larger maximum slope at onsets: the median "
        f"maximum log-slope is **{onset['median']:.1f}/s at {onset['n']} n=1 "
        f"onsets versus {pre['control_median']:.1f}/s at {pre['n_controls']:,} "
        "controls before the shot's first onset** under an identical search "
        f"(rank AUC {pre['rank_auc']:.3f}, Mann-Whitney p "
        f"{pre['mannwhitney_p']:.2f}). {late_share:.0%} of all controls fall "
        "after the shot's first onset, so the pre-onset controls are the "
        f"comparison; against all {every['n_controls']:,} controls the median is "
        f"{every['control_median']:.1f}/s (rank AUC {every['rank_auc']:.3f}, p "
        f"{every['mannwhitney_p']:.2f}; `G#/slope_rank_test_n1`).\n"
        "- The OPERATIONS n=1 amplitudes (CN1BAMP, ILN1BAMP, IUN1BAMP) are "
        "applied-field amplitudes, not a plasma response. Flat-top medians (Gauss, "
        "`D#/operations_n1_probe`): 2014 shots 156785 and 158021 have I-coil "
        f"amplitudes {i1['median']:.1f}/{u1['median']:.1f} and "
        f"{i2['median']:.1f}/{u2['median']:.1f} G and C-coil "
        f"{c1['median']:.2f} and {c2['median']:.2f} G, varying with the coil "
        f"programme (5th to 95th percentile of the I-coil amplitude "
        f"{i1['p5']:.1f}–{i1['p95']:.1f} G on 156785 and "
        f"{i2['p5']:.1f}–{i2['p95']:.1f} G on 158021); 2018 shot 176068 has a C-coil plateau of "
        f"{c3['median']:.1f} G ({c3['share_within_5_percent_of_median']:.0%} of "
        "flat-top samples within 5% of the median) and I-coil "
        f"{i3['median']:.2f}/{u3['median']:.2f} G. Their semantics are "
        "unverified (`sensor_probe.json`), and none was promoted to an input.\n\n"
        "The near-flat 2018 C-coil n=1 plateau points to a pre-programmed or "
        "static applied field (for example error-field correction) rather than "
        "active feedback; this does not establish the cause, and it is a "
        "physical reason to stratify every result by campaign (the campaign "
        "tables do). N1RMS cannot distinguish an RWM from a tearing mode "
        "or an applied-field response. The forest can therefore only learn the "
        "βN/li-and-time trajectory of a Hanson shot, and its labels cannot be "
        "checked against any input. **Next step:** fetch the radial-field sensor "
        "with confirmed semantics (PTDATA ONSBRADIAL, disruption-py's fallback for "
        "DUSBRADIAL, is the lead candidate; none is cached for the zero-trace "
        "2014 shots, `O#`)."
    )


def audit_text(record, comparison, rotation, data_audit):
    """Counts behind the phase bootstrap, the dropped slices and the flat-top starts."""
    eligibility = record["phase_eligibility"]
    point, boot = eligibility["point"], eligibility["bootstrap"]
    missing = record["forecast_label_audit"]["primary_without_elapsed_time"]
    roster = data_audit["roster"]
    outlier = next(iter(data_audit["outliers"].values()))
    start_text = f"{outlier['flattop_start_ms']:.0f}".replace("-", "−")
    first_text = f"{outlier['first_ip_at_least_0p5_ma_ms']:.0f}".replace("-", "−")
    replicates = record["protocol"]["bootstrap_replicates"]
    finite = finite_draw_summary(replicates, record, comparison, rotation)
    return (
        "- **Phase-bootstrap eligibility** is decided on resampled slices: a "
        "duplicated shot can lift a bin to the minimum cell size, so eligibility "
        "is not decided on unique slices (documented, not changed). On the "
        f"reference split {point['eligible_cells']} of "
        f"{point['cells_with_both_classes']} two-class bins reach the minimum, "
        f"holding {point['positive_slices_in_eligible_cells']:,}/"
        f"{point['positive_slices']:,} positives and "
        f"{point['negative_slices_in_eligible_cells']:,}/"
        f"{point['negative_slices']:,} "
        "negatives (the rest sit in bins with only one class, which are not "
        f"scored); {point['pairs_in_eligible_cells']:,} pairs are scored. "
        f"Replaying the {boot['replicates']:,} campaign-stratified draws, "
        f"{boot['replicates_with_a_duplicate_only_cell']:,} resamples score a bin "
        "that is eligible only through duplicates: on average "
        f"{boot['mean_duplicate_only_cells']:.3f} bins, at most "
        f"{boot['max_duplicate_only_cells']}, carrying at most "
        f"{boot['max_pair_share_in_duplicate_only_cells']:.2%} of that resample's "
        "pairs (`E#/phase_eligibility`).\n"
        f"- **Primary slices without elapsed time:** {missing['negative']} "
        f"assumed negatives and {missing['positive']} positives on "
        f"{missing['negative_shots']} shots precede the first |Ip| ≥ 0.5 MA sample, "
        "so no phase bin contains them (`E#/forecast_label_audit/"
        "primary_without_elapsed_time`).\n"
        f"- **Flat-top start of comparison shot {outlier['shot']}** is "
        f"{start_text} ms, the only negative start among "
        f"{roster['shots']} roster shots (next smallest "
        f"{roster['smallest_nonnegative_start_ms']:.0f} ms, median "
        f"{roster['median_start_ms']:.0f} ms). Its saved Ip is a smooth rise from "
        f"{outlier['ip_ma_at_ms']['-300']:.3f} MA at −300 ms to "
        f"{outlier['ip_ma_at_ms']['-200']:.2f} MA at −200 ms and its peak is only "
        f"{outlier['peak_ip_ma']:.2f} MA, so the 50%-of-peak crossing "
        f"({outlier['half_peak_ip_ma']:.2f} MA) falls before t = 0; the first "
        f"|Ip| ≥ 0.5 MA sample is at {first_text} ms. "
        "This is consistent with a fast ramp on a low-current shot rather than "
        "a corrupted time base, but a current this high this early is unusual "
        "for DIII-D timing, so a time-base offset is not excluded; the shot is "
        "flagged, not corrected. Comparison shots enter only alarm incidence, "
        "the interval export and the comparison-negative sensitivity's training "
        "(`D#/outliers`).\n"
        "- **Finite bootstrap draws:** every interval records its finite draws "
        f"(`n_finite`) and a build fails below 90% of {replicates:,}. Over "
        f"{finite['intervals']:,} intervals in E, C and A the smallest finite share "
        f"is {finite['minimum_finite_share']:.1%}; "
        f"{finite['intervals_below_all_draws']} intervals lose draws. Warning "
        "means and medians exist only in resamples that warn an onset and are "
        f"exempt ({finite['conditional_intervals']} intervals), as are "
        f"{finite['undefined_estimates']} intervals whose point estimate is "
        "itself undefined."
    )


def write_documentation(record, comparison):
    """Refresh marked numerical prose and README model lines from saved scores."""
    configs, protocol = record["configs"], record["protocol"]
    forest, elapsed = configs["rwm-brf"], configs["rwm-rule-elapsed-time"]
    shots_record = json.loads((OUT / "shots.json").read_text())
    growth_record = json.loads((OUT / "growth.json").read_text())
    data_audit = json.loads((OUT / "data_audit.json").read_text())
    rotation = json.loads((OUT / "rotation_ablation.json").read_text())
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
            balanced_statement(record)
            + " Sources: `E#/{split_sensitivity,paired,leave_one_run_record_out}`."
        ),
        "abstract": abstract_text(record, shots_record, growth_record),
        "glossary": glossary_text(record),
        "physics": physics_text(growth_record, shots_record, data_audit),
        "audit": audit_text(record, comparison, rotation, data_audit),
        "phase": (
            f"Pooled primary AUROC on the reference split is "
            f"{interval(forest['metrics']['slice_auroc'])} for the forest and "
            f"{interval(elapsed['metrics']['slice_auroc'])} for elapsed time; both "
            "are phase-confounded, so neither is a skill measure. Phase "
            f"control uses {protocol['phase_bin_ms']:g} ms bins with at least "
            f"{protocol['phase_min_slices']} slices per bin. "
            + bin_choice(record)
            + "\n\n"
        )
        + table(
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
            "(`E#/screen_audit` and that screen's `counts`). "
            + candidate_line(record)
        ),
        "absent_collapses": absent_collapse_text(shots_record),
    }
    audit = record["forecast_label_audit"]
    missing, rotation = audit["efit_missing"], audit["rotation_raw"]["rot_core_khz"]
    first, first_time = forest["first_onset_only"], elapsed["first_onset_only"]
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
    clause = {
        "rwm-brf": brief_statement(record),
        "rwm-rule-elapsed-time": "its phase-controlled AUROC is the residual-phase "
        "floor",
        "rwm-rule-betan": "single-scalar rule, not an RWM sensor",
        "rwm-rule-betan-over-li": "single-scalar rule, not an RWM sensor",
    }
    model_lines = []
    screen = candidate_facts(record)
    clause[CANDIDATES] = (
        "retrospective review-editor screen (n=1 RMS above the flat-top median "
        "plus 6 MAD, βN > 4 li, at least 10 ms) with "
        f"{screen['primary_calls']} calls on primary Hanson slices, so its AUROC "
        f"is the no-information value; warns {screen['warned']}/{screen['onsets']} "
        f"onsets and alarms on {screen['alarmed']}/{screen['comparison']} "
        f"({screen['incidence']:.0%}) unlabelled comparison shots; a review "
        "aid, not a forecast comparator"
    )
    for name in (*NAMES, CANDIDATES):
        config = configs[name]
        model_lines.append(
            f"- {name} | 2026_10_03 | Phase-controlled AUROC: "
            f"{interval(config['phase_controlled_auroc'])} | "
            f"AUROC: {interval(config['metrics']['slice_auroc'])} (primary; phase-confounded) | "
            f"AUPRC: {interval(config['metrics']['slice_auprc'])} | "
            f"F1: {interval(config['metrics']['slice_f1'])} | "
            f"Limitation: {clause[name]}"
        )
    models = (
        "## Models\n**stable**: none\n\n**latest**: rwm-brf\n\n**all**:\n\n"
        + "\n".join(model_lines)
        + "\n\n"
        + balanced_statement(record)
        + "\n\nReference split (seed 0), Hanson primary forecasts, assumed "
        f"negatives, {protocol['phase_bin_ms']:g} ms phase bins with at least "
        f"{protocol['phase_min_slices']} slices; intervals are exploratory 95% "
        "shot-bootstrap intervals. Definitions, the split sensitivities and the "
        "post-hoc bin choice are in [protocol and results](../../../docs/labeler/"
        "rwm_baseline.md).\n\n"
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


APPENDIX_NOTE = (
    "% Appendix table: the paper's main-text row group is table_rwm_compact.tex.\n"
)


def write_latex(record, out_dir):
    configs = record["configs"]
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
        APPENDIX_NOTE.rstrip(),
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
    lines += [r"\bottomrule", r"\end{tabular}"]
    protocol = record["protocol"]
    n = configs["rwm-brf"]["counts"]
    caption = (
        balanced_statement(record, latex=True)
        + " "
        + bin_choice(record, latex=True)
        + " Labels come from Hanson's onset list, with assumed negatives. "
        "Phase control compares pairs within campaign and "
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
        r"predictions, unadjusted for multiplicity. Published NSTX results are a "
        r"different machine and are not comparable."
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
        "scope": "appendix",
    }


def write_compact_latex(record, out_dir):
    """The paper's row group: forest, elapsed time and βN/li, at column width."""
    configs, protocol = record["configs"], record["protocol"]
    keys = ("slice_auroc", "slice_auprc", "slice_f1")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{1pt}",
        r"\begin{tabular}{@{}lccccc@{}}",
        r"\toprule",
        r"\multicolumn{6}{@{}l}{RWM, DIII-D (assumed negatives)} \\",
        r"\midrule",
        (
            r"Model / rule & \shortstack{Phase-\\controlled\\AUROC} & "
            r"\shortstack{Within-\\shot\\AUROC} & "
            r"\shortstack{Pooled\\AUROC} & \shortstack{Pooled\\AUPRC} & "
            r"\shortstack{Pooled\\F1} \\"
        ),
        r"\midrule",
    ]
    names = {
        "rwm-brf": "Forest",
        "rwm-rule-elapsed-time": "Elapsed time",
        "rwm-rule-betan-over-li": r"$\beta_N/l_i$",
    }
    cells = {}
    for name, label in names.items():
        config = configs[name]
        within = config["within_shot_auroc"]["primary"]["mean"]
        phase, pooled = config["phase_controlled_auroc"], config["metrics"]
        cells_row = [
            latex_cell(phase, stacked=True, bound_digits=2),
            stack(f"${within:.3f}$"),
            *(latex_cell(pooled[k], stacked=True, bound_digits=2) for k in keys),
        ]
        lines.append(stack(label, align="l") + " & " + " & ".join(cells_row) + r" \\")
        lines.append(r"\addlinespace[2pt]")
        cells[name] = {
            "phase_controlled_auroc": f"configs.{name}.phase_controlled_auroc",
            "within_shot_mean_auroc": f"configs.{name}.within_shot_auroc.primary.mean",
            **{k: f"configs.{name}.metrics.{k}" for k in keys},
        }
    lines += [r"\bottomrule", r"\end{tabular}"]
    n = configs["rwm-brf"]["counts"]
    caption = (
        balanced_statement(record, latex=True)
        + f" Phase control compares pairs within campaign and "
        f"{protocol['phase_bin_ms']:g} ms bins (at least "
        f"{protocol['phase_min_slices']} slices); elapsed time is the "
        "residual-phase floor. Within-shot AUROC is the mean over two-class "
        "Hanson shots on the primary mask (point estimate). Table rows show the "
        f"reference split, {n['positive_slices']} positive and "
        f"{n['negative_slices']:,} assumed-negative slices; pooled scores are "
        r"phase-confounded. Brackets: exploratory 95\% shot-bootstrap intervals."
    )
    lines += [
        r"\caption{" + caption + "}",
        r"\label{tab:rwm-compact}",
        r"\end{table}",
    ]
    target = out_dir / "table_rwm_compact.tex"
    source = OUT / target.name
    tex = "\n".join(lines) + "\n"
    target.write_text(tex)
    source.write_text(tex)
    return {
        "path": str(target),
        "source_path": str(source),
        "cells": cells,
        "caption": caption,
        "caption_words": len(caption.split()),
        "scope": "paper (column width)",
    }


def write_supplemental_latex(record, out_dir):
    """Keep full paired, alarm and snapshot evidence readable in separate tables."""
    artifacts = {}
    n_bootstrap = record["protocol"]["bootstrap_replicates"]

    def write(
        name, columns, header, rows, caption, source, long=False, panel=None, space=None
    ):
        environment = "longtable" if long else "tabular"
        lines = [APPENDIX_NOTE.rstrip()]
        lines += [r"\begingroup"] if long else [r"\begin{table*}[t]", r"\centering"]
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
            " & ".join(paper_name(cell, latex=True) for cell in row)
            + r" \\"
            + (rf" \addlinespace[{space}]" if space else "")
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
            "scope": "appendix",
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
        + balanced_statement(record, latex=True)
        + " "
        + bin_choice(record, latex=True)
        + " "
        r"Ranges are "
        r"point estimates across seeds 0--4, not confidence intervals. Brackets: "
        r"exploratory 95\% shot-bootstrap intervals, unadjusted for multiplicity, "
        r"at fixed predictions; individual scores "
        r"use percentile intervals, paired differences use basic intervals. "
        r"High-$\beta$: $\beta_N\geq0.8$ shot p95; above-proxy: $\beta_N/l_i>4$. "
        r"Holdout retains four records across three dates; broad paired "
        r"intervals are available for seed 0 and holdout only. "
        + rounding_note(summary),
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
    facts = headline_facts(record)
    within_pairs = [*facts["within"].values(), *facts["within_broad"].values()]
    excludes_zero = sum(row["low"] > 0 or row["high"] < 0 for row in within_pairs)
    phase_pairs = [row["phase_controlled_auroc"] for row in record["paired"].values()]
    phase_excluding = sum(row["low"] > 0 or row["high"] < 0 for row in phase_pairs)

    def rule_list(mask):
        return ", ".join(
            sci(row, latex=True) for row in facts[mask].values()
        )

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
        headline(record, latex=True)
        + f". Within-shot AUROC means on the reference split; each of "
        f"{time_mean['n_shots']} two-class "
        r"Hanson shots receives equal weight. Within-shot model means are point "
        r"estimates; "
        r"differences have 95\% basic paired shot-bootstrap intervals "
        f"({n_bootstrap:,} replicates, seed 0), unadjusted for multiplicity. "
        "Forest minus elapsed time, $\\beta_N$ and $\\beta_N/l_i$ is "
        f"{rule_list('within')} on the primary mask and {rule_list('within_broad')} "
        f"on the broad mask ({excludes_zero} of {len(within_pairs)} unadjusted "
        r"intervals exclude zero). "
        f"Elapsed time's {time_mean['mean']:.3f} is an artefact of the primary "
        r"mask's cutoff at the last onset, so within-shot ranking on that mask is "
        r"dominated by phase. Phase control uses primary positive-negative pairs "
        f"within campaign and {record['protocol']['phase_bin_ms']:g} ms "
        r"elapsed-time bins, weighted by pair counts. "
        r"Individual phase intervals are percentile; "
        f"{phase_excluding} of {len(phase_pairs)} basic paired forest-minus-rule "
        r"phase intervals exclude zero on the reference split. One-class cells "
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
        r"above-proxy: $\beta_N/l_i>4$. "
        + campaign_notes(record, latex=True),
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
                label if label.startswith(r"\shortstack") else stack(label, align="l"),
                stack(
                    "/".join(
                        str(n[f"hanson_{k}_shots"])
                        for k in ("detected", "early", "missed")
                    )
                ),
                stack(
                    "/".join(
                        str(n[f"hanson_any_alarm_{k}_shots"])
                        for k in ("detected", "early", "missed")
                    )
                ),
                stack(f"{n['onsets_warned']}/{n['target_onsets']}"),
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
            stack("--"),
            stack("--"),
            stack("--"),
            *(
                latex_range(ranges[k], 0 if k == keys[2] else 3, stacked=True)
                for k in keys
            ),
            stack("--"),
            stack("--"),
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
        "D/E/M: Detected/Early/Missed among "
        f"{record['configs']['rwm-brf']['counts']['hanson_target_shots']} target "
        "shots. The first considered "
        r"alarm decides the primary shot category; any-warning precedence is a "
        r"sensitivity. Per-onset counts retain all alarms. "
        r"Brackets: 95\% percentile shot-bootstrap intervals, including "
        r"detection-minus-reference, conditional on fixed fitted predictions "
        r"and unadjusted for multiplicity. "
        r"Between-model differences elsewhere use basic paired intervals. "
        + alarm_reference_statement(record)
        + r" Comparison and Hanson "
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
        space="4pt",
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
        "Rule baseline — rwm_candidates review screen (reference split, seed 0)": (
            candidate_table(record)
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
    supplemental = write_supplemental_latex(record, out_dir)
    provenance_supplemental = [k for k in supplemental if k != "alarms"]
    provenance = {
        "script": "scripts/labeler/rwm_tables.py",
        "source": "outputs/labeler/rwm/evaluation.json",
        "source_sha256": hashlib.sha256(
            (OUT / "evaluation.json").read_bytes()
        ).hexdigest(),
        "latex": write_latex(record, out_dir),
        "compact_latex": write_compact_latex(record, out_dir),
        "supplemental_latex": supplemental,
        "markdown": str(args.out),
        "sections": list(sections),
        "documentation": write_documentation(record, comparison),
        "balanced_statement": {
            "text": balanced_statement(record),
            "words": len(balanced_statement(record).split()),
            "facts": headline_facts(record),
        },
        "candidate_rule": candidate_facts(record),
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
        "paper_tables": ["table_rwm_compact.tex"],
        "appendix_tables": [
            "table_rwm.tex",
            "table_rwm_alarms.tex",
            *(f"table_rwm_{k}.tex" for k in provenance_supplemental),
        ],
        "other_tables_scope": "appendix or repository-only supplements",
        "nstx_legacy": "text only (different machine, not comparable): E#/legacy",
    }
    provenance["rendered_latex"] = {}
    for name, artifact in {
        "main": provenance["latex"],
        "compact": provenance["compact_latex"],
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
            ("covering_tests", "fix13-covering-tests.log"),
            ("ruff_check", "fix13-ruff.log"),
            ("python_format", "fix13-format.log"),
            ("table_and_record_validation", "fix13-validation.log"),
            ("latex_compile", "fix13-latex.log"),
            ("visual_inspection", "fix13-visual.log"),
            ("baseline_saved_replay", "fix13-rescore.log"),
            ("comparison_saved_replay", "fix13-comparison-replay.log"),
            ("rotation_saved_replay", "fix13-rotation-replay.log"),
            ("growth_regeneration", "fix13-growth.log"),
            ("data_audit", "fix13-data-audit.log"),
            ("figure_regeneration", "fix13-figure.log"),
            ("saved_prediction_preservation", "fix13-predictions.log"),
            ("artifact_hashes", "fix13-artifact-hashes.log"),
        )
        if (tmp_dir / filename).is_file()
    }
    (OUT / "presentation.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"wrote {args.out} and {out_dir / 'table_rwm.tex'}")


if __name__ == "__main__":
    main()
