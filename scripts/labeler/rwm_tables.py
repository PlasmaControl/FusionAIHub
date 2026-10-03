#!/usr/bin/env python
"""Render named RWM tables and a paper LaTeX table from the fix-round JSON."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "labeler" / "rwm"
NAMES = (
    "rwm-brf",
    "rule-time-since-flattop",
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


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in rows])


def scores(configs, prefix="slice", conditional=False):
    keys = [f"{prefix}_auroc", f"{prefix}_auprc"]
    if prefix == "slice":
        keys.append("slice_f1")
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
        + (["F1 (95% CI)"] if prefix == "slice" else [])
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
        ("high_beta", "high-beta"),
        ("above_proxy", "above-proxy"),
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
            "unlabelled shots with an alarm",
        ],
        rows,
    ) + (
        "\n\nDetected, Early and Missed are mutually exclusive on Hanson shots "
        "with an n=1 target: any Detected alarm takes precedence over Early, "
        "then Missed. Early means more than 400 ms before a listed onset. "
        "Unlabelled-shot alarms are incidence, not a verified false-positive rate."
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
        "explanation onset. This tolerance extends beyond the primary slice mask, "
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


def latex_cell(m):
    if m["estimate"] is None:
        return "--"
    point = f"{m['estimate']:.3f}"
    if m["low"] is None or m["high"] is None:
        return point
    return (
        r"\shortstack{"
        + point
        + r"\\{["
        + f"{m['low']:.3f},"
        + r"}\\{"
        + f"{m['high']:.3f}]"
        + "}}"
    )


def write_latex(record, out_dir):
    configs, legacy = record["configs"], record["legacy"]
    labels = {
        "rwm-brf": r"\texttt{rwm-brf}",
        "rule-time-since-flattop": "Elapsed time",
        "rule-betan": r"$\beta_N$",
        "rule-betan-over-li": r"$\beta_N/l_i$",
        "rule-rwm-candidates": "RWM screen",
    }
    keys = ("slice_auroc", "slice_auprc", "slice_f1", "high_beta_auroc")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{1.5pt}",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        (
            r"Model / rule & \shortstack{Primary\\AUROC} & "
            r"\shortstack{Primary\\AUPRC} & \shortstack{Primary\\F1} & "
            r"\shortstack{High-$\beta$\\AUROC} \\"
        ),
        r"\midrule",
    ]
    provenance = {}
    for name in NAMES:
        lines.append(
            labels[name]
            + " & "
            + " & ".join(latex_cell(configs[name]["metrics"][k]) for k in keys)
            + r" \\"
        )
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
        r"\begin{tabular}{@{}lc@{}}",
        r"\toprule",
        (
            r"\multicolumn{2}{@{}l}{\shortstack[l]{Legacy NSTX RUS forest\\"
            r"Piccione et al. (2022)}} \\"
        ),
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
        (
            r"\caption{DIII-D onset forecasting on Hanson shots. Positives are "
            r"within 100 ms before a listed $n=1$ onset; assumed negatives are "
            r"earlier than the last onset, excluding event aftermath. Complete "
            r"negative coverage is unverified. High-$\beta$ uses "
            r"$\beta_N\geq0.8$ times the shot's 95th percentile. Brackets give "
            r"95\% shot-bootstrap intervals from 1,000 resamples of held-out "
            r"predictions. Within-phase discrimination is not distinguishable "
            r"from chance; reversed in 2014. The retrospective candidate screen "
            r"never fires before a listed onset and is constant zero on primary "
            r"slices. The separately sourced Legacy reference uses a different "
            r"machine (NSTX), expert-reviewed stable shots, different inputs and "
            r"validation; its published test results are not comparable to this "
            r"DIII-D benchmark. Legacy F1 and intervals are not available in "
            r"the source digest.}"
        ),
        r"\label{tab:rwm-baseline}",
        r"\end{table}",
    ]
    target = out_dir / "table_rwm.tex"
    target.write_text("\n".join(lines) + "\n")
    return {
        "path": str(target),
        "cells": provenance,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT / "tables.md")
    args = parser.parse_args()
    record = json.loads((OUT / "evaluation.json").read_text())
    configs = record["configs"]
    c = configs["rwm-brf"]
    day_out = record["leave_one_run_day_out"]
    sections = {
        "Piccione-style primary scores — all models": scores(configs),
        "Broader Hanson-negative sensitivity — same models and predictions": scores(
            configs, "broad"
        ),
        "High-beta conditional scores — all models": scores(configs, "high_beta", True),
        "Above no-wall-proxy conditional scores — all models": scores(
            configs, "above_proxy", True
        ),
        "Campaign sensitivity — rwm-brf (95% shot CIs)": grouped_scores(
            c["by_campaign"]
        ),
        "Leave-one-run-day-out — rwm-brf (95% shot CIs)": grouped_scores(
            {"pooled four-day holdout": day_out}
        ),
        "Leave-one-run-day-out — each held-out run day": grouped_scores(
            day_out["by_run_day"], intervals=False
        ),
        "Leave-one-run-day-out — F1 and alarms (95% shot CIs)": (
            grouped_alarm_scores({"pooled four-day holdout": day_out})
        ),
        "Leave-one-run-day-out — F1 and alarms by held-out run day": (
            grouped_alarm_scores(day_out["by_run_day"], intervals=False)
        ),
        **alarm_tables(configs),
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
        ("high_beta", "high-beta"),
        ("above_proxy", "above-proxy"),
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
    runs = {"0": c, **c["split_seeds"]}
    sections["Split sensitivity — rwm-brf (fixed hyperparameters)"] = table(
        [
            "model",
            "fold seed",
            "primary AUROC",
            "high-beta AUROC",
            "detection rate",
            "unlabelled alarm incidence",
        ],
        [
            [
                "rwm-brf",
                s,
                *(
                    f"{(m[k]['estimate'] if s == '0' else m[k]):.3f}"
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
    text = (
        "\n\n".join(f"### {name}\n\n{body}" for name, body in sections.items()) + "\n"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    out_dir = Path(os.environ["LABELER_ROOT"]) / "round4" / "rwm"
    provenance = {
        "script": "scripts/labeler/rwm_tables.py",
        "source": "outputs/labeler/rwm/evaluation.json",
        "latex": write_latex(record, out_dir),
        "markdown": str(args.out),
        "sections": list(sections),
    }
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
            ("latex_compile", "latex.log"),
        )
        if (tmp_dir / filename).is_file()
    }
    (OUT / "presentation.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"wrote {args.out} and {out_dir / 'table_rwm.tex'}")


if __name__ == "__main__":
    main()
