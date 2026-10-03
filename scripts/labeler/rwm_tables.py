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


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in rows])


def scores(configs, prefix="slice", conditional=False):
    keys = [f"{prefix}_auroc", f"{prefix}_auprc"]
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
    ]
    return table(
        [
            "first model minus rule",
            "AUROC difference (95% CI)",
            "AUPRC difference (95% CI)",
        ],
        rows,
    )


def latex_cell(m):
    if m["estimate"] is None:
        return "--"
    point = f"{m['estimate']:.3f}"
    if m["low"] is None or m["high"] is None:
        return point
    return (
        r"\shortstack{" + point + r"\\{[" + f"{m['low']:.3f}, {m['high']:.3f}]" + "}}"
    )


def write_latex(configs, out_dir):
    labels = {
        "rwm-brf": r"\texttt{rwm-brf}",
        "rule-time-since-flattop": "Elapsed time",
        "rule-betan": r"$\beta_N$",
        "rule-betan-over-li": r"$\beta_N/l_i$",
        "rule-rwm-candidates": "RWM screen",
    }
    keys = ("slice_auroc", "slice_auprc", "high_beta_auroc")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        (
            r"Model / rule & \shortstack{Piccione-style\\AUROC} & "
            r"\shortstack{Piccione-style\\AUPRC} & \shortstack{High-$\beta$\\AUROC} \\"
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
        (
            r"\caption{DIII-D onset forecasting on Hanson shots. Positives are "
            r"within 100 ms before a listed $n=1$ onset; assumed negatives are "
            r"earlier than the last onset, excluding event aftermath. Complete "
            r"negative coverage is unverified. High-$\beta$ uses "
            r"$\beta_N\geq0.8$ times the shot's 95th percentile. Brackets give "
            r"95\% shot-bootstrap intervals from 1,000 resamples of held-out "
            r"predictions. The candidate screen uses a retrospective threshold.}"
        ),
        r"\label{tab:rwm-baseline}",
        r"\end{table}",
    ]
    target = out_dir / "table_rwm.tex"
    target.write_text("\n".join(lines) + "\n")
    return {"path": str(target), "cells": provenance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT / "tables.md")
    args = parser.parse_args()
    record = json.loads((OUT / "evaluation.json").read_text())
    configs = record["configs"]
    sections = {
        "Piccione-style primary scores — all models": scores(configs),
        "Broader Hanson-negative sensitivity — same models and predictions": scores(
            configs, "broad"
        ),
        "High-beta conditional scores — all models": scores(configs, "high_beta", True),
        "Above no-wall-proxy conditional scores — all models": scores(
            configs, "above_proxy", True
        ),
        **alarm_tables(configs),
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
    sections["Paired alarm differences — rwm-brf versus rules"] = table(
        [
            "first model minus rule",
            "detection difference",
            "Hanson incidence difference",
            "unlabelled incidence difference",
        ],
        [
            [name, *(interval(m[k]) for k in alarm_keys)]
            for name, m in record["paired"].items()
        ],
    )
    c = configs["rwm-brf"]
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
        "latex": write_latex(configs, out_dir),
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
