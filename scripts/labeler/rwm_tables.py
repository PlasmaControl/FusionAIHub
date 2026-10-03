#!/usr/bin/env python
"""Render the tables of `docs/labeler/rwm_baseline.md` from the JSON records.

Reads `outputs/labeler/rwm/{evaluation,shots,growth}.json` and writes the Markdown
tables to `outputs/labeler/rwm/tables.md` (and stdout), so every number in the document
is a copy of a number in a record and none is typed by hand. The published (Legacy)
numbers are the constants below, from the digest of Piccione et al. 2022.

    python scripts/labeler/rwm_tables.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "labeler" / "rwm"

#: Piccione et al. 2022 (NSTX), digest `.tmp/label_papers/Piccione_2022_Nucl._Fusion_62_
#: 036002.md`: random under-sampling forest on its 28-shot test set (11 unstable, 17
#: stable shots).
LEGACY = {
    "slice_auroc": 0.918,
    "slice_tpr": 0.924,
    "slice_fpr": 0.214,
    "detected": 10,
    "missed": 1,
    "unstable_shots": 11,
    "false_positives": 2,
    "stable_shots": 17,
}


def interval(metric: dict | None, digits: int = 3) -> str:
    """`estimate [low, high]`, or a dash when the metric is undefined."""
    if not metric or metric.get("estimate") is None:
        return "-"
    low, high = metric.get("low"), metric.get("high")
    text = f"{metric['estimate']:.{digits}f}"
    if low is None or high is None:
        return text
    return f"{text} [{low:.{digits}f}, {high:.{digits}f}]"


def plain(value, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def balance(shots: dict) -> str:
    rows = []
    for year, groups in shots["comparison"]["balance"].items():
        for name, label in (
            ("hanson", "Hanson shots"),
            ("matched", "chosen comparison"),
            ("pool_not_chosen", "pool, not chosen"),
        ):
            g = groups[name]
            rows.append(
                [
                    year,
                    label,
                    str(g["shots"]),
                    plain(g["betan_p95_mean"], 2),
                    plain(g["betan_over_li_p95_mean"], 2),
                ]
            )
    return table(
        ["campaign", "set", "shots", "mean flat-top beta_N p95", "mean beta_N/l_i p95"],
        rows,
    )


def headline(configs: dict, names: list[str]) -> str:
    rows = []
    for name in names:
        c = configs[name]
        m = c["metrics"]
        rows.append(
            [
                name,
                interval(m["slice_auroc"]),
                interval(m["slice_auprc"]),
                interval(m["slice_f1"]),
                interval(m["slice_tpr"]),
                interval(m["slice_fpr"]),
            ]
        )
    return table(
        ["model", "AUROC", "AUPRC", "F1", "TPR", "FPR"],
        rows,
    )


def mixed(configs: dict, names: list[str]) -> str:
    rows = []
    for name in names:
        m = configs[name]["metrics"]
        rows.append(
            [
                name,
                interval(m["mixed_auroc"]),
                interval(m["mixed_auprc"]),
                interval(m["mixed_fpr"]),
            ]
        )
    return table(["model", "AUROC", "AUPRC", "FPR"], rows)


def per_shot(configs: dict, names: list[str]) -> str:
    rows = []
    for name in names:
        c = configs[name]
        m, n = c["metrics"], c["counts"]
        rows.append(
            [
                name,
                f"{n['onsets_warned']} of {n['target_onsets']}",
                interval(m["onset_detection_rate"]),
                plain(c.get("chance_detection")),
                interval(m["warning_ms_median"], 0),
                f"{n['hanson_shots_with_a_false_alarm']} of {n['hanson_shots']}",
                interval(m["hanson_false_alarm_shot_rate"]),
                interval(m["hanson_false_alarms_per_shot"], 2),
                f"{n['comparison_shots_with_an_alarm']} of {n['comparison_shots']}",
                interval(m["comparison_false_alarm_shot_rate"]),
                interval(m["comparison_false_alarms_per_shot"], 2),
            ]
        )
    return table(
        [
            "model",
            "onsets warned",
            "detection rate",
            "detection by chance",
            "median warning (ms)",
            "Hanson shots with a false alarm",
            "rate",
            "per shot",
            "comparison shots with an alarm",
            "rate",
            "per shot",
        ],
        rows,
    )


def variants(configs: dict, names: list[str]) -> str:
    rows = []
    for name in names:
        c = configs[name]
        m, n = c["metrics"], c["counts"]
        rows.append(
            [
                name,
                str(n["positive_slices"]),
                interval(m["slice_auroc"]),
                interval(m["slice_auprc"]),
                interval(m["onset_detection_rate"]),
                interval(m["hanson_false_alarm_shot_rate"]),
                interval(m["comparison_false_alarm_shot_rate"]),
            ]
        )
    return table(
        [
            "configuration",
            "positive slices",
            "AUROC",
            "AUPRC",
            "detection rate",
            "Hanson false-alarm shot rate",
            "comparison false-alarm shot rate",
        ],
        rows,
    )


def estimate(value):
    """A point estimate from either a bootstrap dict or a plain number."""
    return value["estimate"] if isinstance(value, dict) else value


def seeds(config: dict) -> str:
    runs = {"0": config["metrics"], **config.get("split_seeds", {})}
    keys = (
        "slice_auroc",
        "slice_auprc",
        "onset_detection_rate",
        "comparison_false_alarm_shot_rate",
    )
    rows = [
        [seed, *(plain(estimate(m[k])) for k in keys)]
        for seed, m in sorted(runs.items())
    ]
    return table(
        [
            "fold seed",
            "AUROC",
            "AUPRC",
            "detection rate",
            "comparison false-alarm shot rate",
        ],
        rows,
    )


def by_campaign(config: dict) -> str:
    rows = []
    for year, part in config["by_campaign"].items():
        n, m = part["counts"], part["metrics"]
        rows.append(
            [
                year,
                f"{n['hanson_shots']} / {n['comparison_shots']}",
                str(n["positive_slices"]),
                plain(m["slice_auroc"]),
                plain(m["slice_auprc"]),
                f"{n['onsets_warned']} of {n['target_onsets']}",
                f"{n['hanson_shots_with_a_false_alarm']} of {n['hanson_shots']}",
                f"{n['comparison_shots_with_an_alarm']} of {n['comparison_shots']}",
            ]
        )
    return table(
        [
            "campaign",
            "Hanson / comparison shots",
            "positive slices",
            "AUROC",
            "AUPRC",
            "onsets warned",
            "Hanson shots with a false alarm",
            "comparison shots with an alarm",
        ],
        rows,
    )


def paired(record: dict) -> str:
    keys = (
        ("slice_auroc", "AUROC"),
        ("slice_auprc", "AUPRC"),
        ("onset_detection_rate", "detection rate"),
        ("hanson_false_alarm_shot_rate", "Hanson false-alarm shot rate"),
        ("comparison_false_alarm_shot_rate", "comparison false-alarm shot rate"),
    )
    rows = [
        [name, *(interval(difference[key]) for key, _ in keys)]
        for name, difference in record.get("paired", {}).items()
    ]
    return table(["first - second", *(label for _, label in keys)], rows)


def scored_time(config: dict) -> str:
    rows = []
    for role in ("hanson", "comparison"):
        records = [r for r in config.get("per_shot", []) if r["role"] == role]
        if not records:
            continue
        span = [r["span_ms"] / 1000.0 for r in records]
        alarms = sum(r["alarms"] for r in records)
        rows.append(
            [
                role,
                str(len(records)),
                plain(sum(span) / len(span), 2),
                plain(min(span), 2),
                plain(max(span), 2),
                plain(sum(span), 1),
                str(alarms),
                plain(alarms / sum(span), 2),
            ]
        )
    return table(
        [
            "shots",
            "count",
            "mean scored span (s)",
            "shortest",
            "longest",
            "total (s)",
            "alarms",
            "alarms per scored second",
        ],
        rows,
    )


def single_features(record: dict) -> str:
    rows = [
        [
            name,
            plain(v["auroc"]),
            "higher" if v["higher_means_unstable"] else "lower",
            plain(v["present"], 3),
        ]
        for name, v in record["single_feature_auroc"].items()
    ]
    return table(
        ["feature", "AUROC", "unstable when", "fraction of slices with a value"], rows
    )


def shot_list(config: dict, role: str = "hanson") -> str:
    rows = []
    for r in config.get("per_shot", []):
        if r["role"] != role:
            continue
        warned = ", ".join("-" if w is None else f"{w:.0f}" for w in r["warning_ms"])
        rows.append(
            [
                str(r["shot"]),
                str(r["campaign"]),
                str(r["onsets"]),
                warned or "-",
                str(r["false_alarms"]),
            ]
        )
    return table(
        ["shot", "campaign", "n=1 onsets", "warning per onset (ms; - missed)", "false"],
        rows,
    )


def legacy(configs: dict) -> str:
    c = configs["rwm-brf"]
    m, n = c["metrics"], c["counts"]
    with_onset = [
        r for r in c.get("per_shot", []) if r["role"] == "hanson" and r["onsets"]
    ]
    detected = sum(any(w is not None for w in r["warning_ms"]) for r in with_onset)
    old = LEGACY
    old_shots = f"{old['detected']} of {old['unstable_shots']} unstable shots"
    old_false = f"{old['false_positives']} of {old['stable_shots']} stable shots"
    new_shots = (
        f"{detected} of {len(with_onset)} shots with an n=1 onset "
        f"({n['onsets_warned']} of {n['target_onsets']} onsets)"
    )
    new_false = (
        f"{n['hanson_shots_with_a_false_alarm']} of {n['hanson_shots']} Hanson shots; "
        f"{n['comparison_shots_with_an_alarm']} of {n['comparison_shots']} unlabelled "
        "comparison shots (upper bound)"
    )
    rows = [
        [
            "Legacy: Piccione et al. 2022, NSTX, 28 test shots",
            f"{old['slice_auroc']:.3f}",
            f"{old['slice_tpr']:.3f}",
            f"{old['slice_fpr']:.3f}",
            old_shots,
            old_false,
        ],
        [
            "Tokamak-SI: rwm-brf, DIII-D, shot-grouped CV",
            interval(m["slice_auroc"]),
            interval(m["slice_tpr"]),
            interval(m["slice_fpr"]),
            new_shots,
            new_false,
        ],
    ]
    return table(
        [
            "setting",
            "slice AUROC",
            "slice TPR",
            "slice FPR",
            "unstable shots detected",
            "false alarms",
        ],
        rows,
    )


def growth(g: dict) -> str:
    rows = [
        [
            "largest 20 ms growth rate of N1RMS, -150 to +30 ms from the onset (per s)",
            plain(g["max_growth_per_s_n1"]["q1"], 0),
            plain(g["max_growth_per_s_n1"]["median"], 0),
            plain(g["max_growth_per_s_n1"]["q3"], 0),
        ],
        [
            "e-folding time at that rate (ms)",
            plain(g["efold_ms_n1"]["q1"], 1),
            plain(g["efold_ms_n1"]["median"], 1),
            plain(g["efold_ms_n1"]["q3"], 1),
        ],
        [
            "time of the maximum growth relative to the onset (ms)",
            plain(g["max_growth_at_ms_n1"]["q1"], 1),
            plain(g["max_growth_at_ms_n1"]["median"], 1),
            plain(g["max_growth_at_ms_n1"]["q3"], 1),
        ],
        [
            "beta_N 100 ms before the onset",
            plain(g["betan"]["m100"]["q1"], 2),
            plain(g["betan"]["m100"]["median"], 2),
            plain(g["betan"]["m100"]["q3"], 2),
        ],
        [
            "beta_N at the onset",
            plain(g["betan"]["at_onset"]["q1"], 2),
            plain(g["betan"]["at_onset"]["median"], 2),
            plain(g["betan"]["at_onset"]["q3"], 2),
        ],
        [
            "beta_N 40 ms after the onset",
            plain(g["betan"]["p40"]["q1"], 2),
            plain(g["betan"]["p40"]["median"], 2),
            plain(g["betan"]["p40"]["q3"], 2),
        ],
    ]
    return table(
        ["quantity (48 n=1 onsets)", "first quartile", "median", "third quartile"], rows
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT / "tables.md")
    args = parser.parse_args()
    record = json.loads((OUT / "evaluation.json").read_text())
    shots = json.loads((OUT / "shots.json").read_text())
    g = json.loads((OUT / "growth.json").read_text())
    configs = record["configs"]
    sections = {
        "selection balance (shots.json)": balance(shots),
        "growth (growth.json)": growth(g),
        "slice scores, Hanson shots (evaluation.json)": headline(
            configs,
            ["rwm-brf", "rwm-nnpu", "rule-betan-over-li", "rule-rwm-candidates"],
        ),
        "slice scores with the comparison shots as negatives": mixed(
            configs,
            ["rwm-brf", "rwm-nnpu", "rule-betan-over-li", "rule-rwm-candidates"],
        ),
        "per-shot alarm scores": per_shot(
            configs,
            ["rwm-brf", "rwm-nnpu", "rule-betan-over-li", "rule-rwm-candidates"],
        ),
        "legacy against Tokamak-SI": legacy(configs),
        "rwm-brf feature and training ablations": variants(
            configs,
            [
                "rwm-brf",
                "rwm-brf-hanson-only",
                "rwm-brf-equilibrium-only",
                "rwm-brf-magnetics-only",
                "rwm-brf-no-rotation",
                "rwm-brf-with-locked-mode",
            ],
        ),
        "rwm-brf target variants": variants(
            configs,
            [
                "rwm-brf-horizon-50",
                "rwm-brf",
                "rwm-brf-horizon-200",
                "rwm-brf-all-modes",
            ],
        ),
        "rwm-nnpu variants": variants(
            configs,
            ["rwm-nnpu-prior-x0.5", "rwm-nnpu", "rwm-nnpu-prior-x2"],
        ),
        "rwm-brf split-seed sensitivity": seeds(configs["rwm-brf"]),
        "paired differences on shared shot resamples": paired(record),
        "rwm-brf by campaign": by_campaign(configs["rwm-brf"]),
        "rwm-brf scored time and alarm rate": scored_time(configs["rwm-brf"]),
        "rwm-nnpu scored time and alarm rate": scored_time(configs["rwm-nnpu"]),
        "single-feature AUROC": single_features(record),
        "rwm-brf per Hanson shot": shot_list(configs["rwm-brf"]),
        "rwm-nnpu per Hanson shot": shot_list(configs["rwm-nnpu"]),
    }
    text = "".join(f"<!-- {name} -->\n{body}\n\n" for name, body in sections.items())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
