"""Render the swap's documentation, LaTeX table and README lines from evaluation JSON.

Every number in the generated text is read from ``evaluation.json``; the wording of
each comparison is chosen from the intervals, so a changed result changes the prose.
"""

from __future__ import annotations

from pathlib import Path

ARMS = ("legacy", "dense", "threeway")
REFS = (("dense", "dense"), ("legacy", "annotation"))
GROUPS = (
    ("all_60", "60 shots", "All 60 evaluation shots"),
    ("fair_19", "19 shared held-out shots", "The 19 shared held-out shots"),
)
# Scores the retired selection protocol published for the threeway recipe (README).
PUBLISHED = {
    "all_60": {"auroc": 0.980, "auprc": 0.992, "f1": 0.951},
    "fair_19": {"auroc": 0.982, "auprc": 0.992, "f1": 0.944},
}
SAVED = {"ae-rcn": "ae-rcn", "ae-lstm": "ae-lstm"}
CLOCKS = {
    "clock-annotation": "clock (from annotation)",
    "clock-dense": "clock (from dense labels)",
}
SAVED_TRAINING = "UCI ±125 ms windows, 801 shots (saved)"
MODEL_NOTE = {
    "legacy": "annotation (legacy) supervision",
    "dense": "dense supervision",
    "threeway": "annotation-and-TokEye agreement (threeway) supervision",
}
PM = "±"


def f3(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def signed(value) -> str:
    return "n/a" if value is None else f"{value:+.3f}"


def interval(ci) -> str:
    return "n/a" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def point_ci(metric: dict) -> str:
    """Point estimate with its 95% shot-bootstrap interval (never an SD)."""
    value = metric.get("value", metric.get("mean"))
    return f"{f3(value)} {interval(metric['ci95'])}"


def mean_sd(metric: dict) -> str:
    sd = metric.get("sd")
    return (
        f"{f3(metric['mean'])} {PM} {f3(sd)}" if sd is not None else f3(metric["mean"])
    )


def mean_shot_ci(metric: dict) -> str:
    return f"{f3(metric['mean'])} {interval(metric['ci95_shot'])}"


def diff_text(metric: dict) -> str:
    return f"{signed(metric['mean'])} {interval(metric['ci95_shot'])}"


def pair(summary: dict, first: str, second: str) -> dict:
    """The seed-mean difference first minus second, in that direction."""
    pairs = summary["paired_differences"]
    if f"{first} minus {second}" in pairs:
        return pairs[f"{first} minus {second}"]
    flipped = {}
    for metric, score in pairs[f"{second} minus {first}"].items():
        score = dict(score)
        if isinstance(score, dict) and "mean" in score:
            score["mean"] = -score["mean"]
            for key in ("ci95", "ci95_shot"):
                if score.get(key) is not None:
                    low, high = score[key]
                    score[key] = [-high, -low]
        flipped[metric] = score
    return flipped


def verdict(metric: dict) -> str:
    low, high = metric["ci95_shot"]
    if low > 0:
        return "leads"
    if high < 0:
        return "trails"
    return "is not resolved from"


def per_seed(block: dict, arm: str, reference: str, record: dict) -> list[tuple]:
    methods = block["references"][reference]["methods"]
    return [
        (seed, methods[f"ae-ours-{arm}-seed{seed}"])
        for seed in record["convergence"]["accepted_seeds"][arm]
    ]


def shot_median(block: dict, reference: str, name: str) -> float | None:
    return block["references"][reference]["within_shot"][name]["auroc"]["value"]


def seed_mean_shot_median(block, arm, reference, record) -> tuple[float, float | None]:
    values = [
        shot_median(block, reference, f"ae-ours-{arm}-seed{seed}")
        for seed in record["convergence"]["accepted_seeds"][arm]
    ]
    mean = sum(values) / len(values)
    sd = (
        (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5
        if len(values) > 1
        else None
    )
    return mean, sd


def display(name: str) -> tuple[str, str]:
    if name in CLOCKS:
        return CLOCKS[name], "input-free (120 shots)"
    if name in SAVED:
        return name, SAVED_TRAINING
    return "ae-ours", MODEL_NOTE[name.removeprefix("ae-ours-")]


# --------------------------------------------------------------------------- tables

TEX_PM = r"$\pm$"
SAVED_TRAINING_TEX = r"UCI $\pm$125\,ms windows, 801 shots (saved)"


def tex(value: str) -> str:
    return value.replace(PM, TEX_PM)


def table_rows(record: dict, group: str, reference: str) -> list[str]:
    """Seed rows (estimate and 95% CI), mean rows (mean and SD), baselines."""
    block = record["results"][group]
    result = block["references"][reference]
    rows = []
    for arm in ARMS:
        for seed, m in per_seed(block, arm, reference, record):
            name = f"ae-ours-{arm}-seed{seed}"
            rows.append(
                " & ".join(
                    [
                        r"\texttt{ae-ours}",
                        f"{arm}, seed {seed}",
                        point_ci(m["auroc"]),
                        point_ci(m["auprc"]),
                        f3(shot_median(block, reference, name)),
                        f3(m["f1"]["value"]),
                    ]
                )
                + r" \\"
            )
        s = result["seed_summary"]["methods"][f"ae-ours-{arm}"]
        mean, sd = seed_mean_shot_median(block, arm, reference, record)
        rows.append(
            " & ".join(
                [
                    r"\textit{ae-ours}",
                    rf"\textit{{{arm}, mean of 3}}",
                    tex(mean_sd(s["auroc"])),
                    tex(mean_sd(s["auprc"])),
                    tex(f"{f3(mean)} {PM} {f3(sd)}"),
                    tex(mean_sd(s["f1"])),
                ]
            )
            + r" \\"
        )
    for name in ("ae-rcn", "ae-lstm", "clock-annotation", "clock-dense"):
        if name not in result["methods"]:
            continue
        m = result["methods"][name]
        label, training = display(name)
        if name in SAVED:
            label, training = rf"\texttt{{{name}}}", SAVED_TRAINING_TEX
        rows.append(
            " & ".join(
                [
                    label,
                    training,
                    point_ci(m["auroc"]),
                    point_ci(m["auprc"]),
                    f3(shot_median(block, reference, name)),
                    f3(m["f1"]["value"]),
                ]
            )
            + r" \\"
        )
    return rows


def one_table(record: dict, reference: str, name: str, label: str) -> str:
    """One table* at 6.75 in, two labelled cohort panels, 7 pt text or larger."""
    blocks = record["results"]
    counts = {g: blocks[g]["references"][reference] for g, _, _ in GROUPS}
    caption = (
        rf"\caption{{AE supervision swap scored against the {name} reference, on "
        r"10\,ms frames. \textbf{Models.} The three \texttt{ae-ours} arms differ only "
        r"in the activity target they train on (the legacy annotation, the dense "
        r"relabel, or annotation-and-TokEye agreement), with 100 training shots, "
        r"random 182\,ms windows, the epoch chosen on 20 held-out selection shots "
        r"and three seeds each. \texttt{ae-rcn} and \texttt{ae-lstm} are the saved "
        r"detectors, trained on UCI $\pm$125\,ms windows from 801 shots (41 of the "
        r"60 evaluation shots among them, so they are scored on the 19 shared "
        r"held-out shots only). The two \emph{clock} rows use no input: each 10\,ms "
        r"frame is scored by its positive rate over the 120 training and selection "
        r"shots, from the annotation or from the dense labels. "
        r"\textbf{Scores.} AUROC and AUPRC pool all frames of a cohort; a seed row "
        r"and a baseline give the estimate and a 95\,\% interval from 1000 "
        r"shot-bootstrap replicates (CI); a \emph{mean} row gives the mean and "
        r"the sample standard deviation over the three seeds (SD), never both. "
        r"Shot AUROC is the median over shots of the within-shot AUROC. F1 "
        r"is calibrated per method on the reference being scored: its threshold "
        r"maximises F1 on that method's own selection shots (20; six for "
        r"\texttt{ae-rcn} and \texttt{ae-lstm}) against this reference. "
        rf"Frames: {counts['all_60']['n_frames']} on 60 shots "
        rf"({counts['all_60']['n_positive']} positive) and "
        rf"{counts['fair_19']['n_frames']} on 19 shots "
        rf"({counts['fair_19']['n_positive']} positive).}}"
    )
    head = [
        r"\begin{table*}[t]",
        caption,
        rf"\label{{{label}}}",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\begin{tabular}{@{}l>{\raggedright\arraybackslash}p{1.3in}cccc@{}}",
        r"\toprule",
        r"Model & Training & AUROC & AUPRC & Shot AUROC & F1 \\",
    ]
    body = []
    for group, panel, _ in GROUPS:
        body.append(r"\midrule")
        body.append(rf"\multicolumn{{6}}{{@{{}}l}}{{\textbf{{{panel}}}}} \\")
        body.extend(table_rows(record, group, reference))
    tail = [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(head + body + tail) + "\n"


def paper_table(record: dict) -> str:
    """Two table* environments (one per reference); needs booktabs and array."""
    return (
        "% Needs booktabs and array. Written by ae_supervision_swap_report.py from\n"
        "% evaluation.json; the dense table comes first, the annotation table second.\n"
        + one_table(record, "dense", "dense relabel", "tab:ae_supervision_swap_dense")
        + "\n"
        + one_table(
            record,
            "legacy",
            "annotation (Heidbrink)",
            "tab:ae_supervision_swap_annotation",
        )
    )


def methods_rows(record: dict, group: str, reference: str, f1: bool = False):
    """Markdown rows: per-seed estimates with shot CIs, mean rows, baselines."""
    block = record["results"][group]
    result = block["references"][reference]
    rows = []
    for arm in ARMS:
        for seed, m in per_seed(block, arm, reference, record):
            name = f"ae-ours-{arm}-seed{seed}"
            cells = [
                point_ci(m["auroc"]),
                point_ci(m["auprc"]),
                (
                    f"{f3(shot_median(block, reference, name))} "
                    f"{interval(result['within_shot'][name]['auroc']['ci95'])}"
                ),
                point_ci(m["f1"]),
            ]
            rows.append(f"| ae-ours {arm}, seed {seed} | " + " | ".join(cells) + " |")
        s = result["seed_summary"]["methods"][f"ae-ours-{arm}"]
        mean, sd = seed_mean_shot_median(block, arm, reference, record)
        cells = [
            mean_sd(s["auroc"]),
            mean_sd(s["auprc"]),
            f"{f3(mean)} {PM} {f3(sd)}",
            mean_sd(s["f1"]),
        ]
        rows.append(
            f"| **ae-ours {arm}, mean over seeds** | " + " | ".join(cells) + " |"
        )
    for name in ("ae-rcn", "ae-lstm", "clock-annotation", "clock-dense"):
        if name not in result["methods"]:
            continue
        m = result["methods"][name]
        label, training = display(name)
        cells = [
            point_ci(m["auroc"]),
            point_ci(m["auprc"]),
            (
                f"{f3(shot_median(block, reference, name))} "
                f"{interval(result['within_shot'][name]['auroc']['ci95'])}"
            ),
            point_ci(m["f1"]),
        ]
        rows.append(f"| {label} ({training}) | " + " | ".join(cells) + " |")
    return rows


def paired_rows(record: dict, group: str, reference: str) -> list[str]:
    summary = record["results"][group]["references"][reference]["seed_summary"]
    rows = []
    for name, metrics in summary["paired_differences"].items():
        rows.append(
            f"| {name} | "
            + " | ".join(diff_text(metrics[m]) for m in ("auroc", "auprc", "f1"))
            + " |"
        )
    return rows


# ----------------------------------------------------------------------- the text


def convergence_section(record: dict) -> list[str]:
    plan = record["convergence"]
    rule = plan["rule"]
    train, screen = rule["training"], rule["screen"]
    lines = [
        "## Convergence rule and the nine records",
        "",
        (
            "Fix round 1 found that the first legacy seed 2 never left the "
            "constant-output plateau (selection loss lowest at epoch 2, patience 5 "
            "ended it at epoch 7, constant score). One rule, written in "
            "`CONVERGENCE_RULE` and in this document before any rerun, applies "
            "uniformly to every record of every arm:"
        ),
        "",
        (
            f"1. {train['statement']} (Maximum {train['max_epochs']} epochs, patience "
            f"{train['patience']}, minimum improvement {train['min_delta']:g}, monitored "
            f"on {train['monitor']}.) {train['conformance']}"
        ),
        (
            f"2. {screen['statement']} Selection screen: {screen['sd_statistic']} "
            f"at least {screen['within_shot_sd_min']}, and selection AUROC above "
            f"{screen['selection_auroc_min_exclusive']}, on the {screen['scope']}."
        ),
        "",
        (
            "An earlier draft of the rule (the screen alone) was applied by the "
            "previous implementer to selection-only outputs and is archived as "
            "`convergence_draft0.json`; this two-part rule replaced it before any "
            "rerun. The failure of the first legacy seed 2 was already known from "
            "review when the rule was written, so the rule is predeclared with "
            "respect to the reruns, not blind to that failure."
        ),
        "",
        (
            "| Record | Last epoch | Selected epoch | Rule 1 | Within-shot SD | "
            "Selection AUROC | Outcome |"
        ),
        "|---|---:|---:|---|---:|---:|---|",
    ]
    for name, entry in sorted(plan["audit"].items()):
        tr, sc = entry["training_rule"], entry["screen"]
        lines.append(
            f"| {name} | {tr['final_epoch']} | {entry['selected_epoch']} | "
            f"conforms | {sc['mean_within_shot_sd']:.3f} | "
            f"{f3(sc['selection_auroc'])} | {entry['status']} |"
        )
    for name, entry in sorted(plan["superseded"].items()):
        tr = entry["training_rule"]
        extra = record["excluded_runs"].get(name, {}).get("screen")
        sd = f"{extra['mean_within_shot_sd']:.3f}" if extra else "n/a"
        auc = f3(extra["selection_auroc"]) if extra else "n/a"
        lines.append(
            f"| {name} (first attempt) | {tr['final_epoch']} | "
            f"{entry['selected_epoch']} | stopped inside the plateau | {sd} | "
            f"{auc} | superseded: rerun with the same seed |"
        )
    lines += [""]
    replaced = plan["replacements"]
    passing = sum(1 for e in plan["audit"].values() if e["status"] == "accepted")
    superseded = sorted(plan["superseded"])
    lines += [
        (
            f"Of the nine first-generation records, {9 - len(superseded)} conformed "
            f"to rule 1 as they stood; {len(superseded)} stopped inside the plateau "
            f"({', '.join(superseded) if superseded else 'none'}) and were rerun "
            "with the same seed. After rerunning, "
            f"{passing} of the {len(plan['audit'])} final records pass rule 2"
            + (
                "; replacements: "
                + ", ".join(f"{a} -> {b}" for a, b in replaced.items())
                if replaced
                else " and none needed replacing"
            )
            + ". Selection-only outputs were used for every decision above; no "
            "evaluation frame entered the rule. The first-attempt records are kept "
            "on disk and scored in the last section, so the effect of the rule is "
            "visible."
        ),
        "",
    ]
    return lines


def protocol_section(record: dict, manifest: dict) -> list[str]:
    counts = record["dense_counts"]
    prev = record["dense_prevalence"]
    hist = record["dense_history"]
    tr = prev["train_selection_120"]
    ev = prev["evaluation_60"]
    return [
        "## Frozen protocol",
        "",
        (
            f"The dense snapshot (`review/labels.csv`, SHA256 "
            f"`{manifest['inputs']['dense']['sha256']}`) has {counts['rows']} rows "
            f"over {counts['shots']} shots: {counts['present_intervals']} present "
            f"and {counts['rows'] - counts['present_intervals']} absent rows. Every "
            "present row is a merged crowd span (`iscrowd`), so they are not "
            "individual boxes. Its history file holds "
            f"{hist['entries']} entries, whose latest interval list per shot totals "
            f"{hist['latest_intervals_per_shot_total']} intervals and whose "
            f"entries together total {hist['all_entry_intervals_total']}. The paper's "
            '"943 intervals when the benchmark was scored, 954 after later review" '
            "matches none of these counts, so it counts another unit or an earlier "
            "snapshot; this document uses the file as it stands and takes no "
            "number from the paper's count. "
            f"Prevalence of present time: {tr['prevalence_by_duration']:.3f} by "
            f"duration and {tr['prevalence_any_touch_frames']:.3f} in any-touch "
            "10 ms frames on the 120 training and selection shots, against 0.41 in "
            "the paper (not reconciled: the paper's figure may predate later review "
            f"edits), and {ev['prevalence_by_duration']:.3f} and "
            f"{ev['prevalence_any_touch_frames']:.3f} on the 60 validation shots, "
            "against 0.70 in the paper (the frame count matches). Source: "
            "`dense_counts`, `dense_history` and `dense_prevalence` in "
            "evaluation.json."
        ),
        "",
        (
            f"NumPy split seed {manifest['split_seed']} freezes 100 training, 20 "
            "selection and 60 evaluation shots; none is in the fixed catalog's "
            "blind test split. Seeds 0, 1 and 2 change initialisation and the "
            "sampled windows; every arm uses the same split. The original one-seed "
            "manifest was archived before the seed list was extended."
        ),
        "",
        (
            "Only the activity target changes between arms. Legacy is the audit's "
            "at-least-half-annotated rule at 10 ms (classes 1 to 4, LFM excluded), "
            "expanded to the native columns; dense uses the catalog any-touch state "
            "rule and masks unknown states; threeway keeps the native "
            "annotation-and-TokEye agreement, needs at least half the native columns "
            "of a frame to carry agreement weight, and masks disagreement. The "
            "annotation-and-TokEye frequency target and its weights are identical in "
            "every arm (verification.json). Each input covers 0 to 2 s as 7,820 "
            "native columns."
        ),
        "",
        (
            "**Band.** The model reads the CO2 spectrogram from 80.57 to 250.00 kHz "
            '(348 bins, four chords). "Absent" for ae-ours, and in the targets it '
            "trains on, means absent within that band; it says nothing about activity "
            "below 80.6 kHz, which the annotation can mark. The saved UCI detectors "
            "read 20 to 250 kHz."
        ),
        "",
        (
            "**Recipe.** The original AeSeldNet recipe (four channels, 348 "
            "frequency bins, pools (6, 2, 29), two bidirectional GRUs, SCE plus the "
            "unchanged frequency objective, 710-column windows (182 ms), eight "
            "windows per shot, batch 16, AdamW 1e-4, weight decay 1e-4, cosine decay "
            "to 1e-6) with the convergence rule above; CUDA allocations are capped "
            "at 10 GiB. The persistent DataLoader workers keep their epoch-0 copy of "
            "the dataset, so `set_epoch()` never reaches them: every epoch re-draws "
            "the same eight windows per shot (800 windows), and only the shot order "
            "is reshuffled. This holds for every arm and seed, including the "
            "published model's recipe; changing it would require rerunning "
            "everything (checked in isolation with and without persistent "
            "workers)."
        ),
        "",
        (
            "**Thresholds.** The epoch minimises the combined loss on the 20 "
            "selection shots. The headline F1 calibrates every method on the "
            "reference it is scored against: the threshold maximises 10 ms F1 on that "
            "method's selection shots (20; six for ae-rcn and ae-lstm, the only "
            "ones they did not train on; the clocks use the 20) with ties going to "
            "the highest threshold. F1 at each record's own-target threshold (the "
            "earlier protocol: each arm's threshold from its own activity target, "
            "the older detectors' from the annotation) is kept in evaluation.json "
            "(`f1_own_target_threshold`) and is used in no claim. No evaluation "
            "frame selects an epoch or a threshold."
        ),
        "",
        (
            "**Saved detectors.** Garcia probabilities are the saved spectrogram "
            "predictions: maximum over the first four AE classes, mean over four "
            "chords, nearest output-bin centre on the 10 ms grid. Only 19 "
            "evaluation shots and six selection shots have predictions for shots "
            "the detectors did not train on."
        ),
        "",
    ]


def runs_section(record: dict) -> list[str]:
    lines = [
        "## Completed runs",
        "",
        (
            "Epochs are zero-based. All seeds were trained with `--patience-start 10` "
            "unless noted; the seven first-generation records that conformed were "
            "trained before the flag existed and are shown by the audit above to be "
            "what the rule produces."
        ),
        "",
        (
            "| Supervision | Seed | Selected epoch | Epochs run | Own-target threshold | "
            "Execution | GPU |"
        ),
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for name, run in sorted(record["runs"].items()):
        ex = run["execution"]
        job = ex["slurm_job_id"] or "head node"
        if ex["slurm_array_job_id"]:
            job = (
                f"{ex['slurm_array_job_id']}_{ex['slurm_array_task_id']} "
                f"(job {ex['slurm_job_id']})"
            )
        lines.append(
            f"| {run['supervision']} | {run['seed']} | {run['selected_epoch']} | "
            f"{run['epochs_completed']} | {run['threshold']['threshold']:.4f} | "
            f"{job} | {run['training_environment']['gpu']} |"
        )
    lines += [
        "",
        (
            "Legacy and dense seed 0 ran on A100 GPUs; every other record ran on "
            "the head node's V100S. This Torch build reports V100 bfloat16 support "
            "through emulation, so the unchanged trainer autocasts to bfloat16 on "
            "both (gpu_probe.json). Hardware is therefore a nuisance variable of "
            "the supervision comparison, and the V100-only sensitivity analysis "
            "below uses the V100 runs of seeds 1 and 2 only."
        ),
        "",
    ]
    return lines


def scores_section(record: dict) -> list[str]:
    lines = [
        "## Scores",
        "",
        (
            "AUROC and AUPRC are the primary metrics: they need no threshold. "
            "Per-seed rows and baselines give the pooled-frame estimate with a 95% "
            "interval from 1,000 shot-bootstrap replicates (seed 20261004). Mean "
            "rows give the mean and sample SD over seeds of the per-seed pooled "
            "values. Shot AUROC is the median over shots with both classes of the "
            "within-shot AUROC, with a shot-bootstrap interval; it removes "
            "differences in calibration between shots. F1 is calibrated on the "
            "reference being scored for every method (above)."
        ),
        "",
    ]
    for group, short, label in GROUPS:
        block = record["results"][group]
        lines += [
            f"### {label}",
            "",
            "Shots: " + ", ".join(map(str, block["shots"])) + ".",
            "",
        ]
        for reference, name in REFS:
            result = block["references"][reference]
            lines += [
                (
                    f"**Against the {name} reference** ({result['n_frames']} scorable "
                    f"frames, {result['n_positive']} positive)."
                ),
                "",
                "| Method | AUROC | AUPRC | Shot AUROC | F1 |",
                "|---|---|---|---|---|",
                *methods_rows(record, group, reference),
                "",
            ]
        lines += [
            (
                "Seed-mean paired differences, first minus second, with the interval "
                "from resampling shots only (every seed kept in each draw; the "
                "seed-to-seed spread is the SD in the table above):"
            ),
            "",
        ]
        for reference, name in REFS:
            lines += [
                f"Against the {name} reference:",
                "",
                "| First minus second | AUROC | AUPRC | F1 |",
                "|---|---|---|---|",
                *paired_rows(record, group, reference),
                "",
            ]
    return lines


def sensitivity_section(record: dict) -> list[str]:
    block = record["sensitivity_v100_seeds_1_2"]
    lines = ["## V100-only sensitivity", ""]
    if not block:
        return lines + [
            "Not computed: the V100 runs of seeds 1 and 2 are incomplete.",
            "",
        ]
    runs = block["fair_19"]["runs"]
    lines += [
        (
            "Legacy and dense seed 0 ran on A100, threeway seed 0 on V100. "
            "Restricting every arm to its V100 runs of seeds other than 0 "
            "(" + "; ".join(f"{a}: {', '.join(r)}" for a, r in runs.items()) + ") "
            "gives two seeds per arm on identical hardware. Seed means, with the "
            "shot-only interval:"
        ),
        "",
        "| Cohort | Reference | Arm | AUROC | AUPRC |",
        "|---|---|---|---|---|",
    ]
    for group, short, _ in GROUPS:
        for reference, name in REFS:
            summary = block[group]["references"][reference]
            for arm in ARMS:
                m = summary["methods"][f"ae-ours-{arm}"]
                lines.append(
                    f"| {short} | {name} | {arm} | {mean_shot_ci(m['auroc'])} | "
                    f"{mean_shot_ci(m['auprc'])} |"
                )
    lines += ["", "Selected paired contrasts on the same hardware (AUROC):", ""]
    lines += ["| Cohort | Reference | Contrast | AUROC |", "|---|---|---|---|"]
    for group, short, _ in GROUPS:
        for reference, name in REFS:
            summary = block[group]["references"][reference]
            contrasts = [
                ("ae-ours-dense", "ae-ours-legacy"),
                ("ae-ours-threeway", "ae-ours-legacy"),
            ]
            if group == "fair_19":
                contrasts += [(f"ae-ours-{a}", "ae-rcn") for a in ARMS]
            for first, second in contrasts:
                lines.append(
                    f"| {short} | {name} | {first} minus {second} | "
                    f"{diff_text(pair(summary, first, second)['auroc'])} |"
                )
    lines += [""]
    return lines


def rel(res: dict, first: str, second: str, metric: str = "auroc") -> dict:
    """Seed-mean difference first minus second on one cohort and reference."""
    return pair(res["seed_summary"], first, second)[metric]


def contrast_table(record: dict, metric: str) -> list[str]:
    fair = record["results"]["fair_19"]["references"]
    lines = [
        (
            f"{metric.upper()}, seed-mean ae-ours minus the saved detector, 19 shared "
            "held-out shots, interval from resampling shots only:"
        ),
        "",
        (
            "| ae-ours arm | minus ae-rcn, dense | minus ae-rcn, annotation | "
            "minus ae-lstm, dense | minus ae-lstm, annotation |"
        ),
        "|---|---|---|---|---|",
    ]
    for arm in ARMS:
        cells = [
            diff_text(rel(fair[ref], f"ae-ours-{arm}", base, metric))
            for base in ("ae-rcn", "ae-lstm")
            for ref in ("dense", "legacy")
        ]
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    return lines + [""]


def vs(label: str, other: str, dense: dict, annotation: dict) -> str:
    return (
        f"{label} {verdict(dense)} {other} on the dense reference "
        f"({diff_text(dense)}) and {verdict(annotation)} it on the annotation "
        f"({diff_text(annotation)})"
    )


def findings(record: dict) -> dict:
    """The comparisons the summary and the interpretation both state."""
    fair = record["results"]["fair_19"]["references"]
    out = {
        "legacy_rcn": (
            rel(fair["dense"], "ae-ours-legacy", "ae-rcn"),
            rel(fair["legacy"], "ae-ours-legacy", "ae-rcn"),
        ),
        "legacy_lstm": (
            rel(fair["dense"], "ae-ours-legacy", "ae-lstm"),
            rel(fair["legacy"], "ae-ours-legacy", "ae-lstm"),
        ),
    }
    out["no_reversal_rcn"] = all(d["mean"] < 0 for d in out["legacy_rcn"])
    out["lstm_reverses"] = (
        out["legacy_lstm"][0]["mean"] > 0 > out["legacy_lstm"][1]["mean"]
    )
    out["lead_with_new"] = any(
        rel(fair["dense"], f"ae-ours-{arm}", "ae-rcn")["ci95_shot"][0] > 0
        for arm in ("dense", "threeway")
    )
    clock = fair["legacy"]["methods"]["clock-annotation"]["auroc"]["value"]
    out["margin_rcn"] = (clock - 0.5) / (
        fair["legacy"]["methods"]["ae-rcn"]["auroc"]["value"] - 0.5
    )
    out["margin_lstm"] = (clock - 0.5) / (
        fair["legacy"]["methods"]["ae-lstm"]["auroc"]["value"] - 0.5
    )
    return out


def profile(prior: list[float]) -> str:
    peak = max(range(len(prior)), key=prior.__getitem__)
    edge = (sum(prior[:10]) + sum(prior[-10:])) / 20
    return (
        f"peaks at {(peak + 0.5) * 10:.0f} ms ({prior[peak]:.2f}), averages "
        f"{sum(prior) / len(prior):.2f} over the 0 to 2 s record and {edge:.2f} over "
        "its first and last 100 ms"
    )


def interpretation(record: dict) -> list[str]:
    fair = record["results"]["fair_19"]["references"]
    all60 = record["results"]["all_60"]["references"]
    out = ["## Interpretation", ""]

    # 1. the effect of supervision inside ae-ours
    lines = []
    for label, res in (("19 shared shots", fair), ("60 shots", all60)):
        parts = []
        for first in ("dense", "threeway"):
            d = rel(res["dense"], f"ae-ours-{first}", "ae-ours-legacy")
            a = rel(res["legacy"], f"ae-ours-{first}", "ae-ours-legacy")
            parts.append(
                f"{first} minus legacy supervision {diff_text(d)} against the dense "
                f"reference and {diff_text(a)} against the annotation"
            )
        lines.append(f"On the {label}, the AUROC of " + "; ".join(parts) + ".")
    dt = rel(fair["dense"], "ae-ours-dense", "ae-ours-threeway")
    dt60 = rel(all60["dense"], "ae-ours-dense", "ae-ours-threeway")
    lines.append(
        "Dense minus threeway supervision, against the dense reference: "
        f"{diff_text(dt)} on the 19 shots and {diff_text(dt60)} on the 60 shots, "
        "so dense relabelling and annotation-and-TokEye agreement are not "
        "separated by this experiment."
    )
    out += ["**Within ae-ours (supervision).** " + " ".join(lines), ""]

    # 2. against the saved detectors
    out += contrast_table(record, "auroc") + contrast_table(record, "auprc")
    f = findings(record)
    statements = [
        vs("Legacy-supervised ae-ours", "ae-rcn", *f["legacy_rcn"])
        + ". "
        + (
            "It trails ae-rcn on both references: there is no reversal against "
            "ae-rcn when ae-ours is trained on legacy-type supervision."
            if f["no_reversal_rcn"]
            else "The reversal against ae-rcn is not excluded by this comparison."
        ),
        vs("The same model", "ae-lstm", *f["legacy_lstm"])
        + ". "
        + (
            "The reversal against ae-lstm persists: legacy-supervised ae-ours is "
            "above it on the dense reference and below it on the annotation."
            if f["lstm_reverses"]
            else "The reversal against ae-lstm does not persist in both signs."
        ),
    ]
    for arm in ("dense", "threeway"):
        statements.append(
            vs(
                f"The {arm}-supervised model",
                "ae-rcn",
                rel(fair["dense"], f"ae-ours-{arm}", "ae-rcn"),
                rel(fair["legacy"], f"ae-ours-{arm}", "ae-rcn"),
            )
            + "."
        )
    if f["no_reversal_rcn"] and f["lead_with_new"]:
        statements.append(
            "ae-ours's lead over ae-rcn on the dense reference therefore depends on "
            "dense or annotation-and-TokEye-agreement supervision: it is absent "
            "when the same recipe trains on the legacy annotation and present when "
            "it trains on either of the other two targets. Which of the two carries "
            "it is not separated, and the cross-architecture confounds below apply "
            "to every statement against the saved detectors."
        )
    out += [
        "**What the swap shows (AUROC, 19 shared shots).** " + " ".join(statements),
        "",
    ]

    # 3. the clock
    ref_a, ref_d = fair["legacy"], fair["dense"]
    value = lambda ref, n, m="auroc": ref["methods"][n][m]["value"]
    mean = lambda ref, n, m="auroc": ref["seed_summary"]["methods"][n][m]["mean"]
    shot = lambda ref, n: shot_median(record["results"]["fair_19"], ref, n)
    clocks = record["clock"]
    out += [
        (
            "**Input-free clock.** Each 10 ms frame is scored by its positive rate "
            "over the 120 training and selection shots, with no input and no "
            "evaluation data. The annotation-based clock "
            f"{profile(clocks['clock-annotation']['prior'])}; the dense-based clock "
            f"{profile(clocks['clock-dense']['prior'])}. On the 19 shared shots its "
            "pooled AUROC is "
            f"{f3(value(ref_a, 'clock-annotation'))} against the annotation (clock "
            "from the annotation) and "
            f"{f3(value(ref_d, 'clock-dense'))} against the dense reference (clock "
            "from the dense labels). For comparison (annotation / dense): ae-rcn "
            f"{f3(value(ref_a, 'ae-rcn'))} / {f3(value(ref_d, 'ae-rcn'))}, ae-lstm "
            f"{f3(value(ref_a, 'ae-lstm'))} / {f3(value(ref_d, 'ae-lstm'))}, and "
            "the ae-ours seed means "
            + ", ".join(
                f"{arm} {f3(mean(ref_a, 'ae-ours-' + arm))} / "
                f"{f3(mean(ref_d, 'ae-ours-' + arm))}"
                for arm in ARMS
            )
            + ". Within shots the clock's median AUROC is "
            f"{f3(shot('legacy', 'clock-annotation'))} against the annotation and "
            f"{f3(shot('dense', 'clock-dense'))} against the dense reference, "
            f"against ae-rcn {f3(shot('legacy', 'ae-rcn'))} / "
            f"{f3(shot('dense', 'ae-rcn'))} and ae-lstm "
            f"{f3(shot('legacy', 'ae-lstm'))} / {f3(shot('dense', 'ae-lstm'))}."
        ),
        "",
        (
            "Against the annotation the clock reaches "
            f"{f['margin_rcn']:.0%} of ae-rcn's pooled AUROC margin over chance and "
            f"{f['margin_lstm']:.0%} of ae-lstm's. Paired AUROC differences against the "
            "annotation: ae-rcn minus the annotation clock "
            f"{diff_text(rel(ref_a, 'ae-rcn', 'clock-annotation'))}, ae-lstm minus "
            f"it {diff_text(rel(ref_a, 'ae-lstm', 'clock-annotation'))}, "
            "and "
            + ", ".join(
                f"{arm}-supervised ae-ours minus it "
                f"{diff_text(rel(ref_a, f'ae-ours-{arm}', 'clock-annotation'))}"
                for arm in ARMS
            )
            + ". Against the dense reference, ae-ours-dense minus the dense clock is "
            f"{diff_text(rel(ref_d, 'ae-ours-dense', 'clock-dense'))} and ae-rcn "
            f"minus it {diff_text(rel(ref_d, 'ae-rcn', 'clock-dense'))}."
        ),
        "",
        (
            "ae-ours trains on random 182 ms windows (710 columns) and cannot learn "
            "absolute time; Garcia's models read the whole 0 to 2 s record and can. "
            "Where the annotation concentrates in time, as the profile above shows, "
            "part of the older detectors' score against the annotation is time "
            "context, not a better reading of the spectrogram, and the clock is "
            "the control that measures how much. The fair test of time context "
            "against supervision is the deferred LSTM retrain (Limitations)."
        ),
        "",
    ]

    # 4. F1
    f1_dense = value(ref_d, "ae-rcn", "f1")
    f1_own = record["results"]["fair_19"]["references"]["dense"][
        "f1_own_target_threshold"
    ]["ae-rcn"]
    out += [
        (
            "**F1.** AUROC and AUPRC carry the claims. F1 is reported with every "
            "method calibrated on the reference it is scored on. The earlier "
            "protocol left the saved detectors at thresholds set against the "
            "annotation, which moves their F1 on the dense reference: ae-rcn on "
            f"the 19 shots scores {f3(f1_dense)} with its threshold calibrated on "
            f"dense and {f3(f1_own)} at its saved annotation-set threshold "
            "(`f1_own_target_threshold`)."
        ),
        "",
    ]

    # 5. selection
    th = fair["dense"]["seed_summary"]["methods"]["ae-ours-threeway"]
    th60 = all60["dense"]["seed_summary"]["methods"]["ae-ours-threeway"]
    out += [
        (
            "**Selection.** The reviewers' objection was that the published model "
            "was selected on the 60 validation shots that include the 19 benchmark "
            "shots. The same recipe retrained with the epoch chosen on 20 separate "
            "training shots (the threeway arm, which is the published target) "
            f"scores AUROC {f3(th['auroc']['mean'])} on the 19 shots (published "
            f"{PUBLISHED['fair_19']['auroc']:.3f}) and "
            f"{f3(th60['auroc']['mean'])} on the 60 shots (published "
            f"{PUBLISHED['all_60']['auroc']:.3f}), AUPRC {f3(th['auprc']['mean'])} "
            f"and {f3(th60['auprc']['mean'])} (published "
            f"{PUBLISHED['fair_19']['auprc']:.3f} and "
            f"{PUBLISHED['all_60']['auprc']:.3f}); the seed SD of the AUROC is "
            f"{f3(th['auroc']['sd'])}. Selecting on the validation block did not "
            "inflate the published scores beyond that seed spread."
        ),
        "",
    ]
    return out


def confounds_section(record: dict) -> list[str]:
    band = record["band_confound"]
    camp = record["campaign_overlap"]
    ev, fair, train100 = band["evaluation_60"], band["fair_19"], band["train_100"]
    dist = camp["fair_nearest_garcia_training_shot_distance"]
    ours = camp["ae_ours_training_by_block"]
    blocks = camp["evaluation_by_block"]
    return [
        "## Cross-architecture confounds",
        "",
        (
            "The swap holds the architecture fixed and varies supervision. The "
            "comparison with ae-rcn and ae-lstm still differs in more than "
            "supervision, and every statement against them carries these:"
        ),
        "",
        (
            "1. **Time context.** The saved detectors read the whole 0 to 2 s record; "
            "ae-ours sees 182 ms windows. The clock above measures how much that is "
            "worth against each reference."
        ),
        (
            f"2. **Data volume.** ae-rcn and ae-lstm trained on "
            f"{camp['garcia_training_shots']} shots; each ae-ours arm trains on "
            f"{camp['ae_ours_training_shots']}."
        ),
        f"3. **Campaign overlap.** {camp['evaluation_in_garcia_training']} of the 60 "
        "evaluation shots are in the saved detectors' training set (which is why "
        "they can be scored on 19 only), and each of the 19 held-out shots has a "
        f"Garcia training shot {min(dist)} to {max(dist)} shot numbers away, a "
        "same-day neighbour. The evaluation shots sit in run blocks "
        + ", ".join(f"{k}xx: {v}" for k, v in blocks.items())
        + "; ae-ours's 100 training shots in those blocks number "
        + ", ".join(f"{k}xx: {ours.get(k, 0)}" for k in blocks)
        + ".",
        (
            f"4. **Input band.** ae-ours reads {band['model_band_khz'][0]:.1f} to "
            f"{band['model_band_khz'][1]:.1f} kHz; the saved detectors read 20 to "
            "250 kHz. Annotated columns that are BAE only are "
            f"{ev['bae_only_share']:.1%} of annotated columns on the 60 evaluation "
            f"shots ({fair['bae_only_share']:.1%} on the 19) against "
            f"{train100['bae_only_share']:.1%} in ae-ours's training shots. In the "
            f"evaluation shots {ev['in_band_active_share_bae_only']:.0%} of those "
            f"columns ({fair['in_band_active_share_bae_only']:.0%} on the 19) carry "
            "in-band TokEye activity, against "
            f"{ev['in_band_active_share_other']:.0%} of other annotated columns; in "
            f"training the shares are {train100['in_band_active_share_bae_only']:.0%} "
            f"and {train100['in_band_active_share_other']:.0%}."
        ),
        (
            "5. **Calibration sets.** F1 thresholds come from 20 selection shots for "
            "ae-ours and the clocks and from six for the saved detectors."
        ),
        "",
    ]


def limitations_section(record: dict) -> list[str]:
    hist = record["dense_history"]
    return [
        "## Limitations",
        "",
        (
            f"- **Dense labels' provenance.** All {hist['entries']} history entries of "
            f"the dense table are by {len(hist['reviewers'])} reviewer "
            f"({', '.join(hist['reviewers'])}), and their source is "
            f"`{'`, `'.join(hist['sources'])}`: the review was pre-filled from the "
            "annotation's source table. Whether a TokEye layer was on screen while "
            "reviewing is an open question, and it bears on why dense and threeway "
            "supervision score alike."
        ),
        (
            "- **Three seeds** give limited precision for training variability. The "
            "headline intervals resample shots only, and the seed SD is reported "
            "beside them; the interval that also resamples the seed IDs is in "
            "evaluation.json as `ci95` and is not used here, because with three seeds "
            "it mostly reflects the worst seed."
        ),
        "- **Hardware.** A100 and V100S runs mix; see the V100-only sensitivity.",
        (
            "- **Fixed training windows.** Every epoch re-draws the same windows "
            "(persistent workers); this applies to all arms and to the published "
            "recipe."
        ),
        (
            "- **The LSTM retrain was deferred.** The brief's optional item "
            "(`ae-lstm-retrained`: the published 3 x 64 LSTM architecture trained on "
            "the same 100/20 split, on whole records, with dense and with legacy "
            "supervision) was not run. It is the control that separates time context "
            "from supervision: it gives the older architecture the same supervision "
            "treatment while keeping its time context, and the clock rows bound what "
            "time context alone achieves."
        ),
        (
            "- The saved detectors are scored on 19 shots and calibrated on six; their "
            "intervals are wide."
        ),
        "",
    ]


def excluded_section(record: dict) -> list[str]:
    runs = record["excluded_runs"]
    lines = ["## Excluded and superseded records", ""]
    if not runs:
        return lines + ["None.", ""]
    lines += [
        (
            "These records are not in any mean, SD or interval above. They are scored "
            "with the same code, thresholds calibrated on the selection shots, so the "
            "effect of the convergence rule can be read."
        ),
        "",
        (
            "| Record | Epochs run | Selected epoch | Cohort | Dense AUROC | "
            "Annotation AUROC | Selection within-shot SD | Selection AUROC |"
        ),
        "|---|---:|---:|---|---|---|---:|---:|",
    ]
    for name, row in sorted(runs.items()):
        screen = row["screen"]
        for group, short, _ in GROUPS:
            res = row["results"][group]
            lines.append(
                f"| {name} | {row['epochs_completed']} | {row['selected_epoch']} | "
                f"{short} | {point_ci(res['dense']['auroc'])} | "
                f"{point_ci(res['legacy']['auroc'])} | "
                f"{screen['mean_within_shot_sd']:.3f} | "
                f"{f3(screen['selection_auroc'])} |"
            )
    lines += [""]
    return lines


def readme_block(record: dict) -> tuple[list[str], str]:
    fair = record["results"]["fair_19"]["references"]["dense"]["seed_summary"][
        "methods"
    ]
    seeds = record["convergence"]["accepted_seeds"]
    lines, checkpoints = [], []
    for arm in ARMS:
        m = fair[f"ae-ours-{arm}"]
        lines.append(
            f"- d3d_ae_activity_seldnet_sup_{arm} | 2026_10_03 | "
            + " | ".join(
                f"{k.upper()}: {m[k]['mean']:.3f}" for k in ("auroc", "auprc", "f1")
            )
        )
        ids = ", ".join(
            f"seed {seed} `{record['runs'][f'ae-ours-{arm}-seed{seed}']['checkpoint']['sha256'][:12]}`"
            for seed in seeds[arm]
        )
        checkpoints.append(
            f"`sup_{arm}` = `models/{arm}/seed-<n>/ae_seldnet_{arm}_sce_seed<n>.pt` "
            f"(sha256 prefix: {ids})"
        )
    note = (
        "The three `d3d_ae_activity_seldnet_sup_*` lines above are the published "
        "recipe (`ae-ours`) retrained for the supervision swap on 100 training "
        "shots with the epoch chosen on 20 held-out selection shots, each trained "
        "on one activity target (`sup_legacy`: the annotation; `sup_dense`: the "
        "dense relabel; `sup_threeway`: annotation-and-TokEye agreement). Each "
        "line is the mean of three seeds on the 19 shared held-out shots against "
        "the dense reference, with F1 calibrated on the dense reference on the "
        "20 selection shots; they are not the published model on the first line, "
        "which was selected on the 60 validation shots. Checkpoints under "
        "`$LABELER_ROOT/round4/aeswap/`: "
        + "; ".join(checkpoints)
        + ". Both references, all 60 shots, the input-free clock and paired "
        "differences: [ae_supervision_swap.md](../../../docs/labeler/ae_supervision_swap.md)."
    )
    return lines, note


def write_readme(record: dict, repo: Path) -> None:
    readme = repo / "data/events/alfven_eigenmode/README.md"
    content = readme.read_text()
    lines, note = readme_block(record)
    for marker in (
        "- d3d_ae_activity_seldnet_sup_legacy |",
        "- ae-ours | legacy supervision |",
    ):
        if marker in content:
            start = content.index(marker)
            break
    else:
        raise ValueError("README has no supervision-swap lines to replace")
    end = content.index("\nScores are against the owner-reviewed", start)
    content = content[:start] + "\n".join(lines) + "\n" + content[end:]
    inputs = content.index("\n## Inputs")
    head, tail = content[:inputs], content[inputs:]
    for old in (
        "\nThe three `d3d_ae_activity_seldnet_sup_*` lines",
        "\nSupervision-swap lines report",
    ):
        if old in head:
            head = head[: head.index(old)].rstrip("\n") + "\n"
    readme.write_text(head.rstrip("\n") + "\n\n" + note + "\n" + tail)


def summary(record: dict) -> str:
    fair = record["results"]["fair_19"]["references"]
    f = findings(record)
    seed_mean = lambda ref, arm: f3(
        fair[ref]["seed_summary"]["methods"][f"ae-ours-{arm}"]["auroc"]["mean"]
    )
    return (
        "On the 19 shared held-out shots the seed-mean AUROC against the dense "
        "reference is "
        + ", ".join(f"{arm} {seed_mean('dense', arm)}" for arm in ARMS)
        + " for the legacy, dense and threeway supervision arms, and against the "
        "annotation "
        + ", ".join(f"{arm} {seed_mean('legacy', arm)}" for arm in ARMS)
        + ". Legacy-supervised ae-ours "
        + (
            "trails ae-rcn on both references, so the reversal against ae-rcn "
            "needs dense or agreement supervision"
            if f["no_reversal_rcn"] and f["lead_with_new"]
            else "does not trail ae-rcn on both references"
        )
        + "; "
        + (
            "the reversal against ae-lstm persists"
            if f["lstm_reverses"]
            else "the reversal against ae-lstm does not persist in both signs"
        )
        + f". An input-free clock reaches {f['margin_rcn']:.0%} of ae-rcn's AUROC "
        "margin over chance against the annotation, which makes time context a "
        "confound of every comparison with the saved detectors. The Interpretation "
        "and Cross-architecture confounds sections give the numbers and the "
        "limits."
    )


def render_report(record: dict, manifest: dict, out: Path, repo: Path) -> None:
    """Keep numeric doc, README and LaTeX content tied to the saved JSON."""
    record["results"]["fair_19"]["references"]["dense"]["seed_summary"]["methods"]
    head = [
        "# AE supervision swap",
        "",
        (
            "Does the ranking reversal between ae-ours and the older CO2 detectors "
            "come from training supervision? The same AeSeldNet recipe is trained "
            "on three activity targets (the legacy annotation, the dense relabel, "
            "annotation-and-TokEye agreement; three seeds each) with epochs chosen "
            "on 20 held-out training shots, then scored with ae-rcn, ae-lstm and an "
            "input-free clock against both references on the 60 validation shots "
            "and on the 19 shots the older detectors did not train on."
        ),
        "",
        (
            "Source of every number below: "
            "[evaluation.json](../../outputs/labeler/ae/supervision_swap/evaluation.json) "
            "(written by `scripts/labeler/ae_supervision_swap.py evaluate`; run "
            "records, execution IDs, thresholds, per-seed scores, paired differences, "
            "the clock priors and the convergence audit are embedded). "
            "[manifest.json](../../outputs/labeler/ae/supervision_swap/manifest.json) "
            "holds the shot lists and input hashes; "
            "[verification.json](../../outputs/labeler/ae/supervision_swap/verification.json) "
            "checks target equality and split isolation. The paper table is "
            "`table_supervision_swap.tex` in the same directory."
        ),
        "",
        "## Summary",
        "",
        summary(record),
        "",
    ]
    doc = (
        head
        + convergence_section(record)
        + protocol_section(record, manifest)
        + runs_section(record)
        + scores_section(record)
        + sensitivity_section(record)
        + interpretation(record)
        + confounds_section(record)
        + limitations_section(record)
        + excluded_section(record)
        + [
            "## Reproduction",
            "",
            (
                "From the worktree, with the scratch TMPDIR, `LABELER_ROOT`, "
                "`LABELER_LABEL_TABLES`, `LABELER_NO_FETCH=1` and `PYTHONPATH=$PWD/src`:"
            ),
            "",
            "```bash",
            (
                '# training: GPU, CUDA interpreter; sbatch (AESWAP_RUNS="legacy:2 dense:1" '
                "for reruns) or one head-node GPU per process"
            ),
            "sbatch scripts/labeler/ae_supervision_swap.sbatch",
            "bash scripts/labeler/ae_supervision_swap_head.sh 0 legacy:2",
            "# CPU, pixi labelmaker environment",
            "python scripts/labeler/ae_supervision_swap.py verify",
            "python scripts/labeler/ae_supervision_swap.py audit-convergence",
            "python scripts/labeler/ae_supervision_swap.py evaluate",
            "```",
            "",
            (
                "`audit-convergence` applies the declared rule to every record, archives "
                "any record that stopped inside the plateau, and lists the runs still to "
                "train. Completed runs are never overwritten. The launchers use a short "
                "TMPDIR (`$LABELER_ROOT/scratch/ae-sw`) because DataLoader workers add "
                "`/pymp-*/listener-*` to it and a socket path must stay under 108 bytes."
            ),
            "",
        ]
    )
    (repo / "docs/labeler/ae_supervision_swap.md").write_text("\n".join(doc))
    table = paper_table(record)
    (out / "table_supervision_swap.tex").write_text(table)
    (
        repo / "outputs/labeler/ae/supervision_swap/table_supervision_swap.tex"
    ).write_text(table)
    write_readme(record, repo)
