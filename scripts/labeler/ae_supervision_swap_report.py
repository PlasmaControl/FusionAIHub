"""Render the swap's documentation, LaTeX table and README lines from evaluation JSON.

Every number in the generated text is read from ``evaluation.json``; the wording of
each comparison is chosen from the intervals, so a changed result changes the prose.
"""

from __future__ import annotations

import re
from pathlib import Path

ARMS = ("legacy", "dense", "threeway")
# One name for the Heidbrink hand annotation everywhere: "legacy annotation".
REFS = (("dense", "dense"), ("legacy", "legacy annotation"))
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
    "clock-annotation": "clock from legacy annotation",
    "clock-dense": "clock from dense relabel",
}
CLOCK_TRAINING = "input-free, 120 shots"
SAVED_TRAINING = "saved predictions, UCI ±125 ms windows, 801 shots"
MODEL_NOTE = {
    "legacy": "legacy annotation",
    "dense": "dense relabel",
    "threeway": "annotation-and-TokEye agreement",
}
#: The target each arm trains on, as the prose names it.
ARM_TARGET = {
    "legacy": "the legacy annotation",
    "dense": "the dense relabel",
    "threeway": "annotation-and-TokEye agreement",
}
#: ae-ours trained on each target, as the prose names the model.
ARM_MODEL = {arm: f"ae-ours trained on {target}" for arm, target in ARM_TARGET.items()}
DAGGER = "†"
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
        return CLOCKS[name], CLOCK_TRAINING
    if name in SAVED:
        return name, SAVED_TRAINING
    return "ae-ours", MODEL_NOTE[name.removeprefix("ae-ours-")]


def pretty(name: str) -> str:
    """A method id as it reads in a row label (the legacy annotation, in full)."""
    return name.replace("clock-annotation", "clock-legacy-annotation")


# --------------------------------------------------------------------------- tables

TEX_PM = r"$\pm$"
SAVED_TRAINING_TEX = r"UCI $\pm$125\,ms windows, 801 shots (saved)"


def tex_minus(text: str) -> str:
    """A hyphen that is a minus sign before a number becomes a TeX minus sign."""
    return re.sub(r"(?<![\w$])-(?=\d)", "$-$", text)


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
                        f"{MODEL_NOTE[arm]}, seed {seed}",
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
                    rf"\textit{{{MODEL_NOTE[arm]}, mean of 3}}",
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
        f1 = f3(m["f1"]["value"]) + (r"$^\dagger$" if name in CLOCKS else "")
        rows.append(
            " & ".join(
                [
                    label,
                    training,
                    point_ci(m["auroc"]),
                    point_ci(m["auprc"]),
                    f3(shot_median(block, reference, name)),
                    f1,
                ]
            )
            + r" \\"
        )
    return rows


def clock_shot_auroc(record: dict) -> float:
    """Median within-shot AUROC of the dense clock against the dense reference."""
    return shot_median(record["results"]["all_60"], "dense", "clock-dense")


def one_table(record: dict, reference: str, name: str, label: str) -> str:
    """One table* at 6.75 in, two labelled cohort panels, 7 pt text or larger."""
    blocks = record["results"]
    counts = {g: blocks[g]["references"][reference] for g, _, _ in GROUPS}
    coarse = record["dense_reference_coarseness"]
    caveat = (
        (
            r"\textbf{Dense reference.} "
            rf"{coarse['single_present_span_in_table']} of {coarse['shots']} "
            r"shots have a single present span, so the input-free dense clock "
            rf"reaches a median Shot AUROC of {clock_shot_auroc(record):.4f} on "
            r"the 60 shots and that column cannot rank methods here. "
        )
        if reference == "dense"
        else ""
    )
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
        r"shots, from the legacy annotation or from the dense relabel. "
        r"\textbf{Scores.} AUROC and AUPRC pool all frames of a cohort; a seed row "
        r"and a baseline give the estimate and a 95\,\% interval from 1000 "
        r"shot-bootstrap replicates (CI); a \emph{mean} row gives the mean and "
        r"the sample standard deviation over the three seeds (SD), never both. "
        r"Shot AUROC is the median over shots of the within-shot AUROC. F1 "
        r"is calibrated per method on the reference being scored: its threshold "
        r"maximises F1 on that method's own selection shots (20; six for "
        r"\texttt{ae-rcn} and \texttt{ae-lstm}) against this reference. "
        r"$^\dagger$\,The clocks' thresholds are tuned on 20 shots that are part of "
        r"the 120 shots the clock itself is built from (in-sample). "
        + caveat
        + rf"Frames: {counts['all_60']['n_frames']} on 60 shots "
        rf"({counts['all_60']['n_positive']} positive) and "
        rf"{counts['fair_19']['n_frames']} on 19 shots "
        rf"({counts['fair_19']['n_positive']} positive).}}"
    )
    head = [
        r"\begin{table*}[t]",
        caption,
        rf"\label{{{label}}}",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{4.2pt}",
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
    return tex_minus(
        "% Needs booktabs and array. Written by ae_supervision_swap_report.py from\n"
        "% evaluation.json; the dense table comes first, the annotation table second.\n"
        + one_table(record, "dense", "dense relabel", "tab:ae_supervision_swap_dense")
        + "\n"
        + one_table(
            record,
            "legacy",
            "legacy annotation",
            "tab:ae_supervision_swap_annotation",
        )
    )


MAIN_ROWS = (
    ("ae-ours", "legacy annotation", "ae-ours-legacy"),
    ("ae-ours", "dense relabel", "ae-ours-dense"),
    ("ae-ours", "annotation-TokEye agreement", "ae-ours-threeway"),
    ("ae-rcn", "saved, 801 shots", "ae-rcn"),
    ("ae-lstm", "saved, 801 shots", "ae-lstm"),
    ("clock", "from the legacy annotation", "clock-annotation"),
    ("clock", "from the dense relabel", "clock-dense"),
)
#: The training labels as the narrow main table names them.
MAIN_TEX_TRAINING = {
    "ae-ours-legacy": "legacy annotation",
    "ae-ours-dense": "dense relabel",
    "ae-ours-threeway": "agreement",
    "ae-rcn": "saved",
    "ae-lstm": "saved",
    "clock-annotation": "legacy annotation",
    "clock-dense": "dense relabel",
}
MAIN_HEAD = (
    "Model",
    "Training labels",
    "Dense AUROC",
    "Dense AUPRC",
    "Legacy-annotation AUROC",
    "Legacy-annotation AUPRC",
    "AUROC minus ae-rcn, dense",
    "AUROC minus ae-rcn, legacy annotation",
)


def main_cells(record: dict) -> list[tuple[str, str, list[str]]]:
    """The main table's rows: both references side by side on the 19 shared shots.

    ae-ours rows give the seed mean and the sample SD; the saved detectors and the
    clocks are single models, so they give the estimate. The last two cells are
    the paired AUROC difference from ae-rcn with its shot-bootstrap interval.
    """
    fair = record["results"]["fair_19"]["references"]
    dense, legacy = fair["dense"], fair["legacy"]
    rows = []
    for model, training, name in MAIN_ROWS:
        cells = []
        for res in (dense, legacy):
            for metric in ("auroc", "auprc"):
                if name.startswith("ae-ours-"):
                    cells.append(mean_sd(res["seed_summary"]["methods"][name][metric]))
                else:
                    cells.append(f3(res["methods"][name][metric]["value"]))
        for res in (dense, legacy):
            cells.append(
                "n/a" if name == "ae-rcn" else diff_text(rel(res, name, "ae-rcn"))
            )
        rows.append((model, training, cells))
    return rows


def main_markdown(record: dict) -> list[str]:
    lines = [
        "| " + " | ".join(MAIN_HEAD) + " |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for model, training, cells in main_cells(record):
        lines.append("| " + " | ".join([model, training, *cells]) + " |")
    return lines


def main_table(record: dict) -> str:
    """The main-text table (``table*``); the long tables stay in the appendix."""
    rows = []
    for (model, _, name), (_, _, cells) in zip(MAIN_ROWS, main_cells(record)):
        label = rf"\texttt{{{model}}}" if model.startswith("ae-") else model
        short = MAIN_TEX_TRAINING[name]
        rows.append(
            " & ".join([f"{label}, {short}", *(tex(c) for c in cells)]) + r" \\"
        )
    coarse = record["dense_reference_coarseness"]
    caption = (
        r"\caption{AE supervision swap on the 19 held-out shots that ae-rcn and "
        r"ae-lstm did not train on, scored against two references (10\,ms frames). "
        r"The three \texttt{ae-ours} rows are one recipe trained on three activity "
        r"targets (the legacy annotation, the dense relabel, or the agreement of the "
        r"annotation with TokEye; 100 training shots, epoch chosen on 20 other "
        r"shots) and give the mean $\pm$ sample standard deviation over three "
        r"seeds. The saved detectors \texttt{ae-rcn} and \texttt{ae-lstm} (801 "
        r"training shots) and the two input-free "
        r"\emph{clocks} (each 10\,ms frame scored by its positive rate over the 120 "
        r"training and selection shots, from the labels named) are single models "
        r"and give the estimate. "
        r"The last two columns are the paired AUROC difference from "
        r"\texttt{ae-rcn} (row minus \texttt{ae-rcn}; seed mean for "
        r"\texttt{ae-ours}) with a 95\,\% interval from 1000 resamplings of the 19 "
        r"shots. The dense reference is temporally coarse "
        rf"({coarse['single_present_span_in_table']} of {coarse['shots']} shots "
        r"have one present span), so the clock rows show how much of each score "
        r"is time context alone. All 60 evaluation shots and the F1 and per-seed "
        r"results are in the appendix tables.}"
    )
    lines = [
        r"\begin{table*}[t]",
        caption,
        r"\label{tab:ae_supervision_swap_main}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}l cc cc cc@{}}",
        r"\toprule",
        (
            r" & \multicolumn{2}{c}{Dense reference} & "
            r"\multicolumn{2}{c}{Legacy annotation} & "
            r"\multicolumn{2}{c}{AUROC minus \texttt{ae-rcn}} \\"
        ),
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}",
        (
            r"Model, training labels & AUROC & AUPRC & AUROC & AUPRC & "
            r"Dense ref. & Legacy-ann. ref. \\"
        ),
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return tex_minus(
        "% Needs booktabs. Written by ae_supervision_swap_report.py from\n"
        "% evaluation.json: the main-text table (19 shared held-out shots).\n"
        + "\n".join(lines)
        + "\n"
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
            rows.append(
                f"| ae-ours, {MODEL_NOTE[arm]}, seed {seed} | "
                + " | ".join(cells)
                + " |"
            )
        s = result["seed_summary"]["methods"][f"ae-ours-{arm}"]
        mean, sd = seed_mean_shot_median(block, arm, reference, record)
        cells = [
            mean_sd(s["auroc"]),
            mean_sd(s["auprc"]),
            f"{f3(mean)} {PM} {f3(sd)}",
            mean_sd(s["f1"]),
        ]
        rows.append(
            f"| **ae-ours, {MODEL_NOTE[arm]}, mean over seeds** | "
            + " | ".join(cells)
            + " |"
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
            point_ci(m["f1"]) + (DAGGER if name in CLOCKS else ""),
        ]
        rows.append(f"| {label} ({training}) | " + " | ".join(cells) + " |")
    return rows


def paired_rows(record: dict, group: str, reference: str) -> list[str]:
    summary = record["results"][group]["references"][reference]["seed_summary"]
    rows = []
    for name, metrics in sorted(summary["paired_differences"].items()):
        rows.append(
            f"| {pretty(name)} | "
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
            f"{entry['selected_epoch']} | stopped before the rule allowed | {sd} | "
            f"{auc} | superseded: rerun with the same seed |"
        )
    lines += [""]
    replaced = plan["replacements"]
    passing = sum(1 for e in plan["audit"].values() if e["status"] == "accepted")
    superseded = sorted(plan["superseded"])
    lines += [
        (
            f"Of the nine first-generation records, {9 - len(superseded)} conformed "
            f"to rule 1 as they stood; {len(superseded)} stopped before the rule allowed "
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
        float32_screen_note(record),
        "",
    ]
    return lines


def float32_screen_note(record: dict) -> str:
    """Whether the accepted records still pass rule 2 on the float32 scores."""
    screen = record["convergence"]["rule"]["screen"]
    shots = [run["fp32_screen"] for run in record["runs"].values()]
    passing = [
        sc["mean_within_shot_sd"] >= screen["within_shot_sd_min"]
        and sc["selection_auroc"] > screen["selection_auroc_min_exclusive"]
        for sc in shots
    ]
    sds = [sc["mean_within_shot_sd"] for sc in shots]
    aucs = [sc["selection_auroc"] for sc in shots]
    failing = len(passing) - sum(passing)
    return (
        "The table above and the decisions it records used the bfloat16 outputs; "
        "every score below is a float32 re-inference of the same checkpoints. "
        f"Re-screened on the float32 selection scores, {sum(passing)} of "
        f"{len(passing)} accepted records pass rule 2 (within-shot SD "
        f"{min(sds):.3f} to {max(sds):.3f}, selection AUROC {min(aucs):.3f} to "
        f"{max(aucs):.3f}); no record changes status."
        if all(passing)
        else "Re-screened on the float32 selection scores, "
        + (
            f"{failing} accepted record{'' if failing == 1 else 's'} "
            f"{'fails' if failing == 1 else 'fail'} rule 2 "
        )
        + "(`fp32_screen` in evaluation.json)."
    )


def clock_note(record: dict) -> str:
    return (
        "The clocks' thresholds are tuned on the 20 selection shots, which are part "
        "of the 120 shots the clock's own positive rates come from, so clock F1 is "
        "in-sample; their AUROC and AUPRC need no threshold."
    )


def n_shots(n: int, kind: str = "evaluation") -> str:
    return f"{n} {kind} shot{'' if n == 1 else 's'}"


def garcia_text(record: dict) -> str:
    """Who saved changes on the shots that differ from the paper's snapshot."""
    rec = record["dense_reconciliation"]
    names = rec["differing_shots_by_name"]
    n_train = len(rec["difference"]["train_selection_120"]["differing_shots"])
    zero = {"train": 0, "selection": 0, "evaluation": 0}
    garcia = names.get("Alvin Garcia", zero)
    chen = names.get("Nathaniel Chen", zero)
    unnamed = names.get("(unnamed)", zero)
    return (
        f"A. Garcia saved changes on {garcia['train'] + garcia['selection']} of the "
        f"{n_train} training and selection shots that differ from the paper's "
        f"snapshot ({garcia['train']} training, {garcia['selection']} selection) "
        f"and on {n_shots(garcia['evaluation'])}; N. Chen on "
        f"{chen['train'] + chen['selection']} of them and "
        f"{n_shots(chen['evaluation'])}; unnamed saves on "
        f"{unnamed['train'] + unnamed['selection']} of them and "
        f"{n_shots(unnamed['evaluation'])} (a shot can carry several)"
    )


def utc(stamp: str) -> str:
    """An ISO timestamp as 'YYYY-MM-DD HH:MM:SS UTC' (the stored times are UTC)."""
    return stamp[:19].replace("T", " ") + " UTC"


def band_source(record: dict) -> str:
    """Where the review-display date comes from, and what supports it."""
    band = record["dense_reconciliation"]["display_band_change"]
    text = band["source"]
    parts = []
    if band.get("file_modified_utc"):
        parts.append(
            "the main checkout's copy of that file was last modified "
            f"{utc(band['file_modified_utc'])}, which fits the date"
        )
    if band.get("commit"):
        garcia = record.get("dense_history", {}).get("by_name", {}).get("Alvin Garcia")
        late = (
            garcia is not None
            and garcia["last_change"] is not None
            and band["commit_utc"] > garcia["last_change"]
        )
        parts.append(
            f"the change was committed in {band['commit']} on {utc(band['commit_utc'])}"
            + (", after A. Garcia's saves" if late else "")
        )
    if parts:
        text += (
            "; " + ", and ".join(parts) + "; what the browser showed was not checked"
        )
    return text


def band_text(record: dict) -> str:
    rec = record["dense_reconciliation"]
    shots = rec["differing_shots"]
    ts = [s for s, v in shots.items() if v["group"] != "evaluation"]
    late = sum(shots[s]["changed_with_60_250_khz_display"] for s in ts)
    early = sum(shots[s]["changed_with_80_250_khz_display"] for s in ts)
    band = rec["display_band_change"]
    return (
        f"**Band.** The model reads the CO2 spectrogram from 80.57 to 250.00 kHz "
        f"(348 bins, four chords). The review display the dense labels were drawn on "
        f"read {band['before']} until {band['date']} and {band['after']} afterwards "
        f"({band_source(record)}). The {len(ts)} training and selection shots that "
        f"differ from the paper's snapshot were changed with the {band['after']} "
        f"display ({late} of {len(ts)}; {early} of them also "
        f"{'carries' if early == 1 else 'carry'} an earlier change made "
        f"with the {band['before']} display), so present frames added to them can "
        "rest on activity between 60 and 80.6 kHz that ae-ours cannot see. "
        "Among the reviewers, "
        + garcia_text(record)
        + ". A. Garcia is the author of ae-rcn and ae-lstm (Limitations). The "
        "arms train on the current table, edits included. "
        '"Absent" in the dense labels, and in every target ae-ours trains on, means '
        "absent within the observed band; it says nothing about activity below "
        "80.6 kHz, which the legacy annotation can mark and which can therefore be "
        "labelled absent. The saved UCI detectors read 20 to 250 kHz."
    )


def counts_text(record: dict, manifest: dict) -> str:
    counts = record["dense_counts"]
    rec = record["dense_reconciliation"]
    snap = rec["paper_snapshot"]
    prev = rec["any_touch_prevalence"]
    diff = rec["difference"]
    ts, ev = diff["train_selection_120"], diff["evaluation_60"]
    ev_shots = ", ".join(map(str, ev["differing_shots"]))
    return (
        f"**Dense label counts.** The paper's 943 intervals are the {snap['rows']} "
        f"rows of the table its audit scored, the ae_xpower v2 review table "
        f"(`{snap['path']}`, SHA256 `{snap['sha256']}`): {snap['present_rows']} "
        f"present and {snap['absent_rows']} absent rows over {snap['shots']} shots. "
        "Its any-touch prevalence (present among present and absent 10 ms frames) "
        f"is {prev['train_selection_120']['paper_snapshot']:.3f} on the 120 "
        f"training and selection shots (the paper's 0.41) and "
        f"{prev['evaluation_60']['paper_snapshot']:.3f} on the 60 evaluation shots "
        "(0.70). The current table "
        f"(`review/labels.csv`, SHA256 `{manifest['inputs']['dense']['sha256']}`) "
        f"has {counts['rows']} rows over {counts['shots']} shots, "
        f"{counts['present_intervals']} present and "
        f"{counts['rows'] - counts['present_intervals']} absent; every present row "
        "is a merged crowd span (`iscrowd`), so rows are not individual boxes. It "
        f"differs from the snapshot on {len(ts['differing_shots'])} of the 120 "
        f"training and selection shots ({ts['frames']} frames, a net "
        f"{ts['net_present']:+d} present frames; prevalence "
        f"{prev['train_selection_120']['current']:.3f}) and on "
        f"{len(ev['differing_shots'])} of the 60 evaluation shots "
        f"({ev['frames']} frames, a net {ev['net_present']:+d}; shot {ev_shots}), "
        f"and on {len(diff['fair_19']['differing_shots'])} of the 19 shared "
        "held-out shots. The evaluation reference therefore matches the paper's "
        "audit; the training labels are a later version of it. The paper's second "
        'count, 954 "after later review", matches neither table. Source: '
        "`dense_counts`, `dense_reconciliation` and `dense_history` in "
        "evaluation.json."
    )


def protocol_section(record: dict, manifest: dict) -> list[str]:
    grid = record["frame_grid"]
    return [
        "## Frozen protocol",
        "",
        counts_text(record, manifest),
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
            "Only the activity target changes between arms. The legacy arm trains on "
            "the legacy annotation (the Heidbrink hand annotation) by the audit's "
            "at-least-half-annotated rule at 10 ms (classes 1 to 4, LFM excluded), "
            "expanded to the native columns; the dense arm uses the catalog "
            "any-touch state rule and masks unknown states; threeway keeps the "
            "native annotation-and-TokEye agreement, needs at least half the native "
            "columns of a frame to carry agreement weight, and masks disagreement. "
            "The annotation-and-TokEye frequency target and its weights are "
            "identical in every arm (verification.json). Each input covers 0 to 2 s "
            "as 7,820 native columns."
        ),
        "",
        band_text(record),
        "",
        (
            f"**Frame grid.** The audit's 10 ms frames spread the "
            f"{grid['columns']:,} native columns evenly over 0 to 2 s "
            "(`frame_index`), and every arm, target and reference here inherits "
            f"that assignment. The recorded column times run from "
            f"{grid['first_ms']:.2f} to {grid['last_ms']:.2f} ms, so "
            f"{grid['columns_in_another_frame']} of the {grid['columns']:,} columns "
            "fall in a neighbouring frame, at most "
            f"{grid['max_abs_offset_ms']:.2f} ms from the uniform centre "
            f"(identical in all {grid['shots_checked']} shots)."
        ),
        "",
        (
            "**Recipe.** The original AeSeldNet recipe (four channels, 348 "
            "frequency bins, pools (6, 2, 29), two bidirectional GRUs, SCE plus the "
            "unchanged frequency objective, 710-column windows (182 ms), eight "
            "windows per shot, batch 16, AdamW 1e-4, weight decay 1e-4, cosine decay "
            "to 1e-6) with the convergence rule above; CUDA allocations are capped "
            "at 10 GiB. The recipe was developed for the threeway target: symmetric "
            "cross entropy was chosen over binary cross entropy on the 60 validation "
            "shots (the model README), and its weights, the learning rate and early "
            "stopping were set for that target; the legacy and dense arms reuse it "
            "unchanged. The persistent DataLoader workers keep their epoch-0 copy of "
            "the dataset, so `set_epoch()` never reaches them: every epoch re-draws "
            "the same eight windows per shot (800 windows), and only the shot order "
            "is reshuffled. This holds for every arm and seed, including the "
            "published model's recipe, and it is not fixed in this round: changing "
            "it would require rerunning everything (checked in isolation with and "
            "without persistent workers)."
        ),
        "",
        (
            "**Thresholds.** The epoch minimises the combined loss on the 20 "
            "selection shots. The headline F1 calibrates every method on the "
            "reference it is scored against: the threshold maximises 10 ms F1 on that "
            "method's selection shots (20; six for ae-rcn and ae-lstm, the only "
            "ones they did not train on; the clocks use the 20) with ties going to "
            "the highest threshold. "
            + clock_note(record)
            + " F1 at each record's own-target threshold (the earlier protocol: "
            "each arm's threshold from its own activity target, the older "
            "detectors' from the legacy annotation) is kept in evaluation.json "
            "(`f1_own_target_threshold`) and is used in no claim. No evaluation "
            "frame selects an epoch or a threshold."
        ),
        "",
        (
            "**Precision.** The training run scored each checkpoint under bfloat16 "
            "autocast. Every score in this document is a float32 re-inference "
            "(autocast off) of the same checkpoints, with the thresholds "
            "recalibrated on the float32 selection scores; the next section gives "
            "the change."
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
            "Epochs are zero-based. The patience column is the epoch from which "
            "non-improving epochs count: records trained after the rule was declared "
            "use 10, earlier records counted from 0 and the audit above shows that "
            "the rule gives them the same stopping epoch."
        ),
        "",
        (
            "| Supervision | Seed | Selected epoch | Epochs run | Patience from | "
            "Own-target threshold (float32) | Execution | GPU |"
        ),
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    on_a100 = []
    for name, run in sorted(record["runs"].items()):
        ex = run["execution"]
        job = ex["slurm_job_id"] or "head node"
        if ex["slurm_array_job_id"]:
            job = (
                f"{ex['slurm_array_job_id']}_{ex['slurm_array_task_id']} "
                f"(job {ex['slurm_job_id']})"
            )
        gpu = run["training_environment"]["gpu"]
        if "A100" in gpu:
            on_a100.append(name)
        start = run.get("training_rule", {}).get("patience_start_epoch", 0)
        lines.append(
            f"| {run['supervision']} | {run['seed']} | {run['selected_epoch']} | "
            f"{run['epochs_completed']} | {start} | "
            f"{run['fp32']['threshold']['threshold']:.4f} | {job} | {gpu} |"
        )
    lines += [
        "",
        (
            f"{', '.join(on_a100) or 'No record'} ran on A100 GPUs; every other "
            "record ran on the head node's V100S. This Torch build reports V100 "
            "bfloat16 support through emulation, so the unchanged trainer autocasts "
            "to bfloat16 on both (gpu_probe.json) when it scores the checkpoint; "
            "every score in this document is instead a float32 re-inference of the "
            "same checkpoints (next section). Hardware is therefore a nuisance "
            "variable of the supervision comparison; the V100-only sensitivity "
            "analysis below restricts every arm to its V100 runs of seeds other "
            "than 0."
        ),
        "",
    ]
    return lines


def where_text(where: str) -> str:
    """'run, cohort key, reference key reference' as it reads in prose."""
    run, group, reference = (part.strip() for part in where.split(","))
    cohort = {key: short for key, short, _ in GROUPS}[group]
    names = dict(REFS)
    return f"{run}, {cohort}, {names[reference.removesuffix(' reference')]} reference"


def precision_section(record: dict) -> list[str]:
    """How much the float32 re-inference moved the scores."""
    check = record["precision_check"]
    worst = check["max_abs_change_per_run"]
    frames = {
        name: run["fp32"]["frame_probability_difference"]
        for name, run in sorted(record["runs"].items())
    }
    lines = [
        "## Inference precision",
        "",
        (
            "The training run scored each checkpoint under bfloat16 autocast. This "
            "Torch build emulates bfloat16 on the V100, and the frame probabilities "
            "it gives differ visibly from float32 ones. Every checkpoint was "
            "therefore re-inferred in float32 (autocast off; `infer-fp32`, files "
            "`probabilities_fp32.npz` and `fp32.json` beside each run), the "
            "thresholds were recalibrated on the float32 selection scores, and all "
            "scores, thresholds and intervals in this document are float32. The "
            "bfloat16 files are untouched. Change in the pooled-frame score, float32 "
            "minus bfloat16 (`precision_check` in evaluation.json): the largest "
            "change of any run in any cohort and reference is "
            f"{worst['auroc']['value']:.4f} "
            f"AUROC ({where_text(worst['auroc']['where'])}) and "
            f"{worst['auprc']['value']:.4f} "
            f"AUPRC ({where_text(worst['auprc']['where'])}). Seed-mean change per arm "
            "(AUROC / AUPRC):"
        ),
        "",
        "| Arm | "
        + " | ".join(f"{short}, {name}" for _, short, _ in GROUPS for _, name in REFS)
        + " |",
        "|---|" + "---|" * (len(GROUPS) * len(REFS)),
    ]
    for arm in ARMS:
        cells = []
        for group, _, _ in GROUPS:
            for reference, _ in REFS:
                d_auroc, d_auprc = check["seed_mean_change"][arm][
                    f"{group}/{reference}"
                ]
                cells.append(f"{d_auroc:+.4f} / {d_auprc:+.4f}")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    over = [f["evaluation_shots_over_0.1"] for f in frames.values()]
    biggest = max(frames.values(), key=lambda f: f["evaluation_max"])
    lines += [
        "",
        (
            "Frame by frame the two precisions disagree more than the pooled scores "
            f"do: on the 60 evaluation shots the largest single-frame difference "
            f"between the two probabilities of a shot exceeds 0.1 on {min(over)} to "
            f"{max(over)} shots per record (largest overall "
            f"{biggest['evaluation_max']:.2f}). Pooled over a cohort, the largest "
            f"change of a run's AUROC is {worst['auroc']['value']:.4f}; the "
            "bfloat16 scores stay on disk (`probabilities.npz`) and the first "
            "evaluation that used them is in the git history of this document."
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
            "reference being scored for every method (above). The dense reference "
            "is temporally coarse (Interpretation): Shot AUROC cannot rank methods "
            "on it."
        ),
        "",
        (
            "**Main table** (19 shared held-out shots; the main-text LaTeX table is "
            "`table_supervision_swap_main.tex`; ae-ours rows are the seed mean and "
            "sample SD, the other rows single models; the last two columns are the "
            "paired AUROC difference from ae-rcn with a shot-bootstrap interval):"
        ),
        "",
        *main_markdown(record),
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
            "| ae-ours arm | minus ae-rcn, dense | minus ae-rcn, legacy annotation | "
            "minus ae-lstm, dense | minus ae-lstm, legacy annotation |"
        ),
        "|---|---|---|---|---|",
    ]
    for arm in ARMS:
        cells = [
            diff_text(rel(fair[ref], f"ae-ours-{arm}", base, metric))
            for base in ("ae-rcn", "ae-lstm")
            for ref in ("dense", "legacy")
        ]
        lines.append(f"| {MODEL_NOTE[arm]} | " + " | ".join(cells) + " |")
    return lines + [""]


def vs(label: str, other: str, dense: dict, annotation: dict) -> str:
    return (
        f"{label} {verdict(dense)} {other} on the dense reference "
        f"({diff_text(dense)}) and {verdict(annotation)} it on the legacy "
        f"annotation ({diff_text(annotation)})"
    )


def lstm_seed_note(record: dict) -> str:
    """How many seeds resolve the legacy-minus-ae-lstm lead, and the seed interval."""
    dense = record["results"]["fair_19"]["references"]["dense"]
    try:
        seeds = record["convergence"]["accepted_seeds"]["legacy"]
        per_seed_ci = [
            dense["paired_differences"][f"ae-ours-legacy-seed{seed} minus ae-lstm"][
                "auroc"
            ]["ci95"]
            for seed in seeds
        ]
        seed_ci = rel(dense, "ae-ours-legacy", "ae-lstm")["ci95"]
    except KeyError:
        return ""
    note = f"{sum(low > 0 for low, _ in per_seed_ci)} of {len(seeds)} seeds"
    if seed_ci[0] <= 0:
        note += "; not resolved once seed variance is included"
    return note


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
    rcn_d, rcn_a = out["legacy_rcn"]
    lstm_d, lstm_a = out["legacy_lstm"]
    out["no_reversal_rcn"] = (
        verdict(rcn_d) != "leads" and rcn_d["mean"] < 0 and rcn_a["mean"] < 0
    )
    out["rcn_resolved"] = verdict(rcn_d) == "trails" and verdict(rcn_a) == "trails"
    out["lstm_sign"] = lstm_d["mean"] > 0 > lstm_a["mean"]
    out["lstm_resolved"] = verdict(lstm_d) == "leads" and verdict(lstm_a) == "trails"
    #: The dense interval's lower end sits within rounding of zero.
    out["lstm_marginal"] = out["lstm_resolved"] and lstm_d["ci95_shot"][0] < 0.005
    out["lstm_seeds"] = lstm_seed_note(record)
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


def seed_sd_multiple(gap: float, sd: float) -> str:
    """A gap in units of the seed SD, as prose."""
    ratio = gap / sd if sd else 0.0
    return "within roughly 1 seed SD" if ratio <= 1.5 else f"about {ratio:.0f} seed SD"


def selection_text(record: dict) -> str:
    """The selection paragraph, derived from the published scores and the retrain."""
    seed = {
        "all_60": record["results"]["all_60"]["references"]["dense"]["seed_summary"][
            "methods"
        ]["ae-ours-threeway"],
        "fair_19": record["results"]["fair_19"]["references"]["dense"]["seed_summary"][
            "methods"
        ]["ae-ours-threeway"],
    }
    gaps = {
        (group, metric): (
            PUBLISHED[group][metric] - seed[group][metric]["mean"],
            seed[group][metric]["sd"],
        )
        for group in seed
        for metric in ("auroc", "auprc")
    }
    beyond = [
        f"{metric.upper()} on {'19' if group == 'fair_19' else '60'} shots"
        for (group, metric), (gap, sd) in gaps.items()
        if gap > sd
    ]
    cells = "; ".join(
        f"{metric.upper()} {gap:+.4f} against SD {sd:.4f} on "
        f"{'19' if group == 'fair_19' else '60'} shots"
        for (group, metric), (gap, sd) in gaps.items()
    )
    g60 = {m: max(gaps[("all_60", m)][0], 0.0) for m in ("auroc", "auprc")}
    spread = {m: seed_sd_multiple(g60[m], gaps[("all_60", m)][1]) for m in g60}
    precision = record["precision_check"]["max_abs_seed_mean_change"]["threeway"]
    return (
        "**Selection.** The reviewers' objection was that the published model "
        "was selected on the 60 validation shots that include the 19 benchmark "
        "shots. The same recipe retrained with the epoch chosen on 20 separate "
        "training shots (the threeway arm, which is the published target) scores "
        f"AUROC {f3(seed['fair_19']['auroc']['mean'])} on the 19 shots (published "
        f"{PUBLISHED['fair_19']['auroc']:.3f}) and "
        f"{f3(seed['all_60']['auroc']['mean'])} on the 60 shots (published "
        f"{PUBLISHED['all_60']['auroc']:.3f}), AUPRC "
        f"{f3(seed['fair_19']['auprc']['mean'])} and "
        f"{f3(seed['all_60']['auprc']['mean'])} (published "
        f"{PUBLISHED['fair_19']['auprc']:.3f} and "
        f"{PUBLISHED['all_60']['auprc']:.3f}). Published minus retrained seed mean, "
        f"against the seed SD of the retrain: {cells}. "
        + (
            f"The gap exceeds the seed SD for {' and '.join(beyond)}, so the "
            "retrained scores are lower than the published ones by more than the "
            "seed spread there. "
            if beyond
            else "No gap exceeds the seed SD. "
        )
        + "The comparison also changes the training data: the retrain uses 100 "
        "training shots, the published model 120. Selection on the validation "
        "block and the 20 extra training shots together therefore account for "
        f"about {g60['auroc']:.3f} AUROC ({spread['auroc']}) and "
        f"{g60['auprc']:.3f} AUPRC ({spread['auprc']}) on the 60 shots; these "
        "are point estimates, not bounds, because the published model is one seed "
        "(its scores are rounded to three decimals, and were "
        "scored under the original bfloat16 inference; float32 moves the retrain's "
        f"seed means by at most {precision['auroc']:.4f} AUROC and "
        f"{precision['auprc']:.4f} AUPRC). The recipe itself (symmetric cross "
        "entropy over binary cross entropy) was chosen on the validation block "
        "(the model README), so only the epoch choice is clean."
    )


def coarse_text(record: dict) -> str:
    coarse = record["dense_reference_coarseness"]
    blocks = record["results"]
    shot60 = shot_median(blocks["all_60"], "dense", "clock-dense")
    shot19 = shot_median(blocks["fair_19"], "dense", "clock-dense")
    return (
        "**The dense reference is temporally coarse.** "
        f"{coarse['single_present_span_in_table']} of {coarse['shots']} shots carry "
        f"a single present span in the table ({coarse['single_run_of_present_frames']} "
        "a single run of present frames on the 10 ms grid), and the input-free dense "
        f"clock reaches a median within-shot AUROC of {shot60:.4f} on the 60 "
        f"evaluation shots ({shot19:.4f} on the 19) against it. Within a shot the "
        "reference is therefore predicted almost perfectly by time alone, so the "
        "Shot AUROC column cannot rank methods on it, and the pooled AUROC against "
        "it mostly measures onset and offset timing between shots."
    )


def interpretation(record: dict) -> list[str]:
    fair = record["results"]["fair_19"]["references"]
    all60 = record["results"]["all_60"]["references"]
    out = ["## Interpretation", ""]
    legacy_model = ARM_MODEL["legacy"]

    # 1. the effect of supervision inside ae-ours
    lines = []
    for label, res in (("19 shared shots", fair), ("60 shots", all60)):
        parts = []
        for first in ("dense", "threeway"):
            d = rel(res["dense"], f"ae-ours-{first}", "ae-ours-legacy")
            a = rel(res["legacy"], f"ae-ours-{first}", "ae-ours-legacy")
            parts.append(
                f"{ARM_MODEL[first]} minus {legacy_model} {diff_text(d)} against the "
                f"dense reference and {diff_text(a)} against the legacy annotation"
            )
        lines.append(f"On the {label}, the AUROC of " + "; ".join(parts) + ".")
    dt = rel(fair["dense"], "ae-ours-dense", "ae-ours-threeway")
    dt60 = rel(all60["dense"], "ae-ours-dense", "ae-ours-threeway")
    lines.append(
        "Dense relabel minus agreement as the training target, against the dense "
        f"reference: {diff_text(dt)} on the 19 shots and {diff_text(dt60)} on the 60 "
        "shots, so dense relabelling and annotation-and-TokEye agreement are not "
        "separated by this experiment."
    )
    out += ["**Within ae-ours (supervision).** " + " ".join(lines), ""]

    # 2. against the saved detectors
    out += contrast_table(record, "auroc") + contrast_table(record, "auprc")
    f = findings(record)
    statements = [
        vs("The model trained on the legacy annotation", "ae-rcn", *f["legacy_rcn"])
        + ". "
        + (
            (
                "It trails ae-rcn on both references: there is no reversal "
                "against ae-rcn when ae-ours is trained on the legacy annotation."
                if f["rcn_resolved"]
                else "Its point estimate is below ae-rcn's on both references, "
                "resolved on the legacy annotation but with an interval that "
                "includes zero on the dense reference; it does not lead ae-rcn "
                "there, so there is no reversal against ae-rcn when ae-ours is "
                "trained on the legacy annotation."
            )
            if f["no_reversal_rcn"]
            else "The reversal against ae-rcn is not excluded by this comparison."
        ),
        vs("The same model", "ae-lstm", *f["legacy_lstm"])
        + ". "
        + (
            "The reversal against ae-lstm persists"
            + (f" ({f['lstm_seeds']})" if f["lstm_seeds"] else "")
            + ": ae-ours trained on the legacy "
            "annotation is above it on the dense reference and below it on the "
            "legacy annotation."
            + (
                " The dense interval's lower end is within 0.005 of zero, so that "
                "resolution is marginal."
                if f["lstm_marginal"]
                else ""
            )
            if f["lstm_resolved"]
            else "The reversal against ae-lstm persists in sign (above on the "
            "dense reference, below on the legacy annotation) but is resolved "
            "only on the legacy annotation; the dense interval includes zero."
            if f["lstm_sign"]
            else "The reversal against ae-lstm does not persist in both signs."
        ),
    ]
    for arm in ("dense", "threeway"):
        statements.append(
            vs(
                f"The model trained on {ARM_TARGET[arm]}",
                "ae-rcn",
                rel(fair["dense"], f"ae-ours-{arm}", "ae-rcn"),
                rel(fair["legacy"], f"ae-ours-{arm}", "ae-rcn"),
            )
            + "."
        )
    if f["no_reversal_rcn"] and f["lead_with_new"]:
        statements.append(
            "ae-ours's lead over ae-rcn on the dense reference therefore depends on "
            "the target: it is absent when the same recipe trains on the legacy "
            "annotation, rather than the best achievable legacy-trained model "
            "(Limitations: the recipe was developed for the agreement target, the "
            "legacy seeds are unstable, and training windows are fixed), and "
            "present when it trains on either of the other two targets. Which of "
            "the two carries it is not separated, and the cross-architecture "
            "confounds below apply to every statement against the saved detectors."
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
            "evaluation data. The clock from the legacy annotation "
            f"{profile(clocks['clock-annotation']['prior'])}; the clock from the "
            f"dense relabel {profile(clocks['clock-dense']['prior'])}. On the 19 "
            "shared shots its pooled AUROC is "
            f"{f3(value(ref_a, 'clock-annotation'))} against the legacy annotation "
            "(clock from the legacy annotation) and "
            f"{f3(value(ref_d, 'clock-dense'))} against the dense reference (clock "
            "from the dense relabel). For comparison (legacy annotation / dense): "
            f"ae-rcn {f3(value(ref_a, 'ae-rcn'))} / {f3(value(ref_d, 'ae-rcn'))}, "
            f"ae-lstm {f3(value(ref_a, 'ae-lstm'))} / {f3(value(ref_d, 'ae-lstm'))}, "
            "and the ae-ours seed means "
            + ", ".join(
                f"{MODEL_NOTE[arm]} {f3(mean(ref_a, 'ae-ours-' + arm))} / "
                f"{f3(mean(ref_d, 'ae-ours-' + arm))}"
                for arm in ARMS
            )
            + ". Within shots the clock's median AUROC is "
            f"{f3(shot('legacy', 'clock-annotation'))} against the legacy "
            f"annotation and {f3(shot('dense', 'clock-dense'))} against the dense "
            f"reference, against ae-rcn {f3(shot('legacy', 'ae-rcn'))} / "
            f"{f3(shot('dense', 'ae-rcn'))} and ae-lstm "
            f"{f3(shot('legacy', 'ae-lstm'))} / {f3(shot('dense', 'ae-lstm'))}."
        ),
        "",
        (
            "Against the legacy annotation the clock reaches "
            f"{f['margin_rcn']:.0%} of ae-rcn's pooled AUROC margin over chance and "
            f"{f['margin_lstm']:.0%} of ae-lstm's. Paired AUROC differences against "
            "the legacy annotation: ae-rcn minus the clock from the legacy "
            f"annotation {diff_text(rel(ref_a, 'ae-rcn', 'clock-annotation'))}, "
            f"ae-lstm minus it {diff_text(rel(ref_a, 'ae-lstm', 'clock-annotation'))}, "
            "and "
            + ", ".join(
                f"ae-ours trained on {ARM_TARGET[arm]} minus it "
                f"{diff_text(rel(ref_a, f'ae-ours-{arm}', 'clock-annotation'))}"
                for arm in ARMS
            )
            + ". Against the dense reference, ae-ours-dense minus the dense clock is "
            f"{diff_text(rel(ref_d, 'ae-ours-dense', 'clock-dense'))} and ae-rcn "
            f"minus it {diff_text(rel(ref_d, 'ae-rcn', 'clock-dense'))}."
        ),
        "",
        coarse_text(record),
        "",
        (
            "ae-ours trains on random 182 ms windows (710 columns) and cannot learn "
            "absolute time; Garcia's models read the whole 0 to 2 s record and can. "
            "Where the legacy annotation concentrates in time, as the profile above "
            "shows, part of the older detectors' score against it is time "
            "context, not a better reading of the spectrogram, and the clock is "
            "the control that measures how much. The fair test of time context "
            "against supervision is the deferred LSTM retrain "
            "(`ae-lstm-retrained`; Limitations)."
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
            "method calibrated on the reference it is scored on; the clocks' F1 is "
            "in-sample (their thresholds are tuned on shots that are part of the "
            "clocks' own 120). The earlier protocol left the saved detectors at "
            "thresholds set against the legacy annotation, which moves their F1 on "
            f"the dense reference: ae-rcn on the 19 shots scores {f3(f1_dense)} with "
            f"its threshold calibrated on dense and {f3(f1_own)} at its saved "
            "annotation-set threshold (`f1_own_target_threshold`)."
        ),
        "",
    ]

    # 5. selection
    out += [selection_text(record), ""]
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


def provenance_text(record: dict) -> str:
    hist = record["dense_history"]
    rec = record["dense_reconciliation"]
    people = "; ".join(
        f"{'unnamed' if name == '(unnamed)' else name} {item['saves']} saves on "
        f"{item['shots_saved']} shots"
        + (
            f" and {item['confirmation_entries']} confirmation entries"
            if item["confirmation_entries"]
            else ""
        )
        for name, item in hist["by_name"].items()
    )
    notes = "; ".join(f'"{note}" ({n} entries)' for note, n in hist["notes"].items())
    times = hist["confirmation_times"]
    batch = f", all written at {utc(times[0])}" if len(times) == 1 else ""
    unnamed = hist["by_name"].get("(unnamed)")
    named_from = min(
        item["first_saved"]
        for name, item in hist["by_name"].items()
        if name != "(unnamed)" and item["first_saved"] is not None
    )
    garcia = hist["by_name"]["Alvin Garcia"]
    edits = garcia["changed_shots"]
    days = {garcia["first_change"][:10], garcia["last_change"][:10]}
    when = (
        f"on {garcia['first_change'][:10]}"
        if len(days) == 1
        else f"from {garcia['first_change'][:10]} to {garcia['last_change'][:10]}"
    )
    differing = len(rec["difference"]["train_selection_120"]["differing_shots"])
    band = rec["display_band_change"]
    return (
        f"- **Dense labels' provenance and reviewers.** The history file holds "
        f"{hist['entries']} entries on {hist['shots']} shots. Its `reviewer` field is "
        f"the login of the review server's process ({', '.join(hist['logins'])}), "
        f"not a person; the person is the `name` field: {people}"
        + (
            f" (the unnamed saves run {unnamed['first_saved'][:10]} to "
            f"{unnamed['last_saved'][:10]}, before the first named save on "
            f"{named_from[:10]})"
            if unnamed
            else ""
        )
        + f". The confirmation note reads {notes}{batch}. The source of every entry is "
        f"`{'`, `'.join(hist['sources'])}`: the review was pre-filled from the "
        "annotation's source table. A. Garcia, the author of ae-rcn and ae-lstm, "
        "therefore helped make the dense labels the training arms learn from: "
        f"a confirmation in his name was recorded for all "
        f"{garcia['shots_confirmed']} shots at the owner's request; his own "
        f"{garcia['saves']} saves cover {garcia['shots_saved']} shots, and the "
        f"{garcia['interval_changing_saves']} saves that changed intervals "
        f"({when}) cover {edits['train']} training, {edits['selection']} selection "
        f"and {edits['evaluation']} evaluation shots. "
        + garcia_text(record)
        + f". The current table differs from the snapshot the paper's audit scored "
        f"on {differing} of the 120 training and selection shots and on none of "
        "the 19 shared shots. Those changes were made with the "
        f"{band['after']} review display ({band['before']} before {band['date']}), "
        "while ae-ours reads 80.6 kHz and above. The arms were trained on the "
        "current table, edits included; no model was retrained without them. "
        "Whether a TokEye layer was on screen while reviewing is an open question, "
        "and it bears on why dense and threeway supervision score alike."
    )


def legacy_limits_text(record: dict) -> str:
    cells = []
    for group, short, _ in GROUPS:
        for reference, name in REFS:
            vals = [
                m["auroc"]["value"]
                for _, m in per_seed(
                    record["results"][group], "legacy", reference, record
                )
            ]
            cells.append(f"{min(vals):.3f} to {max(vals):.3f} ({short}, {name})")
    collapsed = sorted(
        n.removeprefix("ae-ours-legacy-")
        .removesuffix("-superseded1")
        .replace("seed", "seed ")
        for n in record["convergence"]["superseded"]
        if n.startswith("ae-ours-legacy")
    )
    return (
        "- **The legacy arm is not the best achievable legacy-trained model.** The "
        "recipe (the symmetric-cross-entropy weights, the learning rate and the "
        "early stopping) was developed for the agreement target and reused "
        "unchanged for the legacy and dense targets, and the legacy arm is the "
        "unstable one: the pooled AUROC of its three accepted seeds ranges "
        + "; ".join(cells)
        + (
            f"; the first attempt of {', '.join(collapsed)} stopped inside the "
            "constant-output plateau and was rerun under the declared rule."
            if collapsed
            else "; no first attempt was superseded."
        )
        + " A comparison with ae-rcn that rests on the legacy arm therefore says "
        "what this recipe does on the legacy annotation, not what a model tuned "
        "for it would do."
    )


def frame_sensitivity_text(record: dict) -> str:
    """Which arm the bfloat16/float32 frame-level disagreement sits in."""
    seeds = record["convergence"]["accepted_seeds"]
    over = {
        arm: [
            record["runs"][f"ae-ours-{arm}-seed{seed}"]["fp32"][
                "frame_probability_difference"
            ]["evaluation_shots_over_0.1"]
            for seed in seeds[arm]
        ]
        for arm in ARMS
    }
    worst = max(over, key=lambda arm: max(over[arm]))
    rest = max(n for arm in ARMS if arm != worst for n in over[arm])
    counts = ", ".join(map(str, over[worst][:-1])) + f" and {over[worst][-1]}"
    return (
        f"The frame-level disagreement between the two precisions sits in the {worst} "
        f"arm: its three seeds have {counts} of the 60 evaluation shots with a "
        f"single-frame difference above 0.1, against at most {rest} in any other "
        "arm's runs."
        + (
            " The legacy epochs were selected under bfloat16, which is one more "
            "reason that arm is not the best achievable legacy-trained model."
            if worst == "legacy"
            else ""
        )
    )


def limitations_section(record: dict) -> list[str]:
    grid = record["frame_grid"]
    fp32 = record["precision_check"]["max_abs_change_per_run"]
    return [
        "## Limitations",
        "",
        provenance_text(record),
        legacy_limits_text(record),
        (
            "- **Fixed training windows.** Every epoch re-draws the same 800 "
            "windows (eight per shot; persistent DataLoader workers never see "
            "`set_epoch()`); the shot order alone changes. This affects every arm "
            "and seed equally, and the published recipe too, and it is not fixed "
            "in this round, because it would mean rerunning every arm. Its effect "
            "on the arms' order is not measured."
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
            "- **Precision.** Epochs were selected during training under bfloat16 "
            "autocast; only the reported scores, thresholds and intervals are "
            f"float32. The largest change of a run's pooled AUROC is "
            f"{fp32['auroc']['value']:.4f} (Inference precision). "
            + frame_sensitivity_text(record)
        ),
        (
            "- **Dense reference.** It is temporally coarse, so within-shot AUROC "
            "cannot rank methods on it (Interpretation); and the 10 ms frame grid "
            f"is the audit's, with {grid['columns_in_another_frame']} of "
            f"{grid['columns']:,} columns in a neighbouring frame (Frozen "
            "protocol)."
        ),
        (
            "- **Clock F1 is in-sample.** The clocks' thresholds are tuned on 20 "
            "shots that are part of the 120 shots the clocks are built from; their "
            "AUROC and AUPRC are not affected."
        ),
        (
            "- **The LSTM retrain was deferred.** The brief's optional item "
            "(`ae-lstm-retrained`: the published 3 x 64 LSTM architecture trained on "
            "the same 100/20 split, on whole records, with dense and with legacy "
            "supervision) was not run. It remains the control that separates time "
            "context from supervision: it gives the older architecture the same "
            "supervision treatment while keeping its time context, and the clock "
            "rows bound what time context alone achieves."
        ),
        (
            "- The saved detectors are scored on 19 shots and calibrated on six; their "
            "intervals are wide."
        ),
        "",
    ]


def interrupted_lines(record: dict) -> list[str]:
    """One line per launch that died before it finished (never scored)."""
    lines = []
    for name, item in sorted(record.get("interrupted_runs", {}).items()):
        again = (
            f" and the first {item['epochs_recorded']} epochs of the completed run "
            "reproduce its validation losses exactly"
            if item["first_epochs_match_completed_run"]
            else ""
        )
        lines += [
            f"`{name}` (`{item['completed_run']}-interrupted`) is not in the table "
            "and was never scored: a first launch "
            f"of the same seed that stopped after {item['epochs_recorded']} recorded "
            f"epochs (its attempt record still reads `{item['attempt_status']}`"
            + ("" if item["has_run_json"] else " and it has no `run.json`")
            + f"); the seed was trained again from scratch in "
            f"`{item['completed_run']}`{again}.",
            "",
        ]
    return lines


def excluded_section(record: dict) -> list[str]:
    runs = record["excluded_runs"]
    lines = ["## Excluded and superseded records", ""]
    if not runs:
        return lines + ["None.", "", *interrupted_lines(record)]
    lines += [
        (
            "These records are not in any mean, SD or interval above. They are scored "
            "with the same code, thresholds calibrated on the selection shots, so the "
            "effect of the convergence rule can be read."
        ),
        "",
        (
            "| Record | Epochs run | Selected epoch | Cohort | Dense AUROC | "
            "Legacy-annotation AUROC | Selection within-shot SD | Selection AUROC |"
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
    lines += ["", *interrupted_lines(record)]
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
        "on one activity target (`sup_legacy`: the legacy annotation; `sup_dense`: "
        "the dense relabel; `sup_threeway`: annotation-and-TokEye agreement). Each "
        "line is the mean of three seeds on the 19 shared held-out shots against "
        "the dense reference, scored in float32, with F1 calibrated on the dense "
        "reference on the 20 selection shots; they are not the published model on "
        "the first line, which was selected on the 60 validation shots, and the "
        "legacy line is the recipe on the legacy annotation, not the best "
        "achievable legacy-trained model. Checkpoints under "
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


def garcia_shots(record: dict) -> str:
    """The shots whose intervals A. Garcia's saves changed, by cohort."""
    edits = record["dense_history"]["by_name"]["Alvin Garcia"]["changed_shots"]
    return (
        f"{edits['train'] + edits['selection']} training and selection shots and "
        f"{n_shots(edits['evaluation'])}"
    )


def summary(record: dict) -> str:
    fair = record["results"]["fair_19"]["references"]
    f = findings(record)
    seed_mean = lambda ref, arm: f3(
        fair[ref]["seed_summary"]["methods"][f"ae-ours-{arm}"]["auroc"]["mean"]
    )
    precision = record["precision_check"]["max_abs_change_per_run"]
    return (
        "On the 19 shared held-out shots the seed-mean AUROC of ae-ours trained on "
        "the legacy annotation, the dense relabel and annotation-and-TokEye "
        "agreement is "
        + ", ".join(seed_mean("dense", arm) for arm in ARMS)
        + " against the dense reference and "
        + ", ".join(seed_mean("legacy", arm) for arm in ARMS)
        + " against the legacy annotation. ae-ours trained on the legacy annotation "
        + (
            (
                "trails ae-rcn on both references"
                if f["rcn_resolved"]
                else "is below ae-rcn on both references in point estimate (resolved "
                "on the legacy annotation only)"
            )
            + ", so, for this recipe (rather than the best achievable legacy-trained "
            "model), the reversal against ae-rcn needs dense or agreement targets"
            if f["no_reversal_rcn"] and f["lead_with_new"]
            else "does not trail ae-rcn on both references"
        )
        + "; "
        + (
            "the reversal against ae-lstm persists"
            + (f" ({f['lstm_seeds']})" if f["lstm_seeds"] else "")
            + (", marginally on the dense reference" if f["lstm_marginal"] else "")
            if f["lstm_resolved"]
            else "the reversal against ae-lstm persists in sign but is not resolved "
            "on the dense reference"
            if f["lstm_sign"]
            else "the reversal against ae-lstm does not persist in both signs"
        )
        + f". An input-free clock reaches {f['margin_rcn']:.0%} of ae-rcn's AUROC "
        "margin over chance against the legacy annotation, which makes time context "
        "a confound of every comparison with the saved detectors, and the dense "
        "reference is temporally coarse, so within-shot AUROC cannot rank methods "
        "on it. A. Garcia, the author of the saved detectors, saved interval "
        f"changes to {garcia_shots(record)} of the dense labels (Limitations). "
        "Scores are float32; "
        f"the largest change of a run's pooled AUROC from the bfloat16 inference is "
        f"{precision['auroc']['value']:.4f}. The Interpretation and "
        "Cross-architecture confounds sections give the numbers and the limits."
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
            "checks target equality and split isolation. The main-text paper table "
            "is `table_supervision_swap_main.tex` and the two long appendix tables "
            "are `table_supervision_swap.tex`, in the same directory."
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
        + precision_section(record)
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
            "# float32 re-inference of every finished run (GPU, CUDA interpreter)",
            "python scripts/labeler/ae_supervision_swap.py infer-fp32",
            "# CPU, pixi labelmaker environment",
            "python scripts/labeler/ae_supervision_swap.py verify",
            "python scripts/labeler/ae_supervision_swap.py audit-convergence",
            "python scripts/labeler/ae_supervision_swap.py evaluate",
            "```",
            "",
            (
                "`audit-convergence` applies the declared rule to every record, archives "
                "any record that stopped before the rule allowed, and lists the runs still to "
                "train. Completed runs are never overwritten. The launchers use a short "
                "TMPDIR (`$LABELER_ROOT/scratch/ae-sw`) because DataLoader workers add "
                "`/pymp-*/listener-*` to it and a socket path must stay under 108 bytes."
            ),
            "",
        ]
    )
    (repo / "docs/labeler/ae_supervision_swap.md").write_text("\n".join(doc))
    for name, table in (
        ("table_supervision_swap.tex", paper_table(record)),
        ("table_supervision_swap_main.tex", main_table(record)),
    ):
        (out / name).write_text(table)
        (repo / "outputs/labeler/ae/supervision_swap" / name).write_text(table)
    write_readme(record, repo)
