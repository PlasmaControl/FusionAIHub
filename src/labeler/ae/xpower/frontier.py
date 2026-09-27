"""Sweep saved AE candidates on validation shots for the owner's operating point.

    python -m labeler.ae.xpower.frontier [--models DIR]

For each saved candidate under DIR (default `$LABELER_ROOT/models/ae_xpower/v1`),
score only its split.csv validation shots, using its archived review labels.
Reuse `evaluate.shot_frames` for the same 0-2 s frame selection and MHD definition
as validation in `evaluate.run_choose`, and its cells/rates and `scoring.stats`
for the pooled metrics. Infer each shot once, then sweep 0.05, 0.10, ..., 0.95.

The operating-point rule, separately for each candidate: among thresholds whose
validation F1 is within 0.02 of that candidate's best validation F1, the lowest
validation MHD false-positive rate (ties: the higher F1). An exact tie after
both comparisons takes the lower threshold. Undefined rates are left blank in
CSV and shown as n/a in Markdown; no defined F1 or MHD rate means no operating
point can be recommended.

Write validation_frontier.csv and validation_frontier.md atomically in DIR,
marking chosen.json's candidate and whether any (candidate, threshold) reaches
validation F1 >= 0.90 with MHD FP <= 0.05. This is validation evidence for the
owner: it neither changes chosen.json nor reads or scores test-shot data.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch

from ...config import Paths, atomic_path
from ...events.review import labels
from ...scoring import stats
from . import evaluate, model_dir, train

THRESHOLDS = tuple(k / 20 for k in range(1, 20))
COLUMNS = (
    "candidate",
    "chosen",
    "threshold",
    "f1",
    "f1_low",
    "f1_high",
    "precision",
    "recall",
    "fp_rate_mhd",
    "fp_rate_mhd_low",
    "fp_rate_mhd_high",
    "fp_rate_other",
    "operating_point",
    "chosen_rule_threshold",
    "validation_shots",
    "scored_frames",
    "mhd_absent_frames",
)


def _number(value) -> float | None:
    return float(value) if np.isfinite(value) else None


def operating_point(rows: list[dict]) -> dict | None:
    """The candidate's row under the F1-margin/MHD rule, or None if undefined."""
    defined = [row for row in rows if row["f1"] is not None]
    if not defined:
        return None
    best = max(row["f1"] for row in defined)
    eligible = [
        row
        for row in defined
        if row["f1"] >= best - evaluate.F1_MARGIN - 1e-12
        and row["fp_rate_mhd"] is not None
    ]
    return min(
        eligible,
        key=lambda row: (row["fp_rate_mhd"], -row["f1"], row["threshold"]),
        default=None,
    )


def _candidate(paths: Paths, file: Path, chosen: str | None) -> list[dict]:
    model, blob = train.load(file)
    split = train.read_split(file.parent / "split.csv")
    validation = sorted(shot for shot, which in split.items() if which == "val")
    if not validation:
        raise ValueError(f"{file.parent / 'split.csv'}: no validation shots")
    saved = labels.read_saved(file.parent)
    frames = [
        evaluate.shot_frames(s, paths=paths, label=saved[s], model=model, blob=blob)
        for s in validation
    ]
    counts = {
        "validation_shots": len(frames),
        "scored_frames": int(sum(f.scored.sum() for f in frames)),
        "mhd_absent_frames": int(
            sum((f.scored & evaluate.mhd_absent(f)).sum() for f in frames)
        ),
    }
    rows = []
    for threshold in sorted({*THRESHOLDS, blob["threshold"]}):
        for frame in frames:
            frame.said["ae_xpower"] = frame.prob >= threshold
        totals = evaluate.cells(frames, "ae_xpower").sum(axis=0)
        rows.append(
            {
                "candidate": file.parent.name,
                "chosen": file.parent.name == chosen,
                "threshold": threshold,
                "f1": _number(stats.f1(totals)),
                "precision": _number(stats.precision(totals)),
                "recall": _number(stats.recall(totals)),
                "fp_rate_mhd": _number(
                    evaluate.fp_rate(
                        evaluate.cells(frames, "ae_xpower", evaluate.mhd_absent).sum(
                            axis=0
                        )
                    )
                ),
                "fp_rate_other": _number(
                    evaluate.fp_rate(
                        evaluate.cells(frames, "ae_xpower", evaluate.other_absent).sum(
                            axis=0
                        )
                    )
                ),
                "operating_point": False,
                "chosen_rule_threshold": threshold == blob["threshold"],
                "f1_low": None,
                "f1_high": None,
                "fp_rate_mhd_low": None,
                "fp_rate_mhd_high": None,
                **counts,
            }
        )
    point = operating_point(rows)
    if point is not None:
        point["operating_point"] = True
    for row in rows:
        if not (row["operating_point"] or row["chosen_rule_threshold"]):
            continue
        for frame in frames:
            frame.said["ae_xpower"] = frame.prob >= row["threshold"]
        for key, where, metric in (
            ("f1", None, stats.f1),
            ("fp_rate_mhd", evaluate.mhd_absent, evaluate.fp_rate),
        ):
            interval = evaluate._estimate(
                evaluate.cells(frames, "ae_xpower", where), metric
            )
            row[f"{key}_low"] = interval["low"]
            row[f"{key}_high"] = interval["high"]
    return rows


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _interval(row: dict, key: str) -> str:
    return f"{_fmt(row[key])} [{_fmt(row[f'{key}_low'])}, {_fmt(row[f'{key}_high'])}]"


def report_md(rows: list[dict]) -> str:
    """The chosen candidate, each suggested point, and validation feasibility."""
    feasible = any(
        row["f1"] is not None
        and row["fp_rate_mhd"] is not None
        and row["f1"] >= 0.90
        and row["fp_rate_mhd"] <= 0.05
        for row in rows
    )
    lines = [
        "# AE validation frontier",
        "",
        (
            "Validation shots only, each candidate's archived labels and split; "
            "0-2 s with evaluate.run_choose's frame selection and MHD definition."
        ),
        "",
        (
            "Rule per candidate: among thresholds whose validation F1 is within 0.02 "
            "of that candidate's best validation F1, choose the lowest validation "
            "MHD false-positive rate (ties: the higher F1; "
            "exact ties: lower threshold)."
        ),
        "",
        (
            "Any (candidate, threshold) with validation "
            f"F1 >= 0.90 and MHD FP <= 0.05: {'yes' if feasible else 'no'}. "
            "This feasibility line is on point estimates."
        ),
        "",
        (
            "F1 and MHD FP at each operating point and each saved chosen-rule "
            "threshold have 95 % shot-bootstrap intervals (2000 replicates, "
            f"seed {evaluate.SEED}), computed as in evaluate."
        ),
        "",
        (
            "The full 0.05-0.95 sweep is in validation_frontier.csv. "
            "Undefined rates are blank there and n/a here. "
            "The saved choice and test evaluation are unchanged."
        ),
        "",
        (
            "| candidate | best F1 | threshold | F1 | precision | recall | "
            "MHD FP | other-absent FP |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in sorted({row["candidate"] for row in rows}):
        candidate = [row for row in rows if row["candidate"] == name]
        best = max(
            (row["f1"] for row in candidate if row["f1"] is not None), default=None
        )
        marked = f"{name} (chosen)" if candidate[0]["chosen"] else name
        point = next((row for row in candidate if row["operating_point"]), None)
        if point is None:
            lines.append(
                f"| {marked} | {_fmt(best)} | no defined operating point "
                "| n/a | n/a | n/a | n/a | n/a |"
            )
        else:
            metrics = " | ".join(
                _interval(point, key)
                if key in ("f1", "fp_rate_mhd")
                else _fmt(point[key])
                for key in ("f1", "precision", "recall", "fp_rate_mhd", "fp_rate_other")
            )
            lines.append(
                f"| {marked} | {_fmt(best)} | {point['threshold']:.2f} | {metrics} |"
            )
    lines += [
        "",
        "Saved chosen-rule thresholds (the checkpoint's validation choice):",
        "",
        (
            "| candidate | threshold | F1 [95 %] | MHD FP [95 %] | "
            "shots | frames | MHD-absent frames |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        if row["chosen_rule_threshold"]:
            lines.append(
                f"| {row['candidate']} | {row['threshold']:.2f} | "
                f"{_interval(row, 'f1')} | {_interval(row, 'fp_rate_mhd')} | "
                f"{row['validation_shots']} | {row['scored_frames']} | "
                f"{row['mhd_absent_frames']} |"
            )
    return "\n".join([*lines, ""])


def run(paths: Paths, models: Path) -> list[dict]:
    files = sorted(models.glob("*/model.pt"))
    if not files:
        raise ValueError(f"{models}: no saved candidate")
    choice = models / "chosen.json"
    chosen = json.loads(choice.read_text())["candidate"] if choice.is_file() else None
    rows = [row for file in files for row in _candidate(paths, file, chosen)]
    with (
        atomic_path(models / "validation_frontier.csv") as tmp,
        tmp.open("w", newline="") as stream,
    ):
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with atomic_path(models / "validation_frontier.md") as tmp:
        tmp.write_text(report_md(rows))
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--models", type=Path, help="default $LABELER_ROOT/models/ae_xpower/v1"
    )
    args = parser.parse_args(argv)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    paths = Paths.from_env()
    models = args.models or model_dir(paths)
    try:
        run(paths, models)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(f"wrote {models / 'validation_frontier.csv'} and validation_frontier.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
