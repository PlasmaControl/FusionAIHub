"""Render the completed swap's documentation and tables from evaluation JSON."""

from __future__ import annotations

from pathlib import Path


def metric_cell(metric: dict, *, latex: bool = False) -> str:
    mean = metric.get("mean", metric.get("value"))
    if mean is None:
        return r"\textemdash" if latex else "unavailable"
    text = f"{mean:.3f}"
    if metric.get("sd") is not None:
        pm = r"$\pm$" if latex else "±"
        text += f" {pm} {metric['sd']:.3f}"
    if metric["ci95"] is not None:
        lo, hi = metric["ci95"]
        text += f" [{lo:.3f}, {hi:.3f}]"
    return text


def interpretation(record: dict) -> list[str]:
    """Describe supervision effects and clean-selection rankings from saved scores."""
    lines = ["", "## Interpretation", ""]
    fair = record["results"]["fair_19"]["references"]
    for baseline in ("ae-rcn", "ae-lstm"):
        key = f"ae-ours-threeway minus {baseline}"
        dense = fair["dense"]["seed_summary"]["paired_differences"][key]["auroc"]
        legacy = fair["legacy"]["seed_summary"]["paired_differences"][key]["auroc"]
        resolved = dense["ci95"][0] > 0 and legacy["ci95"][1] < 0
        conclusion = (
            "The clean-selection AUROC ranking reversal is supported by both "
            "paired intervals."
            if resolved
            else "The paired intervals do not resolve an AUROC ranking reversal."
        )
        lines += [
            f"On the shared 19 shots, threeway ae-ours minus {baseline} is "
            f"{metric_cell(dense)} against dense and {metric_cell(legacy)} "
            f"against legacy. {conclusion}",
            "",
        ]
    for cohort, label in (("all_60", "60 shots"), ("fair_19", "shared 19 shots")):
        metrics = record["results"][cohort]["references"]["dense"]["seed_summary"][
            "paired_differences"
        ]["ae-ours-dense minus ae-ours-threeway"]
        unresolved = all(m["ci95"][0] <= 0 <= m["ci95"][1] for m in metrics.values())
        conclusion = (
            "None of these paired intervals resolves an improvement over threeway."
            if unresolved
            else "Interpret each improvement using its paired interval."
        )
        lines += [
            f"Dense-supervised minus threeway ae-ours on the {label}, against dense: "
            + "; ".join(
                f"{m.upper()} {metric_cell(metrics[m])}"
                for m in ("auroc", "auprc", "f1")
            )
            + f". {conclusion}",
            "",
        ]
    for cohort, label in (("all_60", "60 shots"), ("fair_19", "shared 19 shots")):
        for reference, result in record["results"][cohort]["references"].items():
            summary = result["seed_summary"]
            # Group insertion order determines the stored subtraction direction.
            pairs = summary["paired_differences"]
            key = "ae-ours-dense minus ae-ours-legacy"
            reverse = key not in pairs
            if reverse:
                key = "ae-ours-legacy minus ae-ours-dense"
            effects = []
            for metric in ("auroc", "auprc", "f1"):
                score = dict(pairs[key][metric])
                if reverse:
                    score["mean"] = -score["mean"]
                    if score["ci95"] is not None:
                        lo, hi = score["ci95"]
                        score["ci95"] = [-hi, -lo]
                effects.append(f"{metric.upper()} {metric_cell(score)}")
            lines += [
                f"Dense-supervised minus legacy-supervised ae-ours on the {label}, "
                f"against {reference}: " + "; ".join(effects) + ".",
                "",
            ]
    for reference, result in fair.items():
        methods = result["seed_summary"]["methods"]
        for metric in ("auroc", "auprc", "f1"):
            ranked = sorted(
                methods, key=lambda n: methods[n][metric]["mean"], reverse=True
            )
            lines += [
                f"On the shared 19 shots against {reference}, the {metric.upper()} "
                "point-estimate ranking is "
                + " > ".join(
                    f"{name} ({methods[name][metric]['mean']:.3f})" for name in ranked
                )
                + ".",
                "",
            ]
    lines += [
        (
            "Legacy supervision has substantial observed training variability. "
            "Its seed 2 selected epoch 2 under the unchanged combined-loss rule; "
            "all completed seeds remain in the mean, SD and pooled intervals. "
            "The contrasts estimate behavior under this frozen training and "
            "selection recipe, including early stopping and target-specific "
            "calibration, rather than the best achievable legacy-trained model."
        ),
        "",
        (
            "These rankings use clean selection for every ae-ours arm. Their "
            "paired intervals above determine which gaps remain uncertain; a "
            "point-estimate ordering alone does not establish a difference. "
            "F1 also reflects calibration against each model's training target. "
            "Garcia probabilities come from fixed saved models, with thresholds "
            "calibrated on the six available selection shots. This experiment "
            "isolates supervision within ae-ours and resolves "
            "its selection overlap, while comparisons across architectures retain "
            "different training recipes."
        ),
        "",
    ]
    return lines


def render_report(record: dict, manifest: dict, out: Path, repo: Path) -> None:
    """Keep numeric doc, README and LaTeX content tied to the saved JSON."""
    doc = [
        "# AE supervision swap",
        "",
        (
            "The three activity-supervision arms completed three seeds each under "
            "clean 100/20 shot selection. Both references are evaluated on all 60 "
            "original validation shots and the 19 shared held-out shots."
        ),
        "",
        (
            "Source for every result below: "
            "[evaluation.json](../../outputs/labeler/ae/supervision_swap/evaluation.json). "
            "It embeds the completed run records, execution IDs, selected epochs, "
            "thresholds, per-seed scores and paired differences. "
            "[manifest.json](../../outputs/labeler/ae/supervision_swap/manifest.json) "
            "contains the exact shot lists and frozen input hashes; "
            "[verification.json](../../outputs/labeler/ae/supervision_swap/verification.json) "
            "checks target equality and split isolation."
        ),
        "",
        "## Frozen protocol",
        "",
        (
            f"The current dense snapshot has {manifest['dense_counts']['rows']} CSV rows, "
            f"{manifest['dense_counts']['present_intervals']} present intervals and "
            f"{manifest['dense_counts']['shots']} shots. Its SHA256 is "
            f"`{manifest['inputs']['dense']['sha256']}`. These observed counts differ "
            "from the brief's anticipated interval count. No source table was edited."
        ),
        "",
        (
            f"NumPy split seed {manifest['split_seed']} freezes 100 training, 20 "
            "selection and 60 evaluation shots. None overlaps the fixed catalog's "
            "blind test split. Seeds 0, 1 and 2 change initialization and sampled "
            "training windows; every arm uses the same split. The original one-seed "
            "manifest was archived before extending its seed list."
        ),
        "",
        (
            "Only activity supervision changes: legacy is the audit's at-least-half "
            "annotation rule at 10 ms, expanded to native columns; dense uses the "
            "catalog any-touch state rule and masks unknown states; threeway retains "
            "native annotation/TokEye agreement and masks disagreement. The original "
            "annotation-and-TokEye frequency target and weights are identical in "
            "every arm. Each input covers 0–2 s with 7,820 native columns."
        ),
        "",
        (
            "The original AeSeldNet recipe uses four CO2 channels, 348 frequency "
            "bins, frequency pools (6, 2, 29), two bidirectional GRUs, SCE plus the "
            "unchanged frequency objective, 710-column windows, eight windows per "
            "shot, batch 16, AdamW 1e-4, weight decay 1e-4, cosine decay to 1e-6, "
            "30 epochs maximum and patience five. The selected epoch minimizes "
            "combined loss on the 20 selection shots. The CUDA interpreter is "
            "`envs/phase3/bin/python`, with `PYTHONPATH=<worktree>/src`; the launcher "
            "retains `AESWAP_PIXI_ENV` as an explicit alternative. CUDA allocations "
            "are capped at 10 GiB. Autocast uses bfloat16 when the runtime reports "
            "support, otherwise float32."
        ),
        "",
        (
            "Each seed's threshold maximizes 10 ms selection F1 against its own "
            "activity target, with the highest threshold breaking ties. Threeway "
            "requires at least half the native columns to have agreement weight "
            "and a majority of those agreeing columns to be positive. The score is "
            "mean native probability; the frozen threshold is used against both "
            "evaluation references. No evaluation frames select epochs or thresholds."
        ),
        "",
        (
            "Garcia probabilities use saved spectrogram predictions: maximum over "
            "the first four AE classes, mean over four chords, nearest output-bin "
            "centre on the 10 ms grid. Only 19 evaluation shots and six selection "
            "shots have available predictions held out from Garcia's training. "
            "The other 41 evaluation shots cannot be scored for these saved models. "
            "Legacy-calibrated older thresholds remain frozen against dense labels."
        ),
        "",
        "## Completed runs and operating points",
        "",
        "Epoch numbers are zero-based, as in the trainer.",
        "",
        "| Supervision | Seed | Selected epoch | Threshold | Execution | GPU |",
        "|---|---:|---:|---:|---|---|",
    ]
    for name, run in sorted(record["runs"].items()):
        ex = run["execution"]
        job = ex["slurm_job_id"] or "head node, GPU 0"
        if ex["slurm_array_job_id"]:
            job = (
                f"{ex['slurm_array_job_id']}_{ex['slurm_array_task_id']} "
                f"(job {ex['slurm_job_id']})"
            )
        gpu = run["training_environment"]["gpu"]
        doc.append(
            f"| {run['supervision']} | {run['seed']} | "
            f"{run['selected_epoch']} | {run['threshold']['threshold']:.9f} | "
            f"{job} | {gpu} |"
        )
    doc += [
        "",
        (
            "Legacy/dense seed 0 ran on A100; fallback runs used V100. The CUDA "
            "runtime probe records that this Torch build reports V100 bfloat16 "
            "support through emulation despite lacking native support. The "
            "as-built trainer therefore selects bfloat16 autocast on V100 too. "
            "The earlier report's hypothetical float32 fallback does not describe "
            "these actual runs. See evaluation.json:gpu_probe and each run's "
            "runtime metadata. Device types and kernel implementations differ; "
            "the observed across-seed SD includes environment variation."
        ),
    ]
    doc += ["", "Older selection thresholds:", ""]
    for name in ("ae-rcn", "ae-lstm"):
        threshold = record["thresholds"][name]
        doc.append(
            f"- {name}: `{threshold['threshold']}`, shots "
            + ", ".join(map(str, threshold["shots"]))
            + "."
        )
    doc += [
        "",
        "## Scores and uncertainty",
        "",
        (
            "Entries for retrains are mean ± sample SD over three seeds, followed "
            "by a 95% pooled bootstrap interval. Each of 1,000 replicates resamples "
            "whole evaluation shots and the observed training seed IDs with "
            "replacement. The statistic averages each seed's pooled-frame metric. "
            "All arms share shot draws and seed indices (seed 20261004); saved "
            "baselines have only shot uncertainty. Frames and seed copies are never "
            "treated as independent observations. Per-seed shot-only intervals "
            "remain in JSON. Three seeds give limited precision for training variability."
        ),
    ]
    tex = ["% Requires booktabs. Retrains: mean +/- sample SD and pooled 95% CI."]
    for group, label in (
        ("all_60", "All 60 evaluation shots"),
        ("fair_19", "Shared 19 held-out shots"),
    ):
        block = record["results"][group]
        doc += [
            "",
            f"### {label}",
            "",
            "Shots: " + ", ".join(map(str, block["shots"])) + ".",
            "",
            "Scorable frames and positives: "
            + "; ".join(
                f"{reference}: {results['n_frames']} frames, "
                f"{results['n_positive']} positives"
                for reference, results in block["references"].items()
            )
            + ".",
            "",
            "| Model / supervision | Reference | AUROC | AUPRC | F1 |",
            "|---|---|---|---|---|",
        ]
        for reference, results in block["references"].items():
            for name, metrics in results["seed_summary"]["methods"].items():
                doc.append(
                    f"| {name} | {reference} | "
                    + " | ".join(
                        metric_cell(metrics[m]) for m in ("auroc", "auprc", "f1")
                    )
                    + " |"
                )
        tex += [
            f"% {label}",
            r"\begin{tabular}{llrrrr}",
            r"\toprule",
            r"Model & Supervision & Dense AUROC & Dense F1 & Legacy AUROC & Legacy F1 \\",
            r"\midrule",
        ]
        dense = block["references"]["dense"]["seed_summary"]["methods"]
        for name in dense:
            if name.startswith("ae-ours-"):
                model, supervision = "ae-ours", name.removeprefix("ae-ours-")
            else:
                model, supervision = name, "legacy (Garcia saved)"
            cells = [
                metric_cell(
                    block["references"][r]["seed_summary"]["methods"][name][m],
                    latex=True,
                )
                for r in ("dense", "legacy")
                for m in ("auroc", "f1")
            ]
            tex.append(f"{model} & {supervision} & " + " & ".join(cells) + r" \\")
        tex += [r"\bottomrule", r"\end{tabular}", ""]
        doc += [
            "",
            "Paired differences use the same pooled draws:",
            "",
            "| First model minus second | Reference | AUROC | AUPRC | F1 |",
            "|---|---|---|---|---|",
        ]
        for reference, results in block["references"].items():
            for pair, metrics in results["seed_summary"]["paired_differences"].items():
                doc.append(
                    f"| {pair} | {reference} | "
                    + " | ".join(
                        metric_cell(metrics[m]) for m in ("auroc", "auprc", "f1")
                    )
                    + " |"
                )
    doc += interpretation(record)
    doc += [
        "",
        "## Reproduction",
        "",
        (
            "From this worktree with the prescribed "
            "scratch TMPDIR, LABELER_ROOT, LABELER_LABEL_TABLES, LABELER_NO_FETCH=1 "
            "and PYTHONPATH=$PWD/src:"
        ),
        "",
        "```bash",
        "# Existing manifest already freezes seeds 0, 1, 2.",
        "sbatch scripts/labeler/ae_supervision_swap.sbatch",
        "# After every run finishes:",
        (
            "pixi run --frozen --no-install --manifest-path "
            "/scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker "
            "python scripts/labeler/ae_supervision_swap.py verify"
        ),
        (
            "pixi run --frozen --no-install --manifest-path "
            "/scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker "
            "python scripts/labeler/ae_supervision_swap.py evaluate"
        ),
        "```",
        "",
        (
            "Completed runs cannot be silently overwritten. A pending "
            "task exceeding 30 minutes is cancelled before running that seed "
            "with CUDA_VISIBLE_DEVICES=0 on the shared head node. A short worktree "
            "symlink resolves multiprocessing socket paths into the prescribed "
            "scratch temp directory; actual temp files stay there."
        ),
        "",
    ]
    (repo / "docs/labeler/ae_supervision_swap.md").write_text("\n".join(doc))
    (out / "table_supervision_swap.tex").write_text("\n".join(tex) + "\n")
    table = repo / "outputs/labeler/ae/supervision_swap/table_supervision_swap.tex"
    table.write_text("\n".join(tex) + "\n")
    readme = repo / "data/events/alfven_eigenmode/README.md"
    content = readme.read_text()
    start = content.index("- ae-ours | legacy supervision |")
    end = content.index("\nScores are against the owner-reviewed", start)
    lines = []
    fair = record["results"]["fair_19"]["references"]["dense"]["seed_summary"][
        "methods"
    ]
    for arm in ("legacy", "dense", "threeway"):
        scores = fair[f"ae-ours-{arm}"]
        lines.append(
            f"- ae-ours | {arm} supervision | clean 100/20 selection | "
            "three seeds | "
            + " | ".join(
                f"{m.upper()}: {scores[m]['mean']:.3f} ± {scores[m]['sd']:.3f}"
                for m in ("auroc", "auprc", "f1")
            )
        )
    lines += [
        "",
        (
            "Supervision-swap lines report seed mean ± sample SD against "
            "dense labels on the shared 19 held-out shots. Pooled intervals, "
            "both references, all 60 shots and paired differences: "
            "[ae_supervision_swap.md](../../../docs/labeler/ae_supervision_swap.md)."
        ),
        "The historical model scores below use the original selection protocol.",
        "",
    ]
    readme.write_text(content[:start] + "\n".join(lines) + content[end:])
