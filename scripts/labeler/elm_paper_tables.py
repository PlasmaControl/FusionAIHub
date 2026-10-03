#!/usr/bin/env python
"""Write paper ELM tables and their shared appendix note from evaluation JSONs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from labeler.config import Paths, git_sha
from labeler.elm import swap_tex

REPO = Path(__file__).resolve().parents[2]
OUTPUTS = REPO / "outputs" / "labeler" / "elm"


def benchmark_table(ours, dsm) -> str:
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & AUROC & AUPRC & F1 \\",
    ]
    for source, record in (("Primary reviewed bins", ours), ("Common DSM bins", dsm)):
        for tag in ("all119", "bes73"):
            res = record["sets"][tag]
            subset = "BES, ELM-O chunks" if tag == "bes73" else "All reviewed shots"
            description = (
                f"{source}, {subset}: {res['n_shots']} shots/{res['bins']:,} bins"
            )
            lines += [r"\midrule", r"\multicolumn{4}{l}{" + description + r"} \\"]
            exposed = False
            for method, _ in (*swap_tex.ROWS, (swap_tex.ALWAYS, swap_tex.ALWAYS)):
                if method not in res["methods"]:
                    continue
                if method in swap_tex.EXPOSED and not exposed:
                    lines.append(swap_tex.exposure_heading(4))
                    exposed = True
                if method == swap_tex.ALWAYS:
                    lines.append(r"\midrule")
                result = res["methods"][method]
                cells = [
                    swap_tex.metric_cell(result, m) for m in ("auroc", "auprc", "f1")
                ]
                lines.append(
                    swap_tex.method_label(method) + " & " + " & ".join(cells) + r" \\"
                )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    counts = ours["sets"]["all119"]["methods"]["elm-ours"]["counts"]
    crowd_share = counts["crowd_bins"] / (counts["tp"] + counts["fn"])
    caption = (
        "ELMy-phase occupancy benchmark on identical reviewed 50 ms interior bins "
        f"within each panel; {100 * crowd_share:.0f}\\% of primary positive bins "
        "are crowd annotations. Primary panels use signal coverage; common panels also "
        "require DSM rows. ELM-O panels use BES chunks. Shot-grouped predictions "
        "and review-tuned thresholds exclude blind-cohort shots for learned detectors. "
        "Brackets show 95\\% shot-bootstrap intervals. $^{\\ddagger}$ marks "
        "supplemental source exposure; $^{\\dagger}$ marks recall $\\geq0.99$. "
        "Companions separate annotation modes and crowd/non-crowd performance."
    )
    return swap_tex.wrap_table("\n".join(lines), caption, "tab:elm-benchmark")


MODES = {
    "crowd_only": "Crowd only",
    "non_crowd_only": "Non-crowd only",
    "mixed": "Mixed crowd/non-crowd",
    "no_present": "No reviewed present span",
}


def annotation_table(ours) -> str:
    lines = []
    for tag in ("all119", "bes73"):
        res = ours["sets"][tag]
        scope = "All reviewed shots" if tag == "all119" else "BES, ELM-O chunks"
        lines += [
            r"\textbf{" + scope + r"}\par\smallskip",
            r"\begin{tabular}{lrrcc}",
            r"\toprule",
            r"Annotation mode & Shots & Bins & F1 & AUROC \\",
            r"\midrule",
        ]
        for mode, label in MODES.items():
            group = res["annotation_modes"][mode]
            result = group["methods"]["elm-ours"]
            f1, auc = (
                ("--", "--")
                if mode == "no_present"
                else tuple(swap_tex.metric_cell(result, m) for m in ("f1", "auroc"))
            )
            lines.append(
                f"{label} & {group['n_shots']} & {group['bins']:,} & {f1} & {auc} "
                + r"\\"
            )
        lines += [
            r"\bottomrule",
            r"\end{tabular}\par\smallskip",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Annotation mode & Precision & Recall & Absent-bin FP fraction \\",
            r"\midrule",
        ]
        for mode, label in MODES.items():
            result = res["annotation_modes"][mode]["methods"]["elm-ours"]
            if mode == "no_present":
                cells = [
                    "--",
                    "--",
                    swap_tex.metric_cell(result, "no_present_false_positive_fraction"),
                ]
            else:
                cells = [
                    swap_tex.metric_cell(result, m) for m in ("precision", "recall")
                ]
                cells.append("--")
            lines.append(label + " & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}\par\medskip"]
    caption = (
        "elm-ours occupancy performance by mutually exclusive annotation mode. "
        "Membership uses each shot's complete reviewed annotations; scoring uses "
        "the panel's covered interior bins. Mixed shots contain both crowd and "
        "individual-span annotations. No-present shots report the fraction of "
        "absent bins called present; positive-class metrics are undefined. "
        "Shot counts include members with no scored bins. "
        "Brackets show 95\\% shot-bootstrap intervals. These occupancy scores "
        "do not validate individual ELM onsets."
    )
    return swap_tex.wrap_table("\n".join(lines), caption, "tab:elm-annotation-modes")


def per_kind_table(ours) -> str:
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & Crowd-bin recall & Non-crowd bin recall & Non-crowd span recall \\",
    ]
    for tag in ("all119", "bes73"):
        res = ours["sets"][tag]
        scope = "All reviewed shots" if tag == "all119" else "BES, ELM-O chunks"
        lines += [r"\midrule", r"\multicolumn{4}{l}{" + scope + r"} \\"]
        for method, _ in (*swap_tex.ROWS, (swap_tex.ALWAYS, swap_tex.ALWAYS)):
            if method not in res["per_kind"]:
                continue
            kinds = res["per_kind"][method]
            cells = [
                swap_tex.cell(kinds[k]["point"], kinds[k]["ci95"])
                for k in (
                    "crowd_bin_recall",
                    "non_crowd_bin_recall",
                    "non_crowd_span_touch_recall",
                )
            ]
            lines.append(
                swap_tex.method_label(method) + " & " + " & ".join(cells) + r" \\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    caption = (
        "Occupancy recall separated by reviewed annotation kind. Crowd and "
        "non-crowd bin recall use positive interior bins; non-crowd span recall "
        "credits any detection touching a sufficiently covered individual span, "
        "after clipping detections to analyzed coverage. Brackets show 95\\% "
        "shot-bootstrap intervals. Span-touch recall does not measure onset timing."
    )
    return swap_tex.wrap_table("\n".join(lines), caption, "tab:elm-per-kind")


def own_target_table(dsm) -> str:
    target = dsm["own_target"]
    context = dsm["model_context"]
    lines = [
        r"\begin{tabular}{lrrc}",
        r"\toprule",
        r"Forecast horizon & Rows & Positive rows & AUROC \\",
        swap_tex.exposure_heading(4),
    ]
    for key, row in target["horizons"].items():
        horizon = re.search(r"\d+", key).group()
        ci = row.get("ci95", row.get("auroc_ci95"))
        if isinstance(ci, dict):
            ci = ci.get("auroc")
        lines.append(
            f"{horizon} ms$^{{\\ddagger}}$ & {row['cases'] + row['controls']:,} & "
            f"{row['cases']:,} & " + swap_tex.cell(row["auroc"], ci) + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    caption = (
        f"{context['refit_input_columns']}-input DSM refit checkpoint, selected "
        f"after epoch {context['refit_checkpoint_epochs']} of a "
        f"{context['refit_run_epochs']}-epoch run. Its survival target is an ELM within "
        "the forecast horizon, queried one ms later. Rows are the source "
        f"early-stopping validation set ({target['shots']} physical shots); "
        "these are model-selection results, not independent validation of the "
        "published 124-input model. Brackets show 95\\% physical-shot bootstrap "
        "intervals. This supplemental source-exposed evaluation is distinct from "
        "the reviewed occupancy benchmark."
    )
    return swap_tex.wrap_table("\n".join(lines), caption, "tab:elm-dsm-own-target")


def native_table(native) -> str:
    cover = native["coverage"]
    n_exact = sum(v["exact_native_export_available"] for v in cover.values())
    n_reconstructed = sum(v["reconstructable"] for v in cover.values())
    exposure = native["exposure"]
    lines = [
        r"\begin{tabular}{lrrrcc}",
        r"\toprule",
        r"Risk horizon & Shots & Rows & Positive rows & AUROC & AUPRC \\",
        swap_tex.exposure_heading(6),
        r"\multicolumn{6}{l}{"
        + f"Coverage: {len(cover)} reviewed; {n_exact} exact exports; "
        + f"{n_reconstructed} reconstructable"
        + r"} \\",
        r"\multicolumn{6}{l}{"
        + f"Reviewed source overlap: {len(exposure['reviewed_optimizer_train'])} "
        + "optimizer-trained; "
        + f"{len(exposure['reviewed_checkpoint_selection'])} checkpoint-selected"
        + r"} \\",
    ]
    panels = {
        "reviewed_exact_export": "Reviewed occupancy, exact native exports",
        "reviewed_reconstructed": "Reviewed occupancy, reconstructed inputs (sensitivity)",
        "own_target": "Source survival target, early-stopping validation",
    }
    for key, label in panels.items():
        panel = native.get(key, {})
        if not panel.get("horizons"):
            continue
        lines += [r"\midrule", r"\multicolumn{6}{l}{" + label + r"} \\"]
        for horizon_key, row in panel["horizons"].items():
            horizon = re.search(r"\d+", horizon_key).group()
            lines.append(
                f"{horizon} ms$^{{\\ddagger}}$ & {row['n_shots']} & "
                f"{row['rows']:,} & {row['cases']:,} & "
                + swap_tex.cell(row["auroc"], row["auroc_ci95"])
                + f" & {row['auprc']:.3f} "
                + r"\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    caption = (
        "Original 124-input DSM checkpoint on native 1 ms inputs, scored without "
        "operating thresholds. Coverage is reported separately for each panel. "
        "Exact source exports provide the faithful comparison; reconstruction is "
        "a sensitivity with source/serving smoothing disagreement. Reviewed panels "
        "score interval occupancy; the survival panel uses source early-stopping "
        "validation. AUROC brackets show 95\\% physical-shot bootstrap intervals. "
        "$^{\\ddagger}$ marks supplemental source exposure, not independent "
        "confirmatory evidence."
    )
    return swap_tex.wrap_table("\n".join(lines), caption, "tab:elm-dsm-native")


def appendix_note(strata, dsm, swap) -> str:
    note = r"""\paragraph{Shared ELM table protocol and limitations.}\label{app:elm-table-notes}
Tables score reviewed interval occupancy in 50 ms bins wholly inside a known
reviewed state and analyzed coverage. Primary panels use available D-alpha/density
coverage; BES panels use ELM-O chunks. Common DSM panels additionally require both
forecast and detection rows. Annotation-mode groups use complete shot annotations
and are mutually exclusive. Mixed shots contain crowd and non-crowd annotations;
no-present shots contribute absent-bin false-positive fractions. The reviewed
positive-bin distribution is predominantly crowd occupancy, so aggregate F1
principally measures ELMy-phase occupancy, not individual ELM detection.

Brackets are 95\% percentile intervals from 1,000 physical-shot bootstrap
resamples, shared across methods and references within a panel. Learned occupancy
detectors use shot-grouped out-of-fold predictions, with thresholds selected on
inner-validation reviewed bins. The fixed cohort test shots are excluded. ELM-O
hard calls use its selected threshold and rank scores use the saved nested sweep
on the same bins. The clock has hard calls only and seeded the review. elm-ours
calls a bin present from its mean score; ELM-O and the clock use any touching
detection. Detections are clipped to panel coverage before span-touch and alarm
counts. The always present rule is a degenerate recall control. $^{\dagger}$ marks
recall $\geq0.99$ while retaining numerical F1; companion precision/recall rows
make these operating points explicit.

$^{\ddagger}$ marks supplemental DSM source exposure. The survival refit
and source-initialized detection retain source fitting or selection exposure;
the historical exposed detection retains upstream preprocessing exposure.
Source normalization was computed before the source split and includes two
blind-cohort shots. For the 60-input refit, five reviewed swap shots were source-training shots. Subsets
outside source fitting and selection have only three shots, or two with BES;
their historical source normalization exposure remains. The clean detection
retrain starts from scratch and fits preprocessing on each outer training
partition, excluding both inner-validation and held-out shots.

DSM deployment adaptations use 60 of 124 source inputs, no D-alpha input,
mean-filled unavailable diagnostics and clipped standardized inputs. The survival
refit was trained on 1 ms rows and is served from 50 ms means; occupancy detection
heads train on reviewed 50 ms rows. CO2 is missing on 75 of the 119 reviewed shots.
Centered NBI smoothing adds 25 ms lookahead, so the survival score is offline,
not a causal forecast. Source-target results use early-stopping validation rows,
not an untouched test set. The 60-input refit's results do not constitute a native
full-input checkpoint comparison. Review annotations and elm-ours share D-alpha evidence, whereas the
60-input DSM adaptations have no D-alpha input; model gaps therefore also reflect input choice. The
individual-onset deliverable remains incomplete without independent adjudication.

Legacy onset bins mark nonzero 1 ms onset counts in each 50 ms bin. Occupancy
sensitivities merge positive intervals across covered gaps of at most
$\tau=100,200,300$ ms and never bridge missing coverage. $|M|$ counts reviewed
present/legacy absent bins and $|P|$ the converse; these are reference
disagreements, not independently verified omitted ELMs. All-covered audits use
at least 25 ms majority occupancy and separate uncertain or mixed review states.
Strict swaps reuse identical reviewed interior bins, predictions and thresholds.
Because learned F1 thresholds were tuned against review labels, only AUROC is
compared across references. Eight overlap shots and the tiny source-unexposed
subsets leave the evidence inconclusive; point rankings cannot establish or rule
out an AE-style ranking-reversal result.

The native full-input checkpoint is evaluated separately on its exact source
exports and on reconstructed 124-input, 1 ms rows. These panels remain historical
and source-exposed. Reconstruction uses available original diagnostics without
mean filling or clipping; finite rows inside the source normalization/filter
domain are retained. Native centered NBI smoothing adds 50 ms lookahead.
Source smoothing concatenates filtered phase rows, whereas reconstruction smooths
within each shot before filtering, so reconstructed inputs are a sensitivity,
not a claim of exact native-export equivalence. The agreement audit and explicit
per-panel coverage must accompany that interpretation.
"""
    audit = strata["checkpoint_selection_audit"]
    n_folds = len(audit["folds"])
    note += (
        "\nCheckpoint selection is noisy: "
        f"{audit['selected_during_warmup']} of {n_folds} selected epochs fall "
        "inside warm-up, and a three-epoch smoothed validation criterion chooses "
        f"a different candidate in {audit['smoothed_candidate_differs']} of "
        f"{n_folds} folds. Inner-validation thresholds span "
        f"{audit['threshold_min']:.3f}--{audit['threshold_max']:.3f}. "
        "The smoothed candidates have not been rescored; the reported numbers "
        "retain the original saved checkpoints.\n"
    )
    context = dsm["model_context"]
    note += (
        f"\nThe survival-refit checkpoint was selected after epoch "
        f"{context['refit_checkpoint_epochs']} from a "
        f"{context['refit_run_epochs']}-epoch run; the one-epoch description refers "
        "to the selected checkpoint, not the run length.\n"
    )
    if swap_tex.bes_point_order_crosses(swap):
        note += (
            "\nThe main AUROC leader is unchanged across reference conversions, "
            "but the clean DSM detector and ELM-O exchange point-AUROC order "
            "in the BES occupancy sensitivities. This is a point-order crossing "
            "on seven overlapping shots, with wide intervals, not evidence of "
            "a population-level ranking reversal.\n"
        )
    return note


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args(argv)
    out = args.out_dir or Paths.from_env().root / "round4" / "elm"
    out.mkdir(parents=True, exist_ok=True)
    files = {key: OUTPUTS / key / "evaluation.json" for key in ("ours", "dsm", "swap")}
    files["strata"] = OUTPUTS / "ours" / "annotation_strata.json"
    native_path = OUTPUTS / "dsm" / "native_evaluation.json"
    if native_path.exists():
        files["native"] = native_path
    records = {key: json.loads(path.read_text()) for key, path in files.items()}
    root_tables = {
        "table_elm_benchmark.tex": benchmark_table(records["ours"], records["dsm"]),
        "table_elm_annotation_modes.tex": annotation_table(records["ours"]),
        "table_elm_per_kind.tex": per_kind_table(records["ours"]),
        "table_elm_dsm_own_target.tex": own_target_table(records["dsm"]),
    }
    if "native" in records:
        root_tables["table_elm_dsm_native.tex"] = native_table(records["native"])
    for name, content in root_tables.items():
        (out / name).write_text(content)
        (OUTPUTS / name).write_text(content)
    note = appendix_note(records["strata"], records["dsm"], records["swap"])
    (out / "elm_table_notes.tex").write_text(note)
    (OUTPUTS / "elm_table_notes.tex").write_text(note)
    swap_tex.write(records["swap"], out)
    swap_output = OUTPUTS / "swap"
    swap_output.mkdir(parents=True, exist_ok=True)
    tables = sorted(out.glob("table_elm_*.tex"))
    for path in tables:
        if path.name not in root_tables:
            (swap_output / path.name).write_bytes(path.read_bytes())
    caption_words = {
        p.name: [
            len(c.split())
            for c in re.findall(
                r"\\caption\{(.*?)\}\n\\label", p.read_text(), re.DOTALL
            )
        ]
        for p in tables
    }
    if any(n > 80 for counts in caption_words.values() for n in counts):
        raise ValueError(f"A table caption exceeds 80 words: {caption_words}")
    manifest = {
        "git": git_sha(),
        "sources": {
            key: {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for key, path in files.items()
        },
        "tables": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tables},
        "caption_word_counts": caption_words,
        "shared_notes": {
            "path": str(OUTPUTS / "elm_table_notes.tex"),
            "sha256": hashlib.sha256(note.encode()).hexdigest(),
        },
        "benchmark_json_paths": [
            f"{key}/evaluation.json:sets.{tag}.methods"
            for key in ("ours", "dsm")
            for tag in ("all119", "bes73")
        ],
        "stratified_json_paths": [
            f"ours/evaluation.json:sets.{tag}.{part}"
            for tag in ("all119", "bes73")
            for part in ("annotation_modes", "per_kind")
        ],
        "own_target_json_path": "dsm/evaluation.json:own_target",
        "native_json_path": "dsm/native_evaluation.json"
        if "native" in records
        else None,
        "swap_json_paths": [
            "swap/evaluation.json:swap.overlap",
            "swap/evaluation.json:swap.overlap_bes",
            "swap/evaluation.json:interval_audit",
        ],
    }
    for target in (out, OUTPUTS):
        (target / "tables.json").write_text(json.dumps(manifest, indent=1))
    print("tables", out, sorted(manifest["tables"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
