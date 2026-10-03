#!/usr/bin/env python
"""Write paper ELM tables and their shared appendix note from evaluation JSONs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

from labeler.config import Paths, git_sha
from labeler.elm import swap_tex

REPO = Path(__file__).resolve().parents[2]
OUTPUTS = REPO / "outputs" / "labeler" / "elm"


def benchmark_table(ours, dsm, feature=None) -> str:
    """The six core rows on one common coverage set in each of two panels."""
    lines = [r"\begin{tabular}{lccc}", r"\toprule", r"Method & AUROC & AUPRC & F1 \\"]
    rows = (
        ("elm-ours", "elm-ours"),
        ("elm-elmo", "elm-elmo"),
        ("elm-dsm-detect", "elm-dsm (detection)"),
        ("elm-clock", "elm-clock"),
        ("always present", "always-present"),
        ("elm-feature-only", "feature-only"),
    )
    for tag in ("all119", "bes73"):
        res = dsm["sets"][tag]
        scope = "All reviewed" if tag == "all119" else "BES subset"
        lines += [
            r"\midrule",
            r"\multicolumn{4}{l}{"
            + f"{scope}: {res['n_shots']} shots/{res['bins']:,} common bins"
            + r"} \\",
        ]
        for key, label in rows:
            if key == "elm-feature-only":
                result = (
                    feature["sets"]["common"][tag]["methods"][key] if feature else None
                )
            else:
                result = res["methods"].get(key)
            cells = (
                [swap_tex.metric_cell(result, m) for m in ("auroc", "auprc", "f1")]
                if result
                else ["--"] * 3
            )
            lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    caption = (
        "Reviewed 50 ms occupancy, 97\\% crowd positives; brackets: eligible "
        "shot-bootstrap intervals. Inputs: ours, FS02--04/fast density; ELM-O, "
        "BES/FS/density (BES only); DSM, 60 diagnostics including PCPHD02/03 photodiodes "
        "on all 119 shots and DENV2F/3F on 115 each (four each mean-filled); clock, "
        "D-alpha/phase gate; "
        "always-present, none; feature-only, log D-alpha max-minus-median. "
        "Clock boundary identity within 1 ms: 56\\% starts, 44\\% ends, "
        "33\\% both. Density calibration remains unresolved."
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


def supplemental_table(dsm) -> str:
    """Keep historical model comparisons outside the six-row main benchmark."""
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Supplemental method & AUROC & AUPRC & F1 \\",
    ]
    for tag in ("all119", "bes73"):
        res = dsm["sets"][tag]
        lines += [
            r"\midrule",
            r"\multicolumn{4}{l}{"
            + f"{res['n_shots']} shots/{res['bins']:,} common bins"
            + r"} \\",
        ]
        for key in ("elm-dsm", "elm-dsm-detect-exposed", "elm-dsm-detect-init"):
            result = res["methods"][key]
            cells = [swap_tex.metric_cell(result, m) for m in ("auroc", "auprc", "f1")]
            lines.append(
                swap_tex.method_label(key) + " & " + " & ".join(cells) + r" \\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return swap_tex.wrap_table(
        "\n".join(lines),
        "Historical DSM variants with source fitting, normalization or checkpoint "
        "selection exposure. Inputs retain mean-filled photodiodes and incomplete "
        "slow CO2, unlike the revised isolated detector. Brackets show eligible "
        "shot-bootstrap intervals; these rows are developmental diagnostics.",
        "tab:elm-supplemental",
    )


def smith_table(record) -> str:
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & AUROC & AUPRC & Occupancy F1 \\",
        r"\midrule",
    ]
    for method, row in record["methods"].items():
        res = row["occupancy_1ms"]
        cells = [swap_tex.metric_cell(res, m) for m in ("auroc", "auprc", "f1")]
        lines.append(method + " & " + " & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule\end{tabular}\par\smallskip",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Method & Tolerance & Precision & Recall & F1 & Median $|e|$ (ms) \\",
        r"\midrule",
    ]
    for method, row in record["methods"].items():
        for tolerance, res in row["events"].items():
            cells = [
                swap_tex.metric_cell(res, m) for m in ("precision", "recall", "f1")
            ]
            error = res["timing_error_ms"]["median_absolute"]
            timing = "--" if error is None else f"{error:.3f}"
            lines.append(
                method
                + rf" & $\pm${tolerance} ms & "
                + " & ".join(cells)
                + " & "
                + timing
                + r" \\",
            )
    lines += [r"\bottomrule\end{tabular}"]
    caption = (
        "Independent Smith hand-labelled windows: frozen review ensemble "
        "(elm-ours), Smith-only grouped-CV onset head, and fixed-setting ELM-O. "
        "Whole 1 ms occupancy cells and one-to-one onset matching have different "
        "targets. Timing errors describe matched events. Brackets show eligible "
        "shot-bootstrap intervals; ELM-O's 0.997/0.980 window-overlap P/R is "
        "distinct from onset matching."
    )
    return swap_tex.wrap_table("\n".join(lines), caption, "tab:elm-smith")


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
        "reviewed_reconstructed": "Any reviewed present within horizon, reconstructed inputs",
        "reviewed_reconstructed_onsets": "Non-crowd starts within horizon (per-ELM shots)",
        "own_target": "Source survival target, early-stopping validation",
    }
    for key, label in panels.items():
        panel = (
            {"horizons": native["reviewed_reconstructed"]["non_crowd_start_horizons"]}
            if key == "reviewed_reconstructed_onsets"
            else native.get(key, {})
        )
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
        "Reconstruction is a sensitivity with source/serving smoothing disagreement. "
        "Reviewed panels use forward targets; the survival panel uses source early-stopping "
        "validation. AUROC brackets show 95\\% physical-shot bootstrap intervals. "
        "$^{\\ddagger}$ marks supplemental source exposure, not independent "
        "confirmatory evidence."
    )
    return swap_tex.wrap_table("\n".join(lines), caption, "tab:elm-dsm-native")


def appendix_note(strata, dsm, swap) -> str:
    return r"""\paragraph{ELM evaluation protocol.}\label{app:elm-table-notes}
Reviewed occupancy uses 50 ms bins wholly inside known spans and signal coverage;
the main panels additionally require DSM rows. Crowd annotations supply 97\% of
positive bins. Review began from the clock: within 1 ms, 56\% of crowd starts,
44\% of ends and 33\% of both edges match its boundaries. This development
reference is dependent on D-alpha evidence; span starts are not verified onsets.

The DSM and legacy onset table were built on WPQH phases with breakthrough-ELM
targets; Finding 1 and low DSM AUROCs partly reflect definition and domain shift
(192721: 1 legacy bin versus 17 non-crowd review spans).

Five shot-grouped folds exclude blind-cohort shots; checkpoint and threshold
selection uses inner-validation shots. The primary occupancy model remains the
frozen original recipe. A trailing-three-epoch selection after warm-up is reported
as a sensitivity, not selected on its held-out performance. Feature-only uses one
within-bin FS02--04 log D-alpha max-minus-median feature and L2 logistic regression
on identical folds. The revised isolated DSM detector has training-only
normalization and random initialization; real photodiodes cover all 119 shots,
with no FS substitutes. DENV2F/3F means fill v2/v3 on 115 shots per chord;
four rejected shots per chord are mean-filled. Numerical
scales are checked against paired slow CO2 and training columns, but physical
calibration remains unresolved. Legacy survival and source-exposed fits stay
supplemental; 50 ms means and centered NBI lookahead differ from native training.
Native risks are scored against reviewed presence or per-ELM starts in $(t,t+h]$.
The four exact-export source-exposed shots remain in the appendix JSON record only.

Eligible intervals use 1,000 physical-shot bootstrap draws. Every metric records
finite and undefined draws; fewer than five positive-bearing shots are descriptive
only and have no population 95\% interval. $^{\ddagger}$ marks source exposure;
$^{\dagger}$ marks recall at least 0.99. Complete numeric precision and recall,
input availability and training exposure remain in the evaluation JSON records.

Smith's independent hand-labelled windows test the frozen review ensemble and
shot-grouped Smith-trained onset head separately. Window occupancy uses 1 ms
cells, so it is not comparable to the crowd-dominated 50 ms review target.
Event metrics use one-to-one onset matching at 2 and 5 ms and report timing errors;
ELM-O uses its published fixed setting. Frozen occupancy transfer fails against
Smith event regions, and the frozen auxiliary onset output is not delivered.
The Smith-trained onset head succeeds on selected windows and remains an
experimental CV trace; catalog physical-onset output is withheld. Run-day overlap
is reported separately from shot overlap.

Finding 1 uses all-covered known 50 ms cells with at least 25 ms reviewed present;
unknown time stays unknown. Finding 2 includes both strict interior and known
all-covered bins with identical saved predictions and thresholds. Only AUROC
compares references, because F1 thresholds were review-tuned. Covered-gap merges
at 100/200/300 ms are definition sensitivities; missing coverage is never bridged.
The eight-shot swap is inconclusive and cannot establish population ranking reversal.
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args(argv)
    out = args.out_dir or Paths.from_env().root / "round4" / "elm"
    out.mkdir(parents=True, exist_ok=True)
    files = {key: OUTPUTS / key / "evaluation.json" for key in ("ours", "dsm", "swap")}
    files["feature"] = OUTPUTS / "ours" / "feature_only.json"
    files["strata"] = OUTPUTS / "ours" / "annotation_strata.json"
    native_path = OUTPUTS / "dsm" / "native_evaluation.json"
    if native_path.exists():
        files["native"] = native_path
    smith_path = OUTPUTS / "smith/evaluation.json"
    if smith_path.exists():
        files["smith"] = smith_path
    records = {key: json.loads(path.read_text()) for key, path in files.items()}
    # Retain the previous generated tables outside git, then publish one current set.
    archive = out / "archive" / "fix4_tables"
    for path in sorted(OUTPUTS.rglob("*.tex")):
        relative = path.relative_to(OUTPUTS)
        destination = archive / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(path, destination)
        path.unlink()
    for path in out.glob("table_elm_*.tex"):
        path.unlink()
    root_tables = {
        "table_elm_benchmark.tex": benchmark_table(
            records["ours"], records["dsm"], records["feature"]
        ),
        "table_elm_annotation_modes.tex": annotation_table(records["ours"]),
        "table_elm_per_kind.tex": per_kind_table(records["ours"]),
        "table_elm_dsm_own_target.tex": own_target_table(records["dsm"]),
        "table_elm_supplemental.tex": supplemental_table(records["dsm"]),
    }
    if "native" in records:
        root_tables["table_elm_dsm_native.tex"] = native_table(records["native"])
    if "smith" in records:
        root_tables["table_elm_smith.tex"] = smith_table(records["smith"])
    for name, content in root_tables.items():
        (out / name).write_text(content)
        target = OUTPUTS if name == "table_elm_benchmark.tex" else OUTPUTS / "appendix"
        target.mkdir(exist_ok=True)
        (target / name).write_text(content)
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
    if any(n > 130 for counts in caption_words.values() for n in counts):
        raise ValueError(f"A table caption exceeds 130 words: {caption_words}")
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
