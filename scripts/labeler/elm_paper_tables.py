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
from labeler.elm import facts as F
from labeler.elm import swap_tex

REPO = Path(__file__).resolve().parents[2]
OUTPUTS = REPO / "outputs" / "labeler" / "elm"


LOWER_BOUND = (
    "These DSM rows are lower bounds on DSM detection skill under our recipe, "
    "not the best achievable DSM performance."
)
CAPTION_CAPS = {
    "table_elm_benchmark.tex": 280,
    "table_elm_common_control.tex": 240,
    "table_elm_swap.tex": 150,
}
DEFAULT_CAP = 130


delta_text = F.delta_text


def seed_sentence(fx: dict, only: str | None = None) -> str:
    """Three further DSM detection seeds with post-warm-up selection, from the JSON."""
    seeds = fx.get("dsm_baselines")
    if not seeds:
        return ""
    bits = []
    for key, label in (
        ("reduced", r"the 60-input $1\times128$ adaptation"),
        ("native", "the native 124-input refit"),
    ):
        if only not in (None, key):
            continue
        row = seeds[key]
        bits.append(
            f"{row['auroc']['mean']:.3f} "
            f"({row['auroc']['min']:.3f}--{row['auroc']['max']:.3f}) for {label} "
            f"({row['n_shots']} shots)"
        )
    count = swap_tex.count_word(len(seeds["seeds"])).capitalize()
    return (
        f"{count} further seeds with post-warm-up selection "
        "(DSM repeats table) give mean (range) AUROC "
        f"{' and '.join(bits)}. "
    )


def metric_row(label, result, keys, blank=()):
    """A table row; `blank` names metrics that the scope does not define."""
    cells = [
        "--" if m in blank or not result else swap_tex.metric_cell(result, m)
        for m in keys
    ]
    return label + " & " + " & ".join(cells) + r" \\"


PRIMARY_KEYS = (
    "auroc",
    "auprc",
    "f1",
    "false_alarm_bin_rate",
    "absent_span_alarm_rate",
)
COMMON_KEYS = ("auroc", "auprc", "f1")


def main_caption(fx: dict) -> str:
    """The primary-benchmark caption; every figure is read from the JSON records."""
    folds = fx["ours_folds"]
    early = min(folds, key=lambda r: r["epoch"])
    thr = [r["threshold"] for r in folds]
    auroc = fx["seed_range"]["all119"]["auroc"]
    quiet = fx["non_crowd_only"]
    prec = F.fmt(*quiet["precision"])
    share = fx["clock_share"]
    text = (
        "Primary reviewed-occupancy benchmark: developmental shot-grouped CV on "
        f"the {fx['shots']['all119']} reviewed shots; a frozen-model score on the "
        "blind split awaits review of those shots. The target is ELMy-period "
        "occupancy, not onsets: "
        f"{100 * fx['crowd_share']:.0f}\\% of positive bins come from crowd spans "
        "and the label set has no onset trace. Scored 50 ms bins lie wholly inside "
        "reviewed spans, so bins straddling a boundary are dropped and the task is "
        "easier than whole-shot detection "
        f"({fx['bins']['all119']:,} all119 bins; {fx['bins']['bes73']:,} bes73). "
        "Labels come from review seeded by the D-alpha clock "
        f"({100 * share['start']:.0f}\\% of crowd starts, {100 * share['end']:.0f}\\% "
        f"of ends and {100 * share['both']:.0f}\\% of both lie within 1 ms of clock "
        "boundaries), which favours D-alpha-input models over ELM-O. "
        "Fold thresholds maximise inner-validation F1; elm-ours calls a bin on its "
        "mean probability, ELM-O and the clock on any touching detected span. "
        "FPR is the fraction of absent bins called present; alarm is the fraction "
        "of absent spans touched. Paired elm-ours minus ELM-O on bes73: "
        f"{delta_text(fx['paired_primary'])}; all include zero and equivalence is "
        f"untested. Fold {early['fold']} (zero-based) selected epoch "
        f"{early['epoch']} of {fx['epochs']} (thresholds "
        f"{min(thr):.3f}--{max(thr):.3f}); four training "
        f"seeds give all119 AUROC {auroc['min']:.3f}--{auroc['max']:.3f}. On "
        f"non-crowd-only shots ({quiet['shots']} shots/{quiet['bins']:,} bins) "
        f"elm-ours precision is {prec}."
    )
    moved = fx.get("moved")
    if moved:
        text += (
            " The last block keeps bes73 spans whose boundaries both moved more "
            f"than 1 ms from the clock ({moved['present_spans']} of "
            f"{moved['all_present_spans']} present and {moved['absent_spans']} "
            f"absent spans; {moved['bins']:,} bins); with so few present spans its "
            "intervals are wide and the comparison inconclusive."
        )
    text += (
        r" $^{\dagger}$ marks recall $\geq0.99$. Brackets: 95\% shot-bootstrap "
        "intervals."
    )
    return text


def common_caption(fx: dict) -> str:
    """The common-bin control caption, with the DSM baselines' status."""
    folds = fx["dsm_folds"]
    epochs = ", ".join(str(r["epoch"]) for r in folds)
    thr = [r["threshold"] for r in folds]
    text = (
        "Secondary common-bin control, requiring valid DSM rows for every method "
        f"({fx['common_bins']['all119']:,} all119; {fx['common_bins']['bes73']:,} "
        r"bes73 bins). The elm-dsm (60-input $1\times128$) row is a detection "
        "adaptation on 50 ms input means, not the native architecture; its "
        f"reported fit is fragile (selected epochs {epochs}; thresholds "
        f"{min(thr):.3f}--{max(thr):.3f}). " + seed_sentence(fx) + LOWER_BOUND + " "
        "Paired elm-ours minus elm-dsm, all119: "
        f"{delta_text(fx['paired_common_dsm']['all119'])}; bes73: "
        f"{delta_text(fx['paired_common_dsm']['bes73'])}. Paired elm-ours minus "
        f"ELM-O on bes73: {delta_text(fx['paired_common_elmo'])}; the latter "
        "intervals include zero and equivalence is untested. "
        r"Brackets: 95\% shot-bootstrap intervals."
    )
    return text


def benchmark_table(ours, dsm, feature=None, fx=None, *, common=False) -> str:
    """Primary occupancy panels, or the secondary DSM common-bin control."""
    keys = COMMON_KEYS if common else PRIMARY_KEYS
    head = (
        r"Method & AUROC & AUPRC & F1 \\"
        if common
        else r"Method & AUROC & AUPRC & F1 & Absent-bin FPR & Absent-span alarm \\"
    )
    lines = [
        r"\begin{tabular}{l" + "c" * len(keys) + "}",
        r"\toprule",
        head,
    ]
    rows = (
        ("elm-ours", "elm-ours"),
        ("elm-elmo", "elm-elmo"),
        ("elm-dsm-detect", r"elm-dsm (60-input $1\times128$)"),
        ("elm-clock", "elm-clock"),
        ("always present", "always-present"),
        ("elm-feature-only", "elm-feature"),
    )
    width = len(keys) + 1
    for tag in ("all119", "bes73"):
        res = (dsm if common else ours)["sets"][tag]
        scope = "All reviewed" if tag == "all119" else "BES subset"
        lines += [
            r"\midrule",
            rf"\multicolumn{{{width}}}{{l}}{{"
            + f"{scope}: {res['n_shots']} shots/{res['bins']:,} "
            + ("common bins" if common else "primary bins")
            + r"} \\",
        ]
        for key, label in rows:
            if key == "elm-dsm-detect" and not common:
                continue
            if key == "elm-feature-only":
                result = (
                    feature["sets"]["common" if common else "primary"][tag]["methods"][
                        key
                    ]
                    if feature
                    else None
                )
            else:
                result = res["methods"].get(key)
            if result is None:
                continue
            lines.append(metric_row(label, result, keys))
    moved = None if common or not fx else fx.get("moved")
    if moved:
        lines += [
            r"\midrule",
            rf"\multicolumn{{{width}}}{{l}}{{BES subset, boundaries moved "
            r"$>1$ ms from the clock: "
            + f"{moved['present_spans']} present and {moved['absent_spans']} absent "
            + f"spans/{moved['bins']:,} bins"
            + r"} \\",
        ]
        for key, label in rows[:2] + rows[3:4]:
            result = moved["methods"].get(key)
            if result is not None:
                lines.append(
                    metric_row(label, result, keys, blank=("absent_span_alarm_rate",))
                )
    lines += [r"\bottomrule", r"\end{tabular}"]
    caption = common_caption(fx) if common else main_caption(fx)
    label = "tab:elm-common-control" if common else "tab:elm-benchmark"
    return swap_tex.wrap_table("\n".join(lines), caption, label)


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
        scope = "All reviewed shots" if tag == "all119" else "BES, elm-elmo chunks"
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
        scope = "All reviewed shots" if tag == "all119" else "BES, elm-elmo chunks"
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
        "intervals. These shots contributed to source model selection; this "
        "supplemental evaluation is distinct from "
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
        for key, label in (
            ("elm-dsm", "elm-dsm survival refit"),
            ("elm-dsm-detect-exposed", "elm-dsm (source statistics, detection)"),
            (
                "elm-dsm-detect-init",
                "elm-dsm (source weights and statistics, detection)",
            ),
        ):
            result = res["methods"][key]
            cells = [swap_tex.metric_cell(result, m) for m in ("auroc", "auprc", "f1")]
            lines.append(label + r"$^{\ddagger}$ & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return swap_tex.wrap_table(
        "\n".join(lines),
        "Historical DSM variants with source fitting, normalization or checkpoint "
        "selection on these shots. Inputs retain mean-filled photodiodes and incomplete "
        "slow CO2, unlike the revised isolated detector. Brackets show eligible "
        "shot-bootstrap intervals; these rows are developmental diagnostics.",
        "tab:elm-supplemental",
    )


def smith_table(record) -> str:
    day_audit = record["onset_run_day_audit"]
    onset_audit = record["onset_window_audit"]
    head = record["methods"]["elm-ours-onset"]["events"]["2"]
    precision = head["point"]["precision"]
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & Window AUROC & Window AUPRC & Window occupancy F1 \\",
        r"\midrule",
    ]
    for method, row in record["methods"].items():
        if method.startswith("elm-ours") and method != "elm-ours-onset":
            continue
        res = row["occupancy_1ms"]
        cells = [
            swap_tex.cell(res["point"][m], res.get("ci95", {}).get(m), ci_digits=3)
            for m in ("auroc", "auprc", "f1")
        ]
        lines.append(method + " & " + " & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule\end{tabular}\par\smallskip",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Method & Tolerance & Recall & Median $|e|$ (ms) \\",
        r"\midrule",
    ]
    for method, row in record["methods"].items():
        if method.startswith("elm-ours") and method != "elm-ours-onset":
            continue
        res = row["events"]["2"]
        recall = swap_tex.cell(
            res["point"]["recall"], res.get("ci95", {}).get("recall"), ci_digits=3
        )
        error = res["timing_error_ms"]["median_absolute"]
        timing = "--" if error is None else f"{error:.3f}"
        lines.append(rf"{method} & $\pm$2 ms & {recall} & {timing} \\")
    lines += [r"\bottomrule\end{tabular}"]
    caption = (
        "Smith selected windows: frozen review-occupancy rows are omitted "
        "for target mismatch (50 ms occupancy versus approximately 8.5 ms windows). "
        "The experimental elm-ours-onset head is developmental "
        f"shot CV with {day_audit['run_days_spanning_multiple_folds']} of "
        f"{day_audit['run_days']} Smith run days crossing folds; overlap with "
        "ELM-O's historical tuning events is unknown because event membership "
        "was not retained. Every metric is conditional on selected windows; "
        "continuous-discharge precision and F1 are unavailable for every method. "
        "Event precision and F1 are not shown: the 10 ms minimum peak "
        f"separation fixes the head's in-window precision at {precision:.3f}, and "
        f"{onset_audit['out_of_window_firings']:,} additional out-of-window "
        f"firings lack negative truth. Head recall is {head['point']['recall']:.3f}; "
        f"matched errors are within {onset_audit['max_absolute_error_ms']:.2f} ms "
        f"and {100 * onset_audit['correct_1ms_cell_fraction']:.0f}\\% are in the "
        "correct 1 ms cell. Only $\\pm2$ ms matching is shown; the head is "
        "withheld from catalog claims. Brackets: shot-bootstrap intervals."
    )
    return swap_tex.wrap_table("\n".join(lines), caption, "tab:elm-smith")


def native_table(native) -> str:
    cover = native["coverage"]
    n_exact = sum(v["exact_native_export_available"] for v in cover.values())
    n_reconstructed = sum(v["reconstructable"] for v in cover.values())
    exposure = native["exposure"]
    scored = native["reviewed_exact_export"]["horizons"]["h50ms"]["shots"]
    n_scored = len(scored)
    unscored = sorted(
        int(shot)
        for shot, v in cover.items()
        if v["exact_native_export_available"] and int(shot) not in set(scored)
    )
    lines = [
        r"\begin{tabular}{lrrrcc}",
        r"\toprule",
        r"Risk horizon & Shots & Rows & Positive rows & AUROC & AUPRC \\",
        swap_tex.exposure_heading(6),
        r"\multicolumn{6}{l}{"
        + f"Coverage: {len(cover)} reviewed; {n_exact} available exact exports "
        + f"({native['reviewed_exact_export']['horizons']['h50ms']['n_shots']} scored)"
        + r"} \\",
        r"\multicolumn{6}{l}{"
        + f"{n_reconstructed} shots have inputs recoverable from stored signals"
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
        f"{n_exact} exact exports are available, but {len(unscored)} "
        f"({', '.join(map(str, unscored))}) has no scored reviewed overlap, leaving "
        f"{n_scored} shots in the exact-export JSON. "
        "$^{\\ddagger}$ marks shots used in source training, normalization or "
        "checkpoint selection; they do not provide independent confirmation."
    )
    return swap_tex.wrap_table("\n".join(lines), caption, "tab:elm-dsm-native")


def native_detection_table(record, fx=None) -> str:
    lines = [r"\begin{tabular}{lccc}", r"\toprule", r"Method & AUROC & AUPRC & F1 \\"]
    for tag in ("all119", "bes73"):
        panel = record["sets"][tag]
        lines += [
            r"\midrule",
            r"\multicolumn{4}{l}{"
            + f"{tag}: {panel['n_shots']} shots/{panel['bins']:,} matched bins"
            + r"} \\",
        ]
        for key, label in (
            ("elm-ours", "elm-ours"),
            ("elm-dsm-detect", r"elm-dsm (60-input $1\times128$)"),
            ("elm-dsm-native-detect", "elm-dsm (124-input [100,1000])"),
            ("elm-elmo", "ELM-O"),
        ):
            result = panel["methods"].get(key)
            if result:
                cells = [
                    swap_tex.metric_cell(result, m) for m in ("auroc", "auprc", "f1")
                ]
                lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    gate_text = record["gate"].replace(">=", r"$\geq$")
    epochs = record["recipe"]["epochs"]
    caption = (
        f"Secondary native DSM detection comparison. The 1 ms audit finds "
        f"{gate_text}; "
        f"{record['sets']['all119']['n_shots']} reviewed shots have complete-input "
        "scored bins. Both DSM rows are occupancy detection "
        r"refits: 60-input $1\times128$ adaptation on 50 ms means, and native "
        "124-input [100,1000] architecture on 1 ms means. All methods use identical "
        "supported shots/bins. "
        "Inner-validation AUPRC selects checkpoints and F1 selects thresholds. "
        r"Brackets: 95\% shot-bootstrap intervals. These are developmental results "
        f"from a fixed {epochs}-epoch refit, with reconstructed native inputs and "
        "within-shot "
        "NBI smoothing. " + (seed_sentence(fx, "native") if fx else "") + LOWER_BOUND
    )
    return swap_tex.wrap_table("\n".join(lines), caption, "tab:elm-native-detection")


def dsm_repeats_table(fx: dict) -> str:
    """The reported DSM detection fits beside three post-warm-up repeats."""
    seeds = fx["dsm_baselines"]
    lines = [r"\begin{tabular}{lccc}", r"\toprule", r"Fit & AUROC & AUPRC & F1 \\"]
    for key, label in (
        ("reduced", r"elm-dsm (60-input $1\times128$)"),
        ("native", "elm-dsm (124-input [100,1000])"),
    ):
        row = seeds[key]
        lines += [
            r"\midrule",
            r"\multicolumn{4}{l}{"
            + f"{label}: {row['n_shots']} shots/{row['bins']:,} bins"
            + r"} \\",
        ]
        reported = row["reported"]["all119"]
        lines.append(
            "reported fit, raw selection & "
            + " & ".join(f"{reported[m]:.3f}" for m in F.METRIC_KEYS)
            + r" \\"
        )
        cells = []
        for m in F.METRIC_KEYS:
            r = row["all119"][m]
            cells.append(f"{r['mean']:.3f} ({r['min']:.3f}--{r['max']:.3f})")
        lines.append(
            "post-warm-up repeats, mean (range) & " + " & ".join(cells) + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    reduced, native = seeds["reduced"], seeds["native"]
    caption = (
        "DSM detection baselines retrained with post-warm-up checkpoint selection. "
        "Each repeat refits the detector with the same outer folds and recipe "
        f"for {swap_tex.count_word(len(seeds['seeds']))} new seeds, choosing the "
        "epoch that ends the best three-epoch mean inner-validation AUPRC window "
        "lying wholly at or "
        f"after the warm-up (epoch {reduced['warmup_epochs']} of "
        f"{reduced['total_epochs']} for the 60-input adaptation, "
        f"{native['warmup_epochs']} of {native['total_epochs']} for the native "
        "refit); the reported fit takes the raw best epoch, which can fall in "
        "the first epochs. Selected epochs span "
        f"{reduced['epochs'][0]}--{reduced['epochs'][1]} (60-input) and "
        f"{native['epochs'][0]}--{native['epochs'][1]} (native); thresholds span "
        f"{reduced['thresholds'][0]:.4f}--{reduced['thresholds'][1]:.3f} and "
        f"{native['thresholds'][0]:.4f}--{native['thresholds'][1]:.3f}, so the "
        "native operating point is erratic. Ranges are over seeds, not "
        "intervals. " + LOWER_BOUND
    )
    return swap_tex.wrap_table("\n".join(lines), caption, "tab:elm-dsm-repeats")


def rejection_table(record) -> str:
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Chord & Screen shots & Screen bins & Clipping shots & Clipping bins \\",
    ]
    for name, row in record["totals"].items():
        lines.append(
            name
            + " & "
            + " & ".join(
                str(row[k])
                for k in (
                    "screen_shots",
                    "screen_bins",
                    "clipping_shots",
                    "clipping_bins",
                )
            )
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    delta = record["metric_change_disabled_minus_enabled"]
    caption = (
        f"Per-chord preprocessing on all119 ({record['n_shots']} shots/"
        f"{record['bins']:,} primary bins). "
        "The density screen zeroes chords with median absolute native magnitude "
        "$>10^{16}$; FS02--04 have no such screen. FS clipping floors native "
        "values at $10^{12}$ before the logarithm. Density divided by $10^{14}$ "
        "is clipped to $[-3,12]$; ten times its high-pass to $[-10,10]$. "
        "Clipping counts exclude screened chords and mark any affected cell; "
        "clipping discards no bins. Disabling only the screen, using frozen "
        "weights/thresholds and raw chords under the same clipping, changes "
        f"AUROC {delta['auroc']:+.5f}, AUPRC {delta['auprc']:+.5f}, "
        f"F1 {delta['f1']:+.5f}; {len(record['affected_shots'])} shots change "
        "inputs. This does not validate "
        "physical calibration or diagnostic failure."
    )
    return swap_tex.wrap_table("\n".join(lines), caption, "tab:elm-rejection")


def pct(value: float) -> str:
    return f"{100 * value:.0f}"


def appendix_note(fx: dict) -> str:
    """The shared appendix note: definitions, panels, thresholds and limits.

    One page at most; every figure is read from the JSON records through `facts`.
    """
    offsets, share = fx["offsets"], fx["clock_share"]
    alarm = fx["alarm"]
    native = fx["native_detection"]
    review_days = fx["review_run_days"]
    paragraphs = [
        r"\paragraph{ELM evaluation notes.}\label{app:elm-table-notes}",
        (
            r"\textbf{Sets.} \emph{all119}: the "
            f"{fx['shots']['all119']} reviewed shots, {fx['bins']['all119']:,} "
            "interior 50 ms bins (wholly inside one reviewed absent, non-crowd or "
            "crowd span and inside signal coverage). \\emph{bes73}: the "
            f"{fx['shots']['bes73']} of them with BES, {fx['bins']['bes73']:,} bins "
            "(ELM-O's coverage). \\emph{Common bins} "
            f"({fx['common_bins']['all119']:,}/{fx['common_bins']['bes73']:,}) also "
            "need valid DSM rows. \\emph{Known all-covered} cells, used only in "
            "the legacy swap, are 50 ms cells with at least 25 ms reviewed present "
            "occupancy inside every method's coverage; unknown time stays unknown. "
            "\\emph{Exact exports} are the source DSM's saved native rows. "
            "$^{\\ddagger}$ marks upstream data reused in source training, "
            "normalization or selection (``source stats'', ``source weights/stats''; "
            "two blind-cohort shots); $^{\\dagger}$ marks recall $\\geq0.99$."
        ),
        (
            r"\textbf{Labels.} "
            f"{pct(fx['crowd_share'])}\\% of positive bins lie in crowd spans "
            "(ELMing periods, not isolated ELMs); non-crowd starts are annotation "
            "boundaries without independent physical-onset truth. The review began "
            f"from the clock: {pct(share['start'])}\\% of crowd starts, "
            f"{pct(share['end'])}\\% of ends and {pct(share['both'])}\\% of both lie "
            f"within 1 ms of its boundaries ({share['crowd_spans']} crowd spans). "
            f"Of {offsets['starts']} non-crowd starts on {offsets['shots']} BES "
            f"shots, {offsets['matched']} (on {offsets['matched_shots']} shots) match "
            "the nearest ELM-O onset within $\\pm50$ ms; reviewed start minus "
            f"ELM-O onset has median ${offsets['median']}$ ms and quartiles "
            f"$[{offsets['p25']}, {offsets['p75']}]$ ms, an annotation offset, not a "
            "physical-onset error."
        ),
        (
            r"\textbf{Panels.} Native-detection panel: "
            f"{native['panel_shots']} shots with complete 124-input rows "
            f"({native['shots_at_least_90']} pass the 112/124-input gate). "
            "Legacy swap: eight overlap shots, seven with BES; the legacy table "
            "and the DSM were built on WPQH phases with breakthrough-ELM targets. "
            "Smith windows are selected windows: precision and F1 there are "
            "conditional, and continuous-discharge precision is unavailable."
        ),
        (
            r"\textbf{Thresholds.} Each fold selects its checkpoint by "
            "inner-validation AUPRC and its threshold by inner-validation F1; "
            "elm-ours thresholds a bin's mean probability, ELM-O and the clock use "
            "any touching detected span, and the DSM adaptation thresholds one "
            "aligned row score. Absent-span alarm rates (any detection touching a "
            "covered absent span): elm-ours "
            f"{alarm['all119/elm-ours']['span_alarm'][0]:.3f}, "
            f"clock {alarm['all119/elm-clock']['span_alarm'][0]:.3f} on all119; "
            f"ELM-O {alarm['bes73/elm-elmo']['span_alarm'][0]:.3f} on bes73. "
            f"Intervals use {fx['replicates']:,} shot-bootstrap draws and need five "
            "denominator-bearing shots per endpoint."
        ),
        r"\textbf{Limits.} " + limits_text(fx, review_days),
    ]
    return "\n\n".join(paragraphs) + "\n"


def limits_text(fx: dict, review_days: dict) -> str:
    """Limits: developmental CV, run days, serving protocol and unverified inputs."""
    text = (
        "Cross-validation is developmental: predictions on outer-fold shots "
        "existed before the recipe was fixed, and "
        f"{review_days['crossing']} of {review_days['days']} run days cross folds. "
    )
    run_day = fx.get("run_day")
    if run_day:
        auroc, f1 = F.fmt(*run_day["auroc"]), F.fmt(*run_day["all119"]["f1"])
        head = run_day["all119"]["headline"]
        text += (
            "Folds grouped by run day, which hold every day whole but do not "
            f"balance annotation kinds, give all119 AUROC {auroc} and F1 {f1} "
            f"against {head['auroc']:.3f} and {head['f1']:.3f} for the shot-grouped "
            "headline, which stays the headline. "
        )
    tiled = fx.get("tiled")
    if tiled:
        change = F.fmt(*tiled["whole_minus_tiled"], signed=True)
        text += (
            "Serving runs each shot whole through the five fold models, with the "
            f"mean fold threshold ({tiled['serving_threshold']:.3f}, unevaluated); "
            f"{tiled['tile_ms']:,} ms tiles with frozen weights give AUROC "
            f"{tiled['tiled_auroc']:.3f} against {tiled['whole_auroc']:.3f} "
            f"(whole minus tiled {change}). "
        )
    text += (
        "Fast-density units and FS01--04 sightlines are unverified; the U-Net's "
        "input scaling and chord screen are numerical choices, not calibrations. "
        "The DSM detection rows are lower bounds on DSM detection skill under "
        "our recipe. A frozen-model blind-split score awaits review of those "
        "shots; no catalog onset is delivered."
    )
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args(argv)
    out = args.out_dir or Paths.from_env().root / "round4" / "elm"
    out.mkdir(parents=True, exist_ok=True)
    files = F.paths(OUTPUTS)
    records = F.load(OUTPUTS)
    fx = F.facts(records)
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
            records["ours"], records["dsm"], records["feature"], fx
        ),
        "table_elm_common_control.tex": benchmark_table(
            records["ours"], records["dsm"], records["feature"], fx, common=True
        ),
        "table_elm_annotation_modes.tex": annotation_table(records["ours"]),
        "table_elm_per_kind.tex": per_kind_table(records["ours"]),
        "table_elm_dsm_own_target.tex": own_target_table(records["dsm"]),
        "table_elm_supplemental.tex": supplemental_table(records["dsm"]),
        "table_elm_native_detection.tex": native_detection_table(
            records["native_detection"], fx
        ),
        "table_elm_rejection.tex": rejection_table(records["rejection"]),
    }
    if fx["dsm_baselines"]:
        root_tables["table_elm_dsm_repeats.tex"] = dsm_repeats_table(fx)
    if "native" in records:
        root_tables["table_elm_dsm_native.tex"] = native_table(records["native"])
    if "smith" in records:
        root_tables["table_elm_smith.tex"] = smith_table(records["smith"])
    for name, content in root_tables.items():
        (out / name).write_text(content)
        target = OUTPUTS if name == "table_elm_benchmark.tex" else OUTPUTS / "appendix"
        target.mkdir(exist_ok=True)
        (target / name).write_text(content)
    note = appendix_note(fx)
    (out / "elm_table_notes.tex").write_text(note)
    (OUTPUTS / "elm_table_notes.tex").write_text(note)
    swap_tex.write(records["swap"], out, ours=records["ours"])
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
    over = {
        name: counts
        for name, counts in caption_words.items()
        if any(n > CAPTION_CAPS.get(name, DEFAULT_CAP) for n in counts)
    }
    if over:
        raise ValueError(f"A table caption exceeds its word cap: {over}")
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
