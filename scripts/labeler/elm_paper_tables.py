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


def benchmark_table(ours, dsm, feature=None, *, common=False) -> str:
    """Primary occupancy panels, or the secondary DSM common-bin control."""
    lines = [r"\begin{tabular}{lccc}", r"\toprule", r"Method & AUROC & AUPRC & F1 \\"]
    rows = (
        ("elm-ours", "elm-ours"),
        ("elm-elmo", "elm-elmo"),
        (
            "elm-dsm-detect",
            r"elm-dsm (60-input $1\times128$)",
        ),
        ("elm-clock", "elm-clock"),
        ("always present", "always-present"),
        ("elm-feature-only", "elm-feature"),
    )
    for tag in ("all119", "bes73"):
        res = (dsm if common else ours)["sets"][tag]
        scope = "All reviewed" if tag == "all119" else "BES subset"
        lines += [
            r"\midrule",
            r"\multicolumn{4}{l}{"
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
            cells = (
                [swap_tex.metric_cell(result, m) for m in ("auroc", "auprc", "f1")]
                if result
                else ["--"] * 3
            )
            lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    all_panel, bes_panel = (
        (dsm if common else ours)["sets"][tag] for tag in ("all119", "bes73")
    )
    deltas = []
    for key, label in (("auroc", "AUROC"), ("auprc", "AUPRC"), ("f1", "F1")):
        row = bes_panel["paired"][f"elm-ours - elm-elmo: {key}"]
        lo, hi = row["ci95"]
        deltas.append(f"{label} {row['value']:+.3f} [{lo:.3f}, {hi:.3f}]")
    if common:
        caption = (
            "Secondary common-bin control, requiring valid DSM rows for every method. "
            "The elm-dsm (60-input $1\\times128$) row is a detection adaptation on "
            "50 ms input means, not the native architecture. Paired elm-ours minus "
            "ELM-O on bes73: " + ", ".join(deltas) + ". Paired intervals include "
            "zero; equivalence is untested. Brackets: 95\\% shot-bootstrap intervals."
        )
    else:
        caption = (
            "Primary occupancy benchmark: developmental shot-grouped CV on the 119 "
            "reviewed shots; no untouched run-day or adjudicated continuous-interval "
            "confirmation. Reviewed occupancy uses 50 ms bins wholly inside reviewed "
            f"spans and signal coverage ({all_panel['bins']:,} all119; "
            f"{bes_panel['bins']:,} bes73). Five folds select thresholds by "
            "inner-validation F1; brackets are 95\\% intervals from 1,000 "
            "shot-bootstrap resamples. Hard calls use bin-mean probability "
            "$\\geq$ threshold for elm-ours; ELM-O and the clock use any "
            "detected-span touch. Review was seeded by the clock. The secondary "
            "60-input DSM adaptation is fragile: fold 4 selected epoch 0; "
            "thresholds range 0.021--0.482. Baselines/controls: always-present "
            "and elm-feature; ELM-O requires BES."
        )
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
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        (
            r"Method & Tolerance & Conditional precision & Recall & Conditional F1 "
            r"& Median $|e|$ (ms) \\"
        ),
        r"\midrule",
    ]
    for method, row in record["methods"].items():
        if method.startswith("elm-ours") and method != "elm-ours-onset":
            continue
        for tolerance, res in (("2", row["events"]["2"]),):
            cells = [
                swap_tex.cell(res["point"][m], res.get("ci95", {}).get(m), ci_digits=3)
                for m in ("precision", "recall", "f1")
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
        "Smith selected windows: frozen review-occupancy rows are omitted "
        "for target mismatch (50 ms occupancy versus approximately 8.5 ms windows). "
        "The experimental elm-ours-onset head is developmental "
        f"shot CV with {day_audit['run_days_spanning_multiple_folds']} of "
        f"{day_audit['run_days']} Smith run days crossing folds; overlap with "
        "ELM-O's historical tuning events is unknown because event membership "
        "was not retained. All metrics are conditional on selected windows; "
        "continuous-discharge precision/F1 are unavailable for every method, "
        "because false positives count only inside windows, with "
        f"{onset_audit['out_of_window_firings']:,} additional out-of-window head "
        f"firings. Head recall is {head['point']['recall']:.3f}; all matched "
        f"errors are within {onset_audit['max_absolute_error_ms']:.2f} ms and "
        f"{100 * onset_audit['correct_1ms_cell_fraction']:.0f}\\% are in the correct 1 ms "
        "cell. Only $\\pm2$ ms matching is shown. The head is withheld from "
        "catalog claims. Brackets: shot-bootstrap intervals."
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
        "Five exact exports are available, but 192751 has no scored reviewed "
        "overlap, leaving four shots in the exact-export JSON. "
        "$^{\\ddagger}$ marks shots used in source training, normalization or "
        "checkpoint selection; they do not provide independent confirmation."
    )
    return swap_tex.wrap_table("\n".join(lines), caption, "tab:elm-dsm-native")


def native_detection_table(record) -> str:
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
    caption = (
        f"Secondary native DSM detection comparison. The 1 ms audit finds at least "
        f"112/124 inputs on {record['shots_at_least_90_percent']} reviewed shots; "
        "37 have complete-input scored bins. Both DSM rows are occupancy detection "
        r"refits: 60-input $1\times128$ adaptation on 50 ms means, and native "
        "124-input [100,1000] architecture on 1 ms means. All methods use identical "
        "supported shots/bins. Native weights start randomly; normalization uses "
        "optimizer-training shots only, with the original outer/inner partitions. "
        "Inner-validation AUPRC selects checkpoints and F1 selects thresholds. "
        r"Native probabilities are averaged per 50 ms bin. Brackets: 95\% "
        "shot-bootstrap intervals. These are developmental results from a fixed "
        "25-epoch refit, with reconstructed native inputs and within-shot NBI smoothing."
    )
    return swap_tex.wrap_table("\n".join(lines), caption, "tab:elm-native-detection")


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
        "Per-chord preprocessing on all119 (119 shots/12,409 primary bins). "
        "The density screen zeroes chords with median absolute native magnitude "
        "$>10^{16}$; FS02--04 have no such screen. FS clipping floors native "
        "values at $10^{12}$ before the logarithm. Density divided by $10^{14}$ "
        "is clipped to $[-3,12]$; ten times its high-pass to $[-10,10]$. "
        "Clipping counts exclude screened chords and mark any affected cell; "
        "clipping discards no bins. Disabling only the screen, using frozen "
        "weights/thresholds and raw chords under the same clipping, changes "
        f"AUROC {delta['auroc']:+.5f}, AUPRC {delta['auprc']:+.5f}, "
        f"F1 {delta['f1']:+.5f}; six shots change inputs. This does not validate "
        "physical calibration or diagnostic failure."
    )
    return swap_tex.wrap_table("\n".join(lines), caption, "tab:elm-rejection")


def appendix_note(strata, dsm, swap, offsets) -> str:
    timing = offsets["offset_ms"]
    timing_note = (
        f"Of {offsets['n_starts']} non-crowd reviewed starts on "
        f"{offsets['n_shots_with_non_crowd_spans']} BES-subset shots, "
        f"{offsets['n_matched']} on {offsets['n_matched_shots']} shots match the "
        "nearest ELM-O onset within $\\pm50$ ms. For that matched subset only, "
        "reviewed start minus ELM-O onset has median "
        f"{timing['median']:.3f} ms and quartiles "
        f"[{timing['p25']:.3f}, {timing['p75']:.3f}] ms; "
        f"{offsets['n_unmatched']} unmatched starts are omitted. Matching is "
        "independent per start and permits onset reuse. This annotation-to-detector "
        "offset is not a measured physical-onset error "
        "(review\\_start\\_offsets.json)."
    )
    return (
        r"""\paragraph{ELM evaluation protocol.}\label{app:elm-table-notes}
Primary reviewed occupancy uses all119 (12,409 bins) and bes73 (6,843 bins),
with 50 ms bins wholly inside known spans and signal coverage. The secondary
common-bin control additionally requires valid survival and repaired detection
DSM rows (11,653 all119; 6,527 bes73). Crowd annotations supply 97\% of positive bins. Review began from the clock:
within 1 ms, 56\% of crowd starts, 44\% of ends and 33\% of both edges match its
boundaries. This development reference depends on D-alpha evidence; reviewed
non-crowd starts are annotation boundaries without independent physical-onset truth.

"""
        + timing_note
        + r"""

Five shot-grouped folds exclude blind-cohort shots; checkpoint and threshold
selection uses inner-validation shots. These are developmental cross-validation
estimates. Preliminary predictions on outer-fold shots were available before the
reported recipe was fixed and could have informed input choice, scaling,
architecture, checkpoint or threshold selection, or the evaluation protocol;
the saved records do not establish whether or how much they influenced those
choices. The primary occupancy checkpoints and thresholds remain frozen.
No independently validated physical-onset detector is delivered, and run days
cross folds in both developmental analyses (16 review days; 21 of 31 Smith days).
Primary absent-span alarm rates are 0.596 for elm-ours and 0.105 for the clock
on all119; ELM-O is 0.440 on bes73. These use sufficiently covered absent spans
and any detection touching them, not the absent-bin false-positive rate
(ours/evaluation.json:sets.*.methods.*.point.absent\_span\_alarm\_rate).
Primary bes73 paired elm-ours minus ELM-O differences are AUROC
+0.023 [-0.014,0.067], AUPRC +0.043 [-0.058,0.129], and F1 +0.002
[-0.052,0.058]; paired CIs include zero and equivalence is untested.
A trailing-three-epoch selection after warm-up is reported as a sensitivity,
not chosen on its outer-fold performance. Feature-only uses one within-bin
FS02--04 log D-alpha max-minus-median feature and L2 logistic regression on
identical folds. The always-present control uses no diagnostic input.

The BES-free U-Net uses FS02--04 and DENV2F/DENV3F; FS01 was omitted because
the retained input cache contains FS02--04 only. ELM-O uses BES, filterscopes
and fast density; the clock uses D-alpha with a plasma-phase gate.
Fast-density physical ordinate units and FS01--04 sightlines (divertor versus
midplane) are unverified in the retained metadata. The U-Net's numerical
preprocessing is fixed: filterscope levels use $(\log_{10}(\max(x,10^{12}))-15)/1.5$;
fast density is divided by $10^{14}$ native ordinate units and clipped to
$[-3,12]$, while ten times its 0.2 s high-pass is clipped to $[-10,10]$.
A chord with median absolute native magnitude above $10^{16}$ is zeroed by
the heuristic failed-digitiser screen. These are numerical choices, not
verified calibrations or a validated diagnostic-failure criterion; paired
slow CO2 cannot determine the fast-channel units. Metadata audits made no
new fetches and did not change saved inputs, clipping, screening or weights.

The detection DSM is a reduced-input adaptation trained and evaluated on
50 ms rows: 60 inputs and one 128-unit layer, fitted with training-only
normalization and random weights. PCPHD02/03 cover all 119 shots without
filterscope substitutes; DENV2F and DENV3F means supply the two density
columns on 115 shots per chord, with four per chord mean-filled.
The source model instead trained on native 1 ms rows with 124 inputs and
layers [100,1000] for WPQH breakthrough-ELM forecasting. The detection
adaptation is not an objective-only retrain of that model.
The secondary native detection refit uses the [100,1000] architecture, random
weights and training-only normalization on complete 124-input 1 ms rows;
all compared methods share its supported shots/bins (native\_detection.json).
The native audit meets the 90\%/30-shot gate (48 shots); complete-input bins
support 37 shots. Source weights and statistics are not reused. The retained
60-input $1\times128$ adaptation remains a separate row. Labels ``source stats''
and ``source weights/stats'' in swap tables denote historical detection variants.
For hard bin calls, elm-ours thresholds its mean probability, while the DSM
adaptation thresholds one aligned row score computed from 50 ms input means.
Historical survival
and detection variants that reuse source weights or statistics remain
supplemental; their normalization includes two blind-cohort shots.
50 ms means and centered-NBI lookahead also differ from native training.
Native risks use forward reviewed-presence or non-crowd-start targets in
$(t,t+h]$. Five exact exports are available; 192751 has no scored reviewed
overlap, leaving four shots in the exact-export JSON.
Those four contributed to source training, normalization or checkpoint
selection and supply descriptive evidence only.

Eligible intervals use 1,000 physical-shot bootstrap draws. Every metric records
finite and undefined draws; at least five denominator-bearing shots are required
per endpoint (negative-bearing shots for false-alarm rates).
$^{\ddagger}$ marks upstream data reused in training, normalization or selection;
$^{\dagger}$ marks recall at least 0.99. Complete precision and recall, input
availability and memberships remain in the evaluation JSON records.

Frozen review-to-Smith transfer shares neither review shots nor review run days.
The overlap with ELM-O's historical tuning events is unknown because event
membership was not retained. Window occupancy uses 1 ms cells and differs from
the crowd-dominated 50 ms review target. Event metrics use one-to-one onset
matching at 2 ms and report timing errors; ELM-O uses its published fixed
setting. Frozen occupancy transfer is omitted for target mismatch: 50 ms
occupancy versus approximately 8.5 ms selected windows with brief ELM regions.
The frozen auxiliary span-start output is also omitted. The experimental
Smith-trained elm-ours-onset head recalls
0.930 of selected-window events; matched errors are within 0.82 ms, with 94\%
in the correct 1 ms cell. Every method's precision/F1 is conditional on selected
windows: false positives are counted only within those windows, while 30,436
additional head firings outside them lack negative truth.
Continuous-discharge precision/F1 are unavailable for every method.
Catalog onset output is withheld.

Per-chord screen and clipping counts and the frozen disabled-screen sensitivity
are reported in the accompanying per-chord table and
ours/rejection\_sensitivity.json. Clipping caps/floors values; it rejects no bins.
The screen zero-fills density features, rather than fitting a physical mean.

Finding 1 uses all-covered known 50 ms cells with at least 25 ms reviewed present;
unknown time stays unknown. Finding 2 includes both strict interior and known
all-covered bins with identical saved predictions and thresholds, intersecting
actual diagnostic coverage for every method. Only AUROC compares references,
because F1 thresholds were review-tuned. Covered-gap merges at 100/200/300 ms
are definition sensitivities; missing coverage is never bridged.
The eight-shot swap is inconclusive and cannot establish population ranking reversal.
"""
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args(argv)
    out = args.out_dir or Paths.from_env().root / "round4" / "elm"
    out.mkdir(parents=True, exist_ok=True)
    files = {key: OUTPUTS / key / "evaluation.json" for key in ("ours", "dsm", "swap")}
    files["feature"] = OUTPUTS / "ours" / "feature_only.json"
    files["strata"] = OUTPUTS / "ours" / "annotation_strata.json"
    files["offsets"] = OUTPUTS / "review_start_offsets.json"
    files["density"] = OUTPUTS / "density_units.json"
    files["filterscope"] = OUTPUTS / "filterscope_metadata.json"
    files["history"] = OUTPUTS / "training_history.json"
    files["native_detection"] = OUTPUTS / "dsm/native_detection.json"
    files["rejection"] = OUTPUTS / "ours/rejection_sensitivity.json"
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
        "table_elm_common_control.tex": benchmark_table(
            records["ours"], records["dsm"], records["feature"], common=True
        ),
        "table_elm_annotation_modes.tex": annotation_table(records["ours"]),
        "table_elm_per_kind.tex": per_kind_table(records["ours"]),
        "table_elm_dsm_own_target.tex": own_target_table(records["dsm"]),
        "table_elm_supplemental.tex": supplemental_table(records["dsm"]),
        "table_elm_native_detection.tex": native_detection_table(
            records["native_detection"]
        ),
        "table_elm_rejection.tex": rejection_table(records["rejection"]),
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
    note = appendix_note(
        records["strata"], records["dsm"], records["swap"], records["offsets"]
    )
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
