#!/usr/bin/env python
"""Render the current ELM protocol and README Models from canonical records."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd

from labeler.config import Paths, git_sha, sha256_of
from labeler.elm import net

REPO = Path(__file__).resolve().parents[2]
OUTPUTS = REPO / "outputs/labeler/elm"
NAMES = {
    "elm-ours": "elm-ours",
    "elm-elmo": "ELM-O",
    "elm-clock": "elm-clock",
    "elm-dsm": "elm-dsm refit‡ (supplemental)",
    "elm-dsm-detect": "elm-dsm detection (isolated)",
    "elm-dsm-detect-exposed": "elm-dsm detection‡ (exposed; supplemental)",
    "elm-dsm-detect-init": "elm-dsm detection init‡ (supplemental)",
    "always present": "always present",
}


def metric(res, key):
    if key not in res["point"]:
        return "--"
    point = res["point"][key]
    if not math.isfinite(point):
        return "--"
    ci = res.get("ci95", {}).get(key)
    value = f"{point:.3f}"
    if ci is not None and math.isfinite(ci[0]):
        value += f" [{ci[0]:.3f}, {ci[1]:.3f}]"
    if key == "f1" and res["point"]["recall"] >= 0.99:
        value += "†"
    return value


def benchmark_rows(record):
    all_set, bes_set = (record["sets"][tag] for tag in ("all119", "bes73"))
    lines = [
        (
            f"All reviewed shots: {all_set['n_shots']} shots/{all_set['bins']:,} bins. "
            f"BES subset, ELM-O chunks: {bes_set['n_shots']} shots/{bes_set['bins']:,} bins."
        ),
        "",
        "| Set | Method | AUROC | AUPRC | F1 | Precision / recall |",
        "|---|---|---|---|---|---|",
    ]
    for tag in ("all119", "bes73"):
        subset = record["sets"][tag]
        for name, res in subset["methods"].items():
            cells = [metric(res, key) for key in ("auroc", "auprc", "f1")]
            p, r = res["point"]["precision"], res["point"]["recall"]
            lines.append(
                f"| {tag} | {NAMES[name]} | "
                + " | ".join(cells)
                + f" | {p:.3f} / {r:.3f} |"
            )
    return "\n".join(lines)


def alarm_rows(ours):
    lines = [
        "| Set | Method | Raw | 25 ms guard, same denominator | Nonempty interiors |",
        "|---|---|---|---|---|",
    ]
    for tag in ("all119", "bes73"):
        for name in ("elm-ours", "elm-elmo"):
            res = ours["sets"][tag]["methods"].get(name)
            if res is None:
                continue
            cells = [
                metric(res, key)
                for key in (
                    "absent_span_alarm_rate",
                    "absent_span_alarm_rate_guard25",
                    "absent_span_interior_alarm_rate_guard25",
                )
            ]
            lines.append(f"| {tag} | {NAMES[name]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def audit_rows(swap):
    audits = {"Onset bins": swap["interval_audit"]}
    audits.update(
        {
            f"Occupancy τ={gap} ms": swap["interval_audit_occupancy"][f"gap_{gap}ms"]
            for gap in (100, 200, 300)
        }
    )
    lines = [
        "| Reference | Known bins | M | P | Legacy recall | Legacy F1 |",
        "|---|---|---|---|---|---|",
    ]
    for title, audit in audits.items():
        res = audit["known_review_majority"]
        lines.append(
            f"| {title} | {res['bins']} | {res['M']} | {res['P']} | "
            + metric(res, "recall")
            + " | "
            + metric(res, "f1")
            + " |"
        )
    return "\n".join(lines)


def ranking_rows(swap):
    lines = [
        "| Set | Reference | M / P | Recall | AUROC order | F1 order |",
        "|---|---|---|---|---|---|",
    ]
    for tag in ("overlap", "overlap_bes"):
        subset = swap["swap"][tag]
        refs = {"Onset bins": subset}
        refs.update(
            {
                f"Occupancy τ={gap} ms": subset["occupancy"][f"gap_{gap}ms"]
                for gap in (100, 200, 300)
            }
        )
        for title, record in refs.items():
            finding = record["finding_1"]
            order = record["legacy"]["ranking"]
            orders = [
                " > ".join(NAMES.get(n, n) for n in order[key])
                for key in ("auroc", "f1")
            ]
            lines.append(
                f"| {tag} | {title} | {finding['M']} / {finding['P']} | "
                + metric(finding, "recall")
                + " | "
                + " | ".join(orders)
                + " |"
            )
    return "\n".join(lines)


def repeat_rows(repeats):
    lines = [
        "| Training seed | all119 AUROC | all119 AUPRC | all119 F1 |",
        "|---|---|---|---|",
    ]
    for row in repeats["results"]:
        res = row["sets"]["all119"]
        lines.append(
            f"| {row['training_seed']} | "
            + " | ".join(metric(res, key) for key in ("auroc", "auprc", "f1"))
            + " |"
        )
    lines += [
        "",
        "| Set | Metric | Three new seeds: min–max | Sample SD |",
        "|---|---|---|---|",
    ]
    for tag, metrics in repeats["seed_ranges"].items():
        for key in ("auroc", "auprc", "f1"):
            spread = metrics[key]["three_new_seeds"]
            lines.append(
                f"| {tag} | {key} | {spread['min']:.3f}–{spread['max']:.3f} | "
                f"{spread['sample_sd']:.3f} |"
            )
    return "\n".join(lines)


def kind_rows(ours):
    lines = [
        "| Set | Method | Crowd-bin recall | Non-crowd span-touch recall |",
        "|---|---|---|---|",
    ]
    for tag in ("all119", "bes73"):
        for name, res in ours["sets"][tag]["methods"].items():
            if name == "always present":
                continue
            lines.append(
                f"| {tag} | {NAMES[name]} | "
                + metric(res, "crowd_bin_recall")
                + " | "
                + metric(res, "non_crowd_span_touch_recall")
                + " |"
            )
    return "\n".join(lines)


def mode_rows(ours):
    lines = [
        (
            "| Annotation mode | Shots / bins | AUROC | F1 | Precision | Recall | "
            "Absent-bin call fraction |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for name, group in ours["sets"]["all119"]["annotation_modes"].items():
        res = group["methods"]["elm-ours"]
        lines.append(
            f"| {name.replace('_', ' ')} | {group['n_shots']} / {group['bins']:,} | "
            + " | ".join(
                (
                    "--"
                    if name == "no_present" and key != "false_alarm_bin_rate"
                    else metric(res, key)
                )
                for key in (
                    "auroc",
                    "f1",
                    "precision",
                    "recall",
                    "false_alarm_bin_rate",
                )
            )
            + " |"
        )
    return "\n".join(lines)


def own_target_rows(dsm):
    lines = [
        "| Horizon (ms) | AUROC | Cases / controls |",
        "|---|---|---|",
    ]
    for res in dsm["own_target"]["horizons"].values():
        point = res["auroc"]
        low, high = res["auroc_ci95"]
        lines.append(
            f"| {res['horizon_ms']:g}‡ | {point:.3f} [{low:.3f}, {high:.3f}] | "
            f"{res['cases']:,} / {res['controls']:,} |"
        )
    return "\n".join(lines)


def filterscope_rows(record):
    lines = ["| Input | Tree PMT | View from metadata |", "|---|---|---|"]
    for name in record["used_channels"]:
        item = record["views"][name]
        view = item["view"] or "unverified: no divertor/midplane field"
        lines.append(f"| {name} | {item['pmt']} | {view} |")
    lines += [
        "",
        (
            f"FS01 has finite corpus samples on {record['fs01_finite_shots']}/119 shots. "
            "It is outside the existing three-channel native-rate cache and all retained "
            "fits; exclusion was a data-pipeline choice. Fetching is allowed this round. "
            "Read-only tree queries verified the PMT aliases but returned only signal "
            "and calibration metadata, with no sightline labels. Each channel's divertor "
            "versus midplane view therefore remains unverified. Source: "
            "`outputs/labeler/elm/filterscope_metadata.json`."
        ),
    ]
    return "\n".join(lines)


def native_rows(record):
    lines = [
        "| Native panel | Horizon (ms) | Scored shots / 1 ms rows | AUROC |",
        "|---|---|---|---|",
    ]
    for panel in ("reviewed_exact_export", "reviewed_reconstructed", "own_target"):
        for name, result in record[panel]["horizons"].items():
            low, high = result["auroc_ci95"]
            lines.append(
                f"| {panel}‡ | {name[1:-2]} | {result['n_shots']} / "
                f"{result['rows']:,} | {result['auroc']:.3f} "
                f"[{low:.3f}, {high:.3f}] |"
            )
    lines += [
        "",
        (
            "The native checkpoint uses all **124 original inputs, including both "
            "photodiodes and 64 BES columns, at 1 ms**. Exact original exports exist "
            "on five reviewed shots, but 192751 has no scored reviewed overlap: "
            "four contribute 11,565 rows. All four have source exposure: 196541 "
            "was optimizer-trained; 190637, 190643 and 192721 were used for "
            "checkpoint selection. This is a separate source-exposed panel, not "
            "the 119-shot 50 ms benchmark."
        ),
        "",
        (
            "Original diagnostic records plus paced photodiode fetches make 45 "
            "shots reconstructable; only 33 contribute reviewed rows after the "
            "original native-domain filter (no mean fill or clipping). Reconstructed "
            "inputs are a **sensitivity**, because within-shot NBI smoothing cannot "
            "reproduce the source's smoothing of concatenated filtered phase rows "
            "across boundaries. The exact export is the faithful native-input "
            "evaluation. Native own-target AUROC uses 326 physical early-stopping "
            "validation shots from the original reversed split, not untouched test "
            "evidence. Every panel has 1000 physical-shot bootstrap replicates."
        ),
        "",
        (
            "Sources: `dsm/native_evaluation.json:{checkpoint,coverage,exposure,"
            "reviewed_exact_export,reviewed_reconstructed,own_target}` and "
            "`dsm/native_fetch.json`. Coverage lists every unavailable original "
            "column per shot. Missing raw diagnostic groups, including complete "
            "BES/ECE/actuator records on later shots, prevent a complete native "
            "reconstruction; 50 ms adapter means cannot recover those 1 ms inputs. "
            "78 photodiode records were fetched on 39 otherwise complete-input "
            "shots, one worker, pace 1, with no authentication error."
        ),
    ]
    return "\n".join(lines)


def main():
    files = {
        "ours": "ours/evaluation.json",
        "dsm": "dsm/evaluation.json",
        "swap": "swap/evaluation.json",
        "feature": "ours/feature_only.json",
        "strata": "ours/annotation_strata.json",
        "smith": "smith/evaluation.json",
        "smoothed": "ours/smoothed_selection.json",
    }
    records = {
        key: json.loads((OUTPUTS / path).read_text())
        for key, path in files.items()
        if (OUTPUTS / path).exists()
    }
    ours, detector = (records[k] for k in ("ours", "dsm"))
    primary = ours["sets"]["all119"]
    cohort_path = Paths.from_env().catalog / "cohort.csv"
    cohort = pd.read_csv(cohort_path).set_index("shot")
    cohort_counts = cohort.split.value_counts().to_dict()
    reviewed_counts = dict(Counter(cohort.loc[primary["shots"], "split"]))
    parameters = net.n_parameters(net.ElmUNet())
    pos = primary["methods"]["elm-ours"]["counts"]
    crowd_share = pos["crowd_bins"] / (pos["tp"] + pos["fn"])
    domain = (
        "The DSM and legacy onset table were built on WPQH phases with "
        "breakthrough-ELM targets; Finding 1 and low DSM AUROCs partly reflect "
        "definition and domain shift (192721: 1 legacy bin versus 17 non-crowd "
        "review spans)."
    )
    lines = [
        "# ELM occupancy and onset evaluation",
        "",
        (
            f"`elm-ours` is a {parameters:,}-parameter 1D U-Net over FS02--FS04 log "
            "D-alpha and DENV2F/3F native density ordinates, without BES. The "
            "reviewed target is occupancy of known ELMy intervals in 50 ms bins, "
            f"with {100 * crowd_share:.2f}% of positive bins supplied by crowd spans. "
            "Reviewed non-crowd starts are not independently verified physical onsets."
        ),
        "",
        domain,
        "",
        (
            f"The fixed cohort is {cohort_counts['train']} train / "
            f"{cohort_counts['val']} validation / {cohort_counts['test']} blind test. "
            f"The {primary['n_shots']} reviewed shots are {reviewed_counts['train']} "
            f"train and {reviewed_counts['val']} validation; five fixed physical-shot "
            "folds reserve 14 inner-validation shots for checkpoint and F1 threshold "
            "selection. No blind-test shots enter fits, normalization or tuning. "
            "These review results are development estimates: earlier development "
            "included outer-fold predictions, and folds share run days. The original "
            "cv2 recipe and thresholds remain frozen; independent Smith evaluation "
            "uses its five-checkpoint ensemble. Smoothed selection is a sensitivity."
        ),
        "",
        "## Reviewed occupancy",
        "",
        f"Original signal-coverage panel: {primary['n_shots']} shots / "
        f"{primary['bins']:,} bins; AUROC "
        + metric(primary["methods"]["elm-ours"], "auroc")
        + "; F1 "
        + metric(primary["methods"]["elm-ours"], "f1")
        + ". "
        "The main paper table uses identical DSM-covered bins for all core rows.",
        "",
        "| Panel / method | Bins | AUROC | AUPRC | F1 |",
        "|---|---:|---|---|---|",
    ]
    model_lines = []
    rows = (
        ("elm-ours", "elm-ours"),
        ("elm-elmo", "elm-elmo"),
        ("elm-dsm-detect", "elm-dsm (detection)"),
        ("elm-clock", "elm-clock"),
        ("always present", "elm-always-present"),
        ("elm-feature-only", "elm-feature-only"),
    )
    for tag in ("all119", "bes73"):
        subset = detector["sets"][tag]
        for key, label in rows:
            if key == "elm-feature-only":
                res = records["feature"]["sets"]["common"][tag]["methods"][key]
            else:
                res = subset["methods"].get(key)
            if res is None:
                continue
            cells = [metric(res, k) for k in ("auroc", "auprc", "f1")]
            lines.append(
                f"| {tag} / {label} | {subset['bins']:,} | " + " | ".join(cells) + " |"
            )
            if tag == "all119" or key == "elm-elmo":
                model_lines.append(
                    f"- {label} | 2026_10_03 | AUROC: {cells[0]} | "
                    f"AUPRC: {cells[1]} | F1: {cells[2]} "
                    f"({subset['n_shots']} shots / {subset['bins']:,} common bins)"
                )
    lines += [
        "",
        (
            "Inputs are explicit in the main caption. The revised isolated DSM "
            "detector uses real PCPHD02/03 where available (otherwise labelled FS "
            "substitutes), and native DENV2F/3F means for CO2 v2/v3. Its remaining "
            "60-column inputs are actuator, magnetic, slow CO2 and ECE diagnostics, "
            "with fold-training-only normalization and mean fill. Native density "
            "physical calibration remains unresolved; scale/paired-unit checks and "
            "old/new DSM scores are in `dsm/evaluation.json`. Historical source-exposed "
            "survival and detection variants are supplemental. Native 124-input "
            "forecasts use forward presence and non-crowd onset targets in (t,t+h]. "
            "The four exact-export shots appear only in the appendix JSON record."
        ),
        "",
        (
            "The clock seeded D-alpha review. Crowd-boundary identity within "
            "1 ms is 56% of starts, 44% of ends and 33% of both; exact counts are in "
            "`ours/annotation_strata.json`. D-alpha sightlines remain unverified. "
            "Bin calls are U-Net mean score at its fold threshold, DSM bin-end score, "
            "or any touching ELM-O/clock span. Detected spans are clipped to panel "
            "coverage. Short inter-ELM gaps and centered 50 ms smoothing contribute "
            "to occupancy disagreements. Per-kind, annotation-mode, guarded alarm, "
            "review non-crowd tau-merge and seed diagnostics remain in appendix records."
        ),
        "",
        (
            "Intervals use 1000 physical-shot bootstrap replicates. Valid and "
            "undefined counts are retained for each metric. Fewer than five shots "
            "with positive targets yields descriptive estimates only, without a "
            "population 95% interval; this includes the two-shot BES subset."
        ),
        "",
        "## Independent Smith evaluation",
        "",
        (
            "`smith/evaluation.json` records 2,316 hand-labelled windows on 211 "
            "shots, exact shot/run-day overlap, frozen checkpoint hashes and "
            "thresholds, 1 ms occupancy metrics and one-to-one event onset matching "
            "at ±2/5 ms with timing errors. Window occupancy differs from 50 ms crowd "
            "occupancy. `elm-ours-onset` is trained only on Smith shots with grouped "
            "CV; review and cohort blind-test shots are excluded. ELM-O uses its "
            "published fixed setting. Its previous 0.997 precision / 0.980 recall "
            "are window-overlap reimplementation scores, not comparable onset "
            "matching scores; the paper digest reports 0.995 / 0.976 on 972 tuning "
            "ELMs. Poor onset agreement means the onset output is not delivered."
        ),
        "",
        "## Reference swap",
        "",
        (
            "The legacy WPQH onset table has 576 shots, eight overlapping review; "
            "seven have BES. Finding 1 on 782 known all-covered bins gives M/P = "
            "60/14 (onset bins) and 34/60 (200 ms covered-gap occupancy). Unknown "
            "review time is excluded. These are definition disagreements, not "
            "adjudicated missing physical events. Covered-gap merges at 100/200/300 "
            "ms never bridge unavailable coverage. Finding 2 uses fixed predictions "
            "on both 641 strict interior bins and 782 known all-covered bins. "
            "Only AUROC supports cross-reference comparisons because F1 thresholds "
            "were review-tuned. Eight-shot evidence is inconclusive; point crossings "
            "do not establish a population ranking reversal. Full values and "
            "bootstrap draw counts are in `swap/evaluation.json`."
        ),
        "",
        "## Artifacts and reproduction",
        "",
        (
            "Small canonical JSON records and the six-row, two-panel main table are "
            "under `outputs/labeler/elm/`; supplemental tables are under `appendix/` "
            "and `swap/`. Large checkpoints, predictions, vector PDF / 150 dpi PNG "
            "figures and table proofs live under `$LABELER_ROOT/round4/elm/`. "
            "Every PNG must be visually inspected; the figure record documents the "
            "next-rank replacement for the ambiguous drop-shaped former panel (b)."
        ),
        "",
        (
            "Use the mandated pixi labelmaker wrapper for CPU scripts and the "
            "phase3 CUDA interpreter for GPU training (CUDA_VISIBLE_DEVICES=1, "
            "at most 12 GB). Set TMPDIR to the ELM scratch directory and "
            "LABELER_NO_FETCH=1 except authorized fetches. Executable entry points:"
        ),
        "",
        "```text",
        "scripts/labeler/elm_ours_evaluate.py --run cv2",
        "scripts/labeler/elm_feature_evaluate.py --run cv2",
        "scripts/labeler/elm_annotation_strata.py --run cv2",
        "scripts/labeler/elm_dsm_evaluate.py --run cv2 --rescore",
        "scripts/labeler/elm_dsm_native.py",
        "scripts/labeler/elm_reference_swap.py --run cv2 --no-tables",
        "scripts/labeler/elm_smith_evaluate.py --help",
        "scripts/labeler/elm_paper_tables.py",
        "scripts/labeler/elm_example_figure.py --run cv2",
        "scripts/labeler/elm_protocol.py",
        "```",
        "",
        (
            "Current code/record hashes identify evaluation provenance; historical "
            "training metadata remains retrospective where noted. The stream report "
            "contains fix history and detailed verification. No production labels "
            "or blind-test splits are modified."
        ),
    ]
    (REPO / "docs/labeler/elm_ours.md").write_text("\n".join(lines))
    readme = REPO / "data/events/edge_localized_mode/README.md"
    text = readme.read_text()
    begin, end = text.index("## Models"), text.index("## Inputs")
    caveat = (
        "Brackets are eligible 95% shot-bootstrap intervals. Review scores are "
        "occupancy-development estimates (97% crowd positives; clock-seeded "
        "D-alpha review), with shot-grouped fits excluding blind-cohort shots. "
        "ELM-O requires BES; the DSM detector includes photodiodes and fast-density "
        "substitutes with unresolved calibration. Historical source-exposed "
        "DSM variants remain supplemental. The independent Smith onset and "
        "occupancy evaluation, run-day overlap and limitations are in "
        "[elm_ours.md](../../../docs/labeler/elm_ours.md); poor physical-onset "
        "validation means no onset output is delivered. " + domain
    )
    section = [
        "## Models",
        "",
        "**stable**: elm-dsm (offline survival adapter)",
        "",
        "**latest**: elm-ours",
        "",
        *model_lines,
        "",
        caveat,
        "",
        "",
    ]
    rendered = text[:begin] + "\n".join(section) + text[end:]
    rendered = rendered.replace(
        "Each ELM is a burst on the divertor D-alpha signal 2-5 ms wide.",
        "D-alpha pulse morphology and duration depend on diagnostic view, ELM "
        "type and detachment; 2--5 ms bursts are a common diagnostic signature, "
        "not a universal definition of an ELM.",
    )
    rendered = rendered.replace(
        "`pcphd02`, `pcphd03` (always mean-filled)",
        "`pcphd02`, `pcphd03` (detection: measured, otherwise FS substitutes)",
    )
    readme.write_text(rendered)
    manifest = {
        "git": git_sha(),
        "sources": {
            key: {"path": path, "sha256": sha256_of(OUTPUTS / path)}
            for key, path in files.items()
            if (OUTPUTS / path).exists()
        },
        "artifacts": {
            str(p.relative_to(REPO)): sha256_of(p)
            for p in (REPO / "docs/labeler/elm_ours.md", readme)
        },
        "metadata": {
            "reviewed_shots": primary["n_shots"],
            "reviewed_split_counts": reviewed_counts,
            "cohort_split_counts": cohort_counts,
            "cohort_sha256": sha256_of(cohort_path),
            "elm_ours_parameters": parameters,
            "crowd_positive_share": crowd_share,
        },
    }
    (OUTPUTS / "protocol.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print("protocol written")


if __name__ == "__main__":
    main()
