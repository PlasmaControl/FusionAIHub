#!/usr/bin/env python
"""Append the current-state-first fix-round-four report from canonical records."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from elm_protocol import metric

from labeler.config import Paths

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/labeler/elm"
REPORT = Path(
    "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/claude-89242e53/r4/reports/elm.md"
)


def read(name):
    return json.loads((OUT / name).read_text())


def main():
    ours, dsm, smith, smooth, swap = map(
        read,
        (
            "ours/evaluation.json",
            "dsm/evaluation.json",
            "smith/evaluation.json",
            "ours/smoothed_selection.json",
            "swap/evaluation.json",
        ),
    )
    _strata, verify, tables = map(
        read,
        ("ours/annotation_strata.json", "fix_round4_verification.json", "tables.json"),
    )
    primary = ours["sets"]["all119"]["methods"]["elm-ours"]
    smoothed_main = smooth["sets"]["all119"]["methods"]["elm-ours"]
    common = dsm["sets"]["all119"]["methods"]["elm-ours"]
    detector = dsm["sets"]["all119"]["methods"]["elm-dsm-detect"]
    feature = read("ours/feature_only.json")["sets"]["primary"]["all119"]["methods"][
        "elm-feature-only"
    ]
    independent = smith["protocol"]["independence"]
    frozen, head, _elmo = (
        smith["methods"][m] for m in ("elm-ours", "elm-ours-onset", "elm-elmo")
    )
    text = f"""## Fix round 4

Status: DONE_WITH_CONCERNS. This is the current state; preceding sections are
historical. Both fourth re-reviews were read. The manuscript was not edited.

The frozen primary occupancy model remains AUROC {metric(primary, "auroc")},
AUPRC {metric(primary, "auprc")} and F1 {metric(primary, "f1")} on
{ours["sets"]["all119"]["bins"]:,} bins / {ours["sets"]["all119"]["n_shots"]} shots.
The main table now uses {dsm["sets"]["all119"]["bins"]:,} identical DSM-covered bins:
elm-ours AUROC {metric(common, "auroc")}, versus repaired isolated DSM
{metric(detector, "auroc")} / F1 {metric(detector, "f1")}. Repairing inputs did
not improve DSM. The feature-only control scores AUROC {metric(feature, "auroc")}
/ F1 {metric(feature, "f1")} on the original primary bins. The separate smoothed
selection run scores AUROC {metric(smoothed_main, "auroc")}
/ F1 {metric(smoothed_main, "f1")}; it does not
replace the primary model or the frozen Smith ensemble.
Sources: `outputs/labeler/elm/ours/evaluation.json`, `dsm/evaluation.json`,
`ours/feature_only.json`, `ours/smoothed_selection.json` (`sets.all119.methods`,
or `sets.primary.all119.methods` for feature-only).

Smith supplies {independent["smith_windows"]:,} hand-labelled windows on
{len(independent["smith_shots"])} shots, spanning {len(independent["smith_run_days"])}
days, versus {len(independent["review_run_days"])} review run days; shot and run-day
overlaps are both zero, with no unresolved dates. Frozen elm-ours window occupancy
AUROC is {metric(frozen["occupancy_1ms"], "auroc")} and F1
{metric(frozen["occupancy_1ms"], "f1")}. Its frozen auxiliary onset F1 is
{metric(frozen["events"]["2"], "f1")} at ±2 ms and
{metric(frozen["events"]["5"], "f1")} at ±5 ms. The independently Smith-trained
elm-ours-onset head has F1 {metric(head["events"]["2"], "f1")} / 
{metric(head["events"]["5"], "f1")}, respectively. The experimental Smith CV
trace is available as benchmark output; physical-onset catalog output remains
withheld, with the exact scope/reason in the record. These are selected-window
metrics, not continuous-discharge false-alarm evidence.
Source: `outputs/labeler/elm/smith/evaluation.json:protocol.independence,methods`.

### What changed and data discipline

- `labeler.elm.smith` and `elm_smith_evaluate.py` implement window-union targets,
  maximum-cardinality one-to-one matching with timing-error tie breaks, whole-shot
  bootstrap intervals/counts, explicit frozen checkpoint/threshold hashes,
  shot/cohort/day overlap checks, and five shot-grouped Smith-only onset folds.
  No review or cohort test shots enter Smith training. Existing cv2 is the frozen
  five-model ensemble; thresholds are not changed on Smith. The ensemble averages
  probability-to-original-threshold ratios and calls at one. Occupancy uses whole
  1 ms cells; this differs from the reviewed 50 ms crowd target. Physical events
  retain every hand onset in their denominator, including window-edge cases.
- A single malformed Smith source (179859) has repeated acquisition time axes.
  Refetch reproduced the defect; the first complete monotone acquisition is an
  explicitly audited derivative, with finite aligned FS/IF channels, source,
  refetch and derivative hashes. No arbitrary sorting mixes acquisitions.
  Frozen inference was rerun and all Smith folds restarted cleanly after archiving
  the aborted fit. All 2,316 windows have complete diagnostic-record coverage.
  The 119 review inputs and their checkpoints/thresholds were unchanged.
- Native DSM risk(t) now uses any reviewed present in (t,t+h], with unknown-time
  coverage excluded, and non-crowd starts inside the same forward horizon on
  per-ELM portions. Crowd/unknown portions cannot become onset controls. The
  exact-export n=4 source-exposed result remains only in native JSON, descriptive
  without CI; no paper-facing exact-export panel is retained.
- Real PCPHD02/03 cover all 119 shots each (111 fetched/cache and eight upstream),
  with no filterscope substitutes. Fetching completed without auth errors.
  DENV2F/3F replace CO2 v2/v3; each is valid on 115 shots, with four fixed-rule
  bad-digitiser rejections per chord. The rejected sets differ. Slow R0/V1 remain
  unavailable on 75 shots. Paired 44-shot 50 ms means agree numerically to about
  1e-5, but fast source metadata says V while the training CO2 contract says
  cm^-2; no physical unit conversion is claimed. One unchanged 40-epoch,
  randomly initialized isolated detector recipe was refit with optimizer-shot
  normalization only. Every saved OOF score reproduces exactly from checkpoints.
- `score`, `methods` and `swap` now record finite/undefined bootstrap draws for
  each metric and suppress population CIs below five positive-bearing shots.
  The two-shot legacy AUROC 0.993 has 763 finite / 237 undefined draws and no CI.
  Cached per-shot count matrices speed paired hard metrics without changing draws
  or results; regression tests compare explicit pooling draw by draw.
- Finding 2 now includes 782 known all-covered bins (724 BES bins), and identical
  detection-covered subsets of 779/721. Three final bins lack detection rows on
  190637, 192751 and 196541; no scores are extrapolated. Complete-score AUROC orders
  are unchanged, while fixed review-tuned F1 point orders can cross. The evidence
  remains inconclusive. Review non-crowd tau merges (100/200/300 ms), without
  bridging missing coverage, are a fixed-prediction definition sensitivity.
- Clock boundary identity is 48/86 starts (56%), 38/86 ends (44%), 28/86 both
  (33%); 27/86 match both ends of the same clock span. The protocol and captions
  disclose this dependence. Every ELM caption, notes and protocol state that DSM
  and legacy annotations were built on WPQH breakthrough-ELM targets, so Finding 1
  and low historical DSM AUROCs partly reflect definition/domain shift (192721:
  one legacy bin versus 17 non-crowd reviewed spans).
- Main output is six core rows (elm-ours, elm-elmo, elm-dsm detection, elm-clock,
  always-present, feature-only) and two identical-coverage panels. All remaining
  numeric tables are consolidated in the appendix. README Models is model lines
  plus one caveat paragraph; names follow task-model. D-alpha morphology is
  qualified by diagnostic view, ELM type and detachment. The model-card BEATS
  claim was replaced by the fact that incomparable validation sets do not
  establish an input-set advantage. Docs contain current state, not fix history.
- Current record git fields identify the current code/record audit and retain
  original generation/training provenance; this does not retroactively claim
  unchanged training sources. `provenance.json` hashes current code and records.
  Duplicate DSM evaluations and obsolete tables are archived externally with
  paths/hashes, rather than duplicated in git.

Input/DSM sources: `dsm/detection_input_audit.json`, `dsm/detection_input_fetch.json`,
`dsm/detection_input_comparison.json`, `dsm/reproducibility.json`,
`dsm/native_evaluation.json`; reference/clock/tau sources: `swap/evaluation.json`
and `ours/annotation_strata.json` (`clock_boundary_identity`,
`review_non_crowd_merge_sensitivity`). Every quantitative result below comes
directly from those records.

### Independent Smith window and event results

| Method | Occupancy AUROC | Occupancy AUPRC | Occupancy F1 |
|---|---|---|---|
"""
    for name, row in smith["methods"].items():
        res = row["occupancy_1ms"]
        text += (
            f"| {name} | "
            + " | ".join(metric(res, k) for k in ("auroc", "auprc", "f1"))
            + " |\n"
        )
    text += (
        "\n| Method | Tolerance ms | Precision | Recall | F1 | "
        "Median error / absolute error ms | p05–p95 ms |\n"
        "|---|---:|---|---|---|---|---|\n"
    )
    for name, row in smith["methods"].items():
        for tolerance, res in row["events"].items():
            timing = res["timing_error_ms"]
            text += f"| {name} | {tolerance} | " + " | ".join(
                metric(res, k) for k in ("precision", "recall", "f1")
            )
            quantiles = timing["quantiles"]
            text += (
                f" | {timing['median']} / {timing['median_absolute']} | "
                f"{quantiles.get('p05')}–{quantiles.get('p95')} |\n"
            )
    text += (
        "\nEvent timing errors are prediction minus hand onset, on matched "
        "events only; full quantiles, mean, counts, CIs and valid/undefined "
        "draw counts are in `smith/evaluation.json:methods.<method>.events`.\n"
    )
    text += (
        "\nELM-O 0.997/0.980 was the prior local 2,316-window overlap score "
        "(TP 2,269 / FP 6 / FN 47); the paper digest's fixed-setting table "
        "reports 0.995/0.976 on 972 tuning ELMs. Corrected first-acquisition "
        "inputs change the local overlap counts. Current corrected overlap "
        "record:\n\n```json\n"
        + json.dumps(smith["reimplemented_elmo_overlap"], indent=1)
        + "\n```\n\nThis overlap score is distinct from ±2/5 ms onset matching. "
        "Smith was previously used for ELM-O, and its historical tuning "
        "overlap is unresolved; independence is established for the frozen "
        "review-trained model, not ELM-O tuning.\n"
    )
    text += (
        "\n### Native forward targets and DSM repair comparison\n\n"
        "| Native target | Horizon | Shots / rows | AUROC |\n|---|---|---|---|\n"
    )
    native = read("dsm/native_evaluation.json")["reviewed_reconstructed"]
    for target in ("horizons", "non_crowd_start_horizons"):
        for horizon, row in native[target].items():
            ci = row["auroc_ci95"]
            value = f"{row['auroc']:.3f}" + (
                f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
            )
            text += (
                f"| {target} | {horizon} | "
                f"{row['n_shots']} / {row['rows']:,} | {value} |\n"
            )
    comparison = read("dsm/detection_input_comparison.json")
    text += (
        "\nOld/new comparison schema and archived baseline paths are preserved "
        "in `dsm/detection_input_comparison.json`; all 119 old AUROC/F1 "
        "0.854934/0.753635 versus repaired 0.844655/0.742399. Input removal "
        "alone did not explain the original gap. The native 124-input "
        "source-selected model, source survival refit and review detector "
        "evaluate different targets/splits and should not be directly ranked.\n"
    )
    text += (
        "\n### GPU sensitivity, figures and verification\n\nSmoothed selection "
        "uses a trailing three-epoch mean of inner-validation AUPRC, with all "
        "three epochs after four warm-up epochs, saving the endpoint weights "
        "and that endpoint's inner-validation F1 threshold. Original "
        "outer/inner shot partitions, seed and recipe are checked identical. "
        "Selected zero-based epochs and thresholds:\n\n```json\n"
        + json.dumps(smooth["selections"], indent=1)
        + "\n```\n"
    )
    large = Paths.from_env().root / "round4/elm"
    peak_bytes = smooth["gpu"]["max_allocated_bytes"]
    text += (
        "\nOne five-fold GPU run used CUDA_VISIBLE_DEVICES=1 with an allocator "
        f"cap below 9.2 GB; peak allocated memory was {peak_bytes:,} bytes. "
        f"Sources: `{large}/smoothed_train.log`, `ours/smoothed_selection.json` "
        f"and `{large}/cv/cv2-smoothed/run.json`. tmpsweep exit 0 after GPU "
        f"completion logged 622 MB→622 MB in `{large}/smoothed_tmpsweep.log`; "
        "Smith aborted/restarted/final GPU sweeps are also recorded. No full "
        "suite, installation, push, merge, rebase or production-store write "
        "was performed.\n"
    )
    pages = read("table_render_verification.json")["png_page_count"]
    text += (
        f"\nFigure `{large}/figures/fig_elm_examples.pdf` and 150-dpi `.png` "
        "were inspected. Panel (b) uses next eligible F1 rank, shot 203941, "
        "replacing 200427; its trace shows burst activity. The selection/"
        "window/source/threshold/inspection record is "
        f"`{large}/figures/fig_elm_examples.json`. Tables and notes compile to "
        f"{pages} inspected PNG pages with zero warnings; {len(tables['tables'])} "
        "tables replace the previous 35. Source/render hashes are in "
        "`tables.json`, `table_render_verification.json` and `provenance.json`.\n"
    )
    for key, step in verify["checks"].items():
        text += (
            f"\n{key}: exit {step['exit_code']}\n\n```text\n"
            + step["output_tail"]
            + "\n```\n"
        )
    text += (
        "\nVerification assertions:\n\n```json\n"
        + json.dumps(verify["assertions"], indent=1)
        + "\n```\n"
    )
    text += (
        "\n### Deviations, concerns and next work\n\nSmith bins are 1 ms rather "
        "than 50 ms because individual hand windows are short; both model "
        "recipes and operating points are identified and frozen before "
        "scoring. Smoothed reselection required a new five-fold replay because "
        "non-selected weights were not saved; it remains a separate "
        "sensitivity, not a retroactively replaced primary result. The "
        "requested 0.997/0.980 ELM-O reference is correctly labelled as prior "
        "reimplementation window-overlap evidence rather than a published "
        "onset score. The one-shot acquisition repair is explicit and all "
        "methods share it.\n\nReview targets remain clock-dependent and crowd "
        "dominated; shot CV shares run days and is developmental. Smith CV has "
        "selected-window negatives and is grouped by shot, not day. "
        "Independent transfer and onset performance must be read with those "
        "target definitions; no catalog physical-onset release is claimed. "
        "DSM calibrated density units and continuous-discharge onset false-"
        "alarm validation remain unresolved. Eight-shot reference-swap "
        "evidence is inconclusive; tiny subsets have no population CI. Next "
        "work would validate the frozen onset recipe on continuous "
        "independently adjudicated discharges grouped by run day and obtain "
        "source calibration metadata.\n"
    )
    log = subprocess.check_output(
        ["git", "log", "--reverse", "--format=%h %s", "8e166437..HEAD"],
        cwd=REPO,
        text=True,
    )
    text += "\n### Commits\n\n```text\n" + log + "```\n"
    # Keep the whole history outside docs; replace only a prior copy of this section.
    existing = REPORT.read_text()
    if "## Fix round 4\n" in existing:
        existing = existing.split("## Fix round 4\n", 1)[0]
    assert verify["passed"] and comparison
    assert swap["all_covered_swap"]["overlap"]["reviewed"]["bins"] == 782
    REPORT.write_text(existing.rstrip() + "\n\n" + text)
    print(REPORT)
    print("headline", metric(primary, "auroc"), metric(detector, "auroc"))


if __name__ == "__main__":
    main()
