#!/usr/bin/env python
"""Append the authoritative fix-round-three report from committed ELM records."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from elm_protocol import benchmark_rows, kind_rows, metric, mode_rows, own_target_rows

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/labeler/elm"
REPORT = Path(
    "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/claude-89242e53/r4/reports/elm.md"
)


def native_rows(native):
    lines = [
        "| Native panel | Horizon ms | Shots / rows | AUROC (95% shot CI) |",
        "|---|---|---|---|",
    ]
    for panel in ("reviewed_exact_export", "reviewed_reconstructed", "own_target"):
        for horizon, result in native[panel]["horizons"].items():
            low, high = result["auroc_ci95"]
            lines.append(
                f"| {panel}‡ | {horizon} | {result['n_shots']} / {result['rows']:,} | "
                f"{result['auroc']:.3f} [{low:.3f}, {high:.3f}] |"
            )
    return "\n".join(lines)


def main():
    records = {
        key: json.loads((OUT / path).read_text())
        for key, path in {
            "ours": "ours/evaluation.json",
            "dsm": "dsm/evaluation.json",
            "strata": "ours/annotation_strata.json",
            "swap": "swap/evaluation.json",
            "native": "dsm/native_evaluation.json",
            "fetch": "dsm/native_fetch.json",
            "verification": "fix_round3_verification.json",
            "tables": "tables.json",
            "renders": "table_render_verification.json",
            "filterscopes": "filterscope_metadata.json",
        }.items()
    }
    ours, detector, _strata, native = (
        records[key] for key in ("ours", "dsm", "strata", "native")
    )
    primary = ours["sets"]["all119"]["methods"]["elm-ours"]
    clean = detector["sets"]["all119"]["methods"]["elm-dsm-detect"]
    positive = primary["counts"]["tp"] + primary["counts"]["fn"]
    fraction = primary["counts"]["crowd_bins"] / positive
    text = f"""## Fix round 3

Status: DONE_WITH_CONCERNS. This section is the current result; earlier sections
describe superseded historical states. Both third re-reviews were read in full.

The reviewed target is ELMy-phase occupancy: {100 * fraction:.2f}% of the
{positive:,} present bins are crowds. Primary elm-ours remains AUROC
{metric(primary, "auroc")} and F1 {metric(primary, "f1")}. The corrected,
source-isolated 40-epoch DSM detection baseline scores AUROC
{metric(clean, "auroc")} and F1 {metric(clean, "f1")} on the common 119-shot
11,653-bin panel. Independently adjudicated ELM onset precision/timing remain
unavailable, and the onset head is explicitly withdrawn.
Sources: `outputs/labeler/elm/ours/evaluation.json:sets.all119.methods.elm-ours`
and `outputs/labeler/elm/dsm/evaluation.json:sets.all119.methods.elm-dsm-detect`.

### What changed

- `methods.span_counts` intersects detected intervals with the specific panel's
  analysed coverage before raw and guarded span metrics. Regression tests cover
  the reviewed 203941 case, disconnected coverage and empty coverage. All
  original primary bin metrics remain unchanged, verified numerically.
- `methods.annotation_summary` and `kind_summary`, the occupancy evaluator and
  new `elm_annotation_strata.py` report physical-shot-grouped per-kind/mode
  confidence intervals, complete shot lists and annotation-count definitions.
- `dsm.shot_rows(..., None)` builds raw rows without upstream statistics or
  clipping; fold-local measured-column normalization and independently seeded
  weights isolate the confirmatory detector. Missing columns are filled only
  at its training-partition mean. Upstream-normalized historical scratch and
  source-initialized fits remain explicitly supplemental; ‡ marks source
  fitting/selection exposure wherever relevant.
- `elm_dsm_native.py` selects the original 124-input checkpoint by matching
  the shipped export weights, evaluates exact original 1 ms inputs separately,
  and adds a raw-input reconstruction sensitivity with exact missing-column,
  native domain-filter, reconstruction-agreement and source-exposure audits.
- `elm_reference_swap.py` includes isolated and exposed detection scores,
  preserves review-tuned operating points and limits cross-reference comparisons
  to AUROC. All legacy overlaps and the inconclusive conclusion are visible.
- Table generators use <=80-word table-specific captions, no filenames or
  person names in captions, one `elm_table_notes.tex`, source-exposure markers,
  and wider swap tables with P/R and 200 ms results in companions. Every table
  is compiled standalone and rendered for visual inspection.
- `elm_protocol.py` regenerates the current-state protocol and README. Models
  lines restore dates, separators and prior slugs; registry model-index naming
  is restored. The report and verification have dedicated reproducible scripts.

### Data, splits and stratified results

The fixed cohort remains 400 train / 50 val / 50 blind test. The 119 reviewed
shots are 69 train / 50 val. No blind-test shots enter new U-Net or confirmatory
DSM fitting, preprocessing or selection. Five fixed shot-grouped outer folds
reserve 14 inner-validation shots each. All intervals use 1000 physical-shot
bootstrap draws; annotation-mode groups resample within the group. The clock
seeded D-alpha review. U-Net inputs share D-alpha with those labels; adapted
DSM is D-alpha-free, so input choice and architecture effects are inseparable.
elm-ours hard calls use bin-mean >= threshold; ELM-O/clock use any-touch.

"""
    text += kind_rows(ours) + "\n\n" + mode_rows(ours)
    text += """

Sources: `ours/evaluation.json:sets.<panel>.per_kind,annotation_modes` and
`ours/annotation_strata.json:sets`. There are 56 crowd-only, 13 non-crowd-only,
20 mixed and 30 no-present shots, accounting for all 119. Of the no-present
shots, 29 contribute scored bins; 194600 has absent/uncertain labels and no
scored bins. Thus 33 non-crowd and 76 crowd shots overlap on 20 shots.

Exact coverage-clipping changes, source `ours/annotation_strata.json:
coverage_clipping_audit`:

- Common BES: ELM-O/190637 absent alarm 1 -> 0; U-Net/194155 raw and guarded
  absent alarm 1 -> 0; clock/203941 non-crowd hit 1 -> 0.
- Common all119: clock/192229 absent alarm 1 -> 0; clock/203941 non-crowd
  hit 1 -> 0. Its 5990–6012 ms detection lies after coverage ends at 5975 ms.

### DSM adapted and native results

"""
    text += benchmark_rows(detector) + "\n\n" + own_target_rows(detector)
    text += """

Sources: `dsm/evaluation.json:sets,detectors,model_context,own_target`.
The limited survival refit is a one-epoch checkpoint selected after the first
epoch of a seven-epoch run. Its own-target results use early-stopping validation
rows, not untouched testing. The isolated 40-epoch retrain is the fair adapted
detection baseline. Historical source exposure is retained only in supplemental
rows, rather than certified as isolated through disclosure.

"""
    text += native_rows(native)
    text += """

Native sources: `dsm/native_evaluation.json` and `dsm/native_fetch.json`.
Exact native exports are the faithful original-input comparison; reconstructed
rows are a separately labelled sensitivity because original NBI smoothing
concatenated filtered phase rows across shot/phase boundaries. Full per-shot
coverage, missing original columns and domain exclusions are in `coverage`;
every native panel uses 124 columns at 1 ms, without mean filling or clipping.
Native own-target rows are the original checkpoint's early-stopping validation
with the source's reversed train/validation split, never a new untouched test.

Fetching used the login node, prescribed `fdp run`, one worker and pace 1.
Native photodiodes were fetched only for eligibility-screened complete-input
shots. No authentication error occurred. Stores under production roots were
read-only; new arrays/checkpoints/signals remain under `round4/elm`.

### Swap findings

Sources: `swap/evaluation.json:overlap,interval_audit,interval_audit_occupancy,
swap.<panel>.comparison`. The onset table has 576 shots / 8 reviewed overlap;
shot-level ground truth 365 / 5, all yes (192751 has no reviewed present span);
independent Smith BES windows 211 / 0. On 782 known bins the legacy reference misses
60 reviewed-present bins (31%) as onset bins, versus 34 (17%) with 200 ms
occupancy; corresponding P counts are 14 and 60. Historical full-overlap AUROC
order and leader remain unchanged under all conversions, with no AUROC
reversal observed. F1 uses thresholds tuned to review, so it cannot provide a
cross-reference ranking comparison. Eight shots cannot show or rule out a
reversal; source-unexposed subsets of three/two shots are still less precise.
The main swap caption explicitly states the evidence is inconclusive.

The added isolated detector preserves the primary eight-shot AUROC order, but
its point order with ELM-O crosses on the seven-shot BES companion under
100/200/300 ms occupancy conversions. Leadership remains unchanged. This new
point crossing is explicitly reported; it does not establish a supported
AE-style reversal and is not erased by historical-family invariance.

### Metadata, checkpoint selection and open limits

`filterscope_metadata.json` verifies FS02/03/04 map to PMT11/12/13. Accessible
tree metadata expose signal/calibration nodes and no sightline/location field;
each divertor/midplane view remains unverified. FS01 has finite corpus samples
on 104/119 shots, but is outside the existing three-channel native-rate input
cache and retained fits. Its exclusion was a pipeline choice, not universal
absence. Prior fetch attempts returning invalid-expression errors were corrected
by reading tree nodes directly; explicit tree cleanup resolved wrapper teardown
errors. Final metadata fetch completed with exit 0 and no auth errors.

`ours/annotation_strata.json:checkpoint_selection_audit` records noisy original
selection: epoch 2 during warm-up in fold 3, a fold-4 single-epoch spike,
thresholds 0.108826–0.911287. A prespecified trailing-three mean after warm-up
changes 15/20 candidate endpoints. Only selected weights were saved; smoothed
evaluation needs all twenty folds retrained, with observed training alone
0.945 GPU-hours. It is not a cheap checkpoint rescore, so instability is
reported and no smoothed-result improvement is claimed. All four seeds remain
reported with unchanged aggregate bin scores.

Onsets remain incomplete, native exact-input coverage is limited and source
exposed, reconstructed-input native evaluation is a sensitivity, fast-density
physical units remain unverified, and run-day/development/input-coupling effects
remain. Next work: independently adjudicate individual onsets and absent spans,
recover sightline metadata, add source-unexposed native coverage, then prespecify
smoothed-checkpoint and run-day validation. No external reviewer rating is claimed.

### Verification and artifacts

Source: `outputs/labeler/elm/fix_round3_verification.json`.

"""
    verify = records["verification"]
    for key in ("tests", "ruff", "new_file_format"):
        result = verify[key]
        text += f"{key}: exit {result['exit_code']}\n\n"
        text += "```text\n" + " ".join(result["command"]) + "\n"
        text += result["output"][-4000:] + "\n```\n\n"
    text += (
        "Primary bin metrics unchanged: "
        f"{verify['primary_bin_metrics_unchanged']['all_unchanged']}. "
        "Current source/document/table/render hashes verified: "
        f"{verify['artifact_hashes_current']['all_current']}. "
        "Only covering test files ran; no full suite. Source/table hashes and "
        "caption word counts are in `tables.json`; standalone compile, image "
        "inspection and render hashes are in `table_render_verification.json`. "
        "Large artifacts, including "
        "each standalone PDF and 150-dpi PNG, are under "
        "`$LABELER_ROOT/round4/elm/table-proofs`. All 35 tables and the shared "
        "appendix note compiled twice and were inspected as rendered images.\n\n"
    )
    text += (
        "Initial delegated read-only inspections omitted TMPDIR before the worker "
        "read the binding rules; they did not intentionally create temporary "
        "outputs. Subsequent commands used the prescribed temporary directory.\n\n"
    )
    commits = subprocess.check_output(
        ["git", "log", "--reverse", "--format=%h %s", "aa4ccbf2..HEAD"],
        cwd=REPO,
        text=True,
    )
    text += "### Commits\n\n```text\n" + commits + "```\n\n"
    text += (
        "All commits use labeler: and the requested Codex co-author trailer. No push.\n"
    )
    prior = REPORT.read_text().split("\n## Fix round 3\n", 1)[0]
    REPORT.write_text(prior.rstrip() + "\n\n" + text)
    print(REPORT)


if __name__ == "__main__":
    main()
