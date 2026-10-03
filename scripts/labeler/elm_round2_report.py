#!/usr/bin/env python
"""Append the ELM round-two report from canonical outputs and validation evidence."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

from labeler.config import Paths

REPO = Path(__file__).resolve().parents[2]
OUTPUTS = REPO / "outputs/labeler/elm"
REPORT = Path(
    "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/claude-89242e53/r4/reports/elm.md"
)


def main():
    spec = importlib.util.spec_from_file_location(
        "elm_protocol", REPO / "scripts/labeler/elm_protocol.py"
    )
    render = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(render)
    paths = {
        "ours": OUTPUTS / "ours/evaluation.json",
        "dsm": OUTPUTS / "dsm/evaluation.json",
        "swap": OUTPUTS / "swap/evaluation.json",
        "seeds": OUTPUTS / "ours/seed_repeats.json",
        "verification": OUTPUTS / "fix_round2_verification.json",
        "records": OUTPUTS / "round2_records.json",
        "protocol": OUTPUTS / "protocol.json",
    }
    records = {key: json.loads(path.read_text()) for key, path in paths.items()}
    if not records["verification"]["passed"]:
        raise ValueError("cannot publish report with failed verification")
    commits = subprocess.check_output(
        ["git", "log", "--reverse", "--format=%h %s", "eb4e3630..HEAD"],
        cwd=REPO,
        text=True,
    ).strip()
    text = """

## Fix round 2

Status: DONE_WITH_CONCERNS. Both re-reviews were read in full. Every Important
and Minor item was addressed with the choices below; density physical units
remain unverified because permitted offline sources lack the required metadata.
No external re-review rating is claimed. The independent read-only code review
found no Important scientific/mathematical defect; its two minor issues
(nonfinite predictions and compiler attribution) were fixed.

### Implementation and choices

1. `elm.swap.occupancy_table` joins positive legacy bins across covered gaps
   ≤100/200/300 ms, never missing coverage. `elm_reference_swap.py` recomputes
   all-covered and strict-bin Finding 1, reference rankings/paired reversals
   (Finding 2), and original-source-unexposed three-shot/two-BES-shot subsets.
   Predictions, calls and previously chosen thresholds remain fixed; hashes
   and per-shot thresholds are recorded in `swap/evaluation.json:
   fixed_prediction_provenance`.
2. `elm.dsm`, DSM spec/card and authoritative physical membership disclose every
   variant's upstream feature-statistics exposure to blind-cohort 190646/190532.
   We chose disclosure rather than new preprocessing fits. Every caption says
   no D-alpha input (pcphd02/03 mean-filled), CO2 missing on 75/119, source
   survival training at 1 ms versus 50 ms-mean serving, and offline risk score
   with 25 ms centered-NBI lookahead (not a causal forecast). Detection heads
   refit the reviewed mean rows. Adapter `training_shots` includes all source
   exposure roles, preventing exposed shots from being called held out.
3. Tables retain numeric F1 with intervals, † high-recall markers and P/R,
   always-present baselines, full short method names and named All/BES panels.
   The benchmark is now committed. Legacy annotation authorship is separated
   from the `labels_format.py`/category `source_formatters` compiler.
4. `methods.span_counts` and `score.rates` add a 25 ms edge guard beside raw
   absent-span touch rate. The same-denominator result and nonempty-interior
   conditional result are both present, with empty-interior counts. Centered
   smoothing and short per-ELM gaps can trigger raw boundary touches. The
   protocol discloses 33 per-ELM/non-crowd shots versus 76 crowd-annotated shots.
5. `train.train_fold` raises a clear error for no finite validation AUPRC;
   scoring refuses NaN/±Inf predictions so sorting cannot make a divergent
   checkpoint appear valid. Three new full CV seeds preserve outer and inner
   physical-shot partitions/hyperparameters. The onset head is dropped from
   paper outputs, with no onset retraining; existing auxiliary loss remains.
6. Input-unit provenance is retained in native-unit metadata sidecars, with
   no speculative rescaling. All 119 prepared input hashes remain unchanged.
   The source cache discarded fast ordinate units, accessible staged fast H5
   sources were absent, and companion slow-CO2 units cannot establish fast
   units. `density_units.json` records evidence and explicit unverified status.
   Superseded DSM snapshots are archived outside git with hashes; one current
   consolidated evaluation contains all three variants.
7. The protocol and README Models are current-state descriptions. The revised
   figure uses distinct non-crowd shading and clock bars, a compact legend,
   occupancy only and per-panel fold thresholds. PDF/150 dpi PNG and table
   proof pages were rendered and visually inspected.

### Data, current metrics and source records

The fixed cohort and reviewed split counts, model size and source hashes are
in `outputs/labeler/elm/protocol.json:metadata`. Every shot/fold/bin count is
recorded in the evaluation or seed JSONs. Production stores and labels were
read only; no fetching or manuscript edits occurred. New training/selection
excludes cohort test shots; inherited DSM source exposure is explicit.

Primary reviewed bins:

"""
    text += render.benchmark_rows(records["ours"])
    text += (
        "\n\nSource: `outputs/labeler/elm/ours/evaluation.json:sets.<set>.methods`.\n"
    )
    text += "\nCommon DSM bins:\n\n" + render.benchmark_rows(records["dsm"])
    text += (
        "\n\nSource: `outputs/labeler/elm/dsm/evaluation.json:sets.<set>.methods`.\n"
    )
    text += "\nRaw and guarded absent-span rates:\n\n" + render.alarm_rows(
        records["ours"]
    )
    text += "\n\nSource: `ours/evaluation.json:sets.<set>.methods.<method>.{point,ci95,counts}`.\n"
    text += "\nAll-covered Finding 1:\n\n" + render.audit_rows(records["swap"])
    text += """

Source: `swap/evaluation.json:interval_audit` and `interval_audit_occupancy`.
Onset M/P 60/14 becomes 34/60 at 200 ms occupancy; recall improves while
reviewed-absent legacy-positive bins increase. Counts are definition-dependent,
not adjudicated physical omissions. Unknown review bins remain excluded.

Strict-bin Finding 1 and Finding 2 rankings:

"""
    text += render.ranking_rows(records["swap"])
    text += """

Source: `swap/evaluation.json:swap.<set>.finding_1`, `comparison` and
`occupancy.gap_<tau>ms`. Full-overlap AUROC order/leader stays unchanged at all
tolerances. F1 reversals vary with the reference: on All-8, clock/refit reverses
under onset and 100 ms occupancy, but not 200/300 ms. BES-7 refit passes ELM-O
by point F1 at 200 ms. No significant rank change is established by these
point orders. Broad shot intervals and tiny overlap limit conclusions.

Five of eight overlap shots were DSM source training shots: 190637,190643,
192721,192751,196541. Captions mark refit/initialized embedding in-sample.
The three-source-unexposed shots are 189885,192732,200385, with a two-shot BES
subset; their complete rows/references are `swap.overlap_dsm_heldout` and
`swap.overlap_bes_dsm_heldout`. They also lie outside normalization membership,
but the models retain blind-cohort source-statistics exposure. On these tiny
subsets, some AUROC orders change; full-overlap invariance is not generalized.

### Training seed repeats

"""
    text += render.repeat_rows(records["seeds"])
    text += """

Source: `outputs/labeler/elm/ours/seed_repeats.json:{results,seed_ranges}`.
The three new seeds change training randomness only, with original fixed
shot lists. Every run has individual 95% shot-bootstrap intervals, selected
epochs, fold thresholds, full validation histories, source snapshots and
large-artifact paths/hashes. Original and repeats all remain reported; no
best-seed selection. Seed spread is not a confidence interval. Source
snapshots distinguish the added nonfinite guard; finite metric arithmetic is
unchanged across runs. Original cv2 provenance remains retrospective.

### Artifacts, tests and lint

Committed benchmark: `outputs/labeler/elm/table_elm_benchmark.tex`.
Committed swap artifacts under `outputs/labeler/elm/swap/` include the main
review/onset/200 ms table, all sensitivity/full-P/R tables, ranking tables and
three-shot DSM subset rows. `tables.json` records every source/table hash.
`round2_records.json` verifies original metrics/intervals against eb4e3630 and
records archived snapshot paths/hashes outside git.

Large artifacts:

"""
    large = Paths.from_env().root / "round4/elm"
    for item in (
        "figures/fig_elm_examples.pdf",
        "figures/fig_elm_examples.png",
        "figures/fig_elm_examples.json",
        "figures/tableproof.pdf",
        "cv/cv2_seed20261004",
        "cv/cv2_seed20261005",
        "cv/cv2_seed20261006",
        "seed_repeats",
        "inputs",
        "archive/fix_round2",
    ):
        text += f"- `{large / item}`\n"
    text += (
        "\nVerification source: `outputs/labeler/elm/fix_round2_verification.json`.\n"
    )
    for key in ("tests", "ruff", "new_file_format"):
        record = records["verification"][key]
        text += (
            f"\n{key}: exit {record['exit_code']}; command `{record['command']}`\n\n"
            + "```text\n"
            + record["output"][-1600:].strip()
            + "\n```\n"
        )
    text += """

Only covering test files ran; no full suite. Model repeats used assigned GPU 0
sequentially, modest memory and at most eight CPU threads. CPU analyses used
mandated frozen/no-install Pixi, the stream worktree/PYTHONPATH and TMPDIR.
Temporary sweep ran after long jobs. No push, merge or rebase.

### Commits and remaining concerns

"""
    text += "```text\n" + commits + "\n```\n"
    text += """

All commits use `labeler:` and the requested Codex co-author trailer.
Remaining concerns: fast-density physical units cannot be verified from the
permitted offline metadata; the eight-shot legacy overlap and two/three-shot
DSM source-unexposed subsets are imprecise; DSM feature-statistics exposure,
heterogeneous occupancy labels, development/run-day dependence and GroupNorm
scope remain scientific limitations. External Opus/Sol ratings require the
controller's actual re-review; no ≥8 score is represented as achieved here.

Next work: retain source ordinate metadata in an authorized fast-signal fetch,
expert adjudication of absent spans/onsets, more reviewed legacy-overlap shots,
source-isolated DSM preprocessing/training and run-day-grouped validation.
"""
    old = REPORT.read_text()
    marker = "\n\n## Fix round 2\n"
    if marker in old:
        old = old[: old.index(marker)]
    REPORT.write_text(old.rstrip() + text)
    print(REPORT)


if __name__ == "__main__":
    main()
