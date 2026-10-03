#!/usr/bin/env python
"""Put the reproducible fix-round-three state first and archive prior reports."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from detach_json import dumps

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ["LABELER_ROOT"]) / "round4/detach"
RESULTS = REPO / "docs/labeler/results"
DEFAULT_REPORT = (
    Path(os.environ["LABELER_ROOT"]) / "scratch/claude-89242e53/r4/reports/detach.md"
)
ARCHIVE = "\n## Archived previous reports\n"


def load(name):
    return json.loads((RESULTS / f"detachment_{name}.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--start-commit", default="71cc3cda")
    args = parser.parse_args()
    r, b, reference, validation = map(
        load, ("round3", "benchmark", "reference", "validation")
    )
    current = r["current"]
    pop, certain = current["assessed"], current["certain"]
    composition = r["composition"]["by_state"]
    paper = b["paper_agreement"]
    end = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True
    ).strip()
    lines = [
        "# Detachment stream — current state",
        "",
        "## Fix round 3",
        "",
        (
            f"**DONE_WITH_CONCERNS** — `{args.start_commit}..{end}` on "
            "`r4-detach`. Paper scope: exploratory coverage and indicator "
            "agreement only; **no independent benchmark**. Expert "
            "adjudication was unavailable while the owner was away."
        ),
        "",
        (
            f"Current export: {pop['bins']:,} assessed bins/{pop['shots']} shots; "
            f"{certain['bins']} certain bins/{certain['shots']} shots, "
            f"{certain['seconds']:.2f} seconds. Certain composition: "
            f"{composition['attached']['bins']} attached, "
            f"{composition['detached']['bins']} detached, "
            f"{composition['marfe']['bins']} MARFE; "
            f"{composition['uncertain']['bins']:,} assessed bins uncertain. "
            "No certain attached-to-detached transition. MARFE means "
            "single-shot MARFE candidates within threshold uncertainty on "
            "199172. Sources: `detachment_round3.json` and "
            "`detachment_benchmark.json` under worktree `docs/labeler/results/`."
        ),
        "",
        (
            "Certainty is an upper-shelf TangTV vote with f_div corroboration. "
            "Afrac supplies no upper-shelf SOL probe votes; Snorkel is "
            "vestigial. Figure 2 contains coverage and upper-shelf Prad–TangTV "
            "κ with shot-bootstrap CI; F1-vs-consensus rows are removed. "
            "The compared Prad votes are constant detached, giving κ=0. "
            "This is indicator agreement, not physical accuracy."
        ),
        "",
        "### Repairs and files",
        "",
        (
            "`core.py`, `signals.py`, `prad.py`, `thresholds.py`: preserve "
            "D-alpha/beam sample availability, centered 250 ms radiation and "
            "heating means, full averaging-window D-alpha coverage, "
            "native-bin and averaged radiation offset gates. Uncovered ELM "
            "windows veto; uncovered heating is not zero. −0.05 MW is a "
            "documented operational offset tolerance, not calibrated "
            "uncertainty. Chen's ~50 ms Prad lead is context, not the "
            "averaging prescription. Both Prad,div/P_in and Prad,div/Prad "
            "diagnostics use averaged signals. The unsourced "
            "AFRAC_MIN_MS=3000 restriction is removed."
        ),
        "",
        (
            "`detach_bins.py`, `detach_refresh_prad.py`, "
            "`detach_regenerate_widths.py`, `detach_label.py`: regenerated "
            "bins, labels, intervals, grids and traces; non-test sensitivity "
            "grids use the same extraction. `detach_benchmark.py`, "
            "`detach_round3_records.py`: tier-specific agreement, composition, "
            "measurement audit and threshold sensitivity. "
            "`detach_reference.py`, `detach_ours.py`, `detach_victor.py`: "
            "bootstrap reference classes frozen before shot resampling; "
            "undefined missing-support class F1 and macro-F1 draws counted. "
            "CNN inputs omit p_in_mw; majority comparisons use paired draws, "
            "with confusion matrices and per-class shot support. Neither "
            "model learns beyond majority; MARFE transfer is unsupported."
        ),
        "",
        (
            "`detach_figure.py`, `detach_paper_panel.py`: coverage/agreement "
            "panel, no empty Jsat timeline, masked dead lower-fan chord 11, "
            "honest uncertain views column (a), separately identified "
            "199172 MARFE witness beside missed 199166 onset with flux "
            "contours/cue values. `detach_protocol.py`, "
            "`detach_docs_tables.py`: current exploratory docs/README without "
            "paper-facing review history or empty tables; MAE to three "
            "decimals. `detach_handoff.py`: per-shot availability and "
            "detach-ui paths/schema contract."
        ),
        "",
        "### Results and sources",
        "",
    ]
    for name in (
        "normalization_change",
        "measurement_gate_audit",
        "afrac_absence",
        "threshold_derivation",
        "threshold_sensitivity",
        "eligibility",
    ):
        lines += [
            f"#### {name} — detachment_round3.json",
            "",
            "```json",
            dumps(r[name], indent=1),
            "```",
            "",
        ]
    lines += [
        "#### Upper-shelf agreement — detachment_benchmark.json",
        "",
        "```json",
        dumps(paper, indent=1),
        "```",
        "",
        (
            "Certain-set changes include all gate and normalization repairs, "
            "not an isolated smoothing ablation. Previous attached support "
            "was single-bin snippets on three shots; none survives. Every "
            "agreement table is stratified by tangtv_tier. Threshold "
            "sensitivity is descriptive and did not select a threshold."
        ),
        "",
    ]
    for name in ("ours", "victor"):
        model = load(name)
        lines += [
            f"#### CNN/majority — detachment_{name}.json",
            "",
            "```json",
            dumps(
                {"model_cv": model["cv_shots"], "majority_cv": model["cv_majority_ci"]},
                indent=1,
            ),
            "```",
            "",
        ]
    manual = reference["manual_front_check"]
    lines += [
        (
            "All three published anchors remain abstentions. The 199166 "
            "3.705 s MARFE onset fails the containing-bin spatial gate; "
            "fG is below 0.8 and no explicit H-L cue exists. Later spatial "
            "hits cannot repair that density cue. The 199172 candidates "
            "lie near fG=0.8, inside the chord estimate's ±10–20% "
            "uncertainty. Sources: `detachment_reference.json`, "
            "`detachment_physics.json`, `detachment_round3.json` and "
            "`figure/marfe_witness.json`."
        ),
        (
            f"Manual front check: {manual['n_paired_valid_bins']} paired bins/"
            f"{manual['n_paired_shots']} shot, MAE {manual['mae_dz']:.3f}. "
            "These annotations are not independent state truth. Source: "
            "`detachment_reference.json`."
        ),
        "",
        "### Artifacts and verification",
        "",
        (
            f"Large artifacts: `{ROOT}`; figures "
            "`figure/fig_detachment_{views,timeline,figure2,marfe_witness}."
            "{pdf,png}`. Every changed PNG was inspected after final "
            "regeneration. Small artifacts: worktree `docs/labeler/results/`, "
            "`docs/labeler/detachment.md`, `docs/labeler/figure2_detach.json` "
            "and `data/events/detachment/`. `HANDOFF.md` describes the "
            "Figure 2/agreement schema changes and "
            "`detachment_availability.json` gives per-shot diagnostic "
            "availability. State codes and interval/grid/trace paths stay "
            "stable for detach-ui."
        ),
        "",
        "```json",
        dumps(validation, indent=1),
        "```",
        "",
    ]
    for key in ("covering_test_log", "lint_log"):
        path = Path(validation[key])
        lines += [
            str(path),
            "",
            "```text",
            *path.read_text().splitlines()[-8:],
            "```",
            "",
        ]
    lines += [
        "### Deviations, concerns and next work",
        "",
        (
            "Controller limits paper use to exploratory coverage/agreement. "
            "MIN_VALID_BINS=20 is a stated eligibility deviation. Extra EFIT "
            "fetching was optional and was not attempted: required repair "
            "and regeneration took priority. Existing ψ coverage is sparse "
            "and gives no upper-shelf SOL probe. The 201081 pinj failure "
            "is a PTSERVER client configuration error, not evidence of "
            "missing physical beam power. Chen states 4 MW NBI; the local "
            "4.4 MW threshold denominator is an unsourced assumption."
        ),
        "",
        (
            "Next: independent attached/detached adjudication and multiple "
            "MARFE shots, radiation/heating calibration checks, frozen "
            "development thresholds, and expanded positioned-probe ψ maps. "
            "Current labels cannot support detector accuracy, transitions "
            "or MARFE transfer claims. No push/merge, manuscript edits or "
            "production-store mutation."
        ),
    ]
    previous = args.report.read_text() if args.report.exists() else ""
    if ARCHIVE in previous:
        previous = previous.split(ARCHIVE, 1)[1]
    args.report.write_text("\n".join(lines) + ARCHIVE + "\n" + previous.lstrip())
    print(f"Current report: {args.report}; HEAD {end}")


if __name__ == "__main__":
    main()
