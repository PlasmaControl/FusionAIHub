# Stage 6 scientific review: final report and artifact handoff

Reviewer: ML scientist / AI conference reviewer. Date: 2026-10-01.

**Final handoff readiness: 8/10. Proceed with the report and local artifact
handoff.** This approves an accurate, reproducible exploratory study, not
detector promotion or acceptance of the full paper. The separate whole-paper
assessment remains approximately **5/10 as-is** on the project's internal
ICML-style scale. H1 passes and the original H2 fails; neither result changed.

## Material inspected

I read the complete report at
`dev/label_paper/reviews/2026-10-01_confinement_kevin_jalal_elm_report.md`,
visually inspected its four embedded PNGs, and checked them against the prior
stage results and frozen files in `runs/labeler/confinement/v1`. I inspected
the unified CLI defaults, relocation manifest, executed notebook, and local
no-BES inference smoke. I made no implementation or manuscript edits.

The report preserves the scientific distinctions that matter:

- Exact source agreement is a provenance/consistency result. Workbook and
  Jalal entries are dependent; the direction of copying remains unverified.
  HDF class semantics are inferred rather than supplied independently.
- The 252 ms within-Kevin L/H conflict is preserved and excluded. Full-bin
  targets, subtype ambiguity, unknown time and shot-wide Test-only isolation
  are described without converting unassessed time into negatives.
- Diagnostic eligibility substantially contracts the test cohort. Five L
  shots, concentrated L bins, two QH shots and four WP shots limit inference.
  No test spectral features are usable; this is a diagnostic-summary baseline.
- The full recipe remains primary. Its L/H F1 scores, wide L interval,
  valid-bootstrap counts and failed H2 are reported accurately. No incremental
  BES benefit or arbitrary missing-modality robustness is established.
- The historical comparison contains only one L bin on one shot, with prior
  historical test exposure. Perfect ranking by both models and degenerate
  bootstrap intervals do not establish superiority or unseen-L generalization.
- The revised ELM overlap is one 300 ms QH interval. Assisted-review dependence
  is explicit. The separate onset archive generates candidates, not adjudicated
  errors or a population confinement–ELM association.

## Independent final verification

All six hashes recorded in `relocation.json` match the relocated files. I also
verified the four frozen pipeline source hashes (`labels.py`, `data.py`,
`train.py`, `apply.py`), all training input hashes, label-output hashes, all five
model/weight pairs, feature CSV, evaluation freeze/prediction hashes, matched
historical predictions, comparison code and comparison input hashes. Mapping
the old output prefixes resolves the historical records; shared diagnostic and
historical-model inputs remain at their original paths. No refitting or retuning
was needed to relocate the experiment.

The run contains approximately 29.2 million apparent bytes. Both previous output
directories are absent. The scientific artifacts are inside the requested
`FusionAIHub/runs/labeler/confinement/v1` directory. This is a portable output
bundle; its external diagnostic inputs still prevent a fully self-contained
public replication, a limitation the report acknowledges.

Fresh validation command:

```bash
OMP_NUM_THREADS=4 pixi run --frozen -e shot-design-cpu python -m pytest \
  tests/labeler/test_confinement_labels.py \
  tests/labeler/test_confinement_elm.py \
  tests/labeler/test_confinement_data.py \
  tests/labeler/test_confinement_train.py \
  tests/labeler/test_confinement_export.py \
  tests/labeler/test_confinement_comparison.py \
  tests/labeler/test_confinement_cli.py -q -W error
```

Result: **34 passed**. The saved notebook has ten executed code cells and zero
error outputs. Its analysis reads the local run. The relocated no-BES smoke on
185915, [2450,2750) ms, produces six category-1 bins and records the exact frozen
model/training hashes. This verifies inference operation, not new label accuracy.

The four report PNGs are byte-identical to their run figures. Their denominators,
units, cohort labels and uncertainty displays are legible and consistent with
the saved results. Source agreement uses assessed duration; balancing is clearly
annotation-stage; detector plots disclose exploratory L support; the ELM plot
separates the single reviewed interval from source-conditioned onset rates and
marks unsupported regimes as unassessed.

`git -C dev/label_paper diff --exit-code` succeeds. Manuscript HEAD remains
`57efc593ac553fa8d91356fa0ba1995c32d57e92`; status contains only the new report and
its four PNGs under `reviews/`. The user's no-manuscript-edit constraint is met.
Recommendations remain in the review rather than manuscript source changes.

## Issues corrected before the final rating

The initial handoff readiness was 7/10 because reproduction commands could
overwrite upstream manifests in the frozen v1 directory before training refused
to refit. The corrected report runs all write steps in a subshell with a fresh
`CONFINEMENT_RUN_DIR`, checks that it does not exist, and stops on errors. Saved
v1 inference/figure examples use explicit paths. It distinguishes exploratory
replication of an opened test from a new confirmatory experiment. This resolves
the substantive handoff blocker without changing frozen code or artifacts.

The report also now distinguishes 1,959 workbook data rows from 1,960 rows
including the header, and explicitly states the possible dependence of assisted
ELM review on detector/archive proposals. No must-fix handoff issue remains.

## Research priorities and paper readiness

The report's proposed order is scientifically appropriate: adjudicate the exact
conflict; obtain independent jointly assessed confinement–ELM labels with
candidate and noncandidate controls; reserve new test shots before opening new
gold; expand independent L/QH/WP support; then compare diagnostic recipes and
temporal models on common cohorts with equal budgets. Timestamp repairs require
original-acquisition validation and a separately declared run. Additional
engineering on this opened test remains exploratory.

For the whole paper, independent multi-reader reference quality, a completed
aligned-cohort result, confirmatory testing, standard comparative baselines and
an externally usable release remain the substantial gaps. High legacy-target
point scores, perfect dependent-file agreement, or the tiny historical
intersection do not close those gaps. The handoff correctly presents this work
as stronger provenance, explicit coverage and a reproducible baseline, with
unresolved scientific questions retained.
