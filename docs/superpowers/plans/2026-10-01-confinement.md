# Confinement reconciliation and detector implementation plan

**Goal:** Compare Kevin Gill's workbook/time labels with Jalal Butt's labels,
merge usable annotations, balance training, and evaluate a detector that works
with BES absent. The user authorized implementation and scientific Sol reviews.

**Architecture:** A separate `labeler.confinement` workflow preserves source
intervals and creates exact reconciliation segments, conservative 50 ms targets,
shot splits, diagnostic features and a versioned model. Existing model artifacts
stay available for comparison. Generated products live under `LABELER_ROOT`.

**Constraints:** Production stores and raw labels are read-only. Unknown is never
negative. No contradictory H/L span is a training target. QH/WP remain recorded
as high-confinement variants. Test-only shots stay held out across sources.
Balancing affects training alone; validation/test retain their prevalence.
Do not infer radial coordinates for raw Thomson channel order. No new dependencies.
Every major scientific outcome needs a Sol rating >=8 before the next stage.

## Step 1: Labels

- [x] Write and run failing tests for half-open HDF intervals, missing/gapped
  samples, conflict exclusion, subtype agreement, full-bin coverage, exact
  class/shot weighting and test-only split isolation.
- [x] Implement `src/labeler/confinement/labels.py`: workbook/CSV/HDF readers,
  source hashes, interval sweep, duration confusion, conservative bin targets,
  deterministic shot splits and balanced weights. Audit duplicated label clips.
- [x] Run the raw-data reconciliation; save source intervals, merged intervals,
  conflicts, per-shot comparison, source audits, targets and a readable report.
- [x] Sol scientific review; fix issues until rating >=8.

## Step 2: Diagnostics and training

- [x] Test missing diagnostic handling, no interpolation across acquisition gaps,
  and train-only balancing after diagnostic coverage exclusions.
- [x] Implement `data.py` for window summaries of BES, D-alpha, Thomson, CO2,
  Mirnov, heating and cached equilibrium features, with explicit missing values.
- [x] Implement `train.py`: shot-disjoint validation selection; balanced weights;
  native missing-value classifier; no-BES and single-modality ablations; threshold
  selected on validation only; frozen test and source/model/split hashes.
- [x] Run extraction/training with the installed shot-design-cpu environment.
- [x] Sol scientific review; fix issues until rating >=8.

## Step 3: Evaluation and handoff

- [x] Evaluate paired baselines on eligible unseen shots, report both class F1,
  prevalence, excluded time, diagnostic subgroups and shot-bootstrap intervals.
- [x] Implement inference CLI and common-schema H/L export of the merged targets.
- [x] Save figures, a reproducible notebook, model card and paper-ready results
  with explicit source-dependence/generalization limits.
- [x] Run focused regression tests and a real-data inference smoke check.
- [x] Final Sol scientific review; revise until rating >=8. Report remaining
  scientific limits without treating reconciled legacy labels as independent GT.

User handoff constraints: no manuscript edits; report with graphs/tables goes in dev/label_paper/reviews/. All new label experiments use runs/labeler/<study>/<version>; current outputs are runs/labeler/confinement/v1. Shared production/cache inputs remain read-only.
