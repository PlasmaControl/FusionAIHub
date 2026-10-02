# Stage 1 scientific review: confinement label reconciliation

Reviewer: ML scientist / AI conference reviewer. Date: 2026-10-01.
Manuscript context: `dev/label_paper`, commit `57efc59`.

**Rating: 8/10. Proceed to the next stage.** This rating concerns the readiness
of an exploratory imported-label audit and binary target construction. It is
not an ICML acceptance rating, independent validation of confinement truth, or
evidence of detector performance. There are no blocking errors in the current
artifacts at that scope.

## Evidence inspected

I read `confinement_analysis.md`, `src/labeler/confinement/labels.py`, its focused
tests, and the saved `labels.json`, source/merged/conflict intervals, targets,
split, and per-shot comparisons under
`/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/confinement/v1`.
I independently recomputed CSV hashes, source counts, workbook containment,
merged interval non-overlap, coverage/conflict constraints, split support,
reserved-shot membership, and training-weight totals from those outputs.

Fresh validation:
`pixi run --frozen -e labelmaker pytest tests/labeler/test_confinement_labels.py -q -W error`
returned **7 passed**. All six saved CSV hashes match the manifest. I did not
rerun every large raw HDF import or rehash every raw HDF file; raw sample and
duplicate-file findings were assessed through the importer, tests, recorded
audit metadata, and saved intervals.

## Why this outcome meets the stage threshold

1. **Dependence is disclosed and concretely demonstrated.** All 1,958 workbook
   intervals on 426 shots are exactly contained in Jalal's 1,964 intervals on
   428 shots. BES overlap totals 84.418002 seconds on 58 shots, with agreement
   in each of L/H/QH/WP. These are source consistency results, not three
   independent expert readings. The inferred HDF class order is identified as
   an inference and has a unique best four-way overlap permutation.
2. **Duration and uncertainty semantics are defensible.** The merged union is
   a non-overlapping partition on 448 shots. Its 800.59104 seconds of matching
   sources, 122.463998 seconds of single-source coverage, and 0.252 seconds of
   binary conflict reproduce the report. Shot 189379's internal BES H/L
   contradiction is retained and excluded, not resolved by a donor priority.
   Imported gaps and blank regime rows do not become L-mode negatives.
3. **Conservative targets avoid invented supervision.** Every saved known
   target has exactly 50 ms of coverage and zero conflict duration. There are
   17,522 known bins on 435 shots and 1,550 bins with some coverage but no
   usable target. QH/WP remain distinct in the interval artifact and both
   support the binary H target. A subtype disagreement does not silently
   become a four-way resolved label.
4. **Splits and weighting avoid immediate leakage and duration dominance.**
   Each supervised shot has a unique split. No known bin from a reserved
   Test-only shot falls outside test. The training class masses are both
   5,771.5; per-shot mass within each class is equal to numerical precision.
   Validation/test weights remain zero. The test contains 434 L bins on 17
   L-labelled shots and 2,884 H bins on 67 H-labelled shots before feature
   eligibility, substantially more negative-shot support than the existing
   reported test but still only a modest number of independent L shots.

## Limits and precise requirements for subsequent claims

- **Preserve the entire reservation roster.** `labels.json` reserves 11 raw
  Test-only shots; `split.csv` contains the nine with usable targets. Shots
  200649 and 200654 carry named EH-B/EH-Li flags but no explicit L/H/QH/WP
  regime. They correctly provide no targets. Feature preparation and any
  later label extension must consult the manifest reservation roster, not
  infer eligibility from absence in `split.csv`. Likewise preserve the 13
  merged shots without any known bin as assessed-but-unusable cases.
- **State the estimand.** This is a selected legacy-annotation domain with a
  stratified split and forced Test-only assignments. Unweighted validation
  and test retain their observed class prevalence; that is not population
  prevalence. Report source/campaign composition and feature eligibility
  when assessing generalization. Separate reserved and seeded test domains
  where support permits, without overstating tiny subgroup results.
- **Do not claim transition timing from this stage.** Mixed and partially
  covered bins are removed, so detector performance on clean bins will not
  measure transition accuracy or whole-discharge coverage. Keep the 50 ms
  imported task distinct from the manuscript's 10 ms reviewed task. Boundary
  evaluation needs explicit reference events and a disclosed tolerance.
- **Treat balancing as a training objective, not calibration.** Recompute
  training weights after modality eligibility; evaluate natural held-out
  bins with both class-specific and shot-level support. Use validation-only
  threshold choice and paired shot-bootstrap intervals. A class-balanced
  loss alone does not establish calibrated H probabilities.
- **Preserve source-independent claim hygiene.** Perfect overlap is
  compatible with shared ancestry and says nothing about errors shared by
  all donors. Training/evaluation against these targets measures agreement
  with reconciled legacy supervision. Independent gold, meaningful standard
  baselines, and an accessible release remain manuscript-level needs.

Nonblocking importer hardening for future inputs: `_table` ignores literal
`0`/`0.0` Test-only markers, and `build` does not union workbook reservation
flags into Jalal's reservation set. Current markers are `1`, `EH-B`, and
`EH-Li`, and the current workbook has no reservation flags, so neither issue
changes these outputs. Define the intended zero-marker semantics and union
reservation sets if additional flagged sources are introduced. Record a source
code digest or commit plus dirty diff for a later public freeze; CSV/raw hashes
alone do not pin the transformation code.

## Next scientific outcome: ELM versus confinement consistency

Use exact reconciled four-regime intervals and reviewed ELM spans, rather than
the binary 50 ms target. Compute joint assessed durations and shared-shot
counts, with ELM present/absent/uncertain/not-observable and missing coverage
reported separately. Do not convert missing ELM review into absence. Quantify
candidate QH/WP–ELM overlaps, preserving the source identity and conflicting
subtypes; separate stable interior overlaps from boundaries by a stated
tolerance or sensitivity analysis.

An ELM-present/QH overlap is a candidate inconsistency requiring review of
diagnostic evidence and definitions, not automatic proof that either table is
wrong. Preserve both original labels and report adjudication separately.
Reviewed ELM spans may have been drafted from the ELM clock; disclose that
dependence and avoid using the consistency relation both to change labels and
to present those changed labels as independent confirmation. This audit can
demonstrate the utility of aligned sources even if no final errors are yet
adjudicated.
