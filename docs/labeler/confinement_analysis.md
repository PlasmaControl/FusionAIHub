# Confinement source reconciliation — 1 October 2026

This is an exploratory audit of imported supervision for the label paper,
not an independently annotated ground-truth dataset. Manuscript context:
`dev/label_paper`, GitHub `fusionlabeldatabase`, commit `57efc59`.

Reproduce: `pixi run --frozen -e labelmaker python -m labeler.confinement labels`.
Products: `runs/labeler/confinement/v1/` (the configured default is
`/scratch/gpfs/nc1514/FusionAIHub/runs/labeler/confinement/v1`). `labels.json` pins every
raw file and all generated CSVs by SHA-256. Raw files and production stores are
never modified.

## Source statistics and dependence

| Supplied source | Labelled shots | Unique intervals | Finding |
| --- | ---: | ---: | --- |
| Jalal Butt CSV | 428 | 1,964 | 434 raw shots; six have no usable explicit interval |
| Kevin Gill workbook | 426 | 1,958 | Every interval is exactly present in Jalal's CSV |
| Kevin Gill BES-time labels | 82 | 171 | 172 files; one is empty |

Kevin's workbook path metadata names `/Users/kevingill/PhD/...`; its header is
L, H, QH, WP. HDF files have no class-name attributes. Interpreting their four
one-hot columns in that order matches every overlap with the workbook/Jalal
labels; all four classes have nonzero overlap, and this permutation has the
unique highest overlap agreement. This supports the interpretation, but is not
independent semantic verification.

The time labels agree with Jalal on all four regimes over **84.418002 s on 58
shots**. The workbook agrees over **800.59104 s on 426 shots**. These are
consistency checks: exact matching strongly suggests shared source history,
without independently establishing the direction of copying.
Do not count the imported files as multiple independent expert readers.

Eight readable BES signal copies exactly duplicate their time and label arrays.
The ninth, `bes_signals_149993at4172.hdf5`, is truncated. Its readable time-label
copy remains usable, and the signal copy is unavailable, with its error recorded.
`bes_signals_192007at1885.hdf5` has zero samples and contributes no labels.

## Conflict and merge

There is no cross-source disagreement on the overlapping assessed time. There
is one within-source conflict: **shot 189379, [3198, 3450) ms**, where Kevin's
`at500` clip says L and `at3198` says H, for **252 ms**. It is recorded in
`conflicts.csv` and excluded from H/L training. This duration exceeds one
training bin and is not merely a sequential transition inside a bin.

The exact union covers **448 shots**: 800.59104 s with matching sources,
122.463998 s with a single source, and 0.252 s with conflicting labels. No gap,
empty file, unknown sample or acquisition timestamp is made into a negative.
QH and WP stay distinct in the interval table, and both map to high confinement
for the binary detector. An unresolved subtype conflict would remain recorded
while still supporting H if every source agreed on the binary state; none occurs
in this import.

## Conservative targets and balancing

A 50 ms target is usable only when the entire half-open bin is covered by one
unambiguous H/L state. Transition bins and partly annotated bins are excluded.
There are **17,522 known bins on 435 shots** and **1,550 excluded bins with some
source coverage**. The other thirteen union shots have no complete known bin.
This is a 50 ms legacy-supervision task, distinct from the paper's 10 ms dense
review task. Complete annotation absence is outside the assessed denominator.

| Split | Shots with known bins | L bins | H bins | L-labelled shots | H-labelled shots |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 300 | 1,803 | 9,740 | 82 | 274 |
| Validation | 63 | 338 | 2,323 | 17 | 58 |
| Test | 72 | 434 | 2,884 | 17 | 67 |

These are annotation counts before diagnostic eligibility. The next stage must
report which shots remain after feature availability. The split is deterministic
by shot, stratified by represented binary classes; every nonempty raw `Test only`
marker reserves the whole shot for test across all sources. Train/validation/test
are disjoint. Training alone gets equal effective H and L weight, with equal
shot weight within each class; validation and test retain natural prevalence.
Weights will be recomputed after diagnostic exclusions.

The paper's existing H-mode result has only 37 L bins on four test shots. These
larger annotation counts address support but are not themselves improved model
performance. The new test remains exploratory; prior model exposure and source
dependence must be disclosed in comparisons.

## Scientific reviews

Sol stage reviews are saved in `confinement_reviews/`. A stage score measures
readiness of this particular outcome, not the overall paper's ICML acceptance
prospects. Outcomes scoring below 8/10 are revised before the next stage.

## Confinement–ELM consistency

Reproduce: `pixi run --frozen -e labelmaker python -m labeler.confinement elm --legacy`.
The `elm_study/` directory pins a snapshot of the current reviewed ELM table,
the exact confinement table, both original-onset source files, joint assessment
intervals and event-level QH inspection candidates.

The revised ELM review has 110 shots but only **one overlaps confinement**:
shot **192751**, QH over **[2000, 2300) ms**. All 300 ms are assessed as
non-ELMing. Removing 50 or 100 ms at each confinement-phase edge leaves 200 or
100 ms, also assessed as non-ELMing. Every other confinement interval lacks
matching reviewed ELM assessment. No reviewed cross-regime comparison or
population claim follows from this sample.

The original onset archive provides a separate, exploratory comparison:

| Regime | Shared shots | Supplied onset-assessed time (s) | Positive onset samples | Onset-positive shots | Samples/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| L | 18 | 5.044 | 0 | 0 | 0 observed |
| H | 0 | 0 | — | — | not assessed |
| QH | 69 | 71.216 | 60 | 15 | 0.843 |
| WP | 65 | 91.521 | 43 | 20 | 0.470 |

Shots can contribute to more than one regime. The 95% shot-bootstrap interval
for QH is [0.264, 1.532] onset samples/s; for WP it is [0.283, 0.677]. No
nonparametric interval is reported for zero observed events, because resampling
all-zero shots cannot supply a useful upper bound. These are source-conditioned
descriptive rates, not population rates or a statistical H-versus-QH comparison.

With a 100 ms boundary guard, QH has **51 onset samples on 11 shots over
56.557 s** (0.902 samples/s); WP has 27 on 13 shots over 67.035 s. Thus the
candidate QH/onset inconsistency is not confined to the first/last 100 ms of
annotated phases. These guards do not independently validate physical phase
boundaries. The inspected candidates are saved in
`elm_study/legacy_qh_onset_candidates.csv`, with exact phase bounds and distance
to a boundary. The guards trim regime phases, not changes in source provenance.

An archived zero records no onset in that supplied millisecond; it is not a
dense non-ELMing phase label. Unknown, uncertain and not-observable reviewed
intervals never become absence. ELM spans drafted with model/rule assistance and
dependent legacy annotations do not establish independent physics validation.
QH/ELM overlap is a candidate inconsistency requiring signal inspection; neither
label set is automatically corrected from the other's nominal definition.

This is a useful paper audit and demonstrates why aligned review coverage is
needed. A strong revised-label association result needs additional independently
reviewed L, standard H and QH phases on common shots.

## Diagnostic eligibility and feature scope

Real extraction is read-only, with no live fetches. `features.json` pins the input
label tables, extractor and consumed native values; `feature_support.csv` records
finite statistical-feature counts by split, shot and binary class. Missing values
stay NaN, and missingness flags are separate from physical measurements.

| Split | Eligible shots | Eligible bins | L bins | L shots | H bins | H shots |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 68 | 2955 | 485 | 26 | 2470 | 59 |
| val | 16 | 745 | 43 | 4 | 702 | 15 |
| test | 28 | 1403 | 103 | 5 | 1300 | 26 |

The test is conditioned on annotations and available local diagnostics; it is not
a population prevalence sample. Diagnostic availability excludes **44 test shots
and 1,915 otherwise known test bins**. The raw Test-only eligible slice contains
155 H bins on three shots and **no L bins**, so it cannot validate L performance.

The test L support is concentrated: shot 189379 contributes 60 bins, 185871
25, 187049 16, 189602 one, and 192007 one. Only three test shots contribute
more than one L bin. The apparent annotation support gain does not remove this
scarcity after diagnostic eligibility. Four-class test support is L 103 bins
on five shots, H 1,167 on 22, QH 37 on two, and WP 94 on four; subtype results
are consequently exploratory and require shot-level uncertainty.

Validation is also sparse: its 43 L bins are distributed 19, 19, three and
two across four shots. Validation-only selection avoids test leakage but can
still be noisy; this is a limit to report, not a reason to tune on test.

D-alpha, raw Thomson summaries and NBI cover all 1,403 eligible test bins.
BES is observed for 614 bins on 26 shots (36 L bins on three shots); Mirnov
is observed for 1,056. Cached beta_N and q_min have limited support; beta_p
has 61 test bins, all H. CO2 and W_MHD have **zero usable bins**. Twelve
CO2 time axes fail strict monotonicity, and the separately identified BES clip
is truncated. No missing measurement is invented to fill these gaps.

The current learner is a **diagnostic-summary baseline**: native amplitude,
fluctuation and channel summaries. Native periodogram bands are attempted only
on sufficiently regular timestamps. All three spectral bands are unavailable
for every eligible test bin. Only BES clip spectra are available in training
(34 bins on one shot) and validation (20 on one shot); no D-alpha or Mirnov
spectral features are available. This run therefore does not establish
spectral-feature generalization, CO2 utility or full-rate spectrogram quality.
All-NaN training columns are pruned using training alone before model fitting.

Raw Thomson channels have no radial coordinates, so their summaries are not
pedestal measurements. Boundary coverage, acquisition gaps and stale samples
are checked; there is no interpolation or temporal stride decimation. Channels
are fixed a priori: first eight filterscopes; eight spread BES/Thomson channels;
four spread Mirnov channels; summed NBI power; optional cached equilibrium
quantities. Every currently eligible row has a non-NBI diagnostic. Dedicated
no-BES training and forced BES-removal evaluation assess that specified shift;
they do not establish robustness to arbitrary missing-diagnostic combinations.

## Frozen detector results

Five models were fitted on the eligible training shots using weighted histogram
gradient boosting with native missing values. Each received the same eight
configurations (learning rate 0.05/0.1, 7/15/23/31 leaves, 150 iterations).
Selection maximized natural-prevalence validation macro-F1. Binary thresholds
were selected from a fixed grid and validation probability boundaries, with
ties favoring 0.5. The primary full model was declared before test scoring;
it is retained despite a slightly higher test score for the no-BES alternative.
There was no test retuning. `training.json` was frozen at
2026-10-01T20:14:50.889992Z before the first test prediction; it pins source,
features, code, model and effective training-weight files.

| Model | Test shots / bins | F1 L | F1 H-family | Macro-F1 | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Primary diagnostic summaries | 28 / 1,403 | 0.925 | 0.994 | 0.960 | 0.950 |
| Dedicated no-BES | 28 / 1,403 | 0.930 | 0.995 | 0.962 | 0.950 |
| D-alpha + NBI | 28 / 1,403 | 0.795 | 0.980 | 0.888 | 0.963 |
| BES only | 26 / 614 | 0.500 | 0.942 | 0.721 | 0.893 |
| Primary with BES removed at test | 28 / 1,403 | 0.935 | 0.995 | 0.965 | 0.950 |
| Always H, primary cohort | 28 / 1,403 | 0.000 | 0.962 | 0.481 | 0.500 |

BES-only has a different eligible cohort; its headline score must not be used
as a direct matched comparison. The saved paired bootstrap for that comparison
uses only shared bins. Full versus dedicated no-BES macro-F1 difference has a
95% paired shot interval of **[-0.044, 0.023]**: no incremental BES benefit is
established. The forced-absence run concerns BES removal only. No-BES models
still depend on the available alternative diagnostics.
Training eligibility also differs: the full recipe uses 68 shots, whereas
no-BES and D-alpha+NBI use 67. These are practical recipe comparisons on a
common test cohort, not causal single-modality ablations; they do not separately
establish the contribution of Thomson, Mirnov or equilibrium features. All fits
use one fixed seed. Shot intervals quantify test-shot uncertainty conditional
on the fitted model, rather than variability from retraining.

The primary confusion matrix (true rows, predicted columns L/H) is
`[[93, 10], [5, 1295]]`. Its AUROC is 0.99888, AUPRC-L 0.98812 and AUPRC-H
0.99991. On whole-shot resampling, 1,993 of 2,000 bootstrap samples contain
both classes. Conditional on that support, 95% intervals are F1-L
**[0.571, 0.990]**, F1-H [0.984, 1.000], macro-F1 [0.785, 0.995] and balanced
accuracy [0.916, 1.000]. Sparse L shots produce substantial uncertainty despite
the high aggregate point estimate. Paired gains over always-H are macro-F1
[0.286, 0.513] and balanced accuracy [0.416, 0.500].

The original manuscript bars are unchanged. **H1 passes** (F1-H >=0.95 and
F1-L >=0.70). **H2 fails**: the paired F1-H improvement over always-H has a
95% interval of **[-0.00043, 0.07969]**, whose lower bound is not positive.
Macro-F1 gains do not replace that pre-existing H2 bar. No existing default
detector is automatically promoted. The result is an improved exploratory
two-class baseline against the constant, with limited independent-shot support
and dependent legacy supervision.

The separately fitted four-class baseline uses 1,401 subtype-known test bins:
macro-F1 **0.845**, balanced accuracy **0.856**, and F1 L/H/QH/WP
0.851/0.982/0.742/0.806. Its 95% macro-F1 interval is [0.622, 0.928], from
1,760 of 2,000 shot resamples containing all four classes. This interval is
conditional on complete class support; QH has two test shots and WP four.
Subtype recognition is descriptive, not an independently validated regime
classifier. No four-class acceptance bar was invented after seeing results.

`evaluation.json` contains natural missingness patterns, BES-present/absent
subgroups, all per-shot results and paired ablations. The raw Test-only slice
has all 155 H bins classified correctly, but no L; balanced metrics and AUROC
there are undefined. Models, frozen thresholds and predictions reside outside
the repository under the run directory.

## Historical detector intersection

The comparison excludes every old RowsCNN training and validation shot and
uses the already frozen old threshold (0.33) and new threshold (0.963534).
Of 28 diagnostic-eligible new test shots / 1,403 bins, eight shots / 355 bins
were unseen by old fitting and validation. Four have old feature files; requiring
all five historical 10 ms frames to be observable leaves only **three shots /
122 common bins**: 185915, 187036 and 189602. Shot 187049 has no whole observed
common bins. The remaining cohort has **one L bin on one shot** and 121 H bins.
All three shared shots were also in the previously exposed historical test.

The primary gets all 122 labels right; the old detector classifies all as H,
missing the single L bin. Both rank this tiny cohort perfectly. This is an
inspection of threshold behavior on shared imported targets, not evidence
of independently validated model superiority. The inputs and temporal context
also differ. There is inadequate common L support for a meaningful historical
comparison: an apparently perfect score and degenerate bootstrap interval
cannot quantify uncertainty on unseen L shots. The artifact retains all
results, 1,427/2,000 two-class bootstrap draws, exact matched timestamps,
excluded cohorts and pinned old/new inputs under `legacy_comparison/`.
Reproduce with `python -m labeler.confinement comparison` in `shot-design-cpu`.

## Artifacts and verification

The executed notebook is `data/events/confinement/example.ipynb`; the model
card is `docs/labeler/confinement_model_card.md`. Standalone PDF/PNG figures
reside under the run's `figures/`, with detector figures generated by
`analysis/confinement_results.py`. Common-schema merged H/L targets are in
`format/`; every known bin is complementary, while unassessed time stays unknown.
Frozen model inference was checked on six real H bins and two fully unavailable
bins: unavailable bins stay category 3 with NaN confidence.

Fresh focused verification: **34 tests passed** with warnings treated as errors.
The earlier relevant source-format/schema/layout regression run passed **47
tests**. Notebook execution completed without cell errors, and the frozen code,
models, weights, predictions and label/feature artifacts were verified by hash.
Raw data, production corpus stores and existing default detectors were preserved.

The GitHub paper clone was updated to `57efc59` before this work. At the user's
request, manuscript edits and added manuscript figures/tables were reverted.
The full report is saved as
`dev/label_paper/reviews/2026-10-01_confinement_kevin_jalal_elm_report.md`;
the manuscript remains unchanged from the pulled revision. No changes were
committed or pushed. Temporary manuscript build output was removed during cleanup.

Sol rated the five scientific stages and final handoff **8/10 each**. These ratings are
outcome-readiness gates. Its separate overall ICML-style assessment was about
**5/10**: independent multi-reader gold, aligned reviewed cross-regime evidence,
confirmatory testing, release access and broader benchmark evidence remain open.
The improved baseline and consistency audit strengthen the paper without closing
those larger research requirements.
