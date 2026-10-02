# Stage 3 scientific review: diagnostic eligibility and features

Reviewer: ML scientist / AI conference reviewer. Date: 2026-10-01.

**Rating: 8/10. Proceed with the frozen exploratory diagnostic-summary baseline.**
This is a pre-fit methodology and coverage gate. It is not a score for detector
performance, satisfaction of H1/H2, independent gold validation, or unrestricted
missing-modality generalization. No current-artifact blocker remains after
explicitly documenting the feature-support limitations discovered in review.

## Evidence and validation

I inspected `src/labeler/confinement/data.py`, data tests, the prospective training
and inference implementation and tests, `features.json`, `features/all.csv`,
`extraction.log`, the added diagnostic-eligibility documentation, and
`feature_support.csv` under
`/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/confinement/v1`.

Independent checks confirm the generated feature hash, extractor hash, and
four input-artifact hashes. I recomputed eligibility, diagnostic patterns,
subtype counts, reserved-test support, L-shot contributions, spectral support,
and nonfinite-feature consistency. The per-feature support CSV agrees with
the saved features. No real fitted model artifacts were present during review.
I did not repeat extraction of the large raw stores.

Fresh validation:

```
OMP_NUM_THREADS=4 pixi run --frozen -e shot-design-cpu python -m pytest \
  tests/labeler/test_confinement_data.py \
  tests/labeler/test_confinement_train.py -q -W error
```

returned **17 passed**. The six data tests also passed in `labelmaker`.
`labelmaker` lacks scikit-learn and therefore cannot collect the training
tests; use the existing `shot-design-cpu` environment for that verification.
No dependency installation was needed.

## Why fitting can proceed

The extractor summarizes native samples within the labeled 50 ms bin, checks
boundary support, rejects acquisition gaps and stale records, and leaves missing
measurements NaN with separate presence flags. It neither interpolates nor
stride-decimates time. Fixed channel selection is independent of labels and
model performance. Thomson quantities are correctly described as channel
summaries without radial-coordinate or pedestal claims; NBI is summed across
valid beam channels. Cached equilibrium values are optional, with no live fetch.

Shot, absolute time, label, source, and split columns are outside the feature
schema. Tests establish train-only balancing after eligibility exclusion,
train-only removal of entirely missing features, frozen split preservation,
test-feature/target invariance of model fitting, and abstention on all-missing
inputs. Prospective model/configuration/threshold choice uses validation only.
The manifest reservation list is checked as well as the supervised split.
An existing freeze prevents silent refitting after test opening.

There are no NBI-only eligible rows in train, validation, or test. Every eligible
test row has D-alpha, Thomson density/temperature, and NBI. Therefore the broad
any-physical-feature eligibility rule does not require restriction for an
NBI-only case in the present dataset. If such rows appear in a future inference
domain, report them as a separate contextual prediction domain; this run has
not validated that situation.

## Coverage is the main scientific limitation

| Split | Eligible shots | Bins | L bins / shots | H bins / shots |
| --- | ---: | ---: | ---: | ---: |
| Train | 68 | 2,955 | 485 / 26 | 2,470 / 59 |
| Validation | 16 | 745 | 43 / 4 | 702 / 15 |
| Test | 28 | 1,403 | 103 / 5 | 1,300 / 26 |

Diagnostic exclusions remove 1,915 otherwise known test bins on 44 shots.
Eligibility is conditioned on legacy annotations and local stores; these are
not population prevalence estimates. The eligible reserved Test-only subgroup
has 155 H bins on three shots and no L bins. It cannot validate L detection,
and unsupported discrimination metrics must remain null.

L support is weaker than bin counts imply: test L bins are 60 on shot 189379,
25 on 185871, 16 on 187049, and one each on 189602 and 192007. Only three
test shots contribute more than one L bin. Validation has 19 + 19 + 3 + 2 L
bins across four shots. Validation-only tuning is leakage-free but consequently
noisy; this is a reason to report uncertainty and all outcomes, not to retune
on test. Four-class test support is L 103/5, standard H 1,167/22, QH 37/2,
and WP 94/4 bins/shots, with two ambiguous subtype bins excluded. QH/WP
results are descriptive, with very limited independent-shot support.

BES covers 614 test bins on 26 shots, including only 36 L bins on three shots.
Mirnov covers 1,056 test bins. Cached beta_N/q_min are highly imbalanced
(only one L bin each), and beta_p covers 61 H bins with no L. These are not
balanced modality-specific benchmarks. CO2 and W_MHD have zero usable bins;
12 CO2 axes fail strict monotonicity, separately from the truncated BES clip.
Rejecting those inputs is preferable to fabricating coverage. Quantization is
only a possible explanation for invalid axes, not a validated repair.

## Spectral scope corrected before fitting

Every eligible test row lacks all three spectral-band features for D-alpha,
BES, and Mirnov. Train/validation spectral support exists only in BES clips:
34 bins on one train shot and 20 bins on one validation shot. No D-alpha or
Mirnov spectral features are available. The added documentation and feature
support table make this limitation explicit.

Accordingly this run is primarily an amplitude/fluctuation/channel-summary
baseline. It cannot establish learned spectrogram quality, spectral-band test
generalization, or CO2 utility. No timestamp repair is required to fit that
limited baseline; a future spectral claim would need a separately validated
sampling/timestamp method and another coverage review before its test opens.

## Requirements for interpreting the next outcome

- Freeze the unchanged feature, label, split, fitting-code, candidate-grid,
  threshold, and bar records before test prediction. Keep the existing H1/H2
  definitions rather than lowering them in response to scarce support.
- Recompute class/shot training weights for each model's actual eligible
  training rows. Keep test metrics on observed prevalence, report class and
  shot denominators, and use paired whole-shot differences on common eligible
  bins. Preserve omitted/all-missing cases in coverage accounting.
- Report H/L F1, P/R, balanced accuracy, PR discrimination, and the paired
  H2 difference from all-H with valid bootstrap replicate counts. Do not
  interpret one-class subgroups as a two-class success. Four-class estimates
  must disclose sparse support and undefined replicate/class metrics.
- No-BES training and forced BES removal test that particular observed shift.
  They do not demonstrate arbitrary missing-diagnostic robustness or missing
  D-alpha/Thomson generalization. Tiny missingness patterns and subtype groups
  should be descriptive. Cached-equilibrium missingness may track campaign
  or source selection even though no explicit source feature is supplied.
- Do not compare these scores to the historical CNN on a different cohort as
  a model improvement. A learned-baseline comparison must share assessed
  windows, eligible domains, and test exposure history. Passing a performance
  bar against dependent legacy targets does not itself create independently
  validated labels or authorize promotion of the existing default detector.

The next scientific gate concerns the complete frozen evaluation, including
failures. This gate permits fitting; it makes no claim that any model will pass.
