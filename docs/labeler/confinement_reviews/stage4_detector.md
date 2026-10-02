# Stage 4 scientific review: frozen confinement detector outcome

Reviewer: ML scientist / AI conference reviewer. Date: 2026-10-01.

**Rating: 8/10. Proceed to a carefully scoped historical comparison.** This is
scientific readiness of a transparent exploratory model result, not a claim that
the detector passed its acceptance bar. **H1 passes; H2 fails.** The failed
clause remains failed, and no default detector should be promoted from this
result. A readiness score must not pressure the experiment into a passing result.

## Independent evidence

I inspected the current training/evaluation code, `training.json`, model and
training-weight artifacts, `evaluation.json`, `predictions.csv`, measured report,
model card, and real/missing inference smoke outputs under
`/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/confinement/v1`.

All feature/input/pipeline/model/weight hashes match the freeze. The freeze was
written at 2026-10-01T20:14:50.889992Z; recorded test opening follows at
20:14:50.928288Z. Each of five model families selected the best validation
macro-F1 among the same eight configurations. Every model's training weights
have equal total class mass and equal shot mass within class. Saved training
and validation rosters are disjoint from eligible test shots. Predictions match
the frozen thresholds. The primary remains `full`, despite a higher descriptive
test macro-F1 for alternatives.

Fresh data/training/comparison/export test validation returned **21 passed**:

```
OMP_NUM_THREADS=4 pixi run --frozen -e shot-design-cpu python -m pytest \
  tests/labeler/test_confinement_data.py \
  tests/labeler/test_confinement_train.py \
  tests/labeler/test_confinement_comparison.py \
  tests/labeler/test_confinement_export.py -q -W error
```

I independently reconstructed confusion matrices and vectorized whole-shot
resampling from saved predictions without the implementation's bootstrap
helper. This exactly reproduces the primary H/L intervals, H2 difference
interval, valid replicate counts, and four-class macro-F1 interval.

## Primary result and unchanged performance requirements

On 1,403 eligible bins from 28 shots, the primary matrix (true rows, predicted
columns L/H) is `[[93,10],[5,1295]]`. F1-L is 0.925373, F1-H 0.994242,
macro-F1 0.959807, and balanced accuracy 0.949533. The point results satisfy
H1's F1-H >=0.95 and F1-L >=0.70 requirements.

However, the paired F1-H gain over all-H has 95% interval
**[-0.000426379,0.079692608]**. Its lower bound is not above zero, so H2 fails.
The positive macro-F1 difference interval [0.285535,0.513481] is useful
descriptive evidence but cannot substitute for H2.

There are 1,993 two-class draws out of 2,000 requested. The saved intervals
condition on those draws: F1-L **[0.571429,0.989696]**, F1-H
[0.984437,0.999523], and macro-F1 [0.785178,0.994540]. That broad L interval
is material. The five L test shots contain 103 bins, but only three contribute
more than one bin. The point F1-L does not establish reliable L performance
across campaigns or independently reviewed shots.

The three eligible reserved Test-only shots have 155 H bins, all correctly
classified, but no L support. Their macro-F1, balanced accuracy, AUROC and
PR discrimination remain null. That subgroup result is not a two-class test.
All 44 excluded test shots and 1,915 missing-diagnostic bins remain visible in
the outcome's coverage accounting.

## Alternatives and what the comparisons mean

Dedicated no-BES macro-F1 is 0.962314; full minus no-BES has interval
[-0.043868,0.023176]. BES benefit is not established. Removing BES at the
frozen full model and threshold gives macro-F1 0.964843; full minus that
forced-absence result has interval [-0.049890,0]. This supplies no evidence
that BES improves this test result, nor does it prove BES is generally harmful.

Full minus D-alpha+NBI macro-F1 has exploratory paired interval
[0.024630,0.231214] on the common 1,403 bins. The compared training domains
are not identical: full uses 68 shots, whereas no-BES and D-alpha+NBI use
67, excluding the BES-only training clip. Consequently these are practical
recipe/domain comparisons, not clean causal tests of one added modality.
The D-alpha+NBI gap cannot separately identify Thomson, Mirnov, or EFIT value.
All models are from one fixed seed; shot-bootstrap intervals condition on the
fitted models rather than quantify model-selection or seed variability.

BES-only has 614 test bins on 26 shots, only 36 L bins on three shots, and
macro-F1 0.720909. Its headline score is not directly comparable to full on
1,403 bins. The saved common-bin comparison correctly uses shared exposure
and has 1,907 two-class bootstrap draws, not 2,000 usable draws.

These missingness outcomes concern the specified BES-removal shift. The
unchanged absence of all test spectral bands, unusable CO2/W_MHD, and complete
D-alpha/Thomson/NBI support preclude claims of spectral generalization or
arbitrary missing-modality robustness.

## Four-class result

The four-class baseline has macro-F1 0.845238 and balanced accuracy 0.856121
on 1,401 subtype-known bins. F1 L/H/QH/WP is
0.850829/0.982097/0.741573/0.806452. Its macro-F1 interval
[0.622460,0.928033] uses 1,760 of 2,000 draws containing every reference
class; the manifest explicitly states that conditioning.

Only two QH and four WP test shots support subtype estimates. The QH F1
interval is [0.400000,0.892857] and WP [0.206897,0.928105]. These are
descriptive subtype-recognition results against imported targets, not reliable
independent confinement-regime validation. No new four-class passing bar was
invented after test opening.

## Inference and manuscript framing

Saved inference smoke files show six observed bins with H suggestions and
complementary L-absent output; two completely missing bins retain category 3
and NaN scores in both H/L exports. Model and freeze hashes match. This checks
schema behavior and the shared extraction/inference path, not physical accuracy
of the smoke shot. The model card correctly calls probabilities uncalibrated
classification scores and forbids automatic default replacement.

The new manuscript audit distinguishes imported source consistency from gold,
uses “exactly matching entries” without claiming a verified direction of copying,
and states that surviving legacy QH/onset candidates lie outside the first/last
100 ms of annotated phases. The ELM-free QH/WPQH motivation is consistent with
[DIII-D's own Burrell-2020 publication description](https://fusion.gat.com/global/diii-d/papers);
that motivation does not adjudicate any candidate.

Current measured documentation and the model card retain H2 failure, wide L
uncertainty, sparse subtype support, and dependent supervision. No scientific
claim-hygiene blocker remains for this stage.

## Requirements for the historical comparison

Keep all current models, thresholds, targets and the primary definition frozen.
Score the historical detector on the same exact eligible assessed windows,
with its own frozen threshold and documented temporal alignment. Audit its
training/validation overlap with the new test shots and separate exposed,
held-out, and unknown-exposure domains. If no genuinely common held-out domain
with both classes exists, report a diagnostic comparison rather than independent
model superiority. Do not reinterpret H2 failure, switch the primary model,
retune after comparison, or promote the default based on a contaminated cohort.
