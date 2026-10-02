# Confinement diagnostic-summary models, v1

These models supply exploratory confinement suggestions from native 50 ms
diagnostic windows. The primary binary model distinguishes L from the H/QH/WP
family. Dedicated no-BES, D-alpha+NBI and BES-only models are available alongside
a descriptive four-class baseline. They were fitted and evaluated on
1 October 2026; no existing default detector is automatically replaced.

Artifacts: `runs/labeler/confinement/v1/model/*.pkl`, with frozen selection,
feature schemas, thresholds and hashes in `training.json`, natural held-out
predictions in `predictions.csv`, and results in `evaluation.json`.
Full source and measured results: [confinement_analysis.md](confinement_analysis.md).

## Inputs and intended use

The extractor summarizes observed BES, filterscope D-alpha, raw Thomson,
Mirnov, NBI and optional cached equilibrium quantities. Missing measurements
remain NaN. CO2 and stored-energy features have no usable support in this run;
spectral features have no eligible test support. Thomson channel summaries are
not radially resolved pedestal measurements. The learner is weighted histogram
gradient boosting; probabilities are uncalibrated classification scores.

Use the full model on diagnostics similar to this corpus. The no-BES variant
supports the tested absence of BES, still requiring alternative measurements.
This run does not validate arbitrary diagnostic combinations, NBI-only inputs,
new devices, campaigns or population-wide prevalence. Entirely unavailable
model inputs produce category 3 (not observable), never an invented L label.
Suggestions require review before being treated as independent expert labels.

## Supervision, selection and validation

Kevin Gill and Jalal Butt's supplied tables contain exactly matching intervals;
their agreement does not establish independent annotation. The merge retains exact
source provenance and masks an internal 252 ms L/H conflict. Complete unknown
and partly assessed bins are excluded. Raw Test-only reservations hold across
sources; train/validation/test shots are disjoint. Training alone has equal
class mass and equal shot mass within each class after diagnostic exclusions.

The full model is the predeclared primary. Eight candidate configurations per
model and binary thresholds were selected on validation alone, then frozen
before test opening. Test support is 28 shots / 1,403 bins, including 103 L bins
on five shots (only three contribute more than one L bin). Full F1-L is 0.925
with 95% shot-bootstrap interval [0.571, 0.990]; F1-H is 0.994, macro-F1 0.960.
There are 1,993 valid two-class bootstrap samples of 2,000 requested.

Original manuscript H1 passes; H2 fails because the paired H-F1 gain over
always-H includes zero [-0.00043, 0.07969]. Macro-F1 gain over always-H is
positive [0.286, 0.513], but does not replace H2. Dedicated no-BES macro-F1 is
0.962; its paired difference from full does not establish a BES benefit.
Four-class macro-F1 is 0.845, with only two QH and four WP test shots. Neither
task has been tested against a new independent gold annotation set.
The full and no-BES training cohorts contain 68 and 67 eligible shots; recipe
comparisons do not isolate causal modality contributions. Fits use one seed,
and intervals quantify held-out shot uncertainty conditional on these models.

## Running the workflow

From the repository root, choose a fresh experiment directory before any write
step. Preserve the saved `v1` record: upstream label/extraction commands can
overwrite files even though fitting refuses to refit an existing freeze.
Source processing and exports use `labelmaker`:

```bash
export CONFINEMENT_RUN_DIR=runs/labeler/confinement/reproduce_20261001
# Use a new, previously nonexistent directory for each experiment.
pixi run --frozen -e labelmaker python -m labeler.confinement labels
pixi run --frozen -e labelmaker python -m labeler.confinement export
pixi run --frozen -e labelmaker python -m labeler.confinement elm --legacy
```

Extraction and training use the installed CPU environment:

```bash
OMP_NUM_THREADS=4 pixi run --frozen -e shot-design-cpu python -m labeler.confinement data
OMP_NUM_THREADS=4 pixi run --frozen -e shot-design-cpu python -m labeler.confinement train
```

The existing freeze prevents refitting silently after test opening. A future
independent experiment needs its own output directory and declared split.
Apply a frozen binary model without modifying source diagnostic stores:

```bash
OMP_NUM_THREADS=4 pixi run --frozen -e shot-design-cpu python -m labeler.confinement apply \
  --shot 185915 --start-ms 2450 --stop-ms 2750 --variant no_bes \
  --model-dir runs/labeler/confinement/v1 \
  --out "runs/labeler/confinement/inference/185915"
```

The output contains predictions, complementary common-schema H/L suggestions
and an inference audit. Missing outputs retain NaN confidence. Training and
inference share the same extractor; tested smoke cases include a real six-bin
H window and a fully missing two-bin window.
