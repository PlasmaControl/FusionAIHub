# RWM baseline on DIII-D: an onset-derived forecasting reference

The owner asked for a reasonable baseline from Piccione et al. (2022) when no
DIII-D RWM model was available. This implements that modest reference: a balanced
random forest, retrained on Hanson shots, forecasting a listed n=1 onset. It is
conditional on assumed negative coverage and conventional positive windows. It is
not a validated RWM detector or an estimate of physical instability duration.

The headline rwm-brf AUROC is **0.760 [0.706, 0.809]**, with AUPRC
**0.163 [0.123, 0.220]**, on Piccione-style pre-onset negatives. The causal
elapsed-time rule has AUROC **0.759 [0.712, 0.815]**. The forest primarily separates
discharge phases; discrimination within the high-beta phase is weak:
**0.602 [0.491, 0.688]** AUROC. Above the no-wall proxy it is
**0.546 [0.416, 0.642]**. These results do not establish an advantage over elapsed
time, beta_N alone, or beta_N/l_i in AUROC. Source: `outputs/labeler/rwm/evaluation.json`,
`configs.<model>.metrics.{slice_auroc,slice_auprc,high_beta_auroc,above_proxy_auroc}`
and `paired.rwm-brf - <rule>`.

## Evidence tiers and the two targets

Hanson's listed onset **points** are the only verified evidence. Complete reviewed
coverage, including negative coverage on those same shots, is unknown. The table
`data/events/resistive_wall_mode/extend_rwm_growth/rwm_windows.csv` retains the
interval-table schema and records `evidence_tier` and `coverage_verified=false` in
its JSON `attrs` column. Its metadata spells out the assumptions.

- **Conventional weak annotation:** the 20 ms before each listed onset. This is a
  convention motivated by the millisecond wall flux-penetration time tau_w. It is
  not a measured growth time or a verified instability interval, for either n=1
  or n=2. The curated point evidence remains in the original onset tables.
- **Assumed absent:** Hanson high-current time outside 100 ms either side of every
  listed onset. This assumes the list is complete on those shots. It is not
  expert-verified absence. The precursor outside the short weak window and the
  immediate aftermath carry no assessed interval row.
- **Unlabelled:** every slice of comparison shots. Screen spans are unlabelled
  candidates for review, never negative ground truth or primary training negatives.

The catalog's weak 20 ms interval and the model's 100 ms forecast target are
different. For forecasting, a slice is positive if an n=1 onset follows within
100 ms. Piccione's digest defines earlier slices of an unstable shot as negative;
it does not designate the post-onset collapse as stable time. For multiple DIII-D
onsets, our primary negative coverage ends at the **last n=1 onset**. Earlier
onsets' following 100 ms and n=2 onsets' surrounding 100 ms are excluded unless a
later n=1 forecast window makes a slice positive. The n=2-only Hanson shots have
no primary n=1 negatives. Every primary negative remains an assumption.

The **broader-negative sensitivity** restores post-last-onset Hanson time and
n=2-only Hanson time as assumed negatives, using the former label mask. It scores
the same newly trained model's held-out predictions; it does not refit on that
broader set. Its rwm-brf AUROC is **0.740 [0.668, 0.799]**, AUPRC
**0.065 [0.043, 0.103]**. The previous headline, trained with comparison slices as
negatives, is superseded. No mixed comparison-as-negative score is reported.
Source: `evaluation.json/configs.rwm-brf.metrics.broad_{auroc,auprc}`.

## Data and input limitations

The roster contains 33 Hanson shots and 132 matched comparison shots from two
campaigns. There are 56 listed points, merged to 48 n=1 and 6 n=2 events; 30 shots
have an n=1 onset. None of the selected shots overlaps the frozen cohort. Primary
slice scoring uses 480 positives and 5,255 assumed negatives; 9,184 Hanson slices
are excluded. Comparison slices remain unlabelled. Sources:
`outputs/labeler/rwm/shots.json/{hanson,comparison,cohort_overlap,slices}` and
`evaluation.json/configs.rwm-brf.counts`.

Comparison shots are matched without reuse within campaign on flat-top p95 beta_N
and beta_N/l_i, from the same run days or RWM-experiment days. The matching is
imperfect, especially in the earlier campaign. It does not create a stable control
cohort or establish exchangeable label missingness. Exact campaign balance and the
leftover pool are recorded in `shots.json/comparison.balance`; per-shot matching is
in `rwm_windows.shots.csv`. We make no optimality claim for this greedy match.

Inputs are beta_N, l_i, q95, qmin, W_MHD, |Ip|, derived beta_N/l_i and beta_N-4l_i,
N1RMS and N2RMS, and ZIPFIT toroidal rotation at the two configured radii. The RMS
features are trailing means, maxima and log slopes. N1RMS is a rotating-mode RMS,
not a direct low-frequency RWM sensor. No DCON wall limits, E-cross-B frequency,
ion collisionality or MHD peak-frequency reconstruction is available in this run.
This feature set is an adaptation of Piccione's, not a reproduction of NSTX inputs.

**DUSBRADIAL is zero in most of the selected 2014 traces** (79 of 100 shots,
including 15 of 20 Hanson shots). The cached data also contain nonzero 2014 traces,
so the earlier claim that every 2014 trace was zero was too broad. The namespace
flags the radial-field data for **176030–176912 as corrupted**, covering every
2018 Hanson shot. DUSBRADIAL is excluded from rwm-brf. Source:
`shots.json/input_audit.{dusbradial_zero_2014_shots,dusbradial_by_role_campaign,
dusbradial_2014_nonzero_shots,dusbradial_corrupted_shot_range}`, with the corruption
provenance in `src/labeler/features/namespace.py` (Fu et al. 2020 note).

A low-frequency n=1 saddle-loop/Bp RWM amplitude was **not available through the
approved FETCH_SPECS or feature namespace**: neither lists a validated locator for
it. No guessed node was fetched, and availability elsewhere in the archive has
not been established. The approved specification inventory and decision are in
`shots.json/input_audit.{fetch_specs,low_frequency_n1_rwm_sensor_specs,
low_frequency_sensor_fetch_attempted,low_frequency_sensor_status}`.

Resampling holds the last available sample and uses trailing windows, but ZIPFIT's
upstream time smoothing is mildly acausal. The high-current analysis span (longest
run with |Ip| above half the whole-shot peak) is retrospective. The candidate
screen's median-plus-MAD threshold also uses the **whole flat-top**: its alarm
results are a retrospective screen comparator, not a causal online baseline.
The elapsed-time rule instead uses the first fixed |Ip| >= 0.5 MA crossing as a
causal operational proxy for flat-top start; no future peak sets that start.

## Growth search: the matched control rejects a growth-time justification

`scripts/labeler/rwm_growth.py` applies exactly the same maximum trailing-log-slope
search to onsets and random flat-top centres on the n=1 Hanson shots. Each search
uses the same 90 offsets from -150 to +28 ms and the same trailing 20 ms fit.
Controls are at least 170 ms from every listed onset and have room for the full
search inside the high-current span. The median maximum is **111.7 per second** at
onsets and **109.8 per second** at 7,704 controls. The median slope evaluated at the
onset itself is **-1.75 per second**. These are search statistics of a nonspecific
signal; they do not identify RWM growth time. The 20 ms annotation therefore rests
on the tau_w convention alone. Source: `outputs/labeler/rwm/growth.json/
{search,controls,max_growth_per_s_n1,control_max_growth_per_s_n1,
slope_at_onset_per_s_n1}`. Per-centre rows are in the CSV paths in that JSON.

## Models and validation

The primary **rwm-brf** fits Hanson positives and Hanson assumed negatives only.
Neither comparison slices nor excluded Hanson time enter its imputer or forest.
The forest reimplements per-tree balanced bootstrapping in NumPy; settings are
fixed at 300 trees, depth 8 and minimum leaf size 5. No architecture or
hyperparameter selection uses an outer held-out shot. The rules score increasing
elapsed time, beta_N, beta_N/l_i, or the binary candidate-screen call; their
orientations are fixed before scoring. No single-feature direction is selected
from held-out outcomes. There is no comparison-negative training variant in the
formal run.

Missing rule inputs rank below every finite score and do not trigger an alarm;
the forest instead uses training-fold median imputation. Comparisons depend on
this missing-data convention, rather than using complete-case subsets per rule.

There are 5 outer and 3 inner folds, grouped by shot and stratified by role and
campaign. Training medians, the ROC slice cutoff, and hysteresis alarm thresholds
come from training shots only. Inner out-of-fold Hanson traces tune the alarm
objective (n=1 detection share minus unexplained-alarm shot incidence); comparison
traces never tune it. The alarm penalty is over full Hanson high-current traces,
including post-last-onset time, and assumes completeness of the onset list. It is
not verified negative coverage. Target n=1 onsets are passed separately from all
n=1/n=2 events used to explain alarms. An n=2 event can explain an alarm but cannot
reward detection of a headline target. Warnings must fall 10–400 ms before a target.

All displayed metric intervals use 1,000 shot-bootstrap resamples, independently
within Hanson and comparison strata. Paired differences use the same shot draws
for both models. These intervals describe sampling of shots at fixed fitted
out-of-fold predictions, not uncertainty from fitting or choosing a new fold split.
The primary split seed is fixed at zero; the split-sensitivity table reports five
seeds. The isolated rotation-benefit claim has been dropped. Protocol and every
fold's thresholds: `evaluation.json/{protocol,configs.<model>.rules_by_fold}`.

**nnPU is a development note only.** The earlier experiment used outer-fold
feedback for network/loss settings and used a labelled-positive fraction that did
not identify the prior among comparison slices. It is excluded from the formal
configuration list, tables, paired comparisons and headline README. Its library
code remains for development; no development-biased number is reported here.
A future PNU experiment would use Hanson assumed negatives as N and comparison
slices as U, sweep a separately justified U prior, and select every setting inside
training folds.

## Conditional scores and their interpretation

Both conditional evaluations use the **primary Piccione-style mask**. High-beta
means beta_N >= 0.8 times the shot's whole-window p95; this retrospective evaluation
stratum is never a predictor or tuning threshold. Above-proxy means beta_N/l_i > 4.
Their positive-slice prevalences are **0.134** and **0.160**, respectively, versus
**0.084** for the primary set. rwm-brf's conditional AUPRCs are
**0.168 [0.123, 0.228]** and **0.176 [0.130, 0.240]**. The paired conditional
intervals against elapsed time, beta_N and beta_N/l_i do not support a forest AUROC
advantage. The time rule has higher AUPRC on this constructed target; this supports
the phase interpretation, rather than RWM-specific discrimination. Sources:
`evaluation.json/configs.<model>.{counts,metrics}` and
`paired.rwm-brf - <rule>.{slice,high_beta,above_proxy}_{auroc,auprc}`.

The forest warns of **1 of 48** headline onsets in the primary split, with an
unexplained alarm on **4 of 33** Hanson shots and an alarm on **1 of 132** unlabelled
comparison shots. Comparison incidence is **0.008 [0.000, 0.023]**, not a false
positive rate conditional on stable shots. Alarm incidence changes substantially
with fold split; no robust operating point or rotation benefit is established.
The uniform-alarm reference assumes independent uniform placements over each
shot's scored span; it is only a rough reference. Detection minus that reference
is **0.003 [-0.031, 0.047]**. The warning-time bootstrap is conditional on detecting
an onset; with only one detection its interval degenerates and conveys no timing
precision. Sources: `evaluation.json/configs.rwm-brf.{counts,metrics,per_shot,split_seeds}`.

This is an assumption-conditioned baseline on a small onset-derived roster. It
cannot establish RWM-stable coverage, calibration, performance on the blind gold
cohort, or generalisation across campaigns. Piccione's expert-reviewed NSTX cohort,
features and validation differ; no pooled DIII-D score is presented as a comparable
published benchmark.

## Paper outputs and reproduction

One figure shows held-out rwm-brf scores and beta_N/l_i for six Hanson shots and two
unlabelled comparison shots, selected by shot number and matching rather than model
performance. Listed onsets and conventional forecast windows are distinct from the
matched reference times on comparison shots. The PDF and 150-dpi PNG are
`$LABELER_ROOT/round4/rwm/rwm_onset_scores.{pdf,png}`; source rows, selection and the
caption are in `outputs/labeler/rwm/figure.json`. The PNG was visually inspected.
The paper table is `$LABELER_ROOT/round4/rwm/table_rwm.tex`, with row models/rules,
primary AUROC/AUPRC and conditional high-beta AUROC, each with CIs. Cell provenance
is in `outputs/labeler/rwm/presentation.json`.

Reproduce with `scripts/labeler/rwm_build.py`, `rwm_growth.py`,
`rwm_evaluate.py --workers 5 --replicates 1000`, `rwm_tables.py`, and `rwm_figure.py`,
using the environment and temp-directory rules of this stream. Stored predictions
and all large outputs live under `$LABELER_ROOT/round4/rwm/`. No fetch is required
for this fix round. The tables that follow are generated by `rwm_tables.py`.
All score cells come from `evaluation.json/configs.<row-model>.metrics.<metric>`;
count/prevalence cells from `counts`; paired cells from `paired.<row-pair>.<metric>`;
split rows from `configs.rwm-brf.{metrics,split_seeds.<seed>.metrics}`.

## Generated score and alarm tables

### Piccione-style primary scores — all models

| model | AUROC (95% CI) | AUPRC (95% CI) |
|---|---|---|
| rwm-brf | 0.760 [0.706, 0.809] | 0.163 [0.123, 0.220] |
| rule-time-since-flattop | 0.759 [0.712, 0.815] | 0.228 [0.212, 0.307] |
| rule-betan | 0.707 [0.639, 0.772] | 0.166 [0.128, 0.246] |
| rule-betan-over-li | 0.723 [0.664, 0.784] | 0.159 [0.127, 0.232] |
| rule-rwm-candidates | 0.500 [0.500, 0.500] | 0.084 [0.070, 0.102] |

### Broader Hanson-negative sensitivity — same models and predictions

| model | AUROC (95% CI) | AUPRC (95% CI) |
|---|---|---|
| rwm-brf | 0.740 [0.668, 0.799] | 0.065 [0.043, 0.103] |
| rule-time-since-flattop | 0.390 [0.333, 0.440] | 0.025 [0.019, 0.032] |
| rule-betan | 0.715 [0.641, 0.776] | 0.064 [0.043, 0.099] |
| rule-betan-over-li | 0.752 [0.688, 0.815] | 0.074 [0.052, 0.115] |
| rule-rwm-candidates | 0.498 [0.495, 0.500] | 0.034 [0.026, 0.041] |

### High-beta conditional scores — all models

| model | AUROC (95% CI) | AUPRC (95% CI) | positive slices | assumed-negative slices | prevalence |
|---|---|---|---|---|---|
| rwm-brf | 0.602 [0.491, 0.688] | 0.168 [0.123, 0.228] | 365 | 2351 | 0.134 |
| rule-time-since-flattop | 0.642 [0.557, 0.732] | 0.255 [0.241, 0.343] | 365 | 2351 | 0.134 |
| rule-betan | 0.563 [0.468, 0.651] | 0.176 [0.132, 0.273] | 365 | 2351 | 0.134 |
| rule-betan-over-li | 0.593 [0.509, 0.677] | 0.165 [0.132, 0.249] | 365 | 2351 | 0.134 |
| rule-rwm-candidates | 0.500 [0.500, 0.500] | 0.134 [0.108, 0.171] | 365 | 2351 | 0.134 |

### Above no-wall-proxy conditional scores — all models

| model | AUROC (95% CI) | AUPRC (95% CI) | positive slices | assumed-negative slices | prevalence |
|---|---|---|---|---|---|
| rwm-brf | 0.546 [0.416, 0.642] | 0.176 [0.130, 0.240] | 365 | 1913 | 0.160 |
| rule-time-since-flattop | 0.597 [0.524, 0.676] | 0.268 [0.254, 0.355] | 365 | 1913 | 0.160 |
| rule-betan | 0.509 [0.386, 0.617] | 0.183 [0.139, 0.279] | 365 | 1913 | 0.160 |
| rule-betan-over-li | 0.512 [0.428, 0.591] | 0.168 [0.132, 0.252] | 365 | 1913 | 0.160 |
| rule-rwm-candidates | 0.500 [0.500, 0.500] | 0.160 [0.125, 0.215] | 365 | 1913 | 0.160 |

### Alarm counts — all models (n=1 targets)

| model | onsets warned | Hanson shots: unexplained alarm | unlabelled shots: any alarm |
|---|---|---|---|
| rwm-brf | 1/48 | 4/33 | 1/132 |
| rule-time-since-flattop | 3/48 | 25/33 | 130/132 |
| rule-betan | 2/48 | 9/33 | 52/132 |
| rule-betan-over-li | 2/48 | 4/33 | 25/132 |
| rule-rwm-candidates | 0/48 | 7/33 | 36/132 |

### Alarm rates — all models (95% shot CIs)

| model | onset detection | Hanson unexplained incidence | alarm incidence on unlabelled shots |
|---|---|---|---|
| rwm-brf | 0.021 [0.000, 0.064] | 0.121 [0.030, 0.242] | 0.008 [0.000, 0.023] |
| rule-time-since-flattop | 0.062 [0.000, 0.130] | 0.758 [0.606, 0.879] | 0.985 [0.962, 1.000] |
| rule-betan | 0.042 [0.000, 0.109] | 0.273 [0.151, 0.424] | 0.394 [0.318, 0.470] |
| rule-betan-over-li | 0.042 [0.000, 0.103] | 0.121 [0.030, 0.242] | 0.189 [0.121, 0.258] |
| rule-rwm-candidates | 0.000 [0.000, 0.000] | 0.212 [0.091, 0.364] | 0.273 [0.197, 0.348] |

### Warning times — all models (detected onsets only)

| model | median warning, ms (95% CI) | uniform-alarm reference | detection minus reference |
|---|---|---|---|
| rwm-brf | 90 [90, 90] | 0.017 [0.003, 0.038] | 0.003 [-0.031, 0.047] |
| rule-time-since-flattop | 130 [36, 145] | 0.089 [0.075, 0.102] | -0.026 [-0.088, 0.046] |
| rule-betan | 201 [16, 385] | 0.033 [0.012, 0.057] | 0.009 [-0.035, 0.069] |
| rule-betan-over-li | 122 [56, 188] | 0.025 [0.006, 0.049] | 0.017 [-0.028, 0.069] |
| rule-rwm-candidates | - | 0.021 [0.006, 0.039] | -0.021 [-0.039, -0.006] |

### Paired differences — rwm-brf versus rules, primary

| first model minus rule | AUROC difference (95% CI) | AUPRC difference (95% CI) |
|---|---|---|
| rwm-brf - rule-time-since-flattop | 0.001 [-0.057, 0.052] | -0.065 [-0.145, -0.029] |
| rwm-brf - rule-betan | 0.053 [-0.034, 0.144] | -0.003 [-0.074, 0.058] |
| rwm-brf - rule-betan-over-li | 0.037 [-0.041, 0.111] | 0.004 [-0.063, 0.056] |
| rwm-brf - rule-rwm-candidates | 0.260 [0.206, 0.309] | 0.079 [0.043, 0.129] |

### Paired differences — rwm-brf versus rules, high-beta

| first model minus rule | AUROC difference (95% CI) | AUPRC difference (95% CI) |
|---|---|---|
| rwm-brf - rule-time-since-flattop | -0.040 [-0.140, 0.041] | -0.087 [-0.185, -0.046] |
| rwm-brf - rule-betan | 0.039 [-0.108, 0.180] | -0.008 [-0.103, 0.061] |
| rwm-brf - rule-betan-over-li | 0.009 [-0.105, 0.111] | 0.003 [-0.076, 0.055] |
| rwm-brf - rule-rwm-candidates | 0.102 [-0.009, 0.188] | 0.033 [-0.010, 0.087] |

### Paired differences — rwm-brf versus rules, above-proxy

| first model minus rule | AUROC difference (95% CI) | AUPRC difference (95% CI) |
|---|---|---|
| rwm-brf - rule-time-since-flattop | -0.051 [-0.195, 0.055] | -0.091 [-0.185, -0.045] |
| rwm-brf - rule-betan | 0.037 [-0.119, 0.197] | -0.007 [-0.095, 0.060] |
| rwm-brf - rule-betan-over-li | 0.034 [-0.099, 0.160] | 0.008 [-0.068, 0.063] |
| rwm-brf - rule-rwm-candidates | 0.046 [-0.084, 0.142] | 0.016 [-0.030, 0.071] |

### Paired alarm differences — rwm-brf versus rules

| first model minus rule | detection difference | Hanson incidence difference | unlabelled incidence difference |
|---|---|---|---|
| rwm-brf - rule-time-since-flattop | -0.042 [-0.102, 0.000] | -0.636 [-0.818, -0.455] | -0.977 [-1.000, -0.947] |
| rwm-brf - rule-betan | -0.021 [-0.093, 0.044] | -0.152 [-0.303, 0.000] | -0.386 [-0.462, -0.303] |
| rwm-brf - rule-betan-over-li | -0.021 [-0.091, 0.045] | 0.000 [-0.121, 0.121] | -0.182 [-0.250, -0.114] |
| rwm-brf - rule-rwm-candidates | 0.021 [0.000, 0.064] | -0.091 [-0.242, 0.061] | -0.265 [-0.341, -0.189] |

### Split sensitivity — rwm-brf (fixed hyperparameters)

| model | fold seed | primary AUROC | high-beta AUROC | detection rate | unlabelled alarm incidence |
|---|---|---|---|---|---|
| rwm-brf | 0 | 0.760 | 0.602 | 0.021 | 0.008 |
| rwm-brf | 1 | 0.787 | 0.651 | 0.042 | 0.182 |
| rwm-brf | 2 | 0.773 | 0.631 | 0.000 | 0.015 |
| rwm-brf | 3 | 0.806 | 0.696 | 0.042 | 0.030 |
| rwm-brf | 4 | 0.793 | 0.664 | 0.000 | 0.000 |
