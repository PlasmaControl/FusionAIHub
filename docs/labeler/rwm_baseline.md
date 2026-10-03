# RWM baseline on DIII-D: an onset-derived forecasting reference

The owner asked for a reasonable baseline from Piccione et al. (2022) when no
DIII-D RWM model was available. This implements that modest reference: a balanced
random forest, retrained on Hanson shots, forecasting a listed n=1 onset. It is
conditional on assumed negative coverage and conventional positive windows. It is
not a validated RWM detector or an estimate of physical instability duration.

The headline rwm-brf AUROC is **0.760 [0.706, 0.809]**, with AUPRC
**0.163 [0.123, 0.220]**, on Piccione-style pre-onset negatives. The causal
elapsed-time rule has AUROC **0.759 [0.712, 0.815]**. The forest primarily separates
discharge phases; primary within-phase discrimination is **not distinguishable from chance;
reversed in 2014**:
**0.602 [0.491, 0.688]** AUROC. Above the no-wall proxy it is
**0.546 [0.416, 0.642]**. These results do not establish an advantage over elapsed
time, beta_N alone, or beta_N/l_i in AUROC. Source: `outputs/labeler/rwm/evaluation.json`,
`configs.<model>.metrics.{slice_auroc,slice_auprc,high_beta_auroc,above_proxy_auroc}`
and `paired.rwm-brf - <rule>`. Primary per-slice F1 is
**0.275 [0.227, 0.337]**, using cutoffs selected in the inner training folds
(`configs.rwm-brf.metrics.slice_f1`).

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
- **Assumed absent:** Hanson high-current time before the **first** listed onset's
  100 ms precursor only. This assumes the list is complete on those shots; it is not
  expert-verified absence. It never establishes mode termination after an onset.
- **Unassessed (category 4):** explicit precursor holes and all physical post-onset
  time through analysis end, except later conventional weak windows. No termination
  or recovery evidence is available, so the physical state does not expire after
  100 ms. Comparison time outside screen spans is also explicitly unassessed.
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

The physical catalog is deliberately stricter: category 0 stops before the first
precursor, whereas the forecast target may assume that no *new listed onset* occurs
between events. Absence of a future onset does not imply physical absence of an
existing mode. The review reader retains category 0 and category 4 rows and their
evidence attributes; saving a tier-less edit against this tiered source is rejected.
The current category-only review editor refuses to save a tiered source; editing
these evidence attributes requires an evidence-aware workflow.
The export-to-reader regression uses shot 156785, including its precursor and
post-onset state.

The regenerated export has 54 conventional weak windows, 33 assumed-absent spans,
87 Hanson unassessed spans, 126 comparison screen spans and 258 comparison
unassessed spans. There are no verified absence intervals. Comparison screen spans
are clipped to their analysis window. Source: `shots.json/windows`.

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
traces never tune it. Primary alarm scoring and tuning ignore alarms after the last
listed n=1/n=2 onset plus 100 ms. This uses the slice mask's exclusion of unsupported
late time with a 100 ms alarm-association tolerance; it is not verified physical
absence. The former full-Hanson-trace objective is independently re-tuned and scored
as a sensitivity. Target n=1 onsets are passed separately from all
n=1/n=2 events used to explain alarms. An n=2 event can explain an alarm but cannot
reward detection of a headline target. Warnings must fall 10–400 ms before a target.

All displayed metric intervals use 1,000 shot-bootstrap resamples, independently
within Hanson and comparison strata. Paired differences use the same shot draws
for both models and **basic bootstrap** intervals (percentile endpoints reflected
around the observed difference); individual metric and campaign CIs use percentile
intervals. These intervals describe sampling of shots at fixed fitted
out-of-fold predictions, not uncertainty from fitting or choosing a new fold split.
The primary split seed is fixed at zero; the split-sensitivity table reports five
seeds. The isolated rotation-benefit claim has been dropped. Protocol and every
fold's thresholds: `evaluation.json/{protocol,configs.<model>.rules_by_fold}`.

The Hanson shots occupy only four logbook run records: 20140421, 20140620A,
20180314 and 20180314A. Primary shot-grouped folds can train on same-run siblings.
The leave-one-run-day-out sensitivity holds out each entire run record and selects
cutoffs and alarms only within the other three records. It uses the Hanson shots
alone; its pooled and campaign intervals resample shots at fixed held-out predictions
and do not estimate generalisation to a population of run days. The two 20180314
records remain separate as specified by the logbook; they are not four independent
calendar dates. Sources: `evaluation.json/leave_one_run_day_out` and
`rwm_windows.shots.csv/run`.

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
advantage. Elapsed time has a higher point AUPRC, but its top-scoring slices are
concentrated in very few shots and the primary basic-bootstrap paired interval
includes zero: forest minus time AUPRC is **-0.065 [-0.100, 0.016]**.
The top 60 elapsed-time scores come from 176078 (48 slices) and 176069 (12 slices),
per `configs.rule-time-since-flattop.top_score_concentration`. This does not establish
an AUPRC advantage. Sources:
`evaluation.json/configs.<model>.{counts,metrics}` and
`paired.rwm-brf - <rule>.{slice,high_beta,above_proxy}_{auroc,auprc}`.

Campaign-specific results expose a reversal hidden by pooling. Primary high-beta
AUROC is **0.311 [0.220, 0.408]** in 2014 versus
**0.698 [0.569, 0.801]** in 2018; the earlier campaign
is below chance. Above-proxy AUROC is **0.277 [0.172, 0.382]**
versus **0.647 [0.526, 0.763]**. The campaign table
reports primary pooled-over-slices and both conditional AUROC/AUPRC with campaign
shot-bootstrap CIs; it resamples all Hanson shots in that campaign, including shots
with no eligible slices for a particular mask. Source:
`evaluation.json/configs.rwm-brf.by_campaign.<year>.{counts,metrics}`.

Leaving out whole run records gives primary AUROC **0.779 [0.731, 0.823]**,
AUPRC **0.181 [0.149, 0.248]** and F1 **0.285 [0.233, 0.344]**. Its high-beta AUROC
is **0.621 [0.536, 0.700]**, with this shot CI above chance, whereas above-proxy
AUROC **0.579 [0.475, 0.677]** still includes chance. It warns of **8/48**
onsets. Thus “not distinguishable from chance; reversed in 2014” describes the
primary shot-fold conditional result, not every sensitivity. Four fixed run records
and assumed labels do not establish run-population or physical RWM discrimination.
All four held-out-run score and alarm rows follow below. Source:
`evaluation.json/leave_one_run_day_out.{metrics,counts,by_run_day}`.

The retrospective screen never fires before a listed onset on these Hanson traces;
all of its calls fall in excluded time and its primary score is constant zero.
Its AUROC of 0.5 is therefore a constant-score reference. The paired forest-minus-screen
row has been removed. Source: `evaluation.json/screen_audit`.

Under primary alarm scoring, the forest warns of **9 of 48** headline onsets
(onset detection **0.188 [0.049, 0.333]**), with an unexplained alarm on
**8 of 33** Hanson shots and an alarm on
**19 of 132** unlabelled comparison shots. Comparison incidence is
**0.144 [0.083, 0.212]**, not a false-positive rate conditional on stable shots.
Piccione-style per-shot categories are **6 Detected / 22 Missed / 2 Early**
among the 30 shots with n=1 targets; 3 n=2-only shots have no headline target.
For multiple onsets, any 10–400 ms detection takes precedence; otherwise an
unexplained alarm more than 400 ms before a future target makes the shot Early,
then Missed. Early alarm counts remain visible even on Detected shots. The comparison
FP column is **alarm incidence on unlabelled shots**, not expert-reviewed stable-shot FPR.
The primary definition ignores 62 alarms on 11 Hanson shots after the last
listed onset plus 100 ms. Its physical post-onset state remains unassessed.

The separately retuned **full-trace sensitivity** retains **1/48** detections,
**4/33** Hanson unexplained-alarm shots and **1/132** comparison alarm shots. Every
one of those Hanson unexplained alarms follows a listed onset (156794, 158015,
176067 and 176087), illustrating the former post-onset penalty. Alarm times and
fold rules are recorded under `configs.rwm-brf.full_trace_alarm_sensitivity`.
Alarm incidence changes with fold split; no robust operating point is established.
The uniform-alarm reference assumes independent uniform placements over each
shot's considered span; it is only a rough reference. Primary detection minus that
reference is **-0.002 [-0.058, 0.052]**. Warning-time intervals
are conditional on detected onsets and do not establish general warning precision.
Sources: `evaluation.json/configs.rwm-brf.{counts,metrics,per_shot,split_seeds}`
and `configs.rwm-brf.full_trace_alarm_sensitivity.{counts,metrics,per_shot,rules_by_fold}`.

This is an assumption-conditioned baseline on a small onset-derived roster. It
cannot establish RWM-stable coverage, calibration, performance on the blind gold
cohort, or generalisation across campaigns. Piccione's expert-reviewed NSTX cohort,
features and validation differ; no pooled DIII-D score is presented as a comparable
published benchmark.

**Legacy (published NSTX reference, Piccione et al. 2022):** RUS forest AUROC
**0.918**, slice TPR **92.4%**, slice FPR **21.4%**, **10/11** unstable test shots
detected and **2/17** stable test shots with FP alarms. This is a **different
machine, expert-reviewed stable shots, not comparable** to this DIII-D benchmark.
Its seven NSTX inputs and training/test validation differ; its F1 and confidence
intervals are unavailable in the source digest. Source: `evaluation.json/legacy`,
transcribed by the committed evaluation script from the local Piccione digest
`FusionAIHub/.tmp/label_papers/Piccione_2022_Nucl._Fusion_62_036002.md`, sections
"Results: time slices" and "Results: per shot" (doi:10.1088/1741-4326/ac44af).

## Paper outputs and reproduction

One figure shows held-out rwm-brf scores and beta_N/l_i for six Hanson shots and two
unlabelled comparison shots, selected by shot number and matching rather than model
performance. Listed onsets and conventional forecast windows are distinct from the
matched reference times on comparison shots. The PDF and 150-dpi PNG are
`$LABELER_ROOT/round4/rwm/rwm_onset_scores.{pdf,png}`; source rows, selection and the
caption are in `outputs/labeler/rwm/figure.json`. The PNG was visually inspected.
On 156796 and 158022, beta_N/l_i collapses about 400 ms before the listed onset,
inside assumed-negative forecast time, visibly straining the onset-list completeness
assumption (`figure.json/caption` and `source_rows_csv`). The legend labels both the
forecast band and the no-wall line.
The paper table is `$LABELER_ROOT/round4/rwm/table_rwm.tex`, with row models/rules,
primary AUROC/AUPRC/F1 and conditional high-beta AUROC, each with CIs, and a separately
sourced Legacy subsection. Cell provenance
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

| model | AUROC (95% CI) | AUPRC (95% CI) | F1 (95% CI) |
|---|---|---|---|
| rwm-brf | 0.760 [0.706, 0.809] | 0.163 [0.123, 0.220] | 0.275 [0.227, 0.337] |
| rule-time-since-flattop | 0.759 [0.712, 0.815] | 0.228 [0.212, 0.307] | 0.267 [0.217, 0.331] |
| rule-betan | 0.707 [0.639, 0.772] | 0.166 [0.128, 0.246] | 0.246 [0.194, 0.312] |
| rule-betan-over-li | 0.723 [0.664, 0.784] | 0.159 [0.127, 0.232] | 0.261 [0.212, 0.328] |
| rule-rwm-candidates | 0.500 [0.500, 0.500] | 0.084 [0.070, 0.102] | 0.000 [0.000, 0.000] |

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

### Campaign sensitivity — rwm-brf (95% shot CIs)

| group | slice mask | Hanson shots | positive slices | assumed-negative slices | prevalence | AUROC (95% shot CI) | AUPRC (95% shot CI) |
|---|---|---|---|---|---|---|---|
| 2014 | primary | 20 | 260 | 2280 | 0.102 | 0.663 [0.596, 0.745] | 0.130 [0.099, 0.190] |
| 2014 | high-beta | 20 | 168 | 754 | 0.182 | 0.311 [0.220, 0.408] | 0.125 [0.091, 0.180] |
| 2014 | above-proxy | 20 | 176 | 705 | 0.200 | 0.277 [0.172, 0.382] | 0.132 [0.095, 0.200] |
| 2018 | primary | 13 | 220 | 2975 | 0.069 | 0.819 [0.744, 0.879] | 0.236 [0.159, 0.336] |
| 2018 | high-beta | 13 | 197 | 1597 | 0.110 | 0.698 [0.569, 0.801] | 0.229 [0.155, 0.332] |
| 2018 | above-proxy | 13 | 189 | 1208 | 0.135 | 0.647 [0.526, 0.763] | 0.244 [0.168, 0.362] |

### Leave-one-run-day-out — rwm-brf (95% shot CIs)

| group | slice mask | Hanson shots | positive slices | assumed-negative slices | prevalence | AUROC (95% shot CI) | AUPRC (95% shot CI) |
|---|---|---|---|---|---|---|---|
| pooled four-day holdout | primary | 33 | 480 | 5255 | 0.084 | 0.779 [0.731, 0.823] | 0.181 [0.149, 0.248] |
| pooled four-day holdout | high-beta | 33 | 365 | 2351 | 0.134 | 0.621 [0.536, 0.700] | 0.176 [0.149, 0.239] |
| pooled four-day holdout | above-proxy | 33 | 365 | 1913 | 0.160 | 0.579 [0.475, 0.677] | 0.186 [0.154, 0.258] |

### Leave-one-run-day-out — each held-out run day

| group | slice mask | Hanson shots | positive slices | assumed-negative slices | prevalence | AUROC (point estimate) | AUPRC (point estimate) |
|---|---|---|---|---|---|---|---|
| 20140421 | primary | 11 | 180 | 1273 | 0.124 | 0.759 | 0.240 |
| 20140421 | high-beta | 11 | 119 | 395 | 0.232 | 0.532 | 0.238 |
| 20140421 | above-proxy | 11 | 113 | 334 | 0.253 | 0.457 | 0.224 |
| 20140620A | primary | 9 | 80 | 1007 | 0.074 | 0.742 | 0.152 |
| 20140620A | high-beta | 9 | 49 | 359 | 0.120 | 0.546 | 0.164 |
| 20140620A | above-proxy | 9 | 63 | 371 | 0.145 | 0.482 | 0.164 |
| 20180314 | primary | 6 | 130 | 1414 | 0.084 | 0.813 | 0.297 |
| 20180314 | high-beta | 6 | 110 | 819 | 0.118 | 0.705 | 0.288 |
| 20180314 | above-proxy | 6 | 104 | 395 | 0.208 | 0.601 | 0.323 |
| 20180314A | primary | 7 | 90 | 1561 | 0.055 | 0.757 | 0.133 |
| 20180314A | high-beta | 7 | 87 | 778 | 0.101 | 0.600 | 0.143 |
| 20180314A | above-proxy | 7 | 85 | 813 | 0.095 | 0.639 | 0.147 |

### Leave-one-run-day-out — F1 and alarms (95% shot CIs)

| held-out group | primary F1 (95% shot CI) | onsets warned | onset detection (95% shot CI) | Early Hanson shots | Hanson shots with an unexplained alarm | Hanson unexplained incidence (95% shot CI) | median warning, ms (95% shot CI) |
|---|---|---|---|---|---|---|---|
| pooled four-day holdout | 0.285 [0.233, 0.344] | 8/48 | 0.167 [0.049, 0.309] | 1/30 | 9/33 | 0.273 [0.152, 0.424] | 214 [140, 343] |

### Leave-one-run-day-out — F1 and alarms by held-out run day

| held-out group | primary F1 (point estimate) | onsets warned | onset detection (point estimate) | Early Hanson shots | Hanson shots with an unexplained alarm | Hanson unexplained incidence (point estimate) | median warning, ms (point estimate) |
|---|---|---|---|---|---|---|---|
| 20140421 | 0.364 | 0/18 | 0.000 | 0/11 | 0/11 | 0.000 | - |
| 20140620A | 0.249 | 4/8 | 0.500 | 0/7 | 5/9 | 0.556 | 214 |
| 20180314 | 0.329 | 1/13 | 0.077 | 0/6 | 0/6 | 0.000 | 140 |
| 20180314A | 0.153 | 3/9 | 0.333 | 1/6 | 4/7 | 0.571 | 343 |

### Alarm counts — all models (n=1 targets)

| model | onsets warned | Hanson shots: unexplained alarm | unlabelled shots: any alarm |
|---|---|---|---|
| rwm-brf | 9/48 | 8/33 | 19/132 |
| rule-time-since-flattop | 3/48 | 4/33 | 128/132 |
| rule-betan | 4/48 | 1/33 | 58/132 |
| rule-betan-over-li | 1/48 | 3/33 | 30/132 |
| rule-rwm-candidates | 0/48 | 0/33 | 36/132 |

### Alarm rates — all models (95% shot CIs)

| model | onset detection | Hanson unexplained incidence | alarm incidence on unlabelled shots |
|---|---|---|---|
| rwm-brf | 0.188 [0.049, 0.333] | 0.242 [0.091, 0.394] | 0.144 [0.083, 0.212] |
| rule-time-since-flattop | 0.062 [0.000, 0.137] | 0.121 [0.030, 0.242] | 0.970 [0.939, 1.000] |
| rule-betan | 0.083 [0.019, 0.163] | 0.030 [0.000, 0.121] | 0.439 [0.356, 0.515] |
| rule-betan-over-li | 0.021 [0.000, 0.064] | 0.091 [0.000, 0.212] | 0.227 [0.167, 0.295] |
| rule-rwm-candidates | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.273 [0.197, 0.348] |

### Warning times — all models (detected onsets only)

| model | median warning, ms (95% CI) | uniform-alarm reference | detection minus reference |
|---|---|---|---|
| rwm-brf | 356 [286, 389] | 0.190 [0.080, 0.302] | -0.002 [-0.058, 0.052] |
| rule-time-since-flattop | 145 [36, 235] | 0.046 [0.023, 0.069] | 0.016 [-0.039, 0.083] |
| rule-betan | 34 [16, 385] | 0.025 [0.006, 0.050] | 0.058 [0.014, 0.115] |
| rule-betan-over-li | 36 [36, 36] | 0.030 [0.006, 0.065] | -0.009 [-0.052, 0.030] |
| rule-rwm-candidates | - | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

### Piccione-style per-shot categories — primary alarm definition

| model | Detected Hanson shots | Missed Hanson shots | Early Hanson shots | Hanson shots without n=1 targets (excluded) | FP on comparison shots (alarm incidence) |
|---|---|---|---|---|---|
| rwm-brf | 6/30 | 22/30 | 2/30 | 3 | 19/132 |
| rule-time-since-flattop | 3/30 | 25/30 | 2/30 | 3 | 128/132 |
| rule-betan | 4/30 | 26/30 | 0/30 | 3 | 58/132 |
| rule-betan-over-li | 1/30 | 26/30 | 3/30 | 3 | 30/132 |
| rule-rwm-candidates | 0/30 | 30/30 | 0/30 | 3 | 36/132 |

Detected, Early and Missed are mutually exclusive on Hanson shots with an n=1 target: any Detected alarm takes precedence over Early, then Missed. Early means more than 400 ms before a listed onset. The FP column reports unlabelled-shot alarm incidence, not a verified stable-shot false-positive rate.

### Alarm definition sensitivity — rwm-brf

| alarm definition | onsets warned | onset detection (95% CI) | Early Hanson shots | Hanson shots with an unexplained alarm | Hanson unexplained incidence (95% CI) | unlabelled shots with an alarm | unlabelled alarm incidence (95% CI) |
|---|---|---|---|---|---|---|---|
| primary: end 100 ms after last n=1/n=2 onset | 9/48 | 0.188 [0.049, 0.333] | 2/30 | 8/33 | 0.242 [0.091, 0.394] | 19/132 | 0.144 [0.083, 0.212] |
| full-trace sensitivity | 1/48 | 0.021 [0.000, 0.064] | 0/30 | 4/33 | 0.121 [0.030, 0.242] | 1/132 | 0.008 [0.000, 0.023] |

The primary alarm window ends 100 ms after the last n=1 or n=2 explanation onset. This tolerance extends beyond the primary slice mask, which ends at the last n=1 target onset. Both alarm definitions are tuned within the inner folds; unlabelled comparisons never tune alarms.

### Legacy NSTX — separately sourced published reference

| published model | slice AUROC | slice TPR | slice FPR | detected unstable shots | false-positive stable shots |
|---|---|---|---|---|---|
| NSTX RUS forest | 0.918 | 92.4% | 21.4% | 10/11 | 2/17 |

Piccione et al. (2022), doi:10.1088/1741-4326/ac44af. Different machine (NSTX), expert-reviewed stable shots, different inputs and validation; these published test results are not comparable to the DIII-D benchmark. F1 and confidence intervals are not available in the source digest. Source in evaluation.json: legacy.source = /scratch/gpfs/nc1514/FusionAIHub/.tmp/label_papers/Piccione_2022_Nucl._Fusion_62_036002.md.

### Paired differences — rwm-brf versus rules, primary

| first model minus rule | AUROC difference (95% basic CI) | AUPRC difference (95% basic CI) |
|---|---|---|
| rwm-brf - rule-time-since-flattop | 0.001 [-0.049, 0.060] | -0.065 [-0.100, 0.016] |
| rwm-brf - rule-betan | 0.053 [-0.038, 0.140] | -0.003 [-0.064, 0.069] |
| rwm-brf - rule-betan-over-li | 0.037 [-0.038, 0.115] | 0.004 [-0.047, 0.071] |

### Paired differences — rwm-brf versus rules, high-beta

| first model minus rule | AUROC difference (95% basic CI) | AUPRC difference (95% basic CI) |
|---|---|---|
| rwm-brf - rule-time-since-flattop | -0.040 [-0.120, 0.061] | -0.087 [-0.128, 0.011] |
| rwm-brf - rule-betan | 0.039 [-0.101, 0.187] | -0.008 [-0.077, 0.087] |
| rwm-brf - rule-betan-over-li | 0.009 [-0.092, 0.123] | 0.003 [-0.049, 0.081] |

### Paired differences — rwm-brf versus rules, above-proxy

| first model minus rule | AUROC difference (95% basic CI) | AUPRC difference (95% basic CI) |
|---|---|---|
| rwm-brf - rule-time-since-flattop | -0.051 [-0.157, 0.092] | -0.091 [-0.138, 0.002] |
| rwm-brf - rule-betan | 0.037 [-0.123, 0.192] | -0.007 [-0.074, 0.081] |
| rwm-brf - rule-betan-over-li | 0.034 [-0.092, 0.167] | 0.008 [-0.046, 0.085] |

### Paired alarm differences — rwm-brf versus rules (95% basic CIs)

| first model minus rule | detection difference | Hanson incidence difference | unlabelled incidence difference |
|---|---|---|---|
| rwm-brf - rule-time-since-flattop | 0.125 [-0.021, 0.270] | 0.121 [-0.091, 0.333] | -0.826 [-0.894, -0.765] |
| rwm-brf - rule-betan | 0.104 [-0.053, 0.245] | 0.212 [0.061, 0.333] | -0.295 [-0.394, -0.197] |
| rwm-brf - rule-betan-over-li | 0.167 [0.015, 0.310] | 0.152 [0.000, 0.303] | -0.083 [-0.182, 0.015] |

### Split sensitivity — rwm-brf (fixed hyperparameters)

| model | fold seed | primary AUROC | high-beta AUROC | detection rate | unlabelled alarm incidence |
|---|---|---|---|---|---|
| rwm-brf | 0 | 0.760 | 0.602 | 0.188 | 0.144 |
| rwm-brf | 1 | 0.787 | 0.651 | 0.167 | 0.280 |
| rwm-brf | 2 | 0.773 | 0.631 | 0.208 | 0.174 |
| rwm-brf | 3 | 0.806 | 0.696 | 0.271 | 0.242 |
| rwm-brf | 4 | 0.793 | 0.664 | 0.125 | 0.091 |
