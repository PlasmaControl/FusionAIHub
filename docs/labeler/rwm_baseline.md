# RWM baseline on DIII-D: current protocol

- **Primary mask:** 100 ms pre-onset positives; assumed negatives end at the last n=1 onset, with aftermath/n=2 exclusions.
- **Broad mask:** same positives and exclusions; adds post-last-onset and n=2-only Hanson time as assumed negatives.
- **Reference split:** fixed shot-fold seed 0, not selected for its score.
- **Run-record holdout:** each of four logbook records is held out in turn; two records share a date.
- **Rate-matched reference:** approximate uniform random alarms with each target shot's alarm count and scored span.
- **Within-shot AUROC:** rank slices separately in each two-class Hanson shot, then average with equal shot weight.

This retrospective forest forecasts listed n=1 onsets in 33 Hanson shots; four
rules were compared. Reference-split pooled AUROC is **0.760 primary / 0.740 broad**.
Primary-mask elapsed time ranks almost perfectly within shot (median **1.0**).
No AUROC advantage over the strongest scalar, or onset-specific warning skill,
was established. Paired within-shot intervals place the forest below beta_N/l_i
on primary; neither beta_N mask nor broad beta_N/l_i is distinguishable.
Negative coverage, physical duration and online input timing remain unverified.
Sources: `E#/configs/<model>/{metrics,within_shot_auroc}` and
`E#/paired/rwm-brf - <scalar>/within_shot_auroc`; E is defined below.

## Records and provenance

Paths below are relative to the repository unless prefixed with `$LABELER_ROOT`.
JSON references use these aliases:

- **E**: [evaluation.json](../../outputs/labeler/rwm/evaluation.json), written by
  `scripts/labeler/rwm_evaluate.py`.
- **S**: [shots.json](../../outputs/labeler/rwm/shots.json), written by
  `scripts/labeler/rwm_build.py`.
- **G**: [growth.json](../../outputs/labeler/rwm/growth.json), written by
  `scripts/labeler/rwm_growth.py`.
- **F**: [figure.json](../../outputs/labeler/rwm/figure.json), written by
  `scripts/labeler/rwm_figure.py`.
- **P**: [presentation.json](../../outputs/labeler/rwm/presentation.json), written
  by `scripts/labeler/rwm_tables.py`.

All score, alarm, campaign, per-run-record and paired tables are in
[tables.md](../../outputs/labeler/rwm/tables.md). Split-0 tables describe a fixed
reference split; supplements give five-split ranges. A range of point estimates
across splits is not a confidence interval. No evaluation JSON is hand-edited.

## Data

The roster contains **33 Hanson shots** (20 in 2014, 13 in 2018) and **132 matched
comparison shots** (80 in 2014, 52 in 2018). The 56 curated points merge into
48 n=1 and 6 n=2 events; 30 Hanson shots have n=1 targets. Selected shots have
**zero overlap** with the fixed 500-shot train/validation/blind-test cohort.
Primary fitting and scoring use **480 positive / 5,255 assumed-negative 10 ms
slices**; 9,184 other Hanson slices are excluded. Comparison slices remain
unlabelled and never enter primary supervised fitting or tuning.
Sources: `S#/{hanson,comparison,cohort_overlap,slices}` and
`E#/configs/rwm-brf/counts`.

Comparison shots are matched without reuse, within campaign, on flat-top p95
beta_N and beta_N/l_i, from the Hanson run records or related RWM experiment
records. This greedy match does not create verified stable controls or ensure
exchangeable missing labels. Campaign balance and the unchosen pool are in
`S#/comparison/balance`; shot-level matches and run IDs are in
`data/events/resistive_wall_mode/extend_rwm_onset_window/rwm_windows.shots.csv`.

Inputs are beta_N, l_i, q95, qmin, W_MHD, |Ip|, beta_N/l_i, beta_N−4l_i, N1RMS,
N2RMS and ZIPFIT toroidal rotation at the configured radii. RMS features use trailing
means, peaks and log slopes: these are trailing calculations on offline inputs,
not proof of real-time availability. N1RMS/N2RMS are postprocessed amplitudes with
uncertain upstream timing; ZIPFIT's upstream time smoothing is acausal, with
unbounded timing bias here. No rotation ablation is needed for this negative
baseline because no skill or rotation benefit is claimed.
Analysis-span selection and the
candidate screen's whole-flat-top threshold are also retrospective. This is a
Piccione-style forest on generic 0D inputs; only beta_N and an RMS amplitude
concept overlap the seven NSTX inputs.
Sources: `E#/configs/rwm-brf/options/columns`, `S#/feature_coverage`,
`src/labeler/rwm/features.py` and `src/labeler/features/namespace.py`.

DUSBRADIAL is excluded: it is zero in 79/100 selected 2014 traces, including
15/20 Hanson traces, and the namespace flags shots 176030–176912 as corrupted,
covering all 2018 Hanson shots. N1RMS is rotating-mode RMS and cannot distinguish
RWM from tearing or applied-field response. No validated low-frequency n=1 RWM-specific
sensor is included. An isolated DIII-D probe tested existing OPERATIONS node
locators CN1BAMP, ILN1BAMP and IUN1BAMP on Hanson shots 156785, 158021 and 176068.
All nine returned finite nonzero Gauss traces with approximately 1 ms cadence;
node names and units do not establish corrected plasma-mode amplitude, filtering
or subtraction of applied-coil fields, so none was promoted to a model input.
No DCON wall limits, E-cross-B frequency, ion collisionality or MHD peak-frequency
reconstruction is available here. Sources: `S#/input_audit` and
[sensor_probe.json](../../outputs/labeler/rwm/sensor_probe.json), generated by
`scripts/labeler/rwm_sensor_probe.py` from an on-disk archive node inventory.

Rotation columns retain the inherited `rot_*_khz` names. Recorded cache units are
`units_from_source=kHz`, and the namespace also declares kHz. The hypothesis that
the physical values are krad/s remains unresolved; no conversion is applied and
no isolated rotation benefit is claimed. The fixed radii are **rho=0.25** (core)
and **rho=0.625** (mid-radius); the latter does not identify a q=2 surface.
In labelled Hanson slices, qmin>2 in about **83% (2014)** and **96% (2018)**
of all labelled slices, so most slices have no q=2 surface; missing qmin remains
in the denominator (2,108/2,540 in 2014; 3,074/3,195 in 2018). The conventional
beta_N≈4l_i no-wall proxy is uncertain for these high-qmin, low-li plasmas,
so transfer of a higher-qmin, 2018-like pattern to 2014 is only a hypothesis
for the conditional failure. Source:
`E#/onset_physics/labelled_slice_qmin` and `src/labeler/features/namespace.py`
(`rot_zipfit`) and `src/labeler/rwm/features.py`.

## Onset physics and sampling times

The generated [onset tables](../../outputs/labeler/rwm/tables.md) list beta_N,
l_i, beta_N/l_i and elapsed time for **all 48 merged n=1 points**, by campaign.
Values at the exact listed onset hold the last original EFIT sample for at most
**50 ms**, using the feature age limit; elapsed time starts at the first
|Ip| ≥0.5 MA sample. These are offline input calculations. At exact onset **five
points are below beta_N/l_i =4** and **four have missing EFIT**, including a
below-proxy 2018 point (176089). Source: `E#/onset_physics/{rows,by_campaign}`.

A separate table shows the **first 10 ms slice in [o−20 ms, o)** and its actual
sample time. This window-start snapshot flags **six below-proxy points** on
156785, 156786, 156787, 156794, 156796 and 158021, and **three missing-EFIT points**
on 156797, 176069 and 176070. Its 2018 finite values are all above the proxy;
that statement does not describe exact onset values. The distinction matters
near rapid beta collapses and missing-data boundaries. Sources:
`E#/onset_physics/{snapshot_scope,onset_scope,rows,by_campaign}`.

## Labels: physical catalog and future-onset target

Only curated onset **points** are confirmed evidence. Onset listing does not
establish reviewed negative coverage, growth duration or mode termination. Original
database events retain unknown confidence/coverage; the original 500-shot screen
export is header-only, not an absence claim.

The `rwm-onset-window` rule exports the physical interval CSV under the
`extend_<model>` layout:
`data/events/resistive_wall_mode/extend_rwm_onset_window/rwm_windows.csv`, with JSON
`attrs.evidence_tier` and `coverage_verified=false`, plus `rwm_windows.meta.json`:

- **Category 1, minimal present slice:** [o, o+10 ms) for each listed n=1 or
  n=2 onset, evidence tier `onset_point_minimal`. One 10 ms slice records minimal
  post-onset presence; it is not a measured duration or termination time. Present
  evidence takes precedence if a later uncertain pre-onset window overlaps it.
- **Category 2, uncertain pre-onset window:** [o−20 ms, o) by convention.
  Hanson `ONSET_TIME` does not specify growth start versus detection/threshold
  crossing, so pre-onset presence and the 20 ms extent remain unverified. Piccione's
  NSTX convention cannot establish this DIII-D meaning. Source searches are in
  `rwm_windows.meta.json/onset_time_provenance`.
- **Category 0, assumed absent:** high-current Hanson time before the **first**
  listed onset's 100 ms precursor, conditional on onset-list completeness. There
  is no post-onset reset to absence and no verified absence interval.
- **Category 4, unassessed:** precursor gaps and time beyond minimal present
  slices, except later uncertain onset windows. The physical state does not expire
  after 100 ms. Comparison time outside uncertain screen spans is category 4 too.
- **Comparison screen spans:** unlabelled candidates for review, never primary
  negatives. The screen requires N1RMS above the whole-flat-top median +6 MAD,
  beta_N >4l_i and at least 10 ms duration. Missing inputs are not observable.

The export has **612 rows on 165 shots**: 54 minimal present slices, 54 uncertain
onset windows, 33 assumed-absent, 87 Hanson unassessed, 126 comparison screen and
258 comparison unassessed spans.
Comparison spans are clipped to analysis coverage. The review reader retains
category 0/4 rows and aligned evidence attributes; the category-only editor rejects
saving tiered sources until an evidence-aware workflow is available.
Sources: `S#/{evidence,windows}`, interval metadata, and
[editor guide](equilibrium_review.md).

The growth search applies identical maximum trailing-log-slope searches to onsets
and random flat-top centres on n=1 Hanson shots: 90 offsets (−150 to +28 ms),
trailing 20 ms fits, controls at least 170 ms from listed onsets. Median maximum
N1RMS slope is **111.7/s at onsets versus 109.8/s at 7,704 controls**; slope at the
onset itself has median **−1.75/s**. Comparable control maxima cannot validate
physical growth extent or pre-onset presence in the 20 ms uncertain window.
Sources: `G#/{search,controls,max_growth_per_s_n1,control_max_growth_per_s_n1,
slope_at_onset_per_s_n1}` and its per-centre CSV paths.

The model target is separate: a slice is positive when a listed **n=1 onset follows
within 100 ms**. Primary assumed negatives are earlier than the last n=1 onset;
earlier onsets' following 100 ms and n=2 onsets' surrounding 100 ms are excluded
unless a later n=1 forecast window makes the slice positive. n=2-only shots have no
primary n=1 negatives. Between-event forecast negatives mean no new listed onset,
not physical absence of an existing mode. The broader-negative sensitivity adds
post-last-onset and n=2-only Hanson time as assumed negatives to the same held-out
scores, without refitting. Sources: `E#/protocol`, `src/labeler/rwm/labels.py`.

## Splits and models

Five outer and three inner folds are shot-grouped and stratified by role/campaign.
All windows of a shot stay together. Imputation medians, slice cutoffs and alarm
rules are selected on training shots only; comparison traces never tune them.
Hyperparameters are fixed across split seeds **0–4**; seeds also change tree
randomisation. Split 0 remains the reference for full score/alarm tables, rather
than being selected for its score. Sources: `E#/protocol` and
`E#/configs/<model>/{fold_seed,rules_by_fold,split_seeds}`.

**Leave-one-run-record-out** retains four logbook records: 20140421 (11 shots),
20140620A (9), 20180314 (6) and 20180314A (7). Each outer fold holds out all sibling
shots of one record; inner shot folds fit/tune only on the other records. This is
Hanson-only. **20180314 and 20180314A remain separate despite sharing a date**;
calendar-day isolation is not claimed. Primary shot folds can train on same-record
siblings. Sources: `E#/leave_one_run_record_out/{protocol,by_run_record}`.

The forest uses NumPy per-tree balanced bootstrapping with **300 trees, depth 8,
minimum leaf size 5**, and training-only median imputation. Neither comparison
slices nor excluded Hanson time enter the imputer or forest. Fixed rules increase
with elapsed time, beta_N, beta_N/l_i or the binary candidate-screen call;
orientations are fixed before scoring. Missing scalar inputs rank below finite
scores and do not trigger an alarm. Elapsed time starts at the first fixed
|Ip| ≥0.5 MA crossing, a causal proxy requiring no future peak. The candidate
screen is retrospective and constant zero on primary Hanson slices; all 62 Hanson
calls occur after a listed onset, so no forest-minus-screen paired row is reported.
Sources: `E#/protocol/forest`, `E#/screen_audit`, `src/labeler/rwm/evaluate.py`.

nnPU is excluded from formal configurations: earlier development used outer-fold
feedback and an unidentified prior among comparison slices. No isolated rotation
benefit or comparison-as-negative training result is claimed.
Source: `E#/protocol/{nnpu,rotation_claim,primary_training}`.

## Metrics and alarm definitions

Slice AUROC/AUPRC use Hanson primary labels. F1 uses the ROC cutoff nearest
(0 FPR, 1 TPR), selected from inner-fold predictions. High-beta conditional scoring
restricts primary slices to beta_N ≥0.8 times the shot's whole-window p95;
above-proxy conditional scoring uses beta_N/l_i >4. These retrospective evaluation
masks are not predictors or tuning thresholds and do not remove discharge phase.
Prevalences are **0.084 primary / 0.134 high-beta / 0.160 above-proxy**.
Sources: `E#/protocol`, `E#/configs/rwm-brf/counts`.

Alarm thresholds use primary-negative score quantiles and hysteresis with a hold
duration. Inner-OOF Hanson traces maximise n=1 detection share minus unexplained
alarm shot incidence. Warnings must be **10–400 ms before an n=1 target**. n=1/n=2
onsets can explain alarms, but n=2 events cannot reward n=1 detection. Primary
Hanson tuning/scoring ends at the last n=1/n=2 explanation onset +100 ms; comparison
alarm traces retain their **full analysis span**. The 100 ms tolerance extends
beyond the primary slice mask, not the evidence for physical absence. Full-Hanson-
trace alarm sensitivity is separately retuned on the same inner folds and forest
scores. Sources: `E#/protocol/{alarm_grid,alarm_tuning,primary_alarm_scope,
full_trace_alarm_sensitivity}`.

Per-shot Detected/Early/Missed categories are mutually exclusive on n=1 target
shots: any accepted warning wins, otherwise an unexplained alarm >400 ms before a
future target makes Early, otherwise Missed. Raw Early counts remain visible on
Detected shots. n=2-only shots are No target. Comparison FP means alarm incidence
on unlabelled shots, not verified stable-shot FPR. Sources:
`E#/protocol/shot_categories`, `E#/configs/<model>/{counts,per_shot}`.

Individual/campaign intervals are percentile CIs from **1,000 shot resamples**,
within Hanson and comparison strata, including shots with no eligible slices in a
conditional mask. Detection-minus-random-reference also uses percentile
shot-bootstrap intervals.
Between-model paired differences share shot draws and use **basic** bootstrap
CIs (reflected percentile endpoints). All intervals condition on fixed fitted OOF
predictions; they do not include refitting, fold selection or run-population
uncertainty. The random-alarm reference keeps each target shot's alarm count and
places alarms independently/uniformly over its scored span, intersecting each
onset's warning window with that span; it is approximate.
Warning-time CIs condition on detected onsets. Sources: `E#/protocol`,
`E#/paired_method`, `src/labeler/rwm/metrics.py` and `evaluate.py/chance_detection`.

## Results

Reference-split within-shot means (30 two-class Hanson shots, equal shot weights;
`E#/configs/<model>/within_shot_auroc/{primary,broad}/mean`):

| Model / rule | Primary mean AUROC | Broad mean AUROC |
|---|---|---|
| rwm-brf | 0.784 | 0.719 |
| Elapsed time | 0.931 | 0.417 |
| beta_N | 0.814 | 0.739 |
| beta_N/l_i | 0.855 | 0.786 |
| Candidate screen | 0.500 | 0.499 |

Forest minus scalar within-shot mean AUROC (95% basic paired shot-bootstrap
intervals; 1,000 replicates, seed 0; 30 shared two-class shots per mask):

| Scalar | Primary difference | Broad difference |
|---|---|---|
| Elapsed time | -0.147 [-0.200, -0.084] | +0.302 [+0.217, +0.384] |
| beta_N | -0.031 [-0.109, +0.045] | -0.020 [-0.095, +0.058] |
| beta_N/l_i | -0.071 [-0.133, -0.010] | -0.067 [-0.137, +0.002] |

Paired within-shot intervals place the forest below beta_N/l_i on primary;
neither beta_N mask nor broad beta_N/l_i is distinguishable. The forest is below
elapsed time on primary and above it on broad. Primary truncation makes time
almost perfect within shot (median **1.0**). These are differences of equal-shot
means, separate from pooled AUROC differences. Source:
`E#/paired/rwm-brf - <scalar>/within_shot_auroc`.

Broad pooled paired forest-minus-time AUROC is
**+0.350 [0.287, 0.414]** (run-record holdout **+0.362 [0.304, 0.428]**), while
forest-minus-beta_N/l_i is **−0.013 [−0.092, 0.072]**. These intervals establish
neither scalar equivalence nor onset-specific warning skill. Sources:
`E#/paired/rwm-brf - <rule>/broad_auroc`,
`E#/leave_one_run_record_out/paired_time/broad_auroc`.

Five-split AUROC point ranges (`E#/split_sensitivity/auroc_ranges`; broad:
`P#/supplemental_latex/split_summary/source/broad_ranges`):

| Scope | Primary | Broad | High-beta conditional | Above-proxy conditional |
|---|---|---|---|---|
| Pooled | 0.760–0.806 | 0.740–0.798 | 0.602–0.696 | 0.546–0.657 |
| 2014 | 0.663–0.735 | 0.693–0.780 | 0.311–0.527 | 0.277–0.508 |
| 2018 | 0.819–0.848 | 0.776–0.812 | 0.698–0.747 | 0.647–0.706 |

The forest has no established AUROC advantage over elapsed time in any of these
three **pooled** strata across the five seeds. Paired forest-minus-time point
ranges are 0.001–0.047, −0.040 to +0.054 and −0.051 to +0.061 respectively; all
paired 95% CIs include zero, with primary seed 3 borderline at
**0.047 [−0.0001, 0.098]**. For high-beta reference split (seed 0) the difference is
**−0.040 [−0.120, 0.061]**. This comparison is conditioned on the truncated
primary negative mask; it cannot establish onset-specific timing skill. Per-seed
paired intervals are in `E#/split_sensitivity/paired_time_by_seed`
and the generated tables; the point ranges are in `paired_time_ranges`.

In **2014 high-beta**, the forest is below chance on the reference split
(**0.311 [0.22, 0.41]**); the scalar rules are near chance (elapsed time **0.567**,
beta_N **0.516**, beta_N/l_i **0.472**). Transfer of a 2018-like pattern across the
different qmin regimes is a hypothesis only. Source:
`E#/configs/<model>/by_campaign/2014/metrics/high_beta_auroc`.

For the **2014 high-beta** stratum, forest-minus-elapsed-time AUROC is negative
on every split (**−0.255 to −0.039**, point estimates; CIs exclude zero on
**2 of 5 seeds**); the reference split difference is
**−0.255 [−0.422, −0.067]**. The 2014 above-proxy differences are also negative
on every split (**−0.262 to −0.031**). This conditional ranking failure is
accompanied by different high-beta target coverage: **7/26** targets in 2014
versus **1/22** in 2018 have no high-beta slice in their 100 ms forecast window.
In 2018, high-beta differences range
**−0.033 to +0.016**, with every CI including zero. Sources:
`E#/split_sensitivity/paired_time_by_campaign`, `E#/onset_physics/by_campaign`.

Leave-one-run-record-out AUROC (95% fixed-prediction shot CI;
`E#/leave_one_run_record_out/{metrics,by_campaign/<year>/metrics}`):

| Scope | Primary | Broad | High-beta conditional | Above-proxy conditional |
|---|---|---|---|---|
| Pooled | 0.779 [0.731, 0.823] | 0.752 [0.690, 0.805] | 0.621 [0.536, 0.700] | 0.579 [0.475, 0.677] |
| 2014 | 0.729 [0.672, 0.814] | 0.735 [0.656, 0.800] | 0.441 [0.369, 0.582] | 0.404 [0.329, 0.536] |
| 2018 | 0.796 [0.744, 0.845] | 0.762 [0.676, 0.843] | 0.661 [0.576, 0.746] | 0.633 [0.545, 0.729] |

Its AUPRC is **0.181 [0.149, 0.248]**, F1 **0.285 [0.233, 0.344]**, and alarms warn
**8/48** onsets. Pooled forest-minus-elapsed-time paired AUROC differences are **0.020
[−0.033, 0.080]** primary, **−0.021 [−0.097, 0.069]** high-beta and **−0.017
[−0.123, 0.105]** above-proxy. Campaign high-beta differences are **−0.126
[−0.245, 0.005]** in 2014 and **−0.070 [−0.117, −0.010]** in 2018. The **2018 primary** holdout
difference is also negative: **−0.049 [−0.073, −0.013]**; both 2018 primary and
high-beta holdout CIs exclude zero. See
`E#/leave_one_run_record_out/{paired_time,paired_time_by_campaign}`. Four fixed
records cannot establish population-of-run performance. Source: `E#/leave_one_run_record_out/{metrics,counts}`.

Reference split-0 scores (95% shot CIs; `E#/configs/<model>/metrics`):

| Model / rule | Primary AUROC | Broad AUROC | Primary AUPRC | Primary F1 |
|---|---|---|---|---|
| rwm-brf | 0.760 [0.706, 0.809] | 0.740 [0.668, 0.799] | 0.163 [0.123, 0.220] | 0.275 [0.227, 0.337] |
| Elapsed time | 0.759 [0.712, 0.815] | 0.390 [0.333, 0.440] | 0.228 [0.212, 0.307] | 0.267 [0.217, 0.331] |
| beta_N | 0.707 [0.639, 0.772] | 0.715 [0.641, 0.776] | 0.166 [0.128, 0.246] | 0.246 [0.194, 0.312] |
| beta_N/l_i | 0.723 [0.664, 0.784] | 0.752 [0.688, 0.815] | 0.159 [0.127, 0.232] | 0.261 [0.212, 0.328] |
| Candidate screen | 0.500 [0.500, 0.500] | 0.498 [0.495, 0.500] | 0.084 [0.070, 0.102] | 0.000 [0.000, 0.000] |

Elapsed time's higher point AUPRC does not establish a difference: forest-minus-
time AUPRC is **−0.065 [−0.100, 0.016]** under the basic paired bootstrap. Its top
60 scores are concentrated in 176078 (48 slices) and 176069 (12). Split-0
broader-negative sensitivity is **AUROC 0.740 [0.668, 0.799] / AUPRC 0.065
[0.043, 0.103]**, from identical predictions. Sources: `E#/paired/rwm-brf -
rwm-rule-elapsed-time/slice_auprc`, `E#/configs/rwm-rule-elapsed-time/
top_score_concentration`, `E#/configs/rwm-brf/metrics/broad_{auroc,auprc}`.
Broad elapsed-time AUROC is **0.390** versus forest **0.740** and beta_N/l_i
**0.752**; the broad paired differences above reverse the forest-versus-time
comparison while retaining no advantage over the best scalar. This is mask
sensitivity, not evidence of onset-specific skill or scalar equivalence.
Source: `E#/configs/{rwm-rule-elapsed-time,rwm-brf,rwm-rule-betan-over-li}/metrics/broad_auroc`.

Five-split forest alarms warn **6–13/48** onsets: detection **0.125–0.271**,
reference difference **−0.002 to +0.057**, median warning **135–356 ms**. No
improvement over the approximate random reference was established (all five CIs
include 0). The run-record holdout warns **8/48**: detection **0.167
[0.049, 0.309]**, reference difference **0.014 [−0.025, 0.053]**, median warning
**214 ms [140, 343]**. Warning intervals condition on detected onsets.
Sources: `E#/split_sensitivity/alarm_ranges` and
`E#/leave_one_run_record_out/{counts,metrics}`.

The reference split (seed 0) warns **9/48** onsets: **6 Detected / 22 Missed /
2 Early** among 30 n=1 target shots, plus 3 No target. Detection is **0.188
[0.049, 0.333]**, approximate reference **0.190 [0.080, 0.302]**, difference
**−0.002 [−0.058, 0.052]**, median warning **356 ms [286, 389]**. All four
rules were compared on the reference split only; their alarm results were not
replayed over seeds 1–4. Source: `E#/configs/<model>/{counts,metrics}`.

Hanson unexplained incidence is **8/33**, **0.242 [0.091, 0.394]**. Comparison
alarm incidence is **19/132**, **0.144 [0.083, 0.212]** on unlabelled shots. In 2014,
matched beta_N/l_i p95 mean is **4.43 versus 5.07 for Hanson** (unchosen pool 4.65),
biasing comparison incidence low for a beta-tracking model. Comparison alarm
traces retain their full span while Hanson traces end at last onset +100 ms,
adding exposure asymmetry. This is not stable-shot FPR. Primary scoring ignores
62 late alarms on 11 Hanson shots. Sources: `E#/configs/rwm-brf/{counts,metrics}`,
`E#/protocol/primary_alarm_scope`, `S#/comparison/balance/2014`.

Separately retuned full-trace alarms warn **1/48**, with **4/33** Hanson unexplained
shots and **1/132** comparison alarm shots. Those four Hanson unexplained alarms
follow listed onsets (156794, 158015, 176067, 176087). Slice scores/calls are
unchanged; alarm thresholds differ. No robust operating point is established.
Source: `E#/configs/rwm-brf/full_trace_alarm_sensitivity`.

**Legacy (Piccione et al., NSTX):** published RUS forest AUROC **0.918**, TPR
**92.4%**, FPR **21.4%**, **10/11** unstable test shots detected and **2/17** stable
test shots with FP alarms. Different machine, expert-reviewed stable shots,
inputs and validation: these results are not comparable to the DIII-D baseline.
Legacy F1/CIs are unavailable in the local digest. Source: `E#/legacy`, transcribed
by the evaluation script from the local Piccione digest, sections time-slice and
per-shot results (doi:10.1088/1741-4326/ac44af).

## Limitations

Negative coverage is assumed, pre-onset physical windows uncertain, and comparisons
unlabelled. The roster is small and campaign-specific; it does not establish stable
coverage, calibration, blind-gold performance or cross-campaign generalisation.
Primary negatives truncate at the last onset; conditional masks retain phase
information. The broad mask changes which scalar ranks best. Fixed-prediction
shot CIs and five
splits do not estimate population-of-run uncertainty. ZIPFIT is acausal,
N1RMS/N2RMS are postprocessed with timing uncertainty,
and the screen and analysis-span selection are retrospective. The validated
low-frequency n=1 RWM-sensitive input is missing despite the isolated candidate
fetch attempt. A stronger study needs an approved sensor locator
validated on Hanson shots and expert review of negative coverage and termination.

The first figure panel, **156785**, has onset beta_N **1.28** and beta_N/l_i
**2.46**, far below the conventional proxy (`E#/onset_physics/rows`).
On figure shots 156796 and 158022, the beta_N/l_i collapses about 400 ms before
the listed onset are of unidentified cause, inside assumed-negative time; a
reason for expert timing/coverage review. The orange beta_N/l_i=4 line uses the
right axis and coincides with score 0.5 on the left; it is not a score threshold.
Sources: `F#/caption`, `F#/source_rows_csv`.

The 2018 run records are RWM-control development with active applied n=1 fields;
probed IUN1BAMP/ILN1BAMP/CN1BAMP look like applied coil-field amplitudes, while
feedback state, I/C-coil currents and NBI torque were not used (run titles in
`rwm_windows.shots.csv`; `sensor_probe.json/probes`).

## Reproduction and paper outputs

Use the stream's frozen/no-install pixi labelmaker runner with this worktree's
`src`, the main manifest, `LABELER_NO_FETCH=1`, and the prescribed TMPDIR. Cached
inputs require no fetching. Build/fit with `rwm_build.py`, `rwm_growth.py`,
`rwm_evaluate.py --workers 5 --replicates 1000`; render with `rwm_tables.py` and
`rwm_figure.py`, all under `scripts/labeler/`.

This round runs `rwm_build.py` for the physical export, then
`rwm_evaluate.py --rescore-saved --workers 5 --replicates 1000` for paired
within-shot intervals. Saved predictions, forecast labels, fitted parameters and
thresholds remain unchanged; there is no refitting, retuning or fetching. Render
with `rwm_tables.py` and `rwm_figure.py`.

Large artifacts live under `$LABELER_ROOT/round4/rwm/`. The figure
`rwm_onset_scores.{pdf,png}` shows six Hanson and two comparison shots selected by
shot number/matching, rather than score. F records caption, source rows and
selection. The figure caption flags 156785 alongside 156796 and 158022; it
illustrates scores, not verified physical growth extent. The main paper table
`table_rwm.tex` uses one aligned tabular for primary, broad and within-shot mean
AUROC, AUPRC, F1, TPR/FPR, and a separate Legacy block. Supplements
(`table_rwm_split_summary.tex`, `table_rwm_within_shot.tex`,
`table_rwm_campaign_pairs.tex`, `table_rwm_alarms.tex`,
`table_rwm_onset_actual.tex`, `table_rwm_onset_window.tex`) hold split/campaign
ranges, pairs, both within-shot masks, alarms and sampled physics.
Small LaTeX sources are committed under `outputs/labeler/rwm/`; compiled PDFs and
150-dpi PNGs live under `$LABELER_ROOT/round4/rwm/`. P records each cell's source,
artifact hashes and compilation/visual checks. Compile with `booktabs` and
`longtable` in a minimal standalone document.

## Appendix: fix history

- **Round 1:** separated confirmed points, conventional weak extent and assumed
  absence; removed comparison-negative fitting and development-biased nnPU
  claims; added phase rules, conditional scores and growth-search controls.
- **Round 2:** tiled explicit unassessed physical time and preserved evidence in
  the reader; added campaign/run-record checks, primary/full alarm scopes,
  per-shot categories, F1 and separately sourced NSTX Legacy results.
- **Round 3:** replaced single-split qualitative headlines with five-split ranges
  and paired elapsed-time results, qualified alarms with their random reference,
  named the holdout by run record, documented matching/span asymmetry and the
  pre-onset convention, and aligned/compiled the paper table. This protocol is the
  current readout; earlier report sections are historical snapshots.

- **Round 4:** added paired campaign/run-record references and five-split alarm
  ranges; described the 2014 conditional failure and seed-3 borderline interval;
  changed unverified pre-onset physical windows to uncertain; added onset physics
  with explicit sampling times, documented offline input timing/rotation units,
  and probed candidate n=1 archive amplitudes without claiming validated sensor
  provenance. The current tables include slice TPR/FPR and labelled paired strata.

- **Round 5:** made the primary-mask time-ranking mechanism explicit; added
  per-shot AUROC and broad paired comparisons; corrected interval descriptions,
  campaign-holdout captions and physics notes; renamed uncertain-window exports
  and regenerated scoped LaTeX sources.

- **Round 6:** shortened the lead/caption; displayed broad and within-shot scores;
  moved ranges to supplements; clarified nonsignificance, 2014 inversion, rule
  multiplicity, unexplained collapses, actuator context, recorded rotation units
  and the two category-2 evidence tiers. Fits and evaluation results are unchanged.

- **Round 7:** added paired shot-bootstrap intervals for within-shot differences
  by saved-prediction replay, qualified the scalar comparisons, added one minimal
  10 ms present slice per onset, renamed task-specific rules and Legacy stable-shot
  keys, and clipped random-alarm warning windows to each scored span. Forecast
  labels and classifier scores remain unchanged.
