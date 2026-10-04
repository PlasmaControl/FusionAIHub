# RWM labels and an equilibrium-scalar timing baseline on DIII-D

<!-- rwm:lead:start -->
No demonstrated skill beyond elapsed time or βN/li under phase control; `rwm-brf` is an equilibrium-scalar timing baseline and no input senses the RWM. Phase-controlled AUROC, seed 0 (the forest's weakest split): forest 0.534 [0.436, 0.625], elapsed time 0.522, βN/li 0.588. Over 5 seeds the forest's margin over elapsed time is +0.012 to +0.107, positive on 5/5 seeds and the holdout, with an interval excluding zero on 1/5. Within shot, forest minus elapsed time, βN and βN/li is −0.147 [−0.200, −0.084], −0.031 [−0.109, 0.045] and −0.071 [−0.133, −0.010]; primary-mask within-shot ranking is dominated by phase (elapsed time 0.931). Run-record holdout, forest minus elapsed time (primary AUROC): 2018 −0.049 [−0.073, −0.013], 2014 +0.019 [−0.052, 0.096], pooled +0.020 [−0.033, 0.080]. Sources: `E#/{split_sensitivity,paired,leave_one_run_record_out}`.
<!-- rwm:lead:end -->

## Abstract

<!-- rwm:abstract:start -->
1. **Task.** From equilibrium scalars on a 10 ms grid, say whether a resistive wall mode (RWM) onset comes within the next 100 ms in a DIII-D discharge that is known to have one.
2. **Shots.** 33 shots from Jeremy Hanson's onset lists (20 in 2014, 13 in 2018; 56 listed onsets merge into 48 n=1 and 6 n=2 events) and 132 matched comparison shots without a listed onset.
3. **Labels.** A 612-row tiered interval table (categories 0 assumed absent, 1 minimal present, 2 uncertain or candidate, 4 unassessed; no verified absence) and forecast labels of 480 positive and 5,255 assumed-negative 10 ms slices.
4. **Result.** No demonstrated skill beyond elapsed time or βN/li under phase control: a balanced random forest after Piccione 2022 (`rwm-brf`) has phase-controlled AUROC 0.534 [0.436, 0.625] against 0.522 for elapsed time and 0.588 for βN/li on seed 0, and within shot, forest minus elapsed time, βN and βN/li is −0.147 [−0.200, −0.084], −0.031 [−0.109, 0.045] and −0.071 [−0.133, −0.010].
5. **Limitation.** No input senses the RWM, so `rwm-brf` is an equilibrium-scalar timing baseline, not an RWM predictor; the next step is a fetch of the radial-field sensor with confirmed semantics.
<!-- rwm:abstract:end -->

## Glossary

<!-- rwm:glossary:start -->
- **Phase-controlled AUROC:** AUROC over positive-negative slice pairs from the same campaign and the same 100 ms elapsed-time bin; a bin needs both classes and at least 5 slices, bins are weighted by pair count, ties count half. It removes the ranking that discharge phase supplies on its own.
- **Residual-phase floor:** the phase-controlled AUROC of elapsed time itself (0.522 [0.511, 0.536]), what time since flat-top still ranks inside one bin. A model must beat it to show more than phase.
- **Primary and broad masks:** primary positives are slices with a merged n=1 onset in the next 100 ms; primary negatives end at the last n=1 onset (aftermath and n=2-only time excluded). The broad mask keeps the same scores and positives and adds post-last-onset and n=2-only Hanson time as negatives.
- **Assumed negatives:** slices of a Hanson shot with no listed onset in the next 100 ms. The onset lists are unverified for completeness, so none is a confirmed absence.
- **Categories 0/1/2/4:** interval labels of the export. 0 assumed absent (before the first onset's 100 ms precursor); 1 minimal present ([o, o+10 ms)); 2 uncertain, either a Hanson pre-onset window ([o−20, o) ms, tier `onset_window_uncertain`) or a comparison-shot screen candidate (tier `unlabelled_screen`); 4 unassessed. None is verified absence.
- **High-β and above-proxy strata:** evaluation masks, βN at least 0.8 times the shot's p95 and βN/li above 4; not predictors.
<!-- rwm:glossary:end -->

## Records and scope

All numerical results come from committed scripts and their JSON records. Paths
below are relative to the repository; large artifacts live under
`$LABELER_ROOT/round4/rwm/`.

- **E**: [evaluation.json](../../outputs/labeler/rwm/evaluation.json), written by
  `scripts/labeler/rwm_evaluate.py`. The summary retains metrics and counts below
  0.6 MB; `external_details` identifies the complete record and SHA-256. Every
  bootstrap interval carries its finite-draw count `n_finite`.
- **S**: [shots.json](../../outputs/labeler/rwm/shots.json), written by
  `scripts/labeler/rwm_build.py`; roster, matching, labels and input audits.
- **G**: [growth.json](../../outputs/labeler/rwm/growth.json), written by
  `scripts/labeler/rwm_growth.py`; identical onset/control slope searches.
- **C**: [comparison_sensitivity.json](../../outputs/labeler/rwm/comparison_sensitivity.json),
  written by `scripts/labeler/rwm_comparison_sensitivity.py`; label-noisy negatives.
- **A**: [rotation_ablation.json](../../outputs/labeler/rwm/rotation_ablation.json),
  written by `scripts/labeler/rwm_rotation_ablation.py`; input dependence.
- **D**: [data_audit.json](../../outputs/labeler/rwm/data_audit.json), written by
  `scripts/labeler/rwm_data_audit.py`; flat-top starts and the saved OPERATIONS
  n=1 probe traces.
- **P**: [presentation.json](../../outputs/labeler/rwm/presentation.json), written
  by `scripts/labeler/rwm_tables.py`; numerical prose, table cells and artifacts.
- **F**: [figure.json](../../outputs/labeler/rwm/figure.json), written by
  `scripts/labeler/rwm_figure.py`; example selection, caption and source rows.
- **O**: [cached_sensor_audit.json](../../outputs/labeler/rwm/cached_sensor_audit.json),
  written by `scripts/labeler/rwm_cached_onsbradial.py`; cache-only sensor search.

Read summaries with `labeler.rwm.records.load_evaluation(path)`; `details=True`
verifies and loads the external payload, including per-shot warnings and source
arrays. C uses the same storage contract. [tables.md](../../outputs/labeler/rwm/tables.md)
contains the full score, split, campaign, run-record, alarm and onset tables.
Reference results use seed 0. Marked numerical blocks and README model lines
regenerate from the JSON records.

## Data and labels

The roster has **33 Hanson shots** (20 in 2014, 13 in 2018) and **132 comparisons**
(80 in 2014, 52 in 2018). The 56 listed onset points merge into 48 n=1 and six
n=2 events; 30 shots have n=1 targets. There is **zero overlap** with the fixed
500-shot train/validation/blind-test cohort. Primary fitting/scoring uses **480
positive / 5,255 assumed-negative 10 ms slices**; 9,184 other Hanson slices are
excluded. Comparisons remain unlabelled and never enter baseline supervised
fitting, cutoff selection or alarm tuning. Sources: `S#/{hanson,comparison,
cohort_overlap,slices}` and `E#/configs/rwm-brf/counts`.

Matching selects four comparisons per Hanson shot without reuse, within campaign,
on flat-top p95 βN and βN/li. The experiment pool is not verified stable;
titles and matching distances are in `rwm_windows.shots.csv` and
`S#/comparison/balance`. In 2014 the matched βN/li p95 mean is
**4.43 versus 5.07** for Hanson (unchosen pool 4.65), potentially biasing incidence
low for a beta-tracking model. Title membership does not establish absence.

Only curated onset points are confirmed evidence. Onset-list completeness,
physical growth duration and termination remain unreviewed. The header-only
fixed-cohort export makes no absence claim.

The physical export is
`data/events/resistive_wall_mode/extend_rwm_onset_window/rwm_windows.csv`, with
metadata and a shot roster beside it. It contains **612 rows on 165 shots**:
54 minimal present, 54 uncertain onset, 33 assumed-absent, 87 Hanson unassessed,
126 comparison-screen and 258 comparison-unassessed spans (`S#/windows`).
Every row has `coverage_verified=false`.

- **Category 1:** [o, o+10 ms), one minimal post-onset present slice, tier
  `onset_point_minimal`. It does not measure duration or termination.
- **Category 2 on Hanson:** [o−20 ms, o), tier `onset_window_uncertain`.
  Pre-onset presence and extent are unverified; present evidence takes precedence
  when windows overlap.
- **Category 0:** high-current time before the first listed onset's 100 ms
  precursor, assuming complete onset listing. It is not verified absence.
- **Category 4:** precursor gaps and time beyond minimal present slices, except
  later uncertain onset windows. Physical state does not expire with the forecast
  horizon; no post-onset recovery is inferred.
- **Category 2 on comparisons:** tier `unlabelled_screen`, clipped to analysis
  coverage. Other comparison time is category 4.

<!-- rwm:absent_collapses:start -->
**Collapses inside category 0.** A βN fall of at least 40% within 100 ms, from βN of at least 2, starts inside the assumed-absent span of 4 of 33 Hanson shots (156796, 158019, 158022, 176092; 5 events, `S#/assumed_absent_collapses`). Neutral-beam power is cached for none of them, so the cause (a beam trip, a mode, a disruption precursor) is not established. Each such span keeps category 0 and carries `attrs.unexplained_beta_collapse`; no span was re-categorised.
<!-- rwm:absent_collapses:end -->

The minimal pre/post windows **deviate from the brief's growth-window positives**:
wall time motivates a millisecond convention but does not establish measured
pre-onset growth. The 20 ms length assumes a wall time τw of about 5 ms, which is
an assumption: the Piccione 2022 digest says only "milliseconds", and no DIII-D
value is cited (\citeph{DIII-D wall time}). The 20 ms uncertain window and the
20 ms growth window both rest on it. Hanson's `ONSET_TIME` is not defined as
growth start versus threshold/detection time; Piccione's NSTX threshold
convention cannot resolve its DIII-D meaning. The source search is preserved in
interval metadata and [sensor_probe.json](../../outputs/labeler/rwm/sensor_probe.json).

**Reader contract:** category 2 mixes two evidence types. Inspect
`attrs.evidence_tier` before analysis; category alone cannot distinguish onset
uncertainty from comparison candidates. Onset-derived intervals additionally
retain `attrs.source_onsets`: contributing `NTOR`, `MODE_TYPE`, original
`ONSET_TIME` in ms and source, including merged duplicate points. The review
reader retains rows and attributes when it loads a tiered source, but its
`save()` refuses tiered sources, so expert review of these labels needs a
tier-preserving save path before it can edit them; the category-only editor
rejects tier-less saves. See [editor guide](equilibrium_review.md).

## What the inputs sense

The forest uses βN, li, q95, qmin, W_MHD, |Ip|, βN/li, βN−4li, the N1RMS
trailing mean, peak and log slope, the N2RMS trailing mean (N2RMS enters no other
way), and ZIPFIT rotation. Rotation is sampled at rho=0.25 and rho=0.625; the
latter is a fixed radius, not an identified q=2 surface. Sources:
`E#/configs/rwm-brf/options/columns`, `S#/feature_coverage`,
`src/labeler/rwm/features.py` and the feature namespace.

<!-- rwm:physics:start -->
No model input senses the RWM, and the evidence that it could is negative:

- `dusbradial` is unusable: zero on 79/100 selected 2014 traces (15/20 Hanson) and flagged corrupted for shots 176030–176912, which covers all 2018 Hanson shots (`S#/input_audit`).
- N1RMS shows no detectably larger maximum slope at onsets: the median maximum log-slope is **111.7/s at 48 n=1 onsets versus 111.7/s at 2,675 controls before the shot's first onset** under an identical search (rank AUC 0.545, Mann-Whitney p 0.28). 65% of all controls fall after the shot's first onset, so the pre-onset controls are the comparison; against all 7,704 controls the median is 109.8/s (rank AUC 0.556, p 0.18; `G#/slope_rank_test_n1`).
- The OPERATIONS n=1 amplitudes (CN1BAMP, ILN1BAMP, IUN1BAMP) are applied-field amplitudes, not a plasma response. Flat-top medians (Gauss, `D#/operations_n1_probe`): 2014 shots 156785 and 158021 have I-coil amplitudes 11.9/11.9 and 13.8/13.7 G and C-coil 0.02 and 0.01 G, varying with the coil programme (5th to 95th percentile of the I-coil amplitude 0.4–17.1 G on 156785 and 4.0–33.6 G on 158021); 2018 shot 176068 has a C-coil plateau of 19.6 G (61% of flat-top samples within 5% of the median) and I-coil 0.05/0.03 G. Their semantics are unverified (`sensor_probe.json`), and none was promoted to an input.

The near-flat 2018 C-coil n=1 plateau points to a pre-programmed or static applied field (for example error-field correction) rather than active feedback; this does not establish the cause, and it is a physical reason to stratify every result by campaign (the campaign tables do). N1RMS cannot distinguish an RWM from a tearing mode or an applied-field response. The forest can therefore only learn the βN/li-and-time trajectory of a Hanson shot, and its labels cannot be checked against any input. **Next step:** fetch the radial-field sensor with confirmed semantics (PTDATA ONSBRADIAL, disruption-py's fallback for DUSBRADIAL, is the lead candidate; none is cached for the zero-trace 2014 shots, `O#`).
<!-- rwm:physics:end -->

<!-- rwm:rotation_units:start -->
The `TROTFIT` units field says kHz, but core magnitudes (median 75, maximum 174) match krad/s, the ZIPFIT convention; kHz would imply supersonic toroidal velocity. Legacy `rot_*_khz` names retain raw values without conversion and do not establish physical units. The forest is invariant to a positive constant unit conversion (`E#/forecast_label_audit/rotation_raw`; namespace metadata).
<!-- rwm:rotation_units:end -->

Trailing/held feature calculations are causal relative to the stored samples;
upstream timing is not established. N1RMS/N2RMS are postprocessed magnetic
amplitudes with uncertain timing. ZIPFIT uses acausal upstream smoothing with
unbounded timing bias here. Analysis-span selection and the screen's
whole-flat-top median/MAD threshold are retrospective. Only βN and the
RMS-amplitude concept overlap Piccione's seven NSTX inputs.

PTDATA ONSBRADIAL is disruption-py's reported fallback for unavailable
DUSBRADIAL, but that behavior is not independently verified here. No trace is
cached on any of the 79 relevant zero-DUSBRADIAL 2014 shots, including 15 Hanson
shots. O inspected existing HDF5 names and locator attributes without a fetch;
`LABELER_NO_FETCH=1`. A validated low-frequency n=1 RWM-specific sensor remains
missing. DCON wall limits, E-cross-B frequency, ion collisionality and MHD peak
frequency are also unavailable. RWM frequencies of order 1/τw differ from kHz
rotating tearing modes.

The onset tables list βN, li, βN/li and elapsed time for all **48 merged n=1
points**. Exact-onset values hold the last original EFIT sample for at most
**50 ms**; elapsed time begins at the first |Ip|≥0.5 MA sample. At exact onset,
five points are below βN/li=4 and four have missing EFIT. The separate snapshot
at the first grid slice in [o−20,o) has six below-proxy and three missing points;
it must not be described as the onset value. The distinction matters near beta
collapses and missing-data boundaries (`E#/onset_physics/{rows,by_campaign}`).

G applies identical maximum trailing-log-slope searches at onsets and random
flat-top centres, using 90 offsets from −150 to +28 ms and trailing 20 ms fits;
controls stay at least 170 ms from listed onsets. Median slope at onset is
−1.75/s. This does not establish pre-onset growth extent. The rank comparison
quoted above uses only controls before each shot's first onset, because most
controls fall after it (`G#/slope_rank_test_n1`). G records the search,
controls and per-centre CSV paths.

<!-- rwm:collapse:start -->
βN falls at least 20% by 40 ms after **26/43 eligible onsets (60.5%)**, relative to the value 100 ms before onset; **5 of 48 n=1 onsets** are unknown. Both reference values require finite bracketing samples separated by at most 50 ms, with no extrapolation. This supports alignment with a plasma event while leaving mode identity and growth-start timing unresolved (`G#/betan`).
<!-- rwm:collapse:end -->

In labelled Hanson slices, qmin>2 for **83% in 2014** and **96% in 2018**
(2,108/2,540 and 3,074/3,195, including missing values in denominators;
`E#/onset_physics/labelled_slice_qmin`). Most have no q=2 surface. The conventional
βN≈4li no-wall proxy is uncertain in these high-qmin, low-li plasmas; a transfer
of 2018-like structure to 2014 is a hypothesis rather than an established cause
of conditional failure.

<!-- rwm:screen:start -->
The retrospective `rwm_candidates` screen serves a concurrent review role, with 62 post-onset Hanson slice calls and calls on 36/132 unlabelled comparisons, so it is excluded from forecasting tables (`E#/screen_audit` and that screen's `counts`). As a rule baseline the `rwm_candidates` screen scores primary AUROC 0.500, warns 0/48 onsets and alarms on 36/132 (27%) unlabelled comparison shots; it makes 0 calls on primary Hanson slices (62 Hanson calls in all, 0 before the first listed onset), so its AUROC is the no-information value, not a measured skill (`E#/configs/rwm-rule-rwm-candidates`, `E#/screen_audit`).
<!-- rwm:screen:end -->

## The baseline: fitting, masks and phase control

`rwm-brf` reimplements the balanced forest rather than copying an unlicensed
repository: 300 trees, depth 8, minimum leaf size 5, with balanced per-tree
sampling. Training medians impute missing inputs. Five outer and three inner
shot-grouped folds keep all windows of a shot together. Inner out-of-fold Hanson
slices select the ROC cutoff nearest (0 FPR,1 TPR); a separate inner objective
selects alarm high/low thresholds and hold duration. The roster is fixed and
none of the blind cohort trains or tunes anything. Sources: `E#/protocol` and
`src/labeler/rwm/evaluate.py`.

<!-- rwm:missingness:start -->
EFIT inputs are missing in 6% of positive slices versus 2% of negatives (at least one missing βN, li, q95, qmin or W_MHD input); median imputation could act as a weak missingness signal (`E#/forecast_label_audit/efit_missing`).
<!-- rwm:missingness:end -->

Primary positives are slices with a merged n=1 onset in the next **100 ms**.
Assumed negatives end at the **last n=1 onset**. Aftermath and other-mode
precursors are excluded; n=2-only shots do not supply primary negatives. The
broad sensitivity retains the same scores/positives/exclusions but adds
post-last-onset and n=2-only Hanson time as assumed negative. These forecast
labels are separate from the physical interval categories.

<!-- rwm:forecast:start -->
Primary forecast labels contain **826/5,255 negatives (15.7%)** after a first n=1 onset and **180/480 positives (37.5%)** preceding repeat onsets. These labels concern the next onset; inter-onset physical state remains category 4 (unassessed) in the interval export. Restricting the same saved reference-split scores to primary slices strictly before the first n=1 onset gives pooled forest AUROC **0.775 [0.722, 0.827]** versus elapsed time **0.773 [0.728, 0.832]**, on 300 positives / 4,429 assumed negatives across 30 shots; no refit or tuning. Sources: `E#/forecast_label_audit` and `E#/configs/{rwm-brf,rwm-rule-elapsed-time}/first_onset_only`.
<!-- rwm:forecast:end -->

Conditional high-beta scoring means βN≥0.8 times the shot's whole-window p95;
above-proxy means βN/li>4. Both are evaluation masks, not predictors or tuning
thresholds, and retain discharge-phase information. Their slice counts and
prevalences are in E and tables.md. Scalar comparators rank elapsed time, βN
and βN/li in fixed directions. nnPU is excluded because its earlier development
used an outer evaluation fold and the unlabelled prior is unidentified.

Phase-controlled AUROC compares only primary positive-negative pairs in the
same campaign and elapsed-time bin. Each bin needs both classes and at least
five eligible slices; slices with no elapsed time are omitted (counted in
Appendix C). Bins are weighted by positive-times-negative pair counts, with half
credit for ties. The headline uses **100 ms** bins. Elapsed time's remaining
ranking is the **residual-phase floor**, not evidence of skill beyond phase.
Narrower bins reduce but do not remove differences within bins.

## Results

<!-- rwm:phase:start -->
Pooled primary AUROC on the reference split is 0.760 [0.706, 0.809] for the forest and 0.759 [0.712, 0.815] for elapsed time; both are phase-confounded, so neither is a skill measure. Phase control uses 100 ms bins with at least 5 slices per bin. The 100 ms bin width was chosen after an earlier run with 200 ms bins, where seed 0's forest-minus-elapsed-time margin was −0.037; it is a post-hoc choice.

| Model / rule | Phase AUROC (95% CI) | Forest minus rule (95% CI) |
|---|---|---|
| rwm-brf | 0.534 [0.436, 0.625] | — |
| Elapsed time | 0.522 [0.511, 0.536] | 0.012 [-0.081, 0.113] |
| βN | 0.529 [0.410, 0.638] | 0.005 [-0.123, 0.133] |
| βN/li | 0.588 [0.471, 0.690] | -0.054 [-0.172, 0.061] |

| Evaluation | Forest phase AUROC | Forest − elapsed time | Forest − βN/li |
|---|---|---|---|
| Seed 0 | 0.534 [0.436, 0.625] | 0.012 [-0.081, 0.113] | -0.054 [-0.172, 0.061] |
| Seed 1 | 0.592 [0.510, 0.680] | 0.070 [-0.017, 0.160] | 0.004 [-0.106, 0.111] |
| Seed 2 | 0.580 [0.497, 0.666] | 0.058 [-0.029, 0.143] | -0.008 [-0.133, 0.104] |
| Seed 3 | 0.629 [0.549, 0.706] | 0.107 [0.030, 0.189] | 0.041 [-0.071, 0.151] |
| Seed 4 | 0.598 [0.506, 0.691] | 0.076 [-0.017, 0.171] | 0.010 [-0.112, 0.128] |
| Run-record holdout | 0.593 [0.517, 0.679] | 0.071 [-0.013, 0.149] | 0.005 [-0.119, 0.102] |

| Time-bin width | Forest | Elapsed time | βN | βN/li |
|---|---|---|---|---|
| 50 ms | 0.536 | 0.518 | 0.529 | 0.588 |
| 100 ms | 0.534 | 0.522 | 0.529 | 0.588 |
| 200 ms | 0.544 | 0.582 | 0.541 | 0.595 |
| 400 ms | 0.557 | 0.561 | 0.558 | 0.592 |
| 800 ms | 0.538 | 0.552 | 0.558 | 0.589 |
| No time control | 0.742 | 0.778 | 0.705 | 0.734 |
<!-- rwm:phase:end -->

All reference-split forest-minus-scalar phase intervals include zero. The split
table shows the split-dependent elapsed-time margin and the four-run holdout. The
bin-width table uses saved reference-split predictions and the same primary
mask/minimum cell size; the 100 ms default was chosen after an earlier run
with 200 ms bins in which seed 0's forest is below elapsed time (the
200 ms row: 0.544 against 0.582 as rounded in the table, −0.0375 unrounded), so it
is a post-hoc choice and the width sensitivity is a point
check, not an independent experiment. “No time control” retains campaign
control; it differs from the pooled primary AUROC
(`E#/configs/<model>/phase_bin_width_sensitivity`). Bootstrap phase intervals
resample shots within campaign, preserving campaign shot counts. Pairs are
formed within one copy of a shot or between different shots; pairs between
copies of the same shot are excluded.

Within-shot mean AUROC weights two-class Hanson shots equally. The primary mask
ends with positives, making elapsed time almost perfect within shot; that point
statistic is therefore confined to the repository supplement. Paired within-shot
differences share shot draws; the lead reports the primary-mask differences
against all three rules. Complete values and class counts are in
`E#/configs/<model>/within_shot_auroc` and `E#/paired`.

The reference forest's **2014 high-beta AUROC is 0.311**, with paired difference
against elapsed time **−0.255 [−0.422, −0.067]**. Its conditional failure and
campaign dependence remain material; no cross-campaign generalisation is
established (`E#/configs/rwm-brf/by_campaign/2014/metrics/high_beta_auroc` and
`E#/split_sensitivity/paired_time_by_campaign/2014/0/high_beta_auroc`).

Individual intervals use **1,000 percentile shot-bootstrap resamples**;
between-model differences use **basic paired intervals** with common draws.
Intervals condition on fixed fitted predictions, are exploratory and unadjusted
for multiple comparisons. F1, TPR and FPR apply each fold's inner-chosen slice
cutoff; neither slice AUROC nor a confidence interval validates an alarm
operating point.

## Appendix A: sensitivities and uncertainty

Five forest seeds change shot partitions/model randomness at fixed
hyperparameters. Leave-one-run-record-out holds each of four logbook records
out of fitting/tuning, including all sibling shots. `20180314` and `20180314A`
share a calendar date, so this is not calendar-day isolation. Run-holdout
intervals are shot-sampling intervals at fixed predictions, not four-run
population intervals. E and tables.md retain per-record and campaign results.

<!-- rwm:sensitivity:start -->
Adding the unverified comparisons as label-noisy training negatives gives phase-controlled AUROC **0.587 [0.506, 0.665]**, a paired change of **+0.053 [-0.030, 0.127]**. It alarms on **3/132** comparison shots and warns **5/48** onsets. The paired interval includes zero; whether verified stable-shot negatives would help is untested. Source: `C#/{phase_controlled_auroc,paired_change,counts}`.
<!-- rwm:sensitivity:end -->

C is one reference-split CV with identical outer shots, inner splits and seeds;
only training includes comparison negatives. Cutoff/alarm tuning retains the
original Hanson scope. It does not replace the baseline or establish stable
controls. This measured sensitivity replaces any causal claim that the
Hanson-only roster explains the phase ranking.

A excludes both rotation columns using the same folds/seeds. Primary AUROC is
**0.752**, change **−0.008 [−0.021,+0.005]**; broad AUROC is **0.754**, change
**+0.014 [−0.001,+0.028]** (no rotation minus baseline;
`A#/{metrics,paired_change}`). Input dependence does not bound upstream timing
bias, and no isolated rotation benefit is claimed.

## Appendix B: alarm interpretation

Hysteresis needs a high crossing, continued low-threshold support and the chosen
hold duration. Gaps reset the alarm. Matching accepts warnings **10–400 ms**
before merged n=1 targets; n=1/n=2 onsets explain alarms separately. Hanson
alarm traces stop at the last explanation onset+100 ms, while comparisons keep
their full span. This truncation is a retrospective scoring convention, not
physical termination evidence (`E#/protocol/primary_alarm_scope`).

<!-- rwm:alarms:start -->
The reference split warns **9/48** onsets; median warning is **356 [286, 389] ms**. The 9 matched warnings have range **32–391 ms** within the **10–400 ms** matching window. The median is close to its upper limit, consistent with early phase-driven alarms; it does not establish onset-specific anticipation. Comparison shots alarmed: **19/132**, **0.144 [0.083, 0.212]**. Hanson shots with unexplained alarms: **8/33**, **0.242 [0.091, 0.394]**. These are unlabelled-shot incidences, not verified false-positive rates. Sources: `E#/configs/rwm-brf/{counts,metrics,per_shot}`; distribution calculation: `P#/documentation/warning_distribution`.
<!-- rwm:alarms:end -->

The first considered alarm assigns Detected/Early/Missed: a matched n=1 warning
is Detected, an unexplained alarm more than 400 ms before a future target is
Early, otherwise Missed. The reference forest gives **2/6/22** across 30 target
shots; any-warning precedence gives **6/2/22** as a labelled sensitivity.
Per-onset counts retain all alarms. Late/other-mode explained alarms cannot
silently change the primary category. Sources: `E#/configs/rwm-brf/counts` and
`E#/protocol/shot_categories`.

The uniform-alarm reference approximately places each target shot's observed
alarm count randomly across its scored span, clipping each matching window to
coverage. It is an approximate rate-matched reference, not a fitted predictor.
No detection improvement over that reference is established across five seeds.
Full-trace tuning is a separately labelled sensitivity on the same fitted slice
scores; it warns only **1/48** onsets, with **1/132** comparisons alarmed
(`E#/configs/rwm-brf/full_trace_alarm_sensitivity`).

Scalar/forest alarm comparisons in tables.md mostly reflect tuning monotone
rules on truncated Hanson traces; they do not establish forest forecasting
skill. The rules are evaluated on the reference split only. Comparison exposure
and matching imbalance further limit comparisons of unlabelled-shot incidence.
Operational forecasting and verified stable-shot false-positive performance
remain unvalidated.

Published NSTX results of Piccione et al. (2022, doi:10.1088/1741-4326/ac44af)
are quoted here only as text: AUROC **0.918**, TPR **92.4%**, FPR **21.4%**,
**10/11** unstable shots detected and **2/17** stable shots alarmed. That is a
different machine with expert-reviewed negatives, different inputs and a
different validation, so it is not comparable to any number above; F1 and
intervals are unavailable in the local digest (`E#/legacy`). No table sets it
beside the DIII-D rows.

## Appendix C: deviations and audits

<!-- rwm:audit:start -->
- **Phase-bootstrap eligibility** is decided on resampled slices: a duplicated shot can lift a bin to the minimum cell size, so eligibility is not decided on unique slices (documented, not changed). On the reference split 43 of 43 two-class bins reach the minimum, holding 466/480 positives and 3,129/5,185 negatives (the rest sit in bins with only one class, which are not scored); 35,059 pairs are scored. Replaying the 1,000 campaign-stratified draws, 14 resamples score a bin that is eligible only through duplicates: on average 0.014 bins, at most 1, carrying at most 0.11% of that resample's pairs (`E#/phase_eligibility`).
- **Primary slices without elapsed time:** 70 assumed negatives and 0 positives on 4 shots precede the first |Ip| ≥ 0.5 MA sample, so no phase bin contains them (`E#/forecast_label_audit/primary_without_elapsed_time`).
- **Flat-top start of comparison shot 157975** is −212 ms, the only negative start among 165 roster shots (next smallest 84 ms, median 189 ms). Its saved Ip is a smooth rise from 0.001 MA at −300 ms to 0.46 MA at −200 ms and its peak is only 0.81 MA, so the 50%-of-peak crossing (0.40 MA) falls before t = 0; the first |Ip| ≥ 0.5 MA sample is at −189 ms. This is consistent with a fast ramp on a low-current shot rather than a corrupted time base, but a current this high this early is unusual for DIII-D timing, so a time-base offset is not excluded; the shot is flagged, not corrected. Comparison shots enter only alarm incidence, the interval export and the comparison-negative sensitivity's training (`D#/outliers`).
- **Finite bootstrap draws:** every interval records its finite draws (`n_finite`) and a build fails below 90% of 1,000. Over 1,865 intervals in E, C and A the smallest finite share is 99.7%; 3 intervals lose draws. Warning means and medians exist only in resamples that warn an onset and are exempt (96 intervals), as are 36 intervals whose point estimate is itself undefined.
<!-- rwm:audit:end -->

Deviations from the brief, with reasons:

- **Fetch route.** `scripts/labeler/rwm_fetch.py` resolves features through
  `labeler.features.resolve_fdp` instead of the `labeler.events.raw` `FETCH_SPECS`
  the brief named. `FETCH_SPECS` covers only co2, ece, ip, pcphd03, filterscopes
  and pinj, while the RWM inputs (n=1/n=2 RMS, βN, li, q95, qmin, W_MHD, ZIPFIT
  rotation, locked mode) are namespace features that `resolve_fdp` already
  resolves; it writes the same per-shot raw cache layout (`raw.cache_path`), so
  nothing downstream differs.
- **Minimal windows.** The [o−20, o) and [o, o+10 ms) windows replace the brief's
  growth-window positives (see Data and labels); τw ≈ 5 ms is an assumption.
- **Evaluation summary size.** The summary cap rose from 0.5 MB to 0.6 MB when
  each bootstrap interval began to carry its finite-draw count.
- **Audit script.** `scripts/labeler/rwm_data_audit.py` is a stand-alone audit;
  it reads only saved arrays and the roster.

## Future work

Phase-matched negative sampling, which draws training negatives at the same
elapsed time and campaign as each positive instead of ending every labelled
span in a positive, would test skill beyond phase more directly than the
comparison-negative sensitivity, and is left as future work.

## Reproduction and paper outputs

Run from this worktree using the frozen/no-install main-manifest pixi labelmaker
environment, this worktree's `src`, prescribed stream TMPDIR,
`LABELER_ROOT` and main `LABELER_LABEL_TABLES`; retain `LABELER_NO_FETCH=1`.
The tables, figure and audits use only cached inputs and saved predictions:

```text
scripts/labeler/rwm_build.py
scripts/labeler/rwm_evaluate.py --rescore-saved --workers 5 --replicates 1000
scripts/labeler/rwm_comparison_sensitivity.py --rescore-saved
scripts/labeler/rwm_rotation_ablation.py --rescore-saved
scripts/labeler/rwm_growth.py
scripts/labeler/rwm_data_audit.py
scripts/labeler/rwm_figure.py
scripts/labeler/rwm_tables.py
```

Replay retains prediction files/fold rules; P records check logs/artifact hashes.

Use **table_rwm_compact.tex** for the paper: one column-width row group (forest,
elapsed time, βN/li). Everything else is appendix or repository supplement:
table_rwm.tex (full scores), table_rwm_alarms.tex (alarms), the split, campaign,
within-shot and onset tables. Small LaTeX sources live under
`outputs/labeler/rwm/`; vector PDFs and 150-dpi PNGs live under
`$LABELER_ROOT/round4/rwm/`.

The score figure selects six Hanson/two comparison examples by shot number and
matching. Its 156785 panel is below the proxy; 156796 and 158022 show βN/li
collapses inside assumed-absent time (see *Collapses inside category 0*; the
caption states the detector). The orange proxy line uses
the right axis, not a score threshold (F). Sensor timing, onset meaning, stable
coverage and termination need expert review before operational claims.
