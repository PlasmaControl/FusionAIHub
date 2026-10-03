# RWM baseline on DIII-D: current protocol

<!-- rwm:lead:start -->
This retrospective baseline has no established skill beyond discharge phase: `rwm-brf` has phase-controlled AUROC **0.534 [0.438, 0.624]** in **100 ms bins**, with at least **5 eligible slices per bin**. Elapsed time's residual value **0.522 [0.520, 0.549]** is the phase floor. Pooled primary AUROC **0.760 [0.706, 0.809]** is phase-confounded. The task is when an RWM comes in a Hanson shot that has one; physical duration, negative coverage and online input timing remain unverified. Sources: `E#/configs/{rwm-brf,rwm-rule-elapsed-time}/phase_controlled_auroc` and `E#/configs/rwm-brf/metrics/slice_auroc`; E is defined below.
<!-- rwm:lead:end -->

## Records and scope

All numerical results come from committed scripts and their JSON records. Paths
below are relative to the repository; large artifacts live under
`$LABELER_ROOT/round4/rwm/`.

- **E**: [evaluation.json](../../outputs/labeler/rwm/evaluation.json), written by
  `scripts/labeler/rwm_evaluate.py`. The summary retains metrics and counts below
  0.5 MB; `external_details` identifies the complete record and SHA-256.
- **S**: [shots.json](../../outputs/labeler/rwm/shots.json), written by
  `scripts/labeler/rwm_build.py`; roster, matching, labels and input audits.
- **G**: [growth.json](../../outputs/labeler/rwm/growth.json), written by
  `scripts/labeler/rwm_growth.py`; identical onset/control slope searches.
- **C**: [comparison_sensitivity.json](../../outputs/labeler/rwm/comparison_sensitivity.json),
  written by `scripts/labeler/rwm_comparison_sensitivity.py`; label-noisy negatives.
- **A**: [rotation_ablation.json](../../outputs/labeler/rwm/rotation_ablation.json),
  written by `scripts/labeler/rwm_rotation_ablation.py`; input dependence.
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
regenerate from E/C.

## Data and physical evidence

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

The minimal pre/post windows **deviate from the brief's growth-window positives**:
wall time motivates a millisecond convention but does not establish measured
pre-onset growth. Hanson's `ONSET_TIME` is not defined as growth start versus
threshold/detection time; Piccione's NSTX threshold convention cannot resolve its
DIII-D meaning. The source search is preserved in interval metadata and
[sensor_probe.json](../../outputs/labeler/rwm/sensor_probe.json).

**Reader contract:** category 2 mixes two evidence types. Inspect
`attrs.evidence_tier` before analysis; category alone cannot distinguish onset
uncertainty from comparison candidates. Onset-derived intervals additionally
retain `attrs.source_onsets`: contributing `NTOR`, `MODE_TYPE`, original
`ONSET_TIME` in ms and source, including merged duplicate points. The review
reader retains rows/attributes; the category-only editor rejects tier-less saves.
See [editor guide](equilibrium_review.md).

## Inputs and onset physics

The forest uses βN, li, q95, qmin, W_MHD, |Ip|, βN/li, βN−4li, N1RMS/N2RMS
trailing means, peaks and log slopes, and ZIPFIT rotation. Rotation is sampled at
rho=0.25 and rho=0.625; the latter is a fixed radius, not an identified q=2
surface. `TROTFIT` reports kHz in its units field, and `rot_*_khz` retains those
units without conversion. Sources: `E#/configs/rwm-brf/options/columns`,
`S#/feature_coverage`, `src/labeler/rwm/features.py` and the feature namespace.

Trailing/held feature calculations are causal relative to the stored samples;
upstream timing is not established. N1RMS/N2RMS are postprocessed magnetic
amplitudes with uncertain timing. ZIPFIT uses acausal upstream smoothing with
unbounded timing bias here. Analysis-span selection and the screen's
whole-flat-top median/MAD threshold are retrospective. Only βN and the
RMS-amplitude concept overlap Piccione's seven NSTX inputs.

DUSBRADIAL is excluded: **79/100** selected 2014 traces are zero, including
**15/20** Hanson traces, and the namespace flags 176030–176912 as corrupted,
covering the 2018 Hanson shots (`S#/input_audit`). A prior isolated probe of
OPERATIONS CN1BAMP, ILN1BAMP and IUN1BAMP produced finite Gauss traces, but node
names/units do not establish corrected plasma-mode amplitude, filtering or
applied-field subtraction. None was promoted to an input. The 2018 records use
active applied n=1 fields; feedback state, I/C-coil currents and NBI torque were
not model inputs.

PTDATA ONSBRADIAL is disruption-py's reported fallback for unavailable
DUSBRADIAL, but that behavior is not independently verified here. No trace is
cached on any of the 79 relevant zero-DUSBRADIAL 2014 shots, including 15 Hanson
shots. O inspected existing HDF5 names and locator attributes without a fetch;
`LABELER_NO_FETCH=1`. A validated low-frequency n=1 RWM-specific sensor remains
missing. DCON wall limits, E-cross-B frequency, ion collisionality and MHD peak
frequency are also unavailable. N1RMS cannot distinguish RWM, tearing and
applied-field response; RWM frequencies of order 1/τw differ from kHz rotating
tearing modes.

The onset tables list βN, li, βN/li and elapsed time for all **48 merged n=1
points**. Exact-onset values hold the last original EFIT sample for at most
**50 ms**; elapsed time begins at the first |Ip|≥0.5 MA sample. At exact onset,
five points are below βN/li=4 and four have missing EFIT. The separate snapshot
at the first grid slice in [o−20,o) has six below-proxy and three missing points;
it must not be described as the onset value. The distinction matters near beta
collapses and missing-data boundaries (`E#/onset_physics/{rows,by_campaign}`).

G applies identical maximum trailing-log-slope searches at onsets and random
flat-top centres, using 90 offsets from −150 to +28 ms and trailing 20 ms fits;
controls stay at least 170 ms from listed onsets. Median maximum N1RMS slope is
**111.7/s at onsets versus 109.8/s at 7,704 controls**; median slope at onset is
−1.75/s. This does not establish pre-onset growth extent. Separately, **βN falls
at least 20% by 40 ms after 56% of onsets**, relative to the value 100 ms before
onset (27/48; `G#/betan/fraction_dropping_20pct_by_p40`). That collapse supports alignment of
listed points with a plasma event while leaving mode identity and growth-start
timing unresolved. G records the search, controls and per-centre CSV paths.

In labelled Hanson slices, qmin>2 for **83% in 2014** and **96% in 2018**
(2,108/2,540 and 3,074/3,195, including missing values in denominators;
`E#/onset_physics/labelled_slice_qmin`). Most have no q=2 surface. The conventional
βN≈4li no-wall proxy is uncertain in these high-qmin, low-li plasmas; a transfer
of 2018-like structure to 2014 is a hypothesis rather than an established cause
of conditional failure.

<!-- rwm:screen:start -->
The retrospective `rwm_candidates` screen serves a concurrent review role, with 62 post-onset Hanson slice calls and calls on 36/132 unlabelled comparisons, so it is excluded from forecasting tables (`E#/screen_audit` and that screen's `counts`).
<!-- rwm:screen:end -->

## Fitting, masks and phase control

`rwm-brf` reimplements the balanced forest rather than copying an unlicensed
repository: 300 trees, depth 8, minimum leaf size 5, with balanced per-tree
sampling. Training medians impute missing inputs. Five outer and three inner
shot-grouped folds keep all windows of a shot together. Inner out-of-fold Hanson
slices select the ROC cutoff nearest (0 FPR,1 TPR); a separate inner objective
selects alarm high/low thresholds and hold duration. The roster is fixed and
none of the blind cohort trains or tunes anything. Sources: `E#/protocol` and
`src/labeler/rwm/evaluate.py`.

Primary positives are slices with a merged n=1 onset in the next **100 ms**.
Assumed negatives end at the **last n=1 onset**. Aftermath and other-mode
precursors are excluded; n=2-only shots do not supply primary negatives. The
broad sensitivity retains the same scores/positives/exclusions but adds
post-last-onset and n=2-only Hanson time as assumed negative. These forecast
labels are separate from the physical interval categories.

Conditional high-beta scoring means βN≥0.8 times the shot's whole-window p95;
above-proxy means βN/li>4. Both are evaluation masks, not predictors or tuning
thresholds, and retain discharge-phase information. Their slice counts and
prevalences are in E and tables.md. Scalar comparators rank elapsed time, βN
and βN/li in fixed directions. nnPU is excluded because its earlier development
used an outer evaluation fold and the unlabelled prior is unidentified.

Phase-controlled AUROC compares only primary positive-negative pairs in the
same campaign and elapsed-time bin. Each cell needs both classes and at least
five eligible slices; missing elapsed time is omitted. Cells are weighted by
positive-times-negative pair counts, with half credit for ties. The headline
uses **100 ms** bins. Elapsed time's remaining ranking is the **residual-phase
floor**, not evidence of skill beyond phase. Narrower bins reduce but do not
remove differences within bins.

<!-- rwm:phase:start -->
| Model / rule | Phase AUROC (95% CI) | Forest minus rule (95% CI) |
|---|---|---|
| rwm-brf | 0.534 [0.438, 0.624] | — |
| Elapsed time | 0.522 [0.520, 0.549] | 0.012 [-0.069, 0.122] |
| βN | 0.529 [0.417, 0.634] | 0.005 [-0.115, 0.131] |
| βN/li | 0.588 [0.476, 0.691] | -0.054 [-0.169, 0.062] |

| Time-bin width | Forest | Elapsed time | βN | βN/li |
|---|---|---|---|---|
| 50 ms | 0.536 | 0.518 | 0.529 | 0.588 |
| 100 ms | 0.534 | 0.522 | 0.529 | 0.588 |
| 200 ms | 0.544 | 0.582 | 0.541 | 0.595 |
| 400 ms | 0.557 | 0.561 | 0.558 | 0.592 |
| 800 ms | 0.538 | 0.552 | 0.558 | 0.589 |
| No time control | 0.742 | 0.778 | 0.705 | 0.734 |
<!-- rwm:phase:end -->

All forest-minus-scalar intervals include zero. The bin-width table uses saved
reference-split predictions and the same primary mask/minimum cell size. “No
time control” retains campaign control; it differs from the pooled primary
AUROC. These are point sensitivities, not tuned bin choices or independent
experiments (`E#/configs/<model>/phase_bin_width_sensitivity`). Bootstrap phase
intervals resample shots within campaign, preserving campaign shot counts.

Within-shot mean AUROC weights two-class Hanson shots equally. The primary
mask ends with positives, making elapsed time almost perfect within shot; that
point statistic is therefore confined to the repository supplement. Paired
within-shot differences share shot draws. Complete values and class counts are
in `E#/configs/<model>/within_shot_auroc` and `E#/paired`.

## Sensitivities and uncertainty

Individual intervals use **1,000 percentile shot-bootstrap resamples**;
between-model differences use **basic paired intervals** with common draws.
Intervals condition on fixed fitted predictions, are exploratory and unadjusted
for multiple comparisons. Shots with no eligible slices still participate in
the appropriate resampling strata. F1, TPR and FPR apply each fold's inner-chosen
slice cutoff; neither slice AUROC nor a confidence interval validates an alarm
operating point.

Five forest seeds change shot partitions/model randomness at fixed
hyperparameters. Leave-one-run-record-out holds each of four logbook records
out of fitting/tuning, including all sibling shots. `20180314` and `20180314A`
share a calendar date, so this is not calendar-day isolation. Run-holdout
intervals are shot-sampling intervals at fixed predictions, not four-run
population intervals. E/tables.md retain per-record and campaign results.

The reference forest's **2014 high-beta AUROC is 0.311**, with paired difference
against elapsed time **−0.255 [−0.422, −0.067]**. Its conditional failure and
campaign dependence remain material; no cross-campaign generalisation is
established (`E#/configs/rwm-brf/by_campaign/2014/metrics/high_beta_auroc` and
`E#/split_sensitivity/paired_time_by_campaign/2014/0/high_beta_auroc`).

<!-- rwm:sensitivity:start -->
Adding the unverified comparisons as label-noisy training negatives gives phase-controlled AUROC **0.587 [0.505, 0.660]**, a paired change of **+0.053 [-0.028, 0.127]**. It alarms on **3/132** comparison shots and warns **5/48** onsets. The paired interval includes zero; whether verified stable-shot negatives would help is untested. Source: `C#/{phase_controlled_auroc,paired_change,counts}`.
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

## Alarm interpretation

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

Legacy Piccione NSTX results are separately sourced: AUROC **0.918**, TPR
**92.4%**, FPR **21.4%**, **10/11** unstable shots detected and **2/17** stable
shots alarmed. Different machine, expert-reviewed negatives, inputs and
validation make these non-comparable; F1/CIs are unavailable in the local
Piccione 2022 digest (doi:10.1088/1741-4326/ac44af; `E#/legacy`).

## Reproduction and paper outputs

Run from this worktree using the frozen/no-install main-manifest pixi labelmaker
environment, this worktree's `src`, prescribed stream TMPDIR,
`LABELER_ROOT` and main `LABELER_LABEL_TABLES`; retain `LABELER_NO_FETCH=1`.
This fix round uses only cached inputs and saved predictions:

```text
scripts/labeler/rwm_build.py
scripts/labeler/rwm_evaluate.py --rescore-saved --workers 5 --replicates 1000
scripts/labeler/rwm_comparison_sensitivity.py --rescore-saved
scripts/labeler/rwm_rotation_ablation.py --rescore-saved
scripts/labeler/rwm_tables.py
scripts/labeler/rwm_figure.py
```

Replay retains prediction files/fold rules; P records check logs/artifact hashes.

Use **table_rwm.tex and table_rwm_alarms.tex** for the paper. Within-shot means,
split/campaign pairs and exact-onset/window snapshots stay in repository
supplements. Small LaTeX sources live under `outputs/labeler/rwm/`; vector PDFs
and 150-dpi PNGs live under `$LABELER_ROOT/round4/rwm/`.

The score figure selects six Hanson/two comparison examples by shot number and
matching. Its 156785 panel is below the proxy; 156796/158022 have earlier
unidentified beta collapses in assumed-negative time. The orange proxy line uses
the right axis, not a score threshold (F). Sensor timing, onset meaning, stable
coverage and termination need expert review before operational claims.
