# RWM baseline on DIII-D: current protocol

This retrospective reference forecasts Jeremy Hanson's listed n=1 onsets using a
Piccione-inspired balanced forest (`rwm-brf`) and fixed scalar rules. It conditions
on assumed negative coverage and conventional forecast windows in 33 Hanson shots;
it does not measure physical instability duration or establish a validated detector.

Across five shot-fold splits, pooled AUROC is **0.760–0.806** and **high-beta
conditional AUROC is 0.602–0.696 (about 0.60–0.70)**. High-beta AUROC is about chance
in 2014 (**0.311–0.527**) and **0.698–0.747 in 2018**. The forest is not
distinguishable from the elapsed-time rule in AUROC in any of the three pooled
strata: forest-minus-time point differences range **0.001–0.047** for primary
slices, **−0.040 to +0.054** for high-beta, and **−0.051 to +0.061** for above-proxy;
every seed's paired 95% interval includes zero. These conditional strata retain
discharge-phase information. Sources: `E#/split_sensitivity/{auroc_ranges,
paired_time_ranges,paired_time_by_seed}`.

Primary split-0 detection is **0.188**, indistinguishable from the rate-matched
random-alarm reference **0.190**: difference **−0.002 [−0.058, 0.052]**. The median
warning is **356 ms [286, 389]**, near the 400 ms acceptance limit and consistent
with phase tracking. This does not establish useful onset-specific timing skill.
Source: `E#/configs/rwm-brf/metrics`.

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
reference split; five-split ranges are the headline. A range of point estimates
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
`data/events/resistive_wall_mode/extend_rwm_growth/rwm_windows.shots.csv`.

Inputs are beta_N, l_i, q95, qmin, W_MHD, |Ip|, beta_N/l_i, beta_N−4l_i, N1RMS,
N2RMS and ZIPFIT toroidal rotation at the configured radii. RMS features use trailing
means, peaks and log slopes. Resampling holds the last available sample, but
ZIPFIT's upstream time smoothing is mildly acausal. Analysis-span selection and the
candidate screen's whole-flat-top threshold are also retrospective. This is an
adaptation of Piccione's NSTX inputs, not a reproduction of them.
Sources: `E#/configs/rwm-brf/options/columns`, `S#/feature_coverage`,
`src/labeler/rwm/features.py` and `src/labeler/features/namespace.py`.

DUSBRADIAL is excluded: it is zero in 79/100 selected 2014 traces, including
15/20 Hanson traces, and the namespace flags shots 176030–176912 as corrupted,
covering all 2018 Hanson shots. N1RMS is rotating-mode RMS and cannot distinguish
RWM from tearing or applied-field response. No validated low-frequency n=1
saddle-loop/Bp amplitude locator exists in the approved disk FETCH_SPECS or feature
namespace. No node is guessed or fetched. No DCON wall limits, E-cross-B frequency,
ion collisionality or MHD peak-frequency reconstruction is available here.
Source: `S#/input_audit`, including `low_frequency_n1_rwm_sensor_specs` and
`low_frequency_sensor_status`.

## Labels: physical catalog and future-onset target

Only curated onset **points** are confirmed evidence. Onset listing does not
establish reviewed negative coverage, growth duration or mode termination. Original
database events retain unknown confidence/coverage; the original 500-shot screen
export is header-only, not an absence claim.

The physical interval CSV is
`data/events/resistive_wall_mode/extend_rwm_growth/rwm_windows.csv`, with JSON
`attrs.evidence_tier` and `coverage_verified=false`, plus `rwm_windows.meta.json`:

- **Category 1, conventional weak:** the nominal [o−20 ms, o] pre-onset window for
  each listed n=1 or n=2 onset. CSV/readers represent intervals as half-open. The
  20 ms extent is a prescribed pre-onset convention motivated by the millisecond
  wall time tau_w, not a measured growth time. We do not extend a point label into
  an assumed post-onset duration: the few milliseconds immediately after onset
  are **category 4**, pending extent/termination evidence.
- **Category 0, assumed absent:** high-current Hanson time before the **first**
  listed onset's 100 ms precursor, conditional on onset-list completeness. There
  is no post-onset reset to absence and no verified absence interval.
- **Category 4, unassessed:** precursor gaps and all physical post-onset time,
  except later conventional weak windows. The physical state does not expire
  after 100 ms. Comparison time outside uncertain screen spans is category 4 too.
- **Comparison screen spans:** unlabelled candidates for review, never primary
  negatives. The screen requires N1RMS above the whole-flat-top median +6 MAD,
  beta_N >4l_i and at least 10 ms duration. Missing inputs are not observable.

The export has **558 rows on 165 shots**: 54 conventional weak, 33 assumed-absent,
87 Hanson unassessed, 126 comparison screen and 258 comparison unassessed spans.
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
physical growth extent, so the weak-window duration rests on convention.
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
conditional mask. Paired differences share shot draws and use **basic** bootstrap
CIs (reflected percentile endpoints). All intervals condition on fixed fitted OOF
predictions; they do not include refitting, fold selection or run-population
uncertainty. The random-alarm reference keeps each target shot's alarm count and
places alarms independently/uniformly over its scored span; it is approximate.
Warning-time CIs condition on detected onsets. Sources: `E#/protocol`,
`E#/paired_method`, `src/labeler/rwm/metrics.py` and `evaluate.py/chance_detection`.

## Results

Five-split AUROC point ranges (`E#/split_sensitivity/auroc_ranges`):

| Scope | Primary | High-beta conditional | Above-proxy conditional |
|---|---|---|---|
| Pooled | 0.760–0.806 | 0.602–0.696 | 0.546–0.657 |
| 2014 | 0.663–0.735 | 0.311–0.527 | 0.277–0.508 |
| 2018 | 0.819–0.848 | 0.698–0.747 | 0.647–0.706 |

The forest is not distinguishable from elapsed time in AUROC in any of these
three **pooled** strata across the five seeds. Paired forest-minus-time point
ranges are 0.001–0.047, −0.040 to +0.054 and −0.051 to +0.061 respectively; all
paired 95% CIs include zero. For high-beta split 0 the difference is
**−0.040 [−0.120, 0.061]**. These results do not show an AUROC advantage over phase
tracking. Per-seed paired intervals are in `E#/split_sensitivity/paired_time_by_seed`
and the generated tables; the point ranges are in `paired_time_ranges`.

Leave-one-run-record-out AUROC (95% fixed-prediction shot CI;
`E#/leave_one_run_record_out/{metrics,by_campaign/<year>/metrics}`):

| Scope | Primary | High-beta conditional | Above-proxy conditional |
|---|---|---|---|
| Pooled | 0.779 [0.731, 0.823] | 0.621 [0.536, 0.700] | 0.579 [0.475, 0.677] |
| 2014 | 0.729 [0.672, 0.814] | 0.441 [0.369, 0.582] | 0.404 [0.329, 0.536] |
| 2018 | 0.796 [0.744, 0.845] | 0.661 [0.576, 0.746] | 0.633 [0.545, 0.729] |

Its AUPRC is **0.181 [0.149, 0.248]**, F1 **0.285 [0.233, 0.344]**, and alarms warn
**8/48** onsets. The pooled high-beta shot CI is above chance; the 2014 conditional
intervals include chance. Four fixed records cannot establish population-of-run
performance. Source: `E#/leave_one_run_record_out/{metrics,counts}`.

Reference split-0 scores (95% shot CIs; `E#/configs/<model>/metrics`):

| Model / rule | Primary AUROC | Primary AUPRC | Primary F1 |
|---|---|---|---|
| rwm-brf | 0.760 [0.706, 0.809] | 0.163 [0.123, 0.220] | 0.275 [0.227, 0.337] |
| Elapsed time | 0.759 [0.712, 0.815] | 0.228 [0.212, 0.307] | 0.267 [0.217, 0.331] |
| beta_N | 0.707 [0.639, 0.772] | 0.166 [0.128, 0.246] | 0.246 [0.194, 0.312] |
| beta_N/l_i | 0.723 [0.664, 0.784] | 0.159 [0.127, 0.232] | 0.261 [0.212, 0.328] |
| Candidate screen | 0.500 [0.500, 0.500] | 0.084 [0.070, 0.102] | 0.000 [0.000, 0.000] |

Elapsed time's higher point AUPRC does not establish a difference: forest-minus-
time AUPRC is **−0.065 [−0.100, 0.016]** under the basic paired bootstrap. Its top
60 scores are concentrated in 176078 (48 slices) and 176069 (12). Split-0
broader-negative sensitivity is **AUROC 0.740 [0.668, 0.799] / AUPRC 0.065
[0.043, 0.103]**, from identical predictions. Sources: `E#/paired/rwm-brf -
rule-time-since-flattop/slice_auprc`, `E#/configs/rule-time-since-flattop/
top_score_concentration`, `E#/configs/rwm-brf/metrics/broad_{auroc,auprc}`.

Primary split-0 alarms warn **9/48** onsets: **6 Detected / 22 Missed / 2 Early**
among 30 n=1 target shots, plus 3 No target. Detection **0.188 [0.049, 0.333]**
equals the rate-matched random-alarm reference **0.190 [0.080, 0.302]** within
uncertainty: difference **−0.002 [−0.058, 0.052]**. The forest is no better than
randomly placed alarms at the same rate under this reference. Median warning
**356 ms [286, 389]** lies near the 400 ms limit, consistent with phase tracking.
Sources: `E#/configs/rwm-brf/{counts,metrics}`.

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

Negative coverage is assumed, physical weak extent conventional, and comparisons
unlabelled. The roster is small and campaign-specific; it does not establish stable
coverage, calibration, blind-gold performance or cross-campaign generalisation.
Conditional masks retain phase information. Fixed-prediction shot CIs and five
splits do not estimate population-of-run uncertainty. ZIPFIT is acausal, the screen
and analysis-span selection retrospective, and the validated low-frequency n=1
RWM-sensitive input is missing. A stronger study needs an approved sensor locator
validated on Hanson shots and expert review of negative coverage and termination.

On figure shots 156796 and 158022, beta_N/l_i collapses about 400 ms before the
listed onset, inside assumed-negative forecast time; this strains onset-list
completeness. The N1RMS maximum-slope control search does not repair that evidence
gap. Sources: `F#/caption`, `F#/source_rows_csv`, `G#/controls`.

## Reproduction and paper outputs

Use the stream's frozen/no-install pixi labelmaker runner with this worktree's
`src`, the main manifest, `LABELER_NO_FETCH=1`, and the prescribed TMPDIR. Cached
inputs require no fetching. Build/fit with `rwm_build.py`, `rwm_growth.py`,
`rwm_evaluate.py --workers 5 --replicates 1000`; render with `rwm_tables.py` and
`rwm_figure.py`, all under `scripts/labeler/`.

To reproduce this round's summaries without refitting/tuning, run
`rwm_evaluate.py --rescore-saved --workers 5 --replicates 1000`, then
`rwm_tables.py`. The evaluation script replays saved fold alarm rules, bootstraps
every seed and campaign, pairs each forest split with the identical fixed
elapsed-time ranks, and writes E. All preceding point scores, split-0 intervals,
alarm outcomes and run-record scores are preserved. Sources:
`E#/protocol/prediction_mode`, `E#/configs/<model>/predictions`,
`E#/leave_one_run_record_out/predictions`.

Large artifacts live under `$LABELER_ROOT/round4/rwm/`. The figure
`rwm_onset_scores.{pdf,png}` shows six Hanson and two comparison shots selected by
shot number/matching, rather than score. F records caption, source rows and
selection. The figure remains unchanged this round. The paper table
`table_rwm.tex` has one-line point/CI cells, labelled Tokamak-SI and Legacy blocks,
five-split campaign ranges and run-record holdout cells. It uses a two-column-wide
`table*` to preserve readable intervals; P records every cell's JSON source and
its short caption. Compile it with `booktabs` in a minimal standalone document.

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
