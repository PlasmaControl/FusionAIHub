# ELM detection and reference audit

`elm-ours` and ELM-O have **similar point estimates; no significant difference detected**
on the BES subset. This is an interval-occupancy evaluation against reviewed annotations,
not a verified count of individual physical ELMs. The legacy audit separately measures
onset-bin presence, full-coverage interval agreement, and proximity to reviewed span starts.

The elm-ours and DSM detection heads use pooled shot-grouped out-of-fold predictions;
the initialized DSM variant retains a pretrained embedding with source-shot overlap
described below. The survival refit is pretrained; clock/ELM-O are rules. Brackets are
95 % percentile intervals from 1,000 shot-bootstrap resamples, with shared draws for
comparisons. No cohort test shot is used for this stream's detector training or tuning.
Sources are relative to `outputs/labeler/elm/`: **O** = `ours/evaluation.json`,
**D** = `dsm/evaluation.json`, **S** = `swap/evaluation.json`, **P** =
`dsm/prefetch_evaluation.json`, **H** = `training_history.json`. The producing scripts are
committed under `scripts/labeler/`; the source revisions and content hashes are recorded.

## Primary benchmark

The two primary sets use 50 ms bins wholly inside one reviewed absent, non-crowd present
or crowd span and analysed time. `all119` uses fetched-signal coverage: 119
shots/12,409 bins. `bes73` uses ELM-O's analysed chunks:
73 shots/6,843 bins. The latter provides an identical-bin
comparison; ELM-O requires BES. Its AUROC/AUPRC are rank metrics from the saved nested eta
sweep, with unhit bins tied below all hit bins. The clock has only hard calls and seeded
the review, so it is not independent. (O: `sets.{all119,bes73}.{n_shots,bins}`.)

| set | method | auroc | auprc | f1 |
|---|---|---|---|---|
| all119 | `elm-ours` | 0.941 [0.906, 0.967] | 0.877 [0.774, 0.949] | 0.830 [0.778, 0.871] |
| all119 | ELM clock | -- | -- | 0.757 [0.680, 0.827] |
| all119 | Always-present rule | 0.500 [0.500, 0.500] | 0.376 [0.322, 0.430] | degenerate (recall ≥ 0.99) |
| bes73 | `elm-ours` | 0.941 [0.898, 0.972] | 0.875 [0.751, 0.961] | 0.845 [0.786, 0.896] |
| bes73 | ELM-O | 0.918 [0.877, 0.951] | 0.833 [0.753, 0.890] | 0.842 [0.789, 0.885] |
| bes73 | ELM clock | -- | -- | 0.708 [0.602, 0.800] |
| bes73 | Always-present rule | 0.500 [0.500, 0.500] | 0.402 [0.325, 0.479] | degenerate (recall ≥ 0.99) |

Source: O: `sets.<set>.methods.<method>.{point,ci95}`. F1 is marked degenerate when
recall ≥ 0.99; numeric results remain in JSON.

On `bes73`, elm-ours minus ELM-O is F1 +0.002 [-0.052, 0.058], AUROC +0.023
[-0.014, 0.067], AUPRC +0.043 [-0.058, 0.129]. No equivalence test was performed.
Against the clock, paired F1 is +0.137 [0.048, 0.233] on `bes73` and +0.073
[-0.003, 0.151] on `all119`; the latter interval includes zero. (O:
`sets.<set>.paired["elm-ours - elm-elmo: <metric>"]` and
`sets.<set>.paired["elm-ours - elm-clock: f1"]`.)

## Non-crowd present spans and annotation disagreements

`iscrowd=0` identifies **non-crowd present spans**, not verified single ELMs. Their
120 durations range 1–497 ms; median
54.5 ms, interquartile range 42–98.25 ms,
90th percentile 205.3 ms. 30 exceed 100 ms and
14 exceed 200 ms. 108 of 125 possible
non-crowd interior bins come from spans longer than 100 ms. Only
47 spans contribute scored bins; only
2 starts fall inside them. Durations and
per-span IDs are recorded, so the long annotations can be inspected. (O: `label_audit`,
`sets.all119.non_crowd_bin_audit`.)

| set | method | crowd_bin_recall | non_crowd_span_touch_recall | absent_span_alarm_rate |
|---|---|---|---|---|
| all119 | `elm-ours` | 0.851 [0.789, 0.903] | 0.683 [0.379, 0.834] | 0.596 [0.478, 0.689] |
| all119 | ELM clock | 0.680 [0.581, 0.789] | 0.233 [0.122, 0.521] | 0.105 [0.063, 0.168] |
| all119 | Always-present rule | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| bes73 | `elm-ours` | 0.877 [0.805, 0.942] | 0.699 [0.277, 0.871] | 0.610 [0.452, 0.742] |
| bes73 | ELM-O | 0.852 [0.788, 0.901] | 0.634 [0.267, 0.859] | 0.440 [0.301, 0.623] |
| bes73 | ELM clock | 0.628 [0.492, 0.771] | 0.215 [0.090, 0.631] | 0.110 [0.053, 0.188] |
| bes73 | Always-present rule | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |

Source: O: `sets.<set>.methods.<method>.{point,ci95}`. A touch is any overlap with a
detected interval on a span with at least half its length analysed. It is not a
millisecond onset match. The former “68 % single-ELM recall” is 82
of 120 non-crowd present spans touched (0.683), with no
single-ELM inference. On `bes73`, the absent-span alarm rate is 0.610 for elm-ours versus
0.440 for ELM-O (CIs above); elm-ours is more permissive by this measure. Detector-positive
absent spans are **annotation disagreements**. The earlier ELM-O inspection of a small
selected subset does not establish how many are physical ELMs, and no expert audit was
added here. No lower/upper bound on physical detection quality is inferred.
(O: `sets.all119.methods.elm-ours.counts`,
`sets.bes73.methods.{elm-ours,elm-elmo}.point.absent_span_alarm_rate`.)

## Span-start timing, separate from interval occupancy

The onset head was trained on Gaussians at reviewed non-crowd span starts. These starts
are not independently verified ELM onset times; the crowd annotations have no individual
ELM timing. Treat the following as diagnostic **span-start timing agreement**.
Each detection is matched at most once within analysed coverage. Unmatched detections
count as false positives only in reviewed absent/non-crowd time; crowd, uncertain and
unlabelled time is ignored. The onset-head threshold is chosen on inner-validation
shots. This scoring mask explains why only a subset of retained clock points enters
the timing precision denominator. (O: `sets.<set>.onset`;
`labeler.elm.methods.onset_counts`, `labeler.elm.onset.match`.)

The clock comparator now uses genuine peak-picker point times, restricted to the original
saved clock present periods. Its original channel and unchanged picker were verified for
all 119 shots; 24,341 points are retained. Period boundaries
are never substituted for peaks. Current signal hashes are recorded, but historical
input identity cannot be proved without contemporaneous signal hashes. The review remains
dependent on the clock. (O: `clock_onset_source`.)

| all119 method | tolerance | span-start precision | span-start recall | span-start F1 |
|---|---|---|---|---|
| elm-ours | ±5 ms | 0.032 [0.011, 0.061] | 0.075 [0.027, 0.171] | 0.045 [0.016, 0.083] |
| elm-ours | ±10 ms | 0.050 [0.019, 0.089] | 0.117 [0.050, 0.245] | 0.070 [0.030, 0.121] |
| elm-clock | ±5 ms | 0.001 [0.000, 0.002] | 0.025 [0.000, 0.078] | 0.001 [0.000, 0.004] |
| elm-clock | ±10 ms | 0.005 [0.003, 0.013] | 0.183 [0.092, 0.423] | 0.011 [0.005, 0.025] |

Source: O: `sets.all119.onset.<method>.tol_<k>ms.{point,ci95,counts}`. The elm-ours
head has weak agreement with span starts; these numbers do not establish physical onset
accuracy. The clock's formerly reported boundary F1 is superseded.

## DSM refit, limited inputs (60 of the original 124)

The stable identifier `elm-dsm` means **DSM refit, limited inputs (60 of the original
124)** throughout the JSON display names and paper tables. It is the labeler's refit,
not the original full-input checkpoint. Its survival training used Hiro's onsets via
`elm_survival_labels.pkl` and the `wpqh_elm_hiro` survival-row split: it is the
**legacy-trained method**, analogous to the AE audit's RCN/LSTM. The detection and
initialized detection variants retain the same limited inputs/embedding and replace
the survival heads with one logit trained on reviewed interval occupancy.
(D: `display_name`, `model_context`, `method_display_names`; S: `legacy_trained_method`.)

On its own survival target, the refit has AUROC 0.758/0.764/0.770/0.777 at
5/10/20/50 ms on 82 phase records from 80 physical shots/142,745 rows, agreeing with
its recorded fit. The source key `test` was used for early stopping, so this is
**validation evidence**, not untouched held-out evidence. Source IDs are `<shot>_<phase>`;
the previous integer rendering hid physical-shot overlap. Source training has 327 phase
records from 300 shots; 15 physical shots cross its train/validation split. The own-target
bootstrap now groups all phases of each physical shot. (D: `own_target.{phase_records,
shots,rows,horizons,selection_role,split_phase_records,split_physical_shot_counts,
physical_shots_in_both_split_sides,bootstrap_unit}`.)

The prior refit's source training includes reviewed shots 190637, 190643, 192721,
192751 and 196541; source validation includes 190643. It also includes cohort blind-test
shot 190646 in training and 190532 in early-stopping validation. No new detector fit or
threshold selection uses those blind-test shots, but the fixed DSM baseline and initialized
detection variant inherit this pretraining. Their comparisons are supplemental and cannot
certify isolation from the blind cohort. The initialized variant holds out reviewed labels
from detection refitting, not all prior shot exposure. (D:
`own_target.{reviewed_shot_ids_in_published_split,cohort_physical_shot_overlap,overlap_scope}`;
`dsm/own_target_correction.json:correction`.)

Ip/Bt missing on 104 review shots before this fix were fetched into
`$LABELER_ROOT/round4/elm/dsm/fetched_features/`, leaving production stores read-only.
Fetch used one worker with `--pace 1`, and stopped on no auth error. Serving diagnostics
and row fingerprints invalidate stale caches; rescore refuses mismatched saved rows.
The photodiodes are mean-filled **on every shot**; CO2 remains absent on its recorded
subset. (P: `row_diagnostics.missing_features`; D: `row_diagnostics.missing_features`;
`dsm/fetch.json` and `dsm/fetch_part1.json`.)

DSM comparisons additionally require its forecast and detection rows, and are cut to
those rows' analysed coverage. These common sets contain
119 shots/11,653 bins and
73 BES shots/6,527 bins; each method is rescored on the identical
bins. ELM-O's sweep scores are now available on the common BES bins. (D:
`sets.<set>.{n_shots,bins,bins_before_restriction,bins_without_rows}`.)

| set | method | auroc | auprc | f1 |
|---|---|---|---|---|
| all119 common | `elm-ours` | 0.939 [0.901, 0.966] | 0.878 [0.775, 0.951] | 0.834 [0.782, 0.875] |
| all119 common | ELM clock | -- | -- | 0.759 [0.683, 0.829] |
| all119 common | DSM refit, limited inputs (60 of the original 124) | 0.777 [0.730, 0.824] | 0.662 [0.592, 0.736] | 0.627 [0.559, 0.692] |
| all119 common | DSM refit, limited inputs (60 of the original 124), detection | 0.850 [0.802, 0.895] | 0.761 [0.671, 0.834] | 0.743 [0.680, 0.798] |
| all119 common | DSM refit, limited inputs (60 of the original 124), detection init | 0.863 [0.817, 0.905] | 0.792 [0.710, 0.857] | 0.738 [0.675, 0.793] |
| all119 common | Always-present rule | 0.500 [0.500, 0.500] | 0.391 [0.332, 0.450] | degenerate (recall ≥ 0.99) |
| bes73 common | `elm-ours` | 0.939 [0.893, 0.972] | 0.876 [0.752, 0.962] | 0.848 [0.790, 0.899] |
| bes73 common | ELM-O | 0.914 [0.869, 0.948] | 0.833 [0.755, 0.892] | 0.841 [0.786, 0.885] |
| bes73 common | ELM clock | -- | -- | 0.714 [0.610, 0.806] |
| bes73 common | DSM refit, limited inputs (60 of the original 124) | 0.715 [0.633, 0.794] | 0.631 [0.536, 0.730] | 0.591 [0.504, 0.674] |
| bes73 common | DSM refit, limited inputs (60 of the original 124), detection | 0.843 [0.788, 0.892] | 0.750 [0.648, 0.845] | 0.762 [0.681, 0.826] |
| bes73 common | DSM refit, limited inputs (60 of the original 124), detection init | 0.853 [0.796, 0.900] | 0.779 [0.681, 0.860] | 0.749 [0.671, 0.811] |
| bes73 common | Always-present rule | 0.500 [0.500, 0.500] | 0.417 [0.336, 0.500] | degenerate (recall ≥ 0.99) |

Source: D: `sets.<set>.methods.<method>.{point,ci95}`. These are measured comparisons
under the recorded serving conditions; input restriction and architecture differences
cannot be separated causally, and this is not a lower bound for the original model.

The risk scale collapses under transfer: prefetch fold thresholds were
4.10478e-06, 0.000463477, 0.000841796, 0.0016784, 0.0625769; after fetching they are 4.98688e-07, 0.000623552, 8.92347e-07, 0.0021375, 0.0152994.
The share of usable rows outside the training filter was
22.658% before and
22.658% after. Rows are retained with normalized inputs clipped
to ±10 rather than removed, a deviation from the original training filter. Quantiles of
all survival horizons and scored forecast-bin risks are in JSON. After fetching, the
50 ms risk median is 5.25614e-6 across usable rows, and 7.36273e-5 across `all119`
scored forecast bins (95th percentile 0.284456). AUROC remains a ranking
measure even when the probability scale is poorly calibrated; threshold-dependent F1
with recall ≥ 0.99 is marked **degenerate**. (P/D: `published_thresholds[].threshold`,
`row_diagnostics.{usable_rows,outside_training_filter_usable_rows,outside_training_filter_usable_row_share,risk_quantiles_usable_rows}`;
D: `sets.<set>.risk_quantiles_scored_forecast_bins`.)

The committed evaluator was run without `--rescore`, writing `evaluation_trained.json`
and `fits.json` directly; a following rescore reproduced all recorded scientific results
exactly (`dsm/reproducibility.json:exact_results_reproduced`). Prefetch metrics are retained
for comparison. This supersedes the old scratch-created fit provenance.
The later phase-ID correction recomputed only source validation intervals and metadata,
preserving the original fit/rescore snapshots and every reviewed-bin result. Both records
received the same correction; this was not a new fit.
(`dsm/own_target_correction.json:correction`, `dsm/reproducibility.json:verification_scope`.)

## Reference swap: overlap first

Hiro's onset table covers 8 of the reviewed shots:
189885, 190637, 190643, 192721, 192732, 192751, 196541, 200385. The shot-level truth
covers 5 reviewed shots and cannot localize bins; Smith's
independent BES windows cover 0. The onset table's grid alignment is
checked. (S: `overlap`.)

The AE-style coverage audit takes **all legacy-covered 50 ms cells intersecting the
review window**, without DSM or diagnostic coverage restrictions. A review state is
assigned by time occupancy ≥25 ms per cell; present is the union of crowd/non-crowd
annotations. Ties prefer present, then absent. Unknown review states are counted
separately. The legacy's binary rows already represent bins holding original onset
samples and align to this grid. Neither comparison measures independent physical ELM
omissions. (S: `conversion`, `interval_audit.definition`.)

| audit | legacy-covered bins | known review bins | legacy-positive bins | M | P |
|---|---|---|---|---|---|
| All-covered majority | 837 | 782 | 196 | 60 | 14 |
| Interior benchmark + DSM rows | 641 | 641 | 111 | 40 | 4 |

Source: S: `interval_audit.{legacy_covered_bins,legacy_present_bins,known_review_majority}`,
`swap.overlap.finding_1`. M is reviewed interval-majority present / legacy onset-bin
absent; P is legacy-positive / reviewed-majority absent. The restricted scored-bin
comparison is a **deviation from the AE audit**, retained for comparable detector
metrics. Boundary exclusion changes the audit count.

The full audit excludes 53 review-uncertain cells, including
47 legacy-positive cells, plus 2 mixed/unlabelled
cells (legacy-positive 0) and
0 not-observable cells. Every excluded cell is listed by shot/time/state in
`interval_audit.excluded_review_states.<state>.bin_list`; none is treated as absent.

**Shot 192751** is explicitly recorded at `interval_audit.per_shot["192751"]`.
Its full legacy-positive extent is 1750–5100 ms; the shot-level truth says it has ELMs.
The review contains an **uncertain crowd at 2478–5115 ms**, not a verified negative
throughout this period. The full audit has 52 legacy-positive cells on it,
47 in uncertain review time; the remaining
5 fall in majority-absent cells. The restricted table contains
4 such cells. This shot cannot support the earlier claim that the legacy
adds almost no ELMs. (S: that per-shot record,
`overlap.shot_level_ground_truth.values_on_overlap["192751"]`,
`swap.overlap.finding_1.per_shot["192751"]`.)

### Original onsets near reviewed starts

Original positive 1 ms onset samples are read from the two `elm_labels_dict` pickles;
their binned truth is checked against the legacy table. The separate question is: is
there a legacy onset within inclusive ±k ms of **each reviewed span start**? This is a
per-annotation proximity query; a legacy onset can serve more than one start. Crowd
starts are not the crowd's individual ELMs. No independent onset adjudication was added.
(S: `onset_agreement.{definition,source_files,per_shot}`.)

| kind | tolerance | matched starts / reviewed spans | share [95 % CI] |
|---|---|---|---|
| non-crowd present | ±5 ms | 8 / 64 | 0.125 [0.040, 0.322] |
| crowd | ±5 ms | 2 / 3 | 0.667 [0.000, 1.000] |
| non-crowd present | ±10 ms | 13 / 64 | 0.203 [0.040, 0.429] |
| crowd | ±10 ms | 2 / 3 | 0.667 [0.000, 1.000] |
| non-crowd present | ±50 ms | 40 / 64 | 0.625 [0.070, 0.892] |
| crowd | ±50 ms | 3 / 3 | 1.000 [1.000, 1.000] |

Source: S: `onset_agreement.summary.tol_<k>ms.<kind>`. The former “61 % single-ELM
bins missed” claim is dropped. M/P quantify onset-bin versus interval occupancy; the
start table gives a separately defined event-proximity statistic, not verified ELM recall.

### Identical predictions under both references

| set | method | AUROC review | AUROC legacy | F1 review | F1 legacy |
|---|---|---|---|---|---|
| overlap | `elm-ours` | 0.963 [0.832, 0.995] | 0.955 [0.834, 0.988] | 0.720 [0.327, 0.904] | 0.614 [0.219, 0.788] |
| overlap | DSM refit, limited inputs (60 of the original 124) | 0.815 [0.645, 0.923] | 0.846 [0.725, 0.941] | degenerate (recall ≥ 0.99) | degenerate (recall ≥ 0.99) |
| overlap | DSM refit, limited inputs (60 of the original 124), detection | 0.761 [0.644, 0.907] | 0.776 [0.648, 0.941] | 0.367 [0.124, 0.561] | 0.312 [0.079, 0.505] |
| overlap | DSM refit, limited inputs (60 of the original 124), detection init | 0.739 [0.639, 0.892] | 0.744 [0.617, 0.910] | 0.350 [0.115, 0.534] | 0.296 [0.073, 0.482] |
| overlap | ELM clock | -- | -- | 0.430 [0.108, 0.659] | 0.406 [0.131, 0.635] |
| overlap | Always-present rule | 0.500 | 0.500 | degenerate (recall ≥ 0.99) | degenerate (recall ≥ 0.99) |
| overlap_bes | `elm-ours` | 0.965 [0.804, 0.998] | 0.951 [0.813, 0.990] | 0.737 [0.240, 0.933] | 0.654 [0.200, 0.832] |
| overlap_bes | ELM-O | 0.603 [0.375, 0.947] | 0.647 [0.399, 0.971] | 0.474 [0.042, 0.894] | 0.535 [0.081, 0.919] |
| overlap_bes | DSM refit, limited inputs (60 of the original 124) | 0.821 [0.692, 0.938] | 0.834 [0.724, 0.969] | degenerate (recall ≥ 0.99) | degenerate (recall ≥ 0.99) |
| overlap_bes | DSM refit, limited inputs (60 of the original 124), detection | 0.716 [0.564, 0.908] | 0.746 [0.598, 0.947] | 0.340 [0.062, 0.563] | 0.315 [0.056, 0.528] |
| overlap_bes | DSM refit, limited inputs (60 of the original 124), detection init | 0.699 [0.582, 0.887] | 0.715 [0.575, 0.910] | 0.325 [0.061, 0.526] | 0.300 [0.054, 0.502] |
| overlap_bes | ELM clock | -- | -- | 0.388 [0.027, 0.639] | 0.382 [0.069, 0.621] |
| overlap_bes | Always-present rule | 0.500 | 0.500 | degenerate (recall ≥ 0.99) | degenerate (recall ≥ 0.99) |

Source: S: `swap.<set>.{reviewed,legacy}.methods.<method>.{point,ci95}`. The restricted
sets have 8 shots/641 bins and
7 BES shots/526 bins. Every score/call is unchanged
between references; only the reference changes. CIs are shot-bootstrap intervals, not
seed variability, and this overlap offers little precision.

On `overlap`, AUROC order is unchanged: review `elm-ours > elm-dsm > elm-dsm-detect > elm-dsm-detect-init`; legacy `elm-ours > elm-dsm > elm-dsm-detect > elm-dsm-detect-init`. The leading method is `elm-ours` / `elm-ours`. F1 pair reversals are exactly `elm-clock` / `elm-dsm`. The legacy-trained DSM refit AUROC is 0.815 against review and 0.846 against legacy (change +0.031). (S: `swap.overlap.{reviewed,legacy}.ranking`, `swap.overlap.comparison.{order_flips,auroc_changes}`.)

On `overlap_bes`, AUROC order is unchanged: review `elm-ours > elm-dsm > elm-dsm-detect > elm-dsm-detect-init > elm-elmo`; legacy `elm-ours > elm-dsm > elm-dsm-detect > elm-dsm-detect-init > elm-elmo`. The leading method is `elm-ours` / `elm-ours`. F1 pair reversals are exactly `elm-clock` / `elm-dsm`. The legacy-trained DSM refit AUROC is 0.821 against review and 0.834 against legacy (change +0.013). (S: `swap.overlap_bes.{reviewed,legacy}.ranking`, `swap.overlap_bes.comparison.{order_flips,auroc_changes}`.)

The preserved prefetch swap also records the original legacy-trained DSM increase
0.821 → 0.851 on the eight-shot overlap. Its AUROC order and leader stayed unchanged;
weaker F1 methods reordered: on eight shots, clock/DSM survival refit,
clock/DSM detection, and clock/DSM initialized detection; on seven BES shots,
clock/DSM survival refit and DSM detection/ELM-O. Source:
`swap/prefetch_evaluation.json:swap.<set>.comparison.order_flips.f1`. Postfetch
statements above use the current record. These finite-sample results neither establish
invariance to reference choice nor demonstrate the AE audit's leader reversal.

Detector-derived clock/ELM-O proxy swaps remain auxiliary, with the producer excluded;
they are dependent references. Their exact orders/pairs are
S: `proxy.<name>.{reviewed,proxy}.ranking` and `proxy.<name>.comparison.order_flips`.

## Model, training, provenance and limitations

- Inputs use **FS02–FS04**, rather than the brief's FS01–FS04, because this run reused
  the already fetched ELM-O signal store, which contains those three channels. FS01 was
  not fetched or evaluated; it is not claimed unavailable on every reviewed shot.
  The other inputs are the two fast density chords and validity mask, with no BES.
- The event model is the committed 1D U-Net. Saved `cv2` folds/thresholds/checkpoints
  are unchanged by this fix. F1 thresholds use each fold's inner-validation shots.
- Shot grouping prevents the same shot crossing a fold; it does **not** prevent run-day
  correlations. H: `same_day_outer_fold_audit` records the number of days crossing folds and
  actual examples. The two examples in the review happened to share folds, but other
  same-day shots do cross them. No run-day CV or second seed was trained in this fix.
- GroupNorm is trained on 4,096 ms crops and inferred on whole shots; its time-dependent
  normalization statistics can differ. No effect-size or equivalence test was run.
- Earlier runs are disclosed in H: `runs.smoke` and `runs.cv1`: `smoke` completed only
  fold 0 with two epochs/five iterations/batch eight; `cv1` saved 40 fold-0 epochs,
  entered fold 1 without completing it, and has no recoverable full configuration.
  Both contain held-out fold-0 predictions. Artifacts do not establish whether those
  predictions affected the final settings, so the final CV is a development estimate,
  not an untouched confirmatory experiment. Only `cv2` metrics are reported.
- `cv2/run.json` now identifies the producing training code as commit `0d16c19`, with
  explicit retrospective provenance and SHA256 hashes of labels, cohort, prepared
  inputs, fold records, checkpoints and predictions. These hashes describe bytes
  observed during the fix; no contemporaneous training snapshot was saved. New runs
  capture provenance automatically. Partial `--only` updates merge compatible complete
  fold records and refuse incompatible settings/inputs/code.
- Reviewed span starts, clock independence, missing DSM photodiodes/CO2, transferred risk
  calibration, and the small legacy overlap remain limitations. No labels were changed.

Source: O: `train_record`, `config`, `event_thresholds`, `onset_thresholds`; H:
`runs`, `same_day_outer_fold_audit`; `$LABELER_ROOT/round4/elm/cv/cv2/run.json:provenance`.

## Paper artifacts and reproduction

`$LABELER_ROOT/round4/elm/table_elm_benchmark.tex` has primary and common-bin panels for
both all-reviewed and BES subsets; `table_elm_swap.tex` has both overlap panels. Captions
state shot/bin counts, interval construction, thresholds, dependence, DSM limitations
and high-recall F1 degeneracy. `outputs/labeler/elm/tables.json` records source paths and
hashes. Example files are `$LABELER_ROOT/round4/elm/figures/fig_elm_examples.{pdf,png,json}`.
They show rule-picked shots 195111 and 200427 at seven-inch two-column placement, with
restored log10 D-alpha (a.u.) and the 2690–2760 ms non-crowd annotation at the D-alpha
drop explicitly marked. No verified ELM or L-H identification is inferred. The PNG was
visually inspected. The complete caption/windows/axis details are in the figure JSON:
`round4/elm/figures/fig_elm_examples.json:{shots,panels,size_inches,caption}`.

Run from this worktree with the environment required by `implementer-rules.md`:

```bash
python scripts/labeler/elm_run_provenance.py --run cv2 --revision 0d16c19 \
  --evidence "Original run trained the working tree first committed at 0d16c19" \
  --history-json outputs/labeler/elm/training_history.json
# Authorized DIII-D fetch: pixi/fdp wrapper, TMPDIR=scratch/tmp-fetch
python scripts/labeler/elm_dsm_fetch.py --workers 1 --pace 1
python scripts/labeler/elm_dsm_evaluate.py --run cv2
python scripts/labeler/elm_dsm_evaluate.py --run cv2 --rescore
python scripts/labeler/elm_ours_evaluate.py --run cv2
python scripts/labeler/elm_reference_swap.py --run cv2
python scripts/labeler/elm_paper_tables.py
python scripts/labeler/elm_example_figure.py --run cv2
```

The `python` lines are arguments to mandated pixi execution, not bare interpreter
commands. GPU training uses the prescribed CUDA environment; no retraining of elm-ours
was performed. Covering tests and Ruff checks are recorded in the fix-round report.
README `stable` pointers are unchanged.
