# ELM occupancy evaluation

`elm-ours` delivers BES-free ELMy-occupancy probability; the requested finer
physical-onset trace is **not delivered to the catalog**. On all119 (12,409 interior 50
ms bins) its AUROC is 0.941 [0.906, 0.967], AUPRC 0.877 [0.774, 0.949] and F1 0.830
[0.778, 0.871]; on bes73 (6,843 bins) AUROC 0.941 [0.898, 0.972] and F1 0.845 [0.786,
0.896]. DSM common-bin results are secondary controls; native detection is scored on a
smaller identical-support panel. Smith onset recall describes selected windows only, and
the legacy swap measures reference-definition disagreement; neither confirms
continuous-discharge physical-onset performance.

## Glossary

- **all119**: the 119 reviewed shots, scored on 12,409 interior 50 ms bins that lie
  wholly inside one reviewed absent, non-crowd or crowd span and inside signal coverage.
- **bes73**: the 73 of them with BES (elm-elmo's coverage), 6,843 bins. It is a scope,
  not a model.
- **common bins**: primary bins that also have valid DSM rows (11,653 / 6,527); the
  secondary control.
- **crowd / non-crowd / absent**: reviewed ELMing-period spans, other present spans, and
  spans without ELMs.
- **known-majority cells**: legacy-swap 50 ms cells with at least 25 ms of legacy
  coverage that at least 25 ms of one reviewed state, present or absent, labels (ties go
  to present); mixed, uncertain and not-observable cells stay out and unknown time stays
  unknown.
- **model names**: `elm-elmo` is our reimplementation of ELM-O; `elm-dsm-survival` is
  the DSM survival refit on source 1 ms rows; `elm-dsm-detect` is the 60-input 1×128
  occupancy-detection adaptation on 50 ms means; `elm-dsm-native-detect` is the
  124-input [100,1000] detection refit on 1 ms means.
- **exact exports**: the source DSM's saved native rows.
- **source stats / source weights**: DSM variants that reuse upstream normalization or
  weights (marked ‡); that normalization includes two blind-cohort shots.
- **†**: recall of at least 0.99.

## Target and labels

The target is ELMy-period occupancy: 97% of positive bins lie in crowd spans, and the
label set has no onset trace. Scored bins lie wholly inside reviewed spans, so
boundary-straddling bins are dropped and the task is easier than whole-shot detection.
Review started from the D-alpha clock: 56% of crowd starts, 44% of ends and 33% of both
(86 crowd spans) lie within 1 ms of its boundaries, which favours D-alpha-input models
over elm-elmo (BES and interferometers).

Of 93 non-crowd reviewed starts on 22 BES-subset shots, 63 (on 15 shots) match the
nearest elm-elmo onset within ±50 ms; the other 30 are omitted and each start is matched
independently. Reviewed start minus elm-elmo onset has median -20 ms and quartiles [-28,
-12] ms on the matched subset. This annotation-to-detector offset is not a measured
physical-onset error: neither reference has independently verified onset truth
(`review_start_offsets.json`, `elm_review_start_offsets.py`).

On non-crowd-only shots (13 shots, 1,003 bins) elm-ours precision is 0.298 [0.132,
0.549]. The absent-span alarm rate (any detection touching a covered absent span) is
0.596 [0.478, 0.689] for elm-ours and 0.105 [0.063, 0.168] for the clock on all119,
0.440 [0.301, 0.623] for elm-elmo on bes73. The centered 50 ms mean carries elm-ours
detections about 25 ms across reviewed edges and the clock's edges seeded the review, so
the raw rate favours the clock; with 25 ms trimmed from each absent-span edge (guard25)
the rates are 0.427 (elm-ours), 0.096 (clock) and 0.307 (elm-elmo, bes73).

## Benchmark

All review results are **developmental shot-CV estimates**: predictions on outer-fold
shots existed before the recipe was fixed and could have informed input choice, scaling,
architecture, selection or the protocol; the saved records do not establish whether they
did. Five shot-grouped folds cover the 119 reviewed shots; no blind-test shot enters
fits, normalization or threshold selection, and a frozen-model score on the blind split
awaits review of those shots. Each fold selects its checkpoint by inner-validation AUPRC
and its threshold by inner-validation F1. Intervals are 1,000-draw shot bootstraps
needing five denominator-bearing shots per endpoint.

elm-ours calls a bin present when its mean probability reaches the fold threshold;
elm-elmo and the clock use any touching detected span. Fold 3 (zero-based) selected
epoch 2 of 25; fold thresholds span 0.170–0.761. Four training seeds give all119 AUROC
0.924–0.949, AUPRC 0.858–0.913 and F1 0.815–0.841. On bes73, paired elm-ours minus
elm-elmo is AUROC +0.023 [-0.014, +0.067], AUPRC +0.043 [-0.058, +0.129], F1 +0.002
[-0.052, +0.058]: every interval includes zero (equivalence untested), while elm-ours
extends BES-free coverage to all 119 shots.

| Coverage / method | Shots / bins | AUROC [95% shot CI] | AUPRC | F1 | Absent-bin FPR | Absent-span alarm | Alarm, guard25 |
|---|---|---|---|---|---|---|---|
| all119 / elm-ours | 119 / 12,409 | 0.941 [0.906, 0.967] | 0.877 | 0.830 | 0.113 | 0.596 | 0.427 |
| all119 / elm-clock | 119 / 12,409 | -- | -- | 0.757 | 0.063 | 0.105 | 0.096 |
| all119 / always-present | 119 / 12,409 | 0.500 [0.500, 0.500] | 0.376 | 0.546 | 1.000 | 1.000 | 0.915 |
| all119 / elm-feature | 119 / 12,409 | 0.833 [0.783, 0.878] | 0.695 | 0.703 | 0.310 | -- | -- |
| bes73 / elm-ours | 73 / 6,843 | 0.941 [0.898, 0.972] | 0.875 | 0.845 | 0.123 | 0.610 | 0.436 |
| bes73 / elm-elmo | 73 / 6,843 | 0.918 [0.877, 0.951] | 0.833 | 0.842 | 0.108 | 0.440 | 0.307 |
| bes73 / elm-clock | 73 / 6,843 | -- | -- | 0.708 | 0.091 | 0.110 | 0.106 |
| bes73 / always-present | 73 / 6,843 | 0.500 [0.500, 0.500] | 0.402 | 0.573 | 1.000 | 1.000 | 0.890 |
| bes73 / elm-feature | 73 / 6,843 | 0.831 [0.770, 0.886] | 0.706 | 0.713 | 0.382 | -- | -- |

### Spans whose reviewed boundaries moved from the clock

43 of 84 bes73 present spans have no boundary within 1 ms of the clock's (an absent span
is the gap between clock spans): 4,125 bins on 57 shots, 1,429 positive, with 117 absent
spans. The intervals are wide and the stratum inconclusive; paired elm-ours minus
elm-elmo AUROC is +0.004 [-0.034, +0.037]. Source: `ours/moved_boundary_stratum.json`.

| bes73 moved-boundary bins / method | AUROC [95% CI] | AUPRC | F1 | FPR |
|---|---|---|---|---|
| elm-ours | 0.938 [0.875, 0.980] | 0.836 | 0.837 | 0.096 |
| elm-clock | -- | -- | 0.516 | 0.136 |
| elm-elmo | 0.934 [0.894, 0.965] | 0.816 | 0.822 | 0.098 |

**Secondary DSM common-bin control** (11,653 / 6,527 bins; tables in
`dsm/evaluation.json:sets` and the paper). The reduced-input elm-dsm detector scores
all119 AUROC 0.845 [0.796, 0.893], AUPRC 0.748, F1 0.742; its reported fit is fragile
(selected epochs 35, 31, 2, 24, 0). Paired elm-ours minus elm-dsm, all119: AUROC +0.094
[+0.055, +0.132], AUPRC +0.131 [+0.068, +0.195], F1 +0.091 [+0.040, +0.139]; bes73:
AUROC +0.099 [+0.058, +0.147], AUPRC +0.146 [+0.075, +0.227], F1 +0.066 [+0.009,
+0.126]. Paired elm-ours minus elm-elmo on bes73: AUROC +0.025 [-0.013, +0.071], AUPRC
+0.044 [-0.056, +0.131], F1 +0.007 [-0.044, +0.063]. These DSM detection rows are lower
bounds on DSM detection skill under our recipe, not the best achievable DSM performance.

## Serving protocol and sensitivities

**Serving.** A new shot runs whole (zero-padded to the network's length multiple)
through each of the five fold models; the score is their mean probability and the
threshold is the mean of the five fold thresholds (0.464, 0.299, 0.761, 0.170, 0.391;
mean 0.417). The reported numbers use the one fold model that never saw each shot, so
the ensemble and this threshold are unevaluated: every reviewed shot is in some fold's
training set and blind-test shots may not be used.

**Tiling.** Group normalisation takes its statistics over the whole input and training
used 4,096 ms crops. Scoring the same frozen models and thresholds on independent 4,096
ms tiles gives all119 AUROC 0.932 against 0.941 whole (whole minus tiled +0.009 [-0.002,
+0.024]). Nothing is refitted. Source: `ours/tiled_inference.json`.

**Run-day folds.** 16 of 94 run days cross the headline's shot-grouped folds. Repeating
the CV with every run day whole inside one fold (94 days, same recipe, folds not tuned
on performance) gives all119 AUROC 0.944 [0.916, 0.964] (headline 0.941), AUPRC 0.896
(0.877), F1 0.860 (0.830); bes73 AUROC 0.933 [0.894, 0.966] (headline 0.941), AUPRC
0.878 (0.875), F1 0.855 (0.845). The greedy day-balanced dealing balances shot counts,
not annotation kinds (shots with non-crowd spans per fold: 3, 4, 6, 8, 12), so this is a
sensitivity beside the headline. Source: `ours/run_day_cv.json`.

## DSM baselines

The detection DSM is a reduced-input adaptation trained and evaluated on 50 ms rows: 60
input columns and one 128-unit layer. PCPHD02/03 means come from a fresh fetch on 111 of
119 shots and from the upstream WPQH PCPHD02/03 export on 8; DENV2F and DENV3F means
supply the two density columns on 115 and 115 shots, with 4 and 4 rejected (failed
digitiser) and mean-filled. The historical source DSM trained on native 1 ms rows with
124 inputs and layers [100, 1000] for WPQH breakthrough-ELM forecasting, so the
detection row is not an objective-only retrain of that model and comparative claims
apply to this adaptation; the native detection refit is compared below. Historical
survival and detection variants that reuse source weights or statistics remain
supplemental. These DSM detection rows are lower bounds on DSM detection skill under our
recipe, not the best achievable DSM performance.

### Native DSM detection comparator

The native [100,1000] ReLU6 architecture, refitted for occupancy (25 epochs, original
folds, inner-validation selection, no source weights or blind-test shots) on the 37
shots with complete 124-input rows, is scored on identical support. The native refit
trains on 24–28 shots per fold with 2–8 inner-validation shots (train/inner-validation
by fold: 24/2, 28/3, 24/7, 24/3, 25/8), against 81–82 and 14 for the headline elm-ours;
its thresholds, chosen on so few shots, are erratic (3.1e-05–0.9999). Inputs and method:
the DSM card's Evaluation section.

| Matched panel / method | Shots / bins | AUROC [95% shot CI] | AUPRC | F1 |
|---|---|---|---|---|
| all119 / elm-ours | 37 / 3,576 | 0.936 [0.862, 0.981] | 0.853 | 0.833 |
| all119 / elm-ours (native-fold training shots) | 37 / 3,576 | 0.863 [0.795, 0.923] | 0.785 | 0.760 |
| all119 / elm-dsm-detect (60-input 1×128) | 37 / 3,576 | 0.823 [0.735, 0.899] | 0.704 | 0.759 |
| all119 / elm-dsm-native-detect (124-input [100,1000]) | 37 / 3,576 | 0.734 [0.610, 0.856] | 0.598 | 0.664 |
| bes73 / elm-ours | 37 / 3,339 | 0.929 [0.848, 0.980] | 0.854 | 0.837 |
| bes73 / elm-ours (native-fold training shots) | 37 / 3,339 | 0.852 [0.780, 0.917] | 0.789 | 0.766 |
| bes73 / elm-dsm-detect (60-input 1×128) | 37 / 3,339 | 0.805 [0.708, 0.889] | 0.706 | 0.763 |
| bes73 / elm-dsm-native-detect (124-input [100,1000]) | 37 / 3,339 | 0.715 [0.584, 0.844] | 0.604 | 0.667 |
| bes73 / elm-elmo | 37 / 3,339 | 0.890 [0.818, 0.948] | 0.808 | 0.818 |

`elm-ours (native-fold training shots)` repeats the elm-ours network, recipe and seeds
on the native folds' own train and inner-validation shots
(`scripts/labeler/elm_native_ours.py`; its fold thresholds span 0.013–0.782). On the
same bins it scores AUROC 0.863 [0.795, 0.923] against 0.936 [0.862, 0.981] for the
headline elm-ours, 0.734 [0.610, 0.856] for the native refit and 0.823 [0.735, 0.899]
for the 60-input adaptation. At equal training shots elm-ours has the higher point AUROC
than the native refit, with overlapping intervals. The headline's lead over the native
refit therefore mixes training size with architecture. This smaller support panel is a
secondary control and does not replace the primary all119/bes73 benchmark.

### DSM baselines retrained with post-warm-up selection

Both detection variants were refitted with three further seeds on the same outer folds
and recipe, choosing the epoch that ends the best three-epoch mean inner-validation
AUPRC window lying wholly at or after the warm-up (6 of 40 epochs for the 60-input
adaptation, 4 of 25 for the native refit); the reported fit takes the raw best epoch.
Ranges are over seeds, not intervals. Selected epochs span 8–37 and 6–24; native
thresholds span 0.0002–1.000, an erratic operating point.

| Detector | Shots / bins | Fit | AUROC | AUPRC | F1 |
|---|---|---|---|---|---|
| elm-dsm-detect (60-input 1×128) | 119 / 11,653 | reported (raw selection) | 0.845 | 0.748 | 0.742 |
| elm-dsm-detect (60-input 1×128) | 119 / 11,653 | post-warm-up repeats, mean (range) | 0.865 (0.857–0.870) | 0.740 (0.722–0.751) | 0.755 (0.749–0.763) |
| elm-dsm-native-detect (124-input [100,1000]) | 37 / 3,576 | reported (raw selection) | 0.734 | 0.598 | 0.664 |
| elm-dsm-native-detect (124-input [100,1000]) | 37 / 3,576 | post-warm-up repeats, mean (range) | 0.715 (0.707–0.725) | 0.598 (0.595–0.600) | 0.626 (0.603–0.642) |

These DSM detection rows are lower bounds on DSM detection skill under our recipe, not
the best achievable DSM performance. Source: `dsm/baseline_seeds.json`.

### Diagnostic rejection and inference sensitivity

Counts use all119 primary coverage (119 shots / 12,409 bins); a bin is affected if any
valid 0.1 ms input cell is screened or clipped. Clipping discards no shot or bin, and
its counts exclude already screened chords. Per-shot magnitudes are in
`ours/rejection_sensitivity.json`.

| Chord | Screen shots / bins | Clipping shots / bins | Threshold |
|---|---|---|---|
| FS02 | 0 / 0 | 67 / 1363 | log floor: native x < 1e12; no upper clip |
| FS03 | 0 / 0 | 71 / 1267 | log floor: native x < 1e12; no upper clip |
| FS04 | 0 / 0 | 54 / 1472 | log floor: native x < 1e12; no upper clip |
| DENV2F | 4 / 463 | 16 / 20 | native x / 1e14 outside [-3,12]; 10*high-pass outside [-10,10] |
| DENV3F | 5 / 523 | 9 / 174 | native x / 1e14 outside [-3,12]; 10*high-pass outside [-10,10] |

The failed-digitiser heuristic applies only to DENV2F/3F (median absolute native 0.1 ms
cell mean above 1e16); the primary U-Net zero-fills a rejected chord's two features.
Using rejected chords raw, with frozen weights, thresholds and bins and no retraining,
changes inputs on 6 shots: AUROC, AUPRC and F1 are 0.936, 0.869, 0.823 (changes -0.005,
-0.008, -0.007) and the false-alarm bin rate moves from 0.113 to 0.122. This does not
establish physical calibration or prove that screened chords are faulty.

## Signal provenance and numerical preprocessing

The retained caches do not establish fast-density physical ordinate units or FS01–04
sightlines (divertor versus midplane); FS01 is not an input to the occupancy U-Net, and
paired slow CO2 checks numerical scales only. Filterscope levels use `(log10(max(x,
1e12)) - 15) / 1.5` with contrast to a 0.5 s running median. Fast density is divided by
`1e14`, clipped to `[-3, 12]`, and ten times its 0.2 s high-pass is clipped to `[-10,
10]`; a chord whose median absolute native magnitude exceeds `1e16` is zeroed. These are
fixed numerical choices, not verified calibration or a validated failure criterion.
Offline audits changed no saved inputs, screening or weights and made no new fetches.
Over the flat-top window (1–4 s) of the 119 reviewed shots the divided fast-density
input has median 0.84, 5–95% range 0.17–1.93, and 0.4% of cells sit at the upper clip
(12; 9 of 238 chords are zeroed or empty in the window; `density_range.json`,
`scripts/labeler/elm_density_range.py`). This is a numerical range only: the ordinate
units and the FS02–04 sightlines are unverified, and no sightline list is retained in
the signal records or the literature digests. Sources: `density_units.json`,
`filterscope_metadata.json`, `src/labeler/elm/inputs.py`.

## Smith onset and transfer

Frozen occupancy transfer is omitted for target mismatch: 50 ms occupancy cannot resolve
approximately 8.5 ms Smith windows. The frozen reviewed-span start output is also
omitted. The experimental elm-ours-onset head is developmental CV: the Smith onset folds
are grouped by run day (none of 31 days crosses folds); overlap with the original
ELM-O's historical tuning events is unknown.

At ±2 ms the head recalls 0.925 of 2,316 hand-labelled windows on 211 shots; every
matched error is within 1.01 ms and 92% lie in the correct 1 ms cell. **Every method's
precision/F1 is conditional on selected windows; continuous-discharge precision/F1 is
unavailable.** 31,079 further firings lie outside the windows and lack negative truth;
the 10 ms minimum peak separation fixes the head's in-window precision, so none is
reported. elm-elmo region-overlap recall is 0.986 after the 179859 time-axis repair. On
the selected windows' 1 ms occupancy, which every method sees only conditionally, window
AUROC is 0.980 [0.978, 0.982] for the head and 0.883 [0.874, 0.891] for elm-elmo.
Traces: `$LABELER_ROOT/round4/elm/smith/cv/pred/`; catalog onsets stay withheld.

## Reference swap

The legacy onset table overlaps eight review shots (seven with BES). Finding 1 is read
on 782 known-majority cells: the legacy table's recall against the review is 0.692 and
its precision 0.906, with |M|=60 review-present cells it marks absent and |P|=14
review-absent cells it marks present; these are definition disagreements, not
adjudicated events. The 641 strict interior bins of the method comparison give |M|=40
and |P|=4. No ranking change: the AUROC order is unchanged. The F1 order of two method
pairs differs (review-tuned thresholds; descriptive). The elm-dsm-survival paired
legacy-minus-review AUROC change is +0.031 [-0.029, +0.132], in-sample on 5/8 shots and
within its interval; no AE Finding-2 analogue is established. One of 11 unadjusted
paired intervals excludes zero: elm-elmo on the BES shots (+0.043 [+0.007, +0.180];
seven shots). The three shots (2 with BES) outside original DSM fitting are too small
for intervals; their values are in the JSON. elm-elmo scores 0.603 [0.375, 0.947] on the
seven BES shots against 0.918 on bes73; the interval contains that value, so no domain
shift is established with seven shots. A fresh fetch of the 8 swap shots' PCPHD02/03
equals the upstream export sample for sample in 16 of 16 records (maximum absolute
difference 0; `dsm/swap_photodiode_agreement.json`), so the two sources agree in scale
and the detection AUROC gap on those shots is not an input-source artefact. Fixed
predictions are also compared on 641 strict interior bins; only AUROC compares
references, and covered-gap merges at 100/200/300 ms never bridge missing coverage.

## Sources and reproduction

Records under `outputs/labeler/elm/`:

- `ours/evaluation.json`, `ours/seed_repeats.json`, `ours/feature_only.json`,
  `ours/annotation_strata.json`: occupancy results, seeds, control and labels.
- `ours/moved_boundary_stratum.json`, `ours/tiled_inference.json`,
  `ours/run_day_cv.json`, `ours/rejection_sensitivity.json`: stratum and sensitivities.
- `dsm/evaluation.json`, `dsm/native_evaluation.json`, `dsm/native_detection.json`,
  `dsm/baseline_seeds.json`: common bins, refits, native targets and comparator,
  repeats.
- `dsm/swap_photodiode_agreement.json`, `density_range.json`: the swap shots' PCPHD02/03
  against a fresh fetch, and the fast-density input's numerical range.
- `smith/evaluation.json`, `swap/evaluation.json`, `review_start_offsets.json`: onset
  scope, swap and timing.
- `training_history.json`, `density_units.json`, `filterscope_metadata.json`: folds and
  signal provenance.

The consolidated `provenance.json` was removed in round seven; each record carries its
own top-level `git` commit (most also a creation stamp and a script digest), which
replaces it. Records without a `git` field (`dsm/native_fetch.json`,
`ours/seed_repeats.json`, `table_render_verification.json`, `training_history.json`)
name their sources and digests instead.

Run through the pixi labelmaker wrapper with `LABELER_NO_FETCH=1`: read the frozen run
from `ours/evaluation.json:run`, then `elm_ours_evaluate.py --run <run>`,
`elm_dsm_evaluate.py --run <run> --rescore`, `elm_feature_evaluate.py`,
`elm_moved_boundary_stratum.py`, `elm_tiled_inference.py`, `elm_run_day_summary.py`
(after `elm_ours_evaluate.py --run cv2_runday`), `elm_dsm_baseline_seeds.py`,
`elm_native_ours.py` (GPU; elm-ours on the native folds) then `elm_native_detect.py
--rescore`, `elm_dsm_fetch.py --native-photodiodes --fresh-shots <swap shots>` then
`elm_dsm_photodiode_agreement.py`, `elm_density_range.py`, `elm_smith_evaluate.py train
--group-by run_day` (GPU) and `elm_smith_evaluate.py evaluate`, `elm_reference_swap.py`,
`elm_example_figure.py --run <run>`, `elm_paper_tables.py` and `elm_protocol.py`.
Figures, checkpoints and predictions live under `$LABELER_ROOT/round4/elm/`; the
original ELM-O benchmark doc is retained with a dated update.
