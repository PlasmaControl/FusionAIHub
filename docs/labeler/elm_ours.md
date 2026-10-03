# ELM occupancy and onset evaluation

`elm-ours` is a 401,714-parameter 1D U-Net over FS02--FS04 log D-alpha and DENV2F/3F native density ordinates, without BES. The reviewed target is occupancy of known ELMy intervals in 50 ms bins, with 97.32% of positive bins supplied by crowd spans. Reviewed non-crowd starts are not independently verified physical onsets.

The DSM and legacy onset table were built on WPQH phases with breakthrough-ELM targets; Finding 1 and low DSM AUROCs partly reflect definition and domain shift (192721: 1 legacy bin versus 17 non-crowd review spans).

The fixed cohort is 400 train / 50 validation / 50 blind test. The 119 reviewed shots are 69 train and 50 validation; five fixed physical-shot folds reserve 14 inner-validation shots for checkpoint and F1 threshold selection. No blind-test shots enter fits, normalization or tuning. These review results are development estimates: earlier development included outer-fold predictions, and folds share run days. The original cv2 recipe and thresholds remain frozen; independent Smith evaluation uses its five-checkpoint ensemble. Smoothed selection is a sensitivity.

## Reviewed occupancy

Original signal-coverage panel: 119 shots / 12,409 bins; AUROC 0.941 [0.906, 0.967]; F1 0.830 [0.778, 0.871]. The main paper table uses identical DSM-covered bins for all core rows.

| Panel / method | Bins | AUROC | AUPRC | F1 |
|---|---:|---|---|---|
| all119 / elm-ours | 11,653 | 0.939 [0.901, 0.966] | 0.878 [0.775, 0.951] | 0.834 [0.782, 0.875] |
| all119 / elm-dsm (detection) | 11,653 | 0.845 [0.796, 0.893] | 0.748 [0.651, 0.832] | 0.742 [0.681, 0.798] |
| all119 / elm-clock | 11,653 | -- | -- | 0.759 [0.683, 0.829] |
| all119 / elm-always-present | 11,653 | 0.500 [0.500, 0.500] | 0.391 [0.332, 0.450] | 0.562 [0.499, 0.621]† |
| all119 / elm-feature-only | 11,653 | 0.833 [0.782, 0.879] | 0.704 [0.612, 0.789] | 0.713 [0.649, 0.770] |
| bes73 / elm-ours | 6,527 | 0.939 [0.893, 0.972] | 0.876 [0.752, 0.962] | 0.848 [0.790, 0.899] |
| bes73 / elm-elmo | 6,527 | 0.914 [0.869, 0.948] | 0.833 [0.755, 0.892] | 0.841 [0.786, 0.885] |
| bes73 / elm-dsm (detection) | 6,527 | 0.840 [0.777, 0.892] | 0.731 [0.612, 0.836] | 0.783 [0.711, 0.842] |
| bes73 / elm-clock | 6,527 | -- | -- | 0.714 [0.610, 0.806] |
| bes73 / elm-always-present | 6,527 | 0.500 [0.500, 0.500] | 0.417 [0.336, 0.500] | 0.588 [0.502, 0.666]† |
| bes73 / elm-feature-only | 6,527 | 0.829 [0.767, 0.888] | 0.712 [0.609, 0.825] | 0.723 [0.638, 0.798] |

Inputs are explicit in the main caption. The revised isolated DSM detector uses real PCPHD02/03 on all 119 shots, with no FS substitutes, and native DENV2F/3F means for CO2 v2/v3 (115 shots per chord; four per chord rejected and mean-filled). Its remaining 60-column inputs are actuator, magnetic, slow CO2 and ECE diagnostics, with fold-training-only normalization and mean fill. Native density physical calibration remains unresolved; scale/paired-unit checks and old/new DSM scores are in `dsm/evaluation.json`. Historical source-exposed survival and detection variants are supplemental. Native 124-input forecasts use forward presence and non-crowd onset targets in (t,t+h]. The four exact-export shots appear only in the appendix JSON record.

The clock seeded D-alpha review. Crowd-boundary identity within 1 ms is 56% of starts, 44% of ends and 33% of both; exact counts are in `ours/annotation_strata.json`. D-alpha sightlines remain unverified. Bin calls are U-Net mean score at its fold threshold, DSM bin-end score, or any touching ELM-O/clock span. Detected spans are clipped to panel coverage. Short inter-ELM gaps and centered 50 ms smoothing contribute to occupancy disagreements. Per-kind, annotation-mode, guarded alarm, review non-crowd tau-merge and seed diagnostics remain in appendix records.

Intervals use 1000 physical-shot bootstrap replicates. Valid and undefined counts are retained for each metric. Fewer than five shots with positive targets yields descriptive estimates only, without a population 95% interval; this includes the two-shot BES subset.

## Independent Smith evaluation

Frozen transfer fails on Smith's event-region target: occupancy AUROC 0.505 [0.493, 0.517], F1 0.257 [0.250, 0.264]†; frozen auxiliary onset F1 0.000 [0.000, 0.000] at ±2 ms and 0.129 [0.102, 0.156] at ±5 ms. The original elm-ours delivers occupancy only. The Smith-trained experimental head succeeds: event precision 1.000 [1.000, 1.000], recall 0.930 [0.915, 0.945], F1 0.964 [0.956, 0.972] at both tolerances; median absolute matched timing error 0.244 ms. Its shot-CV traces are benchmark outputs; catalog physical-onset output is withheld because selected-window negatives do not validate continuous-discharge false alarms. All 2,316 windows have complete input-record coverage, with conservative whole-cell edge exclusions. Smith and review have zero overlapping shots and run days.

`smith/evaluation.json` records 2,316 hand-labelled windows on 211 shots, exact shot/run-day overlap, frozen checkpoint hashes and thresholds, 1 ms occupancy metrics and one-to-one event onset matching at ±2/5 ms with timing errors. Window occupancy differs from 50 ms crowd occupancy. `elm-ours-onset` is trained only on Smith shots with grouped CV; review and cohort blind-test shots are excluded. ELM-O uses its published fixed setting. Its previous 0.997 precision / 0.980 recall are window-overlap reimplementation scores, not comparable onset matching scores; the paper digest reports 0.995 / 0.976 on 972 tuning ELMs. The frozen auxiliary onset output is not delivered; the Smith-trained head has the experimental scope stated above.

## Reference swap

The legacy WPQH onset table has 576 shots, eight overlapping review; seven have BES. Finding 1 on 782 known all-covered bins gives M/P = 60/14 (onset bins) and 34/60 (200 ms covered-gap occupancy). Unknown review time is excluded. These are definition disagreements, not adjudicated missing physical events. Covered-gap merges at 100/200/300 ms never bridge unavailable coverage. Finding 2 uses fixed predictions on both 641 strict interior bins and 782 known all-covered bins. Only AUROC supports cross-reference comparisons because F1 thresholds were review-tuned. Eight-shot evidence is inconclusive; point crossings do not establish a population ranking reversal. Full values and bootstrap draw counts are in `swap/evaluation.json`.

## Artifacts and reproduction

Small canonical JSON records and the six-row, two-panel main table are under `outputs/labeler/elm/`; supplemental tables are under `appendix/` and `swap/`. Large checkpoints, predictions, vector PDF / 150 dpi PNG figures and table proofs live under `$LABELER_ROOT/round4/elm/`. Every PNG must be visually inspected; the figure record documents the next-rank replacement for the ambiguous drop-shaped former panel (b).

Use the mandated pixi labelmaker wrapper for CPU scripts and the phase3 CUDA interpreter for GPU training (CUDA_VISIBLE_DEVICES=1, at most 12 GB). Set TMPDIR to the ELM scratch directory and LABELER_NO_FETCH=1 except authorized fetches. Executable entry points:

```text
scripts/labeler/elm_ours_evaluate.py --run cv2
scripts/labeler/elm_feature_evaluate.py --run cv2
scripts/labeler/elm_annotation_strata.py --run cv2
scripts/labeler/elm_dsm_evaluate.py --run cv2 --rescore
scripts/labeler/elm_dsm_native.py
scripts/labeler/elm_reference_swap.py --run cv2 --no-tables
scripts/labeler/elm_smith_evaluate.py --help
scripts/labeler/elm_paper_tables.py
scripts/labeler/elm_example_figure.py --run cv2
scripts/labeler/elm_protocol.py
```

Current code/record hashes identify evaluation provenance; historical training metadata remains retrospective where noted. The stream report contains fix history and detailed verification. No production labels or blind-test splits are modified.

The separate smoothed-selection sensitivity scores AUROC 0.958 [0.934, 0.975], AUPRC 0.931 [0.888, 0.960], F1 0.830 [0.779, 0.869] on original review bins. It does not replace frozen cv2 or support independent transfer claims.
