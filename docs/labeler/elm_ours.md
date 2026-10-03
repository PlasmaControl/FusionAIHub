# ELM occupancy and onset evaluation

`elm-ours` delivers BES-free ELMy-occupancy probability; the requested
finer physical-onset trace is **not delivered to the catalog**.

Current numbers and what each means (sources below):

- 0.941 AUROC: reviewed occupancy, 119 shots / 12,409 interior 50 ms bins.
- 0.830 [0.778, 0.871] F1: fold thresholds selected on inner validation.
- 0.953: mean per-fold AUROC sensitivity to pooling fold scores.
- 0.939 / 0.941: elm-ours AUROC on DSM-common / BES-only review coverage.
- 0.845 / 0.840: isolated DSM detector AUROC on common / BES-common bins.
- 0.607: native DSM forward-presence AUROC, four shots, no CI.
- 0.505 [0.493, 0.517]: frozen review-to-Smith occupancy AUROC.
- 0.930 [0.915, 0.945]: Smith-trained onset recall in selected windows.
- 0.82 ms / 94%: max matched error / correct 1 ms cell.
- 60 / 14: legacy/review disagreements; eight-shot swap is inconclusive.

## Scope and review benchmark

The 401,714-parameter 1D U-Net reads FS02–04 log D-alpha and DENV2F/3F
density, without BES. The target is reviewed ELMy occupancy; 97% of
positive bins come from crowd spans. Review started from the clock:
within 1 ms, 56% of crowd starts, 44% of ends and 33% of both match it.
Reviewed non-crowd starts sit about 5 ms before BES onsets and are not
independently verified physical onsets.

All reported review fits are **developmental shot-CV estimates**.
Preliminary outer-fold predictions existed before cv2; their influence
is unresolved. Sixteen review run days span folds. Five shot-grouped
folds cover 69 cohort-train and 50 validation shots; no blind-test shot
enters isolated fits, normalization or threshold selection. The original
cv2 checkpoints and thresholds stay frozen.

| Coverage / method | Shots / bins | AUROC [95% shot CI] | AUPRC | F1 |
|---|---:|---|---|---|
| all119 / elm-ours | 119 / 11,653 | 0.939 [0.901, 0.966] | 0.878 [0.775, 0.951] | 0.834 [0.782, 0.875] |
| all119 / elm-dsm (detection) | 119 / 11,653 | 0.845 [0.796, 0.893] | 0.748 [0.651, 0.832] | 0.742 [0.681, 0.798] |
| all119 / elm-clock | 119 / 11,653 | -- | -- | 0.759 [0.683, 0.829] |
| all119 / always-present | 119 / 11,653 | 0.500 [0.500, 0.500] | 0.391 [0.332, 0.450] | 0.562 [0.499, 0.621] |
| all119 / elm-feature | 119 / 11,653 | 0.833 [0.782, 0.879] | 0.704 [0.612, 0.789] | 0.713 [0.649, 0.770] |
| bes73 / elm-ours | 73 / 6,527 | 0.939 [0.893, 0.972] | 0.876 [0.752, 0.962] | 0.848 [0.790, 0.899] |
| bes73 / elm-dsm (detection) | 73 / 6,527 | 0.840 [0.777, 0.892] | 0.731 [0.612, 0.836] | 0.783 [0.711, 0.842] |
| bes73 / elm-clock | 73 / 6,527 | -- | -- | 0.714 [0.610, 0.806] |
| bes73 / always-present | 73 / 6,527 | 0.500 [0.500, 0.500] | 0.417 [0.336, 0.500] | 0.588 [0.502, 0.666] |
| bes73 / elm-feature | 73 / 6,527 | 0.829 [0.767, 0.888] | 0.712 [0.609, 0.825] | 0.723 [0.638, 0.798] |
| bes73 / elm-elmo | 73 / 6,527 | 0.914 [0.869, 0.948] | 0.833 [0.755, 0.892] | 0.841 [0.786, 0.885] |

DSM uses 60 input columns; PCPHD02/03 are measured on all 119 shots.
DENV2F/3F means fill v2/v3 on 115 shots per chord; four per chord are
mean-filled. Density calibration remains unresolved. The native DSM
has layers [100, 1000], while the refit uses one 128-unit layer.
Survival refits and historical exposed/initialized detectors are
supplemental; their upstream normalization includes two blind-cohort
shots. Native presence and non-crowd-start targets are forward (t,t+h].

Intervals use 1,000 physical-shot resamples, with valid/undefined counts.
Each endpoint needs five denominator-bearing shots; false-alarm rates
on negative-only shots retain their intervals. The saved per-fold AUROCs
provide a sensitivity to pooling differently calibrated fold scores.

## Smith onset and transfer

Frozen review-to-Smith transfer is independent: no shared review shots
or run days. The Smith-trained head is developmental shot CV with
substantial within-Smith run-day sharing. Overlap of Smith events with
ELM-O's historical tuning events remains unresolved.

The head recalls 0.930 [0.915, 0.945] of 2,316 hand-labelled windows
on 211 shots. Every matched error is within 0.82 ms; 94% lie in the correct 1 ms cell.
**Precision/F1: not estimable under selected windows.** False positives
are counted only inside windows, each holding one labelled event, while
30,436 additional firings lie outside. Those
firings lack negative truth. The 10 ms minimum peak separation further
restricts in-window false positives. Experimental traces are available
under `$LABELER_ROOT/round4/elm/smith/cv/pred/`; catalog onsets stay withheld.
ELM-O region-overlap recall is now 0.986 after the 179859 time-axis repair;
region overlap, onset matching and 1 ms occupancy are distinct targets.

## Reference swap

The legacy onset table overlaps eight review shots (seven with BES).
On 782 known all-covered 50 ms cells, M=60 and P=14; these are definition disagreements, not adjudicated events.
Unknown time stays unknown. Fixed predictions are also compared on 641
strict interior bins. Only AUROC compares references; F1 thresholds were
review-tuned. Covered-gap merges at 100/200/300 ms are sensitivities that
never bridge missing coverage. Source-unexposed subsets (3 shots; 2 with
BES) are too small for intervals; values are in the JSON. No population
ranking reversal is established.

## Sources and reproduction

All paths below are relative to `outputs/labeler/elm/`:

- `ours/evaluation.json:sets`: original occupancy and per-fold AUROCs.
- `dsm/evaluation.json:{sets,detectors,own_target}`: common bins and refits.
- `dsm/native_evaluation.json`: native forward targets and exposure.
- `ours/feature_only.json`, `ours/annotation_strata.json`: controls and labels.
- `smith/evaluation.json:{methods,onset_window_audit,protocol}`: onset scope.
- `swap/evaluation.json:{swap,interval_audit}`: fixed-prediction swap.

Use the mandated pixi labelmaker wrapper with LABELER_NO_FETCH=1:
`elm_ours_evaluate.py --run cv2`, `elm_dsm_evaluate.py --run cv2 --rescore`,
`elm_smith_evaluate.py evaluate`, `elm_paper_tables.py`,
`elm_example_figure.py --run cv2`, then `elm_protocol.py`.
Figures (PDF / 150 dpi PNG), checkpoints and predictions live under
`$LABELER_ROOT/round4/elm/`; paper tables remain in this worktree.
The original ELM-O benchmark doc is retained with a dated update.
