# Sawtooth evaluation records

These are research candidates: weak expert validation and unavailable radial calibration preclude a stable-model recommendation.

All model predictions below are out of fold on fixed train-cohort shots. Fixed validation shots select checkpoints and crash thresholds. All expert-reviewed shots and all fixed test shots are excluded from training and tuning. Intervals are 95% shot-bootstrap percentiles (1000 replicates). Presence bins are 2 ms, threshold 0.5.

## Data

| Fixed split | Shots | Crash candidates | Train spans | Positive shots |
|---|---:|---:|---:|---:|
| train | 400 | 30528 | 3136 | 361 |
| val | 50 | 4009 | 386 | 46 |
| test | 50 | 6470 | 690 | 49 |

Source: `outputs/labeler/sawtooth/data_summary.json` and `outputs/labeler/sawtooth/cohort_labels.json`.

Population: 16909 corpus files inspected; 13773 usable ECE shots, 1258685 crash candidates, 122055 train spans, 3136 absent/unusable/corrupt records. Source: `outputs/labeler/sawtooth/population_labels.json`.

## Physics labels versus expert spans

| Metric | Value [95% CI] |
|---|---:|
| Bin precision | 0.634 [0.000, 0.680] |
| Bin recall | 0.272 [0.000, 0.937] |
| Bin f1 | 0.381 [0.000, 0.788] |
| Span precision, IoU >= 0.1 | 0.111 [0.000, 0.167] |
| Span recall, IoU >= 0.1 | 0.200 [0.000, 1.000] |
| Span f1, IoU >= 0.1 | 0.143 [0.000, 0.286] |

Any-overlap span recall: 0.400. Supported picks: 47/102 inside positive expert spans. This is not crash precision. True crash precision/recall/F1 are unavailable because expert point times were not supplied. All known spans on shots 186636, 189324 and 190637 are used; category >=2 abstains and waveform coverage clips the assessment. Source: `outputs/labeler/sawtooth/validation.json`.

## Legacy / Tokamak-SI model results

Legacy HL-3 figures are published three-regime window classification: real-time stated accuracy 0.922 (count-derived 0.835); offline stated accuracy 0.956 (count-derived 0.907). The source is OuYang et al., PPCF 67 (2025) 105004. No published crash-tolerance score or interval CI is available.

| Tokamak-SI model | Crash F1, 1 ms | Crash F1, 2 ms | Train-bin AUROC | Train-bin AUPRC | Train-bin F1 |
|---|---:|---:|---:|---:|---:|
| saw-hl3 | 0.382 [0.348, 0.414] | 0.415 [0.378, 0.448] | 0.727 [0.701, 0.749] | 0.462 [0.420, 0.503] | 0.494 [0.461, 0.526] |
| saw-ours | 0.442 [0.412, 0.472] | 0.468 [0.437, 0.499] | 0.838 [0.822, 0.854] | 0.642 [0.599, 0.682] | 0.611 [0.579, 0.642] |

DIII-D baseline three-class window accuracy: 0.544 [0.515, 0.573]. The smaller/longer period boundary is fitted separately in each training fold. ECE replaces the HL-3 SXR pair. Its crash score uses a regime-gated derivative picker because the published architecture does not output point times. These task/input adaptations prevent direct comparison with the legacy accuracy.

Source for every model value: `outputs/labeler/sawtooth/benchmark.json`. Exact shots, boundaries, thresholds and training budgets are in `split_manifest.json` and each model's `*_fold_*.json` records.

## Independent expert check of trained models

| Three-fold ensemble | Train-bin AUROC | Train-bin AUPRC | Train-bin F1 |
|---|---:|---:|---:|
| saw-hl3 | 0.419 [0.162, 0.692] | 0.480 [0.447, 0.537] | 0.538 [0.008, 0.731] |
| saw-ours | 0.677 [0.587, 0.805] | 0.600 [0.483, 0.750] | 0.338 [0.050, 0.580] |

Same reviewed shots/known bins as detector validation. True expert crash scores remain unavailable. Three shots yield fragile confidence intervals. Source: `outputs/labeler/sawtooth/benchmark.json`, expert blocks.

See [the method and limitations](sawtooth_physics.md) for reproduction, signal coverage and scientific qualifications.
