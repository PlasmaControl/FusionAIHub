# Sawtooth evaluation records

These tables describe unvalidated four-state research labels and GPU models. Uncertain and unassessed support never supplies negative targets. Blind expert crash times are unavailable. No model is recommended as latest or stable; algorithmic agreement does not establish independent physical accuracy.

## Data

| Fixed split | Shots | Diagnostic crash points | Definite train candidates | Uncertain candidate spans | Definite train shots |
|---|---:|---:|---:|---:|---:|
| train | 400 | 4391 | 729 | 10310 | 177 |
| val | 50 | 633 | 98 | 1407 | 26 |
| test | 50 | 404 | 90 | 1244 | 27 |

Source: `outputs/labeler/sawtooth/fix2/data_summary.json` → `splits`.

These diagnostic counts describe train/candidate metadata. CSV state spans come from the canonical four-state support; overlapping uncertainty or unassessed support overrides a point's candidate state at export.

| Fixed split | Present (s) | Absent (s) | Uncertain (s) | Unassessed (s) |
|---|---:|---:|---:|---:|
| train | 127.089 | 48.677 | 1673.943 | 272.849 |
| val | 18.616 | 1.929 | 202.798 | 41.487 |
| test | 14.548 | 6.059 | 202.764 | 42.625 |

Source: `outputs/labeler/sawtooth/fix2/data_summary.json` → `splits.<split>.state_seconds`.

## Old rule and physics labels against anchored expert spans

The span annotations for three expert shots were drawn while viewing the old `ece_sawtooth` suggestions and are anchored to that rule. Shot 190637's span may contain edge-originated relaxations. These checks are exploratory, unvalidated comparisons, not independent ground truth. Scores are shown separately for every reviewed shot. Primary metrics retrieve definite-present labels on all known expert bins with valid core ECE. No bootstrap is used for these shots.

| Shot | Assessed / observable known bins | Assessment fraction | Uncertain / observable positive bins |
|---|---:|---:|---:|
| 186636 | 0 / 2341 | 0.000 | 905 / 905 |
| 189324 | 1041 / 2798 | 0.372 | 470 / 1367 |
| 190637 | 271 / 2986 | 0.091 | 1957 / 1957 |

| Rule | Shot | Observable precision | Observable recall | Observable F1 | Observable span F1 | Conditional F1 | Conditional span F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Physics | 186636 | undefined | 0.000 | 0.000 | 0.000 | undefined | undefined |
| ece_sawtooth | 186636 | 1.000 | 0.471 | 0.640 | 0.571 | undefined | undefined |
| Physics | 189324 | 0.862 | 0.656 | 0.745 | 0.286 | 0.926 | 0.286 |
| ece_sawtooth | 189324 | 0.368 | 0.253 | 0.300 | 0.000 | 0.351 | 0.000 |
| Physics | 190637 | undefined | 0.000 | 0.000 | 0.000 | undefined | undefined |
| ece_sawtooth | 190637 | 1.000 | 0.250 | 0.400 | 0.000 | undefined | undefined |

Source: `outputs/labeler/sawtooth/fix2/validation.json` → `expert.by_shot[0:3].new; expert.by_shot[0:3].old (shot-ordered)`.

Observable scores use the same expert-known core-ECE denominator for both rules and both models. Uncertain positive bins count as unresolved misses for definite-present recall/F1; they are never exported as absent. Conditional scores exclude the same uncertain support for both rules. Coverage and uncertain-positive counts expose the cost of abstention. Undefined precision means that the rule made no positive calls.

Expert annotations provide spans rather than point crash times. Picks inside positive spans are a support check; expert crash precision, recall and F1 remain unavailable. Calibrated radial localization and independent crash annotations remain necessary before promoting these labels to ground truth.

Span matching preserves original annotation identities and uses one-to-one IoU inside the stated assessment mask. Its recorded IoU threshold is 0.100; spans with no observable support are excluded.

## Agreement with the old detector

The read-only `heuristics.sawtooth_events` rule and the physics labels both ran on 500 shots. Agreement is measured within their common assessed core-ECE support; it measures detector consistency rather than accuracy.

| Crash metric, ±2 ms | Value [95% shot-bootstrap CI] |
|---|---:|
| precision | 0.099 [0.078, 0.129] |
| recall | 0.104 [0.081, 0.130] |
| f1 | 0.101 [0.079, 0.129] |

Source: `outputs/labeler/sawtooth/fix2/validation.json` → `legacy_agreement`.

## Muscatello reference shots

Shot 141182: local corpus file unavailable; no detector run or physical period/amplitude/timing check was performed.

Shot 141195: local corpus file unavailable; no detector run or physical period/amplitude/timing check was performed.


Source: `outputs/labeler/sawtooth/fix2/muscatello_reference.json` → `by_shot`.

## Models on out-of-fold training-cohort shots

Each whole shot belongs to one outer fold. Inner selection shots from that fold's training portion select the checkpoint and operating thresholds. Fixed validation, expert and blind test shots never select a threshold or checkpoint. Training uses the CUDA environment and stops on inner selection loss patience. Intervals use 1,000 shot-bootstrap replicates.

| Model | Crash F1, ±1 ms | Crash F1, ±2 ms | Bin AUROC | Bin AUPRC | Bin F1 |
|---|---:|---:|---:|---:|---:|
| saw-hl3 | 0.854 [0.809, 0.893] | 0.855 [0.809, 0.893] | 0.705 [0.607, 0.795] | 0.805 [0.731, 0.877] | 0.883 [0.844, 0.913] |
| saw-ours | 0.822 [0.790, 0.850] | 0.828 [0.797, 0.853] | 0.986 [0.977, 0.993] | 0.993 [0.989, 0.997] | 0.978 [0.964, 0.988] |

| Model | Assessed bins | Observable bins | Uncertain bins | Unassessed bins |
|---|---:|---:|---:|---:|
| saw-hl3 | 87868 | 924838 | 836970 | 136353 |
| saw-ours | 87868 | 924838 | 836970 | 136353 |

Source: `outputs/labeler/sawtooth/fix2/benchmark.json` → `Tokamak-SI`.

The inputs differ: `saw-hl3` receives two ECE group means, Mirnov mean and Ip; `saw-ours` receives all 48 ECE channels. The HL-3 paper used core/edge SXR, and verified DIII-D SXR spatial pairing is unavailable. This comparison therefore combines input information with architecture. The adapted HL-3 classifier gates a separate derivative picker because its published network does not output crash times.

## Model comparison with anchored expert spans

Model span scores use all known observable bins; the anchored annotations supply exploratory span targets. The coverage table retains the number of expert-positive bins the label rule called uncertain. No confidence interval is computed for the small expert set.

| Model | Shot | Observable / algorithm-assessed reviewed bins | Uncertain / observable positive bins |
|---|---:|---:|---:|
| saw-hl3 | 186636 | 2341 / 0 | 905 / 905 |
| saw-hl3 | 189324 | 2798 / 1041 | 470 / 1367 |
| saw-hl3 | 190637 | 2986 / 271 | 1957 / 1957 |
| saw-ours | 186636 | 2341 / 0 | 905 / 905 |
| saw-ours | 189324 | 2798 / 1041 | 470 / 1367 |
| saw-ours | 190637 | 2986 / 271 | 1957 / 1957 |

| Model | Shot | Observable reviewed bins | AUROC | AUPRC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| saw-hl3 | 186636 | 2341 | 0.020 | 0.228 | 0.387 | 1.000 | 0.558 |
| saw-hl3 | 189324 | 2798 | 0.139 | 0.325 | 0.489 | 1.000 | 0.656 |
| saw-hl3 | 190637 | 2986 | 0.172 | 0.534 | 0.027 | 0.010 | 0.014 |
| saw-ours | 186636 | 2341 | 0.751 | 0.535 | 0.599 | 0.812 | 0.689 |
| saw-ours | 189324 | 2798 | 0.754 | 0.658 | 0.586 | 0.971 | 0.731 |
| saw-ours | 190637 | 2986 | 0.660 | 0.723 | 0.808 | 0.539 | 0.646 |

Source: `outputs/labeler/sawtooth/fix2/benchmark.json` → `Tokamak-SI.<model>.expert.by_shot`.

saw-hl3 pooled expert AUROC is 0.147, below chance. This anchored-span ranking result supplies no independent physical validation despite its out-of-fold agreement with algorithmic labels.

Source: `outputs/labeler/sawtooth/fix2/benchmark.json` → `Tokamak-SI.saw-hl3.expert.presence.auroc`.

saw-hl3 ranks the anchored span targets below chance on shots 186636, 189324, 190637. Neither this comparison nor agreement with algorithmic training labels establishes physical accuracy.

## Published HL-3 context

| System / dataset | Three-class window accuracy [95% CI] | Macro-F1 [95% CI] |
|---|---:|---:|
| Adapted saw-hl3 / unvalidated DIII-D labels | 0.601 [0.534, 0.667] | 0.576 [0.506, 0.643] |
| Fit-chosen majority / same DIII-D windows | 0.554 [0.472, 0.629] | 0.238 [0.214, 0.257] |
| Observed majority class 2 / same DIII-D windows | 0.554 [0.472, 0.629] | 0.238 [0.214, 0.257] |
| OuYang real time / HL-3 | 0.922 stated; 0.835 count-derived | not reported |
| OuYang offline / HL-3 | 0.956 stated; 0.907 count-derived | not reported |

Confusion rows are algorithmic truth; columns are predictions. Classes are absent (0), short-period (1), long-period (2); the period boundary is fitted on each training fold.

| Truth class | Predicted 0 | Predicted 1 | Predicted 2 | Recall |
|---|---:|---:|---:|---:|
| 0 | 12884 | 3110 | 8346 | 0.529 |
| 1 | 878 | 9561 | 4413 | 0.644 |
| 2 | 8467 | 9810 | 30419 | 0.625 |

Source: `outputs/labeler/sawtooth/fix2/benchmark.json` → `Tokamak-SI.saw-hl3.three_class_*`.

The fit-chosen baseline predicts the natural fitting-window majority selected separately in each fold. The observed majority describes pooled out-of-fold class prevalence; its class is fixed across bootstrap resamples. Neither baseline selects model weights or thresholds.

OuYang et al., PPCF 67 (2025) 105004 reports HL-3 classification, a different task and population from DIII-D crash-tolerance scoring.

Source: `outputs/labeler/sawtooth/fix2/benchmark.json` → `legacy`.

## GPU convergence and inner selection

| Model | Fold | Fit / selection shots | Best / completed epochs | Presence threshold | Crash threshold | Derivative z |
|---|---:|---:|---:|---:|---:|---:|
| saw-hl3 | 0 | 213 / 53 | 4 / 12 | 0.050 | 0.050 | 20.000 |
| saw-hl3 | 1 | 214 / 53 | 4 / 12 | 0.150 | 0.050 | 20.000 |
| saw-hl3 | 2 | 214 / 53 | 1 / 9 | 0.200 | 0.050 | 17.000 |
| saw-ours | 0 | 213 / 53 | 14 / 22 | 0.500 | 0.900 | 0.000 |
| saw-ours | 1 | 214 / 53 | 10 / 18 | 0.300 | 0.950 | 0.000 |
| saw-ours | 2 | 214 / 53 | 18 / 26 | 0.750 | 0.950 | 0.000 |

saw-hl3 fold 0: Tesla V100S-PCIE-32GB; inner selection loss patience exhausted. saw-hl3 fold 1: Tesla V100S-PCIE-32GB; inner selection loss patience exhausted. saw-hl3 fold 2: Tesla V100S-PCIE-32GB; inner selection loss patience exhausted. saw-ours fold 0: Tesla V100S-PCIE-32GB; inner selection loss patience exhausted. saw-ours fold 1: Tesla V100S-PCIE-32GB; inner selection loss patience exhausted. saw-ours fold 2: Tesla V100S-PCIE-32GB; inner selection loss patience exhausted.

Source: `outputs/labeler/sawtooth/fix2/saw-*_fold_*.json`.

## Pending blind annotation and paper example

The 15-shot nonexpert validation queue is in `data/events/sawtooth_oscillation/review/crash_time_queue.csv`; its regime stratification and shot/window list are recorded in `outputs/labeler/sawtooth/fix2/crash_time_queue.json`. The owner is away; blind crash accuracy remains unvalidated. The README contains the blind protocol.

Source: `outputs/labeler/sawtooth/fix2/crash_time_queue.json`.
The 3.25-inch example's trace channels, 300 ms window, vector PDF and 150-dpi PNG paths are recorded in `outputs/labeler/sawtooth/fix2/paper_example.json`. Markers are algorithmic candidates, not expert crash truth.

Source: `outputs/labeler/sawtooth/fix2/paper_example.json`.

## History appendix

The first correction used observable gaps as absence, an auxiliary noncorroboration gate, fixed ECE proxies and different training/inference masking. Its results remain archived under `outputs/labeler/sawtooth/fix/` and are superseded here. The three span comparisons were previously described as independent; the old-suggestion anchoring makes that description incorrect. No blind crash validation has been completed.

See [the method, calibration provenance and reproduction](sawtooth_physics.md).
