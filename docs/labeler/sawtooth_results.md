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
| train | 127.089 | 44.475 | 1678.145 | 272.849 |
| val | 18.616 | 1.454 | 203.273 | 41.487 |
| test | 14.548 | 4.240 | 204.583 | 42.625 |

Source: `outputs/labeler/sawtooth/fix2/data_summary.json` → `splits.<split>.state_seconds`.

## State policy before and after

Absence requires candidate-free support within ±1.5 maximum periods and a recorded core-ECE relaxation test with no periodic pattern. Stable significant negative core edges protect their entire phase without the positive train's period bounds or crossing an observability gap. Undetected ambiguous support stays uncertain. This is a conservative research negative-label policy, not expert-validated absence.

| Scope / run | Present (s) | Absent (s) | Uncertain (s) | Unassessed (s) | Absent in <300 ms state holes (s) | Absent between candidate spans (s) |
|---|---:|---:|---:|---:|---:|---:|
| cohort / before | 33.312 | 1929.254 | 364.139 | 326.679 | 291.034 | 290.996 |
| cohort / after | 160.254 | 50.169 | 2086.001 | 356.960 | 7.872 | 0.000 |
| population / before | 1225.299 | 37419.786 | 9112.752 | 34380.014 | 6710.258 | 6709.009 |
| population / after | 5496.152 | 905.902 | 40897.400 | 34838.397 | 155.385 | 0.000 |

Candidate holes are gaps <300 ms between merged detected present/uncertain candidate spans. Canonical state holes also include tested quiet spans flanked by default uncertainty from context/noise limits; both definitions are reported explicitly.

Source: `outputs/labeler/sawtooth/fix2/state_transition_audit.json` → `scopes`.
Source: `outputs/labeler/sawtooth/fix2/cohort_phase_refinement.json` → `changed_shots`.
Source: `outputs/labeler/sawtooth/fix2/population_phase_refinement.json` → `changed_shots`.

Population: 16909 corpus files inspected; 13648 processed records, 144149 diagnostic crash points and 28714 definite train candidates. 3261 read failures are excluded from label truth. Processed records can have no observable support; those bins are unassessed. The same frozen rule applies to cohort and population, with different evidence availability.

Source: `outputs/labeler/sawtooth/fix2/population_labels.json`.

Previously processed shots now excluded: [186208, 186866]; their screened ECE lacks enough coherent physical core channels. These failures supply no training or scoring truth.

Source: `outputs/labeler/sawtooth/fix2/population_labels.json` → `errors`.

EFIT01 availability: 1,212/13,648 population shots versus 452/500 cohort shots. The prior run had 1,213/13,650 versus 452/500; equality of the rule does not give equality of equilibrium evidence.

Source: `outputs/labeler/sawtooth/fix2/population_labels.json` → `q_sources`.
Source: `outputs/labeler/sawtooth/fix2/cohort_labels.json` → `q_sources`.
Source: `outputs/labeler/sawtooth/fix/population_labels.json` → `q_sources`.
Source: `outputs/labeler/sawtooth/fix/cohort_labels.json` → `q_sources`.

## Nominal geometry coverage and q=1 comparison

Same-shot RF metadata and field/axis support determine nominal R. Other shots use a per-shot hottest physical channel proxy; their radii are null. These comparisons remain physically unvalidated.

| Scope | Mapped shots | Paired inversion / q=1 crashes | Median R difference (m) | Maximum absolute difference (m) |
|---|---:|---:|---:|---:|
| cohort | 4 | 17 | -0.084 | 0.207 |
| population | 10 | 52 | 0.014 | 0.207 |

Source: `outputs/labeler/sawtooth/fix2/population_labels.json` → `radius_geometry_counts; q1_major_radius_comparison`.
Source: `outputs/labeler/sawtooth/fix2/cohort_labels.json` → `radius_geometry_counts; q1_major_radius_comparison`.

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
| recall | 0.106 [0.083, 0.132] |
| f1 | 0.102 [0.080, 0.130] |

Source: `outputs/labeler/sawtooth/fix2/validation.json` → `legacy_agreement`.

## Muscatello reference shots

Shot 141182: local corpus file unavailable; no detector run or physical period/amplitude/timing check was performed.

Shot 141195: local corpus file unavailable; no detector run or physical period/amplitude/timing check was performed.


Source: `outputs/labeler/sawtooth/fix2/muscatello_reference.json` → `by_shot`.

## Models on out-of-fold training-cohort shots

Each whole shot belongs to one outer fold. Inner selection shots from that fold's training portion select the checkpoint and operating thresholds. Fixed validation, expert and blind test shots never select a threshold or checkpoint. Training uses the CUDA environment and stops on inner selection loss patience. Intervals use 1,000 shot-bootstrap replicates.

| Model | Crash F1, ±1 ms | Crash F1, ±2 ms | Bin AUROC | Bin AUPRC | Bin F1 |
|---|---:|---:|---:|---:|---:|
| saw-hl3 | 0.852 [0.805, 0.891] | 0.853 [0.807, 0.892] | 0.753 [0.658, 0.834] | 0.849 [0.770, 0.912] | 0.897 [0.862, 0.926] |
| saw-ours | 0.834 [0.802, 0.862] | 0.836 [0.805, 0.863] | 0.985 [0.974, 0.994] | 0.993 [0.988, 0.998] | 0.976 [0.961, 0.987] |

| Model | Assessed bins | Observable bins | Uncertain bins | Unassessed bins |
|---|---:|---:|---:|---:|
| saw-hl3 | 85767 | 924838 | 839071 | 136353 |
| saw-ours | 85767 | 924838 | 839071 | 136353 |

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
| saw-hl3 | 186636 | 2341 | 0.036 | 0.230 | 0.387 | 1.000 | 0.558 |
| saw-hl3 | 189324 | 2798 | 0.222 | 0.349 | 0.489 | 1.000 | 0.656 |
| saw-hl3 | 190637 | 2986 | 0.167 | 0.523 | 0.032 | 0.012 | 0.017 |
| saw-ours | 186636 | 2341 | 0.625 | 0.457 | 0.433 | 0.503 | 0.465 |
| saw-ours | 189324 | 2798 | 0.728 | 0.634 | 0.593 | 0.971 | 0.737 |
| saw-ours | 190637 | 2986 | 0.722 | 0.806 | 0.831 | 0.763 | 0.796 |

Source: `outputs/labeler/sawtooth/fix2/benchmark.json` → `Tokamak-SI.<model>.expert.by_shot`.

saw-hl3 pooled expert AUROC is 0.159, below chance. This anchored-span ranking result supplies no independent physical validation despite its out-of-fold agreement with algorithmic labels.

Source: `outputs/labeler/sawtooth/fix2/benchmark.json` → `Tokamak-SI.saw-hl3.expert.presence.auroc`.

saw-hl3 ranks the anchored span targets below chance on shots 186636, 189324, 190637. Neither this comparison nor agreement with algorithmic training labels establishes physical accuracy.

## Published HL-3 context

| System / dataset | Three-class window accuracy [95% CI] | Macro-F1 [95% CI] |
|---|---:|---:|
| Adapted saw-hl3 / unvalidated DIII-D labels | 0.651 [0.586, 0.708] | 0.626 [0.554, 0.686] |
| Fit-chosen majority / same DIII-D windows | 0.568 [0.486, 0.643] | 0.241 [0.218, 0.261] |
| Observed majority class 2 / same DIII-D windows | 0.568 [0.486, 0.643] | 0.241 [0.218, 0.261] |
| OuYang real time / HL-3 | 0.922 stated; 0.835 count-derived | not reported |
| OuYang offline / HL-3 | 0.956 stated; 0.907 count-derived | not reported |

Classification covers 85,788 20 ms windows at 2 ms hops, scored on observable & algorithm-assessed window centers. Per-fold fitting period boundaries are [40.64999903840272, 38.44999909044466, 38.599999086896375] ms.

Source: `outputs/labeler/sawtooth/fix2/benchmark.json` → `Tokamak-SI.saw-hl3.three_class_window_protocol`.

Confusion rows are algorithmic truth; columns are predictions. Classes are absent (0), short-period (1), long-period (2); the period boundary is fitted on each training fold.

| Truth class | Predicted 0 | Predicted 1 | Predicted 2 | Recall |
|---|---:|---:|---:|---:|
| 0 | 11692 | 3308 | 7240 | 0.526 |
| 1 | 271 | 11577 | 3004 | 0.779 |
| 2 | 4095 | 12019 | 32582 | 0.669 |

Source: `outputs/labeler/sawtooth/fix2/benchmark.json` → `Tokamak-SI.saw-hl3.three_class_*`.

The fit-chosen baseline predicts the natural fitting-window majority selected separately in each fold. The observed majority describes pooled out-of-fold class prevalence; its class is fixed across bootstrap resamples. Neither baseline selects model weights or thresholds.

OuYang et al., PPCF 67 (2025) 105004 reports HL-3 classification, a different task and population from DIII-D crash-tolerance scoring.

Source: `outputs/labeler/sawtooth/fix2/benchmark.json` → `legacy`.

## GPU convergence and inner selection

| Model | Fold | Fit / selection shots | Best / completed epochs | Presence threshold | Crash threshold | Derivative z |
|---|---:|---:|---:|---:|---:|---:|
| saw-hl3 | 0 | 213 / 53 | 2 / 10 | 0.050 | 0.450 | 20.000 |
| saw-hl3 | 1 | 214 / 53 | 8 / 16 | 0.100 | 0.450 | 20.000 |
| saw-hl3 | 2 | 214 / 53 | 1 / 9 | 0.350 | 0.100 | 17.000 |
| saw-ours | 0 | 213 / 53 | 20 / 28 | 0.950 | 0.950 | 0.000 |
| saw-ours | 1 | 214 / 53 | 12 / 20 | 0.150 | 0.950 | 0.000 |
| saw-ours | 2 | 214 / 53 | 12 / 20 | 0.450 | 0.950 | 0.000 |

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
