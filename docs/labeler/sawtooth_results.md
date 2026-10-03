# Sawtooth evaluation records

These tables describe the corrected four-state physics labels and GPU models. Uncertain and unassessed support never supplies negative targets. Algorithmic agreement does not establish independent physical accuracy.

## Data

| Fixed split | Shots | Diagnostic crash points | Definite train candidates | Uncertain candidate spans | Definite train shots |
|---|---:|---:|---:|---:|---:|
| train | 400 | 5449 | 147 | 10100 | 79 |
| val | 50 | 796 | 16 | 1446 | 6 |
| test | 50 | 494 | 9 | 1010 | 4 |

Source: `outputs/labeler/sawtooth/fix/data_summary.json` → `splits`.

These diagnostic counts describe train/candidate metadata. CSV state spans come from the canonical four-state support; overlapping uncertainty or unassessed support overrides a point's candidate state at export.

| Fixed split | Present (s) | Absent (s) | Uncertain (s) | Unassessed (s) |
|---|---:|---:|---:|---:|
| train | 28.543 | 1557.137 | 290.298 | 246.580 |
| val | 3.498 | 176.580 | 44.829 | 39.923 |
| test | 1.271 | 195.537 | 29.012 | 40.176 |

Source: `outputs/labeler/sawtooth/fix/data_summary.json` → `splits.<split>.state_seconds`.

Population: 16909 corpus files inspected; 13650 processed records, 186396 diagnostic crash points and 6068 definite train candidates. 3259 read failures are excluded from label truth. Processed records can have no observable support; those bins are unassessed. The same frozen rule applies to cohort and population.

Source: `outputs/labeler/sawtooth/fix/population_labels.json`.

## Old rule and physics labels against expert spans

Scores are shown separately for every reviewed shot. Primary metrics retrieve definite-present labels on all known expert bins with valid core ECE. No bootstrap is used for these shots.

| Shot | Assessed / observable known bins | Assessment fraction | Uncertain / observable positive bins |
|---|---:|---:|---:|
| 186636 | 2397 / 2405 | 0.997 | 0 / 905 |
| 189324 | 1109 / 2798 | 0.396 | 1238 / 1367 |
| 190637 | 2968 / 3003 | 0.988 | 0 / 1957 |

| Rule | Shot | Observable precision | Observable recall | Observable F1 | Observable span F1 | Conditional F1 | Conditional span F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Physics | 186636 | undefined | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| ece_sawtooth | 186636 | 1.000 | 0.471 | 0.640 | 0.571 | 0.640 | 0.571 |
| Physics | 189324 | undefined | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| ece_sawtooth | 189324 | 0.368 | 0.253 | 0.300 | 0.000 | 0.107 | 0.286 |
| Physics | 190637 | undefined | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| ece_sawtooth | 190637 | 1.000 | 0.250 | 0.400 | 0.000 | 0.400 | 0.000 |

Source: `outputs/labeler/sawtooth/fix/validation.json` → `expert.by_shot[0:3].new; expert.by_shot[0:3].old (shot-ordered)`.

Observable scores use the same expert-known core-ECE denominator for both rules and both models. Uncertain positive bins count as unresolved misses for definite-present recall/F1; they are never exported as absent. Conditional scores exclude the same uncertain support for both rules. Coverage and uncertain-positive counts expose the cost of abstention. Undefined precision means that the rule made no positive calls.

Expert annotations provide spans rather than point crash times. Picks inside positive spans are a support check; expert crash precision, recall and F1 remain unavailable. Calibrated radial localization and independent crash annotations remain necessary before promoting these labels to ground truth.

Span matching preserves original annotation identities and uses one-to-one IoU inside the stated assessment mask. Its recorded IoU threshold is 0.100; spans with no observable support are excluded.

## Agreement with the old detector

The read-only `heuristics.sawtooth_events` rule and the physics labels both ran on 500 shots. Agreement is measured within their common assessed core-ECE support; it measures detector consistency rather than accuracy.

| Crash metric, ±2 ms | Value [95% shot-bootstrap CI] |
|---|---:|
| precision | 0.162 [0.106, 0.234] |
| recall | 0.005 [0.003, 0.008] |
| f1 | 0.010 [0.006, 0.015] |

Source: `outputs/labeler/sawtooth/fix/validation.json` → `legacy_agreement`.

## Native-rate antialias filtering check

The detector and all thresholds are identical in both preparations. Only the ECE reader changes; these train shots did not select model thresholds.

| Shot | Stride-first candidates | Native-filter candidates | Stride-first crashes | Native-filter crashes | Median timing change (ms) |
|---|---:|---:|---:|---:|---:|
| 185945 | 745 | 825 | 0 | 0 | undefined |
| 186640 | 722 | 748 | 4 | 3 | 0.000217 |
| 189008 | 1056 | 1046 | 0 | 0 | undefined |
| 190992 | 792 | 785 | 8 | 5 | 0.000066 |

Timing change is measured only for one-to-one matched points; undefined means that no pair met the recorded tolerance. This is preprocessing sensitivity rather than a ground-truth timing score.

Source: `outputs/labeler/sawtooth/fix/antialias_impact.json` → `by_shot`.

## Muscatello reference shots

Shot 141182: local corpus file unavailable; no detector run or physical period/amplitude/timing check was performed.

Shot 141195: local corpus file unavailable; no detector run or physical period/amplitude/timing check was performed.


Source: `outputs/labeler/sawtooth/fix/muscatello_reference.json` → `by_shot`.

## Models on out-of-fold training-cohort shots

Each whole shot belongs to one outer fold. Inner selection shots from that fold's training portion select the checkpoint and operating thresholds. Fixed validation, expert and blind test shots never select a threshold or checkpoint. Training uses the CUDA environment and stops on inner selection loss patience. Intervals use 1,000 shot-bootstrap replicates.

| Model | Crash F1, ±1 ms | Crash F1, ±2 ms | Bin AUROC | Bin AUPRC | Bin F1 |
|---|---:|---:|---:|---:|---:|
| saw-hl3 | 0.083 [0.054, 0.117] | 0.084 [0.055, 0.118] | 0.751 [0.664, 0.821] | 0.048 [0.026, 0.079] | 0.101 [0.060, 0.147] |
| saw-ours | 0.469 [0.371, 0.555] | 0.478 [0.379, 0.565] | 0.863 [0.811, 0.904] | 0.173 [0.108, 0.243] | 0.288 [0.208, 0.355] |

| Model | Assessed bins | Observable bins | Uncertain bins | Unassessed bins |
|---|---:|---:|---:|---:|
| saw-hl3 | 792812 | 937972 | 145160 | 123219 |
| saw-ours | 792812 | 937972 | 145160 | 123219 |

Source: `outputs/labeler/sawtooth/fix/benchmark.json` → `Tokamak-SI`.

The inputs differ: `saw-hl3` receives two ECE group means, Mirnov mean and Ip; `saw-ours` receives all 48 ECE channels. The HL-3 paper used core/edge SXR, and verified DIII-D SXR spatial pairing is unavailable. This comparison therefore combines input information with architecture. The adapted HL-3 classifier gates a separate derivative picker because its published network does not output crash times.

## Independent model check per expert shot

Model expert scores use all known observable bins because independent expert annotations resolve algorithmic uncertainty. The coverage table retains the number of expert-positive bins the label rule called uncertain. No confidence interval is computed for the small expert set.

| Model | Shot | Observable / algorithm-assessed reviewed bins | Uncertain / observable positive bins |
|---|---:|---:|---:|
| saw-hl3 | 186636 | 2405 / 2397 | 0 / 905 |
| saw-hl3 | 189324 | 2798 / 1109 | 1238 / 1367 |
| saw-hl3 | 190637 | 3003 / 2968 | 0 / 1957 |
| saw-ours | 186636 | 2405 / 2397 | 0 / 905 |
| saw-ours | 189324 | 2798 / 1109 | 1238 / 1367 |
| saw-ours | 190637 | 3003 / 2968 | 0 / 1957 |

| Model | Shot | Observable reviewed bins | AUROC | AUPRC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| saw-hl3 | 186636 | 2405 | 0.825 | 0.644 | undefined | 0.000 | 0.000 |
| saw-hl3 | 189324 | 2798 | 0.904 | 0.808 | 0.874 | 0.953 | 0.912 |
| saw-hl3 | 190637 | 3003 | 0.169 | 0.496 | undefined | 0.000 | 0.000 |
| saw-ours | 186636 | 2405 | 0.634 | 0.451 | 0.381 | 0.009 | 0.017 |
| saw-ours | 189324 | 2798 | 0.675 | 0.578 | 0.486 | 0.177 | 0.260 |
| saw-ours | 190637 | 3003 | 0.374 | 0.591 | 0.000 | 0.000 | 0.000 |

Source: `outputs/labeler/sawtooth/fix/benchmark.json` → `Tokamak-SI.<model>.expert.by_shot`.

saw-hl3 pooled expert AUROC is 0.487, below chance. This independent ranking failure limits the model's physical validation despite its out-of-fold agreement with algorithmic labels.

Source: `outputs/labeler/sawtooth/fix/benchmark.json` → `Tokamak-SI.saw-hl3.expert.presence.auroc`.

saw-hl3 ranks expert truth below chance on shots 190637. This is an independent validation failure; agreement with its training labels does not resolve it.

saw-ours ranks expert truth below chance on shots 190637. This is an independent validation failure; agreement with its training labels does not resolve it.

## Published HL-3 context

Real time three-regime window accuracy: stated 0.922, count-derived 0.835.
Offline three-regime window accuracy: stated 0.956, count-derived 0.907.

OuYang et al., PPCF 67 (2025) 105004 reports HL-3 classification, a different task and population from DIII-D crash-tolerance scoring.

Source: `outputs/labeler/sawtooth/fix/benchmark.json` → `legacy`.

## GPU convergence and inner selection

| Model | Fold | Fit / selection shots | Best / completed epochs | Presence threshold | Crash threshold | Derivative z |
|---|---:|---:|---:|---:|---:|---:|
| saw-hl3 | 0 | 213 / 53 | 1 / 9 | 0.650 | 0.650 | 10.000 |
| saw-hl3 | 1 | 214 / 53 | 1 / 9 | 0.550 | 0.500 | 10.000 |
| saw-hl3 | 2 | 214 / 53 | 4 / 12 | 0.750 | 0.550 | 10.000 |
| saw-ours | 0 | 213 / 53 | 18 / 26 | 0.400 | 0.900 | 0.000 |
| saw-ours | 1 | 214 / 53 | 30 / 38 | 0.250 | 0.550 | 0.000 |
| saw-ours | 2 | 214 / 53 | 33 / 41 | 0.250 | 0.500 | 0.000 |

saw-hl3 fold 0: Tesla V100S-PCIE-32GB; inner selection loss patience exhausted. saw-hl3 fold 1: Tesla V100S-PCIE-32GB; inner selection loss patience exhausted. saw-hl3 fold 2: Tesla V100S-PCIE-32GB; inner selection loss patience exhausted. saw-ours fold 0: Tesla V100S-PCIE-32GB; inner selection loss patience exhausted. saw-ours fold 1: Tesla V100S-PCIE-32GB; inner selection loss patience exhausted. saw-ours fold 2: Tesla V100S-PCIE-32GB; inner selection loss patience exhausted.

Source: `outputs/labeler/sawtooth/fix/saw-*_fold_*.json`.

The selected derivative threshold reaches the upper requested grid boundary in saw-hl3 fold 0, saw-hl3 fold 1, saw-hl3 fold 2. The adapted baseline's crash performance remains limited despite the broader search.

See [the method, calibration provenance and reproduction](sawtooth_physics.md).
