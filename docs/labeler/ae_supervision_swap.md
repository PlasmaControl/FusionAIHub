# AE supervision swap

Does the ranking reversal between ae-ours and the older CO2 detectors come from training supervision? The same AeSeldNet recipe is trained on three activity targets (the legacy annotation, the dense relabel, annotation-and-TokEye agreement; three seeds each) with epochs chosen on 20 held-out training shots, then scored with ae-rcn, ae-lstm and an input-free clock against both references on the 60 validation shots and on the 19 shots the older detectors did not train on.

Source of every number below: [evaluation.json](../../outputs/labeler/ae/supervision_swap/evaluation.json) (written by `scripts/labeler/ae_supervision_swap.py evaluate`; run records, execution IDs, thresholds, per-seed scores, paired differences, the clock priors and the convergence audit are embedded). [manifest.json](../../outputs/labeler/ae/supervision_swap/manifest.json) holds the shot lists and input hashes; [verification.json](../../outputs/labeler/ae/supervision_swap/verification.json) checks target equality and split isolation. The main-text paper table is `table_supervision_swap_main.tex` and the two long appendix tables are `table_supervision_swap.tex`, in the same directory.

## Summary

On the 19 shared held-out shots the seed-mean AUROC of ae-ours trained on the legacy annotation, the dense relabel and annotation-and-TokEye agreement is 0.807, 0.980, 0.980 against the dense reference and 0.747, 0.685, 0.711 against the legacy annotation. ae-ours trained on the legacy annotation is below ae-rcn on both references in point estimate (resolved on the legacy annotation only), so, for this recipe (rather than the best achievable legacy-trained model), the reversal against ae-rcn needs dense or agreement targets; the reversal against ae-lstm persists (marginally on the dense reference). An input-free clock reaches 88% of ae-rcn's AUROC margin over chance against the legacy annotation, which makes time context a confound of every comparison with the saved detectors, and the dense reference is temporally coarse, so within-shot AUROC cannot rank methods on it. A. Garcia, the author of the saved detectors, saved interval changes to 28 training and selection shots and 0 evaluation shots of the dense labels (Limitations). Scores are float32; the largest change of a run's pooled AUROC from the bfloat16 inference is 0.0103. The Interpretation and Cross-architecture confounds sections give the numbers and the limits.

## Convergence rule and the nine records

Fix round 1 found that the first legacy seed 2 never left the constant-output plateau (selection loss lowest at epoch 2, patience 5 ended it at epoch 7, constant score). One rule, written in `CONVERGENCE_RULE` and in this document before any rerun, applies uniformly to every record of every arm:

1. Early stopping counts non-improving epochs only from zero-based epoch 10, so a plateau in the first ten epochs cannot end a run. (Maximum 30 epochs, patience 5, minimum improvement 1e-06, monitored on combined loss on the 20 selection shots.) A record conforms when replaying this rule on its recorded selection-loss history stops it at the recorded final epoch; a record that stopped earlier is rerun with the same seed.
2. A record whose selected checkpoint is constant (within-shot SD below 0.005) or does not rank its own target above chance (selection AUROC at most 0.5) is excluded and replaced by the arm's next unused seed, starting at 3, which is trained under the same training rule and screened again. Selection screen: mean over selection shots of the population SD of the score at least 0.005, and selection AUROC above 0.5, on the selected checkpoint, selection shots, its own activity target.

An earlier draft of the rule (the screen alone) was applied by the previous implementer to selection-only outputs and is archived as `convergence_draft0.json`; this two-part rule replaced it before any rerun. The failure of the first legacy seed 2 was already known from review when the rule was written, so the rule is predeclared with respect to the reruns, not blind to that failure.

| Record | Last epoch | Selected epoch | Rule 1 | Within-shot SD | Selection AUROC | Outcome |
|---|---:|---:|---|---:|---:|---|
| ae-ours-dense-seed0 | 24 | 19 | conforms | 0.392 | 0.944 | accepted |
| ae-ours-dense-seed1 | 20 | 15 | conforms | 0.378 | 0.922 | accepted |
| ae-ours-dense-seed2 | 18 | 13 | conforms | 0.375 | 0.932 | accepted |
| ae-ours-legacy-seed0 | 22 | 17 | conforms | 0.213 | 0.787 | accepted |
| ae-ours-legacy-seed1 | 22 | 17 | conforms | 0.207 | 0.789 | accepted |
| ae-ours-legacy-seed2 | 29 | 27 | conforms | 0.288 | 0.850 | accepted |
| ae-ours-threeway-seed0 | 18 | 13 | conforms | 0.374 | 0.991 | accepted |
| ae-ours-threeway-seed1 | 14 | 9 | conforms | 0.371 | 0.988 | accepted |
| ae-ours-threeway-seed2 | 29 | 26 | conforms | 0.378 | 0.993 | accepted |
| ae-ours-dense-seed1-superseded1 (first attempt) | 11 | 6 | stopped inside the plateau | 0.374 | 0.920 | superseded: rerun with the same seed |
| ae-ours-legacy-seed2-superseded1 (first attempt) | 7 | 2 | stopped inside the plateau | 0.001 | 0.460 | superseded: rerun with the same seed |

Of the nine first-generation records, 7 conformed to rule 1 as they stood; 2 stopped inside the plateau (ae-ours-dense-seed1-superseded1, ae-ours-legacy-seed2-superseded1) and were rerun with the same seed. After rerunning, 9 of the 9 final records pass rule 2 and none needed replacing. Selection-only outputs were used for every decision above; no evaluation frame entered the rule. The first-attempt records are kept on disk and scored in the last section, so the effect of the rule is visible.

The table above and the decisions it records used the bfloat16 outputs; every score below is a float32 re-inference of the same checkpoints. Re-screened on the float32 selection scores, 9 of 9 accepted records pass rule 2 (within-shot SD 0.206 to 0.392, selection AUROC 0.787 to 0.994); no record changes status.

## Frozen protocol

**Dense label counts.** The paper's 943 intervals are the 943 rows of the table its audit scored, the ae_xpower v2 review table (`/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/models/ae_xpower/v2/review/labels.csv`, SHA256 `5f52a26831cd9173f3d67b1e540ba23ee4aa1adab9159b1312a663cf88eee1de`): 407 present and 536 absent rows over 180 shots. Its any-touch prevalence (present among present and absent 10 ms frames) is 0.409 on the 120 training and selection shots (the paper's 0.41) and 0.696 on the 60 evaluation shots (0.70). The current table (`review/labels.csv`, SHA256 `b4308ff52d766779f33a92e9441a916bb07c854dcc1a98a2f61109bb6431e93a`) has 877 rows over 180 shots, 371 present and 506 absent; every present row is a merged crowd span (`iscrowd`), so rows are not individual boxes. It differs from the snapshot on 29 of the 120 training and selection shots (870 frames, a net +548 present frames; prevalence 0.432) and on 1 of the 60 evaluation shots (4 frames, a net -4; shot 170667), and on 0 of the 19 shared held-out shots. The evaluation reference therefore matches the paper's audit; the training labels are a later version of it. The paper's second count, 954 "after later review", matches neither table. Source: `dense_counts`, `dense_reconciliation` and `dense_history` in evaluation.json.

NumPy split seed 20261003 freezes 100 training, 20 selection and 60 evaluation shots; none is in the fixed catalog's blind test split. Seeds 0, 1 and 2 change initialisation and the sampled windows; every arm uses the same split. The original one-seed manifest was archived before the seed list was extended.

Only the activity target changes between arms. The legacy arm trains on the legacy annotation (the Heidbrink hand annotation) by the audit's at-least-half-annotated rule at 10 ms (classes 1 to 4, LFM excluded), expanded to the native columns; the dense arm uses the catalog any-touch state rule and masks unknown states; threeway keeps the native annotation-and-TokEye agreement, needs at least half the native columns of a frame to carry agreement weight, and masks disagreement. The annotation-and-TokEye frequency target and its weights are identical in every arm (verification.json). Each input covers 0 to 2 s as 7,820 native columns.

**Band.** The model reads the CO2 spectrogram from 80.57 to 250.00 kHz (348 bins, four chords). The review display the dense labels were drawn on read 80-250 kHz until 2026-09-30 and 60-250 kHz afterwards (BAND_KHZ comment in src/labeler/events/review/alfven.py). The 29 training and selection shots that differ from the paper's snapshot were changed with the 60-250 kHz display (29 of 29; 1 of them also carry an earlier change made with the 80-250 kHz display), so present frames added to them can rest on activity between 60 and 80.6 kHz that ae-ours cannot see. Among the reviewers, A. Garcia saved changes on 28 of the 29 training and selection shots that differ from the paper's snapshot (22 training, 6 selection) and on 0 evaluation shots; N. Chen on 6 of them and 0 evaluation shots; unnamed saves on 1 of them and 1 evaluation shot (a shot can carry several). A. Garcia is the author of ae-rcn and ae-lstm (Limitations). The arms train on the current table, edits included. "Absent" in the dense labels, and in every target ae-ours trains on, means absent within the observed band; it says nothing about activity below 80.6 kHz, which the legacy annotation can mark and which can therefore be labelled absent. The saved UCI detectors read 20 to 250 kHz.

**Frame grid.** The audit's 10 ms frames spread the 7,820 native columns evenly over 0 to 2 s (`frame_index`), and every arm, target and reference here inherits that assignment. The recorded column times run from -0.77 to 2000.89 ms, so 372 of the 7,820 columns fall in a neighbouring frame, at most 1.02 ms from the uniform centre (identical in all 180 shots).

**Recipe.** The original AeSeldNet recipe (four channels, 348 frequency bins, pools (6, 2, 29), two bidirectional GRUs, SCE plus the unchanged frequency objective, 710-column windows (182 ms), eight windows per shot, batch 16, AdamW 1e-4, weight decay 1e-4, cosine decay to 1e-6) with the convergence rule above; CUDA allocations are capped at 10 GiB. The recipe was developed for the threeway target: symmetric cross entropy was chosen over binary cross entropy on the 60 validation shots (the model README), and its weights, the learning rate and early stopping were set for that target; the legacy and dense arms reuse it unchanged. The persistent DataLoader workers keep their epoch-0 copy of the dataset, so `set_epoch()` never reaches them: every epoch re-draws the same eight windows per shot (800 windows), and only the shot order is reshuffled. This holds for every arm and seed, including the published model's recipe, and it is not fixed in this round: changing it would require rerunning everything (checked in isolation with and without persistent workers).

**Thresholds.** The epoch minimises the combined loss on the 20 selection shots. The headline F1 calibrates every method on the reference it is scored against: the threshold maximises 10 ms F1 on that method's selection shots (20; six for ae-rcn and ae-lstm, the only ones they did not train on; the clocks use the 20) with ties going to the highest threshold. The clocks' thresholds are tuned on the 20 selection shots, which are part of the 120 shots the clock's own positive rates come from, so clock F1 is in-sample; their AUROC and AUPRC need no threshold. F1 at each record's own-target threshold (the earlier protocol: each arm's threshold from its own activity target, the older detectors' from the legacy annotation) is kept in evaluation.json (`f1_own_target_threshold`) and is used in no claim. No evaluation frame selects an epoch or a threshold.

**Precision.** The training run scored each checkpoint under bfloat16 autocast. Every score in this document is a float32 re-inference (autocast off) of the same checkpoints, with the thresholds recalibrated on the float32 selection scores; the next section gives the change.

**Saved detectors.** Garcia probabilities are the saved spectrogram predictions: maximum over the first four AE classes, mean over four chords, nearest output-bin centre on the 10 ms grid. Only 19 evaluation shots and six selection shots have predictions for shots the detectors did not train on.

## Completed runs

Epochs are zero-based. The patience column is the epoch from which non-improving epochs count: records trained after the rule was declared use 10, earlier records counted from 0 and the audit above shows that the rule gives them the same stopping epoch.

| Supervision | Seed | Selected epoch | Epochs run | Patience from | Own-target threshold (float32) | Execution | GPU |
|---|---:|---:|---:|---:|---:|---|---|
| dense | 0 | 19 | 25 | 0 | 0.2135 | 2952146_1 (job 2952150) | NVIDIA A100-PCIE-40GB |
| dense | 1 | 15 | 21 | 10 | 0.0349 | head node | Tesla V100S-PCIE-32GB |
| dense | 2 | 13 | 19 | 0 | 0.0846 | head node | Tesla V100S-PCIE-32GB |
| legacy | 0 | 17 | 23 | 0 | 0.0242 | 2952146_0 (job 2952149) | NVIDIA A100-PCIE-40GB |
| legacy | 1 | 17 | 23 | 0 | 0.1518 | head node | Tesla V100S-PCIE-32GB |
| legacy | 2 | 27 | 30 | 10 | 0.3202 | head node | Tesla V100S-PCIE-32GB |
| threeway | 0 | 13 | 19 | 0 | 0.3885 | head node | Tesla V100S-PCIE-32GB |
| threeway | 1 | 9 | 15 | 0 | 0.3929 | head node | Tesla V100S-PCIE-32GB |
| threeway | 2 | 26 | 30 | 0 | 0.2947 | head node | Tesla V100S-PCIE-32GB |

ae-ours-dense-seed0, ae-ours-legacy-seed0 ran on A100 GPUs; every other record ran on the head node's V100S. This Torch build reports V100 bfloat16 support through emulation, so the unchanged trainer autocasts to bfloat16 on both (gpu_probe.json) when it scores the checkpoint; every score in this document is instead a float32 re-inference of the same checkpoints (next section). Hardware is therefore a nuisance variable of the supervision comparison; the V100-only sensitivity analysis below restricts every arm to its V100 runs of seeds other than 0.

## Inference precision

The training run scored each checkpoint under bfloat16 autocast. This Torch build emulates bfloat16 on the V100, and the frame probabilities it gives differ visibly from float32 ones. Every checkpoint was therefore re-inferred in float32 (autocast off; `infer-fp32`, files `probabilities_fp32.npz` and `fp32.json` beside each run), the thresholds were recalibrated on the float32 selection scores, and all scores, thresholds and intervals in this document are float32. The bfloat16 files are untouched. Change in the pooled-frame score, float32 minus bfloat16 (`precision_check` in evaluation.json): the largest change of any run in any cohort and reference is 0.0103 AUROC (ae-ours-legacy-seed1, all_60, dense reference) and 0.0203 AUPRC (ae-ours-legacy-seed2, fair_19, legacy reference). Seed-mean change per arm (AUROC / AUPRC):

| Arm | 60 shots, dense | 60 shots, legacy annotation | 19 shared held-out shots, dense | 19 shared held-out shots, legacy annotation |
|---|---|---|---|---|
| legacy | +0.0054 / +0.0020 | +0.0001 / -0.0042 | +0.0058 / +0.0020 | -0.0007 / -0.0068 |
| dense | -0.0002 / -0.0001 | -0.0008 / -0.0037 | -0.0005 / -0.0002 | -0.0020 / -0.0087 |
| threeway | -0.0001 / -0.0000 | -0.0015 / -0.0034 | -0.0003 / -0.0001 | -0.0014 / -0.0021 |

Frame by frame the two precisions disagree more than the pooled scores do: on the 60 evaluation shots the largest single-frame difference between the two probabilities of a shot exceeds 0.1 on 0 to 43 shots per record (largest overall 0.68). Pooled over a cohort, the largest change of a run's AUROC is 0.0103; the bfloat16 scores stay on disk (`probabilities.npz`) and the first evaluation that used them is in the git history of this document.

## Scores

AUROC and AUPRC are the primary metrics: they need no threshold. Per-seed rows and baselines give the pooled-frame estimate with a 95% interval from 1,000 shot-bootstrap replicates (seed 20261004). Mean rows give the mean and sample SD over seeds of the per-seed pooled values. Shot AUROC is the median over shots with both classes of the within-shot AUROC, with a shot-bootstrap interval; it removes differences in calibration between shots. F1 is calibrated on the reference being scored for every method (above). The dense reference is temporally coarse (Interpretation): Shot AUROC cannot rank methods on it.

**Main table** (19 shared held-out shots; the main-text LaTeX table is `table_supervision_swap_main.tex`; ae-ours rows are the seed mean and sample SD, the other rows single models; the last two columns are the paired AUROC difference from ae-rcn with a shot-bootstrap interval):

| Model | Training labels | Dense AUROC | Dense AUPRC | Legacy-annotation AUROC | Legacy-annotation AUPRC | AUROC minus ae-rcn, dense | AUROC minus ae-rcn, legacy annotation |
|---|---|---:|---:|---:|---:|---|---|
| ae-ours | legacy annotation | 0.807 ± 0.038 | 0.910 ± 0.024 | 0.747 ± 0.029 | 0.548 ± 0.008 | -0.055 [-0.149, 0.036] | -0.164 [-0.254, -0.074] |
| ae-ours | dense relabel | 0.980 ± 0.006 | 0.991 ± 0.002 | 0.685 ± 0.014 | 0.323 ± 0.024 | +0.118 [0.066, 0.175] | -0.226 [-0.351, -0.119] |
| ae-ours | annotation-TokEye agreement | 0.980 ± 0.005 | 0.992 ± 0.002 | 0.711 ± 0.021 | 0.372 ± 0.035 | +0.119 [0.070, 0.175] | -0.200 [-0.310, -0.098] |
| ae-rcn | saved, 801 shots | 0.862 | 0.930 | 0.911 | 0.691 | n/a | n/a |
| ae-lstm | saved, 801 shots | 0.736 | 0.855 | 0.898 | 0.668 | -0.125 [-0.200, -0.058] | -0.013 [-0.039, 0.012] |
| clock | from the legacy annotation | 0.818 | 0.905 | 0.860 | 0.558 | -0.043 [-0.091, 0.007] | -0.051 [-0.079, -0.020] |
| clock | from the dense relabel | 0.875 | 0.936 | 0.877 | 0.552 | +0.013 [-0.033, 0.065] | -0.034 [-0.058, -0.013] |

### All 60 evaluation shots

Shots: 170659, 170660, 170661, 170662, 170663, 170664, 170665, 170666, 170667, 170669, 170671, 170672, 170675, 170677, 170678, 170679, 170714, 170715, 170716, 170717, 170718, 170719, 170720, 170721, 170722, 170724, 170725, 170727, 170729, 170730, 170790, 170791, 170792, 170793, 170794, 170795, 170797, 170798, 170799, 170801, 170803, 170805, 170806, 170807, 170808, 170809, 170810, 170811, 170813, 170814, 175239, 175240, 175241, 175244, 175245, 175250, 175251, 175252, 178872, 178879.

**Against the dense reference** (12000 scorable frames, 8352 positive).

| Method | AUROC | AUPRC | Shot AUROC | F1 |
|---|---|---|---|---|
| ae-ours, legacy annotation, seed 0 | 0.885 [0.856, 0.913] | 0.953 [0.934, 0.967] | 0.931 [0.900, 0.949] | 0.869 [0.839, 0.894] |
| ae-ours, legacy annotation, seed 1 | 0.805 [0.762, 0.846] | 0.905 [0.880, 0.928] | 0.797 [0.720, 0.867] | 0.828 [0.802, 0.852] |
| ae-ours, legacy annotation, seed 2 | 0.879 [0.848, 0.908] | 0.947 [0.929, 0.963] | 0.880 [0.845, 0.947] | 0.845 [0.810, 0.876] |
| **ae-ours, legacy annotation, mean over seeds** | 0.856 ± 0.044 | 0.935 ± 0.026 | 0.869 ± 0.068 | 0.847 ± 0.020 |
| ae-ours, dense relabel, seed 0 | 0.984 [0.975, 0.991] | 0.993 [0.987, 0.997] | 0.999 [0.995, 1.000] | 0.960 [0.942, 0.975] |
| ae-ours, dense relabel, seed 1 | 0.980 [0.969, 0.989] | 0.991 [0.984, 0.996] | 0.999 [0.996, 1.000] | 0.954 [0.932, 0.971] |
| ae-ours, dense relabel, seed 2 | 0.988 [0.982, 0.994] | 0.995 [0.991, 0.998] | 0.999 [0.997, 1.000] | 0.965 [0.951, 0.977] |
| **ae-ours, dense relabel, mean over seeds** | 0.984 ± 0.004 | 0.993 ± 0.002 | 0.999 ± 0.000 | 0.960 ± 0.005 |
| ae-ours, annotation-and-TokEye agreement, seed 0 | 0.978 [0.958, 0.990] | 0.990 [0.978, 0.996] | 0.998 [0.994, 1.000] | 0.936 [0.913, 0.958] |
| ae-ours, annotation-and-TokEye agreement, seed 1 | 0.977 [0.959, 0.989] | 0.990 [0.979, 0.996] | 0.998 [0.993, 1.000] | 0.945 [0.923, 0.963] |
| ae-ours, annotation-and-TokEye agreement, seed 2 | 0.969 [0.949, 0.984] | 0.987 [0.974, 0.995] | 0.998 [0.991, 1.000] | 0.946 [0.927, 0.962] |
| **ae-ours, annotation-and-TokEye agreement, mean over seeds** | 0.975 ± 0.005 | 0.989 ± 0.001 | 0.998 ± 0.000 | 0.943 ± 0.005 |
| clock from legacy annotation (input-free, 120 shots) | 0.787 [0.762, 0.812] | 0.890 [0.868, 0.911] | 0.859 [0.849, 0.915] | 0.780 [0.763, 0.799]† |
| clock from dense relabel (input-free, 120 shots) | 0.842 [0.819, 0.868] | 0.916 [0.893, 0.938] | 1.000 [0.999, 1.000] | 0.764 [0.741, 0.789]† |

**Against the legacy annotation reference** (12000 scorable frames, 2348 positive).

| Method | AUROC | AUPRC | Shot AUROC | F1 |
|---|---|---|---|---|
| ae-ours, legacy annotation, seed 0 | 0.706 [0.644, 0.764] | 0.455 [0.362, 0.549] | 0.764 [0.619, 0.869] | 0.392 [0.333, 0.455] |
| ae-ours, legacy annotation, seed 1 | 0.721 [0.666, 0.772] | 0.482 [0.400, 0.559] | 0.822 [0.689, 0.907] | 0.436 [0.365, 0.500] |
| ae-ours, legacy annotation, seed 2 | 0.742 [0.688, 0.787] | 0.449 [0.353, 0.541] | 0.844 [0.781, 0.900] | 0.411 [0.352, 0.475] |
| **ae-ours, legacy annotation, mean over seeds** | 0.723 ± 0.018 | 0.462 ± 0.017 | 0.810 ± 0.041 | 0.413 ± 0.022 |
| ae-ours, dense relabel, seed 0 | 0.638 [0.568, 0.703] | 0.256 [0.190, 0.342] | 0.582 [0.489, 0.804] | 0.370 [0.306, 0.442] |
| ae-ours, dense relabel, seed 1 | 0.626 [0.559, 0.691] | 0.239 [0.183, 0.319] | 0.565 [0.472, 0.808] | 0.401 [0.341, 0.467] |
| ae-ours, dense relabel, seed 2 | 0.638 [0.573, 0.702] | 0.257 [0.193, 0.341] | 0.649 [0.544, 0.817] | 0.415 [0.356, 0.477] |
| **ae-ours, dense relabel, mean over seeds** | 0.634 ± 0.007 | 0.251 ± 0.010 | 0.599 ± 0.044 | 0.395 ± 0.023 |
| ae-ours, annotation-and-TokEye agreement, seed 0 | 0.639 [0.579, 0.698] | 0.266 [0.200, 0.356] | 0.637 [0.576, 0.802] | 0.404 [0.348, 0.467] |
| ae-ours, annotation-and-TokEye agreement, seed 1 | 0.636 [0.577, 0.696] | 0.275 [0.207, 0.358] | 0.733 [0.579, 0.836] | 0.401 [0.344, 0.464] |
| ae-ours, annotation-and-TokEye agreement, seed 2 | 0.659 [0.600, 0.715] | 0.310 [0.228, 0.403] | 0.735 [0.631, 0.851] | 0.377 [0.324, 0.441] |
| **ae-ours, annotation-and-TokEye agreement, mean over seeds** | 0.645 ± 0.013 | 0.284 ± 0.023 | 0.702 ± 0.056 | 0.394 ± 0.015 |
| clock from legacy annotation (input-free, 120 shots) | 0.852 [0.819, 0.882] | 0.520 [0.431, 0.604] | 0.943 [0.873, 0.958] | 0.527 [0.457, 0.589]† |
| clock from dense relabel (input-free, 120 shots) | 0.855 [0.826, 0.884] | 0.509 [0.431, 0.581] | 0.923 [0.874, 0.942] | 0.548 [0.480, 0.610]† |

Seed-mean paired differences, first minus second, with the interval from resampling shots only (every seed kept in each draw; the seed-to-seed spread is the SD in the table above):

Against the dense reference:

| First minus second | AUROC | AUPRC | F1 |
|---|---|---|---|
| ae-ours-legacy minus ae-ours-dense | -0.128 [-0.158, -0.100] | -0.058 [-0.074, -0.044] | -0.113 [-0.132, -0.095] |
| ae-ours-legacy minus ae-ours-threeway | -0.118 [-0.153, -0.084] | -0.054 [-0.071, -0.037] | -0.095 [-0.115, -0.078] |
| ae-ours-legacy minus clock-legacy-annotation | +0.070 [0.027, 0.107] | +0.045 [0.023, 0.066] | +0.067 [0.032, 0.100] |
| ae-ours-legacy minus clock-dense | +0.014 [-0.031, 0.051] | +0.019 [-0.005, 0.042] | +0.083 [0.039, 0.125] |
| ae-ours-dense minus ae-ours-threeway | +0.009 [-0.003, 0.028] | +0.004 [-0.002, 0.015] | +0.017 [0.007, 0.030] |
| ae-ours-dense minus clock-legacy-annotation | +0.197 [0.169, 0.225] | +0.103 [0.084, 0.123] | +0.180 [0.152, 0.204] |
| ae-ours-dense minus clock-dense | +0.142 [0.115, 0.167] | +0.077 [0.057, 0.099] | +0.196 [0.158, 0.228] |
| ae-ours-threeway minus clock-legacy-annotation | +0.188 [0.151, 0.219] | +0.098 [0.077, 0.120] | +0.162 [0.131, 0.190] |
| ae-ours-threeway minus clock-dense | +0.132 [0.096, 0.164] | +0.072 [0.049, 0.096] | +0.179 [0.139, 0.214] |
| clock-legacy-annotation minus clock-dense | -0.056 [-0.067, -0.046] | -0.026 [-0.030, -0.022] | +0.016 [0.007, 0.024] |

Against the legacy annotation reference:

| First minus second | AUROC | AUPRC | F1 |
|---|---|---|---|
| ae-ours-legacy minus ae-ours-dense | +0.089 [0.046, 0.129] | +0.211 [0.164, 0.249] | +0.017 [-0.008, 0.038] |
| ae-ours-legacy minus ae-ours-threeway | +0.078 [0.036, 0.115] | +0.178 [0.131, 0.221] | +0.019 [-0.010, 0.043] |
| ae-ours-legacy minus clock-legacy-annotation | -0.129 [-0.184, -0.078] | -0.058 [-0.148, 0.026] | -0.114 [-0.186, -0.040] |
| ae-ours-legacy minus clock-dense | -0.132 [-0.188, -0.085] | -0.047 [-0.126, 0.026] | -0.136 [-0.203, -0.069] |
| ae-ours-dense minus ae-ours-threeway | -0.011 [-0.029, 0.009] | -0.033 [-0.053, -0.011] | +0.001 [-0.013, 0.019] |
| ae-ours-dense minus clock-legacy-annotation | -0.218 [-0.286, -0.152] | -0.269 [-0.352, -0.182] | -0.132 [-0.208, -0.052] |
| ae-ours-dense minus clock-dense | -0.221 [-0.286, -0.159] | -0.258 [-0.329, -0.184] | -0.153 [-0.224, -0.082] |
| ae-ours-threeway minus clock-legacy-annotation | -0.207 [-0.270, -0.144] | -0.236 [-0.332, -0.138] | -0.133 [-0.204, -0.060] |
| ae-ours-threeway minus clock-dense | -0.210 [-0.268, -0.152] | -0.225 [-0.309, -0.135] | -0.154 [-0.220, -0.088] |
| clock-legacy-annotation minus clock-dense | -0.003 [-0.017, 0.012] | +0.011 [-0.009, 0.032] | -0.021 [-0.041, -0.002] |

### The 19 shared held-out shots

Shots: 170660, 170661, 170663, 170666, 170669, 170677, 170678, 170718, 170725, 170729, 170730, 170792, 170793, 170798, 170801, 170803, 175241, 175245, 178879.

**Against the dense reference** (3800 scorable frames, 2595 positive).

| Method | AUROC | AUPRC | Shot AUROC | F1 |
|---|---|---|---|---|
| ae-ours, legacy annotation, seed 0 | 0.830 [0.754, 0.901] | 0.926 [0.890, 0.958] | 0.888 [0.737, 0.952] | 0.831 [0.777, 0.876] |
| ae-ours, legacy annotation, seed 1 | 0.764 [0.668, 0.847] | 0.882 [0.846, 0.924] | 0.797 [0.637, 0.867] | 0.790 [0.743, 0.836] |
| ae-ours, legacy annotation, seed 2 | 0.827 [0.752, 0.889] | 0.921 [0.888, 0.954] | 0.846 [0.797, 0.892] | 0.826 [0.769, 0.874] |
| **ae-ours, legacy annotation, mean over seeds** | 0.807 ± 0.038 | 0.910 ± 0.024 | 0.844 ± 0.046 | 0.816 ± 0.023 |
| ae-ours, dense relabel, seed 0 | 0.983 [0.961, 0.996] | 0.992 [0.979, 0.999] | 0.999 [0.994, 1.000] | 0.960 [0.927, 0.983] |
| ae-ours, dense relabel, seed 1 | 0.973 [0.946, 0.993] | 0.989 [0.971, 0.998] | 1.000 [0.993, 1.000] | 0.958 [0.921, 0.982] |
| ae-ours, dense relabel, seed 2 | 0.984 [0.968, 0.995] | 0.993 [0.983, 0.998] | 0.999 [0.991, 1.000] | 0.960 [0.931, 0.981] |
| **ae-ours, dense relabel, mean over seeds** | 0.980 ± 0.006 | 0.991 ± 0.002 | 0.999 ± 0.000 | 0.959 ± 0.001 |
| ae-ours, annotation-and-TokEye agreement, seed 0 | 0.983 [0.966, 0.994] | 0.992 [0.982, 0.998] | 0.998 [0.984, 1.000] | 0.934 [0.892, 0.969] |
| ae-ours, annotation-and-TokEye agreement, seed 1 | 0.983 [0.969, 0.993] | 0.993 [0.982, 0.998] | 0.994 [0.970, 1.000] | 0.943 [0.909, 0.969] |
| ae-ours, annotation-and-TokEye agreement, seed 2 | 0.975 [0.954, 0.989] | 0.990 [0.976, 0.996] | 0.990 [0.978, 1.000] | 0.942 [0.907, 0.968] |
| **ae-ours, annotation-and-TokEye agreement, mean over seeds** | 0.980 ± 0.005 | 0.992 ± 0.002 | 0.994 ± 0.004 | 0.940 ± 0.005 |
| ae-rcn (UCI ±125 ms windows, 801 shots (saved)) | 0.862 [0.810, 0.907] | 0.930 [0.906, 0.955] | 0.980 [0.948, 0.993] | 0.864 [0.805, 0.912] |
| ae-lstm (UCI ±125 ms windows, 801 shots (saved)) | 0.736 [0.654, 0.802] | 0.855 [0.835, 0.885] | 0.769 [0.692, 0.897] | 0.800 [0.763, 0.839] |
| clock from legacy annotation (input-free, 120 shots) | 0.818 [0.773, 0.865] | 0.905 [0.881, 0.931] | 0.859 [0.849, 0.945] | 0.809 [0.775, 0.848]† |
| clock from dense relabel (input-free, 120 shots) | 0.875 [0.834, 0.913] | 0.936 [0.909, 0.960] | 1.000 [0.995, 1.000] | 0.799 [0.753, 0.850]† |

**Against the legacy annotation reference** (3800 scorable frames, 796 positive).

| Method | AUROC | AUPRC | Shot AUROC | F1 |
|---|---|---|---|---|
| ae-ours, legacy annotation, seed 0 | 0.728 [0.610, 0.840] | 0.540 [0.355, 0.707] | 0.747 [0.483, 0.910] | 0.429 [0.314, 0.547] |
| ae-ours, legacy annotation, seed 1 | 0.732 [0.620, 0.844] | 0.556 [0.376, 0.710] | 0.822 [0.620, 0.967] | 0.463 [0.319, 0.590] |
| ae-ours, legacy annotation, seed 2 | 0.781 [0.696, 0.868] | 0.550 [0.383, 0.709] | 0.929 [0.715, 0.965] | 0.451 [0.344, 0.562] |
| **ae-ours, legacy annotation, mean over seeds** | 0.747 ± 0.029 | 0.548 ± 0.008 | 0.833 ± 0.092 | 0.447 ± 0.017 |
| ae-ours, dense relabel, seed 0 | 0.690 [0.548, 0.814] | 0.341 [0.185, 0.570] | 0.735 [0.420, 0.904] | 0.420 [0.290, 0.572] |
| ae-ours, dense relabel, seed 1 | 0.670 [0.537, 0.800] | 0.295 [0.167, 0.514] | 0.561 [0.411, 0.936] | 0.428 [0.315, 0.555] |
| ae-ours, dense relabel, seed 2 | 0.697 [0.570, 0.818] | 0.333 [0.191, 0.558] | 0.763 [0.468, 0.922] | 0.446 [0.340, 0.572] |
| **ae-ours, dense relabel, mean over seeds** | 0.685 ± 0.014 | 0.323 ± 0.024 | 0.686 ± 0.110 | 0.431 ± 0.013 |
| ae-ours, annotation-and-TokEye agreement, seed 0 | 0.700 [0.581, 0.822] | 0.349 [0.191, 0.564] | 0.797 [0.527, 0.941] | 0.444 [0.335, 0.576] |
| ae-ours, annotation-and-TokEye agreement, seed 1 | 0.698 [0.582, 0.816] | 0.355 [0.200, 0.570] | 0.835 [0.536, 0.948] | 0.450 [0.341, 0.586] |
| ae-ours, annotation-and-TokEye agreement, seed 2 | 0.735 [0.626, 0.847] | 0.412 [0.224, 0.625] | 0.844 [0.603, 0.968] | 0.442 [0.328, 0.587] |
| **ae-ours, annotation-and-TokEye agreement, mean over seeds** | 0.711 ± 0.021 | 0.372 ± 0.035 | 0.826 ± 0.025 | 0.445 ± 0.004 |
| ae-rcn (UCI ±125 ms windows, 801 shots (saved)) | 0.911 [0.878, 0.939] | 0.691 [0.554, 0.798] | 0.944 [0.870, 0.976] | 0.571 [0.439, 0.660] |
| ae-lstm (UCI ±125 ms windows, 801 shots (saved)) | 0.898 [0.854, 0.930] | 0.668 [0.509, 0.779] | 0.917 [0.835, 0.959] | 0.602 [0.435, 0.718] |
| clock from legacy annotation (input-free, 120 shots) | 0.860 [0.817, 0.903] | 0.558 [0.413, 0.696] | 0.926 [0.844, 0.960] | 0.545 [0.441, 0.641]† |
| clock from dense relabel (input-free, 120 shots) | 0.877 [0.840, 0.914] | 0.552 [0.422, 0.680] | 0.921 [0.862, 0.955] | 0.580 [0.485, 0.673]† |

Seed-mean paired differences, first minus second, with the interval from resampling shots only (every seed kept in each draw; the seed-to-seed spread is the SD in the table above):

Against the dense reference:

| First minus second | AUROC | AUPRC | F1 |
|---|---|---|---|
| ae-ours-legacy minus ae-ours-dense | -0.173 [-0.248, -0.111] | -0.081 [-0.108, -0.053] | -0.143 [-0.178, -0.111] |
| ae-ours-legacy minus ae-ours-threeway | -0.173 [-0.251, -0.108] | -0.082 [-0.111, -0.051] | -0.124 [-0.163, -0.091] |
| ae-ours-legacy minus ae-rcn | -0.055 [-0.149, 0.036] | -0.020 [-0.062, 0.024] | -0.048 [-0.096, 0.004] |
| ae-ours-legacy minus ae-lstm | +0.070 [0.001, 0.144] | +0.055 [0.017, 0.091] | +0.016 [-0.033, 0.072] |
| ae-ours-legacy minus clock-legacy-annotation | -0.011 [-0.095, 0.068] | +0.005 [-0.034, 0.046] | +0.007 [-0.060, 0.070] |
| ae-ours-legacy minus clock-dense | -0.068 [-0.153, 0.012] | -0.026 [-0.068, 0.020] | +0.017 [-0.062, 0.090] |
| ae-ours-dense minus ae-ours-threeway | -0.000 [-0.014, 0.012] | -0.000 [-0.007, 0.006] | +0.019 [0.005, 0.036] |
| ae-ours-dense minus ae-rcn | +0.118 [0.066, 0.175] | +0.061 [0.037, 0.084] | +0.096 [0.055, 0.146] |
| ae-ours-dense minus ae-lstm | +0.244 [0.170, 0.326] | +0.136 [0.110, 0.155] | +0.159 [0.125, 0.194] |
| ae-ours-dense minus clock-legacy-annotation | +0.162 [0.106, 0.210] | +0.086 [0.062, 0.110] | +0.150 [0.091, 0.197] |
| ae-ours-dense minus clock-dense | +0.105 [0.055, 0.149] | +0.055 [0.029, 0.082] | +0.160 [0.087, 0.220] |
| ae-ours-threeway minus ae-rcn | +0.119 [0.070, 0.175] | +0.062 [0.038, 0.084] | +0.076 [0.036, 0.125] |
| ae-ours-threeway minus ae-lstm | +0.244 [0.172, 0.327] | +0.136 [0.110, 0.155] | +0.140 [0.099, 0.181] |
| ae-ours-threeway minus clock-legacy-annotation | +0.162 [0.104, 0.210] | +0.086 [0.063, 0.110] | +0.131 [0.063, 0.185] |
| ae-ours-threeway minus clock-dense | +0.105 [0.055, 0.147] | +0.056 [0.031, 0.081] | +0.141 [0.059, 0.207] |
| ae-rcn minus ae-lstm | +0.125 [0.058, 0.200] | +0.075 [0.045, 0.096] | +0.063 [0.008, 0.114] |
| ae-rcn minus clock-legacy-annotation | +0.043 [-0.007, 0.091] | +0.025 [0.001, 0.046] | +0.055 [-0.033, 0.128] |
| ae-rcn minus clock-dense | -0.013 [-0.065, 0.033] | -0.006 [-0.031, 0.018] | +0.065 [-0.038, 0.152] |
| ae-lstm minus clock-legacy-annotation | -0.082 [-0.142, -0.039] | -0.050 [-0.068, -0.032] | -0.009 [-0.063, 0.037] |
| ae-lstm minus clock-dense | -0.138 [-0.216, -0.091] | -0.081 [-0.099, -0.060] | +0.001 [-0.066, 0.056] |
| clock-legacy-annotation minus clock-dense | -0.056 [-0.076, -0.045] | -0.031 [-0.039, -0.023] | +0.010 [-0.008, 0.025] |

Against the legacy annotation reference:

| First minus second | AUROC | AUPRC | F1 |
|---|---|---|---|
| ae-ours-legacy minus ae-ours-dense | +0.062 [-0.003, 0.140] | +0.225 [0.116, 0.293] | +0.016 [-0.037, 0.051] |
| ae-ours-legacy minus ae-ours-threeway | +0.036 [-0.038, 0.114] | +0.176 [0.070, 0.275] | +0.002 [-0.057, 0.043] |
| ae-ours-legacy minus ae-rcn | -0.164 [-0.254, -0.074] | -0.142 [-0.259, -0.023] | -0.124 [-0.221, -0.016] |
| ae-ours-legacy minus ae-lstm | -0.151 [-0.245, -0.054] | -0.119 [-0.274, 0.031] | -0.155 [-0.265, -0.029] |
| ae-ours-legacy minus clock-legacy-annotation | -0.113 [-0.214, -0.027] | -0.009 [-0.164, 0.125] | -0.097 [-0.214, 0.004] |
| ae-ours-legacy minus clock-dense | -0.130 [-0.218, -0.052] | -0.003 [-0.132, 0.113] | -0.133 [-0.238, -0.044] |
| ae-ours-dense minus ae-ours-threeway | -0.026 [-0.056, -0.004] | -0.049 [-0.086, -0.000] | -0.014 [-0.034, 0.003] |
| ae-ours-dense minus ae-rcn | -0.226 [-0.351, -0.119] | -0.367 [-0.468, -0.215] | -0.140 [-0.231, -0.038] |
| ae-ours-dense minus ae-lstm | -0.213 [-0.335, -0.097] | -0.345 [-0.475, -0.163] | -0.171 [-0.265, -0.062] |
| ae-ours-dense minus clock-legacy-annotation | -0.175 [-0.301, -0.067] | -0.235 [-0.371, -0.075] | -0.113 [-0.242, 0.014] |
| ae-ours-dense minus clock-dense | -0.192 [-0.309, -0.090] | -0.228 [-0.333, -0.083] | -0.149 [-0.269, -0.031] |
| ae-ours-threeway minus ae-rcn | -0.200 [-0.310, -0.098] | -0.318 [-0.426, -0.180] | -0.126 [-0.212, -0.028] |
| ae-ours-threeway minus ae-lstm | -0.187 [-0.294, -0.075] | -0.296 [-0.436, -0.121] | -0.157 [-0.249, -0.048] |
| ae-ours-threeway minus clock-legacy-annotation | -0.149 [-0.264, -0.046] | -0.186 [-0.343, -0.022] | -0.100 [-0.221, 0.033] |
| ae-ours-threeway minus clock-dense | -0.166 [-0.273, -0.069] | -0.179 [-0.309, -0.037] | -0.135 [-0.247, -0.010] |
| ae-rcn minus ae-lstm | +0.013 [-0.012, 0.039] | +0.023 [-0.064, 0.109] | -0.031 [-0.102, 0.066] |
| ae-rcn minus clock-legacy-annotation | +0.051 [0.020, 0.079] | +0.133 [0.034, 0.211] | +0.027 [-0.081, 0.116] |
| ae-rcn minus clock-dense | +0.034 [0.013, 0.058] | +0.139 [0.053, 0.205] | -0.009 [-0.119, 0.074] |
| ae-lstm minus clock-legacy-annotation | +0.038 [-0.006, 0.079] | +0.110 [-0.022, 0.219] | +0.058 [-0.099, 0.175] |
| ae-lstm minus clock-dense | +0.021 [-0.018, 0.060] | +0.116 [-0.003, 0.210] | +0.022 [-0.123, 0.138] |
| clock-legacy-annotation minus clock-dense | -0.017 [-0.033, 0.005] | +0.006 [-0.022, 0.045] | -0.035 [-0.070, 0.000] |

## V100-only sensitivity

Legacy and dense seed 0 ran on A100, threeway seed 0 on V100. Restricting every arm to its V100 runs of seeds other than 0 (legacy: ae-ours-legacy-seed1, ae-ours-legacy-seed2; dense: ae-ours-dense-seed1, ae-ours-dense-seed2; threeway: ae-ours-threeway-seed1, ae-ours-threeway-seed2) gives two seeds per arm on identical hardware. Seed means, with the shot-only interval:

| Cohort | Reference | Arm | AUROC | AUPRC |
|---|---|---|---|---|
| 60 shots | dense | legacy | 0.842 [0.807, 0.875] | 0.926 [0.905, 0.945] |
| 60 shots | dense | dense | 0.984 [0.976, 0.991] | 0.993 [0.988, 0.997] |
| 60 shots | dense | threeway | 0.973 [0.954, 0.987] | 0.988 [0.977, 0.995] |
| 60 shots | legacy annotation | legacy | 0.731 [0.677, 0.778] | 0.465 [0.378, 0.545] |
| 60 shots | legacy annotation | dense | 0.632 [0.566, 0.698] | 0.248 [0.189, 0.329] |
| 60 shots | legacy annotation | threeway | 0.648 [0.588, 0.706] | 0.292 [0.220, 0.379] |
| 19 shared held-out shots | dense | legacy | 0.795 [0.715, 0.866] | 0.902 [0.869, 0.939] |
| 19 shared held-out shots | dense | dense | 0.979 [0.959, 0.994] | 0.991 [0.977, 0.998] |
| 19 shared held-out shots | dense | threeway | 0.979 [0.962, 0.991] | 0.991 [0.979, 0.997] |
| 19 shared held-out shots | legacy annotation | legacy | 0.757 [0.665, 0.852] | 0.553 [0.382, 0.702] |
| 19 shared held-out shots | legacy annotation | dense | 0.683 [0.554, 0.811] | 0.314 [0.180, 0.534] |
| 19 shared held-out shots | legacy annotation | threeway | 0.716 [0.603, 0.832] | 0.384 [0.212, 0.592] |

Selected paired contrasts on the same hardware (AUROC):

| Cohort | Reference | Contrast | AUROC |
|---|---|---|---|
| 60 shots | dense | ae-ours-dense minus ae-ours-legacy | +0.142 [0.109, 0.179] |
| 60 shots | dense | ae-ours-threeway minus ae-ours-legacy | +0.131 [0.092, 0.169] |
| 60 shots | legacy annotation | ae-ours-dense minus ae-ours-legacy | -0.099 [-0.142, -0.053] |
| 60 shots | legacy annotation | ae-ours-threeway minus ae-ours-legacy | -0.084 [-0.124, -0.036] |
| 19 shared held-out shots | dense | ae-ours-dense minus ae-ours-legacy | +0.183 [0.116, 0.264] |
| 19 shared held-out shots | dense | ae-ours-threeway minus ae-ours-legacy | +0.184 [0.113, 0.268] |
| 19 shared held-out shots | dense | ae-ours-legacy minus ae-rcn | -0.066 [-0.162, 0.032] |
| 19 shared held-out shots | dense | ae-ours-dense minus ae-rcn | +0.117 [0.066, 0.174] |
| 19 shared held-out shots | dense | ae-ours-threeway minus ae-rcn | +0.117 [0.068, 0.174] |
| 19 shared held-out shots | legacy annotation | ae-ours-dense minus ae-ours-legacy | -0.073 [-0.161, -0.005] |
| 19 shared held-out shots | legacy annotation | ae-ours-threeway minus ae-ours-legacy | -0.040 [-0.125, 0.040] |
| 19 shared held-out shots | legacy annotation | ae-ours-legacy minus ae-rcn | -0.155 [-0.238, -0.071] |
| 19 shared held-out shots | legacy annotation | ae-ours-dense minus ae-rcn | -0.228 [-0.349, -0.122] |
| 19 shared held-out shots | legacy annotation | ae-ours-threeway minus ae-rcn | -0.195 [-0.303, -0.093] |

## Interpretation

**Within ae-ours (supervision).** On the 19 shared shots, the AUROC of ae-ours trained on the dense relabel minus ae-ours trained on the legacy annotation +0.173 [0.111, 0.248] against the dense reference and -0.062 [-0.140, 0.003] against the legacy annotation; ae-ours trained on annotation-and-TokEye agreement minus ae-ours trained on the legacy annotation +0.173 [0.108, 0.251] against the dense reference and -0.036 [-0.114, 0.038] against the legacy annotation. On the 60 shots, the AUROC of ae-ours trained on the dense relabel minus ae-ours trained on the legacy annotation +0.128 [0.100, 0.158] against the dense reference and -0.089 [-0.129, -0.046] against the legacy annotation; ae-ours trained on annotation-and-TokEye agreement minus ae-ours trained on the legacy annotation +0.118 [0.084, 0.153] against the dense reference and -0.078 [-0.115, -0.036] against the legacy annotation. Dense relabel minus agreement as the training target, against the dense reference: -0.000 [-0.014, 0.012] on the 19 shots and +0.009 [-0.003, 0.028] on the 60 shots, so dense relabelling and annotation-and-TokEye agreement are not separated by this experiment.

AUROC, seed-mean ae-ours minus the saved detector, 19 shared held-out shots, interval from resampling shots only:

| ae-ours arm | minus ae-rcn, dense | minus ae-rcn, legacy annotation | minus ae-lstm, dense | minus ae-lstm, legacy annotation |
|---|---|---|---|---|
| legacy annotation | -0.055 [-0.149, 0.036] | -0.164 [-0.254, -0.074] | +0.070 [0.001, 0.144] | -0.151 [-0.245, -0.054] |
| dense relabel | +0.118 [0.066, 0.175] | -0.226 [-0.351, -0.119] | +0.244 [0.170, 0.326] | -0.213 [-0.335, -0.097] |
| annotation-and-TokEye agreement | +0.119 [0.070, 0.175] | -0.200 [-0.310, -0.098] | +0.244 [0.172, 0.327] | -0.187 [-0.294, -0.075] |

AUPRC, seed-mean ae-ours minus the saved detector, 19 shared held-out shots, interval from resampling shots only:

| ae-ours arm | minus ae-rcn, dense | minus ae-rcn, legacy annotation | minus ae-lstm, dense | minus ae-lstm, legacy annotation |
|---|---|---|---|---|
| legacy annotation | -0.020 [-0.062, 0.024] | -0.142 [-0.259, -0.023] | +0.055 [0.017, 0.091] | -0.119 [-0.274, 0.031] |
| dense relabel | +0.061 [0.037, 0.084] | -0.367 [-0.468, -0.215] | +0.136 [0.110, 0.155] | -0.345 [-0.475, -0.163] |
| annotation-and-TokEye agreement | +0.062 [0.038, 0.084] | -0.318 [-0.426, -0.180] | +0.136 [0.110, 0.155] | -0.296 [-0.436, -0.121] |

**What the swap shows (AUROC, 19 shared shots).** The model trained on the legacy annotation is not resolved from ae-rcn on the dense reference (-0.055 [-0.149, 0.036]) and trails it on the legacy annotation (-0.164 [-0.254, -0.074]). Its point estimate is below ae-rcn's on both references, resolved on the legacy annotation but with an interval that includes zero on the dense reference; it does not lead ae-rcn there, so there is no reversal against ae-rcn when ae-ours is trained on the legacy annotation. The same model leads ae-lstm on the dense reference (+0.070 [0.001, 0.144]) and trails it on the legacy annotation (-0.151 [-0.245, -0.054]). The reversal against ae-lstm persists: ae-ours trained on the legacy annotation is above it on the dense reference and below it on the legacy annotation. The dense interval's lower end is within 0.005 of zero, so that resolution is marginal. The model trained on the dense relabel leads ae-rcn on the dense reference (+0.118 [0.066, 0.175]) and trails it on the legacy annotation (-0.226 [-0.351, -0.119]). The model trained on annotation-and-TokEye agreement leads ae-rcn on the dense reference (+0.119 [0.070, 0.175]) and trails it on the legacy annotation (-0.200 [-0.310, -0.098]). ae-ours's lead over ae-rcn on the dense reference therefore depends on the target: it is absent when the same recipe trains on the legacy annotation, rather than the best achievable legacy-trained model (Limitations: the recipe was developed for the agreement target, the legacy seeds are unstable, and training windows are fixed), and present when it trains on either of the other two targets. Which of the two carries it is not separated, and the cross-architecture confounds below apply to every statement against the saved detectors.

**Input-free clock.** Each 10 ms frame is scored by its positive rate over the 120 training and selection shots, with no input and no evaluation data. The clock from the legacy annotation peaks at 495 ms (0.78), averages 0.22 over the 0 to 2 s record and 0.00 over its first and last 100 ms; the clock from the dense relabel peaks at 405 ms (0.99), averages 0.43 over the 0 to 2 s record and 0.12 over its first and last 100 ms. On the 19 shared shots its pooled AUROC is 0.860 against the legacy annotation (clock from the legacy annotation) and 0.875 against the dense reference (clock from the dense relabel). For comparison (legacy annotation / dense): ae-rcn 0.911 / 0.862, ae-lstm 0.898 / 0.736, and the ae-ours seed means legacy annotation 0.747 / 0.807, dense relabel 0.685 / 0.980, annotation-and-TokEye agreement 0.711 / 0.980. Within shots the clock's median AUROC is 0.926 against the legacy annotation and 1.000 against the dense reference, against ae-rcn 0.944 / 0.980 and ae-lstm 0.917 / 0.769.

Against the legacy annotation the clock reaches 88% of ae-rcn's pooled AUROC margin over chance and 90% of ae-lstm's. Paired AUROC differences against the legacy annotation: ae-rcn minus the clock from the legacy annotation +0.051 [0.020, 0.079], ae-lstm minus it +0.038 [-0.006, 0.079], and ae-ours trained on the legacy annotation minus it -0.113 [-0.214, -0.027], ae-ours trained on the dense relabel minus it -0.175 [-0.301, -0.067], ae-ours trained on annotation-and-TokEye agreement minus it -0.149 [-0.264, -0.046]. Against the dense reference, ae-ours-dense minus the dense clock is +0.105 [0.055, 0.149] and ae-rcn minus it -0.013 [-0.065, 0.033].

**The dense reference is temporally coarse.** 93 of 180 shots carry a single present span in the table (102 a single run of present frames on the 10 ms grid), and the input-free dense clock reaches a median within-shot AUROC of 0.9997 on the 60 evaluation shots (0.9996 on the 19) against it. Within a shot the reference is therefore predicted almost perfectly by time alone, so the Shot AUROC column cannot rank methods on it, and the pooled AUROC against it mostly measures onset and offset timing between shots.

ae-ours trains on random 182 ms windows (710 columns) and cannot learn absolute time; Garcia's models read the whole 0 to 2 s record and can. Where the legacy annotation concentrates in time, as the profile above shows, part of the older detectors' score against it is time context, not a better reading of the spectrogram, and the clock is the control that measures how much. The fair test of time context against supervision is the deferred LSTM retrain (`ae-lstm-retrained`; Limitations).

**F1.** AUROC and AUPRC carry the claims. F1 is reported with every method calibrated on the reference it is scored on; the clocks' F1 is in-sample (their thresholds are tuned on shots that are part of the clocks' own 120). The earlier protocol left the saved detectors at thresholds set against the legacy annotation, which moves their F1 on the dense reference: ae-rcn on the 19 shots scores 0.864 with its threshold calibrated on dense and 0.350 at its saved annotation-set threshold (`f1_own_target_threshold`).

**Selection.** The reviewers' objection was that the published model was selected on the 60 validation shots that include the 19 benchmark shots. The same recipe retrained with the epoch chosen on 20 separate training shots (the threeway arm, which is the published target) scores AUROC 0.980 on the 19 shots (published 0.982) and 0.975 on the 60 shots (published 0.980), AUPRC 0.992 and 0.989 (published 0.992 and 0.992). Published minus retrained seed mean, against the seed SD of the retrain: AUROC +0.0055 against SD 0.0046 on 60 shots; AUPRC +0.0032 against SD 0.0014 on 60 shots; AUROC +0.0017 against SD 0.0047 on 19 shots; AUPRC +0.0005 against SD 0.0017 on 19 shots. The gap exceeds the seed SD for AUROC on 60 shots and AUPRC on 60 shots, so the retrained scores are lower than the published ones by more than the seed spread there. The comparison also changes the training data: the retrain uses 100 training shots, the published model 120. Selection on the validation block and the 20 extra training shots together therefore account for at most about 0.005 AUROC and 0.003 AUPRC on the 60 shots (the published scores are rounded to three decimals, and were scored under the original bfloat16 inference; float32 moves the retrain's seed means by at most 0.0015 AUROC and 0.0034 AUPRC). The recipe itself (symmetric cross entropy over binary cross entropy) was chosen on the validation block (the model README), so only the epoch choice is clean.

## Cross-architecture confounds

The swap holds the architecture fixed and varies supervision. The comparison with ae-rcn and ae-lstm still differs in more than supervision, and every statement against them carries these:

1. **Time context.** The saved detectors read the whole 0 to 2 s record; ae-ours sees 182 ms windows. The clock above measures how much that is worth against each reference.
2. **Data volume.** ae-rcn and ae-lstm trained on 801 shots; each ae-ours arm trains on 100.
3. **Campaign overlap.** 41 of the 60 evaluation shots are in the saved detectors' training set (which is why they can be scored on 19 only), and each of the 19 held-out shots has a Garcia training shot 1 to 2 shot numbers away, a same-day neighbour. The evaluation shots sit in run blocks 1706xx: 16, 1707xx: 23, 1708xx: 11, 1752xx: 8, 1788xx: 2; ae-ours's 100 training shots in those blocks number 1706xx: 0, 1707xx: 0, 1708xx: 2, 1752xx: 0, 1788xx: 0.
4. **Input band.** ae-ours reads 80.6 to 250.0 kHz; the saved detectors read 20 to 250 kHz. Annotated columns that are BAE only are 9.7% of annotated columns on the 60 evaluation shots (9.3% on the 19) against 1.5% in ae-ours's training shots. In the evaluation shots 60% of those columns (45% on the 19) carry in-band TokEye activity, against 93% of other annotated columns; in training the shares are 82% and 86%.
5. **Calibration sets.** F1 thresholds come from 20 selection shots for ae-ours and the clocks and from six for the saved detectors.

## Limitations

- **Dense labels' provenance and reviewers.** The history file holds 639 entries on 180 shots. Its `reviewer` field is the login of the review server's process (nc1514), not a person; the person is the `name` field: (unnamed) 208 entries on 180 shots; Alvin Garcia 210 entries on 180 shots; Nathaniel Chen 221 entries on 180 shots (the unnamed saves run 2026-09-23 to 2026-09-29, before the first named save on 2026-09-29). The confirmation note reads "Review confirmation recorded at user request: all AE examples checked by Alvin Garcia and Nathaniel Chen." (360 entries). The source of every entry is `alfven_eigenmode_format_2026_v1.csv`: the review was pre-filled from the annotation's source table. A. Garcia, the author of ae-rcn and ae-lstm, therefore helped make the dense labels the training arms learn from: he saved all 180 shots, and the 29 saves that changed intervals (on 2026-10-01) cover 22 training, 6 selection and 0 evaluation shots. A. Garcia saved changes on 28 of the 29 training and selection shots that differ from the paper's snapshot (22 training, 6 selection) and on 0 evaluation shots; N. Chen on 6 of them and 0 evaluation shots; unnamed saves on 1 of them and 1 evaluation shot (a shot can carry several). The current table differs from the snapshot the paper's audit scored on 29 of the 120 training and selection shots and on none of the 19 shared shots. Those changes were made with the 60-250 kHz review display (80-250 kHz before 2026-09-30), while ae-ours reads 80.6 kHz and above. The arms were trained on the current table, edits included; no model was retrained without them. Whether a TokEye layer was on screen while reviewing is an open question, and it bears on why dense and threeway supervision score alike.
- **The legacy arm is not the best achievable legacy-trained model.** The recipe (the symmetric-cross-entropy weights, the learning rate and the early stopping) was developed for the agreement target and reused unchanged for the legacy and dense targets, and the legacy arm is the unstable one: the pooled AUROC of its three accepted seeds ranges 0.805 to 0.885 (60 shots, dense); 0.706 to 0.742 (60 shots, legacy annotation); 0.764 to 0.830 (19 shared held-out shots, dense); 0.728 to 0.781 (19 shared held-out shots, legacy annotation); the first attempt of seed 2 stopped inside the constant-output plateau and was rerun under the declared rule. A comparison with ae-rcn that rests on the legacy arm therefore says what this recipe does on the legacy annotation, not what a model tuned for it would do.
- **Fixed training windows.** Every epoch re-draws the same 800 windows (eight per shot; persistent DataLoader workers never see `set_epoch()`); the shot order alone changes. This affects every arm and seed equally, and the published recipe too, and it is not fixed in this round, because it would mean rerunning every arm. Its effect on the arms' order is not measured.
- **Three seeds** give limited precision for training variability. The headline intervals resample shots only, and the seed SD is reported beside them; the interval that also resamples the seed IDs is in evaluation.json as `ci95` and is not used here, because with three seeds it mostly reflects the worst seed.
- **Hardware.** A100 and V100S runs mix; see the V100-only sensitivity.
- **Precision.** Epochs were selected during training under bfloat16 autocast; only the reported scores, thresholds and intervals are float32. The largest change of a run's pooled AUROC is 0.0103 (Inference precision).
- **Dense reference.** It is temporally coarse, so within-shot AUROC cannot rank methods on it (Interpretation); and the 10 ms frame grid is the audit's, with 372 of 7,820 columns in a neighbouring frame (Frozen protocol).
- **Clock F1 is in-sample.** The clocks' thresholds are tuned on 20 shots that are part of the 120 shots the clocks are built from; their AUROC and AUPRC are not affected.
- **The LSTM retrain was deferred.** The brief's optional item (`ae-lstm-retrained`: the published 3 x 64 LSTM architecture trained on the same 100/20 split, on whole records, with dense and with legacy supervision) was not run. It remains the control that separates time context from supervision: it gives the older architecture the same supervision treatment while keeping its time context, and the clock rows bound what time context alone achieves.
- The saved detectors are scored on 19 shots and calibrated on six; their intervals are wide.

## Excluded and superseded records

These records are not in any mean, SD or interval above. They are scored with the same code, thresholds calibrated on the selection shots, so the effect of the convergence rule can be read.

| Record | Epochs run | Selected epoch | Cohort | Dense AUROC | Legacy-annotation AUROC | Selection within-shot SD | Selection AUROC |
|---|---:|---:|---|---|---|---:|---:|
| ae-ours-dense-seed1-superseded1 | 12 | 6 | 60 shots | 0.973 [0.959, 0.984] | 0.618 [0.550, 0.687] | 0.374 | 0.920 |
| ae-ours-dense-seed1-superseded1 | 12 | 6 | 19 shared held-out shots | 0.979 [0.958, 0.993] | 0.663 [0.523, 0.789] | 0.374 | 0.920 |
| ae-ours-legacy-seed2-superseded1 | 8 | 2 | 60 shots | 0.445 [0.389, 0.506] | 0.466 [0.413, 0.514] | 0.001 | 0.460 |
| ae-ours-legacy-seed2-superseded1 | 8 | 2 | 19 shared held-out shots | 0.447 [0.355, 0.549] | 0.511 [0.423, 0.594] | 0.001 | 0.460 |

## Reproduction

From the worktree, with the scratch TMPDIR, `LABELER_ROOT`, `LABELER_LABEL_TABLES`, `LABELER_NO_FETCH=1` and `PYTHONPATH=$PWD/src`:

```bash
# training: GPU, CUDA interpreter; sbatch (AESWAP_RUNS="legacy:2 dense:1" for reruns) or one head-node GPU per process
sbatch scripts/labeler/ae_supervision_swap.sbatch
bash scripts/labeler/ae_supervision_swap_head.sh 0 legacy:2
# float32 re-inference of every finished run (GPU, CUDA interpreter)
python scripts/labeler/ae_supervision_swap.py infer-fp32
# CPU, pixi labelmaker environment
python scripts/labeler/ae_supervision_swap.py verify
python scripts/labeler/ae_supervision_swap.py audit-convergence
python scripts/labeler/ae_supervision_swap.py evaluate
```

`audit-convergence` applies the declared rule to every record, archives any record that stopped inside the plateau, and lists the runs still to train. Completed runs are never overwritten. The launchers use a short TMPDIR (`$LABELER_ROOT/scratch/ae-sw`) because DataLoader workers add `/pymp-*/listener-*` to it and a socket path must stay under 108 bytes.
