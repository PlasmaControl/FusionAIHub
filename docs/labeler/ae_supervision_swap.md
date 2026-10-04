# AE supervision swap

Does the ranking reversal between ae-ours and the older CO2 detectors come from training supervision? The same AeSeldNet recipe is trained on three activity targets (the legacy annotation, the dense relabel, annotation-and-TokEye agreement; three seeds each) with epochs chosen on 20 held-out training shots, then scored with ae-rcn, ae-lstm and an input-free clock against both references on the 60 validation shots and on the 19 shots the older detectors did not train on.

Source of every number below: [evaluation.json](../../outputs/labeler/ae/supervision_swap/evaluation.json) (written by `scripts/labeler/ae_supervision_swap.py evaluate`; run records, execution IDs, thresholds, per-seed scores, paired differences, the clock priors and the convergence audit are embedded). [manifest.json](../../outputs/labeler/ae/supervision_swap/manifest.json) holds the shot lists and input hashes; [verification.json](../../outputs/labeler/ae/supervision_swap/verification.json) checks target equality and split isolation. The paper table is `table_supervision_swap.tex` in the same directory.

## Summary

On the 19 shared held-out shots the seed-mean AUROC against the dense reference is legacy 0.801, dense 0.981, threeway 0.981 for the legacy, dense and threeway supervision arms, and against the annotation legacy 0.748, dense 0.687, threeway 0.713. Legacy-supervised ae-ours is below ae-rcn on both references in point estimate (resolved on the annotation only), so the reversal against ae-rcn needs dense or agreement supervision; the reversal against ae-lstm persists in sign but is not resolved on the dense reference. An input-free clock reaches 88% of ae-rcn's AUROC margin over chance against the annotation, which makes time context a confound of every comparison with the saved detectors. The Interpretation and Cross-architecture confounds sections give the numbers and the limits.

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
| ae-ours-dense-seed1-superseded1 (first attempt) | 11 | 6 | stopped inside the plateau | 0.373 | 0.921 | superseded: rerun with the same seed |
| ae-ours-legacy-seed2-superseded1 (first attempt) | 7 | 2 | stopped inside the plateau | 0.001 | 0.467 | superseded: rerun with the same seed |

Of the nine first-generation records, 7 conformed to rule 1 as they stood; 2 stopped inside the plateau (ae-ours-dense-seed1-superseded1, ae-ours-legacy-seed2-superseded1) and were rerun with the same seed. After rerunning, 9 of the 9 final records pass rule 2 and none needed replacing. Selection-only outputs were used for every decision above; no evaluation frame entered the rule. The first-attempt records are kept on disk and scored in the last section, so the effect of the rule is visible.

## Frozen protocol

The dense snapshot (`review/labels.csv`, SHA256 `b4308ff52d766779f33a92e9441a916bb07c854dcc1a98a2f61109bb6431e93a`) has 877 rows over 180 shots: 371 present and 506 absent rows. Every present row is a merged crowd span (`iscrowd`), so they are not individual boxes. Its history file holds 639 entries, whose latest interval list per shot totals 371 intervals and whose entries together total 1460. The paper's "943 intervals when the benchmark was scored, 954 after later review" matches none of these counts, so it counts another unit or an earlier snapshot; this document uses the file as it stands and takes no number from the paper's count. Prevalence of present time: 0.424 by duration and 0.432 in any-touch 10 ms frames on the 120 training and selection shots, against 0.41 in the paper (not reconciled: the paper's figure may predate later review edits), and 0.691 and 0.696 on the 60 validation shots, against 0.70 in the paper (the frame count matches). Source: `dense_counts`, `dense_history` and `dense_prevalence` in evaluation.json.

NumPy split seed 20261003 freezes 100 training, 20 selection and 60 evaluation shots; none is in the fixed catalog's blind test split. Seeds 0, 1 and 2 change initialisation and the sampled windows; every arm uses the same split. The original one-seed manifest was archived before the seed list was extended.

Only the activity target changes between arms. Legacy is the audit's at-least-half-annotated rule at 10 ms (classes 1 to 4, LFM excluded), expanded to the native columns; dense uses the catalog any-touch state rule and masks unknown states; threeway keeps the native annotation-and-TokEye agreement, needs at least half the native columns of a frame to carry agreement weight, and masks disagreement. The annotation-and-TokEye frequency target and its weights are identical in every arm (verification.json). Each input covers 0 to 2 s as 7,820 native columns.

**Band.** The model reads the CO2 spectrogram from 80.57 to 250.00 kHz (348 bins, four chords), and the dense review display covered 80 to 250 kHz. "Absent" in the dense labels, and in every target ae-ours trains on, means absent within that observed band; it says nothing about activity below 80.6 kHz, which the annotation can mark and which can therefore be labelled absent. The saved UCI detectors read 20 to 250 kHz.

**Recipe.** The original AeSeldNet recipe (four channels, 348 frequency bins, pools (6, 2, 29), two bidirectional GRUs, SCE plus the unchanged frequency objective, 710-column windows (182 ms), eight windows per shot, batch 16, AdamW 1e-4, weight decay 1e-4, cosine decay to 1e-6) with the convergence rule above; CUDA allocations are capped at 10 GiB. The persistent DataLoader workers keep their epoch-0 copy of the dataset, so `set_epoch()` never reaches them: every epoch re-draws the same eight windows per shot (800 windows), and only the shot order is reshuffled. This holds for every arm and seed, including the published model's recipe; changing it would require rerunning everything (checked in isolation with and without persistent workers).

**Thresholds.** The epoch minimises the combined loss on the 20 selection shots. The headline F1 calibrates every method on the reference it is scored against: the threshold maximises 10 ms F1 on that method's selection shots (20; six for ae-rcn and ae-lstm, the only ones they did not train on; the clocks use the 20) with ties going to the highest threshold. F1 at each record's own-target threshold (the earlier protocol: each arm's threshold from its own activity target, the older detectors' from the annotation) is kept in evaluation.json (`f1_own_target_threshold`) and is used in no claim. No evaluation frame selects an epoch or a threshold.

**Saved detectors.** Garcia probabilities are the saved spectrogram predictions: maximum over the first four AE classes, mean over four chords, nearest output-bin centre on the 10 ms grid. Only 19 evaluation shots and six selection shots have predictions for shots the detectors did not train on.

## Completed runs

Epochs are zero-based. The patience column is the epoch from which non-improving epochs count: records trained after the rule was declared use 10, earlier records counted from 0 and the audit above shows that the rule gives them the same stopping epoch.

| Supervision | Seed | Selected epoch | Epochs run | Patience from | Own-target threshold | Execution | GPU |
|---|---:|---:|---:|---:|---:|---|---|
| dense | 0 | 19 | 25 | 0 | 0.2030 | 2952146_1 (job 2952150) | NVIDIA A100-PCIE-40GB |
| dense | 1 | 15 | 21 | 10 | 0.0960 | head node | Tesla V100S-PCIE-32GB |
| dense | 2 | 13 | 19 | 0 | 0.0582 | head node | Tesla V100S-PCIE-32GB |
| legacy | 0 | 17 | 23 | 0 | 0.0235 | 2952146_0 (job 2952149) | NVIDIA A100-PCIE-40GB |
| legacy | 1 | 17 | 23 | 0 | 0.1900 | head node | Tesla V100S-PCIE-32GB |
| legacy | 2 | 27 | 30 | 10 | 0.3216 | head node | Tesla V100S-PCIE-32GB |
| threeway | 0 | 13 | 19 | 0 | 0.3621 | head node | Tesla V100S-PCIE-32GB |
| threeway | 1 | 9 | 15 | 0 | 0.3860 | head node | Tesla V100S-PCIE-32GB |
| threeway | 2 | 26 | 30 | 0 | 0.4773 | head node | Tesla V100S-PCIE-32GB |

ae-ours-dense-seed0, ae-ours-legacy-seed0 ran on A100 GPUs; every other record ran on the head node's V100S. This Torch build reports V100 bfloat16 support through emulation, so the unchanged trainer autocasts to bfloat16 on both (gpu_probe.json). Hardware is therefore a nuisance variable of the supervision comparison; the V100-only sensitivity analysis below restricts every arm to its V100 runs of seeds other than 0.

## Scores

AUROC and AUPRC are the primary metrics: they need no threshold. Per-seed rows and baselines give the pooled-frame estimate with a 95% interval from 1,000 shot-bootstrap replicates (seed 20261004). Mean rows give the mean and sample SD over seeds of the per-seed pooled values. Shot AUROC is the median over shots with both classes of the within-shot AUROC, with a shot-bootstrap interval; it removes differences in calibration between shots. F1 is calibrated on the reference being scored for every method (above).

### All 60 evaluation shots

Shots: 170659, 170660, 170661, 170662, 170663, 170664, 170665, 170666, 170667, 170669, 170671, 170672, 170675, 170677, 170678, 170679, 170714, 170715, 170716, 170717, 170718, 170719, 170720, 170721, 170722, 170724, 170725, 170727, 170729, 170730, 170790, 170791, 170792, 170793, 170794, 170795, 170797, 170798, 170799, 170801, 170803, 170805, 170806, 170807, 170808, 170809, 170810, 170811, 170813, 170814, 175239, 175240, 175241, 175244, 175245, 175250, 175251, 175252, 178872, 178879.

**Against the dense reference** (12000 scorable frames, 8352 positive).

| Method | AUROC | AUPRC | Shot AUROC | F1 |
|---|---|---|---|---|
| ae-ours legacy, seed 0 | 0.882 [0.853, 0.911] | 0.952 [0.934, 0.967] | 0.922 [0.897, 0.949] | 0.864 [0.835, 0.889] |
| ae-ours legacy, seed 1 | 0.795 [0.750, 0.837] | 0.901 [0.876, 0.923] | 0.786 [0.692, 0.846] | 0.826 [0.799, 0.849] |
| ae-ours legacy, seed 2 | 0.875 [0.843, 0.905] | 0.947 [0.928, 0.963] | 0.867 [0.831, 0.940] | 0.842 [0.805, 0.873] |
| **ae-ours legacy, mean over seeds** | 0.851 ± 0.049 | 0.933 ± 0.028 | 0.859 ± 0.069 | 0.844 ± 0.019 |
| ae-ours dense, seed 0 | 0.984 [0.976, 0.991] | 0.993 [0.987, 0.997] | 1.000 [0.996, 1.000] | 0.960 [0.941, 0.975] |
| ae-ours dense, seed 1 | 0.980 [0.969, 0.989] | 0.992 [0.985, 0.996] | 0.999 [0.997, 0.999] | 0.958 [0.939, 0.972] |
| ae-ours dense, seed 2 | 0.989 [0.982, 0.994] | 0.995 [0.991, 0.998] | 0.999 [0.997, 1.000] | 0.964 [0.950, 0.976] |
| **ae-ours dense, mean over seeds** | 0.984 ± 0.004 | 0.993 ± 0.002 | 0.999 ± 0.000 | 0.961 ± 0.003 |
| ae-ours threeway, seed 0 | 0.978 [0.958, 0.990] | 0.990 [0.977, 0.996] | 0.998 [0.994, 1.000] | 0.941 [0.919, 0.961] |
| ae-ours threeway, seed 1 | 0.977 [0.959, 0.989] | 0.990 [0.979, 0.996] | 0.998 [0.993, 1.000] | 0.945 [0.924, 0.963] |
| ae-ours threeway, seed 2 | 0.970 [0.949, 0.984] | 0.987 [0.974, 0.995] | 0.998 [0.991, 1.000] | 0.946 [0.927, 0.962] |
| **ae-ours threeway, mean over seeds** | 0.975 ± 0.004 | 0.989 ± 0.001 | 0.998 ± 0.000 | 0.944 ± 0.003 |
| clock (from annotation) (input-free (120 shots)) | 0.787 [0.762, 0.812] | 0.890 [0.868, 0.911] | 0.859 [0.849, 0.915] | 0.780 [0.763, 0.799] |
| clock (from dense labels) (input-free (120 shots)) | 0.842 [0.819, 0.868] | 0.916 [0.893, 0.938] | 1.000 [0.999, 1.000] | 0.764 [0.741, 0.789] |

**Against the annotation reference** (12000 scorable frames, 2348 positive).

| Method | AUROC | AUPRC | Shot AUROC | F1 |
|---|---|---|---|---|
| ae-ours legacy, seed 0 | 0.706 [0.643, 0.763] | 0.455 [0.363, 0.549] | 0.766 [0.619, 0.869] | 0.387 [0.329, 0.450] |
| ae-ours legacy, seed 1 | 0.719 [0.663, 0.770] | 0.480 [0.398, 0.557] | 0.819 [0.683, 0.910] | 0.433 [0.362, 0.498] |
| ae-ours legacy, seed 2 | 0.744 [0.689, 0.790] | 0.464 [0.366, 0.555] | 0.836 [0.779, 0.902] | 0.411 [0.351, 0.476] |
| **ae-ours legacy, mean over seeds** | 0.723 ± 0.020 | 0.466 ± 0.013 | 0.807 ± 0.036 | 0.410 ± 0.023 |
| ae-ours dense, seed 0 | 0.638 [0.567, 0.706] | 0.257 [0.192, 0.345] | 0.571 [0.489, 0.797] | 0.372 [0.308, 0.446] |
| ae-ours dense, seed 1 | 0.627 [0.559, 0.693] | 0.247 [0.187, 0.327] | 0.565 [0.482, 0.817] | 0.400 [0.340, 0.467] |
| ae-ours dense, seed 2 | 0.639 [0.572, 0.703] | 0.260 [0.194, 0.345] | 0.655 [0.558, 0.814] | 0.412 [0.354, 0.475] |
| **ae-ours dense, mean over seeds** | 0.635 ± 0.007 | 0.255 ± 0.007 | 0.597 ± 0.050 | 0.395 ± 0.020 |
| ae-ours threeway, seed 0 | 0.640 [0.580, 0.698] | 0.273 [0.203, 0.364] | 0.641 [0.577, 0.799] | 0.404 [0.348, 0.467] |
| ae-ours threeway, seed 1 | 0.637 [0.578, 0.697] | 0.276 [0.209, 0.358] | 0.728 [0.582, 0.834] | 0.401 [0.344, 0.464] |
| ae-ours threeway, seed 2 | 0.662 [0.604, 0.717] | 0.313 [0.233, 0.405] | 0.730 [0.635, 0.850] | 0.377 [0.323, 0.441] |
| **ae-ours threeway, mean over seeds** | 0.647 ± 0.014 | 0.287 ± 0.022 | 0.700 ± 0.051 | 0.394 ± 0.015 |
| clock (from annotation) (input-free (120 shots)) | 0.852 [0.819, 0.882] | 0.520 [0.431, 0.604] | 0.943 [0.873, 0.958] | 0.527 [0.457, 0.589] |
| clock (from dense labels) (input-free (120 shots)) | 0.855 [0.826, 0.884] | 0.509 [0.431, 0.581] | 0.923 [0.874, 0.942] | 0.548 [0.480, 0.610] |

Seed-mean paired differences, first minus second, with the interval from resampling shots only (every seed kept in each draw; the seed-to-seed spread is the SD in the table above):

Against the dense reference:

| First minus second | AUROC | AUPRC | F1 |
|---|---|---|---|
| ae-ours-dense minus ae-ours-threeway | +0.010 [-0.003, 0.028] | +0.004 [-0.002, 0.015] | +0.016 [0.006, 0.030] |
| ae-ours-dense minus clock-annotation | +0.197 [0.170, 0.225] | +0.103 [0.084, 0.123] | +0.181 [0.153, 0.204] |
| ae-ours-dense minus clock-dense | +0.142 [0.116, 0.167] | +0.077 [0.057, 0.099] | +0.197 [0.160, 0.228] |
| ae-ours-legacy minus ae-ours-dense | -0.133 [-0.165, -0.104] | -0.060 [-0.076, -0.046] | -0.117 [-0.136, -0.100] |
| ae-ours-legacy minus ae-ours-threeway | -0.124 [-0.159, -0.089] | -0.056 [-0.072, -0.039] | -0.101 [-0.121, -0.084] |
| ae-ours-legacy minus clock-annotation | +0.064 [0.021, 0.102] | +0.043 [0.021, 0.064] | +0.064 [0.029, 0.097] |
| ae-ours-legacy minus clock-dense | +0.008 [-0.037, 0.046] | +0.017 [-0.007, 0.040] | +0.080 [0.036, 0.121] |
| ae-ours-threeway minus clock-annotation | +0.188 [0.151, 0.219] | +0.098 [0.077, 0.120] | +0.164 [0.133, 0.191] |
| ae-ours-threeway minus clock-dense | +0.132 [0.096, 0.164] | +0.072 [0.049, 0.096] | +0.180 [0.141, 0.215] |
| clock-annotation minus clock-dense | -0.056 [-0.067, -0.046] | -0.026 [-0.030, -0.022] | +0.016 [0.007, 0.024] |

Against the annotation reference:

| First minus second | AUROC | AUPRC | F1 |
|---|---|---|---|
| ae-ours-dense minus ae-ours-threeway | -0.012 [-0.029, 0.007] | -0.033 [-0.050, -0.012] | +0.001 [-0.014, 0.019] |
| ae-ours-dense minus clock-annotation | -0.217 [-0.286, -0.150] | -0.265 [-0.352, -0.176] | -0.132 [-0.209, -0.053] |
| ae-ours-dense minus clock-dense | -0.220 [-0.285, -0.158] | -0.254 [-0.329, -0.177] | -0.153 [-0.226, -0.083] |
| ae-ours-legacy minus ae-ours-dense | +0.088 [0.045, 0.128] | +0.212 [0.164, 0.250] | +0.015 [-0.009, 0.035] |
| ae-ours-legacy minus ae-ours-threeway | +0.076 [0.034, 0.113] | +0.179 [0.132, 0.221] | +0.016 [-0.012, 0.040] |
| ae-ours-legacy minus clock-annotation | -0.129 [-0.185, -0.078] | -0.053 [-0.146, 0.030] | -0.117 [-0.188, -0.042] |
| ae-ours-legacy minus clock-dense | -0.132 [-0.188, -0.085] | -0.043 [-0.122, 0.030] | -0.138 [-0.206, -0.073] |
| ae-ours-threeway minus clock-annotation | -0.206 [-0.268, -0.143] | -0.232 [-0.329, -0.133] | -0.133 [-0.204, -0.060] |
| ae-ours-threeway minus clock-dense | -0.209 [-0.267, -0.151] | -0.221 [-0.307, -0.131] | -0.154 [-0.219, -0.087] |
| clock-annotation minus clock-dense | -0.003 [-0.017, 0.012] | +0.011 [-0.009, 0.032] | -0.021 [-0.041, -0.002] |

### The 19 shared held-out shots

Shots: 170660, 170661, 170663, 170666, 170669, 170677, 170678, 170718, 170725, 170729, 170730, 170792, 170793, 170798, 170801, 170803, 175241, 175245, 178879.

**Against the dense reference** (3800 scorable frames, 2595 positive).

| Method | AUROC | AUPRC | Shot AUROC | F1 |
|---|---|---|---|---|
| ae-ours legacy, seed 0 | 0.827 [0.748, 0.897] | 0.925 [0.889, 0.958] | 0.880 [0.725, 0.949] | 0.825 [0.772, 0.871] |
| ae-ours legacy, seed 1 | 0.754 [0.654, 0.839] | 0.879 [0.843, 0.920] | 0.764 [0.589, 0.827] | 0.787 [0.741, 0.832] |
| ae-ours legacy, seed 2 | 0.822 [0.741, 0.887] | 0.920 [0.886, 0.953] | 0.846 [0.771, 0.893] | 0.826 [0.767, 0.877] |
| **ae-ours legacy, mean over seeds** | 0.801 ± 0.041 | 0.908 ± 0.026 | 0.830 ± 0.060 | 0.813 ± 0.022 |
| ae-ours dense, seed 0 | 0.983 [0.962, 0.996] | 0.992 [0.979, 0.999] | 0.999 [0.994, 1.000] | 0.960 [0.927, 0.983] |
| ae-ours dense, seed 1 | 0.974 [0.948, 0.994] | 0.989 [0.972, 0.998] | 0.999 [0.993, 1.000] | 0.959 [0.922, 0.983] |
| ae-ours dense, seed 2 | 0.984 [0.968, 0.995] | 0.993 [0.983, 0.998] | 0.999 [0.991, 1.000] | 0.959 [0.930, 0.981] |
| **ae-ours dense, mean over seeds** | 0.981 ± 0.005 | 0.992 ± 0.002 | 0.999 ± 0.000 | 0.959 ± 0.001 |
| ae-ours threeway, seed 0 | 0.983 [0.966, 0.994] | 0.992 [0.982, 0.998] | 0.998 [0.983, 1.000] | 0.938 [0.900, 0.969] |
| ae-ours threeway, seed 1 | 0.983 [0.969, 0.993] | 0.993 [0.982, 0.998] | 0.994 [0.971, 1.000] | 0.944 [0.912, 0.970] |
| ae-ours threeway, seed 2 | 0.976 [0.956, 0.990] | 0.990 [0.977, 0.996] | 0.991 [0.975, 1.000] | 0.942 [0.907, 0.968] |
| **ae-ours threeway, mean over seeds** | 0.981 ± 0.004 | 0.992 ± 0.001 | 0.994 ± 0.004 | 0.941 ± 0.003 |
| ae-rcn (UCI ±125 ms windows, 801 shots (saved)) | 0.862 [0.810, 0.907] | 0.930 [0.906, 0.955] | 0.980 [0.948, 0.993] | 0.864 [0.805, 0.912] |
| ae-lstm (UCI ±125 ms windows, 801 shots (saved)) | 0.736 [0.654, 0.802] | 0.855 [0.835, 0.885] | 0.769 [0.692, 0.897] | 0.800 [0.763, 0.839] |
| clock (from annotation) (input-free (120 shots)) | 0.818 [0.773, 0.865] | 0.905 [0.881, 0.931] | 0.859 [0.849, 0.945] | 0.809 [0.775, 0.848] |
| clock (from dense labels) (input-free (120 shots)) | 0.875 [0.834, 0.913] | 0.936 [0.909, 0.960] | 1.000 [0.995, 1.000] | 0.799 [0.753, 0.850] |

**Against the annotation reference** (3800 scorable frames, 796 positive).

| Method | AUROC | AUPRC | Shot AUROC | F1 |
|---|---|---|---|---|
| ae-ours legacy, seed 0 | 0.728 [0.608, 0.840] | 0.540 [0.356, 0.706] | 0.748 [0.481, 0.910] | 0.425 [0.313, 0.543] |
| ae-ours legacy, seed 1 | 0.730 [0.617, 0.844] | 0.556 [0.373, 0.712] | 0.817 [0.632, 0.960] | 0.462 [0.320, 0.590] |
| ae-ours legacy, seed 2 | 0.785 [0.699, 0.871] | 0.570 [0.396, 0.724] | 0.941 [0.707, 0.967] | 0.453 [0.348, 0.562] |
| **ae-ours legacy, mean over seeds** | 0.748 ± 0.032 | 0.555 ± 0.015 | 0.835 ± 0.098 | 0.447 ± 0.019 |
| ae-ours dense, seed 0 | 0.692 [0.549, 0.818] | 0.348 [0.181, 0.581] | 0.741 [0.391, 0.919] | 0.420 [0.290, 0.570] |
| ae-ours dense, seed 1 | 0.672 [0.537, 0.802] | 0.308 [0.170, 0.527] | 0.564 [0.395, 0.943] | 0.427 [0.313, 0.555] |
| ae-ours dense, seed 2 | 0.698 [0.569, 0.819] | 0.340 [0.190, 0.571] | 0.764 [0.459, 0.924] | 0.442 [0.337, 0.566] |
| **ae-ours dense, mean over seeds** | 0.687 ± 0.014 | 0.332 ± 0.021 | 0.690 ± 0.110 | 0.430 ± 0.011 |
| ae-ours threeway, seed 0 | 0.702 [0.583, 0.823] | 0.350 [0.192, 0.562] | 0.798 [0.529, 0.938] | 0.445 [0.336, 0.577] |
| ae-ours threeway, seed 1 | 0.699 [0.582, 0.818] | 0.356 [0.203, 0.570] | 0.834 [0.533, 0.941] | 0.450 [0.341, 0.586] |
| ae-ours threeway, seed 2 | 0.737 [0.629, 0.847] | 0.416 [0.228, 0.621] | 0.841 [0.606, 0.966] | 0.443 [0.328, 0.588] |
| **ae-ours threeway, mean over seeds** | 0.713 ± 0.021 | 0.374 ± 0.037 | 0.824 ± 0.023 | 0.446 ± 0.003 |
| ae-rcn (UCI ±125 ms windows, 801 shots (saved)) | 0.911 [0.878, 0.939] | 0.691 [0.554, 0.798] | 0.944 [0.870, 0.976] | 0.571 [0.439, 0.660] |
| ae-lstm (UCI ±125 ms windows, 801 shots (saved)) | 0.898 [0.854, 0.930] | 0.668 [0.509, 0.779] | 0.917 [0.835, 0.959] | 0.602 [0.435, 0.718] |
| clock (from annotation) (input-free (120 shots)) | 0.860 [0.817, 0.903] | 0.558 [0.413, 0.696] | 0.926 [0.844, 0.960] | 0.545 [0.441, 0.641] |
| clock (from dense labels) (input-free (120 shots)) | 0.877 [0.840, 0.914] | 0.552 [0.422, 0.680] | 0.921 [0.862, 0.955] | 0.580 [0.485, 0.673] |

Seed-mean paired differences, first minus second, with the interval from resampling shots only (every seed kept in each draw; the seed-to-seed spread is the SD in the table above):

Against the dense reference:

| First minus second | AUROC | AUPRC | F1 |
|---|---|---|---|
| ae-lstm minus clock-annotation | -0.082 [-0.142, -0.039] | -0.050 [-0.068, -0.032] | -0.009 [-0.063, 0.037] |
| ae-lstm minus clock-dense | -0.138 [-0.216, -0.091] | -0.081 [-0.099, -0.060] | +0.001 [-0.066, 0.056] |
| ae-ours-dense minus ae-lstm | +0.244 [0.171, 0.326] | +0.136 [0.110, 0.155] | +0.159 [0.125, 0.194] |
| ae-ours-dense minus ae-ours-threeway | -0.000 [-0.013, 0.011] | -0.000 [-0.007, 0.006] | +0.018 [0.003, 0.034] |
| ae-ours-dense minus ae-rcn | +0.119 [0.067, 0.176] | +0.062 [0.037, 0.084] | +0.096 [0.055, 0.146] |
| ae-ours-dense minus clock-annotation | +0.162 [0.107, 0.210] | +0.086 [0.062, 0.110] | +0.150 [0.091, 0.197] |
| ae-ours-dense minus clock-dense | +0.106 [0.056, 0.149] | +0.056 [0.030, 0.082] | +0.160 [0.088, 0.220] |
| ae-ours-legacy minus ae-lstm | +0.065 [-0.004, 0.138] | +0.053 [0.016, 0.090] | +0.012 [-0.035, 0.069] |
| ae-ours-legacy minus ae-ours-dense | -0.179 [-0.255, -0.116] | -0.084 [-0.111, -0.055] | -0.147 [-0.180, -0.113] |
| ae-ours-legacy minus ae-ours-threeway | -0.180 [-0.260, -0.112] | -0.084 [-0.114, -0.053] | -0.129 [-0.166, -0.095] |
| ae-ours-legacy minus ae-rcn | -0.061 [-0.159, 0.031] | -0.022 [-0.065, 0.022] | -0.051 [-0.098, 0.001] |
| ae-ours-legacy minus clock-annotation | -0.017 [-0.101, 0.063] | +0.003 [-0.037, 0.045] | +0.004 [-0.063, 0.066] |
| ae-ours-legacy minus clock-dense | -0.074 [-0.162, 0.010] | -0.028 [-0.070, 0.020] | +0.014 [-0.066, 0.087] |
| ae-ours-threeway minus ae-lstm | +0.244 [0.173, 0.328] | +0.136 [0.110, 0.155] | +0.141 [0.101, 0.182] |
| ae-ours-threeway minus ae-rcn | +0.119 [0.070, 0.175] | +0.062 [0.039, 0.084] | +0.078 [0.038, 0.127] |
| ae-ours-threeway minus clock-annotation | +0.162 [0.104, 0.210] | +0.086 [0.063, 0.110] | +0.132 [0.066, 0.185] |
| ae-ours-threeway minus clock-dense | +0.106 [0.056, 0.147] | +0.056 [0.031, 0.081] | +0.143 [0.062, 0.207] |
| ae-rcn minus ae-lstm | +0.125 [0.058, 0.200] | +0.075 [0.045, 0.096] | +0.063 [0.008, 0.114] |
| ae-rcn minus clock-annotation | +0.043 [-0.007, 0.091] | +0.025 [0.001, 0.046] | +0.055 [-0.033, 0.128] |
| ae-rcn minus clock-dense | -0.013 [-0.065, 0.033] | -0.006 [-0.031, 0.018] | +0.065 [-0.038, 0.152] |
| clock-annotation minus clock-dense | -0.056 [-0.076, -0.045] | -0.031 [-0.039, -0.023] | +0.010 [-0.008, 0.025] |

Against the annotation reference:

| First minus second | AUROC | AUPRC | F1 |
|---|---|---|---|
| ae-lstm minus clock-annotation | +0.038 [-0.006, 0.079] | +0.110 [-0.022, 0.219] | +0.058 [-0.099, 0.175] |
| ae-lstm minus clock-dense | +0.021 [-0.018, 0.060] | +0.116 [-0.003, 0.210] | +0.022 [-0.123, 0.138] |
| ae-ours-dense minus ae-lstm | -0.211 [-0.336, -0.095] | -0.336 [-0.471, -0.151] | -0.173 [-0.266, -0.064] |
| ae-ours-dense minus ae-ours-threeway | -0.025 [-0.056, -0.004] | -0.042 [-0.074, 0.002] | -0.016 [-0.036, 0.000] |
| ae-ours-dense minus ae-rcn | -0.224 [-0.351, -0.116] | -0.359 [-0.464, -0.198] | -0.142 [-0.231, -0.040] |
| ae-ours-dense minus clock-annotation | -0.173 [-0.301, -0.064] | -0.226 [-0.369, -0.051] | -0.115 [-0.240, 0.012] |
| ae-ours-dense minus clock-dense | -0.190 [-0.309, -0.088] | -0.220 [-0.334, -0.061] | -0.150 [-0.270, -0.032] |
| ae-ours-legacy minus ae-lstm | -0.150 [-0.244, -0.053] | -0.113 [-0.268, 0.036] | -0.156 [-0.265, -0.027] |
| ae-ours-legacy minus ae-ours-dense | +0.060 [-0.005, 0.141] | +0.223 [0.114, 0.297] | +0.017 [-0.034, 0.053] |
| ae-ours-legacy minus ae-ours-threeway | +0.035 [-0.039, 0.112] | +0.181 [0.074, 0.279] | +0.001 [-0.056, 0.042] |
| ae-ours-legacy minus ae-rcn | -0.163 [-0.254, -0.073] | -0.135 [-0.254, -0.017] | -0.125 [-0.222, -0.017] |
| ae-ours-legacy minus clock-annotation | -0.112 [-0.214, -0.026] | -0.002 [-0.161, 0.131] | -0.098 [-0.213, 0.003] |
| ae-ours-legacy minus clock-dense | -0.129 [-0.219, -0.051] | +0.004 [-0.129, 0.117] | -0.134 [-0.239, -0.045] |
| ae-ours-threeway minus ae-lstm | -0.186 [-0.291, -0.075] | -0.294 [-0.433, -0.121] | -0.156 [-0.247, -0.048] |
| ae-ours-threeway minus ae-rcn | -0.199 [-0.307, -0.097] | -0.316 [-0.425, -0.177] | -0.125 [-0.211, -0.027] |
| ae-ours-threeway minus clock-annotation | -0.147 [-0.261, -0.045] | -0.184 [-0.344, -0.021] | -0.099 [-0.220, 0.034] |
| ae-ours-threeway minus clock-dense | -0.164 [-0.272, -0.069] | -0.177 [-0.308, -0.034] | -0.134 [-0.246, -0.010] |
| ae-rcn minus ae-lstm | +0.013 [-0.012, 0.039] | +0.023 [-0.064, 0.109] | -0.031 [-0.102, 0.066] |
| ae-rcn minus clock-annotation | +0.051 [0.020, 0.079] | +0.133 [0.034, 0.211] | +0.027 [-0.081, 0.116] |
| ae-rcn minus clock-dense | +0.034 [0.013, 0.058] | +0.139 [0.053, 0.205] | -0.009 [-0.119, 0.074] |
| clock-annotation minus clock-dense | -0.017 [-0.033, 0.005] | +0.006 [-0.022, 0.045] | -0.035 [-0.070, 0.000] |

## V100-only sensitivity

Legacy and dense seed 0 ran on A100, threeway seed 0 on V100. Restricting every arm to its V100 runs of seeds other than 0 (dense: ae-ours-dense-seed1, ae-ours-dense-seed2; legacy: ae-ours-legacy-seed1, ae-ours-legacy-seed2; threeway: ae-ours-threeway-seed1, ae-ours-threeway-seed2) gives two seeds per arm on identical hardware. Seed means, with the shot-only interval:

| Cohort | Reference | Arm | AUROC | AUPRC |
|---|---|---|---|---|
| 60 shots | dense | legacy | 0.835 [0.798, 0.869] | 0.924 [0.903, 0.942] |
| 60 shots | dense | dense | 0.984 [0.976, 0.991] | 0.993 [0.988, 0.997] |
| 60 shots | dense | threeway | 0.973 [0.954, 0.987] | 0.988 [0.977, 0.995] |
| 60 shots | annotation | legacy | 0.732 [0.676, 0.779] | 0.472 [0.382, 0.552] |
| 60 shots | annotation | dense | 0.633 [0.566, 0.698] | 0.253 [0.192, 0.337] |
| 60 shots | annotation | threeway | 0.650 [0.590, 0.707] | 0.294 [0.222, 0.380] |
| 19 shared held-out shots | dense | legacy | 0.788 [0.705, 0.861] | 0.899 [0.866, 0.936] |
| 19 shared held-out shots | dense | dense | 0.979 [0.960, 0.994] | 0.991 [0.978, 0.998] |
| 19 shared held-out shots | dense | threeway | 0.980 [0.963, 0.991] | 0.991 [0.979, 0.997] |
| 19 shared held-out shots | annotation | legacy | 0.758 [0.666, 0.854] | 0.563 [0.391, 0.711] |
| 19 shared held-out shots | annotation | dense | 0.685 [0.554, 0.812] | 0.324 [0.179, 0.553] |
| 19 shared held-out shots | annotation | threeway | 0.718 [0.606, 0.833] | 0.386 [0.216, 0.591] |

Selected paired contrasts on the same hardware (AUROC):

| Cohort | Reference | Contrast | AUROC |
|---|---|---|---|
| 60 shots | dense | ae-ours-dense minus ae-ours-legacy | +0.149 [0.114, 0.188] |
| 60 shots | dense | ae-ours-threeway minus ae-ours-legacy | +0.138 [0.098, 0.179] |
| 60 shots | annotation | ae-ours-dense minus ae-ours-legacy | -0.098 [-0.141, -0.051] |
| 60 shots | annotation | ae-ours-threeway minus ae-ours-legacy | -0.082 [-0.121, -0.035] |
| 19 shared held-out shots | dense | ae-ours-dense minus ae-ours-legacy | +0.191 [0.120, 0.277] |
| 19 shared held-out shots | dense | ae-ours-threeway minus ae-ours-legacy | +0.191 [0.117, 0.278] |
| 19 shared held-out shots | dense | ae-ours-legacy minus ae-rcn | -0.073 [-0.172, 0.024] |
| 19 shared held-out shots | dense | ae-ours-dense minus ae-rcn | +0.117 [0.066, 0.175] |
| 19 shared held-out shots | dense | ae-ours-threeway minus ae-rcn | +0.118 [0.069, 0.174] |
| 19 shared held-out shots | annotation | ae-ours-dense minus ae-ours-legacy | -0.073 [-0.162, -0.004] |
| 19 shared held-out shots | annotation | ae-ours-threeway minus ae-ours-legacy | -0.040 [-0.123, 0.039] |
| 19 shared held-out shots | annotation | ae-ours-legacy minus ae-rcn | -0.153 [-0.238, -0.069] |
| 19 shared held-out shots | annotation | ae-ours-dense minus ae-rcn | -0.226 [-0.348, -0.120] |
| 19 shared held-out shots | annotation | ae-ours-threeway minus ae-rcn | -0.193 [-0.300, -0.092] |

## Interpretation

**Within ae-ours (supervision).** On the 19 shared shots, the AUROC of dense minus legacy supervision +0.179 [0.116, 0.255] against the dense reference and -0.060 [-0.141, 0.005] against the annotation; threeway minus legacy supervision +0.180 [0.112, 0.260] against the dense reference and -0.035 [-0.112, 0.039] against the annotation. On the 60 shots, the AUROC of dense minus legacy supervision +0.133 [0.104, 0.165] against the dense reference and -0.088 [-0.128, -0.045] against the annotation; threeway minus legacy supervision +0.124 [0.089, 0.159] against the dense reference and -0.076 [-0.113, -0.034] against the annotation. Dense minus threeway supervision, against the dense reference: -0.000 [-0.013, 0.011] on the 19 shots and +0.010 [-0.003, 0.028] on the 60 shots, so dense relabelling and annotation-and-TokEye agreement are not separated by this experiment.

AUROC, seed-mean ae-ours minus the saved detector, 19 shared held-out shots, interval from resampling shots only:

| ae-ours arm | minus ae-rcn, dense | minus ae-rcn, annotation | minus ae-lstm, dense | minus ae-lstm, annotation |
|---|---|---|---|---|
| legacy | -0.061 [-0.159, 0.031] | -0.163 [-0.254, -0.073] | +0.065 [-0.004, 0.138] | -0.150 [-0.244, -0.053] |
| dense | +0.119 [0.067, 0.176] | -0.224 [-0.351, -0.116] | +0.244 [0.171, 0.326] | -0.211 [-0.336, -0.095] |
| threeway | +0.119 [0.070, 0.175] | -0.199 [-0.307, -0.097] | +0.244 [0.173, 0.328] | -0.186 [-0.291, -0.075] |

AUPRC, seed-mean ae-ours minus the saved detector, 19 shared held-out shots, interval from resampling shots only:

| ae-ours arm | minus ae-rcn, dense | minus ae-rcn, annotation | minus ae-lstm, dense | minus ae-lstm, annotation |
|---|---|---|---|---|
| legacy | -0.022 [-0.065, 0.022] | -0.135 [-0.254, -0.017] | +0.053 [0.016, 0.090] | -0.113 [-0.268, 0.036] |
| dense | +0.062 [0.037, 0.084] | -0.359 [-0.464, -0.198] | +0.136 [0.110, 0.155] | -0.336 [-0.471, -0.151] |
| threeway | +0.062 [0.039, 0.084] | -0.316 [-0.425, -0.177] | +0.136 [0.110, 0.155] | -0.294 [-0.433, -0.121] |

**What the swap shows (AUROC, 19 shared shots).** Legacy-supervised ae-ours is not resolved from ae-rcn on the dense reference (-0.061 [-0.159, 0.031]) and trails it on the annotation (-0.163 [-0.254, -0.073]). Its point estimate is below ae-rcn's on both references, resolved on the annotation but with an interval that includes zero on the dense reference; it does not lead ae-rcn there, so there is no reversal against ae-rcn when ae-ours is trained on legacy-type supervision. The same model is not resolved from ae-lstm on the dense reference (+0.065 [-0.004, 0.138]) and trails it on the annotation (-0.150 [-0.244, -0.053]). The reversal against ae-lstm persists in sign (above on the dense reference, below on the annotation) but is resolved only on the annotation; the dense interval includes zero. The dense-supervised model leads ae-rcn on the dense reference (+0.119 [0.067, 0.176]) and trails it on the annotation (-0.224 [-0.351, -0.116]). The threeway-supervised model leads ae-rcn on the dense reference (+0.119 [0.070, 0.175]) and trails it on the annotation (-0.199 [-0.307, -0.097]). ae-ours's lead over ae-rcn on the dense reference therefore depends on dense or annotation-and-TokEye-agreement supervision: it is absent when the same recipe trains on the legacy annotation and present when it trains on either of the other two targets. Which of the two carries it is not separated, and the cross-architecture confounds below apply to every statement against the saved detectors.

**Input-free clock.** Each 10 ms frame is scored by its positive rate over the 120 training and selection shots, with no input and no evaluation data. The annotation-based clock peaks at 495 ms (0.78), averages 0.22 over the 0 to 2 s record and 0.00 over its first and last 100 ms; the dense-based clock peaks at 405 ms (0.99), averages 0.43 over the 0 to 2 s record and 0.12 over its first and last 100 ms. On the 19 shared shots its pooled AUROC is 0.860 against the annotation (clock from the annotation) and 0.875 against the dense reference (clock from the dense labels). For comparison (annotation / dense): ae-rcn 0.911 / 0.862, ae-lstm 0.898 / 0.736, and the ae-ours seed means legacy 0.748 / 0.801, dense 0.687 / 0.981, threeway 0.713 / 0.981. Within shots the clock's median AUROC is 0.926 against the annotation and 1.000 against the dense reference, against ae-rcn 0.944 / 0.980 and ae-lstm 0.917 / 0.769.

Against the annotation the clock reaches 88% of ae-rcn's pooled AUROC margin over chance and 90% of ae-lstm's. Paired AUROC differences against the annotation: ae-rcn minus the annotation clock +0.051 [0.020, 0.079], ae-lstm minus it +0.038 [-0.006, 0.079], and legacy-supervised ae-ours minus it -0.112 [-0.214, -0.026], dense-supervised ae-ours minus it -0.173 [-0.301, -0.064], threeway-supervised ae-ours minus it -0.147 [-0.261, -0.045]. Against the dense reference, ae-ours-dense minus the dense clock is +0.106 [0.056, 0.149] and ae-rcn minus it -0.013 [-0.065, 0.033].

ae-ours trains on random 182 ms windows (710 columns) and cannot learn absolute time; Garcia's models read the whole 0 to 2 s record and can. Where the annotation concentrates in time, as the profile above shows, part of the older detectors' score against the annotation is time context, not a better reading of the spectrogram, and the clock is the control that measures how much. The fair test of time context against supervision is the deferred LSTM retrain (Limitations).

**F1.** AUROC and AUPRC carry the claims. F1 is reported with every method calibrated on the reference it is scored on. The earlier protocol left the saved detectors at thresholds set against the annotation, which moves their F1 on the dense reference: ae-rcn on the 19 shots scores 0.864 with its threshold calibrated on dense and 0.350 at its saved annotation-set threshold (`f1_own_target_threshold`).

**Selection.** The reviewers' objection was that the published model was selected on the 60 validation shots that include the 19 benchmark shots. The same recipe retrained with the epoch chosen on 20 separate training shots (the threeway arm, which is the published target) scores AUROC 0.981 on the 19 shots (published 0.982) and 0.975 on the 60 shots (published 0.980), AUPRC 0.992 and 0.989 (published 0.992 and 0.992); the seed SD of the AUROC is 0.004. Selecting on the validation block did not inflate the published scores beyond that seed spread.

## Cross-architecture confounds

The swap holds the architecture fixed and varies supervision. The comparison with ae-rcn and ae-lstm still differs in more than supervision, and every statement against them carries these:

1. **Time context.** The saved detectors read the whole 0 to 2 s record; ae-ours sees 182 ms windows. The clock above measures how much that is worth against each reference.
2. **Data volume.** ae-rcn and ae-lstm trained on 801 shots; each ae-ours arm trains on 100.
3. **Campaign overlap.** 41 of the 60 evaluation shots are in the saved detectors' training set (which is why they can be scored on 19 only), and each of the 19 held-out shots has a Garcia training shot 1 to 2 shot numbers away, a same-day neighbour. The evaluation shots sit in run blocks 1706xx: 16, 1707xx: 23, 1708xx: 11, 1752xx: 8, 1788xx: 2; ae-ours's 100 training shots in those blocks number 1706xx: 0, 1707xx: 0, 1708xx: 2, 1752xx: 0, 1788xx: 0.
4. **Input band.** ae-ours reads 80.6 to 250.0 kHz; the saved detectors read 20 to 250 kHz. Annotated columns that are BAE only are 9.7% of annotated columns on the 60 evaluation shots (9.3% on the 19) against 1.5% in ae-ours's training shots. In the evaluation shots 60% of those columns (45% on the 19) carry in-band TokEye activity, against 93% of other annotated columns; in training the shares are 82% and 86%.
5. **Calibration sets.** F1 thresholds come from 20 selection shots for ae-ours and the clocks and from six for the saved detectors.

## Limitations

- **Dense labels' provenance.** All 639 history entries of the dense table are by 1 reviewer (nc1514), and their source is `alfven_eigenmode_format_2026_v1.csv`: the review was pre-filled from the annotation's source table. Whether a TokEye layer was on screen while reviewing is an open question, and it bears on why dense and threeway supervision score alike.
- **Three seeds** give limited precision for training variability. The headline intervals resample shots only, and the seed SD is reported beside them; the interval that also resamples the seed IDs is in evaluation.json as `ci95` and is not used here, because with three seeds it mostly reflects the worst seed.
- **Hardware.** A100 and V100S runs mix; see the V100-only sensitivity.
- **Fixed training windows.** Every epoch re-draws the same windows (persistent workers); this applies to all arms and to the published recipe.
- **The LSTM retrain was deferred.** The brief's optional item (`ae-lstm-retrained`: the published 3 x 64 LSTM architecture trained on the same 100/20 split, on whole records, with dense and with legacy supervision) was not run. It is the control that separates time context from supervision: it gives the older architecture the same supervision treatment while keeping its time context, and the clock rows bound what time context alone achieves.
- The saved detectors are scored on 19 shots and calibrated on six; their intervals are wide.

## Excluded and superseded records

These records are not in any mean, SD or interval above. They are scored with the same code, thresholds calibrated on the selection shots, so the effect of the convergence rule can be read.

| Record | Epochs run | Selected epoch | Cohort | Dense AUROC | Annotation AUROC | Selection within-shot SD | Selection AUROC |
|---|---:|---:|---|---|---|---:|---:|
| ae-ours-dense-seed1-superseded1 | 12 | 6 | 60 shots | 0.973 [0.960, 0.984] | 0.617 [0.549, 0.687] | 0.373 | 0.921 |
| ae-ours-dense-seed1-superseded1 | 12 | 6 | 19 shared held-out shots | 0.979 [0.958, 0.993] | 0.666 [0.527, 0.792] | 0.373 | 0.921 |
| ae-ours-legacy-seed2-superseded1 | 8 | 2 | 60 shots | 0.456 [0.420, 0.495] | 0.463 [0.424, 0.503] | 0.001 | 0.467 |
| ae-ours-legacy-seed2-superseded1 | 8 | 2 | 19 shared held-out shots | 0.434 [0.371, 0.498] | 0.478 [0.422, 0.527] | 0.001 | 0.467 |

## Reproduction

From the worktree, with the scratch TMPDIR, `LABELER_ROOT`, `LABELER_LABEL_TABLES`, `LABELER_NO_FETCH=1` and `PYTHONPATH=$PWD/src`:

```bash
# training: GPU, CUDA interpreter; sbatch (AESWAP_RUNS="legacy:2 dense:1" for reruns) or one head-node GPU per process
sbatch scripts/labeler/ae_supervision_swap.sbatch
bash scripts/labeler/ae_supervision_swap_head.sh 0 legacy:2
# CPU, pixi labelmaker environment
python scripts/labeler/ae_supervision_swap.py verify
python scripts/labeler/ae_supervision_swap.py audit-convergence
python scripts/labeler/ae_supervision_swap.py evaluate
```

`audit-convergence` applies the declared rule to every record, archives any record that stopped inside the plateau, and lists the runs still to train. Completed runs are never overwritten. The launchers use a short TMPDIR (`$LABELER_ROOT/scratch/ae-sw`) because DataLoader workers add `/pymp-*/listener-*` to it and a socket path must stay under 108 bytes.
