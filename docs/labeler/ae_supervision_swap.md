# AE supervision swap

The three activity-supervision arms completed three seeds each under clean 100/20 shot selection. Both references are evaluated on all 60 original validation shots and the 19 shared held-out shots.

Source for every result below: [evaluation.json](../../outputs/labeler/ae/supervision_swap/evaluation.json). It embeds the completed run records, execution IDs, selected epochs, thresholds, per-seed scores and paired differences. [manifest.json](../../outputs/labeler/ae/supervision_swap/manifest.json) contains the exact shot lists and frozen input hashes; [verification.json](../../outputs/labeler/ae/supervision_swap/verification.json) checks target equality and split isolation.

## Convergence rule (declared 2026-10-03, before the convergence reruns)

Fix round 1 found that legacy seed 2 never left the constant-output plateau: its
selection loss was lowest at epoch 2, patience 5 ended it at epoch 7, and its
selected checkpoint outputs one value (within-shot SD 0.001). One rule, declared
here and in `CONVERGENCE_RULE` (`src/labeler/ae/supervision.py`) before any rerun,
applies to all nine records of all three arms:

1. Early stopping counts non-improving epochs only from zero-based epoch 10
   (maximum 30 epochs, patience 5, minimum improvement 1e-6, monitored on the 20
   selection shots). A record conforms when replaying this rule on its recorded
   selection-loss history stops it at the epoch where it stopped; one that stopped
   earlier is archived and rerun with the same seed.
2. A conforming record is screened on the selection shots only: it is excluded when
   its selected checkpoint is constant (mean within-shot SD of the score below
   0.005) or ranks its own target no better than chance (selection AUROC at most
   0.5). An excluded seed is replaced by the arm's next unused seed, starting at 3,
   trained under rule 1 and screened again.

An earlier draft of the rule (the screen alone, without rule 1) was applied by the
previous implementer to selection-only outputs and is archived as
`convergence_draft0.json`; this two-part rule supersedes it before any rerun.

## Frozen protocol

The current dense snapshot has 877 CSV rows, 371 present intervals and 180 shots. Its SHA256 is `b4308ff52d766779f33a92e9441a916bb07c854dcc1a98a2f61109bb6431e93a`. These observed counts differ from the brief's anticipated interval count. No source table was edited.

NumPy split seed 20261003 freezes 100 training, 20 selection and 60 evaluation shots. None overlaps the fixed catalog's blind test split. Seeds 0, 1 and 2 change initialization and sampled training windows; every arm uses the same split. The original one-seed manifest was archived before extending its seed list.

Only activity supervision changes: legacy is the audit's at-least-half annotation rule at 10 ms, expanded to native columns; dense uses the catalog any-touch state rule and masks unknown states; threeway retains native annotation/TokEye agreement and masks disagreement. The original annotation-and-TokEye frequency target and weights are identical in every arm. Each input covers 0–2 s with 7,820 native columns.

The original AeSeldNet recipe uses four CO2 channels, 348 frequency bins, frequency pools (6, 2, 29), two bidirectional GRUs, SCE plus the unchanged frequency objective, 710-column windows, eight windows per shot, batch 16, AdamW 1e-4, weight decay 1e-4, cosine decay to 1e-6, 30 epochs maximum and patience five. The selected epoch minimizes combined loss on the 20 selection shots. The CUDA interpreter is `envs/phase3/bin/python`, with `PYTHONPATH=<worktree>/src`; the launcher retains `AESWAP_PIXI_ENV` as an explicit alternative. CUDA allocations are capped at 10 GiB. Autocast uses bfloat16 when the runtime reports support, otherwise float32.

Each seed's threshold maximizes 10 ms selection F1 against its own activity target, with the highest threshold breaking ties. Threeway requires at least half the native columns to have agreement weight and a majority of those agreeing columns to be positive. The score is mean native probability; the frozen threshold is used against both evaluation references. No evaluation frames select epochs or thresholds.

Garcia probabilities use saved spectrogram predictions: maximum over the first four AE classes, mean over four chords, nearest output-bin centre on the 10 ms grid. Only 19 evaluation shots and six selection shots have available predictions held out from Garcia's training. The other 41 evaluation shots cannot be scored for these saved models. Legacy-calibrated older thresholds remain frozen against dense labels.

## Completed runs and operating points

Epoch numbers are zero-based, as in the trainer.

| Supervision | Seed | Selected epoch | Threshold | Execution | GPU |
|---|---:|---:|---:|---|---|
| dense | 0 | 19 | 0.203014240 | 2952146_1 (job 2952150) | NVIDIA A100-PCIE-40GB |
| dense | 1 | 6 | 0.584994057 | head node, GPU 0 | Tesla V100S-PCIE-32GB |
| dense | 2 | 13 | 0.058173278 | head node, GPU 0 | Tesla V100S-PCIE-32GB |
| legacy | 0 | 17 | 0.023511697 | 2952146_0 (job 2952149) | NVIDIA A100-PCIE-40GB |
| legacy | 1 | 17 | 0.190000283 | head node, GPU 0 | Tesla V100S-PCIE-32GB |
| legacy | 2 | 2 | 0.077075437 | head node, GPU 0 | Tesla V100S-PCIE-32GB |
| threeway | 0 | 13 | 0.362064370 | head node, GPU 0 | Tesla V100S-PCIE-32GB |
| threeway | 1 | 9 | 0.385975281 | head node, GPU 0 | Tesla V100S-PCIE-32GB |
| threeway | 2 | 26 | 0.477263172 | head node, GPU 0 | Tesla V100S-PCIE-32GB |

Legacy/dense seed 0 ran on A100; fallback runs used V100. The CUDA runtime probe records that this Torch build reports V100 bfloat16 support through emulation despite lacking native support. The as-built trainer therefore selects bfloat16 autocast on V100 too. The earlier report's hypothetical float32 fallback does not describe these actual runs. See evaluation.json:gpu_probe and each run's runtime metadata. Device types and kernel implementations differ; the observed across-seed SD includes environment variation.

Older selection thresholds:

- ae-rcn: `0.5281208929558328`, shots 172000, 176039, 176044, 176053, 176548, 176551.
- ae-lstm: `0.5118714645504951`, shots 172000, 176039, 176044, 176053, 176548, 176551.

## Scores and uncertainty

Entries for retrains are mean ± sample SD over three seeds, followed by a 95% pooled bootstrap interval. Each of 1,000 replicates resamples whole evaluation shots and the observed training seed IDs with replacement. The statistic averages each seed's pooled-frame metric. All arms share shot draws and seed indices (seed 20261004); saved baselines have only shot uncertainty. Frames and seed copies are never treated as independent observations. Per-seed shot-only intervals remain in JSON. Three seeds give limited precision for training variability.

### All 60 evaluation shots

Shots: 170659, 170660, 170661, 170662, 170663, 170664, 170665, 170666, 170667, 170669, 170671, 170672, 170675, 170677, 170678, 170679, 170714, 170715, 170716, 170717, 170718, 170719, 170720, 170721, 170722, 170724, 170725, 170727, 170729, 170730, 170790, 170791, 170792, 170793, 170794, 170795, 170797, 170798, 170799, 170801, 170803, 170805, 170806, 170807, 170808, 170809, 170810, 170811, 170813, 170814, 175239, 175240, 175241, 175244, 175245, 175250, 175251, 175252, 178872, 178879.

Scorable frames and positives: dense: 12000 frames, 8352 positives; legacy: 12000 frames, 2348 positives.

| Model / supervision | Reference | AUROC | AUPRC | F1 |
|---|---|---|---|---|
| ae-ours-legacy | dense | 0.711 ± 0.225 [0.466, 0.885] | 0.837 ± 0.157 [0.685, 0.952] | 0.757 ± 0.158 [0.593, 0.877] |
| ae-ours-dense | dense | 0.982 ± 0.008 [0.970, 0.991] | 0.992 ± 0.003 [0.985, 0.997] | 0.955 ± 0.013 [0.932, 0.973] |
| ae-ours-threeway | dense | 0.975 ± 0.004 [0.956, 0.988] | 0.989 ± 0.001 [0.977, 0.996] | 0.948 ± 0.004 [0.922, 0.967] |
| ae-ours-legacy | legacy | 0.629 ± 0.144 [0.469, 0.753] | 0.374 ± 0.162 [0.194, 0.526] | 0.382 ± 0.053 [0.314, 0.463] |
| ae-ours-dense | legacy | 0.632 ± 0.012 [0.562, 0.702] | 0.257 ± 0.003 [0.192, 0.341] | 0.408 ± 0.011 [0.353, 0.472] |
| ae-ours-threeway | legacy | 0.647 ± 0.014 [0.584, 0.706] | 0.287 ± 0.022 [0.212, 0.379] | 0.407 ± 0.007 [0.350, 0.471] |

Paired differences use the same pooled draws:

| First model minus second | Reference | AUROC | AUPRC | F1 |
|---|---|---|---|---|
| ae-ours-legacy minus ae-ours-dense | dense | -0.271 ± 0.230 [-0.524, -0.099] | -0.155 ± 0.159 [-0.310, -0.042] | -0.198 ± 0.146 [-0.346, -0.086] |
| ae-ours-legacy minus ae-ours-threeway | dense | -0.263 ± 0.221 [-0.505, -0.094] | -0.152 ± 0.156 [-0.305, -0.038] | -0.191 ± 0.154 [-0.342, -0.075] |
| ae-ours-dense minus ae-ours-threeway | dense | 0.007 ± 0.011 [-0.009, 0.028] | 0.003 ± 0.004 [-0.004, 0.014] | 0.007 ± 0.009 [-0.013, 0.026] |
| ae-ours-legacy minus ae-ours-dense | legacy | -0.003 ± 0.152 [-0.160, 0.118] | 0.117 ± 0.164 [-0.057, 0.241] | -0.026 ± 0.063 [-0.086, 0.038] |
| ae-ours-legacy minus ae-ours-threeway | legacy | -0.017 ± 0.158 [-0.187, 0.107] | 0.087 ± 0.184 [-0.107, 0.227] | -0.024 ± 0.060 [-0.086, 0.035] |
| ae-ours-dense minus ae-ours-threeway | legacy | -0.015 ± 0.012 [-0.040, 0.010] | -0.030 ± 0.020 [-0.058, -0.002] | 0.002 ± 0.007 [-0.016, 0.018] |

### Shared 19 held-out shots

Shots: 170660, 170661, 170663, 170666, 170669, 170677, 170678, 170718, 170725, 170729, 170730, 170792, 170793, 170798, 170801, 170803, 175241, 175245, 178879.

Scorable frames and positives: dense: 3800 frames, 2595 positives; legacy: 3800 frames, 796 positives.

| Model / supervision | Reference | AUROC | AUPRC | F1 |
|---|---|---|---|---|
| ae-ours-legacy | dense | 0.672 ± 0.209 [0.452, 0.849] | 0.813 ± 0.156 [0.637, 0.934] | 0.739 ± 0.141 [0.570, 0.849] |
| ae-ours-dense | dense | 0.982 ± 0.003 [0.964, 0.995] | 0.992 ± 0.001 [0.980, 0.998] | 0.955 ± 0.007 [0.920, 0.979] |
| ae-ours-threeway | dense | 0.981 ± 0.004 [0.965, 0.992] | 0.992 ± 0.001 [0.980, 0.997] | 0.938 ± 0.005 [0.891, 0.969] |
| ae-rcn | dense | 0.862 [0.810, 0.907] | 0.930 [0.906, 0.955] | 0.350 [0.244, 0.466] |
| ae-lstm | dense | 0.736 [0.654, 0.802] | 0.855 [0.835, 0.885] | 0.382 [0.260, 0.521] |
| ae-ours-legacy | legacy | 0.646 ± 0.145 [0.481, 0.807] | 0.435 ± 0.196 [0.220, 0.655] | 0.411 ± 0.059 [0.312, 0.531] |
| ae-ours-dense | legacy | 0.685 ± 0.017 [0.547, 0.809] | 0.336 ± 0.014 [0.182, 0.554] | 0.443 ± 0.009 [0.339, 0.555] |
| ae-ours-threeway | legacy | 0.713 ± 0.021 [0.593, 0.828] | 0.374 ± 0.037 [0.206, 0.586] | 0.456 ± 0.015 [0.341, 0.593] |
| ae-rcn | legacy | 0.911 [0.878, 0.939] | 0.691 [0.554, 0.798] | 0.571 [0.439, 0.660] |
| ae-lstm | legacy | 0.898 [0.854, 0.930] | 0.668 [0.509, 0.779] | 0.602 [0.435, 0.718] |

Paired differences use the same pooled draws:

| First model minus second | Reference | AUROC | AUPRC | F1 |
|---|---|---|---|---|
| ae-ours-legacy minus ae-ours-dense | dense | -0.311 ± 0.210 [-0.530, -0.136] | -0.179 ± 0.156 [-0.351, -0.061] | -0.217 ± 0.134 [-0.369, -0.111] |
| ae-ours-legacy minus ae-ours-threeway | dense | -0.309 ± 0.205 [-0.526, -0.131] | -0.178 ± 0.154 [-0.348, -0.060] | -0.199 ± 0.136 [-0.360, -0.093] |
| ae-ours-legacy minus ae-rcn | dense | -0.190 ± 0.209 [-0.408, -0.002] | -0.117 ± 0.156 [-0.287, 0.009] | 0.389 ± 0.141 [0.182, 0.562] |
| ae-ours-legacy minus ae-lstm | dense | -0.065 ± 0.209 [-0.287, 0.121] | -0.042 ± 0.156 [-0.212, 0.078] | 0.357 ± 0.141 [0.144, 0.540] |
| ae-ours-dense minus ae-ours-threeway | dense | 0.002 ± 0.006 [-0.014, 0.015] | 0.001 ± 0.002 [-0.006, 0.006] | 0.018 ± 0.002 [-0.012, 0.054] |
| ae-ours-dense minus ae-rcn | dense | 0.120 ± 0.003 [0.072, 0.176] | 0.062 ± 0.001 [0.039, 0.084] | 0.606 ± 0.007 [0.455, 0.728] |
| ae-ours-dense minus ae-lstm | dense | 0.246 ± 0.003 [0.174, 0.326] | 0.137 ± 0.001 [0.111, 0.156] | 0.574 ± 0.007 [0.403, 0.714] |
| ae-ours-threeway minus ae-rcn | dense | 0.119 ± 0.004 [0.069, 0.174] | 0.062 ± 0.001 [0.039, 0.085] | 0.588 ± 0.005 [0.437, 0.714] |
| ae-ours-threeway minus ae-lstm | dense | 0.244 ± 0.004 [0.171, 0.329] | 0.136 ± 0.001 [0.110, 0.155] | 0.556 ± 0.005 [0.393, 0.704] |
| ae-rcn minus ae-lstm | dense | 0.125 [0.058, 0.200] | 0.075 [0.045, 0.096] | -0.032 [-0.079, 0.016] |
| ae-ours-legacy minus ae-ours-dense | legacy | -0.040 ± 0.157 [-0.233, 0.117] | 0.100 ± 0.200 [-0.135, 0.277] | -0.032 ± 0.068 [-0.116, 0.059] |
| ae-ours-legacy minus ae-ours-threeway | legacy | -0.067 ± 0.166 [-0.267, 0.086] | 0.061 ± 0.232 [-0.206, 0.263] | -0.045 ± 0.073 [-0.149, 0.050] |
| ae-ours-legacy minus ae-rcn | legacy | -0.265 ± 0.145 [-0.426, -0.109] | -0.255 ± 0.196 [-0.461, -0.053] | -0.160 ± 0.059 [-0.261, -0.033] |
| ae-ours-legacy minus ae-lstm | legacy | -0.253 ± 0.145 [-0.419, -0.093] | -0.233 ± 0.196 [-0.448, 0.001] | -0.191 ± 0.059 [-0.306, -0.041] |
| ae-ours-dense minus ae-ours-threeway | legacy | -0.027 ± 0.016 [-0.068, -0.002] | -0.038 ± 0.037 [-0.086, 0.015] | -0.013 ± 0.010 [-0.044, 0.012] |
| ae-ours-dense minus ae-rcn | legacy | -0.226 ± 0.017 [-0.357, -0.119] | -0.355 ± 0.014 [-0.462, -0.202] | -0.128 ± 0.009 [-0.218, -0.032] |
| ae-ours-dense minus ae-lstm | legacy | -0.213 ± 0.017 [-0.341, -0.098] | -0.332 ± 0.014 [-0.472, -0.160] | -0.159 ± 0.009 [-0.250, -0.054] |
| ae-ours-threeway minus ae-rcn | legacy | -0.199 ± 0.021 [-0.308, -0.099] | -0.316 ± 0.037 [-0.431, -0.171] | -0.115 ± 0.015 [-0.199, -0.012] |
| ae-ours-threeway minus ae-lstm | legacy | -0.186 ± 0.021 [-0.297, -0.077] | -0.294 ± 0.037 [-0.442, -0.127] | -0.146 ± 0.015 [-0.239, -0.040] |
| ae-rcn minus ae-lstm | legacy | 0.013 [-0.012, 0.039] | 0.023 [-0.064, 0.109] | -0.031 [-0.102, 0.066] |

## Interpretation

On the shared 19 shots, threeway ae-ours minus ae-rcn is 0.119 ± 0.004 [0.069, 0.174] against dense and -0.199 ± 0.021 [-0.308, -0.099] against legacy. The clean-selection AUROC ranking reversal is supported by both paired intervals.

On the shared 19 shots, threeway ae-ours minus ae-lstm is 0.244 ± 0.004 [0.171, 0.329] against dense and -0.186 ± 0.021 [-0.297, -0.077] against legacy. The clean-selection AUROC ranking reversal is supported by both paired intervals.

Dense-supervised minus threeway ae-ours on the 60 shots, against dense: AUROC 0.007 ± 0.011 [-0.009, 0.028]; AUPRC 0.003 ± 0.004 [-0.004, 0.014]; F1 0.007 ± 0.009 [-0.013, 0.026]. None of these paired intervals resolves an improvement over threeway.

Dense-supervised minus threeway ae-ours on the shared 19 shots, against dense: AUROC 0.002 ± 0.006 [-0.014, 0.015]; AUPRC 0.001 ± 0.002 [-0.006, 0.006]; F1 0.018 ± 0.002 [-0.012, 0.054]. None of these paired intervals resolves an improvement over threeway.

Dense-supervised minus legacy-supervised ae-ours on the 60 shots, against dense: AUROC 0.271 ± 0.230 [0.099, 0.524]; AUPRC 0.155 ± 0.159 [0.042, 0.310]; F1 0.198 ± 0.146 [0.086, 0.346].

Dense-supervised minus legacy-supervised ae-ours on the 60 shots, against legacy: AUROC 0.003 ± 0.152 [-0.118, 0.160]; AUPRC -0.117 ± 0.164 [-0.241, 0.057]; F1 0.026 ± 0.063 [-0.038, 0.086].

Dense-supervised minus legacy-supervised ae-ours on the shared 19 shots, against dense: AUROC 0.311 ± 0.210 [0.136, 0.530]; AUPRC 0.179 ± 0.156 [0.061, 0.351]; F1 0.217 ± 0.134 [0.111, 0.369].

Dense-supervised minus legacy-supervised ae-ours on the shared 19 shots, against legacy: AUROC 0.040 ± 0.157 [-0.117, 0.233]; AUPRC -0.100 ± 0.200 [-0.277, 0.135]; F1 0.032 ± 0.068 [-0.059, 0.116].

On the shared 19 shots against dense, the AUROC point-estimate ranking is ae-ours-dense (0.982) > ae-ours-threeway (0.981) > ae-rcn (0.862) > ae-lstm (0.736) > ae-ours-legacy (0.672).

On the shared 19 shots against dense, the AUPRC point-estimate ranking is ae-ours-dense (0.992) > ae-ours-threeway (0.992) > ae-rcn (0.930) > ae-lstm (0.855) > ae-ours-legacy (0.813).

On the shared 19 shots against dense, the F1 point-estimate ranking is ae-ours-dense (0.955) > ae-ours-threeway (0.938) > ae-ours-legacy (0.739) > ae-lstm (0.382) > ae-rcn (0.350).

On the shared 19 shots against legacy, the AUROC point-estimate ranking is ae-rcn (0.911) > ae-lstm (0.898) > ae-ours-threeway (0.713) > ae-ours-dense (0.685) > ae-ours-legacy (0.646).

On the shared 19 shots against legacy, the AUPRC point-estimate ranking is ae-rcn (0.691) > ae-lstm (0.668) > ae-ours-legacy (0.435) > ae-ours-threeway (0.374) > ae-ours-dense (0.336).

On the shared 19 shots against legacy, the F1 point-estimate ranking is ae-lstm (0.602) > ae-rcn (0.571) > ae-ours-threeway (0.456) > ae-ours-dense (0.443) > ae-ours-legacy (0.411).

Legacy supervision has substantial observed training variability. Its seed 2 selected epoch 2 under the unchanged combined-loss rule; all completed seeds remain in the mean, SD and pooled intervals. The contrasts estimate behavior under this frozen training and selection recipe, including early stopping and target-specific calibration, rather than the best achievable legacy-trained model.

These rankings use clean selection for every ae-ours arm. Their paired intervals above determine which gaps remain uncertain; a point-estimate ordering alone does not establish a difference. F1 also reflects calibration against each model's training target. Garcia probabilities come from fixed saved models, with thresholds calibrated on the six available selection shots. This experiment isolates supervision within ae-ours and resolves its selection overlap, while comparisons across architectures retain different training recipes.


## Reproduction

From this worktree with the prescribed scratch TMPDIR, LABELER_ROOT, LABELER_LABEL_TABLES, LABELER_NO_FETCH=1 and PYTHONPATH=$PWD/src:

```bash
# Existing manifest already freezes seeds 0, 1, 2.
sbatch scripts/labeler/ae_supervision_swap.sbatch
# After every run finishes:
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker python scripts/labeler/ae_supervision_swap.py verify
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker python scripts/labeler/ae_supervision_swap.py evaluate
```

Completed runs cannot be silently overwritten. A pending task exceeding 30 minutes is cancelled before running that seed with CUDA_VISIBLE_DEVICES=0 on the shared head node. A short worktree symlink resolves multiprocessing socket paths into the prescribed scratch temp directory; actual temp files stay there.
