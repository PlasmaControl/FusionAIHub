# Confinement benchmark: the BES classifier

Status: **run 2026-10-01, retrained from the paper's recipe.** The owner named the BES-based
confinement classifier as the benchmark for the confinement labels, the way the CO2 LSTM and RCN are
for the AE labels (2026-10-01). Its weights and code were not found, so the recipe below was
retrained on the corpus BES and scored on shots the network never saw. Script:
`scripts/labeler/confinement_bes_benchmark.py` (stages `features`, `train`, `evaluate`); record:
`outputs/labeler/confinement/bes/evaluation.json` (git `2b4da1d`); features, fold models and
out-of-fold predictions: `$LABELER_ROOT/benchmarks/confinement/bes/`. The paper's title, authors and
citation were not in the paste: fill them in below when known.

Updated 2026-10-03: the gap to the published score was taken apart by an ablation on 444 shots
fetched at 1 MHz (the section "The gap to the published score: an ablation" below). The first retrain's
sections stand as run on 2026-10-01; it read the cohort's two blind test shots (190857, 192756),
which no later work reads, so its rerun without them (117 shots) is the ablation's `base`.

Citation: _not given_

## What the paper's model is

A classifier from a block of BES (beam emission spectroscopy) data to a
probability over K = 4 confinement regimes: **L, H, QH, WP QH**.

| Item | Paper |
|---|---|
| Input block | 1024 time samples of the BES array in a 6 (radial rows) x 8 (poloidal columns) configuration, 48 channels |
| Band-pass | 2.5-150 kHz |
| Standardisation | mean and standard deviation of the training set |
| Features | the 1024 samples are split into 2 sub-windows of 512; each is split in 2 segments of 256; FFT of each segment; magnitudes; log10 of the squared magnitudes; the 2 segments averaged. Result: 128 unique spectral features per 512-sample sub-window, per channel (2 x 128 per channel) |
| Network | dropout, 3D convolution (10 kernels of (3, 3, 5) over radial, poloidal, frequency; stride 1; zero padding; groups = 2, one per sub-window), batch norm, LeakyReLU, 3D max-pool (1, 2, 4), flatten, MLP with 2 hidden layers of 60 (LeakyReLU), 4 logits, softmax |
| Loss | cross-entropy over the 4 classes, averaged over the mini-batch |
| Optimiser | Adam with weight decay; one learning rate for the convolution and a different one for the MLP; 60,000 optimiser steps; early stopping after 30 evaluations without improvement (not triggered); the checkpoint with the best validation F1 is kept |
| Software and compute | PyTorch Lightning 1.6.5; 48 A100 GPUs on Perlmutter, one run about 5-10 h |
| Metrics | per-class recall, precision and F1; confusion matrix; one-vs-rest ROC AUC |

**Not in the paste** (equations and symbols were lost, so these need the paper
or the original code): the dropout probability, both learning rates, the weight
decay and Adam betas, the batch size, the parameter counts, the sampling rate and
the window duration in time, the channel selection for the 6 x 8 block, the class
counts (the paper's table 1) and the train/validation/test split.

## Published results (as pasted)

| Confinement regime | F1 | Precision | Recall |
|---|---|---|---|
| L-mode | 0.94 | 0.98 | 0.90 |
| H-mode | 0.97 | 0.95 | 0.98 |
| QH-mode | 0.94 | 0.92 | 0.96 |
| WP QH-mode | 0.90 | 0.95 | 0.86 |
| Average | 0.94 | 0.95 | 0.93 |

These come from the paper's own test split (not described in the excerpt), not from
our labels, so they are the reference for what the model can do, not a number to
compare with ours directly.

## The original is not on disk

No checkpoint or training code for this network was found in the group's project directories. The
nearest relative, `/projects/EKOLEMEN/jzimmerman/bes-ml`, has a four-class path (L 0, H 1, QH 2, WP 3;
channel = row * 8 + column) in plain PyTorch, which is not this network. So the recipe was
reimplemented from the excerpt and retrained, as for the AE detectors that could not be run, except
that here the retraining is ours.

## What was run

| Item | Here |
|---|---|
| Data | the corpus BES (`<shot>_processed.h5`, group `bes`, 64 channels at **500 kHz**) of the 119 shots (185871-196493) of the merged intervals that the corpus holds BES for, out of 448 labelled shots |
| Windows | 1024 samples (2.05 ms) wholly inside one interval of a single regime, one every 2048 samples: 66,532 windows (L 8,209 on 38 shots; H 47,373 on 75; QH 5,608 on 17; WP 5,342 on 20); the one H/L conflict interval is left out |
| Block | channels 8-55, rows 1-6 of the 8 x 8 grid: the paper's 6 x 8 block (which rows is not in the excerpt) |
| Pre-processing | as the paper: band-pass 2.5-150 kHz (4th-order Butterworth, causal), per-channel standardisation by the training windows' standard deviation, 2 sub-windows of 512, 2 segments of 256, FFT, log10 of the squared magnitude of bins 0-127, the two segments averaged: 2 x 128 features per channel |
| Network | dropout, Conv3d (10 kernels (3, 3, 5), zero padding, groups 2), batch norm, LeakyReLU, MaxPool3d (1, 2, 4), MLP 2 x 60, 4 logits: 465,244 parameters |
| Training | cross-entropy, AdamW, 60,000 steps allowed and early stopping after 30 evaluations (every 500 steps) without a better validation loss, as the paper; it stopped at 15,500-16,500 steps. The checkpoint with the best validation macro-F1 is kept (best steps 7,500, 1,500, 12,500, 6,000, 1,500). About 2.3 minutes per fold on one V100S |
| Assumed, not in the excerpt | dropout 0.2, learning rates 1e-3 (convolution) and 1e-4 (MLP), weight decay 0.01, batch 256, natural class frequencies, 500 kHz, rows 1-6, stride 2048 samples. None was tuned |
| Protocol | shot-grouped 5-fold cross-validation, folds dealt within class-presence strata; a fold's test shots are scored by a network trained on three other folds and validated on a sixth of the rest, so every prediction is out of sample. One seed. Intervals are 95 % shot bootstraps (1000 replicates) |

## Results

**All 119 shots, out of sample** (window level, as the paper scores):

| Regime | Windows (shots) | Precision | Recall | F1 [95 % CI] | AUROC | AUPRC | Paper F1 (P / R) |
|---|---|---|---|---|---|---|---|
| L-mode | 8,209 (38) | 0.765 | 0.703 | 0.733 [0.60, 0.84] | 0.945 | 0.783 | 0.94 (0.98 / 0.90) |
| H-mode | 47,373 (75) | 0.946 | 0.939 | 0.943 [0.91, 0.97] | 0.948 | 0.979 | 0.97 (0.95 / 0.98) |
| QH-mode | 5,608 (17) | 0.505 | 0.639 | 0.564 [0.39, 0.71] | 0.942 | 0.496 | 0.94 (0.92 / 0.96) |
| WP QH-mode | 5,342 (20) | 0.495 | 0.452 | 0.473 [0.25, 0.65] | 0.870 | 0.546 | 0.90 (0.95 / 0.86) |
| **Macro** | 66,532 (119) | 0.678 | 0.683 | 0.678 [0.60, 0.75] | 0.926 [0.89, 0.96] | 0.701 | 0.94 (0.95 / 0.93) |

Confusion matrix (windows, rows true):

| true \ predicted | L | H | QH | WP |
|---|---|---|---|---|
| L | 5,773 | 1,527 | 411 | 498 |
| H | 1,462 | 44,493 | 992 | 426 |
| QH | 268 | 222 | 3,583 | 1,535 |
| WP | 43 | 768 | 2,116 | 2,415 |

**On the 27 test shots of the diagnostic-summary models** ([confinement_model_card.md](confinement_model_card.md); the
shots among its 28 test shots that have corpus BES), same predictions: macro F1 0.724 [0.46, 0.86], AUROC
0.955. Only 2 QH and 4 WP shots, so the per-class intervals reach 0.

**Beside the four-class diagnostic-summary baseline**, on the 1,354 50 ms bins of those shots that hold both
(the BES network's window probabilities averaged in the bin; that model's four-class histogram-gradient-boosting
probabilities; truth the bin's merged regime):

| Model | F1 L | F1 H | F1 QH | F1 WP | Macro F1 |
|---|---|---|---|---|---|
| BES network (retrained) | 0.743 | 0.967 | 0.592 | 0.657 | 0.740 |
| Diagnostic-summary baseline | 0.851 | 0.981 | 0.742 | 0.806 | 0.845 |

The paired macro-F1 difference (BES minus baseline) is -0.105, 95 % shot bootstrap
[-0.34, +0.13].

**Agreement-only windows** (intervals where Gill's and Butt's tables agree; 110 shots, 54,817 windows):
macro F1 0.589 [0.50, 0.67]; per class
L 0.354, H 0.943, QH 0.575, WP 0.485. The single-source intervals are all Gill's BES-time files
(5,423 of the 8,209 L-mode windows, 6,252 of the 47,373 H-mode, 40 WP QH, no QH): the network scores them better than the intervals
the two tables share, which is why L-mode drops from 0.73 to 0.35 here.

**Within-shot split (a leaky diagnostic, not a benchmark number).** The same windows with 0.2 s blocks of each
shot assigned at random to train, validation and test (60/20/20), so a test window has training windows of
its own shot beside it: macro F1 0.927 on 13,397 test windows (L 0.95, H 0.99,
QH 0.85, WP 0.91). `train --protocol blocks` reproduces it.

## Reading them

- Across shots the recipe reaches macro F1 0.68 [0.60, 0.75] against the paper's 0.94. H-mode
  is close to the paper (F1 0.94 against 0.97); L-mode is 0.73; QH and WP QH are weak (0.56
  and 0.47) and are mostly taken for each other (1,535 of 5,608 QH windows called WP, 2,116 of 5,342 WP windows
  called QH), though their one-vs-rest AUROC is 0.94 and 0.87.
- The same pipeline scores 0.93 when test windows share shots with training windows, near the paper's 0.94, so the
  pipeline is not what holds the shot-grouped score down: it is generalising to a shot never seen, with 119 shots to learn from
  (QH on 17, WP QH on 20). The excerpt did not say how the paper split its data; its full text says by discharge,
  stratified by regime. [The ablation below](#the-gap-to-the-published-score-an-ablation) tests this with 444 shots and the
  paper's data selection: 0.72 held out by shot, 0.83 without the beam gate.
- Beside the four-class diagnostic-summary baseline the BES network is lower on these bins (0.74 against 0.85), but the interval of
  the paired difference includes zero, and only 2 QH and 4 WP shots are in it.
- Gill and Butt are not independent annotators (their tables agree exactly where both cover a shot), and two thirds of
  the L-mode windows come from Gill's BES-time files alone, which no second table confirms.

## The gap to the published score: an ablation

Status: **run 2026-10-02 to 2026-10-03; no one difference in protocol closes the gap, more shots
help most, and the score stays 0.22 below the paper's.** The first retrain scored 0.678 against the
paper's 0.94. The paper's full text (a digest of Gill et al. 2024 in the group's literature
folder) states the data selection, the split and the optimiser that the pasted excerpt left out,
so the retrain was rerun with each of them put in, on its own and cumulatively, scoring every
row the same way. Script: `scripts/labeler/confinement_bes_ablation.py` (stages `consolidate`,
`run`, `summarize`, `table`); library: `src/labeler/confinement/bes_{windows,protocol}.py`;
records: `outputs/labeler/confinement/bes/ablation.json` (all rows) and
`ablation_rows/<row>.json` (one per row, with its configuration and training log); confident
learning: `scripts/labeler/confinement_bes_confident.py`,
`outputs/labeler/confinement/bes/confident_{base,only_ge}.json`; features, fold networks and
out-of-fold predictions: `$LABELER_ROOT/round4/conf/{datasets,ablation}/`. The cohort's two blind
test shots (190857, 192756) are never read; the first retrain's 119 shots included them, so its
rerun here (`base`, 117 shots) differs from it a little.

### What was varied

The paper (330 discharges 2012-2023, beam emission spectroscopy at 1 MHz; a test set of 44
discharges, 7 to 17 per class) selects and splits its data differently from the first retrain in
the ways below. Each row of the table that follows adds some of them; the letters name them.

| | Factor | First retrain | Paper's protocol, as put in |
|---|---|---|---|
| a | Beam gate | none | the 150L beam at or above 700 kW and the 150R beam at or below 200 kW over the whole window (the BES views the 150L beam; 150R spoils the view); WPQH needs only 400 kW in training, never in test; a window with no beam record fails |
| b | Transition exclusion | none | no window within 20 ms of either end of its interval (the paper ends segments "slightly before" a transition; the 20 ms is ours), none in the 100 ms after an L-mode interval that an H-class interval follows within 10 ms (the paper drops the post-L-H pedestal build-up, 480 ms in its one example; the 100 ms is ours) |
| c | Optimiser | AdamW, weight decay 1e-2, rates 1e-3 (convolution) and 1e-4 (MLP) | Adam, weight decay 1e-3, rates 1e-3 and 1e-5 |
| d | Channel check | none | a shot whose 6 x 8 block has more than 10 % dead channels (median window power below 1 % of the block's) is left out |
| r | Rows | rows 1-6 of the 8 x 8 array | rows 0-5 (the paper truncates 8 x 8 shots to the first six) |
| g | Sampling | 500 kHz corpus BES: a window is 2.05 ms | the native 1 MHz: a window is 1.02 ms |
| e | Shots | the 117 curated shots the corpus holds BES for | every curated shot with BES, fetched at 1 MHz: 444 shots (L 125, H 176, QH 196, WPQH 78); 171438 and 175658 have none in the archive. The extra shots exist only at 1 MHz, so `only_ge` tests e together with g |
| f | Split | shot-grouped 5-fold | by discharge, stratified by each shot's dominant regime, 72.5 / 15 / 12.5 %: five random repeats, the test windows of the repeats pooled |

Windows are 1,024 samples, one every 2,048, wholly inside one interval of one regime. Beam
power comes from the corpus `pinj` or, where it has none, from the raw cache or a fetch
(`scripts/labeler/confinement_zerod_fetch.py`); shot 175658 has no beam record, so it fails the gate.
Dropout 0.2, batch 256, the 60,000-step cap, early stopping after 30 evaluations (every 500
steps) without a better validation loss, and the best-validation-macro-F1 checkpoint are those of
the first retrain in every row. Two further rows are diagnostics: `cum_abcdrf` is the paper's
split on the 500 kHz corpus shots (no fetch needed), and `leak_*` rows draw the same fractions over
0.2 s blocks of windows of every shot, so test windows have training windows of their own shot
beside them (a leaky split, run to show what that does, not a benchmark).

### How a row is scored

Per window, out of sample (a shot is scored by networks that never saw it), with the macro F1 of
the four classes and a 95 % interval from a shot bootstrap (1,000 replicates, seed 20261001).
Every row is scored twice: on its own windows (those its gate and margins keep), and on **the
paper's windows**, the common population every row can be compared on (beam gate, 20 ms inside
the interval, 100 ms after L-mode). `margin_sensitivity` in each row's JSON rescored the own
windows at 0, 20, 50, 100 and 200 ms inside the interval ends: the macro F1 moves by at most
0.05 over that range in every row except `only_a` (0.07) and `cum_abcdrf` (0.40 to 0.53, 37
shots). The paired difference between two rows on the same shots is
`bes_protocol.paired_difference`. Macro AUROC and AUPRC (one against the rest, class means, shot
bootstrap) are in the JSON of the rows run with `--rank`. No row was chosen, tuned or dropped by
its score: every row in the script's chain was run and is reported. The protocol-matched row for
the label paper's comparison table (`cum_abcdrgef`) was named before any all-shot row ran.

### Results

| Row | Configuration | Shots | Windows | Macro F1 [95 % CI], own windows | Macro F1 on the paper's windows | F1 L / H / QH / WPQH (own) | Shots L / H / QH / WPQH (own) |
|---|---|---|---|---|---|---|---|
| `base` | first retrain (shot-grouped 5-fold, no gating) | 117 | 65,052 | 0.717 [0.63, 0.78] | 0.599 [0.44, 0.71] | 0.74 / 0.93 / 0.53 / 0.67 | 38 / 74 / 16 / 20 |
| `only_a` | + beam gating | 95 | 43,584 | 0.610 [0.46, 0.74] | 0.555 [0.39, 0.69] | 0.42 / 0.93 / 0.49 / 0.59 | 21 / 68 / 11 / 11 |
| `only_b` | + transition exclusion | 109 | 58,258 | 0.671 [0.56, 0.76] | 0.541 [0.39, 0.63] | 0.67 / 0.94 / 0.49 / 0.59 | 35 / 73 / 16 / 12 |
| `only_c` | + paper optimiser | 117 | 65,052 | 0.662 [0.57, 0.73] | 0.523 [0.38, 0.64] | 0.69 / 0.93 / 0.46 / 0.58 | 38 / 74 / 16 / 20 |
| `only_d` | + channel check | 111 | 60,278 | 0.705 [0.61, 0.78] | 0.574 [0.42, 0.69] | 0.71 / 0.92 / 0.53 / 0.66 | 38 / 68 / 16 / 20 |
| `only_r` | + paper rows (first six) | 117 | 65,052 | 0.650 [0.55, 0.73] | 0.453 [0.33, 0.57] | 0.72 / 0.93 / 0.44 / 0.52 | 38 / 74 / 16 / 20 |
| `only_g` | + native 1 MHz | 117 | 130,071 | 0.713 [0.62, 0.78] | 0.552 [0.40, 0.67] | 0.77 / 0.93 / 0.52 / 0.63 | 38 / 74 / 16 / 20 |
| `only_f` | + paper split protocol | 51 | 36,800 | 0.810 [0.69, 0.89] | 0.706 [0.54, 0.81] | 0.80 / 0.95 / 0.75 / 0.75 | 17 / 34 / 6 / 9 |
| `only_ge` | + native 1 MHz and every fetched shot | 444 | 446,426 | 0.828 [0.80, 0.86] | 0.780 [0.72, 0.84] | 0.86 / 0.91 / 0.84 / 0.70 | 125 / 176 / 196 / 78 |
| `cum_a` | + beam gating | 95 | 43,584 | 0.610 [0.46, 0.74] | 0.555 [0.39, 0.69] | 0.42 / 0.93 / 0.49 / 0.59 | 21 / 68 / 11 / 11 |
| `cum_ab` | + transition exclusion | 90 | 40,041 | 0.520 [0.35, 0.65] | 0.520 [0.35, 0.65] | 0.25 / 0.94 / 0.45 / 0.44 | 16 / 67 / 11 / 6 |
| `cum_abc` | + paper optimiser | 90 | 40,041 | 0.489 [0.36, 0.62] | 0.489 [0.36, 0.62] | 0.33 / 0.95 / 0.44 / 0.24 | 16 / 67 / 11 / 6 |
| `cum_abcd` | + channel check | 84 | 39,645 | 0.486 [0.34, 0.62] | 0.485 [0.34, 0.62] | 0.32 / 0.94 / 0.43 / 0.25 | 16 / 61 / 11 / 6 |
| `cum_abcdr` | + paper rows | 80 | 37,045 | 0.433 [0.31, 0.57] | 0.435 [0.31, 0.56] | 0.23 / 0.93 / 0.38 / 0.19 | 16 / 57 / 11 / 6 |
| `cum_abcdrg` | + native 1 MHz | 80 | 74,263 | 0.416 [0.29, 0.55] | 0.417 [0.29, 0.55] | 0.15 / 0.91 / 0.37 / 0.23 | 16 / 57 / 11 / 6 |
| `cum_abcdrge` | + every fetched shot | 311 | 232,357 | 0.731 [0.66, 0.80] | 0.706 [0.64, 0.77] | 0.82 / 0.92 / 0.85 / 0.34 | 72 / 122 / 153 / 18 |
| `cum_abcdrgef` | + paper split protocol | 146 | 154,468 | 0.716 [0.61, 0.81] | 0.716 [0.61, 0.81] | 0.71 / 0.89 / 0.80 / 0.46 | 35 / 57 / 70 / 8 |
| `cum_abcdrf` | + paper split protocol (500 kHz, corpus shots) | 37 | 24,942 | 0.483 [0.28, 0.63] | 0.483 [0.28, 0.63] | 0.32 / 0.93 / 0.30 / 0.38 | 5 / 26 / 6 / 5 |
| `leak_abcdr` | within-shot block split, paper criteria (500 kHz; leaky diagnostic) | 75 | 24,249 | 0.960 [0.93, 0.98] | 0.959 [0.92, 0.98] | 0.93 / 1.00 / 0.96 / 0.95 | 13 / 55 / 10 / 4 |
| `leak_abcdrge` | within-shot block split, paper criteria, every shot at 1 MHz (leaky) | 304 | 144,059 | 0.950 [0.93, 0.97] | 0.933 [0.90, 0.95] | 0.95 / 0.99 / 0.98 / 0.88 | 67 / 120 / 150 / 16 |

Macro AUROC and AUPRC [95 % CI] of the rows run with `--rank` (shot bootstrap):

- `base`: AUROC 0.941 [0.911, 0.967], AUPRC 0.741 [0.643, 0.834]
- `only_f`: AUROC 0.975 [0.952, 0.989], AUPRC 0.832 [0.710, 0.918]
- `only_ge`: AUROC 0.955 [0.940, 0.969], AUPRC 0.864 [0.821, 0.908]
- `cum_abcdr`: AUROC 0.827 [0.732, 0.917], AUPRC 0.409 [0.331, 0.569]
- `cum_abcdrge`: AUROC 0.912 [0.869, 0.951], AUPRC 0.730 [0.684, 0.809]
- `cum_abcdrgef`: AUROC 0.942 [0.908, 0.973], AUPRC 0.740 [0.647, 0.867]
- `cum_abcdrf`: AUROC 0.856 [0.704, 0.940], AUPRC 0.441 [0.302, 0.635]
- `leak_abcdr`: AUROC 0.997 [0.993, 1.000], AUPRC 0.976 [0.947, 0.995]
- `leak_abcdrge`: AUROC 0.997 [0.994, 0.999], AUPRC 0.981 [0.966, 0.990]

Rows named `only_*` add one factor to `base`; `cum_*` add them in the order of the letters. The
"shots" column is the shots with a scored window of that class, so the gate's cost reads off it.

### Reading it

- **No single factor closes the gap, and the paper's selection rules lower the score on our
  labels.** Scored on the paper's windows, the one population every row shares (90 shots), the
  117-shot `base` is 0.599 [0.44, 0.71]; adding the gate, the margins, the optimiser, the rows,
  the channel check or 1 MHz one at a time gives 0.555, 0.541, 0.523, 0.453, 0.574 and 0.552:
  none higher, none outside the base's interval. Stacked (`cum_abcdr`, 80 shots) they give 0.435
  [0.31, 0.56]; on each row's own windows the base is 0.717 [0.63, 0.78] and the stack 0.433
  [0.31, 0.57]. L-mode carries the fall: its F1 is 0.74 in the base, 0.42 with the gate alone and
  0.23 in the stack.
- **The gate removes most of the windows of the two classes the paper's selection suits least**
  (`gate_breakdown.json`, from `scripts/labeler/confinement_bes_gate_breakdown.py`). On the 117
  corpus shots 28 % of the L-mode windows pass it (52 % have the 150L beam under 50 kW, ohmic or
  beam-off phases; 20 % lie between 50 and 700 kW), against 75 % of H, 73 % of QH and 54 % of WPQH.
  On all 444 shots 54 % of L, 67 % of H, 68 % of QH and 24 % of WPQH windows pass: 68 % of the WPQH
  windows have the 150L beam under 50 kW and 53 % have 150R over 200 kW, so the gated WPQH
  population is 18 of 78 shots at test. Without a gate the first retrain was in part a beam
  detector (windows without a beam are easy to tell from the others); the paper scores
  beam-on time only.
- **Shots are the largest single factor.** Adding the 327 fetched shots (e) lifts the stacked
  recipe from 0.416 [0.29, 0.55] to 0.731 [0.66, 0.80] (`cum_abcdrg` to `cum_abcdrge`; on the
  paper's windows 0.417 to 0.706 [0.64, 0.77]) and the ungated base from 0.713 to 0.828
  [0.80, 0.86] (`only_g` to `only_ge`; its interval was [0.62, 0.78]). L-mode (0.82), H (0.92) and QH (0.85) are then 0.05 to 0.12 below the paper's per-class
  scores (0.94, 0.97, 0.94); WPQH (0.34 [0.09, 0.56], 18 shots, against the paper's 0.90) is far
  below, and holds the macro F1 down.
- **The paper's split by discharge does not by itself raise a shot-held-out score.** `only_f`
  is 0.810 [0.69, 0.89] on 51 shots against `base` 0.717 [0.63, 0.78], inside each other's
  intervals; at full data the five random splits (`cum_abcdrgef`, 0.716 [0.61, 0.81], 146 shots)
  agree with the 5-fold of the same recipe (`cum_abcdrge`, 0.731 [0.66, 0.80]). A test set of
  about 15 to 40 shots is as noisy as the paper's 44, whose per-class scores rest on 7 to 17 shots.
- **A within-shot split is what reaches the published number.** With the same recipe on the 117
  corpus shots, cutting 0.2 s blocks at random (`leak_abcdr`) gives 0.960 [0.93, 0.98] and
  macro AUROC 0.997, against 0.483 [0.28, 0.63] when the shots are held out (`cum_abcdrf`).
  With every shot at 1 MHz the same split (`leak_abcdrge`, 304 shots, 144,059 windows) gives
  0.950 [0.93, 0.97] (L 0.95, H 0.99, QH 0.98, WPQH 0.88; macro AUROC 0.997, AUPRC 0.981),
  within 0.04 of each of the paper's per-class scores (0.94, 0.97, 0.94, 0.90), against 0.716
  [0.61, 0.81] for the same recipe held out by shot (`cum_abcdrgef`). The recipe reaches the
  published figure exactly when test windows share shots with training windows, and not
  otherwise. The paper says it split by discharge, so this is not claimed as the paper's
  protocol; it says that what separates 0.95 from 0.72 here is generalising to a discharge the
  network has not seen, and that if the paper's test discharges were independent of its
  training ones, something other than the pipeline (labels, selection, unstated settings)
  differs.
- **Label noise is real but small, and unlikely to be the whole 0.2.** Confident learning (below)
  flags 7.7 % of the intervals as probably another class, among them 14 of the 418 QH and
  WPQH intervals (3.3 %) as the other of the two. A model scored on the intervals it agrees with scores
  higher by construction, so no such number is reported.

### The protocol-matched score for the comparison table

The score to set beside the paper's 0.94 is **`cum_abcdrgef`: macro F1 0.716 [0.61, 0.81]**
(per class L 0.71, H 0.89, QH 0.80, WPQH 0.46 on 35, 57, 70 and 8 shots; macro AUROC 0.942
[0.908, 0.973], macro AUPRC 0.740 [0.647, 0.867]; 154,468 windows on 146 test shots over five
random splits by shot). It is the paper's protocol on our labels: beam gate, margins, optimiser,
rows, channel check, native 1 MHz, every fetched shot, split by discharge and held out by shot; it
was named before the all-shot rows ran, and no row was selected after. The first retrain
(0.678 [0.60, 0.75] on 119 shots, 0.717 rerun on 117) is the same network without the protocol,
and `only_ge` (0.828 [0.80, 0.86]) the same without the gate.

It is 0.22 short of the paper's 0.94, outside the 0.05-0.08 a protocol difference could explain,
so the comparison table must carry the remainder in words. What the ablation accounts for: the
shots (0.42 to 0.73 for the stacked recipe), the gate (it removes 46 % of the L-mode and 76 % of
the WPQH windows, and WPQH is where the score is lowest), and the split, but only if test windows
share shots with training windows (0.95). What it does not account for: the paper selects 330
discharges for beam-on BES with a 6 x 8 layout and labels them "standard ELMy H-mode" with H98y2
of at least 1, segments ended before every transition, checked against the logbook, whereas our
intervals merge two annotators' tables with a wider H class and 7.7 % of their intervals look
mislabelled (below); WPQH under the gate has 8 test shots here and 7 in the paper. Dropout, batch
size and the window stride were not stated and were not tuned. The remainder is therefore
attributed to labels, selection and unstated settings; no row here tests the labels directly. The
next test is to restrict the H-mode intervals to H98y2 of at least 1 and rescore (not done).

### Confident learning: intervals the BES network doubts

A label-noise estimate in the manner of Northcutt et al. (2021), written here and applied to
intervals (no code from their repository, `src/labeler/confinement/confident.py`): a class's
threshold is the mean out-of-fold probability of the windows labelled with it; a window is
confidently of the class with the largest probability among those at or above their thresholds;
an interval is flagged when it holds at least 5 kept windows and at least half of them are
confidently one other class. Out-of-fold probabilities come from `only_ge` (every window of 432
shots is predicted by a network that never saw its shot; windows within 20 ms of an interval
end are left out).

- **72 of 931 intervals (7.7 %) are flagged**, 6.7 % of the 746 where Gill's and Butt's tables
  agree and 11.9 % of the 185 from one source. The record
  (`confident_only_ge.json`) lists the 60 most doubted; all 72 are in
  `$LABELER_ROOT/round4/conf/confident/intervals_only_ge.csv`. `confident_base.json` is the same
  on the 117 shots of the first retrain (25 of 251 intervals).
- **QH and WPQH**: 14 of the 418 intervals (3.3 %) are taken for the other of the two (QH read
  as WPQH 7 of 299, WPQH as QH 7 of 119), as the confusion matrices already show. The list
  (shot, interval in ms): QH read as WPQH, 163456 (1500-2500), 174656 (1315-1475), 192711
  (1500-1850), 192710 (1250-1500), 192763 (1175-1841), 169865 (1312-1400), 184822 (1050-1235);
  WPQH read as QH, 161608 (3741-5461), 169848 (1302-1664), 174653 (4363-4799 and 4130-4320),
  174675 (1610-3527), 184968 (2947-3180 and 3185-3260). A reviewer should look at the long ones
  first (161608, 163456 and 174675 each span about 1 s or more).
- The largest group is **QH read as H**: 20 of 299 QH intervals (6.7 %), an interval in which the
  edge oscillation may be weak; L read as H 6 of 218, H as L 2 of 295, H as QH 3 of 295.
- **A second opinion from `confine-ours`**, which reads the 0D signals and not BES
  ([confinement_ours.md](confinement_ours.md)): of the 72 flagged intervals 6 are also
  mostly (over half) not the labelled class by `confine-ours` (4 of them QH/WPQH; 5 of the 6 give the
  same other class as the BES network): shots 154482, 161608, 169848, 174653, 192710 and 192718.
  Those are the likeliest label errors; the other 66 are doubted only by the BES network.
  `confine-ours` is not independent of the labels (it learned from them), so the opinion is
  weak where it agrees with them.
- None of this edits a label. The flagged intervals are for the review page.

### What the ablation shows for the benchmark

The 0.68 of the first retrain was a small-data, no-protocol number. With the paper's protocol and
every shot the recipe reaches 0.72 [0.61, 0.81] on discharges it never saw; without the beam gate,
0.83 [0.80, 0.86]; with test windows allowed to share discharges with training windows, 0.95
[0.93, 0.97]. The paper's 0.94 is matched only in the last case here, which measures what the
recipe can fit, not how it generalises: the first number is the one to report.

## Not done

- The H-mode definition: the paper's H-mode is ELMy H-mode with H98y2 of at least 1; ours is the experts'
  wider class. Restricting the H intervals to H98y2 of at least 1 (EFIT confinement time) and rescoring would
  test whether the label convention explains the rest of the gap. Not run.
- Anything the paper leaves open was set once and not tuned: dropout, batch size, window stride, class
  weights. Every row is one seed per fold or split; a second seed would show the run-to-run spread, which the
  bootstrap intervals (over shots, not over seeds) do not include.
- Two curated shots have no BES in the archive (171438, 175658), so the 444 fetched shots are all there is.
- The leaky rows (`leak_*`) are diagnostics of the split, not benchmark numbers; the 0.2 s blocks leave
  neighbouring windows of one shot on both sides of the split by design.
- `confident_*.json` lists the 60 most doubted intervals only; the other flagged intervals are in
  `$LABELER_ROOT/round4/conf/confident/`. No label was changed.

## Source excerpt, as pasted

Web-page controls ("Zoom In", "Download figure", ...) are removed. Equations,
Greek symbols and some numbers were lost in the paste; they are left as gaps.

> **3. Neural network architecture and training**
>
> **3.1. Setup**
> Our goal is to identify the global confinement state (e.g. L, H, QH, or WP QH) from the evolution of local density fluctuations that exist within the real-time BES data stream. Formally, this amounts to a multivariate time-series classification task, which can be thought of as finding a mapping from , where X represents a 3D block of BES data and Y is its corresponding probability distribution over the different confinement regimes. Our approach is to find a classifier (neural network) parameterized by weights and biases θ, such that  is a good approximation of Y.
>
> In order to do this, we train our classifier on data X and corresponding labels Y via an optimization procedure. Specifically, we seek to minimize the following objective function with respect to θ:
>
> [equation lost]
>
> where the inner sum iterates over the K = 4 classes to compute the cross-entropy loss [33], which measures the dissimilarity between the true distribution of labels Ynj and the predicted probability distribution output by the model. The outer sum explicates averaging the loss over a mini-batch of size N (batch-size) of 3D BES data blocks from the training set. The parameters of the network are updated after the completion of a forward pass of the entire mini-batch via gradient descent:
>
> [equation lost]
>
> where the learning rate α controls the step size used by the optimizer. We save the model whose performance on the validation set is most optimal. We provide further practical details of our implementation in the following section 3.3.
>
> **3.2. Preprocessing**
> Figure 4 illustrates the general workflow for data-driven real-time confinement-regime classification in DIII-D, from data collection (figure 4(a)) to model classification (figure 4(e)). The input to the workflow is a  block of BES data (see figure 4(b)), where 1024 corresponds to a  signal window size, and  refers to the spatial configuration of BES consisting of 6 (radially-spanning) rows and 8 (poloidally-spanning) columns. We explore differing input dimensions in section 4.1. We perform the following preprocessing procedure (figure 4(c)). First, we apply a bandpass filter of  to the signal to filter out unwanted and/or irrelevant information such as the low-frequency component due to the neutral beam and the high-end excess photon and electronic noise [44]. As mentioned earlier, broadband turbulence characteristic of L-mode plasmas can be found in this frequency range. The EHO, signature of QH plasmas, is a low toroidal (n) mode number MHD oscillation also found in this frequency range (), and leaves a trace on the density fluctuations made visible by BES. Similarly, in transition to WP QH, this EHO is often lost to edge broadband MHD, of which associated density fluctuations are visible too with edge channels of BES [23]. It is worthy to note that we experimented with broadening and narrowing this frequency range and saw no major improvements to the overall performance and therefore justify our preprocessing selection empirically as well.
>
> Figure 4. Workflow for data-driven real-time confinement regime classification using high-bandwidth fluctuation measurements in DIII-D. (a) Cross-section of DIII-D tokamak, showing nested magnetic flux surfaces and highlighting 2D BES array location. The BES array is shown in  configuration, which we utilize in this work; i.e. in this example, we utilize signals only from turquoise channels, and ignore red channels. (b) Extract signal window of 2D BES array, containing information about density fluctuations and pedestal turbulence dynamics. (c) Raw signals undergo preprocessing by applying a 2.5–150 kHz bandpass filter, standardizing, and further FFT operations (more detail in figure 5). (d) Neural network consisting of shallow convolutional network and multi-layer perceptron (MLP). (e) Model outputs confinement regime with highest probability for given input BES data.
>
> We then standardize the frequency-filtered signals using the mean and standard deviation of the training set to remove shot-to-shot variation of the signals, and to avoid outliers caused by extraneous events such as beam modulations of the NBI. In order to allow for an easier extraction of this meaningful information, we then utilize the Fourier-transformed signals via the fast Fourier transform [58] (FFT). The signal window is split into 2 sub-windows of length 512  to promote the model learning temporal evolution in the data stream. Then, we further split this sub-window in 2 and perform the FFT on both segments of 256 . We retain only the magnitudes (i.e. absolute value of complex-valued FFT), and take the logarithm of the squared magnitudes to avoid exploding and vanishing gradients [59] during neural network training. Finally, we average these 2 segments to reduce noise, resulting in 128 (i.e. due to symmetry about Nyquist frequency) unique spectral-like features for each 512  window. In conclusion, for each channel in the BES array, we recast our original 1024  time-series data into , the latter of which is the only input to the neural network. This way, the network can extract frequency information and still learn temporal evolution across the 2 distinct sub-windows. Figure 5 visualizes this preprocessing technique for one of the 48 signals in the BES array. As these choices are partly empirically motivated, we explore different values for this preprocessing procedure in appendix.
>
> Figure 5. Preprocessing procedure illustrated for 1 channel of BES array. (a)  time window of raw signal from BES channel. (b) Signal after applying 2.5–150 kHz bandpass filter and dividing into first and second half. (c) Subsequent FFT power spectra for different time windows in signal. (d) Result after squaring, then taking the base-10 logarithm of magnitudes to keep values close to unity. (e) Final result is 2 sets of spectral-like features for 2 halves corresponding to first and second  of original signal. For each channel, (e) is fed to neural network.
>
> **3.3. Neural network architecture and training**
> The neural network consists of a convolutional layer followed by a shallow multi-layer perceptron (MLP). In the convolutional layer, we follow the standard procedure of dropout [60], convolution [61], batch normalization [62], LeakyReLU activation [63], and max-pooling, where we note that the convolution and pooling operations are three-dimensional (3D). For the 3D convolution, we used 10 kernels of shape (3, 3, 5) with stride (1, 1, 1) acting in the radial, poloidal, and frequency dimension, respectively, with zero padding, and groups = 2 (number of sub-windows). For the 3D max-pooling operation, we used a kernel of shape (1, 2, 4) acting in the radial, poloidal, and frequency dimension, respectively, with zero padding. The features output by the convolutional layer are then flattened and serve as input to an MLP with 2 hidden layers each of size 60, utilizing LeakyReLU activation. The output layer is of size 4, where a final softmax [64] operation converts the raw logits from the network into a probability distribution over the different confinement regimes.
>
> The model is optimized using the Adam optimizer [65], with weight decay [66] , , and . The learning rate is set to  for the parameters in the convolutional layer and  for the parameters in the MLP. This is to promote stability in the MLP, with far more parameters than in the convolutional layer (i.e.  parameters versus parameters). We set the training process to terminate after 60 000 optimizer steps, with an early stopping mechanism in place that would halt training if the validation loss did not improve over 30 consecutive evaluations. However, the training concluded at the 60 000 step mark, as the early stopping criterion was not met. We saved the model whose F1 score is highest on the validation set, and used that for testing the model. Our framework was implemented using PyTorch Lightning 1.6.5.
>
> Despite the relatively small architecture used (i.e.  K parameters), the sheer volume of data (i.e. ) necessitates parallel processing to manage the high data throughput efficiently, distribute the workload, and maintain a reasonable training time. Therefore, we utilized distributed training using 48 Nvidia A100 Tensor Core Graphics Processing Units (GPUs) on the Perlmutter supercomputer at the National Energy Research Scientific Computing Center (NERSC), resulting in a single training run taking ∼5–10 h.
>
> **3.4. Metrics for evaluation**
> In order to accurately gauge the performance of the network, we report recall, precision, and F1 score:
>
> [equations lost]
>
> where TP, FP, and TN refer to true positives, false positives, and false negatives, respectively. Recall can be intuited as the true positive rate (TPR) (e.g. out of all true L-modes, how many cases does the model classify correctly), whereas precision can be intuited as the positive predictive power of a model (e.g. out of all the cases where the model predicts L-mode, how many of these predictions are correct). In classification tasks, it is common to use the F1 score, i.e. the geometric mean of the precision and recall, to accurately assess performance in instances where there exists a large class imbalance, such as in our case with BES data belonging to L-mode or WP QH-mode being much more scarce than BES data belonging to H-mode or QH-mode (see table 1).
>
> Also common to binary and multi-class classification tasks is the confusion matrix, a square matrix whose shape is determined by the number of classes in the task. The confusion matrix illustrates the classifications made by the model, in a way such that correct classifications fall along the diagonal, while the off-diagonal elements reflect signs of confusion for the model. A perfect classifier, therefore, would be represented by a purely diagonal confusion matrix. Looking along rows one can infer recall values while looking along columns one can infer precision values.
>
> Finally, we use the receiver operating characteristic (ROC) curve to evaluate the performance of our model. The ROC curve is generated by varying the probability threshold at which we classify an input as belonging to a particular class, which in turn affects the TPR and false positive rate (FPR). An optimal classifier is found by its proximity to the top-left corner of the ROC space, representing a perfect classifier (i.e. FPR = 0, TPR = 1). The area under the ROC curve (AUC) is a measure of the model's overall performance, with AUC = 0.5 indicating performance no better than random chance and AUC = 1.0 indicating perfect classification. To extend this binary classification evaluation to our multi-class scenario, we use the same multi-class model and evaluate its performance on binary classification tasks by considering one class against the rest (e.g. L-mode vs. rest, H-mode vs. rest). This approach allows us to assess the separability and performance of each individual class within the multi-class framework.
