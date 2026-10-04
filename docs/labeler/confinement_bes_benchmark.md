# Confinement benchmark: the BES classifier

Status: **fix round 4, 2026-10-04. Under the paper's selection, split, optimiser, architecture
and training length our reimplementation scores macro F1 0.703 [0.59, 0.79] on discharges held out by shot, against the
paper's 0.94; the remainder is unexplained** (section "The gap to the published score"). The 0.703
pools five random by-discharge splits whose test sets overlap: 142 distinct shots in 200 shot-tests,
106,232 distinct windows of 152,863 pooled; one split like the paper's scores 0.684 +/- 0.074, each
shot counted once 0.736 and probabilities averaged over repeats 0.747 (section "Overlap of the five
test draws").
The owner named the BES-based confinement classifier as the benchmark for the confinement labels,
the way the CO2 LSTM and RCN are for the AE labels (2026-10-01). Its weights and code were not
found, so the recipe below was retrained on the corpus BES and scored on shots the network never
saw. Script: `scripts/labeler/confinement_bes_benchmark.py` (stages `features`, `train`,
`evaluate`); record: `outputs/labeler/confinement/bes/evaluation.json` (git `2b4da1d`); features,
fold models and out-of-fold predictions: `$LABELER_ROOT/benchmarks/confinement/bes/`.

The first retrain's sections ("What the paper's model is" to "Reading them") stand as run on
2026-10-01; it read the cohort's two blind test shots (190857, 192756), which no later work
reads, so its rerun without them (117 shots) is the ablation's `base`. The gap to the published
score was taken apart by an ablation on 444 shots fetched at 1 MHz (2026-10-02 to 2026-10-03) and
redone with the paper's architecture, training length and a block chosen from the channel
positions (2026-10-03, fix round 2). The first retrain's explanation of the gap ("a small-data
number") did not survive the fixed-population rescoring below and is withdrawn. Fix round 3
(2026-10-04) disclosed the overlap of the five test draws behind the Table 3 number, restated the
label hypothesis (our labels are Gill's own), added a per-shot failure list and the geometry
exceptions; nothing was retrained. Fix round 4 (2026-10-04) states the paper's per-class AUC of at
least 0.99 beside our macro AUROC, reads the by-year and BES-time-file comparisons per class, and
marks what our H class contains against the paper's as unverified; nothing was retrained.

Citation: K. Gill, D. Smith, S. Joung, B. Geiger, G. McKee, J. Zimmerman, R. Coffee,
A. Jalalvand and E. Kolemen, "Real-time confinement regime detection in fusion plasmas with
convolutional neural networks and high-bandwidth edge fluctuation measurements", Machine
Learning: Science and Technology 5, 035012 (2024), DOI 10.1088/2632-2153/ad605e (the reference as
cited in OSTI 2587945 [28] and 3030081 [63] of the group's literature folder).

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

The paper's "zero padding" is read two ways: padding with zeros (a same-size output, 465,244
parameters for 6 rows; what the first retrain built) or none (`padding=0` in PyTorch: 227,644
parameters). The paper's parameter count was lost in the paste. The first retrain and the ablation rows
without the `full_` prefix pad; the `full_*` rows do not, and
`tests/labeler/test_confinement_bes_cnn.py` asserts both counts.

**Not in the paste** (equations and symbols were lost): the dropout probability, both learning
rates, the weight decay and Adam betas, the batch size, the parameter counts, the sampling rate and
the window duration in time, the channel selection for the 6 x 8 block, the class counts (its
table 1) and the train/validation/test split. The first retrain had to assume them. The
paper's full text (the group's literature folder) later supplied most: Adam with weight decay 1e-3,
learning rates 1e-3 (convolution) and 1e-5 (MLP), the native 1 MHz, 8 x 8 shots truncated to the first
six rows, a split by discharge, the data selection of factors a to d below, and its table 1: the
labelled seconds and discharges per class (all data: L 60.0 s on 86 discharges, H 196.6 s on 102,
QH 148.1 s on 140, WP QH 35.6 s on 38; its test set: 8.3 s on 11, 21.8 s on 13, 17.4 s on 17 and
9.7 s on 7, which the class-mix reweighting below uses as `PAPER_TEST_MIX`). Still not stated, and
set once and left untuned in every row here: the dropout probability (0.2), the Adam betas, the batch
size (256) and the window stride (2,048 samples).

## Published results (as pasted)

| Confinement regime | F1 | Precision | Recall |
|---|---|---|---|
| L-mode | 0.94 | 0.98 | 0.90 |
| H-mode | 0.97 | 0.95 | 0.98 |
| QH-mode | 0.94 | 0.92 | 0.96 |
| WP QH-mode | 0.90 | 0.95 | 0.86 |
| Average | 0.94 | 0.95 | 0.93 |

The paper also reports a one-vs-rest ROC AUC of at least 0.99 for every class (no macro value, and
no AUPRC); the four per-class F1 above, their precision and recall and this bound are committed as
`outputs/labeler/confinement/bes/gill_2024_published.json`, which the benchmark figure reads.

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
| Network | dropout, Conv3d (10 kernels (3, 3, 5), zero padding, groups 2), batch norm, LeakyReLU, MaxPool3d (1, 2, 4), MLP 2 x 60, 4 logits: 465,244 parameters (padded; the unpadded network has 227,644) |
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
  pipeline is not what holds the shot-grouped score down: it is generalising to a shot never seen. The excerpt
  did not say how the paper split its data; its full text says by discharge, stratified by regime. [The ablation
  below](#the-gap-to-the-published-score-an-ablation) tests what does hold the score down. It is not the number
  of training shots: the 117 shots of this retrain score the same when 327 more are added to the training
  set (their F1 is 0.705 against 0.713 with the 117 alone, `only_ge` against `only_g`).
- Beside the four-class diagnostic-summary baseline the BES network is lower on these bins (0.74 against 0.85), but the interval of
  the paired difference includes zero, and only 2 QH and 4 WP shots are in it.
- Gill and Butt are not independent annotators (their tables agree exactly where both cover a shot), and two thirds of
  the L-mode windows come from Gill's BES-time files alone, which no second table confirms.

## The gap to the published score: an ablation

Status: **run 2026-10-02 to 2026-10-03, reworked in fix rounds 2 to 4 (2026-10-03 to 2026-10-04). Our
reimplementation under the paper's selection and split scores 0.703 [0.59, 0.79] against the paper's
0.94, and the remainder is unexplained. What the ablation shows: no single protocol factor moves the
score by more than 0.05 on 444 shots; the 117 shots of the corpus score 0.44 to 0.46 in the four rows that apply
the paper's whole selection to all fetched shots, and the other shots 0.74 to 0.83 in the same rows,
still below the paper's 0.94; and a split inside each shot reaches 0.90 to 0.95.** The first retrain scored 0.678 against the paper's 0.94. The
paper's full text (a digest of Gill et al. 2024 in the group's literature folder) states the data
selection, the split and the optimiser that the pasted excerpt left out, so the retrain was rerun
with each of them put in, on its own and cumulatively, scoring every row the same way; in fix round
2 the rows were rescored on fixed populations with paired differences, the single factors were rerun
on 444 shots, factor d was rebuilt from the fetched channel positions, and the rows that decide the
comparison-table number were rerun with the paper's architecture and training length. Script:
`scripts/labeler/confinement_bes_ablation.py` (stages `consolidate`, `geometry`, `run`, `summarize`,
`populations`, `audit`, `table`, `tables`, and, from fix round 3, `repeats`, `subsets`, `failures`,
`tables-protocol`, which print the tables of the last three sections of this page from their
records); SLURM array:
`scripts/labeler/confinement_bes_ablation.sbatch`; library:
`src/labeler/confinement/bes_{windows,protocol,cnn,geometry}.py`; records, all under
`outputs/labeler/confinement/bes/`: `ablation.json` (all rows) and `ablation_rows/<row>.json` (one
per row, with its configuration and training log), `populations.json` (fixed populations, paired
steps, label fragmentation, class mix, years), `validation_audit.json`, `geometry.json` and
`geometry_blocks.csv`, `gate_breakdown.json`, `protocol_repeats.json`, `subsets.json`,
`failure_blocks.json` and `failure_blocks.csv`, `confident_*.json`; features, fold networks and
out-of-fold predictions: `$LABELER_ROOT/round4/conf/{datasets,ablation,geometry}/`. The cohort's two
blind test shots (190857, 192756) are never read; the first retrain's 119 shots included them, so
its rerun here (`base`, 117 shots) differs from it a little.

### What was varied

The paper (330 discharges 2012-2023, beam emission spectroscopy at 1 MHz; a test set of 44
discharges, 7 to 17 per class) selects and splits its data differently from the first retrain in the
ways below. Each row of the tables that follow adds some of them; the letters name them.

| | Factor | First retrain | Paper's protocol, as put in |
|---|---|---|---|
| a | Beam gate | none | the 150L beam at or above 700 kW and the 150R beam at or below 200 kW over the whole window (the BES views the 150L beam; 150R spoils the view); WPQH needs only 400 kW in training, never in test; a window with no beam record fails |
| b | Transition exclusion | none | no window within 20 ms of either end of its interval (the paper ends segments "slightly before" a transition; the 20 ms is ours), none in the 100 ms after an L-mode interval that an H-class interval follows within 10 ms (the paper drops the post-L-H pedestal build-up, 480 ms in its one example; the 100 ms is ours) |
| c | Optimiser | AdamW, weight decay 1e-2, rates 1e-3 (convolution) and 1e-4 (MLP) | Adam, weight decay 1e-3, rates 1e-3 and 1e-5 |
| d | Channel layout | rows 1-6 of the 8 x 8 array, no check | in the early-stopped rows a stand-in, **the dead-channel check** (a shot whose 6 x 8 block has more than 10 % dead channels, median window power below 1 % of the block's, is left out); in the `full_*` rows **the real thing, by geometry** (below): each shot's 6 x 8 block is the one covering the pedestal, and a shot whose array does not reach the separatrix is left out |
| r | Rows | rows 1-6 of the 8 x 8 array | rows 0-5 (the paper truncates 8 x 8 shots to the first six); under geometry the block is chosen per shot |
| g | Sampling | 500 kHz corpus BES: a window is 2.05 ms | the native 1 MHz: a window is 1.02 ms |
| e | Shots | the 117 curated shots the corpus holds BES for | every curated shot with BES, fetched at 1 MHz: 444 shots (L 125, H 176, QH 196, WPQH 78); 171438 and 175658 have none in the archive. The extra shots exist only at 1 MHz, so `only_ge` tests e together with g |
| f | Split | shot-grouped 5-fold | by discharge, stratified by each shot's dominant regime, 72.5 / 15 / 12.5 %: five random repeats, the test windows of the repeats pooled |
| p | Architecture and training length | the convolution padded (465,244 parameters); 60,000 steps allowed, early stopping after 30 evaluations (runs stopped between 15,500 and 60,000 steps, most by 31,000, with best checkpoints as early as step 500) | no padding (227,644 parameters), 60,000 steps, **no early stopping**, the checkpoint with the best validation macro F1 kept; run only in the `full_*` rows |

Windows are 1,024 samples, one every 2,048, wholly inside one interval of one regime. Beam power
comes from the corpus `pinj` or, where it has none, from the raw cache or a fetch
(`scripts/labeler/confinement_zerod_fetch.py`); shot 175658 has no beam record, so it fails the
gate. Dropout 0.2, batch 256 and the window stride are those of the first retrain in every row (the
paper does not state them). Rows named `only_*` add one factor to `base`; `cum_*` add them in the
order of the letters; `full_*` are the `cum_*` rows (or `only_c`) rerun with p and, where marked,
the geometry block; `ge_a`, `ge_b`, `ge_c` and `ge_r` add one of a, b, c and r to `only_ge` (all 444
shots at 1 MHz, shot-grouped 5-fold), where every fold has all four classes in validation, to
replace the underpowered single-factor rows on 117 shots. Three further rows are diagnostics:
`cum_abcdrf` is the paper's split on the 500 kHz corpus shots (no fetch needed), and `leak_*` rows
draw the same fractions over blocks of 100,000 samples (0.2 s at 500 kHz, 0.1 s at 1 MHz) of every
shot, so test windows have training windows of their own shot beside them (a leaky split, run to
show what that does, not a benchmark).

### Factor d by geometry

The channel check of the early-stopped rows was a stand-in (dead channels say nothing about where
the array looks). The paper uses a 6 x 8 block that reaches the edge ("8 x 8 shots truncated to the
first six rows") and loses up to 0.6 F1 when edge channels are missing. Here the block is chosen
from each shot's own channel positions, never from a score
(`src/labeler/confinement/bes_geometry.py`, `scripts/labeler/confinement_bes_geometry_fetch.py`):

1. **Fetch.** For each of the 444 shots, the 64 channels' positions `\BES::BES_R` and `BES_Z` (cm)
   and the EFIT01 flux map (`psirz`, `ssimag`, `ssibry`, `r`, `z`, `gtime`; `efit02` where `efit01`
   holds none) were read on the login node under `fdp run`, 3 workers, `--pace 1`, 60-minute
   timeout, with the stop at the first authentication error (none came; five corrupted-record reads
   were retried). 441 shots returned a record; 185920, 186637 and 186641 have none in the archive
   and are left out. The files are `$LABELER_ROOT/round4/conf/geometry/<shot>.npz` (32 MB in all).
   The cohort's blind shots are not fetched.
2. **Flux of each channel.** Bilinear interpolation of `psi_N` (0 on axis, 1 on the separatrix) at
   each channel and EFIT time; a channel's value for the shot is its median over the shot's labelled
   windows.
3. **Block.** A row of 8 channels covers the pedestal when at least 2 of them have `psi_N` in
   0.85-1.0 (the pedestal and the foot of its gradient); the block is the 6 consecutive rows,
   starting at row 0, 1 or 2, that cover the most rows, the lowest start on a tie (so the paper's
   first six rows stand unless a row is displaced from the pedestal). The constants were fixed
   before any score was read.
4. **Drop.** A shot is left out when the block's outermost channel has median `psi_N` below 0.98:
   the array does not reach the last closed flux surface.

Result (`outputs/labeler/confinement/bes/geometry.json`, one line per shot in
`geometry_blocks.csv`): 401 of 444 shots are kept, 405,629 of 446,426 windows. 43 are dropped: 3
without a position record and 40 whose array does not reach the separatrix (median outer `psi_N`
0.62; for 30 of them no row covers the pedestal at all, for 7 six rows do but the array ends inside
0.98). The block starts at row 0 for 349 shots, row 1 for 37 and row 2 for 15; 324 shots have six
rows covering the pedestal, 61 five; 191 shots have a row displaced by more than 1 cm in major
radius from the others. The dropped shots hold 5 L, 13 H, 26 QH and 4 WPQH shots of the 125 / 176 /
196 / 78 fetched, 13 of them corpus shots; by year of the shot, 2013 2, 2014 11, 2015 2, 2017 6,
2018 6, 2019 1, 2021 6, 2022 8 and 2023 1. The dead-channel stand-in of the early-stopped rows
(`only_d`) left out 6 of the 117 corpus shots.

**Exceptions to the 6 x 8-block claim.** The block is chosen by how far the array reaches and how
many rows cover the pedestal band, never by whether a row is one radial row. Two groups of kept
shots do not fit the claim (stage `subsets`, record `subsets.json`):

- **16 shots with a 4 x 16 layout** (179209, 179211 to 179214, 179216, 179219, 179220, 179223,
  179314, 179321, 179328, 179331, 179333, 179334 and 186230; 23,359 windows): their 8-channel rows
  alternate between an inner and an outer radial half (`rows_displaced` = 8, a Z span of 4.7 to
  5.0 cm against a median of 10.9 cm), so the block is 4 poloidal x 16 radial, not a 6 x 8 grid.
- **2 shots with the block in the scrape-off layer** (157082 and 185918; at most one row in the
  `psi_N` band 0.85-1.0, none for 185918; outermost channel at `psi_N` 1.26 and 1.22): the array
  looks outside the pedestal.

They stay in every row as trained, since the rule was fixed before any score was read. Scoring
without them changes nothing that matters (nothing was retrained): the Table 3 row scores 0.703
[0.59, 0.80] on 132 shots (ten of the 18 are in its test sets) against 0.703 [0.59, 0.79] on 142,
and the 5-fold `full_cum_abcdrge` 0.742 [0.67, 0.81] on 299 shots (17 of the 18 are scored; 157082
has no window in it) against 0.736 on 316.

### How a row is scored

Per window, out of sample (a shot is scored by networks that never saw it), with the macro F1 of the
four classes and a 95 % interval from a shot bootstrap (1,000 replicates, seed 20261001). Every row
is scored three ways. **On its own windows** (those its gate and margins keep; the tables' "own"
column). **On the paper's windows**, the beam gate with a 20 ms margin inside the interval and 100
ms after L-mode, which gives rows with different gates one criterion; no population is common to
every row (the shots a row scores differ with its gate, margins, channel rule and split), so the
paper's-window column compares a row with the criterion, not rows with each other. **On fixed
populations**: the 117 shots the corpus holds BES for (the corpus shots, 2021-2023) and the 327
fetched-only shots, each row restricted to the shots it scores in that population, so a change of
score is not a change of shots. A factor's effect is the **paired difference** between two rows on
the shots both score (`bes_protocol.paired_difference`: the same shots drawn in each bootstrap
replicate, so the interval is that of the difference), reported for all common shots and for the
corpus and fetched-only shots separately. `margin_sensitivity` in each row's JSON rescored the own
windows at 0, 20, 50, 100 and 200 ms inside the interval ends: the macro F1 moves by at most 0.05
over that range in every row except `only_a` (0.07) and `cum_abcdrf` (0.40 to 0.53, 37 shots). Macro
AUROC and AUPRC (one against the rest, class means, shot bootstrap) are in the JSON of the rows run
with `--rank`. No row was chosen, tuned or dropped by its score: every row in the script's chain was
run and is reported. The protocol-matched row for the label paper's comparison table was named
before any all-shot row ran (`cum_abcdrgef`); after review its architecture, training length and
block were made the paper's and it was rerun once as `full_cum_abcdrgef`, the row Table 3 now
reports.

**A validation audit** (`validation_audit.json`, stage `audit`): the checkpoint of a run is the best
validation macro F1, a mean over the classes the validation set holds; a split whose validation set
lacks a class is degenerate, and one with fewer than three shots in a class thin, and a row with
either is underpowered for ranking single factors. The audit also lists the test shots per class and
split (the paper's WPQH test set has 7 discharges).

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
| `full_cum_abcdrge` | paper training and geometry block, every fetched shot, 5-fold by shot | 316 | 230,282 | 0.736 [0.66, 0.80] | 0.736 [0.66, 0.80] | 0.83 / 0.90 / 0.82 / 0.39 | 77 / 143 / 137 / 19 |
| `full_cum_abcdrgef` | paper training, geometry block and split: the protocol row | 142 | 152,863 | 0.703 [0.59, 0.79] | 0.703 [0.59, 0.79] | 0.80 / 0.86 / 0.77 / 0.39 | 40 / 69 / 59 / 7 |
| `full_leak_abcdrge` | paper training and geometry block, within-shot block split (leaky) | 307 | 142,613 | 0.902 [0.86, 0.93] | 0.902 [0.86, 0.93] | 0.93 / 0.97 / 0.94 / 0.77 | 72 / 139 / 134 / 17 |
| `ge_a` | every fetched shot at 1 MHz + beam gating | 366 | 268,812 | 0.752 [0.70, 0.80] | 0.744 [0.69, 0.80] | 0.82 / 0.92 / 0.86 / 0.41 | 89 / 161 / 158 / 29 |
| `ge_b` | every fetched shot at 1 MHz + transition exclusion | 432 | 409,558 | 0.851 [0.82, 0.88] | 0.809 [0.74, 0.86] | 0.86 / 0.93 / 0.86 / 0.76 | 119 / 174 / 195 / 66 |
| `ge_c` | every fetched shot at 1 MHz + paper optimiser | 444 | 446,426 | 0.829 [0.80, 0.86] | 0.785 [0.73, 0.84] | 0.85 / 0.91 / 0.84 / 0.71 | 125 / 176 / 196 / 78 |
| `ge_r` | every fetched shot at 1 MHz + paper rows (first six) | 444 | 446,426 | 0.821 [0.79, 0.85] | 0.755 [0.69, 0.81] | 0.84 / 0.91 / 0.83 / 0.70 | 125 / 176 / 196 / 78 |
| `full_only_c` | paper optimiser, architecture and training length (117 shots) | 117 | 65,052 | 0.676 [0.59, 0.75] | 0.561 [0.42, 0.67] | 0.70 / 0.92 / 0.46 / 0.62 | 38 / 74 / 16 / 20 |

Rows named `only_*` add one factor to `base`; `cum_*` add them in the order of the letters; `full_*`
are the `cum_*` rows (or `only_c`) rerun with the paper's architecture and training length and,
where marked, the geometry block; `ge_*` add one factor to `only_ge` on all 444 shots; `leak_*` are
the leaky diagnostics. The "shots" column is the shots with a scored window of that class, so the
gate's cost reads off it.

**The split rows overlap.** `only_f`, `cum_abcdrf`, `cum_abcdrgef` and `full_cum_abcdrgef` score the
pooled test sets of five random draws, drawn independently, so a shot can be tested in several of
them. Their "Windows" count a window once per draw that tests it, and their score weights a shot by
its number of tests; their "Shots" are distinct shots. The distinct windows and shot-tests and the
scores with each shot counted once are in [Overlap of the five test
draws](#overlap-of-the-five-test-draws).

Macro AUROC and AUPRC [95 % CI] of the rows run with `--rank` (shot bootstrap, one against the
rest):

- `base`: AUROC 0.941 [0.911, 0.967], AUPRC 0.741 [0.643, 0.834]
- `only_c`: AUROC 0.930 [0.895, 0.958], AUPRC 0.712 [0.617, 0.802]
- `only_f`: AUROC 0.975 [0.952, 0.989], AUPRC 0.832 [0.710, 0.918]
- `only_ge`: AUROC 0.955 [0.940, 0.969], AUPRC 0.864 [0.821, 0.908]
- `cum_abcdr`: AUROC 0.827 [0.732, 0.917], AUPRC 0.409 [0.331, 0.569]
- `cum_abcdrge`: AUROC 0.912 [0.869, 0.951], AUPRC 0.730 [0.684, 0.809]
- `cum_abcdrgef`: AUROC 0.942 [0.908, 0.973], AUPRC 0.740 [0.647, 0.867]
- `cum_abcdrf`: AUROC 0.856 [0.704, 0.940], AUPRC 0.441 [0.302, 0.635]
- `leak_abcdr`: AUROC 0.997 [0.993, 1.000], AUPRC 0.976 [0.947, 0.995]
- `leak_abcdrge`: AUROC 0.997 [0.994, 0.999], AUPRC 0.981 [0.966, 0.990]
- `full_cum_abcdrge`: AUROC 0.922 [0.885, 0.958], AUPRC 0.760 [0.698, 0.831]
- `full_cum_abcdrgef`: AUROC 0.889 [0.810, 0.949], AUPRC 0.713 [0.615, 0.829]
- `full_leak_abcdrge`: AUROC 0.990 [0.983, 0.995], AUPRC 0.946 [0.910, 0.974]
- `full_only_c`: AUROC 0.929 [0.895, 0.959], AUPRC 0.727 [0.627, 0.817]

#### Fixed populations and paired steps

Each row on the shots it scores in each population: all of them, the corpus shots among them (the
117 shots the corpus holds BES for) and the fetched-only shots (the 327 others), as macro F1 [95 %
CI] (shots).

| Row | All scored shots | Corpus shots | Fetched-only shots |
|---|---|---|---|
| `base` | 0.717 [0.63, 0.78] (117) | 0.717 [0.63, 0.78] (117) | - |
| `only_a` | 0.610 [0.46, 0.74] (95) | 0.610 [0.46, 0.74] (95) | - |
| `only_b` | 0.671 [0.56, 0.76] (109) | 0.671 [0.56, 0.76] (109) | - |
| `only_c` | 0.662 [0.57, 0.73] (117) | 0.662 [0.57, 0.73] (117) | - |
| `only_d` | 0.705 [0.61, 0.78] (111) | 0.705 [0.61, 0.78] (111) | - |
| `only_r` | 0.650 [0.55, 0.73] (117) | 0.650 [0.55, 0.73] (117) | - |
| `only_g` | 0.713 [0.62, 0.78] (117) | 0.713 [0.62, 0.78] (117) | - |
| `only_f` | 0.810 [0.69, 0.89] (51) | 0.810 [0.69, 0.89] (51) | - |
| `only_ge` | 0.828 [0.80, 0.86] (444) | 0.705 [0.62, 0.78] (117) | 0.844 [0.81, 0.87] (327) |
| `cum_a` | 0.610 [0.46, 0.74] (95) | 0.610 [0.46, 0.74] (95) | - |
| `cum_ab` | 0.520 [0.35, 0.65] (90) | 0.520 [0.35, 0.65] (90) | - |
| `cum_abc` | 0.489 [0.36, 0.62] (90) | 0.489 [0.36, 0.62] (90) | - |
| `cum_abcd` | 0.486 [0.34, 0.62] (84) | 0.486 [0.34, 0.62] (84) | - |
| `cum_abcdr` | 0.433 [0.31, 0.57] (80) | 0.433 [0.31, 0.57] (80) | - |
| `cum_abcdrg` | 0.416 [0.29, 0.55] (80) | 0.416 [0.29, 0.55] (80) | - |
| `cum_abcdrge` | 0.731 [0.66, 0.80] (311) | 0.452 [0.35, 0.57] (80) | 0.830 [0.70, 0.88] (231) |
| `cum_abcdrgef` | 0.716 [0.61, 0.81] (146) | 0.446 [0.30, 0.60] (40) | 0.775 [0.61, 0.88] (106) |
| `cum_abcdrf` | 0.483 [0.28, 0.63] (37) | 0.483 [0.28, 0.63] (37) | - |
| `leak_abcdr` | 0.960 [0.93, 0.98] (75) | 0.960 [0.93, 0.98] (75) | - |
| `leak_abcdrge` | 0.950 [0.93, 0.97] (304) | 0.924 [0.84, 0.95] (78) | 0.957 [0.93, 0.97] (226) |
| `full_cum_abcdrge` | 0.736 [0.66, 0.80] (316) | 0.436 [0.32, 0.56] (80) | 0.802 [0.71, 0.86] (236) |
| `full_cum_abcdrgef` | 0.703 [0.59, 0.79] (142) | 0.463 [0.28, 0.66] (34) | 0.742 [0.60, 0.83] (108) |
| `full_leak_abcdrge` | 0.902 [0.86, 0.93] (307) | 0.826 [0.67, 0.89] (78) | 0.928 [0.89, 0.95] (229) |
| `ge_a` | 0.752 [0.70, 0.80] (366) | 0.598 [0.48, 0.71] (97) | 0.766 [0.70, 0.82] (269) |
| `ge_b` | 0.851 [0.82, 0.88] (432) | 0.730 [0.61, 0.82] (109) | 0.864 [0.83, 0.89] (323) |
| `ge_c` | 0.829 [0.80, 0.86] (444) | 0.722 [0.63, 0.79] (117) | 0.843 [0.81, 0.87] (327) |
| `ge_r` | 0.821 [0.79, 0.85] (444) | 0.678 [0.59, 0.76] (117) | 0.844 [0.81, 0.87] (327) |
| `full_only_c` | 0.676 [0.59, 0.75] (117) | 0.676 [0.59, 0.75] (117) | - |

Paired steps: the difference of macro F1 between a row and the row it is compared with, on the shots
both score, with the 95 % interval of the difference from a shot bootstrap; the columns are all
common shots, the corpus shots among them and the fetched-only shots. A step with fewer than 5
common shots in a population is not scored ("-").

| Step | Row against its reference | Common shots | Corpus shots | Other |
|---|---|---|---|---|
| a gate | `only_a` - `base` | -0.071 [-0.16, +0.01] (95) | -0.071 [-0.16, +0.01] (95) | - (0) |
| b margins | `only_b` - `base` | -0.036 [-0.11, +0.04] (109) | -0.036 [-0.11, +0.04] (109) | - (0) |
| c optimiser | `only_c` - `base` | -0.055 [-0.11, -0.01] (117) | -0.055 [-0.11, -0.01] (117) | - (0) |
| d channel check | `only_d` - `base` | -0.011 [-0.04, +0.02] (111) | -0.011 [-0.04, +0.02] (111) | - (0) |
| r rows | `only_r` - `base` | -0.068 [-0.12, -0.02] (117) | -0.068 [-0.12, -0.02] (117) | - (0) |
| g 1 MHz | `only_g` - `base` | -0.005 [-0.04, +0.03] (117) | -0.005 [-0.04, +0.03] (117) | - (0) |
| f paper split | `only_f` - `base` | +0.027 [-0.05, +0.11] (51) | +0.027 [-0.05, +0.11] (51) | - (0) |
| e shots (with g) | `only_ge` - `only_g` | -0.008 [-0.05, +0.04] (117) | -0.008 [-0.05, +0.04] (117) | - (0) |
| a gate, 444 shots | `ge_a` - `only_ge` | -0.046 [-0.09, -0.00] (366) | -0.059 [-0.15, +0.03] (97) | -0.054 [-0.12, -0.01] (269) |
| b margins, 444 shots | `ge_b` - `only_ge` | +0.020 [+0.00, +0.04] (432) | +0.043 [-0.01, +0.10] (109) | +0.013 [-0.01, +0.03] (323) |
| c optimiser, 444 shots | `ge_c` - `only_ge` | +0.001 [-0.01, +0.01] (444) | +0.017 [-0.03, +0.06] (117) | -0.002 [-0.01, +0.01] (327) |
| r rows, 444 shots | `ge_r` - `only_ge` | -0.007 [-0.02, +0.01] (444) | -0.027 [-0.09, +0.03] (117) | -0.001 [-0.02, +0.01] (327) |
| cum a gate | `cum_a` - `base` | -0.071 [-0.16, +0.01] (95) | -0.071 [-0.16, +0.01] (95) | - (0) |
| cum b margins | `cum_ab` - `cum_a` | -0.071 [-0.15, -0.01] (90) | -0.071 [-0.15, -0.01] (90) | - (0) |
| cum c optimiser | `cum_abc` - `cum_ab` | -0.031 [-0.13, +0.06] (90) | -0.031 [-0.13, +0.06] (90) | - (0) |
| cum d channel check | `cum_abcd` - `cum_abc` | -0.004 [-0.04, +0.03] (84) | -0.004 [-0.04, +0.03] (84) | - (0) |
| cum r rows | `cum_abcdr` - `cum_abcd` | -0.051 [-0.10, +0.01] (80) | -0.051 [-0.10, +0.01] (80) | - (0) |
| cum g 1 MHz | `cum_abcdrg` - `cum_abcdr` | -0.018 [-0.07, +0.03] (80) | -0.018 [-0.07, +0.03] (80) | - (0) |
| cum e shots | `cum_abcdrge` - `cum_abcdrg` | +0.037 [-0.05, +0.13] (80) | +0.037 [-0.05, +0.13] (80) | - (0) |
| cum f paper split | `cum_abcdrgef` - `cum_abcdrge` | -0.023 [-0.10, +0.03] (146) | +0.007 [-0.09, +0.06] (40) | -0.100 [-0.19, -0.01] (106) |
| paper training + geometry (5-fold) | `full_cum_abcdrge` - `cum_abcdrge` | -0.015 [-0.05, +0.01] (282) | -0.026 [-0.09, +0.03] (72) | -0.045 [-0.10, +0.00] (210) |
| paper training + geometry (paper split) | `full_cum_abcdrgef` - `cum_abcdrgef` | -0.044 [-0.11, +0.06] (61) | -0.055 [-0.15, +0.11] (15) | -0.012 [-0.09, +0.10] (46) |
| f under paper training | `full_cum_abcdrgef` - `full_cum_abcdrge` | +0.001 [-0.07, +0.08] (142) | +0.071 [-0.07, +0.21] (34) | -0.064 [-0.12, -0.00] (108) |
| paper training alone (optimiser row) | `full_only_c` - `only_c` | +0.014 [-0.02, +0.06] (117) | +0.014 [-0.02, +0.06] (117) | - (0) |
| within-shot split, early-stopped | `leak_abcdrge` - `cum_abcdrgef` | +0.239 [+0.15, +0.33] (142) | +0.484 [+0.31, +0.60] (38) | +0.178 [+0.07, +0.30] (104) |
| within-shot split, paper training | `full_leak_abcdrge` - `full_cum_abcdrgef` | +0.158 [+0.08, +0.25] (139) | +0.313 [+0.18, +0.44] (33) | +0.174 [+0.06, +0.31] (106) |

#### Label fragmentation

The curated intervals of the two populations: how many there are per class, their median length (and
quartiles) in ms and the share shorter than 100 ms.

| Class | Corpus: intervals | median (IQR), ms | under 100 ms | Fetched-only: intervals | median (IQR), ms | under 100 ms |
|---|---|---|---|---|---|---|
| L | 272 | 20 (10-40) | 83 % | 499 | 25 (20-75) | 78 % |
| H | 408 | 40 (10-120) | 74 % | 365 | 30 (16-828) | 65 % |
| QH | 20 | 1212 (338-1457) | 10 % | 392 | 190 (15-928) | 36 % |
| WP | 268 | 40 (30-40) | 95 % | 322 | 40 (20-53) | 82 % |

#### Class mix

- `cum_abcdrgef` as scored 0.716; reweighted to the paper's test mix [8.3, 21.8, 17.4, 9.7]
  (labelled seconds of L, H, QH and WPQH): 0.750 (F1 0.73 / 0.87 / 0.74 / 0.66)
- `full_cum_abcdrgef` as scored 0.703; reweighted to the paper's test mix [8.3, 21.8, 17.4, 9.7]
  (labelled seconds of L, H, QH and WPQH): 0.685 (F1 0.79 / 0.78 / 0.74 / 0.43)

#### By year of the shot

The same rows by year of the shot (the year of the shot's EFIT01 reconstruction insertion,
`$LABELER_ROOT/round4/conf/dates.csv`); macro F1 [95 % CI] (shots). A year holds few shots, and
often one or two classes, so a per-year score is noisy.

| Year of the shot | `only_ge` | `cum_abcdrgef` | `full_cum_abcdrge` | `full_cum_abcdrgef` |
|---|---|---|---|---|
| 2012 | 0.884 [0.84, 0.94] (5) | 0.910 [0.91, 1.00] (2) | 0.851 [0.71, 1.00] (3) | 0.997 [1.00, 1.00] (2) |
| 2013 | 0.000 [0.00, 0.00] (3) | 0.175 [0.18, 0.18] (1) | 0.000 [0.00, 0.00] (1) | 0.043 [0.04, 0.04] (1) |
| 2014 | 0.492 [0.45, 0.99] (63) | 0.961 [0.86, 1.00] (17) | 0.494 [0.49, 1.00] (48) | 0.481 [0.45, 1.00] (20) |
| 2015 | 0.681 [0.56, 0.88] (26) | 0.710 [0.69, 1.00] (7) | 0.357 [0.30, 0.97] (19) | 0.337 [0.25, 0.98] (11) |
| 2016 | 0.731 [0.62, 0.84] (21) | 0.673 [0.44, 0.99] (9) | 0.846 [0.69, 0.96] (14) | 0.802 [0.68, 0.91] (7) |
| 2017 | 0.858 [0.76, 0.92] (52) | 0.509 [0.40, 0.99] (12) | 0.812 [0.67, 0.88] (29) | 0.832 [0.52, 0.94] (14) |
| 2018 | 0.892 [0.81, 0.94] (50) | 0.782 [0.75, 0.98] (21) | 0.683 [0.61, 0.93] (35) | 0.928 [0.74, 0.97] (11) |
| 2019 | 0.835 [0.64, 0.98] (36) | 0.475 [0.38, 1.00] (7) | 0.455 [0.40, 0.98] (26) | 0.848 [0.65, 1.00] (14) |
| 2020 | 0.301 [0.08, 0.82] (5) | - | 0.159 [0.00, 0.45] (5) | 0.533 [0.53, 0.53] (1) |
| 2021 | 0.645 [0.54, 0.75] (85) | 0.610 [0.52, 0.84] (40) | 0.623 [0.55, 0.86] (69) | 0.758 [0.61, 0.88] (32) |
| 2022 | 0.760 [0.66, 0.85] (73) | 0.487 [0.34, 0.66] (22) | 0.438 [0.28, 0.60] (45) | 0.464 [0.27, 0.69] (19) |
| 2023 | 0.471 [0.34, 0.59] (25) | 0.285 [0.18, 0.60] (8) | 0.454 [0.31, 0.68] (22) | 0.360 [0.21, 1.00] (10) |

The Table 3 row by year again, with the F1 of each class and the test shots that hold it (the
stage `tables` prints it from `populations.json`):

| Year of the shot | `full_cum_abcdrgef`: shots | windows | L | H | QH | WP |
|---|---|---|---|---|---|---|
| 2012 | 2 | 1,477 | 1.00 (2) | 1.00 (1) | - | - |
| 2013 | 1 | 458 | 0.04 (1) | - | - | - |
| 2014 | 20 | 15,889 | 0.00 (1) | - | 0.96 (20) | - |
| 2015 | 11 | 6,248 | 0.11 (2) | - | 0.90 (9) | 0.00 (1) |
| 2016 | 7 | 5,279 | 0.64 (4) | 0.87 (4) | 0.89 (3) | - |
| 2017 | 14 | 8,487 | 0.97 (4) | 0.89 (3) | 0.86 (10) | 0.61 (2) |
| 2018 | 11 | 7,268 | 0.91 (6) | 0.90 (3) | 0.97 (7) | - |
| 2019 | 14 | 15,544 | - | 0.85 (14) | - | - |
| 2020 | 1 | 1,068 | - | - | - | 0.53 (1) |
| 2021 | 32 | 57,296 | 0.88 (13) | 0.91 (24) | 0.48 (5) | - |
| 2022 | 19 | 24,990 | 0.26 (4) | 0.77 (13) | 0.52 (4) | 0.32 (3) |
| 2023 | 10 | 8,859 | 0.20 (3) | 0.88 (7) | 0.00 (1) | - |

F1 per class (shots holding the class). The macro F1 is the mean of the F1 of the classes the year
holds, so a class carried by one shot counts for a half (two classes) to a quarter (four) of it:
the 2014 macro F1 of 0.481 is QH 0.96 on 20 shots and one L shot scored 0, and the 2023 value of
0.360 averages three classes, one of them QH on a single shot at 0.00. A year's macro F1 is
therefore a reading only where each class it holds rests on several shots (2021, 2022 and 2016 to
2018).

#### Validation audit

A split is degenerate when its validation set lacks a class, thin when it holds fewer than three
shots of some class (the checkpoint of its network is then chosen on a handful of shots); the last
column is the number of WPQH test shots in each split.

| Row | Degenerate splits | Thin splits | WPQH test shots per split |
|---|---|---|---|
| `base` | - | fold0, fold1, fold2, fold3, fold4 | [4, 4, 3, 4, 5] |
| `only_a` | fold0, fold1 | fold2, fold3, fold4 | [2, 4, 0, 3, 2] |
| `only_b` | - | fold0, fold1, fold2, fold3, fold4 | [2, 2, 1, 3, 4] |
| `only_c` | - | fold0, fold1, fold2, fold3, fold4 | [4, 4, 3, 4, 5] |
| `only_d` | - | fold0, fold1, fold2, fold3, fold4 | [4, 4, 3, 4, 5] |
| `only_r` | - | fold0, fold1, fold2, fold3, fold4 | [4, 4, 3, 4, 5] |
| `only_g` | - | fold0, fold1, fold2, fold3, fold4 | [4, 4, 3, 4, 5] |
| `only_f` | - | split1, split4 | [2, 3, 2, 2, 2] |
| `only_ge` | - | - | [15, 16, 17, 15, 15] |
| `cum_a` | fold0, fold1 | fold2, fold3, fold4 | [2, 4, 0, 3, 2] |
| `cum_ab` | fold0, fold1, fold3, fold4 | fold2 | [1, 2, 0, 3, 0] |
| `cum_abc` | fold0, fold1, fold3, fold4 | fold2 | [1, 2, 0, 3, 0] |
| `cum_abcd` | fold0, fold1, fold3, fold4 | fold2 | [1, 2, 0, 3, 0] |
| `cum_abcdr` | fold0, fold1, fold3, fold4 | fold2 | [1, 2, 0, 3, 0] |
| `cum_abcdrg` | fold0, fold1, fold3, fold4 | fold2 | [1, 2, 0, 3, 0] |
| `cum_abcdrge` | - | fold0, fold1, fold2 | [1, 4, 6, 2, 5] |
| `cum_abcdrgef` | - | split0, split1, split2, split4 | [2, 2, 2, 2, 2] |
| `cum_abcdrf` | - | split0, split1, split2, split3, split4 | [1, 1, 2, 1, 1] |
| `leak_abcdr` | - | - | [4, 2, 2, 3, 4] |
| `leak_abcdrge` | - | - | [11, 11, 13, 10, 9] |
| `full_cum_abcdrge` | - | fold0, fold1, fold2 | [1, 4, 6, 2, 6] |
| `full_cum_abcdrgef` | - | - | [2, 3, 2, 2, 2] |
| `full_leak_abcdrge` | - | - | [10, 12, 13, 10, 10] |
| `ge_a` | - | - | [3, 8, 7, 4, 7] |
| `ge_b` | - | - | [11, 12, 15, 14, 14] |
| `ge_c` | - | - | [15, 16, 17, 15, 15] |
| `ge_r` | - | - | [15, 16, 17, 15, 15] |
| `full_only_c` | - | fold0, fold1, fold2, fold3, fold4 | [4, 4, 3, 4, 5] |

### Reading it

Every difference below is a paired difference on the shots both rows score, from the "Paired steps"
table; "corpus" and "fetched-only" are the fixed populations of that table.

- **The corpus shots score far below the others, and more training shots do not change it.** The 117
  corpus shots score 0.705 [0.62, 0.78] when 327 more shots are added to training (`only_ge`,
  ungated) against 0.713 [0.62, 0.78] without them (`only_g`): paired -0.008 [-0.05, +0.04]. Under
  the paper's whole selection on all fetched shots they score 0.44 to 0.46 (`cum_abcdrge` 0.452 on
  80 shots, `cum_abcdrgef` 0.446 on 40, `full_cum_abcdrge` 0.436 on 80, the Table 3 row
  `full_cum_abcdrgef` 0.463 on 34), while the fetched-only shots score 0.74 to 0.83 in the same
  rows. The claim is for these four rows: the rows with part of the selection score the corpus shots
  differently (`cum_a` 0.610, `cum_ab` 0.520, `ge_a` 0.598, and the 117-shot stack `cum_abcdrg`
  0.416). Adding the 327 shots to the gated 117-shot
  stack moves its corpus shots by +0.037 [-0.05, +0.13] (`cum_abcdrg` to `cum_abcdrge`, 80 shots).
  The all-shot rows score higher than the 117-shot rows because 327 of their 444 shots are
  fetched-only shots that are easier, not because the network has more to learn from. **The first
  retrain's explanation of the gap (a small-data number) and the earlier ranking of the shots as the
  largest single factor are withdrawn.** The other model, `confine-ours`, which reads the 0D signals
  and was trained on all 401 shots, shows the same split (0.689 on the corpus shots, 0.926 on the
  others, [confinement_ours.md](confinement_ours.md)), so the cause is in those shots, not in the
  BES recipe. What it is, the data cannot say. The corpus shots are 20 from 2021, 72 from 2022 and
  25 from 2023; the fetched-only shots are 65 from 2021 and 1 from 2022, the rest earlier, so the
  campaign and the membership of the corpus are almost the same variable. By year of the shot the
  Table 3 row scores 0.76 for 2021 (32 shots) and 0.46 and 0.36 for 2022 and 2023 (19 and 10 shots).
  The 2022 deficit is broad: L 0.26 on 4 shots, H 0.77 on 13, QH 0.52 on 4 and WPQH 0.32 on 3. The
  2023 value rests on L 0.20 (3 shots), H 0.88 (7) and one QH shot at 0.00, so it is thin. The other
  years are not compared: most hold one or two classes on one or two shots each, so a class held by
  one shot carries a half to a quarter of the macro F1 (the per-class table under "By year of the
  shot"; 2014, for one, is QH 0.96 on 20 shots plus one L shot scored 0). The sampling
  rate is not it (500 kHz in the corpus; `only_g` against `base` is -0.005 [-0.04, +0.03]); the
  labels of those shots (the next point) may be.
- **The labelled intervals are short, which the gate and margins punish.** The median L, H and WPQH
  interval of the corpus shots is 20, 40 and 40 ms, and 74 to 95 % of them are under 100 ms
  (fetched-only shots: 25, 30 and 40 ms; 65 to 82 %); only QH differs, 20 intervals of 1.2 s
  (median) on the corpus shots against 392 of 190 ms. A 20 ms margin at each end leaves nothing of
  an interval of 40 ms, so most intervals of these classes give no window once the margins apply,
  and the gate takes more (it fails 46 % of the L-mode and 76 % of the WPQH windows of the 444
  shots, passive-BES windows among them: `gate_breakdown.json`).
- **Single factors, on 444 shots.** The single-factor rows on 117 shots (`only_a` to `only_r`) are
  underpowered: in all five folds of each (two of five for `only_f`) the validation set holds fewer
  than 3 shots of some class (the validation audit), and `only_a` has two folds with a class missing
  altogether, so the best checkpoint of a fold is chosen on a handful of shots. They were rerun from
  `only_ge` on all 444 shots (`ge_a`, `ge_b`, `ge_c`, `ge_r`; no fold is degenerate or thin, and a
  fold has 3 to 17 WPQH test shots). The beam gate lowers the score by 0.046 [-0.09, -0.00]; the
  margins raise it by 0.020 [+0.00, +0.04] (the timing of a label at a transition is its least
  certain part, and the margins drop those windows); the paper's optimiser changes nothing (+0.001
  [-0.01, +0.01]) and neither do the paper's rows (-0.007 [-0.02, +0.01]). By class the gate costs
  WPQH most (F1 0.70 to 0.41; L 0.86 to 0.82, H 0.91 to 0.92, QH 0.84 to 0.86) and the margins help
  WPQH most (0.70 to 0.76). **What the gate step removes.** The gate fails two kinds of window and the
  step does not separate them: windows with the 150L beam on but out of range (50 to 700 kW) or the
  150R beam above 200 kW, and **passive-BES windows**, where the 150L beam is below 50 kW and the BES
  records passive emission only. On the 444 shots the passive windows are 33 % of the L-mode windows,
  11 % of the H, 17 % of the QH and 68 % of the WPQH windows (`gate_breakdown.json`), against the 46 %
  of L and 76 % of WPQH windows the gate fails in all, so most of what it removes is passive. The
  gate steps compare a gated row, scored on beam-on windows, with an ungated row scored on all
  windows, so the WPQH drop (0.70 to 0.41) is largely the removal of passive-BES windows, where the
  beam state itself can serve as a shortcut to the class, not a measure of the 700 and 200 kW
  thresholds; no row varies the thresholds, so this is not tested. The 117-shot rows had given -0.055 for the optimiser and -0.068 for the
  rows; neither survives at 444 shots and the earlier reading that the paper's selection rules each
  lower the score is withdrawn.
- **The stack does not reach the paper.** The paper's selection on all fetched shots gives 0.731
  [0.66, 0.80] with the early-stopped padded network (`cum_abcdrge`) and 0.736 [0.66, 0.80] with the
  paper's architecture, training length and a block chosen by geometry (`full_cum_abcdrge`, 316
  shots; paired -0.015 [-0.05, +0.01] on 282 shots). The paper's architecture, 60,000 steps with no
  early stopping and the geometry block do not raise the score; on the single-factor row they add
  +0.014 [-0.02, +0.06] (`full_only_c` against `only_c`, 117 shots), and under the paper's split
  -0.044 [-0.11, +0.06] on the 61 shots both rows score (`full_cum_abcdrgef` against
  `cum_abcdrgef`). Geometry dropped 43 of the 444 shots (3 without a position record, 40 whose array
  does not reach the separatrix), 13 of them corpus shots.
- **The paper's split is no lift.** The five random splits by discharge give 0.703 [0.59, 0.79]
  pooled (`full_cum_abcdrgef`, 142 distinct shots in 200 shot-tests; one split scores 0.684 +/-
  0.074) against 0.736 for the 5-fold of the same recipe: paired +0.001 [-0.07, +0.08] on the 142
  shots both score. A test set of 7 (WPQH) to 69 (H) shots per class is as noisy as the paper's
  44-shot set; the WPQH class rests on 2 to 3 test shots per split here (7 distinct over the five
  splits) against the paper's 7 on its one test set.
- **Class mix does not explain it.** Reweighting the confusion matrix of the Table 3 row to the
  class mix of the paper's test set (8.3, 21.8, 17.4 and 9.7 labelled seconds of L, H, QH and WPQH)
  gives 0.685, lower than the 0.703 as scored (the early-stopped `cum_abcdrgef` goes from 0.716 to
  0.750). The score is carried down by WPQH (0.39 on 7 shots) and QH (0.77).
- **A within-shot split is what reaches the published number.** With the early-stopped network and
  0.1 s blocks of every shot drawn at random (`leak_abcdrge`) the score is 0.950 [0.93, 0.97] (L
  0.95, H 0.99, QH 0.98, WPQH 0.88; AUROC 0.997); with the paper's architecture and training length
  (`full_leak_abcdrge`, 307 shots) it is 0.902 [0.86, 0.93] and the paper's 0.94 lies between the
  two. Held out by shot, the same recipe gives 0.703. On the corpus shots the within-shot split
  gives 0.826 [0.67, 0.89] against 0.463 held out: paired +0.313 [+0.18, +0.44] on the 33 corpus
  shots both rows score. The paper says it split by discharge, so this is not claimed as its
  protocol; it shows that what separates 0.90 to 0.95 from 0.70 here is generalising to a discharge
  the network has not seen, and that if the paper's test discharges were independent of its training
  ones, something other than the pipeline (the labels, the selection, unstated settings, a 48-GPU
  training) differs.
- **Label noise is real but small, and unlikely to be the whole 0.2.** Confident learning (below)
  flags 8.6 % of the intervals of the gated row as probably another class, among them 13 of the 254
  QH and WPQH intervals (5.1 %) as the other of the two. A model scored on the intervals it agrees
  with scores higher by construction, so no such number is reported.

### The comparison-table score (Table 3)

The score to set beside the paper's 0.94 is **`full_cum_abcdrgef`: macro F1 0.703 [0.59, 0.79]**,
**our reimplementation under the paper's selection and split** (per class L 0.80, H 0.86, QH 0.77,
WPQH 0.39 on 40, 69, 59 and 7 distinct shots; macro AUROC 0.889, AUPRC 0.713; 152,863 pooled
windows, of which 106,232 are distinct, on 142 distinct test shots in 200 shot-tests over five
random splits by shot whose test sets overlap, the splits' own scores 0.79, 0.73, 0.62, 0.66 and
0.62: mean 0.684, sd 0.074). The row was named in advance as the Table 3 row and its number is
kept; read it with the overlap below. Overall and by population: 0.463 [0.28, 0.66] on the 34 corpus shots in it and 0.742 [0.60, 0.83] on
the other 108.

**The ranking score falls short too.** The paper reports a one-vs-rest AUC of at least 0.99 for
every class and no macro value; our macro AUROC on this row is 0.889 [0.81, 0.95] (L 0.937, H 0.925,
QH 0.912, WPQH 0.785 one-vs-rest; AUPRC 0.713 [0.62, 0.83], which the paper does not report). So the
gap is in the ranking of the windows and
not only in the F1 at the argmax. The benchmark figure draws the paper's bound as a hatched bar at
0.99 labelled "per class, no macro value" beside our macro AUROC, and leaves its AUPRC as "not
reported".

The row is the paper's data selection and training put in as the paper's full text
states them: the beam gate (150L at or above 700 kW, 150R at or below 200 kW; WPQH 400 kW in
training), the margins, a 6 x 8 block chosen from each shot's channel positions (401 of the 444
shots), the native 1 MHz BES, Adam with weight decay 1e-3 and rates 1e-3 and 1e-5, the
one-convolution network without padding (227,644 parameters, asserted in
`tests/labeler/test_confinement_bes_cnn.py`), 60,000 steps with no early stopping and the checkpoint
of best validation macro F1 (steps 3,000, 48,000, 31,000, 53,000 and 27,000 in the five splits), and
the split by discharge, stratified by regime, five repeats pooled. The first version of it
(`cum_abcdrgef`, early-stopped padded network, dead-channel check) was named before any all-shot row
ran; after review its architecture, training length and block were made the paper's and it was rerun
once as `full_cum_abcdrgef`; no row was chosen by its score, and the earlier 0.716 is superseded by
this one (it stays in the table above). The first retrain's 0.678 and the ungated `only_ge` row
(0.828) are different protocols and are not offered as comparison-table numbers; the earlier
suggestion to show `only_ge` beside it is dropped.

#### Overlap of the five test draws

The five repeats of the paper's split are drawn independently (`bes_protocol.paper_roles`, seed
`SEED + r`), so a shot can be in the test set of more than one repeat: of the 142 test shots, 99
were tested once, 30 twice, 11 three times and 2 four times (200 shot-tests). Pooling the five
test sets counts a repeated shot's windows once per repeat that tests it (152,863 windows, of which
106,232 are distinct) and weights the shot by its number of tests; each repeat's network is a
different one, so the repeats are not copies. The pooled 0.703 stays the Table 3 number;
three more readings bracket it (stage `repeats`, record `protocol_repeats.json`, all computed from
the out-of-fold predictions on disk; each interval is a 95 % shot bootstrap that draws a shot with
all its repeats):

| Row | Test shots (shot-tests) | Windows pooled | Distinct windows | Pooled (Table 3) | One split, mean +/- sd of five | First repeat only | Probabilities averaged over repeats |
|---|---|---|---|---|---|---|---|
| `only_f` | 51 (70) | 36,800 | 27,804 | 0.810 [0.69, 0.89] (51) | 0.766 +/- 0.061 | 0.832 [0.71, 0.90] (51) | 0.839 [0.73, 0.90] (51) |
| `cum_abcdrf` | 37 (50) | 24,942 | 18,009 | 0.483 [0.28, 0.63] (37) | 0.436 +/- 0.163 | 0.438 [0.27, 0.61] (37) | 0.428 [0.26, 0.60] (37) |
| `cum_abcdrgef` | 146 (190) | 154,468 | 119,142 | 0.716 [0.61, 0.81] (146) | 0.712 +/- 0.083 | 0.713 [0.60, 0.81] (146) | 0.726 [0.60, 0.82] (146) |
| `full_cum_abcdrgef` | 142 (200) | 152,863 | 106,232 | 0.703 [0.59, 0.79] (142) | 0.684 +/- 0.074 | 0.736 [0.61, 0.81] (142) | 0.747 [0.61, 0.82] (142) |

- **One split like the paper's scores 0.684 +/- 0.074** (mean and sample standard deviation of the
  five splits' own macro F1: 0.792, 0.727, 0.619, 0.658, 0.624): the score to expect from one run of
  this protocol, and its spread. The pooled 0.703 lies within one sd of it.
- **Each shot counted once** (the windows of the first repeat that tests it, 106,232 windows):
  0.736 [0.61, 0.81].
- **Probabilities averaged over the repeats that tested a shot** (the same 106,232 windows):
  0.747 [0.61, 0.82].
- The four readings of the Table 3 row span 0.684 to 0.747 and each of the three intervals holds the
  other values; the paper's 0.94 is outside all of them. The other split rows overlap too and are in the table
  (`cum_abcdrf` is the weakest, 0.483 pooled and 0.436 +/- 0.163 per split, on 37 shots).

Windows and shots of the Table 3 row by class (the shots are distinct):

| Class | Windows pooled | Distinct windows | Test shots |
|---|---|---|---|
| L | 23,693 | 14,589 | 40 |
| H | 79,557 | 58,748 | 69 |
| QH | 44,061 | 29,399 | 59 |
| WP | 5,552 | 3,496 | 7 |

It is 0.24 short of the paper's 0.94. The paper's selection, split, optimiser, architecture,
training length and class mix are in, and none of them accounts for a gap that size (the single
factors move the score by at most 0.05, the class mix lowers it, the paper's split changes it by
+0.001 [-0.07, +0.08]), so the remainder is **unexplained**. What still differs, and is not tested:

1. **The paper's selection within Gill's own labels.** Our labels are Gill's own: 2,545 of the
   2,551 merged intervals carry a source from his tables (his workbook, his BES-time files or both)
   and only 6 are Butt's table alone ([below](#whose-labels-and-where-the-score-is-lost)), so what
   differs from the paper is what it kept of them, not who labelled them: "standard ELMy H-mode"
   with H98y2 of at least 1 and no dithering, segments ended slightly before every transition and
   checked against the logbook, 330 discharges. Whether ours is the experts' wider H class (taking in
   ELM-suppressed, ELM-free or negative-triangularity H-modes the paper left out) is **unverified**
   ([below](#what-our-h-class-contains-against-the-papers-is-unverified)); ours merge the sources
   where they differ and include shots outside those 330; and 8.6 % of the intervals look
   mislabelled (below). Restricting the scoring to the intervals of Gill's BES-time files gives no
   lift (0.594 on 26 shots), and the failures are concentrated in 22 shots (below). The corpus
   shots score 0.46 against 0.74 for the others, in this row and in `confine-ours`; whether their
   labels or their conditions differ is not known.
2. **The test set.** One 44-shot test set of 7 to 17 shots per class against five random splits
   that overlap, pooled over 142 distinct shots (200 shot-tests); the WPQH class rests on 2 to 3
   shots per split (7 distinct) against 7.
3. **The shots.** 330 discharges 2012 to 2023 against 401 here (the 444 fetched, less the 43 that
   geometry drops).
4. **Constants of ours.** The 20 ms margins and 100 ms after L-mode (the paper does not give them),
   the pedestal rows and 0.98 criterion of the geometry block; the unstated dropout (0.2), Adam
   betas, batch size (256), window stride (2,048 samples) and class weights, all set once and not
   tuned; one GPU where the paper trained on 48.

### Whose labels, and where the score is lost

Two readings of predictions already on disk bear on the first hypothesis above (stages `subsets`
and `failures`; records `subsets.json`, `failure_blocks.json` and `failure_blocks.csv`; nothing was
retrained).

**Our labels are Gill's own.** Of the 2,551 merged intervals, 2,545 carry a source from Gill's
tables (his workbook, his BES-time files or both) and 6 come from Butt's table alone
(`merged_intervals.csv`, column `sources`: 1,563 intervals from the workbook and Butt's table, 514
from the BES-time files alone, 468 from all three, 6 from Butt's alone). So the labels differ from
the paper's in what the paper kept, not in who set them. If the paper's 0.94 rested on the part
of the labels Gill took from his BES-time files, scoring on that part would lift the score. It does
not: the predictions rescored on the windows of the 982 intervals that carry a BES-time-file source
(alone or with the others), against the windows of the rest (macro F1 [95 % CI] (shots)):

| Row | All scored | Intervals from Gill's BES-time files | Other intervals | Without the 4 x 16 and scrape-off-layer shots |
|---|---|---|---|---|
| `full_cum_abcdrgef` | 0.703 [0.59, 0.79] (142) | 0.594 [0.48, 0.87] (26) | 0.708 [0.59, 0.80] (124) | 0.703 [0.59, 0.80] (132) |
| `full_cum_abcdrge` | 0.736 [0.66, 0.80] (316) | 0.607 [0.53, 0.84] (57) | 0.749 [0.67, 0.82] (273) | 0.742 [0.67, 0.81] (299) |

By class, with the shots that hold each class (the stage `tables-protocol` prints it from
`subsets.json`; "Mean of L, H, QH" is the mean of those three F1, WPQH windows still counting as
errors of the other classes):

| Row | Intervals | L | H | QH | WP | Mean of L, H, QH |
|---|---|---|---|---|---|---|
| `full_cum_abcdrgef` | all scored | 0.80 (40) | 0.86 (69) | 0.77 (59) | 0.39 (7) | 0.806 |
| `full_cum_abcdrgef` | Gill's BES-time files | 0.89 (6) | 0.78 (15) | 0.70 (10) | 0.00 (1) | 0.793 |
| `full_cum_abcdrgef` | other intervals | 0.77 (36) | 0.88 (59) | 0.79 (49) | 0.40 (6) | 0.811 |
| `full_cum_abcdrge` | all scored | 0.83 (77) | 0.90 (143) | 0.82 (137) | 0.39 (19) | 0.851 |
| `full_cum_abcdrge` | Gill's BES-time files | 0.88 (14) | 0.76 (25) | 0.79 (26) | 0.00 (2) | 0.809 |
| `full_cum_abcdrge` | other intervals | 0.82 (66) | 0.93 (126) | 0.83 (113) | 0.42 (17) | 0.858 |

Scoring on the BES-time-file intervals gives **no lift**: the macro F1 is lower (0.594 against 0.708
on the Table 3 row, 0.607 against 0.749 on the 5-fold row; the intervals overlap), but WPQH, which
carries a quarter of the macro F1, rests on one shot in the subset (two in the 5-fold row) and is
scored 0. Over L, H and QH the two subsets are level (0.793 against 0.811 on the Table 3 row, 0.809
against 0.858 on the 5-fold row; no interval is computed for these means), with L higher on the
BES-time-file intervals (0.89 on 6 shots against 0.77 on 36) and H and QH lower, so these intervals
are neither easier nor clearly harder. The last column of the first table drops the geometry
exceptions of the section on factor d: the scores do not move. The H98y2 restriction of the paper's
H class is not run (see Not done).

**Where the 5-fold row fails: whole blocks.** In `full_cum_abcdrge` (316 shots, 230,282 windows; the
row with most shots under the paper's training), a (shot, class) block is a failure when it has at
least 150 scored windows and fewer than 10 % of them are called right. There are 23 such blocks on
22 shots (12 of them corpus shots; L 9 blocks, H 4, QH 5, WPQH 5), holding 15,171 windows, 6.6 % of
the row's. The threshold is a review-list convention set after looking at the predictions (with 20
windows the list has 40 blocks). Five corpus L-mode shots (191376, 196493, 191782, 190508, 189329)
hold 2,295 of the windows, which the network calls mostly H (WP on 196493); QH fails on 190507,
195821 and 190666, WPQH on 190670, 190514 and 195865. 18 of the 23 blocks overlap an interval that confident learning flags
below; both read the same out-of-fold predictions, so the overlap is not independent evidence. The
list is for an expert or logbook check: each row is a stretch the network calls another regime
almost throughout, whether through a label error, a plasma outside the paper's definition (H98y2
below 1, dithering, the QH/WPQH boundary) or a BES problem (beam, geometry); the data cannot say
which, and no label was changed. The failures are not a placement artefact: over each block's own
windows, the outermost block channel's psi_N lies within 0.05 of the shot median for 22 of the 23 blocks
(the largest deviation 0.013), and 21 also match its count of channels in the pedestal band (within 3;
159372 differs by 4, 29 against 25). The exception is 174653 (WPQH, 4130-4799 ms), a geometry case: outer
psi_N 1.19 against 1.02 and 16 channels in the band against 29, the plasma having moved (stage
`failures`, columns `outer_psin_block` to `placement_matches_shot` of `failure_blocks.csv`, record
`failure_blocks.json` key `placement`; the tolerances are conventions fixed before the comparison). "Interval span" is the curated interval the block lies in, "Confident
learning" the flagged intervals that overlap it ("-" for none), "Mostly called" the class most of the
block's windows are called (and their share), the year that of the shot's EFIT01 insertion.

| Shot | Year | Corpus | Class | Windows | Called right | Mostly called | Interval span (ms) | Confident learning |
|---|---|---|---|---|---|---|---|---|
| 179634 | 2019 | no | L | 2,031 | 0.0 % | QH (100 %) | 800-5000 | 800-5000 |
| 185871 | 2021 | yes | H | 1,474 | 3.3 % | WP (55 %) | 2110-5170 | 2110-5170 |
| 184810 | 2021 | no | H | 1,331 | 2.0 % | QH (98 %) | 2239-5006 | 2239-5006 |
| 190507 | 2022 | yes | QH | 1,308 | 8.3 % | WP (88 %) | 1372-4244 | 1799-2000;1372-1787;2010-2079;2090-4244 |
| 190670 | 2022 | yes | WP | 957 | 0.0 % | H (98 %) | 3000-5000 | 3000-5000 |
| 190514 | 2022 | yes | WP | 904 | 0.0 % | QH (99 %) | 3024-4916 | 3024-4916 |
| 195865 | 2023 | yes | WP | 863 | 0.0 % | H (53 %) | 2592-4401 | - |
| 191376 | 2022 | yes | L | 839 | 3.6 % | H (96 %) | 300-2058 | 300-2058 |
| 195821 | 2023 | yes | QH | 661 | 0.0 % | H (71 %) | 1600-3400 | 1600-3400 |
| 190666 | 2022 | yes | QH | 622 | 1.8 % | H (98 %) | 2686-4000 | 2686-4000 |
| 184433 | 2021 | no | L | 560 | 0.0 % | QH (98 %) | 365-1552 | - |
| 196493 | 2023 | yes | L | 463 | 0.0 % | WP (94 %) | 1520-2518 | 1520-2518 |
| 191782 | 2022 | yes | L | 413 | 0.5 % | H (100 %) | 400-2127 | 400-2127 |
| 184433 | 2021 | no | H | 402 | 0.2 % | QH (100 %) | 3063-3927 | 3063-3927 |
| 161608 | 2015 | no | WP | 387 | 0.0 % | QH (39 %) | 3741-5461 | - |
| 190508 | 2022 | yes | L | 325 | 6.8 % | H (50 %) | 834-1539 | - |
| 174653 | 2018 | no | WP | 266 | 0.4 % | QH (98 %) | 4130-4799 | 4363-4799;4130-4320 |
| 189329 | 2022 | yes | L | 255 | 0.4 % | H (100 %) | 1162-1989 | 1162-1212;1842-1989;1242-1417;1642-1818;1442-1618 |
| 172211 | 2017 | no | QH | 239 | 1.7 % | H (98 %) | 1500-2030 | 1500-2030 |
| 154771 | 2013 | no | L | 229 | 0.0 % | QH (70 %) | 2167-2680 | - |
| 182682 | 2020 | no | H | 228 | 0.0 % | WP (92 %) | 1948-3819 | 1948-3819 |
| 182665 | 2020 | no | QH | 216 | 0.0 % | WP (99 %) | 1800-2300 | 1800-2300 |
| 159372 | 2014 | no | L | 198 | 0.0 % | QH (100 %) | 921-1367 | 921-1367 |

**As a diagnostic only, not a result:** the row scored without these blocks gives 0.860 [0.79,
0.90] on 302 shots (215,111 windows; L 0.92, H 0.93, QH 0.88, WPQH 0.70) against 0.736. Dropping the
blocks the network gets most wrong is selection on the outcome, so the number measures nothing; it
shows that the deficit of this row sits in 6.6 % of its windows on 22 shots, and that even
without them the score stays below the paper's 0.94.

### Confident learning: intervals the BES network doubts

A label-noise estimate in the manner of Northcutt et al. (2021), written here and applied to
intervals (no code from their repository, `src/labeler/confinement/confident.py`): a class's
threshold is the mean out-of-fold probability of the windows labelled with it; a window is
confidently of the class with the largest probability among those at or above their thresholds; an
interval is flagged when it holds at least 5 kept windows and at least half of them are confidently
one other class. Out-of-fold probabilities come from **`full_cum_abcdrge`**, the paper-training row
on the beam-gated windows (230,282 windows of 316 shots, each predicted by a network that never saw
its shot; windows outside the gate and within 20 ms of an interval end are left out), so a flagged
interval is one a network reading beam-valid BES doubts. The first version of this analysis used the
ungated `only_ge` row, which scores passive-BES windows too (the 150L beam off, where a "beam-valid"
claim does not hold); its record `confident_only_ge.json` (72 of 931 intervals flagged) is kept as
that earlier, ungated run and is not the number to cite. Script:
`scripts/labeler/confinement_bes_confident.py`; record:
`outputs/labeler/confinement/bes/confident_full_cum_abcdrge.json` (all 48 flagged intervals; the CSV
is `$LABELER_ROOT/round4/conf/confident/intervals_full_cum_abcdrge.csv`).

- **48 of 558 scored intervals (8.6 %) are flagged**, with an estimated label noise of 10.2 % of the
  windows; 8.9 % of the 505 intervals where Gill's and Butt's tables agree and 5.7 % of the 53 from
  one source. A gated network has fewer low-class windows to doubt: 12 of the 108 L-mode intervals
  (11 %) are taken for H, none of the 196 H intervals for L.
- **QH and WPQH**: 13 of the 254 intervals (5.1 %) are taken for the other of the two (QH read as
  WPQH 7 of 212, WPQH read as QH 6 of 42, the thin class). The list (shot, interval in ms): QH read
  as WPQH, 190507 (1799-2000, 1372-1787, 2010-2079 and 2090-4244), 182665 (1800-2300), 174634
  (1700-3000), 195821 (1600-3400); WPQH read as QH, 190514 (3024-4916), 174621 (1837-2653), 169848
  (1302-1664), 174653 (4363-4799 and 4130-4320), 195825 (1400-1500). A reviewer should look at the
  long ones first (190507, 190514, 195821 and 174634 span 1.3 s or more).
- Of the 48 flagged, 10 are also mostly (over half) not the labelled class by `confine-ours`, which
  reads the 0D signals and not BES ([confinement_ours.md](confinement_ours.md)); 4 of them give the
  same other class as the BES network and 5 are QH/WPQH: shots 169848, 174653, 179634, 190514,
  190666, 192098, 192711, 192942, 195821 and 195825. Those are the likeliest label errors; the other
  38 are doubted only by the BES network. `confine-ours` is not independent of the labels (it
  learned from them), so the opinion is weak where it agrees with them.
- None of this edits a label. The flagged intervals are for the review page.

### What our H class contains, against the paper's, is unverified

The paper's H-mode is "standard ELMy H-mode" with H98y2 of at least 1 and no dithering; it leaves
out ELM-suppressed, ELM-free and negative-triangularity H-modes. Our H intervals are the experts'
(merged from Gill's tables), and **what they contain against those exclusions is UNVERIFIED**: no
check here shows that they hold such plasmas, and none shows that they do not. This page and
[confinement_ours.md](confinement_ours.md) say the same, and neither states either reading as fact.
Two cheap checks were possible without fetching (script `scripts/labeler/confinement_h_class_checks.py`,
record `outputs/labeler/confinement/bes/h_class_checks.json`); they are observations:

- **I-coil currents** (the 12 coils of the `rmp` group of the corpus HDF5, on all 117 corpus
  shots; peak |I| over the coils and over each curated interval; 1 kA is a reading convention, not a
  threshold of ELM suppression). Resonant perturbations that suppress ELMs take a few kA, but the
  same coils are driven near 1.5 kA for non-resonant fields in QH-mode work, so a current says "a field is
  applied", not "ELMs are suppressed":

  | Class | Corpus shots | Intervals | Shots with a peak at or above 1 kA | Intervals at or above 1 kA | Median shot peak (A) | Largest shot peak (A) |
  |---|---|---|---|---|---|---|
  | L | 38 | 272 | 9 | 13 | 19 | 1,546 |
  | H | 74 | 408 | 10 | 34 | 27 | 5,101 |
  | QH | 16 | 20 | 12 | 16 | 1,523 | 1,561 |
  | WPQH | 20 | 268 | 15 | 205 | 1,530 | 2,630 |

  Of the 74 corpus shots with an H interval, 10 reach 1 kA while labelled H, five of them 2.9 to
  5.1 kA (189189 and 189191 at 5.1 kA, 195510, 195511 and 195550 at 2.9 to 3.3 kA; the other five
  1.0 to 1.6 kA), sizes at which resonant perturbations are run. So the H class holds some shots
  with an applied field; whether any is ELM-suppressed is not read from the currents. The other 64
  of the 74 stay below 1 kA.
- **EFIT triangularity of 179634** (2019; labelled L over 800-5000 ms and called QH throughout, the
  first row of the failure list above): it would show whether this is a negative-triangularity
  discharge, which the paper left out. **Not run**: the plasma shape is not on disk (the geometry
  records keep only psi_N at the 64 BES channels, and the 0D records the stored energy and
  normalised beta), and fetching was out of scope.

The H98y2 restriction and an ELM-state check stay under "Not done".

### What the labels rest on

The curated intervals were set by experts from other measurements than BES: L and H from the D-alpha
level and its ELM pattern, QH from the edge harmonic oscillation seen in the magnetic spectrogram,
and WPQH from the pedestal width measured by Thomson scattering. The BES network reads none of
these, so it reproduces the labelling rule only through the density fluctuations the same plasma
state leaves in BES, and the QH/WPQH boundary is where that link looks weakest (the first retrain
calls 1,535 of its 5,608 QH windows WPQH); the BES spectra are the network's input, not the basis of
the labels. A reviewer checking a flagged QH or WPQH interval should read the magnetic spectrogram
and the Thomson pedestal width, not the BES spectrum.

## The coverage figure (appendix)

`scripts/labeler/paper/fig_confinement_coverage.py` draws every curated shot outside the cohort's
blind test split (446) as one row, in shot-number order over three panels, and writes
`$LABELER_ROOT/round4/conf/fig_confinement_coverage.{pdf,png}` (a full-page figure, 9 in tall at the
ICML text width; `--table` writes the year marginal as CSV). Each row holds the labelled regimes (L,
H, QH, WPQH) and, in light grey beneath them, the time the beam gate of Gill et al. (2024) is met
(150L at or above 700 kW, 150R at or below 200 kW; blank for 175658, which has no beam record), so
the gate reads apart from the two BES squares left of the row (right, mid grey: native 1 MHz BES
fetched, 444 shots; left, black: the shot is also one of the 117 the corpus holds BES for at 500 kHz). The time axes
end at the last labelled time of the data (6.5 s). The marginal bars give the labelled regime time
by year of the shot and per 0.2 s of shot time. The year is not on disk for shots before 2021; it is
the year the shot's EFIT01 reconstruction was inserted into MDSplus, in the server's local (Pacific)
time (the stamps carry no time zone and fall between 08:00 and 21:59; they are not UTC)
(`confinement_shot_dates_fetch.py` reads `\EFIT01::TOP.RESULTS.GEQDSK:GTIME`; the file is
`$LABELER_ROOT/round4/conf/dates.csv`: 445 of the 446 shots carry a stamp; the one without, 175658,
and four whose stamp lies more than 30 days from their neighbours' (an EFIT run again later: 149996,
195510, 195511, 195550) take the year of their neighbouring shots). Shots per year: 2012 5, 2013 3,
2014 63, 2015 26, 2016 21, 2017 53, 2018 51, 2019 36, 2020 5, 2021 85, 2022 73, 2023 25. The 117
corpus shots are 20 from 2021, 72 from 2022 and 25 from 2023.

**Caption draft.** *Coverage of the confinement labels. Each row is one of the 446 curated
discharges outside the held-out test set, in shot-number order (three panels); the bar is the
labelled regime over the discharge (L blue, H orange, QH green, WPQH pink; unlabelled time blank),
the light-grey strip beneath it the time the beam gate of the reference BES classifier is met (150L
beam at or above 700 kW, 150R at or below 200 kW). The two squares to the left of a row mark the BES
record: the left one black where the shot is in the 500 kHz corpus (117 shots, all 2021-2023), the
right one grey where its native 1 MHz BES was fetched (444 shots; a corpus row shows both, the
others the grey one alone). Bottom: labelled regime time by year of the discharge (left; the year is that of the
EFIT reconstruction's insertion into MDSplus, with the number of discharges above each bar) and per
0.2 s of discharge time for all discharges (right).*

## Not done

- The H-mode definition: the paper's H-mode is ELMy H-mode with H98y2 of at least 1; whether ours is
  the experts' wider class is unverified (see "What our H class contains"). Restricting the H intervals to H98y2 of at least 1 (EFIT confinement
  time) and rescoring would test whether the label convention explains part of the remainder.
  Not run: H98y2 is not on disk (the 0D records hold the density, the stored energy and betaN),
  and a fetch was out of scope in fix round 3.
- No row draws the five test sets of the paper's split without overlap; the per-split mean, the
  first-occurrence score and the repeat-averaged score bracket the effect.
- The corpus-shot deficit cannot be attributed with what is on disk: the 117 corpus shots
  (2021-2023) differ from the rest in campaign, possibly in who set their labels and how (the
  interval-length table shows no finer intervals, so this is not known), in sampling (500 kHz in
  the corpus, 1 MHz fetched) and in the mix of classes, and no row varies one of these while
  holding the others. A label audit of those shots against the logbook would separate the label
  convention from the campaign. Not run.
- Anything the paper leaves open was set once and not tuned: dropout (0.2), the Adam betas, the
  batch size (256), the window stride (2,048 samples), class weights. Every row is one seed per fold or split; a second seed would show the run-to-run
  spread, which the bootstrap intervals (over shots, not over seeds) do not include.
- The paper trained on 48 GPUs (its batch size per device is not stated); one GPU and a batch of
  256 here are a difference of optimisation, not tested.
- The blocks of the `leak_*` rows are diagnostics of the split, not benchmark numbers: the 0.1
  to 0.2 s blocks leave neighbouring windows of one shot on both sides of the split by design.
- Two curated shots have no BES in the archive (171438, 175658), and 3 more (185920, 186637,
  186641) have no position record, so 441 of the 444 shots can be placed by geometry.
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
