# RWM baseline: rwm-brf and rwm-nnpu

Status: **run 2026-10-03 on branch `r4-rwm`.** DIII-D has no published RWM predictor that
we know of, so the owner asked for the paper that comes closest, Piccione et al. 2022
(NSTX, balanced random forest), to be adapted as the baseline "given the labels", and for a
positive-unlabelled variant because Jeremy Hanson's tables list onsets only. This note
records the data, the labels, the models, the protocol and every number with the file it
comes from. The tables below are copied from `outputs/labeler/rwm/tables.md`, which
`scripts/labeler/rwm_tables.py` renders from the JSON records; none was typed by hand.

## Result in brief

- **Labels and inputs exist.** Inputs were fetched for 208 candidate shots (Hanson's 33 and
  175 from the same run days and experiments); 165 are used: the 33 Hanson shots and 132
  comparison shots (4.0 per Hanson shot, matched within campaign on flat-top beta_N and
  beta_N / l_i). The label table `data/events/resistive_wall_mode/extend_rwm_growth/rwm_windows.csv`
  holds 54 growth windows of 20 ms before the 54 distinct listed onsets (48 with n = 1), 84
  absent spans on the Hanson shots, and 128 uncertain spans of the screen on comparison shots.
  The comparison shots are unlabelled, never negative.
- **rwm-brf ranks "n = 1 onset within 100 ms" above other flat-top time on the Hanson
  shots: AUROC 0.798 [0.740, 0.845], AUPRC 0.084 [0.061, 0.116]** (5-fold shot-grouped cross-validation, 480
  positive and 13,848 negative 10 ms slices, 95 % shot-bootstrap intervals). The prevalence is
  0.034, so the AUPRC is about 2.5 times chance. rwm-nnpu is the same (0.794 [0.743, 0.845]).
- **That ranking is not shown to be better than a single number.** beta_N / l_i read
  off as a score has AUROC 0.752 [0.688, 0.815]; the paired difference rwm-brf minus that rule is
  0.046 [-0.018, 0.114] (the interval spans zero). Of the features, the equilibrium scalars
  carry it (single-feature AUROC: l_i 0.777, beta_N / l_i 0.752, beta_N 0.715, W_MHD 0.704).
  The n = 1 and n = 2 RMS and the rotation together add 0.031 [0.008, 0.056] AUROC to the
  equilibrium scalars; the rotation alone adds none (-0.002 [-0.009, 0.006]).
- **As a warning system the models are not shown to beat chance.** rwm-brf warns of 17 of
  48 onsets (detection rate 0.354 [0.217, 0.500]); randomly placed alarms at its own alarm rate
  (0.91 alarms per scored second on the Hanson shots) would warn of 32.8 %. It raises at
  least one alarm with no onset near on 21 of 33 Hanson shots. rwm-nnpu warns of 20 of 48
  (chance 34.9 %).
- **The Legacy numbers are not comparable one to one.** Piccione et al. report AUC 0.918,
  TPR 92.4 % and FPR 21.4 % on a fixed 28-shot test set of NSTX, with 10 of 11 unstable
  shots detected and 2 of 17 stable shots falsely alarmed. Ours is DIII-D, fixed
  hyperparameters, out-of-fold scores over 33 shots that all went (or may have gone) near the
  limit, and a within-shot slice score; the closest like-for-like figure, the score with the
  comparison shots counted as negatives (a lower bound), is AUROC 0.899 [0.866, 0.928].
- **The data are small.** 48 distinct n = 1 onsets on 30 shots (3 of the 33 listed shots have
  only n = 2 onsets) in two campaigns, 20 + 13 shots. See "What the numbers can and cannot
  support" before quoting any of this.

## What the numbers can and cannot support

Can support:

- That the labels, inputs, features and protocol are in place and reproducible: repeated full
  runs of `scripts/labeler/rwm_evaluate.py` gave the same numbers to 1e-9.
- That on Hanson's shots, equilibrium quantities near the no-wall limit (high beta_N,
  low l_i) rank the 100 ms before an n = 1 onset above the rest of the flat-top, at AUROC
  about 0.8 against 0.5 for no information, in both campaigns (out-of-fold point estimates: 2014 0.797, 2018
  0.800).
- A first DIII-D reference to improve on, with its failure modes: low-threshold alarms
  fire before and away from onsets, and the 1 kHz magnetics and the rotation add only about
  0.03 AUROC to the equilibrium scalars.

Cannot support:

- **A ranking of models.** The intervals of rwm-brf, rwm-nnpu and the beta_N / l_i rule
  overlap, and the paired differences on shared shot resamples include zero (table
  "paired differences"). The only measurable differences are feature-group ones:
  dropping the magnetics and rotation costs 0.031 AUROC (interval above zero).
- **A warning-time or alarm claim.** The detection rate is at the chance level of the same
  alarm rate (table "per-shot alarm scores", column "detection by chance"), and the
  false-alarm rate on Hanson shots is high. The alarm rule is fitted on 48 onsets, and the
  split seed alone moves the detection rate between 0.23 and 0.44 (table "split-seed
  sensitivity").
- **A false-alarm rate on unlabelled shots.** The 132 comparison shots are RWM
  experiment days; an alarm on one may be an unlisted RWM. The comparison-shot alarm rate
  (15.9 % of shots for rwm-brf) is an upper bound on a true false-alarm rate, and the
  "mixed" scores that count those slices as negatives are lower bounds.
- **A comparison with NSTX.** Different machine, inputs (no DCON no-wall limit, no
  collisionality), cadence, hyperparameter search (Piccione et al. tuned with 1000 TPE trials,
  here they are fixed), positive definition by a different table, and a fixed test split in
  place of out-of-fold scores.
- **Generalisation across campaigns or devices.** Folds are grouped by shot and
  stratified by campaign, so both campaigns are in every training set. A leave-one-campaign-out
  test was not run; the locked-mode detector, which tags the campaign, shows how easily a model
  could learn the era (below).
- **Anything about RWMs that Hanson did not list.** Onsets he lists are the only positives;
  RWMs he omitted sit in the negatives and the unlabelled sets and lower every score.

## Data

### Hanson shots

`data/events/resistive_wall_mode/raw/rwm_onsets_{2017,2024}.csv`: 56 onsets on 33 shots,
156785-158023 (20 shots, 2014) and 176067-176092 (13 shots, 2018). Two onsets of one shot
within 10 ms are one event (`labels.merge_close`), which leaves 54 distinct events: 48 with
toroidal mode number n = 1 and 6 with n = 2. The baselines forecast n = 1 onsets; the n = 2
ones are excluded from the target (slices from 100 ms before to 100 ms after are neither
positive nor negative), and `rwm-brf-all-modes` counts them as positives. Shots 158015,
158023 and 176092 have only n = 2 onsets. None of the shots is in the corpus or in the frozen
cohort (`outputs/labeler/rwm/shots.json`, `cohort_overlap`: none of the 165 used shots is among
the 500 cohort shots), so no `split == test` shot was trained on or tuned against.

### Comparison shots: how they were chosen

A model fitted on one era against another learns the era, so the comparison shots come
from the same two campaigns. `scripts/labeler/rwm_pool.py` reads the operator logbook's run
record (`labeler.rwm.shots.choose`) and lists, besides the Hanson shots, the candidates
of two kinds: every other plasma shot of a run day that holds a Hanson shot (`same_day`, 20),
and every plasma shot of a run day of the same year whose title is one of the high-beta_N
or RWM experiments of the Hanson days (`same_experiment`, 155; titles in
`shots.EXPERIMENT_TITLES`). That is a pool of 208 shots, 207 with usable inputs (one has
no n = 1 RMS).

`scripts/labeler/rwm_build.py` measures every pool shot's flat-top beta_N and beta_N / l_i
(95th percentile over the window where Ip is at least half its peak) and matches four
comparison shots to each Hanson shot of its campaign, nearest in those two numbers, without
reuse, the Hanson shots with the highest beta_N first so the scarce high-beta_N shots are
not used up (`shots.match_comparison`). That gives 132 comparison shots, 4.0 per Hanson
shot (aim: at least 3), 80 in 2014 and 52 in 2018. The balance after matching:

| campaign | set | shots | mean flat-top beta_N p95 | mean beta_N/l_i p95 |
|---|---|---|---|---|
| 2014 | Hanson shots | 20 | 3.25 | 5.07 |
| 2014 | chosen comparison | 80 | 3.31 | 4.43 |
| 2014 | pool, not chosen | 22 | 4.08 | 4.65 |
| 2018 | Hanson shots | 13 | 3.05 | 4.98 |
| 2018 | chosen comparison | 52 | 3.13 | 5.11 |
| 2018 | pool, not chosen | 20 | 3.36 | 5.63 |

The comparison set is close in beta_N. In 2014 the Hanson shots reach a higher
beta_N / l_i than the comparison shots (5.07 against 4.43), so the match is not perfect
there (the pool holds no closer shots); in 2018 it is close (4.98 against 5.11). Source: `outputs/labeler/rwm/shots.json`; roster of all 208 candidates with the
choice: `data/events/resistive_wall_mode/extend_rwm_growth/rwm_windows.shots.csv`.

### Inputs fetched

`scripts/labeler/rwm_fetch.py` fetched, for all 208 candidates (about 78 shot-minutes, three workers at
pace 1, run on the login node under `fdp run`, log under `$LABELER_ROOT/round4/rwm/`),
into the raw cache `$LABELER_ROOT/raw/<shot>_processed.h5`: `ip`, `bt`, `betan`, `li`,
`q95`, `qmin`, `wmhd`, `n1rms`, `n2rms`, the ZIPFIT toroidal rotation profile
(`rot_zipfit`) and the locked-mode detector (`dusbradial`). Some inputs of the
Hanson shots were already in the cache from the review editor; a rerun fetches only what a
shot lacks. Three shots lacked a signal at the source (157998: no RMS; 176093 and 176094: no
ZIPFIT rotation); 157998 is the unusable shot. `q95` and `dusbradial` were added as
canonical features (`labeler.features.namespace`) with a test. The brief asked for CER
rotation if fetchable; the fitted ZIPFIT profile, which comes on a regular time grid, was
used instead and the raw CER channels were not fetched.

Coverage of the 165 used shots: every shot has every input except one 2018 comparison shot
without rotation; by slice, the EFIT-derived inputs are present on 94 to 99 % and rotation on
71 to 92 % (`shots.json`, `feature_coverage`).

**The locked-mode detector is an era tag, not a feature.** `DUSBRADIAL` is exactly zero on
every 2014 shot and nonzero on 2018 shots, so it separates campaigns whatever the plasma does.
It is stored but kept out of the model; `rwm-brf-with-locked-mode` lets it in (table
"feature and training ablations").

## Labels

An onset is a point; two things are made from it, and they are different.

**The growth window (the catalog label).** The 20 ms before each listed onset, of either
mode number, is category 1, present. `GROWTH_MS` in `src/labeler/rwm/labels.py`; the
measurement behind it is `scripts/labeler/rwm_growth.py` -> `outputs/labeler/rwm/growth.json`.
The mode grows on the wall time tau_w (a few ms on DIII-D), so the growth before the
onset is short. Measured on the n = 1 RMS of the 48 distinct n = 1 onsets, the largest
trailing 20 ms growth rate between 150 ms before and 30 ms after each onset has a median
e-folding time of 9.0 ms
(quartiles 5.8 and 11.0 ms), so 20 ms is about 2.2 e-foldings (1.8 to 3.5).

| quantity (48 n=1 onsets) | first quartile | median | third quartile |
|---|---|---|---|
| largest 20 ms growth rate of N1RMS, -150 to +30 ms from the onset (per s) | 91 | 112 | 174 |
| e-folding time at that rate (ms) | 5.8 | 9.0 | 11.0 |
| time of the maximum growth relative to the onset (ms) | -103.5 | -39.0 | 0.0 |
| beta_N 100 ms before the onset | 2.55 | 2.74 | 3.01 |
| beta_N at the onset | 2.55 | 2.68 | 2.85 |
| beta_N 40 ms after the onset | 1.09 | 1.93 | 2.36 |

**The limits of that justification.** The 1 kHz RMS does not show a growth burst that
separates the onsets from random flat-top times: the 20 ms log-ratio at the onset exceeds
the 90th percentile of 8,696 random control times for 12.5 % of the onsets (10 % is
chance), and the fastest growth sits a median 39 ms before the listed onset. The listed
onsets coincide with the start of the beta_N collapse instead (median beta_N 2.68 at the
onset, 1.93 forty ms later; 56 % of the onsets fall by 20 % within 40 ms). So the RMS cannot
place the start of the growth more sharply than the wall time, and 20 ms is a convention
resting on the wall time: it is the "short growth time" the brief asked for, not a
measured start of the mode. Changing it changes the catalog windows only; the baselines
use the forecast target below.

**Absent.** On the 33 Hanson shots, the high-current window (Ip at least half its peak)
outside 100 ms before to 100 ms after every listed onset is category 0, absent: 84 spans.
This treats the Hanson shots as examined, which is an assumption: the tables list onsets and
say nothing about the rest of a shot. The 80 ms between 100 ms and 20 ms before an onset (the
precursor) and the 100 ms after it (while the mode acts) carry no row, not assessed. Shots outside the
tables are never absent.

**Uncertain.** The `rwm_candidates` screen's spans on the comparison shots (128) are
category 2, uncertain: weak labels, not negatives. The screen flags 62 negative and no
positive slices of the Hanson shots (7 shots) and 681 of the 68,006 comparison slices.

**The forecast target (the baselines).** As Piccione et al. define their stability label,
a slice is positive when an n = 1 onset follows within 100 ms (`labels.HORIZON_MS`), and
negative otherwise; slices within 100 ms after an onset, and within 100 ms either side of
an n = 2 onset, are excluded from both classes. The comparison shots' slices keep the label
"unlabelled" and are used only as described below. Slices are every 10 ms in the flat-top
(the paper uses 5 ms; the EFIT cadence is 20 ms): 14,919 on Hanson shots (480 positive,
13,848 negative, 591 excluded) and 68,006 on comparison shots (`shots.json`, `slices`).

## Features

`labeler.rwm.features` builds one causal row per grid time, as a real-time system would
have seen it: a held value (EFIT 50 ms, ZIPFIT 100 ms maximum age), trailing windows for
rates, nothing looks ahead. The 14 model inputs: beta_N, l_i, q95, qmin, W_MHD, beta_N / l_i,
beta_N - 4 l_i (the no-wall proxy, beta_N,no-wall ~ 4 l_i), Ip; the n = 1 RMS (5 ms window),
its 20 ms peak and its 20 ms log growth rate, the n = 2 RMS; ZIPFIT rotation at rho = 0.25
and 0.625. Missing values are filled with training-fold medians. The locked-mode amplitude
(15th column) is excluded as an era tag.

Single-feature AUROC on the labelled Hanson slices (no fitting; the direction is the better
of the two, so these are optimistic by that choice):

| feature | AUROC | unstable when | fraction of slices with a value |
|---|---|---|---|
| betan | 0.715 | higher | 0.987 |
| li | 0.777 | lower | 0.987 |
| q95 | 0.542 | higher | 0.987 |
| qmin | 0.656 | higher | 0.987 |
| wmhd_mj | 0.704 | higher | 0.987 |
| betan_over_li | 0.752 | higher | 0.987 |
| betan_minus_4li | 0.754 | higher | 0.987 |
| ip_ma | 0.550 | higher | 1.000 |
| n1rms_g | 0.553 | higher | 1.000 |
| n1rms_max_g | 0.530 | higher | 1.000 |
| n1rms_growth_per_s | 0.512 | lower | 1.000 |
| n2rms_g | 0.565 | higher | 1.000 |
| rot_core_khz | 0.680 | higher | 0.854 |
| rot_mid_khz | 0.631 | higher | 0.854 |
| lock_v | 0.599 | lower | 1.000 |

## Models and protocol

- **rwm-brf**: a balanced random forest after Piccione et al. (each tree draws a bootstrap
  of the positive slices plus an equal-size random draw of negatives), 300 trees, maximum
  depth 8, minimum leaf 5, written in numpy (`labeler.rwm.forest`; there is no scikit-learn
  or imbalanced-learn in the environment). The training negatives are the Hanson negatives
  and the unlabelled comparison slices, treated as assumed negatives; `rwm-brf-hanson-only`
  trains on the Hanson shots alone.
- **rwm-nnpu**: the same features through a small network (two layers of 32) under the
  non-negative PU risk of Kiryo et al. 2017 (`labeler.rwm.nnpu`). Labelled = positive
  slices; unlabelled = every other slice except the excluded ones (Hanson negatives and
  comparison slices). The class prior is the share of positive slices among the training
  slices (about 0.006) times `prior_scale` (1, 0.5, 2 for the sensitivity runs). The logistic
  loss replaces Kiryo et al.'s sigmoid loss, which collapsed to a constant output at this
  prior; 15 epochs, weight decay 1e-3.
- **Rule baselines**: `rule-betan-over-li` (beta_N / l_i as a score, nothing fitted) and
  `rule-rwm-candidates` (the screen's call as a 0/1 score).
- **Cross-validation**: 5 outer folds of whole shots, stratified by role (Hanson or
  comparison) and campaign; no shot is on both sides. Inside each training set a 3-fold
  inner cross-validation gives out-of-fold scores, on which the slice cutoff (the ROC
  point nearest (0, 1), as in the paper) and the alarm rule are chosen; a held-out shot
  never sets a threshold that scores it. Per-slice AUROC, AUPRC, TPR, FPR, precision and
  F1 are over the labelled Hanson slices; "mixed" scores add the comparison slices as
  negatives (a lower bound).
- **Alarms** (the paper's hysteresis rule): fire when the score has crossed a high threshold
  and then stays above a low threshold for a hold time. The three parameters are chosen
  on the training shots from a grid (high threshold at the 0.90, 0.95, 0.98, 0.99 quantiles
  of the training negatives' scores, low threshold at 0.5, 0.75, 1.0 of the way up, hold 0,
  20, 50 ms) to maximise the share of onsets warned minus the share of shots with an
  unexplained alarm. An alarm 10 to 400 ms before an onset warns of it; an alarm that is
  neither in that range nor within 100 ms after an onset is a false alarm (the paper has a
  separate "too early" column; here too early counts as false).
- **Intervals**: 95 % percentile intervals from 1000 shot bootstraps, the Hanson and
  comparison shots resampled as two strata (`labeler.rwm.metrics.shot_bootstrap`). The
  paired differences use the same resamples for both models (`paired_bootstrap`).
  The fold seed is 0; the headline model was rerun for seeds 1 to 4 as a check.

## Results

All numbers: `outputs/labeler/rwm/evaluation.json` (per-shot outcomes for rwm-brf,
rwm-nnpu and the rule: `configs.<name>.per_shot`; paired differences: `paired`), counts and
protocol in the same file, shot selection in `outputs/labeler/rwm/shots.json`.

### Slice scores, Hanson shots

| model | AUROC | AUPRC | F1 | TPR | FPR |
|---|---|---|---|---|---|
| rwm-brf | 0.798 [0.740, 0.845] | 0.084 [0.061, 0.116] | 0.149 [0.112, 0.193] | 0.854 [0.768, 0.932] | 0.333 [0.282, 0.388] |
| rwm-nnpu | 0.794 [0.743, 0.845] | 0.083 [0.060, 0.134] | 0.136 [0.104, 0.176] | 0.767 [0.655, 0.874] | 0.329 [0.268, 0.394] |
| rule-betan-over-li | 0.752 [0.688, 0.815] | 0.074 [0.052, 0.115] | 0.141 [0.103, 0.183] | 0.735 [0.602, 0.847] | 0.302 [0.245, 0.363] |
| rule-rwm-candidates | 0.498 [0.495, 0.500] | 0.034 [0.026, 0.041] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.004 [0.001, 0.009] |

With the comparison shots counted as negatives (lower bounds, they may hold unlisted RWMs):

| model | AUROC | AUPRC | FPR |
|---|---|---|---|
| rwm-brf | 0.899 [0.866, 0.928] | 0.047 [0.032, 0.069] | 0.186 [0.161, 0.213] |
| rwm-nnpu | 0.900 [0.867, 0.933] | 0.052 [0.035, 0.091] | 0.172 [0.146, 0.199] |
| rule-betan-over-li | 0.784 [0.718, 0.843] | 0.017 [0.011, 0.027] | 0.265 [0.240, 0.291] |
| rule-rwm-candidates | 0.495 [0.493, 0.498] | 0.006 [0.005, 0.007] | 0.009 [0.005, 0.015] |

The screen flags none of the 480 positive slices, so its AUROC is 0.5 by construction; it
never fires within 100 ms before a listed onset. It is a review aid for large n = 1
excursions, not a forecaster.

### Per-shot scores (the paper's warning and false-alarm metrics)

| model | onsets warned | detection rate | detection by chance | median warning (ms) | Hanson shots with a false alarm | rate | per shot | comparison shots with an alarm | rate | per shot |
|---|---|---|---|---|---|---|---|---|---|---|
| rwm-brf | 17 of 48 | 0.354 [0.217, 0.500] | 0.328 | 330 [276, 367] | 21 of 33 | 0.636 [0.485, 0.788] | 3.03 [1.91, 4.30] | 21 of 132 | 0.159 [0.098, 0.227] | 0.83 [0.36, 1.36] |
| rwm-nnpu | 20 of 48 | 0.417 [0.296, 0.553] | 0.349 | 203 [143, 297] | 22 of 33 | 0.667 [0.514, 0.818] | 3.27 [2.09, 4.46] | 15 of 132 | 0.114 [0.061, 0.174] | 0.59 [0.28, 0.98] |
| rule-betan-over-li | 1 of 48 | 0.021 [0.000, 0.070] | 0.029 | 188 [188, 188] | 5 of 33 | 0.152 [0.030, 0.273] | 0.24 [0.06, 0.45] | 17 of 132 | 0.129 [0.076, 0.182] | 0.30 [0.15, 0.48] |
| rule-rwm-candidates | 0 of 48 | 0.000 [0.000, 0.000] | 0.021 | - | 7 of 33 | 0.212 [0.091, 0.364] | 0.42 [0.12, 0.79] | 36 of 132 | 0.273 [0.197, 0.348] | 0.80 [0.50, 1.14] |

Scored time and alarm rate (the basis of the "detection by chance" column, which is the
expected share of onsets with an alarm in their 390 ms warning range if the shot's alarms
fell at random over its scored span, `labeler.rwm.evaluate.chance_detection`):

| shots | count | mean scored span (s) | shortest | longest | total (s) | alarms | alarms per scored second |
|---|---|---|---|---|---|---|---|
| hanson | 33 | 4.51 | 1.35 | 5.89 | 148.9 | 136 | 0.91 |
| comparison | 132 | 5.14 | 1.56 | 6.87 | 678.7 | 109 | 0.16 |

| shots | count | mean scored span (s) | shortest | longest | total (s) | alarms | alarms per scored second |
|---|---|---|---|---|---|---|---|
| hanson | 33 | 4.51 | 1.35 | 5.89 | 148.9 | 168 | 1.13 |
| comparison | 132 | 5.14 | 1.56 | 6.87 | 678.7 | 78 | 0.11 |

Per Hanson shot (warning time before each n = 1 onset, "-" missed; unexplained alarms):

| shot | campaign | n=1 onsets | warning per onset (ms; - missed) | false |
|---|---|---|---|---|
| 156785 | 2014 | 1 | - | 5 |
| 156786 | 2014 | 2 | 392, - | 0 |
| 156787 | 2014 | 2 | -, 237 | 0 |
| 156790 | 2014 | 1 | 300 | 2 |
| 156791 | 2014 | 1 | - | 0 |
| 156792 | 2014 | 2 | 348, - | 2 |
| 156793 | 2014 | 2 | 166, 395 | 5 |
| 156794 | 2014 | 2 | -, - | 14 |
| 156795 | 2014 | 2 | 276, 363 | 1 |
| 156796 | 2014 | 1 | - | 2 |
| 156797 | 2014 | 2 | -, - | 0 |
| 158012 | 2014 | 1 | - | 14 |
| 158013 | 2014 | 2 | 340, - | 5 |
| 158014 | 2014 | 1 | 367 | 4 |
| 158015 | 2014 | 0 | - | 0 |
| 158018 | 2014 | 1 | - | 3 |
| 158019 | 2014 | 1 | - | 4 |
| 158021 | 2014 | 1 | - | 0 |
| 158022 | 2014 | 1 | - | 5 |
| 158023 | 2014 | 0 | - | 7 |
| 176067 | 2018 | 1 | - | 0 |
| 176068 | 2018 | 4 | -, 112, 390, 383 | 1 |
| 176069 | 2018 | 3 | 50, -, 330 | 3 |
| 176070 | 2018 | 2 | -, - | 4 |
| 176071 | 2018 | 1 | 281 | 1 |
| 176074 | 2018 | 2 | -, - | 9 |
| 176077 | 2018 | 1 | - | 6 |
| 176078 | 2018 | 2 | 195, - | 3 |
| 176085 | 2018 | 1 | - | 0 |
| 176087 | 2018 | 1 | - | 0 |
| 176088 | 2018 | 3 | -, -, - | 0 |
| 176089 | 2018 | 1 | - | 0 |
| 176092 | 2018 | 0 | - | 0 |

| shot | campaign | n=1 onsets | warning per onset (ms; - missed) | false |
|---|---|---|---|---|
| 156785 | 2014 | 1 | - | 7 |
| 156786 | 2014 | 2 | -, 228 | 1 |
| 156787 | 2014 | 2 | -, 297 | 0 |
| 156790 | 2014 | 1 | 370 | 2 |
| 156791 | 2014 | 1 | - | 0 |
| 156792 | 2014 | 2 | 88, - | 0 |
| 156793 | 2014 | 2 | -, 265 | 5 |
| 156794 | 2014 | 2 | -, 56 | 8 |
| 156795 | 2014 | 2 | 166, 143 | 1 |
| 156796 | 2014 | 1 | 380 | 3 |
| 156797 | 2014 | 2 | -, 38 | 0 |
| 158012 | 2014 | 1 | 296 | 9 |
| 158013 | 2014 | 2 | 170, - | 11 |
| 158014 | 2014 | 1 | 367 | 2 |
| 158015 | 2014 | 0 | - | 7 |
| 158018 | 2014 | 1 | - | 1 |
| 158019 | 2014 | 1 | 378 | 9 |
| 158021 | 2014 | 1 | - | 0 |
| 158022 | 2014 | 1 | - | 8 |
| 158023 | 2014 | 0 | - | 3 |
| 176067 | 2018 | 1 | 178 | 7 |
| 176068 | 2018 | 4 | -, -, -, 133 | 5 |
| 176069 | 2018 | 3 | -, -, - | 0 |
| 176070 | 2018 | 2 | -, - | 3 |
| 176071 | 2018 | 1 | 341 | 1 |
| 176074 | 2018 | 2 | -, 101 | 6 |
| 176077 | 2018 | 1 | - | 8 |
| 176078 | 2018 | 2 | 105, 259 | 1 |
| 176085 | 2018 | 1 | - | 0 |
| 176087 | 2018 | 1 | - | 0 |
| 176088 | 2018 | 3 | -, -, - | 0 |
| 176089 | 2018 | 1 | - | 0 |
| 176092 | 2018 | 0 | - | 0 |

### Legacy against Tokamak-SI

"Legacy" is the published NSTX numbers from the digest of Piccione et al. 2022
(`.tmp/label_papers/Piccione_2022_Nucl._Fusion_62_036002.md`, random under-sampling forest,
28 test shots: 11 unstable, 17 stable); "Tokamak-SI" is ours on DIII-D.

| setting | slice AUROC | slice TPR | slice FPR | unstable shots detected | false alarms |
|---|---|---|---|---|---|
| Legacy: Piccione et al. 2022, NSTX, 28 test shots | 0.918 | 0.924 | 0.214 | 10 of 11 unstable shots | 2 of 17 stable shots |
| Tokamak-SI: rwm-brf, DIII-D, shot-grouped CV | 0.798 [0.740, 0.845] | 0.854 [0.768, 0.932] | 0.333 [0.282, 0.388] | 12 of 30 shots with an n=1 onset (17 of 48 onsets) | 21 of 33 Hanson shots; 21 of 132 unlabelled comparison shots (upper bound) |

Read with the cautions above. The Legacy per-shot numbers count shots (an unstable shot is
detected or missed; a stable shot is falsely alarmed or not). Ours count the Hanson shots, 30
of which have an n = 1 onset, so a false alarm there is mostly an alarm elsewhere in the
flat-top of a shot that does contain an onset. The three Hanson shots with no n = 1 onset
are the nearest thing to the paper's stable shots: rwm-brf raises an unexplained alarm on 1
of them (158023) and rwm-nnpu on 2 (158015, 158023), from the per-shot tables below. Shot
counts and onset counts are both given.

### Paired differences

| first - second | AUROC | AUPRC | detection rate | Hanson false-alarm shot rate | comparison false-alarm shot rate |
|---|---|---|---|---|---|
| rwm-brf - rule-betan-over-li | 0.046 [-0.018, 0.114] | 0.010 [-0.016, 0.032] | 0.333 [0.184, 0.491] | 0.485 [0.303, 0.667] | 0.030 [-0.045, 0.106] |
| rwm-nnpu - rule-betan-over-li | 0.041 [-0.027, 0.108] | 0.009 [-0.018, 0.045] | 0.396 [0.263, 0.546] | 0.515 [0.333, 0.697] | -0.015 [-0.091, 0.068] |
| rwm-brf - rwm-nnpu | 0.004 [-0.023, 0.028] | 0.001 [-0.027, 0.011] | -0.062 [-0.227, 0.109] | -0.030 [-0.152, 0.091] | 0.045 [-0.008, 0.098] |
| rwm-brf - rwm-brf-hanson-only | -0.007 [-0.040, 0.022] | -0.004 [-0.029, 0.018] | 0.146 [-0.045, 0.333] | 0.242 [0.030, 0.455] | -0.045 [-0.129, 0.038] |
| rwm-brf - rwm-brf-equilibrium-only | 0.031 [0.008, 0.056] | 0.016 [0.004, 0.031] | 0.083 [-0.085, 0.264] | -0.061 [-0.182, 0.061] | 0.023 [-0.030, 0.076] |
| rwm-brf - rwm-brf-no-rotation | -0.002 [-0.009, 0.006] | -0.000 [-0.010, 0.007] | -0.021 [-0.170, 0.135] | -0.091 [-0.212, 0.000] | -0.083 [-0.136, -0.038] |

Positive = the first model is higher (for the false-alarm rates, worse).

### Ablations and sensitivity

| configuration | positive slices | AUROC | AUPRC | detection rate | Hanson false-alarm shot rate | comparison false-alarm shot rate |
|---|---|---|---|---|---|---|
| rwm-brf | 480 | 0.798 [0.740, 0.845] | 0.084 [0.061, 0.116] | 0.354 [0.217, 0.500] | 0.636 [0.485, 0.788] | 0.159 [0.098, 0.227] |
| rwm-brf-hanson-only | 480 | 0.805 [0.756, 0.852] | 0.089 [0.066, 0.124] | 0.208 [0.095, 0.341] | 0.394 [0.242, 0.546] | 0.205 [0.144, 0.273] |
| rwm-brf-equilibrium-only | 480 | 0.767 [0.710, 0.818] | 0.068 [0.049, 0.096] | 0.271 [0.140, 0.417] | 0.697 [0.545, 0.848] | 0.136 [0.076, 0.197] |
| rwm-brf-magnetics-only | 480 | 0.688 [0.639, 0.740] | 0.059 [0.043, 0.086] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| rwm-brf-no-rotation | 480 | 0.799 [0.743, 0.847] | 0.085 [0.061, 0.119] | 0.375 [0.208, 0.559] | 0.727 [0.576, 0.879] | 0.242 [0.174, 0.318] |
| rwm-brf-with-locked-mode | 480 | 0.799 [0.744, 0.845] | 0.085 [0.063, 0.120] | 0.438 [0.286, 0.583] | 0.788 [0.636, 0.909] | 0.205 [0.136, 0.273] |

Leaving out the rotation changes the slice AUROC by nothing but raises the comparison-shot
alarm rate (brf minus no-rotation: -0.083 [-0.136, -0.038] in the paired table), the one
alarm-level difference between models with an interval clear of zero other than the
Hanson-only comparison. With the magnetics alone the forest ranks slices (AUROC 0.688) but its chosen alarm rules
(thresholds 0.75 to 0.78) never fire, so it warns of no onset and has no false alarm.

| configuration | positive slices | AUROC | AUPRC | detection rate | Hanson false-alarm shot rate | comparison false-alarm shot rate |
|---|---|---|---|---|---|---|
| rwm-brf-horizon-50 | 240 | 0.792 [0.738, 0.840] | 0.042 [0.031, 0.060] | 0.354 [0.229, 0.489] | 0.636 [0.455, 0.788] | 0.167 [0.106, 0.227] |
| rwm-brf | 480 | 0.798 [0.740, 0.845] | 0.084 [0.061, 0.116] | 0.354 [0.217, 0.500] | 0.636 [0.485, 0.788] | 0.159 [0.098, 0.227] |
| rwm-brf-horizon-200 | 958 | 0.806 [0.750, 0.854] | 0.170 [0.124, 0.241] | 0.396 [0.265, 0.535] | 0.758 [0.606, 0.909] | 0.197 [0.129, 0.265] |
| rwm-brf-all-modes | 540 | 0.800 [0.748, 0.844] | 0.088 [0.066, 0.126] | 0.333 [0.216, 0.471] | 0.636 [0.485, 0.788] | 0.189 [0.129, 0.258] |

The horizon changes the AUPRC (the prevalence) and barely the AUROC. The 50 ms horizon has half the
positive slices.

| configuration | positive slices | AUROC | AUPRC | detection rate | Hanson false-alarm shot rate | comparison false-alarm shot rate |
|---|---|---|---|---|---|---|
| rwm-nnpu-prior-x0.5 | 480 | 0.791 [0.738, 0.842] | 0.083 [0.059, 0.140] | 0.333 [0.217, 0.463] | 0.636 [0.455, 0.788] | 0.098 [0.053, 0.152] |
| rwm-nnpu | 480 | 0.794 [0.743, 0.845] | 0.083 [0.060, 0.134] | 0.417 [0.296, 0.553] | 0.667 [0.514, 0.818] | 0.114 [0.061, 0.174] |
| rwm-nnpu-prior-x2 | 480 | 0.792 [0.738, 0.844] | 0.083 [0.060, 0.130] | 0.333 [0.234, 0.442] | 0.636 [0.484, 0.788] | 0.106 [0.061, 0.159] |

| fold seed | AUROC | AUPRC | detection rate | comparison false-alarm shot rate |
|---|---|---|---|---|
| 0 | 0.798 | 0.084 | 0.354 | 0.159 |
| 1 | 0.807 | 0.092 | 0.438 | 0.227 |
| 2 | 0.790 | 0.088 | 0.333 | 0.318 |
| 3 | 0.800 | 0.082 | 0.229 | 0.258 |
| 4 | 0.796 | 0.083 | 0.417 | 0.227 |

By campaign (out-of-fold, rwm-brf; point estimates only):

| campaign | Hanson / comparison shots | positive slices | AUROC | AUPRC | onsets warned | Hanson shots with a false alarm | comparison shots with an alarm |
|---|---|---|---|---|---|---|---|
| 2014 | 20 / 80 | 260 | 0.797 | 0.089 | 10 of 26 | 14 of 20 | 14 of 80 |
| 2018 | 13 / 52 | 220 | 0.800 | 0.080 | 7 of 22 | 7 of 13 | 7 of 52 |

## Deviations from Piccione et al. 2022 and from the brief

1. **Features**: DIII-D scalars at the EFIT cadence on a 10 ms grid, not the paper's
   5 ms inputs. No DCON no-wall and with-wall limit (beta_N / l_i and beta_N - 4 l_i stand in
   for the no-wall limit), no omega_E or nu_ii (ZIPFIT rotation at two radii instead), no
   RMS peak frequency. The n = 1 RMS is the stored 1 kHz RMS, not a 5 ms spectrum.
2. **Hyperparameters fixed** (300 trees, depth 8, minimum leaf 5), not tuned by 1000 TPE
   trials; the forest is a numpy implementation of the same balancing (no imbalanced-learn
   in the environment).
3. **Cross-validation instead of a fixed 28-shot test set**: the 33 Hanson shots are too few
   to hold some out, so every Hanson shot is scored out of fold. The thresholds are chosen
   on inner out-of-fold scores of the training shots.
4. **Early alarms count as false**: the paper reports "early" separately.
5. **Slice labels**: n = 2 onsets are excluded from the n = 1 target (`rwm-brf-all-modes`
   is the sensitivity run). Double-listed onsets are merged. The 100 ms after an onset is
   excluded.
6. **Comparison set**: built from the run-day and experiment pool by matching on beta_N and
   beta_N / l_i (the brief's "same campaigns and similar beta_N"), 4.0 per Hanson shot
   (aim 3). They are unlabelled; the Hanson shots are taken as examined.
7. **Rotation from ZIPFIT, not CER.** Locked-mode amplitude is stored but excluded (era tag).
8. **nnPU**: logistic loss and a small regularised network, because the sigmoid loss
   collapsed at a class prior of 0.006 and a wider network overfit (train AUROC 0.98
   against 0.7 held out in a trial fold). The settings were chosen after trial runs on outer
   fold 0, whose held-out shots are also part of the final out-of-fold scores, so the nnPU
   numbers are mildly optimistic and carry a degree of freedom the forest's do not. No
   cohort test-split shot was involved.
9. **The alarm rule's objective** (detection share minus the share of shots with an
   unexplained alarm) is our choice; the digest does not say how Piccione et al. chose
   (k1, k2, Delta t_w), only that their slice cutoff and hyperparameters minimise the distance
   of the ROC point from (0, 1).

## Open problems and next steps

- A **leave-one-campaign-out** score (train 2014, test 2018 and back) is the test of the era
  worry above and was not run.
- The 100 ms positive window and the 20 ms growth window both rest on conventions; the
  growth window could be set by a magnetics-only detector run on a faster signal (the 1 kHz
  RMS is too coarse to place the start), or by the DIII-D team's own onset definition.
- A true no-wall limit (DCON, or the physics-guided network in the paper) and
  omega_E, nu_ii would test whether the 0.8 is the limit of scalar features.
- Hanson's tables do not say which shots were examined. Confirming that the 33 shots were
  checked end to end would make the absent spans negatives rather than an assumption; a
  second reader on the comparison shots' screen spans would turn some of the unlabelled shots
  into labelled ones, and the false-alarm rates from upper bounds into rates.
- A larger set of examined shots (the corpus has none) is the only fix for the intervals.

## Reproduce

```
python scripts/labeler/rwm_pool.py                       # candidate shots from the logbook
fdp run python scripts/labeler/rwm_fetch.py --shots-csv $LABELER_ROOT/round4/rwm/shots_pool.csv --workers 3 --pace 1
python scripts/labeler/rwm_growth.py                     # growth.json
python scripts/labeler/rwm_build.py                      # selection, slices, label CSV, shots.json
python scripts/labeler/rwm_evaluate.py                   # evaluation.json (about 8 minutes, 6 workers)
python scripts/labeler/rwm_tables.py                     # tables.md
```

Environment `labelmaker`; large outputs (slice table, fetch log, pool, roster) are under
`$LABELER_ROOT/round4/rwm/`. Tests: `tests/labeler/test_rwm_{features,labels,models,shots_data,evaluate}.py`.

## Sources

| Quantity | File |
|---|---|
| Onsets | `data/events/resistive_wall_mode/raw/rwm_onsets_{2017,2024}.csv` |
| Growth numbers | `outputs/labeler/rwm/growth.json` |
| Shot selection, balance, counts, coverage, cohort overlap | `outputs/labeler/rwm/shots.json` |
| Every score, interval, per-shot outcome, paired difference | `outputs/labeler/rwm/evaluation.json` |
| Tables in this note | `outputs/labeler/rwm/tables.md` |
| Labels | `data/events/resistive_wall_mode/extend_rwm_growth/rwm_windows.csv` (+ `.meta.json`, `.shots.csv`) |
| Legacy numbers | `.tmp/label_papers/Piccione_2022_Nucl._Fusion_62_036002.md` |
| Class-prior PU risk | `outside/Kiryo_2017_nnPU.md` |
