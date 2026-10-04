# confine-ours: a dense confinement segmenter over 0D signals

Status: **experimental. Trained and scored 2026-10-03; rescored 2026-10-04 with folds that keep
whole run days together, and the roster relabelled with a confidence floor and tiers.**
`confine-ours` is a U-Time-style 1D U-Net that labels every millisecond of a shot L, H, QH or
WPQH from signals nearly every shot has, so it can run on the whole roster (839 shots) where the
BES classifier (`confine-cnn`, [confinement_bes_benchmark.md](confinement_bes_benchmark.md)) can
run only on the curated shots that have BES. It is the generator of the confinement labels on
the roster outside the curated set, and those labels are **unreviewed**: on the curated shots
only 42 % of its label time falls inside a curated interval, the shots past the curated range
(above 196493) are extrapolation, and its QH and WPQH on shots outside the curated set are weakly
supported (section "Roster labels"). `confine-cnn` is the predecessor it is benchmarked against.
Script:
`scripts/labeler/confinement_ours.py` (stages `build`, `train`, `score`, `apply`); fetch:
`scripts/labeler/confinement_zerod_fetch.py`, `scripts/labeler/confinement_shot_dates_fetch.py`;
library: `src/labeler/confinement/{zerod,unet,segments,scoring}.py`; scores:
`outputs/labeler/confinement/ours/scores.json` (folds by run day, the headline) and
`scores_random_shots.json` (the first version's folds by shot); roster audit:
`outputs/labeler/confinement/ours/roster_audit.json`; labels:
`data/events/confinement/extend_confine_ours/roster.csv` (+ `.meta.json`); models, grids and
probabilities: `$LABELER_ROOT/round4/conf/{zerod,zerod_grid,ours_runday,ours}/`.

## Inputs

One 1 ms grid per shot, seven channels cut from signals the corpus, the raw cache or
`confinement_zerod_fetch.py` hold (`labeler.confinement.zerod`):

| Channel | Source | Transform |
|---|---|---|
| `da_mean`, `da_max`, `da_std` | D-alpha (`filterscopes`, the first of FS01-FS08 that is live) | bin mean, maximum and standard deviation; `log10(x / ref + 0.01)` for the first two, `std / ref` (at most 3) for the third; `ref` is the shot's own 95th percentile of D-alpha, the only per-shot scaling |
| `dens` | `\BCI::DENV2F`, a line-density trace of the CO2 interferometer (not a voltage: 1e13 to 3e14, with a rank correlation of 0.84 with the Thomson peak density on 542 shots; nominally cm^-3, about 2.3 times the Thomson peak, so the scale is not calibrated, `denv2f_check.json`) | 1 ms mean, / 1e14 |
| `betan`, `wmhd` | `\efit01::top.results.aeqdsk` | interpolated across gaps of at most 100 ms; / 1.5 and / 5e5 J |
| `pinj` | the eight beams' power (corpus `pinj`, else raw cache, else fetched) | summed, 1 ms mean, / 3e6 W |

A bin with no value becomes 0 with a 1 on a mask channel (one for each of `dens`, `betan`,
`wmhd`, `pinj`), so the network has 11 input channels and learns what a missing record
means. Every channel except the D-alpha ones has a fixed scale, so absolute levels (a beta_N of 2)
mean the same on every shot. BES is not an input: it exists on too few roster shots.

## Network and training

`UNet1d`: five encoder stages (widths 16, 32, 64, 128, 128; two Conv1d(k = 5) + BatchNorm + ReLU
each) with max-pooling 4, 4, 4, 2, so the bottleneck cell covers 128 ms and its receptive field
about 1.8 s; a decoder with skip connections (nearest-neighbour upsampling, a 3-wide convolution,
concatenation); dropout 0.1; a 1 x 1 convolution to 4 logits per bin. 739,044 parameters.
Loss = cross-entropy + generalized Dice (Perslev et al., 2019, U-Time) + 0.15 x truncated T-MSE
(Farha and Gall, 2019, MS-TCN; tau = 4) on the change of the log-probabilities between
neighbouring bins. Training windows are 4096 ms, 16 per step, drawn so that each holds a bin of a
class chosen uniformly (class-balanced); AdamW, learning rate 1e-3 (one cycle), weight decay
1e-2; up to 5000 steps with validation every 250 and patience 16; the checkpoint with the
best validation macro-F1 is kept. Unlabelled bins carry -1 and count in no loss.
None of this was tuned.

**Data.** The curated confinement intervals (L, H, QH, WPQH; `merged_intervals.csv`, 448 shots)
without the 2 blind test shots of the cohort (190857, 192756), which are never read: 446 shots,
of which the 401 that have a D-alpha record are used (the 45 without one have no input to read
and are in no fold). The labelled bins are 789,441
(L 97,856 on 104 shots; H 384,821 on 156; QH 211,348 on 181; WPQH 95,416 on 71).

**Protocol.** 5-fold cross-validation in which a fold holds out **whole run days**. A shot's
run day is the calendar day, in the MDSplus server's local (Pacific) time, on which its EFIT01
reconstruction was inserted into MDSplus (the stamps carry no time zone and fall between 08:00 and
21:59, so they are not UTC; `confinement_shot_dates_fetch.py` reads it from the tree: no table on disk dates the shots
before 2021; five shots whose date is far from their neighbours' or missing take the day of
the previous shot), which gives 104 groups for the 401 shots. Groups are dealt to the folds,
those holding the rarer classes first, each to the fold that holds the smallest share of the
group's classes (`bes_protocol.deal_groups`): 78, 84, 79, 79 and 81 test shots. The network of a
fold is validated on a sixth of the other days, so every score below is from a network that
never saw the shot **or any shot of its day**. The first version dealt single shots at random
within class-presence strata: of the 218 pairs of scored shots at most 2 shot numbers apart,
169 then fell in different folds, against 2 now. Those scores are kept as
`scores_random_shots.json`, and the table below shows what the leakage added.

## Scores

Out of sample means each shot is scored by the network of the fold that held out its run day
(about 267 shots train it, 55 others validate it, and the rest, about 80, are its test shots).
Intervals are 95 % shot bootstraps (1,000 replicates, seed 20261001). Records:
`outputs/labeler/confinement/ours/scores.json` (run-day folds) and `scores_random_shots.json`
(random-shot folds).

**Per bin, all 401 shots, folds by run day** (789,441 labelled 1 ms bins):

| Regime | Bins (shots) | F1 [95 % CI] | AUROC | AUPRC |
|---|---|---|---|---|
| L-mode | 97,856 (104) | 0.895 [0.842, 0.939] | 0.962 | 0.903 |
| H-mode | 384,821 (156) | 0.953 [0.936, 0.968] | 0.985 | 0.984 |
| QH-mode | 211,348 (181) | 0.892 [0.855, 0.924] | 0.962 | 0.928 |
| WPQH-mode | 95,416 (71) | 0.801 [0.721, 0.865] | 0.948 | 0.837 |
| **Macro** | 789,441 (401) | **0.885 [0.855, 0.914]** | 0.964 | 0.913 |

Leaving out the bins within 20 ms of an interval end (741,626 bins, 391 shots) gives macro F1
0.893 [0.859, 0.921]. Confusion matrix (bins, rows true):

| true \ predicted | L | H | QH | WPQH |
|---|---|---|---|---|
| L | 90,407 | 1,519 | 5,831 | 99 |
| H | 3,155 | 366,650 | 9,431 | 5,585 |
| QH | 6,940 | 5,807 | 187,477 | 11,124 |
| WPQH | 3,739 | 10,330 | 6,441 | 74,906 |

**What the session leakage added.** The same network and data with the first version's random
shot folds scored higher in every class: macro F1 0.934 [0.909, 0.955] (L 0.946, H 0.983, QH
0.933, WPQH 0.874; AUROC 0.976, AUPRC 0.950), 0.049 above the run-day number, and the segment
F1@25 after the mode filter 0.892 against 0.774. Shots of one run day share plasma conditions
and their 0D traces are near copies, so a model that trained on a neighbour had half the
answer. **0.885 is the number to report; 0.934 is not out of sample in the sense a new
campaign would be.** Run days are a lower bound on what a new campaign changes: the 2012-2020
shots and the 2021-2023 shots share a machine configuration only in part, and no fold holds a
campaign out.

**Per shot, segment level, folds by run day** (Farha and Gall's scores; a segment is a run of
one regime over the labelled bins of a shot, in the labels or in the prediction; a predicted
segment is a hit when it overlaps a not yet matched labelled segment of the same regime with
intersection over union of at least 10, 25 or 50 %; the F1 uses hits, false and missed
segments summed over the 401 shots; the edit score is 100 x (1 - Levenshtein distance between
the two segment sequences / the longer length), averaged over shots):

| Output | F1@10 | F1@25 | F1@50 | Edit |
|---|---|---|---|---|
| argmax per bin | 0.723 [0.636, 0.805] | 0.719 [0.632, 0.800] | 0.707 [0.619, 0.789] | 81.0 [77.9, 84.1] |
| + mode filter over 21 bins | 0.779 [0.687, 0.854] | 0.774 [0.681, 0.849] | 0.761 [0.669, 0.838] | 82.5 [79.5, 85.5] |

**Against `confine-cnn` on the same windows** (`comparison_with_confine_cnn` in `scores.json`;
each row of the BES ablation scores its own windows, those its gate, margins and shot rules
keep; the table restricts both networks to the windows and shots both score and bootstraps the
difference over the shots they share, so the interval is that of the paired difference). The
folds of `confine-ours` hold out run days, those of the BES network hold out shots:

| `confine-cnn` row | What it is | Shots | Windows | `confine-cnn` F1 [95 % CI] | `confine-ours` F1 [95 % CI] | Ours minus CNN, paired [95 % CI] |
|---|---|---|---|---|---|---|
| `base` | first-retrain recipe, shot-grouped 5-fold, ungated | 117 | 65,052 | 0.717 [0.63, 0.78] | 0.688 [0.59, 0.77] | -0.029 [-0.13, +0.07] |
| `only_ge` | ungated, 444 shots at 1 MHz, shot-grouped 5-fold | 400 | 385,113 | 0.839 [0.80, 0.87] | 0.885 [0.85, 0.91] | +0.046 [+0.01, +0.09] |
| `cum_abcdrge` | paper selection, early-stopped padded network, shot-grouped 5-fold | 276 | 203,073 | 0.727 [0.66, 0.79] | 0.811 [0.74, 0.87] | +0.084 [+0.02, +0.15] |
| `cum_abcdrgef` | paper selection and split, early-stopped padded network | 127 | 130,591 | 0.702 [0.58, 0.80] | 0.864 [0.76, 0.93] | +0.162 [+0.08, +0.24] |
| `full_cum_abcdrge` | paper selection, architecture and training, shot-grouped 5-fold | 280 | 202,339 | 0.731 [0.65, 0.80] | 0.812 [0.74, 0.88] | +0.081 [+0.02, +0.14] |
| `full_cum_abcdrgef` | **Table 3 row**: paper selection, architecture, training and split (pooled over five overlapping draws) | 124 | 130,711 | 0.702 [0.58, 0.80] | 0.794 [0.71, 0.87] | +0.092 [-0.04, +0.24] |

The two split rows (`cum_abcdrgef`, `full_cum_abcdrgef`) pool the test sets of five random draws that
overlap (142 distinct shots in 200 shot-tests for the Table 3 row), so their windows count a shot once
per draw that tests it and the paired difference weights it the same way
([confinement_bes_benchmark.md](confinement_bes_benchmark.md), "Overlap of the five test draws").

**By population** (per 1 ms bin, all labelled bins, run-day folds): on the 117 shots the corpus
holds BES for, 0.689 [0.59, 0.77] (L 0.84, H 0.94, QH 0.49, WPQH 0.48); on the other 284 curated
shots, 0.926 [0.89, 0.95] (L 0.93, H 0.97, QH 0.93, WPQH 0.88).

## Reading them

- Across run days the segmenter reaches macro F1 0.885 [0.855, 0.914] per bin and a segment F1 of
  0.76 to 0.78 after the mode filter; H-mode is at 0.95, L-mode 0.90, QH 0.89, WPQH 0.80. Its
  main confusion is the one the BES network has: 11,124 of 211,348 QH bins are called WPQH and
  6,441 of 95,416 WPQH bins QH; H and WPQH are also confused (5,585 H bins called WPQH, 10,330
  WPQH bins called H).
- The mode filter lifts the segment F1 by 0.05: the raw output flickers over a few bins at a
  transition. The roster labels are written with the filter (21 bins, minimum segment 20 ms).
- **Against the BES network** `confine-ours` is ahead on the larger sets and level on the
  smallest: paired +0.046 [+0.01, +0.09] on 400 shots without a gate, +0.081 [+0.02, +0.14] on the
  280 shots of the paper-training 5-fold row, -0.029 [-0.13, +0.07] on the 117 corpus shots, and
  +0.092 [-0.04, +0.24] on the 124 test shots of the Table 3 row, whose interval holds zero. The
  margin depends on the windows, and the comparison is not independent: the two score different
  windows (the BES row's gate and margins), hold out different units (run days against shots) and
  both learned from the same labels.
- **The corpus-shot deficit is shared.** `confine-ours` scores 0.689 on the 117 corpus shots
  against 0.926 on the other 284, the same ordering the BES network shows (0.44 to 0.46 under the
  paper's whole selection on the corpus shots against 0.74 to 0.83 on the others, in the four
  all-shot full-selection rows of `confinement_bes_benchmark.md`). Two
  networks that read different signals find the same shots harder, so the cause lies in those
  shots (their labels or their conditions), not in the BES recipe; the data cannot say which.
- **Circularity.** The curated intervals were set by experts reading the same 0D traces (a
  D-alpha level and its ELM pattern separate L from H; steady D-alpha and density with no ELMs
  mark QH), so the 0.885 measures how well the network reproduces the labelling rule from the
  signals, not agreement with an independent measurement. The QH label rests on the edge harmonic
  oscillation seen in the magnetic spectrogram and the WPQH label on the pedestal width from
  Thomson scattering; the network sees neither, and there it is weakest.
- **Training length.** Every fold ran to the 5,000-step cap and the patience of 16 evaluations
  (4,000 steps) never fired, so the networks may be under-trained. The validation macro-F1 does not
  support that: it peaked at steps 1,750, 3,000, 1,000, 1,500 and 2,500 and was lower at step 5,000
  by 0.028, 0.055, 0.080, 0.014 and 0.010 (`ours_runday/fold*_training.json` under
  `$LABELER_ROOT/round4/conf/`), so a longer run is not obviously better; none was run, nor was the
  cap or patience tuned.
- Folds hold whole run days out, so they cover day-to-day variation but not a change of
  campaign: no fold holds out 2012-2017 or 2021-2023. The cohort's blind test shots are not
  read at all.

## Roster labels

`apply` writes `data/events/confinement/extend_confine_ours/roster.csv` and `roster.meta.json`,
in the format of the other extend-model folders (see
[data/events/README.md](../../data/events/README.md)), with these columns (times in ms):

| Column | Meaning |
|---|---|
| `shot`, `t_start`, `t_end` | the segment |
| `category` | 1 high, 2 low, 3 qh, 4 wpqh, **5 uncertain**: a segment whose confidence is under 0.7 |
| `confidence` | mean probability of the segment's class over its bins |
| `predicted` | the segment's class (same numbering), before the floor |
| `source` | `held_out`: a curated shot, read by the fold network that never saw its day; `ensemble`: any other shot, the mean of the five |
| `tier` | `unreviewed`: a QH or WPQH segment on a shot outside the curated set; `model`: the rest |
| `extrapolated` | the shot is past the last curated one (196493): none of these campaigns was seen in training |

The full class probabilities of every labelled roster shot (float16, 1 ms) are in
`$LABELER_ROOT/round4/conf/ours_runday/roster_probs.npz`, the fold networks and their training
records in the same folder. `roster_audit.json` holds every number below.

- A segment is a run of one regime of at least 20 ms after the 21-bin mode filter, over the bins
  where the neutral beams inject at least 200 kW (gaps of at most 50 ms in the beam record are
  bridged, so a modulated beam does not cut the shot into pieces). **Time without beam power is
  not labelled, and not low**: the network was never shown it, and an ohmic L-mode phase before
  the beams fire is outside the labels.
- 805 of the 839 roster shots carry labels: 3,546 segments, 3,611 s. **1,102 of them (31 %) are
  under the 0.7 floor and are category 5.** The others: 792 H, 1,161 L, 295 QH and 196 WPQH.

| Predicted class | Segments | Shots | Seconds | Median segment (s) | Mean confidence | At or above 0.7 |
|---|---|---|---|---|---|---|
| H | 1,139 | 674 | 1,809 | 0.52 | 0.80 | 70 % |
| L | 1,461 | 788 | 941 | 0.42 | 0.85 | 79 % |
| QH | 524 | 316 | 611 | 0.48 | 0.73 | 57 % |
| WPQH | 422 | 243 | 250 | 0.16 | 0.67 | 46 % |

- **Coverage of the curated intervals is 42 %.** On the 401 curated shots 1,850 s are labelled
  and only 42 % of that time (H 54 %, L 22 %, QH 44 %, WPQH 42 %) lies inside a curated
  interval; inside one of the same class it is 39 % (H 51 %, L 19 %, QH 40 %, WPQH 34 %). The
  rest is the network's reading of time nobody curated (beam-on stretches outside the experts'
  intervals), which can be right or wrong. The curated shots' own scores above are per labelled
  bin, so they do not speak to that time.
- **Shots outside the curated set** (404 shots, 1,761 s; 142 of them are past 196493):

  | Predicted class | Segments | Shots | Shots with 1 s or more | Seconds | Mean confidence | At or above 0.7 |
  |---|---|---|---|---|---|---|
  | H | 583 | 360 | 276 | 1,093 | 0.77 | 62 % |
  | L | 704 | 389 | 163 | 498 | 0.79 | 69 % |
  | QH | 142 | 83 | 42 | 137 | 0.54 | 18 % (25 segments) |
  | WPQH | 107 | 59 | 8 | 32 | 0.48 | 4 % (4 segments) |

  Whether the curated training set leaves out ELM-suppressed, ELM-free and negative-triangularity
  H-modes (the exclusions of the paper's H class, Gill et al. 2024) is **unverified**: its H
  intervals are the experts' own, merged from Gill's tables, and nothing here checks what they
  contain ([confinement_bes_benchmark.md](confinement_bes_benchmark.md), "What our H class
  contains, against the paper's, is unverified"; the one check on disk, the I-coil currents of the
  117 corpus shots, finds an applied field of 2.9 to 5.1 kA in labelled H intervals of 5 of them and
  proves nothing about ELM suppression). If such plasmas are not in the set, a plasma without ELMs
  on a shot outside it is likely to be read as QH; if they are, the network has learned them as H.
  Either way the QH and WPQH segments there (249) are on the `unreviewed` tier, and
  only 29 of them clear the floor; treat the other 220 as uncertain. The 142 shots past 196493
  are flagged `extrapolated` (their H and L segments are 60 % and 74 % at or above the floor,
  their QH 17 % and WPQH 3 %, the same as the rest).
- **Cross-check against the D-alpha L-H table** (`suggestions/dalpha_lh/v1`, the rule-based
  H-mode detector that never establishes L-mode), on the 404 shots outside the curated set: of
  the network's H time, 44 % lies where the table says high, 24 % where it is uncertain and 32 %
  where the table marks no H-mode or not observable; of the network's L time, 6 % lies where the
  table says high (14 % uncertain, 80 % unmarked). So the network's L is rarely in the
  detector's H-mode (6 %), but a third of its H lies where the detector found no H-mode or could not
  observe one: either the detector misses H-mode there or the network over-calls it.
- 34 shots have none: 33 have no beam power record in the corpus, the raw cache or the fetch (the
  list and reasons are in `roster.meta.json`), and 200007 has no beam-on stretch of 20 ms.
- A reviewer should look at a sample of the unreviewed and extrapolated segments in the review
  page before these labels enter the catalog; nothing has been reviewed yet.

## Run

```bash
# fetch the density, EFIT and beam-power traces the corpus does not hold, and the shot dates
# (login node, under `fdp run`, at most 3 workers: --part k/3 --pace 1)
python scripts/labeler/confinement_zerod_fetch.py --part 0/3 --pace 1
python scripts/labeler/confinement_shot_dates_fetch.py
# the CUDA environment trains and applies; the library imports in the pixi environment too
PYTHONPATH=src python scripts/labeler/confinement_ours.py build
PYTHONPATH=src python scripts/labeler/confinement_ours.py train        # 5 folds, 1-6 min each
PYTHONPATH=src python scripts/labeler/confinement_ours.py score --cnn-predictions <ablation row dirs>
PYTHONPATH=src python scripts/labeler/confinement_ours.py apply
# the first version's folds by shot (the leakage comparison)
PYTHONPATH=src python scripts/labeler/confinement_ours.py score --grouping shot
```

## Not done

- **No BES input.** It exists on 117 curated shots in the corpus and on the roster outside them
  only by fetch; the segmenter's whole point is to run without it. A BES-fed branch would help
  the QH / WPQH boundary and cost the coverage.
- **The density channel is uncalibrated.** `\BCI::DENV2F` is a density-like trace of the CO2
  interferometer (checked against Thomson on 542 shots: rank correlation 0.84, scale about 2.3
  times the Thomson peak in cm^-3; about 1 shot in 20 has a near-zero trace), not a calibrated
  line-averaged density. Fringe jumps are not unwrapped.
- **Beam-on bins only.** An L-mode before the beams or in a beam-off phase is not labelled; a
  shot with a modulated beam of short duty is bridged, not studied.
- **No independent check of the roster labels outside the curated shots**, and no held-out
  campaign. The 0.885 is for run days of the campaigns the curated set covers; the roster
  includes 142 shots past them.
- Nothing was tuned: one architecture, one loss weighting, one seed per fold; no ablation of
  the channels or the loss terms; no second seed.
- 33 roster shots lack a beam record; fetching their injected power would let the segmenter run
  on them.
