# confine-ours: a dense confinement segmenter over 0D signals

Status: **run 2026-10-03: trained, scored on shots it never saw, applied to the roster.**
`confine-ours` is a U-Time-style 1D U-Net that labels every millisecond of a shot L, H, QH or
WPQH from signals nearly every shot has, so it can run on the whole roster (839 shots) where the
BES classifier (`confine-cnn`, [confinement_bes_benchmark.md](confinement_bes_benchmark.md)) can
run only on the curated shots that have BES. It is the generator of the confinement labels on
the roster outside the curated set; `confine-cnn` is the predecessor it is benchmarked against.
Script:
`scripts/labeler/confinement_ours.py` (stages `build`, `train`, `score`, `apply`); fetch:
`scripts/labeler/confinement_zerod_fetch.py`; library: `src/labeler/confinement/{zerod,unet,segments,scoring}.py`;
scores: `outputs/labeler/confinement/ours/scores.json`; labels:
`data/events/confinement/extend_confine_ours/roster.csv` (+ `.meta.json`); models, grids and
probabilities: `$LABELER_ROOT/round4/conf/{zerod,zerod_grid,ours}/`.

## Inputs

One 1 ms grid per shot, seven channels cut from signals the corpus, the raw cache or
`confinement_zerod_fetch.py` hold (`labeler.confinement.zerod`):

| Channel | Source | Transform |
|---|---|---|
| `da_mean`, `da_max`, `da_std` | D-alpha (`filterscopes`, the first of FS01-FS08 that is live) | bin mean, maximum and standard deviation; `log10(x / ref + 0.01)` for the first two, `std / ref` (at most 3) for the third; `ref` is the shot's own 95th percentile of D-alpha, the only per-shot scaling |
| `dens` | `\BCI::DENV2F`, the CO2 V2 interferometer's fast voltage | 1 ms mean, / 1e14; proportional to the line-integrated density, **fringe jumps are not unwrapped** |
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
(L 97,856 on 104 shots; H 384,821 on 156; QH 211,348 on 181; WPQH 95,416 on 71). **Protocol.** Shot-grouped 5-fold cross-validation, dealt within
class-presence strata (the same `deal` as the BES rows); the network of a fold is validated on a sixth of
the other shots, so every score below is from a network that never saw the shot.

## Scores

Out of sample means each shot is scored by the network of the fold that never saw it (about 267
shots train it, 55 others validate it, and the rest, about 80, are its test shots). Intervals are 95 % shot
bootstraps (1,000 replicates, seed 20261001). Record:
`outputs/labeler/confinement/ours/scores.json`.

**Per bin, all 401 shots** (789,441 labelled 1 ms bins):

| Regime | Bins (shots) | F1 [95 % CI] | AUROC | AUPRC |
|---|---|---|---|---|
| L-mode | 97,856 (104) | 0.946 [0.895, 0.980] | 0.961 | 0.944 |
| H-mode | 384,821 (156) | 0.983 [0.971, 0.990] | 0.993 | 0.992 |
| QH-mode | 211,348 (181) | 0.933 [0.902, 0.959] | 0.984 | 0.951 |
| WPQH-mode | 95,416 (71) | 0.874 [0.820, 0.920] | 0.966 | 0.913 |
| **Macro** | 789,441 (401) | **0.934 [0.909, 0.955]** | 0.976 | 0.950 |

Leaving out the bins within 20 ms of an interval end (741,626 bins, 391 shots) changes
nothing: macro F1 0.935 [0.907, 0.959]. Confusion matrix (bins, rows true):

| true \ predicted | L | H | QH | WPQH |
|---|---|---|---|---|
| L | 90,468 | 1,012 | 6,362 | 14 |
| H | 1,040 | 377,056 | 3,615 | 3,110 |
| QH | 967 | 1,280 | 200,686 | 8,415 |
| WPQH | 1,008 | 3,366 | 8,068 | 82,974 |

**Per shot, segment level** (Farha and Gall's scores; a segment is a run of one regime over the
labelled bins of a shot, in the labels or in the prediction; a predicted segment is a hit when
it overlaps a not yet matched labelled segment of the same regime with intersection over union
of at least 10, 25 or 50 %; the F1 uses hits, false and missed segments summed over the 401
shots; the edit score is 100 x (1 - Levenshtein distance between the two segment sequences /
the longer length), averaged over shots):

| Output | F1@10 | F1@25 | F1@50 | Edit |
|---|---|---|---|---|
| argmax per bin | 0.860 [0.792, 0.915] | 0.856 [0.789, 0.912] | 0.848 [0.777, 0.906] | 86.8 [84.1, 89.6] |
| + mode filter over 21 bins | 0.896 [0.836, 0.941] | 0.892 [0.832, 0.936] | 0.883 [0.821, 0.930] | 88.1 [85.5, 90.6] |

**Beside `confine-cnn`**, on the BES windows both scored: each BES window carries its
interval label and centre time, `confine-ours` is read at the bin that holds that time, and
both are held out by shot. The population is the 117 curated shots the corpus holds BES for
(the 119 of the BES benchmark less the two blind shots). The rows of `confine-cnn` are those
of [the ablation](confinement_bes_benchmark.md#the-gap-to-the-published-score-an-ablation):

| `confine-cnn` row | Windows scored (shots) | `confine-cnn` macro F1 | `confine-ours` macro F1 | Paired difference ours minus cnn |
|---|---|---|---|---|
| `base`, first retrain: every window of the shot | 65,052 (117) | 0.717 [0.63, 0.78] | 0.860 [0.79, 0.91] | +0.143 [+0.056, +0.231] |
| `cum_abcdr`, the paper's recipe, on the windows it keeps (beam gate, 20 ms inside intervals, 100 ms after L-mode, channel check) | 37,045 (80) | 0.433 [0.31, 0.57] | 0.794 [0.67, 0.88] | +0.360 [+0.244, +0.467] |

The paired intervals draw the same shots for both models. Per-class F1 on the second row,
`confine-cnn` against `confine-ours`: L 0.23 against 0.95 (16 shots), H 0.93 against 0.98,
QH 0.38 against 0.80 (11 shots), WPQH 0.19 against 0.45 (6 shots). Further rows
(`cum_abcdrg`, 1 MHz) are in the record.

## Reading them

- Across shots the segmenter reaches macro F1 0.93 per bin and a segment F1 of 0.85 or
  better at every overlap threshold; H-mode is at 0.98, L-mode 0.95, QH 0.93, WPQH 0.87.
  Its main confusion is the one the BES network has: 8,415 of 211,348 QH bins are called
  WPQH and 8,068 of 95,416 WPQH bins QH. L-mode is taken for QH in 6,362 bins (6.5 %).
- The mode filter lifts the segment F1 by 0.03 to 0.04: the raw output flickers over a few bins
  at a transition. The roster labels are written with the filter (21 bins, minimum segment
  20 ms), so the 0.88-0.90 figures are the ones that describe them.
- **It beats `confine-cnn` on shared shots, by 0.14 against the first retrain and by 0.36
  against the paper's recipe on the windows that recipe keeps**, and the paired intervals
  exclude zero. That is a statement about these two models on these labels, not about BES
  against 0D signals: the segmenter reads a 1.8 s context of the signals the labelling experts
  looked at (D-alpha, density, beta_N, stored energy, beam power), while the BES network sees
  one 2 ms window of fluctuations and nothing else, and has 117 shots to learn from, 6 to 16 of
  them for the rarer regimes under the paper's gate.
- **Circularity.** The curated intervals were set by experts reading the same 0D traces (a
  D-alpha level and its ELM pattern separate L from H; steady D-alpha and density with no ELMs
  mark QH), so the 0.93 measures how well the network reproduces the labelling rule from the
  signals, not agreement with an independent measurement. The QH and WPQH boundaries rest on
  the BES spectra (the edge harmonic oscillation, edge broadband) the network never sees, and
  there it is weakest.
- Folds are by shot and dealt at random within class-presence strata, so shots of one session
  (neighbouring numbers, one machine configuration) fall in different folds: the intervals
  cover shot-to-shot variation but not a change of campaign. The cohort's blind test shots
  are not read at all.

## Roster labels

`apply` writes `data/events/confinement/extend_confine_ours/roster.csv` (columns `shot`,
`category`, `t_start`, `t_end`, `confidence`; 1 high, 2 low, 3 qh, 4 wpqh; times in ms) and
`roster.meta.json`, in the format of the other extend-model folders (see
[data/events/README.md](../../data/events/README.md)). The full class probabilities of every
labelled roster shot (float16, 1 ms) are in `$LABELER_ROOT/round4/conf/ours/roster_probs.npz`,
the fold networks and their training records in the same folder.

- A curated shot is predicted by the fold network that never saw it (401 shots); any other roster
  shot by the mean of the five. Only the curated shots can be checked against labels.
- A segment is a run of one regime of at least 20 ms after the 21-bin mode filter, over the bins
  where the neutral beams inject at least 200 kW (gaps of at most 50 ms in the beam record are
  bridged, so a modulated beam does not cut the shot into pieces). **Time without beam power is
  not labelled, and not low**: the network was never shown it, and an ohmic L-mode phase before
  the beams fire is outside the labels. `confidence` is the mean probability of the segment's
  class over its bins.
- 805 of the 839 roster shots carry labels: 3,368 segments, 3,611 s.

| Regime | Segments | Shots | Seconds | Median segment (s) | Mean confidence |
|---|---|---|---|---|---|
| H | 1,024 | 651 | 1,793 | 0.78 | 0.83 |
| L | 1,392 | 790 | 872 | 0.40 | 0.87 |
| QH | 607 | 341 | 697 | 0.43 | 0.77 |
| WPQH | 345 | 222 | 248 | 0.17 | 0.74 |

  A quarter of the segments (25 %) have a confidence below 0.7; the QH and WPQH segments are the
  least sure, as in the scores.
- 34 shots have none: 33 have no beam power record in the corpus, the raw cache or the fetch (the
  list and reasons are in `roster.meta.json`), and 200007 has no beam-on stretch of 20 ms.

## Run

```bash
# fetch the density, EFIT and beam-power traces the corpus does not hold (login node,
# under `fdp run`, at most 3 workers: --part k/3 --pace 1)
python scripts/labeler/confinement_zerod_fetch.py --part 0/3 --pace 1
# the CUDA environment trains and applies; the library imports in the pixi environment too
PYTHONPATH=src python scripts/labeler/confinement_ours.py build
PYTHONPATH=src python scripts/labeler/confinement_ours.py train        # 5 folds, 6 min each (V100S)
PYTHONPATH=src python scripts/labeler/confinement_ours.py score --cnn-predictions <ablation row dirs>
PYTHONPATH=src python scripts/labeler/confinement_ours.py apply
```

## Not done

- **No BES input.** It exists on 117 curated shots in the corpus and on the roster outside them
  only by fetch; the segmenter's whole point is to run without it. A BES-fed branch would help
  the QH / WPQH boundary and cost the coverage.
- **The density channel is a proxy.** `\BCI::DENV2F` is the CO2 V2 interferometer's fast
  voltage: proportional to the line-integrated density, fringe jumps not unwrapped, scale nominal.
  A calibrated line density would be a cleaner input.
- **Beam-on bins only.** An L-mode before the beams or in a beam-off phase is not labelled; a
  shot with a modulated beam of short duty is bridged, not studied.
- **No independent check of the roster labels outside the curated shots.** Their only evidence is
  the fold ensemble's confidence; reviewing a sample in the review page is the next step.
- Nothing was tuned: one architecture, one loss weighting, one seed per fold; no ablation of
  the channels or the loss terms; no second seed.
- 33 roster shots lack a beam record; fetching their injected power would let the segmenter run
  on them.
