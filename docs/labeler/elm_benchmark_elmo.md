# ELM benchmark: ELM-O

Status: **run 2026-10-01 (repository commit 2b4da1d, scripts not yet committed), re-implemented from the paper.** The owner
(2026-10-01) pointed at the ELM-O detector as the benchmark for the ELM labels, and as
a possible later method for curating ELM datasets ("for now benchmark is good"). The
public code has no licence, so the detector in `scripts/labeler/elmo_benchmark.py`
was written from the paper's description and checked against the code, not copied from
it. It was scored twice: on David Smith's hand-labelled windows, which is the paper's own
kind of truth, and on our review labels.

## Result in brief

- **On Smith's windows the re-implementation reproduces the published scores.** At the
  published setting (t = 1 V, eta = 0.997) it finds 2269 of 2316 marked ELMs
  with 6 extra spans: precision 0.997, recall 0.980 (paper 0.995 and
  0.976), over 211 shots. 32 of its 47 misses are in two shots, as in the paper
  (18 of its 23).
- **On our review labels ELM-O as published scores F1 0.842** (precision 0.840, recall 0.844
  over 50 ms bins). That is for the 73 of the 119 review shots that have BES in the corpus
  (387 s analysed); the other 46 hold only a one-sample BES stub, and ELM-O needs BES. It
  finds dense ELMing best (85 % of the bins inside crowd spans) and single marked ELMs
  worse (59 of 93 individual spans; 82 of 93 without its BES vote).
- **Some of its false alarms are label gaps.** 96 of 218 spans marked absent hold an
  ELM-O ELM, 927 ELMs against 6,631 in present spans (precision 0.877). 416 of those 927
  peak within 500 ms of a present span, where the edges are fuzzy; the others sit in long
  absent stretches, and in the six absent spans with the most ELMs the D-alpha trace
  shows bursts under the ELM-O ticks (`outputs/labeler/elm/elmo/absent_alarms.png`). The
  precision against the labels is a lower bound on ELM-O's, and the absent spans with ELMs
  (`absent_with_elms.csv`) are labels worth reading again. How many of the 927 are real ELMs was not
  counted.
- **It would have drafted more of the final labels than `elm_clock` did:** bin recall 0.844
  against 0.621, at about the same precision (0.840 against 0.822). The reviewers
  started from the clock, so that row is not an independent score. Where it can run it needs BES,
  the interferometers and the filterscopes, not D-alpha alone.
- **The BES vote carries the precision.** Without it ELM-O keeps its recall (0.888) and
  drops to precision 0.485; with the code's eta of 0.995 instead of the paper's 0.997 it scores
  F1 0.856 (precision 0.818, recall 0.898), inside the intervals of the published setting.

Reproduce: `python scripts/labeler/elmo_benchmark.py smith`, then `review`, then
`evaluate` (environment `labelmaker`; the inputs come from
`scripts/labeler/elmo_fetch.py`). The record is `outputs/labeler/elm/elmo/evaluation.json`; the
per-window and per-ELM tables are under `$LABELER_ROOT/benchmarks/elm/elmo/`.

## Source

- Paper: F. H. O'Shea, S. Joung, D. R. Smith and R. Coffee, "Automatic
  identification of edge localized modes in the DIII-D tokamak", APL Mach. Learn. 1,
  026102 (2023), doi:10.1063/5.0134001 (CC BY). Local copy: `.tmp/elm_auto.pdf`
  (git-ignored).
- Code: https://github.com/finnoshea/PublicELMO, commit `4423df5` (2023-02-09). Public,
  but the repository has **no licence file**, so do not copy it into this repository.
  Run it from a clone outside the repo, or re-implement it from the description below.
  `elm_finder_pkg.py` is its complete working finder (its README says the rest is a
  "barf-deck" of things tried).

## The algorithm

Rule-based, no machine learning. Inputs over a window in which all instruments
report: two interferometer chords (`denv2f`, `denv3f`, 100 kS/s), three filterscopes
(FS02, FS03, FS04, 50 kS/s) and the 64 BES channels (1 MS/s).

1. Bring every signal to 1 MS/s by spectrum-preserving up-sampling: DCT, zero-pad the
   transform, inverse DCT, scaled by sqrt(new/old length), done in chunks of 200 ms
   (`chunked_lengthen`).
2. BES: the last 32 channels are multiplied by 2 (half the range of the first 32),
   then all 64 are averaged to one trace. Candidate where that trace exceeds
   `t = 1` V.
3. Interferometers and filterscopes: absolute first difference, then candidate where
   the value is above the `eta` quantile of that signal in the window; the paper's
   `eta = 0.997`, the code's default is 0.995.
4. Each interferometer and filterscope candidate mask is widened by 100 samples
   (100 us) either side (`blur_labels(n=100)`, a 201-sample kernel). The paper says
   all six signals are convolved with a 100 us window; the code leaves the BES mask
   unwidened.
5. Per diagnostic type: interferometers, either chord; filterscopes, at least 2 of
   the 3; BES, directly.
6. ELM where all three types have a candidate; gaps of at most 100 samples are
   bridged; the ELM's peak time is the argmax of FS03 inside each span.

`eta` is tuned on the hand-labelled set once and then fixed; the paper stresses it
is not a per-shot parameter. It also means ELMs too small to exceed the percentile
threshold are not labelled when large ELMs share the window (the paper's Fig. 6).

## Published results

Tuning set: 972 hand-labelled ELMs (the paper does not name the labeller), trimmed
regions each holding one ELM with a smaller region marked as the ELM. At `t = 1`,
`eta = 0.997`: precision
0.995, recall 0.976, area under the precision-recall curve 0.971 (5 false
positives, 23 false negatives; 18 of the 23 from two shots where the digitisers'
time registration failed).

| t | eta | Precision | Recall | AUC |
|---|---|---|---|---|
| 0.5 | 0.998 | 0.996 | 0.974 | 0.970 |
| 1.0 | 0.997 | 0.995 | 0.976 | 0.971 |
| 2.0 | 0.997 | 0.997 | 0.975 | 0.972 |
| 5.0 | 0.200 | 0.998 | 0.961 | 0.959 |

Unseen data, one random 200 ms window per shot, expert-checked, precision only
(recall cannot be checked without exhaustive labels):

| Group | Shots | ELMs found | Precision |
|---|---|---|---|
| 1: the shots the tuning ELMs came from (166433-173224), other time windows | 49 | 299 | 0.987 (4 false positives, from two shots, as pairs marking one event 200-300 us apart) |
| 2: shots disjoint from the tuning shots' range | 76 | 392 | 0.977 (383 of 392) |

Group 2 was 200 shots drawn at random from 9,941 plasma-mode discharges longer than
1 s with all the diagnostics present (156562-187328); the draws inside the tuning
shots' range were dropped, leaving 76 (58 before it, 18 after).

The paper compares with a KSTAR U-Net (single filterscope signal, reported as
positive prediction rate and true positive rate): on its training data precision
0.924 and recall 0.935; on two KSTAR test shots (18396 and 29487) precision 0.84 and
0.87, recall 0.96 and 0.88, interpolated from that paper's figures.
Scoring rule used throughout the paper: a detected span is a true positive if it
overlaps a hand-labelled ELM, a false positive if it overlaps none; a hand-labelled
ELM with no overlapping span is a false negative.

## Truth data on disk

David Smith's hand-labelled ELMs (the ELM-O code calls them "Smith labels"; our ELM
README listed them as not obtained, but they are readable under
`/projects/EKOLEMEN/dsmith/data/`):

| File | Events | Shots | Notes |
|---|---|---|---|
| `labeled-elm-events.hdf5` (1.5 GB, 2022-03-20) | 481 | 164 | each group: `time` (ms, 1 MHz), `signals` (64 BES x samples), `labels` (int8, 1 inside the ELM), attribute `shot` |
| `labeled_elm_events_long_windows_20220921.hdf5` (13.7 GB, 2023-03-13) | 2,008 | 199 | same layout, longer windows (5k-29k samples) |

Only BES is inside them. The interferometer and filterscope traces ELM-O also needs
were fetched from the shots for those windows (`scripts/labeler/elmo_fetch.py`, see
below). Which of these events make up the paper's 972 is not recorded here. The code reads two files of this kind, `labeled-elm-events.hdf5`
and a `labeled_elm_events_long_windows_20220527.hdf5`, and the second is not the
file above.

Our own ELM labels are the other truth: `data/events/edge_localized_mode/` holds
Hiro's `wpqh_elm_hiro` onset pickles (breakthrough ELMs of the WPQH experiments, in
the format table as 50 ms bins) and the review labels, where a span is one ELM
(`iscrowd` 0) or a crowd, an ELMing period whose single ELMs are not separated
(`iscrowd` 1).

## What was run

- **Detector** (`detect` in `scripts/labeler/elmo_benchmark.py`): steps 1-6 above. Sample
  counts follow the BES rate: the widening and the gap are 100 us whatever the
  rate, so the corpus BES (500 kS/s) uses half the samples that Smith's windows (1 MS/s) do.
  Up-sampling is in pieces of at most 200 ms.
- **Check against the reference code.** A scratch script (not in the repository, since it
  imports the unlicensed clone) compared the re-implementation with
  `chunked_lengthen` and `Elmo.find_candidates` / `label_candidates` of PublicELMO `4423df5`:
  the up-sampled arrays agree to the last bit on eight lengths, and the detected spans agree
  sample for sample on 60 windows drawn at random from Smith's files, each at eta 0.997, 0.995
  and 0.9 (180 comparisons, no mismatch).
  `tests/labeler/test_elmo_benchmark.py` covers the pure functions.
- **Inputs.** `scripts/labeler/elmo_fetch.py` fetched `\BCI::DENV2F` and `DENV3F`
  (100 kS/s) and `\SPECTROSCOPY::FS02` to `FS04` (50 kS/s) for the 211 Smith shots in the
  union of the two files and for all 119 review shots (BES is Smith's own for his windows and
  the corpus `bes` group for ours; 46 review shots turned out to have no BES, see below).
- **Smith's windows.** 2,489 windows in the two files, 2,316 after dropping 173 of the
  long file's (164 mark an ELM already marked in the short file, 9 repeat an earlier window of
  the long file). The paper's rule scores a window: true positive when a detected span overlaps the marked
  region, false negative when none does, and one false positive for each span that
  overlaps nothing. Confidence intervals resample shots, 1,000 times.
- **Review labels.** The 73 shots of `data/events/edge_localized_mode/review/labels.csv`
  (119 shots) that have BES in the corpus: on the other 46 the `bes` group is a
  one-sample stub, so ELM-O cannot run and they are not scored. ELM-O runs in consecutive
  200 ms chunks over each scored shot's reviewed time (387 s; 2 chunks skipped for missing signals). A 50 ms bin that lies wholly inside one
  labelled span, and inside analysed time, is positive when an ELM-O span overlaps it. Spans of
  category 1 are present, of category 0 absent; uncertain and not-observable time is left out.
  **Detection precision** is a different count: the share of ELM-O ELMs whose FS03 peak falls
  in a present span rather than an absent one. Confidence intervals resample shots.

## Results on Smith's windows

| | Windows (shots) | TP | FP | FN | Precision [95 % CI] | Recall [95 % CI] | Precision x recall [95 % CI] |
|---|---|---|---|---|---|---|---|
| This reimplementation, t = 1, eta = 0.997 | 2,316 (211) | 2269 | 6 | 47 | 0.997 [0.995, 0.999] | 0.980 [0.956, 0.995] | 0.977 [0.954, 0.993] |
| The paper (972 labelled ELMs) | 972 | 949 | 5 | 23 | 0.995 | 0.976 | 0.971 |

The same windows, re-scanned over eta for each BES threshold, best product of precision and
recall (the paper's Table II "AUC" is that product):

| t | eta here | Precision | Recall | P x R | Paper: eta | Precision | Recall | AUC |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.997 | 0.996 | 0.981 | 0.976 | 0.998 | 0.996 | 0.974 | 0.970 |
| 1 | 0.995 | 0.995 | 0.982 | 0.978 | 0.997 | 0.995 | 0.976 | 0.971 |
| 2 | 0.994 | 0.997 | 0.981 | 0.979 | 0.997 | 0.997 | 0.975 | 0.972 |
| 5 | 0.200 | 0.999 | 0.966 | 0.965 | 0.2 | 0.998 | 0.961 | 0.959 |

By file:

| File | Windows | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|---|
| `labeled-elm-events.hdf5` | 481 | 471 | 1 | 10 | 0.998 | 0.979 |
| `labeled_elm_events_long_windows_20220921.hdf5` | 1835 | 1798 | 5 | 37 | 0.997 | 0.980 |

**Misses.** 47 windows have no overlapping span, in 16 shots: 17 had no interferometer
vote at the marked ELM, 22 no filterscope vote and 0 no BES vote (the BES trace peaks at a median
8.8 V inside them). 32 of them are in two shots, 179859 (16 of its 16 windows missed; the interferometers voted at the marked ELM in 16 of those and the filterscopes in 1) and 166597 (16 of its 20 windows missed; the interferometers voted at the marked ELM in 0 of those and the filterscopes in 16). The paper's
misses are concentrated the same way (18 of its 23 in two shots, which it puts down to the digitisers'
time registration); whether that is the cause here was not checked. The FS03 peak and the BES
peak are a median 101 us apart, 133 us in the missed windows; 5 windows are more than a
millisecond apart.

## Results on our review labels

| Detector | Bin precision | Bin recall | Bin F1 | False-alarm bins, all / near / far | Crowd-bin recall | Individual spans hit | Absent spans alarmed | Detection precision |
|---|---|---|---|---|---|---|---|---|
| ELM-O as published (eta 0.997, BES) | 0.840 [0.764, 0.901] | 0.844 | 0.842 [0.786, 0.888] | 11 % / 15 % / 9 % | 0.852 | 63 % (59 of 93) | 44 % (96 of 218) | 0.877 [0.820, 0.926] |
| ELM-O with the code's eta 0.995 | 0.818 [0.736, 0.884] | 0.898 | 0.856 [0.796, 0.901] | 13 % / 19 % / 11 % | 0.907 | 67 % (62 of 93) | 47 % (103 of 218) | 0.863 [0.802, 0.916] |
| ELM-O without the BES vote | 0.485 [0.393, 0.571] | 0.888 | 0.627 [0.542, 0.696] | 63 % / 57 % / 66 % | 0.890 | 88 % (82 of 93) | 76 % (165 of 218) | 0.530 [0.436, 0.612] |
| `elm_clock` (the suggestion the reviewers started from) | 0.822 [0.714, 0.918] | 0.621 | 0.708 [0.587, 0.800] | 9 % / 11 % / 8 % | 0.628 | 22 % (20 of 93) | 11 % (24 of 218) | n/a |

Bin precision, recall and F1 count 50 ms bins (2,750 present, 4,093 absent
for ELM-O as published). Near and far split the absent bins by their distance to the
nearest present span (500 ms). The last row is the suggestion the reviewers began
from, so its agreement with the labels is not an independent score.

**Absent spans with ELMs.** ELM-O finds ELMs in 95 of the 218 absent spans, 46 of them with five
or more (917 ELMs in all, 36 shots; the detection count of 927 below also takes in the absent spans of these shots with under half their length analysed). The worst are listed in
`$LABELER_ROOT/benchmarks/elm/elmo/absent_with_elms.csv`:

| Shot | Absent span (ms) | Analysed (ms) | ELM-O ELMs | Per second |
|---|---|---|---|---|
| 193022 | 9 - 4479 | 4079 | 103 | 25.3 |
| 199789 | 2958 - 5837 | 2879 | 81 | 28.1 |
| 198899 | 24 - 1458 | 1158 | 65 | 56.1 |
| 185961 | 8 - 1404 | 1204 | 45 | 37.4 |
| 200385 | 6 - 4895 | 4195 | 40 | 9.5 |
| 192914 | 10 - 2567 | 2167 | 40 | 18.5 |
| 191677 | 2458 - 4425 | 1967 | 39 | 19.8 |
| 193022 | 5001 - 6109 | 1108 | 36 | 32.5 |

124 of the 927 ELM-O ELMs in absent spans peak within 50 ms of a present span and 416 within
500 ms. `outputs/labeler/elm/elmo/absent_alarms.png` (a scratch plot, not made by the script) shows FS03 in the
six worst absent spans, one per shot: each has bursts at most of the ELM-O ticks, in time
the review left absent.

**ELM rates by kind of reviewed span** (ELM-O as published; per second of analysed time):

| Reviewed span | Spans | Median length (ms) | ELM-O ELMs in them | ELMs per second, quartiles |
|---|---|---|---|---|
| absent | 218 | 624 | 917 | 0.0 / 0.0 / 3.0 |
| individual | 93 | 53 | 197 | 0.0 / 17.2 / 23.8 |
| crowd | 52 | 2296 | 6,434 | 28.5 / 40.5 / 53.9 |

Crowd spans hold the dense ELMing. An individual span is about one ELM wide, so its
per-second figure only restates its width and is not a rate to compare.

## Caveats

- Only 73 of the 119 review shots have BES in the corpus, so every score on the review labels is
  for those: 218 absent, 52 crowd and 93 individual spans. The individual-span
  figures rest on few shots (95 % interval of the hit rate 0.27-0.86, resampling shots); read them as indicative. The labels
  were made starting from `elm_clock`, so its row is not an independent score.
- The corpus BES is 500 kS/s and ELM-O was developed on 1 MS/s; the parameters were scaled
  in microseconds, which is a choice the paper does not make for us. A spot check of three
  corpus shots and six of Smith's windows found the same voltage scale for the 1 V threshold
  (baseline near -0.1 V, ELM peaks of 5 to 9 V in the BES mean).
- An absent span says nothing was marked there. Where the reviewers started from the
  `elm_clock` draft it may only mean the draft found no ELM and nobody added one, and span
  edges are fuzzy: the absent-span alarms are the first to read before calling ELM-O wrong,
  and before changing any label.
- Smith's marked regions are single ELMs in windows chosen to hold one; they do not test
  whether ELM-O copes with crowds, which is what the review labels add.
- No retuning was done (`eta`, `t` are the published values); the Table II rescan is on the
  same windows it scores, as in the paper.
- None of the PublicELMO code was copied. Smith's labels are not ingested as a source of
  our ELM table.

## Combining datasets

The crowd/individual flag is what lets one ELM dataset hold both Jalal's breakthrough
ELMs (isolated individual events in WPQH) and the regular ELMing periods of this
paper's kind; the benchmark should score the two kinds separately. A further dataset
from David Eldon is expected later and is not part of this benchmark.

On this evidence the two kinds behave differently, which is a reason to keep the flag.
ELM-O's bin recall inside crowd spans (a median 2.3 s of ELMing, about
40 ELM-O ELMs per second) is 0.852; it hits 63 % of the single marked ELMs
(88 % without its BES vote); and Smith's windows, one ELM each, score a recall of
0.980. A dataset that mixes the kinds is scored fairly only per kind.
