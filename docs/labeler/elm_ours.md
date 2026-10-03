# ELM detection: `elm-ours`, the lab's ELM model (`elm-dsm`) and the reference swap

Status: **run 2026-10-03 (round four, stream `elm`).** The ELM labels now have what the AE
labels have: a best model (`elm-ours`), a prior model of the lab (`elm-dsm`) and the
published detector ELM-O (`elm-elmo`, [elm_benchmark_elmo.md](elm_benchmark_elmo.md)), all
scored against the reviewed spans with 95 % shot-bootstrap intervals, and a reference swap
against the lab's older ELM annotation. The AE counterpart is
[ae_baselines_benchmark.md](ae_baselines_benchmark.md).

## Result in brief

- **`elm-ours` matches ELM-O without BES and runs on every reviewed shot.** On the 73 shots
  with BES, 50 ms bins: AUROC 0.941 [0.898, 0.972], AUPRC 0.875 [0.751, 0.961], F1 0.845
  [0.786, 0.896], against ELM-O's 0.918, 0.833 and 0.842. The paired differences include
  zero (F1 +0.002 [-0.052, 0.058]; AUROC +0.023 [-0.014, 0.067]), so the claim is parity, not
  superiority. On all 119 shots (12,409 bins, where ELM-O cannot run): AUROC 0.941 [0.906,
  0.967], AUPRC 0.877 [0.774, 0.949], F1 0.830 [0.778, 0.871].
- **It beats the clock and the lab's survival model by a wide margin.** The ELM clock scores
  F1 0.708 on the 73 shots (`elm-ours` minus clock: +0.137 [0.048, 0.233]) and 0.757 on the
  119 (+0.073 [-0.003, 0.151]); the reviewers started from the clock, so that row is not an
  independent score. The survival model as published scores AUROC 0.780 [0.734, 0.829],
  F1 0.638 on the 119; with the objective changed to detection AUROC 0.855 [0.812, 0.896],
  F1 0.735. `elm-ours` is 0.083 [0.043, 0.128] AUROC and 0.099 [0.048, 0.148] F1 above the
  retrained one, with the same shots, folds and bins.
- **The onset head does not work.** The finer trace of ELM onsets finds 7.5 % of the
  reviewed individual-ELM starts within 5 ms (F1 0.045 on 119 shots) where the clock finds
  18 % (F1 0.277). Only 120 individual spans are labelled and the crowds' ELMs are not, so
  the head fires on every unlabelled ELM and counts as wrong. Use the event head; do not
  quote the onset numbers as a result.
- **The reference swap does not reverse the ELM ranking, and its overlap is too small to
  claim much.** The legacy ELM table (Hiro's 50 ms onset bins) covers 8 of the 119 reviewed
  shots (641 bins). Read as a detector of the review it marks 40 review-present bins
  absent (|M| = 40, 27 % of 147) and 4 review-absent bins present (|P| = 4): precision 0.964,
  recall 0.728; it misses 61 % of the single-ELM bins and 18 % of the crowd bins. Scored
  against either reference `elm-ours` is first on AUROC and F1 and every difference to it
  keeps its sign; only weak methods swap places on F1 (the clock and the survival model),
  and every interval on 8 shots is wide.

Source records: `outputs/labeler/elm/ours/evaluation.json` (elm-ours, ELM-O, clock),
`outputs/labeler/elm/dsm/evaluation.json` (elm-dsm on common bins),
`outputs/labeler/elm/swap/evaluation.json` and the `table_elm_*.tex` beside it (reference
swap). Large artefacts (inputs, folds, out-of-fold predictions, DSM rows and scores) are
under `$LABELER_ROOT/round4/elm/`.

## `elm-ours`

**Task.** Per millisecond, is the plasma ELMing (ELMy time: a single ELM or a crowd, an
ELMing period)? Per 50 ms bin the score is the mean event probability over the bin; the
finer onset trace is a second head.

**Inputs** (`labeler/elm/inputs.py`). The fetched filterscope D-alpha channels FS02-FS04
(50 kS/s) and the line density from the two fast interferometer chords DENV2F and DENV3F
(100 kS/s), on a 10 kHz grid; **no BES**, so every reviewed shot with the two records is
covered (all 119, against ELM-O's 73). Eleven channels: log D-alpha level and its contrast
to a 0.5 s running median per filterscope, density and its 0.2 s high-pass per chord, and a
validity mask. The records are those `scripts/labeler/elmo_fetch.py` stored; nothing was
fetched for this stream.

**Model** (`labeler/elm/net.py`). A 1D U-Net in the manner of PhaseNet and U-Time (Zhu and
Beroza 2019; Perslev et al. 2019): a strided stem to 1 kHz, four strided-convolution
encoder levels (stride 4, kernel 7, widths 24/32/48/64/96) and a decoder with skips back to
1 kHz, 401,714 parameters. Two logits per millisecond: ELMy time and ELM onset.

**Labels** (`labeler/elm/labels.py`). The reviewed spans (`review/labels.csv`): present
(single ELM or crowd) is 1, absent is 0, uncertain, not observable and unlabelled time are
ignored. Onsets are the starts of individual spans as a Gaussian (sigma 2 ms), defined only
in absent and individual spans (a crowd's ELMs are unmarked, not absent).

**Training** (`labeler/elm/train.py`, run `cv2`). The 119 reviewed shots (all cohort train
or validation; no cohort test shot is read, `check_no_test` refuses one) are dealt into five
shot-grouped folds, stratified by the kinds of span a shot holds. Per fold: 81 training
shots, 14 inner-validation shots (early stopping and the threshold), 24 held-out shots
(fold 4: 82 and 14 and 23). Masked binary cross-entropy on both heads, random 4,096 ms
crops, D-alpha level shifts, channel dropout and noise as augmentation; 25 epochs of 40
iterations of 16 crops, AdamW (one-cycle schedule, peak lr 2e-3, weight decay 0.01), seed 20261003, on one V100S
(about 8 s per epoch). The best epoch is the one with the highest inner-validation bin AUPRC.
Every shot gets one out-of-fold prediction, from a model that never saw it.

**Scoring** (`labeler/elm/score.py`, `scripts/labeler/elm_ours_evaluate.py`). The bins are
the ELM-O benchmark's own: 50 ms cells `[50k, 50k + 50)` wholly inside one absent,
individual or crowd span and inside analysed time. Hard calls use the fold's threshold
(F1-maximising on that fold's 14 inner-validation shots, applied to the fold's held-out
shots; the five event thresholds are 0.46, 0.30, 0.76, 0.17, 0.39). AUROC and AUPRC are
threshold-free, pooled over the out-of-fold bins. 95 % intervals are percentile intervals
over 1,000 shot resamples with the same draws for every method, so differences are paired.
The ELM-O and clock rows reproduce the ELM-O benchmark's counts exactly (the evaluation
records `elmo_counts_match_published`).

Two sets: `bes73` (73 shots, 6,843 bins, 40 % present; analysed time is ELM-O's chunks) and
`all119` (119 shots, 12,409 bins, 38 % present; analysed time is where the fetched records
have samples).

| set | method | AUROC | AUPRC | F1 | precision | recall |
|---|---|---|---|---|---|---|
| bes73 | `elm-ours` | 0.941 [0.898, 0.972] | 0.875 [0.751, 0.961] | 0.845 [0.786, 0.896] | 0.826 [0.740, 0.896] | 0.864 [0.790, 0.934] |
| bes73 | `elm-elmo` | 0.918 [0.877, 0.951] | 0.833 [0.753, 0.890] | 0.842 [0.789, 0.885] | 0.840 [0.768, 0.899] | 0.844 [0.779, 0.892] |
| bes73 | `elm-clock` | -- | -- | 0.708 [0.602, 0.800] | 0.822 [0.720, 0.914] | 0.621 [0.488, 0.757] |
| all119 | `elm-ours` | 0.941 [0.906, 0.967] | 0.877 [0.774, 0.949] | 0.830 [0.778, 0.871] | 0.818 [0.750, 0.877] | 0.842 [0.780, 0.895] |
| all119 | `elm-clock` | -- | -- | 0.757 [0.680, 0.827] | 0.866 [0.788, 0.930] | 0.673 [0.575, 0.779] |

ELM-O's AUROC and AUPRC come from sweeping its threshold eta, the clock makes hard calls
only (see [elm_benchmark_elmo.md](elm_benchmark_elmo.md)).

**By kind** (the review's crowd spans and single ELMs, `crowd_bin_recall`,
`individual_span_recall`, `absent_span_alarm_rate` in the JSON):

| set | method | crowd bins found | individual spans touched | absent spans touched |
|---|---|---|---|---|
| bes73 | `elm-ours` | 0.877 [0.805, 0.942] | 0.699 [0.277, 0.871] | 0.610 [0.452, 0.742] |
| bes73 | `elm-elmo` | 0.852 [0.788, 0.901] | 0.634 [0.267, 0.859] | 0.440 [0.301, 0.623] |
| bes73 | `elm-clock` | 0.628 [0.492, 0.771] | 0.215 [0.090, 0.631] | 0.110 [0.053, 0.188] |
| all119 | `elm-ours` | 0.851 [0.789, 0.903] | 0.683 [0.379, 0.834] | 0.596 [0.478, 0.689] |
| all119 | `elm-clock` | 0.680 [0.581, 0.789] | 0.233 [0.122, 0.521] | 0.105 [0.063, 0.168] |

`elm-ours` finds the crowds and most of the single ELMs; it also touches more of the spans
marked absent than the clock does (60 % against 10 %). As for ELM-O, many of those spans
hold real ELMs (the ELM-O benchmark found ELM trains in 96 of the 218 absent spans), so the
absent-span alarm rate is partly a label-gap measure; it was not recounted here.

**The onset head** (`onset` in the JSON; a detected onset is a peak of the onset head above
the fold's onset threshold, at least 10 ms from a higher one; a reviewed onset is found when
one lies within the tolerance):

| set | method | tolerance | precision | recall | F1 |
|---|---|---|---|---|---|
| all119 | `elm-ours` | 5 ms | 0.032 | 0.075 | 0.045 [0.016, 0.083] |
| all119 | `elm-ours` | 10 ms | 0.050 | 0.117 | 0.070 [0.030, 0.121] |
| all119 | `elm-clock` | 5 or 10 ms | 0.500 | 0.192 | 0.277 [0.153, 0.482] |

This is a limitation, not a result: only 120 single ELMs carry an onset label, the ELMs
inside crowds carry none, and absent spans that contain ELMs teach the head that a real
onset is background. A better onset detector needs per-ELM labels the review does not
hold (the ELM-O benchmark's labelled windows by David Smith have them but are BES-based and
sit on 211 other shots).

## `elm-dsm`: the lab's ELM survival model

The model is the lab's Deep Survival Machine for the time to the next ELM (auton-survival,
LogNormal, three components, a 128-unit ReLU6 embedding, risk at 5/10/20/50 ms), **as the
labeler fitted it** on the 629,023 survival rows of the `wpqh_elm_hiro` project with its 60
non-BES columns (`labeler.models.d3d_elm_time_to_event_dsm`; the upstream graphs need 64 BES
channels the corpus lacks). No reviewed shot is in its published train or test split (0 and
0, `reviewed_shots_in_published_split`), so nothing leaks.

**Own target** (`own_target` in the DSM record). On the test rows of its own split (82 shots,
142,745 rows, 44 % positive) AUROC is 0.758, 0.764, 0.770, 0.777 at 5, 10, 20, 50 ms; they
agree with the training record. This is the published setting, "legacy" in the paper's
sense.

**On the reviewed bins, as published.** The model is not retrained. Its 50 ms risk at the
row before a bin is the bin's score ("will an ELM start in the next 50 ms", asked before the
bin); the hard call uses the F1-maximising threshold of the fold's inner-validation shots.
The AUROC is the same at all four horizons (0.732-0.744 on 73 shots, 0.780-0.786 on 119), so
the horizon does not matter for detection.

**With the objective changed to detection** (`elm-dsm-detect`). The same 60 inputs and the
same 128-unit embedding, the survival heads replaced by one logit, trained by cross-entropy
on whether the 50 ms ending at a row is ELMy in the review (the horizon-0 case), on the
same folds and inner-validation shots as `elm-ours`, scored at the row ending the bin.
`elm-dsm-detect-init` starts from the published embedding. Detector training uses the
`Detector` of `labeler/elm/dsm.py`: Linear(60, 128, no bias), ReLU6, dropout 0.2, one
logit; AdamW (lr 1e-3, weight decay 1e-4), batch 512, up to 40 epochs, best epoch by
inner-validation AUPRC. Deviation from the published architecture: the survival heads are
replaced by a single logit.

**Common bins.** The DSM serves rows only where the ECE record covers the 50 ms they
summarise and up to 5,975 ms. A comparison with it keeps the bins whose forecast row and
detection row both exist for every method and cuts every method's analysed time to the
time those rows summarise: 6,527 of 6,843 bins (bes73) and 11,653 of 12,409 (all119).
`elm-ours`, `elm-elmo` and the clock are re-scored on them; ELM-O's sweep AUROC and AUPRC
are not available on the restricted bins.

| set | method | AUROC | AUPRC | F1 |
|---|---|---|---|---|
| bes73 | `elm-ours` | 0.939 [0.893, 0.972] | 0.876 [0.752, 0.962] | 0.848 [0.790, 0.899] |
| bes73 | `elm-dsm` as published | 0.732 [0.660, 0.805] | 0.647 [0.554, 0.743] | 0.601 [0.517, 0.673] |
| bes73 | `elm-dsm-detect` | 0.854 [0.803, 0.898] | 0.751 [0.655, 0.847] | 0.759 [0.681, 0.822] |
| bes73 | `elm-dsm-detect-init` | 0.837 [0.782, 0.885] | 0.738 [0.645, 0.826] | 0.741 [0.654, 0.811] |
| bes73 | `elm-elmo` | -- | -- | 0.841 [0.786, 0.885] |
| bes73 | `elm-clock` | -- | -- | 0.714 [0.610, 0.806] |
| all119 | `elm-ours` | 0.939 [0.901, 0.966] | 0.878 [0.775, 0.951] | 0.834 [0.782, 0.875] |
| all119 | `elm-dsm` as published | 0.780 [0.734, 0.829] | 0.666 [0.594, 0.737] | 0.638 [0.577, 0.699] |
| all119 | `elm-dsm-detect` | 0.855 [0.812, 0.896] | 0.762 [0.683, 0.835] | 0.735 [0.672, 0.792] |
| all119 | `elm-dsm-detect-init` | 0.854 [0.812, 0.893] | 0.761 [0.682, 0.834] | 0.736 [0.671, 0.795] |
| all119 | `elm-clock` | -- | -- | 0.759 [0.683, 0.829] |

Retraining for detection raises the survival model's AUROC by 0.07-0.12 and its F1 by
0.10-0.16, mostly through recall (0.93 against 0.73 on 73 shots) at a false-alarm bin rate
of 0.32-0.39 against 0.38-0.50. It does not close the gap to `elm-ours` (paired AUROC
+0.083 [0.043, 0.128], F1 +0.099 [0.048, 0.148] on 119 shots; +0.086 and +0.089 on 73).
Against the clock its F1 is lower on 119 shots (0.735 against 0.759) and higher on 73 (0.759
against 0.714), both inside the intervals.

**What explains the gap is partly the inputs, partly the model.** The corpus serves the
DSM's inputs badly: `ip` and `bt` for 15 of 119 shots, the four CO2 columns for 44, the two
photodiodes never, so 4 to 10 of its 60 columns are mean-filled on every shot (counts in
`rows` of the JSON). On the shots where the CO2 columns are served the DSM is better and
still below `elm-ours` (119-shot AUROC, CO2 served 44 shots / missing 75 shots: `elm-dsm`
0.841 / 0.747, `elm-dsm-detect` 0.917 / 0.818, `elm-ours` 0.953 / 0.931; the intervals of
`elm-dsm-detect` and `elm-ours` overlap in the served stratum, 0.876-0.953 against
0.920-0.981). The DSM never sees D-alpha, the signal an ELM is a burst of, which `elm-ours`
reads at 10 kHz.

## Reference swap

The AE audit scores the same detectors against the lab's older annotation and against the
dense expert labels and shows the older reference reverses their ranking. Here the older
reference is Hiro's ELM table.

**Overlap, first** (`overlap` in the swap JSON).

| legacy reference | content | reviewed shots it covers |
|---|---|---|
| `edge_localized_mode_format_2026_v1.csv` (Hiro, 576 shots) | 50 ms bins, category 1 where an ELM onset falls in the bin | **8 of 119** (189885, 190637, 190643, 192721, 192732, 192751, 196541, 200385); 7 of them have BES |
| `elm_all_ground_truth.csv` (365 shots) | one yes/no per shot (has ELMs) | **5 of 119** (190637, 190643, 192721, 192751, 196541); shot-level, so it cannot score bins |
| D. Smith's labelled windows | BES windows with ELM regions | **0** (211 other shots; the ELM-O benchmark's truth) |

The shot-level table says yes for all five overlapping shots; the review agrees on four and
holds no present span for 192751 (one shot, so not a count of anything). Only Hiro's table
can be scored in bins, and only on 8 shots.

**Conversion** (documented in `labeler/elm/swap.py`). The table's rows are runs of 50 ms
bins on the same `[50k, 50k + 50)` grid as the scored bins (no off-grid edge), category 1
meaning an onset falls in the bin. A scored bin takes the category of the table row that
contains its midpoint; a bin the table does not cover has no legacy value and is dropped.
The legacy reference thus marks a bin present when an onset falls in it; the review marks
it present when it lies wholly inside an individual-ELM span or a crowd. The two agree on a
bin holding one isolated ELM's onset and differ on the rest of a crowd or an ELM longer than
a bin. The comparison is on the 641 bins (8 shots) both cover, with the DSM's rows.

**Finding 1** (`finding_1`): the legacy table read as a detector of the review.

| | value |
|---|---|
| bins both cover | 641 (147 review-present, 111 legacy-present) |
| \|M\| (review-present, legacy-absent) | **40** (27 % of the present bins) [16 %, 59 %] |
| \|P\| (legacy-present, review-absent) | **4** (all in 192751) |
| precision / recall / F1 of legacy against review | 0.964 / 0.728 [0.413, 0.837] / 0.829 [0.532, 0.901] |
| crowd bins marked present by legacy | 94 of 114 (82 %) |
| single-ELM bins marked present by legacy | 13 of 33 (39 %) |

The older table misses 20 single-ELM bins and 20 crowd bins and adds almost nothing (four
bins, all of one shot): it is a conservative subset of the review, which marks ELMy time
and not only onsets.

**Finding 2** (`comparison`, `swap.overlap`): every method's scores and calls, unchanged,
against both references on the same 641 bins.

| method | AUROC, review | AUROC, legacy | F1, review | F1, legacy |
|---|---|---|---|---|
| `elm-ours` | 0.963 [0.83, 1.00] | 0.955 [0.83, 0.99] | 0.720 [0.33, 0.90] | 0.614 [0.22, 0.79] |
| `elm-dsm` as published | 0.821 [0.65, 0.93] | 0.851 [0.72, 0.94] | 0.451 [0.19, 0.67] | 0.360 [0.11, 0.56] |
| `elm-dsm-detect` | 0.772 [0.67, 0.90] | 0.779 [0.66, 0.92] | 0.503 [0.22, 0.71] | 0.404 [0.13, 0.59] |
| `elm-clock` | -- | -- | 0.430 [0.11, 0.66] | 0.406 [0.13, 0.64] |
| always present | 0.500 | 0.500 | 0.373 | 0.295 |

(`table_elm_benchmark.tex`; the 7 BES shots add `elm-elmo` in `table_elm_benchmark_bes.tex`;
the full table with precision, recall and the oracle rows is `table_elm_swap_full.tex`.)

The AUROC order is the same under both references (`elm-ours`, the survival model as
published, then the retrained one). The F1 order differs in one place on the 8 shots: the
clock sits below the three survival-model variants against the review and above them
against the legacy table (F1 0.406 against 0.360-0.404), a gap far inside the intervals; on
the 7 BES shots two pairs swap (also `elm-dsm-detect` and `elm-elmo`). In all 15
comparisons against `elm-ours` (`paired_differences`) the sign of the difference is the same
under both references. **The answer is no reversal, and the evidence is thin:** 8 shots,
641 bins, intervals running across most of the range. For the AE labels the older
annotation reversed the ranking; for the ELM labels the older table is a close,
conservative cut of the review (|P| = 4), so a swap cannot show much. A ranking test with
power needs more reviewed shots among the 576 of Hiro's table (8 are reviewed today); none
was reviewed for this stream.

**Proxy references (not independent; do not read as legacy).** Many shots, a reference built
from a detector: the bins a detector's output marks (the producer is not scored against its
own reference).

| proxy | shots | review (AUROC / F1) | proxy reference (AUROC / F1) | order flips |
|---|---|---|---|---|
| ELM clock ELMy spans (`table_elm_proxy_clock_spans.tex`) | 119 | `elm-ours` 0.939 / 0.834; `elm-dsm` 0.780 / 0.638; `elm-dsm-detect` 0.855 / 0.735 | 0.838 / 0.682; 0.725 / 0.543; 0.750 / 0.573 | AUROC: `elm-dsm-detect` against its `-init` variant only |
| ELM-O onset bins (`table_elm_proxy_elmo_onsets.tex`) | 73 | 0.939 / 0.848; 0.732 / 0.601; 0.854 / 0.759; clock F1 0.714 | 0.916 / 0.816; 0.683 / 0.587; 0.830 / 0.726; clock F1 0.673 | none |

A detector-made reference lowers every method's score a little and does not reorder them.
The clock's rows used the clock's category-1 rows as spans (one per ELMy period); an earlier
attempt that read them as onsets made the reference 0 % present, because they are not onsets.

## Reproduce

Environment `labelmaker` through pixi for everything except training (CPU torch only);
training uses the CUDA venv `$LABELER_ROOT/envs/phase3/bin/python` with `PYTHONPATH=src`,
launched through pixi, as `scripts/labeler/ae_train.sbatch` does.

```
python -m labeler.elm.prepare                           # inputs -> $LABELER_ROOT/round4/elm/inputs
python -m labeler.elm.train --run cv2 --device cuda     # phase3 python; folds + OOF predictions
python scripts/labeler/elm_ours_evaluate.py --run cv2   # outputs/labeler/elm/ours/evaluation.json
python scripts/labeler/elm_dsm_evaluate.py --run cv2    # outputs/labeler/elm/dsm/evaluation.json
python scripts/labeler/elm_reference_swap.py --run cv2  # outputs/labeler/elm/swap/ (json + tex)
```

`elm_dsm_evaluate.py --rescore` recomputes the JSON from the saved scores and fits without
retraining. The tests are `tests/labeler/test_elm_ours.py` and
`tests/labeler/test_elm_dsm_swap.py`. The cv2 run's training record is
`$LABELER_ROOT/round4/elm/cv/cv2/run.json`.

## Deviations and caveats

- No fetching was done for this stream; the inputs are the records `elmo_fetch.py` stored
  for the 119 reviewed shots. `ip` and `bt`, which the survival model reads, are on disk for
  15 of them and were deliberately not fetched for the other 104 (the allowed fetches were
  the filterscopes and the interferometer), so the survival model runs with them
  mean-filled; this lowers its scores (the CO2 strata above show the size of the effect for
  the other missing columns).
- The survival model's inputs are clipped to +-10 normalised units (`spec.Z_LIMIT`).
  Upstream dropped training rows beyond that limit; dropping them here would score a
  different set of bins, and `in_filter` in the saved rows records which rows it would have
  dropped.
- `elm-dsm-detect` replaces the survival heads with one logit (above).
- Thresholds come only from each fold's 14 inner-validation shots (no cohort test shot,
  none of the held-out fold); they differ widely between folds (event thresholds 0.17-0.76),
  so the F1 values carry that noise, and AUROC and AUPRC are the steadier numbers.
- Absent spans contain real ELMs (ELM-O benchmark), so false-alarm rates and precision are
  lower bounds on the models' quality; no label was changed.
- The cv2 model was trained from the working tree that became commit 0d16c19 (the run
  record names its base commit 21183f0).
- The clock is the reviewers' starting point and not independent of the review; its
  rows are context only.
- The reference swap's overlap is 8 shots (5 and 0 for the two other references): the
  ranking claim is a statement about what 8 shots can show.
