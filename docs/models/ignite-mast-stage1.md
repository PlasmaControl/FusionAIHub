---
title: "IGNITE on MAST: stage 1 fixed tokenizers"
---

# IGNITE on MAST: stage 1 fixed tokenizers

**Status: measured (2026-11-23). The assignment `v1` (1,470 values per 5 ms frame) passes the gate on all 57 (task, signal) pairs; five pairs miss the fidelity target, each recorded with its cost, and one input signal (`interferometer-n_e_line`, through one shot) misses its reconstruction target. Stage 1c's learned tokenizers are measured (2026-11-23, see [Learned tokenizers (stage 1c)](#learned-tokenizers-stage-1c)): no network or shared learned tokenizer replaces a fixed codec, and the linear concatenated basis clearly beats the shared one on the four magnetics arrays.**

## What this is

Stage 1b gives each of the 39 cached MAST signals of TokaMark a tokenizer
without learned weights, a frozen assignment that says how many values each
signal keeps per 5 ms frame, and a measured
representation floor per task in TokaMark's own units. Three codecs do the work: a cosine transform per frame
(`frame_dct`), a principal-component basis for profiles and the flux map
(`linear`) and a band transform for the two Mirnov spectrograms (`band_dct`).
Stage 2 can start on these tokenizers alone, and every learned tokenizer of
plan 1c has this page's numbers as its baseline. The design is in
[the design page](./ignite-mast-design.md), the measurements this stage builds
on are in [the stage-0 page](./ignite-mast-stage0.md), and the plan is
`docs/superpowers/plans/2026-11-22-ignite-mast-stage1b-fixed-tokenizers.md`.

The code and its tests are committed, and the long jobs that produce the numbers
(K1 to K7 under [How to run](#how-to-run)) ran on 2026-11-22 and 2026-11-23;
their numbers are under [Results](#results). Nothing on this page is measured
except the dated numbers that carry their split, their number of shots and the
words "measured" or "rough look". A rough look is a small,
single read-only look with the code of the day. It sets an expectation and is
not a result.

## Units and read-outs

The same four units appear in the code and in the tables. A function's
docstring names the unit of every array it takes and returns.

| Unit | Definition | Where it is used |
|---|---|---|
| Physical | the unit of the data | the harness, the codecs' inputs and outputs |
| Model units | the cache's: `(c(x) - mean) / std` with `c` the identity, or `asinh(x / scale)` for the 15 signals the cache compresses (`transform.to_model`) | the cache, the stored `stats.npz` |
| Linear units | the compression undone but the data's own shape kept: `(x - mean) / std` for an uncompressed signal and `x / scale` for an asinh signal (`frames.to_linear`) | the default domain of the cosine transform |
| TokaMark units | `(x - SignalSpec.mean) / SignalSpec.std`, one constant pair per signal | every floor, error and target on this page |

The codecs call their domains `linear`, `compressed` (model units), `signal`
(TokaMark units) and `band` (linear power over a scale per band). A floor, a
fidelity target and an error are an NRMSE in TokaMark units, and lower is
better. "Values per frame" is the count of numbers that one 5 ms frame of a
signal keeps. NRMSE always means TokaMark's evaluator, reached through
`harness.score`, which is the same measurement stage 0 made for TokaMind's
representation.

Missing data is NaN and stays NaN. A frame in which some samples are missing
is encoded from its valid samples. A frame without a valid sample has NaN
values and decodes to NaN. A count of zero examples is reported as "no data",
never as an error of 0.

**The three read-outs.** A value is not one class among 64. It is a point
between two neighbouring bin centres, and the centres of each (token, value)
position are asinh-spaced over a range fitted on training shots, so a model
that predicts a distribution over the bins is trained on an ordered, two-hot
target. The same value is read three ways:

- `unquantized` is the value itself. K is chosen on this read-out.
- `ranged` is the value limited to the range of its bin centres, which is the
  most that a two-hot target over those bins can say. The fidelity target is
  judged on this read-out.
- `nearest` is the nearest bin centre, which is what a single sampled token
  says.

A value beyond the fitted range reads as the outermost bin centre in `ranged`
and `nearest`. It does not read as NaN and does not wrap.

**A floor.** The floor of a task is the NRMSE of decode(encode(true targets)),
scored by the stage-0 harness on TokaMark's own windows of the split, per
signal from the evaluator's per-signal table. It is the error a model would
have if it predicted its token values exactly, so it is a limit for models that
predict in this representation and says nothing about models that do not. A
floor is a fitted quantity: the ranges and the bases are fitted on training
shots, and the floor is read on validation shots. The floor of TokaMind's
representation (stage 0) is the comparison, and the "own mean" floor is the
reference that a tokenizer with more values has to beat (see the rules).

**The "own mean" reference.** It is the assignment that `assign --reference`
writes (`assign.reference_assignment`), scored in the `unquantized` read-out
(`scores/.../tok.ref.unquantized`), and what it keeps depends on the codec:

- `frame_dct`: K = 1, the mean of the valid samples of each 5 ms frame and
  channel, in linear units, repeated over the frame. The example is read, at
  the resolution of a frame.
- `band_dct`: K = 1 in every band, the band's mean power in the frame (the
  frame's columns averaged, valid columns only), flat over the 32 bins of the
  band. The example is read, at the resolution of a frame and a band.
- `linear`: 0 components, in the `signal` domain. There is no token, so the
  decode is the training mean profile (or map) for every frame, whatever the
  example is. It is a constant fitted on training shots and reads nothing of
  the example.

The gate's `own_mean` column is that floor, and rule 7 compares it only for a
setting that keeps more values per frame than the reference does (1 per channel,
1 per band, 0 components). So for a `linear` signal the test is to beat a
profile that does not depend on the example, which is the easiest of the three;
whether it can ever block one is not measured. Checked 2026-11-23 against
`tokenizers/assign.py:156-167` (`reference_assignment`), against the docstrings
of `FrameDCT` ("value 0 is the frame mean"), `BandDCT` ("value 0 is the band's
mean power in units of `band_scale`") and `LinearCodec` ("with `n_values = 0`
... `decode` gives the mean profile for every frame") in `tokenizers/codecs.py`,
and against `tokenizers/gate.py:11-13` and `:221-225` (the floor of `ref.json`
is `own_mean`; a setting with `values > reference_values` blocks unless its
`unquantized` floor is below it).

## The tokenizers

All three codecs turn one signal's window, in physical units, into token values
of shape `(N, F, n_tok, n_val)`: `N` windows, `F` frames of 5 ms, `n_tok` tokens
per frame and `n_val` values per token. The first values of a token are its best
short description, so a ladder of sizes is a ladder of prefixes. A fit gives
every position its bin range (`grid.fit_ranges`). The count of bins per value is
64, a number this stage chose and stage 2 may change, and the assignment
records it.

**`frame_dct`: one token per channel.** It serves the 28 slow signals (129
channels) and the four high-rate series (saddle coils, the two soft X-ray
cameras and D-alpha), 32 signals in all. A token holds the first K cosine
values of one channel's samples in one frame, so `n_tok` is the number of
channels and `n_val` is K. Value 0 is the frame mean in the codec's domain, and
the values are nested. A frame with some missing samples keeps the mean of its
valid samples in value 0 and 0 in the others. K runs over 1, 2, 3, 4, 6, 8, 12
and 16, up to the number of samples in a frame. A high-rate channel has 250
samples per frame and no separate filter: keeping the first K coefficients is
its low-pass filter (see [the departures](#where-this-departs-from-the-design)).
The domain is a per-signal choice between `linear` and, for an asinh signal,
`compressed`; the ladder measures both.

**`linear`: principal components of a profile or map.** It serves the four
profiles (`equilibrium-lcfs_r` and `lcfs_z` with 170 points, Thomson `t_e` and
`n_e` with 120) and the flux map (`equilibrium-psi`, 65 by 65). One time sample
is one example, flattened row-major. `n_values` coefficients are split into
`n_values // values_per_token` tokens of `values_per_token` values each, 4 for a
profile and 2 for the map, so a prefix of the tokens is a coarser profile.
`n_values` runs over 0, 8, 16, 24, 32, 48, 64 and 96, up to the number of
points. The coefficients of a profile with missing points are solved from the
valid points with a ridge, and a profile with no valid point has NaN values.
The ridge is the variance that the kept components leave, per point, and it is 0
for a signal in which no example has a missing point. For a signal where some do
(the Thomson profiles), `LinearCodec.encode_domain` applies it to every row,
complete rows too, and it shrinks each coefficient of a complete row by
`e / (e + ridge)` for its component's eigenvalue `e` (derived from that formula,
not measured). So the floor of such a signal includes the shrinkage, which is
small only for components whose eigenvalue is large against the ridge. With 0
values there is no token and the decode is the mean profile. Boundary
coordinates `lcfs_r` and `lcfs_z` have separate bases. The domain is one of
`signal` (TokaMark units), `linear` and, for an asinh signal,
`compressed`.

**`band_dct`: a cosine transform over the bins of each band.** It serves the
two Mirnov spectrograms, which the cache holds as TokaMark's `(3, 256, T)`
spectrograms at 1.024 ms. The 256 bins make 8 bands of 32 bins (31.25 kHz
each). One token is one (probe, band), so `n_tok` is 24 and `n_val` is the
largest K of any band, at most 32. Encoding takes, per bin, the mean over the
frame's valid columns (4 or 5 of them), a cosine transform across the band's 32
bins, and sets the values from index `k[band]` on to 0, so a band keeps its
own number of values. Decoding gives the same spectrum to every column of the
frame. Value 0 is the band's mean power in units of a per-band scale, and with
all 32 values kept in every band (768 values per frame) the decode is each
bin's frame mean. That setting is the frame oracle, the reference of rule 5. A
band's dropped positions encode to 0, not NaN, so the gate counts them as no
example when it counts bin usage (`bins_used_min`) and does not report every
band codec as using 1 bin of 64.

## Rules

Stated before any number existed. The seven rules below are copied from the
plan's section "Which number decides what", word for word, including its
stage-0 numbers as of 2026-11-22. "Floor" is the NRMSE of decode(encode(true
targets)) through `harness.score`, per signal from the evaluator's per-signal
table. Each rule was fixed before the measurement, 2026-11-22, and none is
changed here to fit a result.

1. **Fidelity target per signal:** 0.2 times the smallest best published score among the tasks that predict the signal. The EFIT scalars of tasks 1-1 and 2-1, the three betas among them, 0.0176 (task 3-3 also predicts the betas, at a larger score); `magnetic_axis_z` 0.0138 (task 4-3); coil currents, solenoid current and plasma current 0.0176; boundary 0.0074; flux map 0.0166; Thomson 0.0429; D-alpha 0.0504; soft X-rays 0.0424. None for the Mirnov spectrograms.

    *Fixed before the measurement, 2026-11-22.*

2. **K by the ladder.** For each signal the ladder is scored on 100 `val_select` shots spread evenly. K is the smallest rung whose `unquantized` floor is at or under the signal's target. Every domain of the signal is run (a series: linear units, and asinh units if the cache compresses it; a profile or map: TokaMark's units, linear units, and asinh units if compressed) and the domain with the smaller K wins; on a tie the earlier one in that order. The ladder tasks are 1-2 (boundary), 1-3 (flux map), 2-1 (currents), 3-1 (Thomson), 3-2 (D-alpha and soft X-rays) and 4-5 (Mirnov); a signal with one sample per frame needs no ladder (K = 1 is exact).

    *Fixed before the measurement, 2026-11-22.*

3. **If no rung meets the target,** K is the smallest rung whose pass-through cost is at most 1%: the direct model's prediction for a task that predicts the signal is passed through the tokenizer and scored, and the cost is that score over the raw prediction's score, minus 1, per signal on the same shots. The miss is recorded with both numbers. If the task has no usable direct model (1a marks it stale, unfinished or not competitive), K is the largest rung and the miss is recorded.

    *Fixed before the measurement, 2026-11-22.*

4. **Signals no task predicts** (magnetics, saddle coils, coil voltages, gas, beam power, requested waveforms, line density): K is the smallest rung whose reconstruction error on 100 `val_select` shots, computed from the cache in TokaMark units (per shot RMSE over valid samples, mean over shots), is at or under 0.0176. The input round-trip is then measured on the same 100 `val_select` shots: the direct models of tasks 1-1, 1-2, 2-1 and 4-1 are scored with every input and actuator passed through its tokenizer, and raw. What decides is the added error `sqrt(max(inputs**2 - raw**2, 0))` over the task's best published score: above 0.2 on any of the four raises K of these 11 signals by one rung, once (the first assignment is kept beside the raised one), and is recorded for stage 2's token-path check. For a model at the best published score this is a cost of 2%; written this way it does not depend on how good the direct model is, which matters because two of the four are behind the published scores (stage 0, `val_report`: 1-1 0.1361 against 0.0941, 2-1 0.1523 against 0.0880; 1-2 is ahead at 0.0362 with the variant `lr1e4`; 4-1 ties at 0.1952). It assumes that the two errors add in quadrature. The plain ratio `inputs / raw - 1` is reported beside it. The round-trip is reported again on `val_report` in the confirmation.

    *Fixed before the measurement, 2026-11-22.*

5. **Mirnov spectrograms.** The reference is the frame oracle: all 32 coefficients in every band. Step a: the lowest band with 4, 8, 16, 32 coefficients, the others whole. Step b: bands 1 to 7 with a common 1, 2, 4, 8, 16, 32, the lowest band at its choice. In step a, K is the smallest rung within 2% (relative, for both signals) of the frame oracle; in step b, the smallest rung within 2% of step a's chosen floor. The benchmark cannot see the upper bands (they hold 0.2% and 23% of the squared error), so step b has a second condition: K is at least the smallest rung whose spectral contrast ratio is 0.9 or more (`ignite.gate.spectral_contrast_ratio` of the decoded against the true frame-mean spectra, in model units, bins 32 to 255, on the same 100 shots from the cache). The threshold 0.9 is the author's; it keeps mode lines above 31 kHz in the tokens although no benchmark task rewards them. The frame oracle, the mean baseline on the same shots and TokaMind's floor go to the owner the day they exist (open decision 1).

    *Fixed before the measurement, 2026-11-22.*

6. **Confirmation.** The chosen assignment is scored on all `val_report` shots for the 14 tasks in the three read-outs. The gate judges the fidelity target on `ranged`. If `ranged` is more than 1% above `unquantized` for a signal, its range is too narrow: reported, not fixed here.

    *Fixed before the measurement, 2026-11-22.*

7. **What blocks stage 2 for a signal:** a tokenizer with more than the reference's values that does not beat the "own mean" reference. Nothing else blocks; a miss of the fidelity target is recorded with its number.

    *Fixed before the measurement, 2026-11-22.*

## Where this departs from the design

Each item names a sentence of [the design](./ignite-mast-design.md) and the
measured reason for the departure. The measurements are rough looks made by the
plan's author on the CPU on 2026-11-22, from cache v1, on training shots spread
evenly over the campaigns. Cache v1 holds 11,188 shots (counted 2026-11-22: 8,963
train, 1,115 val, 1,110 test) and 39 signals. A rough look is pooled RMSE over
valid samples divided by TokaMark's standard deviation. It is not a harness
score and the ladders of this stage replace it.

- **Transform domain.** The design says that after the compression and
  standardization of the data path, the token holds the first K cosine
  coefficients. Here the cosine transform is taken in linear units by default
  and the compression moves into the spacing of the bin centres. Reason, rough
  look of 2026-11-22 on 24 training shots, transform taken in model units: for
  the upper soft X-ray camera the error with K = 1, 2, 4, 8 is 0.105, 0.128,
  0.079, 0.060, so K = 2 is worse than K = 1. A truncation in asinh units is
  not a projection in physical units. Also by reasoning, not measured: a mean
  taken on a compressed scale is biased low. The domain is a per-signal switch,
  both are measured by the ladder, and the lower floor in TokaMark units wins,
  which also answers stage 0's open question whether asinh about zero is the
  right compression.
- **Trend path.** The design says that a high-rate channel's trend is low-pass
  filtered and decimated to the slow family's rate. Here no separate filter
  exists: the channel's 250 samples per frame go through the same cosine
  transform, and keeping the first K coefficients is the low-pass filter. Reason:
  one code path, exact at frame edges. This is a choice and has no measurement
  behind it. The design leaves the filter to the builder.
- **Bands.** The design says that a 50 kHz channel has one band and a 500 kHz
  channel has ten, and leaves the band definition to the builder. Here a band is
  32 of TokaMark's bins (31.25 kHz), so the 256 bins make 8 equal bands. The
  transform works in linear power, which is what the design asks for (expected
  linear power per bin per frame). Reason for linear power over whitened or log
  power, rough look of 2026-11-22 on 40 training shots (19 and 36 of them have
  the two signals, whole shots and not windows): truncating in the cache's
  per-bin whitened units gave errors of 0.93 and up to 74, because the per-bin
  scale varies by a factor of 1,000 between neighbouring bins near 0 Hz.
- **The fidelity target is not applied to task 4-5.** The design says
  reconstruction NRMSE at most a fifth of the best published score for that
  task. Reason, rough look of 2026-11-22 on the same 40 training shots: a
  perfect prediction of every 5 ms frame's mean spectrum scores 0.676 and 0.422
  against mean baselines of 0.915 and 0.573 (toroidal-field and poloidal
  probes), which is 74% of the mean baseline for both, so frame statistics
  cannot reach a fifth of 0.6510. Open decision 1 of the plan, and
  [Limits](#limits) below, say more.
- **A miss is not always closed by more coefficients.** The design says a miss
  has to be closed before the first complete table, by more coefficients for a
  fixed transform. Here, where no rung up to 16 values meets the target, rule 3
  chooses K by what the miss costs a prediction. Reason, rough look of
  2026-11-22 on 24 training shots, cosine truncation per frame, error with
  K = 1, 2, 4, 8: soft X-ray lower camera 0.098, 0.081, 0.069, 0.065, upper
  camera 0.105, 0.128, 0.079, 0.060, D-alpha 0.221, 0.155, 0.122, 0.091,
  against targets of 0.0424 and 0.0504. The missing part is expected to be
  fluctuation inside a frame, and closing it would take most of the 250 samples
  per frame (an expectation, not a measurement). Signals that miss are
  published with their numbers and with the pass-through cost that says what the
  miss costs on predictions.
- **Three read-outs of a token value.** The design measures the gate by
  decoding from unquantized values. Here K is chosen on `unquantized`, the
  fidelity target is judged on `ranged`, and all three are reported. Reason:
  the two-hot target of stage 2 can express no more than `ranged`, and a single
  sampled token says `nearest`. No measurement is behind this choice.
- **Number of bins: 64 per value, asinh-spaced.** The design says each value
  sits on a fine scale of ordered bins and leaves the number to the builder.
  The plan chose 64 per value on 2026-11-22, with the bin centres asinh-spaced.
  No measurement is behind the number 64. Stage 2 owns the loss (the number of
  bins per value and the smoothing of the bin targets) and may change it, and
  the assignment records the number it was made with.
- **No collapse rule.** The design says (in "What blocks and what does not"): "A
  tokenizer that has collapsed, or that does not beat replacing each window by
  its own mean, blocks." Here only the second half blocks (rule 7). Reason: the
  design's own account of this tokenizer is "It has no weights, so it cannot
  collapse", and collapse is a learned codebook that uses a few codes. A fixed
  grid that wastes its bins shows in `bins_used_min`, a column of the gate table
  that has no threshold, and as a `ranged` floor above the `unquantized` one
  (rule 6, which the gate prints as a line). So a fixed tokenizer with a badly
  used grid passes the gate with its numbers shown. The collapse rule binds for
  the learned tokenizers of plan 1c. No measurement is behind this choice.
- **The "own mean" is a frame mean, not a window mean.** The design says "replacing
  each window by its own mean". Here it is defined per codec as above: the mean
  of each 5 ms frame, the mean power of each band and frame, or, for the
  profiles and the flux map, the training mean profile with no use of the
  example. Reason: the reference has to be a point on the ladder of its own
  codec, with K = 1 or 0 components, so that "more values than the reference" is
  a count the gate can make (`values_per_frame`). A frame mean is a finer
  reference than a window mean, so beating it is harder for the signals that
  have one, and the profile reference is an easier one (see above). No
  measurement is behind this choice.

## How to run

The controller runs these jobs from `/home/nvidia/FusionAIHub` with
`IGNITE_MAST_ROOT` unset. In the commands `RT="scripts/mast/py.sh
scripts/mast/run_tokenizer.py"` and `S1=/models/intermediate/mast/stage1`.
Logs go to `$S1/logs/`. The durations are the plan author's estimates from
stage-0 timings (floor scoring of task 4-1 over 1,110 shots: 4,131 s; cache
build over 11,188 shots: 225 s, both measured 2026-11-22). The durations this
code took are in the paragraph after the table.

Sub-commands of `run_tokenizer.py`: `fit`, `score` (instruments `floor`,
`passthrough`, `inputs` and `raw`; read modes `unquantized`, `ranged` and
`nearest`), `ladder` (`--task` or `--inputs`), `assign` and `gate`. The task
names are `task_1-1` to `task_4-5`. What each one writes under `$S1`:

- `fit/v1/` holds `fit.npz` and `fit.json`, the ranges and bases fitted on 2,000
  training shots spread evenly (`--n-shots`).
- `ladders/<task>.csv` and `.json`, and `ladders/inputs.csv` and `.json` for the
  signals no task predicts.
- `assignments/<name>.json`, which `assign` refuses to overwrite.
- `scores/<task>/<predictor>/<split>/` holds `score.json` and `signals.csv`. The
  predictor is `tok.<name>.<mode>` (floor), `pass.<name>.<mode>.<variant>`,
  `inputs.<name>.<mode>.<variant>` or `raw.<variant>`, and a ladder pass is
  `tok.ladder-<domain>-<i>` (with `pass.<pass>.unquantized.<variant>` for its
  pass-through).
- `gate/<name>.csv` and `gate/<name>.md`.

| Job | After | Command | GPU | Duration | What is read from the output |
|---|---|---|---|---|---|
| K1 fit | the `fit` command | `CUDA_VISIBLE_DEVICES="" $RT fit --workers 64` | none | 15 to 40 min | 39 lines. No signal says "no data". `n_skipped` of `magnetics-b_field_tor_probe_cc_field` is about a third of 2,000 and of Thomson about a fifth; the soft X-ray cameras skip about 5% (102 and 104 of 2,000 shots, measured 2026-11-23 at K1; sampling-rate mismatch). If a signal has under 500 shots, stop and report. |
| K2 ladders | K1, the `ladder` command | `$RT ladder --task T [--variant V --gpu N --share-gpu]` for the six ladder tasks (`task_1-2`, `task_1-3`, `task_2-1`, `task_3-1`, `task_3-2`, `task_4-5`), and `$RT ladder --inputs`, all at once | one GPU for a few minutes per task with a variant; may be shared | 15 to 60 min each, about 1 h together | `$S1/ladders/*.csv`. Per signal: the floor falls with K; the first rung equals the "own mean" reference. For task 4-5: the frame oracle (the row `k0 = 32`), the mean baseline on the same shots, the contrast per rung. These three numbers and TokaMind's floor (0.3605 on `val_report`) go to the owner the same day (open decision 1). |
| K3 assign | K2 | `$RT assign --name v1` and `$RT assign --name ref --reference` | none | seconds | The printed table: which signals miss their target, and the total of values per frame (stage 2's sequence length follows from it). |
| K4 input round-trip | K3 | for tasks 1-1, 1-2, 2-1, 4-1: `$RT score --task T --split val_select --n-shots 100 --instrument raw --variant V --gpu N --share-gpu`, and the same with `--instrument inputs --assignment $S1/assignments/v1.json --mode unquantized` | one each, may be shared | 8 passes of 3 to 10 min | Per task the added error of rule 4 over the task's best published score, and the ratio `inputs / raw - 1`. Above 0.2 on any task: `$RT assign --name v1r --from v1 --raise-inputs`, repeat the four `inputs` passes with `v1r` once, and use `v1r` from here on. Both costs are recorded. |
| K5 confirmation | K4 | for the 14 tasks: `CUDA_VISIBLE_DEVICES="" $RT score --task T --split val_report --assignment A --mode M --num-workers 16` for the three read-outs, and with `--assignment $S1/assignments/ref.json --mode unquantized` | none | 56 passes of 10 to 40 min; about 2 to 3 h in waves of 14 | The last line of each pass. A pass that ends with a non-finite score or an error stops the wave. |
| K6 costs on `val_report` | K5 | `--instrument raw` and `--instrument passthrough --mode ranged` for every task with a signal that misses its target (expected: 3-2, 4-1, 4-2) and for 1-2 as a control; `--instrument inputs --mode ranged` and `raw` for 1-1, 1-2, 2-1, 4-1 | one each, may be shared | about 14 passes of 15 to 45 min; about 2 h on 4 GPUs | Per task and signal the pass-through cost; per task the input round-trip cost. A pass-through cost above 1% on `val_report` for a signal chosen by rule 3 is reported, and K is not changed again. |
| K7 gate | K5, the `gate` command | `CUDA_VISIBLE_DEVICES="" $RT gate --assignment $S1/assignments/<name>.json` | none | 5 to 15 min | The table, any `range too narrow` lines (rule 6: `ranged` more than 1% above `unquantized`; reported, no verdict changes), and the last line. `exit 1` means a row blocks: a transform with more values than its reference can only lose to it through the read-out or a bug, so the controller stops and reports instead of lowering K. |

The variant `V` of a task is the one whose score the headroom table shows for it
(`scripts/mast/py.sh scripts/mast/run_headroom.py --table`, and `--table-all`
once plan 1a has run). A task whose direct row is marked stale, unfinished or
not competitive has no usable model for the pass-through of rule 3, and its
ladder runs without `--variant`. For the input round-trip of rule 4 a model that
is behind the published scores is still used (rule 4 says how its number is
read); a stale or unfinished one is not. No variant is tuned: stage 0's base
configuration was fixed before any score existed, and its variants are single
measurements (`lr1e4` on 1-2, `h050b256` on 4-4, `s90k` on 1-1). Tasks 1-3 and
3-2 get their direct model from plan 1a's job J4; if it is not there when K2
runs, their ladders run without pass-through and are run again when it is (only
a signal that misses its target needs it).

Measured durations (2026-11-22 and 2026-11-23, node13): K1 7 min (64 workers);
K2 65 min for the six task ladders and the inputs together (02:12 to 03:17 UTC);
K5 94 min for the 56 passes in four waves of 14 on the CPU (03:23 to 04:57); K6
56 min for 16 passes, two per GPU on three GPUs (03:47 to 04:43; the 3-3 pair
was added to the 14 planned because K5 showed a Thomson miss on 3-3); K7 6 s.

After K7 the controller fills the four tables below from
`$S1/assignments/<name>.json`, `$S1/gate/<name>.md` and the stored scores, sets
the status line, and commits with the numbers, their date, split and shot counts
in the body.

## Results

Four tables, each with its date, split and number of shots in the line above it.
What they say, in short (all measured 2026-11-23):

- **Nothing blocks stage 2.** The gate passes all 57 (task, output signal)
  pairs of the 14 tasks with `v1`: no tokenizer with more values than its
  own-mean reference loses to it.
- **Five pairs miss the fidelity target** (a fifth of the task's best published
  score, judged on `ranged`): D-alpha on 3-2 (0.1380 against 0.0504), the lower
  soft X-ray camera on 4-1 and 4-2 (0.0479 against 0.0424 and 0.0428), and the
  Thomson profiles on 3-3 (n_e 0.0444 and t_e 0.0436 against 0.0429; on 3-1,
  whose target is 0.0662, both meet it). Passing each task's direct model
  through the tokenizer costs at most +0.24% on any of these signals (n_e on
  3-3) and +0.22% on D-alpha, the one signal whose K rule 3 chose, so the misses
  cost a model's predictions under 1% (K6).
- **One large pass-through cost, on a signal that meets its target.** On task
  2-1 (100 `val_select` shots, K2), passing the direct model's prediction
  through the tokenizer costs 15.5% on the solenoid current and 1.0% on the coil
  currents, both at K = 1. Both meet their target of 0.0176 (floors 0.0140 and
  0.0171), and rule 2 takes the smallest K that meets the target without looking
  at the cost, so K stays at 1. The cost is real for a model that predicts the
  solenoid current better than the task's average: the 2-1 model's raw error on
  it is 0.0248. Task 2-1 was not in K6. Whether rule 2 should also bound the
  pass-through cost is a question for the owner, not settled by this plan.
- **The inputs pass through cheaply.** With every input and actuator sent
  through its tokenizer, the four direct models change by -0.63% to +1.21% on
  `val_report`; the largest added error is 0.145 of the best published score
  (2-1), under rule 4's 0.2. On the 100 `val_select` shots of K4 the largest
  added error is 0.174 (1-2, +1.87%), the closest any task came to 0.2.
- **One input misses its target, through one shot.** `interferometer-n_e_line`
  reconstructs at 0.2267 with 16 values per frame against a target of 0.0176
  (rule 4, mean over 100 `val_select` shots of the per-shot RMSE; 0.32 to 0.39
  at the smaller rungs), so it got the largest rung of its domain. One shot of
  the 100, 16575, has spikes of about 1e22 m⁻² and carries that mean (measured
  2026-11-23 at K2). The rule was fixed before the measurement and the shot was
  not dropped (ruling B28), so this input spends 15 more values per frame than
  a clean one would (about 1% of the 1,470). The input round-trips above include
  it. Cleaning interferometer spikes is left to stage 2's token store.
- **Rule 6 (range too narrow, reported, not fixed here).** Thomson t_e is 2.0%
  above its `unquantized` floor in `ranged` on 3-1 and 3-3. The gate also prints
  six rows of one-sample-per-frame signals (`x_point_r`, `x_point_z` and
  `minor_radius` on 1-1 and 2-1) whose `unquantized` floor is exactly 0, so any
  value clipped by the range shows; the largest is `minor_radius` at 0.0004,
  2% of its target.
- **Against TokaMind's representation.** The task floors of `v1` are below
  TokaMind's floor on 1-2, 1-3, 2-1, 2-2, 2-3, 4-3 and 4-4, and above it on
  3-1, 3-2, 3-3, 4-1, 4-2 and 4-5 (1-1 is exact in both). A floor is what a
  perfect prediction of the tokens scores; K6 above is what the gap costs a real
  model, which is under 1% on the five tasks K6 measured (1-2, 3-2, 3-3, 4-1,
  4-2). On 4-5 the floor
  (0.5133) is 0.79 of the best published score (0.6510), close to the three
  quarters the frame statistics led us to expect (open decision 1, under
  Limits).

### The assignment

One row per signal, from `assignments/<name>.json`. A signal that missed its
target carries the miss in the note.

Measured 2026-11-23 (job K3, `assign --name v1`, from the ladders on 100 `val_select` shots and the input reconstruction errors on the same shots).

| Signal | Codec | K | Domain | Values per frame | Note |
|---|---|---|---|---|---|
| `equilibrium-beta_normal` | `frame_dct` | 1 | linear | 1 | one sample per frame: K = 1 is exact |
| `equilibrium-beta_pol` | `frame_dct` | 1 | linear | 1 | one sample per frame: K = 1 is exact |
| `equilibrium-beta_tor` | `frame_dct` | 1 | linear | 1 | one sample per frame: K = 1 is exact |
| `equilibrium-bphi_rmag` | `frame_dct` | 1 | linear | 1 | one sample per frame: K = 1 is exact |
| `equilibrium-bvac_rmag` | `frame_dct` | 1 | linear | 1 | one sample per frame: K = 1 is exact |
| `equilibrium-elongation` | `frame_dct` | 1 | linear | 1 | one sample per frame: K = 1 is exact |
| `equilibrium-elongation_axis` | `frame_dct` | 1 | linear | 1 | one sample per frame: K = 1 is exact |
| `equilibrium-lcfs_r` | `linear` | 32/4 | signal | 32 | meets 0.0074 with 0.0069; pass-through cost -0.4% |
| `equilibrium-lcfs_z` | `linear` | 32/4 | signal | 32 | meets 0.0074 with 0.0068; pass-through cost -0.7% |
| `equilibrium-magnetic_axis_r` | `frame_dct` | 1 | linear | 1 | one sample per frame: K = 1 is exact |
| `equilibrium-magnetic_axis_z` | `frame_dct` | 1 | linear | 1 | one sample per frame: K = 1 is exact |
| `equilibrium-minor_radius` | `frame_dct` | 1 | linear | 1 | one sample per frame: K = 1 is exact |
| `equilibrium-psi` | `linear` | 16/2 | signal | 16 | meets 0.0166 with 0.0146; pass-through cost -0.2% |
| `equilibrium-q95` | `frame_dct` | 1 | linear | 1 | one sample per frame: K = 1 is exact |
| `equilibrium-triangularity_lower` | `frame_dct` | 1 | linear | 1 | one sample per frame: K = 1 is exact |
| `equilibrium-triangularity_upper` | `frame_dct` | 1 | linear | 1 | one sample per frame: K = 1 is exact |
| `equilibrium-x_point_r` | `frame_dct` | 1 | linear | 2 | one sample per frame: K = 1 is exact |
| `equilibrium-x_point_z` | `frame_dct` | 1 | linear | 2 | one sample per frame: K = 1 is exact |
| `gas_injection-total_injected` | `frame_dct` | 1 | linear | 1 | meets 0.0176 with 0.0079; no pass-through measurement |
| `interferometer-n_e_line` | `frame_dct` | 16 | linear | 16 | misses 0.0176: floor 0.2267; no pass-through measurement |
| `magnetics-b_field_pol_probe_ccbv_field` | `frame_dct` | 4 | linear | 160 | meets 0.0176 with 0.0133; no pass-through measurement |
| `magnetics-b_field_pol_probe_obr_field` | `frame_dct` | 1 | linear | 18 | meets 0.0176 with 0.0159; no pass-through measurement |
| `magnetics-b_field_pol_probe_obv_field` | `frame_dct` | 2 | linear | 36 | meets 0.0176 with 0.0124; no pass-through measurement |
| `magnetics-b_field_pol_probe_omv_voltage` | `band_dct` | 32/16/16/16/16/16/16/16 | band | 432 | magnetics-b_field_tor_probe_cc_field: frame oracle 0.6267, step a floor 0.6267 at k0 32, step b floor 0.6267 at k 16, contrast 0.9348, contrast sets K; magnetics-b_field_pol_probe_omv_voltage: frame oracle 0.4017, step a floor 0.4017 at k0 32, step b floor 0.4030 at k 16, contrast 0.9414, contrast sets K |
| `magnetics-b_field_tor_probe_cc_field` | `band_dct` | 32/16/16/16/16/16/16/16 | band | 432 | magnetics-b_field_tor_probe_cc_field: frame oracle 0.6267, step a floor 0.6267 at k0 32, step b floor 0.6267 at k 16, contrast 0.9348, contrast sets K; magnetics-b_field_pol_probe_omv_voltage: frame oracle 0.4017, step a floor 0.4017 at k0 32, step b floor 0.4030 at k 16, contrast 0.9414, contrast sets K |
| `magnetics-b_field_tor_probe_saddle_voltage` | `frame_dct` | 1 | linear | 12 | meets 0.0176 with 0.0128; no pass-through measurement |
| `magnetics-flux_loop_flux` | `frame_dct` | 2 | linear | 30 | meets 0.0176 with 0.0072; no pass-through measurement |
| `pf_active-coil_current` | `frame_dct` | 1 | linear | 10 | meets 0.0176 with 0.0171; pass-through cost 1.0% |
| `pf_active-coil_voltage` | `frame_dct` | 4 | linear | 16 | meets 0.0176 with 0.0160; no pass-through measurement |
| `pf_active-solenoid_current` | `frame_dct` | 1 | linear | 1 | meets 0.0176 with 0.0140; pass-through cost 15.5% |
| `pulse_schedule-i_plasma` | `frame_dct` | 2 | linear | 2 | meets 0.0176 with 0.0099; no pass-through measurement |
| `pulse_schedule-n_e_line` | `frame_dct` | 1 | linear | 1 | meets 0.0176 with 0.0100; no pass-through measurement |
| `soft_x_rays-horizontal_cam_lower` | `frame_dct` | 2 | linear | 36 | meets 0.0424 with 0.0422; pass-through cost -0.0% |
| `soft_x_rays-horizontal_cam_upper` | `frame_dct` | 4 | linear | 72 | meets 0.0424 with 0.0416; pass-through cost -0.1% |
| `spectrometer_visible-filter_spectrometer_dalpha_voltage` | `frame_dct` | 2 | linear | 6 | misses 0.0504: floor 0.1430; pass-through cost 0.2% |
| `summary-ip` | `frame_dct` | 4 | linear | 4 | meets 0.0176 with 0.0147; pass-through cost 0.0% |
| `summary-power_nbi` | `frame_dct` | 8 | linear | 8 | meets 0.0176 with 0.0147; no pass-through measurement |
| `thomson_scattering-n_e` | `linear` | 32/4 | signal | 32 | meets 0.0429 with 0.0391; pass-through cost 0.3% |
| `thomson_scattering-t_e` | `linear` | 48/4 | signal | 48 | meets 0.0429 with 0.0398; pass-through cost 0.1% |

Total: 1470 values per 5 ms frame.

**The assignment `v1b` (ruling C29).** The solenoid current and the coil currents of task 2-1 are at K 1 in `v1`,
where passing the direct model's prediction through the tokenizer costs 15.5% on the solenoid current
(see [Results](#results) above, which left to the owner whether rule 2 should also bound the pass-through cost).
Ruling C29 (2026-11-23; `assignments/v1b.json` written 06:29:53 UTC and confirmed 06:30 to 06:31, ledger entry
06:38) applied the stated default of that owner item, since the owner had not said no
and the numbers it needs were already measured: K 2 for both currents, the "2 to 4 values per frame" of the item.
That is 11 more values per frame (1 for the solenoid current, 10 for the coil currents), so 1,481 values per frame
instead of 1,470, in `assignments/v1b.json`; `v1` is untouched and only these two settings differ. Measured
2026-11-23 on all 520 `val_report` shots of task 2-1 (the `s90k` model, `ranged`, `pass / raw - 1` on the same
shots): the solenoid current costs +16.15% with `v1` and -0.34% with `v1b`, the coil currents +0.71% and -0.27%,
the task's mean +0.17% and -0.01%. The input round-trip on `val_report`, `ranged`, is in
[The input round-trip of `v1`, `v1b` and `v2c`](#the-input-round-trip-of-v1-v1b-and-v2c): task 1-2, the task the plan reads it on, goes from
+1.21% with `v1` to +0.24% with `v1b`, and no task is above
+0.24%. The gate on `v1b` (2026-11-23, 07:40 UTC, all
three read-outs on `val_report`) says `57 rows, 0 not scored, 0 block`, as for `v1`, and the two tables differ in
two rows only, the two currents of task 2-1: `unquantized` floor 0.0161 to 0.0075 for the coil currents and 0.0133
to 0.0021 for the solenoid current. If the owner says no, `v1b` is deleted (one file and its scores) and stage 2
starts on `v1`; the cost of keeping it is 11 values per frame on two signals (0.75%).

### The gate

One row per (task, output signal) pair, the columns of `gate_rows`. `target` is
a fifth of the best published score of the task (empty for the spectrograms),
the three floors and `own_mean` are on `val_report`, and `bins_used_min`,
`worst_channel`, `worst_channel_error` and `contrast` are measured from the
cache on 200 `val_report` shots.

Measured 2026-11-23 (job K7, `gate --assignment v1.json`, 04:59 UTC): the floors are the stored K5 scores on all `val_report` shots of each task (351 to 553 shots), the cache-side columns are from 200 `val_report` shots. Last line: `57 rows, 0 not scored, 0 block`.

| task | signal | codec | k | domain | values_per_frame | target | unquantized | ranged | nearest | own_mean | bins_used_min | worst_channel | worst_channel_error | contrast | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| task_1-1 | equilibrium-elongation | frame_dct | 1 | linear | 1 | 0.0188 | 0.0000 | 0.0000 | 0.0271 | 0.0000 | 0.8906 | 0 | 0.0000 | no data | pass |
| task_1-1 | equilibrium-elongation_axis | frame_dct | 1 | linear | 1 | 0.0188 | 0.0000 | 0.0000 | 0.0323 | 0.0000 | 0.7969 | 0 | 0.0000 | no data | pass |
| task_1-1 | equilibrium-triangularity_upper | frame_dct | 1 | linear | 1 | 0.0188 | 0.0000 | 0.0000 | 0.0268 | 0.0000 | 0.8438 | 0 | 0.0000 | no data | pass |
| task_1-1 | equilibrium-triangularity_lower | frame_dct | 1 | linear | 1 | 0.0188 | 0.0000 | 0.0000 | 0.0318 | 0.0000 | 0.7188 | 0 | 0.0000 | no data | pass |
| task_1-1 | equilibrium-x_point_r | frame_dct | 1 | linear | 2 | 0.0188 | 0.0000 | 0.0000 | 0.0946 | 0.0000 | 0.4062 | 0 | 0.0000 | no data | pass |
| task_1-1 | equilibrium-x_point_z | frame_dct | 1 | linear | 2 | 0.0188 | 0.0000 | 0.0000 | 0.0218 | 0.0000 | 0.1719 | 0 | 0.0000 | no data | pass |
| task_1-1 | equilibrium-minor_radius | frame_dct | 1 | linear | 1 | 0.0188 | 0.0000 | 0.0003 | 0.0244 | 0.0000 | 0.8438 | 0 | 0.0000 | no data | pass |
| task_1-1 | equilibrium-magnetic_axis_r | frame_dct | 1 | linear | 1 | 0.0188 | 0.0000 | 0.0000 | 0.0220 | 0.0000 | 0.8750 | 0 | 0.0000 | no data | pass |
| task_1-1 | equilibrium-magnetic_axis_z | frame_dct | 1 | linear | 1 | 0.0188 | 0.0000 | 0.0000 | 0.0289 | 0.0000 | 0.9375 | 0 | 0.0000 | no data | pass |
| task_1-1 | equilibrium-q95 | frame_dct | 1 | linear | 1 | 0.0188 | 0.0000 | 0.0000 | 0.0287 | 0.0000 | 0.7969 | 0 | 0.0000 | no data | pass |
| task_1-1 | equilibrium-beta_tor | frame_dct | 1 | linear | 1 | 0.0188 | 0.0000 | 0.0000 | 0.0538 | 0.0000 | 0.6719 | 0 | 0.0000 | no data | pass |
| task_1-1 | equilibrium-beta_pol | frame_dct | 1 | linear | 1 | 0.0188 | 0.0000 | 0.0000 | 0.0467 | 0.0000 | 0.7344 | 0 | 0.0000 | no data | pass |
| task_1-1 | equilibrium-beta_normal | frame_dct | 1 | linear | 1 | 0.0188 | 0.0000 | 0.0000 | 0.0510 | 0.0000 | 0.7188 | 0 | 0.0000 | no data | pass |
| task_1-1 | equilibrium-bvac_rmag | frame_dct | 1 | linear | 1 | 0.0188 | 0.0000 | 0.0000 | 0.0228 | 0.0000 | 0.9219 | 0 | 0.0000 | no data | pass |
| task_1-1 | equilibrium-bphi_rmag | frame_dct | 1 | linear | 1 | 0.0188 | 0.0000 | 0.0000 | 0.1252 | 0.0000 | 0.3438 | 0 | 0.0000 | no data | pass |
| task_1-2 | equilibrium-lcfs_r | linear | 32 | signal | 32 | 0.0082 | 0.0068 | 0.0068 | 0.0101 | 0.2274 | 0.7031 | 142 | 0.0104 | no data | pass |
| task_1-2 | equilibrium-lcfs_z | linear | 32 | signal | 32 | 0.0082 | 0.0068 | 0.0068 | 0.0093 | 0.2127 | 0.6719 | 156 | 0.0104 | no data | pass |
| task_1-3 | equilibrium-psi | linear | 16 | signal | 16 | 0.0231 | 0.0130 | 0.0130 | 0.0355 | 0.6829 | 0.6250 | 3789 | 0.0482 | no data | pass |
| task_2-1 | pf_active-coil_current | frame_dct | 1 | linear | 10 | 0.0176 | 0.0161 | 0.0161 | 0.0259 | 0.0161 | 0.8438 | 5 | 0.1334 | no data | pass |
| task_2-1 | pf_active-solenoid_current | frame_dct | 1 | linear | 1 | 0.0176 | 0.0133 | 0.0133 | 0.0232 | 0.0133 | 0.9219 | 0 | 0.0176 | no data | pass |
| task_2-1 | summary-ip | frame_dct | 4 | linear | 4 | 0.0176 | 0.0139 | 0.0139 | 0.0290 | 0.0715 | 0.7344 | 0 | 0.0121 | no data | pass |
| task_2-1 | equilibrium-elongation | frame_dct | 1 | linear | 1 | 0.0176 | 0.0000 | 0.0000 | 0.0271 | 0.0000 | 0.8906 | 0 | 0.0000 | no data | pass |
| task_2-1 | equilibrium-elongation_axis | frame_dct | 1 | linear | 1 | 0.0176 | 0.0000 | 0.0000 | 0.0325 | 0.0000 | 0.7969 | 0 | 0.0000 | no data | pass |
| task_2-1 | equilibrium-triangularity_upper | frame_dct | 1 | linear | 1 | 0.0176 | 0.0000 | 0.0000 | 0.0270 | 0.0000 | 0.8438 | 0 | 0.0000 | no data | pass |
| task_2-1 | equilibrium-triangularity_lower | frame_dct | 1 | linear | 1 | 0.0176 | 0.0000 | 0.0000 | 0.0322 | 0.0000 | 0.7188 | 0 | 0.0000 | no data | pass |
| task_2-1 | equilibrium-x_point_r | frame_dct | 1 | linear | 2 | 0.0176 | 0.0000 | 0.0000 | 0.0946 | 0.0000 | 0.4062 | 0 | 0.0000 | no data | pass |
| task_2-1 | equilibrium-x_point_z | frame_dct | 1 | linear | 2 | 0.0176 | 0.0000 | 0.0000 | 0.0218 | 0.0000 | 0.1719 | 0 | 0.0000 | no data | pass |
| task_2-1 | equilibrium-minor_radius | frame_dct | 1 | linear | 1 | 0.0176 | 0.0000 | 0.0004 | 0.0249 | 0.0000 | 0.8438 | 0 | 0.0000 | no data | pass |
| task_2-1 | equilibrium-magnetic_axis_r | frame_dct | 1 | linear | 1 | 0.0176 | 0.0000 | 0.0000 | 0.0224 | 0.0000 | 0.8750 | 0 | 0.0000 | no data | pass |
| task_2-1 | equilibrium-magnetic_axis_z | frame_dct | 1 | linear | 1 | 0.0176 | 0.0000 | 0.0000 | 0.0291 | 0.0000 | 0.9375 | 0 | 0.0000 | no data | pass |
| task_2-1 | equilibrium-q95 | frame_dct | 1 | linear | 1 | 0.0176 | 0.0000 | 0.0000 | 0.0297 | 0.0000 | 0.7969 | 0 | 0.0000 | no data | pass |
| task_2-1 | equilibrium-beta_tor | frame_dct | 1 | linear | 1 | 0.0176 | 0.0000 | 0.0000 | 0.0539 | 0.0000 | 0.6719 | 0 | 0.0000 | no data | pass |
| task_2-1 | equilibrium-beta_pol | frame_dct | 1 | linear | 1 | 0.0176 | 0.0000 | 0.0000 | 0.0476 | 0.0000 | 0.7344 | 0 | 0.0000 | no data | pass |
| task_2-1 | equilibrium-beta_normal | frame_dct | 1 | linear | 1 | 0.0176 | 0.0000 | 0.0000 | 0.0514 | 0.0000 | 0.7188 | 0 | 0.0000 | no data | pass |
| task_2-1 | equilibrium-bvac_rmag | frame_dct | 1 | linear | 1 | 0.0176 | 0.0000 | 0.0000 | 0.0233 | 0.0000 | 0.9219 | 0 | 0.0000 | no data | pass |
| task_2-1 | equilibrium-bphi_rmag | frame_dct | 1 | linear | 1 | 0.0176 | 0.0000 | 0.0000 | 0.1267 | 0.0000 | 0.3438 | 0 | 0.0000 | no data | pass |
| task_2-2 | equilibrium-lcfs_r | linear | 32 | signal | 32 | 0.0074 | 0.0069 | 0.0069 | 0.0104 | 0.2447 | 0.7031 | 142 | 0.0104 | no data | pass |
| task_2-2 | equilibrium-lcfs_z | linear | 32 | signal | 32 | 0.0074 | 0.0068 | 0.0068 | 0.0094 | 0.2196 | 0.6719 | 156 | 0.0104 | no data | pass |
| task_2-3 | equilibrium-psi | linear | 16 | signal | 16 | 0.0166 | 0.0126 | 0.0126 | 0.0359 | 0.6990 | 0.6250 | 3789 | 0.0482 | no data | pass |
| task_3-1 | thomson_scattering-t_e | linear | 48 | signal | 48 | 0.0662 | 0.0446 | 0.0455 | 0.0648 | 0.9144 | 0.5781 | 5 | 0.2371 | no data | pass |
| task_3-1 | thomson_scattering-n_e | linear | 32 | signal | 32 | 0.0662 | 0.0407 | 0.0407 | 0.0631 | 0.8931 | 0.3281 | 119 | 0.1859 | no data | pass |
| task_3-2 | spectrometer_visible-filter_spectrometer_dalpha_voltage | frame_dct | 2 | linear | 6 | 0.0504 | 0.1380 | 0.1380 | 0.1407 | 0.1903 | 0.6250 | 2 | 0.1528 | no data | pass, misses the fidelity target (0.1380 against 0.0504) |
| task_3-2 | soft_x_rays-horizontal_cam_lower | frame_dct | 2 | linear | 36 | 0.0504 | 0.0396 | 0.0396 | 0.0543 | 0.0455 | 0.5156 | 4 | 0.1406 | no data | pass |
| task_3-2 | soft_x_rays-horizontal_cam_upper | frame_dct | 4 | linear | 72 | 0.0504 | 0.0359 | 0.0359 | 0.0405 | 0.0544 | 0.6250 | 0 | 0.1503 | no data | pass |
| task_3-3 | thomson_scattering-t_e | linear | 48 | signal | 48 | 0.0429 | 0.0428 | 0.0436 | 0.0608 | 0.7020 | 0.5781 | 5 | 0.2371 | no data | pass, misses the fidelity target (0.0436 against 0.0429) |
| task_3-3 | thomson_scattering-n_e | linear | 32 | signal | 32 | 0.0429 | 0.0444 | 0.0444 | 0.0647 | 0.7981 | 0.3281 | 119 | 0.1859 | no data | pass, misses the fidelity target (0.0444 against 0.0429) |
| task_3-3 | equilibrium-beta_tor | frame_dct | 1 | linear | 1 | 0.0429 | 0.0000 | 0.0000 | 0.0540 | 0.0000 | 0.6719 | 0 | 0.0000 | no data | pass |
| task_3-3 | equilibrium-beta_pol | frame_dct | 1 | linear | 1 | 0.0429 | 0.0000 | 0.0000 | 0.0468 | 0.0000 | 0.7344 | 0 | 0.0000 | no data | pass |
| task_3-3 | equilibrium-beta_normal | frame_dct | 1 | linear | 1 | 0.0429 | 0.0000 | 0.0000 | 0.0510 | 0.0000 | 0.7188 | 0 | 0.0000 | no data | pass |
| task_4-1 | soft_x_rays-horizontal_cam_lower | frame_dct | 2 | linear | 36 | 0.0424 | 0.0479 | 0.0479 | 0.0628 | 0.0548 | 0.5156 | 4 | 0.1406 | no data | pass, misses the fidelity target (0.0479 against 0.0424) |
| task_4-1 | soft_x_rays-horizontal_cam_upper | frame_dct | 4 | linear | 72 | 0.0424 | 0.0415 | 0.0415 | 0.0461 | 0.0583 | 0.6250 | 0 | 0.1503 | no data | pass |
| task_4-2 | soft_x_rays-horizontal_cam_lower | frame_dct | 2 | linear | 36 | 0.0428 | 0.0479 | 0.0479 | 0.0628 | 0.0548 | 0.5156 | 4 | 0.1406 | no data | pass, misses the fidelity target (0.0479 against 0.0428) |
| task_4-2 | soft_x_rays-horizontal_cam_upper | frame_dct | 4 | linear | 72 | 0.0428 | 0.0415 | 0.0415 | 0.0461 | 0.0583 | 0.6250 | 0 | 0.1503 | no data | pass |
| task_4-3 | equilibrium-magnetic_axis_z | frame_dct | 1 | linear | 1 | 0.0138 | 0.0000 | 0.0000 | 0.0283 | 0.0000 | 0.9375 | 0 | 0.0000 | no data | pass |
| task_4-4 | summary-ip | frame_dct | 4 | linear | 4 | 0.0623 | 0.0148 | 0.0148 | 0.0297 | 0.0757 | 0.7344 | 0 | 0.0121 | no data | pass |
| task_4-5 | magnetics-b_field_tor_probe_cc_field | band_dct | 32/16/16/16/16/16/16/16 | band | 432 | no target | 0.5314 | 0.5314 | 0.5326 | 0.6749 | 0.4375 | 256 | 12.0899 | 0.9338 | pass (no fidelity target) |
| task_4-5 | magnetics-b_field_pol_probe_omv_voltage | band_dct | 32/16/16/16/16/16/16/16 | band | 432 | no target | 0.4952 | 0.4953 | 0.4964 | 0.6598 | 0.3750 | 256 | 5.2086 | 0.9417 | pass (no fidelity target) |

### The floors per task

The 14 tasks, scored on all `val_report` shots with the chosen assignment.

Measured 2026-11-23 (job K5, 56 passes, all `val_report` shots of each task; the shot count is in the last column). The own-mean reference is the `ref` assignment's `unquantized` floor. The mean baseline and TokaMind's floor are stage 0's scores on the same split.

| Task | unquantized | ranged | nearest | own-mean reference | A fifth of the best published score | Mean baseline | TokaMind's floor | Shots |
|---|---|---|---|---|---|---|---|---|
| task_1-1 | 0.0000 | 0.0000 | 0.0425 | 0.0000 | 0.0188 | 0.7599 | 0.0000 | 520 |
| task_1-2 | 0.0068 | 0.0068 | 0.0097 | 0.2200 | 0.0082 | 1.0005 | 0.0239 | 525 |
| task_1-3 | 0.0130 | 0.0130 | 0.0355 | 0.6829 | 0.0231 | 0.9312 | 0.0294 | 527 |
| task_2-1 | 0.0024 | 0.0024 | 0.0401 | 0.0056 | 0.0176 | 0.8168 | 0.0037 | 520 |
| task_2-2 | 0.0068 | 0.0068 | 0.0099 | 0.2321 | 0.0074 | 0.9935 | 0.0217 | 525 |
| task_2-3 | 0.0126 | 0.0126 | 0.0359 | 0.6990 | 0.0166 | 0.9249 | 0.0282 | 527 |
| task_3-1 | 0.0426 | 0.0431 | 0.0640 | 0.9037 | 0.0662 | 0.9940 | 0.0284 | 450 |
| task_3-2 | 0.0712 | 0.0712 | 0.0785 | 0.0967 | 0.0504 | 0.6197 | 0.0288 | 532 |
| task_3-3 | 0.0174 | 0.0176 | 0.0554 | 0.3000 | 0.0429 | 0.8412 | 0.0091 | 445 |
| task_4-1 | 0.0447 | 0.0447 | 0.0545 | 0.0566 | 0.0424 | 0.4961 | 0.0353 | 531 |
| task_4-2 | 0.0447 | 0.0447 | 0.0545 | 0.0566 | 0.0428 | 0.4961 | 0.0353 | 531 |
| task_4-3 | 0.0000 | 0.0000 | 0.0283 | 0.0000 | 0.0138 | 0.7591 | 0.0162 | 510 |
| task_4-4 | 0.0148 | 0.0148 | 0.0297 | 0.0757 | 0.0623 | 1.0364 | 0.0351 | 553 |
| task_4-5 | 0.5133 | 0.5133 | 0.5145 | 0.6673 | — | 0.6852 | 0.3605 | 351 |

### The pass-through and input round-trip costs

Pass-through (rule 3 and K6): the prediction of a direct model sent through the
tokenizer, per task and signal. Input round-trip (rule 4 and K4, K6): the
model's inputs and actuators sent through the tokenizer, per task. The added
error is `sqrt(max(inputs**2 - raw**2, 0))` over the best published score of
the task, and it is filled for the input round-trip only.

Measured 2026-11-23 (job K4 on 100 `val_select` shots, `unquantized`; job K6 on all `val_report` shots of each task, `ranged`). The direct model of each task is the variant the headroom table shows for it.

| Measure | Task | Signal | Split | Shots | Raw score | Score through the tokenizer | `through / raw - 1` | Added error over the best published score |
|---|---|---|---|---|---|---|---|---|
| input round-trip (K4, `unquantized`) | task_1-1 (`s90k`) | all | val_select | 88 | 0.1250 | 0.1253 | +0.24% | 0.092 |
| input round-trip (K4, `unquantized`) | task_1-2 (`lr1e4`) | all | val_select | 92 | 0.0369 | 0.0375 | +1.87% | 0.174 |
| input round-trip (K4, `unquantized`) | task_2-1 (`s90k`) | all | val_select | 89 | 0.1336 | 0.1340 | +0.32% | 0.121 |
| input round-trip (K4, `unquantized`) | task_4-1 (`h050b256`) | all | val_select | 95 | 0.1621 | 0.1625 | +0.27% | 0.056 |
| input round-trip (K6, `ranged`) | task_1-1 (`s90k`) | all | val_report | 520 | 0.1235 | 0.1237 | +0.22% | 0.086 |
| input round-trip (K6, `ranged`) | task_1-2 (`lr1e4`) | all | val_report | 525 | 0.0362 | 0.0366 | +1.21% | 0.137 |
| input round-trip (K6, `ranged`) | task_2-1 (`s90k`) | all | val_report | 520 | 0.1347 | 0.1353 | +0.45% | 0.145 |
| input round-trip (K6, `ranged`) | task_4-1 (`h050b256`) | all | val_report | 531 | 0.1727 | 0.1716 | -0.63% | 0.000 |
| pass-through (K6, `ranged`) | task_3-2 (`base`) | `soft_x_rays-horizontal_cam_lower` | val_report | 532 | 0.2122 | 0.2121 | -0.04% | — |
| pass-through (K6, `ranged`) | task_3-2 (`base`) | `soft_x_rays-horizontal_cam_upper` | val_report | 532 | 0.1542 | 0.1540 | -0.09% | — |
| pass-through (K6, `ranged`) | task_3-2 (`base`) | `spectrometer_visible-filter_spectrometer_dalpha_voltage` | val_report | 532 | 0.3263 | 0.3270 | +0.22% | — |
| pass-through (K6, `ranged`) | task_3-3 (`base`) | `equilibrium-beta_normal` | val_report | 445 | 0.1893 | 0.1893 | +0.00% | — |
| pass-through (K6, `ranged`) | task_3-3 (`base`) | `equilibrium-beta_pol` | val_report | 445 | 0.1813 | 0.1813 | +0.00% | — |
| pass-through (K6, `ranged`) | task_3-3 (`base`) | `equilibrium-beta_tor` | val_report | 445 | 0.1909 | 0.1909 | +0.00% | — |
| pass-through (K6, `ranged`) | task_3-3 (`base`) | `thomson_scattering-n_e` | val_report | 445 | 0.2688 | 0.2695 | +0.24% | — |
| pass-through (K6, `ranged`) | task_3-3 (`base`) | `thomson_scattering-t_e` | val_report | 445 | 0.2665 | 0.2666 | +0.04% | — |
| pass-through (K6, `ranged`) | task_4-1 (`h050b256`) | `soft_x_rays-horizontal_cam_lower` | val_report | 531 | 0.1652 | 0.1650 | -0.09% | — |
| pass-through (K6, `ranged`) | task_4-1 (`h050b256`) | `soft_x_rays-horizontal_cam_upper` | val_report | 531 | 0.1802 | 0.1801 | -0.05% | — |
| pass-through (K6, `ranged`) | task_4-2 (`h050b256`) | `soft_x_rays-horizontal_cam_lower` | val_report | 531 | 0.1656 | 0.1654 | -0.09% | — |
| pass-through (K6, `ranged`) | task_4-2 (`h050b256`) | `soft_x_rays-horizontal_cam_upper` | val_report | 531 | 0.1813 | 0.1812 | -0.06% | — |
| pass-through (K6, `ranged`) | task_1-2 (`lr1e4`) | `equilibrium-lcfs_r` | val_report | 525 | 0.0427 | 0.0425 | -0.37% | — |
| pass-through (K6, `ranged`) | task_1-2 (`lr1e4`) | `equilibrium-lcfs_z` | val_report | 525 | 0.0297 | 0.0295 | -0.71% | — |

## Learned tokenizers (stage 1c)

Stage 1c's learned tokenizers are built and measured. Jobs M1 to M10 of
[How to run stage 1c](#how-to-run-stage-1c) ran on 2026-11-23, and ruling C45 added one training and one comparison (08:54 to 08:57 UTC). The bases and
networks were fitted on the 2,000
training shots of the fit and compared on 100 `val_select` shots (62 to 100 scored per signal), and where a rule
asks for it the comparison was confirmed on all 557 `val_report` shots. The six tables under
[Results of the comparison](#results-of-the-comparison) hold every number. Each error is a tokenizer floor in
TokaMark units, the score of a perfect prediction of the tokens, and not an IGNITE score. What the tables say:

- **No network tokenizer beats its linear counterpart.** Of the 13 networks, four collapsed under rule 7
  (ruling C33), also after their 40,000-step runs: `net.series` on `interferometer-n_e_line` and the three Mirnov
  per-stream arms. The other nine lost the comparison for the slot each competes for (ruling C41) on all 15
  signal rows of their five comparisons (`val_select`, 62 to 100 shots; 13 rows clearly, the two rows of the Mirnov
  concatenated network by "b beats a"). Their nine retries at learning rate 1e-4 lost the same 15 rows with the
  same words (rule 8), so no seeds 1 and 2 were trained. One retry, `net.flux.s0.lr1e4`, ended at 20,000 steps
  with a plateau of "no" and was extended once under ruling C45: `net.flux.s0.lr1e4_long` has a best `val_mse` of
  0.01282 against 0.01360, and `c-net0rl` still loses to `c-pca` on the flux (0.01531 against 0.002548; gap of b
  against a 0.834, 0.823 to 0.844), so every verdict is unchanged. The checkpoints and the comparison use
  the same 100 `val_select` shots, which favours the networks (inferred, see [The trainings](#the-trainings)), and
  they still lose.
- **The linear concatenated arm wins on the four magnetics arrays (rule 6).** `c-pca` clearly beats the best shared
  arm `s-klt` on `ccbv`, `obr`, `obv` and `flux`, with gaps 0.643, 0.475, 0.635 and 0.791 on `val_select`
  (100 shots) and 0.633, 0.474, 0.631 and 0.788 on `val_report` (557 shots); the low end of every 95% interval is
  above 0.44 on both splits. Its cost is a missing channel: dropping one raises the error of the others by a
  factor of 3.510 on `flux` (1.069 on `ccbv`, 1.009 on `obr`, 1.000 on `obv`; `val_report`), where the per-channel
  arms are unaffected by construction.
- **Own beats shared in linear form.** On `val_report` (557 shots) `o-klt` has a lower error than `s-klt` on all
  four arrays (gaps 0.182 on `ccbv`, 0.008 on `obr`, 0.335 on `obv`, 0.426 on `flux`), so the owner's point that
  one encoder fed more data does better does not hold in linear form on the magnetics. On the two Mirnov signals
  the shared (`b-klt`) and own (`bo-klt`) arms tie (`val_select`, 62 shots), which ruling C17 expected.
- **Nothing replaces the cosine transform (rule 4) or the linear baselines (rule 5).** `s-klt` loses to `v1` on 13
  of the 17 series signals (clearly on 7), ties on 3 and beats it on one, the solenoid current at `v1`'s K 1 (gap
  0.028, 89 shots). At `v1b`'s K 2 (ruling C42) `v1b` clearly beats `s-klt.b` on both currents (solenoid current
  0.002212 against 0.005913, coil currents 0.007586 against 0.009755; `val_select`, 89 shots). The profile and map
  networks lose clearly to `v1` at both learning rates (76 to 92 shots), and the Mirnov networks collapsed or lost
  to `bc-pca`.
- **`v2` equals `v1b`, and `v2c` exists.** No learned arm replaced anything, so `v2` was not built as a separate
  file. `v2c` is `v1b` with `pca.ccbv`, `pca.obr`, `pca.obv` and `pca.flux` at `v1b`'s values per frame (160, 18, 36
  and 30). Its input round-trip on `val_report` (`ranged`, `inputs / raw - 1` on the same shots) is
  +0.08% on task 1-1 (520 shots), +0.11% on 1-2 (525),
  +0.01% on 2-1 (520) and -0.63% on 4-1 (531), against +0.13%,
  +0.24%, +0.03% and -0.63% for `v1b`
  (the table under [The input round-trip of `v1`, `v1b` and `v2c`](#the-input-round-trip-of-v1-v1b-and-v2c)).
- **The token count is reported and not adopted (open decision 2).** On `val_report` (557 shots) `c-pca` reaches
  0.0176 with 40 values per frame on `ccbv` and 12 on `flux`, where `s-klt` needs 160 and 30; on `obr` (18) and
  `obv` (36) it needs what `o-klt` needs, and `s-klt` needs 72 on `obv`. `v2c` keeps `v1b`'s values per frame.
- **Open decision 1 now applies, taken as the plan's author recommends.** Stage 2's smallest pretraining runs on
  both `v2` and `v2c` and decides on the scores of tasks 1-1 and 1-2; nothing is adopted here. Stage 2 starts on
  `v1b` (ruling C29, see [Rules of the comparison](#rules-of-the-comparison)).

The numbers of the design (sizes, probabilities, thresholds) are the plan's choices, and the measured facts it cites
(rulings C15, C17 and C26) carry their date, their split and their number of shots. The plan is
`docs/superpowers/plans/2026-11-22-ignite-mast-stage1c-learned-tokenizers.md`.

**What a recoder is.** A learned tokenizer of this stage is a recoder placed
behind one of the three fixed codecs above, run at its top rung: the first 16
cosine values of a frame for a series channel, the 32 values of a band for a
Mirnov spectrogram, and the first 96 principal components for a profile or the
flux map. Those are the three front ends with 16, 32 and 96 values, and they are
not learned. The recoder turns a front end's values into an ordered list of token
values and back. It is linear (a principal-component basis of the front values,
fitted on the CPU in minutes; the arms are named `klt` when it takes one stream
at a time and `pca` when it takes all of them at once) or a small network (an
encoder, values bounded to (-1, 1), a quantizer that is skipped for a random half
of the training examples, and a decoder). A stream is one channel of a series
signal, one (probe, band) pair of a spectrogram, or a whole profile or map. The
shared tokenizer takes one stream at a time and is told which stream it has; the
control takes all the streams of a signal at once. A `RecodedCodec` wraps the front
end and the recoder as a codec of this stage, so the floor instrument above scores
it unchanged. Its examples are the front values of the 2,000 training shots of
the fit (`store/v1/train`) and of 100 `val_select` shots spread evenly
(`store/v1/val`), each stream standardized.

**The two orderings of the values.** The values of a token are ordered by
precision: each is bounded to (-1, 1) and is read at 8 levels, at 64 levels (which
contain the 8, and are the 64 bins of this stage's grid) or unrounded, and each
training example draws one of the three with probabilities 1/4, 1/4 and 1/2. They
are also ordered by length: the list of values is cut at a random length during
training and the rest set to zero, so one trained network gives every K up to its
size and can be set beside the fixed transform at each K. A prefix of the values
is therefore a coarser version of the same example.

**The identity features.** A network that serves many streams is told which one it
has by 18 features built from the registry row of the stream (`identity.py`): the
family (slow, high rate, profile, map), the logarithm of the sampling interval,
whether the cache compresses the channel, the unit (T, V, A, Wb, W or none of
them), the channel's place in its array, whether R and Z are known and their
values, whether it is a band codec and the band's index. A learned vector per
stream sits beside them. The probes have no position in cache v1: measured
2026-11-22 from the registry of cache v1 (6,521 rows), R and Z are filled only for
the flux-map points (4,225 rows) and the two Thomson signals (120 rows each). The
magnetic probes, flux loops, saddle coils, soft X-ray chords and D-alpha channels
have a name, a unit and an order in their array and no position, so for them the
table part of the features says only the family, the sampling interval, the unit,
the compression and the order, and the rest sits in the learned vector, which does
not transfer to another machine. Open decision 3 of the plan, taken as its author
recommends: this is accepted for stage 1 and stated here, and positions from the
FAIR-MAST geometry come with the next cache version, when the shared tokenizers
are retrained.

### Shared against specialized

Each comparison has three arms, because shared against concatenated changes two
things at once: how much data the tokenizer sees, and whether it sees the channels
together.

- **Shared:** a per-stream tokenizer fitted on every stream of the family (`klt.series`,
  `net.series` for the series signals; `klt.band`, `net.band` for the two Mirnov
  spectrograms).
- **Own:** a per-stream tokenizer fitted on the signal's own diagnostic only
  (`klt.mag`, `net.mag` on the four magnetics signals; `klt.band.cc`,
  `klt.band.omv` and their network arms on one Mirnov signal each).
- **Concatenated:** the control built the old way, one tokenizer per signal that
  takes all its streams in one input and has no stream identity (`pca.<array>` and
  `net.<array>` for the four magnetics arrays; `pca.band.<signal>` and
  `net.bandcat.<signal>` for the Mirnov signals). It gets the same number of
  values per frame as the arm it is compared with.

What each pair measures: shared against own is the owner's point, that one encoder
fed more data does better. Own against concatenated is what the design calls the
cost, the structure across channels that a per-channel tokenizer cannot use.
Shared against concatenated is the design's own question: "If a specialized one
clearly beats the shared one, that signal uses the specialized tokenizer and the
result is recorded." Each pair is also set beside the fixed
transform of the stage-1b assignment `v1` where the rules below say so. The
profiles and the flux map have one tokenizer per signal already, so for them
there is one learned arm against the linear baseline.

The 24 arms, as `train_tokenizer.py arms` lists them (listed 2026-11-23, 11 linear
and 13 network). "Tokens times values" is per stream for an arm that takes one
stream at a time and per frame for a concatenated arm. The linear concatenated
arms keep all the front values of their streams (16 per stream, 32 for the Mirnov
signals); the network ones keep 8 per stream, except the Mirnov control (below).

| Arm | Signals | Takes | Kind | Tokens times values |
|---|---|---|---|---|
| `klt.series` | the 17 series signals with more than one sample per frame (163 channels) | one stream | linear | 1 x 16 |
| `klt.mag` | the 4 magnetics signals (91 channels) | one stream | linear | 1 x 16 |
| `pca.ccbv` | `ccbv` (40 channels) | all streams | linear | 160 x 4 |
| `pca.obr` | `obr` (18 channels) | all streams | linear | 48 x 6 |
| `pca.obv` | `obv` (18 channels) | all streams | linear | 48 x 6 |
| `pca.flux` | `flux` (15 channels) | all streams | linear | 40 x 6 |
| `klt.band` | both Mirnov signals (48 streams) | one stream | linear | 1 x 32 |
| `klt.band.cc` | `cc` (24 streams) | one stream | linear | 1 x 32 |
| `klt.band.omv` | `omv` (24 streams) | one stream | linear | 1 x 32 |
| `pca.band.cc` | `cc` (24 streams) | all streams | linear | 192 x 4 |
| `pca.band.omv` | `omv` (24 streams) | all streams | linear | 192 x 4 |
| `net.series` | the 17 series signals (163 channels) | one stream | network | 1 x 16 |
| `net.mag` | the 4 magnetics signals (91 channels) | one stream | network | 1 x 16 |
| `net.ccbv` | `ccbv` (40 channels) | all streams | network | 40 x 8 |
| `net.obr` | `obr` (18 channels) | all streams | network | 24 x 6 |
| `net.obv` | `obv` (18 channels) | all streams | network | 24 x 6 |
| `net.flux` | `flux` (15 channels) | all streams | network | 20 x 6 |
| `net.band` | both Mirnov signals (48 streams) | one stream | network | 1 x 32 |
| `net.band.cc` | `cc` (24 streams) | one stream | network | 1 x 32 |
| `net.band.omv` | `omv` (24 streams) | one stream | network | 1 x 32 |
| `net.bandcat.cc` | `cc` (24 streams) | all streams | network | 54 x 8 |
| `net.bandcat.omv` | `omv` (24 streams) | all streams | network | 54 x 8 |
| `net.profile` | the 4 profiles (`lcfs_r`, `lcfs_z`, Thomson `t_e` and `n_e`) | one stream | network | 24 x 4 |
| `net.map` | the flux map (`equilibrium-psi`) | one stream | network | 48 x 2 |

The short names are `ccbv`, `obr` and `obv` for the three poloidal probe arrays
(`magnetics-b_field_pol_probe_<name>_field`), `flux` for
`magnetics-flux_loop_flux`, `cc` for `magnetics-b_field_tor_probe_cc_field` and
`omv` for `magnetics-b_field_pol_probe_omv_voltage`. The last line of the listing
is "24 arms". The arms of the Mirnov signals have two controls on purpose:
`pca.band.*` is the linear one and `net.bandcat.*` the network one, and the second
holds 54 tokens of 8 (432 values) where the first holds 192 tokens of 4 (768
values, of which an assignment keeps 432, 108 tokens).

### Rules of the comparison

Stated before any number existed. The nine rules below are copied from the plan's
section "Which number decides what", word for word, including its stage-0 numbers
as of 2026-11-22 (the stage-0 page holds the current ones). Each was fixed before
the measurement, 2026-11-22, and none is changed here to fit a result. They add to
the seven rules of [Rules](#rules) above and do not replace them.

1. **The gap.** For two tokenizers A and B of one signal: over the shots both have, `gap = (mean error of B - mean error of A) / mean error of B`, with a 95% interval from 2,000 paired resamples of the shots (seed 0).

    *Fixed before the measurement, 2026-11-22.*

2. **"A beats B":** the interval lies above 0; and, if A or B is a network, the gap is larger than the spread of the gap over three seeds of that network (largest minus smallest). **"A clearly beats B":** A beats B and the gap is at least 0.10. The threshold 0.10 is the author's reading of the design's word "clearly".

    *Fixed before the measurement, 2026-11-22.*

3. **Same size.** Two per-channel tokenizers are compared at the same number of values per token, the K of assignment `v1`. A concatenated tokenizer is compared at the same number of values per frame for the signal (channels times K). Each comparison is also reported at half and at a quarter of that size for the concatenated arm, and with the number of values each arm needs to reach 1b's target for the signal.

    *Fixed before the measurement, 2026-11-22.*

4. **Learned shared tokenizer against the fixed transform** (series channels; the design: "It replaces the fixed transform for any signal where it has the lower error"). A learned shared tokenizer (linear or network, the better on `val_select`) replaces the cosine transform for a signal if it beats it on `val_select` and again on `val_report`. Otherwise the cosine transform stays: it has no weights, cannot collapse and works at any frame duration.

    *Fixed before the measurement, 2026-11-22.*

5. **Learned tokenizer against its linear baseline** (profiles, map, Mirnov bands; the design: "A learned tokenizer that does not beat its linear baseline is replaced by the baseline"). The baseline is 1b's codec for the signal (principal components; the band transform). The learned one is used only if it beats the baseline on `val_select` and again on `val_report`.

    *Fixed before the measurement, 2026-11-22.*

6. **Shared against specialized** (the four magnetics signals; the two Mirnov signals). Reported per signal: shared against own, own against concatenated, shared against concatenated, in linear form and as networks. "The specialized one clearly beats the shared one" is judged between the best shared arm and the best concatenated arm (each chosen on `val_select`), by rule 2 on `val_select` and again on `val_report`. For the magnetics, which no task predicts, two more numbers are reported and decide nothing alone: the input round-trip of the direct models of tasks 1-1 and 1-2 (1b's instrument I) with each arm's tokens, and the damage of one missing channel to the other channels. What follows from a clear win is open decision 1; this plan writes `v2` (shared) and `v2c` (the clear winners swapped in) and decides nothing further.

    *Fixed before the measurement, 2026-11-22.*

7. **Collapse.** A network is collapsed if, for any token, the first value's use of the 8 levels has a normalised entropy under 0.3 on the `val_select` examples (`ignite.gate.utilization`; 0.3 is baseline IGNITE's `gate_min_code_entropy`). A collapsed tokenizer takes part in no comparison and is reported. The count of values whose entropy is at least 0.3 is reported as "active values" and decides nothing.

    *Fixed before the measurement, 2026-11-22.*

8. **A network that loses at seed 0** is trained once more with learning rate 1e-4 before the loss stands (stage 0, direct model: a rate of 3e-4 was unstable on task 1-2 and 1e-4 was not, while on tasks 1-1 and 2-1 the lower rate was worse by 0.0056 each; so the lower rate is a second try, not the default). **A network that wins at seed 0** gets seeds 1 and 2 before the win stands. A run whose validation error still falls by more than 1% in its last quarter is extended once to twice its steps.

    *Fixed before the measurement, 2026-11-22.*

9. **Stage 2 does not wait.** Stage 2 starts on 1b's `v1`. `v2` replaces it only where rules 4 and 5 say so. If nothing learned beats anything, `v2` equals `v1`, and that is a result.

    *Fixed before the measurement, 2026-11-22.*

Written 2026-11-23, after the rules above and without changing them: stage 2 starts on `v1b`, not on `v1` (ruling
C29, see [The assignment](#the-assignment)), and job M9 built `v2c` from `v1b`. Nothing learned replaced anything,
so `v2` equals `v1b`.

**Readings the controller added before any network was trained.** Four readings, written
on 2026-11-23. Three of them (rulings C11, C15 and C17) came before any production
store, recoder or comparison existed. The fourth, C33, was written at 07:15 UTC,
after the production store (written by 07:10:34) and the linear recoders (07:10:38
to 07:10:58) were built and before any network was trained. None changes a rule above.

- **`interferometer-n_e_line` is one shot's row (ruling C11).** Its cache-side error
  in a comparison is carried by one shot, 16575, one of the 100 `val_select` shots
  of stage 1b's job K2 (measured 2026-11-23, see [Results](#results)). Its row
  of `s-klt` against `v1` is an exact re-basis at K = 16 (the `v1` setting of this
  signal is the cosine transform at K = 16) and a tie by construction, so it says
  nothing about learned tokenizers.
- **The input round-trip cites K4's line (ruling C15).** Rule 6's round-trip is
  read from K4 (measured 2026-11-23, `val_select`, 100 shots asked and 88, 92, 89
  and 95 scored, `unquantized`, in the table of
  [The pass-through and input round-trip costs](#the-pass-through-and-input-round-trip-costs)):
  the raw score and the score through `v1`, for task 1-1 (`s90k`) 0.12497 and
  0.12527, task 1-2 (`lr1e4`) 0.03686 and 0.03755, task 2-1 (`s90k`) 0.13358 and
  0.13400, task 4-1 (`h050b256`) 0.16211 and 0.16255. These are the numbers the
  reading cites, instead of the `val_report` scores of 2026-11-22 that the plan
  quoted; the conclusion is the same (read the round-trip mainly on task 1-2, see
  [How to run stage 1c](#how-to-run-stage-1c)).
- **Task 4-5's floor is bounded by the frame oracle (ruling C17).** Measured
  2026-11-23 on `val_select`, 62 shots with 4-5 windows, 21,598 windows: the frame
  oracle (all 32 values in every band) scores 0.5142 for the task, 0.6267 on
  `magnetics-b_field_tor_probe_cc_field` and 0.4017 on
  `magnetics-b_field_pol_probe_omv_voltage` (the last two are also in the note of
  [The assignment](#the-assignment)), and the floor on `cc` is flat in k
  (1b's step a and step b both give 0.6267). So a tie between Mirnov tokenizers
  under rules 5 and 6 is expected whatever the tokenizer is (inferred, not
  measured). Had the owner answered open decision 1 of stage 1b with per-column low
  bins before M4, the Mirnov arms would have been planned again; the ledger records no
  answer, and M4 to M7 ran them as planned (07:44 to 08:04 UTC). The
  Mirnov networks stay fourth in the cut order.
- **Rule 7 is judged on distinct examples, and "any token" means no token is in use
  (ruling C33).** Written 2026-11-23 at 07:15 UTC, before any network was trained.
  A signal is collapsed when no token's first value reaches 0.3 normalised entropy,
  read on the signal's distinct validation examples (an exact duplicate row, with
  its stream identity, counted once). The plan's literal reading (some token's first
  value under 0.3, on all rows) is recorded beside it as `collapsed_any_token`, and
  the last line of a training names any signal where the two differ. Reasons:
  `ignite.gate.utilization`, which rule 7 cites, pools the token positions and does
  not judge each token; the plan's own sentence that unused later values do not
  block says that an idle token is not a collapse; and an input that barely varies
  (a dead channel, NBI off in most frames) is not a collapse of the network (about
  88% identical rows give a normalised entropy of about 0.29 on a healthy network: the
  ruling's calculation, not measured on a network). A one-token arm is the plan's rule unchanged. The
  arm's verdict stays "any signal collapsed" (exit 3). Both readings, the entropy
  of each token's first value, the active tokens, the distinct and all examples and
  the share of the most common row are recorded per signal in `result.json`, so the
  literal reading can be applied afterwards without training again.

**Readings added during the measurement.** Eight more readings were written on 2026-11-23.
C37 (07:31 UTC) came as M6 started (07:30:56) and before any network result, so it had the same
results in view as C33. C39 to C43 (07:43 to 08:10 UTC) came after M6's results were seen. C44 and
C45 (08:53 UTC) were written after the scoped review of this page and with the results of jobs
M1 to M10 in view; C45's own result came at 08:57 UTC. None changes a rule above. Each says what
was decided, why, and what it costs if wrong.

- **A signal that cannot vary has no verdict (ruling C37).** When a training exits 3
  (collapsed), a signal with fewer than 2 distinct finite validation rows
  (`n_distinct_examples` under 2) has its verdict read as "no verdict" (its input
  cannot vary, and no data is not a negative), and the arm's verdict is read again from
  its other signals. Two distinct rows can reach a normalised entropy of ln 2 / ln 8 =
  0.333, above 0.3, so 2 is the smallest count at which the verdict can be the
  network's. No code changed. It was applied to no arm: every signal that failed has at
  least 8,576 distinct validation rows (`n_distinct_examples` in `result.json`,
  2026-11-23). If wrong, an arm whose only failing signal never varies is kept, and it
  can win only on fidelity.
- **Rule 8's extension covers the collapsed arms (ruling C39).** Every M6 run whose
  plateau is "no" was extended once (ruling C39), by `--steps 40000 --variant long`, the four
  collapsed ones included, because rule 8 triggers on the validation curve and not on a verdict.
  Ten arms were extended and three were not (`net.mag`, `net.flux`, `net.bandcat.cc`).
  From then on the long run is the arm's seed-0 run, for the collapse verdict and for
  M7, whether or not its `val_mse` is lower: choosing between the two on `val_select`
  would select twice on the same 100 shots. No arm's run is extended again (ruling C45,
  below, extends one retry whose lineage had not used its extension) and none gets a
  further try for a collapse; the s0 runs stay on disk and are in
  [the trainings table](#the-trainings). A retry at 1e-4 or seeds 1 and 2 use the
  steps of the run they repeat. The four collapsed arms were collapsed again after
  40,000 steps. Measured 2026-11-23 from `train_log.csv`: a long run repeats its s0
  run's first 20 validation rows exactly for 5 of the 10 arms and differs for
  the other 5 by up to 0.066 in `val_mse`, each first differing at step 1,000 (the first
  logged step). So GPU training of those five is not repeatable at a fixed seed (cause not
  established), and a long run is a second training with the same seed and not a continuation;
  a network's seed spread (rule 2) cannot be assumed small. If wrong, a collapsed arm got one training more
  than the plan gives it; had its long run not collapsed, it would enter the comparisons,
  where it can win only on fidelity, and the page would still show its s0 collapse.
- **Which pairs M7 runs (ruling C40).** M7 runs the plan's pairs (M5's with the network in
  place of the linear arm, without the scaled `.h` and `.q` arms that networks do not have,
  plus `s-net0:s-klt`, `b-net0:b-klt`, `p-net0:v1` and `m-net0:v1`) and two more,
  `c-net0:c-pca` and `bc-net0:bc-pca`. Reason: rule 4 picks "the better on `val_select`"
  of the linear and network shared arms, and rule 6 picks the best concatenated arm the
  same way, for which the plan lists no paired comparison. The two extra pairs need no new
  floor. An assignment whose source `recode` refuses (collapsed, exit 2) is not built, and
  its pairs are skipped and logged. If wrong, two extra tables that decide nothing the
  others do not let one derive.
- **Which networks "lost" at seed 0 (ruling C41).** A network loses when the comparison
  for the slot it competes for goes against it. For `p-net0` and `m-net0` (rule 5) that is
  `v1`; for the concatenated networks `c-net0` and `bc-net0` (rule 6's best concatenated
  arm, chosen on `val_select`) it is `c-pca` and `bc-pca`; for `o-net0` (rule 6's own arm)
  it is `o-klt`, so the pair `o-net0:o-klt` was added (cache side, no new floor). By that
  reading all nine trained networks lost, and each was trained once more at learning rate
  1e-4 (`--variant lr1e4`) with the steps of the run it repeats (40,000 for six, 20,000 for
  `net.mag`, `net.flux` and `net.bandcat.cc`), recoded to `o-net0r`, `c-net0r`, `bc-net0r`,
  `p-net0r` and `m-net0r`, and compared in the same pairs plus `o-net0r:o-klt`. A retry that
  still loses leaves the loss standing; one that won would get seeds 1 and 2. If wrong,
  the cost is nine trainings (109 to 604 s each, 2026-11-23) that rule 8 may not have
  required; a retry that flipped a loss is read by the same rules.
- **Rule 4 is read again at the K that `v1b` has (ruling C42).** Rule 4's only `val_select`
  win in linear form is `s-klt` on the solenoid current, measured at `v1`'s K 1 (rule 3),
  while `v2` is built from `v1b`, where the two currents have K 2 (ruling C29). A win at
  K 1 says nothing about K 2: the `unquantized` floor of the solenoid current on
  `val_report` is 0.0021 with `v1b` and 0.0133 with `v1`. So the two currents were
  compared again at `v1b`'s K (`s-klt.b`, recoded from `v1b` with `klt.series`, against
  `v1b`, floors of task 2-1 on `val_select`), and a signal enters `v2` only if it beats
  `v1b` there and again on `val_report`. Every other series signal keeps `v1`'s verdict,
  since `v1b` equals `v1` on it. If wrong, the cost is two floor passes on the CPU, and a
  verdict on the two currents judged at the K that stage 2 uses (`v1b`'s K 2) instead of the plan's
  (`v1`'s K 1, rule 3), at which `s-klt` beat `v1` on the solenoid current and was not confirmed on
  `val_report`.
- **The input round-trip of `v2c` is scored on four tasks (ruling C43).** `v2` equals `v1b`
  unless a retry won (rule 4 none by C42, rule 5 all lost at seed 0), and `v1b`'s round-trip
  was already measured (ruling C29). `v2c` is scored with `--instrument inputs` on all
  `val_report` shots: `ranged` on tasks 1-1 (`s90k`), 1-2 (`lr1e4`), 2-1 (`s90k`) and 4-1
  (`h050b256`), like for like with the `v1` of K6 and the `v1b` of C29; and `unquantized`,
  the plan's mode, with `v1b` and `v2c` on 1-1 and 1-2. The magnetics arrays are inputs to
  12 of 14 tasks (all but 3-1 and 3-2), so 2-1 and 4-1 were added to the plan's 1-1 and 1-2.
  No task predicts a magnetics signal, so `v2c` changes no task's output and needs no floor
  (M9 asks for floors "for every task with a changed output signal"). The gate on `v2` is
  `v1b`'s (07:40 UTC, 57 rows, 0 block). If wrong, 8 GPU passes (52 s to 22 minutes each, 2026-11-23, `m9_inputs.meta`).
- **A network floor pass over 3 times its linear arm's pass is recorded and stops nothing (ruling C44).**
  Written 2026-11-23 at 08:53 UTC, after the review of this page and with the results of M1 to M10 in
  view. Ruling C7 (see [the departures](#where-stage-1c-departs-from-the-design)) says that a network
  floor pass taking more than 3 times the pass of the matching linear arm stops the controller and makes
  it report. In M7 (`logs/m7.meta`: seven CPU floor passes started together at 08:01:38) `p-net0` took 40 s
  on task 2-2 against 13 s for `v1`, which is 3.1 times, and the controller did not stop and did not
  report. The ruling: no stop now. The condition guards M7's running time; these passes took seconds,
  were timed under shared load, and the retries' passes took 16 to 134 s (`logs/m8r.meta`). The page states
  the crossing in the departure of ruling C7. If wrong, nothing in a verdict: the guard is about time only.
- **Rule 8's one extension also covers a retry (ruling C45).** Written 2026-11-23 at 08:53 UTC, after the
  review of this page and with the results of M1 to M10 in view. Rule 8 extends a run with a plateau of
  "no" once. `net.flux.s0.lr1e4` ended at 20,000 steps with a "no" (best `val_mse` 0.01360 at step 19,000),
  and its lineage had not used the extension, because `net.flux.s0` plateaued at 20,000 steps and ruling
  C39 left it alone. The five other retries that end with a "no" (`net.bandcat.omv`, `net.ccbv`, `net.map`,
  `net.obr` and `net.profile`, all at 40,000 steps) repeat C39's long runs, so their lineage has used its
  one extension and they get none. The job: `net.flux --lr 1e-4 --steps 40000 --variant lr1e4_long` on GPU 1;
  `recode` `c-net0rl`, which is `c-net0r` with that flux run; `c-net0rl:c-pca` (the slot of ruling C41) on
  `val_select-n100`, with CPU floors for what is missing; read by rules 2 and 7 like every other network.
  Result (2026-11-23, 08:54:00 to 08:57:23 UTC, `logs/c45.meta`, every step exit 0): `net.flux.s0.lr1e4_long`
  has a best `val_mse` of 0.01282 at step 25,000 of 40,000, a plateau of "yes", is not collapsed (20 of 20
  tokens and 6 of 6 values active) and took 180 s, against 0.01360 at step 19,000 of 20,000 for
  `net.flux.s0.lr1e4`. No floor pass was missing. In `c-net0rl:c-pca` (`val_select`, 100 shots) the flux has
  0.01531 against `c-pca`'s 0.002548, gap of b against a 0.834 (0.823 to 0.844), "b clearly beats a"; the
  other three arrays are unchanged from `c-net0r:c-pca`, since the same recoders are used. The flux loss
  stands (0.01573 before the extension) and every verdict of stage 1c is unchanged. If wrong: about 4 min of
  GPU 1 and one CPU floor wave, estimated when the ruling was written (measured: 180 s and no floor pass),
  for a comparison that could not change the verdict unless the extension closed a sixfold gap (flux 0.01573
  against `c-pca`'s 0.002548 in `c-net0r:c-pca`); it did not.

### Where stage 1c departs from the design

Each item names what [the design](./ignite-mast-design.md) says and the reason for
the departure. The first nine are decisions of the plan's author, written
2026-11-22; no measurement is behind them unless the item says so. The rulings of
the controller that follow them changed the method on 2026-11-23, each with the
number behind it. A number from a trial is a rough look on the trial store of
2026-11-23: 8 training shots for the fit and 2 `val_select` shots (11806 and
30467) for the validation examples. It is not a result.

- **A learned tokenizer is a recoder behind a fixed transform.** The design says
  "encoder + finite scalar quantizer + decoder" on one channel, profile or map in
  one frame. Here the first layer of the encoder and the last layer of the decoder
  are the fixed transform of stage 1b at its top rung (16 cosine values, 32 per
  band, 96 principal components) and are not learned. Reasons: any sampling rate,
  missing samples and the units are handled once, by code that 1b has already
  tested and scored; the error of the top rung is known from 1b's ladders, so
  what the learned part can add at most is known before it is trained; and a
  linear recoder is then exactly the best linear tokenizer of that input,
  computable on the CPU. The cost: the learned tokenizer cannot use what the top
  rung has dropped (for the soft X-ray cameras and D-alpha, the part above
  1.6 kHz).
- **The values of a token are ordered in two ways.** The design says that for a
  learned tokenizer the order is precision, a second quantizer level encoding what
  the first missed, with the later parts dropped at random during training. Here
  the precision order is the 8 and 64 levels or no rounding of a bounded value
  (the 64-level reading is the design's second level, and the three are drawn with
  probabilities 1/4, 1/4 and 1/2, which keeps the design's "quantizer skipped for
  a random half"). The length order, a random cut of the list of values, is the
  author's addition: it lets one trained network give every K up to its size, so
  it can be compared with the fixed transform at each K.
- **The quantizer is this plan's own few lines, not baseline IGNITE's
  `SpectroQuantizer`.** Reason: the token value that stage 2 predicts is the
  bounded value itself, which the library hides. Baseline files are not touched.
- **"Built the old way" means one tokenizer per cached signal, all its channels in
  one input, no channel identity.** The control uses the same network body and the
  same training recipe as the shared tokenizer and differs only in what an example
  is. Baseline IGNITE's own `SlowTSCodec` is not used (another architecture,
  another time base, other losses): a difference would then mix the architecture
  with the question. There is one control per probe array (four for the
  magnetics) and not one for all 91 magnetics channels, because a codec of stage
  1b serves one signal; the linear comparison also reports the joint basis of all
  91 channels at the same number of values, as a number only.
- **Three arms, not two.** Reason: shared against concatenated changes two things
  at once, how much data the tokenizer sees and whether it sees the channels
  together, so each comparison has a third arm, the per-channel tokenizer fitted on
  the signal's own diagnostic only (see
  [Shared against specialized](#shared-against-specialized)).
- **The loss counts every channel equally.** Squared error in example units, each
  channel scaled to unit variance, the same in every arm. TokaMark's score weights
  channels by their physical variance. The comparison is read in TokaMark units
  regardless, and [Limits](#limits) says where the two disagree.
- **"Statistics first" for the fluctuation path is met by the input, not by an
  adversarial decoder.** A band example is the frame-mean power spectrum of the
  band, which is already a statistic.
- **Fluctuation tokens for the 50 kHz channels are left out.** Reason: their
  targets in the benchmark are waveforms, a token that holds band power cannot
  lower a waveform error, and a codec that does not decode to a waveform does not
  fit the instruments of stage 1b, so no stage-1 number can judge them. The
  stage-2 plan adds the band as input tokens and judges it by an ablation. Job
  M10 measures what the band holds as a statistic instead.
- **The Mirnov control cannot measure what the design's risk row hopes.** The design
  says the Mirnov-only control measures what the cost is for mode numbers read from
  the phase between coils. The cached spectrograms hold power only, so the control
  measures whether the three probes' power spectra are better coded together. The
  phase question needs raw Mirnov waveforms and is left to a later plan.
- **The plan's open decisions (ruling C0).** The plan names five open decisions for
  the owner, who approved the plan on 2026-11-22 (16:17 UTC) without answering them.
  They are taken as the plan's author recommends until the owner says otherwise.
  (1) A shared tokenizer is adopted only on rule 6's measured win; a clear win of
  the control goes into a second assignment `v2c` while `v2` keeps the shared
  tokenizer, and stage 2 decides on the scores of tasks 1-1 and 1-2. (2) The
  number of values per frame that each arm needs to reach 0.0176 is reported and
  not adopted on. (3) Channel identity without positions is accepted for stage 1
  (above). (4) No fluctuation tokens for the 50 kHz channels in stage 1. (5) If
  stage 1 is late, the cut order is: the learned map tokenizer (trained only if
  stage 1b's linear baseline misses its target), the extra seeds, the learned
  profile tokenizer, the Mirnov networks, the series networks; the linear arms
  are never cut. Reason: each changes the goal, the bar or what is published, and
  the owner can reverse any of them.
- **Token sizes (rulings C4 and C5).** The Mirnov network control `net.bandcat.*` is
  54 tokens of 8 (432 values), not 24 tokens of 8 (192): `v1` keeps 432 values per
  frame on each Mirnov signal (3 x (32 + 7 x 16), job K3, measured 2026-11-23),
  and at 192 values `recode` would exit with "holds" and the network control
  could not be built at the size of `v1`. The linear `pca.band.*` arms keep all
  768 values (24 x 32, in 192 tokens of 4), of which an assignment holds 432 (108
  tokens). The concatenated arms of `obr`, `obv` and `flux` use tokens of 6, linear
  and network alike: `pca.<array>` keeps all S x 16 values and `net.<array>` keeps
  S x 8, so `v1`'s 18, 36 and 30 values per frame are exact (3, 6 and 5 tokens) and
  every arm total divides by 6 (288 and 144 for `obr` and `obv`, 240 and 120 for
  `flux`). `ccbv` keeps tokens of 4 and 8, because 160 is exact in both. Reason:
  rule 3 asks for equal values per frame, and rounding down to tokens of 8 would
  take 18, 36 and 30 values to 16, 32 and 24, a handicap to the concatenated
  control of 11%, 11% and 20% (arithmetic from the token sizes, not measured)
  that biases a tie, where rounding up would favour it.
- **Network floors run on the CPU (ruling C7).** `score --instrument floor` keeps
  stage 1b's refusal of `--gpu`, `--share-gpu` and `--variant` before any file is
  read, and the script limits torch to one thread in its floor workers. Reason: the
  alternative makes a refusal depend on reading the assignment file, and the
  networks are small (tokens of 8, at most 432 values). If a network floor pass
  takes more than 3 times the pass of the matching linear arm, the controller stops
  and reports. Measured 2026-11-23 (`$S1/logs/m7.meta`: seven CPU floor passes started together at
  08:01:38, timed start to end under that shared load): `p-net0` took 40 s on task 2-2 against 13 s for
  `v1` (3.1 times), and 17 s against 13 s on 3-3; `m-net0` took 38 s against 19 s on 2-3; `bc-net0` took
  136 s on 4-5, with no linear pass beside it (M4's `bc-pca` took 322 s among 24 passes, not like for like).
  The 2-2 pass crossed the 3 times and the controller did not stop. Ruling C44 (2026-11-23, written after
  the review) records the crossing and takes no stop: the guard is about M7's running time, and these
  passes took seconds under shared load. The retries' passes took 16 to 134 s (`$S1/logs/m8r.meta`).
- **Concatenated linear recoders are fitted on zero-filled rows (ruling C21).** A
  missing stream is 0 in example units when a concatenated recoder encodes, and
  `fit_linear` now fits on rows filled the same way, not on a pairwise covariance.
  The plan's text "used pairwise" is superseded for concatenated recoders.
  Reason, rough look, 2-shot trial store, 2026-11-23 (the 40 `ccbv` probes, a basis
  of 640 front values fitted on the trial's 672 training frames, the share left
  over measured on its 167 validation frames of 2 `val_select` shots): fitted
  pairwise it is 0.0074, 0.0072, 0.0071 and 0.0071 at 40, 80, 160 and 320 values,
  fitted on zero-filled rows 0.0006, 0.0002, 0.0001 and 0.0000, so the pairwise
  fit was 12 to 36 times higher at 40 and 80 values. The cause (these three
  figures are on the trial's 672 training rows, not on its validation frames):
  69% of the trial's `ccbv` rows have a missing stream (4.5% of the entries), and
  a pairwise covariance with that many gaps is not positive semi-definite (131
  negative eigenvalues of 640), which puts real energy last in the order of the basis. The
  per-stream recoders and stage 1b's profile codecs are not affected (their rows
  are finite, and a profile's missing points are solved by a ridge). The cost,
  inferred and not measured: a zero-filled basis spends some components on which
  streams are missing, and a masked fit might keep a little more per value, so the
  linear controls are slightly pessimistic.
- **The left-over of the joint basis uses the same zero-fill (ruling C23).**
  `compare.left_over` fits on training rows filled as above, through the one helper
  that `fit_linear` calls, so the own concatenated basis of `joint` equals
  `train_tokenizer.py linear --arm pca.<signal>`. Reason: a third copy of the fit
  would measure the pairwise defect of the previous item for the own basis and for
  the joint one.
- **`joint` counts a frame when every listed signal has a finite stream in it
  (ruling C25).** A missing stream is 0 in the encode and counts in neither sum, as
  in `linear`'s left-over and in the previous items. Reason, rough look, 2-shot
  trial store, 2026-11-23, `val_select`: 0 of 167 validation frames have all 91
  magnetics streams finite (dead probes), so a count of complete frames has no
  data. The three numbers of `joint` are therefore fractions over the finite
  entries, not over complete frames; they use one mask, so their order can be
  compared.
- **`joint` matches the frames of different signals by position in the shot
  (ruling C26).** The stores drop the start time, so this assumes that the signals
  of one shot start together. Measured 2026-11-23 on all 11,188 shot files of
  cache v1 (the start time of every signal; no score is involved): the four
  magnetics signals share one start time exactly in all 11,156 shots that have
  them, and all series signals of a shot share one in 11,188 of 11,188 shots. The
  two Mirnov spectrograms start 1.022 ms later, under one frame, and `joint`
  already refuses to mix them with the 16-value magnetics. If a later cache version
  gave signals their own start times, the rows of `joint` would misalign without a
  warning, and the error would work against the joint basis.
- **The 50 kHz band's NaN rules (rulings C13 and C14).** A frame with a missing
  sample is NaN in every band value and is left out of every sum, checked before
  the transform: the cosine transform of a frame with some missing samples gives
  the frame mean and zeros, which would enter the sums as zero band power. The
  saddle coils' `band_to_trend` is empty by design: `v1` gives them K 1, and the
  trend path then holds no value above the mean to divide by (the line is NaN for
  K under 2); their share below 5 kHz and their persistence are read. D-alpha and
  the soft X-ray cameras (K 2 or 4 in `v1`) have trend power from cosine value 1
  on. Reason: a number that is not measured stays empty, and no data is not a
  negative.

### How to run stage 1c

The controller runs these jobs from `/home/nvidia/FusionAIHub` with
`IGNITE_MAST_ROOT` unset, after stage 1b's jobs K1 to K3 (the fit `fit/v1` and the
assignment `v1`) and after the tasks of plan 1c that the table names (Task n is
that plan's task n). In the commands `RT="scripts/mast/py.sh
scripts/mast/run_tokenizer.py"`, `TT="scripts/mast/py.sh
scripts/mast/train_tokenizer.py"` and `S1=/models/intermediate/mast/stage1`. Every
job without a GPU is started with `CUDA_VISIBLE_DEVICES=""`. Logs go to
`$S1/logs/`. This stage writes under `$S1`: `store/v1/{train,val}/`,
`recoders/<name>/` and `compare/`; assignments and scores go where stage 1b puts
them.

Which script holds which sub-command (checked in the argument parsers of both
scripts): `run_tokenizer.py` (`$RT`) holds `fit`, `score`, `ladder`, `assign`,
`recode`, `gate`, `compare` and `seeds`, and `train_tokenizer.py` (`$TT`) holds
`store`, `linear`, `joint`, `arms`, `train` and `fluct`.

`recode` refuses, with exit 2 before it writes anything, a network recoder whose
`result.json` is missing (its training did not finish) or does not say
`collapsed` false (rule 7), and a recoder whose recorded `fit_sha256` or
`stats_sha256` differs from the fit and statistics in use (after a `fit --force`
its basis is the wrong one); `score`, `gate` and `compare` refuse that second case
for an assignment that names such a recoder. `seeds` refuses pairs that do not all
have the same `b`, or that repeat an `a`, prints each pair's `n_shots`, and gives no
verdict for a signal whose pairs were measured over different shots. It reads "b
beats a" against the spread of the `gap_ba` and "a beats b" against that of the
`gap`.

Every command starts with the `OMP_NUM_THREADS` it is launched with, and each job
gets its own count so that workers times threads stay under the node's 352 cores:
16 for a job that runs alone in one process, 2 per worker for the jobs of 64
workers, 8 for each of the 11 linear arms run at once and for the trainings that
share a GPU, and 4 per pass for the floor passes, many of which run at once (the
floor worker itself keeps torch at one thread, see the departures). Reason,
measured 2026-11-23: one CPU test of the training loop took 86 to 111 s at
torch's default thread count on this 352-core node under load, and 1.6 s with 8
threads.

Every duration in the table is the plan author's estimate, written before any job ran. The jobs'
measured times are in `$S1/logs/*.meta` (2026-11-23, UTC): M1 to M3 07:10:19 to 07:10:58, M4's 24 floor
passes 07:46:02 to 07:51:39, M5 to 07:52:41, M6 07:30:56 to 07:40:54 (the long runs of ruling C39 07:43:48
to 08:00:14), M7 08:00:35 to 08:04:47, M8 08:06:01 to 08:23:31, M9's input round-trip 08:10:19 to 08:33:38,
M10 07:47:12 to 07:47:38, and ruling C45's extension 08:54:00 to 08:57:23. Unlike the stage-0 trainings, whose speed was set
by the loaders, these hold their examples in memory (about 2,000 shots times 100
frames times 163 channels of 16 values: 2 GB for the series family, an estimate),
so several can share one GPU.

| Job | After | Command | GPU | Duration | What is read from the output |
|---|---|---|---|---|---|
| M1 store | Task 3, K3 | `OMP_NUM_THREADS=2 $TT store --workers 64` | none | 10 to 30 min | 24 lines. No signal says "no data"; the skipped shots per signal match 1b's fit. |
| M2 linear arms | M1 | `OMP_NUM_THREADS=8 $TT linear --arm A` for the 11 linear arms, all at once | none | 1 to 5 min each | Per signal the share left at 1, 2, 4, 8, 16 values per stream. It falls from left to right. |
| M3 joint basis | M1, Task 4 | `OMP_NUM_THREADS=16 $TT joint --signals ccbv,obr,obv,flux` | none | minutes | Per size three shares: the joint basis of the 91 channels, each array's own basis, the per-channel basis. This is the upper limit of what coding channels together can give in linear form. |
| M4 recoded assignments and their floors | M2, Task 4 | `OMP_NUM_THREADS=16 $RT recode --name s-klt --from v1 --arm klt.series`; `o-klt` (`klt.mag`); `c-pca` (`pca.ccbv,pca.obr,pca.obv,pca.flux`), `c-pca.h` and `c-pca.q` (the same with `--scale 0.5`, `0.25`); `b-klt` (`klt.band`); `bo-klt` (`klt.band.cc,klt.band.omv`); `bc-pca` (`pca.band.cc,pca.band.omv`), `bc-pca.h`, `bc-pca.q`. Then M5's `compare` once: it prints every `score` command whose floor is missing (`OMP_NUM_THREADS=4 $RT score --task T --split val_select --n-shots 100 --assignment A --mode unquantized`); run those in waves | none | about 20 passes of 3 to 15 min; about 1 h | `recode` prints equal values per frame before and after (a concatenated network arm may be a few values short: tokens of 8). The last line of each pass. |
| M5 linear comparison | M4 | `OMP_NUM_THREADS=16 $RT compare --a A --b B --drop-channel --reach 0.0176` for the pairs: `s-klt:v1` (rule 4); `s-klt:o-klt`, `c-pca:o-klt`, `c-pca:s-klt`, `c-pca.h:s-klt`, `c-pca.q:s-klt`, `c-pca:v1` (rule 6, magnetics); `b-klt:v1` (rule 5); `b-klt:bo-klt`, `bc-pca:bo-klt`, `bc-pca:b-klt`, `bc-pca.h:b-klt`, `bc-pca.q:b-klt` (rule 6, Mirnov) | none | minutes each | The 13 tables. They go to the owner the day they exist, with M3's line: this is the linear answer to shared against specialized, and the networks cannot change what the data holds, only how much of it a tokenizer of that size gets. |
| M6 networks, seed 0 | Task 6, M1 | `OMP_NUM_THREADS=8 $TT train --arm A --seed 0 --gpu N --share-gpu` for the 13 network arms, two or three per GPU | 4, shared | 5 to 20 min each; under 1 h | The last line of each run. `collapsed yes` (exit 3): that arm takes part in no comparison, and `recode` refuses it. A last line that ends `(rule 7 readings differ on: ...; C33 decides)` names a signal that the plan's literal reading would call collapsed and C33's does not (or the reverse): read `usage` in its `result.json`. `plateau no`: run it once more with `--steps 40000 --variant long` (rule 8). |
| M7 network comparison | M6 | `OMP_NUM_THREADS=16 $RT recode` to `s-net0` (`net.series.s0`), `o-net0` (`net.mag.s0`), `c-net0` (the four `net.<array>.s0`), `b-net0` (`net.band.s0`), `bo-net0` (`net.band.cc.s0,net.band.omv.s0`), `bc-net0` (`net.bandcat.cc.s0,net.bandcat.omv.s0`), `p-net0` (`net.profile.s0`), `m-net0` (`net.map.s0`); the missing floors as in M4 (`OMP_NUM_THREADS=4` per pass, on the CPU: see the departures above); then `OMP_NUM_THREADS=16 $RT compare` for the pairs of M5 with the network in place of the linear arm, and `s-net0:s-klt`, `b-net0:b-klt`, `p-net0:v1`, `m-net0:v1` (rule 5) | none | about 12 passes of 5 to 20 min; about 1 h | The tables, as in M5. A network's best step and plateau were chosen on the same 100 `val_select` shots as these tables read, so its win is read with that in mind (see [The trainings](#the-trainings)). |
| M8 rule 8 | M7 | For a network that lost: `OMP_NUM_THREADS=8 $TT train --arm A --seed 0 --lr 1e-4 --variant lr1e4 --gpu N --share-gpu`, then its comparison again. For a network that won: seeds 1 and 2, their assignments and floors, then `OMP_NUM_THREADS=16 $RT seeds --pairs a0:b,a1:b,a2:b --label val_select-n100` | 4, shared | up to 26 runs and their floors; 1 to 2 h | The gaps of the three seeds, their spread and the word with that spread. |
| M9 `v2`, `v2c` and their confirmation | M8 | `OMP_NUM_THREADS=16 $RT recode --name v2 --from v1 --arm <the winners by rules 4 and 5> --signals <their signals>`; `v2c` from `v2` with the concatenated winners of rule 6. For every task with a changed output signal: 1b's K5 passes on `val_report` in the three read-outs (floors, `OMP_NUM_THREADS=4` per pass); `OMP_NUM_THREADS=16 $RT compare --label val_report` for the decided pairs; for tasks 1-1 and 1-2 `OMP_NUM_THREADS=8 $RT score --task T --split val_report --instrument inputs --assignment A --mode unquantized --variant V --gpu N` with `v2` and `v2c`; `OMP_NUM_THREADS=16 $RT gate --assignment $S1/assignments/v2.json` | 4 passes with a GPU | 1 to 2 h without a GPU, 4 passes of 15 to 45 min with one, the gate 5 to 15 min | A choice made on `val_select` stands only if `val_report` gives the same word. The gate's last line. If `v2` equals `v1`, nothing is scored again and the page says so. |
| M10 the 50 kHz band | Task 7, K3 | `OMP_NUM_THREADS=2 $TT fluct --workers 64` | none | 5 to 15 min | Four lines; they go to the stage-2 plan. |

The variant `V` of a direct model is the one the headroom table shows for the task
(1b says how it is chosen). As of 2026-11-22 the direct model of task 1-1 is behind
the best published score (0.1361 on `val_report` against 0.0941; 0.1235 with three
times the steps) and that of task 1-2 is ahead with learning rate 1e-4 (0.0362
against 0.0412): the input round-trip of rule 6 is therefore read mainly on task
1-2, and its number for task 1-1 is reported with that limit. Task 1-2's margin,
0.0037 at the interval's near end, is under the 0.0041 that separated two
trainings of one configuration on task 1-1 (both measured 2026-11-23, see "How
firm the verdicts are" in
[the stage-0 page](./ignite-mast-stage0.md#the-other-seven-tasks-stage-1a)), so by
that guide the verdict of task 1-2 is not firm.

After M9 the controller filled the tables below (2026-11-23), set the status line of this
page and committed with the numbers, their date, split and shot counts in the body.
Stage 2 does not wait for any of this: it starts on `v1` (on `v1b` since ruling C29, see
[Rules of the comparison](#rules-of-the-comparison)).

### Results of the comparison

Six tables (the linear arms, the networks, the trainings with a sub-table of its collapsed runs, the
decisions, the input round-trip of `v1`, `v1b` and `v2c`, and the 50 kHz band), filled
on 2026-11-23 by a script that reads the files stored under `$S1` and types no number
by hand. Each has its date, split and number of shots in the line above it, as the
tables of [Results](#results) are. A cell that is empty is a quantity nobody stored
for that row, except an empty "Values to reach 0.0176" cell on a series array, which means that no
size reached 0.0176 (the text under each table says which); it is not 0 and not a good
result, and "no data" is written where a quantity should exist and does not. Every error
is a tokenizer floor, not an IGNITE score.

#### The linear arms

One row per signal and pair, from the comparison files of M5
(`$S1/compare/<a>__<b>.<label>.csv`, one per pair, with the label
`val_select-n100` first and `val_report` for the confirmation). The gap is that
of rule 1, `(mean_b - mean_a) / mean_b`, positive when `a` has the lower error,
with its 95% interval over 2,000 paired resamples of the shots; `gap_ba` is the
same with the roles swapped. `source` is the task whose stored
floor is read, or `cache` for a signal no task predicts. The two missing-channel
ratios (`drop_a`, `drop_b`) are the mean error with one channel dropped over the
mean error without it, written for the series arrays under `--drop-channel`. The
two reach columns (`reach_a`, `reach_b`) are the smallest size of the arm, taken
at 1/8, 1/4, 1/2, 1 and 2 times its values, whose error is at or under 0.0176
(`--reach`), empty if none is.

Measured 2026-11-23: the 13 pairs of job M5 (`compare --drop-channel --reach 0.0176`, 07:51 to
07:52 UTC, label `val_select-n100`: 100 `val_select` shots asked, 62 to 100 scored per signal, the
shots named in the "Shots" column); `s-klt.b:v1b` of ruling C42 (task 2-1, 89 shots); and the four
`val_report` pairs of M9 (the confirmation of rule 6, all 557 `val_report` shots, the four magnetics arrays
only). 18 pairs, 123 rows. The column "Label" is the split of the row. Tokenizers
a and b hold the same values per frame, except the `.h` and `.q` arms, whose halves and quarters are rounded to
whole tokens (rule 3).

| Pair (a:b) | Label | Signal | Source | Values per frame (a) | Values per frame (b) | Shots | Mean error (a) | Mean error (b) | Gap | Gap, low end | Gap, high end | Gap of b against a | Low end (b against a) | High end (b against a) | Word at one seed | Missing-channel ratio (a) | Missing-channel ratio (b) | Values to reach 0.0176 (a) | Values to reach 0.0176 (b) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `s-klt:v1` | `val_select-n100` | `gas_injection-total_injected` | cache | 1 | 1 | 88 | 0.007792 | 0.007902 | 0.014 | -0.034 | 0.053 | -0.014 | -0.056 | 0.032 | tie |  |  |  |  |
| `s-klt:v1` | `val_select-n100` | `interferometer-n_e_line` | cache | 16 | 16 | 91 | 0.2267 | 0.2267 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | tie |  |  |  |  |
| `s-klt:v1` | `val_select-n100` | `magnetics-b_field_pol_probe_ccbv_field` | cache | 160 | 160 | 100 | 0.01627 | 0.01327 | -0.226 | -0.280 | -0.177 | 0.185 | 0.150 | 0.219 | b clearly beats a | 1.000 | 1.000 | 160 | 160 |
| `s-klt:v1` | `val_select-n100` | `magnetics-b_field_pol_probe_obr_field` | cache | 18 | 18 | 100 | 0.01596 | 0.01588 | -0.005 | -0.008 | -0.003 | 0.005 | 0.003 | 0.008 | b beats a | 1.000 | 1.000 | 18 | 18 |
| `s-klt:v1` | `val_select-n100` | `magnetics-b_field_pol_probe_obv_field` | cache | 36 | 36 | 100 | 0.01861 | 0.01242 | -0.498 | -0.523 | -0.473 | 0.333 | 0.321 | 0.343 | b clearly beats a | 1.000 | 1.000 | 72 | 36 |
| `s-klt:v1` | `val_select-n100` | `magnetics-b_field_tor_probe_saddle_voltage` | cache | 12 | 12 | 98 | 0.01311 | 0.01278 | -0.026 | -0.033 | -0.020 | 0.025 | 0.019 | 0.032 | b beats a | 1.000 | 1.000 | 12 | 12 |
| `s-klt:v1` | `val_select-n100` | `magnetics-flux_loop_flux` | cache | 30 | 30 | 100 | 0.01221 | 0.007239 | -0.687 | -0.720 | -0.655 | 0.407 | 0.396 | 0.419 | b clearly beats a | 1.000 | 1.000 | 30 | 30 |
| `s-klt:v1` | `val_select-n100` | `pf_active-coil_current` | task_2-1 | 10 | 10 | 89 | 0.01718 | 0.01714 | -0.003 | -0.011 | 0.005 | 0.003 | -0.005 | 0.011 | tie |  |  |  |  |
| `s-klt:v1` | `val_select-n100` | `pf_active-coil_voltage` | cache | 16 | 16 | 99 | 0.0168 | 0.01597 | -0.052 | -0.091 | -0.043 | 0.050 | 0.041 | 0.083 | b beats a | 1.000 | 1.000 | 16 | 16 |
| `s-klt:v1` | `val_select-n100` | `pf_active-solenoid_current` | task_2-1 | 1 | 1 | 89 | 0.01362 | 0.01402 | 0.028 | 0.018 | 0.040 | -0.029 | -0.041 | -0.019 | a beats b |  |  |  |  |
| `s-klt:v1` | `val_select-n100` | `pulse_schedule-i_plasma` | cache | 2 | 2 | 98 | 0.01329 | 0.00989 | -0.344 | -0.452 | -0.267 | 0.256 | 0.210 | 0.311 | b clearly beats a |  |  |  |  |
| `s-klt:v1` | `val_select-n100` | `pulse_schedule-n_e_line` | cache | 1 | 1 | 99 | 0.01176 | 0.009983 | -0.178 | -0.288 | -0.101 | 0.151 | 0.092 | 0.224 | b clearly beats a |  |  |  |  |
| `s-klt:v1` | `val_select-n100` | `soft_x_rays-horizontal_cam_lower` | task_4-1 | 36 | 36 | 95 | 0.0519 | 0.05067 | -0.024 | -0.035 | -0.016 | 0.024 | 0.016 | 0.034 | b beats a |  |  |  |  |
| `s-klt:v1` | `val_select-n100` | `soft_x_rays-horizontal_cam_upper` | task_4-1 | 72 | 72 | 95 | 0.04907 | 0.04804 | -0.021 | -0.033 | -0.014 | 0.021 | 0.014 | 0.032 | b beats a |  |  |  |  |
| `s-klt:v1` | `val_select-n100` | `spectrometer_visible-filter_spectrometer_dalpha_voltage` | task_3-2 | 6 | 6 | 95 | 0.15 | 0.143 | -0.049 | -0.058 | -0.041 | 0.046 | 0.039 | 0.055 | b beats a |  |  |  |  |
| `s-klt:v1` | `val_select-n100` | `summary-ip` | task_2-1 | 4 | 4 | 89 | 0.01894 | 0.01468 | -0.291 | -0.321 | -0.261 | 0.225 | 0.207 | 0.243 | b clearly beats a |  |  |  |  |
| `s-klt:v1` | `val_select-n100` | `summary-power_nbi` | cache | 8 | 8 | 90 | 0.01705 | 0.01473 | -0.158 | -0.201 | -0.119 | 0.136 | 0.106 | 0.167 | b clearly beats a |  |  |  |  |
| `s-klt:o-klt` | `val_select-n100` | `gas_injection-total_injected` | cache | 1 | 1 | 88 | 0.007792 | 0.007902 | 0.014 | -0.034 | 0.053 | -0.014 | -0.056 | 0.032 | tie |  |  |  |  |
| `s-klt:o-klt` | `val_select-n100` | `interferometer-n_e_line` | cache | 16 | 16 | 91 | 0.2267 | 0.2267 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | tie |  |  |  |  |
| `s-klt:o-klt` | `val_select-n100` | `magnetics-b_field_pol_probe_ccbv_field` | cache | 160 | 160 | 100 | 0.01627 | 0.01314 | -0.238 | -0.286 | -0.194 | 0.193 | 0.162 | 0.222 | b clearly beats a | 1.000 | 1.000 | 160 | 160 |
| `s-klt:o-klt` | `val_select-n100` | `magnetics-b_field_pol_probe_obr_field` | cache | 18 | 18 | 100 | 0.01596 | 0.01585 | -0.007 | -0.009 | -0.005 | 0.007 | 0.005 | 0.009 | b beats a | 1.000 | 1.000 | 18 | 18 |
| `s-klt:o-klt` | `val_select-n100` | `magnetics-b_field_pol_probe_obv_field` | cache | 36 | 36 | 100 | 0.01861 | 0.01223 | -0.521 | -0.546 | -0.496 | 0.343 | 0.331 | 0.353 | b clearly beats a | 1.000 | 1.000 | 72 | 36 |
| `s-klt:o-klt` | `val_select-n100` | `magnetics-b_field_tor_probe_saddle_voltage` | cache | 12 | 12 | 98 | 0.01311 | 0.01278 | -0.026 | -0.033 | -0.020 | 0.025 | 0.019 | 0.032 | b beats a | 1.000 | 1.000 | 12 | 12 |
| `s-klt:o-klt` | `val_select-n100` | `magnetics-flux_loop_flux` | cache | 30 | 30 | 100 | 0.01221 | 0.007112 | -0.717 | -0.751 | -0.684 | 0.418 | 0.406 | 0.429 | b clearly beats a | 1.000 | 1.000 | 30 | 30 |
| `s-klt:o-klt` | `val_select-n100` | `pf_active-coil_current` | task_2-1 | 10 | 10 | 89 | 0.01718 | 0.01714 | -0.003 | -0.011 | 0.005 | 0.003 | -0.005 | 0.011 | tie |  |  |  |  |
| `s-klt:o-klt` | `val_select-n100` | `pf_active-coil_voltage` | cache | 16 | 16 | 99 | 0.0168 | 0.01597 | -0.052 | -0.091 | -0.043 | 0.050 | 0.041 | 0.083 | b beats a | 1.000 | 1.000 | 16 | 16 |
| `s-klt:o-klt` | `val_select-n100` | `pf_active-solenoid_current` | task_2-1 | 1 | 1 | 89 | 0.01362 | 0.01402 | 0.028 | 0.018 | 0.040 | -0.029 | -0.041 | -0.019 | a beats b |  |  |  |  |
| `s-klt:o-klt` | `val_select-n100` | `pulse_schedule-i_plasma` | cache | 2 | 2 | 98 | 0.01329 | 0.00989 | -0.344 | -0.452 | -0.267 | 0.256 | 0.210 | 0.311 | b clearly beats a |  |  |  |  |
| `s-klt:o-klt` | `val_select-n100` | `pulse_schedule-n_e_line` | cache | 1 | 1 | 99 | 0.01176 | 0.009983 | -0.178 | -0.288 | -0.101 | 0.151 | 0.092 | 0.224 | b clearly beats a |  |  |  |  |
| `s-klt:o-klt` | `val_select-n100` | `soft_x_rays-horizontal_cam_lower` | task_4-1 | 36 | 36 | 95 | 0.0519 | 0.05067 | -0.024 | -0.035 | -0.016 | 0.024 | 0.016 | 0.034 | b beats a |  |  |  |  |
| `s-klt:o-klt` | `val_select-n100` | `soft_x_rays-horizontal_cam_upper` | task_4-1 | 72 | 72 | 95 | 0.04907 | 0.04804 | -0.021 | -0.033 | -0.014 | 0.021 | 0.014 | 0.032 | b beats a |  |  |  |  |
| `s-klt:o-klt` | `val_select-n100` | `spectrometer_visible-filter_spectrometer_dalpha_voltage` | task_3-2 | 6 | 6 | 95 | 0.15 | 0.143 | -0.049 | -0.058 | -0.041 | 0.046 | 0.039 | 0.055 | b beats a |  |  |  |  |
| `s-klt:o-klt` | `val_select-n100` | `summary-ip` | task_2-1 | 4 | 4 | 89 | 0.01894 | 0.01468 | -0.291 | -0.321 | -0.261 | 0.225 | 0.207 | 0.243 | b clearly beats a |  |  |  |  |
| `s-klt:o-klt` | `val_select-n100` | `summary-power_nbi` | cache | 8 | 8 | 90 | 0.01705 | 0.01473 | -0.158 | -0.201 | -0.119 | 0.136 | 0.106 | 0.167 | b clearly beats a |  |  |  |  |
| `c-pca:o-klt` | `val_select-n100` | `magnetics-b_field_pol_probe_ccbv_field` | cache | 160 | 160 | 100 | 0.005814 | 0.01314 | 0.558 | 0.519 | 0.590 | -1.260 | -1.438 | -1.078 | a clearly beats b | 1.072 | 1.000 | 40 | 160 |
| `c-pca:o-klt` | `val_select-n100` | `magnetics-b_field_pol_probe_obr_field` | cache | 18 | 18 | 100 | 0.008377 | 0.01585 | 0.472 | 0.438 | 0.501 | -0.893 | -1.004 | -0.781 | a clearly beats b | 1.011 | 1.000 | 18 | 18 |
| `c-pca:o-klt` | `val_select-n100` | `magnetics-b_field_pol_probe_obv_field` | cache | 36 | 36 | 100 | 0.006787 | 0.01223 | 0.445 | 0.421 | 0.467 | -0.803 | -0.876 | -0.728 | a clearly beats b | 1.000 | 1.000 | 36 | 36 |
| `c-pca:o-klt` | `val_select-n100` | `magnetics-flux_loop_flux` | cache | 30 | 30 | 100 | 0.002548 | 0.007112 | 0.642 | 0.629 | 0.655 | -1.792 | -1.901 | -1.692 | a clearly beats b | 3.518 | 1.000 | 12 | 30 |
| `c-pca:s-klt` | `val_select-n100` | `gas_injection-total_injected` | cache | 1 | 1 | 88 | 0.007902 | 0.007792 | -0.014 | -0.056 | 0.032 | 0.014 | -0.034 | 0.053 | tie |  |  |  |  |
| `c-pca:s-klt` | `val_select-n100` | `interferometer-n_e_line` | cache | 16 | 16 | 91 | 0.2267 | 0.2267 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | tie |  |  |  |  |
| `c-pca:s-klt` | `val_select-n100` | `magnetics-b_field_pol_probe_ccbv_field` | cache | 160 | 160 | 100 | 0.005814 | 0.01627 | 0.643 | 0.603 | 0.673 | -1.799 | -2.062 | -1.521 | a clearly beats b | 1.072 | 1.000 | 40 | 160 |
| `c-pca:s-klt` | `val_select-n100` | `magnetics-b_field_pol_probe_obr_field` | cache | 18 | 18 | 100 | 0.008377 | 0.01596 | 0.475 | 0.442 | 0.504 | -0.906 | -1.018 | -0.793 | a clearly beats b | 1.011 | 1.000 | 18 | 18 |
| `c-pca:s-klt` | `val_select-n100` | `magnetics-b_field_pol_probe_obv_field` | cache | 36 | 36 | 100 | 0.006787 | 0.01861 | 0.635 | 0.620 | 0.649 | -1.743 | -1.852 | -1.628 | a clearly beats b | 1.000 | 1.000 | 36 | 72 |
| `c-pca:s-klt` | `val_select-n100` | `magnetics-b_field_tor_probe_saddle_voltage` | cache | 12 | 12 | 98 | 0.01278 | 0.01311 | 0.025 | 0.019 | 0.032 | -0.026 | -0.033 | -0.020 | a beats b | 1.000 | 1.000 | 12 | 12 |
| `c-pca:s-klt` | `val_select-n100` | `magnetics-flux_loop_flux` | cache | 30 | 30 | 100 | 0.002548 | 0.01221 | 0.791 | 0.784 | 0.799 | -3.793 | -3.970 | -3.628 | a clearly beats b | 3.518 | 1.000 | 12 | 30 |
| `c-pca:s-klt` | `val_select-n100` | `pf_active-coil_current` | task_2-1 | 10 | 10 | 89 | 0.01714 | 0.01718 | 0.003 | -0.005 | 0.011 | -0.003 | -0.011 | 0.005 | tie |  |  |  |  |
| `c-pca:s-klt` | `val_select-n100` | `pf_active-coil_voltage` | cache | 16 | 16 | 99 | 0.01597 | 0.0168 | 0.050 | 0.041 | 0.083 | -0.052 | -0.091 | -0.043 | a beats b | 1.000 | 1.000 | 16 | 16 |
| `c-pca:s-klt` | `val_select-n100` | `pf_active-solenoid_current` | task_2-1 | 1 | 1 | 89 | 0.01402 | 0.01362 | -0.029 | -0.041 | -0.019 | 0.028 | 0.018 | 0.040 | b beats a |  |  |  |  |
| `c-pca:s-klt` | `val_select-n100` | `pulse_schedule-i_plasma` | cache | 2 | 2 | 98 | 0.00989 | 0.01329 | 0.256 | 0.210 | 0.311 | -0.344 | -0.452 | -0.267 | a clearly beats b |  |  |  |  |
| `c-pca:s-klt` | `val_select-n100` | `pulse_schedule-n_e_line` | cache | 1 | 1 | 99 | 0.009983 | 0.01176 | 0.151 | 0.092 | 0.224 | -0.178 | -0.288 | -0.101 | a clearly beats b |  |  |  |  |
| `c-pca:s-klt` | `val_select-n100` | `soft_x_rays-horizontal_cam_lower` | task_4-1 | 36 | 36 | 95 | 0.05067 | 0.0519 | 0.024 | 0.016 | 0.034 | -0.024 | -0.035 | -0.016 | a beats b |  |  |  |  |
| `c-pca:s-klt` | `val_select-n100` | `soft_x_rays-horizontal_cam_upper` | task_4-1 | 72 | 72 | 95 | 0.04804 | 0.04907 | 0.021 | 0.014 | 0.032 | -0.021 | -0.033 | -0.014 | a beats b |  |  |  |  |
| `c-pca:s-klt` | `val_select-n100` | `spectrometer_visible-filter_spectrometer_dalpha_voltage` | task_3-2 | 6 | 6 | 95 | 0.143 | 0.15 | 0.046 | 0.039 | 0.055 | -0.049 | -0.058 | -0.041 | a beats b |  |  |  |  |
| `c-pca:s-klt` | `val_select-n100` | `summary-ip` | task_2-1 | 4 | 4 | 89 | 0.01468 | 0.01894 | 0.225 | 0.207 | 0.243 | -0.291 | -0.321 | -0.261 | a clearly beats b |  |  |  |  |
| `c-pca:s-klt` | `val_select-n100` | `summary-power_nbi` | cache | 8 | 8 | 90 | 0.01473 | 0.01705 | 0.136 | 0.106 | 0.167 | -0.158 | -0.201 | -0.119 | a clearly beats b |  |  |  |  |
| `c-pca.h:s-klt` | `val_select-n100` | `gas_injection-total_injected` | cache | 1 | 1 | 88 | 0.007902 | 0.007792 | -0.014 | -0.056 | 0.032 | 0.014 | -0.034 | 0.053 | tie |  |  |  |  |
| `c-pca.h:s-klt` | `val_select-n100` | `interferometer-n_e_line` | cache | 16 | 16 | 91 | 0.2267 | 0.2267 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | tie |  |  |  |  |
| `c-pca.h:s-klt` | `val_select-n100` | `magnetics-b_field_pol_probe_ccbv_field` | cache | 80 | 160 | 100 | 0.009392 | 0.01627 | 0.423 | 0.386 | 0.455 | -0.733 | -0.834 | -0.628 | a clearly beats b | 1.219 | 1.000 | 40 | 160 |
| `c-pca.h:s-klt` | `val_select-n100` | `magnetics-b_field_pol_probe_obr_field` | cache | 6 | 18 | 100 | 0.06278 | 0.01596 | -2.933 | -3.498 | -2.466 | 0.746 | 0.711 | 0.778 | b clearly beats a | 1.170 | 1.000 |  | 18 |
| `c-pca.h:s-klt` | `val_select-n100` | `magnetics-b_field_pol_probe_obv_field` | cache | 18 | 36 | 100 | 0.01819 | 0.01861 | 0.023 | -0.026 | 0.066 | -0.023 | -0.071 | 0.025 | tie | 1.000 | 1.000 | 36 | 72 |
| `c-pca.h:s-klt` | `val_select-n100` | `magnetics-b_field_tor_probe_saddle_voltage` | cache | 12 | 12 | 98 | 0.01278 | 0.01311 | 0.025 | 0.019 | 0.032 | -0.026 | -0.033 | -0.020 | a beats b | 1.000 | 1.000 | 12 | 12 |
| `c-pca.h:s-klt` | `val_select-n100` | `magnetics-flux_loop_flux` | cache | 12 | 30 | 100 | 0.01451 | 0.01221 | -0.188 | -0.243 | -0.132 | 0.158 | 0.116 | 0.196 | b clearly beats a | 7.747 | 1.000 | 12 | 30 |
| `c-pca.h:s-klt` | `val_select-n100` | `pf_active-coil_current` | task_2-1 | 10 | 10 | 89 | 0.01714 | 0.01718 | 0.003 | -0.005 | 0.011 | -0.003 | -0.011 | 0.005 | tie |  |  |  |  |
| `c-pca.h:s-klt` | `val_select-n100` | `pf_active-coil_voltage` | cache | 16 | 16 | 99 | 0.01597 | 0.0168 | 0.050 | 0.041 | 0.083 | -0.052 | -0.091 | -0.043 | a beats b | 1.000 | 1.000 | 16 | 16 |
| `c-pca.h:s-klt` | `val_select-n100` | `pf_active-solenoid_current` | task_2-1 | 1 | 1 | 89 | 0.01402 | 0.01362 | -0.029 | -0.041 | -0.019 | 0.028 | 0.018 | 0.040 | b beats a |  |  |  |  |
| `c-pca.h:s-klt` | `val_select-n100` | `pulse_schedule-i_plasma` | cache | 2 | 2 | 98 | 0.00989 | 0.01329 | 0.256 | 0.210 | 0.311 | -0.344 | -0.452 | -0.267 | a clearly beats b |  |  |  |  |
| `c-pca.h:s-klt` | `val_select-n100` | `pulse_schedule-n_e_line` | cache | 1 | 1 | 99 | 0.009983 | 0.01176 | 0.151 | 0.092 | 0.224 | -0.178 | -0.288 | -0.101 | a clearly beats b |  |  |  |  |
| `c-pca.h:s-klt` | `val_select-n100` | `soft_x_rays-horizontal_cam_lower` | task_4-1 | 36 | 36 | 95 | 0.05067 | 0.0519 | 0.024 | 0.016 | 0.034 | -0.024 | -0.035 | -0.016 | a beats b |  |  |  |  |
| `c-pca.h:s-klt` | `val_select-n100` | `soft_x_rays-horizontal_cam_upper` | task_4-1 | 72 | 72 | 95 | 0.04804 | 0.04907 | 0.021 | 0.014 | 0.032 | -0.021 | -0.033 | -0.014 | a beats b |  |  |  |  |
| `c-pca.h:s-klt` | `val_select-n100` | `spectrometer_visible-filter_spectrometer_dalpha_voltage` | task_3-2 | 6 | 6 | 95 | 0.143 | 0.15 | 0.046 | 0.039 | 0.055 | -0.049 | -0.058 | -0.041 | a beats b |  |  |  |  |
| `c-pca.h:s-klt` | `val_select-n100` | `summary-ip` | task_2-1 | 4 | 4 | 89 | 0.01468 | 0.01894 | 0.225 | 0.207 | 0.243 | -0.291 | -0.321 | -0.261 | a clearly beats b |  |  |  |  |
| `c-pca.h:s-klt` | `val_select-n100` | `summary-power_nbi` | cache | 8 | 8 | 90 | 0.01473 | 0.01705 | 0.136 | 0.106 | 0.167 | -0.158 | -0.201 | -0.119 | a clearly beats b |  |  |  |  |
| `c-pca.q:s-klt` | `val_select-n100` | `gas_injection-total_injected` | cache | 1 | 1 | 88 | 0.007902 | 0.007792 | -0.014 | -0.056 | 0.032 | 0.014 | -0.034 | 0.053 | tie |  |  |  |  |
| `c-pca.q:s-klt` | `val_select-n100` | `interferometer-n_e_line` | cache | 16 | 16 | 91 | 0.2267 | 0.2267 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | tie |  |  |  |  |
| `c-pca.q:s-klt` | `val_select-n100` | `magnetics-b_field_pol_probe_ccbv_field` | cache | 40 | 160 | 100 | 0.01703 | 0.01627 | -0.047 | -0.109 | 0.010 | 0.045 | -0.010 | 0.098 | tie | 1.087 | 1.000 | 40 | 160 |
| `c-pca.q:s-klt` | `val_select-n100` | `magnetics-b_field_pol_probe_obr_field` | cache | 6 | 18 | 100 | 0.06278 | 0.01596 | -2.933 | -3.498 | -2.466 | 0.746 | 0.711 | 0.778 | b clearly beats a | 1.170 | 1.000 |  | 18 |
| `c-pca.q:s-klt` | `val_select-n100` | `magnetics-b_field_pol_probe_obv_field` | cache | 6 | 36 | 100 | 0.1063 | 0.01861 | -4.710 | -5.473 | -4.035 | 0.825 | 0.801 | 0.846 | b clearly beats a | 1.000 | 1.000 |  | 72 |
| `c-pca.q:s-klt` | `val_select-n100` | `magnetics-b_field_tor_probe_saddle_voltage` | cache | 12 | 12 | 98 | 0.01278 | 0.01311 | 0.025 | 0.019 | 0.032 | -0.026 | -0.033 | -0.020 | a beats b | 1.000 | 1.000 | 12 | 12 |
| `c-pca.q:s-klt` | `val_select-n100` | `magnetics-flux_loop_flux` | cache | 6 | 30 | 100 | 0.06952 | 0.01221 | -4.693 | -5.229 | -4.199 | 0.824 | 0.808 | 0.839 | b clearly beats a | 1.893 | 1.000 | 12 | 30 |
| `c-pca.q:s-klt` | `val_select-n100` | `pf_active-coil_current` | task_2-1 | 10 | 10 | 89 | 0.01714 | 0.01718 | 0.003 | -0.005 | 0.011 | -0.003 | -0.011 | 0.005 | tie |  |  |  |  |
| `c-pca.q:s-klt` | `val_select-n100` | `pf_active-coil_voltage` | cache | 16 | 16 | 99 | 0.01597 | 0.0168 | 0.050 | 0.041 | 0.083 | -0.052 | -0.091 | -0.043 | a beats b | 1.000 | 1.000 | 16 | 16 |
| `c-pca.q:s-klt` | `val_select-n100` | `pf_active-solenoid_current` | task_2-1 | 1 | 1 | 89 | 0.01402 | 0.01362 | -0.029 | -0.041 | -0.019 | 0.028 | 0.018 | 0.040 | b beats a |  |  |  |  |
| `c-pca.q:s-klt` | `val_select-n100` | `pulse_schedule-i_plasma` | cache | 2 | 2 | 98 | 0.00989 | 0.01329 | 0.256 | 0.210 | 0.311 | -0.344 | -0.452 | -0.267 | a clearly beats b |  |  |  |  |
| `c-pca.q:s-klt` | `val_select-n100` | `pulse_schedule-n_e_line` | cache | 1 | 1 | 99 | 0.009983 | 0.01176 | 0.151 | 0.092 | 0.224 | -0.178 | -0.288 | -0.101 | a clearly beats b |  |  |  |  |
| `c-pca.q:s-klt` | `val_select-n100` | `soft_x_rays-horizontal_cam_lower` | task_4-1 | 36 | 36 | 95 | 0.05067 | 0.0519 | 0.024 | 0.016 | 0.034 | -0.024 | -0.035 | -0.016 | a beats b |  |  |  |  |
| `c-pca.q:s-klt` | `val_select-n100` | `soft_x_rays-horizontal_cam_upper` | task_4-1 | 72 | 72 | 95 | 0.04804 | 0.04907 | 0.021 | 0.014 | 0.032 | -0.021 | -0.033 | -0.014 | a beats b |  |  |  |  |
| `c-pca.q:s-klt` | `val_select-n100` | `spectrometer_visible-filter_spectrometer_dalpha_voltage` | task_3-2 | 6 | 6 | 95 | 0.143 | 0.15 | 0.046 | 0.039 | 0.055 | -0.049 | -0.058 | -0.041 | a beats b |  |  |  |  |
| `c-pca.q:s-klt` | `val_select-n100` | `summary-ip` | task_2-1 | 4 | 4 | 89 | 0.01468 | 0.01894 | 0.225 | 0.207 | 0.243 | -0.291 | -0.321 | -0.261 | a clearly beats b |  |  |  |  |
| `c-pca.q:s-klt` | `val_select-n100` | `summary-power_nbi` | cache | 8 | 8 | 90 | 0.01473 | 0.01705 | 0.136 | 0.106 | 0.167 | -0.158 | -0.201 | -0.119 | a clearly beats b |  |  |  |  |
| `c-pca:v1` | `val_select-n100` | `magnetics-b_field_pol_probe_ccbv_field` | cache | 160 | 160 | 100 | 0.005814 | 0.01327 | 0.562 | 0.522 | 0.595 | -1.282 | -1.471 | -1.094 | a clearly beats b | 1.072 | 1.000 | 40 | 160 |
| `c-pca:v1` | `val_select-n100` | `magnetics-b_field_pol_probe_obr_field` | cache | 18 | 18 | 100 | 0.008377 | 0.01588 | 0.473 | 0.439 | 0.502 | -0.896 | -1.008 | -0.783 | a clearly beats b | 1.011 | 1.000 | 18 | 18 |
| `c-pca:v1` | `val_select-n100` | `magnetics-b_field_pol_probe_obv_field` | cache | 36 | 36 | 100 | 0.006787 | 0.01242 | 0.454 | 0.430 | 0.475 | -0.831 | -0.906 | -0.755 | a clearly beats b | 1.000 | 1.000 | 36 | 36 |
| `c-pca:v1` | `val_select-n100` | `magnetics-flux_loop_flux` | cache | 30 | 30 | 100 | 0.002548 | 0.007239 | 0.648 | 0.635 | 0.661 | -1.841 | -1.950 | -1.740 | a clearly beats b | 3.518 | 1.000 | 12 | 30 |
| `b-klt:v1` | `val_select-n100` | `magnetics-b_field_pol_probe_omv_voltage` | task_4-5 | 432 | 432 | 62 | 0.4028 | 0.403 | 0.001 | 0.000 | 0.001 | -0.001 | -0.001 | 0.000 | tie |  |  |  |  |
| `b-klt:v1` | `val_select-n100` | `magnetics-b_field_tor_probe_cc_field` | task_4-5 | 432 | 432 | 62 | 0.6267 | 0.6267 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | tie |  |  |  |  |
| `b-klt:bo-klt` | `val_select-n100` | `magnetics-b_field_pol_probe_omv_voltage` | task_4-5 | 432 | 432 | 62 | 0.4028 | 0.4028 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | tie |  |  |  |  |
| `b-klt:bo-klt` | `val_select-n100` | `magnetics-b_field_tor_probe_cc_field` | task_4-5 | 432 | 432 | 62 | 0.6267 | 0.6267 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | tie |  |  |  |  |
| `bc-pca:bo-klt` | `val_select-n100` | `magnetics-b_field_pol_probe_omv_voltage` | task_4-5 | 432 | 432 | 62 | 0.4026 | 0.4028 | 0.000 | -0.001 | 0.002 | 0.000 | -0.002 | 0.001 | tie |  |  |  |  |
| `bc-pca:bo-klt` | `val_select-n100` | `magnetics-b_field_tor_probe_cc_field` | task_4-5 | 432 | 432 | 62 | 0.631 | 0.6267 | -0.007 | -0.017 | -0.001 | 0.007 | 0.001 | 0.017 | b beats a |  |  |  |  |
| `bc-pca:b-klt` | `val_select-n100` | `magnetics-b_field_pol_probe_omv_voltage` | task_4-5 | 432 | 432 | 62 | 0.4026 | 0.4028 | 0.000 | -0.001 | 0.002 | 0.000 | -0.002 | 0.001 | tie |  |  |  |  |
| `bc-pca:b-klt` | `val_select-n100` | `magnetics-b_field_tor_probe_cc_field` | task_4-5 | 432 | 432 | 62 | 0.631 | 0.6267 | -0.007 | -0.017 | -0.001 | 0.007 | 0.001 | 0.017 | b beats a |  |  |  |  |
| `bc-pca.h:b-klt` | `val_select-n100` | `magnetics-b_field_pol_probe_omv_voltage` | task_4-5 | 216 | 432 | 62 | 0.4097 | 0.4028 | -0.017 | -0.035 | -0.005 | 0.017 | 0.005 | 0.034 | b beats a |  |  |  |  |
| `bc-pca.h:b-klt` | `val_select-n100` | `magnetics-b_field_tor_probe_cc_field` | task_4-5 | 216 | 432 | 62 | 0.6307 | 0.6267 | -0.006 | -0.013 | -0.002 | 0.006 | 0.002 | 0.013 | b beats a |  |  |  |  |
| `bc-pca.q:b-klt` | `val_select-n100` | `magnetics-b_field_pol_probe_omv_voltage` | task_4-5 | 108 | 432 | 62 | 0.4223 | 0.4028 | -0.048 | -0.084 | -0.020 | 0.046 | 0.020 | 0.078 | b beats a |  |  |  |  |
| `bc-pca.q:b-klt` | `val_select-n100` | `magnetics-b_field_tor_probe_cc_field` | task_4-5 | 108 | 432 | 62 | 0.6311 | 0.6267 | -0.007 | -0.012 | -0.004 | 0.007 | 0.004 | 0.012 | b beats a |  |  |  |  |
| `s-klt.b:v1b` | `val_select-n100` | `pf_active-coil_current` | task_2-1 | 20 | 20 | 89 | 0.009755 | 0.007586 | -0.286 | -0.329 | -0.248 | 0.222 | 0.199 | 0.248 | b clearly beats a |  |  |  |  |
| `s-klt.b:v1b` | `val_select-n100` | `pf_active-solenoid_current` | task_2-1 | 2 | 2 | 89 | 0.005913 | 0.002212 | -1.673 | -1.733 | -1.615 | 0.626 | 0.618 | 0.634 | b clearly beats a |  |  |  |  |
| `c-pca:s-klt` | `val_report` | `magnetics-b_field_pol_probe_ccbv_field` | cache | 160 | 160 | 557 | 0.005597 | 0.01526 | 0.633 | 0.622 | 0.643 | -1.727 | -1.803 | -1.647 | a clearly beats b | 1.069 | 1.000 | 40 | 160 |
| `c-pca:s-klt` | `val_report` | `magnetics-b_field_pol_probe_obr_field` | cache | 18 | 18 | 557 | 0.007949 | 0.01513 | 0.474 | 0.457 | 0.489 | -0.903 | -0.958 | -0.840 | a clearly beats b | 1.009 | 1.000 | 18 | 18 |
| `c-pca:s-klt` | `val_report` | `magnetics-b_field_pol_probe_obv_field` | cache | 36 | 36 | 557 | 0.006573 | 0.0178 | 0.631 | 0.624 | 0.637 | -1.709 | -1.753 | -1.662 | a clearly beats b | 1.000 | 1.000 | 36 | 72 |
| `c-pca:s-klt` | `val_report` | `magnetics-flux_loop_flux` | cache | 30 | 30 | 557 | 0.002426 | 0.01145 | 0.788 | 0.785 | 0.791 | -3.718 | -3.781 | -3.655 | a clearly beats b | 3.510 | 1.000 | 12 | 30 |
| `c-pca:o-klt` | `val_report` | `magnetics-b_field_pol_probe_ccbv_field` | cache | 160 | 160 | 557 | 0.005597 | 0.01249 | 0.552 | 0.541 | 0.561 | -1.231 | -1.280 | -1.181 | a clearly beats b | 1.069 | 1.000 | 40 | 160 |
| `c-pca:o-klt` | `val_report` | `magnetics-b_field_pol_probe_obr_field` | cache | 18 | 18 | 557 | 0.007949 | 0.01501 | 0.470 | 0.452 | 0.485 | -0.888 | -0.943 | -0.826 | a clearly beats b | 1.009 | 1.000 | 18 | 18 |
| `c-pca:o-klt` | `val_report` | `magnetics-b_field_pol_probe_obv_field` | cache | 36 | 36 | 557 | 0.006573 | 0.01184 | 0.445 | 0.435 | 0.455 | -0.802 | -0.834 | -0.770 | a clearly beats b | 1.000 | 1.000 | 36 | 36 |
| `c-pca:o-klt` | `val_report` | `magnetics-flux_loop_flux` | cache | 30 | 30 | 557 | 0.002426 | 0.006565 | 0.630 | 0.625 | 0.636 | -1.706 | -1.750 | -1.664 | a clearly beats b | 3.510 | 1.000 | 12 | 30 |
| `c-pca:v1` | `val_report` | `magnetics-b_field_pol_probe_ccbv_field` | cache | 160 | 160 | 557 | 0.005597 | 0.01271 | 0.560 | 0.549 | 0.570 | -1.272 | -1.323 | -1.219 | a clearly beats b | 1.069 | 1.000 | 40 | 160 |
| `c-pca:v1` | `val_report` | `magnetics-b_field_pol_probe_obr_field` | cache | 18 | 18 | 557 | 0.007949 | 0.01503 | 0.471 | 0.453 | 0.486 | -0.891 | -0.947 | -0.829 | a clearly beats b | 1.009 | 1.000 | 18 | 18 |
| `c-pca:v1` | `val_report` | `magnetics-b_field_pol_probe_obv_field` | cache | 36 | 36 | 557 | 0.006573 | 0.01204 | 0.454 | 0.444 | 0.463 | -0.832 | -0.864 | -0.799 | a clearly beats b | 1.000 | 1.000 | 36 | 36 |
| `c-pca:v1` | `val_report` | `magnetics-flux_loop_flux` | cache | 30 | 30 | 557 | 0.002426 | 0.006687 | 0.637 | 0.632 | 0.643 | -1.757 | -1.802 | -1.714 | a clearly beats b | 3.510 | 1.000 | 12 | 30 |
| `o-klt:s-klt` | `val_report` | `magnetics-b_field_pol_probe_ccbv_field` | cache | 160 | 160 | 557 | 0.01249 | 0.01526 | 0.182 | 0.167 | 0.196 | -0.222 | -0.244 | -0.200 | a clearly beats b | 1.000 | 1.000 | 160 | 160 |
| `o-klt:s-klt` | `val_report` | `magnetics-b_field_pol_probe_obr_field` | cache | 18 | 18 | 557 | 0.01501 | 0.01513 | 0.008 | 0.007 | 0.009 | -0.008 | -0.009 | -0.007 | a beats b | 1.000 | 1.000 | 18 | 18 |
| `o-klt:s-klt` | `val_report` | `magnetics-b_field_pol_probe_obv_field` | cache | 36 | 36 | 557 | 0.01184 | 0.0178 | 0.335 | 0.330 | 0.339 | -0.503 | -0.513 | -0.493 | a clearly beats b | 1.000 | 1.000 | 36 | 72 |
| `o-klt:s-klt` | `val_report` | `magnetics-flux_loop_flux` | cache | 30 | 30 | 557 | 0.006565 | 0.01145 | 0.426 | 0.421 | 0.432 | -0.744 | -0.759 | -0.728 | a clearly beats b | 1.000 | 1.000 | 30 | 30 |

An empty "Missing-channel ratio" or "Values to reach 0.0176" cell is a quantity the comparison does not compute
for that row: `--drop-channel` and `--reach` are computed only for a series signal of several channels read
from the cache (`_is_series_array` in `scripts/mast/run_tokenizer.py`: the four magnetics arrays, the saddle coils
and the coil voltages). They are not computed for a signal whose floor is read from a task, nor for a one-channel
signal (gas, line density, the two requested waveforms, beam power). On such a series signal an empty "Values to
reach 0.0176" cell means that no size of that arm, at 1/8, 1/4, 1/2, 1 and 2 times its values, reached 0.0176. A ratio of
1.000 for a per-channel arm is by construction, since a missing channel does not touch the others. A gap of
0.000 with a "tie" word is an exact re-basis (`interferometer-n_e_line`, ruling C11) or two arms whose floors
agree to 3 decimals (the Mirnov signals, whose floor is bounded by the frame oracle, ruling C17).

#### The networks

The same rows and columns as the linear arms, from the comparison files of M7 and
M8, with the gap of each seed and its spread added from `seeds`, which prints
`n_shots ...; gaps ...; spread ...; gaps b over a ...; spread ...; word` per signal and
writes nothing, so those columns are copied from its print. The spread is the
largest gap over the seeds minus the smallest (rule 2): that of the gaps for "a beats
b" and that of the gaps of b against a for "b beats a" (the two differ, since the
former gap is over the mean error of b and the latter over that of a). The word is the
verdict of the first pair with those spreads, and "shots differ" instead of a word
when the pairs were measured over different numbers of shots.

Measured 2026-11-23 on 100 `val_select` shots (62 to 100 scored per signal; the "Shots" column): the six
pairs of job M7 (08:04 UTC, networks at learning rate 3e-4) and the eight pairs of job M8 (08:22 to 08:23 UTC;
names ending in `r` are the learning rate 1e-4 retries of ruling C41), and the pair `c-net0rl:c-pca` of ruling
C45 (08:57 UTC, its own comparison, run after the review of the page). 15 pairs,
50 rows. The networks behind the names, after ruling C39: `o-net0` is `net.mag.s0`; `c-net0` is
`net.ccbv.s0.long`, `net.obr.s0.long`, `net.obv.s0.long` and `net.flux.s0`; `bc-net0` is `net.bandcat.cc.s0` and
`net.bandcat.omv.s0.long`; `p-net0` is `net.profile.s0.long`; `m-net0` is `net.map.s0.long`. The `r` assignments
are built from the lr1e4 runs of the same arms. `c-net0rl` (ruling C45, `$S1/assignments/c-net0rl.json`) is
`c-net0r` with the flux recoder replaced by `net.flux.s0.lr1e4_long`; its other three arrays use the same recoders as
`c-net0r`, which is why their rows in `c-net0rl:c-pca` equal those of `c-net0r:c-pca`.

| Pair (a:b) | Signal | Source | Values per frame (a) | Values per frame (b) | Shots | Mean error (a) | Mean error (b) | Gap | Gap, low end | Gap, high end | Gap of b against a | Low end (b against a) | High end (b against a) | Word at one seed | Missing-channel ratio (a) | Missing-channel ratio (b) | Values to reach 0.0176 (a) | Values to reach 0.0176 (b) | Gap, seed 0 | Gap, seed 1 | Gap, seed 2 | Spread of the gaps | Spread of the gaps of b against a | Word with the spreads |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `c-net0:o-net0` | `magnetics-b_field_pol_probe_ccbv_field` | cache | 160 | 160 | 100 | 0.01828 | 0.03766 | 0.515 | 0.446 | 0.566 | -1.061 | -1.306 | -0.804 | a clearly beats b | 2.120 | 1.000 | 320 |  |  |  |  |  |  |  |
| `c-net0:o-net0` | `magnetics-b_field_pol_probe_obr_field` | cache | 18 | 18 | 100 | 0.01892 | 0.0289 | 0.345 | 0.227 | 0.450 | -0.527 | -0.817 | -0.293 | a clearly beats b | 1.798 | 1.000 | 36 |  |  |  |  |  |  |  |
| `c-net0:o-net0` | `magnetics-b_field_pol_probe_obv_field` | cache | 36 | 36 | 100 | 0.0187 | 0.03788 | 0.506 | 0.321 | 0.610 | -1.025 | -1.562 | -0.472 | a clearly beats b | 1.000 | 1.000 | 72 |  |  |  |  |  |  |  |
| `c-net0:o-net0` | `magnetics-flux_loop_flux` | cache | 30 | 30 | 100 | 0.01709 | 0.03583 | 0.523 | 0.493 | 0.552 | -1.097 | -1.230 | -0.973 | a clearly beats b | 11.305 | 1.000 | 30 |  |  |  |  |  |  |  |
| `c-net0:v1` | `magnetics-b_field_pol_probe_ccbv_field` | cache | 160 | 160 | 100 | 0.01828 | 0.01327 | -0.377 | -0.589 | -0.227 | 0.274 | 0.185 | 0.371 | b clearly beats a | 2.120 | 1.000 | 320 | 160 |  |  |  |  |  |  |
| `c-net0:v1` | `magnetics-b_field_pol_probe_obr_field` | cache | 18 | 18 | 100 | 0.01892 | 0.01588 | -0.191 | -0.357 | -0.059 | 0.161 | 0.055 | 0.263 | b clearly beats a | 1.798 | 1.000 | 36 | 18 |  |  |  |  |  |  |
| `c-net0:v1` | `magnetics-b_field_pol_probe_obv_field` | cache | 36 | 36 | 100 | 0.0187 | 0.01242 | -0.505 | -1.090 | -0.178 | 0.336 | 0.151 | 0.522 | b clearly beats a | 1.000 | 1.000 | 72 | 36 |  |  |  |  |  |  |
| `c-net0:v1` | `magnetics-flux_loop_flux` | cache | 30 | 30 | 100 | 0.01709 | 0.007239 | -1.360 | -1.560 | -1.178 | 0.576 | 0.541 | 0.609 | b clearly beats a | 11.305 | 1.000 | 30 | 30 |  |  |  |  |  |  |
| `c-net0:c-pca` | `magnetics-b_field_pol_probe_ccbv_field` | cache | 160 | 160 | 100 | 0.01828 | 0.005814 | -2.144 | -2.720 | -1.723 | 0.682 | 0.633 | 0.731 | b clearly beats a | 2.120 | 1.072 | 320 | 40 |  |  |  |  |  |  |
| `c-net0:c-pca` | `magnetics-b_field_pol_probe_obr_field` | cache | 18 | 18 | 100 | 0.01892 | 0.008377 | -1.259 | -1.584 | -0.997 | 0.557 | 0.499 | 0.613 | b clearly beats a | 1.798 | 1.011 | 36 | 18 |  |  |  |  |  |  |
| `c-net0:c-pca` | `magnetics-b_field_pol_probe_obv_field` | cache | 36 | 36 | 100 | 0.0187 | 0.006787 | -1.756 | -2.813 | -1.195 | 0.637 | 0.544 | 0.738 | b clearly beats a | 1.000 | 1.000 | 72 | 36 |  |  |  |  |  |  |
| `c-net0:c-pca` | `magnetics-flux_loop_flux` | cache | 30 | 30 | 100 | 0.01709 | 0.002548 | -5.706 | -6.168 | -5.285 | 0.851 | 0.841 | 0.860 | b clearly beats a | 11.305 | 3.518 | 30 | 12 |  |  |  |  |  |  |
| `bc-net0:bc-pca` | `magnetics-b_field_pol_probe_omv_voltage` | task_4-5 | 432 | 432 | 62 | 0.4114 | 0.4026 | -0.022 | -0.031 | -0.014 | 0.021 | 0.014 | 0.030 | b beats a |  |  |  |  |  |  |  |  |  |  |
| `bc-net0:bc-pca` | `magnetics-b_field_tor_probe_cc_field` | task_4-5 | 432 | 432 | 62 | 0.6418 | 0.631 | -0.017 | -0.028 | -0.008 | 0.017 | 0.007 | 0.027 | b beats a |  |  |  |  |  |  |  |  |  |  |
| `p-net0:v1` | `equilibrium-lcfs_r` | task_2-2 | 32 | 32 | 92 | 0.01333 | 0.007005 | -0.903 | -1.007 | -0.812 | 0.475 | 0.448 | 0.502 | b clearly beats a |  |  |  |  |  |  |  |  |  |  |
| `p-net0:v1` | `equilibrium-lcfs_z` | task_2-2 | 32 | 32 | 92 | 0.01211 | 0.006733 | -0.799 | -0.857 | -0.746 | 0.444 | 0.427 | 0.462 | b clearly beats a |  |  |  |  |  |  |  |  |  |  |
| `p-net0:v1` | `thomson_scattering-n_e` | task_3-3 | 32 | 32 | 76 | 0.1477 | 0.04245 | -2.480 | -2.759 | -2.207 | 0.713 | 0.688 | 0.734 | b clearly beats a |  |  |  |  |  |  |  |  |  |  |
| `p-net0:v1` | `thomson_scattering-t_e` | task_3-3 | 48 | 48 | 76 | 0.06265 | 0.03641 | -0.721 | -0.900 | -0.584 | 0.419 | 0.369 | 0.474 | b clearly beats a |  |  |  |  |  |  |  |  |  |  |
| `m-net0:v1` | `equilibrium-psi` | task_2-3 | 16 | 16 | 92 | 0.04613 | 0.01294 | -2.565 | -4.901 | -1.011 | 0.720 | 0.503 | 0.831 | b clearly beats a |  |  |  |  |  |  |  |  |  |  |
| `o-net0:o-klt` | `magnetics-b_field_pol_probe_ccbv_field` | cache | 160 | 160 | 100 | 0.03766 | 0.01314 | -1.866 | -2.011 | -1.732 | 0.651 | 0.634 | 0.668 | b clearly beats a | 1.000 | 1.000 |  | 160 |  |  |  |  |  |  |
| `o-net0:o-klt` | `magnetics-b_field_pol_probe_obr_field` | cache | 18 | 18 | 100 | 0.0289 | 0.01585 | -0.823 | -1.157 | -0.595 | 0.451 | 0.373 | 0.536 | b clearly beats a | 1.000 | 1.000 |  | 18 |  |  |  |  |  |  |
| `o-net0:o-klt` | `magnetics-b_field_pol_probe_obv_field` | cache | 36 | 36 | 100 | 0.03788 | 0.01223 | -2.096 | -2.195 | -2.000 | 0.677 | 0.667 | 0.687 | b clearly beats a | 1.000 | 1.000 |  | 36 |  |  |  |  |  |  |
| `o-net0:o-klt` | `magnetics-flux_loop_flux` | cache | 30 | 30 | 100 | 0.03583 | 0.007112 | -4.037 | -4.301 | -3.804 | 0.801 | 0.792 | 0.811 | b clearly beats a | 1.000 | 1.000 |  | 30 |  |  |  |  |  |  |
| `c-net0r:o-net0r` | `magnetics-b_field_pol_probe_ccbv_field` | cache | 160 | 160 | 100 | 0.01807 | 0.03384 | 0.466 | 0.405 | 0.510 | -0.873 | -1.040 | -0.681 | a clearly beats b | 2.168 | 1.000 | 320 |  |  |  |  |  |  |  |
| `c-net0r:o-net0r` | `magnetics-b_field_pol_probe_obr_field` | cache | 18 | 18 | 100 | 0.01782 | 0.03393 | 0.475 | 0.327 | 0.575 | -0.904 | -1.355 | -0.485 | a clearly beats b | 1.885 | 1.000 | 36 |  |  |  |  |  |  |  |
| `c-net0r:o-net0r` | `magnetics-b_field_pol_probe_obv_field` | cache | 36 | 36 | 100 | 0.01948 | 0.03723 | 0.477 | 0.302 | 0.576 | -0.912 | -1.358 | -0.433 | a clearly beats b | 1.000 | 1.000 | 72 |  |  |  |  |  |  |  |
| `c-net0r:o-net0r` | `magnetics-flux_loop_flux` | cache | 30 | 30 | 100 | 0.01573 | 0.03287 | 0.522 | 0.494 | 0.549 | -1.090 | -1.215 | -0.978 | a clearly beats b | 10.690 | 1.000 | 30 | 60 |  |  |  |  |  |  |
| `c-net0r:v1` | `magnetics-b_field_pol_probe_ccbv_field` | cache | 160 | 160 | 100 | 0.01807 | 0.01327 | -0.362 | -0.519 | -0.244 | 0.266 | 0.196 | 0.342 | b clearly beats a | 2.168 | 1.000 | 320 | 160 |  |  |  |  |  |  |
| `c-net0r:v1` | `magnetics-b_field_pol_probe_obr_field` | cache | 18 | 18 | 100 | 0.01782 | 0.01588 | -0.122 | -0.442 | 0.075 | 0.109 | -0.081 | 0.307 | tie | 1.885 | 1.000 | 36 | 18 |  |  |  |  |  |  |
| `c-net0r:v1` | `magnetics-b_field_pol_probe_obv_field` | cache | 36 | 36 | 100 | 0.01948 | 0.01242 | -0.568 | -1.116 | -0.251 | 0.362 | 0.200 | 0.527 | b clearly beats a | 1.000 | 1.000 | 72 | 36 |  |  |  |  |  |  |
| `c-net0r:v1` | `magnetics-flux_loop_flux` | cache | 30 | 30 | 100 | 0.01573 | 0.007239 | -1.172 | -1.349 | -1.009 | 0.540 | 0.502 | 0.574 | b clearly beats a | 10.690 | 1.000 | 30 | 30 |  |  |  |  |  |  |
| `c-net0r:c-pca` | `magnetics-b_field_pol_probe_ccbv_field` | cache | 160 | 160 | 100 | 0.01807 | 0.005814 | -2.108 | -2.577 | -1.743 | 0.678 | 0.635 | 0.720 | b clearly beats a | 2.168 | 1.072 | 320 | 40 |  |  |  |  |  |  |
| `c-net0r:c-pca` | `magnetics-b_field_pol_probe_obr_field` | cache | 18 | 18 | 100 | 0.01782 | 0.008377 | -1.127 | -1.731 | -0.747 | 0.530 | 0.427 | 0.634 | b clearly beats a | 1.885 | 1.011 | 36 | 18 |  |  |  |  |  |  |
| `c-net0r:c-pca` | `magnetics-b_field_pol_probe_obv_field` | cache | 36 | 36 | 100 | 0.01948 | 0.006787 | -1.870 | -2.866 | -1.332 | 0.652 | 0.571 | 0.741 | b clearly beats a | 1.000 | 1.000 | 72 | 36 |  |  |  |  |  |  |
| `c-net0r:c-pca` | `magnetics-flux_loop_flux` | cache | 30 | 30 | 100 | 0.01573 | 0.002548 | -5.173 | -5.592 | -4.792 | 0.838 | 0.827 | 0.848 | b clearly beats a | 10.690 | 3.518 | 30 | 12 |  |  |  |  |  |  |
| `bc-net0r:bc-pca` | `magnetics-b_field_pol_probe_omv_voltage` | task_4-5 | 432 | 432 | 62 | 0.4116 | 0.4026 | -0.022 | -0.031 | -0.015 | 0.022 | 0.015 | 0.030 | b beats a |  |  |  |  |  |  |  |  |  |  |
| `bc-net0r:bc-pca` | `magnetics-b_field_tor_probe_cc_field` | task_4-5 | 432 | 432 | 62 | 0.6388 | 0.631 | -0.012 | -0.022 | -0.001 | 0.012 | 0.001 | 0.022 | b beats a |  |  |  |  |  |  |  |  |  |  |
| `p-net0r:v1` | `equilibrium-lcfs_r` | task_2-2 | 32 | 32 | 92 | 0.01161 | 0.007005 | -0.657 | -0.712 | -0.610 | 0.397 | 0.379 | 0.416 | b clearly beats a |  |  |  |  |  |  |  |  |  |  |
| `p-net0r:v1` | `equilibrium-lcfs_z` | task_2-2 | 32 | 32 | 92 | 0.01077 | 0.006733 | -0.599 | -0.621 | -0.578 | 0.375 | 0.366 | 0.383 | b clearly beats a |  |  |  |  |  |  |  |  |  |  |
| `p-net0r:v1` | `thomson_scattering-n_e` | task_3-3 | 32 | 32 | 76 | 0.1404 | 0.04245 | -2.307 | -2.580 | -2.049 | 0.698 | 0.672 | 0.721 | b clearly beats a |  |  |  |  |  |  |  |  |  |  |
| `p-net0r:v1` | `thomson_scattering-t_e` | task_3-3 | 48 | 48 | 76 | 0.05547 | 0.03641 | -0.523 | -0.646 | -0.431 | 0.344 | 0.301 | 0.392 | b clearly beats a |  |  |  |  |  |  |  |  |  |  |
| `m-net0r:v1` | `equilibrium-psi` | task_2-3 | 16 | 16 | 92 | 0.04857 | 0.01294 | -2.754 | -4.996 | -1.186 | 0.734 | 0.543 | 0.833 | b clearly beats a |  |  |  |  |  |  |  |  |  |  |
| `o-net0r:o-klt` | `magnetics-b_field_pol_probe_ccbv_field` | cache | 160 | 160 | 100 | 0.03384 | 0.01314 | -1.575 | -1.714 | -1.446 | 0.612 | 0.591 | 0.632 | b clearly beats a | 1.000 | 1.000 |  | 160 |  |  |  |  |  |  |
| `o-net0r:o-klt` | `magnetics-b_field_pol_probe_obr_field` | cache | 18 | 18 | 100 | 0.03393 | 0.01585 | -1.140 | -1.465 | -0.902 | 0.533 | 0.474 | 0.594 | b clearly beats a | 1.000 | 1.000 |  | 18 |  |  |  |  |  |  |
| `o-net0r:o-klt` | `magnetics-b_field_pol_probe_obv_field` | cache | 36 | 36 | 100 | 0.03723 | 0.01223 | -2.043 | -2.184 | -1.910 | 0.671 | 0.656 | 0.686 | b clearly beats a | 1.000 | 1.000 |  | 36 |  |  |  |  |  |  |
| `o-net0r:o-klt` | `magnetics-flux_loop_flux` | cache | 30 | 30 | 100 | 0.03287 | 0.007112 | -3.622 | -3.863 | -3.431 | 0.784 | 0.774 | 0.794 | b clearly beats a | 1.000 | 1.000 | 60 | 30 |  |  |  |  |  |  |
| `c-net0rl:c-pca` | `magnetics-b_field_pol_probe_ccbv_field` | cache | 160 | 160 | 100 | 0.01807 | 0.005814 | -2.108 | -2.577 | -1.743 | 0.678 | 0.635 | 0.720 | b clearly beats a | 2.168 | 1.072 | 320 | 40 |  |  |  |  |  |  |
| `c-net0rl:c-pca` | `magnetics-b_field_pol_probe_obr_field` | cache | 18 | 18 | 100 | 0.01782 | 0.008377 | -1.127 | -1.731 | -0.747 | 0.530 | 0.427 | 0.634 | b clearly beats a | 1.885 | 1.011 | 36 | 18 |  |  |  |  |  |  |
| `c-net0rl:c-pca` | `magnetics-b_field_pol_probe_obv_field` | cache | 36 | 36 | 100 | 0.01948 | 0.006787 | -1.870 | -2.866 | -1.332 | 0.652 | 0.571 | 0.741 | b clearly beats a | 1.000 | 1.000 | 72 | 36 |  |  |  |  |  |  |
| `c-net0rl:c-pca` | `magnetics-flux_loop_flux` | cache | 30 | 30 | 100 | 0.01531 | 0.002548 | -5.010 | -5.406 | -4.645 | 0.834 | 0.823 | 0.844 | b clearly beats a | 10.849 | 3.518 | 30 | 12 |  |  |  |  |  |  |

The seed columns are empty: no network beat its comparison at seed 0, at learning rate 1e-4 or with the flux
extension of ruling C45, so rule 8 trained
no seeds 1 and 2, and the spreads and the "word with the spreads" are not defined. The empty ratio and reach
cells are explained under the linear table. 9 pairs of the plan were not run, because `recode` refused a collapsed source (rule 7, ruling C33, exit 2) and the assignment was not built, so they have no rows: `s-net0:v1`, `s-net0:o-net0`, `c-net0:s-net0`, `s-net0:s-klt` (`s-net0`: `net.series`, collapsed on `interferometer-n_e_line`); `b-net0:v1`, `b-net0:bo-net0`, `bc-net0:b-net0`, `b-net0:b-klt` (`b-net0`: `net.band`, collapsed on both signals); `bc-net0:bo-net0` (`bo-net0`: `net.band.cc` and `net.band.omv`, each collapsed on its own signal). (`b-net0:bo-net0` needs both `b-net0` and `bo-net0`.)

#### The trainings

One row per trained network, from `$S1/recoders/<name>/result.json` (its keys are
`best_step`, `best_val_mse`, `best_val_full`, `steps`, `seconds`, `plateau`,
`collapsed` and `usage`). `val_mse` is the validation squared error in example
units at the best step, averaged over the prefix lengths of the values, and
`val_full` the same at all values. `plateau` is yes if the best `val_mse` of the
last quarter of the log rows is not more than 1% under the best before it, and no
if it still fell by more than that, in which case the run is repeated with twice
the steps (rule 8). `collapsed` is
yes, no or unchecked (rule 7, as ruling C33 reads it: no token's first value is in
use on the signal's distinct validation examples; a signal without validation
examples has no verdict, which is unchecked and not a "no", because no data is not a
negative). An arm is collapsed when any of its signals is. "Collapsed (any token)"
is the plan's literal reading (`collapsed_any_token`: some token's first value under
0.3 on all rows), kept beside the verdict so the two can be compared. The column
is `any` over the arm's per-signal `collapsed_any_token`, because `result.json` has no
arm-level key for it; "active
tokens" is the count of tokens whose first value reaches 0.3, and decides nothing,
like the active values (the count per signal of the values whose normalized entropy
is at least 0.3).

A network's best step, and its plateau, are chosen on its validation store, which is
the same 100 `val_select` shots (`cache.pick_evenly`) that the `val_select-n100`
comparison reads (the floors and the compare cache side pick the same 100). Rule 4's
choice between a linear and a network arm is also made on them. Inferred, not
measured: the selection over about 20 checkpoints is small and favours the
networks. Rules 4 and 5 confirm a replacement on `val_report`, which bounds it.

Measured 2026-11-23, 33 rows: the 13 `.s0` runs of job M6 (07:30 to 07:41 UTC), the 10 `.s0.long`
runs of ruling C39 (07:43 to 08:00 UTC, 40,000 steps), the 9 `.s0.lr1e4` runs of job M8 (08:06 to 08:17 UTC) and the
one `.s0.lr1e4_long` run of ruling C45 (08:54 to 08:57 UTC, 40,000 steps), all
on the 2,000 training shots of the fit and validated on 100 `val_select` shots. "Seconds" is wall-clock with two to
four runs sharing a GPU, except for the run of ruling C45, whose log does not say what else ran on its GPU 1. "Active tokens" and "Active values" are "n of m", and for an arm of several signals the
smallest to the largest over its signals ("4 to 11 of 32"); for the values, "of m" is the values per token. The collapsed column was computed again from
`first_value_entropy_by_token` (a signal is collapsed when no entry reaches 0.3, ruling C33) and agrees with the
`collapsed` key of each of the 33 `result.json` files and with the "COLLAPSED (C33)" lines of the training logs.

| Name | Steps | Best step | `val_mse` | `val_full` | Plateau | Collapsed | Collapsed (any token) | Active tokens | Active values | Seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| `net.band.cc.s0` | 20000 | 19000 | 0.8695 | 0.7177 | no | yes | yes | 0 of 1 | 9 of 32 | 191 |
| `net.band.omv.s0` | 20000 | 20000 | 0.161 | 0.03809 | no | yes | yes | 0 of 1 | 3 of 32 | 205 |
| `net.band.s0` | 20000 | 18000 | 0.4243 | 0.2742 | no | yes | yes | 0 of 1 | 4 to 11 of 32 | 205 |
| `net.bandcat.cc.s0` | 20000 | 3000 | 1.658 | 1.596 | yes | no | yes | 5 of 54 | 0 of 8 | 99 |
| `net.bandcat.omv.s0` | 20000 | 20000 | 0.2917 | 0.1782 | no | no | yes | 2 of 54 | 0 of 8 | 201 |
| `net.ccbv.s0` | 20000 | 20000 | 0.01994 | 0.0006649 | no | no | no | 40 of 40 | 8 of 8 | 165 |
| `net.flux.s0` | 20000 | 16000 | 0.01387 | 0.0002369 | yes | no | no | 20 of 20 | 6 of 6 | 130 |
| `net.mag.s0` | 20000 | 15000 | 0.003158 | 0.0005037 | yes | no | no | 1 of 1 | 16 of 16 | 239 |
| `net.map.s0` | 20000 | 19000 | 0.7396 | 0.673 | no | no | yes | 47 of 48 | 0 of 2 | 122 |
| `net.obr.s0` | 20000 | 17000 | 0.03282 | 0.0009109 | no | no | no | 24 of 24 | 6 of 6 | 193 |
| `net.obv.s0` | 20000 | 18000 | 0.02604 | 0.0008569 | no | no | no | 24 of 24 | 6 of 6 | 175 |
| `net.profile.s0` | 20000 | 19000 | 0.1052 | 0.02126 | no | no | yes | 22 of 24 | 1 to 2 of 4 | 171 |
| `net.series.s0` | 20000 | 20000 | 0.08416 | 0.07168 | no | yes | yes | 0 to 1 of 1 | 5 to 15 of 16 | 257 |
| `net.band.cc.s0.long` | 40000 | 40000 | 0.8168 | 0.6112 | no | yes | yes | 0 of 1 | 7 of 32 | 633 |
| `net.band.omv.s0.long` | 40000 | 37000 | 0.1562 | 0.03495 | no | yes | yes | 0 of 1 | 3 of 32 | 637 |
| `net.band.s0.long` | 40000 | 40000 | 0.3656 | 0.1875 | no | yes | yes | 0 of 1 | 6 to 10 of 32 | 458 |
| `net.bandcat.omv.s0.long` | 40000 | 31000 | 0.2733 | 0.1468 | no | no | yes | 1 of 54 | 0 of 8 | 619 |
| `net.ccbv.s0.long` | 40000 | 37000 | 0.01891 | 0.0005274 | no | no | no | 40 of 40 | 8 of 8 | 224 |
| `net.map.s0.long` | 40000 | 23000 | 0.7344 | 0.6839 | yes | no | yes | 47 of 48 | 1 of 2 | 356 |
| `net.obr.s0.long` | 40000 | 36000 | 0.02717 | 0.0009033 | no | no | no | 24 of 24 | 6 of 6 | 617 |
| `net.obv.s0.long` | 40000 | 33000 | 0.02453 | 0.0008547 | no | no | no | 24 of 24 | 6 of 6 | 436 |
| `net.profile.s0.long` | 40000 | 40000 | 0.1023 | 0.01856 | no | no | yes | 22 of 24 | 2 of 4 | 624 |
| `net.series.s0.long` | 40000 | 37000 | 0.07989 | 0.06713 | yes | yes | yes | 0 to 1 of 1 | 6 to 14 of 16 | 712 |
| `net.bandcat.cc.s0.lr1e4` | 20000 | 5000 | 1.596 | 1.571 | yes | no | yes | 7 of 54 | 0 of 8 | 229 |
| `net.bandcat.omv.s0.lr1e4` | 40000 | 34000 | 0.2496 | 0.1431 | no | no | yes | 5 of 54 | 0 of 8 | 385 |
| `net.ccbv.s0.lr1e4` | 40000 | 39000 | 0.0181 | 0.0003929 | no | no | no | 40 of 40 | 7 of 8 | 603 |
| `net.flux.s0.lr1e4` | 20000 | 19000 | 0.0136 | 0.0001627 | no | no | no | 20 of 20 | 6 of 6 | 109 |
| `net.mag.s0.lr1e4` | 20000 | 15000 | 0.003147 | 0.0002541 | yes | no | no | 1 of 1 | 16 of 16 | 372 |
| `net.map.s0.lr1e4` | 40000 | 31000 | 0.72 | 0.686 | no | no | no | 48 of 48 | 2 of 2 | 556 |
| `net.obr.s0.lr1e4` | 40000 | 40000 | 0.02684 | 0.0007336 | no | no | no | 24 of 24 | 6 of 6 | 604 |
| `net.obv.s0.lr1e4` | 40000 | 30000 | 0.02649 | 0.0007707 | yes | no | no | 24 of 24 | 6 of 6 | 600 |
| `net.profile.s0.lr1e4` | 40000 | 36000 | 0.09943 | 0.01864 | no | no | yes | 22 to 23 of 24 | 3 of 4 | 551 |
| `net.flux.s0.lr1e4_long` | 40000 | 25000 | 0.01282 | 0.0001253 | yes | no | no | 20 of 20 | 6 of 6 | 180 |

The 8 collapsed runs and the signal that makes each so, read from the same entropies:

| Run | Signal that collapses | Active tokens | Active values | Distinct rows (of all rows) |
|---|---|---|---|---|
| `net.band.cc.s0` | `magnetics-b_field_tor_probe_cc_field` | 0 of 1 | 9 of 32 | 145,906 of 146,136 |
| `net.band.omv.s0` | `magnetics-b_field_pol_probe_omv_voltage` | 0 of 1 | 3 of 32 | 224,136 of 224,136 |
| `net.band.s0` | `magnetics-b_field_tor_probe_cc_field` | 0 of 1 | 4 of 32 | 145,906 of 146,136 |
| `net.band.s0` | `magnetics-b_field_pol_probe_omv_voltage` | 0 of 1 | 11 of 32 | 224,136 of 224,136 |
| `net.series.s0` | `interferometer-n_e_line` | 0 of 1 | 5 of 16 | 8,576 of 8,637 |
| `net.band.cc.s0.long` | `magnetics-b_field_tor_probe_cc_field` | 0 of 1 | 7 of 32 | 145,906 of 146,136 |
| `net.band.omv.s0.long` | `magnetics-b_field_pol_probe_omv_voltage` | 0 of 1 | 3 of 32 | 224,136 of 224,136 |
| `net.band.s0.long` | `magnetics-b_field_tor_probe_cc_field` | 0 of 1 | 6 of 32 | 145,906 of 146,136 |
| `net.band.s0.long` | `magnetics-b_field_pol_probe_omv_voltage` | 0 of 1 | 10 of 32 | 224,136 of 224,136 |
| `net.series.s0.long` | `interferometer-n_e_line` | 0 of 1 | 6 of 16 | 8,576 of 8,637 |

Where "Collapsed" is no and "Collapsed (any token)" is yes, some tokens are idle while others are in use
(`net.bandcat.*`: 1 to 7 of 54 tokens active; `net.profile`: 22 or 23 of 24; `net.map`: 47 of 48), and ruling C33
keeps the arm. A "Plateau" of no means that the validation error still fell by more than 1% over the last
quarter. Rule 8 extended each M6 run with a no once (ruling C39). The five retries at 1e-4 that end at 40,000 steps
with a no (`net.bandcat.omv.s0.lr1e4`, `net.ccbv.s0.lr1e4`, `net.map.s0.lr1e4`, `net.obr.s0.lr1e4` and
`net.profile.s0.lr1e4`) repeat C39's long runs, so their lineage has used its one extension. `net.flux.s0.lr1e4`
ended at 20,000 steps with a no, and its lineage had not been extended (`net.flux.s0` ended with a yes at 20,000
steps, so ruling C39 left it alone); ruling C45 extended it once, giving `net.flux.s0.lr1e4_long`: best `val_mse`
0.01282 at step 25,000 of 40,000 against 0.01360 at step 19,000 of 20,000, plateau yes, not collapsed, 180 s. Its
comparison `c-net0rl:c-pca`, in the networks table, shows the flux loss standing (0.01531 against 0.002548, and
0.01573 before the extension).

#### The decisions

One row per signal that a rule decides, from M9 and the confirmation on
`val_report`. A choice made on `val_select` stands only if `val_report` gives the
same word. `v2` equals `v1b` (see below), so no `v2` file was built or scored; the confirmation on `val_report`
was run for rule 6 on the four magnetics arrays only.

Measured 2026-11-23 (job M9 and its confirmation, ruling C42; the `c-net0rl` rows are ruling C45's): 61 rows,
the comparison files of the two tables above read per signal. A signal has several rows when a rule reads several
comparisons on it. The rows of `c-net0`, `c-net0r`, `c-net0rl`, `bc-net0` and `bc-net0r` are the concatenated networks
in the slot that ruling C41 gives them (rule 6's best concatenated arm). The rule and the split with its number of shots are in the "Rule"
cell, and "Error of a" and "Error of b" are the mean errors on that split. The word is the comparison's word at one
seed (rule 2, without a spread since no network won). `v2` was not built as a file: it equals `v1b`, because no learned arm
beat `v1b` under rules 4 and 5 (the one win over `v1`, `s-klt` on the solenoid current at `v1`'s K 1, was judged
again at `v1b`'s K 2 by ruling C42, where `v1b` clearly beats `s-klt.b`), so "Held by `v2`" is `v1b`'s setting in
`assignments/v1b.json` and "Held by `v2c`" is the setting in `assignments/v2c.json`; `v2c` differs from `v1b` in the
four magnetics arrays only.

| Signal | Rule | Tokenizer a | Tokenizer b | Error of a | Error of b | Word | Held by `v2` | Held by `v2c` |
|---|---|---|---|---|---|---|---|---|
| `gas_injection-total_injected` | 4, `val_select`, 88 shots | `s-klt` | `v1` | 0.007792 | 0.007902 | tie | `v1b`'s `frame_dct`, K 1 | `v1b`'s `frame_dct`, K 1 |
| `interferometer-n_e_line` | 4, `val_select`, 91 shots | `s-klt` | `v1` | 0.2267 | 0.2267 | tie | `v1b`'s `frame_dct`, K 16 | `v1b`'s `frame_dct`, K 16 |
| `magnetics-b_field_pol_probe_ccbv_field` | 4, `val_select`, 100 shots | `s-klt` | `v1` | 0.01627 | 0.01327 | b clearly beats a | `v1b`'s `frame_dct`, K 4 | `pca.ccbv` (`recoded_concat`, 160) |
| `magnetics-b_field_pol_probe_obr_field` | 4, `val_select`, 100 shots | `s-klt` | `v1` | 0.01596 | 0.01588 | b beats a | `v1b`'s `frame_dct`, K 1 | `pca.obr` (`recoded_concat`, 18) |
| `magnetics-b_field_pol_probe_obv_field` | 4, `val_select`, 100 shots | `s-klt` | `v1` | 0.01861 | 0.01242 | b clearly beats a | `v1b`'s `frame_dct`, K 2 | `pca.obv` (`recoded_concat`, 36) |
| `magnetics-b_field_tor_probe_saddle_voltage` | 4, `val_select`, 98 shots | `s-klt` | `v1` | 0.01311 | 0.01278 | b beats a | `v1b`'s `frame_dct`, K 1 | `v1b`'s `frame_dct`, K 1 |
| `magnetics-flux_loop_flux` | 4, `val_select`, 100 shots | `s-klt` | `v1` | 0.01221 | 0.007239 | b clearly beats a | `v1b`'s `frame_dct`, K 2 | `pca.flux` (`recoded_concat`, 30) |
| `pf_active-coil_current` | 4, `val_select`, 89 shots | `s-klt` | `v1` | 0.01718 | 0.01714 | tie | `v1b`'s `frame_dct`, K 2 | `v1b`'s `frame_dct`, K 2 |
| `pf_active-coil_voltage` | 4, `val_select`, 99 shots | `s-klt` | `v1` | 0.0168 | 0.01597 | b beats a | `v1b`'s `frame_dct`, K 4 | `v1b`'s `frame_dct`, K 4 |
| `pf_active-solenoid_current` | 4, `val_select`, 89 shots | `s-klt` | `v1` | 0.01362 | 0.01402 | a beats b | `v1b`'s `frame_dct`, K 2 | `v1b`'s `frame_dct`, K 2 |
| `pulse_schedule-i_plasma` | 4, `val_select`, 98 shots | `s-klt` | `v1` | 0.01329 | 0.00989 | b clearly beats a | `v1b`'s `frame_dct`, K 2 | `v1b`'s `frame_dct`, K 2 |
| `pulse_schedule-n_e_line` | 4, `val_select`, 99 shots | `s-klt` | `v1` | 0.01176 | 0.009983 | b clearly beats a | `v1b`'s `frame_dct`, K 1 | `v1b`'s `frame_dct`, K 1 |
| `soft_x_rays-horizontal_cam_lower` | 4, `val_select`, 95 shots | `s-klt` | `v1` | 0.0519 | 0.05067 | b beats a | `v1b`'s `frame_dct`, K 2 | `v1b`'s `frame_dct`, K 2 |
| `soft_x_rays-horizontal_cam_upper` | 4, `val_select`, 95 shots | `s-klt` | `v1` | 0.04907 | 0.04804 | b beats a | `v1b`'s `frame_dct`, K 4 | `v1b`'s `frame_dct`, K 4 |
| `spectrometer_visible-filter_spectrometer_dalpha_voltage` | 4, `val_select`, 95 shots | `s-klt` | `v1` | 0.15 | 0.143 | b beats a | `v1b`'s `frame_dct`, K 2 | `v1b`'s `frame_dct`, K 2 |
| `summary-ip` | 4, `val_select`, 89 shots | `s-klt` | `v1` | 0.01894 | 0.01468 | b clearly beats a | `v1b`'s `frame_dct`, K 4 | `v1b`'s `frame_dct`, K 4 |
| `summary-power_nbi` | 4, `val_select`, 90 shots | `s-klt` | `v1` | 0.01705 | 0.01473 | b clearly beats a | `v1b`'s `frame_dct`, K 8 | `v1b`'s `frame_dct`, K 8 |
| `pf_active-coil_current` | 4 (C42, at `v1b`'s K 2), `val_select`, 89 shots | `s-klt.b` | `v1b` | 0.009755 | 0.007586 | b clearly beats a | `v1b`'s `frame_dct`, K 2 | `v1b`'s `frame_dct`, K 2 |
| `pf_active-solenoid_current` | 4 (C42, at `v1b`'s K 2), `val_select`, 89 shots | `s-klt.b` | `v1b` | 0.005913 | 0.002212 | b clearly beats a | `v1b`'s `frame_dct`, K 2 | `v1b`'s `frame_dct`, K 2 |
| `equilibrium-lcfs_r` | 5 (lr 3e-4), `val_select`, 92 shots | `p-net0` | `v1` | 0.01333 | 0.007005 | b clearly beats a | `v1b`'s `linear`, K 32/4 | `v1b`'s `linear`, K 32/4 |
| `equilibrium-lcfs_z` | 5 (lr 3e-4), `val_select`, 92 shots | `p-net0` | `v1` | 0.01211 | 0.006733 | b clearly beats a | `v1b`'s `linear`, K 32/4 | `v1b`'s `linear`, K 32/4 |
| `thomson_scattering-n_e` | 5 (lr 3e-4), `val_select`, 76 shots | `p-net0` | `v1` | 0.1477 | 0.04245 | b clearly beats a | `v1b`'s `linear`, K 32/4 | `v1b`'s `linear`, K 32/4 |
| `thomson_scattering-t_e` | 5 (lr 3e-4), `val_select`, 76 shots | `p-net0` | `v1` | 0.06265 | 0.03641 | b clearly beats a | `v1b`'s `linear`, K 48/4 | `v1b`'s `linear`, K 48/4 |
| `equilibrium-lcfs_r` | 5 (lr 1e-4), `val_select`, 92 shots | `p-net0r` | `v1` | 0.01161 | 0.007005 | b clearly beats a | `v1b`'s `linear`, K 32/4 | `v1b`'s `linear`, K 32/4 |
| `equilibrium-lcfs_z` | 5 (lr 1e-4), `val_select`, 92 shots | `p-net0r` | `v1` | 0.01077 | 0.006733 | b clearly beats a | `v1b`'s `linear`, K 32/4 | `v1b`'s `linear`, K 32/4 |
| `thomson_scattering-n_e` | 5 (lr 1e-4), `val_select`, 76 shots | `p-net0r` | `v1` | 0.1404 | 0.04245 | b clearly beats a | `v1b`'s `linear`, K 32/4 | `v1b`'s `linear`, K 32/4 |
| `thomson_scattering-t_e` | 5 (lr 1e-4), `val_select`, 76 shots | `p-net0r` | `v1` | 0.05547 | 0.03641 | b clearly beats a | `v1b`'s `linear`, K 48/4 | `v1b`'s `linear`, K 48/4 |
| `equilibrium-psi` | 5 (lr 3e-4), `val_select`, 92 shots | `m-net0` | `v1` | 0.04613 | 0.01294 | b clearly beats a | `v1b`'s `linear`, K 16/2 | `v1b`'s `linear`, K 16/2 |
| `equilibrium-psi` | 5 (lr 1e-4), `val_select`, 92 shots | `m-net0r` | `v1` | 0.04857 | 0.01294 | b clearly beats a | `v1b`'s `linear`, K 16/2 | `v1b`'s `linear`, K 16/2 |
| `magnetics-b_field_tor_probe_cc_field` | 5 | `b-net0` (`net.band.s0.long`) | `v1` | no data | no data | collapsed (C33), no comparison | `v1b`'s `band_dct`, K 32/16/16/16/16/16/16/16 | `v1b`'s `band_dct`, K 32/16/16/16/16/16/16/16 |
| `magnetics-b_field_tor_probe_cc_field` | 5 | `bo-net0` (`net.band.cc.s0.long`) | `v1` | no data | no data | collapsed (C33), no comparison | `v1b`'s `band_dct`, K 32/16/16/16/16/16/16/16 | `v1b`'s `band_dct`, K 32/16/16/16/16/16/16/16 |
| `magnetics-b_field_pol_probe_omv_voltage` | 5 | `b-net0` (`net.band.s0.long`) | `v1` | no data | no data | collapsed (C33), no comparison | `v1b`'s `band_dct`, K 32/16/16/16/16/16/16/16 | `v1b`'s `band_dct`, K 32/16/16/16/16/16/16/16 |
| `magnetics-b_field_pol_probe_omv_voltage` | 5 | `bo-net0` (`net.band.omv.s0.long`) | `v1` | no data | no data | collapsed (C33), no comparison | `v1b`'s `band_dct`, K 32/16/16/16/16/16/16/16 | `v1b`'s `band_dct`, K 32/16/16/16/16/16/16/16 |
| `magnetics-b_field_pol_probe_omv_voltage` | 5, `val_select`, 62 shots | `b-klt` | `v1` | 0.4028 | 0.403 | tie | `v1b`'s `band_dct`, K 32/16/16/16/16/16/16/16 | `v1b`'s `band_dct`, K 32/16/16/16/16/16/16/16 |
| `magnetics-b_field_tor_probe_cc_field` | 5, `val_select`, 62 shots | `b-klt` | `v1` | 0.6267 | 0.6267 | tie | `v1b`'s `band_dct`, K 32/16/16/16/16/16/16/16 | `v1b`'s `band_dct`, K 32/16/16/16/16/16/16/16 |
| `magnetics-b_field_pol_probe_omv_voltage` | 6 (lr 3e-4; the concatenated slot ruling C41 sets), `val_select`, 62 shots | `bc-net0` | `bc-pca` | 0.4114 | 0.4026 | b beats a | `v1b`'s `band_dct`, K 32/16/16/16/16/16/16/16 | `v1b`'s `band_dct`, K 32/16/16/16/16/16/16/16 |
| `magnetics-b_field_tor_probe_cc_field` | 6 (lr 3e-4; the concatenated slot ruling C41 sets), `val_select`, 62 shots | `bc-net0` | `bc-pca` | 0.6418 | 0.631 | b beats a | `v1b`'s `band_dct`, K 32/16/16/16/16/16/16/16 | `v1b`'s `band_dct`, K 32/16/16/16/16/16/16/16 |
| `magnetics-b_field_pol_probe_omv_voltage` | 6 (lr 1e-4; the concatenated slot ruling C41 sets), `val_select`, 62 shots | `bc-net0r` | `bc-pca` | 0.4116 | 0.4026 | b beats a | `v1b`'s `band_dct`, K 32/16/16/16/16/16/16/16 | `v1b`'s `band_dct`, K 32/16/16/16/16/16/16/16 |
| `magnetics-b_field_tor_probe_cc_field` | 6 (lr 1e-4; the concatenated slot ruling C41 sets), `val_select`, 62 shots | `bc-net0r` | `bc-pca` | 0.6388 | 0.631 | b beats a | `v1b`'s `band_dct`, K 32/16/16/16/16/16/16/16 | `v1b`'s `band_dct`, K 32/16/16/16/16/16/16/16 |
| `magnetics-b_field_pol_probe_ccbv_field` | 6 (lr 3e-4; the concatenated slot ruling C41 sets), `val_select`, 100 shots | `c-net0` | `c-pca` | 0.01828 | 0.005814 | b clearly beats a | `v1b`'s `frame_dct`, K 4 | `pca.ccbv` (`recoded_concat`, 160) |
| `magnetics-b_field_pol_probe_obr_field` | 6 (lr 3e-4; the concatenated slot ruling C41 sets), `val_select`, 100 shots | `c-net0` | `c-pca` | 0.01892 | 0.008377 | b clearly beats a | `v1b`'s `frame_dct`, K 1 | `pca.obr` (`recoded_concat`, 18) |
| `magnetics-b_field_pol_probe_obv_field` | 6 (lr 3e-4; the concatenated slot ruling C41 sets), `val_select`, 100 shots | `c-net0` | `c-pca` | 0.0187 | 0.006787 | b clearly beats a | `v1b`'s `frame_dct`, K 2 | `pca.obv` (`recoded_concat`, 36) |
| `magnetics-flux_loop_flux` | 6 (lr 3e-4; the concatenated slot ruling C41 sets), `val_select`, 100 shots | `c-net0` | `c-pca` | 0.01709 | 0.002548 | b clearly beats a | `v1b`'s `frame_dct`, K 2 | `pca.flux` (`recoded_concat`, 30) |
| `magnetics-b_field_pol_probe_ccbv_field` | 6 (lr 1e-4; the concatenated slot ruling C41 sets), `val_select`, 100 shots | `c-net0r` | `c-pca` | 0.01807 | 0.005814 | b clearly beats a | `v1b`'s `frame_dct`, K 4 | `pca.ccbv` (`recoded_concat`, 160) |
| `magnetics-b_field_pol_probe_obr_field` | 6 (lr 1e-4; the concatenated slot ruling C41 sets), `val_select`, 100 shots | `c-net0r` | `c-pca` | 0.01782 | 0.008377 | b clearly beats a | `v1b`'s `frame_dct`, K 1 | `pca.obr` (`recoded_concat`, 18) |
| `magnetics-b_field_pol_probe_obv_field` | 6 (lr 1e-4; the concatenated slot ruling C41 sets), `val_select`, 100 shots | `c-net0r` | `c-pca` | 0.01948 | 0.006787 | b clearly beats a | `v1b`'s `frame_dct`, K 2 | `pca.obv` (`recoded_concat`, 36) |
| `magnetics-flux_loop_flux` | 6 (lr 1e-4; the concatenated slot ruling C41 sets), `val_select`, 100 shots | `c-net0r` | `c-pca` | 0.01573 | 0.002548 | b clearly beats a | `v1b`'s `frame_dct`, K 2 | `pca.flux` (`recoded_concat`, 30) |
| `magnetics-b_field_pol_probe_ccbv_field` | 6 (lr 1e-4, flux extended once, ruling C45; the concatenated slot ruling C41 sets), `val_select`, 100 shots | `c-net0rl` | `c-pca` | 0.01807 | 0.005814 | b clearly beats a | `v1b`'s `frame_dct`, K 4 | `pca.ccbv` (`recoded_concat`, 160) |
| `magnetics-b_field_pol_probe_obr_field` | 6 (lr 1e-4, flux extended once, ruling C45; the concatenated slot ruling C41 sets), `val_select`, 100 shots | `c-net0rl` | `c-pca` | 0.01782 | 0.008377 | b clearly beats a | `v1b`'s `frame_dct`, K 1 | `pca.obr` (`recoded_concat`, 18) |
| `magnetics-b_field_pol_probe_obv_field` | 6 (lr 1e-4, flux extended once, ruling C45; the concatenated slot ruling C41 sets), `val_select`, 100 shots | `c-net0rl` | `c-pca` | 0.01948 | 0.006787 | b clearly beats a | `v1b`'s `frame_dct`, K 2 | `pca.obv` (`recoded_concat`, 36) |
| `magnetics-flux_loop_flux` | 6 (lr 1e-4, flux extended once, ruling C45; the concatenated slot ruling C41 sets), `val_select`, 100 shots | `c-net0rl` | `c-pca` | 0.01531 | 0.002548 | b clearly beats a | `v1b`'s `frame_dct`, K 2 | `pca.flux` (`recoded_concat`, 30) |
| `magnetics-b_field_pol_probe_ccbv_field` | 6, `val_select`, 100 shots | `c-pca` | `s-klt` | 0.005814 | 0.01627 | a clearly beats b | `v1b`'s `frame_dct`, K 4 | `pca.ccbv` (`recoded_concat`, 160) |
| `magnetics-b_field_pol_probe_obr_field` | 6, `val_select`, 100 shots | `c-pca` | `s-klt` | 0.008377 | 0.01596 | a clearly beats b | `v1b`'s `frame_dct`, K 1 | `pca.obr` (`recoded_concat`, 18) |
| `magnetics-b_field_pol_probe_obv_field` | 6, `val_select`, 100 shots | `c-pca` | `s-klt` | 0.006787 | 0.01861 | a clearly beats b | `v1b`'s `frame_dct`, K 2 | `pca.obv` (`recoded_concat`, 36) |
| `magnetics-flux_loop_flux` | 6, `val_select`, 100 shots | `c-pca` | `s-klt` | 0.002548 | 0.01221 | a clearly beats b | `v1b`'s `frame_dct`, K 2 | `pca.flux` (`recoded_concat`, 30) |
| `magnetics-b_field_pol_probe_ccbv_field` | 6, `val_report`, 557 shots | `c-pca` | `s-klt` | 0.005597 | 0.01526 | a clearly beats b | `v1b`'s `frame_dct`, K 4 | `pca.ccbv` (`recoded_concat`, 160) |
| `magnetics-b_field_pol_probe_obr_field` | 6, `val_report`, 557 shots | `c-pca` | `s-klt` | 0.007949 | 0.01513 | a clearly beats b | `v1b`'s `frame_dct`, K 1 | `pca.obr` (`recoded_concat`, 18) |
| `magnetics-b_field_pol_probe_obv_field` | 6, `val_report`, 557 shots | `c-pca` | `s-klt` | 0.006573 | 0.0178 | a clearly beats b | `v1b`'s `frame_dct`, K 2 | `pca.obv` (`recoded_concat`, 36) |
| `magnetics-flux_loop_flux` | 6, `val_report`, 557 shots | `c-pca` | `s-klt` | 0.002426 | 0.01145 | a clearly beats b | `v1b`'s `frame_dct`, K 2 | `pca.flux` (`recoded_concat`, 30) |
| `magnetics-b_field_pol_probe_omv_voltage` | 6, `val_select`, 62 shots | `bc-pca` | `b-klt` | 0.4026 | 0.4028 | tie | `v1b`'s `band_dct`, K 32/16/16/16/16/16/16/16 | `v1b`'s `band_dct`, K 32/16/16/16/16/16/16/16 |
| `magnetics-b_field_tor_probe_cc_field` | 6, `val_select`, 62 shots | `bc-pca` | `b-klt` | 0.631 | 0.6267 | b beats a | `v1b`'s `band_dct`, K 32/16/16/16/16/16/16/16 | `v1b`'s `band_dct`, K 32/16/16/16/16/16/16/16 |

What each rule decides. Rule 4: the cosine transform stays on all 17 series signals, `v1b`'s K 2 on the two
currents included (ruling C42); `v2c` swaps four of them by rule 6. Rule 5: the profiles, the map and the Mirnov band
transform stay. The linear Mirnov arm `b-klt` ties with `v1` on both signals (62 shots); the per-stream Mirnov networks
collapsed (`b-net0`, `bo-net0`, with no comparison and so "no data" in the error cells); the concatenated one, `bc-net0`,
was compared only in the rule-6 slot that ruling C41 gives it and lost to `bc-pca` at both learning rates. No Mirnov
network was compared with `v1` itself. That `bc-net0` would lose to it too is inferred, not measured: its errors (0.4114
on `omv`, 0.6418 on `cc_field`) are above `v1`'s in `b-klt:v1` (0.403 and 0.6267) on 62 `val_select` shots, but no paired
comparison was run. `bc-pca` against `b-klt` ties on `omv` and loses on `cc_field`. Rule 6: the concatenated linear
arm clearly beats the shared one on the four magnetics arrays on both splits, so `v2c` holds `pca.ccbv`, `pca.obr`,
`pca.obv` and `pca.flux`; on the two Mirnov signals the best concatenated arm does not clearly beat the shared one. The
concatenated networks lost to the linear concatenated arm in the slot of ruling C41 on every row (`c-net0`, `c-net0r`
and `c-net0rl` against `c-pca`; `bc-net0` and `bc-net0r` against `bc-pca`), so no network is the best concatenated arm.
The `val_report` confirmation of rules 4 and 5, and of rule 6 on the Mirnov signals, was not run, because nothing beat its
baseline on `val_select` at the K the baseline holds in `v1b` (the one win at `v1`'s K 1, `s-klt` on the solenoid
current, lost at K 2, ruling C42), so there is nothing to confirm (no data, not a pass).

#### The input round-trip of `v1`, `v1b` and `v2c`

Measured 2026-11-23 with `score --instrument inputs` on all `val_report` shots of each task (the direct model of
stage 0 named in the first column, `through / raw - 1` on the same shots): the `v1` rows by stage 1b's job K6 (03:47 to
04:43 UTC, `logs/k6.meta`), the `v1b` `ranged` rows by ruling C29 (06:33 to 07:00 UTC, `logs/v1b_gate.meta`), and the
`v2c` rows and the `v1b` `unquantized` rows by job M9 under ruling C43 (08:10 to 08:34 UTC, `logs/m9_inputs.meta`). `ranged` is like for like
with the K6 and C29 rows; `unquantized` was run with `v1b` and `v2c` on tasks 1-1 and 1-2. The magnetics arrays are
inputs to 12 of 14 tasks. One trained direct model and one scoring pass give each row, so whether +0.11% against
+0.24% on task 1-2 is a difference is not tested.

Rule 6 asks for the round-trip "with each arm's tokens". Only the assignments stage 2 can start on were scored (`v1`,
`v1b`, `v2c`). No assignment holding `s-klt` or `o-klt` on the magnetics was scored, so the round-trip of the shared and
own arms is not measured (no data, not a tie).

| Task (direct model) | Tokenizer | Mode | Shots | Raw score | Score through the tokenizer | `through / raw - 1` |
|---|---|---|---|---|---|---|
| `task_1-1` (`s90k`) | `v1` | `ranged` | 520 | 0.1235 | 0.1237 | +0.22% |
| `task_1-1` (`s90k`) | `v1b` | `ranged` | 520 | 0.1235 | 0.1236 | +0.13% |
| `task_1-1` (`s90k`) | `v1b` | `unquantized` | 520 | 0.1235 | 0.1236 | +0.13% |
| `task_1-1` (`s90k`) | `v2c` | `ranged` | 520 | 0.1235 | 0.1236 | +0.08% |
| `task_1-1` (`s90k`) | `v2c` | `unquantized` | 520 | 0.1235 | 0.1236 | +0.08% |
| `task_1-2` (`lr1e4`) | `v1` | `ranged` | 525 | 0.03618 | 0.03662 | +1.21% |
| `task_1-2` (`lr1e4`) | `v1b` | `ranged` | 525 | 0.03618 | 0.03627 | +0.24% |
| `task_1-2` (`lr1e4`) | `v1b` | `unquantized` | 525 | 0.03618 | 0.03627 | +0.24% |
| `task_1-2` (`lr1e4`) | `v2c` | `ranged` | 525 | 0.03618 | 0.03623 | +0.11% |
| `task_1-2` (`lr1e4`) | `v2c` | `unquantized` | 525 | 0.03618 | 0.03623 | +0.11% |
| `task_2-1` (`s90k`) | `v1` | `ranged` | 520 | 0.1347 | 0.1353 | +0.45% |
| `task_2-1` (`s90k`) | `v1b` | `ranged` | 520 | 0.1347 | 0.1347 | +0.03% |
| `task_2-1` (`s90k`) | `v2c` | `ranged` | 520 | 0.1347 | 0.1347 | +0.01% |
| `task_4-1` (`h050b256`) | `v1` | `ranged` | 531 | 0.1727 | 0.1716 | -0.63% |
| `task_4-1` (`h050b256`) | `v1b` | `ranged` | 531 | 0.1727 | 0.1716 | -0.63% |
| `task_4-1` (`h050b256`) | `v2c` | `ranged` | 531 | 0.1727 | 0.1716 | -0.63% |

#### The 50 kHz band

One row per series signal with 250 samples per 5 ms frame (four on cache v1), from
`$S1/compare/fluct.csv`, which `fluct` writes (job M10) with 25 separate columns
`bin_00` to `bin_24`, shown here as one. It is a statistic of the
power above the trend path, by frequency, and no tokenizer is built. With 250
samples of 0.02 ms in a 5 ms frame, group `i` is the band from `i` to `i + 1`
kHz (derived from the frame length, not measured). `band_to_trend` is the band's
power over the power of the trend values, and it is empty for the saddle coils,
because `v1` gives them K 1 (ruling C14). `persistence` is the median, over the
(shot, channel) pairs for which it is defined, of the correlation between the
logarithm of one frame's band power and the next frame's.

Measured 2026-11-23 (job M10, `fluct --workers 64`, 07:47 UTC, 26 s; 100 `val_select` shots, K of the trend
from `v1`). 4 rows.

| Signal | K of the trend (`k_trend`) | Channel-frames (`n_frames`) | Band to trend (`band_to_trend`) | Share below 5 kHz (`share_below_5khz`) | Persistence (`persistence`) | Share of the band in each 1 kHz group (`bin_00` to `bin_24`) |
|---|---|---|---|---|---|---|
| `magnetics-b_field_tor_probe_saddle_voltage` | 1 | 105,675 |  | 0.995 | 0.738 | 0.993, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000 |
| `soft_x_rays-horizontal_cam_lower` | 2 | 155,196 | 3.202 | 0.434 | 0.742 | 0.204, 0.064, 0.070, 0.058, 0.038, 0.038, 0.036, 0.032, 0.031, 0.031, 0.029, 0.028, 0.027, 0.027, 0.027, 0.027, 0.026, 0.026, 0.026, 0.026, 0.026, 0.025, 0.025, 0.026, 0.026 |
| `soft_x_rays-horizontal_cam_upper` | 4 | 155,196 | 0.269 | 0.527 | 0.583 | 0.256, 0.089, 0.065, 0.053, 0.064, 0.051, 0.038, 0.032, 0.027, 0.026, 0.022, 0.020, 0.018, 0.018, 0.018, 0.018, 0.021, 0.018, 0.022, 0.022, 0.026, 0.021, 0.020, 0.018, 0.017 |
| `spectrometer_visible-filter_spectrometer_dalpha_voltage` | 2 | 28,266 | 1.172 | 0.883 | 0.815 | 0.709, 0.098, 0.037, 0.024, 0.016, 0.013, 0.010, 0.008, 0.008, 0.007, 0.006, 0.006, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004 |

The saddle coils' `band_to_trend` is empty by definition (K 1, ruling C14), not missing data. Inside a 5 ms frame
the lower soft X-ray camera holds 3.202 times as much power above its 2-value trend as in it, D-alpha 1.172 times,
and the upper camera, at K 4, keeps 0.269 of the trend's power in the band; the saddle coils keep 0.995 of their
band power below 5 kHz. These are statistics of the inputs (no tokenizer is built, and no score is involved), and
they go to the stage-2 plan.

## Limits

- **The fit is on 2,000 training shots.** The ranges and bases are fitted on
  2,000 training shots spread evenly over the sorted shot list
  (`cache.pick_evenly`), the default of `fit --n-shots`. The first 200 shots by
  number are one old campaign and do not track the rest. The ladders are scored
  on 100 `val_select` shots, also spread evenly, and the confirmation on all
  `val_report` shots.
- **The floors are fitted quantities.** Ranges and bases are fitted, K is chosen
  on `val_select`, and the floors are reported on `val_report`. They are not
  measured on the test split, and nothing in this stage reads the test split.
  Stage 0's floors differ between the splits by up to 0.002 on every task but
  4-5 (measured 2026-11-22, TokaMind's floor, test against `val_report`), and
  the same predictor differs between the two splits by up to 0.04 (measured
  2026-11-22 in [stage 0](./ignite-mast-stage0.md), `val_report` against test:
  persistence on task 4-5 scores 0.9097 and 0.8701, the mean predictor on 4-5
  0.6852 and 0.6533, persistence on 4-1 0.2978 and 0.3247).
- **The rough looks of 2026-11-22 are not results.** Every rough look of
  2026-11-22 quoted on this page was made on the CPU from cache v1 on 24 to 96
  training shots with the code of that day (24 shots for the cosine truncation,
  72 fitted and 24 tested for the principal components, 40 for the
  spectrograms, 96 for the
  Thomson missing-data count). They set expectations only, and the ladders of
  this stage replace them. The rough look of 2026-11-23 on 2 shots, below, is a
  separate one, made on `val_select` shots with the code of the trial.
- **Task 4-5's frame statistics (open decision 1 of the plan).** The design
  scores task 4-5 from frame statistics, one value per 5 ms frame and bin. The
  rough look of 2026-11-22 on 40 training shots says three things about it:
  the error of the Mirnov task sits in the lowest frequencies (99.8% of the mean
  baseline's squared error for the toroidal-field probes and 77% for the
  poloidal ones is below 31 kHz); a perfect prediction of every frame's mean
  spectrum would still score about 74% of the mean baseline, because the
  targets are single 1 ms periodograms that scatter inside a frame; and
  TokaMind's own floor is lower (0.3605 on `val_report`, measured 2026-11-22),
  which the plan reads as because it keeps detail inside 5 ms (the plan's
  reading, inferred, not measured). What stage 0 measured on 2026-11-22 on
  `val_report` (351 shots): the direct model, trained on the 1 ms columns,
  scores 0.6475 against the best published 0.6510 (a tie) and is below the mean
  baseline on the same shots by a paired 0.0377 (95% interval 0.0298 to 0.0461,
  carried by a minority of the shots). Whether that 5% sits in the frame means
  or inside the frames is not measured. With frame statistics the floor of this
  tokenizer on 4-5 is expected to be about three quarters of the published
  scores, and the "fifth of the best published score" target cannot be met for
  this task by any choice of K. The plan's recommendation is to keep frame
  statistics, report the frame-oracle floor beside the score, add no 1 ms detail
  to the tokens unless a model is shown to predict it, and expect a small margin
  on 4-5. The owner may instead want the task met on its own terms or dropped
  from the tokenizer work.
- **A rough look on 2 shots of task 4-5 (2026-11-23).** Task 9's trial on
  2026-11-23 scored the frame oracle (one value per 5 ms frame, all 32
  coefficients in every band, `k0 = 32`) on 2 `val_select` shots (11806 and
  30467) with a trial-scale fit (8 training shots, of which 3 hold the first
  signal and 5 the second). The NRMSE is 2.18 for
  `magnetics-b_field_tor_probe_cc_field` against a mean baseline of 2.64 on the
  same windows, and 0.364 against 0.413 for
  `magnetics-b_field_pol_probe_omv_voltage`. So the frame oracle scores 82% and
  88% of the mean baseline on these 2 shots, and 74% on the 40 shots above.
  A frame mean averages 4 to 5 STFT columns, so it cannot reproduce the
  column-to-column noise that the NRMSE counts. This is a rough look on 2 shots,
  with a trial-scale fit. It is to be measured on 100 `val_select` shots at the
  jobs K2 and K3, and it is not a result.
- **Frames across signals.** A spectrogram column is assigned to the frame that
  holds its centre, counted from the window's first column, while series frames
  are counted from the window's first sample. How TokaMark places the first
  column against the cut time was not checked, and stage 2 needs it before it
  mixes these tokens in one frame.
- **The fluctuation band of the 50 kHz channels is not covered.** The soft X-ray
  cameras, D-alpha and the saddle coils have a fast fluctuation that no fixed
  transform here keeps at K of at most 16. No benchmark task scores the band. Plan 1c's job M10 measured how much of each channel's
  power the band holds (see [The 50 kHz band](#the-50-khz-band)); no tokenizer for it is built, and the
  numbers go to the stage-2 plan.
- **No data is not a negative.** In the gate table a "no data" cell, in the
  columns measured from the cache, is a signal that none of the 200 `val_report`
  shots read for them has (a signal that the fit has no training shot for gets
  the same cells). In the ladders the note `no data` is a signal that none of
  the 100 `val_select` shots has, and `not fitted: no training shot has it`
  tells the other case apart. An empty cell is a score nobody has stored yet.
  None of them is a good result and none is 0.
- **The learned tokenizers' loss counts channels equally.** The loss of stage 1c's
  networks is a squared error in example units, each channel scaled to unit
  variance, while TokaMark weights channels by their physical variance. The
  comparison is read in TokaMark units, and where the two disagree for a signal the
  weighting of TokaMark is the one that counts for the benchmark. Not measured.
- **The Mirnov comparison is on power spectra.** The cached spectrograms hold power
  only, so the comparison of the two Mirnov signals says nothing about the phase
  between probes, from which a mode number is read. That needs raw waveforms, which
  cache v1 does not hold.
- **No magnetics signal is a benchmark target.** The magnetics arrays and the
  Mirnov signals are inputs only, so a tokenizer for them is judged by
  reconstruction and by the input round-trip of four tasks (1-1 and 1-2, the
  direct models of rule 6, and 2-1 and 4-1, added by ruling C43; `val_report`, one direct model and one pass
  each), and no benchmark score says what the dynamics model
  gains or loses from one choice or the other (open decision 1 of the plan).
- **A learned tokenizer sees only what the front end keeps.** A recoder sits behind
  the top rung of a fixed codec (16 cosine values, 32 per band, 96 principal
  components), so it cannot use what that rung drops (for the 50 kHz channels, the
  part above 1.6 kHz), and what it can add at most is known from the ladder of
  stage 1b at that rung.
