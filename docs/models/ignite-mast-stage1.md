---
title: "IGNITE on MAST: stage 1 fixed tokenizers"
---

# IGNITE on MAST: stage 1 fixed tokenizers

**Status: measured (2026-11-23). The assignment `v1` (1,470 values per 5 ms frame) passes the gate on all 57 (task, signal) pairs; five pairs miss the fidelity target, each recorded with its cost, and one input signal (`interferometer-n_e_line`, through one shot) misses its reconstruction target. Stage 1c's learned tokenizers are built and not measured yet.**

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

Stage 1c's learned tokenizers are built and not measured yet. The code and its
tests are committed. None of the controller jobs M1 to M10 under
[How to run stage 1c](#how-to-run-stage-1c) has run, so every table under
[Results of the comparison](#results-of-the-comparison) is empty and says "Not
measured yet." Nothing in this section is a result, and the only numbers in it
are the ones that carry their date, their split, their number of shots and the
words "measured" or "rough look". The plan is
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

**Readings the controller added before any measurement.** Three readings, written
on 2026-11-23 before any production store, recoder or comparison existed. None
changes a rule above.

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
  measured). If the owner answers open decision 1 of stage 1b with per-column low
  bins before M4 runs, the Mirnov arms are planned again before M4 to M7. The
  Mirnov networks stay fourth in the cut order.

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
  and reports; no network floor has been timed yet.
- **Concatenated linear recoders are fitted on zero-filled rows (ruling C21).** A
  missing stream is 0 in example units when a concatenated recoder encodes, and
  `fit_linear` now fits on rows filled the same way, not on a pairwise covariance.
  The plan's text "used pairwise" is superseded for concatenated recoders.
  Reason, rough look, 2-shot trial store, 2026-11-23 (the 40 `ccbv` probes, a basis
  of 640 front values fitted on the trial's 672 training frames, the share left
  over measured on its 167 validation frames of 2 `val_select` shots): fitted
  pairwise it is 0.0074, 0.0072, 0.0071 and 0.0071 at 40, 80, 160 and 320 values,
  fitted on zero-filled rows 0.0006, 0.0002, 0.0001 and 0.0000, so the pairwise
  fit was 12 to 36 times higher at 40 and 80 values. The cause: 69% of the trial's
  `ccbv` rows have a missing stream (4.5% of the entries), and a pairwise
  covariance with that many gaps is not positive semi-definite (131 negative
  eigenvalues of 640), which puts real energy last in the order of the basis. The
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

Every command starts with `OMP_NUM_THREADS=16`, or with `OMP_NUM_THREADS=8` where
three or four jobs share a GPU. Reason, measured 2026-11-23: one CPU test of the
training loop took 86 to 111 s at torch's default thread count on this 352-core
node under load, and 1.6 s with 8 threads.

Every duration in the table is the plan author's estimate, not a measurement;
none was measured for this code. Unlike the stage-0 trainings, whose speed was set
by the loaders, these hold their examples in memory (about 2,000 shots times 100
frames times 163 channels of 16 values: 2 GB for the series family, an estimate),
so several can share one GPU.

| Job | After | Command | GPU | Duration | What is read from the output |
|---|---|---|---|---|---|
| M1 store | Task 3, K3 | `OMP_NUM_THREADS=16 $TT store --workers 64` | none | 10 to 30 min | 24 lines. No signal says "no data"; the skipped shots per signal match 1b's fit. |
| M2 linear arms | M1 | `OMP_NUM_THREADS=16 $TT linear --arm A` for the 11 linear arms, all at once | none | 1 to 5 min each | Per signal the share left at 1, 2, 4, 8, 16 values per stream. It falls from left to right. |
| M3 joint basis | M1, Task 4 | `OMP_NUM_THREADS=16 $TT joint --signals ccbv,obr,obv,flux` | none | minutes | Per size three shares: the joint basis of the 91 channels, each array's own basis, the per-channel basis. This is the upper limit of what coding channels together can give in linear form. |
| M4 recoded assignments and their floors | M2, Task 4 | `OMP_NUM_THREADS=16 $RT recode --name s-klt --from v1 --arm klt.series`; `o-klt` (`klt.mag`); `c-pca` (`pca.ccbv,pca.obr,pca.obv,pca.flux`), `c-pca.h` and `c-pca.q` (the same with `--scale 0.5`, `0.25`); `b-klt` (`klt.band`); `bo-klt` (`klt.band.cc,klt.band.omv`); `bc-pca` (`pca.band.cc,pca.band.omv`), `bc-pca.h`, `bc-pca.q`. Then M5's `compare` once: it prints every `score` command whose floor is missing (`OMP_NUM_THREADS=16 $RT score --task T --split val_select --n-shots 100 --assignment A --mode unquantized`); run those in waves | none | about 20 passes of 3 to 15 min; about 1 h | `recode` prints equal values per frame before and after (a concatenated network arm may be a few values short: tokens of 8). The last line of each pass. |
| M5 linear comparison | M4 | `OMP_NUM_THREADS=16 $RT compare --a A --b B --drop-channel --reach 0.0176` for the pairs: `s-klt:v1` (rule 4); `s-klt:o-klt`, `c-pca:o-klt`, `c-pca:s-klt`, `c-pca.h:s-klt`, `c-pca.q:s-klt`, `c-pca:v1` (rule 6, magnetics); `b-klt:v1` (rule 5); `b-klt:bo-klt`, `bc-pca:bo-klt`, `bc-pca:b-klt`, `bc-pca.h:b-klt`, `bc-pca.q:b-klt` (rule 6, Mirnov) | none | minutes each | The 13 tables. They go to the owner the day they exist, with M3's line: this is the linear answer to shared against specialized, and the networks cannot change what the data holds, only how much of it a tokenizer of that size gets. |
| M6 networks, seed 0 | Task 6, M1 | `OMP_NUM_THREADS=8 $TT train --arm A --seed 0 --gpu N --share-gpu` for the 13 network arms, three or four per GPU | 4, shared | 5 to 20 min each; under 1 h | The last line of each run. `collapsed yes` (exit 3): that arm takes part in no comparison. `plateau no`: run it once more with `--steps 40000 --variant long` (rule 8). |
| M7 network comparison | M6 | `OMP_NUM_THREADS=16 $RT recode` to `s-net0` (`net.series.s0`), `o-net0` (`net.mag.s0`), `c-net0` (the four `net.<array>.s0`), `b-net0` (`net.band.s0`), `bo-net0` (`net.band.cc.s0,net.band.omv.s0`), `bc-net0` (`net.bandcat.cc.s0,net.bandcat.omv.s0`), `p-net0` (`net.profile.s0`), `m-net0` (`net.map.s0`); the missing floors as in M4, on the CPU (see the departures above); then `OMP_NUM_THREADS=16 $RT compare` for the pairs of M5 with the network in place of the linear arm, and `s-net0:s-klt`, `b-net0:b-klt`, `p-net0:v1`, `m-net0:v1` (rule 5) | none | about 12 passes of 5 to 20 min; about 1 h | The tables, as in M5. |
| M8 rule 8 | M7 | For a network that lost: `OMP_NUM_THREADS=8 $TT train --arm A --seed 0 --lr 1e-4 --variant lr1e4 --gpu N --share-gpu`, then its comparison again. For a network that won: seeds 1 and 2, their assignments and floors, then `OMP_NUM_THREADS=16 $RT seeds --pairs a0:b,a1:b,a2:b --label val_select-n100` | 4, shared | up to 26 runs and their floors; 1 to 2 h | The gaps of the three seeds, their spread and the word with that spread. |
| M9 `v2`, `v2c` and their confirmation | M8 | `OMP_NUM_THREADS=16 $RT recode --name v2 --from v1 --arm <the winners by rules 4 and 5> --signals <their signals>`; `v2c` from `v2` with the concatenated winners of rule 6. For every task with a changed output signal: 1b's K5 passes on `val_report` in the three read-outs; `OMP_NUM_THREADS=16 $RT compare --label val_report` for the decided pairs; for tasks 1-1 and 1-2 `OMP_NUM_THREADS=16 $RT score --task T --split val_report --instrument inputs --assignment A --mode unquantized --variant V --gpu N` with `v2` and `v2c`; `OMP_NUM_THREADS=16 $RT gate --assignment $S1/assignments/v2.json` | 4 passes with a GPU | 1 to 2 h without a GPU, 4 passes of 15 to 45 min with one, the gate 5 to 15 min | A choice made on `val_select` stands only if `val_report` gives the same word. The gate's last line. If `v2` equals `v1`, nothing is scored again and the page says so. |
| M10 the 50 kHz band | Task 7, K3 | `OMP_NUM_THREADS=16 $TT fluct --workers 64` | none | 5 to 15 min | Four lines; they go to the stage-2 plan. |

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

After M9 the controller fills the five tables below, sets the status line of this
page and commits with the numbers, their date, split and shot counts in the body.
Stage 2 does not wait for any of this: it starts on `v1`.

### Results of the comparison

Five tables, each with column heads only. Each one is filled with its date, split
and number of shots in the line above it, as the tables of
[Results](#results) are, and none holds a number before the job that produces it
has run. A cell that stays empty is a score nobody has stored; it is not 0 and not
a good result.

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

| Pair (a:b) | Signal | Source | Values per frame (a) | Values per frame (b) | Shots | Mean error (a) | Mean error (b) | Gap | Gap, low end | Gap, high end | Gap of b against a | Low end (b against a) | High end (b against a) | Word at one seed | Missing-channel ratio (a) | Missing-channel ratio (b) | Values to reach 0.0176 (a) | Values to reach 0.0176 (b) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

Not measured yet.

#### The networks

The same rows and columns as the linear arms, from the comparison files of M7 and
M8, with the gap of each seed and its spread added from `seeds`, which prints
`gaps ...; spread ...; word` per signal and writes nothing, so those columns are
copied from its print. The spread is the largest gap over the seeds minus the
smallest (rule 2), and the word is the verdict of the first pair with that spread.

| Pair (a:b) | Signal | Source | Values per frame (a) | Values per frame (b) | Shots | Mean error (a) | Mean error (b) | Gap | Gap, low end | Gap, high end | Gap of b against a | Low end (b against a) | High end (b against a) | Word at one seed | Missing-channel ratio (a) | Missing-channel ratio (b) | Values to reach 0.0176 (a) | Values to reach 0.0176 (b) | Gap, seed 0 | Gap, seed 1 | Gap, seed 2 | Spread of the gaps | Word with the spread |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

Not measured yet.

#### The trainings

One row per trained network, from `$S1/recoders/<name>/result.json` (its keys are
`best_step`, `best_val_mse`, `best_val_full`, `steps`, `seconds`, `plateau`,
`collapsed` and `usage`). `val_mse` is the validation squared error in example
units at the best step, averaged over the prefix lengths of the values, and
`val_full` the same at all values. `plateau` is yes if the best `val_mse` of the
last quarter of the log rows is not more than 1% under the best before it, and no
if it still fell by more than that, in which case the run is repeated with twice
the steps (rule 8). `collapsed` is
yes, no or unchecked (rule 7): a signal without validation examples has no
verdict, which is unchecked and not a "no", because no data is not a negative.
The active values are the count per signal of the values whose normalized entropy
is at least 0.3, and decide nothing.

| Name | Steps | Best step | `val_mse` | `val_full` | Plateau | Collapsed | Active values | Seconds |
|---|---|---|---|---|---|---|---|---|

Not measured yet.

#### The decisions

One row per signal that a rule decides, from M9 and the confirmation on
`val_report`. A choice made on `val_select` stands only if `val_report` gives the
same word. If `v2` equals `v1`, nothing was scored again and this table says so.

| Signal | Rule | Tokenizer a | Tokenizer b | Error of a | Error of b | Word | Held by `v2` | Held by `v2c` |
|---|---|---|---|---|---|---|---|---|

Not measured yet.

#### The 50 kHz band

One row per series signal with 250 samples per 5 ms frame (four on cache v1), from
`$S1/compare/fluct.csv`, which `fluct` writes (job M10). It is a statistic of the
power above the trend path, by frequency, and no tokenizer is built. With 250
samples of 0.02 ms in a 5 ms frame, group `i` is the band from `i` to `i + 1`
kHz (derived from the frame length, not measured). `band_to_trend` is the band's
power over the power of the trend values, and it is empty for the saddle coils,
because `v1` gives them K 1 (ruling C14). `persistence` is the median, over the
(shot, channel) pairs for which it is defined, of the correlation between the
logarithm of one frame's band power and the next frame's.

| Signal | K of the trend (`k_trend`) | Channel-frames (`n_frames`) | Band to trend (`band_to_trend`) | Share below 5 kHz (`share_below_5khz`) | Persistence (`persistence`) | Share of the band in each 1 kHz group (`bin_00` to `bin_24`) |
|---|---|---|---|---|---|---|

Not measured yet.

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
  transform here keeps at K of at most 16. No benchmark task scores the band, and
  plan 1c holds it as its last item.
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
  reconstruction and by the input round-trip of two tasks only (1-1 and 1-2, the
  direct models of rule 6), and no benchmark score says what the dynamics model
  gains or loses from one choice or the other (open decision 1 of the plan).
- **A learned tokenizer sees only what the front end keeps.** A recoder sits behind
  the top rung of a fixed codec (16 cosine values, 32 per band, 96 principal
  components), so it cannot use what that rung drops (for the 50 kHz channels, the
  part above 1.6 kHz), and what it can add at most is known from the ladder of
  stage 1b at that rung.
