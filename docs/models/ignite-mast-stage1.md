---
title: "IGNITE on MAST: stage 1 fixed tokenizers"
---

# IGNITE on MAST: stage 1 fixed tokenizers

**Status: code complete, not measured yet (2026-11-23).**

## What this is

Stage 1b gives each of the 39 cached MAST signals of TokaMark a tokenizer
without learned weights, a frozen assignment that says how many values each
signal keeps per 5 ms frame, and a measured representation floor per task in
TokaMark's own units. Three codecs do the work: a cosine transform per frame
(`frame_dct`), a principal-component basis for profiles and the flux map
(`linear`) and a band transform for the two Mirnov spectrograms (`band_dct`).
Stage 2 can start on these tokenizers alone, and every learned tokenizer of
plan 1c has this page's numbers as its baseline. The design is in
[the design page](./ignite-mast-design.md), the measurements this stage builds
on are in [the stage-0 page](./ignite-mast-stage0.md), and the plan is
`docs/superpowers/plans/2026-11-22-ignite-mast-stage1b-fixed-tokenizers.md`.

The code and its tests are committed. The long jobs that produce the numbers
(K1 to K7 under [How to run](#how-to-run)) have not produced a result yet, so
every results table below has its column heads and no rows. Nothing on this
page is measured except the dated numbers that carry their split, their number
of shots and the words "measured" or "rough look". A rough look is a small,
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
valid points with a small ridge, and a profile with no valid point has NaN
values. With 0 values there is no token and the decode is the mean profile.
Boundary coordinates `lcfs_r` and `lcfs_z` have separate bases. The domain is
one of `signal` (TokaMark units), `linear` and, for an asinh signal,
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

## How to run

The controller runs these jobs from `/home/nvidia/FusionAIHub` with
`IGNITE_MAST_ROOT` unset. In the commands `RT="scripts/mast/py.sh
scripts/mast/run_tokenizer.py"` and `S1=/models/intermediate/mast/stage1`.
Logs go to `$S1/logs/`. The durations are the plan author's estimates from
stage-0 timings (floor scoring of task 4-1 over 1,110 shots: 4,131 s; cache
build over 11,188 shots: 225 s, both measured 2026-11-22). None was measured for
this code.

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
| K1 fit | the `fit` command | `CUDA_VISIBLE_DEVICES="" $RT fit --workers 64` | none | 15 to 40 min | 39 lines. No signal says "no data". `n_skipped` of `magnetics-b_field_tor_probe_cc_field` is about a third of 2,000 and of Thomson about a fifth; the soft X-ray cameras skip a few shots (sampling-rate mismatch). If a signal has under 500 shots, stop and report. |
| K2 ladders | K1, the `ladder` command | `$RT ladder --task T [--variant V --gpu N --share-gpu]` for the six ladder tasks (`task_1-2`, `task_1-3`, `task_2-1`, `task_3-1`, `task_3-2`, `task_4-5`), and `$RT ladder --inputs`, all at once | one GPU for a few minutes per task with a variant; may be shared | 15 to 60 min each, about 1 h together | `$S1/ladders/*.csv`. Per signal: the floor falls with K; the first rung equals the "own mean" reference. For task 4-5: the frame oracle (the row `k0 = 32`), the mean baseline on the same shots, the contrast per rung. These three numbers and TokaMind's floor (0.3605 on `val_report`) go to the owner the same day (open decision 1). |
| K3 assign | K2 | `$RT assign --name v1` and `$RT assign --name ref --reference` | none | seconds | The printed table: which signals miss their target, and the total of values per frame (stage 2's sequence length follows from it). |
| K4 input round-trip | K3 | for tasks 1-1, 1-2, 2-1, 4-1: `$RT score --task T --split val_select --n-shots 100 --instrument raw --variant V --gpu N --share-gpu`, and the same with `--instrument inputs --assignment $S1/assignments/v1.json --mode unquantized` | one each, may be shared | 8 passes of 3 to 10 min | Per task the added error of rule 4 over the task's best published score, and the ratio `inputs / raw - 1`. Above 0.2 on any task: `$RT assign --name v1r --from v1 --raise-inputs`, repeat the four `inputs` passes with `v1r` once, and use `v1r` from here on. Both costs are recorded. |
| K5 confirmation | K4 | for the 14 tasks: `CUDA_VISIBLE_DEVICES="" $RT score --task T --split val_report --assignment A --mode M --num-workers 16` for the three read-outs, and with `--assignment $S1/assignments/ref.json --mode unquantized` | none | 56 passes of 10 to 40 min; about 2 to 3 h in waves of 14 | The last line of each pass. A pass that ends with a non-finite score or an error stops the wave. |
| K6 costs on `val_report` | K5 | `--instrument raw` and `--instrument passthrough --mode ranged` for every task with a signal that misses its target (expected: 3-2, 4-1, 4-2) and for 1-2 as a control; `--instrument inputs --mode ranged` and `raw` for 1-1, 1-2, 2-1, 4-1 | one each, may be shared | about 14 passes of 15 to 45 min; about 2 h on 4 GPUs | Per task and signal the pass-through cost; per task the input round-trip cost. A pass-through cost above 1% on `val_report` for a signal chosen by rule 3 is reported, and K is not changed again. |
| K7 gate | K5, the `gate` command | `CUDA_VISIBLE_DEVICES="" $RT gate --assignment $S1/assignments/<name>.json` | none | 5 to 15 min | The table and the last line. `exit 1` means a row blocks: a transform with more values than its reference can only lose to it through the read-out or a bug, so the controller stops and reports instead of lowering K. |

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

After K7 the controller fills the four tables below from
`$S1/assignments/<name>.json`, `$S1/gate/<name>.md` and the stored scores, sets
the status line, and commits with the numbers, their date, split and shot counts
in the body.

## Results

Four tables. Each carries its date, split and number of shots in the line above
it once it is filled. Until then each has column heads and no rows.

### The assignment

One row per signal, from `assignments/<name>.json`. A signal that missed its
target carries the miss in the note.

| Signal | Codec | K | Domain | Values per frame | Note |
|---|---|---|---|---|---|

Not measured yet.

### The gate

One row per (task, output signal) pair, the columns of `gate_rows`. `target` is
a fifth of the best published score of the task (empty for the spectrograms),
the three floors and `own_mean` are on `val_report`, and `bins_used_min`,
`worst_channel`, `worst_channel_error` and `contrast` are measured from the
cache on 200 `val_report` shots.

| task | signal | codec | k | domain | values_per_frame | target | unquantized | ranged | nearest | own_mean | bins_used_min | worst_channel | worst_channel_error | contrast | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

Not measured yet.

### The floors per task

The 14 tasks, scored on all `val_report` shots with the chosen assignment.

| Task | unquantized | ranged | nearest | A fifth of the best published score | Mean baseline | TokaMind's floor |
|---|---|---|---|---|---|---|

Not measured yet.

### The pass-through and input round-trip costs

Pass-through (rule 3 and K6): the prediction of a direct model sent through the
tokenizer, per task and signal. Input round-trip (rule 4 and K4, K6): the
model's inputs and actuators sent through the tokenizer, per task. The added
error is `sqrt(max(inputs**2 - raw**2, 0))` over the best published score of
the task, and it is filled for the input round-trip only.

| Measure | Task | Signal | Split | Shots | Raw score | Score through the tokenizer | `through / raw - 1` | Added error over the best published score |
|---|---|---|---|---|---|---|---|---|

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
  the same predictor differs between the two splits by up to 0.04.
- **The rough looks of 2026-11-22 are not results.** Every rough look quoted on
  this page was made on the CPU from cache v1 on 24 to 96 training shots with
  the code of that day (24 shots for the cosine truncation, 72 fitted and 24
  tested for the principal components, 40 for the spectrograms, 96 for the
  Thomson missing-data count). They set expectations only, and the ladders of
  this stage replace them.
- **Task 4-5's frame statistics (open decision 1 of the plan).** The design
  scores task 4-5 from frame statistics, one value per 5 ms frame and bin. The
  rough look of 2026-11-22 on 40 training shots says three things about it:
  the error of the Mirnov task sits in the lowest frequencies (99.8% of the mean
  baseline's squared error for the toroidal-field probes and 77% for the
  poloidal ones is below 31 kHz); a perfect prediction of every frame's mean
  spectrum would still score about 74% of the mean baseline, because the
  targets are single 1 ms periodograms that scatter inside a frame; and
  TokaMind's own floor is lower (0.3605 on `val_report`, measured 2026-11-22)
  because it keeps detail inside 5 ms. What stage 0 measured on 2026-11-22 on
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
- **No data is not a negative.** A "no data" cell is a signal that no fitted
  shot has, and an empty cell is a score nobody has stored yet. Neither is a
  good result and neither is 0.
