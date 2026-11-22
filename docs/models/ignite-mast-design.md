---
title: "IGNITE on MAST: design"
---

# IGNITE on MAST: design (2026-11-21, amended 2026-11-22)

**Status: design. Stage 0 is built and measured on 2026-11-22 (the direct
model's rows of the headroom table are still being trained): see
[stage 0 results](./ignite-mast-stage0.md). Nothing else described here is
built.** On 2026-11-22 Nathaniel
answered the request for approval with "ok yes" and raised the bar. That is
read here as approval of the design with the changes listed under
[What changed on 2026-11-22](#what-changed-on-2026-11-22). The next document
is an implementation plan. Decisions are marked **Decided** (Nathaniel, with
the date) or **Proposed** (the reasoning is given, and it can be overruled).
Measurements carry the date they were taken on
[node13](../clusters/nvidia.md).

## Intent

node13 is available for about three weeks from 2026-11-21 (to about
2026-12-12). Private DIII-D data cannot be put on it, so the work is
[IGNITE](./ignite.md) trained from nothing on public MAST data. Three things
are wanted, in this order (Decided, 2026-11-21):

1. An IGNITE that works on MAST: an autoregressive simulator that is
   generative, physically accurate and ready for control use, and whose own
   output is the best result on the TokaMark benchmark by a clear margin,
   ahead of both TokaMind and GenToka (Decided, 2026-11-22: "this must be by
   far the best or else it wont be significant").
2. The groundwork for one model that serves both DIII-D and MAST although the
   channels and diagnostics differ.
3. Changes to the architecture where they can be shown to improve the model.

The data arrives in steps. TokaMark v1 (11,573 shots) is on disk now and is
enough to build everything. The full FAIR-MAST archive is expected around
2026-11-25. The FusionEquilibrium challenge data comes later and is not part of
this design.

## What changed on 2026-11-22

Two messages from Nathaniel changed the design after its first version.

- **The bar is raised.** The model has to beat GenToka as well as TokaMind,
  "by far", and it has to do so as an autoregressive, control-ready
  simulator, "which nothing else can do". The first version had a regression
  head as its headline and asked the generated output only to reach
  TokaMind's level. That is replaced. See [The bar](#the-bar), goals G4, G4b
  and G7, [Three ways to read the simulator](#three-ways-to-read-the-simulator),
  [Control readiness](#control-readiness) and
  [How a change is judged](#how-a-change-is-judged).
- **The confinement-time scaling is for MAST only for now** and a low
  priority, "but as long as it can theoretically scale to d3d that would be
  great". The formulation that allows this is in [Time base](#time-base).

Three consequences are proposals made in amending the design. They are marked
Proposed where they appear.

- Precision is taken from expected values over ordered bins, not from more
  codes ([Tokenizers](#tokenizers)).
- The slow-series tokenizer is a fixed transform by default, and the learned
  shared tokenizer is compared against it at the gate. This changes how the
  shared-tokenizer decision of 2026-11-21 is carried out for one family, so
  it is flagged for Nathaniel.
- Stage 0 gains a headroom study. It says, per task, how far the published
  models are from what the data allows ([Order of work](#order-of-work)).

## The bar

TokaMark (arXiv 2602.10132) defines 14 tasks in 4 groups over 39 MAST signals.
Every score is a normalized RMSE (NRMSE, lower is better). Two models have
published results: TokaMind (arXiv 2602.15084, a 6.9M-parameter transformer)
and GenToka (arXiv 2611.08153, 5 November 2026, a conditional diffusion
transformer). GenToka is the best published result on 11 of the 14 tasks, so
it is the real bar, not TokaMind.

| Task | Predicts | From | Input → output | TokaMind paper | TokaMind re-run by GenToka | GenToka | Our TokaMind replication |
|---|---|---|---|---|---|---|---|
| 1-1 | 15 equilibrium scalars | magnetics, coil currents, Ip | same 5 ms | 0.150 | 0.1363 | **0.0941** | 0.1503 |
| 1-2 | plasma boundary (R, Z) | same | same 5 ms | 0.043 | 0.0439 | **0.0412** | 0.0433 |
| 1-3 | flux map ψ(R, Z), 65×65 | same | same 5 ms | 0.122 | 0.1200 | **0.1156** | 0.1198 |
| 2-1 | coil currents, Ip, 15 scalars | same, plus future coil voltages and NBI power | 5 → 25 ms | 0.181 | 0.1289 | **0.0880** | 0.1809 |
| 2-2 | plasma boundary | same | 5 → 25 ms | 0.043 | 0.0408 | **0.0370** | 0.0439 |
| 2-3 | flux map | same | 5 → 25 ms | 0.092 | 0.0866 | **0.0828** | 0.0917 |
| 3-1 | Thomson Te, ne profiles | one Thomson sample, plus future Ip and density requests, NBI, gas | 5 → 50 ms | 0.346 | 0.3477 | **0.3311** | 0.3457 |
| 3-2 | D-alpha, two soft X-ray cameras | Thomson, D-alpha, soft X-ray, same actuators | 5 → 50 ms | 0.269 | 0.2529 | **0.2522** | 0.2671 |
| 3-3 | Thomson profiles, three betas | whole history of magnetics, coils, Ip, line density | history → 5 ms | 0.220 | 0.2415 | **0.2147** | 0.2197 |
| 4-1 | two soft X-ray cameras | whole history of 14 signals | history → 100 ms | 0.213 | **0.2122** | 0.2298 | 0.2147 |
| 4-2 | two soft X-ray cameras | as 4-1 plus Thomson | history → 100 ms | 0.214 | **0.2143** | 0.2302 | 0.2155 |
| 4-3 | magnetic axis height | as 4-1 | history → 100 ms | 0.080 | 0.0781 | **0.0691** | 0.0810 |
| 4-4 | plasma current | as 4-1 | history → 100 ms | 0.349 | 0.3470 | **0.3116** | 0.3502 |
| 4-5 | Mirnov power spectra | as 4-1 | history → 100 ms | 0.651 | 0.7377 | 0.7487 | 0.6499 |
| Mean | | | | 0.212 | 0.2134 | **0.2033** | 0.2124 |

Three things in these numbers shape the design:

- **Pretraining has bought little so far.** In TokaMind's own table the model
  trained from scratch matches or beats the fine-tuned one on 7 of the 9
  tasks in groups 1 to 3 and is never more than 0.007 worse. GenToka from
  scratch averages 0.2146 against 0.2033 fine-tuned. A much larger
  model on the same 1.3 hours of plasma will not win on size alone. Whether
  IGNITE's size helps is a hypothesis this work tests, with a from-scratch
  control beside the pretrained model on the probe tasks.
- **Nobody has solved task 4-5.** Predicting the mean scores 0.653. Every
  model is at or above that.
- **The published TokaMind numbers disagree with each other** (task 2-1: 0.181
  in the paper, 0.129 in GenToka's re-run). The reference is therefore our own
  replication with today's benchmark code
  (`/models/tokamind/run_all_tasks.sh`, finished 2026-11-22 04:36 UTC). It
  reproduces the paper: all 14 tasks are within 0.0022 of the paper's
  fine-tuned column (largest difference task 1-3, 0.1198 against 0.122), and
  the mean is 0.2124 against 0.212. Task 2-1 comes out at 0.1809, so the
  paper's 0.181 is what today's code gives and GenToka's 0.129 for TokaMind
  on that task is not reproduced here.
- **A few tasks carry the mean.** Task 4-5 alone is 26% of GenToka's summed
  score (0.749 of 2.846). The six tasks of groups 1 and 2 together are 16%
  (0.459). A model that only matched the best published number on every task
  would average 0.194, which is 4.7% below GenToka. A clear margin on the
  mean therefore needs real gains, and the largest levers are task 4-5 and
  groups 1 and 2.
- **Where a margin could come from.** These are hypotheses, and the headroom
  study of stage 0 tests each of them before any large run depends on it.
  First, both published models represent signals by a truncated set of
  cosine-transform coefficients. TokaMind keeps coefficients up to 99.9% of
  explained variance for inputs and 99.5% for outputs (its `embeddings.yaml`,
  read 2026-11-22), and GenToka denoises coefficients of the same kind (its
  truncation was not checked). Groups 1 and 2 predict EFIT's outputs from
  EFIT's own inputs, which is close to a deterministic map, so the
  representation may be a visible part of scores between 0.04 and 0.12.
  Second, GenToka scores one generated sample. Where the target has spread
  that cannot be predicted, the mean of the predictive distribution scores
  better than one sample, by up to a factor of √2. GenToka's losses to
  TokaMind on tasks 4-1, 4-2 and 4-5 are consistent with that. Its near tie
  on task 3-2 is not, so the size of this effect is unknown. Third, on task
  4-5 the number to beat is the mean baseline, and a prediction of the
  conditional mean spectrum from the recent spectra is cheap to test.
- **Where no margin is likely.** Thomson profiles carry measurement scatter.
  The soft X-ray channels vary by 0.55 of a per-channel standard deviation
  at time scales shorter than 0.1 ms (spike of 2026-11-21), which no
  forecast 5 to 100 ms ahead can follow. TokaMind and GenToka are within
  0.3% of each other on task 3-2 and within 5% on 3-1, which is what a
  shared noise floor would look like.

## Goals

- **G1. A MAST data path for IGNITE**, reading the TokaMark stores now and
  FAIR-MAST when it arrives.
- **G2. Tokenizers shared across signals of the same kind**, one per signal
  family, that pass a fidelity gate.
- **G3. A dynamics model pretrained on MAST tokens** that rolls a shot forward
  under given actuators, as IGNITE does on DIII-D.
- **G4. The simulator's own output is the best result on TokaMark by a clear
  margin** (Decided, 2026-11-22; the numbers are Proposed). All 14 tasks are
  scored under the benchmark's own protocol from what the autoregressive
  model itself predicts, decoded from its tokens, with no regression head.
  Proposed reading of "by far the best": a mean NRMSE at least 10% below
  GenToka's 0.2033 (0.183 or lower); better, beyond noise, than both
  GenToka's published score and our TokaMind replication on at least 10 of
  the 14 tasks; and worse beyond noise on none. For scale, GenToka's own
  margin over its re-run of TokaMind is 4.7%.
- **G4b. One set of weights, and honest samples.** Reported beside G4 in
  every table: the same simulator with no per-task adaptation (one set of
  weights for all 14 tasks), the score of a single sample (what GenToka
  reports), and the calibration of the sampled spread.
- **G5. Each architecture change measured** against the unchanged version.
- **G6. Hooks for the two later projects**: channel identity that comes from a
  table, not from code, and time expressed so that the same weights can run
  at another machine's time scale.
- **G7. Control readiness, measured** (Decided, 2026-11-22; the checks are
  Proposed). The model steps under any actuator trajectory, is causal, holds
  over a whole shot, responds to actuators as the machine did on held-out
  shots, and its speed is known. See
  [Control readiness](#control-readiness).

## Non-goals

- Training on DIII-D data, or a joint DIII-D + MAST model. That needs data
  that cannot be on this machine. This design only makes it possible later.
- Fitting or testing a confinement-time-scaled time representation on a
  second machine. The formulation is built so that it can extend to DIII-D
  (Decided, 2026-11-22; see [Time base](#time-base)), and one test of it runs
  on MAST. A measurement spike informs it; see
  [Later projects](#later-projects).
- A controller. This work delivers a simulator that a controller can use,
  and the evaluations that say how far it can be trusted. A small tracking
  demonstration inside the simulator is a stretch item.
- The FusionEquilibrium challenge and GenToka's two extra task groups
  (actuator inversion, disruption prediction).
- Tokamak-SI for MAST. Nathaniel asked on 2026-11-22 for the Tokamak-SI event
  labels to be made for MAST, where they could become a second benchmark. It
  is wanted, but it is a separate track with its own design, and nothing in
  this design waits for it. A feasibility study is under
  `/models/intermediate/spikes/tokamaksi_mast/`. Where the two tracks meet is
  noted under [Later projects](#later-projects).
- Tokenizing the beam emission (2 MHz, in 19% of shots) and OMAHA (10 MHz, in
  21% of shots) channels in the first version. No benchmark task uses them and
  they are most of the bytes. The high-rate tokenizer is built so they can be
  added.
- The benchmark's temporal split as a headline. It needs its own pretraining
  run (see [Splits](#splits-and-leakage)) and is a stretch item.

## Anti-goals

- **Not a benchmark regressor with a generative label.** The headline numbers
  come from the simulator's own autoregressive output. A regression head is a
  diagnostic column and is never the headline.
- **No margin that comes from scoring differently.** Every table says which
  read-out produced each number. A mean over samples always has the
  single-sample score beside it, because GenToka reports one sample.
- **No test leakage and nothing from the future.** Test shots never reach a
  tokenizer, a normalization statistic or a pretraining batch. No prediction
  uses a measurement later than the frame being predicted. Nothing is
  normalized or conditioned with a statistic of the shot being predicted.
  IGNITE v4 z-scored actuators per shot, which uses the future. A shot's
  measured confinement time is an outcome of the shot for the same reason.
- **No silent filling.** Missing data is masked and carried as missing. A
  zero is never written where a measurement is absent.
- **The DIII-D path does not change.** The existing IGNITE code keeps its
  behaviour and `tests/ignite` stays green. New work is added beside it.
- No data or runs in the repository, and no disturbance of the TokaMind
  replication jobs.

## Constraints

**Time.** About 21 days, with the largest data delivery inside the window.

**Machine.** 4 GPUs with 286 GB each, 352 cores, 1.4 TB RAM, aarch64, the pixi
`nvidia` environment. Everything written goes under `/models`. Access ends, so
whatever is worth keeping must be copied off the node before 2026-12-12.

**Data** (profiled 2026-11-21, `/models/intermediate/spikes/data_profile/PROFILE.md`):

- 11,573 shots, 4,737 s of plasma in total (1.3 hours), median shot 386 ms.
  At 5 ms that is 941,846 plasma frames. DIII-D IGNITE v4 trained on 8,752
  shots of several seconds each.
- The stores are Zarr v2, float64. Sampling rates span 200 Hz (EFIT, Thomson)
  to 10 MHz. One cold pass over everything takes about 4.5 minutes with 48
  processes.
- Quality: EFIT is 38% NaN overall (7% inside the plasma), Thomson 48% NaN.
  Dead-channel rates are 3% to 21% depending on the probe array. There are
  sentinel values (X-point −9.99, saturation at ±5 V). 15 arrays in 10 stores
  have missing chunk files, which Zarr returns as NaN without an error. Soft
  X-rays are sampled at 500 kHz instead of 50 kHz in a minority of shots. The
  plasma current changes sign between shots.
- What FAIR-MAST adds (from its metadata tables, 2026-11-21): level 2 is the
  same 11,573 shots as TokaMark, so it adds no shots. Level 1 has 15,969
  shots, 4,720 of them not in TokaMark, and 50 raw sources against level 2's
  10, but under raw MAST signal names that level 2 has not mapped. More
  pretraining data therefore means doing that mapping for the extra shots.
- The stores carry IMAS names, units and channel names for every signal.
  They carry positions only for Thomson (major radius) and the EFIT grid.
  There are no probe positions for the magnetics (checked on shot 30420).

**Benchmark protocol** (read from the TokaMark source, 2026-11-21):

- A window is cut every 1 ms. Inputs cover the input length before the cut.
  For the "non-Markovian" tasks (3-3 and all of group 4) the input is the
  whole shot up to the cut, with at least 150 ms of history. Actuators are
  given over the input window and the output window. Outputs start at the cut.
  Group 1 is the special case where the output window is the input window.
- Every window length is a whole number of 5 ms: 1, 5, 10, 20 and 30.
- The score divides each signal's RMSE, per shot, by one standard deviation
  for the whole signal (all channels together) from the benchmark's statistics
  file, averages over shots, then over the task's signals. NaN targets are
  ignored. Predictions are compared in physical units.
- Task 4-5 is scored on linear power spectra: 512-sample windows without
  overlap (1.024 ms), 256 frequency bins, for 6 Mirnov channels.
- The random split has 8,963 training, 1,115 validation and 1,110 test shots.
  385 shots with negative plasma current are held outside all three.
- Any model can be scored: the evaluator takes arrays of targets and
  predictions per window (`WindowMetricsAccumulator.add_batch`, then
  `compute_metrics`).

**Code** (dev-peter's IGNITE, mapped 2026-11-21): the DIII-D assumptions are
constants, not configuration. The frame is 50 ms in four places, the layout is
20 seed frames plus 80 predicted, the actuators are 88 named DIII-D channels,
each tokenizer concatenates all channels of one diagnostic and has no notion
of channel identity, normalization statistics are tensors in DIII-D channel
order, and the dynamics model raises on a missing modality and fills absent
ones with placeholder codes.

## Approaches

Three ways to build the tokenizer layer were considered. The dynamics model
and the benchmark work are the same under all three.

**A. Straight port.** One tokenizer per MAST diagnostic, as on DIII-D. Lowest
risk in the code. It means more than twenty tokenizers, each trained on a
slice of 1.3 hours of plasma, and nothing carries over to another machine.

**B. One tokenizer per signal family, tokens per channel.** About four
tokenizers: high-rate channels, slow time series, profiles and 2-D maps. Each
channel in each frame is a training example, and a channel-identity embedding
tells the tokenizer which channel it is looking at. The high-rate tokenizer
then learns from about 70 channels instead of 3 to 18 (more once beam
emission is added), a missing channel is simply absent, and adding DIII-D
later means new identity rows on the same weights. The cost is that structure
across channels, such as a mode number read from the phase between coils, is
no longer captured inside the tokenizer and has to be learned by the dynamics
model's attention.

**C. One tokenizer for everything.** The most sharing, but 65×65 flux maps and
500 kHz waveforms in one network is unproven and too risky for three weeks.

**Picked: B** (Decided, 2026-11-21). It is the idea Nathaniel raised ("a
single encoder that trains a single instance on all the high frequency
modality ... so you have much more data for the individual encoders"), and he
confirmed the family split with one token per channel per frame as the way to
carry it out. Two tokenizers of kind A are kept as controls so that shared
against specialized is measured through the same gate. The amendment of
2026-11-22 keeps B. It changes how the slow-series family carries it out (a
fixed transform by default; see [Tokenizers](#tokenizers)).

## Design

### Time base

- **Decided (2026-11-21):** the world-model frame on MAST is 5 ms, one tenth
  of the DIII-D frame, "for now". One frame is then exactly one EFIT or
  Thomson sample, every benchmark window is a whole number of frames, and
  TokaMind itself cuts its inputs into 5 ms chunks (its `chunk_length`).
- **Proposed:** the model's context is up to 100 frames (0.5 s), which is a
  whole MAST shot. The seed length is random during training, from 1 frame to
  the full context, instead of IGNITE's fixed 20. The benchmark needs both
  ends: group 2 forecasts from a single frame, group 4 from the whole history.
  TokaMind truncates history to 50 chunks, so full history is a place to gain.
- **Proposed:** frames are anchored to the benchmark's cut time, not to a
  fixed grid. Windows are cut every 1 ms, so there are five phases of the
  5 ms grid. Training samples all five, which is also augmentation.
- **Proposed:** the frame duration is an input to the dynamics model (an
  embedding of its logarithm). It is constant at 5 ms in the first version.

**Time scale across machines (Decided, 2026-11-22):** the confinement-time
scaling is worked out on MAST only for now and is a low priority, but the
formulation has to be able to extend to DIII-D. The formulation **proposed**
for that has four parts. Each is something this design needs anyway.

1. **Time enters in physical units.** A frame's position is encoded from its
   time in seconds (a rotary encoding), not from a frame index. A 5 ms MAST
   frame and a 50 ms DIII-D frame then sit on one time axis under the same
   weights.
2. **The frame duration and a reference time scale for the machine are both
   inputs.** The reference is one constant per device: its typical energy
   confinement time. On MAST that is 45 ms, the median of FAIR-MAST's
   `generic_max_energy_time` over 500 shots (the definition of that column
   was not verified). It is not the measured confinement time of the shot
   being predicted, which is an outcome of that shot. With both inputs the
   model can form the frame duration in confinement times, which is the
   quantity Nathaniel's proposal scales with. Whether 5 ms on MAST and 50 ms
   on DIII-D are the same fraction of a confinement time has not been
   checked, and with both as inputs they do not have to be.
3. **Time resolution is nested inside the frame.** A token's values are
   ordered from the frame mean to finer detail, and any prefix of them is a
   valid, coarser token (see [Tokenizers](#tokenizers)). This is the
   Matryoshka part. The same number of values describes a 5 ms frame and a
   50 ms frame at the same resolution relative to the frame.
4. **The slow-series tokenizer does not depend on the frame duration.** It is
   a fixed transform, so another frame duration needs no tokenizer training.

MAST alone can test part of this: train at 5 and 20 ms frames and test at
10 ms, which shows whether the duration input carries the model to a frame
length it has not seen (stage 3). MAST alone cannot test whether the
confinement time is the right unit. The spike found that it does not
organize time scales within MAST
([first measurement](#time-scale-a-first-measurement)). That needs DIII-D.

### Data path

- **Standardized cache.** One pass turns each shot's signals into arrays of
  standardized values under `/models/intermediate/`, with NaN marking a
  sample that is not valid. The model never reads Zarr during training.
  As built on 2026-11-22 (`/models/intermediate/mast/cache/v1`, 11,188
  shots, 137.8 GB) the arrays are float32, not the float16 first planned:
  float16 could not hold an `asinh`-compressed heavy-tailed signal to 1e-3 of
  the benchmark's standard deviation, and float32 keeps the cache exact to
  about 1e-6, so the choice of training units stays open. A per-signal dtype
  is a question for the raw 500 kHz waveforms.
- **Validity is explicit.** NaNs, sentinel values, dead channels and missing
  chunk files all become "not valid". A channel that is invalid for a frame
  produces no token for that frame. Two cases are flagged instead of
  invalidated in cache v1, because the benchmark scores them: clipped
  (saturated) samples stay valid and are marked in a `<signal>__saturated`
  dataset, and a channel that is constant within a shot stays valid and is
  listed in `const_channels` (two thirds of those are constant by
  construction, such as boundary anchor points and programmed references).
- **Normalization** uses one set of statistics per channel, computed on
  training shots only and fixed. Signals with heavy tails (soft X-ray,
  D-alpha, Mirnov power) get a monotone compression before standardizing; the
  choice per family is left to the builder. The sign of the plasma current is
  kept as recorded.
- **The channel registry** is a table with one row per channel: device,
  IMAS name, signal, channel index and name, family, sampling rate, units,
  position where known, normalization statistics and quality flags. Nothing
  in the model code names a channel. This table is what the cross-device work
  extends: a DIII-D channel is a new row, and the IMAS name gives a
  device-independent description of what the channel measures.

### Splits and leakage

- **Proposed:** the headline uses the benchmark's random split, which is what
  both published tables report. Tokenizers, statistics and pretraining see
  the 8,963 training shots only. The 1,115 validation shots select models. The
  1,110 test shots are read once, for the final table.
- The temporal split cannot share this pretraining: 2,488 of the random
  split's training shots are test shots of the temporal split (measured
  2026-11-21). Reporting it needs a second pretraining run on its own 6,525
  training shots. That is a stretch item.
- The 385 negative-current shots stay out of the first version. Whether
  adding the 308 "training" ones helps is a later measurement.
- When FAIR-MAST arrives, every TokaMark validation and test shot number is
  excluded from anything trained on it.

### Tokenizers

Every tokenizer is fixed before the dynamics model is trained and has to pass
a gate first, as in IGNITE's Phase A. The learned ones follow its pattern: an
encoder, a finite scalar quantizer and a decoder. Four things are new and
common to all four families:

- **One example is one channel (or one profile, or one map) in one frame.**
  The network's weights are shared by every channel of the family.
- **Channel identity** enters as an embedding built from the registry row (a
  small network over the metadata) plus a learned per-channel vector that can
  be dropped during training, so the tokenizer still works for a channel it
  has no learned vector for.
- **A token's content is a short list of bounded, ordered values, and
  precision comes from expected values over ordered bins** (Proposed,
  2026-11-22). For each value the dynamics model predicts a distribution over
  ordered bins. Its training target is a smoothed histogram built from the
  unquantized value (the two neighbouring bins, weighted by distance), not a
  single code index. The mean of the predicted distribution is then a
  continuous number that is not limited by the bin width, and a sample is
  still a sample. The latents of a finite scalar quantizer already have this
  form: each dimension is an ordered scale with 5 to 8 levels. What changes
  is the target and the read-out. DreamerV3 predicts rewards and values this
  way, and Farebrother et al. (2024) found classification over bins to beat
  squared-error regression for value functions (both cited from memory;
  neither was checked against the paper library). Reason for needing it: the
  current slow-signal tokenizer is too lossy for this benchmark by an order
  of magnitude. Its recorded error on DIII-D is 0.68 in normalized units,
  worse than the 0.56 of replacing each window by its time mean
  ([v4 diagnosis](./ignite-v4-diagnosis.md)), where the boundary task needs
  about 0.04.
- **Tokens are nested from coarse to fine.** For a time series the order is
  time resolution: the first value of a token is the frame mean, and each
  further one adds finer detail inside the frame (cosine-transform
  coefficients in order of frequency). For a learned tokenizer the order is
  precision: a second quantizer level encodes what the first missed. In both
  cases the later parts are dropped at random during training, so a prefix
  is always a valid, coarser token, and a further part can be added later
  without changing what the dynamics model has already learned. The nesting
  in time is the Matryoshka idea on the axis Nathaniel proposed it for (see
  [Time base](#time-base)).

Also proposed for every learned tokenizer: the quantizer is skipped for a
random half of the training examples, so the decoder and the dynamics model
both learn to accept unquantized encoder outputs. This is what lets the model
read its inputs at full precision and take its own expected output back in as
the next input (see
[Three ways to read the simulator](#three-ways-to-read-the-simulator)).

The four families:

| Family | Signals in the first version | One example | Tokens per frame (estimate) |
|---|---|---|---|
| Slow series | 91 magnetic probes and flux loops (5 kHz), 15 coil currents and voltages, Ip, NBI power, line density, gas, the two requested waveforms (4 kHz), 15 EFIT scalars (200 Hz) | one channel's 5 ms of samples, resampled to a fixed length | 1 per channel, about 127 |
| High-rate | saddle coils (12), soft X-ray cameras (36), D-alpha (3) at 50 kHz; Mirnov coils (6 scored, 11 available) at 500 kHz | see below | about 170 |
| Profiles | Thomson Te and ne (120 radial points), plasma boundary R and Z | one profile at one time, with its coordinates and validity mask | about 8 per profile, 24 |
| 2-D maps | flux map ψ (65×65) | one map at one time | about 16 |

That is about 340 tokens per frame, against 1,209 on DIII-D. All counts are
estimates for the builder to settle.

**The slow-series tokenizer is a fixed transform by default** (Proposed,
2026-11-22; it changes how the decision of 2026-11-21 is carried out for this
family, so it is flagged). For each channel and frame, after the compression
and standardization of the data path, the token holds the first K
cosine-transform coefficients of the frame, each on a fine scale of ordered
bins. K = 1 is the frame mean. It is still one token per channel per frame,
and K is set per signal by the gate. Reasons:

- It has no weights, so it cannot collapse, and its error is known before
  anything is trained. For a signal sampled once per frame (the EFIT scalars)
  K = 1 is exact.
- At a 5 ms frame there is little for an encoder to learn here. Between 2%
  (magnetics) and 18% (coil currents) of a slow signal's variance lies inside
  a frame, and for magnetics cosine coefficients beat segment means at equal
  count (spike of 2026-11-21).
- It is the same for every channel and every machine, which is the limiting
  case of a shared tokenizer, and it works at any frame duration.

The learned shared tokenizer for this family is still trained, and it goes
through the same gate with the same number of values per token. It replaces
the fixed transform for any signal where it has the lower error. A learned,
shared tokenizer stays the default where there is structure to learn: the
fluctuation path over all high-rate channels, which is the case Nathaniel
raised, and the profiles and maps.

**The high-rate family has two paths**, because the benchmark scores soft
X-ray and D-alpha as waveforms while IGNITE's spectrogram tokenizer keeps only
the statistics of the fluctuations:

- A **trend** path. The channel is low-pass filtered and decimated to the slow
  family's rate, and then goes through the slow-series tokenizer like any
  other slow channel. This puts yet more channels through that one encoder.
- A **fluctuation** path. The short-time spectrum (log power) is cut into
  bands of fixed width in hertz, and each band becomes a token from one
  shared band encoder that is told which band it has. A 50 kHz channel has
  one band and a 500 kHz channel has ten. Different sampling rates become
  different numbers of tokens instead of different networks. This follows
  FISHER's sub-band design and keeps IGNITE's "statistics first" training for
  this path.

**The gate** extends `gate.py`, whose metrics are already generic. Its
thresholds were calibrated on DIII-D and are re-derived here. Per family it
checks reconstruction error per signal in the benchmark's own units, code
usage (a tokenizer that collapses to a few codes fails, which has happened
three times on DIII-D through wrong statistics), and for the fluctuation path
the existing spectral statistics. Results are reported per channel, including
the worst one. Each learned profile or map tokenizer is also compared with a
linear baseline: principal components fitted on the training shots, with the
same number of values. A proposed target for any signal a task predicts:
reconstruction NRMSE at most a fifth of the best published score for that
task (about 0.007 for the boundary and 0.017 for the flux map), measured on
the path the headline uses, which is decoding from unquantized values.

What blocks and what does not (Proposed):

- A tokenizer that has collapsed, or that does not beat replacing each window
  by its own mean, blocks. The dynamics model is not trained on it. A learned
  tokenizer that does not beat its linear baseline is replaced by the
  baseline.
- The fidelity target matters more than it did in the first version, because
  the headline is now decoded from tokens: the tokenizer's error is a floor
  under the headline score. Errors add in quadrature, so a floor at a fifth
  of a score costs about 2% of it. Missing the target still does not block
  the start of stage 2. Pretraining starts on its date with the best
  tokenizer available for each signal, and the miss is recorded with its
  number. It has to be closed before the first complete table: by more
  coefficients for a fixed transform, and by a further level or the linear
  baseline for a learned tokenizer.

**Controls:** a magnetics-only slow tokenizer and a Mirnov-only fluctuation
tokenizer, built the old way with all channels concatenated, go through the
same gate. If a specialized one clearly beats the shared one, that signal uses
the specialized tokenizer and the result is recorded.

### Dynamics model

Kept from IGNITE: masked-token prediction over frozen codes (MaskGIT), the
transformer that alternates attention within a frame and causal attention
across frames, the FlashAttention-4 dispatch, and scheduled sampling at 0.75,
which was the best setting in dev-peter's study.

Changed, each behind a switch so it can be measured against the old behaviour
where the old behaviour can run at all:

1. **A token is identified by its registry row**, not by its position in a
   fixed layout. The embedding of a token is built from its values, its
   channel identity and its role. Each value enters as a weighted mix of the
   embeddings of its two nearest bins, so a measured value, a sampled bin and
   an expected value all enter the same way, at full precision.
2. **Absent tokens are masked** out of attention and out of the loss. No
   placeholder codes.
3. **Roles replace the fixed actuator vector.** Each token in each frame is
   either observed, given (an actuator, known for future frames too) or to be
   predicted. Which channels play which role is set per batch. Reason: on
   MAST the same channel changes role between tasks. Coil currents are an
   input in group 1 and an output in task 2-1. IGNITE's summed actuator vector
   is kept as the control. Actuators use the fixed training-set statistics.
4. **The values of a token are predicted in order.** The backbone predicts
   the distribution of the first value. A small head predicts each further
   one and is told the ones before it, so the backbone runs once.
5. **Training covers the conditioning patterns the benchmark uses.** Random
   seed length, random horizon, and whole signal groups dropped from the
   context at random, so that "equilibrium from magnetics alone in the same
   frame" and "forecast from one frame" are ordinary training cases. Actuator
   tokens are dropped at random too, which also enables classifier-free
   guidance at sampling time.
6. **The loss is weighted per signal group**, not per token. With a uniform
   per-modality weight, IGNITE v4 put 8/15 of its loss on 33 of its 1,209
   tokens.
7. **Rotary position encoding on physical time** instead of the learned table
   of 100 frame positions, so that neither the context length nor the frame
   duration is fixed by a table. The
   [time-scale formulation](#time-base) needs it, so it is adopted unless it
   measures worse.
8. **Frame duration and the machine's reference time scale** as inputs, as
   above.
9. **The model's own expected output is a training input** (Proposed,
   2026-11-22). IGNITE's scheduled sampling feeds sampled codes back in.
   Here a share of the fed-back frames carry expected values instead, so
   that the mean roll-out is a regime the model was trained in.

**Size.** Three sizes are trained: about 30M, about 100M and about 300M
parameters (the last is the shape of IGNITE v4). The choice is made on
validation loss and on validation NRMSE for a few benchmark tasks. With about
7,300 non-overlapping 100-frame sequences in the training shots, the largest
may well not be the best. Throughput at these sizes has not been measured on
this machine and no figure survives from v4, so the first step of the plan
measures it.

### Three ways to read the simulator

A benchmark score needs one predicted value for every target sample. The
simulator predicts distributions over token values, and there are three ways
to turn those into predictions. All three are reported. The first is the
headline.

- **Mean roll-out (the headline).** The model steps forward one frame at a
  time. At each frame the predicted tokens are replaced by the expected
  values of their distributions. Those are decoded to give the prediction,
  and fed back as the input for the next frame. It is deterministic and
  costs one pass per frame. It estimates the conditional mean, which is what
  an RMSE rewards.
- **Sampled roll-out.** Each frame is sampled by iterative unmasking and fed
  back as a sample, as IGNITE does today. It is scored as one sample, which
  is GenToka's protocol, and as the mean of several. Calibration is measured
  on it.
- **Regression read-out (a diagnostic).** Heads on the backbone's hidden
  states, trained with squared error. This was the headline of the first
  version of this design. It is not autoregressive. It shows how much of
  what the backbone knows the token path delivers. A gap between it and the
  mean roll-out is a defect of the token path to be fixed, not a better
  number to report.

Group 1 has nothing to roll: its output window is its input window, so the
targets are inferred inside the frame, by masked prediction from the observed
tokens. Groups 2 to 4 roll forward between 1 and 20 frames under the task's
actuators.

Feeding back expected values is not the same as averaging sampled futures
when the dynamics are nonlinear over the horizon. The two are therefore
compared on a common subsample of windows for every task, and the table says
which one each number is.

### TokaMark adaptation and scoring

- **Strict protocol.** For each task the model is given exactly the task's
  input signals over the task's input window, the task's actuators over input
  and output windows, and nothing else.
- **Adaptation without new outputs.** For the headline, the pretrained
  simulator is trained further on one task's conditioning pattern, with the
  token objective it was pretrained with and no added output network. That
  gives one adapted copy per task, as TokaMind and GenToka have. Models are
  selected on validation NRMSE of the mean roll-out. The unadapted simulator
  is scored on all 14 tasks too (goal G4b). The task patterns are part of
  its pretraining mix, so none of its weights belong to one task.
- **Inputs at full precision.** Observed values enter as their continuous
  values, not as sampled bins, so input precision is not limited by the bin
  width.
- **Scoring goes through the benchmark's own code.** Test windows are produced
  by TokaMark's dataset class in test mode and scored by TokaMark's evaluator,
  so that window selection, NaN rules and normalization are theirs by
  construction. Our own faster loader is for training and validation only.
- **The harness is checked before any IGNITE number is believed:** a trivial
  predictor (the benchmark's mean and persistence baselines) scored through
  our harness must reproduce the published baseline column.
- **What gets reported per task:**

| Column | What it is |
|---|---|
| TokaMind (paper, our replication), GenToka (paper) | the references |
| Simulator, adapted, mean roll-out | pretrained, adapted per task, strict protocol. **The headline (goal G4).** |
| Simulator, one set of weights, mean roll-out | the pretrained simulator with no per-task adaptation (goal G4b) |
| Simulator, one sample and mean of several | sampled roll-out of the adapted model. One sample is what GenToka reports. |
| From scratch | same network, tokenizers and recipe, no pretraining. Shows what pretraining bought. |
| Regression read-out | the diagnostic described above |
| Direct supervised model | the headroom model of stage 0, reading raw inputs. It is not a simulator. It shows what a specialized model reaches. |
| Full context | extra signals allowed. Labelled as outside the protocol. |

- **"From scratch" means** the same network, the same frozen MAST tokenizers
  and the same recipe, with the dynamics weights initialized at random. It
  isolates what dynamics pretraining bought. What the whole stack buys or
  costs against reading raw signals is a different question, answered by the
  direct supervised model of stage 0.
- **The first complete table** needs the references, the headline column and
  the one-set-of-weights column for all 14 tasks at one model size, and the
  sample columns and the from-scratch column for six probe tasks (1-1, 1-2,
  2-1, 3-1, 4-1, 4-4). Model size is chosen on validation scores of the probe
  tasks, not by building three tables. If time runs short the columns are
  dropped in this order: full context, the regression read-out, then
  from-scratch beyond the probe tasks. The headline, the one-set-of-weights
  column and the single-sample scores are goals and are not dropped.
- **Cost.** There are roughly 300 to 400 test windows per shot and 1,110
  test shots. The mean roll-out is one pass per frame, at most 20 passes per
  window, and every test window is scored with it. A sampled roll-out costs
  several unmasking passes per frame for every sample. The sample columns
  are therefore scored on a fixed random subsample of the windows of every
  test shot, the same for every model, with its size stated.
- **Wins are counted with their noise.** The first table is one run per task,
  like the references. The evaluator writes per-shot scores, so each
  comparison with our TokaMind replication also gets a paired bootstrap over
  test shots. Run-to-run noise is estimated with three seeds on the probe
  tasks, for our model and for the TokaMind replication. A gap inside that
  noise is reported as a tie (task 1-2 is 0.0433 against 0.043 already).
  For GenToka only the published numbers exist here. A win over GenToka is
  counted when the upper end of our bootstrap interval is below its
  published score, and its own run-to-run noise stays unknown.
- **Task 4-5** is scored from the fluctuation path: the decoder gives the
  expected linear power in each frequency bin for each frame, at the
  benchmark's resolution. An average taken on a compressed scale is biased
  low in linear power, and the builder has to correct for that. The number
  to beat is the mean baseline, 0.653. The cheap predictors of stage 0 (the
  recent mean spectrum shrunk toward the overall mean, and a linear model on
  band powers) say how much is there before the tokenizer is built.

### Generative quality

"Generative but physically accurate" is checked beyond the benchmark:

- IGNITE's existing roll-out metrics (`eval_dynamics.py`), which need no
  DIII-D detectors, on 20-frame seeds rolled out 80 frames.
- Calibration of sampled ensembles for the scalar signals: does the truth
  fall inside the sampled spread as often as it should.
- The spectral statistics of the fluctuation tokens, as in the gate.

### Control readiness

"Control ready" is read here as five things that can be checked (a Proposed
reading of the requirement of 2026-11-22):

1. **It steps.** A small interface resets the simulator from any number of
   seed frames and then advances it one frame per call. Actuator values for
   the next 5 ms go in. The next frame of every modelled signal comes out in
   physical units, as a mean or as a sample. Which channels are actuators is
   set by roles, so it is configuration and not code.
2. **It is causal, and that is tested.** Changing an actuator after a frame
   changes no output at or before that frame. No input is scaled with a
   statistic of its own shot.
3. **It holds over a whole shot.** From a 50 ms seed the model is rolled to
   the end of each test shot under the recorded actuators. The error is
   reported against horizon for the quantities a controller tracks (plasma
   current, boundary, axis position, the betas, line density), beside
   persistence. The benchmark stops at 100 ms. The median shot is 386 ms.
4. **It responds to actuators as the machine did.** There are two checks on
   held-out shots. First, the score with the recorded actuators is compared
   with the score when the actuators are held at their last seed value, or
   taken from another shot. The difference is how much the model uses them.
   Second, around recorded actuator changes (a neutral beam switching, a gas
   step, a change in the requested current) the predicted response of the
   betas, the density and the current is compared with the measured one in
   sign, size and delay.
5. **Its speed is known.** Wall time per simulated frame on one GPU, for one
   roll-out and for a batch, with both read-outs. Whether a mean roll-out
   step runs faster than the 5 ms it simulates is measured, not promised.

Because roles are set per batch, the reverse question (which actuators lead
to a wanted future) is the same model with the roles exchanged. It is not
scored in the first version.

What this does not establish: all five checks use recorded shots, so they
test the response to actuator trajectories that operators actually ran. They
say nothing about requests far from those, and nothing about closing a loop
on the machine. No controller is trained here.

### How a change is judged

The headline is now the simulator's own output, so the benchmark and the
simulator share one path, and most changes move both the same way. Two kinds
of conflict remain. A change can lower the error of the mean roll-out and
damage the samples (dynamics that are too smooth, a spread that collapses).
And a change can buy benchmark score by making the model less of a
simulator.

**Rule (Decided, 2026-11-21: the simulator wins).** A change is not kept for
the headline if it breaks causality, removes the ability to sample or to
step, or adds an output network that belongs to one task. Among changes that
keep the model a simulator, validation NRMSE of the mean roll-out on the six
probe tasks decides, and calibration and the roll-out metrics have to stay
within run-to-run noise. Calibration means that the truth falls inside the
central 90% of samples about 90% of the time.

**Bar (Decided, 2026-11-22).** Goal G4. It replaces the bar of 2026-11-21,
which asked the generated output to reach TokaMind's level on at least 8 of
14 tasks.

**Adapted or not.** The first version asked whether the bar is judged on the
simulator after adaptation to each task, or with no adaptation. The answer of
2026-11-22 ("ok yes") is read as: after adaptation, which is how GenToka's
numbers are produced. The unadapted single model is scored beside it. If it
meets the bar by itself, that is the stronger result and the one to lead
with.

**If the bar is out of reach.** The headroom table of stage 0 shows, per
task, what a specialized model reaches from the raw inputs. Both published
tables say pretraining adds little on top of that. If the table shows that
the numbers of G4 cannot be reached, it goes back to Nathaniel with the
measurements. The bar is not lowered silently.

**If the token path is not precise enough** for some signal, the fallbacks in
order are more coefficients or a further quantizer level for that signal,
then a continuous output for its tokens (IGNITE already has a Gaussian
`latent_head`; a small per-token diffusion head is the stronger version of
the same idea).

### Where the code goes

**Proposed:** new modules beside the existing IGNITE ones, importing the
unchanged building blocks (the quantizer, transformer blocks, the MaskGIT
sampler, the attention dispatch) and not editing them in ways that change the
DIII-D path. Tests go under `tests/ignite/`. Docs are part of each change.
Module names are left to the builder.

## Order of work

The work is four sub-projects under this one design. The dates are targets.

| Stage | By | What exists at the end |
|---|---|---|
| 0. Harness, data and headroom | 2026-11-24 | the standardized cache and registry; the scoring harness, checked against the published baselines; the headroom table for the six probe tasks and task 4-5 |
| 1. Tokenizers | 2026-11-27 | the fixed-transform slow tokenizer with K per signal; the learned profile and map tokenizers against their linear baselines; the fluctuation path; the learned slow tokenizer and the two controls as comparisons; the headroom table for the other seven tasks |
| 2. Dynamics and first table | 2026-12-02 | pretraining at three sizes; adaptation per task; the step interface and the causality test; the first complete table |
| 3. Improve | 2026-12-09 | the measured architecture changes; the other control-readiness checks; the frame-duration test (train at 5 and 20 ms, test at 10 ms); the 4,720 extra FAIR-MAST level 1 shots mapped and added, if they have arrived and the pretraining check says more data helps; temporal split if there is time |
| 4. Close | 2026-12-12 | results written into the docs; weights, token caches and tables copied to a private Hugging Face repository under Nathaniel's account (Decided, 2026-11-21; the upload is run when he says so) |

**The headroom table** (new on 2026-11-22) has one row per task and says how
far the published models are from what the data allows. Its columns are: the
error that TokaMind's own output representation leaves when it encodes and
decodes the true targets, scored through the evaluator; cheap predictors
(persistence shrunk toward the mean, a linear model); and a direct supervised
model that reads the raw inputs at full precision. It needs no tokenizer and
no pretraining, so it is the first result of the project, and it decides
where the effort goes.

The first complete table does not wait for the fluctuation path. Thirteen of
the 14 tasks can be scored with the trend path alone. Only task 4-5 needs
spectra. That task is 26% of the mean, though, so the fluctuation path is no
longer the last tokenizer built. It follows the slow family directly.

Four checks decide what happens next, and each has a fallback:

- **Headroom check, end of stage 0.** If the direct supervised model cannot
  reach the numbers of goal G4 on the probe tasks, the bar goes back to
  Nathaniel with the table before stage 2 is committed to it.

- **Tokenizer check.** If a shared tokenizer fails the gate for a signal and
  the specialized control passes, that signal uses the specialized one.
- **Token-path check, early in stage 2.** On tasks 1-1 and 1-2, the mean
  read-out of a small simulator must come within 10% of the direct
  supervised model that reads the raw inputs. If it does not, the token path
  is fixed (more coefficients or levels, or a continuous output) before any
  large pretraining run.
- **Pretraining check.** If the pretrained and the from-scratch model agree
  on validation for the probe tasks, GPU time goes to data (FAIR-MAST) and to
  the token path, not to the largest model.

## Risks

- **The bar may be out of reach on the mean.** A mean 10% below GenToka
  needs, for example, a third off the errors of groups 1 and 2 with every
  other task at the best published score, or a smaller gain there together
  with a clear win over the mean baseline on task 4-5. Several tasks look
  limited by noise that no model removes. Nobody knows yet where the floor
  is. The headroom table is the first result for that reason, and a miss is
  reported, not hidden.
- **Data, not parameters, is the limit.** Covered above. The from-scratch
  column makes the answer visible either way.
- **Precision of the token path is the main technical risk.** The headline
  is decoded from tokens, and IGNITE's slow tokenizer on DIII-D was an order
  of magnitude too coarse for these scores. Expected values over ordered
  bins, the fixed transform, the linear baselines, the gate target and the
  continuous-output fallback all exist for this. The token-path check stops
  a large run from being built on a path that is not precise enough.
- **A mean roll-out is not a sampled future.** It can drift from the mean of
  samples over long horizons, and training the model to take its own
  expected values back in could blunt its samples. Both are measured: the
  two read-outs on a common subsample, and calibration with and without
  that training.
- **The comparison with GenToka rests on its published numbers.** Its code
  and per-shot scores were not looked for, so the run-to-run noise on its
  side is unknown, and it scores one sample where our headline is a mean.
  The single-sample column is the like-for-like comparison.
- **Three weeks is short for this, and shorter with the raised bar.** The
  order of work still puts a complete, comparable table first and the
  refinements after it. Control readiness beyond the step interface and the
  causality test comes after the first table.
- **Cross-channel physics moves into the dynamics model.** The Mirnov-only
  control measures what that costs.
- **FAIR-MAST may arrive late or need more mapping work than expected.**
  Nothing before stage 3 depends on it.

## Later projects

Each gets its own design. This one only leaves the hooks.

- **One model for DIII-D and MAST.** The registry, the identity embedding
  built from metadata, masking of absent tokens and roles per batch are the
  parts that make a second device a matter of data. Probe positions for the
  MAST magnetics are not in the TokaMark stores and would have to come from
  FAIR-MAST or the machine description.
- **Time scale.** The formulation is in [Time base](#time-base). What is left
  for later is a second machine. What a first measurement says is below.
- **Tokamak-SI for MAST** (asked for on 2026-11-22; its own design). Event
  labels on the same shots would give this simulator a second kind of test:
  whether rolled-out shots show events at the rate and at the times the
  labels do. They could also enter the model as one more token family.
- The diagnostics that only FAIR-MAST level 1 has (cameras among them) and
  the FusionEquilibrium challenge.

### Time scale: a first measurement

A throwaway measurement on 500 MAST shots (flat-top only, 2026-11-21,
`/models/intermediate/spikes/timescale/RESULTS.md`) asked how a nested,
confinement-time-scaled representation should be formulated.

- **Within MAST the confinement time does not organize the time scales.**
  The spread of autocorrelation times across shots is larger in units of the
  confinement time than in milliseconds for every signal family but one, and
  their rank correlation with it is at most 0.21. The confinement time spans
  only a factor of about four on one machine, so this does not test the idea
  across machines. That test needs DIII-D.
- **Signal families differ far more than shots do.** Autocorrelation times
  run from about 2.5 ms (Mirnov amplitude) through about 19 ms (soft X-ray,
  D-alpha) to 33-41 ms (magnetics, coil currents, density, EFIT scalars).
- **At a 5 ms frame the slow families lose almost nothing inside a frame**
  (magnetics 1.8% of their variance). Soft X-ray, D-alpha and the Mirnov
  amplitude keep 25% to 56% of theirs inside it.
- Interpretation, not measurement: this points to nesting time resolution
  per family, inside the frame (one value per frame for slow signals, finer
  nested detail for fast ones), rather than one global frame length.
- One test of frame duration as a model input needs no second machine: train
  at 5 and 20 ms frames and test at 10 ms. It is in stage 3. The
  fixed-transform slow tokenizer makes it cheap, because no tokenizer has to
  be retrained for another frame duration.

## Left to the builder

Layer counts and widths, tokens per profile and per map, the filter and band
definitions of the high-rate paths, the compression used before
standardizing, the number of coefficients and of bins per signal, how the bin
targets are smoothed, the share of expected-value feedback in training,
masking and dropout probabilities, learning rates and schedules, batch sizes,
the cache file format, the direct supervised model of the headroom table, the
size of the window subsample for sampled scores, the form of the step
interface, module and test names, and how the three model sizes are laid out
across the four GPUs.

## Calls made in writing this

These were not discussed and are easy to overrule:

- The headline is the random split with pretraining on TokaMark training
  shots only, so the comparison with TokaMind and GenToka is like for like. A
  model pretrained on extra FAIR-MAST shots is reported as a separate row.
- Beam emission, OMAHA and the negative-current shots are left out of the
  first version.
- Actuators become tokens with a role, with the old summed vector as control.
- New modules beside the old ones, rather than making the DIII-D code
  configurable in place.
- A tokenizer that misses its fidelity target does not hold up pretraining.
  One that has collapsed, or loses to the window mean or to its linear
  baseline, does.
- "From scratch" keeps the frozen tokenizers.
- One run per task, with a paired bootstrap over test shots and three seeds
  on the probe tasks to size the noise.
- The design lives here, next to the other IGNITE specs, and is not committed
  until asked.

Added on 2026-11-22, in reading the messages of that day:

- "ok yes" is read as approval of the design with the raised bar.
- "By far the best" is read as a mean at least 10% below GenToka's, wins
  beyond noise over both models on at least 10 of 14 tasks, and no loss
  beyond noise.
- The bar is judged on the simulator adapted per task, with the unadapted
  single model beside it.
- The headline read-out is the mean roll-out, with one sample and the mean
  of several beside it.
- "Control ready" is read as the five checks of
  [Control readiness](#control-readiness), with no controller trained.
- The slow-series tokenizer is a fixed transform by default. This is the one
  proposal that touches something Nathaniel had confirmed.
- Precision comes from expected values over ordered bins. Tokens are nested
  in time resolution for series and in precision for learned tokenizers.
- The tokenizer target is tightened from a quarter to a fifth of the best
  published score.
- The machine's reference time scale is one constant per device, not the
  measured confinement time of each shot.
- Tokamak-SI for MAST is a separate track with its own design.
