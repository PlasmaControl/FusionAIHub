---
title: "IGNITE on MAST: stage 0 results"
---

# IGNITE on MAST: stage 0 results (2026-11-22)

**Status: stage 0 is built and measured, and its headroom check did not pass.**
The scoring harness reproduces the published baselines, the standardized cache
exists, and TokaMind's representation floor, two cheap predictors and one direct
supervised model per task are scored on the seven stage-0 tasks. Our replication
of TokaMind is complete. In the table printed at 23:54, after the three direct-model
runs that were training at 13:45 had finished, the direct model is behind the best published
score on tasks 1-1 and 2-1, ahead on tasks 1-2, 4-1 and 4-4 (on 4-4 by less than
the difference between the splits) and tied on tasks 3-1 and 4-5, and its
projected 14-task mean is 0.1929 against the bar of 0.183. In the table printed
at 13:45, with those three runs still training, task 4-1 was a tie and the
projected mean was 0.1954. That print is kept below, as printed, and the
changes of the 23:54 print are
[given beside it](#the-table-printed-at-2354).
By [the design](./ignite-mast-design.md) the bar therefore goes back to
Nathaniel with this table before stage 2 is committed to it. It is not changed
on this page ([What it says](#what-it-says)). Every finished test of the
training budget lowered a score (three times the steps on tasks 1-1, 2-1 and
4-5, four times the batch on task 4-4, and a batch of 256 on tasks 4-1 and 4-5).
The rows of tasks 1-1 and 2-1 are therefore not converged models, and the
batch-64 runs of tasks 4-1, 4-4 and 4-5 were short of budget; tasks 1-2 and
3-1 have no finished test of that kind. The three runs that were still training
at 13:45 ended between 16:12 and 20:20 and are
[named below](#runs-that-were-still-training-when-the-table-was-printed). This
page records what was run, with the numbers read from the stored score files.
All times are UTC on 2026-11-22, measured on [node13](../clusters/nvidia.md).

The seven stage-0 tasks are the six probe tasks (1-1, 1-2, 2-1, 3-1, 4-1, 4-4)
and task 4-5. Every score is the benchmark's task NRMSE (lower is better),
written by TokaMark's own evaluator (`tokamark.evaluator.compute_metrics`) and
never recomputed here.

## What was run

| What | When | Where it is |
|---|---|---|
| Scoring harness: the test-split check against the published `mean` and `persistence` columns (passed at 03:21), then the same baselines on `val_report` | 02:45 to 03:40 | `scripts/mast/check_harness.py` |
| Standardized cache v1 | built 03:46 to 03:50 | `/models/intermediate/mast/cache/v1/` |
| TokaMind representation floor, on `test` and `val_report` | 03:38 to 05:16 | `scripts/mast/run_headroom.py --predictor floor` |
| Two cheap predictors (`shrunk`, `ridge`), on `val_report` | first fit 04:46 to 05:43; second fit, after the code review, 08:23 to 09:36 | `--predictor shrunk`, `--predictor ridge` |
| Our TokaMind replication, all 14 tasks | finished 04:36 | `/models/tokamind/run_all_tasks.sh` |
| Direct supervised model: seven base runs | 05:08 to 09:50 | `--predictor direct --train` |
| Direct supervised model: thirteen variant runs | from 05:53; three were still training when the table was printed at 13:45 and finished at 16:12, 19:23 and 20:20 | `--predictor direct --variant <name> --train` |
| Code review of the whole branch and its one fix wave | 05:57 to 08:53 | commits `459ae466` to `752b15a7` |
| Headroom table | first print 09:55; printed again at 13:45, after five more variants had finished, and at 23:54, after the three that were training at 13:45 had finished | `scripts/mast/run_headroom.py --table` |

Scores are stored as
`/models/intermediate/mast/stage0/scores/<task>/<predictor>/<split>/score.json`,
beside the evaluator's own per-shot and per-signal files. Each record carries
the hash of the scoring code (`harness.py`, `tasks.py`, `baselines.py`) and, for
a headroom predictor, the hash of the predictor and of the files it was
computed from (a fit, a checkpoint, TokaMind's coefficient indices). The `git_revision` in a record
is the HEAD of the checkout when the scoring code was loaded. It can predate
uncommitted code, so the hashes are the provenance and the revision is a hint.
All 96 scores stored when the table was printed (2026-11-22 13:45) carry the
same scoring-code hash, `173e2747...`. Measured 2026-11-23 by the review of the
whole branch: all 135 records stored then, the three late direct scores
included, carry `173e2747` as well.

Splits. Everything fitted (`shrunk`, `ridge`, the direct model) is scored on
`val_report`, the validation shots at odd positions of the sorted list; models
and ridge's penalty are selected on `val_select`, the other half. The rule for
the test shots is that nothing fitted reads them. In stage 0 they were scored
only by predictors with nothing fitted: the mean and persistence for the
harness check, and the floor. The published scores are on the test split.

## The harness check

**Measured 2026-11-22, passed at 03:21.** The harness scored the two baselines
of the TokaMark paper on the test split (six streams of 48 workers, 38 minutes,
code revision `9ad04499`). All 20 rows that the paper's table publishes agree
with the published value to within 0.0004, against a tolerance of 0.002. The
harness runs the windows through float32 once, and its scores differ from
TokaMark's own loop by at most 2e-8.

Mean baseline (every output sample is the signal's mean), all 14 tasks:

| Task | Ours | Published | Task | Ours | Published |
|---|---|---|---|---|---|
| 1-1 | 0.7566 | 0.757 | 3-3 | 0.8412 | 0.841 |
| 1-2 | 0.9984 | 0.998 | 4-1 | 0.5114 | 0.511 |
| 1-3 | 0.9267 | 0.927 | 4-2 | 0.5114 | 0.511 |
| 2-1 | 0.8174 | 0.817 | 4-3 | 0.7313 | 0.731 |
| 2-2 | 0.9914 | 0.991 | 4-4 | 1.0270 | 1.027 |
| 2-3 | 0.9230 | 0.923 | 4-5 | 0.6533 | 0.653 |
| 3-1 | 0.9837 | 0.984 | | | |
| 3-2 | 0.6297 | 0.630 | | | |

Persistence baseline (the last finite input sample repeated), on the six tasks
for which the paper publishes it:

| Task | Ours | Published |
|---|---|---|
| 3-1 | 0.5364 | 0.536 |
| 3-2 | 0.3651 | 0.365 |
| 4-1 | 0.3247 | 0.325 |
| 4-2 | 0.3247 | 0.325 |
| 4-4 | 0.7178 | 0.718 |
| 4-5 | 0.8701 | 0.870 |

**Two persistence rows, and why.** TokaMark's `scripts/persistence.py` scores
persistence with an edited task: the inputs are reduced to the outputs that are
also inputs, the actuators are removed and the task is made markovian. In
sparse test mode the loader drops a window when all its inputs and all its
actuators are NaN. Without actuators, the published persistence column is
therefore scored only on windows that have a finite input: for task 3-1,
285,885 of the 332,362 test windows (86%), where it scores 0.536 against 0.687
on all windows. The predictor `persistence` reproduces the published
definition, on the benchmark's windows, and is what the check judged.
`persistence_all` scores the same rule on all of the task's windows. It was not
judged, and it is the fair baseline beside predictors that are scored on every
window. On the test split it gives 3-1 0.6868, 3-2 0.3990, 4-1 0.3095, 4-2
0.3095, 4-4 0.7178 and 4-5 0.7743. Both rows are in the headroom table. On
task 4-4 the two agree (0.7178 on test and 0.7157 on `val_report`); why was not
checked.

The same baselines on `val_report` (nothing is judged there) are in the table
below. One thing found on the way: TokaMark's `get_task_metadata` gives the
same dictionary to every role of a signal, so the `length_in_time_stamps` of
an input that is also an output holds the output's length (15 inputs in 7
tasks). The task specs (`tasks.py`) compute the lengths from seconds and `dt`
instead.

## The cache

**Measured 2026-11-22** (`manifest.json` of
`/models/intermediate/mast/cache/v1/`, built from a clean tree at `55b637be`,
TokaMark revision `146b0a0f`).

- 11,188 shot files: 8,963 train, 1,115 val, 1,110 test, none failed and none
  skipped. 137.8 GB, written in 226 s with 96 workers.
- The standardization statistics are fitted on 2,000 training shots (seed 0),
  all of which loaded (statistics hash `e90bbf2f149d...`).
- Of the 436,332 (shot, signal) pairs (39 signals), 408,373 are `ok`, 27,567
  are `absent` (the shot has no such dataset), 240 are `all_invalid` and 152 are
  `dt_mismatch`. No signal was lost.
- Checked against TokaMark's loader (`tests/ignite/mast/test_cache.py`): the
  worst error is 4e-7 of a signal's standard deviation, and no sample that the
  loader has is missing from the cache.

Three deviations from the design, each with its reason:

1. **float32 instead of float16.** With float16 the fidelity test failed at
   its limit of 1e-3 of TokaMark's standard deviation, on asinh-compressed,
   heavy-tailed signals (`cc_field` 3.3e-3, `omv_voltage` 1.6e-3, `beta_pol`
   1.5e-3). The cost is about 135 GB where float16 would have taken 68 GB.
2. **A channel whose finite samples are all equal in a shot stays valid and is
   flagged** (`const_channels`); only a channel with no finite sample is dead.
   Two thirds of what an all-equal rule catches are values that are constant
   by construction and are benchmark targets (the LCFS anchor points) or
   programmed references, which the benchmark scores.
3. **Saturated D-alpha samples are flagged** (`<signal>__saturated`) and stay
   valid, where the design said not valid, because TokaMark scores them.

Known gaps. 76 shots (63 train, 8 val, 5 test) deliver the two soft X-ray
cameras at 500 kHz instead of 50 kHz (`dt_mismatch`) and have no soft X-ray
datasets in the cache. `magnetics-b_field_tor_probe_cc_field` is absent in
3,940 shots. Thomson scattering is absent in about 2,180 shots.

**Nothing in stage 0 reads the cache.** The stage-0 predictors take their
windows from TokaMark's loader through the harness, so the headroom numbers
below do not depend on the cache or on its deviations. The cache is the input
of stage 1.

## The floor: what TokaMind's representation costs

**Definition.** TokaMind's own targets (standardized, filled with its
`interpolate` rule, transformed by the 3-D cosine transform, truncated to the
coefficients its fine-tune configuration keeps) are decoded back and scored as
if they were a prediction. The floor is the error TokaMind would have if it
predicted its kept coefficients exactly. It is a limit for models that predict
in that representation and says nothing about models that do not.

Two limits of the definition. First, the floor is a strict lower bound only on
windows whose targets have no missing sample. Where samples are missing,
TokaMind fills them before the transform and the evaluator scores only the
finite ones, so the kept coefficients of the filled target are not the
coefficients that fit the scored samples best, and a model of TokaMind's kind
could score below the floor on such windows. This is expected to matter most
on tasks 3-1 and 4-5 (a reading made when the code review of this stage raised
the point, not a measurement): neither how often targets are partly missing
nor how far below the floor a model could then score was measured. Second, the
floor is scored on the benchmark's windows. If TokaMind's own evaluation drops
windows (its `SelectValidWindowsTransform`), the replication score and the
floor are on different sets of windows, and the shares below compare across
them. Whether it drops any was not checked.

Coefficients kept per output signal, read from each replication run's
`dct3d.yaml`: task 1-1 all of them (so its floor is exactly 0); task 1-2
`lcfs_r` 10 of 170 and `lcfs_z` 11 of 170; task 2-1 13 of the 18 outputs in
full, `x_point_r` and `x_point_z` 8 of 10, `coil_current` 37 of 1,000,
`solenoid_current` 5 of 100, `ip` 8 of 100; task 3-1 `n_e` 456 of 1,200 and
`t_e` 927 of 1,200; task 4-1 both soft X-ray cameras 8,192 of 90,000; task 4-4
`ip` 29 of 400; task 4-5 both spectrograms 8,192 of 75,264.

**Measured 2026-11-22**, the floor against our TokaMind replication (test
split, same shots as the published scores). The replication is TokaMind's
fine-tuned model re-run with today's benchmark code; it is within 0.0022 of the
paper's fine-tuned column on every task.

| Task | Floor, test | Floor, `val_report` | TokaMind replication, test | Floor as share of the replication |
|---|---|---|---|---|
| 1-1 | 0.0000 | 0.0000 | 0.1503 | 0% |
| 1-2 | 0.0236 | 0.0239 | 0.0433 | 55% |
| 2-1 | 0.0038 | 0.0037 | 0.1809 | 2% |
| 3-1 | 0.0279 | 0.0284 | 0.3457 | 8% |
| 4-1 | 0.0369 | 0.0353 | 0.2147 | 17% |
| 4-4 | 0.0349 | 0.0351 | 0.3502 | 10% |
| 4-5 | 0.3366 | 0.3605 | 0.6499 | 52% |

Reading (see [What it says](#what-it-says) for the answer to the question this
table is for). On tasks 1-1, 2-1, 3-1, 4-1 and 4-4 the representation costs
TokaMind little, so most of its error is prediction error. On tasks 1-2 and 4-5
the representation is about half of its score. The floor differs between the two
splits by at most 0.002 on every task except 4-5, where it is 0.3366 on test and
0.3605 on `val_report`.

## The cheap predictors

Two predictors that look at no network, no codec and no more than the last 5 ms
of each input (`headroom/cheap.py`). Both are fitted on the same 3,000
training shots (`default_rng(0).choice` over the training split) in
TokaMark-standardized units and scored on `val_report`. The command scores
every fitted predictor blind: it hands the predictor a copy of each batch whose
targets are NaN, so a predictor cannot read what it is scored on
(`run_headroom.py`, `_blind_to_targets`). The command does this since
`a34cd797` (08:21). The second fit of the cheap predictors started after it
(08:23), so every stored `shrunk` and `ridge` score was made blind; the
first-fit ridge scores in the table below, which are no longer stored, were
not. A
direct run is scored by the process that trained it, with the code loaded at
its start. Of the 16 direct scores of the table of 13:45, 13 come from runs
that started before that commit (05:08 to 07:42: the seven base runs, `h050`
on tasks 4-1 and 4-4, `s90k` on 1-1, 1-2 and 2-1, `lr1e4` on 1-2) and were
made without the blinding. Six of them are rows of the headroom table (tasks
1-1, 1-2, 2-1, 3-1, 4-1 and 4-5). In the table of 23:54 four of the seven rows
(tasks 1-1, 1-2, 2-1 and 3-1) are from such runs; the rows of tasks 4-1 and 4-5
are from runs that started at about 13:02 (their finish time minus their wall
time), after that commit, and the row of task 4-4 is a run of 08:23. What is
known about these 13:

- *By the code:* the direct model's prediction function builds its tensors
  from the inputs and actuators of a batch and never reads the targets
  (`direct._predict_fn`, `_inputs_to_tensors`), and `direct.py` is
  byte-identical at the commits involved (sha256 `b13327d2…`, the hash each
  score stores).
- *Measured on one of them, 2026-11-22 14:08 to 14:12:* the row of task 1-2
  (`lr1e4`) was scored again with the blinding. With the 48 loader workers of
  the original run the score is 0.036184312776439465, equal to the stored one
  in every digit. With 64 workers it is 0.036184062141816686 in two runs (the
  second value was read from the score file before the next run replaced it;
  `analysis/rescore_1-2_lr1e4.txt` under the stage root records it from the
  session's output). So the
  blinding changes nothing for this predictor, and the number of loader
  workers changes a score by 2.5e-7 here, which no table on this page shows.
  The stored score of that row is now the blind one (same value, date 14:11).
- The other 12 were not scored again.

The two cheap predictors are:

- **`shrunk`**: persistence shrunk toward the per-channel training mean.
  The prediction of output frame k is `m + a_k (p - m)`, with `m` the training
  mean of the channel, `p` the last finite input sample and `a_k` the
  least-squares slope pooled over channels and windows. `a = 0` is the mean
  predictor and `a = 1` is persistence. Where there is no anchor, or the output
  is not an input, it predicts `m`.
- **`ridge`**: ridge regression from frame means. The features are the
  standardized mean of each input over its last 5 ms, a validity flag per
  channel, the frame means of each actuator over its whole window (the future
  part included, because the future actuator program is what the task is
  conditioned on) and a constant. The targets are the frame means of each
  output channel. The penalty is the one of the ten decades 1e-8 to 10 with
  the smallest squared error on 200 shots spread evenly over the sorted
  `val_select` list. The fit stores the chosen penalty, and an edge flag when
  the choice is the smallest or the largest candidate (the optimum is then not
  bracketed).

Limits, stated because they bound what these rows can say. Both work on 5 ms
frames, so neither can predict structure inside a frame. The ridge fit counts a
target frame that is missing as 0 after centring, which pulls the fit toward the
mean wherever targets are missing. For the two spectrogram signals the output
window is one frame, because the content of a spectrogram is the band
structure per bin and not its time course within 20 ms. Tasks 1-1 and 1-2 have no
output that is also an input, so `shrunk` there is only the per-channel
training mean (checked with `tasks.load_task` on the seven tasks, 2026-11-22:
task 2-1 has three of its 18 outputs among its inputs, and in tasks 3-1, 4-1,
4-4 and 4-5 every output is an input). Both predictors are fit on 3,000 of the 8,963 training shots with one
seed and one regularization search, and neither is tuned beyond that.

The numbers below are from the second fit of both predictors (08:23 to
09:36, at code `a34cd797`). The first fit (04:46 to 05:43) chose
ridge's penalty on the first 200 `val_select` shots by shot number, which are
one old campaign (see the direct model below), and from the candidates 1e-4 to
10 only; the code review of this stage found both, and on five of the seven
tasks the choice had been the smallest candidate. The second fit changed
`shrunk` on no task to four decimals, and ridge by at most 0.0061:

| Task | Ridge, first fit (its penalty) | Ridge, second fit | Penalty chosen | Edge flag |
|---|---|---|---|---|
| 1-1 | 0.3053 (1e-4) | 0.2992 | 1e-8 | lower |
| 1-2 | 0.0830 (1e-4) | 0.0809 | 1e-8 | lower |
| 2-1 | 0.3318 (1e-4) | 0.3268 | 1e-8 | lower |
| 3-1 | 0.4877 (1e-4) | 0.4876 | 1e-5 | none |
| 4-1 | 0.2359 (1e-3) | 0.2359 | 1e-3 | none |
| 4-4 | 0.4574 (1e-4) | 0.4574 | 1e-4 | none |
| 4-5 | 0.6876 (1e-2) | 0.6882 | 1e-3 | none |

On tasks 1-1, 1-2 and 2-1 the choice is still the smallest candidate, 1e-8.
The ridge score there is "ridge at the smallest penalty offered", not ridge at
its best penalty; between the first fit's smallest candidate (1e-4) and 1e-8
the score moved by 0.0021 to 0.0061.

**Measured 2026-11-22**, in the headroom table below. Two findings, each with an inferred reading marked as such.

- *A per-channel training mean scores about 0.22 on task 1-2, against 1.0005
  for TokaMark's scalar mean.* A per-channel training mean alone, which is what `shrunk`
  falls back to without an anchor, scores about 0.22 on `val_report` on task
  1-2 (`shrunk` 0.2204), against 1.0005 for the scalar mean, so about 0.22 of
  the 1.0005 remains when the mean is taken per channel. Inferred, not
  measured: most of the scalar mean's error is then the fixed shape of the
  boundary across channels, not its motion; the 0.22 that remains is not
  attributed.
  Ridge on frame means scores 0.0809 there, about twice the best published
  score (0.0412, test split, a different split).
- *How close the published models are to the cheap references.* Measured: on
  task 4-1 our replication (0.2147, test) is 0.07 below `shrunk` (0.2842,
  `val_report`; different splits). On task 4-5 the mean scores 0.6852 on
  `val_report` and 0.6533 on test; our replication (0.6499) and the best
  published score (0.6510) are 0.0034 and 0.0023 below the mean on test, and
  persistence on all windows scores 0.8011 on `val_report`. For the published
  models **no paired test was run**: the papers' per-shot scores are not
  available, and our replication's were not put beside the mean's shot by
  shot. One score on task 4-5 is known only to about +-0.06 over shots
  (bootstrap of `shrunk` on 351 `val_report` shots, measured 2026-11-22, 95%
  interval 0.634 to 0.756). Their differences of 0.002 to 0.003 from the mean
  are therefore neither a win nor a tie: the page does not say that the
  published models beat the mean on task 4-5, nor that they do not.
  For our own predictors the paired test exists, on the 351 `val_report` shots
  of task 4-5 (difference of the per-shot scores, bootstrap over shots, 2,000
  resamples, seed 0; `analysis/paired_bootstrap.py` under the stage root,
  measured 2026-11-22): `shrunk` is above the mean by 0.0073 (95% interval
  0.0061 to 0.0087) and ridge by 0.0031 (0.0014 to 0.0047). Both cheap
  predictors are worse than the benchmark's mean on that task, by a small
  amount that is outside the noise over shots.
  *Inferred, not measured:* that these small margins mean the task is near a
  noise floor, or is not predicted at all. Nothing on this page measures a
  noise floor, and the direct model below does score under the mean there.

## The direct model

One transformer per task, trained from scratch on the task's training windows
(`headroom/direct.py`). It is not the simulator and never becomes a headline
number: it shows what a specialized model reaches when nothing is quantized or
truncated (no codec, no coefficient truncation, no frame means). Its tokens
are the simulator's tokens, one channel over one 5 ms frame, so a gap between
this model and the later simulator points at the token path and not at how the
data are cut.

Configuration of the base variant (`config.json` of each run, read 2026-11-22):
`d_model` 384, 8 layers, 8 heads, 5 ms patches, 30,000 steps, batch 256 for the
markovian tasks (1-1, 1-2, 2-1, 3-1) and batch 64 for the non-markovian tasks
of group 4 (4-1, 4-4, 4-5), learning rate 3e-4 with 1,000 warm-up steps and
cosine decay, weight decay 0.05, an exponential moving average of the weights
(0.999), seed 0. The models have 14.4 to 18.2 million parameters. A window is
132 tokens on task 1-1, 455 on 1-2, 290 on 2-1, 2,684 on 3-1 and 5,050 to 5,750
on group 4, where the task gives 150 ms of history (30 patches per channel)
against 5 ms on the markovian tasks (training logs and `tasks.load_task`,
measured 2026-11-22). The training loss is in TokaMark-standardized units (mean
over signals of the signal's mean squared error) and is not on the NRMSE scale.
Every 2,500 steps the averaged weights are scored on 200 shots evenly spaced
over `val_select`; the lowest becomes `best.pt`. The score in the table is
`best.pt` on `val_report`.

The first version of the trainer validated on the first 200 shots of
`val_select` by shot number. Those are one old campaign (shots 11806 to
18744). **Measured 2026-11-22** on task 4-5, only 65 of them have windows and
`cc_field` is nearly constant there: the per-shot mean baseline scores 0.30 on
`val_report` shots of that range, 0.77 on later shots and 0.685 on all. Choosing
the best checkpoint on such shots would not have tracked the report split, so
the validation shots are now spread evenly over the sorted list and the
trainings were restarted after that change (05:08).

### Variants and when each was chosen

Seven base runs and thirteen variant runs were started. A variant changes one or two
options of the base configuration. A variant chosen after a `val_report` score
was visible is weaker evidence than one chosen before, so the order of events
is given for each. In every case the table takes the variant by its
`val_select` error only.

| Variant | What differs | Tasks | The question it asks | Chosen |
|---|---|---|---|---|
| `s90k` | 90,000 steps | 1-1, 1-2, 2-1 | Is the base run under-trained? | Before any direct score existed (started 05:53) |
| `h050` | 50 ms of history instead of 150 ms | 4-1, 4-4 | How much of the score needs the 150 ms history? | Before any direct score existed (started 05:53) |
| `h050b256` | 50 ms of history and batch 256 | 4-1 (stopped, then started again), 4-4 | Is a difference between `h050` and base the history or the number of windows seen? | Defined before any direct score existed; started 08:23 (4-4) and 08:27 (4-1), when the `val_report` scores of task 1-2 (base and `s90k`) and of `h050` on task 4-4 were visible. The run of task 4-1 was stopped at 09:22 and started again at 13:02, after the `val_report` score of this variant on task 4-4 (0.2859) was visible |
| `lr1e4` | learning rate 1e-4 | 1-2 | Is the instability of the base run (below) the learning rate? | After the base run's `val_report` score on 1-2 (0.0493) was visible; chosen from the training loss and the `val_select` curve (started 07:42) |
| `lr1e4` | learning rate 1e-4 | 1-1, 2-1 | Is the distance to the published scores the learning rate? | After the `val_select` curve of 1-2's `lr1e4` was visible and before any `val_report` score of the direct model on 1-1 or 2-1 existed (started 08:31) |
| `s90k` | 90,000 steps | 4-5 | Is the base run under-trained? | The condition (the base run beats the mean on `val_report`) was fixed before any score existed; the kind of variant was chosen after the base score (0.6475) was visible (started 09:26) |
| `b256` | batch 256, history unchanged | 4-5 | Does the gain that batch 256 gave on task 4-4 carry to task 4-5? | After the `val_report` scores of `h050b256` on task 4-4 and of the base run on task 4-5 were visible (started 13:02) |

Task 3-1 has no variant. Its `val_select` error was lowest at step 17,500
(0.3503) and flat to slightly worse for the remaining 12,500 steps (0.3546 at
the end). *Inferred, not tested:* more steps alone are unlikely to help a
curve that turned at step 17,500, although task 2-1 below shows that a curve
that looks flat under this schedule can still gain from three times the
steps. A possible reason to expect more on that task is the loss: its target
tails are measured (below), their effect on the score was not tested, and a
change of the loss is a change of the model's code and not a variant.

Two runs were stopped by hand. The `s90k` run of task 1-2 was stopped at about
step 30,700 of 90,000 (the last line of its log). Its training loss had been
above 0.7 since step 16,600 and at 0.99 to 1.01, the loss of predicting
the mean, since about step 26,000 (4,700 steps); its
`best.pt` (step 5,000) is scored and marked ‡. The `h050b256` run of task 4-1
was stopped at step 1,300, with no checkpoint, when its speed (0.45
steps per second, while ten trainings and the second fit of the cheap
predictors shared the machine; from the session's notes of 09:22, because the
run's log was overwritten when the variant was started again and its
`train_log.csv` has no row) put its end 18 hours away; its directory was
moved to
`/models/intermediate/mast/stage0/direct_stopped/` and it is in no table.
Neither stop was planned, and what the stopped run of task 1-2 would have
reached is not known. The variant of task 4-1 was started again from step 0
at 13:02, on a GPU of its own, once the same variant had become the best run
of task 4-4 (below).

### What the validation curves show

`val_nrmse` on the 200 `val_select` shots, from each run's `train_log.csv`
(**measured 2026-11-22**; one seed per run).

- **Task 1-2 is unstable at learning rate 3e-4, and stable at 1e-4.** The base
  run reached 0.0522 at step 2,500 and was between 0.11 and 0.55 for the rest
  of the run, so its `best.pt` is its first validation. `s90k`, whose cosine
  keeps the rate near its peak for longer, reached 0.0462 at step 5,000; its
  training loss then spiked from step 10,800, was between 0.72 and 0.82 from
  step 17,500 to step 22,000, and from about step 26,000 sat at 0.99 to 1.01,
  the loss of predicting the mean (validation 0.78 at step 17,500, then 0.89,
  1.53, 1.02, 0.99 and 1.00 at the five validations after it). No
  step was skipped for a non-finite loss in either run. With the rate at 1e-4 and nothing else changed
  the curve falls at every validation but the last: 0.0593, 0.0466, 0.0442,
  0.0429, 0.0421 over the first 12,500 steps, 0.0390 at step 27,500 and 0.0391
  at step 30,000. The targets
  of this task have no outliers (largest standardized value 3.0), so outlier
  targets are ruled out as the cause. Not checked: the bfloat16 autocast.
- **Tasks 1-1 and 4-5 were still improving when the base run ended.** Task 1-1:
  0.1550 at step 17,500 and 0.1430 at step 30,000, the last and lowest value.
  Task 4-5: 0.6980 at step 20,000, then 0.6933, 0.6934 and 0.6903 at the last
  three validations. Their `s90k` variants ask how far that goes. On task
  1-1 the answer is: further than 30,000 steps, and not far. With 90,000
  steps the error is 0.1315 at the last step, still the lowest value of the
  run (0.1326, 0.1322, 0.1319, 0.1316, 0.1315 over the last 10,000 steps), and
  the `val_report` score is 0.1235 against 0.1361 for the base run. On the
  same 520 shots the paired difference is -0.0127 (95% interval -0.0153 to
  -0.0102), and `s90k` is the lower one in 418 shots.
- **Task 2-1** was lowest at step 27,500 (0.1525) and the same to four
  decimals at step 30,000, after 0.1575, 0.1562 and 0.1584 at steps 20,000 to
  25,000. It was close to flat at the end, and it was still short of steps:
  with 90,000 steps the error is 0.1377 at the last step, the lowest value of
  the run (0.1405, 0.1394, 0.1407, 0.1399, 0.1387, 0.1377 over the last 12,500
  steps), and the `val_report` score is 0.1347 against 0.1523. On the same 520
  shots the paired difference is -0.0176 (95% interval -0.0217 to -0.0140),
  and `s90k` is the lower one in 445 shots. A curve that looks flat under a
  cosine schedule near its end does not show that the model is trained out.
- **Task 3-1** was lowest at step 17,500, as above.
- **The lower learning rate helps on task 1-2 only.** With 1e-4 and nothing
  else changed, task 1-1 reaches 0.1505 on `val_select` (base 0.1430) and
  0.1418 on `val_report` (base 0.1361); task 2-1 reaches 0.1589, at its last
  step and still falling (base 0.1525), and 0.1579 on `val_report` (base
  0.1523). On the `val_report` shots the paired difference `lr1e4` minus base
  is +0.0056 on both tasks (95% intervals +0.0041 to +0.0073 and +0.0023 to
  +0.0089 over 520 shots each). So on these two tasks the default rate is not
  what keeps the model behind the published scores, and at 30,000 steps the
  lower rate is the slower one.
- **Tasks 4-1 and 4-4: one third of the history gives the same error.** On
  `val_select`, base and `h050` are 0.1865 and 0.1835 on task 4-1 and 0.3097
  and 0.3140 on task 4-4. On the `val_report` shots the paired difference base
  minus `h050` is -0.0015 on task 4-1 (95% interval -0.0059 to +0.0035 over 531
  shots) and -0.0035 on task 4-4 (-0.0114 to +0.0046 over 553 shots; paired
  bootstrap over shots, `analysis/paired_bootstrap.py` under the stage root).
  Neither interval excludes zero. For the simulator this says that, on these
  two tasks at this model size, a 50 ms context costs no more than 0.006 (task
  4-1) and 0.011 (task 4-4), the ends of the two intervals, and nothing that
  this comparison can tell from zero. It
  says nothing about longer horizons, and it is one seed.
- **Task 4-4: batch 256 is the largest gain any variant made.** `h050b256`
  has the history of `h050` and four times its batch, at the same 30,000
  steps. Its `val_select` error is 0.2787 at step 25,000 (0.2854, 0.2826,
  0.2787, 0.2804, 0.2799 over the last five validations) against 0.3140 for
  `h050` and 0.3097 for the base run, and its `val_report` score is 0.2859
  against 0.3230 and 0.3195. On the same 553 shots the paired difference to
  `h050` is -0.0371 (95% interval -0.0473 to -0.0275) and to the base run
  -0.0336 (-0.0429 to -0.0248; lower in 404 shots). The two runs that differ
  only in the batch are `h050` and `h050b256`. A step of batch 256 sees four
  times the windows, so the run saw four times the data and had a less noisy
  gradient; which of the two gives the gain was not separated. What it says,
  for task 4-4 and one seed: at batch 64 and 30,000 steps the runs of this
  task were short of training, and their scores understate this model.
  *Inferred at the time, measured since 23:54:* the same holds for the base
  runs of tasks 4-1 and 4-5, which have the same batch. The same variant on task
  4-1 and a batch-256 run on task 4-5 were started when this was seen, to
  measure it; the next two bullets give what they found.
- **Task 4-1: batch 256 with 50 ms of history beats base and `h050` on most
  shots.** `h050b256` ended at 0.1572 on `val_select` (step 30,000, the lowest
  value of its run) against 0.1835 for `h050` and 0.1865 for the base run, and
  its `val_report` score is 0.1727 against 0.1952 and 0.1937. On the same 531
  shots the paired difference base minus `h050b256` is +0.0211 (95% interval
  +0.0145 to +0.0271; the base run is the lower one on 90 of 531 shots; median
  per-shot difference +0.0128), and `h050` minus `h050b256` is +0.0225 (+0.0157
  to +0.0286; `h050` is the lower one on 81 of 531; median +0.0128). The
  published 0.2122 is above the upper end of the interval of this score
  (0.1923), so the verdict of the table moved from tie to ahead.
- **Task 4-5: batch 256 is better than base and than 90,000 steps, by little.**
  `b256` ended at 0.6752 on `val_select` (step 20,000, the 8th of 12
  validations; 0.6785 at step 30,000), `s90k` at 0.6812 (step 75,000, the 30th
  of 36; 0.6830 at step 90,000), and the base run at 0.6903. On `val_report` the
  scores are 0.6361, 0.6406 and 0.6475. On the same 351 shots the paired
  difference base minus `b256` is +0.0113 (95% interval +0.0066 to +0.0161; the
  base run is the lower one on 105 of 351 shots; median +0.0025), and `s90k`
  minus `b256` is +0.0044 (+0.0003 to +0.0084; `s90k` is the lower one on 136
  of 351; median +0.0006). The intervals exclude zero, the median per-shot
  differences are 0.0025 and 0.0006, and the verdict stays a tie. Three times
  the steps, the question that the earlier bullet on tasks 1-1 and 4-5 left
  open, helped here by 0.0069 on `val_report` (0.6475 to 0.6406), much less than
  on tasks 1-1 and 2-1.

Batch 256 has now helped on all three tasks on which it was tried (4-4, 4-1 and
4-5). On tasks 4-4 and 4-1 it came together with 50 ms of history, on task 4-5
alone. The paired differences above are over shots, with 2,000 resamples and
seed 0 (`analysis/paired_bootstrap.py TASK A B val_report`); there is one seed
per run.

The selection shots and the report shots differ by more than most differences
between variants: the same checkpoint scores 0.6903 on its 200 `val_select`
shots and 0.6475 on `val_report` on task 4-5, 0.3503 and 0.3267 on task 3-1,
0.1430 and 0.1361 on task 1-1, 0.1865 and 0.1937 on task 4-1. A choice between
variants that are closer than that on `val_select` is a choice within noise;
on task 4-1 the table's variant at 13:45 (`h050`) was the one with the higher
`val_report` score, by 0.0015. In the table of 23:54 the variant of task 4-1 is
`h050b256`, which has the lowest value of both.

**Target tails.** The loss is a mean squared error in standardized units, and
some tasks have targets far from the bulk. Measured 2026-11-22 on 153,600
training windows per task, taken from the trainer's own batches
(`analysis/outliers.py` and its outputs under the stage root; the output for
task 1-2 is from a second run at 14:01, which gave the same two figures,
because the first was not kept): on task 1-2 no
target exceeds 3.0 and the top 1% of windows hold 1.9% of the target
mean-square; on task 1-1 the largest target is 43.2 (`beta_normal`; five other
signals exceed 15), 145 windows from 49 shots have a target above 10, and the
top 1% of windows hold 13.1%; on task 3-1 a Thomson `t_e` target exceeds 10 in
4,311 windows (2.8%), 608 windows have a target above 30, the largest target
is 64.3 (`n_e`; 60.5 for `t_e`), and the top 1% of windows hold 11.8%. *Inferred, not tested:* a squared loss weights those
windows by their square, while the benchmark's score is a mean over shots of a
per-shot RMSE, so on tasks 1-1 and 3-1 part of the model's capacity goes to
windows that the score weights less. A loss matched to the metric, or a robust
one, was not tried in stage 0.

### How the runs were made

- **Shared GPUs.** The base runs of tasks 3-1, 4-1, 4-4 and 4-5 started at
  05:08, one per GPU. From 05:53 on every GPU held two or more trainings
  (`--share-gpu`; GPUs 0, 1 and 3 from 05:50), so every throughput number
  after that time describes a
  shared GPU. Sharing changes the speed of a run, not what it computes. Median
  training throughput in windows per second (`train_log.csv`, 12 values per
  run): 872 on task 1-1,
  1,334 on 1-2, 668 on 2-1, 693 on 3-1, and 200 to 205 on the group-4 base runs
  (323 and 383 for `h050`). On tasks 1-1, 1-2 and 2-1 the speed is bounded by
  TokaMark's loader (the GPUs were mostly idle in the first probes, by the
  session's notes; no stored output); on group 4
  it is bounded by the GPU (a window is 5,050 to 5,750 tokens under full
  attention). Which of the two bounds task 3-1 was not measured.
- **Duration.** The base trainings took 1 h 46 min (task 1-2, 6,339 s) to
  3 h 59 min (task 2-1, 14,355 s), validations included. Scoring `val_report`
  took 85 s to 6 minutes on tasks 1-1, 1-2, 2-1 and 3-1, and 22 to 35 minutes
  on the group-4 base runs. The 90,000 steps of task 1-1 took 7 h 7 min
  (25,594 s), those of task 2-1 7 h 51 min (28,235 s) and `h050b256` on task
  4-4 3 h 43 min (13,367 s), all on shared GPUs.
- **Which code trained a checkpoint.** A run writes `trained_by.json` (the
  hashes of the model's code and of the scoring code) before its first step,
  and the score copies them. The 13 runs that started before that existed
  (commit `a34cd797`) got the file afterwards, by hand, with the hashes that
  `git diff` shows unchanged between their start commits and that commit. For
  those runs the record rests on the git history and on the working tree
  having been clean at their start, not on the run itself.

### Runs that were still training when the table was printed

Three variant runs had no score when the table was printed (13:45). The
`val_nrmse` values are from each run's `train_log.csv` at that time, on the 200
`val_select` shots. The expected ends are the remaining steps at the speed of
the last 2,000 steps and do not include the final scoring. None of these
numbers is a result.

| Task | Variant | Step at 13:45 | Latest `val_nrmse` (at step) | Best `val_nrmse` of the base run | Training expected to end |
|---|---|---|---|---|---|
| 4-5 | `s90k` | 62,500 of 90,000 | 0.6857 (62,500); lowest so far 0.6855 (60,000) | 0.6903 | about 15:15 |
| 4-1 | `h050b256` | 5,000 of 30,000 | 0.2029 (2,500) | 0.1865 | about 17:00 |
| 4-5 | `b256` | 3,400 of 30,000 | 0.7038 (2,500) | 0.6903 | about 19:00 |

When a run finishes, the command scores it, and `--table` takes it if its
`val_select` error is the lowest of its task. All three have finished (from
`/models/intermediate/mast/stage0/direct_streams.log` and each run's
`train_log.csv`, measured 2026-11-22), and the table was printed again at 23:54:

| Task | Variant | Finished | Exit | Wall time | Steps | Lowest `val_nrmse` | At | Last validation |
|---|---|---|---|---|---|---|---|---|
| 4-5 | `s90k` | 16:12:23 | 0 | 24,355 s | 90,000 of 90,000 | 0.6812 | step 75,000 (30th of 36) | 0.6830 at 90,000 |
| 4-1 | `h050b256` | 19:23:54 | 0 | 22,889 s | 30,000 of 30,000 | 0.1572 | step 30,000 (12th of 12) | the same |
| 4-5 | `b256` | 20:20:19 | 0 | 26,274 s | 30,000 of 30,000 | 0.6752 | step 20,000 (8th of 12) | 0.6785 at 30,000 |

The expected ends above were 15:15, 17:00 and 19:00, so the runs ended 57
minutes, 2 hours 24 minutes and 1 hour 20 minutes later than the estimate (the
speed of the last 2,000 steps at 13:45; why the runs were slower than that was
not examined). Their scores are in the
[variant tables of the 23:54 print](#the-table-printed-at-2354). The sections
above that say "still training" describe 13:45, and the table printed at 13:45,
next, is kept as it was printed.

Limits to carry with every direct number: the score is on validation shots
and the published scores it is compared with are on test shots; one seed per
run; one model size, and a training budget that every test of it found too
small (tasks 1-1 and 2-1 improved with three times the steps and task 4-4 with
four times the batch, batch 256 lowered the score on task 4-1 by 0.021 and on
task 4-5 by 0.011, and 90,000 steps lowered it on task 4-5 by 0.0069; a model
with more steps or a different size would score better); and the windows come from TokaMark's loader. A training that stops
before its `max_steps` is marked in the table (‡), and a direct score worse
than our TokaMind replication is marked "not competitive", because a weak
model is not a headroom finding. The table takes the variant with the lowest
best `val_nrmse` on `val_select` among those with a stored score, and
`--table` lists every variant, what differs in its `config.json`, its best
`val_nrmse` and its `val_report` score, including a variant whose selection
error is the lowest but which has no stored score yet (it cannot enter the
table).

## The headroom table

Printed on 2026-11-22 13:45 UTC by

```bash
scripts/mast/py.sh scripts/mast/run_headroom.py --table
```

which reads the stored scores, scores nothing and writes
`/models/intermediate/mast/stage0/headroom_table.csv`. What follows, down to
the line of the projected mean, is the Markdown the command printed, unchanged.

Scores on the benchmark's windows, NRMSE (lower is better). `test` is TokaMark's test split, `val` is `val_report`.

| task | mean test | mean val | persistence test | persistence val | persistence (all windows) test | persistence (all windows) val | floor test | floor val |
|---|---|---|---|---|---|---|---|---|
| 1-1 | 0.7566 | 0.7599 | undefined | undefined | undefined | undefined | 0.0000 | 0.0000 |
| 1-2 | 0.9984 | 1.0005 | undefined | undefined | undefined | undefined | 0.0236 | 0.0239 |
| 2-1 | 0.8174 | 0.8168 | undefined | undefined | undefined | undefined | 0.0038 | 0.0037 |
| 3-1 | 0.9837 | 0.9940 | 0.5364 | 0.5350 | 0.6868 | 0.6936 | 0.0279 | 0.0284 |
| 4-1 | 0.5114 | 0.4961 | 0.3247 | 0.2978 | 0.3095 | 0.2806 | 0.0369 | 0.0353 |
| 4-4 | 1.0270 | 1.0364 | 0.7178 | 0.7157 | 0.7178 | 0.7157 | 0.0349 | 0.0351 |
| 4-5 | 0.6533 | 0.6852 | 0.8701 | 0.9097 | 0.7743 | 0.8011 | 0.3366 | 0.3605 |

Fitted predictors and the direct model on `val_report` (with the 95% interval of the direct model's mean over shots), against the published scores on the test split.

| task | shrunk val | ridge val | direct val [95% interval] | direct variant | TokaMind replication test | best published test | by | direct / best | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1-1 | 0.7004 | 0.2992 | 0.1235 [0.1148, 0.1342] | s90k | 0.1503 | 0.0941 | gentoka | 1.31 | behind |
| 1-2 | 0.2204 | 0.0809 | 0.0362 [0.0350, 0.0375] | lr1e4 | 0.0433 | 0.0412 | gentoka | 0.88 | ahead |
| 2-1 | 0.6374 | 0.3268 | 0.1347 [0.1270, 0.1434] | s90k | 0.1809 | 0.0880 | gentoka | 1.53 | behind |
| 3-1 | 0.6771 | 0.4876 | 0.3267 [0.3091, 0.3461] | base | 0.3457 | 0.3311 | gentoka | 0.99 | tie |
| 4-1 | 0.2842 | 0.2359 | 0.1952 [0.1776, 0.2144] | h050 | 0.2147 | 0.2122 | tokamind_rerun | 0.92 | tie |
| 4-4 | 0.6758 | 0.4574 | 0.2859 [0.2656, 0.3062] | h050b256 | 0.3502 | 0.3116 | gentoka | 0.92 | ahead |
| 4-5 | 0.6925 | 0.6882 | 0.6475 [0.5934, 0.7063] | base | 0.6499 | 0.6510 | tokamind_paper | 0.99 | tie |

missing: no score is stored. undefined: the benchmark defines no persistence for the task. † stale: the stored score is not the score of what is loaded now. The scoring code (harness, tasks, baselines), the predictor's code, its fit or checkpoint, or TokaMind's coefficient files differ from those it was made with; or the record or its provenance is unreadable or cannot be computed now; or the per-shot file does not match the score; for a direct score also 'trained by other code than the code now loaded' (the checkpoint was trained by other code than the predictor and scoring code now loaded) and 'training code not recorded' (no `trained_by.json`). A score is not marked when TokaMark's code (windows, statistics, splits, the metric) or the Zarr data root changed: the record holds neither. ‡ unfinished: the direct model's training stopped before `max_steps`. not competitive: the direct model's `val_report` score is worse than our TokaMind replication's test score on the task.

Direct variants (the table takes the one with the lowest best val_nrmse among those with a stored score):

Task 1-1:

| variant | differs from base | best val_nrmse (val_select) | val_report | in the table |
|---|---|---|---|---|
| base | - | 0.1430 | 0.1361 | no |
| lr1e4 | lr=0.0001 | 0.1505 | 0.1418 | no |
| s90k | max_steps=90000 | 0.1315 | 0.1235 | yes |

Task 1-2:

| variant | differs from base | best val_nrmse (val_select) | val_report | in the table |
|---|---|---|---|---|
| base | - | 0.0522 | 0.0493 | no |
| lr1e4 | lr=0.0001 | 0.0390 | 0.0362 | yes |
| s90k | max_steps=90000 | 0.0462 | 0.0435 ‡ | no |

- 1-2 s90k val_report: trained 30000 of 90000 steps (unfinished)

Task 2-1:

| variant | differs from base | best val_nrmse (val_select) | val_report | in the table |
|---|---|---|---|---|
| base | - | 0.1525 | 0.1523 | no |
| lr1e4 | lr=0.0001 | 0.1589 | 0.1579 | no |
| s90k | max_steps=90000 | 0.1377 | 0.1347 | yes |

Task 3-1:

| variant | differs from base | best val_nrmse (val_select) | val_report | in the table |
|---|---|---|---|---|
| base | - | 0.3503 | 0.3267 | yes |

Task 4-1:

| variant | differs from base | best val_nrmse (val_select) | val_report | in the table |
|---|---|---|---|---|
| base | - | 0.1865 | 0.1937 | no |
| h050 | history_s=0.05 | 0.1835 | 0.1952 | yes |
| h050b256 | batch_size=256, history_s=0.05 | 0.2029 | no score | no |

Task 4-4:

| variant | differs from base | best val_nrmse (val_select) | val_report | in the table |
|---|---|---|---|---|
| base | - | 0.3097 | 0.3195 | no |
| h050 | history_s=0.05 | 0.3140 | 0.3230 | no |
| h050b256 | batch_size=256, history_s=0.05 | 0.2787 | 0.2859 | yes |

Task 4-5:

| variant | differs from base | best val_nrmse (val_select) | val_report | in the table |
|---|---|---|---|---|
| base | - | 0.6903 | 0.6475 | yes |
| b256 | batch_size=256 | 0.7038 | no score | no |
| s90k | max_steps=90000 | 0.6855 | no score | no |

† stale: the stored score is not the score of what is loaded now. The scoring code (harness, tasks, baselines), the predictor's code, its fit or checkpoint, or TokaMind's coefficient files differ from those it was made with; or the record or its provenance is unreadable or cannot be computed now; or the per-shot file does not match the score; for a direct score also 'trained by other code than the code now loaded' (the checkpoint was trained by other code than the predictor and scoring code now loaded) and 'training code not recorded' (no `trained_by.json`). A score is not marked when TokaMark's code (windows, statistics, splits, the metric) or the Zarr data root changed: the record holds neither. ‡ unfinished: the training stopped before `max_steps`.

projected mean over the 14 tasks (direct model on the 7 stage-0 tasks, best published score on the other 7): 0.1954; the bar is 0.183, matching the best published score everywhere would give 0.194

A first table was printed at 09:55, before five of the variants above had a
score. Three rows and the projected mean changed between the two prints; the
other four rows are the same in both.

| | Print of 09:55 | Print of 13:45 |
|---|---|---|
| Task 1-1 | 0.1361 [0.1272, 0.1468], base, behind | 0.1235 [0.1148, 0.1342], `s90k`, behind |
| Task 2-1 | 0.1523 [0.1438, 0.1615], base, behind | 0.1347 [0.1270, 0.1434], `s90k`, behind |
| Task 4-4 | 0.3195 [0.2987, 0.3409], base, tie | 0.2859 [0.2656, 0.3062], `h050b256`, ahead |
| Projected mean | 0.1999 | 0.1954 |

Reading the columns. `mean`, `persistence` and the floor are given on both
splits beside every `val_report` score, because the same predictor differs
between the two splits by up to 0.04 (**measured 2026-11-22**: persistence on task
4-5 0.9097 on `val_report` against 0.8701 on test; mean on 4-5 0.6852 against
0.6533; mean on 4-3 0.7591 against 0.7313; persistence on 4-1 0.2978 against
0.3247). The columns of fitted predictors and the direct model are on
`val_report`; the published scores, our replication and the verdict are on the
test split. **The verdict column therefore compares a `val_report` score with a
test-split published score.** "Best published" is the lowest of the TokaMind paper, the
TokaMind re-run by GenToka and GenToka, and "by" says whose it is. The verdict
is "ahead" if the whole 95% interval (bootstrap of the mean over shots, 2,000
resamples, seed 0) is below the best published score, "behind" if it is above,
and "tie" otherwise. The projected mean is the 14-task mean with the direct
model's score on the seven stage-0 tasks and the best published score on the
other seven. It is not computed while any stage-0 task has no direct score. The
bar is 0.183, which is 10% below GenToka's 0.2033; a model that only matched
the best published score on every task would score 0.194.

Rules for cells that are not plain numbers. A cell with no stored score reads
"missing" and is never printed as 0. "undefined" is a persistence cell for a
task where the benchmark defines no persistence. A score that is stored but
not current (the scoring code, the predictor, its fit or its checkpoint changed
since) is shown with † and named in the `stale` column of the CSV; a direct
score of an unfinished training is shown with ‡. In the print of 13:45 no cell
of the two tables carries a mark; in the variants listing the `s90k` run of
task 1-2, stopped at a third of its steps, carries ‡, and the three runs that
were still training read "no score".

### The table printed at 23:54

The same command was run again at 23:54 UTC, after the three runs of the
section above had finished. Its first table (the benchmark's
windows: mean, persistence, floor) and its direct variants of tasks 1-1, 1-2,
2-1, 3-1 and 4-4 are the same as in the print of 13:45, to the last digit, and
are not repeated. What follows is the Markdown of the 23:54 print for what
changed (the two rows of the second table, the variant tables of tasks 4-1 and
4-5, and the line of the projected mean), unchanged and in the order of the
print.

Fitted predictors and the direct model on `val_report` (with the 95% interval of the direct model's mean over shots), against the published scores on the test split.

| task | shrunk val | ridge val | direct val [95% interval] | direct variant | TokaMind replication test | best published test | by | direct / best | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 4-1 | 0.2842 | 0.2359 | 0.1727 [0.1553, 0.1923] | h050b256 | 0.2147 | 0.2122 | tokamind_rerun | 0.81 | ahead |
| 4-5 | 0.6925 | 0.6882 | 0.6361 [0.5833, 0.6938] | b256 | 0.6499 | 0.6510 | tokamind_paper | 0.98 | tie |

Task 4-1:

| variant | differs from base | best val_nrmse (val_select) | val_report | in the table |
|---|---|---|---|---|
| base | - | 0.1865 | 0.1937 | no |
| h050 | history_s=0.05 | 0.1835 | 0.1952 | no |
| h050b256 | batch_size=256, history_s=0.05 | 0.1572 | 0.1727 | yes |

Task 4-5:

| variant | differs from base | best val_nrmse (val_select) | val_report | in the table |
|---|---|---|---|---|
| base | - | 0.6903 | 0.6475 | no |
| b256 | batch_size=256 | 0.6752 | 0.6361 | yes |
| s90k | max_steps=90000 | 0.6812 | 0.6406 | no |

projected mean over the 14 tasks (direct model on the 7 stage-0 tasks, best published score on the other 7): 0.1929; the bar is 0.183, matching the best published score everywhere would give 0.194

Compared with the print of 13:45, two rows and the projected mean changed; the
other five rows are the same.

| | Print of 13:45 | Print of 23:54 |
|---|---|---|
| Task 4-1 | 0.1952 [0.1776, 0.2144], `h050`, direct / best 0.92, tie | 0.1727 [0.1553, 0.1923], `h050b256`, direct / best 0.81, ahead |
| Task 4-5 | 0.6475 [0.5934, 0.7063], base, direct / best 0.99, tie | 0.6361 [0.5833, 0.6938], `b256`, direct / best 0.98, tie |
| Projected mean | 0.1954 | 0.1929 |

No score in the print of 23:54 carries a † (stale) mark. Only the `s90k` run of
task 1-2 carries ‡ (unfinished), as in the print of 13:45. In the variants
listing of task 4-5 the `s90k` run (0.6406) is not in the table, because its
`val_select` error (0.6812) is above that of `b256` (0.6752).

## What it says

Measured and inferred are kept apart. **Measured** statements are numbers in the
table above, with the limits given with them. **Inferred** statements are
stated as such.

**Question 1: per task, is the direct model ahead of, tied with or behind the
best published score?** Measured, from the table printed at 23:54: behind on
tasks 1-1 and 2-1, ahead on tasks 1-2, 4-1 and 4-4, and tied on tasks 3-1 and
4-5. At 13:45 task 4-1 was a tie, and the verdicts of the other six rows were the same. The
direct score is on `val_report`, the published score on the test split, and the
verdict is the table's rule on the 95% interval over shots. The rows below are
those of 23:54; the column of the mean baseline does not depend on the direct
model's variant and is the same as at 13:45.

| Task | Direct on `val_report` [95% interval] (variant) | Best published on test (by) | Verdict | Mean baseline, `val_report` minus test |
|---|---|---|---|---|
| 1-1 | 0.1235 [0.1148, 0.1342] (`s90k`) | 0.0941 (GenToka) | behind, by 0.021 at the nearer end of the interval | +0.003 |
| 1-2 | 0.0362 [0.0350, 0.0375] (`lr1e4`) | 0.0412 (GenToka) | ahead, by 0.0037 at the nearer end of the interval | +0.002 |
| 2-1 | 0.1347 [0.1270, 0.1434] (`s90k`) | 0.0880 (GenToka) | behind, by 0.039 at the nearer end of the interval | -0.001 |
| 3-1 | 0.3267 [0.3091, 0.3461] (base) | 0.3311 (GenToka) | tie | +0.010 |
| 4-1 | 0.1727 [0.1553, 0.1923] (`h050b256`) | 0.2122 (TokaMind re-run) | ahead, by 0.0199 at the nearer end of the interval, which is more than the mean baseline's difference between the splits and less than persistence's (0.027); it was a tie at 13:45 (`h050`, 0.1952 [0.1776, 0.2144]) | -0.015 |
| 4-4 | 0.2859 [0.2656, 0.3062] (`h050b256`) | 0.3116 (GenToka) | ahead, by 0.0054 at the nearer end of the interval, which is less than the difference between the splits | +0.009 |
| 4-5 | 0.6361 [0.5833, 0.6938] (`b256`) | 0.6510 (TokaMind paper) | tie (it was a tie at 13:45 with the base run, 0.6475 [0.5934, 0.7063]) | +0.032 |

What the rows say beyond the verdict, each measured unless marked:

- *Tasks 1-1 and 2-1 are where the direct model is clearly behind, with three
  times the steps too.* On both the mean baseline differs between the two
  splits by 0.003 or less, so the split is unlikely to be the explanation
  (inferred from the mean baseline; the direct model has no test score). On
  task 1-1
  the direct model (0.1235) is below GenToka's re-run of TokaMind (0.1363),
  the TokaMind paper and our replication (0.150, 0.1503); GenToka is at
  0.0941. On task 2-1 the direct model (0.1347) is below the TokaMind paper
  and our replication (0.181, 0.1809) and 0.006 above GenToka's re-run of
  TokaMind (0.1289), which lies inside its interval; GenToka is at 0.0880.
  These are the two stage-0 tasks on which GenToka's gain over its own
  TokaMind re-run is largest (0.042 and 0.041). Three times the steps lowered
  the two scores by 0.013 and 0.018 and the lower learning rate raised both
  (paired differences in the section on the direct model), and 0.029 and
  0.047 remain to GenToka. Both long runs were still improving at their last
  step, so where they would end is not known.
- *Against our TokaMind replication* (test split) the direct model's
  `val_report` score is lower on all of the seven tasks, and no row carries
  the "not competitive" mark. The replication's score is above the direct
  model's interval on tasks 1-1, 1-2, 2-1, 4-1 and 4-4 (on task 4-1 by 0.0224,
  0.2147 against an upper end of 0.1923; at 13:45 it was 0.0003, against an
  upper end of 0.2144) and inside it on tasks 3-1 and 4-5. The two scores are on
  different splits.
- *Task 4-1: the verdict depends on the batch too, and the margin is of the
  size of the split difference.* With the base batch of 64 the base run
  scores 0.1937 and `h050` 0.1952, and the row was a tie. With batch 256 at
  50 ms of history it is 0.1727 [0.1553, 0.1923], and the whole interval is
  below the published 0.2122 (TokaMind re-run). The margin is 0.0199. On this
  task the mean baseline is lower on `val_report` than on test by 0.0153 (0.4961
  against 0.5114), persistence by 0.0269 (0.2978 against 0.3247) and the floor
  by 0.0016 (0.0353 against 0.0369). By the rule at the end of this page (the
  margin against the mean baseline's difference) the verdict stands, by a
  factor of 1.3. The margin is smaller than persistence's difference, so
  *inferred, not measured:* part of the 0.0199 may be the `val_report` shots
  being easier than the test shots, and the row is read as: the direct model is
  below the best published score on task 4-1, by about the amount by which the
  splits differ. The variant was defined before any direct score existed.
- *Task 4-4: the verdict depends on the batch.* With the base batch of 64 the
  score is 0.3195 and the row was a tie. With batch 256 at 50 ms of history it
  is 0.2859, and the whole interval is below GenToka's 0.3116; on the same
  553 shots the paired difference to the base run is -0.0336 (95% interval
  -0.0429 to -0.0248). The margin of the "ahead" is 0.0054, and on this task
  the mean baseline differs by 0.0094 between the splits (persistence by
  0.0021 in the other direction, the floor by 0.0002). By the limit stated at
  the end of this page a margin smaller than that is not determined, so the
  row is read as: the direct model is at or below the best published score on
  task 4-4. The variant was defined before any direct score existed.
- *Task 4-5: the direct model is below the mean baseline, which no other entry
  of its row is by more than 0.004.* On the 351 `val_report` shots the
  paired difference direct minus mean is -0.0377 (95% interval -0.0461 to
  -0.0298; paired bootstrap over shots), 5.5% of the mean's score. The gain is
  skewed: the direct model is the lower one in 205 of the 351 shots, but
  the median difference per shot is -0.0013 against a mean of -0.0377, so the
  shots with large differences carry it. These figures are those of the base
  run, the row of 13:45; the paired bootstrap against the mean was not run for
  `b256`, the row of 23:54. The two means differ by 0.0491 (0.6852 against
  0.6361, 7.2% of the mean's score). On the test split our replication and the best published score are 0.5%
  and 0.4% below the mean, and GenToka's own score (0.7487) and its TokaMind
  re-run (0.7377) are above it. The verdict stays a tie, because the direct
  score is on other shots than the published one and the mean differs by 0.032
  between the splits. *Inferred, not measured:* if the same relative gain held
  on the test shots the direct model would score about 0.617 there (with the
  base run's 5.5%; the 7.2% of `b256` would give about 0.606, the same
  inference).
- *Task 1-2 depends on the learning rate, not on the model.* At the default rate
  the base run scores 0.0493 and the verdict would be "behind"; at 1e-4 the
  score is 0.0362. On the same 525 shots the paired difference between the
  two runs is -0.0131 (95% interval -0.0142 to -0.0121). The variant was
  chosen after the base run's `val_report` score was visible, which is the
  weaker order of events; it was chosen on the training loss and the
  `val_select` curve, and the table took it by `val_select`. The `lr1e4` runs
  of tasks 1-1 and 2-1 asked whether the default rate holds those tasks back
  in the same way. It does not: both are worse than their base runs, by
  +0.0056 on `val_report` each (95% intervals +0.0041 to +0.0073 and +0.0023
  to +0.0089, paired). Tasks 3-1, 4-1, 4-4 and 4-5 were not run at the lower
  rate.

**Question 2: how much of each published score is TokaMind's representation
floor?** Measured on 2026-11-22, test split, against our replication, which is
within 0.0022 of the paper's TokaMind number on every task:

| Task | Floor | TokaMind replication | Floor / replication | Best published (by) | Floor / best published |
|---|---|---|---|---|---|
| 1-1 | 0.0000 | 0.1503 | 0% | 0.0941 (GenToka) | 0% |
| 1-2 | 0.0236 | 0.0433 | 55% | 0.0412 (GenToka) | 57% |
| 2-1 | 0.0038 | 0.1809 | 2% | 0.0880 (GenToka) | 4% |
| 3-1 | 0.0279 | 0.3457 | 8% | 0.3311 (GenToka) | 8% |
| 4-1 | 0.0369 | 0.2147 | 17% | 0.2122 (TokaMind re-run) | 17% |
| 4-4 | 0.0349 | 0.3502 | 10% | 0.3116 (GenToka) | 11% |
| 4-5 | 0.3366 | 0.6499 | 52% | 0.6510 (TokaMind paper) | 52% |

The answer is in two groups. On tasks 1-1, 2-1, 3-1, 4-1 and 4-4 the floor is
between 0% and 17% of TokaMind's score, so the representation is not what holds
TokaMind back there; its error is prediction error. On task 1-2 the floor is
0.0236 of 0.0433 (55%), and on task 4-5 it is 0.3366 of 0.6499 (52%): about
half of TokaMind's NRMSE on those two tasks is what its representation cannot
express, before any prediction is made. The last column is a fact about TokaMind's
representation only. GenToka's truncation was not checked, so for GenToka's
scores (the best published on 1-1, 1-2, 2-1, 3-1 and 4-4) the floor is a
reference and not its own floor.

Inferred, not measured: if the representation error and the prediction error
were independent, errors would add in quadrature, and the floor would be
`(0.0236 / 0.0433)^2` = 30% of the squared error on task 1-2 and 27% on task
4-5, and 3% or less on the others. That independence was not tested, and
NRMSE is a mean over shots of a mean over signals, so the quadrature statement
is an approximation. Also inferred: for a model that predicts only the kept
coefficients the floor is close to a lower bound, so on tasks 1-2 and 4-5 the
room that remains for such a model is the part above the floor. The direct
model does not predict in that representation, which is what its rows are
there to measure on those tasks. The floor on `val_report` has the same
pattern, and the one split-dependent value is task 4-5 (0.3605 against 0.3366).

**Question 3: is 0.183 within reach according to the projected mean, and
which tasks carry it?** **By this direct model, no.** The projected mean is
0.1929 in the print of 23:54 (the direct model's `val_report` score on the
seven stage-0 tasks and the best published score on the other seven). The bar
is 0.183. For scale: the best published score on every task gives 0.194,
GenToka 0.2033, our TokaMind replication 0.2124. The projected mean was 0.1999
in the print of 09:55, before the longer runs of tasks 1-1 and 2-1 and the
batch-256 run of task 4-4 had finished, and 0.1954 in the print of 13:45; the
runs of tasks 4-1 and 4-5 that finished after that took it to 0.1929.

Where the difference from the best published scores comes from (direct minus
best published, per task, in the print of 23:54): 1-1 +0.0294, 1-2 -0.0050,
2-1 +0.0467, 3-1 -0.0044, 4-1 -0.0395, 4-4 -0.0257, 4-5 -0.0149; together
-0.013 (summed from these differences). At 13:45 the sum was +0.020, with 4-1 at
-0.0170 and 4-5 at -0.0035. Tasks 1-1 and 2-1 are +0.076 of it, and the other
five together give back 0.090 (0.056 at 13:45).

What 0.183 asks for. The 14 scores would have to sum to 2.562. With the best
published score on the other seven tasks (sum 0.985), the seven stage-0 tasks
would have to sum to 1.577. Their best published scores sum to 1.729 and the
direct model's to 1.716 (1.750 at 13:45; 0.1929 times 14 less 0.985). So the
bar asks for 0.153 below the best published scores over these seven tasks, or
the same amount found among the other seven, which have no headroom row yet.
The direct model is 0.013 below them, which leaves 0.139 to find (0.173 at
13:45).

**The headroom check of the design therefore did not pass.** Its text is: "If
the direct supervised model cannot reach the numbers of goal G4 on the probe
tasks, the bar goes back to Nathaniel with the table before stage 2 is
committed to it." The bar is not changed on this page; the decision is his.

**The per-task conditions of G4 give a second, independent reason.** Goal G4
(`ignite-mast-design.md`, "Proposed reading of 'by far the best'") has three
numbers: a mean at or below 0.183 (question 3, above); better, beyond noise,
than both GenToka's published score and our TokaMind replication on at least 10
of the 14 tasks; and worse beyond noise on none. "Beyond noise" is read by the
two rules of this page: the interval rule of the table (the comparison score is
outside the direct model's 95% interval over `val_report` shots) and the
split-difference rule in the limits below (a verdict whose margin, the distance
from the nearer end of the interval to the comparison score, is smaller than the
mean baseline's `|mean val - mean test|` on the task is undetermined). Both are
applied here to each comparison, against GenToka's score and against our
replication separately. Measured 2026-11-23 from the scores stored for the
print of 23:54 (GenToka's scores are the last column of `_PUBLISHED` in
`tasks.py`); each cell gives the interval verdict of the direct model, its
margin, and what the split-difference rule says:

| task | direct val [95% interval] | mean baseline, val minus test | against GenToka | against our replication |
|---|---|---|---|---|
| 1-1 | 0.1235 [0.1148, 0.1342] | 0.0033 | behind, 0.0207, stands (GenToka 0.0941) | ahead, 0.0161, stands (0.1503) |
| 1-2 | 0.0362 [0.0350, 0.0375] | 0.0021 | ahead, 0.0037, stands (0.0412) | ahead, 0.0058, stands (0.0433) |
| 2-1 | 0.1347 [0.1270, 0.1434] | 0.0006 | behind, 0.0390, stands (0.0880) | ahead, 0.0375, stands (0.1809) |
| 3-1 | 0.3267 [0.3091, 0.3461] | 0.0103 | tie (0.3311) | tie (0.3457) |
| 4-1 | 0.1727 [0.1553, 0.1923] | 0.0153 | ahead, 0.0375, stands (0.2298) | ahead, 0.0224, stands (0.2147) |
| 4-4 | 0.2859 [0.2656, 0.3062] | 0.0093 | ahead, 0.0054, undetermined (0.3116) | ahead, 0.0440, stands (0.3502) |
| 4-5 | 0.6361 [0.5833, 0.6938] | 0.0319 | ahead, 0.0549, stands (0.7487) | tie (0.6499) |

On the seven tasks that have a row, the direct model is better than both beyond
noise on 2 (tasks 1-2 and 4-1). By the interval rule alone it would be 3, with
task 4-4, but its margin against GenToka (0.0054) is smaller than the split
difference (0.0093), so by the second rule that comparison is undetermined. It
is worse beyond noise than GenToka on 2, tasks 1-1 and 2-1, and worse than our
replication on none. So the condition "worse beyond noise on none" already
fails on its own, on tasks 1-1 and 2-1, independently of the mean. For the
count of "10 of 14": the other seven tasks have no direct row on this page (the
14-task table of plan 1a covers them), and 2 plus all seven of them is 9, so
this direct model cannot reach 10 of 14 (3 plus seven is exactly 10, with the
interval rule alone). The verdict that the headroom check did not pass is
therefore unchanged, and it now has two reasons: the mean (0.1929 against
0.183) and G4's per-task conditions. The 0.0033, 0.0006 and 0.0093 in the table
are unrounded differences of the mean baseline's two columns; the text of the
limits below, which used the printed four-digit values, gives 0.003, 0.001 and
0.0094 for the same differences.

What the result shows and what it does not:

- *Shown (measured):* one supervised model per task with 14 to 18 million
  parameters, a squared loss and one seed is below the best published score
  beyond its interval on tasks 1-2, 4-1 and 4-4 (on 4-4 by 0.0054, less than the
  0.0094 by which the mean baseline differs between the two splits, so by the
  rule in the limits below that verdict is undetermined; on 4-1 by 0.0199,
  more than the 0.0153 of the mean baseline, so it stands by that rule), at its
  level on tasks 3-1 and 4-5 (a tie by the interval over shots), and behind on
  tasks 1-1 and 2-1. Summed over the seven tasks it is 0.013 below the best
  published scores (0.020 above them at 13:45), each of which is the best of
  three published numbers.
- *Shown (measured): where the training budget was tested, it was short.*
  Every finished test of the
  training budget lowered a score: three times the steps on tasks 1-1 and 2-1
  (by 0.013 and 0.018, with the validation error still falling at step
  90,000), four times the batch on task 4-4 (by 0.034), and, since 23:54, a
  batch of 256 on task 4-1 (by 0.021 against the base run, and 0.0225 against
  `h050`) and on task 4-5 (by 0.011), and three times the steps on task 4-5 (by
  0.0069, and 0.0044 above `b256`). The first three runs moved the projected
  mean from 0.1999 to 0.1954 and the last two from 0.1954 to 0.1929. *Not
  shown:* that the other two rows are short of budget. On tasks 1-2 (`lr1e4`)
  and 3-1 the validation error was flat at the end of the run at the settings
  used, and the one longer run of task 1-2 became unstable and was stopped.
- *Not shown: that 0.183 cannot be reached.* The direct model is a lower bound
  on what the data allows. Where it is behind, a published model already does
  better, so the distance is a limit of this model. Its other measured limits
  are in the section on the direct model: it is unstable at its default
  learning rate on task 1-2, its squared loss meets heavy-tailed targets on
  tasks 1-1 and 3-1, and it has one size.
- *Not shown either: that 0.183 can be reached.* Three rows of the table (tasks
  1-2, 4-1 and 4-4) are below the best published score beyond their interval,
  by 0.004, 0.020 and 0.005 at the nearer end of the interval (0.005, 0.040 and
  0.026 in the score), and for task 4-4 the difference between the splits leaves
  that verdict undetermined. The bar needs 0.153 below the best published
  scores in sum, and tasks 1-1 and 2-1 are 0.076 above them.
- *Inferred, not measured:* task 4-5 is where the most room is. It is 26% of
  GenToka's summed score, half of TokaMind's score there is its representation
  floor, and the direct model is 5.5% below the mean baseline on validation
  shots (7.2% with `b256`, by the difference of the two means; the paired
  figure was not computed) where the published models are within 0.5% of it on
  test shots. A gain of that relative size on the test shots would be about
  0.034 of the 0.153 (0.045 for 7.2%).

The three runs that were still training at 13:45 were on tasks 4-1 and 4-5, and
they have finished. At 13:45 the seven-task sum would have had to fall by 0.173
and this page said, *inferred, not measured,* that a gain of the size of the
largest change a variant had made (0.034, batch 256 on task 4-4, 10.5% of its
base score) on both tasks would be about 0.09, so that the runs were not
expected to change the answer by themselves. They changed the sum by 0.034
(0.0225 on task 4-1 and 0.0114 on task 4-5, direct score of 13:45 minus that
of 23:54), and the sum still has to fall by 0.139 to reach the bar. The
answer did not change: 0.1929 is above 0.183.

## Limits that apply to the whole page

- **Validation shots against published test scores.** Every fitted number is on
  `val_report`, every published number is on the test split, and the same
  predictor differs between the two by up to 0.04. Only differences larger than
  that are readable without the baselines of the table.
- **One seed.** The direct model is one run per task, and run-to-run noise is
  measured in stage 2, not here. The interval of the direct model is over
  shots, not over seeds.
- **The direct model's size and training budget.** One model size (14 to 18
  million parameters), one loss, 30,000 steps for the base runs. The headroom
  it reports is a lower bound on what a specialized model can reach, and where
  it is behind a published score it shows a limit of this model and not of the
  data. On tasks 4-1, 4-4 and 4-5 the base batch is 64, so 30,000 steps see a
  quarter of the windows that 30,000 steps of batch 256 would. Measured since
  the first print: on task 4-4 batch 256 lowers the score by 0.034, on task 4-1
  by 0.021 and on task 4-5 by 0.011 (against the base runs, the last two in the
  print of 23:54), and on tasks 1-1 and 2-1 three times the steps lower it by
  0.013 and 0.018 with the validation error still falling at the last step (on
  task 4-5 by 0.0069). The rows of tasks 1-1 and 2-1 are therefore not
  converged models, and the batch-64 runs of tasks 4-1, 4-4 and 4-5 were short
  of budget; the rows of tasks 4-1, 4-4 and 4-5 are now batch-256 runs. Tasks
  1-2 and 3-1 have no finished test of the budget, so nothing is claimed about
  them either way.
- **A verdict is no finer than the difference between the splits.** The
  interval of the direct model is over `val_report` shots and the published
  score is on test shots. The same predictor differs between the two splits
  by up to 0.04 (the columns `mean test` and `mean val` give that difference
  per task). A verdict whose margin, the distance from the nearer end of the
  interval to the published score, is smaller than that difference on its task
  is undetermined, whatever the column says. From commit `e3957208` on, the
  table prints the outcome of this rule in its own column, `split rule`
  (`split_rule` in the CSV), beside the verdict: "stands" or "undetermined" for
  an ahead or behind verdict (undetermined also when a mean baseline score is
  missing), empty for a tie. The `verdict` column itself still uses the
  interval only. Printed 2026-11-23 from the stored scores, it reads stands on
  tasks 1-1, 1-2, 2-1 and 4-1, undetermined on task 4-4, and is empty on tasks
  3-1 and 4-5. In the print of 13:45 and in that
  of 23:54 the two "behind" verdicts have margins of 0.021 (task 1-1) and 0.039
  (task 2-1)
  where the mean baseline differs by 0.003 and 0.001 between the splits, so
  they stand. The "ahead" on task 1-2 has a margin of 0.0037 where the mean
  baseline differs by 0.0021 and the floor by 0.0003: it stands by this rule,
  by less than a factor of two. The "ahead" on task 4-4 has a margin of 0.0054
  where the mean baseline differs by 0.0094: by this rule it is undetermined.
  The "ahead" on task 4-1, new in the print of 23:54, has a margin of 0.0199
  where the mean baseline differs by 0.0153: it stands by this rule, by a
  factor of 1.3. Persistence differs by 0.0269 on that task, more than the
  margin, which the rule does not use.
- **No data is not a negative.** A "missing" cell is a score nobody has stored
  yet. It is not a bad score and it is not 0.
- **The cheap predictors are cheap.** They say what trivial means predict, not
  what a good model can.
- **The floor is TokaMind's.** It describes TokaMind's coefficient choice on
  its own fine-tune configuration, not a property of the task.
- **What "current" covers for a stored score.** A score is kept, and shown
  without a mark, when the hashes of our scoring code, of the predictor and of
  its fit or checkpoint are the ones now loaded. TokaMark's own code and the
  data root are neither recorded in a score nor compared. The cache manifest
  records TokaMark at revision `146b0a0f8738` when the cache was built, and
  the data root is `/models/mast/tokamark/v1`. No score records the TokaMark
  revision it was computed with, so that the scores were made at the same
  revision is not recorded anywhere; after an update of TokaMark or a move of
  the
  data, the stored scores have to be computed again with `--force`.
- **The hashes are of the code as loaded.** A training or scoring process
  stamps its results with the files it read at start, and its loader workers
  run the same code because they are forked from it (Python 3.12 here). With a
  start method that re-imports modules, a file edited during a run could
  differ between the parent and its workers.
- **Runs from a git worktree use the main checkout's code on node13.**
  `scripts/mast/py.sh` starts Python with `-I`, which ignores `PYTHONPATH`,
  and the environment's `.pth` file names the main checkout's `src`. Everything
  on this page was run from the main checkout.

## The other seven tasks (stage 1a)

Stage 1a trained the direct model on the other seven tasks of TokaMark (1-3,
2-2, 2-3, 3-2, 3-3, 4-2 and 4-3) with the code and the rules of the seven
stage-0 tasks, and printed the 14-task table. **Every score in this section is
the score of the direct supervised model, trained from scratch on one task, which is
an instrument of this work. It is not IGNITE and it has no pretraining.** Its
rows are not tuned results. In NRMSE terms (lower is better), each score is an
upper bound on the NRMSE that this model reaches on its task at the stated
budget: a longer, larger or differently trained model may reach a lower one.

**Measured 2026-11-23.** The table was printed at 05:24 UTC by

```bash
scripts/mast/py.sh scripts/mast/run_headroom.py --table-all > /models/intermediate/mast/stage0/headroom_table_all.md
```

from the scores stored under `/models/intermediate/mast/stage0/scores`. The
command also wrote `headroom_table_all.csv` beside the Markdown. The section
was written on top of revision `8af4ca49` (`git rev-parse --short HEAD`, the
page's revision before the section existed) and first committed as `22e16535`.
These are revisions of the page, not of the scores: each score carries the
hashes of the code that made it (the scoring code and the predictor, as
described under "What was run").

What follows is the Markdown of the print for the seven new rows of both
tables (the other seven rows are those of the 23:54 print above, and the
14-task print repeats them), unchanged except for three things. Two lines of
the output are left out: the first, which only names the root the command
writes under, and the second, "Headroom table of 2026-11-23 05:24 UTC, from the
scores under ...". The print's heading "Direct variants (the table takes the
one with the lowest best val_nrmse among those with a stored score):" is
replaced by this page's own headings.

Scores on the benchmark's windows, NRMSE (lower is better). `test` is TokaMark's test split, `val` is `val_report`.

| task | mean test | mean val | persistence test | persistence val | persistence (all windows) test | persistence (all windows) val | floor test | floor val |
|---|---|---|---|---|---|---|---|---|
| 1-3 | 0.9267 | 0.9312 | undefined | undefined | undefined | undefined | 0.0293 | 0.0294 |
| 2-2 | 0.9914 | 0.9935 | undefined | undefined | undefined | undefined | 0.0218 | 0.0217 |
| 2-3 | 0.9230 | 0.9249 | undefined | undefined | undefined | undefined | 0.0297 | 0.0282 |
| 3-2 | 0.6297 | 0.6197 | 0.3651 | 0.3534 | 0.3990 | 0.3906 | 0.0296 | 0.0288 |
| 3-3 | 0.8412 | 0.8412 | undefined | undefined | undefined | undefined | 0.0089 | 0.0091 |
| 4-2 | 0.5114 | 0.4961 | 0.3247 | 0.2978 | 0.3095 | 0.2806 | 0.0369 | 0.0353 |
| 4-3 | 0.7313 | 0.7591 | undefined | undefined | undefined | undefined | 0.0177 | 0.0162 |

Fitted predictors and the direct model on `val_report` (with the 95% interval of the direct model's mean over shots), against the published scores on the test split.

| task | shrunk val | ridge val | direct val [95% interval] | direct variant | TokaMind replication test | best published test | by | direct / best | verdict | split rule |
|---|---|---|---|---|---|---|---|---|---|---|
| 1-3 | 0.6807 | 0.2228 | 0.1138 [0.1035, 0.1256] | lr1e4 | 0.1198 | 0.1156 | gentoka | 0.98 | tie |  |
| 2-2 | 0.2323 | 0.0904 | 0.0400 [0.0387, 0.0414] | lr1e4 | 0.0439 | 0.0370 | gentoka | 1.08 | behind | undetermined |
| 2-3 | 0.6914 | 0.2067 | 0.0813 [0.0741, 0.0901] | lr1e4 | 0.0917 | 0.0828 | gentoka | 0.98 | tie |  |
| 3-2 | 0.3842 | 0.3402 | 0.2309 [0.2218, 0.2414] | base | 0.2671 | 0.2522 | gentoka | 0.92 | ahead | stands |
| 3-3 | 0.7736 | 0.3399 | 0.2194 [0.2000, 0.2430] | base | 0.2197 | 0.2147 | gentoka | 1.02 | tie |  |
| 4-2 | 0.2842 | 0.2366 | 0.1735 [0.1564, 0.1922] | h050b256 | 0.2155 | 0.2140 | tokamind_paper | 0.81 | ahead | stands |
| 4-3 | 0.8253 | 0.2019 | 0.0836 [0.0758, 0.0923] | h050b256 | 0.0810 | 0.0691 | gentoka | 1.21 | behind; not competitive | undetermined |

missing: no score is stored. undefined: the benchmark defines no persistence for the task. † stale: the stored score is not the score of what is loaded now. The scoring code (harness, tasks, baselines), the predictor's code, its fit or checkpoint, or TokaMind's coefficient files differ from those it was made with; or the record or its provenance is unreadable or cannot be computed now; or the per-shot file does not match the score; for a direct score also 'trained by other code than the code now loaded' (the checkpoint was trained by other code than the predictor and scoring code now loaded) and 'training code not recorded' (no `trained_by.json`). A score is not marked when TokaMark's code (windows, statistics, splits, the metric) or the Zarr data root changed: the record holds neither. ‡ unfinished: the direct model's training stopped before `max_steps`. not competitive: the direct model's `val_report` score is worse than our TokaMind replication's test score on the task. split rule: for an ahead or behind verdict, 'stands' if its margin (the distance from the nearer end of the interval to the published score) is larger than the mean baseline's difference between the splits on the task, else 'undetermined'; empty for any other verdict.

No cell of the seven rows carries a † or a ‡ and no cell is missing; every
chosen run trained all of its `max_steps` (30,000). The one mark of the verdict
column is "not competitive" on task 4-3: the direct score (0.0836, on
`val_report`) is above our TokaMind replication's test score (0.0810). The mean
baseline of that task is 0.0278 higher on `val_report` than on test (0.7591
against 0.7313), so the two scores are on different splits and the margin is
0.0026. The mean, persistence and floor columns of task 4-2 equal those of task
4-1 in every digit printed (as the harness check above shows for the mean and
persistence on test); why was not checked. The partial answer is in the
benchmark's own published numbers: `tasks.py:40-41` hold the published mean and
persistence of tasks 4-1 and 4-2 as the same pair, `(0.511, 0.325, ...)` for
both (mean 0.511 and persistence 0.325), so the published baselines of the two
tasks are equal; their TokaMind and GenToka scores differ (0.213 against 0.214,
0.2122 against 0.2143 and 0.2298 against 0.2302). That the benchmark publishes
equal baselines agrees with our equal scores but does not show why the two
tasks give them; that was not checked.

### The variant of each task and its learning rate

The table takes, per task, the variant with the lowest best `val_nrmse` on
`val_select`, as in stage 0, and `val_report` is not used to choose. The base
configuration has learning rate 3e-4; `lr1e4` has 1e-4 and nothing else
changed; `h050b256` is the base configuration with `history_s` 0.05 (50 ms of
history instead of 150 ms) and batch 256, at the base learning rate 3e-4. The
configurations were fixed before any direct score of these tasks existed
(mean-baseline scores existed from stage 0's harness check), each carried over
from a stage-0 task where it was measured (see the next paragraph). Only tasks
1-3 and 2-3 chose between two learning rates (3e-4 and 1e-4); task 2-2 ran
`lr1e4` only, tasks 3-2 and 3-3 ran the base configuration only, and the two
variants of tasks 4-2 and 4-3 differ in history and batch, not in learning
rate. The batch sizes, read from each run's `config.json` on 2026-11-23, are:
256 on tasks 1-3 (both variants), 2-2, 2-3 (both variants) and 3-2; 64 on task
3-3; and, on tasks 4-2 and 4-3, 64 on the base runs (and on the `s90k` run of
4-3) and 256 on the `h050b256` runs. So 3-3 is the one chosen row of the seven
at batch 64.

"Still falling" below is the outcome of a rule, not a finding about the curve.
The rule is the plan's: on `val_select`, from each run's `train_log.csv`, the
run's lowest `val_nrmse` comes at the last or the second-last of its 12
validations (applied by the controller on 2026-11-23 at 05:28 UTC). A run can
meet it with a curve that is flat at the end. Read from the `train_log.csv` of
the chosen runs on 2026-11-23: task 2-2 has 0.0435, 0.0434 and 0.0435 at steps
25,000, 27,500 and 30,000; task 2-3 has 0.0883 at 27,500 and 0.0884 at 30,000
(0.08834 and 0.08845); task 4-2 has 0.1645 at both 27,500 and 30,000 (0.16454
and 0.16448). On tasks 3-2 and 3-3 the last value is still lower than the
earlier ones (3-2: 0.2268 at 27,500 and 0.2250 at 30,000; 3-3: 0.2120 and
0.2086).

| Task | Variant chosen | Learning rate | Best `val_nrmse` (`val_select`) | Still falling by the plan's rule? | Direct on `val_report` |
|---|---|---|---|---|---|
| 1-3 | `lr1e4` | 1e-4 | 0.1333 at step 25,000 | no | 0.1138 |
| 2-2 | `lr1e4` | 1e-4 | 0.0434 at step 27,500 (0.0435 at the last validation) | yes | 0.0400 |
| 2-3 | `lr1e4` | 1e-4 | 0.0883 at step 27,500 | yes | 0.0813 |
| 3-2 | base | 3e-4 | 0.2250 at step 30,000 | yes | 0.2309 |
| 3-3 | base | 3e-4 | 0.2086 at step 30,000 | yes | 0.2194 |
| 4-2 | `h050b256` | 3e-4 | 0.1645 at step 30,000 | yes | 0.1735 |
| 4-3 | `h050b256` | 3e-4 | 0.0930 at step 22,500 | no | 0.0836 |

Five of the seven chosen runs meet the rule (tasks 2-2, 2-3, 3-2, 3-3 and 4-2).
Inferred, not measured: on those five tasks the direct model's score is an
upper bound on the NRMSE that the configuration reaches at a longer budget,
not its best. The ground for the inference is that every extension of a run
that met this rule lowered the NRMSE: the `s90k` runs of tasks 1-1 and 2-1
(`val_select` 0.1430 to 0.1315 and 0.1525 to 0.1377; the base runs of both are
described under "What the validation curves show" above, 1-1 lowest at the last
validation and 2-1 at the second-last), of task 4-5 (0.6903 to 0.6812, same
passage, the base run lowest at the last validation) and of task 4-3 (0.1142 to
0.1000, below; the base run is lowest at the 11th of 12). Four runs are not a
measure of how often it holds: on task 2-2 the last three validations span
0.0001 at four decimals and on tasks 2-3 and 4-2 the last two differ by 0.0001
or less (above). The base run of task 2-1 was flat in the same way (0.1525 at
steps 27,500 and 30,000 to four decimals) and its extension lowered the
NRMSE; the extension of the flat 2-2 run (`lr1e4s90k`) is under way and has no
number yet.

The plan ran, per task: `lr1e4` alone on 2-2 (the same output signals as task
1-2, where 3e-4 was unstable and 1e-4 was measured to be better); base and
`lr1e4` on 1-3 and 2-3 (no stage-0 task predicts the flux map, and the lower
rate had helped one stage-0 task and hurt two); base alone on 3-2 and 3-3; and
base and `h050b256` on 4-2 and 4-3 (batch 256 at 50 ms of history was the
largest gain on task 4-4). None was tuned. The variant tables below are as printed. A "-" under
"differs from base" means different things in different rows. On the base row it says there is nothing to differ from. On a variant that has a
base run to compare with, it would say the `config.json` is the same as the
base run's (the case of the `b256` run of task 1-1, discussed under "Runs
beyond the plan's list"). On task 2-2, which has no base run, it says only
that there is no base run to compare with, and `lr1e4` does differ from the
base configuration (its learning rate is 1e-4). After the re-print that adds
`lr1e4s90k` to task 2-2's table, that row will also show "-", for this last
reason and not because it equals anything.

Task 1-3:

| variant | differs from base | best val_nrmse (val_select) | val_report | in the table |
|---|---|---|---|---|
| base | - | 0.1348 | 0.1149 | no |
| lr1e4 | lr=0.0001 | 0.1333 | 0.1138 | yes |

Task 2-2:

| variant | differs from base | best val_nrmse (val_select) | val_report | in the table |
|---|---|---|---|---|
| lr1e4 | - | 0.0434 | 0.0400 | yes |

Task 2-3:

| variant | differs from base | best val_nrmse (val_select) | val_report | in the table |
|---|---|---|---|---|
| base | - | 0.0887 | 0.0796 | no |
| lr1e4 | lr=0.0001 | 0.0883 | 0.0813 | yes |

Task 3-2:

| variant | differs from base | best val_nrmse (val_select) | val_report | in the table |
|---|---|---|---|---|
| base | - | 0.2250 | 0.2309 | yes |

Task 3-3:

| variant | differs from base | best val_nrmse (val_select) | val_report | in the table |
|---|---|---|---|---|
| base | - | 0.2086 | 0.2194 | yes |

Task 4-2:

| variant | differs from base | best val_nrmse (val_select) | val_report | in the table |
|---|---|---|---|---|
| base | - | 0.1873 | 0.2004 | no |
| h050b256 | batch_size=256, history_s=0.05 | 0.1645 | 0.1735 | yes |

Task 4-3:

| variant | differs from base | best val_nrmse (val_select) | val_report | in the table |
|---|---|---|---|---|
| base | - | 0.1142 | 0.1071 | no |
| h050b256 | batch_size=256, history_s=0.05 | 0.0930 | 0.0836 | yes |
| s90k | max_steps=90000 | 0.1000 | 0.0907 | no |

On task 2-3 the base run has the lower `val_report` score (0.0796 against
0.0813) and `lr1e4` the lower `val_select` error (0.0883 against 0.0887), so the
table's choice is the one with the higher report score, by 0.0017 and within
the gap between the selection shots and the report shots that this page
measures for other tasks. On task 1-3 the two variants are 0.0015 apart on
`val_select` and 0.0011 on `val_report`.

### Verdicts per task

Measured, in the table printed at 05:24, direct model on `val_report` against
the best published score on the test split (the table's interval rule, and
the split rule beside it):

- *Ahead:* task 3-2, 0.2309 against GenToka's 0.2522 (stands), and task 4-2,
  0.1735 against the TokaMind paper's 0.2140 (stands).
- *Tie:* tasks 1-3 (0.1138 against 0.1156), 2-3 (0.0813 against 0.0828) and 3-3
  (0.2194 against 0.2147), all against GenToka.
- *Behind:* task 2-2, 0.0400 against GenToka's 0.0370 (the split rule gives
  undetermined), and task 4-3, 0.0836 against GenToka's 0.0691 (undetermined,
  and "not competitive").

Across the 14 rows of the full table (the seven above and the seven stage-0
rows of the 23:54 print, which the 05:24 print repeats) the verdicts, of the
direct model on `val_report` against the best published score on test, are:
ahead 5 (tasks 1-2, 3-2, 4-1, 4-2 and 4-4; 4-4 is undetermined by the split
rule), tie 5 (tasks 1-3, 2-3, 3-1, 3-3 and 4-5) and behind 4 (tasks 1-1, 2-1,
2-2 and 4-3). One row of the 14 is "not competitive" (4-3) and none is stale
or unfinished.

**How firm the verdicts are.** Measured 2026-11-23. The intervals of the table
are over shots, and the split rule compares the margin with the difference
between the `val_report` and test shots. Neither includes the difference between two
trainings of one configuration. The one measurement of that difference is the
repeat of task 1-1 below: two runs of the same configuration and seed differ
by +0.0041 (95% interval +0.0024 to +0.0058, over shots). It is one pair of
runs on one task, a sample and not a variance, and whether it scales with the
score was not measured; the scores of tasks 1-2 and 2-2 (0.0362 and 0.0400)
are 0.27 and 0.29 of the 1-1 base run's 0.1361, so on them 0.0041 is a guide to
what to read with, not a measured value. With that figure in hand, the
14 rows read as follows (every margin is the distance from the nearer end of
the row's interval to the published score, in the print of 05:24):

- Two verdicts have a margin under 0.0041. Task 2-2, behind: 0.0387 - 0.0370
  = 0.0017. Task 1-2, ahead: 0.0412 - 0.0375 = 0.0037 (the row's interval is
  0.0350 to 0.0375, the published score GenToka's 0.0412).
- Two "stands" verdicts clear the split rule by less than 0.0041. Task 3-2: the
  margin is 0.0108 (0.2522 - 0.2414) against the mean baseline's difference
  between the splits of 0.0100 (0.6297 - 0.6197), a clearance of 0.0008. Task
  1-2: the margin 0.0037 against 0.0021 (1.0005 - 0.9984), a clearance of
  0.0016. The other two "stands" aheads clear it by more: task 4-1 by 0.0046
  (0.0199 - 0.0153) and task 4-2 by 0.0065 (0.0218 - 0.0153, where 0.0218 =
  0.2140 - 0.1922).
- Two variant choices are closer than that on `val_select`. Task 1-3: 0.1348
  against 0.1333, 0.0015 apart. Task 2-3: 0.0887 against 0.0883, 0.0004 apart;
  there the other variant has the lower `val_report` score (0.0796 against the
  table's 0.0813), so the printed score of 2-3 depends on a choice that is
  smaller than the repeat difference.

The split direction behind the two new "ahead (stands)" verdicts is a second
reading, and it works for the model. On both tasks `val_report` is the easier
split (the mean baseline is lower on it: 0.6197 against 0.6297 on 3-2 and
0.4961 against 0.5114 on 4-2), which is the direction that favours the direct
model, since its score is on `val_report` and the published scores are on test.
The margin passes the split rule, by a factor of 1.08 on 3-2 (0.0108 against
0.0100) and of 1.42 on 4-2 (0.0218 against 0.0153). Persistence differs by
0.0117 on 3-2 (0.3651 against 0.3534) and by 0.0269 on 4-2 (0.3247 against
0.2978): more than the margin on each task (by 0.0009 and 0.0051), which the
rule does not use. This is the same reading as for the "ahead" on task 4-1 in
stage 0, and it is the counterpart of the 4-3 mark above, where the split
works against the model.

This qualifies the limit "One seed" under "Limits that apply to the whole
page", which says that run-to-run noise is measured in stage 2, not here. Stage
2 still has to measure it, and this page gives no variance. But one pair of
runs was measured here, and the verdicts above are to be read with its 0.0041
beside them.

### Runs beyond the plan's list

Each comparison here is a paired bootstrap over shots (difference of the
per-shot scores, seed 0, 10,000 resamples, `val_report`, 95% intervals). The
script is the controller's scratch file `tmp/stage1a/paired_bootstrap.py`,
not tracked in git.

- **Task 4-3, base extended to 90,000 steps (`s90k`).** The plan's rule
  extends a chosen run that is behind and still falling; the base run of 4-3
  was the chosen variant when the extension was started (2026-11-22, 23:51 UTC),
  and `h050b256` was still training. Its `val_report` score is 0.0907 over 510
  shots, with the best checkpoint at step 87,500 and a `val_select` error of
  0.1000. Against `h050b256` (0.0836) the paired difference, `s90k` minus `h050b256`, is +0.0071 (95%
  interval +0.0044 to +0.0099; `s90k` is the lower one on 191 of 510 shots), and
  against the base run (0.1071) it is -0.0164 (-0.0205 to -0.0123). Against the
  base run `h050b256` is lower by 0.0235 (0.1071 - 0.0836) and `s90k` by 0.0164
  (0.1071 - 0.0907), so `h050b256` is 0.0071 lower than `s90k` (0.0907 -
  0.0836). The two runs differ in more than one thing, and this comparison does
  not separate them. `h050b256` ran 30,000 steps at batch 256, which is 7.68
  million windows (30,000 x 256). `s90k` ran 90,000 steps at batch 64 (the base
  `config.json` of task 4-3 has `batch_size` 64), which is 5.76 million windows
  (90,000 x 64). History (50 ms against 150 ms), batch and windows seen are not
  separated between the two runs; stage 0 made the same caveat for task 4-4,
  where the runs that differ only in the batch are `h050` and `h050b256` (see
  "What the validation curves show"). The difference +0.0071 (+0.0044 to
  +0.0099) is a difference of two single runs, 1.7 times the repeat difference
  of task 1-1 below (+0.0041) and with a lower interval end (+0.0044) of the same
  size, so the repeat is a reason not to read more into it than that
  `h050b256` came out lower in these two runs. `h050b256` stays the table's
  variant, because its `val_select` error (0.0930) is the lowest.
- **Task 1-1, a second run of the base configuration (`b256`).** The run was
  started on 2026-11-23 (00:13 UTC) as a test of batch 256, on the premise,
  wrong as shown below, that the base run of task 1-1 had batch 64, with the aim
  of closing the gap to GenToka (0.0941) on task 1-1, and it ran 6,901 s on one
  GPU. **Its
  `config.json` is the same as the base run's in every option** (batch 256,
  base history, learning rate 3e-4, 30,000 steps, seed 0; the base run of
  task 1-1 already has batch 256, as the configuration above says), and the
  hashes of the predictor and the scoring code are the same in the two
  `trained_by.json` files; only the recorded git revision differs. So it is a repeat of the base run and not a variant of
  it (the variant table of task 1-1 lists it with "-" under "differs from
  base"), and it says nothing about batch size. It scores 0.1402 on
  `val_report` over 520 shots (`val_select` 0.1483), against 0.1361 for the base
  run (0.1430). The paired difference repeat minus base is +0.0041 (95% interval
  +0.0024 to +0.0058), and against `s90k` (0.1235) it is +0.0167 (+0.0143 to
  +0.0194). So two runs of one configuration, with the same seed, differ by
  0.0041 on this task, with an interval that excludes zero: that is the size of
  run-to-run difference seen here, one pair of runs on one task. What makes the
  two runs differ (loader order, kernel nondeterminism, GPU sharing) was not
  separated. It does not change the table: the row of task 1-1 stays `s90k`
  (0.1235), the variant with the lowest `val_select` error (0.1315).
- **Task 2-2, `lr1e4` extended to 90,000 steps (`lr1e4s90k`).** The chosen run of
  task 2-2 meets the still-falling rule and is behind (undetermined by the split rule), so the
  plan's extension rule applies to it, with the run's own learning rate of
  1e-4 (the plan's `s90k` would have run the base configuration, which task 2-2
  never trained). It started on 2026-11-23 at 05:28 UTC on one GPU and
  is expected to take about 5 hours. **It is running, and no number of it is
  given here.** The table is printed again when it ends, and the row of task
  2-2 is taken from it only if its `val_select` error is the lowest.

### The projected 14-task mean

Printed 2026-11-23 05:24 UTC, as printed:

```text
projected mean over the 14 tasks (direct model on all 14 tasks): 0.1899; the bar is 0.183, matching the best published score everywhere would give 0.194; 1 row(s) have a direct score that is stale, unfinished or not competitive (see the marks above)
```

The projected mean is **0.1899**, on `val_report` shots, direct model with no
pretraining; the published scores it is compared with are on test shots. It is
the mean of the direct model's `val_report` score over all 14 tasks, each with
the variant the table chose. The one row the line counts is task 4-3 (not
competitive). In the print of 23:54 above the mean was 0.1929, with the direct
model on the seven stage-0 tasks and the best published score on the other
seven. No stage-0 row changed between the two prints; the variant table of
task 1-1 has one more run (`b256`, above), which is not the table's choice. The
answer to the bar's question is unchanged: the projected mean, 0.1899, is above
the bar of 0.183. By the rule of the design the bar goes to the owner with the
table; it is not changed on this page.

### What it does not say

The direct model is one architecture (about 14 to 18 million parameters, one
squared loss), trained on one task at a time, one seed per run. Its budget is
30,000 steps for 12 of the 14 chosen rows; the rows of tasks 1-1 and 2-1 are
90,000-step `s90k` runs, which the 14-task mean includes. Of the seven new
tasks, the learning rate was chosen from two values on 1-3 and 2-3 only. The
batch size of the chosen run, read from each `config.json` on 2026-11-23, is 256
on 13 of the 14 rows and 64 on task 3-3 (see "The variant of each task and its
learning rate" for the seven new rows). A score of it behind the best
published one on a task is not evidence that the task has no headroom: it shows
a limit of this model at this budget, not of the data. A score of it ahead of
the best published one is evidence of headroom only as far as its margin
survives the split rule and the run-to-run difference ("How firm the verdicts
are", above). Each score is an upper bound on the NRMSE that this model reaches
on its task: a longer, larger or differently trained model may reach a lower
one. Five of the seven chosen runs (tasks 2-2, 2-3, 3-2, 3-3 and 4-2) meet the
plan's rule for "still falling" (lowest validation at the last or second-last
of 12), though on 2-2, 2-3 and 4-2 the last validations are flat. Inferred, not
measured: on those five tasks the score is not the best this configuration
reaches, because every extension of a run that met this rule lowered the NRMSE
(tasks 1-1, 2-1, 4-5 and 4-3), which is four runs, so the reading is a
guess about trend and not a measured value for these five. The scores are on
`val_report` shots and the published scores are on test shots; the verdict is read with the split rule beside it, and two
of the seven new verdicts (tasks 2-2 and 4-3) are undetermined by it. A cell
that is missing is "not scored", never 0. The scores of this section say
nothing about IGNITE: stages 1b and 1c cite them as the direct model's score
per task, and a prediction or round-trip instrument of the direct model is
used for a task only if its row is neither marked nor "not competitive". Of the
seven rows above, six qualify on that rule; for task 4-3 the instrument is not
available.
