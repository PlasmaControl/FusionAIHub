---
title: "IGNITE on MAST: stage 0 results"
---

# IGNITE on MAST: stage 0 results (2026-11-22)

**Status: stage 0 is built and measured, except the direct supervised model.**
The scoring harness reproduces the published baselines, the standardized cache
exists, TokaMind's representation floor and two cheap predictors are scored on
the seven stage-0 tasks, and our replication of TokaMind is complete. The direct
model is training on four of the seven tasks and is not yet trained on the
other three, so its rows of the headroom table, the verdicts and the projected
mean are empty. They are written into this page when the trainings have
finished. The plan is [the design](./ignite-mast-design.md); this page records
what was run, with the numbers read from the stored score files. All times are
UTC on 2026-11-22, measured on [node13](../clusters/nvidia.md).

The seven stage-0 tasks are the six probe tasks (1-1, 1-2, 2-1, 3-1, 4-1, 4-4)
and task 4-5. Every score is the benchmark's task NRMSE (lower is better),
written by TokaMark's own evaluator (`tokamark.evaluator.compute_metrics`) and
never recomputed here.

## What was run

| What | When | Where it is |
|---|---|---|
| Scoring harness, checked against the published `mean` and `persistence` columns | 02:45 to 03:40 | `scripts/mast/check_harness.py` |
| Standardized cache v1 | built 03:46 to 03:50 | `/models/intermediate/mast/cache/v1/` |
| TokaMind representation floor, on `test` and `val_report` | 03:38 to 05:16 | `scripts/mast/run_headroom.py --predictor floor` |
| Two cheap predictors (`shrunk`, `ridge`), on `val_report` | 04:48 to 05:12 (`shrunk`, seven tasks), 04:54 to 04:57 (`ridge`, four tasks so far) | `--predictor shrunk`, `--predictor ridge` |
| Our TokaMind replication, all 14 tasks | finished 04:36 | `/models/tokamind/run_all_tasks.sh` |
| Direct supervised model, one per task | training; tasks 3-1, 4-1, 4-4 and 4-5 started at 05:08 | `--predictor direct --train` |
| Headroom table | 05:19 | `scripts/mast/run_headroom.py --table` |

Scores are stored as
`/models/intermediate/mast/stage0/scores/<task>/<predictor>/<split>/score.json`,
beside the evaluator's own per-shot and per-signal files. Each record carries
the hash of the scoring code (`harness.py`, `tasks.py`, `baselines.py`) and, for
a headroom predictor, the hash of the predictor and of the files it was
computed from (a fit, a checkpoint, TokaMind's coefficient indices). The `git_revision` in a record
is the HEAD of the checkout when the scoring code was loaded. It can predate
uncommitted code, so the hashes are the provenance and the revision is a hint.
All 77 scores stored when the table was printed (2026-11-22 05:19) carry the
same scoring-code hash, `173e2747...`.

Splits. Everything fitted (`shrunk`, `ridge`, the direct model) is scored on
`val_report`, the validation shots at odd positions of the sorted list; models
are selected on `val_select`, the other half. Only predictors with nothing
fitted (the mean, persistence, the floor) are scored on the test shots, which
are read once, for the final table. The published scores are on the test split.

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
TokaMark-standardized units and scored on `val_report`.

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
  output channel.

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

**Measured 2026-11-22**, in the headroom table below. Two findings, the second with an inferred reading marked as such.

- *TokaMark's mean baseline is one scalar per signal, and that is most of its
  error on task 1-2.* A per-channel training mean alone, which is what `shrunk`
  falls back to without an anchor, scores about 0.22 on `val_report` on task
  1-2 (`shrunk` 0.2204), against 1.0005 for the scalar mean. The error of the
  mean baseline on that task is the fixed shape of the boundary, not its motion.
  Ridge on frame means scores 0.0830 there, about twice the best published
  score (0.0412, test split, a different split).
- *How close the published models are to the cheap references.* Measured: on
  task 4-1 our replication (0.2147, test) is 0.07 below `shrunk` (0.2842,
  `val_report`; different splits). On task 4-5 the mean scores 0.6852 on
  `val_report` and 0.6533 on test; our replication (0.6499) and the best
  published score (0.6510) are 0.0034 and 0.0023 below the mean on test, and
  `shrunk` on `val_report` is 0.6925, 0.0073 above the mean on the same shots.
  Persistence on all windows scores 0.8011 there. **No paired test was run**,
  and one score on task 4-5 is known only to about +-0.06 over shots
  (bootstrap of `shrunk` on 351 `val_report` shots, measured 2026-11-22, 95%
  interval 0.634 to 0.756). Differences of 0.002 to 0.007 are therefore
  neither a win nor a tie: the page does not say that the published models
  beat the mean on task 4-5, nor that they do not.
  *Inferred, not measured:* that these small margins mean the tasks are near a
  noise floor, or are not predicted at all. Nothing on this page measures a
  noise floor.

Ridge scores for tasks 4-1, 4-4 and 4-5 were not stored when the table below
was printed, so no statement is made about them.

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
markovian tasks (3-1 from its `config.json`; 1-1, 1-2 and 2-1 from
`default_config`, their trainings have not started) and batch 64 for the
non-markovian tasks of group 4 (4-1, 4-4, 4-5), learning rate 3e-4
with 1,000 warm-up steps and cosine decay, weight decay 0.05, an exponential
moving average of the weights (0.999), seed 0. The training loss is in
TokaMark-standardized units (mean over signals of the signal's mean squared
error) and is not on the NRMSE scale. Every 2,500 steps the averaged weights
are scored on 200 shots evenly spaced over `val_select`; the lowest becomes
`best.pt`. The score in the table is `best.pt` on `val_report`.

The first version of the trainer validated on the first 200 shots of
`val_select` by shot number. Those are one old campaign (shots 11806 to
18744). **Measured 2026-11-22** on task 4-5, only 65 of them have windows and
`cc_field` is nearly constant there: the per-shot mean baseline scores 0.30 on
`val_report` shots of that range, 0.77 on later shots and 0.685 on all. Choosing
the best checkpoint on such shots would not have tracked the report split, so
the validation shots are now spread evenly over the sorted list and the
trainings were restarted after that change.

Status when this page was written (2026-11-22 05:20): the trainings of tasks
3-1, 4-1, 4-4 and 4-5, started at 05:08, were at steps 2,200, 2,000, 1,700 and
1,500 of 30,000 (2.6 to 3.4 steps per second, 145 to 820 windows per second;
the speed is bounded by TokaMark's loader, which leaves the GPUs mostly idle).
The trainings of tasks 1-1, 1-2 and 2-1 start later. No direct score is stored
yet.

Limits to carry with every direct number: the score is on validation shots
and the published scores it is compared with are on test shots; one seed per
task; one configuration at one training budget (a model that needs more steps
or a different size would score better); and the windows come from TokaMark's
loader. A training that stops before 30,000 steps is marked in the table
(‡), and a direct score worse than our TokaMind replication is marked "not
competitive", because a weak model is not a headroom finding. When more than
one variant is trained, the table takes the variant with the lowest best
`val_nrmse` on `val_select` and this page lists every variant, what differs
in its `config.json`, its best `val_nrmse` and its `val_report` score. Only the
base variant exists today. `--table` prints this listing for every task that
has a trained variant, including a variant whose selection error is the lowest
but which has no stored score yet (it cannot enter the table).

## The headroom table

Printed on 2026-11-22 05:19 UTC by

```bash
scripts/mast/py.sh scripts/mast/run_headroom.py --table
```

which reads the stored scores, scores nothing and writes
`/models/intermediate/mast/stage0/headroom_table.csv`. The table is the
Markdown the command prints, unchanged. Direct-model cells are empty today.

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
| 1-1 | 0.7004 | 0.3053 | missing | none | 0.1503 | 0.0941 | gentoka | missing | missing |
| 1-2 | 0.2204 | 0.0830 | missing | none | 0.0433 | 0.0412 | gentoka | missing | missing |
| 2-1 | 0.6374 | 0.3318 | missing | none | 0.1809 | 0.0880 | gentoka | missing | missing |
| 3-1 | 0.6771 | 0.4877 | missing | none | 0.3457 | 0.3311 | gentoka | missing | missing |
| 4-1 | 0.2842 | missing | missing | none | 0.2147 | 0.2122 | tokamind_rerun | missing | missing |
| 4-4 | 0.6758 | missing | missing | none | 0.3502 | 0.3116 | gentoka | missing | missing |
| 4-5 | 0.6925 | missing | missing | none | 0.6499 | 0.6510 | tokamind_paper | missing | missing |

missing: no score is stored. undefined: the benchmark defines no persistence for the task. † stale: stored by other code than the code now loaded. ‡ unfinished: the direct model's training stopped before `max_steps`. not competitive: the direct model's `val_report` score is worse than our TokaMind replication's test score on the task.

projected mean: not computed, 7 of 7 stage-0 tasks have no direct score

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
score of an unfinished training is shown with ‡. No cell carries either mark on
2026-11-22 05:19.

## What it says

Measured and inferred are kept apart. **Measured** statements are numbers in the
table above, with the limits given with them. **Inferred** statements are
stated as such.

**Question 1: per task, is the direct model ahead of, tied with or behind the
best published score?** Written when the trainings have finished: the
direct-model rows are empty today, and no statement about the direct model is
made from missing numbers. What the existing rows show is the distance that the
direct model has to cover. Measured: the cheapest references are far from the
best published scores on tasks 1-1, 1-2, 2-1, 3-1 and 4-4 (ridge on `val_report`:
0.3053, 0.0830, 0.3318, 0.4877 against 0.0941, 0.0412, 0.0880, 0.3311; `shrunk`
on task 4-4 0.6758 against 0.3116), and within about 0.07 on task 4-1 (`shrunk` 0.2842
against 0.2122) and 0.04 on task 4-5 (`shrunk` 0.6925, the mean 0.6852 on
`val_report`, against 0.6510). These compare validation scores with test scores.

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
which tasks carry it?** Written when the trainings have finished. The
projected mean is not computed today: 7 of the 7 stage-0 tasks have no direct
score. The spec sends the bar back to Nathaniel with this table before stage 2
is committed to it if the answer is no, and that decision is not made from this
page until the numbers exist. For scale, task 4-5 alone is 26% of GenToka's
summed score, and groups 1 and 2 together 16% ([the
design](./ignite-mast-design.md#the-bar)), which is where the largest
differences between the direct model and the best published score will move the
projected mean.

## Limits that apply to the whole page

- **Validation shots against published test scores.** Every fitted number is on
  `val_report`, every published number is on the test split, and the same
  predictor differs between the two by up to 0.04. Only differences larger than
  that are readable without the baselines of the table.
- **One seed.** The direct model is one run per task, and run-to-run noise is
  measured in stage 2, not here. The interval of the direct model is over
  shots, not over seeds.
- **The direct model's training budget.** One configuration, 30,000 steps. The
  headroom it reports is a lower bound on what a specialized model can reach.
  On tasks 4-1, 4-4 and 4-5 the batch is 64, so 30,000 steps see a quarter of
  the windows that 30,000 steps of batch 256 would (`train_log.csv`, measured
  2026-11-22: 135 to 185 windows per second on group 4, 812 to 865 on task 3-1).
- **No data is not a negative.** A "missing" cell is a score nobody has stored
  yet. It is not a bad score and it is not 0.
- **The cheap predictors are cheap.** They say what trivial means predict, not
  what a good model can.
- **The floor is TokaMind's.** It describes TokaMind's coefficient choice on
  its own fine-tune configuration, not a property of the task.
