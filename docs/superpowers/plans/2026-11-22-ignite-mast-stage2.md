# IGNITE on MAST, stage 2 ("Dynamics and first table"): index of the three implementation plans

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> Written 2026-11-22 by the plan's author. Plan 2a was approved by the owner on 2026-11-23 (17:13 UTC, "approve"). Plans 2b and 2c are scoped here and are written, and put to the owner, after 2a's review.

Only plan 2a is written (`2026-11-22-ignite-mast-stage2a-first-run.md`). Plans 2b and 2c are scoped here and written after 2a's review. Nothing of stage 2 has been implemented or run. Plan 2a was written against the text of plan 1b, whose Task 1 was about to be dispatched on 2026-11-22; none of 1b's code existed then. Every number below carries its date and source and says whether it was measured or inferred.

**Spec:** `docs/models/ignite-mast-design.md`, row "2. Dynamics and first table" of "Order of work" (target 2026-12-02): "pretraining at three sizes; adaptation per task; the step interface and the causality test; the first complete table". The token-path check and the pretraining check under that table also belong to stage 2.

## What stage 2 delivers

- An IGNITE dynamics model trained on MAST tokens under actuators. It is a masked-token transformer that alternates attention within a 5 ms frame with causal attention across frames, over stage 1b's frozen tokenizers. It is scored through the stage-0 harness on TokaMark tasks. This is the first deliverable, and plan 2a reaches it after its Task 4 (training) and Task 5 (scoring).
- The token-path check on tasks 1-1 and 1-2, decided by a rule fixed before any number exists (2a, rule 3).
- A simulator that steps frame by frame in physical units under an actuator trajectory, with a causality test (2b).
- Pretraining at three sizes, adaptation per task, the from-scratch column, the pretraining check, run-to-run noise, and the first complete 14-task table (2c).

| Order | Plan | What it delivers | Tasks | Needs |
|---|---|---|---|---|
| 1 | 2a, first run (written, 89 kB) | layout, token store and its check against the benchmark's windows, the dynamics model with nine switches, conditioning patterns, the trainer, the mean roll-out by recomputation, the harness predictor, the first run (size S), the token-path check | 7 | 1b's code (Tasks 1 to 8), its fit (K1) and an assignment (`v1` after K2 and K3, or the provisional `v0`) |
| 2 | 2b, steps and causality (not written) | the cached step, sampling, the simulator in physical units, the causality test | about 4 (estimate) | 2a's Tasks 1 to 5 |
| 3 | 2c, sizes and first table (not written) | three sizes, adaptation, from-scratch, pretraining check, noise, the first complete table | about 5 (estimate) | 2a with a `pass` (or the owner's word on `marginal`), 2b's cached step, 1b's gate (K7) |

2b can be built while 2a's first run and the token-path runs train. 2c's large runs start only after the token-path verdict.

## Plan 2b: steps, sampling, causality

2b turns the trained network into a simulator a controller could call. It has four parts:

- **The cached step.** Keys and values of the attention across frames are cached per row, so each new frame costs one pass over that frame. It must reproduce 2a's `mean_rollout`, which recomputes the whole history at every step, to 1e-4 of each value's bin range on every value. Without it the 20 steps of a group-4 window cost about 20 times as much as one, which is too slow for the test split at size M or L (about 300 to 400 windows per test shot and 1,110 test shots, design "Cost").
- **Sampling.** The ordered head, told the earlier values of a token, gives one sample per value. 2b adds the sampled roll-out (one sample, and the mean of several), the design's second read-out.
- **The simulator in physical units.** The interface is `reset(seed frames, roles)` and `step(actuator values for the next 5 ms) -> next frame of every modelled signal in physical units, as a mean or a sample`. Which channels are actuators is configuration, as in the design's "Control readiness", item 1. Seeds of up to 100 frames are accepted.
- **The causality test.** Changing an actuator after frame h changes no output at or before frame h. This is tested exactly (equal bytes) on the cached step and on the harness predictor. A check also confirms that no input is scaled with a statistic of its own shot.

2b also measures wall time per simulated frame (item 5 of "Control readiness") for one roll-out and for a batch, with both read-outs. Its rules are pass or fail, without thresholds to tune: equality to tolerance for the cached step, exact equality for causality.

## Plan 2c: sizes, adaptation, first table

**Sizes.** The design asks for about 30M, 100M and 300M parameters. 2a defines S (width 384, depth 12), M (640, 16) and L (1024, 18), whose blocks hold 28.3M, 104.9M and 302.0M parameters (inferred from 16·d² per block). Each size's budget comes from 2a's job J3, which measures token positions per second and GPU memory per 1,000 positions. Until J3 runs, the author's inference is 0.5, 0.13 and 0.045 million positions per second, at 600,000, 270,000 and 150,000 positions per step. At 20,000 steps that is about 7 h, 12 h and 19 h on one GPU. The size is chosen on validation loss and on validation NRMSE (normalized root-mean-square error) of the six probe tasks, not by building three tables.

**Adaptation.** Each task gets one adapted copy: the pretrained model is trained further on the task's conditioning pattern with the token objective and no new output network, and selected on `val_select` NRMSE of the mean roll-out (2a's `--select nrmse`). The unadapted model is scored on all 14 tasks as the one-set-of-weights column.

**From scratch.** The same network, tokenizers and recipe from random weights, on the six probe tasks (1-1, 1-2, 2-1, 3-1, 4-1, 4-4). 2a's token-path runs are already two of these cells at size S.

**The pretraining check.** It is stated in 2c before any number. The author's proposed form: pretraining and from-scratch "agree" when, on at least five of the six probe tasks, the 95% paired-bootstrap interval over `val_report` shots of the ratio (adapted over from-scratch) contains 1.0. Agreement sends GPU time to data (the FAIR-MAST shots) and to the token path, not to the largest size.

**Noise.** Three seeds on the probe tasks, for our model and for the TokaMind replication; a gap inside that noise is a tie. A win over GenToka counts when the upper end of our bootstrap interval is below its published score.

**The first complete table.** The headline (adapted, mean roll-out) and the one-set-of-weights column on all 14 tasks at one size. The sample columns and from-scratch on the six probe tasks. The test split is read once, here. If time runs short, columns are dropped in the design's order: full context, the regression read-out, then from-scratch beyond the probe tasks. Task 4-5 is scored through 1b's band transform, which is stage 1's fluctuation path; it decodes to linear power, as the design asks.

## Dependencies on stage 1

**On plan 1b (required).**

- 2a's Tasks 1 and 3 need no 1b code; they test against a stand-in codec.
- 2a's Task 2 needs 1b's `assign` and `fit` and the three codecs (1b's Tasks 1 to 5 and 7).
- 2a's Task 5 needs `floor.blind_to_targets` and, in `run_tokenizer.py`, `score_one`, `provenance_of` and `split_label` (1b's Tasks 6 and 8).
- Any store needs 1b's fit (Task 8, job K1). The first run's store needs the assignment `v1` (Task 9, jobs K2 and K3). The token-path runs need K4's choice between `v1` and `v1r`.
- The token-path report's diagnosis numbers need K5 (the tokenizer floor on `val_report`) and K6 (the cost of tokenized inputs), or K4's `unquantized` pair.
- 2c's size-M and size-L runs need 1b's gate (Task 10, job K7).

2a consumes from 1b, by name: `Codec` (`encode` to `(N, F, n_tok, n_val)`, `decode`, `grids`, `domain`); `BandDCT`'s token order `probe * n_bands + band`; `assign.Setting`, `load_assignment`, `save_assignment`, `assignment_sha256`, `codec_kind`; `fit.load_fit`, `fit_sha256`, `build_codecs`; `frames.FRAME_S`, `frame_index`; `GridSet`'s asinh centres and 64 bins; `tokenizers.code_sha256`; `paths.stage1_dir`; the results tree `<stage1>/scores/<task>/<predictor>/<label>/` with the names `tok.<a>.<mode>`, `inputs.<a>.<mode>.<v>` and `raw.<v>`. If 1b's review changes one of these, the matching lines of 2a change with it.

**On plan 1c (not required).** Stage 2 does not wait for 1c. 1c's assignments `v2` and `v2c` enter as another assignment file: a new store (2a's J1 and J2) and new runs, since a checkpoint refuses a store of another layout. Recommendation: a learned tokenizer replaces `v1` in stage 2 only if 1c's rules choose it before 2c's size runs start, and only after the token-path runs (J6 to J8) have been repeated on it; otherwise it waits for stage 3.

## What 2b and 2c need 2a to keep

- `Layout.sha256()` is the identity of a tokenizer set, carried by the store, every checkpoint and every predictor.
- `Dynamics.forward(history, query, rows, *, told=None)` with `Frames(values, state, t)`, the five token states, and `t` in seconds relative to the cut. 2b's cached step is tested against `rollout.mean_rollout`.
- `simulator_predictor(..., history_s=None, tokenizers=None)`, `train(..., select_fn=None)` and `TrainConfig.init` for adaptation from a pretrained checkpoint.
- The predictor names `sim.<run>.<checkpoint>.mean` (`.all` for tracking all signals). 2b adds `.sample1` and `.sampleN`.
- `checks.ratio_ci` and `checks.per_shot`, reused by the pretraining check and the noise estimates.

## Open decisions for the owner, across stage 2

1. **(2a) A provisional assignment `v0`.** Recommendation: build it and its store as soon as 1b's K1 and 2a's Task 2 exist, for the store check and the throughput job. Start the first run on `v0` only if `v1` is still more than about 2 h away when 2a's Task 4 is merged. A run on `v0` is a path check and fills no table.
2. **(2a) How the token-path check is read.** Recommendation: a size-S simulator from random weights on the one task, with the same steps, batch, learning rate and selection (200 `val_select` shots) as the direct variant the headroom table chooses. That is `s90k` on 1-1 (0.1235 on `val_report`) and `lr1e4` on 1-2 (0.0362), measured 2026-11-22. A `marginal` verdict goes to the owner before any M or L run.
3. **(all) The bar of 0.183.** With direct-model probes on 13 of 14 tasks (task 2-2 at its published 0.0370), the projected 14-task mean is 0.1913, against 0.1939 for the best published score per task and 0.2033 for GenToka (2026-11-22, `val_report`). Recommendation: keep 0.183 and decide at the token-path check, the first number that prices the token path.
4. **(2c) The budget of the three sizes.** Recommendation: equal wall-clock, the three runs at once on three GPUs with steps set from J3 so that each ends within about 24 h, rather than equal token positions. With about 1.1 million training frames, the design already expects the data to limit before the parameters do, and the dates leave one wave of size runs.
5. **(2c) The history used on group 4.** The direct probes with 50 ms of history and batch 256 beat the 150 ms base on 4-1, 4-2 and 4-4 (2026-11-22). Recommendation: adaptation may choose between 50 and 150 ms of history per group-4 task on `val_select`. Using less of the window than the task offers stays inside the strict protocol. The owner confirms that reading.
6. **(2c) Learned tokenizers in stage 2.** Recommendation as under "On plan 1c" above.

## Departures from the design (all in 2a; 2b and 2c add theirs when written)

- Attention uses masks through torch's `scaled_dot_product_attention`, without the FlashAttention-4 dispatch, which takes no mask. Measured 2026-11-21: as fast or faster at 512 tokens and 16 frames; at 80 frames, 2.63 ms against 2.03 ms.
- A fifth token state, carried: attended to, with no value and no loss. It keeps a signal that is observed before the cut and not predicted after it inside the attention across frames.
- The mean read-out reads each value without the earlier values of its token. This departs in detail from change 4; the head is trained both ways, and sampling uses the told form.
- Training clips hold at most 64 frames (44 before the cut, 20 after) rather than a context of up to 100. No task needs more than 30 before and 20 after, and rotary time encoding lets inference use 100.
- The identity of a token is a learned embedding per layout row, with no network over registry metadata. This departs in part from change 1: MAST has one device and no magnetics positions.
- The provisional assignment `v0`, which the design does not have.
- A difference the design does not foresee: spectrogram frames in the store follow one cut per phase, so training sees five column offsets per shot, while scoring sees one per window.
- A reading, not a departure: the token-path check's "small simulator" is size S trained from scratch on the one task with the reference's budget, as in decision 2.

## Size of the plans

2a is 89 kB against the target of about 70 kB. Of its bytes, 39% are the Interfaces blocks (signatures, with the rule each one obeys), 16% are test names with their assertions (113 tests in 7 tasks), and 27% are the argument at the top (decisions, measured facts, departures, rules). Its prose was cut and its commit lines were shortened. Going further means removing interface detail an implementer needs or moving a task to another plan, which is a decision for the review.
