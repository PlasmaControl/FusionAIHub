# IGNITE on MAST, stage 2a: token store, dynamics model, first training run, token-path check. Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> Approved by the owner on 2026-11-23 (17:13 UTC, "approve"). The open decisions for the owner are taken as the author recommends until the owner says otherwise. The status block below lists what changed between drafting and approval; the controller's pre-flight rulings in the plan's ledger apply it before any task is dispatched.

Nothing here has been run except the small read-only looks under "Measured facts". This is the first of three stage-2 plans; the index is `2026-11-22-ignite-mast-stage2.md`. It stands alone: it needs stage 1b's tokenizer code and one assignment, and nothing of plans 2b and 2c. It was written against the text of plan 1b (`docs/superpowers/plans/2026-11-22-ignite-mast-stage1b-fixed-tokenizers.md`), none of whose code existed on 2026-11-22.

> **Status since drafting (2026-11-23, 08:35 UTC).** The rest of this draft is unchanged. When it is approved, the author folds in these facts before any task is dispatched:
>
> - **Stage 1b and 1c are done and merged on `nvidia`.** K4 kept `v1`: no `v1r` (none of tasks 1-1, 1-2, 2-1 and 4-1 adds more than 0.2 of the best published score; 1b ledger, 03:22 UTC). Ruling C29 then replaced `v1` with **`v1b`** (`$S1/assignments/v1b.json`). `v1b` is `v1` with the currents' K raised. It uses the same fit (`$S1/fit/v1`) and passed the gate at 07:40 UTC (57 rows, 0 block). Only task 2-1's current rows changed. On task 2-1 `val_report` (520 shots), the solenoid current's input cost over raw went from +16.15% with `v1` to -0.34% with `v1b`, and the task mean from +0.17% to -0.01%.
> - **Open decision 1 is moot.** An assignment exists, so there is no provisional `v0`. The author's recommendation is to drop Task 2's `provisional` command, its test and job J0, and to start the first run on `v1b`.
> - **Stage 1c's verdict.** No network tokenizer beat its linear counterpart: 4 collapsed and 9 lost, again at lr 1e-4. Rule 4 replaces nothing, so `v2 = v1b`. The concatenated control `v2c` is `v1b` with PCA tokens on the four magnetics arrays (`ccbv`, `obr`, `obv`, `flux`). It clearly beat the per-signal tokens on both splits. 1c's open decision 1 is taken at its default: the smallest pretraining runs on both `v1b` and `v2c`, and tasks 1-1 and 1-2 decide between them. This adds a paired J4 and J6 on `v2c` (another J1 store build, 10 to 40 minutes, inferred). On `val_report`, `v2c`'s ranged input cost is +0.08% on 1-1, +0.11% on 1-2 and +0.01% on 2-1, and -0.63% on 4-1, the same as `v1b` (1c ledger, C43 result).
> - **Open decision 3's numbers moved.** `--table-all` of 10:36 UTC projects 0.1897 over the 14 tasks, with the direct model on all 14 (0.1899 at 05:24). Task 2-2's last variant (`lr1e4s90k`, 90,000 steps) finished at 10:36 with 0.0371 [0.0359, 0.0385] on `val_report`, against GenToka's 0.0370. That moves 2-2 from behind to tie, and the verdicts are now 5 ahead, 6 tied and 3 behind (1-1, 2-1, 4-3). Recorded on the stage-0 page in e0f87f2c, 84f89b8e and 4f8312e7.
> - **GPU training does not always repeat at a fixed seed.** Stage 1c's long runs matched their seed-0 runs exactly for 5 of 10 networks. The band, flux-map and profile networks differed from the first logged step, by up to 0.066 in val_mse. The cause is not established (1c ledger, correction at 08:40). This draft does not rely on repeatability. Any comparison of two stage-2 runs therefore needs a measured spread, not an assumed zero.
> - **Commit trailer.** It is `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`, not Fable.

**Goal:** A small IGNITE dynamics model is trained on MAST tokens under actuators, rolled out frame by frame and scored through the stage-0 benchmark harness, after the fewest tasks that can carry that; and the design's token-path check on tasks 1-1 and 1-2 is decided by a rule written here before any number exists.

**Architecture:** A new package `ignite/mast/dynamics/`. A `Layout` turns stage 1b's frozen assignment into rows (one token position per row, with its values). A token store holds every training and validation shot as packed token values per 5 ms frame, at the five frame phases that the benchmark's cut times can have. The dynamics model is IGNITE's masked-token transformer (attention within a frame, causal attention across frames for each row) rebuilt with attention masks, a state per token and ordered bins. A trainer draws clips from the store under a conditioning pattern. A predictor encodes a benchmark window with the same tokenizers, rolls the model forward on expected values and decodes, so that `harness.score` can score it.

**Tech Stack:** Python 3, numpy, h5py, pandas, torch 2.15 (`torch.nn.functional.scaled_dot_product_attention`, bfloat16 autocast), pytest; stage 1b's `ignite/mast/tokenizers/`; TokaMark's loader and evaluator only through `harness.score`.

**Spec:** `docs/models/ignite-mast-design.md`: "Dynamics model" (what is kept, the nine changes, "Size"), "Three ways to read the simulator", "TokaMark adaptation and scoring" (strict protocol, inputs at full precision, scoring through the benchmark's code), "Time base", the token-path check under "Order of work", and row "2. Dynamics and first table".

## Open decisions for the owner

The plan does not decide these; each has the author's recommendation.

1. **A first run on a provisional assignment.** Stage 1b's `v1` exists after its jobs K1, K2 and K3 (about 2 h of jobs after 1b's Task 9 is merged, inferred). A provisional `v0`, with kept values per signal from 1b's rough looks (Task 2's table), needs only K1. A run on `v0` proves the path and measures throughput; it cannot give the token-path verdict, a table cell, the size choice, or initial weights for a run on `v1` (the rows differ, and the trainer refuses). Recommendation: build `v0` and its store as soon as 1b's K1 and this plan's Task 2 exist, and use them at once for the store check (J2) and the throughput job (J3), which takes both off the critical path; start the first training run on `v0` only if this plan's Task 4 is done while `v1` is still more than about 2 h away, otherwise on `v1`.
2. **How the token-path check is read.** The design asks that "the mean read-out of a small simulator" come within 10% of the direct supervised model on tasks 1-1 and 1-2. This plan reads it as: a size-S simulator trained from random weights on the one task's pattern, with the same training budget and the same selection as the direct variant it is compared with (rule 3). The reference is the variant the headroom table chooses: `s90k` on 1-1 (`val_report` 0.1235, 90,000 steps) and `lr1e4` on 1-2 (0.0362), both measured 2026-11-22. On 1-1 that reference is itself behind the best published score (0.1235 on `val_report` against 0.0941 on test), so a pass there does not mean being competitive there. The pretrained, unadapted model's score is reported beside it and decides nothing. Recommendation: accept this reading; a `marginal` verdict goes to the owner before any size-M or size-L run starts.
3. **The bar (0.183).** With direct-model probes on 13 of the 14 tasks (2026-11-22, `val_report`; task 2-2 at its published 0.0370), the projected 14-task mean is 0.1913 against the target 0.183 (best published per task 0.1939, GenToka 0.2033). Recommendation: keep 0.183 and decide at the token-path check, the first number that says what the token path costs; this plan does not change it.

## Global Constraints

- Machine: node13, aarch64, 4 GPUs of 286 GiB, 352 cores, 1.4 TB RAM. Data and runs go under `/models/intermediate/mast/`, never into the repository. Nothing is ever written under `/models/tokamind/`. Nothing is uploaded anywhere. Temporary files go under `/models/intermediate/mast/trials/`, not `/tmp`.
- Work in the main checkout `/home/nvidia/FusionAIHub` on branch `nvidia`. Commit locally. Never `git push`, `reset --hard`, `clean`, `stash`, `checkout -- <file>` or `commit --amend`. `git add` only this task's files, by path; never `-A` or `.`. `.gitignore` excludes `*.h5`, `*.pt`, `*.npy`, `*.npz`: tests build their fixtures in memory or in `tmp_path`.
- Commit subjects start with `ignite.mast: `. The body says what changed and the measured facts behind it. The message ends with `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`.
- Two environments. `scripts/mast/py.sh` is `/models/envs/ignite-mast/bin/python -I` with `tokamark` and torch 2.15. `pixi run --frozen -e nvidia ...` (always `--frozen`) has no `tokamark`, so tests that need it skip there. `pyproject.toml` and `pixi.lock` do not change. No new dependency. No scikit-learn.
- No module under `src/tokamak_foundation_model/ignite/mast/` imports `tokamark`, `MAST_tools`, `mmt` or `zarr` at import time, and no module of `dynamics/` imports `ignite/mast/tokenizers/` at import time (functions that need a codec take it as an argument, or import inside the function). A test that needs `tokamark` starts with `pytest.importorskip("tokamark")`; a test that needs the data carries `@pytest.mark.real_data`.
- Hashed files do not change: `ignite/mast/harness.py`, `tasks.py`, `baselines.py`, `headroom/floor.py`, `headroom/cheap.py`, `headroom/direct.py`, and every `*.py` directly in `ignite/mast/tokenizers/`. Every stored score carries their sha256, and one changed byte marks all stored scores stale.
- Files under `src/tokamak_foundation_model/ignite/` outside `ignite/mast/` do not change. They may be imported, or their approach re-implemented under `ignite/mast/dynamics/`.
- Tests run with `-W error` on the CPU: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/<file> -q -W error`, and the whole directory with `CUDA_VISIBLE_DEVICES="" pixi run --frozen -e nvidia python -m pytest tests/ignite/mast -q -W error`. An implementer never runs the whole `tests/ignite` suite (some of its tests use a GPU).
- An implementer runs no training beyond a CPU trial of the `tiny` model on a few shots, no store build over the cache, no scoring pass over a split, and uses no GPU. A trial writes under `/models/intermediate/mast/trials/<task name>/` (set `IGNITE_MAST_ROOT` to it, and pass `--cache-dir /models/intermediate/mast/cache/v1` and `--fit-dir /models/intermediate/mast/stage1/fit/v1`, since both default to paths under the root). Long jobs are the controller's ("Controller jobs"). A GPU process is started only by the controller, with `--gpu N` for a GPU it checked with `nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader`.
- NRMSE (normalized root-mean-square error) always means TokaMark's evaluator through `harness.score`. Everything is trained on training shots, chosen on `val_select` and reported on `val_report`. Nothing in this plan reads the test split; the store build and the `score` command refuse it.
- A subset of shots is always spread evenly over the sorted shot list (`cache.pick_evenly`): the first 200 shots by number are one old campaign.
- Units. "Physical" is the unit of the data, "model units" the cache's (`transform.to_model`), "token values" what a stage-1b `Codec.encode` returns. A docstring names the unit and shape of every array a function takes and returns. Time is in seconds.
- Missing data is NaN and stays NaN; it is never filled with zeros or a placeholder value. A count of zero examples is reported as "no data", never as an error of 0.
- Read files in parts (at most 150 lines per read; grep first). Never print arrays or long logs.
- Docs are part of a change. A number is written with its date, its split, the number of shots, its limits and whether it is measured or inferred.

## Review Focus

- A token without data: a shot without Thomson profiles or without the `cc_field` Mirnov signal, clip frames before the shot's first sample, a row with nothing to attend to. There is no token, no loss and no NaN in any output. [Task 3, `test_a_token_with_nothing_to_attend_to_gives_finite_logits`; Task 4, `test_absent_values_carry_no_loss`; Task 5, `test_a_window_with_a_missing_input_signal_still_predicts_finite_values`]
- The future reaching a prediction while all frames are trained in parallel: a predicted frame must not see history at or after its own time, and a masked value must not be among its own inputs. [Task 3, `test_a_query_frame_does_not_see_history_at_or_after_its_time`; Task 4, `test_a_masked_value_is_not_an_input_of_its_own_prediction`]
- A frame grid that differs between training and scoring: a store frame must hold exactly the samples the benchmark's window holds for a cut on that phase, at every sampling rate, including the 5 ms signals and the spectrogram columns. [Task 1, `test_history_frames_are_counted_back_from_the_cut`; Task 2, `test_a_store_frame_equals_the_harness_window_of_the_same_cut`, `test_a_five_millisecond_signal_gives_the_sample_nearest_the_frame_start`]
- Tokenizers that do not belong to the weights: a store, a checkpoint and a predictor built on different assignments or fits. Each carries the layout's hash and refuses the others. [Task 4, `test_training_refuses_a_store_of_another_layout`; Task 5, `test_the_predictor_refuses_tokenizers_that_do_not_match_the_checkpoint`]
- A predictor that reads what it must not: the window's targets, or the test split. [Task 5, `test_the_predictor_does_not_read_the_targets`, `test_score_refuses_the_test_split`]

## Measured facts this plan relies on (2026-11-22)

"Author's look" means a read-only CPU look at cache v1 by the plan's author (`/models/intermediate/mast/trials/stage2-plan/look_align.py`, `look_grid.py`); they print counts only.

- Cache v1 (`/models/intermediate/mast/cache/v1/`): 11,188 shots (8,963 train, 1,115 val, 1,110 test), 39 signals, float32 in model units, NaN where not valid, each signal with its own `t0` and `dt`. Author's look, all shots: the training shots hold 5,524.0 s, 1,104,808 frames of 5 ms (validation 136,349). Median shot 550 ms (5th percentile 365 ms, 95th 1,130 ms); this includes the time before and after the plasma that the cache holds.
- Author's look, 240 training and validation shots spread evenly: all series signals of a shot share one `t0` (240 of 240). The two Mirnov spectrograms start 1.022 ms later than the series (4,000 of 4,000 shots looked at); their step is 1.024 ms.
- The benchmark's window rule, read from `/models/tokamind/tokamark/src/tokamark/tools/TokaMark_dataset.py`: cut times are `first valid input time + input_length + k * stride`, stride 1 ms for all 14 tasks; for a signal with time base `(times[0], dt)` the input ends (exclusive) at index `round((t_cut - times[0]) / dt)` and the output starts at `round((t_cut + delta - times[0]) / dt)`; samples are consecutive from there; an index beyond the array is clipped to its ends. Author's look, shot 27024, tasks 2-1 and 4-5 (3 and 12 windows, spectrogram columns included): every array of a harness window equals the cache slice this rule gives (largest relative difference 7.5e-6, no validity mismatch).
- Author's look, the same 240 shots: the first cut of a task lies a whole number of milliseconds after the shot's first series sample in 240 of 240 shots for 12 tasks and 238 of 240 for tasks 3-1 and 3-2 (largest offset 4.25 ms). Five frame phases 1 ms apart therefore cover the benchmark's cut times.
- Consequences. A 0.02, 0.2 or 0.25 ms signal has a whole number of samples per frame, so its frames are exact at every cut. A 5 ms signal (the EFIT signals, EFIT being the equilibrium reconstruction; Thomson profiles) gives a frame the one sample nearest the frame's start, up to 2.5 ms either side. A spectrogram gives a frame 4 or 5 columns, and which column is the first after a cut depends on a rounding against that cut; against a fixed 5 ms grid the benchmark's columns sit up to half a column (0.512 ms) early or late. What a column's stored time means (start, centre or end of its 1.024 ms) was not determined.
- The 14 tasks (`tasks.load_task`, author's look): group 1 has 5 ms in and the same 5 ms out (`delta` minus 5 ms) without actuators; group 2 5 ms in, 25 ms out; 3-1 and 3-2 5 ms in, 50 ms out; 3-3 150 ms in, 5 ms out; group 4 150 ms in, 100 ms out. The five signals that are an actuator in some task (8 channels) are `GIVEN_SIGNALS` (Task 1). `toy_task` (`tests/ignite/mast/conftest.py`) is markovian, and its actuator is not an input.
- Stage 0's direct supervised model (stage-0 page and `/models/intermediate/mast/stage0/analysis/seven_summary.py`, measured 2026-11-22): width 384, 8 layers, 14.4 to 18.2 million parameters, base 30,000 steps, batch 256 (64 for group 4), learning rate 3e-4, warm-up 1,000, validation every 2,500 steps on 200 `val_select` shots, bfloat16 autocast, about 0.12 GB of GPU memory per 1,000 tokens times batch under full attention. Median training throughput on shared GPUs: 668 to 1,334 windows per second on tasks 1-1, 1-2 and 2-1 (bound by TokaMark's loader), 200 to 205 on group 4 (5,050 to 5,750 tokens per window, bound by the GPU); 90,000 steps on task 1-1 took 7 h 7 min. Chosen variants on `val_report`: 1-1 `s90k` 0.1235, 1-2 `lr1e4` 0.0362, 2-1 `s90k` 0.1347; on group 4 the variant with 50 ms of history and batch 256 (`h050b256`) beats the base on 4-1, 4-2 and 4-4.
- `ignite/attention.py: attention(q, k, v, causal=False)` has no padding-mask argument. Measured 2026-11-21 (`docs/clusters/nvidia.md`): torch's `scaled_dot_product_attention` is as fast as or faster than FlashAttention-4 at 512 tokens within a frame and at 16 frames, and slower at 80 frames (2.63 ms against 2.03 ms).
- Inferred, not measured: with one row per channel, per band and per group of profile values, `v0` has 284 rows per frame (129 slow, 51 high-rate, 48 Mirnov, 16 boundary, 24 flux map, 16 Thomson) and about 1,500 values; the store of training and validation shots at five phases is then about 37 GB.

## Decisions of the plan's author, and where they leave the design

- **Attention with masks, without the FlashAttention-4 dispatch (departs from "kept from IGNITE: the FlashAttention-4 dispatch").** Absent tokens must be masked out of attention (change 2), and the dispatch takes no mask. Stage 2 calls `scaled_dot_product_attention` with boolean masks; measured above, the difference is small at these lengths.
- **A fifth state of a token: carried (the design names observed, given, to be predicted, and absent).** A carried token has no value, is attended to, has no loss and is not fed back. Reason: attention across frames is per row, so a signal observed before the cut and not predicted after it (the magnetics in task 2-1) would otherwise have no token after the cut, and what it said could reach the predicted rows only through rows present on both sides.
- **The mean read-out reads each value without the earlier values of its token (departs in detail from change 4, "a small head predicts each further one and is told the ones before it").** The head is trained both ways: told the earlier values (for sampling, plan 2b) and not told (for the mean roll-out). Reason: the score rewards the mean of each value, and feeding expected values through a conditional distribution does not give that mean. The switch `value_head` keeps the two pure forms.
- **One conditioning pattern per batch, and the batch holds only the pattern's rows.** The design sets roles per batch; rows outside the pattern then cost nothing. Batch size follows from a budget of token positions per step.
- **Training clips of at most 64 frames (44 before the cut, 20 after) (departs from "context up to 100 frames").** No benchmark task needs more than 30 frames before the cut and 20 after. The model has no table of positions (rotary encoding on time), so a longer context at inference needs no change; plan 2b's step interface keeps up to 100 frames.
- **Spectrogram frames in the store.** The store cuts a shot's columns as the benchmark would for a cut at the phase's first frame boundary, and counts frames from there; the predictor counts frames from each window's own cut. Both follow the benchmark's rule, but a frame can differ by one column at its edge, so training sees five column offsets per shot and scoring one per window. Equality is tested on the frames whose columns coincide.
- **Each of the nine changes has a switch** in `ModelConfig` or `TrainConfig`, each control setting has a unit test, and no ablation run is among this plan's jobs (stage 3 measures the changes).
- **Identity of a token is a learned embedding per layout row** (signal, token index), the rows built from the registry (departs in part from change 1, "identified by its registry row, not by its position in a fixed layout", and from the identity network over registry metadata). MAST has one device and its magnetics have no positions, so a network over metadata has nothing to generalize across yet; it returns when a second machine does.
- **One bin-embedding table per slot index, shared by all rows** (`n_val_max × n_bins × d_model`, Task 3); the row embedding carries the identity. A table per row would hold about 1,500 × 64 embeddings for `v0` (inferred) and a row's rare bins would be trained by few values.
- **The provisional assignment `v0`** is this plan's own file under the stage-2 tree, written with 1b's `assign.Setting` and `save_assignment`; it is not one of 1b's assignments.
- **Results tree.** `/models/intermediate/mast/stage2/`: `assignments/`, `tokens/<assignment>/`, `runs/<run>/`, `scores/<task>/<predictor>/<label>/`, `checks/`, `logs/`.

## Which number decides what

Stated before any number exists.

1. **The store agrees with the benchmark.** Job J2 compares store frames with harness windows on 16 shots (8 training, 8 validation, spread evenly) for tasks 1-1, 2-1, 3-1, 3-3 and 4-5. Every token value of a series, profile or map signal must be equal (largest absolute difference at most 1e-4 of the value's bin range), and every raw input array must equal the cache slice at the rebuilt cut. A spectrogram value must be equal on every frame whose columns coincide; the share of coinciding frames is printed. One unequal value stops stage 2 until it is explained: no run is started on that store.
2. **The first run is a pipeline proof.** Its training loss must fall, its validation loss must be finite for every probe-task pattern, and a checkpoint must score a finite NRMSE on tasks 1-1, 1-2 and 2-1 on 100 `val_select` shots. The mean baseline is scored on the same shots beside it; no rule hangs on that comparison. Scores of a run on `v0` are never copied into a results table.
3. **The token-path check.** For each of tasks 1-1 and 1-2 the reference is the direct variant the headroom table chooses (2026-11-22: `s90k` on 1-1, `lr1e4` on 1-2). A size-S simulator is trained from random weights on the task's pattern with that variant's budget, from its `config.json`: steps, batch and learning rate (1-1: 90,000, 256, 3e-4; 1-2: 30,000, 256, 1e-4), validated every 2,500 steps on the NRMSE of the mean roll-out on the same 200 `val_select` shots the direct model used. A run whose validation loss becomes non-finite, or ends above twice its minimum, is repeated once at a third of the learning rate (run name with `-lowlr`), and the repeat is the one used. The selected checkpoint is scored on all `val_report` shots. The ratio is that score over the reference's `val_report` score, computed from per-shot scores on the common shots, with a 95% interval from a paired bootstrap over shots (2,000 draws, seed 0). `pass`: the ratio is at most 1.10 for both tasks. `fail`: for some task the ratio is above 1.10 and the interval's lower end is above 1.10. `marginal`: otherwise. A run whose budget differs from its reference's gets no verdict.
4. **What the check decides.** On `pass`, plan 2c's size-M and size-L runs may start. On `marginal`, the owner is told first. On `fail`, no M or L run starts, and the report holds three stored numbers per task that say where the loss is: the tokenizer floor of the task's outputs (1b's `tok.<assignment>.ranged` on `val_report`), the cost of tokenized inputs (1b's `inputs.<assignment>.ranged.<variant>` against `raw.<variant>` on `val_report` from job K6, else the `unquantized` pair on `val_select-n100` from job K4), and the simulator's training and validation loss (fitting or generalizing). The fix follows the design's order (more kept values or bins where the first two numbers carry the gap, then a continuous output) and is a new short plan.
5. **Throughput.** Job J3 measures token positions per second and GPU memory per 1,000 positions for the three sizes; plan 2c sets batch and steps from them.

## File structure

```
src/tokamak_foundation_model/ignite/mast/paths.py                (modify: stage2_dir)
src/tokamak_foundation_model/ignite/mast/dynamics/__init__.py    (create: docstring, code_sha256)
src/tokamak_foundation_model/ignite/mast/dynamics/layout.py      (create: rows, states, groups, packing)
src/tokamak_foundation_model/ignite/mast/dynamics/windows.py     (create: benchmark window <-> token values)
src/tokamak_foundation_model/ignite/mast/dynamics/store.py       (create: token store, clip sampler, store check)
src/tokamak_foundation_model/ignite/mast/dynamics/bins.py        (create: ordered bins in torch)
src/tokamak_foundation_model/ignite/mast/dynamics/model.py       (create: the dynamics network, sizes)
src/tokamak_foundation_model/ignite/mast/dynamics/patterns.py    (create: conditioning patterns)
src/tokamak_foundation_model/ignite/mast/dynamics/train.py       (create: batches, loss, trainer)
src/tokamak_foundation_model/ignite/mast/dynamics/rollout.py     (create: mean roll-out by recomputation)
src/tokamak_foundation_model/ignite/mast/dynamics/predictor.py   (create: the harness predictor)
src/tokamak_foundation_model/ignite/mast/dynamics/checks.py      (create: per-shot scores, ratio, token-path verdict)
scripts/mast/run_dynamics.py                                     (create: provisional, tokens, train, score, check)
tests/ignite/mast/test_dyn_{layout,windows,store,bins,model,patterns,train,rollout,predictor,checks,command}.py   (create)
docs/models/ignite-mast-stage2.md, mkdocs.yml                    (create, modify)
```

Unit tests use a stand-in codec written in the test file: it follows 1b's `Codec` interface (attributes `name`, `dt`, `values_shape`, `domain`, `n_tok`, `n_val`, `grids`; methods `encode`, `decode`), and its two values are the frame mean and the frame slope, so no test depends on a fit. Tests that load the real tokenizers are `real_data` and skip while `<stage1>/fit/v1` is missing. Tasks 1 and 3 therefore need no 1b code merged; Task 2 needs 1b's Task 7 (`assign`, `fit`), Task 5 needs 1b's Tasks 6 and 8.

---

### Task 1: the package, `stage2_dir`, the layout and window tokens

**Files:**
- Modify: `src/tokamak_foundation_model/ignite/mast/paths.py`, `tests/ignite/mast/test_paths.py`
- Create: `src/tokamak_foundation_model/ignite/mast/dynamics/__init__.py`, `dynamics/layout.py`, `dynamics/windows.py`
- Test: `tests/ignite/mast/test_dyn_layout.py`, `tests/ignite/mast/test_dyn_windows.py`

**Interfaces:**
- Consumes (plan 1b, text): `Codec` with `name`, `dt`, `values_shape`, `domain`, `n_tok`, `n_val`, `grids: GridSet` (`lo`, `hi`, `width` each `(n_tok, n_val)`, `n_bins: int = 64`), `encode(x) -> (N, F, n_tok, n_val)` for physical `x (N, *values_shape, T)` with `F = frame_index(T, dt).max() + 1`, `decode(values, n) -> (N, *values_shape, n)`; `BandDCT` has `domain == "band"`, token index `probe * n_bands + band`, `n_val = max(k)`, values from `k[band]` on set to 0. `assign.Setting(codec, k, domain="linear", n_bins=64, source="")` with `k` = `(K,)` for `frame_dct`, `(n_values, values_per_token)` for `linear`, one K per band for `band_dct`; `assign.load_assignment(path) -> dict[str, Setting]`, `assign.assignment_sha256(a) -> str`; `fit.load_fit(dir) -> tuple[dict[str, SignalFit], dict]`, `fit.fit_sha256(dir) -> str`, `fit.build_codecs(signals, stats, fits, assignment) -> dict[str, Codec]`; `frames.FRAME_S = 0.005`, `frames.frame_len(dt)`, `frames.frame_index(n, dt)` (sample `j` in frame `floor((j + 0.5) * dt / FRAME_S)`). Stage 0 (code): `registry.all_signals() -> list[SignalSpec]` (needs `tokamark`), `transform.load_stats(path)`, `TaskSpec` (`markovian`, `input_length`, `output_length`, `delta`, `stride`, `inputs`, `actuators`, `outputs`, `n_in(s, history_s=None)`, `n_out(s)`, `n_act(s, history_s=None)`; `n_in` of a markovian task raises `KeyError` for a signal that is not an input), `WindowBatch` (`shot_ids`, `window_indices`, `inputs`, `actuators`, `targets`, `last_valid`), `paths.mast_root()`.
- Produces:
  - `paths.stage2_dir() -> Path`: `mast_root() / "stage2"`.
  - `dynamics.code_sha256(directory: Path | None = None) -> str`: sha256 over the bytes of every `*.py` directly in `dynamics/` (or `directory`), in name order.
  - In `layout.py`:
    - `ABSENT, OBSERVED, GIVEN, PREDICT, CARRIED = 0, 1, 2, 3, 4` (the state of a token in a frame; int8).
    - `GROUPS: tuple[str, ...]` = `("equilibrium", "boundary", "flux_map", "thomson", "coils", "magnetics", "saddle", "density", "soft_x_rays", "d_alpha", "mirnov", "actuators")` and `GROUP_OF: dict[str, str]`, one group for each of the 39 cached signals. `actuators` holds the five `GIVEN_SIGNALS`; `coils` holds the coil currents, the solenoid current and the plasma current.
    - `GIVEN_SIGNALS: tuple[str, ...]` = `("pf_active-coil_voltage", "summary-power_nbi", "pulse_schedule-i_plasma", "pulse_schedule-n_e_line", "gas_injection-total_injected")`.
    - `@dataclass(frozen=True, eq=False) class Layout` with `signals: tuple[str, ...]`; per row `row_signal`, `row_token`, `row_group`, `row_n_val` (each `(N,)` int64); per value `value_row`, `value_slot` (each `(W,)` int64) and `lo`, `hi`, `width` (each `(W,)` float64); `n_bins: int`; `assignment_sha256: str`; `fit_sha256: str`. Rows are in the order of `signals`, then token index; values in row order, then slot order. Properties `n_rows`, `n_values`, `n_val_max`. Methods `rows_of(signal) -> np.ndarray`; `values_of(rows: np.ndarray) -> np.ndarray` (packed value indices of the rows, in the given row order); `sha256() -> str` (over all fields); `pack(per_signal: Mapping[str, np.ndarray], n_frames: int, lead: tuple[int, ...] = ()) -> np.ndarray` (`name -> (*lead, n_frames, n_tok, n_val)` to float32 `(*lead, n_frames, W)`, NaN for a signal not given); `unpack(values, signal) -> np.ndarray` (`(..., F, W)` to `(..., F, n_tok, n_val)`, 0 in slots the row does not hold). Both are index operations on precomputed arrays, without a Python loop over rows.
    - `row_values(codec, setting) -> np.ndarray`: `(n_tok,)` int64, the values each token holds: `codec.n_val`, except for a codec with `domain == "band"`, where token `i` holds `setting.k[i % len(setting.k)]`.
    - `build_layout(signals: Sequence[SignalSpec], codecs: Mapping[str, object], assignment: Mapping[str, object], *, assignment_sha256: str, fit_sha256: str) -> Layout`. `KeyError` naming a signal without an entry in `GROUP_OF`; `ValueError` naming the signal for a held value whose `lo`, `hi` or `width` is NaN, or for codecs whose `grids.n_bins` differ.
    - `save_layout(layout, path)`, `load_layout(path) -> Layout` (JSON).
    - `load_tokenizers(assignment_path: Path, fit_dir: Path, cache_root: Path) -> tuple[Layout, dict[str, object]]`: the layout and the codecs by signal name, from `registry.all_signals()`, `transform.load_stats(cache_root / "stats.npz")`, `fit.load_fit`, `assign.load_assignment` and `fit.build_codecs`. It imports `tokenizers` inside the function.
  - In `windows.py`:
    - `frame_period(dt: float) -> tuple[int, int]`: the smallest number of samples that is a whole number of frames, and that number of frames: `(250, 1)` for 0.02 ms, `(1, 1)` for 5 ms, `(625, 128)` for 1.024 ms.
    - `encode_future(codec, x) -> np.ndarray`: `codec.encode(x)`; frames counted from the first sample.
    - `encode_history(codec, x, n_frames: int) -> np.ndarray`: `x (B, *values_shape, T)` physical, the last sample just before the cut, to `(B, n_frames, n_tok, n_val)`, frames counted back from the cut: `x` is padded on the left with NaN to a multiple of `frame_period(dt)[0]` samples, encoded, and the last `n_frames` frames are kept (frames the array does not reach are NaN).
    - `seed_frames(task, history_s: float | None = None) -> int`: `round(history / FRAME_S)`, history defaulting to `input_length`. `horizon_frames(task) -> int`: `ceil(output_length / FRAME_S)`. `in_frame(task) -> bool`: `abs(delta + input_length) < 1e-9` (group 1: the output window is the last input frame).
    - `actuator_split(task, s: SignalSpec, history_s: float | None = None) -> int`: the actuator samples before the cut, `task.n_act(s, history_s) - round((task.delta + task.output_length) / s.dt)`.
    - `encode_window(batch: WindowBatch, task, layout, codecs, history_s: float | None = None) -> np.ndarray`: float32 `(B, S + H, W)` with S and H from the two functions above (`(B, S, W)` when `in_frame`). Input signals fill the S frames before the cut (`encode_history`); actuators fill all frames (their array split at `actuator_split`: `encode_history` before, `encode_future` after); output-only signals are NaN. It does not read `batch.targets`.
    - `encode_targets(batch, task, layout, codecs) -> np.ndarray`: float32 `(B, H, W)` (`(B, 1, W)` when `in_frame`) holding only the output signals' values.
    - `decode_window(values, task, layout, codecs) -> dict[str, np.ndarray]`: `(B, H, W)` token values to one physical array per output signal, `(B, *values_shape, task.n_out(s))`, by `codec.decode`.

- [ ] **Step 1: Write the failing tests**

`test_paths.py` gains `test_stage2_dir_is_under_the_mast_root`: with `IGNITE_MAST_ROOT` set to `tmp_path`, `stage2_dir()` is `tmp_path / "stage2"`.

`test_dyn_layout.py`, with stand-in codecs for a series of 3 channels with 2 values, a profile of 2 tokens with 4 values, and a band signal of 2 probes, 8 bands and `k = (4, 2, 2, 2, 2, 2, 2, 2)`:
- `test_rows_follow_the_signal_order_and_values_follow_the_rows`: `n_rows == 3 + 2 + 16`; `row_signal` and `value_row` are non-decreasing; `value_slot` restarts at 0 at every row.
- `test_a_band_row_holds_its_bands_values`: `row_values` is 4 for tokens 0 and 8 and 2 elsewhere; `n_values == 3*2 + 2*4 + 2*(4 + 7*2)`.
- `test_pack_and_unpack_are_inverse_on_the_held_slots`: `unpack(pack(x), name)` equals `x` on held slots and is 0 beyond; a signal left out of `pack` is all NaN.
- `test_values_of_keeps_the_given_row_order`: for rows `[5, 0]` the indices of row 5 come first.
- `test_a_signal_without_a_group_is_refused` (`KeyError` naming it); `test_a_value_without_a_range_or_another_bin_count_is_refused` (`ValueError`, both cases).
- `test_the_layout_survives_a_save_and_its_hash_changes_with_a_range`.
- `test_every_cached_signal_has_a_group` (needs `tokamark`): every name of `registry.all_signals()` is in `GROUP_OF`; every `GIVEN_SIGNALS` name is an actuator of some task and in group `actuators`.
- `test_the_code_hash_changes_with_a_file`: over a copy of the package in `tmp_path`, one changed byte changes `code_sha256(copy)`.

`test_dyn_windows.py`, with the stand-in codec, `toy_task`, `toy_batch` and hand-built tasks:
- `test_frame_period_of_the_five_sampling_rates`: `(250, 1)`, `(25, 1)`, `(20, 1)`, `(1, 1)`, `(625, 128)` for 0.02, 0.2, 0.25, 5 and 1.024 ms.
- `test_history_frames_are_counted_back_from_the_cut`: a 1.024 ms signal of 146 columns with values 0, 1, 2, ...: the last frame's mean is that of columns 141 to 145, the frame before it that of columns 136 to 140 (counted from the first column, the last frame would hold 4 columns).
- `test_history_and_future_agree_for_whole_frames`: for a 0.25 ms signal of 600 samples, `encode_history(x, 30)` equals `encode_future(x)`.
- `test_a_history_shorter_than_the_frames_asked_for_is_padded_with_nan`.
- `test_an_actuator_is_split_at_the_cut`: on the markovian `toy_task` (its actuator is not an input), an actuator that is 1 before the cut and 2 after gives frame means 1 before and 2 after; `actuator_split` is 20 for 0.25 ms.
- `test_encode_window_does_not_read_the_targets`: NaN targets give the same result; output-only rows are NaN.
- `test_a_group_one_window_has_one_frame`: `delta == -input_length` gives `(B, 1, W)`.
- `test_decode_window_returns_the_target_shapes`, including 98 spectrogram columns from 20 frames.
- `test_encode_targets_then_decode_is_the_codecs_round_trip`: equals `codec.decode(codec.encode(target), n)`.

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_dyn_layout.py tests/ignite/mast/test_dyn_windows.py tests/ignite/mast/test_paths.py -q -W error`
Expected: an import error for `dynamics`.

- [ ] **Step 3: Implement `stage2_dir`, the package, `layout.py` and `windows.py`**

`stage2_dir` follows `stage0_dir`. The package docstring says what stage 2 is and names the five token states.

- [ ] **Step 4: Run the tests**

Run the command of Step 2. Expected: all passed (19 new tests; `test_every_cached_signal_has_a_group` skips under pixi).

- [ ] **Step 5: Commit**

```bash
M=src/tokamak_foundation_model/ignite/mast
git add $M/paths.py $M/dynamics/{__init__,layout,windows}.py tests/ignite/mast/test_{paths,dyn_layout,dyn_windows}.py
git commit -m "ignite.mast: stage-2 package, the token layout and window tokens"
```

### Task 2: the token store, its check against the benchmark, and the commands `provisional` and `tokens`

**Files:**
- Create: `src/tokamak_foundation_model/ignite/mast/dynamics/store.py`, `scripts/mast/run_dynamics.py`
- Test: `tests/ignite/mast/test_dyn_store.py`, `tests/ignite/mast/test_dyn_command.py`

**Interfaces:**
- Consumes: Task 1; `cache.read_shot(path, names=None, *, stats_sha256=None) -> dict[str, CachedSignal | None]` (`values` float32 `(*values_shape, T)` in model units, `t0`, `dt`), `cache.pick_evenly(shots, n)`, `transform.to_physical(z, st)`, `tasks.split_shots(split)`, `tasks.SPECTROGRAM_SIGNALS`, `paths.cache_dir("v1")`, `paths.prepare_write_root(path)`; `harness.iter_windows(task, shots, *, test_mode, history_s=None, batch_size=64, num_workers=32)`; 1b's `assign.Setting`, `assign.save_assignment(a, path, notes=None)` (`notes` one string per signal), `assign.codec_kind(spec)`, `paths.stage1_dir()`; from `scripts/mast/run_headroom.py`, loaded by path as 1b's command loads it: `Refused`.
- Produces in `store.py`:
  - `N_PHASES = 5`, `PHASE_S = 0.001`.
  - `start_index(t: float, t0: float, dt: float) -> int`: `int(round((t - t0) / dt))`, the benchmark's rule.
  - `reference_time(signals: Mapping[str, CachedSignal | None]) -> float | None`: the smallest `t0` among the present signals that are not spectrograms; None if there is none.
  - `encode_shot(signals, stats, layout, codecs, phase: int) -> np.ndarray | None`: float32 `(F, W)`, or None for a shot without a series signal or shorter than one frame. Frame `m` starts at `b_m = reference_time + phase * PHASE_S + m * FRAME_S`; F is the number of whole frames before the last series sample. A signal with `n` samples per frame gives frame `m` the samples `start_index(b_m, t0, dt) + 0 .. n - 1`. A spectrogram gives the columns from `start_index(b_0, t0, dt)` on, as many as `frames.frame_index` assigns to frames below F. An index outside the stored array reads as NaN (no clipping to the ends). Each signal is converted to physical units and encoded with one call of `codec.encode`. A signal the shot lacks is NaN in all its values.
  - `build(root: Path, cache_root: Path, shots: Mapping[str, Sequence[int]], layout, codecs, *, workers: int, force: bool = False) -> dict`: writes `root/layout.json`, one `root/shots/<shot>.h5` per shot (attributes `shot`, `split`, `t_ref`, `layout_sha256`; datasets `phase0` to `phase4`, float32 `(F_p, W)`) and `root/manifest.json` (`layout_sha256`, `assignment_sha256`, `fit_sha256`, `stats_sha256`, `tokenizer_code_sha256`, `dynamics_code_sha256`, `n_rows`, `n_values`, `counts` per split with `shots` and `frames`, `no_data` as `[shot, reason]` pairs, `seconds_per_shot`, `size_bytes`, `date`). A process pool as `cache.build` uses; the codecs reach each worker once through the pool's initializer. A file whose `layout_sha256` is current is kept unless `force`. Keys of `shots` are `train` and `val` only; `test` raises `ValueError`. Returns the manifest.
  - `class TokenStore`: `TokenStore(root: Path, shots: Sequence[int], layout)` reads the shots' arrays into memory (thread pool); `ValueError` naming "layout" if the manifest's `layout_sha256` is not `layout.sha256()`. Attributes `shots: np.ndarray`, `n_frames: np.ndarray` (phase-0 frames per shot). `clip(shot, phase, first, n) -> np.ndarray`: float32 `(n, W)`, NaN for frames outside the shot. `sample(rng: np.random.Generator, batch: int, before: int, after: int, *, need_before: np.ndarray | None = None, need_after: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]`: clips `(B, before + after, W)` and `(B, 3)` int64 rows `(shot, phase, cut)`. A shot is drawn with probability proportional to its frames, a phase uniformly, a cut uniformly among the frame boundaries with at least one frame of the shot before it and, if `after > 0`, one after it. A clip is redrawn (at most 20 times, then `RuntimeError` naming the counts) if `need_before` (value indices) has no finite value before the cut or `need_after` none after it; with `after == 0` both are looked for in the last frame.
  - `check_against_harness(task, shot: int, cache_root: Path, layout, codecs, *, every: int = 20, history_s: float | None = None) -> dict`: for every `every`-th evaluation window of the shot (`harness.iter_windows(..., test_mode=True)`), the cut is rebuilt from the cache as `first valid input time + history + window_index * stride` and then verified: the window's raw input arrays must equal the cache slices at that cut. The store frames for the cut (`encode_shot` at the cut's phase) are compared with `windows.encode_window` and `encode_targets` of the harness window. Returns `n_windows`, `n_compared`, `raw_unequal` (windows whose raw inputs differ from the cache slice), `off_grid` (cuts not a whole millisecond after the reference time; counted, not compared), `n_unequal` (series, profile and map values differing by more than 1e-4 of their bin range, or finite in one and not the other), `band_frames`, `band_frames_coinciding`, `band_unequal` (on coinciding frames).
- Produces in `scripts/mast/run_dynamics.py` (run with `scripts/mast/py.sh`):
  - `PROVISIONAL: dict[str, tuple]`, the `k` of `v0` per codec kind and sampling: cosine transform `(1,)` for one sample per frame, `(4,)` for 20 or 25 samples per frame, `(8,)` for 250; linear basis `(32, 4)` for the boundary and Thomson signals and `(48, 2)` for the flux map; band transform `(16, 8, 8, 8, 8, 8, 8, 8)`. Domains: `linear` for series, `signal` for the linear basis, `band` for spectrograms.
  - `provisional --name NAME [--fit-dir DIR] [--cache-dir DIR]`: builds every codec once with `fit.build_codecs` (so a setting the fit cannot carry fails here), writes `<stage2>/assignments/<NAME>.json` with `assign.save_assignment` and the note "provisional, from the rough looks of plan 1b; not chosen by a ladder" for every signal, prints one line per signal and the totals of rows and values per frame. Refuses (exit 2) the name `v1` and any name under `<stage1>/assignments/`.
  - `tokens --assignment FILE [--fit-dir DIR] [--cache-dir DIR] [--workers 64] [--shots N,N,...] [--force]`: builds `<stage2>/tokens/<file stem>/` for the training and validation shots (`--shots` restricts to the given training shots, for trials); prints the root before its first write, then per split the shots, frames, shots without data, seconds per shot and the size. Exit 1 if more than 1% of the shots have no data.
  - `tokens --assignment FILE --check [--n-shots 16] [--tasks task_1-1,task_2-1,task_3-1,task_3-3,task_4-5]`: `check_against_harness` on half training, half validation shots spread evenly; one line per task and the last line `<n> compared, <u> unequal, <r> raw unequal, <c> of <b> band frames coincide, <o> windows off the grid`; exit 1 if `u`, `r` or `band_unequal` is not 0.
  - `main(argv: Sequence[str] | None = None) -> int`.

- [ ] **Step 1: Write the failing tests**

`test_dyn_store.py`, stand-in codecs, shots built in memory as `CachedSignal`s with identity statistics:
- `test_a_frame_holds_the_samples_from_the_benchmarks_start_index`: a 0.25 ms ramp with `t0 = -0.0602`; at phase 3 the mean of frame 0 is the mean of samples 12 to 31.
- `test_a_five_millisecond_signal_gives_the_sample_nearest_the_frame_start`: samples 0, 1, 2, ... at 5 ms; frame `m` holds `m` at phases 0, 1, 2 and `m + 1` at phases 3, 4.
- `test_spectrogram_columns_start_at_the_rounded_index_of_the_first_boundary`: with the spectrogram's `t0` 1.022 ms after the series, the first column read at phase 0 is index -1 (outside the array, NaN) and at phase 2 index 1; frame 0 holds the columns `frames.frame_index` gives it.
- `test_an_index_outside_the_array_is_nan_not_the_edge_value`.
- `test_a_signal_the_shot_lacks_is_nan_and_the_others_are_not`; `test_a_shot_without_series_gives_none`.
- `test_build_writes_five_phases_and_a_manifest` (`tmp_path`, 3 shots, `workers=1`): shapes, `counts`; a second call keeps the files.
- `test_build_refuses_the_test_split`; `test_the_store_refuses_another_layout`.
- `test_a_clip_beyond_the_shot_is_padded_with_nan`.
- `test_sample_draws_cuts_inside_the_shot_and_redraws_for_missing_data`: with `need_after` naming a signal one of two shots has, every clip is from that shot; a need no shot meets raises `RuntimeError`.
- `test_sampling_is_reproducible_from_the_generator`.
- `test_a_store_frame_equals_the_harness_window_of_the_same_cut` (`real_data`, needs `tokamark` and `<stage1>/fit/v1`): one validation shot, tasks 1-1, 2-1 and 4-5, a `v0`-style assignment built in memory: `n_unequal == 0`, `raw_unequal == 0`, `band_unequal == 0`, `n_compared > 0`, and `band_frames_coinciding > 0` for 4-5.

`test_dyn_command.py`:
- `test_provisional_refuses_the_name_of_a_real_assignment` (exit 2 for `v1`).
- `test_the_provisional_table_covers_every_kind_and_rate` (needs `tokamark`): every signal of `registry.all_signals()` gets a setting, and `k[0]` of a cosine transform is at most its samples per frame.
- `test_tokens_prints_the_root_before_writing` (builder patched; `IGNITE_MAST_ROOT` in `tmp_path`).
- `test_tokens_check_exits_1_on_an_unequal_value` (the check patched to return one).

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_dyn_store.py tests/ignite/mast/test_dyn_command.py -q -W error`
Expected: an import error for `store`.

- [ ] **Step 3: Implement `store.py` and the two commands**

- [ ] **Step 4: Run the tests**

Run the command of Step 2. Expected: all passed (17 tests; the `real_data` test skips until 1b's fit exists).

- [ ] **Step 5: A trial on three shots, on the CPU** (when 1b's job K1 has run; otherwise the task is reviewed without it)

With `IGNITE_MAST_ROOT=/models/intermediate/mast/trials/stage2a-task2` and the cache and fit paths of the Global Constraints: `provisional --name v0`, `tokens --assignment <root>/stage2/assignments/v0.json --shots <three training shots>`, then `tokens --assignment ... --check --n-shots 2`. Record in the commit body the rows and values per frame of `v0`, seconds per shot, bytes per frame, and the check's last line.

- [ ] **Step 6: Commit**

```bash
git add src/tokamak_foundation_model/ignite/mast/dynamics/store.py scripts/mast/run_dynamics.py tests/ignite/mast/test_dyn_{store,command}.py
git commit -m "ignite.mast: the token store, its check against the benchmark's windows, and a provisional assignment"
```

### Task 3: ordered bins in torch and the dynamics model

This task needs only Task 1 and can be built beside Task 2.

**Files:**
- Create: `src/tokamak_foundation_model/ignite/mast/dynamics/bins.py`, `dynamics/model.py`
- Test: `tests/ignite/mast/test_dyn_bins.py`, `tests/ignite/mast/test_dyn_model.py`

**Interfaces:**
- Consumes: `Layout` and the five token states (Task 1). To read, not to import: `ignite/dynamics.py` (`FactorizedSTBlock.forward_two_stream`, `_MHA.forward_with_history`, `DynamicsBackbone.act_embed`) for the two-stream arrangement re-implemented here with masks.
- Produces in `bins.py`:
  - `class BinTable(torch.nn.Module)`: `BinTable(lo, hi, width: np.ndarray, n_bins: int)` for `(W,)` arrays; buffers only. `centres() -> Tensor` `(W, n_bins)`: `width * sinh(linspace(asinh(lo / width), asinh(hi / width), n_bins))`, 1b's `GridSet` formula. `two_hot(values) -> Tensor`: `(..., W)` to `(..., W, n_bins)`; the two neighbouring centres get weights linear in the value; beyond the range, weight 1 on the outermost centre; NaN gives a row of zeros (not NaN, unlike `GridSet`). `nearest(values) -> Tensor` (one-hot, zeros for NaN). `expected(logits) -> Tensor`: `(..., W, n_bins)` to `(..., W)`, the mean of the centres under the softmax. `sample(logits, generator: torch.Generator) -> Tensor`: one centre per value from the softmax. `take(index: Tensor) -> BinTable`.
- Produces in `model.py`:
  - `@dataclass(frozen=True) class ModelConfig` with `d_model: int`, `depth: int`, `n_heads: int`, `ffn_mult: int = 4`, `d_value_head: int = 256` (hidden width of the per-value head), `frame_s: float = 0.005`, `tau_s: float = 0.045` (the machine's reference time scale; 45 ms on MAST), `rotary_periods: tuple[float, float] = (0.2, 30.0)` (shortest and longest period, in units of `tau_s`), and the switches, default first: `time_encoding: "rotary" | "learned"` (a table of 128 frame positions), `value_embed: "two_hot" | "nearest"`, `absent: "masked" | "placeholder"`, `actuators: "tokens" | "summed"`, `value_head: "both" | "ordered" | "independent"`, `time_scale_inputs: bool = True`. An unknown value raises `ValueError` naming the allowed ones.
  - `SIZES: dict[str, dict]` = `{"tiny": dict(d_model=64, depth=2, n_heads=4, d_value_head=64), "S": dict(d_model=384, depth=12, n_heads=8), "M": dict(d_model=640, depth=16, n_heads=10), "L": dict(d_model=1024, depth=18, n_heads=16)}`. Inferred from 16 `d_model` squared per block (two attentions and a feed-forward layer of width 4): 28, 105 and 302 million parameters in the blocks; the code prints the exact count.
  - `class Frames(NamedTuple)`: `values: Tensor` float32 `(B, F, w)`, the packed values of the rows in use, NaN where there is no value; `state: Tensor` int8 `(B, F, n)`; `t: Tensor` float32 `(B, F)`, each frame's start in seconds relative to the cut.
  - `class Dynamics(torch.nn.Module)`: `Dynamics(cfg: ModelConfig, layout: Layout)`.
    - `forward(history: Frames, query: Frames, rows: Tensor, *, told: Tensor | None = None) -> Tensor`: logits `(B, Q, w, n_bins)` for every value of every query token. `rows (n,)` int64 are the layout rows in use, the same for both streams; `w = len(layout.values_of(rows))`. `history` may have zero frames. `told (B, Q, w)` holds the values the head is told when it predicts a later value of the same token (NaN: not told); None is the read without them.
    - `n_parameters() -> int`; `bins: BinTable` over all the layout's values.
  - The rules the tests pin:
    - A token enters as the sum of its row's embedding, its state's embedding, and either its value embedding (states `OBSERVED`, `GIVEN`) or a mask embedding (`PREDICT`, `CARRIED`). The value embedding sums over the held slots `j` the two-hot weights of the value times a table `E[j]` of `n_bins` embeddings; `E` (`n_val_max, n_bins, d_model`) is shared by all rows, whose identity is the row embedding (two lookups per value, weighted, added into its row with `index_add`). With `time_scale_inputs`, a vector computed from `log(frame_s / tau_s)` is added to every token.
    - Attention within a frame: among the tokens of one frame of one stream that are not `ABSENT`, without position encoding.
    - Attention across frames, per row: a history token attends to the history tokens of its row at its own time and earlier; a query token to the history tokens of its row at strictly earlier times, and to itself. `ABSENT` tokens are never attended to. A token left with nothing to attend to attends to itself, which is put into the mask before the call, never repaired after it. Rotary encoding of the angle `t / tau_s` at periods spaced logarithmically between `rotary_periods`. The history and query tokens of a row lie in one sequence, so one call serves both streams.
    - The head predicts slot `v` of a token from the token's final state, an embedding of `v`, and the sum of the value embeddings of the told slots before `v`.
    - `actuators="summed"`: `GIVEN` tokens leave the attention, and a linear map of the frame's given values (NaN as 0) is added to every token of the frame, as IGNITE's `act_embed` does. `absent="placeholder"`: `ABSENT` tokens are attended to and carry only the state's embedding.
    - Pre-normalization blocks, as IGNITE's. No tensor has a dimension of `n_val_max` other than `E`.

- [ ] **Step 1: Write the failing tests**

`test_dyn_bins.py`:
- `test_centres_two_hot_and_expected_agree_with_gridset` (skipped if `tokenizers.grid` is not importable): random ranges and finite values, including beyond the range, agree with `GridSet` to 1e-5.
- `test_centres_follow_the_asinh_formula` (no dependency): against the formula written in the test.
- `test_a_value_beyond_the_range_enters_as_the_outermost_bin`; `test_nan_gives_a_row_of_zeros_and_no_warning`.
- `test_expected_of_a_two_hot_target_is_the_value`: logits `log(two_hot(v))` for values inside the range.
- `test_a_sample_is_a_centre_and_is_reproducible_from_the_generator`; `test_take_selects_values`.

`test_dyn_model.py`, a hand-built `Layout` of 6 rows (value counts 1, 2, 2, 4, 1, 3), size `tiny`, float32 on the CPU with `torch.manual_seed`:
- `test_logits_have_one_row_of_bins_per_value`: shape `(B, Q, w, n_bins)` for a subset of 4 rows, all finite.
- `test_a_query_frame_does_not_see_history_at_or_after_its_time`: changing history values at `t >= t_q` changes no logit of query frame `q` (at most 1e-5); changing a history frame before `t_q` changes some.
- `test_a_history_frame_does_not_see_later_history`: removing a history frame after `t_q` changes no logit of query frame `q`.
- `test_a_query_frame_does_not_see_other_query_frames`.
- `test_an_absent_token_changes_nothing`: the values of an `ABSENT` token and of an `ABSENT` row change no logit of the other rows.
- `test_a_token_with_nothing_to_attend_to_gives_finite_logits`: empty history, one token not absent.
- `test_a_carried_token_passes_what_its_row_saw_before` (depth 1): one row `OBSERVED` in the history and `CARRIED` in the query, one `PREDICT` row absent from the history: changing the first row's history values changes the second's logits; with the first row `ABSENT` in the query it does not.
- `test_an_observed_value_enters_at_full_precision`: two values inside one bin give different logits.
- `test_the_head_is_told_only_earlier_slots`: changing `told` at slot 2 of a token changes the logits of its slots 3 and later only.
- `test_without_told_values_the_slots_do_not_depend_on_each_other`.
- `test_the_time_encoding_depends_on_time_not_on_the_frame_index`: shifting all `t` by a constant changes no logit; twice the spacing changes some.
- `test_every_switch_runs_in_its_control_setting` (parametrized over the seven control settings): finite logits of the right shape.
- `test_an_unknown_switch_value_is_refused`.
- `test_the_sizes_have_the_expected_block_parameters`: `16 * d_model**2 * depth` is 28.3, 104.9 and 302.0 million for S, M, L (computed without building); `tiny` builds and `n_parameters() > 0`.
- `test_bfloat16_autocast_on_the_cpu_gives_finite_logits`.

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_dyn_bins.py tests/ignite/mast/test_dyn_model.py -q -W error`
Expected: an import error for `bins`.

- [ ] **Step 3: Implement `bins.py` and `model.py`**

- [ ] **Step 4: Run the tests**

Run the command of Step 2. Expected: all passed (22 tests, one of them with 7 parametrized cases).

- [ ] **Step 5: Commit**

```bash
git add src/tokamak_foundation_model/ignite/mast/dynamics/{bins,model}.py tests/ignite/mast/test_dyn_{bins,model}.py
git commit -m "ignite.mast: the dynamics model with token states, attention masks and ordered bins"
```

### Task 4: conditioning patterns, the loss, the trainer and the command `train`

After this task and jobs J1 and J2 the controller starts the first training run.

**Files:**
- Create: `src/tokamak_foundation_model/ignite/mast/dynamics/patterns.py`, `dynamics/train.py`
- Modify: `scripts/mast/run_dynamics.py`, `tests/ignite/mast/test_dyn_command.py`
- Test: `tests/ignite/mast/test_dyn_patterns.py`, `tests/ignite/mast/test_dyn_train.py`

**Interfaces:**
- Consumes: Tasks 1 to 3; `tasks.load_task`, `tasks.ALL_TASKS`, `tasks.PROBE_TASKS`, `tasks.split_shots`; from `scripts/mast/run_headroom.py` (by path): `Refused`, `_look_up_gpu(gpu: int) -> tuple[str, list[int]] | None`, `_refuse_busy_gpu(gpu: int, pids: list[int], share: bool = False) -> bool`, `_take_lock(run_dir) -> Path`, `_release_lock(lock)`, `_write_json_atomic(path, record)`, `LOCK`, `TRAINED_BY`.
- Produces in `patterns.py`:
  - `@dataclass(frozen=True) class Pattern` with `name: str`; `context`, `given`, `predict`, `track: tuple[str, ...]` (signal names; `track` contains `predict`); `before: int` (frames before the cut, at least 1); `after: int` (frames after it; 0 for a pattern that predicts inside its last frame). `rows(layout) -> np.ndarray`: the layout rows of all four sets, sorted.
  - `task_pattern(task, history_s: float | None = None, track: str = "outputs") -> Pattern`: `context` the task's inputs, `given` its actuators, `predict` its outputs, `before` and `after` from `windows.seed_frames`, `horizon_frames` and `in_frame`; `track` the outputs, or the inputs and outputs with `"all"`.
  - `@dataclass(frozen=True) class PatternMix` with `p_task: float = 0.5`, `max_before: int = 44`, `max_after: int = 20`, `p_in_frame: float = 0.2`, `p_context: float = 0.7`, `p_predict: float = 0.5`, `p_no_given: float = 0.1`, `p_track_all: float = 0.5`.
  - `draw_pattern(rng, layout, mix, task_patterns: Sequence[Pattern]) -> Pattern`: with probability `p_task` one of `task_patterns` uniformly, its `given` emptied with probability `p_no_given`. Otherwise a pattern over signal groups: each group but `actuators` is in the context with probability `p_context` and predicted with probability `p_predict` (at least one of each); `given` is `GIVEN_SIGNALS`, or empty with probability `p_no_given`; with probability `p_in_frame` it predicts inside its last frame (no predicted group is then in the context), else `before` is uniform in 1 to `max_before` and `after` in 1 to `max_after`; `track` is `predict`, or the context and `predict` with probability `p_track_all`.
  - `states(pattern, layout, rows) -> tuple[np.ndarray, np.ndarray]`: the states `(F_h, n)` of the history stream and `(Q, n)` of the query stream before missing data and masking. With `after > 0`, the history stream is all `before + after` frames and the query stream the `after` frames. History before the cut: context `OBSERVED`, given `GIVEN`, others `CARRIED`. History after the cut: given `GIVEN`, tracked `OBSERVED` (they hold fed-back values), others `CARRIED`. Query: given `GIVEN`, tracked `PREDICT`, others `CARRIED`. With `after == 0` the history is the first `before - 1` frames and the query the last frame: context `OBSERVED`, given `GIVEN`, predicted `PREDICT`.
- Produces in `train.py`:
  - `@dataclass(frozen=True) class TrainConfig` with `assignment: str` (the assignment file, resolved), `fit_dir: str`, `cache_root: str`, `size: str = "S"`, `model: dict = {}` (overrides of `ModelConfig`), `pattern: str = "mix"` (or a task name), `mix: dict = {}`, `steps: int = 20000`, `positions_per_step: int = 600_000`, `batch_max: int = 1024`, `batch: int | None = None` (a fixed batch; `positions_per_step` then unused), `lr: float = 3e-4`, `warmup: int = 1000`, `weight_decay: float = 0.05`, `ema: float = 0.999`, `grad_clip: float = 1.0`, `p_full_mask: float = 0.5`, `feedback_final: float = 0.75`, `feedback_ramp: int = 2000`, `expected_share: float = 0.5` (change 9; 0 is IGNITE's sampled feedback), `p_told: float = 0.5`, `loss_weight: str = "group"` (control `"token"`), `group_weights: dict = {}` (1 per group), `val_every: int = 1000`, `save_every: int = 1000`, `n_val_shots: int = 200`, `n_val_clips: int = 256`, `seed: int = 0`, `init: str | None = None`.
  - `batch_size(cfg, pattern, n_rows) -> int`: `cfg.batch` if set, else `positions_per_step // (n_rows * (history frames + query frames))`, clamped to 4 .. `batch_max`.
  - `make_batch(clip: Tensor, pattern, layout, rows: np.ndarray, *, generator, p_full_mask) -> tuple[Frames, Frames, Tensor, Tensor]`: from the true values `(B, F, w)` of the rows in use, the history frames, the query frames, the targets `(B, Q, w)` and the loss mask `(B, Q, w)` bool. A token whose state needs a value and whose data is NaN becomes `ABSENT`. In the query, with probability `p_full_mask` nothing is revealed; otherwise, per sample, a share `cos(pi * u / 2)` (`u` uniform) of the `PREDICT` tokens stays masked and the rest become `OBSERVED` with their true values. A `PREDICT` token whose target is NaN becomes `CARRIED`. The loss mask is true for the held values of tokens still `PREDICT`; the query's `values` are NaN wherever the token is `PREDICT` or `CARRIED`.
  - `feed_back(model, history, query, rows: Tensor, *, share, expected_share, generator) -> Frames`: one pass without gradients with every `PREDICT` token masked; for a share `share` of the samples the history values of the tracked tokens after the cut are replaced by the model's output, as expected values for a share `expected_share` of those samples and as one sample per value for the others. `ABSENT` stays `ABSENT`. With `share == 0` or no frame after the cut, the history is returned unchanged and no pass is made.
  - `loss_fn(logits, targets, mask, bins: BinTable, value_group: Tensor, weights: Tensor, how: str = "group") -> tuple[Tensor, dict[str, float]]`: cross-entropy between the logits and the target's two-hot weights, per value; `"group"`: the mean over the masked values of each group, then the weighted mean over the groups that have any; `"token"`: the mean over all masked values. Returns the loss and the per-group means; with no masked value, a zero that carries gradients and an empty dictionary.
  - `validate(model, store, patterns, layout, cfg, device) -> dict[str, float]`: per pattern the loss on `n_val_clips` clips drawn with a fixed seed, every predicted token masked, true history, read without told values; plus `"mean"`.
  - `train(cfg, run_dir: Path, store_root: Path, *, device: str, max_seconds: float | None = None) -> dict`: one pattern drawn per step; `told` holds the targets of the masked values for a share `p_told` of the samples (none with `value_head="independent"`, all with `"ordered"`), NaN elsewhere. The feedback share rises linearly from 0 to `feedback_final` over `feedback_ramp` steps. AdamW (betas 0.9, 0.95), linear warm-up then cosine decay, bfloat16 autocast on a GPU, gradient clipping, an exponential moving average of the weights. Training shots `tasks.split_shots("train")`; validation shots `cache.pick_evenly(tasks.split_shots("val_select"), n_val_shots)`; validation patterns the six probe-task patterns for `pattern == "mix"`, else the task's own. Writes `config.json` (the configuration, the layout hash, the store manifest's hashes), `layout.json`, `train_log.csv` (`step, lr, loss, feedback_share, positions, positions_per_s, gpu_gb_peak, wall_s`, one `loss_<group>` per group, and at validation steps `val_mean` and one `val_<pattern>` per pattern), `last.pt`, `step_<n>.pt` every `save_every` steps, `best.pt` at the lowest `val_mean`, and `trained_by.json` (`dynamics_code_sha256`, `tokenizer_code_sha256`, git revision, date) before the first step. A checkpoint holds the averaged and raw weights, the optimizer state, the step, the configuration and the layout. With `init`, the weights come from that checkpoint, whose layout must have the same hash (`ValueError` naming "layout"). A run directory with a `last.pt` resumes from it if the configuration is the same apart from `steps`; otherwise `ValueError`. Returns the last log row.
  - `load_model(checkpoint: Path, device: str) -> tuple[Dynamics, Layout, dict]`: the averaged weights in evaluation mode, the layout and the configuration.
- Produces in the script: `train --run NAME --assignment FILE --size S [--fit-dir DIR] [--cache-dir DIR] [--pattern mix|task_X-Y] [--steps N] [--batch B] [--positions-per-step P] [--lr X] [--seed K] [--init CKPT] [--set key=value ...] [--gpu N] [--share-gpu] [--device cpu] [--max-hours H] [--throughput]`. The run directory is `<stage2>/runs/<NAME>`, the store `<stage2>/tokens/<assignment stem>`. It takes the lock, refuses a busy GPU as `run_headroom.py` does, prints the resolved directories before its first write and the last log row at the end. `--throughput` runs the given steps (default 300) without validation or checkpoints and prints `positions_per_s` (median after the first 50 steps) and `gb_per_1000_positions` (peak memory over positions per step); it writes only `<stage2>/checks/throughput_<size>.json`. `--set` overrides a field of `TrainConfig`, of `model` or of `mix` by name; an unknown name exits 2.

- [ ] **Step 1: Write the failing tests**

`test_dyn_patterns.py` (hand-built tasks and a layout that has the given signals):
- `test_the_pattern_of_a_forecast_task`: 1 frame before, 5 after, `track == predict`.
- `test_the_pattern_of_a_group_one_task_predicts_inside_its_frame`: `before == 1`, `after == 0`.
- `test_the_pattern_of_a_long_history_task`: 30 and 20 for 150 ms and 100 ms; 60 before with `history_s=0.3`.
- `test_track_all_adds_the_inputs`.
- `test_states_before_and_after_the_cut`: an input that is not an output is `OBSERVED` before the cut and `CARRIED` after it in both streams; an output that is not an input is `CARRIED` before the cut, `OBSERVED` in the history after it and `PREDICT` in the query; an actuator is `GIVEN` everywhere.
- `test_states_of_a_pattern_that_predicts_inside_its_frame`: history `before - 1` frames, query one.
- `test_a_drawn_pattern_is_well_formed` (500 draws): `predict` not empty and inside `track`; no given signal in `context` or `predict`; `before` and `after` within the mix's limits; with `p_task=1` every draw is a task pattern up to its `given`.
- `test_drawing_is_reproducible_from_the_generator`.
- `test_the_fourteen_task_patterns_use_known_signals` (needs `tokamark`): every name is in `GROUP_OF`; the given signals of all tasks together are `GIVEN_SIGNALS`.

`test_dyn_train.py` (size `tiny`, a store of 3 synthetic shots in `tmp_path`, CPU):
- `test_batch_size_follows_the_position_budget`; `test_a_fixed_batch_wins`.
- `test_a_masked_value_is_not_an_input_of_its_own_prediction`: every value under the loss mask is NaN in the query and its token is `PREDICT`.
- `test_revealed_tokens_are_observed_and_carry_no_loss`; `test_with_full_masking_nothing_is_revealed`.
- `test_missing_data_makes_a_token_absent_and_a_missing_target_carries_no_loss`.
- `test_absent_values_carry_no_loss`: changing the targets of unmasked values does not change the loss.
- `test_the_group_loss_weighs_groups_not_values`: one group of 100 values at loss 1 and one of 1 value at loss 3 give 2; with `"token"`, 103/101.
- `test_a_batch_without_a_masked_value_gives_a_zero_loss_with_gradients`.
- `test_the_loss_is_smallest_at_the_two_hot_target`.
- `test_feedback_replaces_only_tracked_values_after_the_cut`: values before the cut, given values and absent tokens unchanged; with `share=1` every tracked value after the cut differs from the truth for an untrained model.
- `test_feedback_with_share_zero_makes_no_pass` (forward calls counted).
- `test_training_lowers_the_loss_on_a_tiny_store`: 150 steps, one pattern; the mean loss of the last 20 steps is below 0.7 of the first 20.
- `test_a_run_writes_its_files_and_resumes`: after 20 steps the files exist; a second call with `steps=30` continues at step 21.
- `test_a_run_refuses_another_configuration_in_its_directory`.
- `test_training_refuses_a_store_of_another_layout`; `test_init_refuses_a_checkpoint_of_another_layout`.
- `test_load_model_returns_the_averaged_weights`.

`test_dyn_command.py` gains `test_train_refuses_an_unknown_override` (exit 2) and `test_train_prints_the_directories_before_writing` (trainer patched).

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_dyn_patterns.py tests/ignite/mast/test_dyn_train.py tests/ignite/mast/test_dyn_command.py -q -W error`
Expected: an import error for `patterns`.

- [ ] **Step 3: Implement `patterns.py`, `train.py` and the command**

The clip is gathered to the pattern's rows (`layout.values_of`) on the CPU before it goes to the device. Follow `headroom/direct.py` for the lock, the atomic writes and the log.

- [ ] **Step 4: Run the tests**

Run the command of Step 2. Expected: all passed (29 new tests).

- [ ] **Step 5: A trial on the CPU**

With `IGNITE_MAST_ROOT=/models/intermediate/mast/trials/stage2a-task4` and the three-shot store of Task 2's trial rebuilt there: `train --run trial --assignment <root>/stage2/assignments/v0.json --fit-dir ... --cache-dir ... --size tiny --pattern mix --steps 200 --device cpu --set positions_per_step=20000 --set val_every=100 --set n_val_shots=1`. Record the first and last training loss and the wall time in the commit body. This is the only training an implementer runs.

- [ ] **Step 6: Commit**

```bash
git add src/tokamak_foundation_model/ignite/mast/dynamics/{patterns,train}.py scripts/mast/run_dynamics.py tests/ignite/mast/test_dyn_{patterns,train,command}.py
git commit -m "ignite.mast: conditioning patterns, the token loss and the dynamics trainer"
```

### Task 5: the mean roll-out, the harness predictor and the command `score`

After this task a checkpoint of the first run is scored through the benchmark harness.

**Files:**
- Create: `src/tokamak_foundation_model/ignite/mast/dynamics/rollout.py`, `dynamics/predictor.py`
- Modify: `scripts/mast/run_dynamics.py`, `tests/ignite/mast/test_dyn_command.py`
- Test: `tests/ignite/mast/test_dyn_rollout.py`, `tests/ignite/mast/test_dyn_predictor.py`

**Interfaces:**
- Consumes: Tasks 1 to 4; `harness.PredictFn`, `harness.score`; `baselines.mean_predictor(task) -> PredictFn`; 1b's `floor.blind_to_targets(predict) -> PredictFn` (keeps the function's attributes); from `scripts/mast/run_tokenizer.py` (by path): `score_one(task, predict_fn, *, predictor, out_dir, split, label, shots, provenance, records, force=False, num_workers=32, batch_size=64) -> dict` (scores with `history_s=getattr(predict_fn, "history_s", None)`, or returns the kept record), `provenance_of(assignment_path, fit_dir, cache_root, checkpoint=None) -> dict[str, str]`, `split_label(split, n_shots) -> str`.
- Produces in `rollout.py`:
  - `predict_frame(model, history: Frames, frame: Frames, rows: Tensor) -> Tensor`: `(B, w)` values of one query frame: the expected value of every value of every `PREDICT` token, read without told values, and the frame's own values elsewhere.
  - `mean_rollout(model, history: Frames, future: Frames, rows: Tensor) -> Tensor`: `(B, H, w)`. The frames of `future` in time order: `predict_frame` with the history so far, then the frame is appended to the history with its `PREDICT` tokens turned `OBSERVED` and holding the expected values. It recomputes the history at every step; plan 2b's cached step must give the same numbers.
- Produces in `predictor.py`:
  - `window_frames(values: np.ndarray, pattern, layout, rows: np.ndarray, device: str) -> tuple[Frames, Frames]`: from `windows.encode_window`'s array, the history frames and the frames to predict, with the states of `patterns.states` and missing data applied as `train.make_batch` applies it, nothing revealed; `t` counts from the cut.
  - `simulator_predictor(task, checkpoint: Path, *, assignment_path: Path, fit_dir: Path, cache_root: Path, device: str = "cpu", track: str = "outputs", history_s: float | None = None, batch: int = 256, tokenizers: tuple[Layout, Mapping[str, object]] | None = None) -> PredictFn`: loads the model, and the tokenizers (`layout.load_tokenizers` unless given); `ValueError` naming "layout" if their layout's hash is not the checkpoint's. The function encodes the window, rolls out (or predicts inside the frame for a pattern with `after == 0`), decodes with `windows.decode_window` on the CPU and returns physical arrays with the targets' shapes, moving `batch` windows at a time to the device. It reads `batch.inputs` and `batch.actuators` only. A prediction that is not finite raises `RuntimeError` naming the signal. It carries `history_s` (the task's `input_length` if None) and `name`.
- Produces in the script: `score --task T --split val_select|val_report [--n-shots N] (--run NAME [--checkpoint best|last|step_<n>] [--track outputs|all] [--history-s X] [--fit-dir DIR] [--cache-dir DIR] | --baseline mean) [--gpu N] [--share-gpu] [--device cpu] [--force] [--num-workers 32]`. With `--run`, the assignment, fit and cache paths come from the run's `config.json` unless given; the predictor's name is `sim.<run>.<checkpoint>.mean`, with `.all` appended for `--track all`; it is wrapped in `blind_to_targets`; provenance `provenance_of(assignment, fit, cache, checkpoint)` plus `dynamics_code_sha256`; records `run`, `checkpoint_step`, `track`, `history_s`. With `--baseline mean`, `baselines.mean_predictor(task)` under the name `mean`, on the CPU. Scores go through `score_one` into `<stage2>/scores/<task>/<predictor>/<label>/`; a subset is `cache.pick_evenly(tasks.split_shots(split), n)`. `--split test` exits 2 with "the test split is read once, by plan 2c's final table". The last line is `task, predictor, label, NRMSE, shots, windows, seconds`.

- [ ] **Step 1: Write the failing tests**

`test_dyn_rollout.py` (size `tiny`, hand-built layout, CPU):
- `test_predict_frame_keeps_the_given_values_and_fills_the_predicted_ones`.
- `test_the_rollout_feeds_its_own_expected_values_back`: with the forward calls recorded, the history at step `h` holds the values returned for the earlier steps, for the tracked rows.
- `test_the_first_step_equals_one_parallel_pass`: the roll-out's first frame equals the expected values of `model.forward` on that frame with the true history.
- `test_a_later_actuator_does_not_change_an_earlier_frame`: changing the given values of frame `h` and later leaves the frames before `h` exactly equal.
- `test_predicted_values_lie_inside_the_bin_range`.

`test_dyn_predictor.py` (stand-in codecs passed with `tokenizers=`, `toy_task` and hand-built tasks, a `tiny` checkpoint from 5 training steps in `tmp_path`):
- `test_the_prediction_has_the_targets_shapes_and_is_finite` for a forecast, a group-1 and a long-history task.
- `test_the_predictor_does_not_read_the_targets`: NaN targets give the same prediction.
- `test_a_window_with_a_missing_input_signal_still_predicts_finite_values`: one input all NaN, and a window with every input NaN.
- `test_a_missing_actuator_frame_is_absent_not_zero`.
- `test_the_predictor_refuses_tokenizers_that_do_not_match_the_checkpoint` (`ValueError` naming "layout").
- `test_window_frames_and_make_batch_agree_on_states`: same clip and pattern, full masking.
- `test_the_predictor_scores_through_the_harness` (`real_data`, needs `tokamark` and 1b's fit): an untrained `tiny` model on a `v0`-style assignment gives a finite NRMSE from `harness.score` on one validation shot of task 2-1.

`test_dyn_command.py` gains `test_score_refuses_the_test_split` (exit 2), `test_score_names_the_predictor_after_run_checkpoint_and_track`, and `test_score_of_the_mean_baseline_needs_no_run` (`score_one` patched).

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_dyn_rollout.py tests/ignite/mast/test_dyn_predictor.py tests/ignite/mast/test_dyn_command.py -q -W error`
Expected: an import error for `rollout`.

- [ ] **Step 3: Implement `rollout.py`, `predictor.py` and the command**

- [ ] **Step 4: Run the tests**

Run the command of Step 2. Expected: all passed (15 new tests; one skips without the fit).

- [ ] **Step 5: A trial on the CPU**

With the trial root of Task 4: `score --run trial --task task_1-1 --split val_select --n-shots 2 --device cpu --checkpoint last`, the same for `task_2-1`, and `score --baseline mean --task task_2-1 --split val_select --n-shots 2`. Record the three last lines in the commit body with the words "tiny model, 200 steps, a path check".

- [ ] **Step 6: Commit**

```bash
git add src/tokamak_foundation_model/ignite/mast/dynamics/{rollout,predictor}.py scripts/mast/run_dynamics.py tests/ignite/mast/test_dyn_{rollout,predictor,command}.py
git commit -m "ignite.mast: the mean roll-out and the simulator as a harness predictor"
```

### Task 6: training on one task with selection on NRMSE, and the token-path check

**Files:**
- Create: `src/tokamak_foundation_model/ignite/mast/dynamics/checks.py`
- Modify: `src/tokamak_foundation_model/ignite/mast/dynamics/train.py`, `scripts/mast/run_dynamics.py`, `tests/ignite/mast/test_dyn_train.py`, `tests/ignite/mast/test_dyn_command.py`
- Test: `tests/ignite/mast/test_dyn_checks.py`

**Interfaces:**
- Consumes: Tasks 1 to 5; the evaluator's `shots_metrics.csv` (columns `shot_id`, `NRMSE_mean`; a shot's value is the mean of its rows, as `headroom/table.py: _per_shot` reads it) at `<score dir>/<task>/shots_metrics.csv`; `baselines.read_kept_score(out_dir, task, predictor, split) -> tuple[dict | None, str]`; `headroom.table.predictor_name("direct", variant)` and `table.direct_run_dir(stage0, task, variant)` (its `config.json` holds `cfg` with `max_steps`, `batch_size`, `lr`, `val_every`, `val_shots`); 1b's stage-1 predictor names `tok.<a>.<mode>`, `inputs.<a>.<mode>.<variant>`, `raw.<variant>` under `<stage1>/scores/<task>/<predictor>/<label>/`; `paths.stage0_dir()`, `paths.stage1_dir()`.
- Produces in `train.py`: `TrainConfig` gains `select: str = "loss"` (or `"nrmse"`), `select_task: str | None = None`, `select_shots: int = 200`; `train` gains `select_fn: Callable[[Path], float] | None = None`. With `select == "nrmse"`, at every validation step `step_<n>.pt` is written (whatever `save_every` is) and `select_fn` is called on it, its result is logged as `val_nrmse`, and `best.pt` is the checkpoint with the lowest `val_nrmse`. `select == "nrmse"` without a function raises `ValueError`.
- Produces in `checks.py`:
  - `per_shot(score_dir: Path, task: str) -> pd.Series`: indexed by `shot_id`; `FileNotFoundError` if the file is missing.
  - `ratio_ci(ours: pd.Series, reference: pd.Series, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float, int]`: on the common shots, the ratio of the two means, the 2.5% and 97.5% points of a bootstrap over shots (the same drawn shots for both), and the number of shots. `ValueError` if fewer than 95% of the reference's shots are in `ours`.
  - `token_path_verdict(ratios: Mapping[str, tuple[float, float, float]], limit: float = 1.10) -> str`: `"pass"`, `"fail"` or `"marginal"` by rule 3.
  - `token_path_report(stage2: Path, stage1: Path, stage0: Path, assignment: str, direct_variants: Mapping[str, str]) -> dict`: per task (1-1, 1-2): the stored `val_report` score of `sim.scr-S-<assignment>-<task>.best.mean` (or of the `-lowlr` run when it exists), the direct variant's, the ratio and interval, whether the run's `steps`, `batch` and `lr` equal the variant's `max_steps`, `batch_size` and `lr` (a mismatch sets the verdict to `"not comparable"`), the verdict, the score of `sim.pre-S-<assignment>.best.mean` (reported, decides nothing), and rule 4's numbers where stored (each `"not stored"` otherwise, including a score `read_kept_score` does not keep).
- Produces in the script:
  - `train --select nrmse [--select-task T] [--select-shots 200]`: `select_fn` from `simulator_predictor` (the tokenizers loaded once and passed with `tokenizers=`) and `harness.score` on `cache.pick_evenly(tasks.split_shots("val_select"), n)` (for n = 200 these are the direct model's validation shots), writing under `<run>/select/step_<n>/`. `--select-task` defaults to the task of `--pattern`; with `--pattern mix` it is required.
  - `check token-path --assignment FILE --direct task_1-1=VARIANT,task_1-2=VARIANT`: prints the report as a table and the last line `token-path check: <verdict>`; writes `<stage2>/checks/token_path_<stem>.json` and `.md`. Exit 0 for `pass` and `marginal`, 1 for `fail`, 2 if a needed score is not stored, a task has no variant, the verdict is `not comparable`, or the assignment's notes say "provisional".

- [ ] **Step 1: Write the failing tests**

`test_dyn_checks.py`:
- `test_per_shot_is_the_mean_of_the_shots_rows` (a small CSV in `tmp_path`).
- `test_the_ratio_is_paired_by_shot_id`: two series in different shot orders give the ratio of the means on the common shots.
- `test_the_interval_of_two_equal_series_is_one`: `(1.0, 1.0, 1.0)`.
- `test_too_few_common_shots_is_refused`.
- `test_the_verdicts`: both ratios 1.05: `pass`; 1.20 with interval 1.12 to 1.28: `fail`; 1.12 with interval 1.05 to 1.19: `marginal`; 1.08 with interval 1.02 to 1.14 and the other 1.0: `pass`.
- `test_a_missing_score_is_named_not_stored`.
- `test_a_budget_that_differs_from_the_reference_is_not_comparable`.

`test_dyn_train.py` gains `test_selection_on_nrmse_keeps_the_checkpoint_with_the_lowest_value` (a `select_fn` returning 3, 1, 2: `best.pt` is the second) and `test_selection_on_nrmse_without_a_function_is_refused`.

`test_dyn_command.py` gains `test_the_token_path_check_refuses_a_provisional_assignment` and `test_the_token_path_check_exits_by_its_verdict` (report patched).

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_dyn_checks.py tests/ignite/mast/test_dyn_train.py tests/ignite/mast/test_dyn_command.py -q -W error`
Expected: an import error for `checks`.

- [ ] **Step 3: Implement `checks.py`, the selection in `train.py` and the two command changes**

- [ ] **Step 4: Run the tests**

Run the command of Step 2. Expected: all passed (11 new tests).

- [ ] **Step 5: Commit**

```bash
git add src/tokamak_foundation_model/ignite/mast/dynamics/{checks,train}.py scripts/mast/run_dynamics.py tests/ignite/mast/test_dyn_{checks,train,command}.py
git commit -m "ignite.mast: training on one task with selection on NRMSE, and the token-path check"
```

### Task 7: the results page

**Files:**
- Create: `docs/models/ignite-mast-stage2.md`
- Modify: `mkdocs.yml`

**Interfaces:**
- Consumes: the commands of this plan and their outputs; the stage-1 page's form if it exists, else the stage-0 page's.
- Produces: a page with the status line "not yet measured" and the sections "What stage 2 builds", "Frames against the benchmark's windows" (this plan's measured facts), "The first run" and "First scores" (empty tables for J3 to J5; a run on a provisional assignment is called a path check), "Token-path check" (rules 3 and 4, marked "fixed before the measurement, 2026-11-22", and an empty table), "Where this departs from the design" (the departures listed under "Decisions of the plan's author") and "Limits" (five column offsets per shot in training; the read without told values; clips of at most 64 frames). Each empty cell names the stored file it will come from.

- [ ] **Step 1: Write the page and add its nav line after the stage-1 page's (or the stage-0 page's)**

- [ ] **Step 2: Check the docs build**

Run: `mkdir -p /models/intermediate/mast/trials/stage2a-docs && uvx --with-requirements requirements-docs.txt --from 'mkdocs<2' mkdocs build -d /models/intermediate/mast/trials/stage2a-docs/site 2>&1 | grep -i "ignite-mast-stage2"`
Expected: no line naming the page as a warning or error. If `uvx` cannot fetch its packages, check every relative link by hand and say so in the commit body.

- [ ] **Step 3: Commit**

```bash
git add docs/models/ignite-mast-stage2.md mkdocs.yml
git commit -m "ignite.mast: docs, the stage-2 results page with the first-run and token-path sections"
```

---

## The shortest path to a trained and scored model

The first deliverable needs tokens of the training shots, a dynamics model conditioned on actuators, a training loop, a roll-out and a predictor the harness can score: Tasks 1 to 5, one reviewer's gate each. Tokens come from a store (one build job, J1), not from tokenizers inside TokaMark's loader, which delivered 668 to 1,334 training windows per second on the short tasks (stage 0, measured 2026-11-22) and never a window longer than 250 ms or wider than one task's signals; pretraining needs clips of all 39 signals. The window encoder (Task 1) comes before the store because the store's check against the benchmark (J2) needs it, and a disagreement found after training started would cost the run. The model (Task 3) needs only Task 1. Training starts after Task 4; its checkpoints are scored once Task 5 is done. Nothing of the first deliverable waits for the cached step, sampling, the three sizes, adaptation, the token-path verdict or the page.

What the first run needs from stage 1b:

| Needed | From plan 1b | For |
|---|---|---|
| `assign`, `fit` and the three codecs as code | Tasks 1 to 5 and 7 | this plan's Task 2 (Tasks 1 and 3 use stand-ins) |
| the fit of ranges and bases | Task 8, job K1 | any store, also `v0` |
| the assignment `v1` | Task 9, jobs K2 and K3 | the store of the first run |
| the choice between `v1` and `v1r` | job K4 | the store of the token-path runs and all later runs |
| `floor.blind_to_targets`, `score_one`, `provenance_of`, `split_label` | Tasks 6 and 8 | this plan's Task 5 |
| the gate's verdict | Task 10, job K7 | plan 2c's size-M and size-L runs, not the first run |

So the first run starts when this plan's Tasks 1 to 4 are merged, an assignment exists (`v1` after 1b's K3, or `v0` after K1 and J0), and J1 and J2 have run on it (J2 needs no store and runs beside J1). It does not wait for K4: if K4 chooses `v1r`, the store is rebuilt for the token-path runs (another J1, 10 to 40 minutes, inferred) and the first run stays a pipeline proof. Estimate, not a measurement: about 1 h after `v1` exists if Tasks 1 to 4 are merged by then; about 8 to 12 h if this plan starts only when `v1` exists (four tasks at about 2 to 3 h each with their reviews, then the hour of jobs).

## Controller jobs

From `/home/nvidia/FusionAIHub` with `IGNITE_MAST_ROOT` unset. `RD="scripts/mast/py.sh scripts/mast/run_dynamics.py"`, `S1=/models/intermediate/mast/stage1`, `S2=/models/intermediate/mast/stage2`. `A` is an assignment file: `$S2/assignments/v0.json` (open decision 1), `$S1/assignments/v1.json`, or the one K4 names. Logs go to `$S2/logs/`. Every duration is the author's inference from stage-0 timings (cache build 225 s; direct model 200 windows per second of about 5,400 tokens; 90,000 steps on 1-1 in 7 h 7 min), not a measurement of this code; J3 replaces the training ones.

| Job | After | Command | GPU | Duration | What is read from the output |
|---|---|---|---|---|---|
| J0 provisional | 1b K1, Task 2 | `$RD provisional --name v0` | none | seconds | Rows and values per frame (expected 284 and about 1,500). |
| J1 store | Task 2, an assignment | `CUDA_VISIBLE_DEVICES="" $RD tokens --assignment A --workers 64` | none | 10 to 40 min; about 37 GB | Per split: shots and frames (training: 8,963 shots, about 1.10 million phase-0 frames); shots without data (none expected); seconds per shot. Exit 1 stops. |
| J2 store check | Task 2, an assignment | `CUDA_VISIBLE_DEVICES="" $RD tokens --assignment A --check` | none | 5 to 15 min | The last line: 0 unequal, 0 raw unequal; the share of band frames that coincide (about a half, inferred); windows off the grid (under 1% expected). Rule 1. |
| J3 throughput | Task 4, J1 | `$RD train --run thr --assignment A --size S --throughput --gpu N`, then `--size M --set positions_per_step=270000` and `--size L --set positions_per_step=150000` | one, not shared | 5 to 10 min each | `positions_per_s` and `gb_per_1000_positions` per size. Inferred: 0.5, 0.13 and 0.045 million positions per second; 0.23, 0.52 and 0.94 GB per 1,000 positions. A size above 250 GB is run again at half the positions. The three files go to plan 2c. |
| J4 first run | Task 4, J1, J2 | `$RD train --run pre-S-<stem of A> --assignment A --size S --pattern mix --steps 20000 --gpu N` | one | 7 h (inferred: 20,000 steps of 600,000 positions) | `train_log.csv`: the loss falls; `val_mean` and the six `val_<pattern>` are finite and fall; `feedback_share` reaches 0.75 at step 2,000 without the validation loss jumping above twice its earlier value. A non-finite loss stops the run and is reported with its step. |
| J5 first scores | Task 5, a checkpoint of J4 | `$RD score --run pre-S-<stem> --checkpoint last --task T --split val_select --n-shots 100 --gpu N --share-gpu` and `$RD score --baseline mean --task T --split val_select --n-shots 100`, for tasks 1-1, 1-2, 2-1 | shared with J4 | 2 to 10 min each | Rule 2: finite NRMSE, with the mean baseline beside it. Repeated at later checkpoints at the controller's choice; the scores go to the page's "First scores" with the assignment named. |
| J6 token-path runs | Task 6, J1 and J2 on the store K4 names | `$RD train --run scr-S-<stem>-task_1-1 --assignment A --size S --pattern task_1-1 --steps 90000 --batch 256 --lr 3e-4 --set val_every=2500 --select nrmse --select-shots 200 --gpu N`; for 1-2: `--steps 30000 --batch 256 --lr 1e-4`, otherwise the same | one each, may be shared | 1-1: 2 to 4 h; 1-2: 1 to 2 h (inferred) | `val_nrmse` per validation and the step of `best.pt`. Rule 3: a diverged run is repeated once at a third of the learning rate as `<run>-lowlr`. |
| J7 token-path scores | J6 | `$RD score --run scr-S-<stem>-<task> --checkpoint best --task <task> --split val_report --gpu N --share-gpu` for both tasks; the same for `pre-S-<stem>` (best checkpoint of J4 on the same store) for both tasks | shared | 5 to 20 min each | The last lines. |
| J8 token-path check | J7, 1b K5 and K6 (or K4) | `$RD check token-path --assignment A --direct task_1-1=s90k,task_1-2=lr1e4` (the variants the headroom table chooses on the day; `scripts/mast/py.sh scripts/mast/run_headroom.py --table-all`) | none | seconds | The verdict and rule 4. `pass`: plan 2c's size-M and size-L runs may start; `marginal`: the owner first; `fail`: stop and report the three numbers per task. |

After J5 and after J8 the controller fills the tables of `docs/models/ignite-mast-stage2.md` and commits (`ignite.mast: stage-2 first run measured`, `ignite.mast: stage-2 token-path check measured`), with the numbers, their date, split and shot counts in the body.

## What this plan leaves to later plans

- The cached step, sampling, the simulator in physical units and the causality test: plan 2b. This plan's roll-out recomputes the history at every step: affordable for groups 1 to 3, too slow for the 20 steps of group 4 at the larger sizes.
- The three sizes with their budgets, adaptation, the from-scratch column, the pretraining check, noise and the first complete table: plan 2c.
- A fix of the token path if the check fails (rule 4).
- Encoding spectrogram frames relative to each clip's own cut in the loader, which would remove the five column offsets per shot; worth doing only if task 4-5 or the group-4 inputs are shown to suffer from it.
- Runs on 1c's learned tokenizers (`v2`, `v2c`): the same commands with another assignment file; the index says when.
- Ablation runs of the switches (stage 3).
