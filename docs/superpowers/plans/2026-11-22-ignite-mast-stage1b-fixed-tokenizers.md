# IGNITE on MAST, stage 1b: fixed tokenizers, linear baselines, the gate and the floors. Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> Approved by the owner on 2026-11-22 (16:17 UTC, "yes i approve the plan. please carry it out"). The open decisions for the owner are taken as the author recommends until the owner says otherwise. The stage-0 numbers quoted in this text are those of the day it was written; `docs/models/ignite-mast-stage0.md` holds the current ones. Written 2026-11-22 by the plan's author. Nothing here has been run except the small read-only looks under "Measured facts". It is the second of three stage-1 plans: 1a (`2026-11-22-ignite-mast-stage1a-headroom-seven.md`) runs first and in parallel with this one; 1c (`2026-11-22-ignite-mast-stage1c-learned-tokenizers.md`) builds on this one.

**Goal:** Each of the 39 cached signals has a tokenizer without learned weights (a cosine transform per frame, a principal-component basis, or a band transform), a frozen assignment that says how many values it keeps, and a measured representation floor per task in TokaMark's own units, so that stage 2 can start on these tokenizers alone and every learned tokenizer of 1c has a baseline to beat.

**Architecture:** A new package `ignite/mast/tokenizers/`. A `Codec` turns one signal's window, in physical units, into token values of shape `(N, F, n_tok, n_val)` (F frames of 5 ms) and back. Three codecs have no learned weights: `FrameDCT` (slow series and the trend path of high-rate channels), `LinearCodec` (profiles and the flux map) and `BandDCT` (Mirnov spectrograms). A floor predictor decodes the encoded true targets and is scored by the stage-0 harness, which is the measurement stage 0 made for TokaMind's representation. A command fits ranges and bases on training shots, runs ladders over the number of kept values, writes the assignment by rules fixed in this plan, and builds the gate table.

**Tech Stack:** Python 3, numpy, scipy (`scipy.fft`), pandas, h5py, pytest; torch only through the stage-0 direct model used as an instrument; TokaMark's loader and evaluator through `harness.score`.

**Spec:** `docs/models/ignite-mast-design.md`: sections "Tokenizers" (common features, the family table, the fixed transform, the two high-rate paths, the gate, what blocks), "Time base", "Task 4-5" under the benchmark protocol, and row "1. Tokenizers" of "Order of work".

## Open decisions for the owner

These change the goal, the bar or what is published. The plan does not decide them; each has the author's recommendation.

1. **Task 4-5 and frame statistics.** A rough look on 40 training shots (see "Measured facts") says three things about the Mirnov spectrogram task as TokaMark scores it. (a) Its error sits in the lowest frequencies: 99.8% of the mean baseline's squared error for the toroidal-field probes and 77% for the poloidal ones is below 31 kHz, and the first bin alone (0 to 1 kHz) holds 86% and 37%. (b) A perfect prediction of every 5 ms frame's mean spectrum would still score about 74% of the mean baseline, because the targets are single 1 ms periodograms that scatter inside a frame, and that scatter does not compress in time. (c) TokaMind's own representation floor is lower (0.36 on `val_report`) because it keeps detail inside 5 ms. The design scores 4-5 from frame statistics. With that choice the floor of our tokenizer on 4-5 is about three quarters of the published scores, and the "fifth of the best published score" target cannot be met for this task by any choice of K. What stage 0 has measured since (2026-11-22, `val_report`, 351 shots): the direct model, which is trained on the 1 ms columns, scores 0.6475, a tie with the best published score (0.6510, itself 0.002 under the published mean baseline of 0.653), and is below the mean baseline on the same shots by a paired 0.0377 (95% interval 0.0298 to 0.0461, carried by a minority of the shots). So about 5% is predicted by anything so far, while a perfect prediction of the frame means would take off about 26%: frame statistics do not stand between this project and a win on 4-5, they stand between it and the "fifth of the best published score" target. Whether the direct model's 5% sits in the frame means or inside the frames is not measured. Recommendation: keep frame statistics, as designed; report the frame-oracle floor beside the score; do not add 1 ms detail to the tokens unless a model is shown to predict it; and expect a small margin on 4-5, not a large one. The owner may instead want the task met on its own terms (per-column values for the lowest bins, about 120 more values per frame), or dropped from the tokenizer work.
2. **The linear baseline may stand as the tokenizer.** If the principal-component basis meets the fidelity target for the boundary and the flux map (a rough look says it does with 24 to 48 values), 1c's learned profile and map tokenizers are a comparison and not a need. Recommendation: stage 2 starts on the linear baselines; the learned ones replace them only by the rule in 1c.
3. **What is published about misses.** Signals that miss the fidelity target (expected: the soft X-ray cameras and D-alpha, where the missing part is fast fluctuation) go into the results page with their numbers and with the pass-through measurement that says what the miss costs on predictions. Recommendation: publish both numbers per signal; do not raise K to meet the target where the pass-through cost is under 1%.

## Global Constraints

- Machine: node13, aarch64, 4 GPUs of 286 GiB, 352 cores, 1.4 TB RAM. Data and runs go under `/models/intermediate/mast/`, never into the repository. Nothing is ever written under `/models/tokamind/`. Nothing is uploaded anywhere. Temporary files do not go to `/tmp`; use a directory under `/models/intermediate/mast/trials/`.
- Work in the main checkout `/home/nvidia/FusionAIHub` on branch `nvidia` (a git worktree run on this machine imports the main checkout's code). Commit locally. Never `git push`, `reset --hard`, `clean`, `stash`, `checkout -- <file>` or `commit --amend`. `git add` only this task's files, by path; never `-A` or `.`. `.gitignore` excludes `*.h5`, `*.pt`, `*.npy`, `*.npz`: tests build their fixtures in memory or in `tmp_path`.
- Commit subjects start with `ignite.mast: `. The body says what changed and the measured facts behind it. The message ends with `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`.
- Two environments. `scripts/mast/py.sh` is `/models/envs/ignite-mast/bin/python -I` and has `tokamark`. `pixi run --frozen -e nvidia ...` (always `--frozen`) has no `tokamark`, so tests that need it skip there. `pyproject.toml` and `pixi.lock` do not change. No new dependency. No scikit-learn.
- No module under `src/tokamak_foundation_model/ignite/mast/` imports `tokamark`, `MAST_tools`, `mmt` or `zarr` at import time. A test that needs `tokamark` starts with `pytest.importorskip("tokamark")`; a test that needs the data carries `@pytest.mark.real_data`.
- Hashed files do not change in this plan: `ignite/mast/harness.py`, `tasks.py`, `baselines.py`, `headroom/floor.py`, `headroom/cheap.py`, `headroom/direct.py`. Every stored score carries their sha256, and one changed byte marks all stored scores stale.
- Files under `src/tokamak_foundation_model/ignite/` outside `ignite/mast/` do not change; they may be imported.
- Tests run with `-W error` on the CPU: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/<file> -q -W error`, and the whole directory with `CUDA_VISIBLE_DEVICES="" pixi run --frozen -e nvidia python -m pytest tests/ignite/mast -q -W error`. An implementer never runs the whole `tests/ignite` suite (some of its tests use a GPU).
- An implementer runs no fit over the whole cache, no scoring pass over a split and no training, and uses no GPU. A trial is at most a few shots on the CPU and writes under `/models/intermediate/mast/trials/<task name>/` (set `IGNITE_MAST_ROOT` to it). Long jobs are the controller's (section "Controller jobs").
- NRMSE always means TokaMark's evaluator through `harness.score`. Everything fitted is fitted on training shots, chosen on `val_select` and reported on `val_report`. Nothing in this plan reads the test split: the ranges and bases are fitted, so our floors are fitted quantities.
- A subset of shots is always spread evenly over the sorted shot list (`cache.pick_evenly`): the first 200 shots by number are one old campaign and do not track the rest.
- Units. "Physical" is the unit of the data. "Model units" are the cache's (`transform.to_model`). "Linear units" are defined in Task 1. "TokaMark units" are `(x - SignalSpec.mean) / SignalSpec.std`, one constant pair per signal. A function's docstring names the unit of every array it takes and returns.
- Missing data is NaN and stays NaN; a count of zero examples is reported as "no data", never as an error of 0.
- Read files in parts (at most 150 lines per read; grep first). Never print arrays or long logs.
- Docs are part of a change. A number is written with its date, its split, the number of shots, its limits and whether it is measured or inferred.

## Review Focus

- A frame in which some samples are missing, and a frame in which all are: the first is encoded from its valid samples, the second has no token, and the floor predictor still returns a finite number wherever the harness will look. [Task 3, `test_a_partly_valid_frame_falls_back_to_the_mean_of_its_valid_samples`, `test_a_frame_without_a_valid_sample_has_no_values`; Task 6, `test_the_floor_is_finite_where_the_target_is_missing`]
- A Thomson profile with missing points (the usual case: 68% of profiles in a 96-shot look), with fewer valid points than kept components, or with none. [Task 4, `test_a_profile_with_missing_points_is_encoded_from_its_valid_points`, `test_fewer_valid_points_than_components_still_gives_finite_values`]
- A value beyond the fitted range (a validation shot more extreme than every training shot): it reads as the outermost bin centre in the `ranged` and `nearest` modes, not as NaN and not wrapped. [Task 2, `test_a_value_outside_the_range_reads_as_the_outermost_centre`]
- A window whose length is not a whole number of frames (98 spectrogram columns for 20 frames: 4 or 5 columns per frame): encode and decode return exactly the window's length. [Task 1, `test_frame_index_of_spectrogram_columns`; Task 5, `test_a_window_of_98_columns_decodes_to_98_columns`]
- A signal that a shot does not have (`cache.read_shot` returns None for it: absent, or refused for a sampling-rate mismatch as the soft X-ray cameras are in 76 shots) during the fit: skipped and counted, never turned into zeros. [Task 7, `test_the_fit_skips_and_counts_a_shot_without_the_signal`]

## Measured facts this plan relies on (2026-11-22)

All "rough looks" below were made by the plan's author on the CPU from cache v1, on training shots spread evenly over the campaigns, as pooled RMSE over valid samples divided by TokaMark's std. They are not harness scores and they set expectations only; the ladders of this plan replace them.

- Cache v1: `/models/intermediate/mast/cache/v1/` holds `manifest.json` (keys used here: `stats_sha256`, `tokamark_revision`), `registry.csv`, `stats.npz` and `shots/<shot>.h5` for 11,188 shots (8,963 train, 1,115 val, 1,110 test). 39 signals. Families by `registry.family_of`: slow 28 signals with 129 channels (one sample per frame for the 15 EFIT signals, 20 for the 0.25 ms signals, 25 for the 0.2 ms magnetics); high-rate 6 signals (saddle coils 12 channels, the two soft X-ray cameras 18 each and D-alpha 3, all at 0.02 ms, 250 samples per frame; and the two Mirnov signals, which the cache holds as TokaMark's spectrograms `(3, 256, T)` at 1.024 ms); profile 4 signals (`equilibrium-lcfs_r`, `lcfs_z` with 170 points; `thomson_scattering-t_e`, `n_e` with 120); map 1 signal (`equilibrium-psi`, 65 by 65).
- 15 signals are compressed with asinh in the cache (`manifest.json: asinh_signals`): the three betas, `bphi_rmag`, `x_point_r`, `x_point_z`, the line density, the two Mirnov spectrograms, the saddle voltage, the coil voltage, the two soft X-ray cameras, D-alpha and Thomson `t_e`.
- What each task predicts and the best published score (`tasks.best_published`): 1-1 EFIT scalars, 0.0941; 1-2 boundary, 0.0412; 1-3 flux map, 0.1156; 2-1 coil currents, solenoid current, plasma current (100 samples at 0.25 ms) and EFIT scalars (5 frames), 0.0880; 2-2 boundary, 0.0370; 2-3 flux map, 0.0828; 3-1 Thomson, 0.3311; 3-2 D-alpha and soft X-rays (2,500 samples), 0.2522; 3-3 Thomson and betas, 0.2147; 4-1 and 4-2 soft X-rays (5,000 samples), 0.2122 and 0.2140; 4-3 `magnetic_axis_z`, 0.0691; 4-4 plasma current (400 samples), 0.3116; 4-5 Mirnov spectrograms (98 columns), 0.6510. Magnetic probes, flux loops, saddle coils, coil voltages, gas, beam power, the requested waveforms and the line density are inputs or actuators only.
- TokaMind's representation floor (stage 0, `val_report` / test): 1-2 0.0239 / 0.0236, 2-1 0.0037 / 0.0038, 3-1 0.0284 / 0.0279, 4-1 0.0353 / 0.0369, 4-4 0.0351 / 0.0349, 4-5 0.3605 / 0.3366; 1-1 0.0000 on test.
- Rough look, cosine truncation per frame (24 shots, transform taken in model units), error with K = 1, 2, 4, 8 kept values: coil currents 0.078, 0.030, 0.013, 0.006; plasma current 0.076, 0.038, 0.016, 0.009; the 40 `ccbv` probes 0.071, 0.032, 0.014, 0.008; flux loops 0.030, 0.008, 0.003, 0.002; beam power 0.053, 0.034, 0.025, 0.014; soft X-ray lower camera 0.098, 0.081, 0.069, 0.065; upper camera 0.105, 0.128, 0.079, 0.060; D-alpha 0.221, 0.155, 0.122, 0.091. For the upper camera K = 2 is worse than K = 1: a truncation in asinh units is not a projection in physical units.
- Rough look, missing data (24 shots): a frame is partly valid in under 1% of frames for the line density, the soft X-ray cameras, D-alpha and beam power, and never elsewhere among series. The boundary and the flux map are wholly valid or wholly missing. Thomson profiles (96 shots): 10 of 5,562 complete, 1,756 wholly missing, 3,796 partly valid, most with 75 to 97% of the 120 points.
- Rough look, principal components (fitted on 72 shots, tested on 24, complete profiles only), error with 16, 32, 48, 64 values: `lcfs_r` 0.0093, 0.0072, 0.0058, 0.0046; `lcfs_z` 0.0109, 0.0078, 0.0064, 0.0051; flux map 0.0180, 0.0064, 0.0042, 0.0033. A basis fitted on all training shots should be better.
- Rough look, Mirnov spectrograms (40 shots, of which 19 and 36 have the two signals; whole shots, not windows). Mean baseline 0.915 and 0.573 (toroidal-field probes, poloidal probes). Each 5 ms frame replaced by its mean spectrum (all 256 bins): 0.676 and 0.422, which is 74% of the mean baseline for both. Each 100 ms window replaced by its mean spectrum: 0.902 and 0.554. Share of the mean baseline's squared error below 31 kHz: 0.998 and 0.767; in the first bin alone: 0.860 and 0.367. Keeping the first 2, 4, 8, 16, 32 cosine coefficients across the 32 bins of the lowest band (other bands whole), in linear power: 0.877, 0.833, 0.760, 0.688, 0.676 and 0.532, 0.512, 0.477, 0.438, 0.422. Keeping time detail inside the frame for the lowest band (2, 3, 5 of 5 time coefficients, all 32 bins): 0.539, 0.358, 0.014 and 0.349, 0.277, 0.158: no compression in time. Truncating in the cache's per-bin whitened units instead of linear power: errors of 0.93 and up to 74, because the per-bin scale varies by a factor of 1,000 between neighbouring bins near 0 Hz.
- `scripts/mast/run_headroom.py` stores direct-model runs at `/models/intermediate/mast/stage0/direct/<task>/<variant>/best.pt` with `config.json`; `direct.direct_predictor(task, checkpoint, device)` rebuilds the predictor. Floor scoring of task 4-1 over 1,110 test shots took 4,131 s; the cache build over all shots took 225 s.

## Decisions of the plan's author, and where they leave the design

- **Transform domain (departs from the design, "Tokenizers", the sentence "after the compression and standardization of the data path, the token holds the first K cosine-transform coefficients").** The cosine transform is taken in linear units by default, not in the asinh-compressed units, and the compression moves into the spacing of the bin centres. Reason: measured above, a truncation in asinh units can get worse with more coefficients in physical units, and a mean taken on a compressed scale is biased low. The domain is a per-signal switch and both are measured by the ladder; the lower floor in TokaMark units wins. This also answers stage 0's open question whether asinh about zero is the right compression, by measurement.
- **Trend path (the design says "low-pass filtered and decimated to the slow family's rate").** No separate filter: a high-rate channel's 250 samples per frame go through the same cosine transform, and keeping the first K coefficients is the low-pass filter. One code path, exact at frame edges. The design leaves the filter to the builder.
- **Bands (the design says "A 50 kHz channel has one band and a 500 kHz channel has ten").** A band is 32 of TokaMark's bins, 31.25 kHz, so the 256 bins make 8 equal bands. The design leaves the band definition to the builder.
- **The band transform works in linear power, not in whitened or log power.** Measured above. The design asks for expected linear power per bin per frame.
- **The fidelity target is not applied to task 4-5** (departs from "reconstruction NRMSE at most a fifth of the best published score for that task"). Measured above: frame statistics cannot meet it. Open decision 1.
- **A miss is not always closed by more coefficients** (departs from "It has to be closed before the first complete table: by more coefficients for a fixed transform"). Where no rung up to 16 values meets the target, the missing part is fluctuation inside a frame (expected for the soft X-ray cameras and D-alpha, see the rough look above), and closing it would take most of the 250 samples per frame. Rule 3 below chooses K by what the miss costs a prediction instead. Open decision 3.
- **Three read-outs of a token value.** `unquantized` is the value itself; `ranged` is the value limited to the range of its bin centres, which is the most a two-hot target over those bins can say; `nearest` is the nearest bin centre, which is what a single sampled token says. The design measures the gate "decoding from unquantized values"; this plan chooses K on `unquantized` and judges the fidelity target on `ranged`, and reports all three.
- **Number of bins: 64 per value, asinh-spaced** (the design says "a fine scale of ordered bins" and leaves the number to the builder). Stage 2 may change it; the assignment records it.
- **The "own mean" reference of the gate.** For a series it is the frame mean (K = 1), for a band the band's mean power (K = 1), for a profile or map the mean profile of the training shots (0 components). A tokenizer with more values than that must beat it or it blocks.
- **One boundary basis per coordinate.** `lcfs_r` and `lcfs_z` get separate bases. A joint (R, Z) basis would need fewer values and is left to a later plan.
- **Results tree.** `/models/intermediate/mast/stage1/`: `fit/v1/`, `assignments/`, `ladders/`, `scores/<task>/<predictor>/<split>/`, `gate/`.

## Which number decides what

Stated before any number exists. "Floor" is the NRMSE of decode(encode(true targets)) through `harness.score`, per signal from the evaluator's per-signal table.

1. **Fidelity target per signal:** 0.2 times the smallest best published score among the tasks that predict the signal. The EFIT scalars of tasks 1-1 and 2-1, the three betas among them, 0.0176 (task 3-3 also predicts the betas, at a larger score); `magnetic_axis_z` 0.0138 (task 4-3); coil currents, solenoid current and plasma current 0.0176; boundary 0.0074; flux map 0.0166; Thomson 0.0429; D-alpha 0.0504; soft X-rays 0.0424. None for the Mirnov spectrograms.
2. **K by the ladder.** For each signal the ladder is scored on 100 `val_select` shots spread evenly. K is the smallest rung whose `unquantized` floor is at or under the signal's target. Every domain of the signal is run (a series: linear units, and asinh units if the cache compresses it; a profile or map: TokaMark's units, linear units, and asinh units if compressed) and the domain with the smaller K wins; on a tie the earlier one in that order. The ladder tasks are 1-2 (boundary), 1-3 (flux map), 2-1 (currents), 3-1 (Thomson), 3-2 (D-alpha and soft X-rays) and 4-5 (Mirnov); a signal with one sample per frame needs no ladder (K = 1 is exact).
3. **If no rung meets the target,** K is the smallest rung whose pass-through cost is at most 1%: the direct model's prediction for a task that predicts the signal is passed through the tokenizer and scored, and the cost is that score over the raw prediction's score, minus 1, per signal on the same shots. The miss is recorded with both numbers. If the task has no usable direct model (1a marks it stale, unfinished or not competitive), K is the largest rung and the miss is recorded.
4. **Signals no task predicts** (magnetics, saddle coils, coil voltages, gas, beam power, requested waveforms, line density): K is the smallest rung whose reconstruction error on 100 `val_select` shots, computed from the cache in TokaMark units (per shot RMSE over valid samples, mean over shots), is at or under 0.0176. The input round-trip is then measured on the same 100 `val_select` shots: the direct models of tasks 1-1, 1-2, 2-1 and 4-1 are scored with every input and actuator passed through its tokenizer, and raw. What decides is the added error `sqrt(max(inputs**2 - raw**2, 0))` over the task's best published score: above 0.2 on any of the four raises K of these 11 signals by one rung, once (the first assignment is kept beside the raised one), and is recorded for stage 2's token-path check. For a model at the best published score this is a cost of 2%; written this way it does not depend on how good the direct model is, which matters because two of the four are behind the published scores (stage 0, `val_report`: 1-1 0.1361 against 0.0941, 2-1 0.1523 against 0.0880; 1-2 is ahead at 0.0362 with the variant `lr1e4`; 4-1 ties at 0.1952). It assumes that the two errors add in quadrature. The plain ratio `inputs / raw - 1` is reported beside it. The round-trip is reported again on `val_report` in the confirmation.
5. **Mirnov spectrograms.** The reference is the frame oracle: all 32 coefficients in every band. Step a: the lowest band with 4, 8, 16, 32 coefficients, the others whole. Step b: bands 1 to 7 with a common 1, 2, 4, 8, 16, 32, the lowest band at its choice. In step a, K is the smallest rung within 2% (relative, for both signals) of the frame oracle; in step b, the smallest rung within 2% of step a's chosen floor. The benchmark cannot see the upper bands (they hold 0.2% and 23% of the squared error), so step b has a second condition: K is at least the smallest rung whose spectral contrast ratio is 0.9 or more (`ignite.gate.spectral_contrast_ratio` of the decoded against the true frame-mean spectra, in model units, bins 32 to 255, on the same 100 shots from the cache). The threshold 0.9 is the author's; it keeps mode lines above 31 kHz in the tokens although no benchmark task rewards them. The frame oracle, the mean baseline on the same shots and TokaMind's floor go to the owner the day they exist (open decision 1).
6. **Confirmation.** The chosen assignment is scored on all `val_report` shots for the 14 tasks in the three read-outs. The gate judges the fidelity target on `ranged`. If `ranged` is more than 1% above `unquantized` for a signal, its range is too narrow: reported, not fixed here.
7. **What blocks stage 2 for a signal:** a tokenizer with more than the reference's values that does not beat the "own mean" reference. Nothing else blocks; a miss of the fidelity target is recorded with its number.

## File structure

```
src/tokamak_foundation_model/ignite/mast/paths.py                 (modify: stage1_dir)
src/tokamak_foundation_model/ignite/mast/tokenizers/__init__.py   (create: docstring, code_sha256)
src/tokamak_foundation_model/ignite/mast/tokenizers/frames.py     (create: frames, linear units)
src/tokamak_foundation_model/ignite/mast/tokenizers/grid.py       (create: ordered bins)
src/tokamak_foundation_model/ignite/mast/tokenizers/codecs.py     (create: Codec, FrameDCT, LinearCodec, BandDCT)
src/tokamak_foundation_model/ignite/mast/tokenizers/floor.py      (create: the three instruments as PredictFn)
src/tokamak_foundation_model/ignite/mast/tokenizers/assign.py     (create: settings, ladders, selection rules)
src/tokamak_foundation_model/ignite/mast/tokenizers/fit.py        (create: ranges and bases from the cache)
src/tokamak_foundation_model/ignite/mast/tokenizers/ladder.py     (create: ladder passes, cache-side errors, the choice of K)
src/tokamak_foundation_model/ignite/mast/tokenizers/gate.py       (create: the stage-1 gate)
scripts/mast/run_tokenizer.py                                     (create: fit, score, ladder, assign, gate)
tests/ignite/mast/test_tok_{frames,grid,codecs,floor,assign,fit,ladder,gate,command}.py   (create)
docs/models/ignite-mast-stage1.md, mkdocs.yml                     (create, modify)
```

### Task 1: the package, `stage1_dir`, frames and linear units

**Files:**
- Modify: `src/tokamak_foundation_model/ignite/mast/paths.py` (add one function after `stage0_dir`)
- Create: `src/tokamak_foundation_model/ignite/mast/tokenizers/__init__.py`, `src/tokamak_foundation_model/ignite/mast/tokenizers/frames.py`
- Test: `tests/ignite/mast/test_tok_frames.py`; one test appended to `tests/ignite/mast/test_paths.py`

**Interfaces:**
- Consumes (existing): `paths.mast_root() -> Path`; `transform.ChannelStats(compress: str, scale: np.ndarray, mean: np.ndarray, std: np.ndarray)` where `compress` is `"none"` or `"asinh"` and the arrays have the signal's `values_shape`; `transform.to_model(x, st)` maps physical `(*values_shape, T)` to `(c(x) - mean) / std` with `c(x) = x` or `asinh(x / scale)`; `headroom.cheap.FRAME_S = 0.005` and `cheap.frame_len(dt) = max(1, round(FRAME_S / dt))`.
- Produces:
  - `paths.stage1_dir() -> Path`: `mast_root() / "stage1"`.
  - `tokenizers.code_sha256() -> str`: sha256 over the bytes of every `*.py` file directly in the `tokenizers` package directory, in sorted name order. It does not descend into sub-packages: plan 1c adds `tokenizers/learned/`, which carries its own hash, so that adding or changing learned-tokenizer code does not mark the scores of this plan stale.
  - `frames.FRAME_S = 0.005`; `frames.frame_len(dt: float) -> int`.
  - `frames.frame_index(n: int, dt: float) -> np.ndarray`: `(n,)` int64, sample `j` belongs to frame `floor((j + 0.5) * dt / FRAME_S)`, counted from the first sample.
  - `frames.to_linear(x: np.ndarray, st: ChannelStats) -> np.ndarray`: physical `(..., *values_shape, T)` to linear units, float64, NaN kept. Linear units are `(x - mean) / std` for `compress == "none"` and `x / scale` for `"asinh"`.
  - `frames.from_linear(u: np.ndarray, st: ChannelStats) -> np.ndarray`: the inverse.
  - `frames.linear_from_model(z: np.ndarray, st: ChannelStats) -> np.ndarray`: model units to linear units (`z` for `"none"`, `sinh(z * std + mean)` for `"asinh"`).
  - `frames.model_from_linear(u: np.ndarray, st: ChannelStats) -> np.ndarray`: the inverse.

- [ ] **Step 1: Write the failing tests** (`tests/ignite/mast/test_tok_frames.py`)

```python
"""Frames of 5 ms and the linear units the transforms work in."""

import numpy as np

from tokamak_foundation_model.ignite.mast import paths, tokenizers, transform
from tokamak_foundation_model.ignite.mast.headroom import cheap
from tokamak_foundation_model.ignite.mast.tokenizers import frames

NONE = transform.ChannelStats(compress="none", scale=np.ones(2), mean=np.array([1.0, -2.0]), std=np.array([2.0, 4.0]))
ASINH = transform.ChannelStats(compress="asinh", scale=np.array([0.5, 2.0]), mean=np.array([0.1, 0.2]),
                               std=np.array([1.5, 0.7]))
X = np.array([[1.0, 3.0, np.nan], [-2.0, 2.0, 6.0]])           # 2 channels, 3 samples, physical


def test_frames_are_stage0s_frames():
    assert frames.FRAME_S == cheap.FRAME_S == 0.005
    for dt, n in ((0.005, 1), (0.00025, 20), (0.0002, 25), (0.00002, 250)):
        assert frames.frame_len(dt) == cheap.frame_len(dt) == n


def test_frame_index_of_whole_frames():
    assert frames.frame_index(100, 0.00025).tolist() == np.repeat(np.arange(5), 20).tolist()
    assert frames.frame_index(3, 0.005).tolist() == [0, 1, 2]
    assert frames.frame_index(5000, 0.00002).max() == 19 and frames.frame_index(5000, 0.00002).dtype == np.int64


def test_frame_index_of_spectrogram_columns():
    idx = frames.frame_index(98, 0.001024)                      # 100 ms of TokaMark's Mirnov columns
    assert np.bincount(idx).tolist() == [5, 5, 5, 5, 4, 5, 5, 5, 5, 5, 5, 5, 4, 5, 5, 5, 5, 5, 5, 5]
    assert (np.diff(idx) >= 0).all()


def test_linear_units_undo_the_compression_and_keep_nan():
    u = frames.to_linear(X, NONE)
    assert np.allclose(u[0, :2], [0.0, 1.0]) and np.isnan(u[0, 2]) and np.allclose(u[1], [0.0, 1.0, 2.0])
    u = frames.to_linear(X, ASINH)
    assert np.allclose(u[0, :2], [2.0, 6.0]) and np.isnan(u[0, 2]) and np.allclose(u[1], [-1.0, 1.0, 3.0])


def test_the_conversions_are_inverses_and_agree_with_the_cache():
    for st in (NONE, ASINH):
        u = frames.to_linear(X, st)
        assert np.allclose(frames.from_linear(u, st), X, equal_nan=True)
        z = transform.to_model(X, st)
        assert np.allclose(frames.linear_from_model(z, st), u, equal_nan=True)
        assert np.allclose(frames.model_from_linear(u, st), z, equal_nan=True)
        assert frames.to_linear(np.stack([X, X, X]), st).shape == (3, 2, 3)      # a leading batch axis


def test_the_package_hash_changes_with_its_files(tmp_path, monkeypatch):
    first = tokenizers.code_sha256()
    assert len(first) == 64 and first == tokenizers.code_sha256()
```

and in `tests/ignite/mast/test_paths.py`:

```python
def test_stage1_dir_is_under_the_root(monkeypatch, tmp_path):
    monkeypatch.setenv("IGNITE_MAST_ROOT", str(tmp_path))
    assert paths.stage1_dir() == tmp_path / "stage1"
```

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_tok_frames.py -q -W error`
Expected: an import error (`No module named '...mast.tokenizers'`).

- [ ] **Step 3: Implement the signatures of "Produces"**

`paths.py` is being edited by another change on this branch; read it as it is now and add only the one function. In `frames.py` the statistics broadcast as `st.mean[..., None]` against the time axis.

- [ ] **Step 4: Run the tests**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_tok_frames.py tests/ignite/mast/test_paths.py -q -W error`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/tokamak_foundation_model/ignite/mast/paths.py src/tokamak_foundation_model/ignite/mast/tokenizers/__init__.py src/tokamak_foundation_model/ignite/mast/tokenizers/frames.py tests/ignite/mast/test_tok_frames.py tests/ignite/mast/test_paths.py
git commit -m "ignite.mast: tokenizers package, 5 ms frames and linear units"
```

### Task 2: ordered bins

**Files:**
- Create: `src/tokamak_foundation_model/ignite/mast/tokenizers/grid.py`
- Test: `tests/ignite/mast/test_tok_grid.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `READ_MODES = ("unquantized", "ranged", "nearest")`.
  - `@dataclass(frozen=True) class GridSet` with fields `lo: np.ndarray`, `hi: np.ndarray`, `width: np.ndarray` (each `(n_tok, n_val)`, `lo < hi`, `width > 0`) and `n_bins: int = 64`. The centres of position `(i, j)` are `width * sinh(linspace(asinh(lo / width), asinh(hi / width), n_bins))`: strictly increasing, first `lo`, last `hi`, spaced evenly near zero and logarithmically beyond `width`.
    - `centres(self) -> np.ndarray`: `(n_tok, n_val, n_bins)`.
    - `two_hot(self, values: np.ndarray) -> np.ndarray`: `values (..., n_tok, k)` with `k <= n_val` to `(..., n_tok, k, n_bins)` float64; the two neighbouring centres get weights linear in the value, a value beyond the range gets weight 1 on the outermost centre, a NaN value gets a row of NaN.
    - `expected(self, weights: np.ndarray) -> np.ndarray`: `(..., n_tok, k, n_bins)` to `(..., n_tok, k)`.
    - `index(self, values: np.ndarray) -> np.ndarray`: index of the nearest centre, int64, -1 for NaN.
    - `read(self, values: np.ndarray, mode: str) -> np.ndarray`: `unquantized` returns the values; `ranged` is `expected(two_hot(values))` (the value limited to `[lo, hi]`); `nearest` is the nearest centre. NaN stays NaN. An unknown mode raises `ValueError` naming `READ_MODES`.
  - `fit_ranges(values: np.ndarray, margin: float = 0.05) -> tuple[np.ndarray, np.ndarray, np.ndarray]`: `values (N, n_tok, n_val)` with NaN for missing to `(lo, hi, width)`, each `(n_tok, n_val)`. `lo` and `hi` are the minimum and maximum of the finite values, each moved outward by `margin` times their span; `width` is 1.4826 times the median absolute deviation about the median, and at least `1e-3` times the span. A position whose span is 0 (a constant) gets `lo = v - 1`, `hi = v + 1`, `width = 1`. A position with no finite value gets NaN in all three.

- [ ] **Step 1: Write the failing tests**

```python
"""Ordered bins: two-hot targets, expected values, and the three read-outs."""

import numpy as np
import pytest

from tokamak_foundation_model.ignite.mast.tokenizers import grid


def one(lo=-10.0, hi=10.0, width=1.0, n_bins=5):
    return grid.GridSet(lo=np.array([[lo]]), hi=np.array([[hi]]), width=np.array([[width]]), n_bins=n_bins)


def test_centres_are_asinh_spaced_between_lo_and_hi():
    c = one().centres()
    assert c.shape == (1, 1, 5)
    assert np.allclose(c[0, 0], np.sinh(np.linspace(np.arcsinh(-10.0), np.arcsinh(10.0), 5)))
    assert np.isclose(c[0, 0, 0], -10.0) and np.isclose(c[0, 0, -1], 10.0) and (np.diff(c[0, 0]) > 0).all()


def test_two_hot_has_two_neighbours_and_its_mean_is_the_value():
    g = one()
    c = g.centres()[0, 0]
    v = np.array([[[c[1]]], [[0.5 * (c[1] + c[2])]], [[0.25 * c[3]]], [[10.0]]])        # (4, 1, 1)
    w = g.two_hot(v)
    assert w.shape == (4, 1, 1, 5) and np.allclose(w.sum(-1), 1.0) and ((w > 0).sum(-1) <= 2).all()
    assert np.allclose(w[0, 0, 0], [0, 1, 0, 0, 0]) and np.allclose(w[1, 0, 0], [0, 0.5, 0.5, 0, 0])
    assert np.allclose(w[2, 0, 0], [0, 0, 0.75, 0.25, 0])
    assert np.allclose(g.expected(w), v)


def test_a_value_outside_the_range_reads_as_the_outermost_centre():
    g = one()
    v = np.array([[[-50.0]], [[1e9]], [[np.nan]]])
    for mode in ("ranged", "nearest"):
        r = g.read(v, mode)
        assert np.allclose(r[:2, 0, 0], [-10.0, 10.0]) and np.isnan(r[2, 0, 0])
    assert g.index(v)[:, 0, 0].tolist() == [0, 4, -1]
    assert np.array_equal(g.read(v, "unquantized"), v, equal_nan=True)


def test_nearest_is_the_nearest_centre_and_ranged_is_exact_inside():
    g = one()
    c = g.centres()[0, 0]
    v = np.array([[[0.49 * (c[2] + c[3])]], [[0.51 * (c[2] + c[3])]]])
    assert np.allclose(g.read(v, "nearest")[:, 0, 0], [c[2], c[3]])
    assert np.allclose(g.read(v, "ranged"), v)


def test_a_prefix_of_the_values_is_read_with_the_first_grids():
    g = grid.GridSet(lo=np.array([[-1.0, -4.0, -9.0]]), hi=np.array([[1.0, 4.0, 9.0]]), width=np.ones((1, 3)), n_bins=9)
    assert np.allclose(g.read(np.array([[5.0, 5.0]]), "ranged"), [[1.0, 4.0]])


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="unquantized"):
        one().read(np.zeros((1, 1)), "rounded")


def test_fit_ranges_covers_the_data_with_a_margin():
    rng = np.random.default_rng(0)
    v = rng.standard_normal((5000, 2, 3))
    v[:, 1, 2] = 7.0                         # a constant position
    v[:, 0, 1] = np.nan                      # a position that was never seen
    v[0, 0, 0] = 40.0                        # an outlier: the range covers it, the width does not grow with it
    lo, hi, width = grid.fit_ranges(v)
    assert lo.shape == hi.shape == width.shape == (2, 3)
    span = 40.0 - np.nanmin(v[:, 0, 0])
    assert np.isclose(hi[0, 0], 40.0 + 0.05 * span) and np.isclose(lo[0, 0], np.nanmin(v[:, 0, 0]) - 0.05 * span)
    assert 0.8 < width[0, 0] < 1.2
    assert (lo[1, 2], hi[1, 2], width[1, 2]) == (6.0, 8.0, 1.0)
    assert np.isnan(lo[0, 1]) and np.isnan(hi[0, 1]) and np.isnan(width[0, 1])
```

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_tok_grid.py -q -W error`
Expected: an import error for `grid`.

- [ ] **Step 3: Implement `GridSet` and `fit_ranges`**

No warning may escape for NaN input or for a position that is all NaN (`-W error` turns numpy's "All-NaN slice" warning into a failure): mask first and restore NaN.

- [ ] **Step 4: Run the tests**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_tok_grid.py -q -W error`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/tokamak_foundation_model/ignite/mast/tokenizers/grid.py tests/ignite/mast/test_tok_grid.py
git commit -m "ignite.mast: ordered bins with two-hot targets and three read-outs"
```

### Task 3: the cosine transform per frame (`FrameDCT`)

**Files:**
- Create: `src/tokamak_foundation_model/ignite/mast/tokenizers/codecs.py`
- Test: `tests/ignite/mast/test_tok_codecs.py`

**Interfaces:**
- Consumes: Task 1's `frames.frame_index`, `to_linear`, `from_linear`, `linear_from_model`; Task 2's `GridSet` and `READ_MODES`; `transform.ChannelStats`, `transform.to_model`, `transform.to_physical(z, st)`.
- Produces (all in `codecs.py`):
  - `dct_values(x: np.ndarray, k: int) -> np.ndarray`: `x (..., n)` with NaN for missing to `(..., k)`. For a frame without NaN: the first `k` coefficients of `scipy.fft.dct(x, type=2, norm="ortho")` divided by `sqrt(n)`, padded with zeros if `k > n`; so value 0 is the frame mean and the mean square of the frame is the sum of the squared values. For a frame with some NaN: value 0 is the mean of the valid samples and the others are 0. For a frame with no valid sample: NaN in every value.
  - `dct_decode(values: np.ndarray, n: int) -> np.ndarray`: `(..., k)` to `(..., n)`; the inverse of the above for any `k` (missing coefficients are zero, coefficients beyond `n` are ignored); a NaN row decodes to NaN.
  - `class Codec`: the base class. Attributes `name: str`, `dt: float`, `values_shape: tuple[int, ...]`, `stats: ChannelStats`, `domain: str` (`"linear"`, `"compressed"` or `"signal"`), `signal_mean: float = 0.0`, `signal_std: float = 1.0` (TokaMark's constants of the signal), `n_tok: int`, `n_val: int`, `grids: GridSet | None`. An unknown domain raises `ValueError`. Methods implemented once in the base:
    - `to_domain(self, x: np.ndarray) -> np.ndarray` and `from_domain(self, u)`: physical `(..., *values_shape, T)` to and from the codec's domain: `frames.to_linear` for `"linear"`, `transform.to_model` for `"compressed"`, `(x - signal_mean) / signal_std` for `"signal"`.
    - `domain_from_model(self, z: np.ndarray) -> np.ndarray`: model units (the cache) to the domain.
    - `encode(self, x: np.ndarray) -> np.ndarray`: physical `(N, *values_shape, T)` to values `(N, F, n_tok, n_val)`, `F = frame_index(T, dt).max() + 1`; calls `encode_domain(to_domain(x))`.
    - `decode(self, values: np.ndarray, n: int) -> np.ndarray`: values to physical `(N, *values_shape, n)`.
    - `roundtrip(self, x: np.ndarray, mode: str = "unquantized") -> np.ndarray`: `decode(read(encode(x)), T)` where `read` is the identity for `unquantized` and `self.grids.read(values, mode)` otherwise (`ValueError` if `grids` is None and the mode needs them). NaN where the frame has no token.
    - Subclasses implement `encode_domain(self, u: np.ndarray) -> np.ndarray` and `decode_domain(self, values: np.ndarray, n: int) -> np.ndarray`.
  - `class FrameDCT(Codec)`: `FrameDCT(name: str, dt: float, values_shape: tuple[int, ...], stats: ChannelStats, k: int, domain: str = "linear", grids: GridSet | None = None, signal_mean: float = 0.0, signal_std: float = 1.0)`. One token per channel: `n_tok = prod(values_shape)`, `n_val = k`. Each channel's samples of one frame go through `dct_values`.

- [ ] **Step 1: Write the failing tests**

```python
"""Codecs without learned weights. This file grows in Tasks 4 and 5."""

import numpy as np
import pytest

from tokamak_foundation_model.ignite.mast import transform
from tokamak_foundation_model.ignite.mast.tokenizers import codecs, grid

RNG = np.random.default_rng(0)


def stats(compress, c):
    return transform.ChannelStats(compress=compress, scale=np.full(c, 2.0), mean=np.full(c, 0.5), std=np.full(c, 1.5))


def test_the_first_value_is_the_frame_mean_and_values_are_nested():
    x = RNG.standard_normal((7, 25))
    v = codecs.dct_values(x, 25)
    assert np.allclose(v[:, 0], x.mean(-1)) and np.allclose((v ** 2).sum(-1), (x ** 2).mean(-1))
    assert np.allclose(codecs.dct_decode(v, 25), x)
    assert np.allclose(codecs.dct_decode(v[:, :1], 25), np.repeat(x.mean(-1, keepdims=True), 25, -1))
    err = [np.mean((codecs.dct_decode(v[:, :k], 25) - x) ** 2) for k in (1, 2, 4, 8, 16, 25)]
    assert all(a > b for a, b in zip(err, err[1:])) and err[-1] < 1e-20
    assert np.allclose(codecs.dct_values(x, 4), v[:, :4])                    # a prefix is the same values


def test_a_frame_of_one_sample_is_exact_with_one_value():
    v = codecs.dct_values(np.array([[3.5]]), 4)
    assert v.tolist() == [[3.5, 0.0, 0.0, 0.0]] and codecs.dct_decode(v, 1).tolist() == [[3.5]]


def test_a_partly_valid_frame_falls_back_to_the_mean_of_its_valid_samples():
    v = codecs.dct_values(np.array([[1.0, np.nan, 3.0, np.nan]]), 3)
    assert v.tolist() == [[2.0, 0.0, 0.0]] and np.allclose(codecs.dct_decode(v, 4), 2.0)


def test_a_frame_without_a_valid_sample_has_no_values():
    v = codecs.dct_values(np.full((1, 4), np.nan), 2)
    assert np.isnan(v).all() and np.isnan(codecs.dct_decode(v, 4)).all()


@pytest.mark.parametrize("compress,domain", [("none", "linear"), ("asinh", "linear"), ("asinh", "compressed")])
def test_frame_dct_with_every_coefficient_is_the_identity(compress, domain):
    c = codecs.FrameDCT("s", 0.00025, (3,), stats(compress, 3), k=20, domain=domain)
    x = RNG.standard_normal((4, 3, 100))                                     # 5 frames of 20 samples
    v = c.encode(x)
    assert v.shape == (4, 5, 3, 20) and (c.n_tok, c.n_val) == (3, 20)
    assert np.allclose(c.decode(v, 100), x) and np.allclose(c.roundtrip(x), x)


def test_frame_dct_with_one_value_holds_the_frame_mean_in_its_domain():
    st = stats("asinh", 2)
    x = np.abs(RNG.standard_normal((1, 2, 40))) * 10
    lin = codecs.FrameDCT("s", 0.00025, (2,), st, k=1, domain="linear").roundtrip(x)
    com = codecs.FrameDCT("s", 0.00025, (2,), st, k=1, domain="compressed").roundtrip(x)
    means = x.reshape(1, 2, 2, 20).mean(-1)
    assert np.allclose(lin.reshape(1, 2, 2, 20)[..., 0], means)              # the linear domain keeps the mean
    assert (com.reshape(1, 2, 2, 20)[..., 0] < means).all()                  # a mean on the asinh scale is low


def test_roundtrip_reads_through_the_grids():
    st = stats("none", 1)
    g = grid.GridSet(lo=np.array([[-1.0]]), hi=np.array([[1.0]]), width=np.array([[1.0]]), n_bins=3)
    c = codecs.FrameDCT("s", 0.005, (1,), st, k=1, grids=g)
    x = st.mean[0] + st.std[0] * np.array([[[0.2, 5.0, np.nan]]])            # linear units 0.2, 5.0, missing
    lin = lambda a: (a - st.mean[0]) / st.std[0]
    assert np.allclose(lin(c.roundtrip(x, "ranged"))[0, 0, :2], [0.2, 1.0])
    assert np.allclose(lin(c.roundtrip(x, "nearest"))[0, 0, :2], [0.0, 1.0])
    assert np.isnan(c.roundtrip(x, "ranged")[0, 0, 2])
    with pytest.raises(ValueError, match="grids"):
        codecs.FrameDCT("s", 0.005, (1,), st, k=1).roundtrip(x, "ranged")
```

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_tok_codecs.py -q -W error`
Expected: an import error for `codecs`.

- [ ] **Step 3: Implement `dct_values`, `dct_decode`, `Codec` and `FrameDCT`**

Frames are the groups of `frames.frame_index(T, dt)`. `transform.to_model` and `transform.to_physical` accept exactly `(*values_shape, T)` and raise on a leading batch axis: fold the batch axis into the time axis before the call and unfold after it. Do not edit `transform.py`.

- [ ] **Step 4: Run the tests**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_tok_codecs.py -q -W error`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/tokamak_foundation_model/ignite/mast/tokenizers/codecs.py tests/ignite/mast/test_tok_codecs.py
git commit -m "ignite.mast: the cosine transform per frame as a tokenizer without weights"
```

### Task 4: principal components for profiles and maps (`LinearCodec`)

**Files:**
- Modify: `src/tokamak_foundation_model/ignite/mast/tokenizers/codecs.py`
- Test: `tests/ignite/mast/test_tok_codecs.py` (append)

**Interfaces:**
- Consumes: Task 3's `Codec` (its `to_domain`, `from_domain`, `domain_from_model`, `roundtrip`).
- Produces:
  - `fit_pca(x: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]`: `x (N, D)` with NaN for missing, to `(mean (D,), basis (n, D), eigenvalues (n,), total_variance)` with `n = min(n_components, D)`. The mean of a point is over the rows where it is valid. The covariance of two points is taken over the rows where both are valid, about the two points' means over those same rows (pairwise-complete). The basis rows are orthonormal eigenvectors in order of decreasing eigenvalue; negative eigenvalues (possible with pairwise covariance) are set to 0. `total_variance` is the trace of the covariance. A point that is never valid, or a pair never valid together, contributes 0.
  - `class LinearCodec(Codec)`: `LinearCodec(name, dt, values_shape, stats, profile_mean: np.ndarray, basis: np.ndarray, eigenvalues: np.ndarray, n_values: int, values_per_token: int, domain: str = "linear", ridge: float = 0.0, grids: GridSet | None = None, signal_mean: float = 0.0, signal_std: float = 1.0)`. One example is one sample in time (these signals have one sample per frame), flattened row-major to `D = prod(values_shape)` points. `n_tok = n_values // values_per_token` (`ValueError` if not a multiple or if `n_values > len(basis)`), `n_val = values_per_token`; token `i` holds components `i * n_val` to `(i + 1) * n_val - 1`, so a prefix of the tokens is a coarser profile. With `B` the first `n_values` basis rows restricted to the valid points `v` and `e` their eigenvalues, the values of a row `r` are the solution `a` of `(B B^T + lam * diag(1 / e)) a = B (r - profile_mean)[v]` with `lam = max(ridge, 1e-10 * eigenvalues[0])`, and components with `e == 0` are fixed to 0. For a row without a missing point the values of the first tokens do not depend on `n_values`; for a row with missing points they do, slightly. A row with no valid point has NaN values. `decode` returns `profile_mean + a @ basis[:n_values]` at every point (also where the input was missing). `n_values = 0` is allowed: no tokens, and the decode is the mean profile.

- [ ] **Step 1: Write the failing tests** (append to `tests/ignite/mast/test_tok_codecs.py`)

```python
def low_rank(n, d=30, seed=0):
    """Rows of dimension d that lie exactly in a 4-dimensional subspace about the mean 3."""
    rng = np.random.default_rng(seed)
    true = np.linalg.qr(rng.standard_normal((d, 4)))[0].T
    return 3.0 + (rng.standard_normal((n, 4)) * np.array([8.0, 4.0, 2.0, 1.0])) @ true, true


def linear_codec(n_values, per_token, **kw):
    x, _ = low_rank(4000)
    mean, basis, eig, _ = codecs.fit_pca(x, 6)
    st = transform.ChannelStats(compress="none", scale=np.ones(30), mean=np.zeros(30), std=np.ones(30))
    return codecs.LinearCodec("p", 0.005, (30,), st, mean, basis, eig, n_values, per_token, **kw), x


def test_fit_pca_finds_the_subspace_in_order_of_variance():
    x, true = low_rank(4000)
    mean, basis, eig, total = codecs.fit_pca(x, 6)
    assert mean.shape == (30,) and basis.shape == (6, 30) and eig.shape == (6,)
    assert np.allclose(mean, x.mean(0)) and np.allclose(basis @ basis.T, np.eye(6), atol=1e-8)
    assert np.allclose(eig[:4], [64.0, 16.0, 4.0, 1.0], rtol=0.15) and (np.abs(eig[4:]) < 1e-8).all()
    assert np.isclose(total, eig.sum()) and np.allclose(true @ basis[:4].T @ basis[:4], true, atol=1e-6)
    assert codecs.fit_pca(x, 96)[1].shape == (30, 30)                        # never more components than points


def test_fit_pca_uses_every_pair_of_valid_points():
    x, _ = low_rank(20000)
    holes = x.copy()
    holes[np.random.default_rng(1).random(x.shape) < 0.3] = np.nan
    holes[:, 29] = np.nan                                                    # a point that is never valid
    mean, basis, eig, _ = codecs.fit_pca(holes, 2)
    ref_mean, ref, _, _ = codecs.fit_pca(x[:, :29], 2)                       # the same 29 points without holes
    assert mean[29] == 0.0 and np.abs(basis[:, 29]).max() < 1e-12
    assert np.allclose(mean[:29], ref_mean, atol=0.1)
    assert np.linalg.norm(ref @ basis[:, :29].T @ basis[:, :29] - ref) < 0.1


def test_a_low_rank_profile_is_exact_with_enough_values_and_tokens_are_nested():
    c, x = linear_codec(4, 2)
    w = np.moveaxis(x[:12].reshape(3, 4, 30), 1, 2)                          # (N=3, D=30, T=4 frames)
    v = c.encode(w)
    assert v.shape == (3, 4, 2, 2) and (c.n_tok, c.n_val) == (2, 2)
    assert np.allclose(c.decode(v, 4), w, atol=1e-6)
    coarse = v.copy()
    coarse[:, :, 1] = 0.0                                                    # drop the second token
    assert np.allclose(c.decode(coarse, 4), linear_codec(2, 2)[0].roundtrip(w), atol=1e-6)


def test_zero_values_decode_to_the_mean_profile():
    c, x = linear_codec(0, 2)
    w = x[:2].T[None]                                                        # (1, 30, 2)
    assert c.encode(w).shape == (1, 2, 0, 2)
    assert np.allclose(c.roundtrip(w), np.repeat(c.profile_mean[None, :, None], 2, -1))


def test_a_profile_with_missing_points_is_encoded_from_its_valid_points():
    c, x = linear_codec(4, 2)
    w = x[:1].T[None].copy()                                                 # (1, 30, 1)
    whole = c.encode(w)
    w[0, ::3, 0] = np.nan                                                    # a third of the points missing
    v = c.encode(w)
    assert np.allclose(v, whole, atol=1e-5)                                  # exact low rank: 20 points are enough
    assert np.isfinite(c.decode(v, 1)).all()                                 # the decoder fills the missing points


def test_fewer_valid_points_than_components_still_gives_finite_values():
    c, x = linear_codec(4, 2, ridge=0.5)
    w = np.full((1, 30, 2), np.nan)
    w[0, :2, 0] = x[0, :2]                                                   # frame 0: two valid points; frame 1: none
    v = c.encode(w)
    assert np.isfinite(v[0, 0]).all() and np.isnan(v[0, 1]).all()
    out = c.decode(v, 2)
    assert np.isfinite(out[0, :, 0]).all() and np.isnan(out[0, :, 1]).all()


def test_a_map_is_flattened_row_major():
    st = transform.ChannelStats(compress="none", scale=np.ones((3, 4)), mean=np.zeros((3, 4)), std=np.ones((3, 4)))
    c = codecs.LinearCodec("m", 0.005, (3, 4), st, np.zeros(12), np.eye(12), np.ones(12), 12, 4)
    w = RNG.standard_normal((2, 3, 4, 5))
    v = c.encode(w)
    assert v.shape == (2, 5, 3, 4) and np.allclose(v[1, 2].ravel(), w[1, :, :, 2].ravel(), atol=1e-6)
    assert np.allclose(c.decode(v, 5), w, atol=1e-6)


def test_the_signal_domain_is_tokamarks_standardization():
    st = transform.ChannelStats(compress="none", scale=np.ones(30), mean=np.full(30, 5.0), std=np.linspace(1.0, 3.0, 30))
    c = codecs.LinearCodec("p", 0.005, (30,), st, np.zeros(30), np.eye(30), np.ones(30), 30, 30, domain="signal",
                           signal_mean=2.0, signal_std=4.0)
    w = RNG.standard_normal((2, 30, 1))
    assert np.allclose(c.to_domain(w), (w - 2.0) / 4.0)
    assert np.allclose(c.encode(w)[:, 0, 0, :], (w[:, :, 0] - 2.0) / 4.0, atol=1e-6)
    assert np.allclose(c.domain_from_model(transform.to_model(w[0], st)), (w[0] - 2.0) / 4.0)
    with pytest.raises(ValueError, match="multiple"):
        codecs.LinearCodec("p", 0.005, (30,), st, np.zeros(30), np.eye(30), np.ones(30), 10, 4)
```

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_tok_codecs.py -q -W error -k "pca or profile or map or signal_domain or zero_values or fewer_valid"`
Expected: FAIL with `AttributeError: module ... has no attribute 'fit_pca'`.

- [ ] **Step 3: Implement `fit_pca` and `LinearCodec`**

`fit_pca` accumulates the pairwise sums and pair counts in chunks of rows (the flux map has D = 4,225: three D by D float64 accumulators, and the input is never copied whole), then `np.linalg.eigh`.

- [ ] **Step 4: Run the tests**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_tok_codecs.py -q -W error`
Expected: 17 passed.

- [ ] **Step 5: Commit**

```bash
git add src/tokamak_foundation_model/ignite/mast/tokenizers/codecs.py tests/ignite/mast/test_tok_codecs.py
git commit -m "ignite.mast: principal components as the linear baseline for profiles and maps"
```

### Task 5: the band transform for spectrograms (`BandDCT`)

**Files:**
- Modify: `src/tokamak_foundation_model/ignite/mast/tokenizers/codecs.py`
- Test: `tests/ignite/mast/test_tok_codecs.py` (append)

**Interfaces:**
- Consumes: Task 3's `Codec`, `dct_values`, `dct_decode`; `frames.frame_index`.
- Produces:
  - `BAND_BINS = 32`.
  - `class BandDCT(Codec)`: `BandDCT(name: str, dt: float, values_shape: tuple[int, int], stats: ChannelStats, k: Sequence[int], grids: GridSet | None = None)` for a spectrogram signal of shape `(P, n_bins)` with `n_bins` a multiple of 32 (`ValueError` otherwise, and if `len(k) != n_bins // 32`). `domain = "band"`: linear power divided by `band_scale (P, n_bands)`, the mean of `stats.scale` over the band's bins; `to_domain`, `from_domain` and `domain_from_model` are overridden accordingly. One token per (probe, band): `n_tok = P * n_bands` with token index `probe * n_bands + band`; `n_val = max(k)`. Encoding: the mean over the frame's columns of each bin (valid columns only), then `dct_values` across the band's 32 bins, then the values from index `k[band]` on are set to 0. Decoding: `dct_decode` to 32 bins, the same spectrum for every column of the frame.

- [ ] **Step 1: Write the failing tests** (append; add `from tokamak_foundation_model.ignite.mast.tokenizers import frames` to the imports)

```python
def band(k, probes=2):
    shape = (probes, 256)
    st = transform.ChannelStats(compress="asinh", scale=np.full(shape, 2.0), mean=np.zeros(shape), std=np.ones(shape))
    return codecs.BandDCT("m", 0.001024, shape, st, k)


def power(*shape):
    return np.abs(RNG.standard_normal(shape)) + 0.1


def test_all_coefficients_give_each_frames_mean_spectrum():
    c = band([32] * 8)
    x = power(1, 2, 256, 98)
    v = c.encode(x)
    assert v.shape == (1, 20, 16, 32) and (c.n_tok, c.n_val, c.domain) == (16, 32, "band")
    out = c.decode(v, 98)
    idx = frames.frame_index(98, 0.001024)
    for f in (0, 4, 19):
        assert np.allclose(out[..., idx == f], x[..., idx == f].mean(-1, keepdims=True))


def test_a_window_of_98_columns_decodes_to_98_columns():
    c = band([4] * 8)
    x = power(3, 2, 256, 98)
    assert c.encode(x).shape == (3, 20, 16, 4) and c.roundtrip(x).shape == x.shape


def test_the_first_value_is_the_band_mean_power_over_the_band_scale():
    c = band([1] * 8)
    x = power(1, 2, 256, 5)                                                  # one frame of 5 columns
    v = c.encode(x)
    assert np.allclose(v[0, 0, :, 0], x[0].mean(-1).reshape(2, 8, 32).mean(-1).ravel() / 2.0)
    assert np.allclose(c.decode(v, 5)[0, 1, 32:64, 3], x[0, 1, 32:64].mean())


def test_k_is_per_band():
    c = band([32, 2, 2, 2, 2, 2, 2, 1])
    x = power(1, 2, 256, 5)
    v = c.encode(x)
    assert c.n_val == 32 and (v[0, 0, 1, 2:] == 0).all() and (v[0, 0, 7, 1:] == 0).all() and (v[0, 0, 9, 2:] == 0).all()
    assert np.allclose(c.decode(v, 5)[0, :, :32], x[0, :, :32].mean(-1, keepdims=True))    # the lowest band is whole


def test_the_band_transform_averages_linear_power():
    c = band([32] * 8, probes=1)
    x = np.ones((1, 1, 256, 5))
    x[..., 0] = 1e4                                                          # one column four orders above the others
    assert np.allclose(c.roundtrip(x), (1e4 + 4.0) / 5.0)
    assert np.allclose(c.domain_from_model(transform.to_model(x[0], c.stats)), x[0] / 2.0)


def test_a_column_that_is_missing_is_left_out_of_the_frame_mean():
    c = band([32] * 8, probes=1)
    x = power(1, 1, 256, 5)
    x[..., 2] = np.nan
    out = c.roundtrip(x)
    assert np.allclose(out[..., 0], np.nanmean(x, -1)) and np.isnan(c.roundtrip(np.full((1, 1, 256, 5), np.nan))).all()


def test_a_spectrogram_whose_bins_do_not_make_whole_bands_is_refused():
    st = transform.ChannelStats(compress="asinh", scale=np.ones((1, 100)), mean=np.zeros((1, 100)), std=np.ones((1, 100)))
    with pytest.raises(ValueError, match="32"):
        codecs.BandDCT("m", 0.001024, (1, 100), st, [1, 1, 1])
```

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_tok_codecs.py -q -W error -k "band or column or spectrogram or coefficients_give"`
Expected: FAIL with `AttributeError: ... has no attribute 'BandDCT'`.

- [ ] **Step 3: Implement `BandDCT`**

Add `"band"` to the domains the base class accepts. The frame mean is taken with a mask and a count, not with `np.nanmean` on an all-NaN slice (`-W error`).

- [ ] **Step 4: Run the tests**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_tok_codecs.py -q -W error`
Expected: 24 passed.

- [ ] **Step 5: Commit**

```bash
git add src/tokamak_foundation_model/ignite/mast/tokenizers/codecs.py tests/ignite/mast/test_tok_codecs.py
git commit -m "ignite.mast: a band transform in linear power for the Mirnov spectrograms"
```

### Task 6: the three instruments as harness predictors

**Files:**
- Create: `src/tokamak_foundation_model/ignite/mast/tokenizers/floor.py`
- Test: `tests/ignite/mast/test_tok_floor.py`

**Interfaces:**
- Consumes: `Codec.roundtrip(x, mode)` (Task 3); `harness.WindowBatch` (a dataclass with `shot_ids`, `window_indices`, `inputs`, `actuators`, `targets`, `last_valid`; arrays `(B, *values_shape, T)` float32 in physical units, NaN for missing); `harness.PredictFn = Callable[[WindowBatch], dict[str, np.ndarray]]` (the harness raises if a prediction is missing, has another shape, or is not finite where the target is finite); `tasks.TaskSpec` (`.inputs`, `.actuators`, `.outputs`: tuples of `SignalSpec(name, dt, values_shape, mean, std)`); the fixtures `toy_task` and `toy_batch(n, seed)` of `tests/ignite/mast/conftest.py` (inputs `summary-ip` with 20 samples and `magnetics-flux_loop_flux` with 25, actuator `pf_active-coil_voltage` with 120, output `summary-ip` with 100 samples at 0.25 ms).
- Produces:
  - `floor_predictor(task: TaskSpec, codecs: Mapping[str, Codec], mode: str = "unquantized") -> PredictFn`: for each output signal, `roundtrip` of the true targets; where the result is not finite, the signal's `SignalSpec.mean`; float32. A missing codec raises `KeyError` naming the signals when the predictor is built.
  - `blind_to_targets(predict: PredictFn) -> PredictFn`: calls `predict` on a copy of the batch whose targets are NaN arrays of the same shapes; keeps the function's attributes (`history_s`).
  - `passthrough_predictor(inner: PredictFn, task: TaskSpec, codecs: Mapping[str, Codec], mode: str = "unquantized") -> PredictFn`: `roundtrip` of the prediction of `blind_to_targets(inner)`; keeps `inner`'s attributes.
  - `input_roundtrip_predictor(inner: PredictFn, task: TaskSpec, codecs: Mapping[str, Codec], mode: str = "unquantized") -> PredictFn`: calls `blind_to_targets(inner)` on a copy of the batch in which every input and actuator array is replaced by its `roundtrip`, with NaN kept wherever the original sample was NaN; `last_valid` is passed on unchanged; keeps `inner`'s attributes. A missing codec for an input or actuator raises `KeyError` when the predictor is built.

- [ ] **Step 1: Write the failing tests**

```python
"""The floor, the prediction pass-through and the input round-trip, as harness predictors."""

import numpy as np
import pytest

from tokamak_foundation_model.ignite.mast import transform
from tokamak_foundation_model.ignite.mast.tokenizers import codecs, floor


def toy_codecs(task, k):
    out = {}
    for s in (*task.inputs, *task.actuators, *task.outputs):
        st = transform.ChannelStats(compress="none", scale=np.ones(s.values_shape),
                                    mean=np.full(s.values_shape, s.mean), std=np.full(s.values_shape, s.std))
        out[s.name] = codecs.FrameDCT(s.name, s.dt, s.values_shape, st, k=k)
    return out


def test_the_floor_is_the_frame_mean_at_one_value_and_exact_at_all(toy_task, toy_batch):
    b = toy_batch(6)
    exact = floor.floor_predictor(toy_task, toy_codecs(toy_task, 20))(b)["summary-ip"]
    assert exact.dtype == np.float32 and np.allclose(exact, b.targets["summary-ip"], rtol=1e-5)
    coarse = floor.floor_predictor(toy_task, toy_codecs(toy_task, 1))(b)["summary-ip"]
    t = b.targets["summary-ip"].reshape(6, 1, 5, 20)
    assert np.allclose(coarse.reshape(6, 1, 5, 20), t.mean(-1, keepdims=True), rtol=1e-5)


def test_the_floor_is_finite_where_the_target_is_missing(toy_task, toy_batch):
    b = toy_batch(4)
    b.targets["summary-ip"][0, 0, :20] = np.nan                              # a frame without a sample
    b.targets["summary-ip"][1, 0, 25] = np.nan                               # a frame with one missing sample
    out = floor.floor_predictor(toy_task, toy_codecs(toy_task, 4))(b)["summary-ip"]
    assert np.isfinite(out).all()
    assert np.allclose(out[0, 0, :20], toy_task.outputs[0].mean)
    assert np.allclose(out[1, 0, 20:40], np.nanmean(b.targets["summary-ip"][1, 0, 20:40]), rtol=1e-5)


def test_a_task_signal_without_a_codec_is_refused_when_the_predictor_is_built(toy_task):
    with pytest.raises(KeyError, match="summary-ip"):
        floor.floor_predictor(toy_task, {})
    with pytest.raises(KeyError, match="pf_active-coil_voltage"):
        floor.input_roundtrip_predictor(lambda b: {}, toy_task, {"summary-ip": toy_codecs(toy_task, 1)["summary-ip"]})


def test_passthrough_tokenizes_the_prediction_and_never_shows_the_targets(toy_task, toy_batch):
    seen = {}

    def inner(batch):
        seen["blind"] = all(np.isnan(a).all() for a in batch.targets.values())
        return {"summary-ip": np.tile(np.arange(100, dtype=np.float32), (len(batch.shot_ids), 1, 1))}

    inner.history_s = 0.05
    p = floor.passthrough_predictor(inner, toy_task, toy_codecs(toy_task, 1))
    out = p(toy_batch(3))["summary-ip"]
    assert seen["blind"] and p.history_s == 0.05 and out.shape == (3, 1, 100)
    assert np.allclose(out[0, 0, :20], 9.5) and np.allclose(out[2, 0, 80:], 89.5)


def test_the_input_roundtrip_changes_inputs_and_actuators_only_and_keeps_missing_samples(toy_task, toy_batch):
    seen = {}

    def inner(batch):
        seen.update(inputs=batch.inputs, actuators=batch.actuators, last_valid=batch.last_valid, targets=batch.targets)
        return {"summary-ip": np.zeros_like(batch.targets["summary-ip"])}

    b = toy_batch(2)
    b.inputs["summary-ip"][0, 0, 3] = np.nan
    original = b.inputs["summary-ip"].copy()
    floor.input_roundtrip_predictor(inner, toy_task, toy_codecs(toy_task, 1))(b)
    ip = seen["inputs"]["summary-ip"]
    assert np.isnan(ip[0, 0, 3]) and np.allclose(ip[0, 0, :3], np.nanmean(original[0, 0]), rtol=1e-5)
    volts = b.actuators["pf_active-coil_voltage"]
    assert np.allclose(seen["actuators"]["pf_active-coil_voltage"][1, 2, :20], volts[1, 2, :20].mean(), rtol=1e-4, atol=1e-2)
    assert seen["last_valid"] is b.last_valid and np.isnan(seen["targets"]["summary-ip"]).all()
    assert np.array_equal(b.inputs["summary-ip"], original, equal_nan=True)   # the caller's batch is not modified
```

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_tok_floor.py -q -W error`
Expected: an import error for `floor`.

- [ ] **Step 3: Implement the four functions of "Produces"**

Copies of a batch are made with `dataclasses.replace`. Attributes are carried over with `wrapper.__dict__.update(getattr(inner, "__dict__", {}))`.

- [ ] **Step 4: Run the tests**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_tok_floor.py -q -W error`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/tokamak_foundation_model/ignite/mast/tokenizers/floor.py tests/ignite/mast/test_tok_floor.py
git commit -m "ignite.mast: tokenizer floor, prediction pass-through and input round-trip as harness predictors"
```

### Task 7: settings, ladders, selection rules and the fit

**Files:**
- Create: `src/tokamak_foundation_model/ignite/mast/tokenizers/assign.py`, `src/tokamak_foundation_model/ignite/mast/tokenizers/fit.py`
- Test: `tests/ignite/mast/test_tok_assign.py`, `tests/ignite/mast/test_tok_fit.py`

**Interfaces:**
- Consumes: `registry.family_of(s: SignalSpec) -> str` (`"slow" | "high_rate" | "profile" | "map"`); `registry.all_signals() -> list[SignalSpec]` (the 39 signals; needs `tokamark` when called); `tasks.SPECTROGRAM_SIGNALS`; `tasks.ALL_TASKS`, `tasks.load_task(name)`, `tasks.best_published(name) -> float`; `transform.load_stats(path) -> dict[str, ChannelStats]`; `cache.read_shot(path: Path, names: Sequence[str] | None = None, *, stats_sha256: str | None = None) -> dict[str, CachedSignal | None]` where `CachedSignal.values` is float32 `(*values_shape, T)` in model units with NaN for missing; `cache.pick_evenly(shots, n) -> list[int]`; Task 2's `fit_ranges`, `GridSet`; Tasks 3 to 5's codecs and `fit_pca`; `frames.frame_len`.
- Produces in `assign.py`:
  - `DCT_RUNGS = (1, 2, 3, 4, 6, 8, 12, 16)`, `PCA_RUNGS = (0, 8, 16, 24, 32, 48, 64, 96)`, `BAND_RUNGS = (1, 2, 4, 8, 16, 32)`, `LOW_BAND_RUNGS = (4, 8, 16, 32)`, `VALUES_PER_TOKEN = {"profile": 4, "map": 2}`, `INPUT_ONLY_TARGET = 0.0176`, `N_BINS = 64`.
  - `@dataclass(frozen=True) class Setting` with fields `codec: str` (`"frame_dct" | "linear" | "band_dct"`), `k: tuple[int, ...]` (`(K,)` for `frame_dct`; `(n_values, values_per_token)` for `linear`; one K per band for `band_dct`), `domain: str = "linear"` (`"band"` for `band_dct`), `n_bins: int = N_BINS`, `source: str = ""` (empty for the three codecs of this plan; plan 1c names a fitted basis file or a checkpoint here; it is part of the JSON that `assignment_sha256` hashes, and `save_assignment` / `load_assignment` keep it).
  - `codec_kind(spec: SignalSpec) -> str`: `"linear"` for the profile and map families, `"band_dct"` for `tasks.SPECTROGRAM_SIGNALS`, else `"frame_dct"`.
  - `domains(spec: SignalSpec, compress: str) -> tuple[str, ...]`: `frame_dct`: `("linear",)`, plus `"compressed"` if `compress == "asinh"`; `linear`: `("signal", "linear")`, plus `"compressed"` if asinh; `band_dct`: `("band",)`.
  - `rungs(spec: SignalSpec, domain: str) -> tuple[Setting, ...]`: the ladder of a signal in rising order. `frame_dct`: one setting per K of `DCT_RUNGS` with `K <= frame_len(spec.dt)`. `linear`: one per n of `PCA_RUNGS` with `n <= prod(values_shape)`, with the family's values per token. `band_dct`: `band_rungs(None)`.
  - `band_rungs(low: int | None) -> tuple[Setting, ...]`: the two-step ladder of "Which number decides what", rule 5. With `low=None` the settings `(k0, 32, 32, 32, 32, 32, 32, 32)` for k0 in `LOW_BAND_RUNGS`; with `low=k0` the settings `(k0, k, k, k, k, k, k, k)` for k in `BAND_RUNGS`.
  - `choose_rung(floors: Sequence[float], target: float) -> tuple[int, bool]`: the index of the first floor at or under the target and True; if none, the last index and False. NaN never meets a target.
  - `choose_within(floors: Sequence[float], reference: float, tolerance: float = 0.02) -> int`: the index of the first floor at or under `reference * (1 + tolerance)`; the last index if none.
  - `reference_assignment(signals: Sequence[SignalSpec]) -> dict[str, Setting]`: the "own mean" reference of every signal: `Setting("frame_dct", (1,))`, `Setting("linear", (0, values_per_token), "signal")`, `Setting("band_dct", (1,) * 8, "band")`.
  - `uniform_assignment(signals: Sequence[SignalSpec], compress: Mapping[str, str], rung: int = -1) -> dict[str, Setting]`: every signal at rung `rung` (a Python index, limited to the ladder's length) of the ladder of its first domain.
  - `predicted_by() -> dict[str, tuple[str, ...]]` (signal name to the tasks that predict it) and `targets() -> dict[str, float | None]` for the 39 signals: 0.2 times the smallest `best_published` over the tasks that predict the signal; `INPUT_ONLY_TARGET` for a signal no task predicts; None for `tasks.SPECTROGRAM_SIGNALS`. Both need `tokamark`.
  - `assignment_sha256(a: Mapping[str, Setting]) -> str`: sha256 of the JSON of the settings with sorted keys.
  - `save_assignment(a: Mapping[str, Setting], path: Path, notes: Mapping[str, str] | None = None) -> None` (JSON `{"settings": ..., "notes": ..., "sha256": ...}`) and `load_assignment(path: Path) -> dict[str, Setting]` (raises `ValueError` if the stored sha256 does not match the settings).
- Produces in `fit.py`:
  - `MAX_COMPONENTS = 96`.
  - `@dataclass class SignalFit` with fields `name: str`, `kind: str`, `n_shots: int` (shots that had the signal), `n_skipped: int` (shots where it was None), `n_examples: int` (frames, or for `linear` profiles, with at least one valid sample, over all shots), `partly_valid: bool` (some example had missing points and valid ones), `ranges: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]` (domain to `(lo, hi, width)`, each `(n_tok, n_val)` at the signal's highest rung), `pca: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, float]]` (domain to the result of `fit_pca`; empty unless `kind == "linear"`).
  - `fit_signal(spec: SignalSpec, stats: ChannelStats, shots_values: Iterable[np.ndarray | None], kind: str | None = None) -> SignalFit`: `shots_values` yields, per shot, the cached values in model units or None. For each domain of the signal: build the codec at the highest rung (for `linear`, after `fit_pca` on every example with at least one valid point, with `MAX_COMPONENTS` components), encode every shot from model units, and `fit_ranges` over all values. `kind=None` means `assign.codec_kind(spec)`. A signal no shot has gives `n_shots == 0` and empty dicts.
  - `fit_all(cache_root: Path, shots: Sequence[int], *, workers: int, names: Sequence[str] | None = None) -> dict[str, SignalFit]`: reads `stats.npz` and the shot files of `cache_root` (with `stats_sha256` from its `manifest.json`), and fits each signal.
  - `save_fit(fits: Mapping[str, SignalFit], out_dir: Path, meta: dict) -> None` writes `fit.npz` and `fit.json`; `load_fit(out_dir: Path) -> tuple[dict[str, SignalFit], dict]`; `fit_sha256(out_dir: Path) -> str` is the sha256 of `fit.npz`.
  - `build_codec(spec: SignalSpec, stats: ChannelStats, fit: SignalFit, setting: Setting) -> Codec`: the codec of the setting with its grids (the first `n_val` columns of the fitted ranges; for `linear` the first `n_values` components regrouped into tokens) and `n_bins`. For `linear`, `ridge` is `(total_variance - eigenvalues[:n_values].sum()) / D` if `fit.partly_valid`, else 0. `signal_mean` and `signal_std` are `spec.mean` and `spec.std`. A domain the signal was not fitted in raises `ValueError` naming the domain; a `setting.codec` other than the three of this plan raises `ValueError` naming the codec (plan 1c's builder handles its own kinds before it falls back to this function).
  - `build_codecs(signals: Sequence[SignalSpec], stats: Mapping[str, ChannelStats], fits: Mapping[str, SignalFit], assignment: Mapping[str, Setting]) -> dict[str, Codec]`.

- [ ] **Step 1: Write the failing tests**

`tests/ignite/mast/test_tok_assign.py`:

```python
"""Settings, ladders and the selection rules (pure functions of numbers)."""

import math

import pytest

from tokamak_foundation_model.ignite.mast import tasks
from tokamak_foundation_model.ignite.mast.tokenizers import assign

IP = tasks.SignalSpec("summary-ip", 0.00025, (1,), 3.5e5, 3.0e5)
KAPPA = tasks.SignalSpec("equilibrium-elongation", 0.005, (1,), 1.6, 0.3)
LCFS = tasks.SignalSpec("equilibrium-lcfs_r", 0.005, (170,), 0.9, 0.4)
PSI = tasks.SignalSpec("equilibrium-psi", 0.005, (65, 65), 0.0, 0.1)
MIRNOV = tasks.SignalSpec(tasks.SPECTROGRAM_SIGNALS[1], 0.001024, (3, 256), 5.5, 266.0)


def test_the_kind_and_the_domains_of_a_signal():
    assert [assign.codec_kind(s) for s in (IP, KAPPA, LCFS, PSI, MIRNOV)] == [
        "frame_dct", "frame_dct", "linear", "linear", "band_dct"]
    assert assign.domains(IP, "none") == ("linear",) and assign.domains(IP, "asinh") == ("linear", "compressed")
    assert assign.domains(LCFS, "none") == ("signal", "linear")
    assert assign.domains(LCFS, "asinh") == ("signal", "linear", "compressed")
    assert assign.domains(MIRNOV, "asinh") == ("band",)


def test_the_ladders():
    assert [s.k for s in assign.rungs(IP, "linear")] == [(1,), (2,), (3,), (4,), (6,), (8,), (12,), (16,)]
    assert [s.k for s in assign.rungs(KAPPA, "linear")] == [(1,)]            # one sample per frame: K = 1 is exact
    assert [s.k for s in assign.rungs(LCFS, "signal")] == [(n, 4) for n in (0, 8, 16, 24, 32, 48, 64, 96)]
    assert assign.rungs(PSI, "signal")[4] == assign.Setting("linear", (32, 2), "signal", 64)
    assert [s.k[0] for s in assign.band_rungs(None)] == [4, 8, 16, 32] and assign.band_rungs(None)[0].k[1:] == (32,) * 7
    assert [s.k for s in assign.band_rungs(16)][1] == (16, 2, 2, 2, 2, 2, 2, 2) and assign.band_rungs(16)[0].domain == "band"


def test_the_smallest_rung_that_meets_the_target_is_chosen():
    assert assign.choose_rung([0.05, 0.02, 0.017, 0.01], 0.0176) == (2, True)
    assert assign.choose_rung([0.0176], 0.0176) == (0, True)                 # at the target counts
    assert assign.choose_rung([0.05, 0.03], 0.0176) == (1, False)            # no rung meets it: the largest, and a miss
    assert assign.choose_rung([math.nan, 0.01], 0.02) == (1, True)           # a rung that was not scored never meets
    assert assign.choose_within([0.9, 0.7, 0.685, 0.676], 0.676) == 2        # within 2% of the reference
    assert assign.choose_within([0.9, 0.7], 0.5) == 1


def test_the_reference_and_the_uniform_assignments():
    ref = assign.reference_assignment([IP, LCFS, PSI, MIRNOV])
    assert ref["summary-ip"] == assign.Setting("frame_dct", (1,))
    assert ref["equilibrium-lcfs_r"] == assign.Setting("linear", (0, 4), "signal")
    assert ref["equilibrium-psi"].k == (0, 2) and ref[MIRNOV.name] == assign.Setting("band_dct", (1,) * 8, "band")
    compress = {"summary-ip": "none", "equilibrium-lcfs_r": "none", MIRNOV.name: "asinh", "equilibrium-elongation": "none"}
    top = assign.uniform_assignment([IP, LCFS, MIRNOV], compress)
    assert top["summary-ip"].k == (16,) and top["equilibrium-lcfs_r"] == assign.Setting("linear", (96, 4), "signal")
    assert top[MIRNOV.name].k == (32,) * 8
    assert assign.uniform_assignment([IP], compress, rung=1)["summary-ip"].k == (2,)
    assert assign.uniform_assignment([KAPPA], compress, rung=3)["equilibrium-elongation"].k == (1,)


def test_an_assignment_is_stored_with_its_hash(tmp_path):
    a = {"summary-ip": assign.Setting("frame_dct", (4,)), "equilibrium-psi": assign.Setting("linear", (32, 2), "signal")}
    path = tmp_path / "v1.json"
    assign.save_assignment(a, path, notes={"summary-ip": "K=4 meets 0.0176 with 0.0161"})
    assert assign.load_assignment(path) == a
    b = dict(a, **{"summary-ip": assign.Setting("frame_dct", (6,))})
    assert assign.assignment_sha256(a) != assign.assignment_sha256(b) and len(assign.assignment_sha256(a)) == 64
    named = dict(a, **{"summary-ip": assign.Setting("frame_dct", (4,), source="basis/x.npz")})
    assert assign.Setting("frame_dct", (4,)).source == "" and assign.assignment_sha256(named) != assign.assignment_sha256(a)
    path.write_text(path.read_text().replace('"domain": "signal"', '"domain": "linear"'))
    with pytest.raises(ValueError, match="sha256"):
        assign.load_assignment(path)


def test_the_fidelity_targets_are_a_fifth_of_the_best_published_score():
    pytest.importorskip("tokamark")
    t = assign.targets()
    assert len(t) == 39
    assert t["equilibrium-lcfs_r"] == pytest.approx(0.2 * tasks.best_published("task_2-2"))
    assert t["equilibrium-psi"] == pytest.approx(0.2 * tasks.best_published("task_2-3"))
    assert t["summary-ip"] == pytest.approx(0.2 * tasks.best_published("task_2-1"))
    assert t["soft_x_rays-horizontal_cam_lower"] == pytest.approx(0.2 * tasks.best_published("task_4-1"))
    assert t["magnetics-flux_loop_flux"] == assign.INPUT_ONLY_TARGET
    assert all(t[name] is None for name in tasks.SPECTROGRAM_SIGNALS)
    assert assign.predicted_by()["summary-ip"] == ("task_2-1", "task_4-4")
```

`tests/ignite/mast/test_tok_fit.py`:

```python
"""Ranges and bases fitted from cached values (synthetic here; the cache itself in the trial)."""

import numpy as np
import pytest

from tokamak_foundation_model.ignite.mast import tasks, transform
from tokamak_foundation_model.ignite.mast.tokenizers import assign, codecs, fit

RNG = np.random.default_rng(0)
IP = tasks.SignalSpec("summary-ip", 0.00025, (1,), 3.5e5, 3.0e5)
LCFS = tasks.SignalSpec("equilibrium-lcfs_r", 0.005, (30,), 0.9, 0.4)


def st(compress, shape):
    return transform.ChannelStats(compress=compress, scale=np.ones(shape), mean=np.zeros(shape), std=np.ones(shape))


def series(n=3, t=400):
    return [RNG.standard_normal((1, t)).astype(np.float32) for _ in range(n)]


def test_the_fit_skips_and_counts_a_shot_without_the_signal():
    f = fit.fit_signal(IP, st("none", 1), [*series(2), None], kind="frame_dct")
    assert (f.n_shots, f.n_skipped, f.n_examples, f.partly_valid) == (2, 1, 40, False)
    lo, hi, width = f.ranges["linear"]
    assert lo.shape == hi.shape == width.shape == (1, 16) and (lo < hi).all() and (width > 0).all()
    assert set(f.ranges) == {"linear"} and f.pca == {}


def test_a_signal_no_shot_has_is_reported_and_not_fitted():
    f = fit.fit_signal(IP, st("none", 1), [None, None], kind="frame_dct")
    assert (f.n_shots, f.n_skipped, f.n_examples) == (0, 2, 0) and f.ranges == {} and f.pca == {}


def test_an_asinh_signal_is_fitted_in_both_domains():
    f = fit.fit_signal(IP, st("asinh", 1), series(), kind="frame_dct")
    assert set(f.ranges) == {"linear", "compressed"}


def test_a_profile_fit_has_a_basis_and_score_ranges_per_domain():
    shots = [RNG.standard_normal((30, 50)).astype(np.float32) for _ in range(4)]
    shots[0][::2, :10] = np.nan                                              # partly valid profiles
    f = fit.fit_signal(LCFS, st("none", 30), shots, kind="linear")
    assert set(f.pca) == set(f.ranges) == {"signal", "linear"} and f.partly_valid and f.n_examples == 200
    mean, basis, eig, total = f.pca["signal"]
    assert basis.shape == (30, 30) and f.ranges["signal"][0].shape == (1, 30)


def test_build_codec_makes_the_codec_of_the_setting():
    f = fit.fit_signal(IP, st("none", 1), series(), kind="frame_dct")
    c = fit.build_codec(IP, st("none", 1), f, assign.Setting("frame_dct", (4,)))
    assert isinstance(c, codecs.FrameDCT) and (c.n_tok, c.n_val) == (1, 4)
    assert c.grids.lo.shape == (1, 4) and c.grids.n_bins == 64 and np.array_equal(c.grids.lo, f.ranges["linear"][0][:, :4])
    with pytest.raises(ValueError, match="compressed"):
        fit.build_codec(IP, st("none", 1), f, assign.Setting("frame_dct", (4,), domain="compressed"))
    with pytest.raises(ValueError, match="frame_klt"):
        fit.build_codec(IP, st("none", 1), f, assign.Setting("frame_klt", (4,)))


def test_build_codec_groups_components_into_tokens_and_sets_the_ridge():
    shots = [RNG.standard_normal((30, 50)).astype(np.float32) for _ in range(4)]
    whole = fit.fit_signal(LCFS, st("none", 30), shots, kind="linear")
    c = fit.build_codec(LCFS, st("none", 30), whole, assign.Setting("linear", (8, 4), "signal"))
    assert isinstance(c, codecs.LinearCodec) and (c.n_tok, c.n_val, c.ridge) == (2, 4, 0.0)
    assert c.grids.lo.shape == (2, 4) and (c.signal_mean, c.signal_std) == (0.9, 0.4)
    assert np.array_equal(c.grids.lo.ravel(), whole.ranges["signal"][0][0, :8])
    shots[0][::2, :10] = np.nan
    holes = fit.fit_signal(LCFS, st("none", 30), shots, kind="linear")
    assert fit.build_codec(LCFS, st("none", 30), holes, assign.Setting("linear", (8, 4), "signal")).ridge > 0.0


def test_a_fit_is_stored_and_read_back(tmp_path):
    fits = {"summary-ip": fit.fit_signal(IP, st("asinh", 1), series(), kind="frame_dct"),
            "equilibrium-lcfs_r": fit.fit_signal(LCFS, st("none", 30), [RNG.standard_normal((30, 50)).astype(np.float32)],
                                                 kind="linear")}
    fit.save_fit(fits, tmp_path, {"n_shots": 3})
    back, meta = fit.load_fit(tmp_path)
    assert meta["n_shots"] == 3 and set(back) == set(fits) and len(fit.fit_sha256(tmp_path)) == 64
    assert np.array_equal(back["summary-ip"].ranges["compressed"][1], fits["summary-ip"].ranges["compressed"][1])
    assert np.array_equal(back["equilibrium-lcfs_r"].pca["signal"][1], fits["equilibrium-lcfs_r"].pca["signal"][1])
    assert back["equilibrium-lcfs_r"].n_examples == 50
```

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_tok_assign.py tests/ignite/mast/test_tok_fit.py -q -W error`
Expected: import errors for `assign` and `fit`.

- [ ] **Step 3: Implement `assign.py` and `fit.py`**

A `linear` signal is fitted in two passes over the shots: the basis, then the ranges of the scores at `min(MAX_COMPONENTS, D)` components in one token (hence range arrays of shape `(1, n)`). Ranges fitted at the highest rung serve every lower rung. `fit_all` reads the shot files with `transform.map_shots(fn, shots, args, workers)` (`fn((shot, *args))` in order, spawn start method, so `fn` is a module-level function); one signal over 2,000 shots fits in memory (the flux map: about 2 GB).

- [ ] **Step 4: Run the tests**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_tok_assign.py tests/ignite/mast/test_tok_fit.py -q -W error`
Expected: 13 passed.

- [ ] **Step 5: Trial on four cached shots (CPU, about a minute)**

```bash
CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh - <<'EOF'
from tokamak_foundation_model.ignite.mast import cache, paths, tasks
from tokamak_foundation_model.ignite.mast.tokenizers import fit
shots = cache.pick_evenly(tasks.split_shots("train"), 4)
fits = fit.fit_all(paths.cache_dir(), shots, workers=4)
for name, f in sorted(fits.items()):
    print(name, f.kind, f.n_shots, f.n_skipped, f.n_examples, f.partly_valid, sorted(f.ranges))
EOF
```

Expected: 39 lines; `n_shots + n_skipped == 4` on every line; the four profile and map signals have kind `linear`, the two `magnetics-b_field_*_probe_*` spectrogram signals `band_dct`; the 15 asinh signals of the cache list `compressed` among their domains (for the two spectrogram signals only `band`). Nothing is written. Put the 39 lines in the report.

- [ ] **Step 6: Commit**

```bash
git add src/tokamak_foundation_model/ignite/mast/tokenizers/assign.py src/tokamak_foundation_model/ignite/mast/tokenizers/fit.py tests/ignite/mast/test_tok_assign.py tests/ignite/mast/test_tok_fit.py
git commit -m "ignite.mast: tokenizer settings, ladders, selection rules and the fit of ranges and bases"
```

### Task 8: the command: `fit` and `score`

**Files:**
- Create: `scripts/mast/run_tokenizer.py`
- Test: `tests/ignite/mast/test_tok_command.py`

**Interfaces:**
- Consumes (existing, quoted from the code at 752b15a7):
  - `harness.score(task: TaskSpec, predict_fn: PredictFn, out_dir: Path, *, split: str, shots: list[int] | None = None, history_s: float | None = None, **kw) -> ScoreResult`; `kw` is `batch_size: int = 64`, `num_workers: int = 32`. `ScoreResult` has `task`, `task_nrmse`, `per_signal` (a DataFrame with columns `feature_name`, `NRMSE`), `per_shot`, `n_windows`, `out_dir`.
  - `baselines.write_score_json(result: ScoreResult, predictor: str, split: str) -> Path` writes `score.json` (keys `task, predictor, split, nrmse, n_shots, n_windows, git_revision, date, code_sha256`) into `result.out_dir`; `baselines.read_kept_score(out_dir: Path, task: str, predictor: str, split: str) -> tuple[dict | None, str]` returns the stored record only if it names this task, predictor and split and was written by the scoring code now loaded.
  - `direct.direct_predictor(task: TaskSpec, checkpoint: Path, device) -> PredictFn` (its function carries `history_s`); `headroom.table.direct_run_dir(root: Path, task_name: str, variant: str) -> Path` and `table.predictor_name(predictor: str, variant: str) -> str` (`"direct"` for the variant `"base"`, else `"direct.<variant>"`).
  - `cache.pick_evenly(shots: Sequence[int], n: int) -> list[int]`; `tasks.split_shots(split: str) -> list[int]`; `tasks.load_task(name: str) -> TaskSpec`; `transform.load_stats(path: Path) -> dict[str, ChannelStats]` and `transform.stats_sha256(path: Path) -> str` (both take the path of `stats.npz`); `paths.stage1_dir()`, `paths.stage0_dir()`, `paths.cache_dir()`, `paths.tokamark_zarr_root()`, `paths.prepare_write_root(path: Path) -> Path`.
  - Task 6's `floor_predictor`, `passthrough_predictor`, `input_roundtrip_predictor`, `blind_to_targets`; Task 7's `fit.fit_all`, `save_fit`, `load_fit`, `fit_sha256`, `build_codecs`, `assign.load_assignment`, `assignment_sha256`; Task 1's `tokenizers.code_sha256()`.
- Produces (functions of the script; the tests load it by path):
  - `STAGE1_SPLITS = ("val_select", "val_report")`, `INSTRUMENTS = ("floor", "passthrough", "inputs", "raw")`.
  - `predictor_name(instrument: str, assignment_name: str | None, mode: str | None, variant: str | None = None) -> str`: `tok.<assignment>.<mode>`, `pass.<assignment>.<mode>.<variant>`, `inputs.<assignment>.<mode>.<variant>`, `raw.<variant>`. The assignment's name is its file name without `.json`.
  - `split_label(split: str, n_shots: int | None) -> str`: the split, or `<split>-n<N>` for a subset.
  - `kept_score(out_dir: Path, task: str, predictor: str, label: str, provenance: Mapping[str, str], force: bool) -> tuple[dict | None, str]`: `baselines.read_kept_score` and then every key of `provenance` equal to the stored value; else `(None, reason)`.
  - `score_one(task: TaskSpec, predict_fn: PredictFn, *, predictor: str, out_dir: Path, split: str, label: str, shots: list[int] | None, provenance: Mapping[str, str], records: Mapping[str, object], force: bool = False, num_workers: int = 32, batch_size: int = 64) -> dict`: returns the kept record, or scores with `harness.score(..., history_s=getattr(predict_fn, "history_s", None))`, writes `score.json` (`write_score_json(result, predictor, label)`, then the provenance and the records added, written atomically) and `signals.csv` (the `per_signal` table) and returns the record.
  - `provenance_of(assignment_path: Path | None, fit_dir: Path | None, cache_root: Path, checkpoint: Path | None = None) -> dict[str, str]`: in this order `tokenizer_code_sha256`, `assignment_sha256`, `fit_sha256`, `stats_sha256` (the first four only with an assignment), `tokamark_revision` (from the cache's `manifest.json`), `data_root` (`str(paths.tokamark_zarr_root())`), and `checkpoint_sha256` with a checkpoint. `ValueError` naming "statistics" if `stats.npz`, the manifest and the fit's `meta["stats_sha256"]` do not carry one hash.
  - `main(argv: list[str] | None = None) -> int` with two sub-commands in this task:
    - `fit [--cache-dir DIR] [--n-shots 2000] [--workers 64] [--force]`: fits on `cache.pick_evenly(tasks.split_shots("train"), n)`, writes `<stage1>/fit/v1/` with `fit.save_fit`; the meta holds `date`, `n_shots`, `shots`, `stats_sha256` and `tokamark_revision` (from the manifest), `tokenizer_code_sha256`, `cache_dir`. Prints one line per signal (`n_shots`, `n_skipped`, `n_examples`; "no data" for a signal no shot had). Exit 2 if a fit exists and `--force` is not given.
    - `score --task T --split S [--assignment FILE --mode M] [--n-shots N] [--instrument I] [--variant V] [--gpu N] [--share-gpu] [--cache-dir DIR] [--stage0-dir DIR] [--num-workers 32] [--batch-size 64] [--force]`: `S` is one of `STAGE1_SPLITS` (the test split is not a choice). `floor` (the default) uses no GPU and refuses `--gpu`; the other three need `--gpu` and `--variant` (default `base`), and `raw` takes no assignment. Scores go to `<stage1>/scores/<task>/<predictor>/<label>/`. Refusals (exit 2) come before any file is read. The last line printed is `<task> <predictor> <label>: nrmse <v>, <n> windows, <s> s`, or `kept ...`.

- [ ] **Step 1: Write the failing tests**

```python
"""The tokenizer command. Its functions are tested with the harness replaced; runs on data are trials."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tokamak_foundation_model.ignite.mast import harness, paths, tasks, transform
from tokamak_foundation_model.ignite.mast.tokenizers import assign, fit

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "mast" / "run_tokenizer.py"
IP = tasks.SignalSpec("summary-ip", 0.00025, (1,), 3.5e5, 3.0e5)
PROV = {"tokenizer_code_sha256": "a" * 64, "assignment_sha256": "b" * 64, "fit_sha256": "c" * 64,
        "stats_sha256": "d" * 64}


def load():
    spec = importlib.util.spec_from_file_location("run_tokenizer_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def ip_fit():
    st = transform.ChannelStats(compress="none", scale=np.ones(1), mean=np.zeros(1), std=np.ones(1))
    shot = np.random.default_rng(0).standard_normal((1, 400)).astype(np.float32)
    return fit.fit_signal(IP, st, [shot], kind="frame_dct")


def fake_score(calls):
    def score(task, predict_fn, out_dir, *, split, shots=None, history_s=None, **kw):
        calls.append({"split": split, "shots": shots, "history_s": history_s, **kw})
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return harness.ScoreResult(task=task.name, task_nrmse=0.25,
                                   per_signal=pd.DataFrame({"feature_name": ["summary-ip"], "NRMSE": [0.25]}),
                                   per_shot=pd.DataFrame({"shot_id": [1, 2], "NRMSE": [0.2, 0.3]}),
                                   n_windows=8, out_dir=Path(out_dir))
    return score


def test_names_of_predictors_and_splits():
    rt = load()
    assert rt.predictor_name("floor", "v1", "ranged") == "tok.v1.ranged"
    assert rt.predictor_name("passthrough", "v1", "unquantized", "lr1e4") == "pass.v1.unquantized.lr1e4"
    assert rt.predictor_name("inputs", "v1", "nearest", "base") == "inputs.v1.nearest.base"
    assert rt.predictor_name("raw", None, None, "base") == "raw.base"
    assert rt.split_label("val_report", None) == "val_report" and rt.split_label("val_select", 100) == "val_select-n100"


def test_a_score_is_stored_with_its_provenance_and_kept_while_nothing_changed(tmp_path, monkeypatch, toy_task):
    rt = load()
    calls = []
    monkeypatch.setattr(rt.harness, "score", fake_score(calls))

    def predict(batch):
        return {}

    predict.history_s = 0.05
    out = tmp_path / "scores" / "task_toy" / "tok.v1.ranged" / "val_select-n100"
    kw = dict(predictor="tok.v1.ranged", out_dir=out, split="val_select", label="val_select-n100", shots=[1, 2],
              provenance=PROV, records={"mode": "ranged"})
    record = rt.score_one(toy_task, predict, **kw)
    assert record["nrmse"] == 0.25 and record["split"] == "val_select-n100" and record["predictor"] == "tok.v1.ranged"
    assert record["assignment_sha256"] == "b" * 64 and record["mode"] == "ranged"
    assert calls[0]["split"] == "val_select" and calls[0]["shots"] == [1, 2] and calls[0]["history_s"] == 0.05
    assert json.loads((out / "score.json").read_text()) == record
    assert pd.read_csv(out / "signals.csv").to_dict("records") == [{"feature_name": "summary-ip", "NRMSE": 0.25}]
    assert rt.score_one(toy_task, predict, **kw) == record and len(calls) == 1               # kept
    rt.score_one(toy_task, predict, **{**kw, "provenance": {**PROV, "assignment_sha256": "e" * 64}})
    assert len(calls) == 2                                                                   # another assignment
    rt.score_one(toy_task, predict, force=True, **kw)
    assert len(calls) == 3


def test_provenance_names_the_code_the_assignment_the_fit_and_the_data(tmp_path):
    rt = load()
    cache_root, fit_dir = tmp_path / "cache", tmp_path / "fit"
    cache_root.mkdir()
    fit_dir.mkdir()
    (cache_root / "stats.npz").write_bytes(b"statistics")
    sha = transform.stats_sha256(cache_root / "stats.npz")
    (cache_root / "manifest.json").write_text(json.dumps({"stats_sha256": sha, "tokamark_revision": "146b0a0f8738"}))
    assign.save_assignment({"summary-ip": assign.Setting("frame_dct", (4,))}, tmp_path / "v1.json")
    fit.save_fit({"summary-ip": ip_fit()}, fit_dir, {"stats_sha256": sha})
    p = rt.provenance_of(tmp_path / "v1.json", fit_dir, cache_root)
    assert list(p) == ["tokenizer_code_sha256", "assignment_sha256", "fit_sha256", "stats_sha256", "tokamark_revision",
                       "data_root"]
    assert p["stats_sha256"] == sha and p["tokamark_revision"] == "146b0a0f8738"
    assert p["data_root"] == str(paths.tokamark_zarr_root()) and p["fit_sha256"] == fit.fit_sha256(fit_dir)
    assert list(rt.provenance_of(None, None, cache_root, tmp_path / "v1.json")) == [
        "tokamark_revision", "data_root", "checkpoint_sha256"]                  # `raw`: any file stands for a checkpoint
    (cache_root / "stats.npz").write_bytes(b"other statistics")
    with pytest.raises(ValueError, match="statistics"):
        rt.provenance_of(tmp_path / "v1.json", fit_dir, cache_root)


def test_the_test_split_and_a_wrong_use_of_the_gpu_are_refused(tmp_path, monkeypatch, capsys):
    rt = load()
    monkeypatch.setenv("IGNITE_MAST_ROOT", str(tmp_path))
    base = ["score", "--task", "task_1-1", "--assignment", str(tmp_path / "v1.json"), "--mode", "ranged"]
    with pytest.raises(SystemExit):
        rt.main([*base, "--split", "test"])
    assert "invalid choice" in capsys.readouterr().err
    assert rt.main([*base, "--split", "val_report", "--instrument", "passthrough"]) == 2
    assert "--gpu" in capsys.readouterr().err
    assert rt.main([*base, "--split", "val_report", "--gpu", "0"]) == 2          # the floor uses no GPU
    assert "--gpu" in capsys.readouterr().err
    assert not (tmp_path / "stage1").exists()                                    # refused before anything is written


def test_fit_writes_once_and_refuses_to_overwrite(tmp_path, monkeypatch, capsys):
    rt = load()
    monkeypatch.setenv("IGNITE_MAST_ROOT", str(tmp_path))
    cache_root = tmp_path / "cache" / "v1"
    cache_root.mkdir(parents=True)
    (cache_root / "manifest.json").write_text(json.dumps({"stats_sha256": "d" * 64, "tokamark_revision": "146b0a0f8738"}))
    monkeypatch.setattr(rt.tasks, "split_shots", lambda split: list(range(30000, 30040)))
    seen = {}

    def fit_all(root, shots, *, workers, names=None):
        seen.update(root=Path(root), shots=list(shots), workers=workers)
        return {"summary-ip": ip_fit()}

    monkeypatch.setattr(rt.fit, "fit_all", fit_all)
    assert rt.main(["fit", "--n-shots", "4", "--workers", "2"]) == 0
    assert seen == {"root": cache_root, "shots": [30000, 30013, 30026, 30039], "workers": 2}
    fits, meta = fit.load_fit(tmp_path / "stage1" / "fit" / "v1")
    assert set(fits) == {"summary-ip"} and meta["n_shots"] == 4 and meta["stats_sha256"] == "d" * 64
    assert meta["shots"] == [30000, 30013, 30026, 30039] and len(meta["tokenizer_code_sha256"]) == 64
    assert rt.main(["fit", "--n-shots", "4", "--workers", "2"]) == 2
    assert "--force" in capsys.readouterr().err
```

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_tok_command.py -q -W error`
Expected: FAIL (`FileNotFoundError` for `run_tokenizer.py`).

- [ ] **Step 3: Implement the script**

Call `harness.score`, `fit.fit_all`, `tasks.split_shots` and `tasks.load_task` through their modules, so that the tests can replace them. Codecs are built only for the signals the instrument touches (the task's outputs for `floor` and `passthrough`; its inputs and actuators for `inputs`), so a signal without data stops only the tasks that use it, with a message that names it.

For `passthrough`, `inputs` and `raw`, select the GPU as `scripts/mast/run_headroom.py` does under `if is_direct:` in its `main`. Load that script by path (`importlib.util.spec_from_file_location`; `py.sh` runs Python with `-I`) and use its `_look_up_gpu(gpu: int) -> tuple[str, list[int]] | None` (the GPU's uuid and its compute pids; None after it printed why), `_refuse_busy_gpu(gpu: int, pids: list[int], share: bool = False) -> bool` and `stale_reason(root: Path, task_name: str, name: str, split: str) -> str`. Set `CUDA_VISIBLE_DEVICES` to the uuid before `headroom.direct` is imported; the device is then `"cuda:0"`. The checkpoint is `table.direct_run_dir(stage0_dir, task, variant) / "best.pt"`; exit 2 if it does not exist. Store `direct_stale_reason = stale_reason(stage0_dir, task, table.predictor_name("direct", variant), "val_report")` among the records and print it as a warning if it is not empty. The inner predictor is always wrapped by `floor.blind_to_targets`.

- [ ] **Step 4: Run the tests**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_tok_command.py -q -W error`
Expected: 5 passed.

- [ ] **Step 5: Trial: a fit on 8 shots and one floor on 2 shots (CPU, about 5 minutes)**

```bash
export IGNITE_MAST_ROOT=/models/intermediate/mast/trials/stage1b-task8
CACHE=/models/intermediate/mast/cache/v1
CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh scripts/mast/run_tokenizer.py fit --cache-dir $CACHE --n-shots 8 --workers 8
CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh - <<'EOF'
import os
from pathlib import Path
from tokamak_foundation_model.ignite.mast import registry, transform
from tokamak_foundation_model.ignite.mast.tokenizers import assign
stats = transform.load_stats(Path("/models/intermediate/mast/cache/v1/stats.npz"))
top = assign.uniform_assignment(registry.all_signals(), {n: s.compress for n, s in stats.items()})
assign.save_assignment(top, Path(os.environ["IGNITE_MAST_ROOT"]) / "top.json")
EOF
CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh scripts/mast/run_tokenizer.py score --task task_4-4 \
    --assignment $IGNITE_MAST_ROOT/top.json --mode unquantized --split val_select --n-shots 2 \
    --num-workers 4 --cache-dir $CACHE
```

Expected: the fit prints 39 lines and writes `$IGNITE_MAST_ROOT/stage1/fit/v1/fit.npz`; the score ends with `task_4-4 tok.top.unquantized val_select-n2: nrmse <v>, <n> windows, <s> s` with `v` finite and under 0.02 (16 of 20 values per frame of the plasma current; the rough look gave 0.009 with 8). Running the score line again prints `kept`. Report the lines and the two durations.

- [ ] **Step 6: Commit**

```bash
git add scripts/mast/run_tokenizer.py tests/ignite/mast/test_tok_command.py
git commit -m "ignite.mast: the tokenizer command fits ranges and bases and scores floors with their provenance"
```

### Task 9: ladders and the assignment

**Files:**
- Create: `src/tokamak_foundation_model/ignite/mast/tokenizers/ladder.py`
- Modify: `scripts/mast/run_tokenizer.py` (sub-commands `ladder` and `assign`)
- Test: `tests/ignite/mast/test_tok_ladder.py`

**Interfaces:**
- Consumes: Task 7's `assign.Setting`, `rungs`, `band_rungs`, `domains`, `codec_kind`, `choose_rung(floors, target) -> tuple[int, bool]`, `choose_within(floors, reference, tolerance=0.02) -> int`, `save_assignment`, `fit.build_codec`; Task 5's `BandDCT`; `Codec.roundtrip`; `transform.to_physical(z: np.ndarray, st: ChannelStats) -> np.ndarray` and `transform.to_model` (both take exactly `(*values_shape, T)`); `frames.frame_index`; `harness.iter_windows(task: TaskSpec, shots: list[int], *, test_mode: bool, history_s: float | None = None, batch_size: int = 64, num_workers: int = 32, shuffle: bool = False, stride_s: float | None = None) -> Iterator[WindowBatch]` (`test_mode=True` gives the benchmark's evaluation windows); `harness.score_batches(task_name: str, batches: Iterable[WindowBatch], predict_fn: PredictFn, out_dir: Path) -> ScoreResult`; `baselines.mean_predictor(task: TaskSpec) -> PredictFn`; `ignite.gate.spectral_contrast_ratio(recon, target, band_bins: Optional[int] = None, prefix: str = "") -> Dict[str, float]` (arrays `(B, C, F, T)`; keys `spectral_contrast_ratio` and `spectral_contrast_gt`; per column the contrast is the maximum minus the median over frequency).
- Produces in `ladder.py`:
  - `LADDER_TASKS = ("task_1-2", "task_1-3", "task_2-1", "task_3-1", "task_3-2", "task_4-5")`.
  - `values_per_frame(spec: SignalSpec, setting: Setting) -> int`: `prod(values_shape) * K` for `frame_dct`, `n_values` for `linear`, `P * sum(k)` for `band_dct`. Plan 1c adds two codec names whose `k` holds kept values, and this function counts them so that the gate needs no change later: `recoded` (one tokenizer per channel) counts like the signal's own kind (`assign.codec_kind(spec)`): `prod(values_shape) * k[0]`, `k[0]`, or `P * sum(k)`; `recoded_concat` (all channels in one) is `k[0]`. Any other name raises `ValueError`.
  - `passes(task: TaskSpec, compress: Mapping[str, str]) -> list[tuple[str, int, dict[str, Setting]]]`: `(domain, rung index, settings of the task's output signals)` for each pass of a task without spectrogram outputs. The domains are the union of the outputs' `assign.domains`, in first-seen order; a signal that lacks a domain keeps its first one, and a signal whose ladder is shorter stays on its last rung. A pass whose settings equal an earlier pass's is left out.
  - `run_ladder(task: TaskSpec, passes, build: Callable[[dict[str, Setting]], dict[str, Codec]], score: Callable[[str, Mapping[str, Codec], str], Mapping[str, float] | None]) -> pd.DataFrame`: `score(kind, codecs, name)` returns the per-signal NRMSE of kind `"floor"` or `"pass"` (prediction pass-through), or None if that kind is not measured; `score("raw", {}, "raw")` is asked once. Columns `signal, domain, k, values_per_frame, floor, raw, passed`; one row per distinct (signal, setting); `k` is K for a series and the number of components for a profile or map; NaN for what was not measured. The pass names are `ladder-<domain>-<rung index>`.
  - `shot_error(codec: Codec, spec: SignalSpec, z_model: np.ndarray, mode: str = "unquantized") -> float | None`: one shot's cached values `(*values_shape, T)` in model units; the RMSE of the round-trip over the valid samples, in TokaMark units; None if there is no valid sample. For a series the samples after the last whole frame are left out; a spectrogram (a codec whose `domain` is `"band"`; the class is not looked at, so that plan 1c's codecs can be given) keeps all its columns (its frames have 4 or 5).
  - `band_contrast(codec: Codec, z_model: np.ndarray, first_bin: int = 32) -> tuple[float, float] | None` (any codec whose `domain` is `"band"`; it uses only the `Codec` methods, so plan 1c's codecs can be given): for one shot's cached spectrogram `(P, n_bins, T)`: the frame-mean spectra (mean of linear power over each frame's valid columns) and their round-trip, both converted to model units, bins from `first_bin` on, given to `spectral_contrast_ratio` as `(1, P, bins, frames)`. Returns `(ratio, weight)` with `weight = spectral_contrast_gt * P * frames`; None if no frame is valid.
  - `choose(rows: pd.DataFrame, target: float, domains: Sequence[str], max_cost: float = 0.01) -> tuple[str, int, str]`: one signal's rows; returns `(domain, rung index in that domain's rows, note)` by rules 2 and 3. A domain meets the target at `choose_rung(floors, target)`; among the domains that meet it the smallest index wins, then the earlier domain. If none meets it: the cost of a rung is `passed / raw - 1`, the smallest index with a cost at or under `max_cost` wins, then the earlier domain; if no cost is measured or none is that small, the last rung of the domain whose last floor is lowest. The note starts with `meets <target> with <floor>` or `misses <target>: floor <floor>` (four decimals) and then says `pass-through cost <x.x%>`, or `no pass-through measurement`. `ValueError` naming "no ladder" for an empty table.
  - `choose_band(rows: pd.DataFrame, tolerance: float = 0.02, min_contrast: float = 0.9) -> tuple[tuple[int, ...], str]`: the rows of the spectrogram signals (columns `signal, step, k0, k, floor, contrast`; step `"a"` has one row per `k0`, step `"b"` one row per common `k` of bands 1 to 7). Rule 5: per signal, the step-a index is `choose_within(floors_a, floor at k0 = 32)`; the largest index over the signals gives `k0`. Per signal, the step-b index is the larger of `choose_within(floors_b, that signal's step-a floor at the chosen k0)` and the first index whose `contrast` is at least `min_contrast`; the largest over the signals gives `k`. Returns `(k0, k, k, k, k, k, k, k)` and a note with each signal's frame oracle, its two chosen floors and its contrast (four decimals). A signal without rows is left out of the choice and named in the note.
  - `build_assignment(signals: Sequence[SignalSpec], compress: Mapping[str, str], ladders: pd.DataFrame, targets: Mapping[str, float | None]) -> tuple[dict[str, Setting], dict[str, str]]`: the assignment and one note per signal. A signal with one sample per frame gets `Setting("frame_dct", (1,))` and the note `one sample per frame: K = 1 is exact`; the spectrogram signals share the result of `choose_band`; every other signal goes through `choose`. `ValueError` naming the signal if it has no rows.
- Produces in the script:
  - `ladder --task T [--n-shots 100] [--variant V --gpu N [--share-gpu]] [--cache-dir DIR] [--stage0-dir DIR] [--num-workers 32] [--batch-size 64]`: loads the evaluation windows of `cache.pick_evenly(tasks.split_shots("val_select"), n)` once into a list (`harness.iter_windows(..., test_mode=True, history_s=<the direct model's, if a variant is given>)`) and scores every pass from memory with `harness.score_batches`, under `<stage1>/scores/<task>/tok.<pass name>.unquantized/val_select-n<N>/`. With a variant, the direct model's raw predictions are computed once (blind to the targets), stored by each batch's first (shot, window) pair, and every pass is also scored as pass-through. The mean baseline is scored on the same windows (column `mean_baseline`). For a task with spectrogram outputs the passes are step a, then step b at the `k0` that `choose_band`'s step-a rule gives, and each step-b rung gets its `contrast` from the cache (`band_contrast` over the same shots, pooled as the weighted mean of the ratios). Writes `<stage1>/ladders/<task>.csv` and `<task>.json` (shots, `fit_sha256`, `tokenizer_code_sha256`, variant, `checkpoint_sha256`).
  - `ladder --inputs [--n-shots 100] [--workers 64] [--cache-dir DIR]`: rule 4. For each signal no task predicts, every domain and rung: the mean over the shots of `shot_error` (shots without the signal are skipped and counted). Writes `<stage1>/ladders/inputs.csv` with the same columns (`floor` holds the error).
  - `assign --name NAME [--from OLD --raise-inputs] [--reference] [--cache-dir DIR]`: reads the seven ladder files (exit 2 naming the missing ones, or one made with another fit), applies `build_assignment` with `assign.targets()`, writes `<stage1>/assignments/<NAME>.json` with the notes and prints one line per signal (codec, K, domain, values per frame, note) and the total of values per frame. `--from OLD --raise-inputs` copies `OLD` with each input-only signal one rung higher (rule 4). `--reference` writes `assign.reference_assignment`. An existing name is refused (exit 2).

- [ ] **Step 1: Write the failing tests**

```python
"""Ladder passes, cache-side errors and the choice of K: rules 2 to 5 of the plan, as code."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from tokamak_foundation_model.ignite.mast import tasks, transform
from tokamak_foundation_model.ignite.mast.tokenizers import assign, codecs, ladder

IP = tasks.SignalSpec("summary-ip", 0.00025, (1,), 3.5e5, 3.0e5)
KAPPA = tasks.SignalSpec("equilibrium-elongation", 0.005, (1,), 1.6, 0.3)
LCFS = tasks.SignalSpec("equilibrium-lcfs_r", 0.005, (170,), 0.9, 0.4)
CC, OMV = (tasks.SignalSpec(n, 0.001024, (3, 256), 0.0, 1.0) for n in tasks.SPECTROGRAM_SIGNALS)
KS = [1, 2, 3, 4, 6, 8, 12, 16]
IP_FLOORS = [0.076, 0.038, 0.025, 0.016, 0.012, 0.009, 0.005, 0.002]
SXR_FLOORS = [0.098, 0.081, 0.075, 0.069, 0.067, 0.065, 0.060, 0.055]


def rows(signal, domain, ks, floors, raw=np.nan, passed=None):
    passed = [np.nan] * len(ks) if passed is None else passed
    return pd.DataFrame({"signal": signal, "domain": domain, "k": ks, "floor": floors, "raw": raw, "passed": passed})


def band_rows(signal, low, rest, contrast):
    a = pd.DataFrame({"signal": signal, "step": "a", "k0": [4, 8, 16, 32], "k": 32, "floor": low, "contrast": np.nan})
    b = pd.DataFrame({"signal": signal, "step": "b", "k0": np.nan, "k": [1, 2, 4, 8, 16, 32], "floor": rest,
                      "contrast": contrast})
    return pd.concat([a, b])


def spectro(contrast_cc, contrast_omv):
    return pd.concat([
        band_rows(CC.name, [0.760, 0.700, 0.688, 0.676], [0.6890, 0.6885, 0.6882, 0.6881, 0.6880, 0.6880], contrast_cc),
        band_rows(OMV.name, [0.477, 0.438, 0.424, 0.422], [0.470, 0.440, 0.428, 0.425, 0.424, 0.424], contrast_omv)])


def unit_stats(*shape):
    return transform.ChannelStats(compress="none", scale=np.ones(shape), mean=np.zeros(shape), std=np.ones(shape))


def test_the_passes_of_a_task_cover_every_domain_and_rung(toy_task):
    p = ladder.passes(toy_task, {"summary-ip": "none"})
    assert [(d, i) for d, i, _ in p] == [("linear", i) for i in range(8)]
    assert p[3][2] == {"summary-ip": assign.Setting("frame_dct", (4,))}
    two = dataclasses.replace(toy_task, outputs=(IP, KAPPA))
    p = ladder.passes(two, {"summary-ip": "asinh", "equilibrium-elongation": "none"})
    assert len(p) == 16 and p[8][:2] == ("compressed", 0) and p[8][2]["summary-ip"].domain == "compressed"
    assert all(s["equilibrium-elongation"] == assign.Setting("frame_dct", (1,)) for _, _, s in p)
    assert ladder.values_per_frame(LCFS, assign.Setting("linear", (32, 4), "signal")) == 32
    assert ladder.values_per_frame(CC, assign.Setting("band_dct", (16, 8, 8, 8, 8, 8, 8, 8), "band")) == 3 * 72
    assert ladder.values_per_frame(tasks.SignalSpec("c", 0.00025, (10,), 0.0, 1.0), assign.Setting("frame_dct", (4,))) == 40
    coils = tasks.SignalSpec("c", 0.00025, (10,), 0.0, 1.0)                 # the two codec names of plan 1c
    assert ladder.values_per_frame(coils, assign.Setting("recoded", (4,), source="klt.series")) == 40
    assert ladder.values_per_frame(coils, assign.Setting("recoded_concat", (24,), source="pca.c")) == 24
    assert ladder.values_per_frame(LCFS, assign.Setting("recoded", (32,), "signal", source="net.profile.s0")) == 32
    assert ladder.values_per_frame(CC, assign.Setting("recoded", (16, 8, 8, 8, 8, 8, 8, 8), "band", source="klt.band")) == 3 * 72
    with pytest.raises(ValueError):
        ladder.values_per_frame(coils, assign.Setting("something", (4,)))


def test_run_ladder_scores_every_pass_and_keeps_what_was_not_measured_missing(toy_task):
    seen = []

    def build(settings):
        return {"summary-ip": codecs.FrameDCT("summary-ip", 0.00025, (1,), unit_stats(1), k=settings["summary-ip"].k[0])}

    def score(kind, tokenizers, name):
        seen.append((kind, name))
        return {"summary-ip": 0.08 / tokenizers["summary-ip"].n_val} if kind == "floor" else None

    table = ladder.run_ladder(toy_task, ladder.passes(toy_task, {"summary-ip": "none"}), build, score)
    assert table["k"].tolist() == KS and table["values_per_frame"].tolist() == KS
    assert table["floor"].tolist() == pytest.approx([0.08 / k for k in KS])
    assert table["passed"].isna().all() and table["raw"].isna().all() and (table["domain"] == "linear").all()
    assert [n for kind, n in seen if kind == "floor"] == [f"ladder-linear-{i}" for i in range(8)]
    assert seen.count(("raw", "raw")) == 1


def test_rule_2_the_smallest_rung_that_meets_the_target():
    lin = rows("summary-ip", "linear", KS, IP_FLOORS)
    domain, index, note = ladder.choose(lin, 0.0176, ("linear",))
    assert (domain, index) == ("linear", 3) and note.startswith("meets 0.0176 with 0.0160")
    com = rows("summary-ip", "compressed", KS, [0.070, 0.030, 0.017, 0.015, 0.012, 0.009, 0.005, 0.002])
    assert ladder.choose(pd.concat([lin, com]), 0.0176, ("linear", "compressed"))[:2] == ("compressed", 2)
    tie = pd.concat([lin, rows("summary-ip", "compressed", KS, IP_FLOORS)])
    assert ladder.choose(tie, 0.0176, ("linear", "compressed"))[:2] == ("linear", 3)      # a tie: the earlier domain


def test_rule_3_when_no_rung_meets_the_target_the_pass_through_cost_decides():
    name = "soft_x_rays-horizontal_cam_lower"
    sxr = rows(name, "linear", KS, SXR_FLOORS, raw=0.2000,
               passed=[0.2100, 0.2030, 0.2016, 0.2010, 0.2008, 0.2005, 0.2003, 0.2001])
    domain, index, note = ladder.choose(sxr, 0.0424, ("linear",))
    assert (domain, index) == ("linear", 2)                    # 0.2016 / 0.2000 - 1 = 0.8%, the first at or under 1%
    assert note.startswith("misses 0.0424: floor 0.0750") and "pass-through cost 0.8%" in note
    domain, index, note = ladder.choose(rows(name, "linear", KS, SXR_FLOORS), 0.0424, ("linear",))
    assert (domain, index) == ("linear", 7) and "no pass-through measurement" in note
    costly = rows(name, "linear", KS, SXR_FLOORS, raw=0.2, passed=[0.25] * 8)
    assert ladder.choose(costly, 0.0424, ("linear",))[1] == 7                  # no rung within 1%: the largest


def test_a_ladder_that_was_not_scored_is_not_a_choice():
    with pytest.raises(ValueError, match="no ladder"):
        ladder.choose(rows("summary-ip", "linear", [], []), 0.0176, ("linear",))


def test_rule_5_two_steps_for_the_spectrograms_and_the_contrast_condition():
    k, note = ladder.choose_band(spectro([0.30, 0.55, 0.80, 0.93, 0.99, 1.0], [0.35, 0.60, 0.85, 0.95, 0.99, 1.0]))
    assert k == (16, 8, 8, 8, 8, 8, 8, 8)        # the benchmark alone would take 4; the contrast asks for 8
    assert "0.6760" in note and "0.4220" in note                               # the frame oracle of both signals
    assert ladder.choose_band(spectro([0.95] * 6, [0.95] * 6))[0] == (16, 4, 4, 4, 4, 4, 4, 4)


def test_the_cache_side_error_is_in_tokamark_units_and_skips_what_is_missing():
    spec = tasks.SignalSpec("s", 0.00025, (1,), 0.0, 2.0)
    z = np.tile(np.r_[np.zeros(10), np.ones(10)], 3)[None].astype(np.float32)   # 3 frames, each half 0 and half 1
    c1 = codecs.FrameDCT("s", 0.00025, (1,), unit_stats(1), k=1)
    assert ladder.shot_error(c1, spec, z) == pytest.approx(0.25)               # rms 0.5 about the frame mean, std 2
    assert ladder.shot_error(codecs.FrameDCT("s", 0.00025, (1,), unit_stats(1), k=20), spec, z) == pytest.approx(0.0, abs=1e-9)
    z[0, 20:40] = np.nan
    assert ladder.shot_error(c1, spec, z) == pytest.approx(0.25)               # a frame without data adds no zero
    assert ladder.shot_error(c1, spec, z[:, :50]) == pytest.approx(0.25)       # a last partial frame is left out
    assert ladder.shot_error(c1, spec, np.full((1, 60), np.nan, np.float32)) is None


def test_the_contrast_falls_when_a_spectral_line_is_averaged_away():
    st = transform.ChannelStats(compress="asinh", scale=np.ones((1, 256)), mean=np.zeros((1, 256)), std=np.ones((1, 256)))
    p = np.ones((1, 256, 50))                                                  # linear power, 11 frames
    p[0, 100] = 100.0                                                          # a line at bin 100
    z = transform.to_model(p, st).astype(np.float32)
    whole = ladder.band_contrast(codecs.BandDCT("m", 0.001024, (1, 256), st, [32] * 8), z)
    flat = ladder.band_contrast(codecs.BandDCT("m", 0.001024, (1, 256), st, [1] * 8), z)
    assert whole[0] == pytest.approx(1.0, abs=1e-4) and whole[1] > 0 and flat[0] < 0.5
    assert ladder.band_contrast(codecs.BandDCT("m", 0.001024, (1, 256), st, [1] * 8),
                                np.full((1, 256, 50), np.nan, np.float32)) is None


def test_build_assignment_applies_the_rules_signal_by_signal():
    pca = [0, 8, 16, 24, 32, 48, 64, 96]
    ladders = pd.concat([
        rows("summary-ip", "linear", KS, IP_FLOORS),
        rows("equilibrium-lcfs_r", "signal", pca, [0.20, 0.012, 0.0093, 0.0080, 0.0072, 0.0058, 0.0046, 0.0030]),
        rows("equilibrium-lcfs_r", "linear", pca, [0.20, 0.013, 0.0100, 0.0090, 0.0080, 0.0070, 0.0050, 0.0030]),
        spectro([0.95] * 6, [0.95] * 6)])
    compress = {"summary-ip": "none", "equilibrium-elongation": "none", "equilibrium-lcfs_r": "none",
                CC.name: "asinh", OMV.name: "asinh"}
    targets = {"summary-ip": 0.0176, "equilibrium-elongation": 0.0176, "equilibrium-lcfs_r": 0.0074,
               CC.name: None, OMV.name: None}
    a, notes = ladder.build_assignment([IP, KAPPA, LCFS, CC, OMV], compress, ladders, targets)
    assert a["summary-ip"] == assign.Setting("frame_dct", (4,))
    assert a["equilibrium-elongation"] == assign.Setting("frame_dct", (1,))
    assert a["equilibrium-lcfs_r"] == assign.Setting("linear", (32, 4), "signal")
    assert a[CC.name] == a[OMV.name] == assign.Setting("band_dct", (16, 4, 4, 4, 4, 4, 4, 4), "band")
    assert set(notes) == set(a) and notes["equilibrium-elongation"].startswith("one sample per frame")
    with pytest.raises(ValueError, match="summary-ip"):
        ladder.build_assignment([IP], compress, ladders[ladders["signal"] != "summary-ip"], targets)
```

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_tok_ladder.py -q -W error`
Expected: an import error for `ladder`.

- [ ] **Step 3: Implement `ladder.py` and the two sub-commands**

In the script, the `score` callable given to `run_ladder` closes over the list of batches: `"floor"` is `harness.score_batches(task.name, batches, floor.floor_predictor(task, codecs), out_dir).per_signal` as a dict; `"pass"` the same with `floor.passthrough_predictor(lookup, task, codecs)`, where `lookup(batch)` returns the stored raw prediction of the batch with that first (shot, window) pair (the predictor hands `lookup` a copy, so the object's identity cannot be the key); `"raw"` scores `lookup` itself. Without a variant, `"pass"` and `"raw"` return None.

- [ ] **Step 4: Run the tests**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_tok_ladder.py tests/ignite/mast/test_tok_command.py -q -W error`
Expected: 14 passed.

- [ ] **Step 5: Trial: one ladder on 2 shots and the input ladder on 4 (CPU, about 5 minutes)**

```bash
export IGNITE_MAST_ROOT=/models/intermediate/mast/trials/stage1b-task9
CACHE=/models/intermediate/mast/cache/v1
CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh scripts/mast/run_tokenizer.py fit --cache-dir $CACHE --n-shots 8 --workers 8
CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh scripts/mast/run_tokenizer.py ladder --task task_4-4 --n-shots 2 --num-workers 4 --cache-dir $CACHE
CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh scripts/mast/run_tokenizer.py ladder --inputs --n-shots 4 --workers 4 --cache-dir $CACHE
```

Expected: `$IGNITE_MAST_ROOT/stage1/ladders/task_4-4.csv` has 8 rows for `summary-ip` with `k` 1, 2, 3, 4, 6, 8, 12, 16, a `floor` that falls from row to row (about 0.08 at K = 1 in the rough look), `mean_baseline` far above every floor, and empty `raw` and `passed`. `inputs.csv` has rows for 11 signals; a signal none of the 4 shots has shows "no data", not 0. Report both tables' first lines and the durations.

- [ ] **Step 6: Commit**

```bash
git add src/tokamak_foundation_model/ignite/mast/tokenizers/ladder.py scripts/mast/run_tokenizer.py tests/ignite/mast/test_tok_ladder.py
git commit -m "ignite.mast: ladders over the number of kept values, and the assignment by the rules of the plan"
```

### Task 10: the gate

**Files:**
- Create: `src/tokamak_foundation_model/ignite/mast/tokenizers/gate.py`
- Modify: `scripts/mast/run_tokenizer.py` (sub-command `gate`)
- Test: `tests/ignite/mast/test_tok_gate.py`

**Interfaces:**
- Consumes: `GridSet.index(values) -> np.ndarray` (nearest centre, -1 for NaN); `Codec.encode`, `roundtrip`, `grids`; `transform.to_physical`; Task 9's `ladder.values_per_frame(spec, setting) -> int`, `ladder.band_contrast(codec, z_model, first_bin=32) -> tuple[float, float] | None`; Task 7's `assign.reference_assignment(signals)`; Task 8's `kept_score` and `provenance_of` (in the script); `tasks.best_published(task: str) -> float`; `baselines.read_kept_score`.
- Produces in `gate.py`:
  - `bin_usage(index: np.ndarray, n_bins: int) -> np.ndarray`: `index (N, n_tok, n_val)` with -1 for missing, to the fraction of the `n_bins` bins that each position `(n_tok, n_val)` uses; NaN for a position without an example.
  - `channel_errors(codec: Codec, spec: SignalSpec, z_model: np.ndarray, mode: str = "unquantized") -> tuple[np.ndarray, np.ndarray]`: for one shot's cached values in model units: per channel (the flattened `values_shape`) the sum of squared round-trip errors in TokaMark units and the number of valid samples; the same samples as `ladder.shot_error`.
  - `cache_side(codec: Codec, spec: SignalSpec, shots_values: Iterable[np.ndarray | None]) -> dict`: keys `n_shots`, `n_skipped` (shots where the signal is None), `bins_used_min` (the smallest `bin_usage` over the positions, from `codec.grids.index` of the encoded values of all shots; None without grids), `worst_channel` (flat index) and `worst_channel_error` (that channel's RMSE pooled over the shots), `contrast` (the weighted mean of `band_contrast` over the shots for a codec whose `domain` is `"band"`, else None). With no shot: `n_shots` 0 and None elsewhere.
  - `verdict(*, values: int, reference_values: int, floor: float | None, ranged: float | None, own_mean: float | None, target: float | None) -> str`, in this order: `not scored` if `floor` or `ranged` is None, or if `values > reference_values` and `own_mean` is None; `blocks: does not beat its own mean (<floor> against <own_mean>)` if `values > reference_values` and not `floor < own_mean`; `pass (no fidelity target)` if `target` is None; `pass` if `ranged <= target`; else `pass, misses the fidelity target (<ranged> against <target>)`. Numbers with four decimals.
  - `gate_rows(signals_by_task: Mapping[str, Sequence[SignalSpec]], assignment: Mapping[str, Setting], floors: Mapping[tuple[str, str], Mapping[str, float] | None], best_published: Mapping[str, float], cache_side: Mapping[str, dict]) -> pd.DataFrame`: one row per (task, output signal). `floors[(task, read)]` holds the per-signal NRMSE for `read` in `("unquantized", "ranged", "nearest", "ref")`, or None if that score is not stored or is stale. Columns `task, signal, codec, k, domain, values_per_frame, target, unquantized, ranged, nearest, own_mean, bins_used_min, worst_channel, worst_channel_error, contrast, verdict`; `target` is `0.2 * best_published[task]`, None for the spectrogram signals; `own_mean` is the `ref` floor; a missing number is NaN.
  - `to_markdown(table: pd.DataFrame) -> str`: numbers with four decimals; a missing number is printed as `not scored` (the score columns) or `no data` (the cache-side columns), never as 0.
- Produces in the script: `gate --assignment FILE [--label val_report] [--n-shots 200] [--workers 64] [--cache-dir DIR] [--stage0-dir DIR]`: for the 14 tasks, reads `signals.csv` of `tok.<name>.<mode>` in the three read-outs and of `tok.ref.unquantized` under `<label>`, each only if `kept_score` keeps it with the current provenance (a stale score is `not scored` and is listed under the table with its reason); runs `cache_side` for every output signal over `cache.pick_evenly(tasks.split_shots("val_report"), n)`; prints the table, and under it one line per task: the task's floor in the three read-outs, a fifth of its best published score, and stage 0's `mean` and TokaMind `floor` scores on `val_report` where `baselines.read_kept_score` keeps them under `<stage0>/scores/<task>/<predictor>/val_report/`. Writes `<stage1>/gate/<name>.csv` and `.md`. The last line is `<n> rows, <m> not scored, <b> block`; exit 1 if `b > 0`.

- [ ] **Step 1: Write the failing tests**

```python
"""The stage-1 gate: bin usage, the worst channel, and the verdict of each (task, signal) pair."""

import numpy as np
import pytest

from tokamak_foundation_model.ignite.mast import tasks, transform
from tokamak_foundation_model.ignite.mast.tokenizers import assign, codecs, gate, grid

SPEC = tasks.SignalSpec("s", 0.00025, (2,), 0.0, 2.0)
ST = transform.ChannelStats(compress="none", scale=np.ones(2), mean=np.zeros(2), std=np.ones(2))


def shot():
    z = np.zeros((2, 40), np.float32)                                         # 2 channels, 2 frames
    z[1, :20:2] = 1.0                                                         # channel 1 alternates in its first frame
    return z


def test_bin_usage_counts_the_bins_each_value_uses():
    idx = np.array([[[0, 5]], [[1, 5]], [[2, 5]], [[-1, 5]]])                 # 4 examples, 1 token, 2 values
    assert gate.bin_usage(idx, 8).tolist() == [[0.375, 0.125]]
    assert np.isnan(gate.bin_usage(np.full((3, 1, 1), -1), 8)).all()          # no example is no data, not 0


def test_channel_errors_are_per_channel_in_tokamark_units():
    c = codecs.FrameDCT("s", 0.00025, (2,), ST, k=1)
    sq, n = gate.channel_errors(c, SPEC, shot())
    assert n.tolist() == [40, 40] and sq[0] == pytest.approx(0.0, abs=1e-12)
    assert sq[1] == pytest.approx(20 * 0.25 / 4.0)                            # 20 samples 0.5 from the frame mean, std 2


def test_the_cache_side_pass_counts_shots_and_names_the_worst_channel():
    g = grid.GridSet(lo=np.full((2, 1), -1.0), hi=np.full((2, 1), 1.0), width=np.ones((2, 1)), n_bins=8)
    c = codecs.FrameDCT("s", 0.00025, (2,), ST, k=1, grids=g)
    out = gate.cache_side(c, SPEC, [shot(), None, shot()])
    assert (out["n_shots"], out["n_skipped"], out["worst_channel"]) == (2, 1, 1)
    assert out["worst_channel_error"] == pytest.approx(np.sqrt(1.25 / 40)) and out["contrast"] is None
    assert out["bins_used_min"] == pytest.approx(1 / 8)                       # channel 0 is always 0: one bin of 8
    none = gate.cache_side(c, SPEC, [None])
    assert (none["n_shots"], none["n_skipped"], none["worst_channel_error"]) == (0, 1, None)


def test_the_verdicts():
    v = gate.verdict
    kw = dict(values=4, reference_values=1)
    assert v(floor=0.010, ranged=0.011, own_mean=0.076, target=0.0176, **kw) == "pass"
    assert v(floor=0.060, ranged=0.061, own_mean=0.098, target=0.0424, **kw) == (
        "pass, misses the fidelity target (0.0610 against 0.0424)")
    assert v(floor=0.100, ranged=0.100, own_mean=0.098, target=0.0424, **kw) == (
        "blocks: does not beat its own mean (0.1000 against 0.0980)")
    assert v(floor=0.098, ranged=0.098, own_mean=0.098, target=None, **kw).startswith("blocks")   # equal is not better
    assert v(values=1, reference_values=1, floor=0.0, ranged=0.0, own_mean=0.0, target=0.0176) == "pass"
    assert v(values=216, reference_values=24, floor=0.676, ranged=0.680, own_mean=0.900, target=None) == (
        "pass (no fidelity target)")
    assert v(floor=None, ranged=None, own_mean=None, target=0.0176, **kw) == "not scored"
    assert v(floor=0.010, ranged=None, own_mean=0.050, target=0.0176, **kw) == "not scored"
    assert v(floor=0.010, ranged=0.011, own_mean=None, target=0.0176, **kw) == "not scored"


def test_gate_rows_judge_each_task_signal_pair_and_keep_missing_scores_missing(toy_task):
    a = {"summary-ip": assign.Setting("frame_dct", (4,))}
    floors = {("task_toy", "unquantized"): {"summary-ip": 0.010}, ("task_toy", "ranged"): {"summary-ip": 0.011},
              ("task_toy", "nearest"): {"summary-ip": 0.020}, ("task_toy", "ref"): {"summary-ip": 0.076},
              ("task_other", "unquantized"): None}
    side = {"summary-ip": {"n_shots": 200, "n_skipped": 0, "bins_used_min": 0.5, "worst_channel": 0,
                           "worst_channel_error": 0.012, "contrast": None}}
    t = gate.gate_rows({"task_toy": toy_task.outputs, "task_other": toy_task.outputs}, a, floors,
                       {"task_toy": 0.0880, "task_other": 0.3116}, side).set_index("task")
    row = t.loc["task_toy"]
    assert row["target"] == pytest.approx(0.0176) and row["verdict"] == "pass" and row["values_per_frame"] == 4
    assert row["own_mean"] == 0.076 and row["nearest"] == 0.020
    assert t.loc["task_other", "verdict"] == "not scored" and np.isnan(t.loc["task_other", "unquantized"])
    text = gate.to_markdown(t.reset_index())
    assert "not scored" in text and "0.0110" in text and "| 0.0000 |" not in text
```

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_tok_gate.py -q -W error`
Expected: an import error for `gate`.

- [ ] **Step 3: Implement `gate.py` and the sub-command**

The local module is `tokenizers/gate.py`; `ignite/gate.py` is the baseline's and is only imported (through `ladder.band_contrast`). `gate_rows` stores a missing number as NaN in the table and gives it to `verdict` as None (`None if np.isnan(v) else float(v)`), so that a missing score is never compared as a number. The cache-side pass reads the shots with `transform.map_shots(fn, shots, args, workers)` as `fit.fit_all` does.

- [ ] **Step 4: Run the tests**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_tok_gate.py tests/ignite/mast/test_tok_command.py -q -W error`
Expected: 10 passed.

- [ ] **Step 5: Trial on the Task 8 trial root (CPU, about 2 minutes)**

```bash
export IGNITE_MAST_ROOT=/models/intermediate/mast/trials/stage1b-task8      # holds the fit, top.json and one score
CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh scripts/mast/run_tokenizer.py gate --assignment $IGNITE_MAST_ROOT/top.json \
    --label val_select-n2 --n-shots 4 --workers 4 --cache-dir /models/intermediate/mast/cache/v1; echo "exit $?"
```

Expected: a table of 57 rows (15 EFIT scalars in task 1-1, 18 signals in 2-1, 5 in 3-3, 3 in 3-2, 2 each in 1-2, 2-2, 3-1, 4-1, 4-2, 4-5, 1 each in 1-3, 2-3, 4-3, 4-4). The row of task 4-4 has a number under `unquantized` and `not scored` under `ranged`; every verdict is `not scored`; the cache-side columns hold numbers for the signals the 4 shots have. Last lines: `57 rows, 57 not scored, 0 block` and `exit 0`. If the trial root is gone, run Task 8's trial again first.

- [ ] **Step 6: Commit**

```bash
git add src/tokamak_foundation_model/ignite/mast/tokenizers/gate.py scripts/mast/run_tokenizer.py tests/ignite/mast/test_tok_gate.py
git commit -m "ignite.mast: the stage-1 gate judges every (task, signal) pair from stored floors"
```

### Task 11: the results page

**Files:**
- Create: `docs/models/ignite-mast-stage1.md`
- Modify: `mkdocs.yml` (one nav line after `- models/ignite-mast-stage0.md`)

**Interfaces:**
- Consumes: the commands of Tasks 8 to 10 and this plan's sections "Decisions of the plan's author", "Which number decides what" and "Measured facts".
- Produces: the page the controller fills after the long jobs. Later plans (1c) add sections to it.

- [ ] **Step 1: Write the page** with these sections, in this order:
  1. "What this is": one paragraph (tokenizers without learned weights for the 39 signals; status line `Status: code complete, not measured yet (<date>)`).
  2. "Units and read-outs": physical, model units, linear units, TokaMark units; `unquantized`, `ranged`, `nearest`; what a floor is and that it is scored by the stage-0 harness on TokaMark's windows.
  3. "The tokenizers": the three codecs, one paragraph each, with the shape of the token values.
  4. "Rules": the seven rules of "Which number decides what", copied, each marked "fixed before the measurement, 2026-11-22".
  5. "Where this departs from the design": the bullets of "Decisions of the plan's author" that name a design sentence, each with its measured reason and the date of the measurement.
  6. "How to run": the commands of "Controller jobs" below.
  7. "Results": four tables with their column heads and no rows: the assignment (signal, codec, K, domain, values per frame, note), the gate (the columns of `gate_rows`), the per-task floors (task, three read-outs, a fifth of the best published score, mean baseline, TokaMind's floor), the pass-through and input round-trip costs. Under each: `Not measured yet.`
  8. "Limits": the fit is on 2,000 training shots; floors are fitted quantities and are not measured on the test split; the rough looks of 2026-11-22 are not results; task 4-5's frame statistics (open decision 1).

  Every number already on the page carries its date, its split, its number of shots and "measured" or "rough look".

- [ ] **Step 2: Add the nav line** in `mkdocs.yml`: `      - models/ignite-mast-stage1.md` directly after the line `      - models/ignite-mast-stage0.md` (same indentation).

- [ ] **Step 3: Check the docs build**

Run: `mkdir -p /models/intermediate/mast/trials/stage1b-docs && uvx --with-requirements requirements-docs.txt --from 'mkdocs<2' mkdocs build -d /models/intermediate/mast/trials/stage1b-docs/site 2>&1 | grep -i "ignite-mast"`
Expected: no line that names `ignite-mast-stage1.md` as a warning or an error (no output at all is the usual case). If `uvx` cannot fetch its packages, check by hand that every relative link of the new page names an existing file and say so in the report.

- [ ] **Step 4: Commit**

```bash
git add docs/models/ignite-mast-stage1.md mkdocs.yml
git commit -m "ignite.mast: docs page for the stage-1 tokenizers, with the rules fixed before the measurement"
```

## Controller jobs

The controller runs these after the tasks they depend on, from `/home/nvidia/FusionAIHub`, with `IGNITE_MAST_ROOT` unset. `RT="scripts/mast/py.sh scripts/mast/run_tokenizer.py"` and `S1=/models/intermediate/mast/stage1`. Logs go to `$S1/logs/`. Durations are the author's estimates from stage-0 timings (floor scoring of task 4-1 over 1,110 shots: 4,131 s; cache build over 11,188 shots: 225 s); none was measured for this code.

| Job | After | Command | GPU | Duration | What is read from the output |
|---|---|---|---|---|---|
| K1 fit | Task 8 | `CUDA_VISIBLE_DEVICES="" $RT fit --workers 64` | none | 15 to 40 min | 39 lines. No signal says "no data". `n_skipped` of `magnetics-b_field_tor_probe_cc_field` is about a third of 2,000 and of Thomson about a fifth; the soft X-ray cameras skip a few shots (sampling-rate mismatch). If a signal has under 500 shots, stop and report. |
| K2 ladders | K1, Task 9 | `$RT ladder --task T [--variant V --gpu N --share-gpu]` for the six ladder tasks, and `$RT ladder --inputs`, all at once | one GPU for a few minutes per task with a variant; may be shared | 15 to 60 min each, about 1 h together | `$S1/ladders/*.csv`. Per signal: the floor falls with K; the first rung equals the "own mean" reference. For task 4-5: the frame oracle (the row `k0 = 32`), the mean baseline on the same shots, the contrast per rung. These three numbers and TokaMind's floor (0.3605 on `val_report`) go to the owner the same day (open decision 1). |
| K3 assign | K2 | `$RT assign --name v1` and `$RT assign --name ref --reference` | none | seconds | The printed table: which signals miss their target, and the total of values per frame (stage 2's sequence length follows from it). |
| K4 input round-trip | K3 | for tasks 1-1, 1-2, 2-1, 4-1: `$RT score --task T --split val_select --n-shots 100 --instrument raw --variant V --gpu N --share-gpu`, and the same with `--instrument inputs --assignment $S1/assignments/v1.json --mode unquantized` | one each, may be shared | 8 passes of 3 to 10 min | Per task the added error of rule 4 over the task's best published score, and the ratio `inputs / raw - 1`. Above 0.2 on any task: `$RT assign --name v1r --from v1 --raise-inputs`, repeat the four `inputs` passes with `v1r` once, and use `v1r` from here on. Both costs are recorded. |
| K5 confirmation | K4 | for the 14 tasks: `CUDA_VISIBLE_DEVICES="" $RT score --task T --split val_report --assignment A --mode M --num-workers 16` for the three read-outs, and with `--assignment $S1/assignments/ref.json --mode unquantized` | none | 56 passes of 10 to 40 min; about 2 to 3 h in waves of 14 | The last line of each pass. A pass that ends with a non-finite score or an error stops the wave. |
| K6 costs on `val_report` | K5 | `--instrument raw` and `--instrument passthrough --mode ranged` for every task with a signal that misses its target (expected: 3-2, 4-1, 4-2) and for 1-2 as a control; `--instrument inputs --mode ranged` and `raw` for 1-1, 1-2, 2-1, 4-1 | one each, may be shared | about 14 passes of 15 to 45 min; about 2 h on 4 GPUs | Per task and signal the pass-through cost; per task the input round-trip cost. A pass-through cost above 1% on `val_report` for a signal chosen by rule 3 is reported, and K is not changed again. |
| K7 gate | K5, Task 10 | `CUDA_VISIBLE_DEVICES="" $RT gate --assignment $S1/assignments/<name>.json` | none | 5 to 15 min | The table and the last line. `exit 1` means a row blocks: a transform with more values than its reference can only lose to it through the read-out or a bug, so the controller stops and reports instead of lowering K. |

The variant `V` of a task is the one whose score the headroom table shows for it (`scripts/mast/py.sh scripts/mast/run_headroom.py --table`, and `--table-all` once plan 1a has run); a task whose direct row is marked stale, unfinished or not competitive has no usable model for the pass-through of rule 3, and its ladder runs without `--variant`. For the input round-trip of rule 4 a model that is behind the published scores is still used (rule 4 says how its number is read); a stale or unfinished one is not. No variant is tuned: stage 0's base configuration was fixed before any score existed, and its variants are single measurements (`lr1e4` on 1-2, `h050b256` on 4-4, `s90k` on 1-1). Tasks 1-3 and 3-2 get their direct model from plan 1a's job J4; if it is not there when K2 runs, their ladders run without pass-through and are run again when it is (only a signal that misses its target needs it).

After K7 the controller fills the four tables of `docs/models/ignite-mast-stage1.md` from `$S1/assignments/<name>.json`, `$S1/gate/<name>.md` and the stored scores, sets the status line, and commits (`ignite.mast: stage-1 fixed tokenizers measured`), with the numbers, their date, split and shot counts in the body.

## What this plan leaves to later plans

- The learned tokenizers and the comparison of shared against specialized: plan 1c.
- The fluctuation band of the 50 kHz channels (soft X-rays, D-alpha, saddle coils). No benchmark task scores it; plan 1c holds it as its last item.
- Time detail inside a frame for the Mirnov spectrograms (open decision 1).
- A token cache for stage 2 (the encoded values of every shot). Stage 2's plan decides its layout once the assignment is frozen.
- The number of bins per value (64 here) and the smoothing of the bin targets: stage 2 owns the loss.
- A joint (R, Z) basis for the boundary.
- A range that proves too narrow on `val_report` (rule 6): reported here, fixed with the next fit.
- Frames across signals: a spectrogram column is assigned to the frame that holds its centre, counted from the window's first column, while series frames are counted from the window's first sample. How TokaMark places the first column against the cut time was not checked; stage 2 needs it before it mixes these tokens in one frame.
