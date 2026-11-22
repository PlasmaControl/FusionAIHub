# IGNITE on MAST, stage 1c: learned tokenizers and the shared-versus-specialized comparisons. Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> Approved by the owner on 2026-11-22 (16:17 UTC, "yes i approve the plan. please carry it out"). The open decisions for the owner are taken as the author recommends until the owner says otherwise. The stage-0 numbers quoted in this text are those of the day it was written; `docs/models/ignite-mast-stage0.md` holds the current ones. Written 2026-11-22 by the plan's author. Nothing here has been run except the read-only looks under "Measured facts". It is the third of three stage-1 plans and builds on 1b (`2026-11-22-ignite-mast-stage1b-fixed-tokenizers.md`): every interface of 1b named here is quoted from that plan, not from code, because 1b is not implemented yet. If 1b's review changes one of them, the matching line here changes with it.

**Goal:** For each signal family there is a tokenizer with fitted or learned weights that is shared by every channel of the family, measured in TokaMark's units against the fixed transform or linear baseline of 1b at the same number of values, and against controls built the old way (all channels of one diagnostic concatenated), so that the owner's idea of one shared tokenizer per family is accepted or rejected per signal on numbers, and stage 2 gets a final assignment `v2`.

**Architecture:** A sub-package `ignite/mast/tokenizers/learned/`. Every tokenizer of this plan is a *recoder* placed behind the top rung of a fixed codec of 1b: the frame's first 16 cosine coefficients of a series channel, the 32 coefficients of a spectrogram band, or the first 96 principal components of a profile or map. A recoder turns those numbers into an ordered list of values and back. It is either linear (a principal-component basis, fitted on the CPU in minutes) or a small network (encoder, bounded values, a quantizer that is skipped for half of the examples, decoder), and it takes either one channel at a time (the shared tokenizer, told which channel it has) or all channels of a signal at once (the control). A `RecodedCodec` wraps front end and recoder as a `Codec` of 1b, so 1b's floor, ladder, gate and instruments score it unchanged. A comparison command pairs two assignments over the same shots and applies rules fixed in this plan.

**Tech Stack:** Python 3, numpy, pandas, torch (the networks; CPU in tests and trials), pytest; 1b's `tokenizers` package; the stage-0 harness through 1b's command.

**Spec:** `docs/models/ignite-mast-design.md`: "Approach B" (one tokenizer per family, shared by every channel), "Tokenizers" (common features, the learned slow tokenizer, the two high-rate paths, the gate, what blocks, "Controls"), the risk "Cross-channel physics moves into the dynamics model", and row "1. Tokenizers" of "Order of work".

## Open decisions for the owner

These change the goal, the bar or what is published. The plan does not decide them; each has the author's recommendation.

1. **What a win of the concatenated control means.** The design says: "If a specialized one clearly beats the shared one, that signal uses the specialized tokenizer and the result is recorded." For the magnetic probe arrays the author expects the control to win on reconstruction at equal values per frame, because the probes of one array are strongly correlated and a per-channel tokenizer cannot use that (this is the cost the design itself names). Adopting the control for a signal gives up, for that signal, channel identity that comes from a table (goal G6), tokens that survive a missing channel, and the per-channel sequence the dynamics model was designed around. No stage-1 number says what the dynamics model gains or loses: every magnetics signal is an input only, so the benchmark cannot score its reconstruction. Recommendation: stage 1 records the result with its numbers, `v2` keeps the shared tokenizer, the control goes into a second assignment `v2c`, and stage 2 runs its smallest pretraining on both and decides on the scores of tasks 1-1 and 1-2 (one extra small run). The owner may instead apply the design's sentence to the reconstruction numbers alone.
2. **Token count as a criterion.** The comparison also reports how many values per frame each arm needs to reach the input target of 1b (0.0176). If the control needs a fraction of the shared tokenizer's values, the frame could be much shorter than the design's estimate of about 127 slow tokens. Whether sequence length may be traded against G6 is not the builder's call. Recommendation: report the count; do not adopt on it in stage 1.
3. **Channel identity without positions.** Measured 2026-11-22: the registry of cache v1 has R and Z only for the flux-map points and the Thomson channels. The magnetic probes, flux loops, saddle coils, soft X-ray chords and D-alpha channels have a name (`AMB_CCBV01` ...), a unit and an order in their array, and no position. The design builds identity "from the registry row" plus a learned vector per channel; for these channels the table part can say only family, sampling rate, unit, compression and order, and the rest sits in the learned vector, which does not transfer to another machine. Recommendation: accept this for stage 1 and state it; add positions from the FAIR-MAST geometry in the next cache version and retrain the shared tokenizers then (they train in minutes).
4. **Fluctuation tokens for the 50 kHz channels are not delivered in stage 1** (a departure from the stage-1 row "the fluctuation path", see "Decisions"). The saddle coils, soft X-ray cameras and D-alpha get the trend path only; the fluctuation path is built for the Mirnov signals. Task 7 measures what the missing band holds. Recommendation: accept; the stage-2 plan adds the band as input tokens and judges it by an ablation on tasks 3-2, 4-1 and 4-2, the only way a benchmark number can see it.
5. **Cut order if stage 1 is late.** The trainings are short, but the plan has eight tasks after 1b's eleven and the date is 2026-11-27. Recommendation, first cut first: the learned map tokenizer (train it only if 1b's linear baseline misses its target); the extra seeds; the learned profile tokenizer; the Mirnov networks (the linear arms stay); the series networks (the linear arms stay). The linear arms (Tasks 1 to 4, CPU only) answer the shared-versus-specialized question in linear form and are not cut.

## Global Constraints

- Machine: node13, aarch64, 4 GPUs of 286 GiB, 352 cores, 1.4 TB RAM. Data and runs go under `/models/intermediate/mast/`, never into the repository. Nothing is ever written under `/models/tokamind/`. Nothing is uploaded anywhere. Temporary files do not go to `/tmp`; use a directory under `/models/intermediate/mast/trials/`.
- Work in the main checkout `/home/nvidia/FusionAIHub` on branch `nvidia` (a git worktree run on this machine imports the main checkout's code). Commit locally. Never `git push`, `reset --hard`, `clean`, `stash`, `checkout -- <file>` or `commit --amend`. `git add` only this task's files, by path; never `-A` or `.`. `.gitignore` excludes `*.h5`, `*.pt`, `*.npy`, `*.npz`: tests build their fixtures in memory or in `tmp_path`.
- Commit subjects start with `ignite.mast: `. The body says what changed and the measured facts behind it. The message ends with `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`.
- Two environments. `scripts/mast/py.sh` is `/models/envs/ignite-mast/bin/python -I` and has `tokamark` and torch 2.15. `pixi run --frozen -e nvidia ...` (always `--frozen`) has no `tokamark`, so tests that need it skip there. `pyproject.toml` and `pixi.lock` do not change. No new dependency. No scikit-learn, no `vector_quantize_pytorch` in new code.
- No module under `src/tokamak_foundation_model/ignite/mast/` imports `tokamark`, `MAST_tools`, `mmt` or `zarr` at import time. No module of `tokenizers/learned/` except `quantizer.py`, `nets.py` and `train.py` imports torch at import time (the linear arms and the comparison run without it).
- Hashed files do not change in this plan: `ignite/mast/harness.py`, `tasks.py`, `baselines.py`, `headroom/floor.py`, `headroom/cheap.py`, `headroom/direct.py`, and every `*.py` file directly in `ignite/mast/tokenizers/` (1b's `tokenizers.code_sha256()` covers exactly those). One changed byte there marks every stored score of 1b stale. All new library code of this plan goes into `ignite/mast/tokenizers/learned/`; the scripts under `scripts/mast/` may change.
- Files under `src/tokamak_foundation_model/ignite/` outside `ignite/mast/` do not change; they may be imported.
- Tests run with `-W error` on the CPU: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/<file> -q -W error`, and the whole directory with `CUDA_VISIBLE_DEVICES="" pixi run --frozen -e nvidia python -m pytest tests/ignite/mast -q -W error`. An implementer never runs the whole `tests/ignite` suite (some of its tests use a GPU).
- An implementer builds no store over the whole cache, runs no scoring pass over a split and no training beyond the unit tests' own and a trial of at most 200 steps of one arm on the CPU, and uses no GPU. A trial writes under `/models/intermediate/mast/trials/<task name>/` (set `IGNITE_MAST_ROOT` to it). Long jobs are the controller's (section "Controller jobs").
- NRMSE always means TokaMark's evaluator through `harness.score`. Everything fitted or trained uses training shots, is chosen on `val_select` and reported on `val_report`. Nothing in this plan reads the test split.
- A subset of shots is always spread evenly over the sorted shot list (`cache.pick_evenly`): the first 200 shots by number are one old campaign and do not track the rest.
- Units. "Physical", "model units" (the cache's), "linear units" and "TokaMark units" are 1b's. "Front values" are what a front-end codec's `encode_domain` returns. "Example units" are front values after this plan's per-stream standardization (`store.Norm`). A function's docstring names the unit of every array it takes and returns.
- Missing data is NaN and stays NaN; a count of zero examples is reported as "no data", never as an error of 0.
- Read files in parts (at most 150 lines per read; grep first). Never print arrays or long logs.
- The tests of this plan were written on 2026-11-22 without being run (1b's code did not exist). If a test fails against an implementation that follows the task's Interfaces block, the implementer reports the test and the measured value to the controller and changes neither a threshold nor an assertion.
- Docs are part of a change. A number is written with its date, its split, the number of shots, its limits and whether it is measured or inferred.

## Review Focus

- A channel that is missing in a frame. A per-channel tokenizer gives that channel no token and leaves every other channel's token exactly as it was; the concatenated control still gives its tokens, from the channel's mean. [Task 2, `test_a_missing_channel_has_no_token_and_changes_no_other_token`, `test_the_concatenated_codec_fills_a_missing_channel`]
- The values are ordered. A prefix of the values, the rest set to zero, decodes to a coarser version of the same example, for the linear recoder and for the network. [Task 2, `test_a_prefix_of_the_values_decodes_with_an_error_that_falls_with_its_length`; Task 5, `test_dropped_values_are_zero_and_the_rest_keep_their_precision`; Task 6, `test_a_short_training_learns_a_low_rank_signal_and_keeps_its_best_weights`]
- Equal size in a comparison. The concatenated arm gets the same number of values per frame as the per-channel arm it is compared with, and the comparison is paired over the same shots; a shot one side lacks is left out of both. [Task 3, `test_recoding_an_assignment_keeps_the_values_per_frame`; Task 4, `test_the_gap_is_paired_over_the_shots_both_sides_have`]
- No score of 1b goes stale. An assignment without a recoder has exactly 1b's provenance keys; one with a recoder carries the recoder's bytes. [Task 3, `test_provenance_of_a_plain_assignment_is_unchanged`, `test_provenance_names_the_recoders_bytes`]
- A learned tokenizer whose first value does not vary is called collapsed and blocks; unused later values do not. [Task 4, `test_a_constant_first_value_is_a_collapse`]

## Measured facts this plan relies on (2026-11-22)

- Read from the registry of cache v1 (`/models/intermediate/mast/cache/v1/registry.csv`, 6,521 rows, columns `device, signal, channel, channel_name, family, dt, units, imas_path, r, z, tm_mean, tm_std, roles, sat_abs, valid_fraction, dead_fraction, compress, scale, mean, std`). Series signals with more than one sample per 5 ms frame: 17 signals, 163 channels. Slow family, 20 samples per frame (0.25 ms): `gas_injection-total_injected`, `interferometer-n_e_line` (asinh), `pf_active-coil_current` (10 channels), `pf_active-coil_voltage` (4, asinh), `pf_active-solenoid_current`, `pulse_schedule-i_plasma`, `pulse_schedule-n_e_line`, `summary-ip`, `summary-power_nbi`. Slow family, 25 samples per frame (0.2 ms), the four magnetics signals: `magnetics-b_field_pol_probe_ccbv_field` (40 channels), `..._obr_field` (18), `..._obv_field` (18), `magnetics-flux_loop_flux` (15); 91 channels together. High-rate family, 250 samples per frame (0.02 ms), all asinh: `magnetics-b_field_tor_probe_saddle_voltage` (12), `soft_x_rays-horizontal_cam_lower` (18), `soft_x_rays-horizontal_cam_upper` (18), `spectrometer_visible-filter_spectrometer_dalpha_voltage` (3).
- The two Mirnov signals (`magnetics-b_field_tor_probe_cc_field`, `magnetics-b_field_pol_probe_omv_voltage`) are in the cache as TokaMark's spectrograms, 3 probes by 256 bins at 1.024 ms: power only. The phase between probes, from which a mode number is read, is not in the data any tokenizer of this plan sees.
- Positions: `r` and `z` are filled for `equilibrium-psi` (4,225 rows) and the two Thomson signals (120 rows each), and for no other signal. Channel names exist for probes and chords (`AMB_CCBV01` to `AMB_CCBV40`, `XSX_HCAML#1` ...). Units in the registry: T, V, A, Wb, W, eV, m, `1 / m ** 2`, `Wb / rad`, `count`, `u * a` and empty.
- Rough looks made for 1b (24 training shots, pooled error in TokaMark units, not harness scores): cosine truncation per frame with 1, 2, 4, 8 values: the 40 `ccbv` probes 0.071, 0.032, 0.014, 0.008; flux loops 0.030, 0.008, 0.003, 0.002; soft X-ray lower camera 0.098, 0.081, 0.069, 0.065; D-alpha 0.221, 0.155, 0.122, 0.091. Principal components of whole profiles with 16, 32, 48, 64 values: `lcfs_r` 0.0093, 0.0072, 0.0058, 0.0046; flux map 0.0180, 0.0064, 0.0042, 0.0033.
- The stage environment has torch 2.15.0 and `vector_quantize_pytorch`; baseline IGNITE's quantizer (`ignite/quantizer.py`) wraps `vector_quantize_pytorch.FSQ` and hides the bounded value before rounding.
- `ignite.gate.utilization(codes, cfg=None, *, codebook_size=None, min_utilization=None, min_code_entropy=None) -> dict` takes integer level indices `(B, n_tok, fsq_dim)` and returns, among others, `dim_entropy_nats` (one marginal entropy per dimension) and `min_dim_entropy`; with `cfg.fsq_levels` it normalises by `log(levels)`.
- `scripts/mast/run_headroom.py` has `class Refused(Exception)`, `LOCK = "train.lock"`, `_take_lock(run_dir: Path) -> Path` (raises `Refused` if another live process holds the run) and `_release_lock(lock: Path) -> None`; 1b's command loads that script by path for its GPU checks.
- TokaMark's evaluator writes, under a score directory, `<task>/shots_metrics.csv` with one row per (shot, signal): columns `shot_id, feature_name, n_windows, NRMSE_mean, NMAE_mean, RMSE_mean, MAE_mean, nan_fraction_mean`.

## Decisions of the plan's author, and where they leave the design

- **A learned tokenizer is a recoder behind a fixed transform** (the design says "encoder + finite scalar quantizer + decoder" on one channel, profile or map in one frame). The first layer of the encoder and the last layer of the decoder are 1b's fixed transform at its top rung (16 cosine coefficients, 32 per band, 96 principal components) and are not learned. Reasons: any sampling rate, missing samples and the units are handled once, by code that 1b has already tested and scored; the error of the top rung is known from 1b's ladders, so what the learned part can add at most is known before it is trained; and a linear recoder is then exactly the best linear tokenizer of that input, computable on the CPU. The cost: the learned tokenizer cannot use what the top rung has dropped (for the soft X-ray cameras and D-alpha, the part above 1.6 kHz).
- **The values of a token are ordered in two ways** (the design: "For a learned tokenizer the order is precision: a second quantizer level encodes what the first missed ... the later parts are dropped at random during training, so a prefix is always a valid, coarser token"). Precision: each value is bounded to (-1, 1) and is read at 8 levels, at 64 levels (which contain the 8, and are the 64 bins of 1b's grid), or unrounded; the 64-level reading is the design's second level. Each training example draws one of the three with probabilities 1/4, 1/4, 1/2, which keeps the design's "quantizer skipped for a random half". Length, which the design names only for fixed transforms: the list of values is also cut at a random length during training, so one trained network gives every K up to its size and can be compared with the fixed transform at each K. That second ordering is the author's addition.
- **The quantizer is this plan's own few lines, not baseline IGNITE's `SpectroQuantizer`.** The token value that stage 2 predicts is the bounded value itself, which the library hides. Baseline files are not touched.
- **"Built the old way" means: one tokenizer per cached signal, all its channels in one input, no channel identity.** The control uses the same network body and the same training recipe as the shared tokenizer and differs only in what an example is. Baseline IGNITE's own `SlowTSCodec` (another architecture, another time base, other losses) is not used: a difference would then mix the architecture with the question. One control per probe array (four for the magnetics), not one for all 91 magnetics channels, because a codec of 1b serves one signal; the linear comparison also reports the joint basis of all 91 channels at the same number of values, as a number only.
- **Three arms, not two.** Shared against concatenated changes two things at once: how much data the tokenizer sees, and whether it sees the channels together. So each comparison has a third arm: the per-channel tokenizer fitted on the signal's own diagnostic only. Shared against own measures the owner's point (more data for one encoder). Own against concatenated measures what the design calls the cost (structure across channels).
- **The loss counts every channel equally** (squared error in example units, each channel scaled to unit variance), the same in every arm. TokaMark's score weights channels by their physical variance. The comparison is read in TokaMark units regardless; the page says so where the two disagree.
- **"Statistics first" for the fluctuation path** is met by the input, not by an adversarial decoder: a band example is the frame-mean power spectrum of the band, which is already a statistic.
- **Fluctuation tokens for the 50 kHz channels are left out** (departs from the stage-1 row). Their targets in the benchmark are waveforms, and a token that holds band power cannot lower a waveform error; a codec that does not decode to a waveform does not fit 1b's instruments; so no stage-1 number can judge them. Task 7 measures the size of the band instead. Open decision 4.
- **The Mirnov control cannot measure what the design's risk row hopes** ("The Mirnov-only control measures what that costs", about mode numbers read from the phase between coils). The cached spectrograms hold power only. The control measures whether the three probes' power spectra are better coded together. The phase question needs raw Mirnov waveforms and is left to a later plan.
- **Results tree.** Under `/models/intermediate/mast/stage1/`: `store/v1/{train,val}/`, `recoders/<name>/`, `compare/`; assignments and scores go where 1b puts them.

## Which number decides what

Stated before any number exists. All errors are in TokaMark units. For a signal that a task predicts, the error of a tokenizer is its floor through `harness.score` (1b's instrument F), per shot from the evaluator's `shots_metrics.csv`. For a signal no task predicts, it is 1b's cache-side error per shot (`ladder.shot_error`). Choices are made on the 100 `val_select` shots of 1b's ladders and confirmed on all `val_report` shots.

1. **The gap.** For two tokenizers A and B of one signal: over the shots both have, `gap = (mean error of B - mean error of A) / mean error of B`, with a 95% interval from 2,000 paired resamples of the shots (seed 0).
2. **"A beats B":** the interval lies above 0; and, if A or B is a network, the gap is larger than the spread of the gap over three seeds of that network (largest minus smallest). **"A clearly beats B":** A beats B and the gap is at least 0.10. The threshold 0.10 is the author's reading of the design's word "clearly".
3. **Same size.** Two per-channel tokenizers are compared at the same number of values per token, the K of assignment `v1`. A concatenated tokenizer is compared at the same number of values per frame for the signal (channels times K). Each comparison is also reported at half and at a quarter of that size for the concatenated arm, and with the number of values each arm needs to reach 1b's target for the signal.
4. **Learned shared tokenizer against the fixed transform** (series channels; the design: "It replaces the fixed transform for any signal where it has the lower error"). A learned shared tokenizer (linear or network, the better on `val_select`) replaces the cosine transform for a signal if it beats it on `val_select` and again on `val_report`. Otherwise the cosine transform stays: it has no weights, cannot collapse and works at any frame duration.
5. **Learned tokenizer against its linear baseline** (profiles, map, Mirnov bands; the design: "A learned tokenizer that does not beat its linear baseline is replaced by the baseline"). The baseline is 1b's codec for the signal (principal components; the band transform). The learned one is used only if it beats the baseline on `val_select` and again on `val_report`.
6. **Shared against specialized** (the four magnetics signals; the two Mirnov signals). Reported per signal: shared against own, own against concatenated, shared against concatenated, in linear form and as networks. "The specialized one clearly beats the shared one" is judged between the best shared arm and the best concatenated arm (each chosen on `val_select`), by rule 2 on `val_select` and again on `val_report`. For the magnetics, which no task predicts, two more numbers are reported and decide nothing alone: the input round-trip of the direct models of tasks 1-1 and 1-2 (1b's instrument I) with each arm's tokens, and the damage of one missing channel to the other channels. What follows from a clear win is open decision 1; this plan writes `v2` (shared) and `v2c` (the clear winners swapped in) and decides nothing further.
7. **Collapse.** A network is collapsed if, for any token, the first value's use of the 8 levels has a normalised entropy under 0.3 on the `val_select` examples (`ignite.gate.utilization`; 0.3 is baseline IGNITE's `gate_min_code_entropy`). A collapsed tokenizer takes part in no comparison and is reported. The count of values whose entropy is at least 0.3 is reported as "active values" and decides nothing.
8. **A network that loses at seed 0** is trained once more with learning rate 1e-4 before the loss stands (stage 0, direct model: a rate of 3e-4 was unstable on task 1-2 and 1e-4 was not, while on tasks 1-1 and 2-1 the lower rate was worse by 0.0056 each; so the lower rate is a second try, not the default). **A network that wins at seed 0** gets seeds 1 and 2 before the win stands. A run whose validation error still falls by more than 1% in its last quarter is extended once to twice its steps.
9. **Stage 2 does not wait.** Stage 2 starts on 1b's `v1`. `v2` replaces it only where rules 4 and 5 say so. If nothing learned beats anything, `v2` equals `v1`, and that is a result.

## File structure

```
src/tokamak_foundation_model/ignite/mast/tokenizers/learned/__init__.py    (create: docstring, code_sha256)
src/tokamak_foundation_model/ignite/mast/tokenizers/learned/identity.py    (create: stream features from the registry)
src/tokamak_foundation_model/ignite/mast/tokenizers/learned/store.py       (create: front ends, examples from the cache, Norm)
src/tokamak_foundation_model/ignite/mast/tokenizers/learned/recode.py      (create: Recoder, LinearRecoder, RecodedCodec, save/load)
src/tokamak_foundation_model/ignite/mast/tokenizers/learned/build.py       (create: codecs of an assignment that names recoders)
src/tokamak_foundation_model/ignite/mast/tokenizers/learned/compare.py     (create: paired gaps, verdicts, collapse, missing channel)
src/tokamak_foundation_model/ignite/mast/tokenizers/learned/quantizer.py   (create: levels, random precision and length)
src/tokamak_foundation_model/ignite/mast/tokenizers/learned/nets.py        (create: TokenNet, NetRecoder)
src/tokamak_foundation_model/ignite/mast/tokenizers/learned/train.py       (create: examples, the training loop)
src/tokamak_foundation_model/ignite/mast/tokenizers/learned/fluct.py       (create: the fluctuation band of a 50 kHz channel, as a statistic)
scripts/mast/train_tokenizer.py                                            (create: store, linear, arms, joint, train, fluct)
scripts/mast/run_tokenizer.py                                              (modify: recoded settings, recode, compare, seeds)
tests/ignite/mast/test_lrn_{store,recode,command,compare,nets,train,fluct}.py   (create)
docs/models/ignite-mast-stage1.md                                          (modify: the sections of this plan)
```

Words used below. A **stream** is what one example of a per-channel tokenizer covers: one channel of a series signal, one (probe, band) pair of a spectrogram signal (stream index `probe * 8 + band`, 1b's token index), or a whole profile or map (one stream). **S** is the number of streams of a signal and **D** the number of front values per stream (16, 32 or 96). An **arm** is one recoder with a name, stored in `<stage1>/recoders/<name>/`.

---

### Task 1: streams, their identity and the example store

**Files:**
- Create: `src/tokamak_foundation_model/ignite/mast/tokenizers/learned/__init__.py`, `learned/identity.py`, `learned/store.py`
- Test: `tests/ignite/mast/test_lrn_store.py`

**Interfaces:**
- Consumes (1b, as that plan defines them): `tasks.SignalSpec(name, dt, values_shape, mean, std)`; `transform.ChannelStats(compress, scale, mean, std)`; `codecs.Codec` with attributes `name, dt, values_shape, stats, domain, signal_mean, signal_std, n_tok, n_val, grids` and methods `domain_from_model(z)` (model units to the codec's domain), `encode_domain(u) -> (N, F, n_tok, n_val)`, `decode_domain(values, n)`; `codecs.FrameDCT(name, dt, values_shape, stats, k, domain="linear", grids=None, signal_mean=0.0, signal_std=1.0)`; `codecs.BandDCT`; `codecs.LinearCodec`; `assign.Setting(codec, k, domain="linear", n_bins=64, source="")`; `fit.SignalFit`; `fit.build_codec(spec, stats, fit, setting) -> Codec`; `frames.frame_len(dt) -> int`. Existing code: `cache.read_shot(path: Path, names: Sequence[str] | None = None, *, stats_sha256: str | None = None) -> dict[str, CachedSignal | None]` (`CachedSignal.values` is float32 `(*values_shape, T)` in model units, NaN for missing); `transform.map_shots(fn, shots: Sequence[int], args: tuple, workers: int) -> Iterator` (`fn((shot, *args))` per shot, in order); `registry.load_registry(path: Path) -> pd.DataFrame`.
- Produces:
  - `learned.code_sha256() -> str`: sha256 over the bytes of every `*.py` file directly in `tokenizers/learned/`, in sorted name order.
  - `identity.N_FEAT = 18`, `identity.UNITS = ("T", "V", "A", "Wb", "W")`, `identity.N_BANDS = 8`.
  - `identity.stream_features(rows: pd.DataFrame, kind: str, n_streams: int) -> np.ndarray`: the registry rows of one signal (sorted by `channel` inside) to `(n_streams, N_FEAT)` float32. `kind` is 1b's codec kind of the signal. Columns: 0 to 3, the family as one-hot in the order slow, high_rate, profile, map; 4, `log10(dt) + 3`; 5, 1 if `compress == "asinh"`; 6 to 10, the unit as one-hot over `UNITS`; 11, 1 if the unit is none of them; 12, the stream's place in its array, `i / (n - 1)` over the channels (the probes for `band_dct`), 0 for a single one; 13, 1 if the channel's `r` and `z` are finite; 14 and 15, `r` and `z` (0 if not); 16, 1 for `band_dct`; 17, the band index divided by 8. For `frame_dct` a stream is a channel (`n_streams` must equal the number of rows). For `band_dct` the rows are probes times bins and a stream is (probe, band): `n_streams = P * 8`, stream `p * 8 + b`. For `linear` there is one stream; columns 12 to 15 are 0. Another `n_streams` raises `ValueError`.
  - `@dataclass(frozen=True) class identity.StreamTable` with `signals: dict[str, tuple[int, int]]` (signal to its first row and its number of streams) and `features: np.ndarray` `(n, N_FEAT)`; `len(table)` is `n`; `index_of(self, name: str, n_streams: int) -> np.ndarray` gives the `(n_streams,)` int64 rows of the signal, all -1 for a signal the table does not hold, `ValueError` if it holds the signal with another number of streams.
  - `identity.build_table(registry: pd.DataFrame, signals: Sequence[tuple[str, str, int]]) -> StreamTable`: `(name, kind, n_streams)` in order; a signal without registry rows raises `ValueError` naming it.
  - `store.D_FRONT = {"frame_dct": 16, "band_dct": 32, "linear": 96}`.
  - `store.front_setting(spec: SignalSpec, kind: str, domain: str, values_per_token: int = 1) -> Setting`: the top rung: `Setting("frame_dct", (min(16, frame_len(spec.dt)),), domain)`; `Setting("band_dct", (32,) * 8, "band")`; `Setting("linear", (n, values_per_token), domain)` with `n` the largest multiple of `values_per_token` that is at most `min(96, prod(values_shape))`.
  - `store.front_codec(spec, stats, fit, kind, domain, values_per_token=1) -> Codec`: `fit.build_codec(spec, stats, fit, front_setting(...))`.
  - `store.to_streams(front: Codec, values: np.ndarray) -> np.ndarray` and `store.from_streams(front, streams)`: front values `(..., n_tok, n_val)` to `(..., S, D)` and back. The identity for `FrameDCT` and `BandDCT` (S = `n_tok`, D = `n_val`); for a `LinearCodec` all tokens become one stream (S = 1, D = `n_tok * n_val`).
  - `store.stream_values(front: Codec, z_model: np.ndarray) -> np.ndarray`: one shot's cached values `(*values_shape, T)` in model units to `(F, S, D)` float32 front values, NaN for a stream without a token in that frame. For a series (`FrameDCT`) only whole frames are kept (`F = T // frame_len(dt)`); a spectrogram keeps every frame.
  - `@dataclass class store.Store` with `name: str`, `values: np.ndarray` `(M, S, D)` float32, `shots: np.ndarray` `(M,)` int64 (the shot of each frame), `n_skipped: int = 0`.
  - `store.collect(name: str, front: Codec, per_shot: Iterable[tuple[int, np.ndarray | None]]) -> Store`: None is a shot without the signal: skipped and counted.
  - `store.build_store(cache_root: Path, shots: Sequence[int], fronts: Mapping[str, Codec], *, workers: int) -> dict[str, Store]`: reads `shots/<shot>.h5` with `cache.read_shot(..., stats_sha256=<the manifest's>)` through `transform.map_shots`.
  - `store.save_store(stores: Mapping[str, Store], out_dir: Path, meta: dict) -> None` (one `<signal>.npz` each and `store.json`; refuses an existing `store.json` with `FileExistsError`) and `store.load_store(out_dir: Path, names: Sequence[str] | None = None) -> tuple[dict[str, Store], dict]`.
  - `@dataclass(frozen=True) class store.Norm` with `center: np.ndarray` `(S, D)` and `scale: np.ndarray` `(S,)`; `apply(self, values)` gives example units `(values - center) / scale[:, None]`, `invert(self, e)` the reverse; both take `(..., S, D)` and keep NaN.
  - `store.fit_norm(values: np.ndarray) -> Norm`: `center` is the mean over the finite examples of each `(stream, value)`; `scale` is the root of the mean of `(values - center) ** 2` over the examples and the D values of the stream. A stream without a finite example gets `center` 0 and `scale` 1; a stream whose scale is 0 gets `scale` 1.

- [ ] **Step 1: Write the failing tests**

```python
"""Streams, their identity features and the store of examples (plan 1c)."""

import numpy as np
import pandas as pd
import pytest

from tokamak_foundation_model.ignite.mast import tasks, tokenizers, transform
from tokamak_foundation_model.ignite.mast.tokenizers import assign, codecs, learned
from tokamak_foundation_model.ignite.mast.tokenizers.learned import identity, store

RNG = np.random.default_rng(0)


def rows(signal, n, family="slow", dt=0.0002, units="T", compress="none", r=np.nan, z=np.nan):
    return pd.DataFrame({"signal": signal, "channel": np.arange(n), "channel_name": [f"c{i}" for i in range(n)],
                         "family": family, "dt": dt, "units": units, "compress": compress, "r": r, "z": z})


def series_front(n_ch=2, k=16):
    shape = (n_ch,)
    st = transform.ChannelStats(compress="none", scale=np.ones(shape), mean=np.zeros(shape), std=np.ones(shape))
    return codecs.FrameDCT("m", 0.00025, shape, st, k)


def test_the_code_hash_covers_this_sub_package_only():
    assert len(learned.code_sha256()) == 64 and learned.code_sha256() == learned.code_sha256()
    assert learned.code_sha256() != tokenizers.code_sha256()


def test_features_of_a_probe_array():
    f = identity.stream_features(rows("m", 5), "frame_dct", 5)
    assert f.shape == (5, identity.N_FEAT) and f.dtype == np.float32
    assert np.array_equal(f[:, :4], np.tile([1, 0, 0, 0], (5, 1))) and np.allclose(f[:, 4], np.log10(0.0002) + 3)
    assert np.array_equal(f[:, 5:12], np.tile([0, 1, 0, 0, 0, 0, 0], (5, 1)))        # not asinh; unit T
    assert np.allclose(f[:, 12], [0, 0.25, 0.5, 0.75, 1]) and not f[:, 13:].any()
    g = identity.stream_features(rows("m", 2, units="eV", r=[0.2, 0.3], z=[-1.0, 1.0]), "frame_dct", 2)
    assert np.array_equal(g[:, 11], [1, 1]) and np.array_equal(g[:, 13], [1, 1])
    assert np.allclose(g[:, 14], [0.2, 0.3]) and np.allclose(g[:, 15], [-1, 1])
    with pytest.raises(ValueError):
        identity.stream_features(rows("m", 5), "frame_dct", 4)


def test_features_of_a_spectrogram_signal_are_per_probe_and_band():
    f = identity.stream_features(rows("s", 768, family="high_rate", dt=0.001024, units="", compress="asinh"), "band_dct", 24)
    assert f.shape == (24, identity.N_FEAT) and f[:, 1].all() and f[:, 5].all() and f[:, 11].all() and f[:, 16].all()
    assert np.allclose(f[:, 17], np.tile(np.arange(8) / 8, 3)) and np.allclose(f[:, 12], np.repeat([0, 0.5, 1], 8))


def test_a_profile_is_one_stream_without_a_position():
    f = identity.stream_features(rows("p", 120, family="profile", dt=0.005, units="eV", r=1.0, z=0.0), "linear", 1)
    assert f.shape == (1, identity.N_FEAT) and f[0, 2] == 1 and not f[0, 12:].any()


def test_the_table_gives_each_signal_its_rows():
    reg = pd.concat([rows("a", 3), rows("b", 2)])
    t = identity.build_table(reg, [("a", "frame_dct", 3), ("b", "frame_dct", 2)])
    assert len(t) == 5 and t.features.shape == (5, identity.N_FEAT) and t.signals == {"a": (0, 3), "b": (3, 2)}
    assert t.index_of("b", 2).tolist() == [3, 4] and t.index_of("c", 4).tolist() == [-1, -1, -1, -1]
    with pytest.raises(ValueError):
        t.index_of("a", 2)
    with pytest.raises(ValueError, match="zz"):
        identity.build_table(reg, [("zz", "frame_dct", 1)])


def test_the_front_end_is_the_top_rung_of_the_fixed_codec():
    ip = tasks.SignalSpec("summary-ip", 0.00025, (1,), 0.0, 1.0)
    assert store.front_setting(ip, "frame_dct", "linear") == assign.Setting("frame_dct", (16,), "linear")
    psi = tasks.SignalSpec("equilibrium-psi", 0.005, (65, 65), 0.0, 1.0)
    assert store.front_setting(psi, "linear", "signal", 2) == assign.Setting("linear", (96, 2), "signal")
    small = tasks.SignalSpec("p", 0.005, (30,), 0.0, 1.0)
    assert store.front_setting(small, "linear", "signal", 4).k == (28, 4)
    mirnov = tasks.SignalSpec("s", 0.001024, (3, 256), 0.0, 1.0)
    assert store.front_setting(mirnov, "band_dct", "band") == assign.Setting("band_dct", (32,) * 8, "band")


def test_stream_values_of_a_series_keep_whole_frames_only():
    front = series_front()
    z = RNG.standard_normal((2, 50)).astype(np.float32)             # two whole frames of 20 samples, and 10 more
    v = store.stream_values(front, z)
    assert v.shape == (2, 2, 16) and v.dtype == np.float32
    assert np.allclose(v[1, 0, 0], z[0, 20:40].mean(), atol=1e-6)   # value 0 is the frame mean (1b)
    z[1, 20:40] = np.nan
    holed = store.stream_values(front, z)
    assert np.isnan(holed[1, 1]).all() and np.isfinite(holed[1, 0]).all() and np.isfinite(holed[0]).all()


def test_collect_skips_and_counts_a_shot_without_the_signal():
    front = series_front()
    shots = [(30000, RNG.standard_normal((2, 40)).astype(np.float32)), (30001, None),
             (30002, RNG.standard_normal((2, 60)).astype(np.float32))]
    s = store.collect("m", front, shots)
    assert s.values.shape == (5, 2, 16) and s.n_skipped == 1 and s.shots.tolist() == [30000, 30000, 30002, 30002, 30002]
    empty = store.collect("m", front, [(30001, None)])
    assert empty.values.shape == (0, 2, 16) and empty.n_skipped == 1


def test_a_store_is_saved_once_and_read_back(tmp_path):
    s = store.collect("m", series_front(), [(30000, RNG.standard_normal((2, 40)).astype(np.float32))])
    store.save_store({"m": s}, tmp_path / "train", {"shots": [30000]})
    back, meta = store.load_store(tmp_path / "train")
    assert np.array_equal(back["m"].values, s.values) and back["m"].shots.tolist() == [30000, 30000]
    assert meta["shots"] == [30000] and store.load_store(tmp_path / "train", names=[])[0] == {}
    with pytest.raises(FileExistsError):
        store.save_store({"m": s}, tmp_path / "train", {})


def test_the_norm_gives_each_stream_unit_scale_and_survives_missing_and_constant_streams():
    v = RNG.standard_normal((500, 3, 4)).astype(np.float32) * 7.0 + 2.0
    v[:, 1] = np.nan
    v[:, 2] = 5.0
    v[0, 0] = np.nan
    n = store.fit_norm(v)
    e = n.apply(v)
    assert n.center.shape == (3, 4) and n.scale.shape == (3,)
    assert np.sqrt(np.nanmean(e[:, 0] ** 2)) == pytest.approx(1.0, abs=1e-4) and n.scale[0] == pytest.approx(7.0, rel=0.1)
    assert n.scale[1] == 1.0 and not n.center[1].any() and n.scale[2] == 1.0 and not e[:, 2].any()
    assert np.isnan(e[0, 0]).all() and np.isnan(e[:, 1]).all()
    assert np.allclose(n.invert(e)[1:, [0, 2]], v[1:, [0, 2]], atol=1e-4)
```

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_lrn_store.py -q -W error`
Expected: an import error for `tokenizers.learned`.

- [ ] **Step 3: Implement the signatures of "Produces"**

`learned/__init__.py` holds the docstring and `code_sha256` only (written like `tokenizers.code_sha256` of 1b, over its own directory). The functions passed to `transform.map_shots` are module-level (they are pickled). Nothing here imports torch.

- [ ] **Step 4: Run the tests**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_lrn_store.py -q -W error`
Expected: 10 passed.

- [ ] **Step 5: Trial on four cached shots (CPU, about a minute)**

```bash
CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh - <<'EOF'
from tokamak_foundation_model.ignite.mast import cache, paths, registry, tasks, transform
from tokamak_foundation_model.ignite.mast.tokenizers import fit
from tokamak_foundation_model.ignite.mast.tokenizers.learned import identity, store
root = paths.cache_dir()
kinds = {"magnetics-b_field_pol_probe_ccbv_field": ("frame_dct", "linear", 1),
         "magnetics-b_field_pol_probe_omv_voltage": ("band_dct", "band", 1),
         "equilibrium-lcfs_r": ("linear", "signal", 4)}
shots = cache.pick_evenly(tasks.split_shots("train"), 4)
specs = {s.name: s for s in registry.all_signals()}
stats = transform.load_stats(root / "stats.npz")
fits = fit.fit_all(root, shots, workers=4, names=list(kinds))
fronts = {n: store.front_codec(specs[n], stats[n], fits[n], *kinds[n]) for n in kinds}
for n, s in store.build_store(root, shots, fronts, workers=4).items():
    e = store.fit_norm(s.values).apply(s.values)
    print(n, s.values.shape, s.n_skipped, bool((abs(e[e == e]) < 1e4).all()))
t = identity.build_table(registry.load_registry(root / "registry.csv"),
                         [(n, k[0], fronts[n].n_tok if k[0] != "linear" else 1) for n, k in kinds.items()])
print(len(t), t.index_of("magnetics-b_field_pol_probe_omv_voltage", 24)[:3].tolist())
EOF
```

Expected: three lines with the shapes `(M, 40, 16)`, `(M, 24, 32)` and `(M, 1, 96)` (M a few hundred frames; the Mirnov signal may be skipped in some of the four shots), each ending in `True`; then `65 [40, 41, 42]`. Nothing is written.

- [ ] **Step 6: Commit**

```bash
git add src/tokamak_foundation_model/ignite/mast/tokenizers/learned/__init__.py \
        src/tokamak_foundation_model/ignite/mast/tokenizers/learned/identity.py \
        src/tokamak_foundation_model/ignite/mast/tokenizers/learned/store.py tests/ignite/mast/test_lrn_store.py
git commit -m "ignite.mast: streams of a signal, their identity features and the store of examples"
```

---

### Task 2: the linear recoder and the recoded codec

**Files:**
- Create: `src/tokamak_foundation_model/ignite/mast/tokenizers/learned/recode.py`, `learned/build.py`
- Test: `tests/ignite/mast/test_lrn_recode.py`

**Interfaces:**
- Consumes: Task 1's `store.Norm`, `fit_norm`, `to_streams`, `from_streams`, `front_codec`, `identity.StreamTable`, `identity.N_BANDS`; 1b's `codecs.Codec` (its `roundtrip(x, mode="unquantized")` is `decode(read(encode(x)), T)`; `encode(x)` is `encode_domain(to_domain(x))`), `codecs.fit_pca(x: np.ndarray, n_components: int) -> tuple[mean, basis, eigenvalues, total_variance]` (`x (N, D)` with NaN for missing, `basis (D, n_components)` with orthonormal columns in order of falling variance), `grid.GridSet(lo, hi, width, n_bins=64)` (fields `(n_tok, n_val)`; centres `width * sinh(linspace(asinh(lo / width), asinh(hi / width), n_bins))`), `grid.fit_ranges(values (N, n_tok, n_val), margin=0.05) -> (lo, hi, width)`, `assign.Setting`, `fit.build_codec`, `fit.SignalFit`.
- Produces in `recode.py`:
  - `class Recoder`: the base. Attributes `kind: str` (`"linear"` or `"net"`), `concat: bool`, `d_in: int` (D for a per-stream recoder, S times D for a concatenated one), `n_tok: int`, `n_val: int`, and the property `n_values = n_tok * n_val`. Methods `encode(self, e: np.ndarray, ident: np.ndarray) -> np.ndarray` (`e (M, d_in)` finite, in example units, `ident (M,)` int64 rows of the recoder's table; returns `(M, n_values)` float32) and `decode(self, values: np.ndarray, ident: np.ndarray) -> np.ndarray` (`(M, n_values)` to `(M, d_in)`). The values are ordered: a prefix with the rest set to 0 decodes to a coarser example.
  - `class LinearRecoder(Recoder)`: `LinearRecoder(mean: np.ndarray, basis: np.ndarray, *, concat: bool = False, n_val: int | None = None)` with `mean (d_in,)`, `basis (d_in, n_values)`. `n_val=None` means one token of `n_values` values; otherwise `n_values` must be a multiple of `n_val` (`ValueError`). `encode` is `(e - mean) @ basis`, `decode` is `values @ basis.T + mean`; `ident` is not used.
  - `fit_linear(e: np.ndarray, n_values: int, *, concat: bool = False, n_val: int | None = None) -> LinearRecoder`: through `codecs.fit_pca` (so examples with NaN entries, which a concatenated example has when a channel is missing, are used pairwise).
  - `class RecodedCodec(Codec)`: `RecodedCodec(front: Codec, norm: Norm, recoder: Recoder, ident: np.ndarray, n_keep: Sequence[int], grids: GridSet | None = None)`. `name, dt, values_shape, stats, domain, signal_mean, signal_std` are the front's, and `to_domain`, `from_domain`, `domain_from_model` call the front's. `ident (S,)` are the streams' rows in the recoder's table. `n_keep` is how many values are kept: one number, or for a band front with a per-stream recoder one per band (stream `s` keeps `n_keep[s % 8]`). Layout: with `recoder.n_tok == 1`, `n_val = max(n_keep)` and the values from a stream's `n_keep` on are 0 (1b's convention for bands); `n_tok` is S per-stream and 1 concatenated. With `recoder.n_tok > 1`, `n_keep` must be a multiple of `recoder.n_val` (`ValueError` naming "multiple"), `n_val = recoder.n_val`, and the tokens per example are `n_keep // n_val` (times S for a per-stream recoder). The property `values_per_frame` is the number of kept values per frame over the whole signal. `ValueError` if `recoder.d_in` does not fit the front or `max(n_keep) > recoder.n_values`.
    - `encode_domain(self, u)`: front values, to streams, to example units, through the recoder. Per-stream: a stream without a token in a frame gets NaN values and no other stream is touched. Concatenated: a missing stream is filled with 0 in example units (its mean) and the frame gets its tokens; only a frame in which every stream is missing gets NaN.
    - `decode_domain(self, values, n)`: kept values padded with zeros to the recoder's size, through the recoder, out of example units, back to front values, then the front's `decode_domain`. NaN values decode to NaN.
  - `@dataclass class Loaded` with `recoder: Recoder`, `norms: dict[str, Norm]`, `table: StreamTable`, `meta: dict`, `ranges: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]`.
  - `save_recoder(out_dir: Path, recoder: Recoder, norms: Mapping[str, Norm], table: StreamTable, meta: dict, ranges: Mapping[str, tuple] | None = None) -> None`: writes `recoder.json` (`kind`, `concat`, `d_in`, `n_tok`, `n_val`, the table's `signals`, and `meta`, which holds at least `signals: {name: {"kind", "domain", "n_streams", "values_per_token"}}`), `norm.npz`, `table.npz`, `ranges.npz` if given, and `basis.npz` for a linear recoder (a network's `best.pt` is put beside them by the trainer). `ranges[name]` is `(lo, hi, width)` of the recoder's values for that signal, each `(S, n_values)` per-stream and `(1, n_values)` concatenated. Refuses an existing `recoder.json` (`FileExistsError`).
  - `load_recoder(directory: Path, device: str = "cpu") -> Loaded`: a `"net"` recoder is built by Task 5's `nets.load_net_recoder(directory: Path, table: StreamTable, device: str = "cpu") -> Recoder`, imported only then (until Task 5 exists, a `"net"` directory fails with the import error).
  - `recoder_sha256(directory: Path) -> str`: sha256 over the bytes of those files that exist among `basis.npz`, `best.pt`, `norm.npz`, `ranges.npz`, `recoder.json`, `table.npz`, in that order.
- Produces in `build.py`:
  - `RECODED = "recoded"` (a per-stream recoder) and `RECODED_CONCAT = "recoded_concat"` (a concatenated one): the two codec names of a setting that names a recoder. 1b's `ladder.values_per_frame` knows both, so 1b's gate counts their values without reading the recoder.
  - `build_codec(spec, stats, fit, setting: Setting, recoders_root: Path, loaded: dict[str, Loaded] | None = None, device: str = "cpu") -> Codec`: `fit.build_codec(spec, stats, fit, setting)` unless `setting.codec` is one of the two names. Then (`ValueError` naming "concat" if the name and the recoder's `concat` disagree): the recoder of `recoders_root / setting.source` (kept in `loaded` by source if given); the front from `store.front_codec` with the kind, domain and values per token that the recoder's `meta["signals"][spec.name]` records; `n_keep = setting.k`; the grids from the stored ranges, cut and regrouped to the codec's `(n_tok, n_val)`, with `setting.n_bins`; for a `"net"` recoder instead the even grid `lo = -1`, `hi = 1`, `width = 1000`. `ValueError` naming the signal if the recoder does not list it; `ValueError` naming "domain" if `setting.domain` is not the recorded one.
  - `build_codecs(signals: Sequence[SignalSpec], stats: Mapping[str, ChannelStats], fits: Mapping[str, SignalFit], assignment: Mapping[str, Setting], recoders_root: Path, device: str = "cpu") -> dict[str, Codec]`.
  - `sources(assignment: Mapping[str, Setting]) -> list[str]`: the distinct non-empty `source` names, sorted.
  - `sources_sha256(assignment, recoders_root: Path) -> str`: sha256 over each source name and its `recoder_sha256`, in that order.

- [ ] **Step 1: Write the failing tests**

```python
"""The linear recoder, the recoded codec, and the codecs of an assignment that names a recoder (plan 1c)."""

import numpy as np
import pytest

from tokamak_foundation_model.ignite.mast import tasks, transform
from tokamak_foundation_model.ignite.mast.tokenizers import assign, codecs, fit, grid
from tokamak_foundation_model.ignite.mast.tokenizers.learned import build, identity, recode, store

RNG = np.random.default_rng(0)
SPEC = tasks.SignalSpec("m", 0.00025, (3,), 0.0, 1.0)
META = {"signals": {"m": {"kind": "frame_dct", "domain": "linear", "n_streams": 3, "values_per_token": 1}}}
TABLE = identity.StreamTable({"m": (0, 3)}, np.zeros((3, identity.N_FEAT), dtype=np.float32))


def stats(shape=(3,)):
    return transform.ChannelStats(compress="none", scale=np.ones(shape), mean=np.zeros(shape), std=np.ones(shape))


def low_rank(m=4000, d=16, scales=(3.0, 2.0, 1.0), noise=0.01):
    q, _ = np.linalg.qr(RNG.standard_normal((d, d)))
    return (RNG.standard_normal((m, len(scales))) * np.array(scales)) @ q[:, :len(scales)].T \
        + noise * RNG.standard_normal((m, d))


def window(n=6):
    """`n` windows of 3 channels and 100 samples (5 frames); neighbouring channels are correlated."""
    w = RNG.standard_normal((n, 3, 100)).cumsum(axis=-1) * 0.1
    return w + 0.5 * np.roll(w, 1, axis=1)


def pieces(concat=False, n_values=16, n_val=None):
    front = codecs.FrameDCT("m", 0.00025, (3,), stats(), 16)
    fv = front.encode(window(40)).reshape(-1, 3, 16)                 # 200 frames
    norm = store.fit_norm(fv)
    e = norm.apply(fv)
    r = recode.fit_linear(e.reshape(len(e), -1) if concat else e.reshape(-1, 16), n_values, concat=concat, n_val=n_val)
    return front, norm, r, np.arange(3)


def test_a_linear_recoder_finds_the_directions_of_the_data():
    e = low_rank()
    r = recode.fit_linear(e, 4)
    assert (r.kind, r.concat, r.d_in, r.n_tok, r.n_val, r.n_values) == ("linear", False, 16, 1, 4, 4)
    ident = np.zeros(len(e), dtype=np.int64)
    v = r.encode(e, ident)
    assert v.shape == (4000, 4) and v.dtype == np.float32
    assert np.sqrt(np.mean((r.decode(v, ident) - e) ** 2)) < 0.02          # three directions hold all but the noise
    assert np.allclose(v.var(axis=0)[:3], [9.0, 4.0, 1.0], rtol=0.15)       # ordered by variance
    with pytest.raises(ValueError):
        recode.fit_linear(e, 6, n_val=4)


def test_a_prefix_of_the_values_decodes_with_an_error_that_falls_with_its_length():
    e = low_rank()
    r = recode.fit_linear(e, 16)
    ident = np.zeros(len(e), dtype=np.int64)
    v = r.encode(e, ident)
    errors = []
    for keep in (1, 2, 3, 16):
        cut = v.copy()
        cut[:, keep:] = 0.0
        errors.append(float(np.sqrt(np.mean((r.decode(cut, ident) - e) ** 2))))
    assert errors[0] > errors[1] > errors[2] > errors[3] and errors[3] < 1e-5
    assert errors[0] == pytest.approx(np.sqrt(5.0 / 16), rel=0.1)           # directions 2 and 3 are left: 4 + 1 over 16


def test_the_fitted_basis_is_never_worse_than_keeping_the_first_coordinates():
    e = low_rank(scales=(3.0, 2.0, 1.0, 0.5), noise=0.05)
    r = recode.fit_linear(e, 16)
    ident = np.zeros(len(e), dtype=np.int64)
    v = r.encode(e, ident)
    for keep in (1, 2, 4, 8):
        cut = v.copy()
        cut[:, keep:] = 0.0
        fitted = np.mean((r.decode(cut, ident) - e) ** 2)
        first = e[:, keep:].var(axis=0).sum() / 16                          # coordinates from `keep` on replaced by their mean
        assert fitted <= first + 1e-6


def test_a_full_recoder_changes_nothing_and_fewer_values_give_a_coarser_window():
    front, norm, r, ident = pieces()
    x = window()
    full = recode.RecodedCodec(front, norm, r, ident, (16,))
    assert (full.n_tok, full.n_val, full.domain, full.name, full.values_per_frame) == (3, 16, "linear", "m", 48)
    assert full.encode(x).shape == (6, 5, 3, 16) and np.allclose(full.roundtrip(x), front.roundtrip(x), atol=1e-4)
    four, one = (recode.RecodedCodec(front, norm, r, ident, (k,)) for k in (4, 1))
    assert four.encode(x).shape == (6, 5, 3, 4) and four.values_per_frame == 12
    err = lambda c: float(np.sqrt(np.mean((c.roundtrip(x) - x) ** 2)))
    assert err(full) < err(four) < err(one)
    with pytest.raises(ValueError):
        recode.RecodedCodec(front, norm, r, ident, (17,))


def test_a_missing_channel_has_no_token_and_changes_no_other_token():
    front, norm, r, ident = pieces()
    c = recode.RecodedCodec(front, norm, r, ident, (4,))
    x = window()
    clean = c.encode(x)
    x[2, 1, 20:40] = np.nan                                                  # window 2, channel 1, frame 1
    holed = c.encode(x)
    assert np.isnan(holed[2, 1, 1]).all()
    others = np.ones(clean.shape, dtype=bool)
    others[2, 1, 1] = False
    assert np.allclose(holed[others], clean[others], rtol=0, atol=1e-6)
    back = c.roundtrip(x)
    assert np.isnan(back[2, 1, 20:40]).all() and np.isfinite(back[2, 0]).all() and np.isfinite(back[2, 1, :20]).all()


def test_the_concatenated_codec_fills_a_missing_channel():
    front, norm, r, ident = pieces(concat=True, n_values=24, n_val=4)
    assert (r.concat, r.d_in, r.n_tok, r.n_val) == (True, 48, 6, 4)
    c = recode.RecodedCodec(front, norm, r, ident, (12,))
    assert (c.n_tok, c.n_val, c.values_per_frame) == (3, 4, 12)
    x = window()
    clean = c.roundtrip(x)
    x[2, 1, 20:40] = np.nan
    values = c.encode(x)
    assert values.shape == (6, 5, 3, 4) and np.isfinite(values).all()
    back = c.roundtrip(x)
    assert np.isfinite(back).all()                                           # the missing channel is decoded from the others
    assert not np.allclose(back[2, 0, 20:40], clean[2, 0, 20:40], rtol=0, atol=1e-6)   # and the others pay for it
    x[2, :, 20:40] = np.nan
    assert np.isnan(c.encode(x)[2, 1]).all() and np.isnan(c.roundtrip(x)[2, :, 20:40]).all()
    with pytest.raises(ValueError, match="multiple"):
        recode.RecodedCodec(front, norm, r, ident, (6,))


def test_a_band_front_keeps_one_number_of_values_per_band():
    shape = (2, 256)
    st = transform.ChannelStats(compress="asinh", scale=np.full(shape, 2.0), mean=np.zeros(shape), std=np.ones(shape))
    front = codecs.BandDCT("s", 0.001024, shape, st, (32,) * 8)
    x = RNG.gamma(2.0, 1.0, (4, 2, 256, 98))
    fv = front.encode(x).reshape(-1, 16, 32)
    norm = store.fit_norm(fv)
    r = recode.fit_linear(norm.apply(fv).reshape(-1, 32), 32)
    c = recode.RecodedCodec(front, norm, r, np.arange(16), (8, 2, 2, 2, 2, 2, 2, 2))
    values = c.encode(x)
    assert (c.n_tok, c.n_val, c.values_per_frame) == (16, 8, 2 * (8 + 7 * 2)) and values.shape == (4, 20, 16, 8)
    assert values[:, :, 0, 2:].any() and not values[:, :, 1, 2:].any() and not values[:, :, 9, 2:].any()
    assert c.roundtrip(x).shape == x.shape
    with pytest.raises(ValueError):
        recode.RecodedCodec(front, norm, r, np.arange(16), (8, 2, 2))


def test_a_recoder_is_saved_once_with_what_it_was_fitted_on(tmp_path):
    front, norm, r, ident = pieces()
    recode.save_recoder(tmp_path / "klt.x", r, {"m": norm}, TABLE, {**META, "date": "2026-11-22"})
    got = recode.load_recoder(tmp_path / "klt.x")
    e = RNG.standard_normal((5, 16)).astype(np.float32)
    zero = np.zeros(5, dtype=np.int64)
    assert np.allclose(got.recoder.encode(e, zero), r.encode(e, zero), atol=1e-6)
    assert np.array_equal(got.norms["m"].scale, norm.scale) and got.meta["signals"]["m"]["domain"] == "linear"
    assert got.table.index_of("m", 3).tolist() == [0, 1, 2] and got.ranges == {}
    recode.save_recoder(tmp_path / "klt.y", recode.LinearRecoder(r.mean + 1.0, r.basis), {"m": norm}, TABLE, META)
    first = recode.recoder_sha256(tmp_path / "klt.x")
    assert len(first) == 64 and first != recode.recoder_sha256(tmp_path / "klt.y")
    with pytest.raises(FileExistsError):
        recode.save_recoder(tmp_path / "klt.x", r, {"m": norm}, TABLE, META)


def test_build_codec_reads_the_recoder_an_assignment_names(tmp_path):
    st = stats()
    sfit = fit.fit_signal(SPEC, st, [RNG.standard_normal((3, 200)).astype(np.float32) for _ in range(4)], kind="frame_dct")
    front, norm, r, ident = pieces()
    e = norm.apply(front.encode(window(40)).reshape(-1, 3, 16))
    values = r.encode(e.reshape(-1, 16), np.zeros(len(e) * 3, dtype=np.int64)).reshape(-1, 3, 16)
    recode.save_recoder(tmp_path / "klt.x", r, {"m": norm}, TABLE, META, {"m": grid.fit_ranges(values)})
    plain = assign.Setting("frame_dct", (4,))
    assert isinstance(build.build_codec(SPEC, st, sfit, plain, tmp_path), codecs.FrameDCT)
    named = assign.Setting("recoded", (4,), "linear", source="klt.x")
    c = build.build_codec(SPEC, st, sfit, named, tmp_path)
    assert isinstance(c, recode.RecodedCodec) and (c.n_tok, c.n_val) == (3, 4)
    assert c.grids.lo.shape == (3, 4) and c.grids.n_bins == 64
    assert np.isfinite(c.roundtrip(window(), "nearest")).all()
    with pytest.raises(ValueError, match="domain"):
        build.build_codec(SPEC, st, sfit, assign.Setting("recoded", (4,), "compressed", source="klt.x"), tmp_path)
    with pytest.raises(ValueError, match="concat"):
        build.build_codec(SPEC, st, sfit, assign.Setting("recoded_concat", (4,), "linear", source="klt.x"), tmp_path)
    other = tasks.SignalSpec("other", 0.00025, (3,), 0.0, 1.0)
    with pytest.raises(ValueError, match="other"):
        build.build_codec(other, st, sfit, named, tmp_path)
    assert build.sources({"m": named, "n": plain}) == ["klt.x"] and build.sources({"n": plain}) == []
    assert build.sources_sha256({"m": named}, tmp_path) != build.sources_sha256({"n": plain}, tmp_path)
```

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_lrn_recode.py -q -W error`
Expected: an import error for `recode`.

- [ ] **Step 3: Implement `recode.py` and `build.py`**

The recoder is called on the finite examples only; rows are put back in place afterwards. `recode.py` and `build.py` do not import torch; `load_recoder` imports `nets` inside the `"net"` branch.

- [ ] **Step 4: Run the tests**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_lrn_recode.py -q -W error`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/tokamak_foundation_model/ignite/mast/tokenizers/learned/recode.py \
        src/tokamak_foundation_model/ignite/mast/tokenizers/learned/build.py tests/ignite/mast/test_lrn_recode.py
git commit -m "ignite.mast: a recoder behind a fixed codec's top rung is a codec; linear recoders are fitted and stored"
```

---

### Task 3: the arms, the store and linear commands, and assignments that name a recoder

**Files:**
- Create: `scripts/mast/train_tokenizer.py` (sub-commands `store`, `linear`, `arms`)
- Modify: `scripts/mast/run_tokenizer.py` (codecs through `learned.build`; provenance; sub-command `recode`), `src/tokamak_foundation_model/ignite/mast/tokenizers/learned/build.py` (`recode_assignment`)
- Test: `tests/ignite/mast/test_lrn_command.py`

**Interfaces:**
- Consumes: Tasks 1 and 2; 1b's command `scripts/mast/run_tokenizer.py` (its `provenance_of(assignment_path: Path | None, fit_dir: Path | None, cache_root: Path, checkpoint: Path | None = None) -> dict[str, str]` returns, for an assignment, the keys `tokenizer_code_sha256, assignment_sha256, fit_sha256, stats_sha256, tokamark_revision, data_root` in that order; its sub-commands `fit`, `score`, `ladder`, `assign`, `gate` build codecs with `fit.build_codecs(signals, stats, fits, assignment)`); 1b's `assign.load_assignment(path)`, `save_assignment(a, path, notes=None)`, `assign.codec_kind(spec)`, `fit.load_fit(out_dir) -> (fits, meta)` (the meta holds `shots`), `paths.stage1_dir()`, `cache.pick_evenly`, `tasks.split_shots`, `tasks.SPECTROGRAM_SIGNALS`.
- Produces in `build.py`:
  - `recode_assignment(base: Mapping[str, Setting], loaded: Loaded, source: str, *, scale: float = 1.0, signals: Sequence[str] | None = None) -> dict[str, Setting]`: `base` with the recoder's signals (or the listed ones among them) replaced by `Setting(RECODED or RECODED_CONCAT, k, base domain, base n_bins, source)`. Per-stream: `k` is the base setting's K (`(K,)` of `frame_dct`, the eight numbers of `band_dct`, `(n_values,)` of `linear`). Concatenated: `k = (m,)` with `m` the base's values per frame for the signal (S times K; P times `sum(k)`) times `scale`, rounded down to a multiple of the recoder's `n_val`, and at least one token. `ValueError` naming: "scale" for a per-stream recoder with `scale != 1`; "domain" if the base setting's domain is not the one the recoder recorded; "recoded" if the base setting already names a recoder; "holds" if `k` asks for more values than the recoder has.
- Produces in `train_tokenizer.py`:
  - `MAGNETICS` (the four magnetics signals in the order ccbv, obr, obv, flux loops), `BANDS = tasks.SPECTROGRAM_SIGNALS`, `SHORT` (`{"ccbv", "obr", "obv", "flux"}` to the four, `"cc"` and `"omv"` to the two Mirnov signals).
  - `@dataclass(frozen=True) class Arm` with `name: str`, `signals: tuple[str, ...]`, `concat: bool`, `net: bool`, `n_tok: int`, `n_val: int`.
  - `arms(series: Sequence[str], profiles: Sequence[str], maps: Sequence[str], streams: Mapping[str, int]) -> dict[str, Arm]`, the 24 arms of this plan. Linear (11): `klt.series` (all series signals, per-stream, 1 x 16), `klt.mag` (`MAGNETICS`, 1 x 16), `pca.ccbv`, `pca.obr`, `pca.obv`, `pca.flux` (one signal, concatenated, all `S * 16` values in tokens of 4), `klt.band` (`BANDS`, 1 x 32), `klt.band.cc`, `klt.band.omv` (one signal, 1 x 32), `pca.band.cc`, `pca.band.omv` (concatenated, all `S * 32` values in tokens of 4). Networks (13): `net.series`, `net.mag` (1 x 16), `net.ccbv`, `net.obr`, `net.obv`, `net.flux` (concatenated, S tokens of 8), `net.band`, `net.band.cc`, `net.band.omv` (1 x 32), `net.bandcat.cc`, `net.bandcat.omv` (concatenated, S tokens of 8), `net.profile` (all profile signals, per-stream, 24 x 4), `net.map` (48 x 2).
  - `net_name(arm: str, seed: int, variant: str | None = None) -> str`: `<arm>.s<seed>`, with `.<variant>` appended if given.
  - `left_over(recoder: Recoder, e: np.ndarray, ident: np.ndarray, sizes: Sequence[int]) -> list[float]`: for each size, the mean square of `e - decode(first size values of encode(e))` over the mean square of `e`.
  - `main(argv: list[str] | None = None) -> int` with, in this task:
    - `store [--assignment FILE] [--n-val 100] [--workers 64] [--cache-dir DIR] [--force]`: the fronts of every signal whose kind is `band_dct` or `linear`, or `frame_dct` with more than one sample per frame (24 signals on cache v1), each in the domain (and, for `linear`, the values per token) of its setting in the assignment (default `<stage1>/assignments/v1.json`). Training frames from the shots of 1b's fit (`<stage1>/fit/v1`), validation frames from `cache.pick_evenly(tasks.split_shots("val_select"), n)`. Writes `<stage1>/store/v1/train/` and `.../val/`; the meta holds `date`, `shots`, `assignment_sha256`, `fit_sha256`, `stats_sha256`, `learned_code_sha256` and per signal `kind`, `domain`, `n_streams`, `values_per_token`. One printed line per signal: frames, S, D, skipped shots; "no data" for a signal without a frame.
    - `linear --arm NAME`: norms from the training store (`fit_norm` per signal); examples pooled over the arm's signals (per-stream: every finite stream; concatenated: one row per frame with NaN for a missing stream); `fit_linear`; the table from the cache's `registry.csv`; ranges with `grid.fit_ranges` over the training values; `save_recoder` into `<stage1>/recoders/<NAME>/` with the store's meta. Prints per signal `left_over` on the validation store at 1, 2, 4, 8, 16 values per stream (concatenated: S times those). Exit 2 for an unknown arm, a network arm, or an existing directory.
    - `arms`: one line per arm (name, signals, streams, concatenated, linear or network, tokens x values) and which recoder directories exist.
- Produces in `run_tokenizer.py`:
  - `recoders_root() -> Path`: `paths.stage1_dir() / "recoders"`. Every sub-command builds its codecs with `learned.build.build_codecs(signals, stats, fits, assignment, recoders_root(), device)`; `score --instrument floor` accepts `--gpu` exactly when the assignment names a network recoder (its codecs then run on that GPU).
  - `provenance_of(..., recoders_root: Path | None = None)`: for an assignment that names no recoder, exactly 1b's keys in 1b's order. Otherwise `learned_code_sha256` and `sources_sha256` follow `stats_sha256`.
  - `recode --name NEW --from OLD --arm RECODER[,RECODER...] [--scale 1.0] [--signals a,b]`: `recode_assignment` on `<stage1>/assignments/OLD.json`, once per recoder directory in the order given (their signals must not overlap after `--signals`), saved as `NEW.json` with one note per changed signal; prints per changed signal the values per frame before and after. An existing `NEW` is refused (exit 2).

- [ ] **Step 1: Write the failing tests**

```python
"""The arms of plan 1c, recoded assignments and their provenance."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from tokamak_foundation_model.ignite.mast import tasks, transform
from tokamak_foundation_model.ignite.mast.tokenizers import assign, fit
from tokamak_foundation_model.ignite.mast.tokenizers.learned import build, identity, recode, store

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "mast"
IP = tasks.SignalSpec("summary-ip", 0.00025, (1,), 3.5e5, 3.0e5)


def load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def loaded(concat, n_tok, n_val, signals):
    recoder = types.SimpleNamespace(kind="linear", concat=concat, n_tok=n_tok, n_val=n_val, n_values=n_tok * n_val)
    meta = {"signals": {n: dict(zip(("kind", "domain", "n_streams", "values_per_token"), v)) for n, v in signals.items()}}
    return recode.Loaded(recoder, {}, None, meta, {})


def test_the_arms_cover_the_three_sides_of_each_comparison():
    tt = load("train_tokenizer")
    series = list(tt.MAGNETICS) + ["summary-ip"]
    streams = {**dict(zip(tt.MAGNETICS, (40, 18, 18, 15))), "summary-ip": 1, **{n: 24 for n in tt.BANDS},
               "equilibrium-lcfs_r": 1, "equilibrium-psi": 1}
    a = tt.arms(series, ["equilibrium-lcfs_r"], ["equilibrium-psi"], streams)
    assert a["klt.series"].signals == tuple(series) and not a["klt.series"].concat
    assert (a["klt.series"].n_tok, a["klt.series"].n_val) == (1, 16) and a["klt.mag"].signals == tuple(tt.MAGNETICS)
    assert a["pca.ccbv"].concat and a["pca.ccbv"].signals == (tt.SHORT["ccbv"],)
    assert (a["pca.ccbv"].n_tok, a["pca.ccbv"].n_val) == (160, 4) and (a["net.ccbv"].n_tok, a["net.ccbv"].n_val) == (40, 8)
    assert (a["pca.band.cc"].n_tok, a["pca.band.cc"].n_val) == (192, 4) and a["klt.band"].signals == tuple(tt.BANDS)
    assert (a["net.bandcat.omv"].n_tok, a["net.bandcat.omv"].n_val, a["net.bandcat.omv"].concat) == (24, 8, True)
    assert (a["net.profile"].n_tok, a["net.profile"].n_val) == (24, 4) and (a["net.map"].n_tok, a["net.map"].n_val) == (48, 2)
    assert sum(not x.net for x in a.values()) == 11 and sum(x.net for x in a.values()) == 13
    assert tt.net_name("net.series", 2) == "net.series.s2" and tt.net_name("net.mag", 0, "lr1e4") == "net.mag.s0.lr1e4"


def test_recoding_an_assignment_keeps_the_values_per_frame():
    base = {"m": assign.Setting("frame_dct", (2,)), "s": assign.Setting("band_dct", (16, 2, 2, 2, 2, 2, 2, 2), "band"),
            "p": assign.Setting("linear", (32, 4), "signal"), "x": assign.Setting("frame_dct", (4,))}
    per = loaded(False, 1, 16, {"m": ("frame_dct", "linear", 40, 1)})
    got = build.recode_assignment(base, per, "klt.series")
    assert got["m"] == assign.Setting("recoded", (2,), "linear", source="klt.series") and got["x"] == base["x"]
    cat = loaded(True, 160, 4, {"m": ("frame_dct", "linear", 40, 1)})
    assert build.recode_assignment(base, cat, "pca.ccbv")["m"] == assign.Setting(
        "recoded_concat", (80,), "linear", source="pca.ccbv")                      # 40 channels x K = 2
    assert build.recode_assignment(base, cat, "pca.ccbv", scale=0.25)["m"].k == (20,)
    assert build.recode_assignment(base, cat, "pca.ccbv", scale=0.01)["m"].k == (4,)   # at least one token
    band = loaded(False, 1, 32, {"s": ("band_dct", "band", 24, 1)})
    assert build.recode_assignment(base, band, "klt.band")["s"].k == (16, 2, 2, 2, 2, 2, 2, 2)
    bandcat = loaded(True, 24, 8, {"s": ("band_dct", "band", 24, 1)})
    assert build.recode_assignment(base, bandcat, "net.bandcat.cc.s0")["s"].k == (88,)  # 3 x 30 = 90, down to tokens of 8
    profile = loaded(False, 24, 4, {"p": ("linear", "signal", 1, 4)})
    assert build.recode_assignment(base, profile, "net.profile.s0")["p"].k == (32,)
    assert build.recode_assignment(base, loaded(False, 1, 16, {"m": ("frame_dct", "linear", 40, 1),
                                                               "x": ("frame_dct", "linear", 3, 1)}),
                                   "klt.series", signals=["x"])["m"] == base["m"]
    for match, args, kw in (("scale", (base, per, "klt.series"), {"scale": 0.5}),
                            ("domain", ({"m": assign.Setting("frame_dct", (2,), "compressed")}, per, "klt.series"), {}),
                            ("recoded", (got, per, "klt.series"), {}),
                            ("holds", ({"m": assign.Setting("frame_dct", (16,))},
                                       loaded(True, 40, 8, {"m": ("frame_dct", "linear", 40, 1)}), "net.ccbv.s0"), {})):
        with pytest.raises(ValueError, match=match):
            build.recode_assignment(*args, **kw)


def provenance_fixture(tmp_path):
    cache_root, fit_dir = tmp_path / "cache", tmp_path / "fit"
    cache_root.mkdir()
    fit_dir.mkdir()
    (cache_root / "stats.npz").write_bytes(b"statistics")
    sha = transform.stats_sha256(cache_root / "stats.npz")
    (cache_root / "manifest.json").write_text(json.dumps({"stats_sha256": sha, "tokamark_revision": "146b0a0f8738"}))
    st = transform.ChannelStats(compress="none", scale=np.ones(1), mean=np.zeros(1), std=np.ones(1))
    shot = np.random.default_rng(0).standard_normal((1, 400)).astype(np.float32)
    fit.save_fit({"summary-ip": fit.fit_signal(IP, st, [shot], kind="frame_dct")}, fit_dir, {"stats_sha256": sha})
    return cache_root, fit_dir


def test_provenance_of_a_plain_assignment_is_unchanged(tmp_path):
    rt = load("run_tokenizer")
    cache_root, fit_dir = provenance_fixture(tmp_path)
    assign.save_assignment({"summary-ip": assign.Setting("frame_dct", (4,))}, tmp_path / "v1.json")
    p = rt.provenance_of(tmp_path / "v1.json", fit_dir, cache_root, recoders_root=tmp_path / "recoders")
    assert list(p) == ["tokenizer_code_sha256", "assignment_sha256", "fit_sha256", "stats_sha256", "tokamark_revision",
                       "data_root"]


def test_provenance_names_the_recoders_bytes(tmp_path):
    rt = load("run_tokenizer")
    cache_root, fit_dir = provenance_fixture(tmp_path)
    table = identity.StreamTable({"summary-ip": (0, 1)}, np.zeros((1, identity.N_FEAT), dtype=np.float32))
    norm = store.Norm(np.zeros((1, 16)), np.ones(1))
    meta = {"signals": {"summary-ip": {"kind": "frame_dct", "domain": "linear", "n_streams": 1, "values_per_token": 1}}}
    for name, shift in (("klt.x", 0.0), ("klt.y", 1.0)):
        recode.save_recoder(tmp_path / "recoders" / name, recode.LinearRecoder(np.full(16, shift), np.eye(16)),
                            {"summary-ip": norm}, table, meta)
        assign.save_assignment({"summary-ip": assign.Setting("recoded", (4,), source=name)}, tmp_path / f"{name}.json")
    px, py = (rt.provenance_of(tmp_path / f"{n}.json", fit_dir, cache_root, recoders_root=tmp_path / "recoders")
              for n in ("klt.x", "klt.y"))
    assert list(px) == ["tokenizer_code_sha256", "assignment_sha256", "fit_sha256", "stats_sha256",
                        "learned_code_sha256", "sources_sha256", "tokamark_revision", "data_root"]
    assert px["sources_sha256"] != py["sources_sha256"] and len(px["learned_code_sha256"]) == 64
    assign.save_assignment({"summary-ip": assign.Setting("recoded", (4,), source="gone")}, tmp_path / "gone.json")
    with pytest.raises(FileNotFoundError):
        rt.provenance_of(tmp_path / "gone.json", fit_dir, cache_root, recoders_root=tmp_path / "recoders")


def test_left_over_falls_to_zero_with_all_values():
    tt = load("train_tokenizer")
    e = np.random.default_rng(0).standard_normal((500, 16))
    left = tt.left_over(recode.fit_linear(e, 16), e, np.zeros(500, dtype=np.int64), [1, 4, 16])
    assert 1.0 > left[0] > left[1] > left[2] and left[2] < 1e-8
```

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_lrn_command.py -q -W error`
Expected: failures for the missing script and the missing `recode_assignment`.

- [ ] **Step 3: Implement the script, `recode_assignment` and the changes to `run_tokenizer.py`**

Read `scripts/mast/run_tokenizer.py` in parts first and keep its conventions (`paths.prepare_write_root` before the first write, refusals with exit 2 before any file is read, atomic writes). `train_tokenizer.py` does not import torch at import time.

- [ ] **Step 4: Run the tests, and 1b's command tests, which must not change**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_lrn_command.py tests/ignite/mast/test_tok_command.py tests/ignite/mast/test_tok_gate.py -q -W error`
Expected: 5 passed for this task's file, and 1b's two files as before.

- [ ] **Step 5: Trial on eight cached shots (CPU, about five minutes)**

```bash
export IGNITE_MAST_ROOT=/models/intermediate/mast/trials/stage1c-task3
C=/models/intermediate/mast/cache/v1
P='env CUDA_VISIBLE_DEVICES= scripts/mast/py.sh'
$P scripts/mast/run_tokenizer.py fit --n-shots 8 --workers 8 --cache-dir $C
$P - <<'EOF'
from pathlib import Path
from tokamak_foundation_model.ignite.mast import paths, registry, transform
from tokamak_foundation_model.ignite.mast.tokenizers import assign
stats = transform.load_stats(Path("/models/intermediate/mast/cache/v1/stats.npz"))
signals = registry.all_signals()
out = paths.prepare_write_root(paths.stage1_dir() / "assignments")
assign.save_assignment(assign.uniform_assignment(signals, {s.name: stats[s.name].compress for s in signals}, rung=1),
                       out / "v1.json")
EOF
$P scripts/mast/train_tokenizer.py store --n-val 2 --workers 8 --cache-dir $C
$P scripts/mast/train_tokenizer.py linear --arm klt.mag
$P scripts/mast/train_tokenizer.py linear --arm pca.ccbv
$P scripts/mast/run_tokenizer.py recode --name t-klt --from v1 --arm klt.mag
$P scripts/mast/run_tokenizer.py recode --name t-pca --from v1 --arm pca.ccbv
$P scripts/mast/train_tokenizer.py arms
```

Expected: `store` prints 24 lines; each `linear` prints one line per signal with five fractions between 0 and 1 that fall from left to right; `recode` prints the four magnetics signals with 80, 36, 36 and 30 values per frame before and after (K = 2 at rung 1), then the `ccbv` signal with 80 and 80; `arms` lists 24 arms, two of them present.

- [ ] **Step 6: Commit**

```bash
git add scripts/mast/train_tokenizer.py scripts/mast/run_tokenizer.py tests/ignite/mast/test_lrn_command.py \
        src/tokamak_foundation_model/ignite/mast/tokenizers/learned/build.py
git commit -m "ignite.mast: the arms of the tokenizer comparison; an assignment can name a recoder and carries its bytes"
```

---

### Task 4: the comparison

**Files:**
- Create: `src/tokamak_foundation_model/ignite/mast/tokenizers/learned/compare.py`
- Modify: `scripts/mast/run_tokenizer.py` (sub-commands `compare`, `seeds`), `scripts/mast/train_tokenizer.py` (sub-command `joint`)
- Test: `tests/ignite/mast/test_lrn_compare.py`

**Interfaces:**
- Consumes: 1b's `ladder.shot_error(codec: Codec, spec: SignalSpec, z_model: np.ndarray, mode: str = "unquantized") -> float | None` (one shot's cached values in model units to the round-trip RMSE in TokaMark units; None without a valid sample), `ladder.values_per_frame(spec, setting) -> int`, `assign.predicted_by() -> dict[str, tuple[str, ...]]`, `tasks.best_published(task) -> float`, `codecs.fit_pca`, the command's `kept_score(out_dir, task, predictor, label, provenance, force) -> tuple[dict | None, str]` and `predictor_name("floor", name, mode)` (`tok.<name>.<mode>`), and the score tree `<stage1>/scores/<task>/<predictor>/<label>/` in which the evaluator leaves `<task>/shots_metrics.csv` (columns `shot_id, feature_name, n_windows, NRMSE_mean, ...`); existing `ignite.gate.utilization(codes, cfg=None, *, codebook_size=None, min_utilization=None, min_code_entropy=None) -> dict` (integer codes `(B, n_tok, fsq_dim)`; returns `dim_entropy_nats`; `cfg.fsq_levels` gives the levels); Task 2's `RecodedCodec`.
- Produces in `compare.py` (no torch at import time; `ignite.gate` is imported inside `usage`):
  - `paired_gap(a: Mapping[int, float], b: Mapping[int, float], *, n_boot: int = 2000, seed: int = 0) -> dict`: keys `n_shots, mean_a, mean_b, gap, lo, hi` over the shots with a finite value on both sides; `gap = (mean_b - mean_a) / mean_b`; `lo`, `hi` the 2.5 and 97.5 percentiles over resamples of those shots; NaN for `gap`, `lo`, `hi` with fewer than two shots.
  - `beats(g: Mapping[str, float], *, seed_spread: float | None = None) -> bool` (rule 2) and `clearly_beats(g, *, seed_spread=None, margin: float = 0.10) -> bool`.
  - `word(ab: Mapping[str, float], ba: Mapping[str, float], seed_spread: float | None = None) -> str`: `a clearly beats b`, `a beats b`, `b clearly beats a`, `b beats a`, `tie` or `no data`.
  - `seed_spread(gaps: Sequence[float]) -> float`: largest minus smallest; NaN for fewer than two.
  - `per_shot_harness(score_dir: Path, task: str, signal: str) -> dict[int, float]`: `NRMSE_mean` per `shot_id` for that `feature_name`; `{}` if the file is missing.
  - `per_shot_cache(codec: Codec, spec: SignalSpec, per_shot: Iterable[tuple[int, np.ndarray | None]], mode: str = "unquantized") -> dict[int, float]`: `ladder.shot_error` per shot; a shot without the signal or without a valid sample is left out.
  - `values_to_reach(curve: Mapping[int, float], target: float) -> int | None`: the smallest size whose error is at or under the target.
  - `scaled_k(k: int, factor: float, step: int, limit: int) -> int | None`: `k * factor` rounded down to a multiple of `step`; None if that is under `step` or over `limit`.
  - `level_index(values: np.ndarray, levels: int) -> np.ndarray`: `rint((v + 1) / 2 * (levels - 1))` as int64.
  - `usage(values: np.ndarray, levels: int = 8, min_entropy: float = 0.3) -> dict` (rule 7): `values (M, n_tok, n_val)` in [-1, 1], examples with NaN dropped; keys `first_value_entropy` (the smallest over the tokens of the normalised entropy of value 0, from `utilization` on one token at a time), `active_values` (the number of value positions whose smallest normalised entropy over the tokens is at least `min_entropy`), `collapsed`.
  - `missing_stream_cost(codec: Codec, spec: SignalSpec, z_model: np.ndarray, channel: int) -> tuple[float, float] | None`: the round-trip RMSE in TokaMark units over the valid samples of the other channels, with the shot as it is and with `channel` set to NaN; None without a valid sample. Series signals only.
  - `left_over(train: np.ndarray, val: np.ndarray, sizes: Sequence[int]) -> list[float]`: principal components of `train (M, d)` (NaN allowed), and for each size the share of the mean square of `val` about the training mean that the first `size` components leave; rows of `val` with NaN are dropped.
- Produces in the scripts:
  - `run_tokenizer.py compare --a NAME --b NAME [--label val_select-n100] [--signals x,y] [--drop-channel] [--reach TARGET] [--workers 64] [--cache-dir DIR]`: for each signal whose setting differs between the two assignments (or the listed ones). If a task predicts the signal: the per-shot errors from the stored floors `tok.<NAME>.unquantized` under the label, of the task with the smallest best published score among those that predict it, each read only if `kept_score` keeps it (otherwise the row says `not scored` and the command prints the `score` command that is missing). If no task predicts it: `per_shot_cache` over the label's shots (`cache.pick_evenly` of the label's split and size). Writes `<stage1>/compare/<a>__<b>.<label>.csv` (columns `signal, source, values_a, values_b, n_shots, mean_a, mean_b, gap, lo, hi, gap_ba, lo_ba, hi_ba, one_seed, drop_a, drop_b, reach_a, reach_b`, where `source` is the task or `cache` and `one_seed` is `word` without a seed spread) and prints it. The last four columns are filled only for a series signal with more than one channel whose source is `cache`. With `--drop-channel`: `drop_a`, `drop_b`, the mean over the shots of `missing_stream_cost` for channel `S // 2`, as the ratio of the second number to the first. With `--reach TARGET`: `reach_a`, `reach_b`, `values_to_reach` over the mean cache-side errors at 1/8, 1/4, 1/2, 1 and 2 times the setting's values (`scaled_k` with `step` 1 and `limit` 16 for a per-channel setting, the recoder's `n_val` and `n_values` for a concatenated one; a size that does not exist is left out), in values per frame.
  - `run_tokenizer.py seeds --pairs a0:b,a1:b,a2:b --label L`: reads the stored comparison files of the pairs and prints per signal the gaps, their spread and `word` with that spread. Exit 2 naming a missing file.
  - `train_tokenizer.py joint --signals a,b,c,d [--sizes 1,2,4,8]`: on the stores, in example units: `left_over` of the joint basis of all streams of the listed signals (frames in which every listed signal is present), of each signal's own concatenated basis, and of the pooled per-stream basis, each at `k` values per stream in total for the listed `k`. Prints one line per size.

- [ ] **Step 1: Write the failing tests**

```python
"""Paired gaps, verdicts, collapse and the cost of a missing channel (plan 1c)."""

import numpy as np
import pandas as pd
import pytest

from tokamak_foundation_model.ignite.mast import tasks, transform
from tokamak_foundation_model.ignite.mast.tokenizers import codecs
from tokamak_foundation_model.ignite.mast.tokenizers.learned import compare, recode, store

RNG = np.random.default_rng(0)
SPEC = tasks.SignalSpec("m", 0.00025, (3,), 0.0, 1.0)


def window(n=6):
    w = RNG.standard_normal((n, 3, 100)).cumsum(axis=-1) * 0.1
    return w + 0.5 * np.roll(w, 1, axis=1)


def codec(concat, n_keep):
    shape = (3,)
    st = transform.ChannelStats(compress="none", scale=np.ones(shape), mean=np.zeros(shape), std=np.ones(shape))
    front = codecs.FrameDCT("m", 0.00025, shape, st, 16)
    fv = front.encode(window(40)).reshape(-1, 3, 16)
    norm = store.fit_norm(fv)
    e = norm.apply(fv)
    r = recode.fit_linear(e.reshape(len(e), -1), 24, concat=True, n_val=4) if concat else recode.fit_linear(e.reshape(-1, 16), 16)
    return recode.RecodedCodec(front, norm, r, np.arange(3), (n_keep,))


def test_the_gap_is_paired_over_the_shots_both_sides_have():
    rng = np.random.default_rng(1)
    b = {s: float(v) for s, v in enumerate(rng.uniform(0.05, 0.15, 100))}
    a = {s: 0.8 * v for s, v in b.items()}
    a[100] = 0.01                                        # a shot only one side has
    del a[0]
    a[1] = float("nan")
    g = compare.paired_gap(a, b)
    assert g["n_shots"] == 98 and g["gap"] == pytest.approx(0.2) and g["lo"] == pytest.approx(0.2) == g["hi"]
    assert compare.paired_gap(a, b) == g                 # seeded
    assert compare.beats(g) and compare.clearly_beats(g) and not compare.beats(compare.paired_gap(b, a))
    assert not compare.clearly_beats(g, margin=0.25) and not compare.beats(g, seed_spread=0.3)
    assert compare.word(g, compare.paired_gap(b, a)) == "a clearly beats b"
    assert compare.word(compare.paired_gap(b, a), g) == "b clearly beats a"
    swing = {s: v * (1.2 if s % 2 else 0.8) for s, v in b.items()}
    n = compare.paired_gap(swing, b)
    assert n["lo"] < 0 < n["hi"] and not compare.beats(n) and compare.word(n, compare.paired_gap(b, swing)) == "tie"
    one = compare.paired_gap({1: 0.1}, {1: 0.2})
    assert one["n_shots"] == 1 and np.isnan(one["lo"]) and not compare.beats(one) and compare.word(one, one) == "no data"


def test_per_shot_errors_are_read_from_the_evaluators_table(tmp_path):
    (tmp_path / "task_x").mkdir()
    pd.DataFrame({"shot_id": [1, 1, 2], "feature_name": ["a", "b", "a"], "n_windows": 5,
                  "NRMSE_mean": [0.1, 0.2, 0.3]}).to_csv(tmp_path / "task_x" / "shots_metrics.csv", index=False)
    assert compare.per_shot_harness(tmp_path, "task_x", "a") == {1: 0.1, 2: 0.3}
    assert compare.per_shot_harness(tmp_path, "task_y", "a") == {}


def test_per_shot_cache_errors_skip_a_shot_without_the_signal():
    c = codec(False, 4)
    shots = [(30000, window(1)[0].astype(np.float32)), (30001, None), (30002, np.full((3, 100), np.nan, dtype=np.float32))]
    got = compare.per_shot_cache(c, SPEC, shots)
    assert list(got) == [30000] and 0.0 < got[30000] < 1.0


def test_values_to_reach_and_the_seed_spread():
    curve = {20: 0.05, 40: 0.02, 80: 0.012, 160: 0.010}
    assert compare.values_to_reach(curve, 0.0176) == 80 and compare.values_to_reach(curve, 0.001) is None
    assert compare.seed_spread([0.12, 0.10, 0.15]) == pytest.approx(0.05) and np.isnan(compare.seed_spread([0.1]))
    assert compare.scaled_k(2, 0.5, 1, 16) == 1 and compare.scaled_k(2, 0.25, 1, 16) is None
    assert compare.scaled_k(80, 0.125, 4, 640) == 8 and compare.scaled_k(80, 2, 8, 320) == 160
    assert compare.scaled_k(16, 2, 1, 16) is None


def test_a_constant_first_value_is_a_collapse():
    v = np.random.default_rng(0).uniform(-1, 1, (4000, 3, 4))
    u = compare.usage(v)
    assert not u["collapsed"] and u["active_values"] == 4 and u["first_value_entropy"] > 0.95
    v[:, :, 3] = 0.01                                    # an unused last value is not a collapse
    u = compare.usage(v)
    assert not u["collapsed"] and u["active_values"] == 3
    v[:, 1, 0] = -0.3                                    # token 1 never moves its first value
    v[5] = np.nan
    u = compare.usage(v)
    assert u["collapsed"] and u["first_value_entropy"] == 0.0
    assert compare.level_index(np.array([-1.0, -0.99, 0.01, 1.0]), 8).tolist() == [0, 0, 4, 7]


def test_a_missing_channel_costs_the_concatenated_codec_and_not_the_per_channel_one():
    shots = [w.astype(np.float32) for w in window(8)]
    per = [compare.missing_stream_cost(codec(False, 4), SPEC, z, 1) for z in shots[:1]]
    assert per[0][0] == pytest.approx(per[0][1], abs=1e-6)
    cat = codec(True, 12)
    costs = np.array([compare.missing_stream_cost(cat, SPEC, z, 1) for z in shots])
    assert costs[:, 1].mean() > costs[:, 0].mean()
    assert compare.missing_stream_cost(cat, SPEC, np.full((3, 100), np.nan, dtype=np.float32), 1) is None


def test_left_over_of_a_low_rank_signal():
    q, _ = np.linalg.qr(RNG.standard_normal((16, 16)))
    e = (RNG.standard_normal((8000, 3)) * [3.0, 2.0, 1.0]) @ q[:, :3].T + 0.01 * RNG.standard_normal((8000, 16))
    left = compare.left_over(e[:4000], e[4000:], [1, 3, 16])
    assert left[0] == pytest.approx(5.0 / 14.0, rel=0.1) and left[1] < 1e-3 and left[2] < 1e-8
```

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_lrn_compare.py -q -W error`
Expected: an import error for `compare`.

- [ ] **Step 3: Implement `compare.py` and the three sub-commands**

`usage` gives `utilization` a `types.SimpleNamespace(codebook_size=levels ** min(n_val, 20), fsq_levels=(levels,) * n_val)` and `min_utilization=0.0`, one token at a time, and divides `dim_entropy_nats` by `log(levels)`.

- [ ] **Step 4: Run the tests**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_lrn_compare.py tests/ignite/mast/test_lrn_command.py -q -W error`
Expected: 12 passed.

- [ ] **Step 5: Trial on the Task 3 trial root (CPU, about two minutes)**

```bash
export IGNITE_MAST_ROOT=/models/intermediate/mast/trials/stage1c-task3
P='env CUDA_VISIBLE_DEVICES= scripts/mast/py.sh'
$P scripts/mast/run_tokenizer.py compare --a t-klt --b v1 --label val_select-n2 --workers 4 --cache-dir /models/intermediate/mast/cache/v1
$P scripts/mast/run_tokenizer.py compare --a t-pca --b t-klt --label val_select-n2 --drop-channel --reach 0.0176 --workers 4 --cache-dir /models/intermediate/mast/cache/v1
$P scripts/mast/train_tokenizer.py joint --signals ccbv,obr,obv,flux --sizes 1,2
```

Expected: the first table has the four magnetics signals with `source` `cache`, `n_shots` 2 and finite means; the second has one row (`ccbv`) with `values_a` and `values_b` both 80, `drop_b` equal to 1 (a per-channel tokenizer), `drop_a` at least 1, and `reach_a`, `reach_b` each a number of values per frame or empty; `joint` prints two lines of three fractions each. With two shots the verdicts mean nothing. If the trial root is gone, run Task 3's trial again first.

- [ ] **Step 6: Commit**

```bash
git add src/tokamak_foundation_model/ignite/mast/tokenizers/learned/compare.py scripts/mast/run_tokenizer.py \
        scripts/mast/train_tokenizer.py tests/ignite/mast/test_lrn_compare.py
git commit -m "ignite.mast: two tokenizers are compared pairwise over shots, with collapse and missing-channel checks"
```

---

### Task 5: the quantizer and the token network

**Files:**
- Create: `src/tokamak_foundation_model/ignite/mast/tokenizers/learned/quantizer.py`, `learned/nets.py`
- Test: `tests/ignite/mast/test_lrn_nets.py`

**Interfaces:**
- Consumes: Task 1's `identity.N_FEAT`, `identity.StreamTable` (`features (n, N_FEAT)`); Task 2's `recode.Recoder` (attributes `kind, concat, d_in, n_tok, n_val`, property `n_values`; `encode(e (M, d_in), ident (M,)) -> (M, n_values)` float32 and `decode(values, ident) -> (M, d_in)`, numpy, example units), `recode.save_recoder(out_dir, recoder, norms, table, meta, ranges=None)`, `recode.load_recoder(directory, device="cpu") -> Loaded`, which calls this task's `load_net_recoder(directory, table, device)` for a recoder of kind `"net"`.
- Produces in `quantizer.py`:
  - `LEVELS = 8`, `FINE_LEVELS = 64`, `P_PRECISION = (0.25, 0.25, 0.5)`, `P_FULL = 0.5`.
  - `round_levels(v: torch.Tensor, levels: int) -> torch.Tensor`: the nearest of `levels` evenly spaced numbers from -1 to 1; the gradient passes as if nothing was rounded.
  - `apply(v: torch.Tensor, precision: torch.Tensor, n_keep: torch.Tensor) -> torch.Tensor`: `v (B, n_values)` in [-1, 1]. `precision (B,)` int64: 0 reads the example's values at `LEVELS`, 1 at `FINE_LEVELS`, 2 leaves them unrounded. `n_keep (B,)` int64: the values from index `n_keep` on are exactly 0.
  - `draw(batch: int, n_values: int, generator: torch.Generator, device=None) -> tuple[torch.Tensor, torch.Tensor]`: `precision` with the probabilities `P_PRECISION`; `n_keep` equal to `n_values` with probability `P_FULL`, otherwise uniform over 1 to `n_values`.
- Produces in `nets.py`:
  - `symlog(x: torch.Tensor) -> torch.Tensor`: `sign(x) * log1p(|x|)`.
  - `class TokenNet(torch.nn.Module)`: `TokenNet(d_in: int, n_values: int, n_streams: int, *, d_feat: int = identity.N_FEAT, d_model: int = 512, depth: int = 4, p_drop_ident: float = 0.1)`; keeps `d_in` and `n_values` as attributes.
    - Identity: `self.learned = nn.Embedding(n_streams, d_model)`, initialised to zero, and a two-layer network over the table's features. `identity(self, feats: torch.Tensor, ident: torch.Tensor, drop: torch.Tensor | None = None) -> torch.Tensor` gives `(B, d_model)`: the feature network's output plus the learned vector of row `ident`; the learned vector is left out where `ident == -1` or `drop` is True. With `n_streams == 0` (the concatenated control) there are no identity parts and it returns zeros.
    - `encode(self, e, feats, ident, drop=None) -> torch.Tensor`: `(B, n_values)` in (-1, 1). `decode(self, v, feats, ident, drop=None) -> torch.Tensor`: `(B, d_in)` in example units.
    - `forward(self, e, feats, ident, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]`: `(e_hat, v)`. In training mode every random draw comes from `generator` (`ValueError` naming "generator" without one): `drop` with probability `p_drop_ident` per example, `quantizer.draw`, and `e_hat = decode(quantizer.apply(encode(e), precision, n_keep))`. In eval mode the values pass unchanged.
    - Body (the author's choice; the tests do not pin it): encoder `Linear(d_in, d_model)` on `symlog(e)`, plus the identity vector, then `depth` residual blocks (`LayerNorm`, `Linear(d_model, 2 * d_model)`, `GELU`, `Linear(2 * d_model, d_model)`), `LayerNorm`, `Linear(d_model, n_values)`, `tanh`. The decoder is `Linear(n_values, d_model)` plus the identity vector, the same blocks, `LayerNorm`, `Linear(d_model, d_in)`. No dropout layer.
  - `class NetRecoder(recode.Recoder)`: `NetRecoder(net: TokenNet, table: StreamTable, *, concat: bool, n_tok: int, n_val: int, device: str = "cpu", batch: int = 65536)`; `kind = "net"`, `d_in = net.d_in`. Puts the net in eval mode; `encode` and `decode` take and return numpy, run without gradients in batches, and look the features up in `table.features` (zeros for `ident == -1`).
  - `save_net(path: Path, net: TokenNet, config: dict) -> None`: `{"state": net.state_dict(), "config": config}` with `torch.save`, written to a temporary name and renamed. `config` holds `d_in, n_values, n_streams, d_model, depth, p_drop_ident, concat, n_tok, n_val`.
  - `load_net_recoder(directory: Path, table: StreamTable, device: str = "cpu") -> NetRecoder`: from `directory / "best.pt"` with `torch.load(..., map_location=device, weights_only=True)`.

- [ ] **Step 1: Write the failing tests**

```python
"""The quantizer of the learned tokenizers and the token network (plan 1c)."""

import numpy as np
import pytest
import torch

from tokamak_foundation_model.ignite.mast.tokenizers.learned import identity, nets, quantizer, recode, store

META = {"signals": {"m": {"kind": "frame_dct", "domain": "linear", "n_streams": 3, "values_per_token": 1}}}


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def level_positions(x, n):
    pos = (quantizer.round_levels(x, n).detach().numpy() + 1) / 2 * (n - 1)
    assert np.abs(pos - np.rint(pos)).max() < 1e-4
    return np.unique(np.rint(pos))


def test_rounding_keeps_the_ends_and_passes_the_gradient():
    v = torch.tensor([-1.0, -0.99, 0.01, 0.99, 1.0], requires_grad=True)
    r = quantizer.round_levels(v, 8)
    assert torch.allclose(r, torch.tensor([-1.0, -1.0, 1 / 7, 1.0, 1.0]), atol=1e-6)
    r.sum().backward()
    assert torch.allclose(v.grad, torch.ones(5))
    x = torch.linspace(-1, 1, 1001)
    assert len(level_positions(x, 8)) == 8 and len(level_positions(x, 64)) == 64
    coarse = quantizer.round_levels(x, 8)
    assert torch.allclose(quantizer.round_levels(coarse, 64), coarse, atol=1e-6)   # the 8 levels are among the 64


def test_dropped_values_are_zero_and_the_rest_keep_their_precision():
    v = torch.tensor([[0.30, -0.52, 0.11, 0.90]] * 3)
    out = quantizer.apply(v, torch.tensor([2, 0, 1]), torch.tensor([2, 4, 3]))
    assert torch.allclose(out[0, :2], v[0, :2], atol=1e-7) and out[0, 2:].abs().sum() == 0        # unrounded, two kept
    assert torch.allclose(out[1], torch.tensor([3 / 7, -3 / 7, 1 / 7, 1.0]), atol=1e-6)           # 8 levels, all kept
    assert torch.allclose(out[2, :3], torch.tensor([19 / 63, -33 / 63, 7 / 63]), atol=1e-6) and out[2, 3] == 0


def test_the_draw_skips_rounding_for_half_and_keeps_all_values_for_more_than_half():
    precision, n_keep = quantizer.draw(20000, 8, gen())
    assert 0.48 < (precision == 2).float().mean() < 0.52 and 0.23 < (precision == 0).float().mean() < 0.27
    assert 0.54 < (n_keep == 8).float().mean() < 0.585 and n_keep.min() == 1 and n_keep.max() == 8
    again = quantizer.draw(20000, 8, gen())
    assert torch.equal(again[0], precision) and torch.equal(again[1], n_keep)


def test_the_net_gives_bounded_values_and_decodes_them():
    torch.manual_seed(0)
    net = nets.TokenNet(16, 8, 5, d_model=32, depth=1)
    e = 50.0 * torch.randn(64, 16)                                   # far outside unit scale
    feats, ident = torch.zeros(64, identity.N_FEAT), torch.arange(64) % 5
    net.eval()
    e_hat, v = net(e, feats, ident)
    assert v.shape == (64, 8) and v.abs().max() <= 1 and e_hat.shape == (64, 16) and torch.isfinite(e_hat).all()
    assert torch.allclose(net.decode(v, feats, ident), e_hat)        # eval mode: the values pass unchanged
    net.train()
    with pytest.raises(ValueError, match="generator"):
        net(e, feats, ident)
    a, _ = net(e, feats, ident, generator=gen(1))
    b, _ = net(e, feats, ident, generator=gen(1))
    assert torch.equal(a, b) and not torch.allclose(a, e_hat)


def test_identity_without_a_table_row_is_the_dropped_identity():
    torch.manual_seed(0)
    net = nets.TokenNet(16, 8, 5, d_model=32, depth=1).eval()
    with torch.no_grad():
        net.learned.weight.normal_()
    feats = torch.randn(4, identity.N_FEAT)
    known, unknown = torch.arange(4), torch.full((4,), -1)
    dropped = net.identity(feats, known, drop=torch.ones(4, dtype=torch.bool))
    assert torch.allclose(net.identity(feats, unknown), dropped)
    assert not torch.allclose(net.identity(feats, known), dropped)
    assert not torch.allclose(dropped[0], dropped[1])                # the table's features still tell streams apart
    control = nets.TokenNet(48, 24, 0, d_model=32, depth=1)
    assert control.identity(feats, unknown).abs().sum() == 0


def test_the_net_recoder_is_a_recoder_and_is_stored(tmp_path):
    torch.manual_seed(0)
    table = identity.StreamTable({"m": (0, 3)}, np.random.default_rng(0).standard_normal((3, identity.N_FEAT)).astype(np.float32))
    config = dict(d_in=16, n_values=8, n_streams=3, d_model=32, depth=1, p_drop_ident=0.1, concat=False, n_tok=1, n_val=8)
    net = nets.TokenNet(16, 8, 3, d_model=32, depth=1)
    r = nets.NetRecoder(net, table, concat=False, n_tok=1, n_val=8, batch=7)
    e = np.random.default_rng(1).standard_normal((20, 16)).astype(np.float32)
    ident = np.arange(20) % 3
    v = r.encode(e, ident)
    assert isinstance(r, recode.Recoder) and r.kind == "net" and (r.d_in, r.n_values) == (16, 8) and not net.training
    assert v.shape == (20, 8) and v.dtype == np.float32 and np.abs(v).max() <= 1 and r.decode(v, ident).shape == (20, 16)
    whole = nets.NetRecoder(net, table, concat=False, n_tok=1, n_val=8)
    assert np.allclose(whole.encode(e, ident), v, atol=1e-6)         # the batch size changes nothing
    nets.save_net(tmp_path / "best.pt", net, config)
    back = nets.load_net_recoder(tmp_path, table)
    assert np.allclose(back.encode(e, ident), v, atol=1e-6) and (back.concat, back.n_tok, back.n_val) == (False, 1, 8)
    recode.save_recoder(tmp_path, r, {"m": store.Norm(np.zeros((3, 16)), np.ones(3))}, table, META)
    loaded = recode.load_recoder(tmp_path)
    assert loaded.recoder.kind == "net" and np.allclose(loaded.recoder.decode(v, ident), r.decode(v, ident), atol=1e-6)
```

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_lrn_nets.py -q -W error`
Expected: an import error for `nets`.

- [ ] **Step 3: Implement `quantizer.py` and `nets.py`**

- [ ] **Step 4: Run the tests, and Task 2's, which load a network recoder through this task**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_lrn_nets.py tests/ignite/mast/test_lrn_recode.py -q -W error`
Expected: 6 passed for this task's file, and Task 2's file as before.

- [ ] **Step 5: Commit**

```bash
git add src/tokamak_foundation_model/ignite/mast/tokenizers/learned/quantizer.py \
        src/tokamak_foundation_model/ignite/mast/tokenizers/learned/nets.py tests/ignite/mast/test_lrn_nets.py
git commit -m "ignite.mast: a token network with ordered, bounded values and a quantizer skipped for half of the examples"
```

---

### Task 6: training a network recoder

**Files:**
- Create: `src/tokamak_foundation_model/ignite/mast/tokenizers/learned/train.py`
- Modify: `scripts/mast/train_tokenizer.py` (sub-command `train`)
- Test: `tests/ignite/mast/test_lrn_train.py`

**Interfaces:**
- Consumes: Task 1's `store.Store` (`values (M, S, D)`, NaN for a stream without a token), `store.Norm`, `fit_norm`, `load_store`, `identity.StreamTable.index_of(name, n_streams)`, `identity.build_table`; Task 5's `TokenNet`, `NetRecoder`, `save_net`, `load_net_recoder`; Task 2's `recode.save_recoder`; Task 3's `Arm`, `arms`, `net_name`; Task 4's `compare.usage(values (M, n_tok, n_val)) -> dict` (key `collapsed`); from `scripts/mast/run_headroom.py`, loaded by path as 1b's command does: `_look_up_gpu(gpu: int) -> tuple[str, list[int]] | None` (the GPU's uuid and its compute pids; None after it printed why), `_refuse_busy_gpu(gpu: int, pids: list[int], share: bool = False) -> bool`, `_take_lock(run_dir: Path) -> Path` (raises `Refused` if a live process holds the run), `_release_lock(lock: Path) -> None`.
- Produces in `train.py`:
  - `@dataclass(frozen=True) class TrainConfig` with `d_model: int = 512`, `depth: int = 4`, `steps: int = 20000`, `batch: int = 4096`, `lr: float = 3e-4`, `weight_decay: float = 0.01`, `warmup: int = 500`, `clip: float = 1.0`, `val_every: int = 1000`, `seed: int = 0`, `p_drop_ident: float = 0.1`.
  - `examples(stores: Mapping[str, Store], norms: Mapping[str, Norm], table: StreamTable, *, concat: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]`: `(e, mask, ident)` in example units. Per-stream: one row per (frame, stream) whose D values are finite, signals in the order of `stores`; `e (M, D)` float32, `mask` all True, `ident` the stream's row in the table. Concatenated (exactly one signal, `ValueError` otherwise): one row per frame with at least one finite stream; `e (M, S * D)` with the streams one after the other, 0 where a stream is missing and `mask` False there; `ident` all -1.
  - `prefix_lengths(n_values: int) -> list[int]`: the powers of two below `n_values`, then `n_values`.
  - `train(net_config: dict, cfg: TrainConfig, train_ex: tuple, val_ex: tuple, table: StreamTable, out_dir: Path, device: str = "cpu", log: Callable[[str], None] = print) -> dict`: `net_config` is the `config` of `nets.save_net` and is what the net is built from (the sub-command fills its `d_model`, `depth` and `p_drop_ident` from the `TrainConfig`). AdamW, a linear warm-up over `warmup` steps and a constant rate after it, gradient clipping at `clip`; the examples are held as tensors on the device and each batch is drawn with replacement by a `torch.Generator` seeded with `cfg.seed` (the same generator feeds the net's `forward`). The loss is the mean of `(e_hat - e) ** 2` over the entries where `mask` is True. Every `val_every` steps and after the last step, in eval mode over all validation examples: `val_full` (all values, unrounded) and `val_mse`, the mean over `prefix_lengths(n_values)` of the same error with the values after the prefix set to 0. One row per validation in `out_dir / "train_log.csv"` (`step, train_loss, val_mse, val_full`; `train_loss` is the mean since the row before); `best.pt` is written through `save_net` at each new lowest `val_mse`. A loss that is not finite raises `FloatingPointError` naming the step. Returns `{"best_step", "best_val_mse", "best_val_full", "steps", "seconds"}` (`best_val_full` is `val_full` at the best step).
  - `plateau(log: pd.DataFrame, rel: float = 0.01) -> bool`: True if the lowest `val_mse` of the last quarter of the rows is not more than `rel` (as a share) under the lowest of the rows before it; False with fewer than four rows.
- Produces in the script: `train --arm NAME [--seed 0] (--gpu N [--share-gpu] | --cpu) [--lr 3e-4] [--steps 20000] [--variant V] [--force]`.
  - Refused with exit 2 before the store is read: an unknown or a linear arm; neither `--gpu` nor `--cpu`; `--lr` or `--steps` other than the default without `--variant`; a run directory that has `result.json`, without `--force`; a busy GPU without `--share-gpu` (the two functions above; set `CUDA_VISIBLE_DEVICES` to the uuid before torch is imported).
  - The run directory is `<stage1>/recoders/<net_name(arm, seed, variant)>/`. Norms from the training store per signal; the table over the arm's signals; `examples` of the training and the validation store; the net has `d_in` D (S times D concatenated), `n_values = arm.n_tok * arm.n_val`, `n_streams = len(table)` (0 concatenated).
  - Written in this order: `config.json` (arm, seed, variant, the `TrainConfig`, the net's config) and `trained_by.json` (`learned_code_sha256`, `tokenizer_code_sha256`, the store's `assignment_sha256`, `fit_sha256` and `stats_sha256`, `argv`, `date`) before the first step; `train.lock` for the duration; `train_log.csv` and `best.pt`; then `recoder.json`, `norm.npz`, `table.npz` through `save_recoder` with the store's meta (no ranges: a network's values have the even grid); last `result.json`: what `train` returned, `plateau`, and the collapse check of rule 7 on the validation examples' values (per-stream arm: `compare.usage` once per signal, pooled over its streams; concatenated: over the signal's tokens).
  - Last printed line: `<name>: best val_mse <v> at step <s> of <n>, plateau <yes|no>, collapsed <yes|no>, <t> s`. Exit 3 if collapsed, else 0.

- [ ] **Step 1: Write the failing tests**

```python
"""Examples for and the training of a network recoder (plan 1c)."""

import numpy as np
import pandas as pd
import pytest

from tokamak_foundation_model.ignite.mast.tokenizers.learned import identity, nets, store, train

RNG = np.random.default_rng(0)
TABLE = identity.StreamTable({"m": (0, 3)}, RNG.standard_normal((3, identity.N_FEAT)).astype(np.float32))
CONFIG = dict(d_in=16, n_values=8, n_streams=3, d_model=64, depth=1, p_drop_ident=0.1, concat=False, n_tok=1, n_val=8)


def test_examples_per_stream_and_concatenated():
    values = RNG.standard_normal((11, 3, 4)).astype(np.float32)
    values[0, 1] = np.nan                                  # one stream missing in one frame
    values[10] = np.nan                                    # a frame without any stream
    stores = {"m": store.Store("m", values, np.zeros(11, dtype=np.int64))}
    norms = {"m": store.Norm(np.zeros((3, 4)), np.full(3, 2.0))}
    e, mask, ident = train.examples(stores, norms, TABLE, concat=False)
    assert e.shape == (29, 4) and mask.all() and np.isfinite(e).all() and e.dtype == np.float32
    assert np.bincount(ident).tolist() == [10, 9, 10] and np.allclose(e[0], values[0, 0] / 2.0)
    e, mask, ident = train.examples(stores, norms, TABLE, concat=True)
    assert e.shape == (10, 12) and mask.sum() == 116 and (ident == -1).all()
    assert not mask[0, 4:8].any() and (e[0, 4:8] == 0).all() and np.allclose(e[1, 4:8], values[1, 1] / 2.0)
    with pytest.raises(ValueError):
        train.examples({**stores, "n": stores["m"]}, {**norms, "n": norms["m"]}, TABLE, concat=True)
    assert train.prefix_lengths(16) == [1, 2, 4, 8, 16] and train.prefix_lengths(24) == [1, 2, 4, 8, 16, 24]


def rank_two(m):
    q, _ = np.linalg.qr(np.random.default_rng(1).standard_normal((16, 16)))
    e = (RNG.standard_normal((m, 2)) * [1.0, 0.5]) @ q[:, :2].T + 0.01 * RNG.standard_normal((m, 16))
    e = (e / 0.28).astype(np.float32)                      # about unit variance per entry
    return e, np.ones_like(e, dtype=bool), RNG.integers(0, 3, m)


def prefix_error(recoder, e, ident, k):
    v = recoder.encode(e, ident)
    v[:, k:] = 0
    return float(np.mean((recoder.decode(v, ident) - e) ** 2))


def test_a_short_training_learns_a_low_rank_signal_and_keeps_its_best_weights(tmp_path):
    cfg = train.TrainConfig(d_model=64, depth=1, steps=400, batch=256, lr=1e-3, warmup=20, val_every=100)
    train_ex, val_ex = rank_two(4000), rank_two(1000)
    res = train.train(CONFIG, cfg, train_ex, val_ex, TABLE, tmp_path, log=lambda line: None)
    variance = float(np.mean(val_ex[0] ** 2))
    assert res["best_val_mse"] < 0.5 * variance and res["steps"] == 400
    log = pd.read_csv(tmp_path / "train_log.csv")
    assert log["step"].tolist() == [100, 200, 300, 400] and list(log.columns) == ["step", "train_loss", "val_mse", "val_full"]
    assert res["best_step"] == int(log.loc[log["val_mse"].idxmin(), "step"])
    r = nets.load_net_recoder(tmp_path, TABLE)
    assert prefix_error(r, val_ex[0], val_ex[2], 8) == pytest.approx(res["best_val_full"], rel=1e-3)
    assert prefix_error(r, val_ex[0], val_ex[2], 1) > prefix_error(r, val_ex[0], val_ex[2], 8)   # a prefix is coarser


def test_a_loss_that_is_not_finite_stops_the_training(tmp_path):
    e, mask, ident = rank_two(500)
    cfg = train.TrainConfig(d_model=64, depth=1, steps=50, batch=64, val_every=50)
    with pytest.raises(FloatingPointError):
        train.train(CONFIG, cfg, (np.full_like(e, np.inf), mask, ident), (e, mask, ident), TABLE, tmp_path,
                    log=lambda line: None)


def test_plateau_reads_the_last_quarter():
    def log(values):
        return pd.DataFrame({"step": range(1, len(values) + 1), "val_mse": values})
    assert train.plateau(log([1.0, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.399]))
    assert not train.plateau(log([1.0, 0.8, 0.6, 0.5, 0.45, 0.4, 0.36, 0.33]))
    assert not train.plateau(log([1.0, 0.5, 0.4]))
```

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_lrn_train.py -q -W error`
Expected: an import error for `train`.

- [ ] **Step 3: Implement `train.py` and the sub-command**

- [ ] **Step 4: Run the tests**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_lrn_train.py tests/ignite/mast/test_lrn_command.py -q -W error`
Expected: 4 passed for this task's file (the training test takes under a minute on the CPU), and the command file as before. The bound `0.5 * variance` was set by reasoning (the two directions hold all but 0.1% of the variance), not by a run: if a correct implementation misses it, report the measured value.

- [ ] **Step 5: Trial on the Task 3 trial root (CPU, 5 to 10 minutes)**

```bash
export IGNITE_MAST_ROOT=/models/intermediate/mast/trials/stage1c-task3
P='env CUDA_VISIBLE_DEVICES= scripts/mast/py.sh'
$P scripts/mast/train_tokenizer.py train --arm net.mag --seed 0 --cpu --steps 200 --variant trial; echo "exit $?"
$P scripts/mast/run_tokenizer.py recode --name t-net --from v1 --arm net.mag.s0.trial
$P scripts/mast/run_tokenizer.py compare --a t-net --b v1 --label val_select-n2 --workers 4 --cache-dir /models/intermediate/mast/cache/v1
```

Expected: the last line of `train` names `net.mag.s0.trial` with a finite `val_mse` under 1 (example units have about unit variance), exit 0 or 3; `compare` prints the four magnetics signals with finite means. After 200 steps the network is expected to lose to the cosine transform; the trial checks the path, not the result. If the trial root is gone, run Task 3's trial again first.

- [ ] **Step 6: Commit**

```bash
git add src/tokamak_foundation_model/ignite/mast/tokenizers/learned/train.py scripts/mast/train_tokenizer.py \
        tests/ignite/mast/test_lrn_train.py
git commit -m "ignite.mast: a network recoder is trained on the stored examples, keeps its best weights and is checked for collapse"
```

---

### Task 7: what the trend path leaves out of the 50 kHz channels

No tokenizer is built here (open decision 4). The task measures the band above the trend, so that the stage-2 plan can decide on fluctuation tokens for these channels with a number. Nothing in stage 1 is decided by it.

**Files:**
- Create: `src/tokamak_foundation_model/ignite/mast/tokenizers/learned/fluct.py`
- Modify: `scripts/mast/train_tokenizer.py` (sub-command `fluct`)
- Test: `tests/ignite/mast/test_lrn_fluct.py`

**Interfaces:**
- Consumes: 1b's `codecs.dct_values(x: np.ndarray, k: int) -> np.ndarray` (`x (..., n)`; for a frame without NaN the first `k` coefficients of `scipy.fft.dct(x, type=2, norm="ortho")` divided by `sqrt(n)`, so the mean square of the frame is the sum of the squared values), `frames.frame_len(dt) -> int`, `Codec.domain_from_model(z)`, `assign.load_assignment`; Task 1's `store.front_codec`; `cache.read_shot`, `cache.pick_evenly`, `tasks.split_shots`, `transform.map_shots`.
- Produces in `fluct.py` (numpy only):
  - `N_FLUCT_BINS = 25`.
  - `fluct_power(u: np.ndarray, k_trend: int, n_bins: int = N_FLUCT_BINS) -> np.ndarray`: `u (..., n)`, the samples of one frame in the codec's domain, to `(..., n_bins)`: the squares of all `n` values of `dct_values(u, n)`, those under index `k_trend` set to 0, summed in `n_bins` groups of `n // n_bins` neighbours. With 250 samples of 0.02 ms a group is 1 kHz wide. A frame with a NaN gives NaN; `n % n_bins != 0` raises `ValueError`.
  - `frame_persistence(p: np.ndarray) -> float`: `p (F,)`, one number per frame: the correlation of `log10(p[1:])` with `log10(p[:-1])` over the pairs in which both are finite and positive; NaN with fewer than three pairs or without variation.
  - `summarize(per_shot: Iterable[np.ndarray], frame_len: int, k_trend: int) -> dict`: each item `(C, T)` in the codec's domain; whole frames only; a frame with a NaN is left out. Keys: `n_frames` (channel-frames used), `share_by_bin` (`(N_FLUCT_BINS,)`, the band's power per group as a share of its total), `share_below_5khz` (the first five groups), `band_to_trend` (the band's power over the power of the values 1 to `k_trend - 1`; NaN for `k_trend < 2`), `persistence` (the median over shots and channels of `frame_persistence` of the band's total power per frame). Without a frame: `n_frames` 0 and NaN elsewhere.
- Produces in the script: `fluct [--n-shots 100] [--workers 64] [--cache-dir DIR]`: for every series signal with 250 samples per frame (four on cache v1), with `k_trend` the signal's K in `<stage1>/assignments/v1.json` and the values of `cache.pick_evenly(tasks.split_shots("val_select"), n)` through the front codec's `domain_from_model`. Writes `<stage1>/compare/fluct.csv` (`signal, k_trend, n_frames, band_to_trend, share_below_5khz, persistence, bin_00` to `bin_24`) and prints it without the bin columns; a signal without a frame is printed as "no data".

- [ ] **Step 1: Write the failing tests**

```python
"""The band above the trend of a 50 kHz channel, as a statistic (plan 1c)."""

import numpy as np
import pytest

from tokamak_foundation_model.ignite.mast.tokenizers.learned import fluct

I = np.arange(250)


def wave(k, amplitude):
    return amplitude * np.cos(np.pi * (I + 0.5) * k / 250)          # cosine coefficient k of a 5 ms frame


def test_the_band_power_adds_up_to_the_frames_mean_square():
    u = np.random.default_rng(0).standard_normal((3, 250))
    assert np.allclose(fluct.fluct_power(u, 0).sum(-1), (u ** 2).mean(-1))
    u = wave(2, 2.0) + wave(37, 0.5)                                 # a trend and a tone at 3.7 kHz
    p = fluct.fluct_power(u, 8)
    assert p.shape == (25,) and p[3] == pytest.approx(0.125) and np.delete(p, 3).max() < 1e-12
    assert fluct.fluct_power(u, 0)[0] == pytest.approx(2.0)          # without a trend path the trend is in the band
    assert np.isnan(fluct.fluct_power(np.where(I == 7, np.nan, u), 8)).all()
    with pytest.raises(ValueError):
        fluct.fluct_power(np.zeros(240), 8)


def test_persistence_tells_a_slow_level_from_noise():
    rng = np.random.default_rng(1)
    assert fluct.frame_persistence(10.0 ** np.cumsum(0.1 * rng.standard_normal(2000))) > 0.9
    assert abs(fluct.frame_persistence(10.0 ** rng.standard_normal(2000))) < 0.2
    assert np.isnan(fluct.frame_persistence(np.array([1.0, np.nan, 2.0, 0.0])))
    assert np.isnan(fluct.frame_persistence(np.ones(10)))


def test_the_summary_of_white_noise_is_flat_and_skips_frames_with_gaps():
    z = np.random.default_rng(2).standard_normal((2, 2500 + 30))    # ten whole frames per channel
    z[0, :250] = np.nan
    s = fluct.summarize([z], 250, 10)
    assert s["n_frames"] == 19 and s["share_by_bin"].sum() == pytest.approx(1.0) and s["share_by_bin"][0] == 0.0
    assert s["share_below_5khz"] == pytest.approx(4 / 24, abs=0.03)
    assert s["band_to_trend"] == pytest.approx(240 / 9, rel=0.4)
    none = fluct.summarize([np.full((2, 500), np.nan)], 250, 10)
    assert none["n_frames"] == 0 and np.isnan(none["share_below_5khz"]) and np.isnan(none["persistence"])
```

- [ ] **Step 2: Run them to see them fail**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_lrn_fluct.py -q -W error`
Expected: an import error for `fluct`.

- [ ] **Step 3: Implement `fluct.py` and the sub-command**

- [ ] **Step 4: Run the tests**

Run: `CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh -m pytest tests/ignite/mast/test_lrn_fluct.py -q -W error`
Expected: 3 passed.

- [ ] **Step 5: Trial (CPU, about two minutes)**

Run: `IGNITE_MAST_ROOT=/models/intermediate/mast/trials/stage1c-task3 CUDA_VISIBLE_DEVICES="" scripts/mast/py.sh scripts/mast/train_tokenizer.py fluct --n-shots 4 --workers 4 --cache-dir /models/intermediate/mast/cache/v1`
Expected: four lines (saddle coils, two soft X-ray cameras, D-alpha), each with `n_frames` above 0, a share between 0 and 1 and a persistence between -1 and 1, or "no data" for a camera that the four shots lack. If the trial root is gone, run Task 3's trial again first.

- [ ] **Step 6: Commit**

```bash
git add src/tokamak_foundation_model/ignite/mast/tokenizers/learned/fluct.py scripts/mast/train_tokenizer.py \
        tests/ignite/mast/test_lrn_fluct.py
git commit -m "ignite.mast: the band above the trend of the 50 kHz channels is measured as a statistic"
```

---

### Task 8: the results page

**Files:**
- Modify: `docs/models/ignite-mast-stage1.md` (created by plan 1b, Task 11)

**Interfaces:**
- Consumes: this plan's sections "Decisions of the plan's author", "Which number decides what", "Measured facts" and "Controller jobs"; the commands of Tasks 3 to 7.
- Produces: the sections the controller fills after the long jobs.

- [ ] **Step 1: Add these sections after 1b's "Results"** and before its "Limits", in this order:
  1. "Learned tokenizers": what a recoder is and what it sits behind (the three front ends with 16, 32 and 96 values); the two orderings of the values; the identity features, with the sentence that the probes have no position in cache v1.
  2. "The comparison of shared against specialized": the three arms and what each pair measures; the table of the 24 arms (name, signals, per channel or concatenated, linear or network, tokens times values), from `train_tokenizer.py arms`.
  3. "Rules": the nine rules of "Which number decides what", copied, each marked "fixed before the measurement, 2026-11-22".
  4. "Where this departs from the design": the bullets of "Decisions of the plan's author", each with its reason.
  5. "How to run": the commands of "Controller jobs".
  6. "Results of the comparison": five tables with their column heads and no rows: the linear arms (the columns of the comparison file, one row per signal and pair); the networks (the same, with the seeds' gaps and their spread); the trainings (name, steps, best step, `val_mse`, plateau, collapsed, active values, seconds); the decisions (signal, rule, the two numbers, the word, which tokenizer `v2` and `v2c` hold); the 50 kHz band (the columns of `fluct.csv`). Under each: `Not measured yet.`
  And add to 1b's "Limits": the loss counts channels equally while TokaMark weights them by variance; the Mirnov comparison is on power spectra and says nothing about the phase between probes; no magnetics signal is a benchmark target, so its tokenizer is judged by reconstruction and by the input round-trip of two tasks only; a learned tokenizer sees only what the front end keeps.

  Every number on the page carries its date, its split, its number of shots and "measured" or "rough look".

- [ ] **Step 2: Check the docs build**

Run: `mkdir -p /models/intermediate/mast/trials/stage1c-docs && uvx --with-requirements requirements-docs.txt --from 'mkdocs<2' mkdocs build -d /models/intermediate/mast/trials/stage1c-docs/site 2>&1 | grep -i "ignite-mast"`
Expected: no line that names `ignite-mast-stage1.md` as a warning or an error. If `uvx` cannot fetch its packages, check by hand that every relative link of the page names an existing file and say so in the report.

- [ ] **Step 3: Commit**

```bash
git add docs/models/ignite-mast-stage1.md
git commit -m "ignite.mast: docs, the stage-1 page gets the learned tokenizers and the rules of the comparison"
```

---

## Controller jobs

The controller runs these after the tasks they depend on and after 1b's jobs K1 to K3 (the fit `fit/v1` and the assignment `v1`), from `/home/nvidia/FusionAIHub`, with `IGNITE_MAST_ROOT` unset. `RT="scripts/mast/py.sh scripts/mast/run_tokenizer.py"`, `TT="scripts/mast/py.sh scripts/mast/train_tokenizer.py"`, `S1=/models/intermediate/mast/stage1`; every job without a GPU is started with `CUDA_VISIBLE_DEVICES=""`. Logs go to `$S1/logs/`. Trainings are started by the controller, never by an implementer. Durations are the author's estimates; none was measured for this code. Unlike the stage-0 trainings, whose speed was set by the loaders, these hold their examples in memory (about 2,000 shots times 100 frames times 163 channels of 16 values: 2 GB for the series family), so several can share one GPU.

| Job | After | Command | GPU | Duration | What is read from the output |
|---|---|---|---|---|---|
| M1 store | Task 3, K3 | `$TT store --workers 64` | none | 10 to 30 min | 24 lines. No signal says "no data"; the skipped shots per signal match 1b's fit. |
| M2 linear arms | M1 | `$TT linear --arm A` for the 11 linear arms, all at once | none | 1 to 5 min each | Per signal the share left at 1, 2, 4, 8, 16 values per stream. It falls from left to right. |
| M3 joint basis | M1, Task 4 | `$TT joint --signals ccbv,obr,obv,flux` | none | minutes | Per size three shares: the joint basis of the 91 channels, each array's own basis, the per-channel basis. This is the upper limit of what coding channels together can give in linear form. |
| M4 recoded assignments and their floors | M2, Task 4 | `$RT recode --name s-klt --from v1 --arm klt.series`; `o-klt` (`klt.mag`); `c-pca` (`pca.ccbv,pca.obr,pca.obv,pca.flux`), `c-pca.h` and `c-pca.q` (the same with `--scale 0.5`, `0.25`); `b-klt` (`klt.band`); `bo-klt` (`klt.band.cc,klt.band.omv`); `bc-pca` (`pca.band.cc,pca.band.omv`), `bc-pca.h`, `bc-pca.q`. Then M5's `compare` once: it prints every `score` command whose floor is missing (`$RT score --task T --split val_select --n-shots 100 --assignment A --mode unquantized`); run those in waves | none | about 20 passes of 3 to 15 min; about 1 h | `recode` prints equal values per frame before and after (a concatenated network arm may be a few values short: tokens of 8). The last line of each pass. |
| M5 linear comparison | M4 | `$RT compare --a A --b B --drop-channel --reach 0.0176` for the pairs: `s-klt:v1` (rule 4); `s-klt:o-klt`, `c-pca:o-klt`, `c-pca:s-klt`, `c-pca.h:s-klt`, `c-pca.q:s-klt`, `c-pca:v1` (rule 6, magnetics); `b-klt:v1` (rule 5); `b-klt:bo-klt`, `bc-pca:bo-klt`, `bc-pca:b-klt`, `bc-pca.h:b-klt`, `bc-pca.q:b-klt` (rule 6, Mirnov) | none | minutes each | The 13 tables. They go to the owner the day they exist, with M3's line: this is the linear answer to shared against specialized, and the networks cannot change what the data holds, only how much of it a tokenizer of that size gets. |
| M6 networks, seed 0 | Task 6, M1 | `$TT train --arm A --seed 0 --gpu N --share-gpu` for the 13 network arms, three or four per GPU | 4, shared | 5 to 20 min each; under 1 h | The last line of each run. `collapsed yes` (exit 3): that arm takes part in no comparison. `plateau no`: run it once more with `--steps 40000 --variant long` (rule 8). |
| M7 network comparison | M6 | `recode` to `s-net0` (`net.series.s0`), `o-net0` (`net.mag.s0`), `c-net0` (the four `net.<array>.s0`), `b-net0` (`net.band.s0`), `bo-net0` (`net.band.cc.s0,net.band.omv.s0`), `bc-net0` (`net.bandcat.cc.s0,net.bandcat.omv.s0`), `p-net0` (`net.profile.s0`), `m-net0` (`net.map.s0`); the missing floors as in M4, with `--gpu N --share-gpu`; then `compare` for the pairs of M5 with the network in place of the linear arm, and `s-net0:s-klt`, `b-net0:b-klt`, `p-net0:v1`, `m-net0:v1` (rule 5) | 4, shared | about 12 passes of 5 to 20 min; about 1 h | The tables, as in M5. |
| M8 rule 8 | M7 | For a network that lost: `$TT train --arm A --seed 0 --lr 1e-4 --variant lr1e4 --gpu N --share-gpu`, then its comparison again. For a network that won: seeds 1 and 2, their assignments and floors, then `$RT seeds --pairs a0:b,a1:b,a2:b --label val_select-n100` | 4, shared | up to 26 runs and their floors; 1 to 2 h | The gaps of the three seeds, their spread and the word with that spread. |
| M9 `v2`, `v2c` and their confirmation | M8 | `recode --name v2 --from v1 --arm <the winners by rules 4 and 5> --signals <their signals>`; `v2c` from `v2` with the concatenated winners of rule 6. For every task with a changed output signal: 1b's K5 passes on `val_report` in the three read-outs; `compare --label val_report` for the decided pairs; for tasks 1-1 and 1-2 `$RT score --task T --split val_report --instrument inputs --assignment A --mode unquantized --variant V --gpu N` with `v2` and `v2c`; `$RT gate --assignment $S1/assignments/v2.json` | 4 passes with a GPU | 1 to 2 h without a GPU, 4 passes of 15 to 45 min with one, the gate 5 to 15 min | A choice made on `val_select` stands only if `val_report` gives the same word. The gate's last line. If `v2` equals `v1`, nothing is scored again and the page says so. |
| M10 the 50 kHz band | Task 7, K3 | `$TT fluct --workers 64` | none | 5 to 15 min | Four lines; they go to the stage-2 plan. |

The variant `V` of a direct model is the one the headroom table shows for the task (1b says how it is chosen). As of 2026-11-22 the direct model of task 1-1 is behind the best published score (0.1361 on `val_report` against 0.0941; 0.1235 with three times the steps) and that of task 1-2 is ahead with learning rate 1e-4 (0.0362 against 0.0412): the input round-trip of rule 6 is therefore read mainly on task 1-2, and its number for task 1-1 is reported with that limit.

After M9 the controller fills the tables of the page, sets its status line, and commits (`ignite.mast: stage-1 learned tokenizers measured`), with the numbers, their date, split and shot counts in the body. Stage 2 does not wait for any of this: it starts on `v1`.

## What this plan leaves to later plans

- Fluctuation tokens for the 50 kHz channels: the stage-2 plan, as an ablation on tasks 3-2, 4-1 and 4-2 (open decision 4), with M10's numbers.
- The phase between Mirnov probes: needs raw waveforms, which cache v1 does not hold.
- Probe and chord positions in the registry (open decision 3): the next cache version; the shared tokenizers are retrained then.
- The paired stage-2 pretraining on `v2` and `v2c` (open decision 1).
- A control over all 91 magnetics channels as a codec: here only M3's number.
- A learned tokenizer on the raw samples of a frame instead of the front values: only if a signal misses its target at the top rung and M10 or 1b's ladder says the missing part matters.
- A specialized control for the profiles and the flux map: each of them is already one tokenizer per signal.

