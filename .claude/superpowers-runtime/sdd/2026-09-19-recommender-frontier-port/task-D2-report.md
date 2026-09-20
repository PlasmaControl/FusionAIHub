# Task D2 report: `shot_design.simulate.decode` + `shot_design.simulate.report`

## What was implemented

Two new modules under `src/shot_design/simulate/`:

- **`decode.py`** (151 lines): `decode_modalities(codecs, arms, names)` decodes a
  `SimulationArms`' `real`/`proposed`/`gt` token arrays through each modality's frozen
  Phase-A codec (`eval_dynamics.decode_flat_chunked`) and reduces the family-specific
  decoded shape down to `(F, C)` (video: `(F, 1)`) for the panel/report:
  - `spectro` -> `band_power(dec, freq_axis_khz(cfg), band)`: mean `|value|` over a
    frequency sub-band. `mhr`/`mirnov` get the brief's 10-60 kHz band; every other
    spectro modality gets the full band (no named band exists for it in the brief).
  - `video` -> `_reduce_video`: per-frame mean over channel/space/time -> `(F, 1)`.
  - `slowts`/`fastts` -> `_reduce_series`: mean over the intra-frame time axis -> `(F, C)`.
  - A name absent from `codecs` is skipped, matching `eval_dynamics.decode_all`'s own
    convention (no frozen codec for it: placeholder or simply not requested).
  - The device to decode on is inferred from `arms`' own tensors (`_device_of`), since the
    brief's signature (`decode_modalities(codecs, arms, names)`) has no device parameter.

- **`report.py`** (175 lines): `write(out_dir, arms, decoded, meta, actuators=None)` writes
  `simulation.h5`, `panels/<m>.png`, and `report.md`, plus `frac_static(proposed,
  seed_frames)` (controller ruling 1).

## How the spectro frequency axis was derived

Fully derivable from the codec's own config, no fallback needed for the default/native
per-codec geometry:

- `ignite.config.STFT_FS = 500_000.0` Hz is the spectro STFT's fixed sampling rate
  (`ignite/config.py:46`), independent of `n_fft` (the module docstring at
  `config.py:116` states the covered band is "UNCHANGED at 0-250 kHz... Nyquist is set
  by STFT_FS, not n_fft").
- `ignite/data.py`'s `log_power_stft` computes the STFT at `cfg.stft_n_fft`, drops the DC
  bin (bin 0), then `_crop_pad_freq_time` keeps the **low** `cfg.freq_bins` of the
  remaining bins unchanged ("cropped from the low end... bins are already DC-dropped and
  ordered low->high", `data.py:74-75`).
- So decoded array index `i` (0-indexed) is raw STFT bin `i + 1`, at frequency
  `(i + 1) * STFT_FS / cfg.stft_n_fft`. `decode.freq_axis_khz(cfg)` implements exactly
  this and is unit-tested against the arithmetic directly
  (`test_freq_axis_khz_matches_stft_bin_math`).
- **Fallback case**: `SpectroCodecConfig.band_pool > 0` mean-pools the STFT bins into
  `band_pool` equal bands as the LAST step of `log_power_stft` (`config.py` comment on
  `band_pool`), which breaks the 1:1 index<->bin mapping in a way not recoverable from
  `cfg` alone (which pooled bins land in the codec's kept range depends on
  `eff_freq_bins`/pooling internals not exposed on `cfg`). `freq_axis_khz` returns `None`
  in that case (tested: `test_freq_axis_khz_none_when_band_pooled`), and `band_power`
  falls back to a full-band mean, documented in both docstrings and in the h5
  `reduction` attr.

This satisfies the brief/ruling: the axis IS recoverable from the codec cfg for the
non-pooled (default) case, which is what production `mhr`/`mirnov` codecs use per the
manifest inspected during D1/D2 (`band_pool: 0` in every codec entry I could find
referenced in `config.py`'s defaults) — the pooled-axis fallback path exists for
robustness but was not exercised against a real pooled checkpoint (none was available
to inspect in this environment).

## The video reduction

`decode._reduce_video` collapses a decoded video array `(F, C, Tv, H, W)` to `(F, 1)` via
a single `.reshape(F, -1).mean(axis=1, keepdims=True)` — a per-frame scalar mean over
every non-frame axis (channel, intra-frame time, height, width). This is disclosed in two
places per ruling 2: the `decode.py` module docstring and function docstring, and
`report.write`'s h5-level `reduction` attribute (see "h5 layout" note below for why it's
file-level rather than per-group).

## Tests and results

Two test files, split as: `tests/shot_design/test_simulate_report.py` (report.write, h5
groups/attrs, panels, markdown table, the qualitative sentence, `frac_static`) and
`tests/shot_design/test_simulate_decode.py` (`freq_axis_khz`, `band_power` pure-numpy
reduction, and a decode-through-a-real-codec smoke test on a **tiny untrained
SpectroCodec** — see below).

**RED** (both modules absent):

```
$ PIXI_CACHE_DIR=/tmp/pixi-cache-nchen pixi run --frozen -e shot-design-frontier \
    pytest tests/shot_design/test_simulate_report.py tests/shot_design/test_simulate_decode.py -q -p no:cacheprovider
...
ImportError: cannot import name 'report' from 'shot_design.simulate'
ImportError: cannot import name 'decode' from 'shot_design.simulate'
Interrupted: 2 errors during collection
```
Expected: neither `decode.py` nor `report.py` existed yet.

**GREEN** (after implementing both modules):

```
$ PIXI_CACHE_DIR=/tmp/pixi-cache-nchen pixi run --frozen -e shot-design-frontier \
    pytest tests/shot_design/test_simulate_report.py tests/shot_design/test_simulate_decode.py -q -p no:cacheprovider
............                                                             [100%]
12 passed, 1 warning in 6.03s
```

**Full suite once, before commit:**

```
$ PIXI_CACHE_DIR=/tmp/pixi-cache-nchen pixi run --frozen -e shot-design-frontier \
    pytest tests/shot_design -q -p no:cacheprovider
FAILED tests/shot_design/test_mcp.py::test_the_project_mcp_config_points_at_this_server
1 failed, 1686 passed, 20 skipped, 2 warnings in 122.48s
```
The one failure is the known pre-existing failure named in the task instructions
(stale `/scratch/gpfs/nc1514` path check, unrelated to this change) — confirmed it is
not something D2 introduced (it fails on `git rev-parse` against a Stellar path that
does not exist on this Frontier checkout).

### On ruling 3's "tiny synthetic codec" decode test

Built one: a `SpectroCodecConfig(channels=1, freq_bins=16, time_frames=4, patch_f=4,
patch_t=4, d_model=64, enc_depth=1, dec_depth=1, heads=1)` -> `SpectroCodec(cfg).eval()`,
entirely untrained (random init), decoding random FSQ code indices through
`eval_dynamics.decode_flat_chunked` on CPU. This is genuinely cheap (~1s, no checkpoint,
no GPU) and is exercised end-to-end through `decode.decode_modalities` in
`test_decode_modalities_reduces_spectro_to_F_C` (covering both the `mhr` banded path and
the `co2` full-band fallback from the same codec, aliased under two names) and
`test_decode_modalities_skips_names_without_a_codec`.

One wrinkle worth recording: `d_model` could not be arbitrarily small — `d_model=8`
crashed inside `x_transformers`' `Encoder` (a residual-dimension mismatch, 64 vs 8) that
traced to internal defaults in that library, not anything in this repo's code.
`d_model=64` works cleanly, so that's what the tiny codec uses; I did not chase the root
cause further since a working tiny codec was the actual goal.

## Controller-ruling decisions and their exact effect

1. **`frac_static`** — implemented in `report.py`, computed from `arms.proposed` only,
   over the predicted region (`codes[seed_frames:]` vs `codes[seed_frames-1:-1]`).
   Guarded for `seed_frames <= 0` or `seed_frames >= F` (returns `nan` for that
   modality rather than raising or slicing incorrectly). Unit-tested for both the
   all-constant (1.0) and every-frame-different (0.0) cases exactly as the ruling asked.

2. **`report.write` gets `actuators: dict[str, np.ndarray] | None = None`** as a 5th,
   keyword-only-by-convention parameter (Python doesn't enforce keyword-only here since
   the brief's four positional parameters had to stay unchanged and adding a 5th
   positional-or-keyword param after them is the minimal-diff way to do that; callers
   passing it positionally would still work, but D3 is expected to pass it as
   `actuators=...`). When given, writes `actuators/real` and `actuators/proposed`
   datasets (only the two brief-named keys, ignoring any other keys the dict might
   carry); when `None`, the `actuators` group is omitted entirely. Both branches are
   tested (`test_report_writes_actuators_when_given`,
   `test_report_skips_actuators_group_when_not_given`).

3. **`decode_modalities` tests** — see "Tests and results" above: a pure-numpy
   `band_power` test using an explicit synthetic `(F, C, Fr, Tb)` array and frequency
   axis (I used `(F, C, Fr, Tb)` rather than the ruling's literal `(F, freq, time)`
   phrasing, since `(F, C, Fr, Tb)` is `decode_flat_chunked`'s REAL output shape for a
   spectro codec and testing the actual contract seemed more valuable than a shape that
   doesn't occur anywhere in the pipeline), a `freq_axis_khz` derivation test, and the
   tiny-synthetic-codec decode test described above.

## Ambiguity calls (per "Before You Begin")

- **`report.write`'s return value**: the brief says `-> Path` but doesn't say which path
  among `simulation.h5` / `panels/` / `report.md`. I return `out_dir` (the root all three
  live under) since that's the one piece of information a caller can't otherwise
  reconstruct without knowing the internal layout, and it's the most useful handle for
  D3's CLI to report back to a user ("wrote simulation to <out_dir>"). The brief's own
  test doesn't inspect the return value beyond existence, so this doesn't affect
  correctness either way.
- **Where the `reduction` h5 attr lives**: ruling 2 says "say so in the h5 attrs
  (reduction)" for the video case specifically, but `report.write` receives no per-modality
  family information (only the already-reduced `decoded` dict), so it cannot write a
  video-specific attr per group without new plumbing the brief never asked for. I wrote
  one file-level `f.attrs["reduction"]` documenting all three families' reductions
  (spectro band-power, video per-frame mean, slowts/fastts intra-frame mean) — it
  satisfies the literal ask (an attr named `reduction` disclosing that video is not raw
  frames) without adding an unrequested family-plumbing parameter to `report.write`.
- **Table header text**: the brief's Interfaces line reads "skill (= token_acc -
  persistence_acc)" as a description, not literal header text; the markdown header uses
  the bare column name `skill` (matching the other five column names' style), with the
  formula documented in `report.py`'s module docstring and the `_render_report` code
  itself instead of in the header text.

## Files changed

- `src/shot_design/simulate/decode.py` (new, 151 lines)
- `src/shot_design/simulate/report.py` (new, 175 lines)
- `tests/shot_design/test_simulate_decode.py` (new, 114 lines)
- `tests/shot_design/test_simulate_report.py` (new, 109 lines)

Commit: `9e755e8` — "shot_design.simulate: decode to physical units, panels and honest
report" (exact message from the brief's Step 4). Committed with explicit paths only; the
pre-existing uncommitted edit under `.claude/superpowers/plans/` was never staged
(verified via `git status --short` before and after commit).

## Self-review findings

- **Completeness**: all h5 groups (`tokens/{real,proposed,gt}/<m>`,
  `decoded/<m>/{real,proposed,gt}`, `actuators/{real,proposed}` when given) and attrs
  (all of `meta` plus `reduction`) present; panels drawn per modality present in
  `decoded` with the seed/predict boundary as a vertical dashed line; report table
  columns in the brief's exact order (`modality | frac_static | token_acc |
  persistence_acc | skill | divergence_vs_real`); qualitative sentence emitted iff any
  modality's skill `< 0`, verified both ways by test.
- **Quality**: every function name states what it returns/does truthfully
  (`freq_axis_khz`, `band_power`, `_reduce_video`, `_reduce_series`, `frac_static`); no
  dead code, no stray `print`s; confirmed no unused imports via an AST-based scan (the
  only "unused" hit was `from __future__ import annotations`, which is a no-op import
  by design).
- **Discipline (YAGNI)**: `decode.py` does not attempt video/slowts/fastts codec
  construction in tests (out of scope per the ruling's explicit escape valve) and does
  not add configuration knobs the brief never asked for (e.g. no configurable band for
  non-mhr/mirnov modalities — full band is the only sane default with no named
  alternative). `report.py` does not add extra artifacts beyond the three named.
- **Testing**: every h5/png/md assertion in `test_simulate_report.py` reads back a real
  file written to `tmp_path` (no mocking of h5py/matplotlib); pytest output is pristine
  (12/12 focused, 1686 passed / 20 skipped / 1 pre-existing-unrelated failure on the
  full suite); ran `awk 'length>88'` over all four new files with zero hits.

## Concerns

- The `freq_axis_khz` fallback (`band_pool > 0`) is implemented and unit-tested in
  isolation but was never exercised against a REAL production codec checkpoint with
  `band_pool` set (no such checkpoint was available in this environment to inspect) — if
  a real `mhr`/`mirnov` codec ever ships with `band_pool > 0`, the band-power panel for
  it will silently widen to the full band rather than erroring, which is the documented
  and intended behavior but is worth a human's attention the first time it actually
  triggers on production data.
- `report.write`'s return-value semantics (returns `out_dir`, not the h5 path) is my own
  reading of an underspecified return type; if D3's CLI expects the h5 path specifically,
  that's a one-line change (`return out_dir / "simulation.h5"`) — flagging so it's not
  assumed silently correct.

## Fix round 1 (implementer edits, committed by the controller as 50cbe54)

The implementer completed the edits at ~22:25 EDT on 2026-09-19 but stalled waiting on its
own background suite run and never committed or reported; the controller verified and
committed the working tree unchanged two hours later.

**Important 1 (decode device):** `decode._device_of(codec, arms)` now returns
`next(codec.parameters()).device`, falling back to the arms' tensor device only for a
parameter-free codec. Test `test_decode_modalities_uses_the_codecs_device_not_the_arms_device`
records the `device` handed to `eval_dynamics.decode_flat_chunked` and asserts it equals the
codec's parameter device.

**Important 2 (untested reductions):** `test_reduce_video_returns_per_frame_mean`,
`test_reduce_series_returns_per_frame_channel_mean`,
`test_reduce_series_passes_through_when_already_F_C` with hand-computed expectations.

**Minors:** `core.ARM_LABELS = ("real", "proposed", "gt")` shared by decode.py and report.py
(duplicate tuple removed); `_write_panel` docstring states the channel-mean choice.

**Tests (controller-run):**
`PIXI_CACHE_DIR=/tmp/pixi-cache-nchen pixi run --frozen -e shot-design-frontier pytest tests/shot_design/test_simulate_decode.py tests/shot_design/test_simulate_report.py tests/shot_design/test_simulate_core.py -q -p no:cacheprovider`
→ `25 passed, 1 warning` (the pre-existing import-time `torch.jit.script` DeprecationWarning).
No new line exceeds 88 columns (awk over the four files).
