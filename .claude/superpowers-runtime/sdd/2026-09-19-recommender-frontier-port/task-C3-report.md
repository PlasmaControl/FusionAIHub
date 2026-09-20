# Task C3 report: loader reads `actuator_dim`; G-ENC gate against the v4 cache (GPU)

Commits `66439f1` and `2cb1065` on `nathan_dev` (parent `fcd03b8`). Not pushed.

## What was implemented

### Step 0a/0b: `frame_codes_dirs` production-cache-first

- `src/shot_design/shotdb/build.py::frame_codes_dirs(paths)` now returns, in order:
  `(Path(production), data_root/"frame_codes", bundle_dir(paths)/"frame_codes")`, with
  `production = ignite.model_cfg().get("frame_codes_cache")` and the first entry left out
  entirely (never an empty string) when the pinned generation has no `frame_codes_cache` key.
  The old v2 entry `<models_dir>/IGNITE/frame_codes` is gone (C2's report flagged this exact
  defect as out-of-scope follow-up #2; this task is that follow-up).
  Import of `ignite` stays lazy (`from . import ignite` inside the function), matching the
  file's existing convention (`build.py:1074, 1293`) and confirmed no circular import (`build.py`
  imports only `config`, `schema`, `features`, `legacy_raw`, `text`, `reader`).
- `src/shot_design/design/program_reference.py::_cache_path` now iterates
  `build.frame_codes_dirs(paths)` instead of keeping its own `roots` list, so `build` (the
  `has_frame_codes` column), `corpus select` and `_cache_path` read one function and cannot
  disagree about the same shot. `program_reference` imports `build` at module level (`build`
  does not import `design`, confirmed by grep); the direct `model_cfg`/`bundle_dir` imports
  from `shotdb.ignite` were removed since nothing else in the file used them.

### Step 0b-ii: hermeticity fixture

- `tests/shot_design/conftest.py` gained an autouse fixture,
  `_no_production_frame_codes_cache`, that patches `ignite.model_cfg` to strip
  `frame_codes_cache` for every test by default. A test that wants the production cache
  patches `ignite.model_cfg` again afterwards (its own `monkeypatch.setattr` runs later and
  wins) — `tests/shot_design/test_ignite_v4.py`'s `prod_cache` fixture does exactly this and
  still passes.
- The pin test itself (`assert "frame_codes_cache" not in ignite.model_cfg()`) is NOT in
  `conftest.py`: pytest's default `python_files` collection pattern does not collect
  `conftest.py`, confirmed with `pytest --collect-only`. It lives in the new
  `tests/shot_design/test_build_frame_codes_dirs.py` instead, alongside the two Step 0a tests.
- `test_ignite_v4.py`'s `prod_cache` fixture simplified: it used to patch both
  `ignite.model_cfg` and `program_reference.model_cfg` (two modules, two imports of the same
  function). Now that `_cache_path` reads `build.frame_codes_dirs` — which does the lazy
  `from . import ignite` itself — only `ignite.model_cfg` needs patching.

### Step 0c: `FULL_SHOT_FRAMES` 239 -> 219

- `scripts/shot_design/g_enc.py`: `FULL_SHOT_FRAMES = 219` (was 239), documented as the
  FALLBACK when a cache's own `n_frames` cannot be read. `main`'s v4 flow computes
  `expect_frames = int(ref["n_frames"])` when present, else the constant.
- `tests/shot_design/test_seed.py`'s hardcoded-239 test renamed to
  `test_g_enc_compare_refuses_a_short_shot_against_a_full_shots_frame_count` and its
  `match="239"` changed to `match=str(g_enc.FULL_SHOT_FRAMES)`, decoupling it from the literal.

### Step 1/2: `load_model` actuator_dim — already fixed, test added

- `tests/ignite/test_load_model_actuator_dim.py` (new): builds an 88-channel
  `MaskGITDynamics` checkpoint with no `cfg_actuator_dim` key and asserts
  `eval_dynamics.load_model` infers `actuator_dim=88` from
  `backbone.act_embed.weight.shape[1]`. **PASSED as written** — Task C0's merge already
  carries this inference (confirmed by reading `eval_dynamics.py:146-181`); Step 3 was
  correctly skipped per the brief's amendment.
- One correction to the brief's literal snippet: `tuple(x) for x in mods` raises
  `TypeError: 'ModalitySpec' object is not iterable` (`ModalitySpec` is a `@dataclass`, not
  a NamedTuple); used `dataclasses.astuple(x)` instead.

### Step 4: G-ENC rewrite and the v4 run

`scripts/shot_design/g_enc.py`'s `main()` was rewritten for a v4 default flow while leaving
the historical v2 gate's functions (`compare`, `verdict`, `is_diagnostic`, `DEFAULT_SHOTS`,
`ACT_TOL`/`ACT_MIN_PASS`) byte-identical, because `tests/shot_design/test_seed.py` pins their
exact (STRICT, bit-identical) behaviour and the v4 bundle ships no `frame_codes/` of its own to
compare against with that mechanism (confirmed empty via `ls` on
`/lustre/orion/fus187/proj-shared/nchen/shot_design/models/IGNITE_v4/`, codecs + dynamics
checkpoint only). New, additive pieces:

- `--cache-dir` (default `ignite.model_cfg().get("frame_codes_cache")`) and `--shots` (default
  `DEFAULT_V4_SHOTS = (190000, 190090, 204346, 190735, 190736)` — verified present in the
  production cache on 2026-09-19; the brief's third shot, 199597, is confirmed absent from that
  cache and was dropped, same finding C2's report already recorded).
- `fraction_verdict(result, families)`: per-modality exact-match-fraction pass bar —
  `FRACTION_THRESHOLD_STRICT = 1.0` for the eight slow/fast time-series modalities,
  `FRACTION_THRESHOLD_LOOSE = 0.99` for the five spectro + two video ones — replacing
  `verdict`'s all-bit-identical rule for the default run only. This is the criterion the design
  spec (`.claude/superpowers/specs/2026-09-19-recommender-frontier-port-design.md` §3) sets for
  the v4 gate. A modality requested and not produced is still an unconditional failure, same as
  `verdict`.
- `encode_one` decoupled from `argparse.Namespace` (explicit keyword args) so it is called with
  per-shot `expect_frames` rather than the old fixed constant.
- Report written to `--out` (default `<data_root>/gates/g_enc_v4.json`), gate name `"G-ENC-v4"`.
- `scripts/slurm_frontier/shot_design_genc.sh` (new): `-p batch -q debug -t 01:00:00
  --gres=gpu:1`, modeled on `shot_design_simulate.sh`'s header and `_shot_design_common.sh`
  sourcing convention; body is exactly the brief's
  `srun "$PY" scripts/shot_design/g_enc.py --device cuda --out "$ROOT/runs/genc_v4.json"`.

## Two bugs found and fixed while actually running Step 4 on Frontier

Both were pre-existing, both blocked the gate from producing a real result, neither is
speculative — each was hit by a real `sbatch`/`srun` run and reproduced/fixed with a test.

**1. `sbatch` copies the script; `dirname "$0"` cannot find `_shot_design_common.sh`.**
Job 5514036 failed in 2s: `/var/spool/slurmd/job5514036/slurm_script: line 16:
/var/spool/slurmd/job5514036/_shot_design_common.sh: No such file or directory`. `sbatch`
always runs a copy of the submitted script from a spool directory, so `$0` is that spool path,
not the repo — universal SLURM behaviour, not cluster-specific. This affects every existing
`shot_design_*.sh` wrapper that sources `_shot_design_common.sh` via `dirname "$0"`
(`shot_design_build.sh`, `shot_design_encode.sh`, `shot_design_census.sh`,
`shot_design_simulate.sh`) — none has ever appeared in `sacct` history, so this had never
actually been exercised via real `sbatch` before (only `sbatch --test-only`, a resource/syntax
check that never executes the script body). Fixed **only in my own new
`shot_design_genc.sh`**: it now sources via `$SLURM_SUBMIT_DIR` (the directory `sbatch` was
invoked from) when set, falling back to `dirname "$0"` for a local, non-sbatch run — the same
pattern `eval_dynamics.sh` already uses (`PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"`). The other
four wrappers were left untouched (out of this task's file list); flagged below as a concern
since it blocks Task F2's real encode jobs identically.

**2. `ignite.filled_channels` crashes on the `text_embed` group.**
After the sourcing fix, job 5514051 reached real Python and crashed on shot 190000:
`KeyError: "Unable to synchronously open object (object 'xdata' doesn't exist)"`, inside
`filled_channels`'s `for name in f` / `f[name]["xdata"]`. Inspecting
`/lustre/orion/fus187/proj-shared/foundation_model/190000_processed.h5` directly showed every
top-level group has `xdata`/`ydata` **except** `text_embed` (`input`/`total` — the
text-embedding campaign's output, injected into the corpus since 2026-09, per this session's
own memory of that campaign closing). `filled_channels` iterated every top-level group
unconditionally and had no prior test covering a group without `xdata`.
Fix in `src/shot_design/shotdb/ignite.py`: `if "xdata" in f[name] and
f[name]["xdata"].shape[0] > 1` — skip a group with no `xdata` rather than erroring, since it is
not a signal `filled_channels` has an opinion about. New regression test
`tests/shot_design/test_ignite.py::test_filled_channels_skips_a_group_with_no_xdata` builds a
tiny H5 with one real modality group and a `text_embed`-shaped group and asserts the latter is
ignored. **Confirmed RED before the fix** (`git stash` the fix, same `KeyError`), **GREEN
after**. This is a real, general corpus-schema bug independent of Task C3 — every shot in the
corpus now has a `text_embed` group, so this would have broken `ignite.frame_codes` /
`design.seed.encode_frame_codes` for ANY caller (G-ENC, the F-phase encode jobs, `shot_design
add`), not just this gate.

## TDD evidence

RED (Step 0a, before `frame_codes_dirs`/`_cache_path` change):
```
FAILED tests/shot_design/test_build_frame_codes_dirs.py::test_frame_codes_dirs_production_cache_first_then_v4_bundle
FAILED tests/shot_design/test_build_frame_codes_dirs.py::test_frame_codes_dirs_without_a_production_cache
AssertionError: assert PosixPath('/prod/frame_codes') == PosixPath('/tmp/.../frame_codes')
```
GREEN after Step 0b: both pass; hermeticity test
(`test_tests_never_see_the_production_cache`) passes once the `conftest.py` fixture landed.

RED (`test_load_model_actuator_dim.py`, first attempt): `TypeError: 'ModalitySpec' object is
not iterable` from the brief's `tuple(x)`; fixed with `dataclasses.astuple(x)`, then GREEN —
the underlying `load_model` behaviour needed no change (Task C0 already there).

RED (`test_ignite.py::test_filled_channels_skips_a_group_with_no_xdata`, `ignite.py` stashed
back to pre-fix): `KeyError: "Unable to synchronously open object (object 'xdata' doesn't
exist)"`. GREEN after the one-line guard.

Full required suite, once, before committing:
```
pixi run --frozen -e shot-design-frontier pytest tests/shot_design tests/ignite -q \
    --ignore=tests/ignite/test_train_codec.py
1 failed, 2263 passed, 25 skipped, 4 warnings in 380.21s (0:06:20)
FAILED tests/shot_design/test_mcp.py::test_the_project_mcp_config_points_at_this_server
```
That failure is the same pre-existing, unrelated one C2's report already confirmed
(`.mcp.json` names a Stellar checkout path; `git -C` on it exits 128 here).

## G-ENC-v4 result: PASS (job 5514064, 4m09s, MI250X)

Two earlier submissions failed before reaching the gate logic at all (the two bugs above,
jobs 5514036 and 5514051); job 5514064 (after both fixes) ran the real gate:

```
device cuda, torch 2.10.0+rocm7.1, gpu AMD Instinct MI250X
cache_dir /lustre/orion/fus187/proj-shared/models/ignite_prod_v4/frame_codes
G-ENC PASS -> /lustre/orion/fus187/proj-shared/nchen/shot_design/runs/genc_v4.json
```

Per-shot, per-modality (agreement %, bit-identical column from the report):

| modality | 190000 | 190090 | 204346 | 190735 | 190736 |
|---|---|---|---|---|---|
| ece | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| bes | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| mhr | 99.998 | 100.00 | 100.00 | 100.00 | 100.00 |
| co2 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| **mirnov** | **100.00** | **99.998** | **100.00** | **99.998** | **100.00** |
| tangtv_lower | 99.763 | 99.784 | 100.00 | 100.00 | 100.00 |
| tangtv_upper | 100.00 | 99.264 | 100.00 | 99.526 | 99.797 |
| ts_core_density | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| ts_core_temp | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| ts_tangential_density | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| ts_tangential_temp | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| cer_ti | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| cer_rot | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| mse | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| filterscopes | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| actuators (88 ch) | 88/88 | 88/88 | 88/88 | 88/88 | 88/88 |

`mirnov` — the modality C2 newly added to the encode path — clears its 99% loose bar with room
to spare: the two misses are one mismatched token each (of ~42000+), the same
bin-boundary/cross-vendor scatter the module docstring already documents for `mhr`/`co2`. Every
video modality (`tangtv_lower`/`upper`) also clears 99% on every shot; the worst is
`tangtv_upper` on 190090 at 99.264%. No modality on any shot came within reach of a threshold
miss, so no fraction needed recording as a failure per the brief's conditional instruction, and
`docs/models/ignite.md` was not touched (that path does not exist in this repo yet; it is
Task H2's file to create, per the brief itself).

All 5 shots: `n_frames: 219` read from each cache's own key (Step 0c in effect), actuators
88/88 bit-identical, `max_delta_z: 0.0` on all five. `report["passed"] = true`,
`failed_shots: []`.

## Files changed

Commit `66439f1`:
- `src/shot_design/shotdb/build.py`, `src/shot_design/design/program_reference.py`
- `tests/shot_design/test_build_frame_codes_dirs.py` (new)
- `tests/shot_design/conftest.py`, `tests/shot_design/test_ignite_v4.py` (both required by the
  hermeticity fixture / the `_cache_path` refactor; not in the brief's literal file list but
  direct, necessary consequences of Step 0b/0b-ii)

Commit `2cb1065`:
- `scripts/shot_design/g_enc.py`, `scripts/slurm_frontier/shot_design_genc.sh` (new)
- `tests/ignite/test_load_model_actuator_dim.py` (new)
- `tests/shot_design/test_seed.py` (Step 0c's renamed/decoupled test)
- `src/shot_design/shotdb/ignite.py`, `tests/shot_design/test_ignite.py` (the `filled_channels`
  bug found and fixed while actually running Step 4 — not in the brief's file list, but
  required to get a real G-ENC result at all)

`src/tokamak_foundation_model/ignite/eval_dynamics.py` needed no change (Step 2/3), so it was
not touched or added to either commit. The pre-existing uncommitted
`.claude/superpowers/plans/2026-09-19-recommender-frontier-port.md` was never staged.

## Self-review of the diff

- `frame_codes_dirs`'s new production-cache entry is `Path(production)` only when the yaml key
  is truthy; a generation without the key (hypothetically a future v5 with no production cache
  yet) degrades to exactly the two-entry tuple the tests for that case pin.
- `_cache_path` no longer has its own notion of "the two writable roots" — checked there is no
  other caller of the removed `roots`-building logic (grep for `frame_codes_dirs` and
  `_cache_path` turned up only `build.py`, `program_reference.py` and the two test files).
- The `fraction_verdict` per-family threshold reads `families = ignite.model_cfg()["families"]`
  once per `main()` call, not per shot — correct, since the family table does not vary by shot.
- `expect_frames` per shot: verified against a cache lacking `n_frames` (impossible for a
  `validate_cache`-passing cache, since that check runs inside `compare` before frames are used)
  falls back to the constant rather than raising early — matches the brief's Step 0c wording.
- Ran the line-length scan (`awk` over `git diff | grep '^\+'`, and a separate raw scan for the
  three new untracked files, since `git diff` does not show untracked content) to zero
  violations on every line this task added; pre-existing lines >88 in `g_enc.py`'s historical
  v2 docstring section were deliberately left untouched (they predate this task and the rule
  is "every new line").
- Confirmed no other caller of `ignite.filled_channels` besides `_resolve_input` (single call
  site), so the added guard cannot change behaviour anywhere else.

## Concerns / follow-ups (nothing blocking; G-ENC-v4 PASSED)

1. **The `dirname "$0"` / sbatch-spool bug is real and un-fixed in four pre-existing wrappers**
   (`shot_design_build.sh`, `shot_design_encode.sh`, `shot_design_census.sh`,
   `shot_design_simulate.sh`). None has ever run via real `sbatch` (`sacct` history is empty for
   all four); only `sbatch --test-only` has exercised them, which never invokes the script body.
   Task F2's 8-GCD encode job uses `shot_design_encode.sh` and will hit the identical failure the
   moment it is submitted for real. Recommend fixing all four with the same
   `${SLURM_SUBMIT_DIR:-...}` pattern before Task F2 runs, or at minimum flagging it explicitly
   in that task's brief.
2. **`filled_channels`'s `text_embed` gap likely affects other unaudited callers.** This task
   fixed the one call site it hit (`_resolve_input` -> G-ENC), but did not audit whether other
   code assumes every top-level HDF5 group has `xdata` the way `filled_channels` did (e.g.
   dataset-length probing in `data_loader.py` or the F-phase batch encoder) — worth a
   `grep -rn 'for name in f\b'`-style sweep if `text_embed` groups are new to more of the
   pipeline than this one function.
3. **`docs/models/ignite.md` does not exist in this repo yet.** The brief's Step 4 note assumed
   it as a place to record a spectro miss; since G-ENC-v4 passed outright there was nothing to
   record, but Task H2 (which the brief itself names as that file's owner) should be aware no
   such file exists to extend.
4. Ruff itself (`ruff check`) is not installed in either the `shot-design-frontier` pixi
   environment or the default one on this login node (`ruff: command not found` from both);
   line-length compliance was verified manually per the binding rule, not cross-checked with the
   tool.
