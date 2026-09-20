# Task D3 report: `shot_design simulate` CLI

## What I implemented

- `src/shot_design/simulate/cli.py` (new): `run(args) -> int`, the orchestration
  described in the brief: `program.load_program` -> `program.export_ignite`
  (design seed, `torch.load`'d) -> `program_reference.reference(...).cache`
  (real arm) -> `core.load_dynamics` -> `core.actuator_arms` -> `core.run_paired`
  -> `shotdb.ignite.load_codecs` + `decode.decode_modalities` ->
  `report.write(..., actuators={"real": real_act, "proposed": prop_act})`.
  `status.json` is written atomically (`tempfile.mkstemp` in the same dir +
  `os.replace`) first thing (`state: running`) and last thing in both the
  success path (`state: complete, report: "report.md"`) and the `except
  Exception` path (`state: failed, error: str(exc)`), which always returns 1.
  `DEFAULT_DECODE = ("filterscopes", "mhr", "mirnov", "ts_core_density",
  "ts_core_temp")` lives here as the single source of truth; the CLI's
  `--decode` has no argparse `default=` so it falls through to this tuple.
- `src/shot_design/cli.py`: added `cmd_simulate(args)` (lazy `from .simulate
  import cli as simulate_cli` inside the function, matching `cmd_model`/
  `cmd_encode`'s existing pattern of keeping heavy submodule imports out of
  module-import time) and the `simulate` subparser (`ident`, `--seed`, `--k0`,
  `--n-predict`, `--decode`, `--device`, `--out`), registered with
  `p.set_defaults(func=cmd_simulate)`.
- `tests/shot_design/test_simulate_cli.py` (new): 11 tests.

## How the reference cache is windowed

`program.export_ignite` -> `_evaluate` (program.py ~L191-388) slices
`ref.cache["codes"]`/`ref.cache["actuators"]` at `[context:display_end]` where
`context = display_start - SEED_FRAMES` and `display_start`/`display_end` come
from `program.start_s`/`program.end_s` rounded to 50 ms frames, THEN clamped to
the reference's own bounds. `_evaluate`'s `can_export` check (feeding
`export_ignite`'s guard) requires `errors` to be empty, and `_window`'s error
conditions are exactly "the window needed clamping" (`start < SEED_FRAMES`,
`end - start` out of `[1, MAX_PREDICTION_FRAMES]`, or window past the
reference's frame count). So whenever `export_ignite` succeeds, the clamps are
provably no-ops and `display_start == round(start_s/FRAME_S)`, `display_end ==
round(end_s/FRAME_S)` hold exactly. `simulate/cli.py`'s
`_windowed_real_actuators` recomputes `context`/`display_end` from
`prog.start_s`/`prog.end_s` (public `act.FRAME_S`, public `program.SEED_FRAMES`)
and slices `ref.cache["actuators"][context:display_end]` to hand `core
.actuator_arms` a real-arm actuator array over the SAME window the design seed
was cut from -- `design_seed["codes"]` itself needs no re-slicing since
`_evaluate` copies `ref.cache["codes"][context:display_end]` unedited into the
exported seed, so it already IS the windowed ground truth `run_paired` scores
against.

I did not call `_evaluate`/`_window` directly (both private); duplicating the
public-input arithmetic that a successful export already proves clamp-free
seemed the minimal, correct alternative to importing a private function. This
is a "read program.py's private logic, derive the equivalent public
computation" adaptation the task explicitly invited ("if slicing the reference
cache... is not derivable from program.py, report NEEDS_CONTEXT") -- I
consider it derivable and did derive it, but flagging the reasoning here since
it is a judgment call rather than a literal quoted interface.

## Where `dynamics_sha256` / `codec_generation` come from

- `dynamics_sha256`: `shotdb_ignite.bundle_identity(paths).get("manifest_sha256")
  or ""`. `bundle_identity`'s own docstring: re-pinning rewrites codecs, the
  dynamics file AND the manifest together, so the manifest's sha256 is "what
  identifies the weights a database was built with" -- there is no separate
  per-file dynamics-only sha256 exposed by any existing public helper (the
  per-file entry lives inside the raw `codecs/MANIFEST.json`'s `sha256` dict,
  keyed by `model_cfg()["dynamics_file"]`, but reading that directly would be
  re-deriving rather than reusing the existing Task-C1 helper the brief points
  at). Falls back to `""` (never `None`, which `h5py.File.attrs` cannot store)
  when no bundle is pinned.
- `codec_generation`: `shotdb_ignite.model_cfg().get("generation", "v2")` --
  matches `bundle_identity`'s own convention for the same field.
- `window_s`: `[prog.start_s, prog.end_s]` -- a 2-element list, matching the
  shape `tests/shot_design/test_simulate_report.py` (D2, already committed)
  uses for this exact meta key (`"window_s": [1.0, 5.0]`); I initially assumed
  a scalar duration and corrected after reading that test.
- `dynamics_step`: the third element `core.load_dynamics` returns (`step`).
- `design_id`: `prog.id` (the loaded `DesignProgram`'s own id, equal to the
  CLI's `ident` argument since `load_program` asserts that).

## Tests and results

`tests/shot_design/test_simulate_cli.py`, 11 tests, all via `cli.main([...])`
(the real entry point) with `program.load_program`, `program.export_ignite`,
`program_reference.reference`, `core.load_dynamics`, `core.run_paired`,
`decode.decode_modalities` monkeypatched per the brief's Step 1, plus (my own
addition, for hermeticity, not required by name in the brief but not forbidden
either) `shotdb.ignite.load_codecs` faked to `{}` since `decode_modalities` is
already faked and does not need real codec weights -- a tiny real
`codecs/MANIFEST.json` (`fake_bundle` fixture) still lets `bundle_identity`
hash real bytes. `report.write` and `core.actuator_arms` are NOT faked, so
`status.json`, `report.md` and `simulation.h5` are real files read back from
`tmp_path`.

- `test_simulate_writes_running_then_complete_status_and_a_report`: mid-run
  snapshot (via the fake `load_program`) shows `state: "running"`; final state
  `complete`, `report.md` + `simulation.h5` exist.
- `test_simulate_leaves_a_failed_status_and_exits_1` (parametrized over each of
  the 6 faked call sites): exit 1, `status["state"] == "failed"`, error text
  contains the fake's message, `report` stays `null`.
- `test_simulate_rejects_k0_plus_n_predict_over_the_seeds_frame_count`: `--k0
  50 --n-predict 80` (130) against a 100-frame seed -> failed status naming
  both numbers.
- `test_simulate_writes_actuators_into_the_h5`: `actuators/real`,
  `actuators/proposed` groups present; `design_id`, `codec_generation`,
  `dynamics_step` attrs correct.
- `test_simulate_default_out_dir_is_data_root_outputs_ident_simulation`,
  `test_simulate_out_flag_overrides_the_default_directory`: default vs `--out`.

Full suite: `PIXI_CACHE_DIR=/tmp/pixi-cache-nchen pixi run --frozen -e
shot-design-frontier pytest tests/shot_design -q -p no:cacheprovider` ->
**1701 passed, 20 skipped, 1 failed** (`test_mcp.py::
test_the_project_mcp_config_points_at_this_server`, the pre-existing failure
named in the task as expected/ignorable; unrelated to this change).

Ruff: the pixi env(s) available (`shot-design-frontier`, `default`) do not
have a `ruff` binary/module installed, so I could not run it directly; I
checked line length manually (`grep -n '.\{89,\}'` and diffing only added
lines against the pre-existing file) on all three changed/new files and fixed
every line I added that exceeded 88 chars. I also manually re-checked for
unused imports (removed one: `simulate.cli` was imported in the test but never
referenced) since I could not run ruff's linter itself.

## TDD evidence

RED: temporarily reverted `src/shot_design/simulate/cli.py` (moved aside) and
`src/shot_design/cli.py` (git checkout) with the new test file already in
place, then ran:

```
PIXI_CACHE_DIR=/tmp/pixi-cache-nchen pixi run --frozen -e shot-design-frontier \
  pytest tests/shot_design/test_simulate_cli.py -q -p no:cacheprovider
```

Failing output (expected reason -- the module and subcommand do not exist yet):

```
ImportError while importing test module '.../tests/shot_design/test_simulate_cli.py'.
...
E   ImportError: cannot import name 'cli' from 'shot_design.simulate'
1 error in 2.88s
```

GREEN: restored both files, reran the same command:

```
...........
11 passed, 1 warning in 5.70s
```

Then re-ran after two follow-up fixes (a test bug where the mid-run status
lookup assumed the default `--out` path, breaking the `--out`-override test;
and line-length/unused-import cleanup) -- final green:

```
11 passed, 1 warning in 5.89s
```

and the full-suite run above.

## Files changed

- `/lustre/orion/fus187/scratch/nchen/FusionAIHub/src/shot_design/simulate/cli.py` (new)
- `/lustre/orion/fus187/scratch/nchen/FusionAIHub/src/shot_design/cli.py` (modified: `cmd_simulate` + `simulate` subparser only)
- `/lustre/orion/fus187/scratch/nchen/FusionAIHub/tests/shot_design/test_simulate_cli.py` (new)

Commit: `b8ea4f4` "shot_design: simulate subcommand with status file" on
`nathan_dev`, containing exactly these three files (verified via `git status
--porcelain` before and after that the unrelated uncommitted
`.claude/superpowers/plans/2026-09-19-recommender-frontier-port.md` edit was
never staged).

## Self-review findings

- Completeness: all six CLI flags present and wired (`ident`, `--seed`, `--k0`,
  `--n-predict`, `--decode`, `--device`, `--out`); status transitions
  running->complete and running->failed both covered; exit codes 0/1 both
  asserted through `cli.main`; default out dir
  `data_root/outputs/<ident>/simulation` matches the brief; `report.write`
  receives `arms`, `decoded`, `meta` (all 5 required keys), and
  `actuators={"real": real_act, "proposed": prop_act}` built from
  `core.actuator_arms`'s own return value (not re-derived).
- Quality/YAGNI: no extra flags or output formats beyond the brief; `run()` is
  a single linear try/except, no premature abstraction; `_windowed_real_actuators`
  is the only helper, isolating the one piece of derived-from-private-code
  arithmetic with its reasoning in the docstring.
- Discipline: `simulate/cli.py` stays at 151 lines (well under ~200);
  `cmd_simulate` in the main `cli.py` keeps the heavy import lazy, consistent
  with `cmd_model`/`cmd_encode`.
- Testing: fakes limited to the brief's six call sites plus my one addition
  (`load_codecs`, justified above); `status.json`/`report.md`/`simulation.h5`
  are real files under `tmp_path`, never mocked; `main()` is the entry point
  under test throughout, not `simulate.cli.run` called directly.

## Concerns

1. `dynamics_sha256` is populated from the whole-manifest digest
   (`bundle_identity`'s `manifest_sha256`), not a dynamics-file-specific
   sha256. I believe this is the intended reuse of the Task-C1 helper (see
   above), but flagging it since the field name suggests something narrower
   than what it holds.
2. I could not run `ruff` in either available pixi environment (binary/module
   absent) to mechanically confirm style/lint compliance; I verified line
   length and unused imports by hand instead. Worth a follow-up ruff pass in
   an environment that has it installed.
3. `shotdb.ignite.load_codecs` is faked in my test even though the brief's
   Step-1 list of monkeypatches does not name it. Without faking it, the test
   would need a full real pinned bundle with real codec checkpoints for the
   default 5-modality `--decode` list, which is exactly the kind of real-data
   dependency the brief says to avoid; I believe this is within the spirit of
   "small fakes" but calling it out as a deviation from the literal list.

## Fix round 1

Addressed all four code-review findings, touching only `simulate/cli.py`, the
`simulate` subparser in `shot_design/cli.py`, and `test_simulate_cli.py` (plus
`simulate/__init__.py`, see the note on finding 2 below).

1. **Traceback discarded on failure (Important).** `run()`'s `except Exception`
   block now also prints `traceback.format_exc()` to stderr (`status.json`'s
   `error` field is unchanged: still the short `str(exc)`, since D4's poller
   reads that field and a full traceback there would be noisy). TDD: added
   `test_simulate_prints_a_traceback_to_stderr_on_failure` (capsys, `raise_in
   = "load_program"`, asserts both `"Traceback"` and `"fake_load_program"` are
   in captured stderr) *before* the fix -- confirmed it failed against the old
   code (`'Traceback' in 'shot_design simulate: boom in load_program\n'` ->
   assertion error), then added the `traceback.format_exc()` print and
   confirmed it passes.
2. **`--decode` help text duplicated the default list (Minor).** Checked
   `simulate/cli.py`'s top-level imports: it imports `torch` directly and `.
   core`, which imports `tokamak_foundation_model.ignite.eval_dynamics` and
   `dynamics_config` -- exactly the heavy machinery `cmd_simulate`'s own
   docstring says every non-`simulate` invocation must not pay to import.
   `shot_design/cli.py`'s `build_parser()` runs on *every* CLI invocation
   (not gated per-subcommand), so a lazy import inside the parser-building
   code would still defeat that invariant. Moved `DEFAULT_DECODE` into
   `simulate/__init__.py` (previously just a docstring, no imports) and made
   `simulate/cli.py` import it from there (`from . import DEFAULT_DECODE,
   core, decode, report`, re-exported via `__all__` for backward
   compatibility with anything importing `simulate.cli.DEFAULT_DECODE` -- grep
   found no such caller). `shot_design/cli.py` now does `from . import
   simulate as simulate_pkg` at module top (importing only the package's
   `__init__.py`, not its submodules) and builds the help text as
   `f"... (default: {','.join(simulate_pkg.DEFAULT_DECODE)})"`. Note: this
   touches a fourth file, `simulate/__init__.py`, which the fix-round brief's
   own wording for this finding explicitly names as the intended location
   ("if it is heavy, move `DEFAULT_DECODE` into a tiny import-light spot such
   as `src/shot_design/simulate/__init__.py`"); it does not intersect the
   concurrent UI implementer's file set.
3. **`raise_in` test not parametrized over `load_codecs` (Minor).** Added
   `"load_codecs"` to `test_simulate_leaves_a_failed_status_and_exits_1`'s
   parametrize list and made `fake_load_codecs` raise
   `ValueError("boom in load_codecs")` when `calls["raise_in"] ==
   "load_codecs"`, matching every other fake's pattern.
4. **`dynamics_sha256` key name (Minor).** Renamed to `bundle_manifest_sha256`
   in the `meta` dict `run()` builds (still populated from
   `bundle_identity(paths).get("manifest_sha256")`). Grepped `report.py` and
   the rest of `shot_design/` for any consumer keyed on the literal string
   `"dynamics_sha256"`: none exists -- `report.write` iterates `meta.items()`
   generically into h5 attrs, and `test_simulate_report.py` (D2's test file,
   out of this task's scope) only uses `"dynamics_sha256"` as arbitrary test
   fixture data for that generic-attrs behavior, never asserted by key name,
   so it did not need updating and was left untouched.

### Tests and results

```
PIXI_CACHE_DIR=/tmp/pixi-cache-nchen pixi run --frozen -e shot-design-frontier \
  python -m pytest tests/shot_design/test_simulate_cli.py -q -p no:cacheprovider
# 13 passed
```

Also ran the full `tests/shot_design` suite once before committing:

```
PIXI_CACHE_DIR=/tmp/pixi-cache-nchen pixi run --frozen -e shot-design-frontier \
  python -m pytest tests/shot_design -q -p no:cacheprovider
# 1 failed, 1728 passed, 20 skipped
# FAILED tests/shot_design/test_mcp.py::test_the_project_mcp_config_points_at_this_server
```

That failure is the pre-existing, known one named in the fix-round brief
(unrelated `git rev-parse --show-toplevel` path mismatch in the MCP config
test); `test_ui_simulate.py` (owned by the concurrent UI implementer) passed
in full, so no exclusion was needed there.
