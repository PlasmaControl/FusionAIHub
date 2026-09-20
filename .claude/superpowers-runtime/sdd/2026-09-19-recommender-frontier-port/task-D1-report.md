# Task D1 report: `shot_design.simulate.core`

## Status: DONE

## What was implemented

- `src/shot_design/simulate/__init__.py` — one-line module docstring.
- `src/shot_design/simulate/core.py` (122 lines) — `SimulationArms` dataclass and three
  functions: `load_dynamics`, `actuator_arms`, `run_paired`.
- `tests/shot_design/test_simulate_core.py` — the brief's Step 1 test, unchanged in
  substance (two long assert lines split to respect the 88-column limit; see below).

### `load_dynamics(paths, device)`

Reads `shotdb.ignite.model_cfg()["dynamics_file"]`, joins it onto
`shotdb.ignite.bundle_dir(paths)`, and calls
`tokamak_foundation_model.ignite.eval_dynamics.load_model(ckpt, device)`, which returns
`(model, cfg, step)` and infers `actuator_dim` from the checkpoint itself. Not exercised by
a test in this task (no bundle exists in the hermetic test environment); the brief's test
list only covers `actuator_arms` and `run_paired`.

### `actuator_arms(reference_cache, design_seed, k0, n_predict)`

```python
total = k0 + n_predict
real = reference_cache["actuators"][:total].float()
proposed = design_seed["actuators"][:total].float().clone()
proposed[:k0] = real[:k0]
return real, proposed
```

The seed frames `[0, k0)` are forced to agree between the two arms even though
`design_seed["actuators"]` already carries the reference values there (per `program.py`,
below) — this is a defensive floor, not a workaround for a bug, and it is what the test
`test_actuator_arms_share_the_seed_frames` actually exercises: `ref` and `des` are two
*independently* random tensors in that test, so without the explicit copy the seed-frame
equality assertion would fail.

### `run_paired(model, cfg, codes, real_act, prop_act, *, seed, temperature=1.0, decode_steps=10)`

Uses `MaskGITDynamics.rollout` (`src/tokamak_foundation_model/ignite/maskgit.py:582`)
directly, exactly as `eval_dynamics.rollout_shot` (`:265`) drives it:
`seed_codes = {n: codes[n][:k0].long().unsqueeze(0) ...}`, actuators unsqueezed to a batch
dim and cast to float, `model.rollout(seed_codes, actuators, n_predict=..., temperature=...)`.

**Determinism**: `torch.manual_seed(seed)` is called immediately before each of the two
`rollout` calls (real, then proposed), exactly as the brief specifies, rather than passing an
explicit `torch.Generator`. This works because `rollout`'s `generate_frame` draws from the
*global* RNG via `torch.multinomial(..., generator=None)` when no generator is supplied, and
`rollout_shot`'s own docstring guarantees "the rollout consumes an IDENTICAL number of RNG
draws whichever [actuator conditioning] is used" — so resetting the seed before each call
makes the two trajectories differ only through the actuator conditioning, and two full
`run_paired` calls with the same `seed` reproduce bit-identical results (verified by
`test_run_paired_is_deterministic_and_reports_token_metrics`, `a.real["ece"] == b.real["ece"]`).

One wrinkle not stated in the interface line: `rollout`/`generate_frame` take no
`decode_steps` argument — `generate_frame` reads `self.cfg.maskgit_decode_steps` off the
model's *own* bound config object. Since the test constructs `model, cfg = _model()` with
`model.cfg is cfg` (confirmed by reading `MaskGITDynamics.__init__`, which does
`self.cfg = cfg`), `run_paired` sets `cfg.maskgit_decode_steps = decode_steps` before calling
`rollout` — this mutates the caller's config object in place, which is documented inline as
the only way to thread `decode_steps` through the existing `rollout` API. Flagging this as a
minor side effect worth knowing about for D2/D3/D4, not a bug: callers that reuse a `cfg`
object across several `run_paired` calls with different `decode_steps` will see the last
value stick.

Metrics are computed over the predicted region only (`[k0:]`), per the brief:
- `divergence_vs_real[m] = 1 - fraction(real[m][k0:] == proposed[m][k0:])`
- `token_accuracy[m] = fraction(real[m][k0:] == gt[m][k0:])`
- `persistence_accuracy[m] = fraction(gt[m][k0-1] repeated == gt[m][k0:])`

## `export_ignite` actuators reading (confirmed)

Read `src/shot_design/design/program.py:340-500` and `:661-667` (`export_ignite` itself).
Confirmed by direct inspection, not inference:

```python
output = ref.cache["actuators"][context:display_end].clone() if ref.cache else None
...
output[SEED_FRAMES:, row] = torch.tensor(z, dtype=torch.float16)   # per edited row
...
artifact = {"codes": ..., "actuators": output, "n_frames": ..., "vocabs": ...}
```

`output` starts as a **clone of the reference shot's own real actuators** and then has each
*edited* channel overwritten from `SEED_FRAMES` (=20) onward with the proposal's z-scored
values (against the reference's own statistics, per `design.actuators.z_score`/`apply`).
Unedited channels stay exactly the reference's values. So:

- The `.pt` file `export_ignite` writes (the "design seed") holds the **proposed** arm:
  reference actuators with the proposal's edits burned in.
- The **real** arm is the *unedited* reference cache's own `actuators` field
  (`program_reference.reference(shot, paths).cache["actuators"]`) over the identical slice.

This matches the brief's stated hypothesis exactly ("if `output` there is the reference's
real actuators with the proposal applied, then the real arm is `reference cache["actuators"]`
over the same slice") — no adaptation was needed, this is the confirmed reading.

## Tests and results

TDD evidence:

**RED** — `pixi run --frozen -e shot-design-frontier pytest tests/shot_design/test_simulate_core.py -q -p no:cacheprovider`
```
ImportError: cannot import name 'core' from 'shot_design.simulate'
1 error in 2.74s
```
Expected: `core.py` did not exist yet, only the `__init__.py` stub.

**GREEN** — same command after implementing `core.py`:
```
..                                                                       [100%]
2 passed, 1 warning in 4.63s
```
(warning is an unrelated pre-existing `torch.jit.script` deprecation notice.)

**Full suite once, before commit** — `pixi run --frozen -e shot-design-frontier pytest tests/shot_design -q -p no:cacheprovider`:
```
1 failed, 1667 passed, 20 skipped, 2 warnings in 131.74s
FAILED tests/shot_design/test_mcp.py::test_the_project_mcp_config_points_at_this_server
```
This is exactly the known pre-existing failure called out in the task instructions; nothing
else regressed.

## Files changed

- `src/shot_design/simulate/__init__.py` (new)
- `src/shot_design/simulate/core.py` (new, 122 lines, under the 200-line cap)
- `tests/shot_design/test_simulate_core.py` (new)

Commit: `6b1e7d9` — "shot_design.simulate: seed assembly and paired real/proposed IGNITE
rollout with token metrics" (message exactly as the brief's Step 4 specifies). Only these
three files were staged and committed (`git add`/`git commit -- <explicit paths>`); the
pre-existing uncommitted edit to `.claude/superpowers/plans/2026-09-19-recommender-frontier-port.md`
and the untracked `.claude/superpowers-runtime/sdd/...` scratch directories were left
untouched, confirmed by `git status --short` before and after commit.

## Test-code adaptation

None needed for `DynamicsConfig` field names or `rollout`'s signature — `k0_seed`,
`n_predict`, `ModalitySpec(name, family, n_tok, codebook_size)` and
`rollout(seed_codes, actuators, n_predict=, temperature=, generator=, sampler=, text=)` all
matched the brief's test code as written. The only change to the given test was purely
cosmetic (splitting two lines that exceeded 88 columns): line 62's compound assert became two
asserts, and line 63 kept its comment on its own line before the assert, both with identical
runtime behavior.

## Self-review

- **Completeness**: all three functions plus the dataclass with every listed field are
  present; metrics are computed strictly over `[k0:]`; persistence baseline is
  `gt[k0-1]` repeated over the predicted frames compared token-wise to `gt[k0:]`, matching
  Step 3 verbatim.
- **Quality**: names are truthful (`_token_fraction_equal` does exactly that); no dead code;
  the `cfg.maskgit_decode_steps` mutation is called out with an inline comment rather than
  left as a silent surprise.
- **Discipline (YAGNI)**: did not add a `device` parameter to `run_paired` (not in the
  interface, and the CPU-only test doesn't need it — `load_dynamics` is the one function that
  takes `device`, per the brief); did not add extra validation of frame counts beyond what
  the given tests require.
- **Testing**: both tests exercise the real tiny `MaskGITDynamics`/`DynamicsConfig` end to
  end (constructing an actual model with `d_model=32, depth=1, n_heads=2` and running a real
  forward pass through `rollout`), not mocks. Ran the exact test file plus the full
  `tests/shot_design` suite once; output above is pristine except the one pre-known failure.
- Line lengths: verified with `awk 'length>88'` over all three new/changed files — zero
  violations (ruff itself is not installed in the `shot-design-frontier` pixi env, so the
  awk check stood in for it as the task instructions anticipated).

## Concerns

- `run_paired` mutates the `cfg` object passed in (`cfg.maskgit_decode_steps = decode_steps`)
  because `MaskGITDynamics.rollout`/`generate_frame` read decode steps off the model's own
  bound config rather than accepting it as a call argument. This is necessary given the
  existing `rollout` API and is documented inline, but D2/D3/D4 callers should be aware that
  passing the same `cfg` object into two `run_paired` calls with different `decode_steps`
  will leave the second value in place afterward (no restore-on-exit). Not a correctness bug
  for this task's scope, just a note for whoever wires up the CLI/route next.
- `load_dynamics` has no test in this task (by design — it touches the real Frontier bundle
  path and the task rules forbid reading production data roots from a test). Its correctness
  rests on matching `shotdb.ignite.model_cfg()`/`bundle_dir()`'s established contract and
  `eval_dynamics.load_model`'s documented return shape; it will get its first real exercise
  in D2/D3.

  **Superseded in Fix round 1 below** — the reviewer correctly pointed out this reasoning
  didn't hold: the `paths` fixture builds a real `models/` dir under `tmp_path`, so
  `load_dynamics` is fully testable without touching production. It now has a test.

## Fix round 1 (commit `a07eb62`, base `6b1e7d9`)

Addressed all four Important findings from the coordinator's review, plus both minors.

**1. Unrestored `cfg.maskgit_decode_steps` mutation.** `run_paired` now saves
`original_decode_steps = cfg.maskgit_decode_steps` before the two rollouts and restores it
in a `finally` block wrapping both `model.rollout(...)` calls, so an exception mid-rollout
still restores the caller's config. Added
`test_run_paired_restores_the_shared_cfgs_decode_steps`, which calls `run_paired` with
`decode_steps=2` and asserts `cfg.maskgit_decode_steps` is back to its original value
(10, the `DynamicsConfig` default the test's `_model()` never overrides) afterward.

**2. `load_dynamics` untested.** The reviewer was right and my original reasoning was
wrong — `paths` (the `tests/shot_design/conftest.py` fixture) already creates a real
`models/` directory under `tmp_path`, and `bundle_dir(paths)` /
`eval_dynamics.load_model` both work on that tmp path with no production access. Added
`test_load_dynamics_reads_the_pinned_bundle_checkpoint(paths)`, following the pattern in
`tests/ignite/test_load_model_actuator_dim.py`: build a tiny `MaskGITDynamics` from the
same `_model()` helper, `torch.save` a checkpoint dict (`model`, `cfg_depth`, `cfg_d_model`,
`cfg_n_heads`, `cfg_k0`, `cfg_n_predict`, `modalities`, `step`) under
`shotdb_ignite.bundle_dir(paths) / shotdb_ignite.model_cfg()["dynamics_file"]`, then call
`core.load_dynamics(paths, "cpu")` and assert `loaded_cfg.actuator_dim == 88` and
`step == 7`. Needed `cfg_k0`/`cfg_n_predict` in the checkpoint (absent from the referenced
example, which relies on `DynamicsConfig`'s k0=20/n_predict=80 defaults matching
`load_model`'s own defaults) because `_model()`'s config uses k0_seed=2/n_predict=3 —
without those two keys `load_model` falls back to its 20/80 defaults and the checkpoint's
`frame_embed` table (sized off `max_frames=5`) fails to load into a 100-frame model with a
`size mismatch` `RuntimeError`. First run of this test surfaced exactly that error, which is
how the missing keys were found.

**3. Short code windows failing opaquely.** Added `_check_frame_counts(codes, names,
needed)` (also covers the minor "missing modality" finding): for each modality name in
`cfg.modalities`, raises `ValueError` if the name is absent from `codes`, or if
`codes[name].shape[0] < needed`, naming the modality and both the actual and required frame
counts. `run_paired` calls it before touching `gt`/`seed_codes`, and separately checks
`real_act`/`prop_act` for the same `k0+n_predict` floor. Added
`test_run_paired_rejects_code_windows_shorter_than_k0_plus_n_predict`: `_codes(4)` against
a config needing 5 frames raises `ValueError` whose message contains both `"4"` and `"5"`.

**4. `actuators: None` raising a bare `TypeError`.** `actuator_arms` now reads
`reference_cache.get("actuators")` / `design_seed.get("actuators")` and raises `ValueError`
naming which dict is missing actuators ("reference_cache has no actuators; the real arm
needs measured controls" / the equivalent for `design_seed`) before ever touching `.float()`
or `.clone()`. Added `test_actuator_arms_rejects_missing_reference_actuators` and
`test_actuator_arms_rejects_missing_design_actuators`, matching each message with
`pytest.raises(ValueError, match=...)`.

**Minors**: `actuator_arms`'s docstring now states explicitly that `reference_cache` must
already be windowed to the same `[context:display_end]` slice as the design seed (it does
not re-align two differently-windowed caches). The missing-modality error is covered by
finding 3's `_check_frame_counts`, shared by `run_paired`.

### Tests and results (Fix round 1)

Focused file, after implementing the fixes:
```
PIXI_CACHE_DIR=/tmp/pixi-cache-nchen pixi run --frozen -e shot-design-frontier pytest tests/shot_design/test_simulate_core.py -q -p no:cacheprovider
.......                                                                  [100%]
7 passed, 1 warning in 5.23s
```
(The one intermediate RED before this: adding `test_load_dynamics_...` without `cfg_k0`/
`cfg_n_predict` in the checkpoint dict failed with `RuntimeError: size mismatch for
backbone.tok.frame_embed.weight: ... torch.Size([5, 32]) ... torch.Size([100, 32])` —
expected, since `load_model` was falling back to its 20/80 frame defaults; fixed by adding
those two keys, as described under finding 2 above.)

Full suite, once, before commit:
```
PIXI_CACHE_DIR=/tmp/pixi-cache-nchen pixi run --frozen -e shot-design-frontier pytest tests/shot_design -q -p no:cacheprovider
1 failed, 1672 passed, 20 skipped, 2 warnings in 123.16s (0:02:03)
FAILED tests/shot_design/test_mcp.py::test_the_project_mcp_config_points_at_this_server
```
1672 passed vs. 1667 before this round (+5 new tests: decode_steps restore, short-window
guard, two missing-actuator guards, load_dynamics). Same single pre-existing known failure,
nothing else regressed.

### Files changed (Fix round 1)

- `src/shot_design/simulate/core.py` (+64/-14 net; `_check_frame_counts`, `actuator_arms`
  and `run_paired` guards, `try/finally` restore, docstring note)
- `tests/shot_design/test_simulate_core.py` (+54 lines: five new tests, `dataclasses`/
  `pytest`/`shotdb_ignite` imports)

Commit: `a07eb62` — "shot_design.simulate: restore decode_steps, guard short windows and
missing actuators, test load_dynamics", committed with explicit paths
(`git add`/`git commit -- src/shot_design/simulate/core.py
tests/shot_design/test_simulate_core.py`); `git status --short` before and after confirms
only those two files changed, the unrelated plan edit and untracked scratch dirs untouched.

### Self-review (Fix round 1)

- All four Important findings addressed with a guard + a test that fails without the guard
  (verified each new test against the pre-fix `core.py` mentally matches the reviewer's
  described failure mode; the `load_dynamics` test additionally caught a real bug in my
  first attempt at it, the missing `cfg_k0`/`cfg_n_predict` keys).
- `core.py` is now 172 lines, still under the 200-line cap.
- Re-checked line lengths with `awk 'length>88'` on both changed files: zero violations.
- No leftover dead code or unused imports (`dataclasses` and `pytest` are both used only in
  the test file, where they're new).

### Concerns (Fix round 1)

None outstanding. The `cfg.maskgit_decode_steps` threading-through-mutation itself is still
the only way to reach that setting given `rollout`'s existing signature, but it is now
scoped and restored, which was the substance of finding 1.
