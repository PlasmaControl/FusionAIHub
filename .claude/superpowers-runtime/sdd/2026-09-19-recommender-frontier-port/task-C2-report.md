# Task C2 report: `mirnov` through the encode path and cache-first frame codes

Commit `fcd03b8` on `nathan_dev` (parent `b6b86be`). Not pushed.

## What was implemented

### 1. `src/shot_design/design/program_reference.py`

- `_cache_path(shot, paths)` now searches, in order:
  `model_cfg()["frame_codes_cache"]` -> `paths.data_root/"frame_codes"` -> `bundle_dir(paths)/"frame_codes"`,
  then the source-digest `reference_cache` fallback as before. The production root is only
  prepended when the key is present and non-empty, so a paths file without it behaves as today.
  `model_cfg` is now imported into this module (the fixture patches `pr.model_cfg`).
- `validate_cache` now checks the GENERATION FIRST: right after the four-key check, `vocabs` must
  be a dict whose key set equals `model.production_vocabs`, and every value must equal the pinned
  one. Only then come `n_frames`, the actuator block and the per-modality token checks. Rationale
  in the code comment: a v2 cache and a v4 one are otherwise indistinguishable, and reported the
  old way round the first complaint a v2 cache draws is "actuators must be float16", which sends
  the reader hunting for a bug in a file whose only fault is its age.
  The per-modality message is unchanged in wording (`... vocabulary N does not match pinned
  production vocabulary M ...`), which is what `test_program.py`'s wrong-generation test greps.
- The stale `"must contain all 14 production modalities"` message now reads `len(specs)` (15).
- The redundant `set(vocabs) != set(codes)` check was removed: both sides are now pinned to the
  same yaml, so it could not fire.

### 2. `src/shot_design/shotdb/ignite.py`

- New `production_cache_path(shot)` -> `model.frame_codes_cache/<shot>.pt` or None. Read-only.
- New `_cached_frame_codes(shot, codecs, max_frames)`: loads that file (`weights_only=True`),
  RAISES on a vocab that is not the pinned one (a pin disagreeing with itself must not be
  re-encoded around), returns None when the cache lacks any requested modality (so the encoder
  takes over for ALL of them - half-cached/half-encoded codes would mix two machines' arithmetic
  inside one frame), else returns `{name: (F, n_tok) int64}` for the requested names only.
- `frame_codes(..., use_cache: bool = True)` consults it before touching the corpus.

### 3. `src/shot_design/design/seed.py`

- `MODALITIES` and `VIDEO_MODALITIES` are no longer hand-written tuples; they are derived from
  `ignite.model_cfg()["families"]` (yaml key order IS the canonical token order, taken from the
  pinned bundle's `codecs/MANIFEST.json`). 14 -> 15 with `mirnov`.
- `wanted_modalities` re-reads `model.families` per call, so an unknown-name rejection and the
  `include_video=False` filter both follow the pinned generation rather than a second copy of the
  list.
- `encode_frame_codes(..., use_cache: bool = True)` forwards to `ignite.frame_codes`.
- Module docstring: 14 -> 15 modalities, v4 vocabs (1000 everywhere) instead of v2's
  32768/64000/1000, "all 15 slots", `n_frames` 239 -> 219 (v4's frame 0 is at 1.0 s), and the
  VERIFIED paragraph is now explicitly labelled as a generation-v2 measurement - v4 has not been
  through G-ENC.

### 4. `src/tokamak_foundation_model/ignite/train_dynamics.py`

- `FROZEN_CODEC_CKPTS` gains exactly the brief's row
  `"mirnov": ("spectro", "eval_runs/ignite_d5_mirnov/codec_best.pt")`, with a comment saying why
  it is still needed after C0: `resolve_codec_path` tries `tmpl` first, but `load_frozen_codecs`
  reads the family as `FROZEN_CODEC_CKPTS[n][0]` unconditionally. Nothing was restructured;
  `actuator_frames` untouched.

### 5. `scripts/shot_design/g_enc.py` (one line + comment, beyond the brief's file list)

- `encode_one` passes `use_cache=False`. WITHOUT THIS THE GATE WOULD BE BROKEN by this task: it
  would be handed production's own frame codes for any shot in `frame_codes_cache` and compare
  them against the shipped cache they were copied from, passing whatever the encoder does. Same
  reason `tests/shot_design/test_ignite.py::test_frame_codes_reproduce_the_production_cache_bit_for_bit`
  now passes `use_cache=False`.

## Tests

New, in `tests/shot_design/test_ignite_v4.py` (plus a `prod_cache` fixture and a `_payload`
helper that builds a shipped-layout cache from `model_cfg()`):

- `test_cache_path_prefers_the_production_v4_cache` - a file exists in BOTH the production root
  and `data_root/frame_codes`; production wins. A shot in neither returns None.
- `test_validate_cache_rejects_a_v2_vocabulary` - `ece: 32768` on an otherwise 1-frame,
  float32-actuator cache still reports `ece`, i.e. the generation is reported before structure.
- `test_frame_codes_reads_the_production_cache_instead_of_re_encoding` - no corpus file exists
  for the shot, so an encode could not even start; `max_frames` is honoured.
- `test_frame_codes_encodes_when_the_cache_is_refused` - `use_cache=False` reaches the encoder
  (FileNotFoundError from `_resolve_input`).
- `test_frame_codes_refuses_a_cached_shot_from_another_generation`.

Updated:

- `test_seed.py::test_wanted_modalities_drops_video_on_request_and_rejects_an_unknown_name`:
  14 -> 15, 12 -> 13, `mirnov` present, and the tuple equals `model_cfg()["families"]`.
- `test_seed.py::test_encode_frame_codes_allows_a_named_partial_run_for_diagnostics`: its stub
  `model_cfg` now also carries `families` (it previously returned only `t0_start_s`).
- `test_ignite.py::test_codecs_expose_the_encode_then_quantize_contract`: the encodable set is
  v4's 13 names (v2's 12 + `mirnov`).

### The `model_cfg()` monkeypatching in the brief does not work

The brief's `monkeypatch.setitem(ignite.model_cfg(), "frame_codes_cache", ...)` is a no-op:
`config.load_yaml` caches by (size, mtime) but returns `copy.deepcopy(...)` on every call, so each
`model_cfg()` hands back a fresh dict. The tests patch the FUNCTION instead, on both modules that
imported it by name (`ignite.model_cfg` and `pr.model_cfg`). This changed the test mechanics only,
not the design, so it was not escalated.

### RED evidence (before the implementation)

```
E       AttributeError: <module 'shot_design.design.program_reference' ...> has no attribute 'model_cfg'
tests/shot_design/test_ignite_v4.py:209: AttributeError          (x5, the prod_cache fixture)
E       AssertionError: assert ('ece', 'bes'...v_upper', ...) == ('ece', 'bes'...v_lower', ...)
E         At index 4 diff: 'tangtv_lower' != 'mirnov'
E         Right contains one more item: 'filterscopes'
tests/shot_design/test_seed.py:216: AssertionError
FAILED tests/shot_design/test_seed.py::test_wanted_modalities_drops_video_on_request_and_rejects_an_unknown_name
ERROR tests/shot_design/test_ignite_v4.py::test_cache_path_prefers_the_production_v4_cache
ERROR tests/shot_design/test_ignite_v4.py::test_validate_cache_rejects_a_v2_vocabulary
ERROR tests/shot_design/test_ignite_v4.py::test_frame_codes_reads_the_production_cache_instead_of_re_encoding
ERROR tests/shot_design/test_ignite_v4.py::test_frame_codes_encodes_when_the_cache_is_refused
ERROR tests/shot_design/test_ignite_v4.py::test_frame_codes_refuses_a_cached_shot_from_another_generation
```

Separately, the pre-existing v2 expectation (the RED this task inherited):

```
E       AssertionError: assert {'bes', 'cer_...rscopes', ...} == {'bes', 'cer_...rscopes', ...}
E         Extra items in the left set:
E         'mirnov'
tests/shot_design/test_ignite.py:192: AssertionError
1 failed, 1 passed, 1 warning in 5.61s
```

### GREEN

```
tests/shot_design/test_ignite_v4.py test_seed.py test_ignite.py test_program.py
125 passed, 6 skipped, 1 warning in 13.02s
```

## Step 4: live check (Frontier login node, CPU)

The brief's third shot, 199597, is NOT in the production cache
(`/lustre/orion/fus187/proj-shared/models/ignite_prod_v4/frame_codes` holds 8752 files spanning
190000-204999; 199597 is absent). The brief's snippet verbatim:

```
190000 219 (219, 88) 15 /lustre/orion/fus187/proj-shared/models/ignite_prod_v4/frame_codes/190000.pt
190090 219 (219, 88) 15 /lustre/orion/fus187/proj-shared/models/ignite_prod_v4/frame_codes/190090.pt
199597 NO CACHE
```

Re-run with 204346 (a G-ENC gate shot) as the third, plus the cache-first encode path exercised
with a codec set that includes `mirnov`:

```
190000 219 (219, 88) 15
190090 219 (219, 88) 15
204346 219 (219, 88) 15
frame_codes -> 14 modalities, mirnov (4, 192) int64
```

`validate_cache` accepts all three unmodified, i.e. the real production caches are exactly what
the new generation check demands: 15 modalities, every vocab 1000, `(F, 88)` float16 actuators.
`n_frames` is 219, not the 239 the seed docstring used to claim (v4's frame 0 is at 1.0 s).

## Full suite before committing

```
pixi run --frozen -e shot-design-frontier pytest tests/shot_design tests/ignite -q \
    --ignore=tests/ignite/test_train_codec.py
1 failed, 2258 passed, 25 skipped, 4 warnings in 373.35s (0:06:13)
FAILED tests/shot_design/test_mcp.py::test_the_project_mcp_config_points_at_this_server
```

That failure is PRE-EXISTING and unrelated: `.mcp.json` names a Stellar checkout
(`/scratch/gpfs/nc1514/FusionAIHub`) as the server `cwd`, and `git -C` on it exits 128 here.
Confirmed by stashing the whole diff (`git stash push src tests scripts`) and re-running it:
`1 failed in 1.12s`.

`tests/labeler/test_labels_layout_integration.py` and `tests/ignite/test_train_codec.py` were not
run, per the task instructions.

Ruff (`uvx ruff check --line-length 88`) passes on every changed shot_design file, the test files
and `g_enc.py`. `train_dynamics.py` reports 57 findings both with and without my change
(verified by stashing), i.e. the added row introduces none.

## Self-review of the diff

- `program_reference._cache_path` is the only behavioural change visible to the UI: a shot in the
  production corpus is now served from there rather than re-encoded. `prepare_reference` returns
  early for such shots, which is the intended effect on Frontier.
- The reordering inside `validate_cache` was checked against every existing caller expectation in
  `test_program.py`: the `vocab` corruption case, the four-way wrong-generation parametrisation
  (which greps for the modality name, "vocabulary" and "production"), `missing_modality`,
  `token_shape`, `token_dtype`, `frames`, `actuator_shape`, `nonfinite` and the extra-key case all
  still pass, and their messages all still contain "cache" (the UI-level assertion).
- `_cached_frame_codes` returns only the modalities the caller's `codecs` names, sliced to
  `max_frames`, as int64 - the same shape/dtype contract `_frames` returns, so
  `seed.encode_frame_codes` (which casts to int32 for the payload) is unaffected.
- `_frames` itself needed no change to accept `mirnov`: the family is `spectro`,
  `train_codec.SPECTRO_MODALITIES` already lists it, and shot_design's `load_codecs` reads the
  family from the bundle manifest rather than from `FROZEN_CODEC_CKPTS`. The only thing that was
  stale was the test's 12-name expectation.
- `MODALITIES`/`VIDEO_MODALITIES` are now computed at import time from the yaml (two `load_yaml`
  calls, cached). `wanted_modalities` re-reads per call so a stub or a re-pin is picked up.

## Concerns / follow-ups (nothing blocking)

1. **`_cache_path` now reads an absolute machine path during tests.** The `paths` fixture is
   tmp_path-based, but the production root is not under it, so a test using a shot number that
   exists in `/lustre/.../ignite_prod_v4/frame_codes` (190000-204999) would silently pick up a
   real cache. Nothing collides today (`test_program.py` uses 990091), and the full suite passes,
   but it is a hermeticity trap for whoever adds the next program-reference test.
2. **`shotdb.build.frame_codes_dirs` is still v2 and does NOT know about the production cache.**
   It hardcodes `<models_dir>/IGNITE/frame_codes` - the v2 bundle name, not `bundle_dir(paths)`
   (`IGNITE_v4`) - and feeds `has_frame_codes`, `corpus select`'s preference and the MCP
   `describe_shot` provenance lookup. So `shot_design` will report "no frame codes" for all 8752
   shots production has already encoded. Out of this task's scope (changing it moves build,
   select, cli and mcp tests), but it is the same defect this task fixed in `_cache_path` and
   should be its own task.
3. **G-ENC's `FULL_SHOT_FRAMES = 239` is a v2 constant.** v4's full shot is 219 frames
   (`t0_start_s: 1.0`), so the gate as written would reject every v4 cache on frame count before
   comparing a token. Untouched here; flagged for whoever re-runs G-ENC against v4.
4. **The `use_cache` opt-out is mine, not the brief's.** The brief asked for cache-first in
   `frame_codes` unconditionally; taken literally that would have turned both G-ENC and
   `test_frame_codes_reproduce_the_production_cache_bit_for_bit` into tautologies (comparing
   production's cache with a copy of itself). The default is cache-first as specified; the two
   parity call sites opt out explicitly and say why in a comment.
5. **`mirnov`'s `FROZEN_CODEC_CKPTS` path is unverified.**
   `eval_runs/ignite_d5_mirnov/codec_best.pt` is the brief's string and follows the `d5` naming of
   its siblings, but it is a repo-relative fallback that is only consulted when `codec_tmpl`
   misses; nothing here resolved it against a real file. The v4 path in practice is the pinned
   bundle via `tmpl`, which does load (`tests/ignite/test_v4_assets_load.py`, untouched).
6. `tests/ignite/test_v4_assets_load.py`, `.claude/` and `scripts/slurm_frontier/` were not
   touched. The modified `.claude/superpowers/plans/...md` in `git status` was already dirty at
   the start of this task and is NOT in the commit.
