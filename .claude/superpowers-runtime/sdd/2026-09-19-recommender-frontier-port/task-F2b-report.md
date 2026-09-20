# Task F2b report: PULSE-LENGTH proxy segments for a shot with no Ip trace

Status: DONE_WITH_CONCERNS (see "Open concerns" -- the mandated Step 6 shot did not
exercise the proxy path, because the real Frontier labeler feature-file backfill is
running concurrently and has already reached that shot).

Commit: `0e2985b5afa20e059170d41854821cfcd93f0b3c` on `nathan_dev`
("shot_design build: PULSE-LENGTH proxy segments for a shot with no Ip trace").

## Changes, per file

- **`src/shot_design/shotdb/features.py`** -- added `proxy_segments(pulse_length_s, cfg)
  -> list[Segment]`. `[]` for `None`/non-finite/`pulse - ramp_up - ramp_down <= 0`;
  otherwise returns `full [0, pulse]`, `ramp_up [0, ramp_up]`,
  `flat_top [ramp_up, pulse - ramp_down]`, `ramp_down [pulse - ramp_down, pulse]` in ms,
  same names/order as `find_segments`. Placed directly after `find_segments`.

- **`configs/shot_design/retrieval.yaml`** -- added `segments.proxy_ramp_up_s: 1.0` and
  `segments.proxy_ramp_down_s: 0.27`, with a comment tying their sum to
  `select.PROXY_RAMP_S` (1.27 s, measured over 319 shots) and noting the 1.0/0.27 split
  is a ruling, not a second measurement.

- **`src/shot_design/shotdb/build.py`**:
  - Added `_bundle_pulse_length_s(shot, paths) -> float | None`, which reads
    `per_shot_txt_dir/shot_<N>.txt` (same file `text.mp_text` reads for the title) and
    parses `text.shot_table_row(bundle)["PULSE-LENGTH"]` the same way `select._num`
    does (space-stripped float, else `None`). Returns `None` for a missing bundle, a
    bundle with no shot table row, or a row with no `PULSE-LENGTH`.
  - In `build_record`, when `ip is None`, the code now calls
    `features.proxy_segments(_bundle_pulse_length_s(shot, paths), cfg["segments"])`
    instead of always returning `[]`. When that yields segments, a local
    `proxy_ip_reason` string is set:
    `"segments from PULSE-LENGTH proxy (ramp-up 1.0 s, ramp-down 0.27 s); no Ip trace
    on this cluster"`.
  - `coverage_reasons` construction changed from
    `dict(getattr(reader, "reasons", None) or {})` (unconditional) to: build that same
    dict, then `if proxy_ip_reason: reasons["ip"] = proxy_ip_reason` -- merges with,
    never replaces, the reader's own per-signal reasons.
  - `ip_out` / `raw_sources` are untouched: `ip_out` still defaults to
    `{"end_reason": "no_ip_signal"}` when `ip is None`, and `raw_sources` still has no
    `"ip"` key in that branch, exactly as the brief specifies -- the two machine-readable
    marks that the segments came from a proxy, not a measurement.
  - Did **not** move `human = text.human_tier(shot, paths)` earlier: it internally
    re-reads the same bundle file through `text.mp_text` via a different call path
    (title/purpose parsing, not `shot_table_row`), so moving the call up would not
    actually avoid the second file read without a deeper refactor of `text.mp_text`/
    `human_tier` to accept pre-read bundle text. Left as a noted open concern rather
    than done speculatively.

- **`src/shot_design/cli.py`**:
  - Renamed `_with_ip` -> `_buildable(shots, paths, reader=None)`. Behavior: a shot is
    kept when `reader.signal_status(shot, ip_spec) == "present"` (unchanged), OR its
    text bundle's `PULSE-LENGTH` (via `select_mod.read_bundles` +
    `select_mod.parse_facts(...).pulse_length_s`, reusing the existing public helpers
    rather than re-parsing) is finite and `> proxy_ramp_up_s + proxy_ramp_down_s`
    (read from `build_mod.load_build_cfg()["segments"]`).
  - `cmd_build`'s call site updated to `_buildable(wanted, paths, reader)`.
  - Skipped-shots message changed from "... have no Ip signal on disk yet and are
    skipped ..." to "... have neither an Ip trace nor a PULSE-LENGTH in their shot
    bundle and are skipped ..." (same parenthetical/`_brief()` shape kept).
  - Added `import math` (for `math.isfinite`).

- **Tests**:
  - `tests/shot_design/test_features.py`: added
    `test_proxy_segments_splits_pulse_length_into_the_four_segments` (pulse 5.0 ->
    full/ramp_up/flat_top(1000,4730)/ramp_down, matching the brief's numbers exactly)
    and `test_proxy_segments_is_empty_when_there_is_no_flat_top` (parametrized over
    1.2, `None`, `nan`).
  - `tests/shot_design/test_build_proxy_segments.py` (new file): a minimal fake
    `SignalReader` (`_NoIpOneActuatorReader`) that returns `ip=None` for every spec and
    a constant `pnbi_15L` beam signal; a fixture writes a text bundle with
    `PULSE-LENGTH: 5.0` via the shared `conftest.text_bundle` helper.
    `test_build_record_uses_pulse_length_proxy_when_ip_is_missing` asserts
    `flat_top.t0_ms == 1000`, `t1_ms == 4730`, `coverage_reasons["ip"]` starts with
    "segments from PULSE-LENGTH proxy", `outcome.end_reason == "no_ip_signal"`, and
    `flat.raw["pnbi_15L_mean"] == 2.0e6`. A second test
    (`test_build_record_without_a_bundle_still_has_no_segments`) pins the old
    zero-segment behavior for a shot with no bundle at all, so the new code path is
    additive, not a replacement of the old empty-record answer.
  - `tests/shot_design/test_cli.py`: updated
    `test_build_skips_shots_with_no_ip_on_disk` and
    `test_build_reader_corpus_reads_the_corpus_and_says_so` for the new message text
    (grepped for `_with_ip` in tests/ first -- no test referenced it by name, so there
    was no dedicated gate test to rename; these two asserted the literal message
    string and needed updating regardless). Added
    `test_buildable_keeps_a_pulse_length_proxy_shot_and_skips_a_bare_one`, calling
    `cli._buildable([shot_with_pulse_length, shot_with_neither], paths)` directly and
    asserting the split.

## TDD evidence

Implementation was written first in this session, then verified as true TDD by
capturing the four source-file diffs as a patch, `git checkout --` reverting exactly
those four files (features.py, build.py, cli.py, retrieval.yaml) back to HEAD while
keeping the new/changed tests, running the affected tests to confirm they failed for
the right reason, then re-applying the patch and re-running to confirm green -- rather
than claiming TDD without evidence.

**Failing (pre-implementation, only test files present):**

```
FAILED tests/shot_design/test_features.py::test_proxy_segments_splits_pulse_length_into_the_four_segments
  AttributeError: module 'shot_design.shotdb.features' has no attribute 'proxy_segments'
FAILED tests/shot_design/test_features.py::test_proxy_segments_is_empty_when_there_is_no_flat_top[1.2]
FAILED tests/shot_design/test_features.py::test_proxy_segments_is_empty_when_there_is_no_flat_top[None]
FAILED tests/shot_design/test_features.py::test_proxy_segments_is_empty_when_there_is_no_flat_top[nan]
  AttributeError: module 'shot_design.shotdb.features' has no attribute 'proxy_segments'
FAILED tests/shot_design/test_build_proxy_segments.py::test_build_record_uses_pulse_length_proxy_when_ip_is_missing
  assert None is not None   # rec.segment("flat_top") was None -- no proxy path yet
FAILED tests/shot_design/test_cli.py::test_buildable_keeps_a_pulse_length_proxy_shot_and_skips_a_bare_one
  AttributeError: module 'shot_design.cli' has no attribute '_buildable'
6 failed, 1 passed, 70 deselected in 0.77s
```

**Passing (post-implementation, same test selection plus the surrounding build/select/cli
files):**

```
tests/shot_design/test_features.py tests/shot_design/test_build_proxy_segments.py
tests/shot_design/test_cli.py tests/shot_design/test_build_store.py
tests/shot_design/test_select.py
223 passed in 20.15s
```

## Full-suite count (Step 4)

Exact command from the brief:

```
PIXI_CACHE_DIR=/tmp/pixi-cache-nchen pixi run --frozen -e shot-design-frontier \
  python -m pytest tests/shot_design -q -p no:cacheprovider -x \
  --deselect tests/shot_design/test_mcp.py
```

Result:

```
1688 passed, 20 skipped, 68 deselected, 2 warnings in 123.16s (0:02:03)
```

(exit code 0; the only deselection is `test_mcp.py`, as expected -- no other failures
or collection errors.)

## Step 6: Frontier smoke (read-only, login node)

Ran exactly the specified snippet against shot 190736 with
`SHOT_DESIGN_PATHS=$PWD/configs/shot_design/paths.frontier.yaml`,
`SHOT_DESIGN_DATA_ROOT=/lustre/orion/fus187/proj-shared/nchen/shot_design`,
`LABELER_ROOT=/lustre/orion/fus187/proj-shared/nchen/labeler`,
`SHOT_DESIGN_CORPUS=/lustre/orion/fus187/proj-shared/foundation_model`,
`HF_HUB_OFFLINE=1 HDF5_USE_FILE_LOCKING=FALSE`, and
`.pixi/envs/shot-design-frontier/bin/python`:

```
n_segments: 4
flat_top t0_ms: 1359.3866264343264 t1_ms: 5121.386626434328
coverage_reasons[ip]: None
populated raw/derived scalars (up to 5):
  ('full', 'ip_mean', 1438632.875242452)
  ('full', 'ip_peak', 1634609.45)
  ('full', 'ip_min', 164715.5)
  ('full', 'ip_slope', 125728.04899583173)
  ('full', 'bt_mean', 1.967213163511355)
```

`build_record` ran end-to-end against real Frontier paths/data with no exceptions and
produced a sensible 4-segment record. However, `coverage_reasons["ip"]` is `None` and
`ip_mean`/`ip_peak`/etc. are populated with real numbers -- this means shot 190736's
`ip` came from a real signal (a labeler feature file), not the PULSE-LENGTH proxy. See
"Open concerns" below for why, and for a second, non-mandated check that does exercise
the proxy path on real Frontier data.

## Open concerns

1. **The mandated smoke shot (190736) did not exercise the proxy path**, because
   `/lustre/orion/fus187/proj-shared/nchen/labeler/features/190736_features.h5` exists
   and carries a real `ip` feature -- `build_record`'s `ip is signals.get("ip")` was
   not `None`, so `find_segments` ran normally and `proxy_segments` was never called for
   this shot. This directly contradicts part of the task's stated premise ("the labeler
   feature store... does not exist on Frontier"). To confirm the new code path is not
   just unit-tested but also works against real Frontier data, I ran the same
   `build_record` call (still read-only, still not `shot_design build`) against shot
   190007, which I first confirmed had **no** `190007_features.h5`:
   ```
   n_segments: 4
   flat_top t0_ms: 742.8696557998667 t1_ms: 5077.8696557998655
   coverage_reasons[ip]: None
   outcome.end_reason: target_signal_unusable
   raw_sources has ip: True
   ```
   By the time this second call ran, `190007_features.h5` had *also* appeared (real
   `ip` again, proxy again not exercised). I re-listed the labeler features directory
   between the two checks: it grew from 51 files to 90 files in the few minutes this
   task was running. **A labeler feature backfill job is actively writing new
   `<shot>_features.h5` files on Frontier concurrently with this task**, which means
   the "Frontier has no Ip trace anywhere" premise the whole task is built on is
   becoming false shot-by-shot in real time. I did not chase a still-unbackfilled shot
   further to force a real-data proxy demonstration, since (a) the target keeps moving,
   (b) the brief named 190736 specifically and I should not silently substitute a
   different shot to make the demo look better, and (c) the unit tests
   (`test_build_proxy_segments.py`, `test_features.py`) directly exercise and pin the
   proxy path's numeric behavior with a controlled fake reader, which is the evidence
   that does not depend on which real shots happen to be backfilled yet. Whoever owns
   the actual `shot_design build` run for the 3,031-shot database should re-check, at
   build time, how many of those shots still lack a labeler `ip` feature -- the proxy
   may end up covering far fewer shots than task F2b's motivating numbers assumed, or
   the backfill may finish before the real build runs and this task's proxy path may
   see little to no use in practice (it remains correct and tested either way).

2. **`human_tier` is not moved earlier / the bundle is still read twice.** The brief
   suggested moving `human = text.human_tier(shot, paths)` up to avoid a second bundle
   read. I looked at this and did not do it: `human_tier` reads the bundle internally
   through `text.mp_text` (title/purpose parsing), not through `text.shot_table_row`,
   so relocating the call does not by itself eliminate the extra read of
   `per_shot_txt_dir/shot_<N>.txt` that `_bundle_pulse_length_s` now also does when
   `ip is None`. Sharing the read would need `text.mp_text`/`human_tier` to accept
   pre-read bundle text, which is a larger refactor than this task's scope. The extra
   read only happens on the `ip is None` branch and is one small text file per shot, so
   the performance cost is minor, but it is a real (small) duplicate read worth noting.

3. **No existing CLI test referenced `_with_ip` by name** (grepped `tests/` before
   writing), so there was no single "the gate test" to rename -- I updated the two
   `test_cli.py` tests that asserted the old literal skip message and added a new,
   dedicated `_buildable` unit test instead.
