### Task F2b: proxy segments for a corpus shot with no Ip trace (Frontier)

**Why this task exists.** Frontier has no plasma-current signal anywhere: the FAITH corpus
groups are diagnostics + actuators (no `ip`), the labeler feature store
(`$LABELER_ROOT/features/<shot>_features.h5`, which `CorpusSignalReader._read_features`
reads for the `ip` spec) does not exist on Frontier, and the two sources labeler's `ip` feature
knows (`archive` = d3d_fusion_data, `fdp` = MDSplus) are Stellar-only. So on Frontier
`build_record` sees `ip is None`, `find_segments` is never called, every record has zero
segments (no flat-top scalars, no waveform shapes, no derived labels), and `cmd_build`'s
`_with_ip` gate skips all 3,031 shots ("nothing to build", jobs 5515059 and 5515085).

The DIII-D shot table row in each shot's text bundle carries `PULSE-LENGTH` (seconds) and
`IP-(MA)`. `shotdb/select.py` already uses `PULSE-LENGTH - PROXY_RAMP_S` (1.27 s, measured on
319 shots, see the comment above `PROXY_RAMP_S`) as the flat-top proxy for rule (d). This task
gives `build` the same proxy so a Frontier build produces real segments, recorded honestly as
proxy segments.

**Files:**
- Modify: `src/shot_design/shotdb/features.py` — add `proxy_segments(pulse_length_s, cfg) -> list[Segment]`
- Modify: `configs/shot_design/retrieval.yaml` — under `segments:` add `proxy_ramp_up_s: 1.0` and `proxy_ramp_down_s: 0.27` with a comment: their sum is `select.PROXY_RAMP_S` (1.27 s, measured ramp-up + ramp-down over 319 shots); the split is a ruling (DIII-D Ip ramp-up to 85 % is ~1 s), not a measurement
- Modify: `src/shot_design/shotdb/build.py::build_record` — when `ip is None`, read the shot's `PULSE-LENGTH` (see how `select.py` ~line 320–345 gets `row = text.shot_table_row(bundle)` for a shot; reuse the same bundle lookup — `human_tier` is computed a few lines later in `build_record`, so move it up if that avoids a second read) and call `features.proxy_segments`; when it yields segments, set `coverage_reasons["ip"] = "segments from PULSE-LENGTH proxy (ramp-up 1.0 s, ramp-down 0.27 s); no Ip trace on this cluster"` (merge with the reader's `reasons`, never overwrite them); `ip_out` stays `{"end_reason": "no_ip_signal"}` and `raw_sources` still lacks `ip` — those two are the machine-readable marks that the segments are proxies
- Modify: `src/shot_design/cli.py` — rename `_with_ip` → `_buildable(shots, paths, reader)`; a shot is buildable when Ip is `present` OR its bundle's `PULSE-LENGTH` parses to a finite number > `proxy_ramp_up_s + proxy_ramp_down_s`; keep the skipped-message shape but say "have neither an Ip trace nor a PULSE-LENGTH in their shot bundle"; keep `--all` semantics
- Test: `tests/shot_design/test_features.py` (proxy_segments), the build test file that already exercises `build_record` with a fake/tmp reader (look in `tests/shot_design/test_build_*.py`; add there or a new `tests/shot_design/test_build_proxy_segments.py`), and the CLI build gate test if one exists (grep `_with_ip` in tests/)

**Interfaces:**
- `proxy_segments(pulse_length_s: float | None, cfg: dict) -> list[Segment]` — `cfg` is the `segments` block of retrieval.yaml (the same dict `find_segments` receives). Returns `[]` for None/non-finite/`pulse - ramp_up - ramp_down <= 0`. Otherwise, in ms, with the same names and order `find_segments` produces: `full` [0, pulse], `ramp_up` [0, ramp_up], `flat_top` [ramp_up, pulse - ramp_down], `ramp_down` [pulse - ramp_down, pulse]. Segment is `shot_design.schema.Segment(name, t0_ms, t1_ms)`.
- Time origin: corpus `xdata` is seconds with t=0 at breakdown (DIII-D convention); `PULSE-LENGTH` counts from t=0.

- [ ] **Step 1: failing tests.** `test_proxy_segments_splits_pulse_length_into_the_four_segments`: pulse 5.0 with the yaml defaults → full (0, 5000), ramp_up (0, 1000), flat_top (1000, 4730), ramp_down (4730, 5000) (compare with `pytest.approx`). `test_proxy_segments_is_empty_when_there_is_no_flat_top`: 1.2 → [], None → [], float("nan") → []. Build test: a `build_record` call with a reader whose `read_shot` returns no `ip` but returns one actuator signal, and a text bundle carrying `PULSE-LENGTH` 5.0 → record has a `flat_top` segment with `t0_ms == 1000`, `coverage_reasons["ip"]` starts with "segments from PULSE-LENGTH proxy", `outcome.end_reason == "no_ip_signal"`, and the actuator's `_mean` scalar over flat_top is populated. CLI gate test: `_buildable` keeps that shot and skips one with no Ip and no PULSE-LENGTH.
- [ ] **Step 2:** run, confirm they fail for the right reason.
- [ ] **Step 3:** implement the four files above. Keep every new line ≤ 88 columns.
- [ ] **Step 4:** `PIXI_CACHE_DIR=/tmp/pixi-cache-nchen pixi run --frozen -e shot-design-frontier python -m pytest tests/shot_design -q -p no:cacheprovider -x --deselect tests/shot_design/test_mcp.py` (the one `test_mcp.py` git-rev-parse failure is known and pre-existing). Report the count.
- [ ] **Step 5:** commit with explicit pathspecs (other agents share this checkout; retry on `index.lock`; never `git add -A`, never stash): `git commit -m "shot_design build: PULSE-LENGTH proxy segments for a shot with no Ip trace"`.
- [ ] **Step 6: Frontier smoke (read-only, login node).** With `SHOT_DESIGN_PATHS=$PWD/configs/shot_design/paths.frontier.yaml SHOT_DESIGN_DATA_ROOT=/lustre/orion/fus187/proj-shared/nchen/shot_design LABELER_ROOT=/lustre/orion/fus187/proj-shared/nchen/labeler SHOT_DESIGN_CORPUS=/lustre/orion/fus187/proj-shared/foundation_model HF_HUB_OFFLINE=1 HDF5_USE_FILE_LOCKING=FALSE` and `.pixi/envs/shot-design-frontier/bin/python`, call `build_record(190736, paths, load_build_cfg(), make_reader("corpus", paths))` in a `python -` snippet and print `len(rec.segments)`, the flat_top t0/t1, `rec.coverage_reasons.get("ip")`, and 5 populated raw scalar keys. Do NOT run `shot_design build` and do not write under the data root. Paste the output into the report.
