# Task fix-C1 — report

Worktree `/scratch/gpfs/nc1514/FusionAIHub-build`, branch `recommender-fix`. Implementer: Claude
(Fable 5.1), replacing the Codex agent named in the brief. Every command below was run from the
worktree with the brief's environments; nothing was written outside the worktree and `/tmp`, no
SLURM job was submitted, and the data roots under `/scratch/gpfs/EKOLEMEN/nc1514/` were only read.

## Final verification (at HEAD `c5a8245`)

```
tests/labelmaker  -W error   1345 passed, 2 skipped  (176 s)
tests/ideate      -W error   833 passed, 1 failed    (251 s; the one failure is the pre-existing .mcp.json cwd test, below)
ruff (recommender-scoped: src/labelmaker tests/labelmaker src/ideate tests/ideate
      scripts/labelmaker scripts/ideate)                             All checks passed!
ruff (the brief's exact command: src tests scripts/labelmaker scripts/ideate)
                                                                     Found 1160 errors
```

* Baselines: labelmaker 1256 passed / 2 skipped in main at a8ad02c; +34 from L10, +50 from Lfix,
  +5 from this task = 1345 expected. ideate 778 in main; +51 from Ifix, +4 from this task = 833
  expected, of which one is the pre-existing failure below.
* The one ideate failure is `tests/ideate/test_mcp.py::test_the_project_mcp_config_points_at_this_server`:
  it asserts `.mcp.json`'s absolute `cwd` (`/scratch/gpfs/nc1514/FusionAIHub`) resolves to the
  checkout the test runs in, and therefore fails in every worktree but the main one. Known and
  pre-existing (it did at ce621c6 too); not skipped, not "fixed".
* The 1160 ruff findings from the brief's exact command are ALL in `src/tokamak_foundation_model/**`
  (625 in `ignite/`, 218 at the package root, 53 + 51 + 17 + 15 + 9 + 6 + 4 in `models/*` and
  `utils/`), `tests/ignite/**` (94) and `tests/e2e/**`. The same per-directory distribution is
  reported at the main checkout's HEAD (fdcee38) with the same command, so they are pre-existing
  and outside both briefs (neither brief names those packages; Lfix 6 was about
  `scripts/labelmaker`, which is clean). 904 are auto-fixable but 146 only with `--unsafe-fixes`,
  and none of that code is this workstream's, so I did not touch it. Every directory the
  recommender workstream owns is ruff-clean.

## Commits (this task; repo style, trailer on each)

| sha | subject |
|---|---|
| `9ad0d4a` | labelmaker: merge recommender-Lfix - per-source coverage and the sources record, on the L10 stage split |
| `acb4d1e` | labelmaker: a point on the detector's own edge is clipped into coverage, not refused with its whole step |
| `e878cb6` | ideate: a text source that ran is not an observation, and the sources contract is guarded from labelmaker's own writer |
| `22a03dd` | ideate: corpus select says what is deterministic about its file, and what 507 frame-code shots counts |
| `c5a8245` | ideate: the G-ENC diagnostic marker is one named rule, with a test |

Plus the report commit that follows this file. Already on the branch when I started (the
controller's merges of the partial work): `a4dbf02` (recommender-L: d5ad174, 63e91a5, f97d3bb,
5103c49, 6b33fde), `16aab1a` (recommender-Ifix: 1e34242, ce417b6, 5729250), and the Lfix commits
brought in by `9ad0d4a` (cf66024, ad79a33, 988de33).

## Step 1 — the merge

One conflict, `src/labelmaker/events/pipeline.py` lines 543–584: L10 kept `_span` beside its new
`plan_shot`; Lfix deleted `_span`. Every `_span` call site had already been replaced by
`events/coverage.py` (`finite_span`, `feature_spans`, `intersect`, `UNKNOWN`) in the parts git
merged cleanly, so `_span` was dead code: resolved by dropping it and keeping `plan_shot`
(`grep -rn '_span\b' src tests scripts` shows only docstring mentions of the old name). The body of
`finish_shot` (L10's split) already carried Lfix's `ran` dict and its `schema.write_sources` call,
so `events/driver.py` and `scripts/labelmaker/tokeye_masks.py` — which call `finish_shot` — write
`events/<shot>_sources.parquet` through the same code as `process_shot`. Suite after resolution:
1340 passed / 2 skipped = 1256 + 34 + 50, i.e. both sides' tests survived. Committed as `9ad0d4a`.

## Step 2 — audit, deliverable by deliverable

Legend: **done** = present in the partial commits and I read the code and its tests; **finished** =
I completed it in this task. Every line names the test that fails if the behaviour is broken.

### Lfix-C1

**1. Evidence policy in `events/windows.py` — done (cf66024).** `DIAGNOSTIC_EVIDENCE = ("detector",
"heuristic")`, `FAMILY_SOURCES` (coherent_mode/pickup ← tokeye_track, elm ← tokeye_transient,
elm_free ← elm_clock, sawtooth ← ece_sawtooth, lh_transition ← dalpha_lh), applied once in
`EventTable.from_frame` via `diagnostic_rows`, imported by `features/resolve_events.py` and stamped
into the served array's `attrs`; the windows.py:17 docstring rewritten to state the policy.
Guarding tests, `tests/labelmaker/test_events_evidence.py`:
`test_the_real_elm_tuning_note_is_not_an_elm` (the verbatim 185980 entry through the real
`text_weak.text_events` and lexicon, alongside a `label_forecast` at 1 s, a genuine `elm_clock`
row and a genuine `tokeye_transient` row; asserts `elm_rate_hz == 0` in (0.0, 0.34) and in
(0.9, 1.24), `time_since_last_elm_s == ELM_AGE_CAP_S`, and that the genuine rows still count),
`test_selecting_by_phenomenon_alone_is_the_defect_this_pins` (reproduces the critic's 2.941 Hz
without the policy), `test_the_real_sawtooth_piggyback_note_is_not_a_crash` (`n_sawtooth == 0` in
both windows), `test_a_human_database_or_model_row_is_not_a_detection`,
`test_a_model_row_wearing_the_trackers_name_is_not_a_track`,
`test_a_text_only_shot_resolves_to_zero_events_and_honest_coverage` (whole `resolve_events` path),
`test_the_evidence_policy_is_two_named_constants`.
*Deviation from the brief's wording:* the brief says `elm ← elm_clock (and tokeye_transient only if
that is the documented ELM path)`. In the code, `transients.transients_to_events` writes `elm`
rows with `source = "tokeye_transient"` and `elm_free` rows with `source = "elm_clock"`
(`transients.py:105–108`), and `docs/LABELMAKER.md` tabulates the same, so the policy maps
`elm ← tokeye_transient`, `elm_free ← elm_clock`. Verified by reading the writer.

**2. Per-source coverage — done (ad79a33).** `events/coverage.py`: `finite_span` (first to last
FINITE sample; any channel finite counts for a multi-channel feature), `feature_spans`,
`intersect` (one unknown input makes the intersection unknown), `UNKNOWN = (nan, nan)`.
`heuristics.actuator_intervals` takes `{name: span}`, each row's `diag` names the axis it was
measured on; `nbi_counter` takes the intersection of torque, power and current; the L-H detector
gets the intersection of D-alpha, line density and injected power; the sawtooth detector the ECE
finite span. Tests: `test_events_coverage.py::test_each_actuator_event_carries_its_own_axis`
(the critic's three 198658 axes: gas −10…94.8576, NBI 0…13.1001, RMP −1.06286…10.20114 s; asserts
each row carries its own and that the union would over-claim NBI by > 80 s),
`test_nbi_counter_takes_the_intersection_of_the_three_it_needs`,
`test_trailing_nan_padding_is_not_coverage`,
`test_one_dead_channel_does_not_end_a_multi_channel_observation`,
`test_a_single_span_is_still_accepted_and_is_still_the_old_answer`; at the pipeline end
`test_events_pipeline.py::test_each_actuator_source_records_its_own_axis` and
`test_the_lh_detector_covers_only_where_all_three_inputs_were_measured`.

**3. Track extent vs coverage — done for tracks (ad79a33), finished for point events (acb4d1e).**
`tracks.tracks_to_events` clips the extent into `t_cov` with `attrs["clipped"] = true`
(`duration_ms` in attrs stays the untrimmed measurement); `heuristics._actuator_event` likewise;
`schema.Event.__post_init__` refuses `t1_s > t_cov1_s` when both are finite. Tests:
`test_events_coverage.py::test_the_stitched_transform_really_does_run_past_the_record` (through the
real `masks.prep`/`col_times_s`: 0.962 ms past a 40 ms record),
`test_a_track_that_runs_past_the_record_is_clipped_and_says_so`,
`test_every_track_row_ends_inside_its_own_coverage`, `test_a_block_with_no_known_coverage_clips_nothing`,
`test_the_coverage_invariant_only_fires_on_two_finite_bounds`,
`test_an_extent_wholly_outside_its_coverage_is_not_repaired`.
*What was incomplete:* two other producers put POINTS past the bound by construction and were left
to the refusal — and `finish_shot` isolates failures per STEP, so one refused row would have cost
a shot its whole ELM clock (both `tokeye_transient` and `elm_clock`, recorded as skipped in the
sources file) or its every L-H claim. (i) `transients.elm_events` returns column times, and the
transform's first and last columns lie outside the record (`masks.COL_ORIGIN = -3`). (ii)
`heuristics.lh_transitions` clears its beam gate on `[when − 5 ms, when]`, so a transition within
5 ms of the NBI record's last finite sample lies past the intersection coverage that deliverable 2
now passes it — reproduced by the new test as `ValueError: t1_s 0.30005 against t_cov1_s 0.298`.
Fix: `coverage.clip_point_to_coverage` moves such a point onto the bound and says so; both
producers write `attrs["clipped"] = true` and keep the measured column (`col`) or instant
(`t_measured_s`); the extent rule for intervals is unchanged. Tests:
`test_events_coverage.py::test_a_point_is_moved_onto_the_bound_it_overran_and_says_so`,
`test_an_elm_on_the_transforms_edge_column_would_otherwise_be_refused` (pins the defect),
`test_an_elm_on_an_edge_column_is_moved_into_coverage_and_says_so`;
`test_events_heuristics.py::test_a_transition_just_after_the_beam_record_ends_is_kept_and_clipped`,
`test_a_transition_inside_its_coverage_is_not_marked_clipped`. Sawtooth crashes are sample times
of a finite trace and QH candidates are intersections with `elm_free` (itself bounded by the
coverage), so both lie inside their coverage by construction; I checked and changed neither.

**4. Per-source completion record — done (ad79a33), guarded further here.** `schema.write_sources`
/ `read_sources`, `SOURCE_COLUMNS` in the contract's exact order and dtypes, `SOURCE_KEY =
(source, diag, channel, pass_name)`, merged and atomic like `write_events`; `Paths.sources_file`;
`coverage.source_records` turns `ran` + `ShotResult.skipped` + the events into rows (a step that
owns two sources gets two rows; `norm`, `qh_flattop`, `nbi_counter` name themselves). Tests:
`test_events_schema.py::test_the_sources_contract_is_the_documented_columns_and_dtypes`,
`test_a_source_that_ran_and_found_nothing_is_a_row`, `test_a_skipped_step_is_a_row_carrying_its_reason`,
`test_a_merge_replaces_one_key_and_keeps_the_others`, `test_the_same_key_may_not_be_written_twice_in_one_call`,
`test_read_sources_of_a_missing_file_is_an_empty_typed_frame`, `test_a_record_that_cannot_be_true_is_refused`;
`test_events_pipeline.py::test_every_source_that_ran_is_a_row_even_with_no_events`,
`test_a_skipped_step_reaches_the_sources_file_with_its_reason`,
`test_a_failed_step_leaves_both_of_its_sources_visible`,
`test_a_rerun_of_one_channel_replaces_only_that_channels_rows`, `test_write_false_writes_no_sources_file`.
*Added here:* the driver identity test
`test_tokeye_masks.py::test_the_driver_writes_exactly_what_process_shot_writes` now also compares
the sources file the two schedules write (nothing asserted it before), and the cross-package test
`tests/ideate/test_labels_join.py::test_a_sources_file_labelmaker_itself_wrote_is_ingested_as_is`
writes through labelmaker's real `schema.write_sources` and ingests through `ideate labels join`,
asserting the two packages' column tuples, dtype maps and file paths agree (the other ideate tests
build the table through ideate's own fixture writer).
*Note, not a deviation:* the pipeline never emits `status == "error"` — a step that raises is
recorded `skipped` with the exception as `reason` (per-step isolation makes "ran and failed" and
"did not run" the same skip). The contract keeps `error`, and ideate handles it
(`test_mcp.py::test_a_source_that_failed_is_reported_rather_than_counted_as_coverage`).

**5. Docs — done (988de33, reviewed; one paragraph added in acb4d1e).** `docs/LABELMAKER.md` now has:
the evidence-kinds table with "may become a diagnostic feature" per kind; "a `text` row does not
describe what a diagnostic showed" with the two real entries; the policy as `DIAGNOSTIC_EVIDENCE` +
`FAMILY_SOURCES`; the coverage section (per source and quantity, finite samples, NaN = unknown ≠
absent, clipping — I added the point-event sentence); the sources-table section with the column
meanings and the three-way reading (ok/0 = observed silence, skipped = no observation, no row =
not processed); the "Known limits" entry on shot-scope text prevalence (0.6 %, a MISS) and "a hit is
not an assertion". `grep -rn "diagnostic showed" docs src` finds no remaining claim that every
non-forecast row describes what a diagnostic showed in the labelmaker docs; the one in
`docs/IDEATE.md:90` is fixed in `e878cb6`. No test guards prose.

**6. Ruff over `scripts/labelmaker` — done (988de33), verified.** `ruff check scripts/labelmaker` at
HEAD: `All checks passed!`. The WIP commit's removal of the `# noqa: E402` (pin_unet.py,
elm_write_normalization.py) and `# noqa: PLC0415` (retrain_tearing_dsm.py) suppressions was right:
under the repository's configuration ruff does not flag those imports (it permits imports after
`sys.path` manipulation, and PLC0415 is not an enabled rule), so the directives were the "unused
suppression" findings the critic counted, and the script logic is unchanged (the `worst(order,
m=m, s=s)` default-binding in elm_write_normalization.py is the loop-variable-capture fix, behaviour
preserved). The other findings the critic saw in `scripts/labelmaker` are gone at HEAD.

### Ifix-C1

**1. G-ENC strictness — done (1e34242), one untested piece finished (c5a8245).** (a) `verdict`
fails a requested modality with `equal is None` ("not encoded"); (b) `encode_frame_codes` raises
`CheckpointMissing` for a requested modality without a codec, `allow_partial=True` is the named
diagnostic escape hatch whose cache then fails the gate; (c) `validate_cache` checks the four
keys, int32 codes, float16 (F, 88) actuators, `n_frames` against every tensor, tokens inside
their vocab, matching shapes/vocabs, the gate's 239 frames; (e) the 190735/190736 wording narrowed
in both `g_enc.py` and `design/seed.py`; (f) `gates/g_enc.json` untouched (mtime 2026-09-07
16:12:22, `passed: false`, `failed_shots: [190090, 204346]`), header section "WHAT COUNTS AS THE
GATE, AND WHAT DOES NOT". Tests, all in `tests/ideate/test_seed.py` against the real 202537
structure (`test_the_synthetic_reference_matches_the_shipped_202537_cache` pins the fixture to the
shipped file, read-only): `test_g_enc_fails_when_a_requested_modality_was_never_encoded`,
`test_g_enc_does_not_fail_for_a_modality_nobody_asked_for`, `test_g_enc_fails_for_a_single_changed_token`,
`test_g_enc_fails_at_81_of_88_actuator_channels_and_passes_at_82`,
`test_g_enc_compare_refuses_a_cache_whose_n_frames_lies_about_its_tensors`,
`test_g_enc_compare_refuses_a_short_shot_against_the_gates_239_frames`,
`test_g_enc_compare_refuses_a_dtype_that_is_not_the_shipped_one`,
`test_g_enc_compare_refuses_a_token_outside_its_own_vocabulary`,
`test_g_enc_compare_refuses_a_token_width_that_is_not_the_shipped_one`,
`test_g_enc_passes_only_when_every_requested_modality_is_bit_identical`,
`test_g_enc_header_does_not_claim_a_demonstrated_input_difference`,
`test_encode_frame_codes_refuses_to_silently_drop_a_requested_modality`,
`test_encode_frame_codes_allows_a_named_partial_run_for_diagnostics`.
*What was unverified:* the report's `"diagnostic": true` marker and its stdout notice were computed
inline in `main()`, unreachable by any test. `is_diagnostic(shots, no_video=, allow_partial=)` is
now the one named rule `main` calls (same expression, no behaviour change), pinned by
`test_g_enc_marks_anything_narrower_than_the_three_shot_gate_as_a_diagnostic`.

**2. MCP/DB coverage states — done (5729250), one policy hole finished (e878cb6).**
(a) Contract: `src/ideate/labels/event_sources.py` (`SOURCES_COLUMNS`/`SOURCES_DTYPES` exactly the
brief's list, `write_sources`/`source_row` fixture writer, `sources_union`, `shot_summary`,
`coverage_span`, `covers`); `labels join` ingests every existing file into
`db/event_sources.parquet` and records `n_sources_ok/skipped/error`, per-shot
`has_observed_products`, `n_shots_unprocessed`. Tests: `test_event_sources.py` (8 + 1 mine),
`test_labels_join.py::test_the_join_ingests_every_source_file_that_exists_and_counts_the_rest`,
`test_a_join_with_no_source_files_at_all_still_writes_a_typed_empty_table`, and the cross-package
test named under Lfix 4. (b) `get_events`: `status` ∈ {unindexed, unprocessed, uncovered,
observed} with a caveat each; payload split `events` / `text_mentions` / `forecasts`; window
validated (`_window`: reversed, zero-width, NaN, non-numeric → error dict with caveat). Tests,
`test_mcp.py`: `test_a_shot_the_database_does_not_hold_is_unindexed_not_quiet`,
`test_an_indexed_shot_nobody_processed_is_unprocessed`,
`test_a_window_outside_every_sources_coverage_is_uncovered_and_names_the_span`,
`test_a_source_that_ran_and_saw_nothing_says_so_in_as_many_words`,
`test_a_source_that_failed_is_reported_rather_than_counted_as_coverage`,
`test_a_reversed_or_non_finite_window_is_an_error_not_a_silent_empty`,
`test_a_text_row_is_a_lexicon_hit_and_never_lands_in_events`.
*What was incomplete:* labelmaker records `text` as a source that RAN over the shot's span, and
`shot_summary`/`coverage_span` counted any `ok` row, so a shot with a logbook entry and no detector
run would have come back `observed` — "0 detections inside coverage" — on the strength of a lexicon.
`NON_DIAGNOSTIC_SOURCES = ("text",)` now excludes it from `has_observed_products` and from
coverage (`n_sources_ok` still counts it). Tests:
`test_event_sources.py::test_a_text_source_that_ran_is_not_an_observation_and_covers_nothing`,
`test_mcp.py::test_a_shot_where_only_the_text_ran_is_unprocessed_not_observed`.
(c) `has_frame_codes` refresh: `build.refresh_frame_codes` (column and `record_json`), called by
`labels join`, count recorded in `manifest.json["labels_join"]`. Tests:
`test_labels_join.py::test_the_join_refreshes_has_frame_codes_from_the_directory`,
`test_refreshing_twice_changes_nothing_and_a_missing_table_is_not_an_error`,
`test_write_tables_records_the_refreshed_count_in_its_own_manifest_block`. **The production
re-run was already made by the Ifix implementer** (allowed by its brief): `db/manifest.json` carries
`labels_join.frame_codes = {n_shots 500, n_has_frame_codes 500, n_was_true 13, n_changed 487,
refreshed true}` written 2026-09-08T03:18:12+00:00, and `db/shots.parquet` reads
`has_frame_codes: {True: 500}` (I read both; wrote nothing). `db/event_sources.parquet` exists with
the contract's 13 columns and **0 rows** — correct, because no `events/<shot>_sources.parquet`
existed at join time (the labelmaker events directory holds only the three pilot `_events.parquet`),
so every one of the 500 is `unprocessed` today, which is the truth the old empty list hid.
(d) Instructions: `mcp/server.py` scopes the caveats promise to application-level replies and
documents the framework's schema-validation path; `get_events`/`describe_shot` docstrings updated;
`test_mcp.py::test_the_instructions_do_not_promise_a_caveat_the_transport_cannot_deliver` exercises
it over the real stdio transport. `.mcp.json` carries no instructions (command, args, cwd only),
so there was nothing to change in it. `docs/IDEATE.md`'s "every other row is a claim about what a
diagnostic showed" is replaced (e878cb6) by the three lists, the four statuses and the scoped
caveats promise.

**3. Frame-code provenance — done (ce417b6), verified.** `design/provenance.py` (sidecar schema
`ideate-frame-codes-provenance-v1`: device, torch_threads, torch_version, git_sha, bundle path +
codec-manifest sha + HF revision, input fingerprint with `kind` = size+mtime (sha256 optional), n_frames,
modalities, encoded_at, run_manifest, backfilled); `encode_frame_codes` writes one per cache and
never lets it fail the encode; `ideate encode` settles the run-manifest path first so cache and
manifest name each other; `scripts/ideate/frame_codes_provenance.py --backfill`; `describe_shot`
returns a `frame_codes` block and caveats a cache without a sidecar; `ideate coverage` prints the
device census. Tests: `test_provenance.py` (14, including `test_backfill_leaves_the_payload_byte_identical`,
`test_backfill_never_overwrites_a_sidecar_a_real_encode_wrote`, `test_a_backfilled_sidecar_does_not_describe_the_process_doing_the_backfill`),
`test_seed.py::test_encode_frame_codes_writes_the_shipped_dict_structure` (the `.pt` still has
exactly the four keys, and the sidecar is read back beside it),
`test_mcp.py::test_describe_shot_says_which_device_encoded_the_frame_codes`,
`test_cli.py::test_coverage_splits_the_encoded_shots_by_the_device_that_encoded_them`.
**The production backfill was already made by the Ifix implementer:** 500 `frame_codes/<shot>.json`
beside the 500 `.pt` (device cpu 490 / cuda 10; `backfilled: true` on all 500; `run_manifest: null`
on 70 — the OOM-killed array tasks' shots, whose manifests never got written; `torch_version` and
`git_sha` null because the run manifests never recorded them, not inferred). I read the census;
wrote nothing. *Deviation, allowed by the brief:* the input fingerprint is size+mtime, declared as
such in `input_fingerprint.kind`.

**4. Determinism wording — not started by Ifix; finished (22a03dd).** `store_fingerprint` returns
`frame_codes_by_dir` beside the union count; `summarize` carries it into the document;
`format_summary`'s verification block prints the frame-code count labelled as the union over the
frame-code directories (production store + shipped bundle, a shot in both counted once) with the
per-directory split under it, and two lines: the `shots:` rows are byte-identical run to run over
the same census, store and seed; the header (`created`, `summary.from_list`,
`summary.feature_store`) is regenerated each run — diff the rows, not the file. Tests,
`test_select.py`: `test_the_summary_says_whether_the_list_is_final_and_which_store_verified_it`
(asserts the 507 = 500 + 10 split and both statements are printed),
`test_the_store_fingerprint_counts_the_files_and_takes_the_newest_mtime`,
`test_the_fingerprint_of_a_store_that_is_not_there_is_empty`,
`test_an_unfinalized_summary_still_carries_the_keys_with_nothing_in_them`,
`test_cli_select_is_deterministic_across_two_runs` — which now holds the claim it prints: the text
from `shots:` onward is compared byte for byte (it compared shot order before) and the printed
summary is asserted to say so. I did not re-run `corpus select --finalize` on the real list: the
committed `recommender_v1.yaml` is not part of any deliverable, the critic already measured the
property on it, and the CLI test holds it on synthetic inputs.

## Production writes

None by me. Both writes the Ifix brief allowed were already made by the Ifix implementer before the
kill and are verified above (sidecars: 500/500; `labels join`: `has_frame_codes` 500/500,
`event_sources.parquet` present and empty). The command that will give `db/event_sources.parquet`
its rows once the labelmaker events production run has written `events/<shot>_sources.parquet` for
the 500 (that run is SLURM work outside this task):

```
cd /scratch/gpfs/nc1514/FusionAIHub && pixi run -e ideate-cpu python -m ideate labels join --list recommender_v1
```

(`labels join` defaults: `--labelmaker-root $LABELMAKER_ROOT`, `--db` from paths.yaml; it also
refreshes `has_frame_codes` and rewrites `manifest.json`'s `labels` and `labels_join` blocks.)

## Additions beyond the briefs' literal text, and why

* Point-event clipping in `transients` and `heuristics.lh_transitions` (Lfix 3): without it the
  new schema refusal, meeting a real edge-column ELM or a transition within 5 ms of the beam
  record's end, loses the shot's whole step in the 500-shot production run.
* `NON_DIAGNOSTIC_SOURCES` in ideate (Ifix 2): the same defect-1 principle applied to the coverage
  table, which otherwise reports a text-only shot as observed silence.
* The cross-package contract test (Lfix 4 / Ifix 2a): the only test that fails if the two
  packages' definitions of the sources file drift.
* The driver identity test now covers the sources file (Lfix 4): `finish_shot` writes it for both
  schedules and nothing asserted so.
* `is_diagnostic` (Ifix 1f): the one piece of the strictness work no test reached.
* `docs/IDEATE.md`: the API-documentation half of the critic's defect 2 lived there, not only in
  the server docstring.

## Left undone, with reasons

* The 1160 ruff findings in `src/tokamak_foundation_model/**`, `tests/ignite/**`, `tests/e2e/**`:
  pre-existing at the main checkout's HEAD, not this workstream's code, 146 fixable only unsafely.
  The brief's ruff command therefore does not come back clean; every recommender directory does.
* `test_the_project_mcp_config_points_at_this_server`: pre-existing worktree-relative failure,
  reported, not skipped.
* The optional login-node `process_shot` re-run on 198658 with `write=False` (allowed by the Lfix
  brief, ~3 min CPU): not run. It is required by no deliverable, and the synthetic-shot pipeline
  tests (`test_events_pipeline.py`, `test_tokeye_masks.py`) exercise the merged `plan_shot →
  prep_block → infer_block → describe_block → finish_shot` path end to end, sources file included.
* `db/event_sources.parquet` stays empty until the labelmaker events production run; command above.
