# Final-review fix round 1 -- report

Repo `/lustre/orion/fus187/scratch/nchen/FusionAIHub`, branch `nathan_dev`, worked
directly on top of HEAD 77a0678 (no worktree). Findings addressed: I1, I2, I3
(added mid-task), I4 (added mid-task), M1, M2, M3, M4, M9, M10, M11.

## I1 -- `scripts/slurm_frontier/shot_design_simulate.sh:4`

Dropped `#SBATCH -q debug`. Added a comment explaining why (Frontier's debug QOS
allows only one submitted job per user, `QOSMaxSubmitJobsPU=1`, which broke the
three-design demo loop and a second concurrent UI simulation; the job is
`-t 01:00:00` on `batch`, already inside debug's own 2h cap) and that a one-off
run can still opt in with `sbatch -q debug scripts/slurm_frontier/shot_design_simulate.sh <ident>`.

- Test: `tests/shot_design/test_slurm_frontier_scripts.py::test_simulate_wrapper_does_not_set_debug_qos_so_the_demo_loop_can_submit`
- Docs corrected: `docs/reference/slurm-scripts.md` (table row), `docs/shot-design/simulation.md`
  (both said `batch -q debug`). `docs/clusters/frontier.md` had no such mention.
  `scripts/shot_design/demo_frontier.sh` needed no change.
- Commit: `b5d4e59` `shot_design slurm: drop -q debug from the simulate wrapper`

## I2 -- `src/shot_design/simulate/cli.py:106-107`

`cfg.max_frames` is a property (`k0_seed + n_predict`); reading it after
`cfg.k0_seed, cfg.n_predict = args.k0, args.n_predict` always passed trivially.
Now `trained = cfg.max_frames` is captured immediately after `core.load_dynamics`,
before the assignment, and `total > trained` raises
`ValueError(f"--k0 + --n-predict = {total} exceeds the checkpoint's trained horizon {trained}")`.

- Test: `tests/shot_design/test_simulate_cli.py::test_simulate_rejects_k0_plus_n_predict_over_the_checkpoints_trained_horizon`
  (design seed 200 frames, reference window widened to 140 frames so neither of
  the other two length guards fires first; only the new checkpoint-horizon guard
  can be what raises). Also added `max_frames=100` to the fixture's default fake
  `cfg` so the other 13 pre-existing tests in the file keep exercising the new
  code path instead of hitting `AttributeError`.
- Commit: `dd61784` `shot_design simulate: guard k0+n_predict against the checkpoint's trained horizon`

## I3 (found at run time, build job 5516457) -- `src/shot_design/shotdb/build.py::_check_publish` (~1168)

Added an early return: if `db_dir / "shots.parquet"` does not exist, return `None`
(same as the `FileNotFoundError` path) before validating manifest fields. A
directory `shot_design labels join` wrote a `labels`-only manifest.json into,
but that was never built, is not a database to protect.

- Test: `tests/shot_design/test_build_store.py::test_a_join_only_manifest_does_not_block_the_first_build`
  (db_dir with only `manifest.json` = `{"labels": {"a": 1}}` plus the three join
  parquet files; `build.build(...)` proceeds without `--force` and the three
  join tables survive the publish).
- Commit: `1cd96d8` `shot_design build: a join-only manifest does not block the first build`

## I4 (found at run time, build job 5517542, 18:11) -- `src/shot_design/shotdb/text.py::build_logs_subset` (~797)

Frontier has no `sql/` layer by decision, so `paths.logs_jsonl` names a file
that never exists there, and the function raised `FileNotFoundError` on every
build/add. Added a guard at the top: if `paths.logs_jsonl` is `None` or does
not exist, `_log.warning(...)` once and return 0, leaving any existing subset
cache untouched. Verified `select.mpid_index` already guards the same way
(`if not logs_jsonl or not Path(logs_jsonl).exists(): return out`, line ~1279)
so no change was needed there. Added one sentence to
`docs/shot-design/database-build.md`'s Build section spelling out that
`build_logs_subset` is a deliberate no-op on Frontier (the existing "Logs
(informational only)" section already said the logbook is absent by decision).

- Test: `tests/shot_design/test_text.py::test_build_logs_subset_without_a_logs_jsonl_file_is_a_noop`
  (plain `paths` fixture with no `text_fixtures`, so `logs.jsonl` was never
  written; asserts `n == 0`, no subset file written, one warning logged).
- Commit: `45931ae` `shot_design text: build without logs.jsonl when the corpus has no sql layer`

## M1 -- `src/shot_design/design/assistant.py:305`

`client.cfg.get("provider") == "off"` -> `client.off`, which already normalises
`""`/`None`/`"none"`/`"false"`/casing.

- Test: `tests/shot_design/test_assistant.py::test_run_design_treats_every_off_spelling_as_unavailable`
  (parametrized `"None"`, `"FALSE"`; asserts `run_design`'s own message
  `"needs a configured model"`, which the old code only produced for the exact
  literal `"off"` -- other spellings fell through to `chat()`'s differently
  worded refusal).

## M2 -- `src/shot_design/llm/client.py:133 available()`

After the `agy` branch, added `if provider not in self._providers: return False, f"unknown llm provider {provider!r}; known providers: {...}"` before falling
into the ollama endpoint logic, mirroring `chat()`'s own refusal.

- Test: `tests/shot_design/test_llm_client.py::test_available_names_an_unknown_provider_instead_of_falling_through_to_ollama`

## M9 -- `src/shot_design/cli.py::cmd_llm` (~682)

`client.cfg.get(provider, {}).get("bin", provider)` -> `(client.cfg.get(provider) or {}).get("bin", provider)`, so a provider block present but `null` (the
style `llm.yaml` uses for absent Frontier-only settings) no longer raises
`AttributeError`.

- Test: `tests/shot_design/test_cli_llm.py::test_llm_status_does_not_crash_when_the_provider_block_is_null`

M1+M2+M9 commit: `ef8b165` `shot_design llm: normalise provider-off and unknown-provider handling`
(note: this commit's diff to `src/shot_design/cli.py` also carries the M3 and
M4 hunks below -- see Concerns.)

## M3 -- `src/shot_design/cli.py` `design show` parser (~2266-2276)

`--references` and `--actuation-csv` are now in one
`s.add_mutually_exclusive_group()` on the `design show` subparser.

- Test: `tests/shot_design/test_design_show_cli.py::test_references_and_actuation_csv_together_is_a_usage_error`
  (`SystemExit(2)`, message contains "not allowed with argument").

## M4 -- `src/shot_design/cli.py::_evalsets_prompt` (~704-724)

`except OSError` -> `except (OSError, yaml.YAMLError)`. For a `dict` entry,
`entry["text"]` -> `entry.get("text")`; `None` prints
`"prompt has no text: {slug!r} in evalset {name!r}"` to stderr and returns 1
instead of raising `KeyError`.

- Tests: `tests/shot_design/test_evalsets_cli.py::test_malformed_evalset_yaml_fails_cleanly_instead_of_raising`,
  `::test_evalset_entry_without_text_prints_a_message_instead_of_a_traceback`
  (both monkeypatch `config.CONFIG_DIR` to a tmp dir with a hand-written
  evalset file; no real corpus/config touched).

M3+M4 commit: `9aec926` `shot_design cli: mutually exclusive design-show flags; evalsets robustness`
(this commit's `src/shot_design/cli.py` hunk is empty -- see Concerns; the
actual `cli.py` code changes for M3/M4 are inside `ef8b165`).

## M10 -- `src/shot_design/shotdb/build.py::_bundle_pulse_length_s` (~333-352)

Now imports `select` and calls `select._num(raw)` instead of its own
`float(str(raw).replace(" ", ""))`/`except ValueError` copy.

- Test: `tests/shot_design/test_build_proxy_segments.py::test_bundle_pulse_length_s_delegates_to_selects_num`
  (monkeypatches `select._num` to a sentinel return value and asserts
  `_bundle_pulse_length_s` returns that same sentinel -- proves the call graph,
  not just input/output equivalence on today's inputs).
- Commit: `a01564f` `shot_design build: share select._num for the bundle pulse-length parse`

## M11 -- `tests/shot_design/test_simulate_report.py:48`

Renamed the meta literal `dynamics_sha256` -> `bundle_manifest_sha256` (what
`simulate/cli.py:127` actually writes). Inert change (nothing in `report.write`
reads or asserts that key), corrects a stale contract in the test.

- Commit: `f2dd3b4` `shot_design tests: rename simulate meta literal to bundle_manifest_sha256`

## Concerns

- **Commit/hunk mismatch on `src/shot_design/cli.py`**: M1/M2/M9 (commit
  `ef8b165`) and M3/M4 (commit `9aec926`) all touch `src/shot_design/cli.py`.
  Because `git add <path>` stages the whole file's current working-tree state,
  the M3 (`design show` mutually-exclusive group) and M4 (`_evalsets_prompt`
  hardening) code hunks ended up bundled into the `ef8b165` commit alongside
  M1/M2/M9's `cli.py` change (the `cmd_llm` fix), even though `ef8b165`'s
  message only describes the llm-side findings. `9aec926` therefore has no
  `cli.py` diff of its own -- only its two test files. All code is correct and
  every listed test passes; this is a commit-message/hunk-boundary accuracy
  issue only (`git show ef8b165 -- src/shot_design/cli.py` shows all three
  hunks: `cmd_llm`, `_evalsets_prompt`, and the mutually-exclusive group). Left
  as-is rather than rewriting history; flagging for awareness.
- `tests/shot_design/test_docs_scratch_db.py` (4 tests) fails on this checkout
  independent of this work -- it reads `docs/SHOT_DESIGN.md`, which commit
  `2284331` ("docs: reorganise into Docusaurus sections...") removed without
  updating this test. Not touched; out of this brief's scope. Confirmed
  pre-existing (file is absent at HEAD, test unchanged by this round).
- Slurm build job 5517542 was reported running against this checkout's `.pixi`
  env during the work; `pyproject.toml`/`pixi.*`/`.pixi/` were not touched, and
  no bare `pixi` command was run (only the pinned
  `.pixi/envs/shot-design-frontier/bin/python` and, read-only for a line-length
  check, `.pixi/envs/fdp/bin/ruff`).

## Final test tail

Targeted files (per the brief's list, adjusted to what exists):

```
tests/shot_design/test_simulate_cli.py tests/shot_design/test_simulate_report.py
tests/shot_design/test_llm_client.py tests/shot_design/test_cli.py tests/shot_design/test_cli_llm.py
tests/shot_design/test_assistant*.py tests/shot_design/test_slurm_frontier_scripts.py
tests/shot_design/test_build_proxy_segments.py tests/shot_design/test_features.py
tests/shot_design/test_design_show_cli.py tests/shot_design/test_evalsets_cli.py
tests/shot_design/test_build_store.py tests/shot_design/test_text.py
-> 277 passed in 25.15s
```

Whole `tests/shot_design -q -p no:cacheprovider --deselect tests/shot_design/test_mcp.py`:

```
FAILED tests/shot_design/test_docs_scratch_db.py::test_the_section_names_every_variable_the_shot_design_features_pin
FAILED tests/shot_design/test_docs_scratch_db.py::test_the_section_gives_both_ways_to_build_a_scratch_database
FAILED tests/shot_design/test_docs_scratch_db.py::test_the_section_quotes_the_origin_labels_the_code_prints
FAILED tests/shot_design/test_docs_scratch_db.py::test_the_section_says_what_the_publish_guard_refuses_and_how_to_override_it
4 failed, 1710 passed, 23 skipped, 68 deselected, 2 warnings in 119.10s (0:01:59)
```

All 4 failures are the pre-existing `docs/SHOT_DESIGN.md`-removal breakage
described above, unrelated to any finding in this round.
