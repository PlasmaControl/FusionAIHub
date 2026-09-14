# Task I-prev — report

Worktree `/scratch/gpfs/nc1514/FusionAIHub-Iprev`, branch `recommender-Iprev` cut from
`recommender` @ 3cced76. Four commits, one per deliverable.

```
0b8f332 ideate: document the pixi activation env and the publish guard
ca7eac3 ideate: build times its phases and records them in the manifest
1070996 ideate: every writing command names the data root it resolved
7e05cd8 ideate: guard rebuilds of different or larger databases
```

`tests/ideate` was 1176 at the branch cut and is 1206 now: +13 from deliverable 1 (1189 was the
baseline this session started from), +10 from deliverable 2, +3 from deliverable 3, +4 from
deliverable 4.

## Deliverable 1 — publish guard (7e05cd8, written by the previous implementer)

Reviewed against the brief rather than rewritten; nothing was missing, so no code changed.
`src/ideate/shotdb/build.py::_check_publish` is called twice — once on the requested count
before anything is read, once on the successfully-built count before the publish — and
`tests/ideate/test_publish_guard.py` (13 tests) pins each clause of the brief:

| brief clause | where it is pinned |
| --- | --- |
| refuses a different `shot_source` | `test_guard_refuses_without_touching_existing_files[list:other]` |
| refuses a smaller `n_shots` (a `--limit` pilot, a `shots:1` build) | same test, `[shots:1]` and `[list:recommender_v1]` cases |
| refuses on the count actually built, not the count requested | `test_guard_uses_successfully_built_count` |
| a corrupt or unreadable manifest counts as different | `test_corrupt_manifest_requires_force_and_records_unknown_history` (5 shapes of damage) and `test_unreadable_manifest_refuses_before_writing` |
| no file touched on refusal | every one of the above snapshots `(bytes, mtime_ns)` of the whole data root and re-compares it |
| non-zero exit, message naming both sources, both counts and `--force` | `test_cli_force_records_previous_database_and_add_remains_an_upsert` (exit 1 through `cli.main`, which turns the `ValueError` into a stderr line) |
| same source, ≥ as many shots still publishes (the 500 → 504 rebuild) | `test_same_source_equal_or_larger_rebuild_still_publishes[2 and 3 shots]` |
| `--force` records `forced_over: {shot_source, n_shots, built_at}` | the CLI test above, asserted key-by-key against the previous manifest |
| `add` and `labels join` are NOT guarded | the CLI test ends with an `add` that publishes; `cmd_labels` never calls `build()` — it calls `refresh_frame_codes` and `join.write_tables` |

Two observations recorded rather than changed: `_check_publish` carries no return annotation,
and a `--force` over a corrupt manifest writes `forced_over` with three nulls (there is nothing
readable to record). Neither contradicts the brief.

## Deliverable 2 — every writing command names its root (1070996)

`config.data_root_origin()` mirrors `load_paths`'s environment precedence and returns one of
`IDEATE_DATA_ROOT env`, `IDEATE_PATHS=<file>`, `configs/ideate/paths.yaml default`.
`cli._announce_root(command, destination, paths)` prints

```
ideate build: data root /x/y (IDEATE_DATA_ROOT env) -> db /x/y/db
```

on stderr before the first write of `build`, `add`, `labels join`, `encode`, `corpus scan` and
`corpus select`. A config too broken to resolve (`IDEATE_DATA_ROOT=""`) degrades the line to
`ideate <command>: no data root resolved -> <destination>` rather than failing a fully-explicit
`corpus scan`/`corpus select` run that never needed a root.

Red → green: `tests/ideate/test_write_root.py` **10 failed → 10 passed**. The red run failed for
the right reasons — `config` had no `data_root_origin`, and each command's first `os.mkdir`
under the data root happened with stderr still empty. Ordering is tested, not just presence:
`watch_first_write` hooks `os.mkdir` and records what stderr held at the first directory created
under the root (every writer in scope reaches its output through a
`mkdir(parents=True, exist_ok=True)`, and `Path.mkdir` calls `os.mkdir` even when the directory
exists). Full ideate suite at this commit: **1199 passed in 208.41s, exit 0**.

## Deliverable 3 — build phase timing (ca7eac3)

`build.PHASES` names six phases in the order they run — `read_records`, `segment`,
`scalar_embedding`, `text_embedding`, `write_tables`, `publish` — and the `_phase` context
manager accumulates wall seconds into `manifest["phase_seconds"]`, logged at the end as
`build phases (wall s): ...`. The publish cannot time itself into the manifest it moves, so the
published manifest is rewritten afterwards through a sibling `manifest.json.part` renamed over
it.

Red → green: `tests/ideate/test_build_phases.py` **3 failed → 3 passed** (red: no
`build.PHASES`, no `phase_seconds` key, no phase log record). The tests pin the key set *and*
its order against `build.PHASES`, that the values are non-negative floats, that the *published*
manifest carries a non-zero `publish`, and that the log line names every phase. Full ideate
suite at this commit: **1202 passed in 230.56s, exit 0**.

## Deliverable 4 — docs (0b8f332)

`docs/IDEATE.md` § "Scratch databases and the pixi activation env": the three variables
`pixi run -e ideate*` pins over the caller's, the two ways to build a scratch database (the
env's interpreter directly with the variables exported, or `IDEATE_PATHS` with
`IDEATE_DATA_ROOT` taken out of the environment), the stderr root line as the way to check
before trusting a run, and exactly what the guard refuses and what `--force` records.

Red → green: `tests/ideate/test_docs_scratch_db.py` **4 failed → 4 passed** (red: no such
section). The tests tie the prose to the code — the pinned variables are read back out of
`pyproject.toml`'s `[tool.pixi.feature.ideate.target.unix.activation.env]`, and the origin
labels the section quotes are compared with what `config.data_root_origin()` returns — so the
section fails rather than drifts if either changes.

## Verification

Run from the worktree with `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src`:

| what | result |
| --- | --- |
| `HF_HUB_OFFLINE=1 pixi run ... -e ideate-cpu python -m pytest tests/ideate -q -W error -p no:cacheprovider` | `1206 passed in 253.28s (0:04:13)`, exit 0 |
| `pixi run ... -e labelmaker python -m pytest tests/labelmaker -q -W error -p no:cacheprovider` | `1580 passed, 2 skipped in 290.92s (0:04:50)`, exit 0 (the XRootD atexit FutureWarning prints after the summary) |
| `pixi run ... -e labelmaker ruff check src/ideate tests/ideate` | `All checks passed!`, exit 0 |

No writing command was run outside the suites' `tmp_path` fixtures;
`/scratch/gpfs/EKOLEMEN/nc1514/` was not touched; nothing was created under
`/scratch/gpfs/nc1514` outside this worktree; `recommender` was neither checked out nor
committed to.

## Deviations

1. **Deliverable 1 was not written here.** The previous implementer (Codex, cut off by its usage
   quota) committed it as 7e05cd8. This session reviewed it against the brief clause by clause
   (table above) and changed nothing, so there is no red → green of its own to report beyond the
   one its commit body states (11 failed / 2 passed → 81 passed in scope).
2. **Deliverable 2's test file was rewritten, not finished.** It was left untracked and never
   run. Its `origin == "default"` case drove a whole `cli build` with `config.CONFIG_DIR`
   monkeypatched at `tmp_path`, which would have redirected `labels.yaml`, `flags.yaml`,
   `evalsets/` and the shot lists as well — six other modules read `config.CONFIG_DIR`. The
   three origins are now unit tests of the accessor, and the command-level tests cover the two
   origins a real run has. Its `== [expected]` assertion on the whole of stderr was also wrong
   for `corpus select`, which legitimately prints "no logbook at ..." first in the fixture; the
   assertion is now on the first line.
3. **The IGNITE encode is not one of the six phases.** It already times itself into
   `manifest["ignite"]["elapsed_s"]`; counting it again inside `write_tables` would have made
   that phase the answer to every question on an encoding build. Recorded in `PHASES`'s comment
   and in the commit body.
4. **The published manifest is written twice** (once into the staging directory, once over the
   published copy through a renamed `.part` file), because the publish is the last phase and
   cannot appear in the manifest it is moving.
5. **Commit trailer.** Every commit ends with `Co-Authored-By: Claude Fable 5.1
   <noreply@anthropic.com>` as the brief binds, although this session ran on Claude Opus 5.
