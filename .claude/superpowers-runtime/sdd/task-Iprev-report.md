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

## Post-review fixes (2026-09-14)

Applied after the review of `3cced76..0a4eb9a` (`.superpowers/sdd/review-Iprev.md`, now tracked
here). Two commits, TDD throughout — every test below was written or extended first and seen red
for the stated reason.

```
80b7b8d ideate: total the build beside its phases, and survive a cross-device cache move
2885afc ideate: the default root origin names the paths file it actually read
```

### Finding 1 (MEDIUM) — the untimed residual is now visible from the manifest

The published manifest carries `build_elapsed_s` beside `phase_seconds`: the wall of `build()`
from entry to the moment that manifest is serialised, rounded to 6 dp like the phases. What the
six phases do not cover — `coverage_report`, the manifest construction, and on an encoding build
the whole `if encode:` block and the ignite table writes — is `build_elapsed_s -
sum(phase_seconds.values())`. The key is a sibling of `phase_seconds`, so the "keys are exactly
`PHASES`" assertion still holds; it is `None` in the first (staged) manifest, which a crash in
the rewrite window leaves behind together with the two zeroed phases the comment now names. The
log line is `build phases (wall s): …, total 12.3 (untimed 1.4)`.

`tests/ideate/test_build_phases.py`: **2 failed / 2 passed → 4 passed** (red: no
`build_elapsed_s` key; no total on the phase log line).

### Finding 2 (MEDIUM-LOW) — the default origin label names the file that was read

`config.data_root_origin()` returns `f"{CONFIG_DIR / 'paths.yaml'} default"` instead of the
literal `configs/ideate/paths.yaml default`. `CONFIG_DIR` follows `IDEATE_CONFIG_DIR`, so the old
label could name a file that settled nothing. `docs/IDEATE.md` now says
`<repo>/configs/ideate/paths.yaml default` and explains why the path is given in full; the drift
test derives the expected prose from what the function returns (label minus the repo root) rather
than repeating a literal.

The new `IDEATE_CONFIG_DIR` case copies the fixture `paths.yaml` into another directory, sets the
variable **and** `config.CONFIG_DIR` (the variable is read once, at import: setting it inside a
live process cannot move `CONFIG_DIR`), and pins the label against the file `load_paths` then
actually opens — `load_paths().data_root == paths.data_root`. The monkeypatch is narrow in the
sense deviation 2 of the original report was about: six other readers use `config.CONFIG_DIR`,
and this test calls only `data_root_origin` and `load_paths`, no command.

`tests/ideate/test_write_root.py`: **2 failed / 8 passed → 11 passed**.
`tests/ideate/test_docs_scratch_db.py`: **1 failed / 3 passed → 4 passed**.

### Finding 5 (LOW) — the staged text subset survives a cross-device move

`build._move_onto(src, dst)` keeps `os.replace` as the fast path (atomic, and both paths are on
one filesystem under the stock paths file) and falls back to `shutil.move` on `errno.EXDEV` only
— the case an `IDEATE_PATHS` file that puts `text_cache_dir` on another mount produces. The
fallback copies and so is not atomic, which is stated where it is written and is acceptable for a
cache the next build rewrites.

### Finding 3 (LOW) — `manifest.json.part` is pruned

One line: `"manifest.json.part"` is now in `BUILD_FILES`, so `_build_owned` claims an orphan left
by a crash between writing it and renaming it over `manifest.json`, and the next `_publish`
removes it. The finding's *second* half — wrapping the rewrite so an ENOSPC after a successful
publish logs instead of raising — was **not** done: it changes what a failed build reports to the
CLI, which is more than a cleanup and outside these fixes.

New module `tests/ideate/test_build_staging.py` covers both: **2 failed → 2 passed** (red: EXDEV
propagated out of `build`; `_publish` left the orphaned `.part` in place). The second test also
asserts a foreign file (`corpus_coverage.parquet`) is still untouched.

### Not changed

The guard semantics (`_check_publish`), the announcement format beyond the origin label, and
everything under `docs/superpowers/plans/**`. Review findings 4, 6, 7, 8 and 9 were out of scope
for this pass; 4 is addressed in prose only (the `phase_seconds` comment now names the two keys a
lost rewrite leaves at zero).

### Verification

Run from the worktree with `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src HF_HUB_OFFLINE=1`:

| what | result |
| --- | --- |
| `pytest test_publish_guard.py test_write_root.py test_build_phases.py test_docs_scratch_db.py test_build_staging.py -q -W error` | `34 passed in 8.23s`, exit 0 |
| `pytest tests/ideate -q -W error -p no:cacheprovider` | `1210 passed in 191.17s (0:03:11)`, exit 0 |
| `ruff check src/ideate tests/ideate` (env `labelmaker`) | `All checks passed!`, exit 0 |

`tests/ideate` went 1206 → 1210 (+1 phases, +1 write_root, +2 staging). `tests/labelmaker` was
not re-run: nothing under it was touched by these fixes. No `ideate` writing command was run
outside the suites' `tmp_path` fixtures, `/scratch/gpfs/EKOLEMEN/` was not touched, nothing was
created under `/scratch/gpfs/nc1514` outside this worktree, and `recommender` was neither checked
out nor committed to.
