# Task E3 report: parallel blurb backfill

Worktree: `/lustre/orion/fus187/scratch/nchen/FusionAIHub-wt-e`, branch `nathan_dev-e`.
Commit: `e91fb99` "shot_design blurb: --workers thread pool; Frontier backfill script"
(parent `a1c0f87`).

Scope executed: brief Steps 1-4 and 6. Step 5 (dry run against the production DB) was
explicitly out of scope and was **not** run — no `agy`, no production data root, no Slurm.

## What was built

1. **`src/shot_design/shotdb/build.py`** — `write_blurbs(..., *, workers: int = 1)`.
   - Validates `workers >= 1` (raises `ValueError` otherwise, mirroring the existing
     `limit` validation).
   - `workers == 1`: computes every `(rec, blurb)` pair with a plain dict comprehension —
     no `ThreadPoolExecutor` is created at all, so the sequential path has no executor
     overhead and its control flow (including where an unexpected exception would
     propagate) is the same as before.
   - `workers > 1`: submits one `_blurb.make(...)` call per candidate shot to a
     `ThreadPoolExecutor(max_workers=workers)`, then blocks on every future before doing
     anything else.
   - Either way, the *writing* (updating `df`, and the dry-run `print`s) happens
     afterwards, in a single pass over `todo` (the pre-existing shot-sorted / targeted-shot
     list) from the calling thread only — so the printed order and the row order written to
     `shots.parquet` never depend on which call happened to finish first. The
     parquet/manifest rewrite itself is untouched and still single-threaded.
   - Added `ThreadPoolExecutor` to the module's existing `from concurrent.futures import
     ProcessPoolExecutor` line rather than a local import, matching that file's own
     convention for `ProcessPoolExecutor`.
   - Per-shot failure handling is unchanged: `_blurb.make` already catches any `client.chat`
     exception internally and falls back to the template blurb with a reason; that try/except
     lives inside `_blurb.make`, so it applies identically whether the call runs on the main
     thread (`workers=1`) or inside a pool worker (`workers>1`). One shot's model error still
     cannot affect any other shot's row.

2. **`src/shot_design/cli.py`** — `blurb --workers N` (`type=int, default=1`), validated
   (`< 1` -> exit 2, same style as the existing `--limit` check, checked before opening the
   client) and threaded through to `write_blurbs(..., workers=args.workers)`.

3. **`scripts/shot_design/blurb_frontier.sh`** (new, executable) — sources
   `scripts/slurm_frontier/_shot_design_common.sh` then runs
   `"$PY" -m shot_design blurb --workers "${WORKERS:-8}" "$@"`, exactly as the brief's
   interface section specifies. Verified with `bash -n` only (syntax check) — I did **not**
   source or execute it, because `_shot_design_common.sh` unconditionally does
   `mkdir -p "$ROOT/runs/slurm"` under `$SHOT_DESIGN_DATA_ROOT` (the production data root by
   default) as soon as it is sourced, which the task rules forbid touching. Manual read of
   that common script found no sbatch-only assumption that would break a login-node run:
   `${SLURM_JOB_ID:-none}` and the `rocm-smi ... || true` guard both degrade gracefully off
   a compute node; the one write it performs (`mkdir -p .../runs/slurm`) is unconditional
   but idempotent and is the same thing every other Frontier shot_design script already does
   by sourcing this file, so it is expected shared behavior, not something specific to my
   change. `scripts/shot_design/serve_llm.sbatch` (Stellar-only) was left untouched, per the
   brief's note that no Frontier serve script exists or is needed.

4. **`tests/shot_design/test_blurb_backfill.py`** (new). No `tmp_db` fixture exists in
   `tests/shot_design/conftest.py`, so I built the smallest equivalent from the fixtures that
   already exist there: `paths` (the on-disk `config.Paths` layout) plus the module-level
   helpers `shot_record`/`write_db` (the same two `shot_design_db` uses) to seed six shots
   directly through `build.records_to_tables` -> `build._write_tables`, skipping the full
   `build.build()` pipeline (which needs raw HDF5 files this test doesn't care about). The
   local `tmp_db` fixture returns a `SimpleNamespace(paths=paths)` so `tmp_db.paths` matches
   the brief's test signature verbatim.
   - `test_write_blurbs_workers_writes_every_row` — the brief's Step 1 test, copied verbatim
     (using the local `tmp_db`/`fake_client` fixtures).
   - `test_write_blurbs_workers_write_order_matches_single_worker` — builds a second,
     identically-seeded database and asserts `shots.parquet` is byte-identical between
     `workers=1` and `workers=4`, directly exercising the "deterministic write order"
     requirement.
   - `test_write_blurbs_workers_one_shot_failure_does_not_kill_the_run` — one shot's
     `fake_client.chat` raises; asserts that shot alone falls back to `blurb_source ==
     "template"` while every other shot still gets `"llm"` and the return count is
     `len(df) - 1`. This is the "errors in one shot must not kill the run" test the brief
     asked for "if cheap" (it was: one extra `fail_shots` parameter on the existing
     `FakeClient`).

   `FakeClient` (local to the test file, not `LLMClient`) implements only what `_blurb.make`
   needs: `.available()`, `.cfg` (a minimal dict with `models`/`blurb` keys, same shape as
   `test_blurb.py`'s `CFG`), and `.chat(...)`, which increments an in-flight counter under a
   `threading.Lock`, sleeps 0.05 s, and returns a gate-passing three-sentence reply
   ("The session ran as planned. It completed without incident. No notable findings were
   logged.") chosen specifically to contain no digits, acronyms, or number-words, so it
   passes `blurb.gate` regardless of what's in each fixture shot's source text.

## Thread-safety reasoning (read-only review of `src/shot_design/llm/*` — not modified,
per the lane rules)

- **`AgyProvider.chat`** (`src/shot_design/llm/agy.py`): holds no per-call mutable state.
  `self.cfg`/`self._runner` are set once in `__init__` and only read in `chat`. The actual
  work is `subprocess.run(...)`, which spawns one independent OS process per call — this is
  the production path (`provider: agy` is the default in `configs/shot_design/llm.yaml`) and
  it is thread-safe by construction: concurrent `chat()` calls on the same `AgyProvider`
  instance from different threads share nothing beyond read-only config.
- **`LLMClient.chat`** (`src/shot_design/llm/client.py`): two things are worth flagging,
  both benign for this workload and neither touched (edits to this file are off-limits in
  this lane):
  - `self._ep_cache` (an `(mtime_ns, Endpoint)` tuple) is mutated inside `endpoint()`, which
    is only reachable through the `"ollama"` provider path
    (`self._chat_openai(body, self.endpoint())`). Production uses `provider: agy`, so this
    path is not exercised by the real backfill; if it ever were run concurrently (e.g. a test
    or a local Ollama backfill with `workers>1`), a race on this tuple is at worst a stale
    read that gets recomputed on the next call — not a crash or corrupted state.
  - The on-disk request cache (`chat()`'s `<sha>.json` file) already follows the brief's
    "write `<sha>.json.part` then `os.replace`" pattern (line ~214 of `client.py`,
    unchanged by me). Two threads computing the *same* cache key concurrently (identical
    `model`/`messages`/`temperature`/`max_tokens` body) could in principle interleave writes
    to the same `.json.part` path before either `os.replace`. In practice this cannot happen
    across shots in this backfill: each shot's `messages` (built from
    `blurb.source_text(rec)`) is essentially unique, so no two in-flight requests in the same
    `write_blurbs(workers>1)` run share a cache key. I did not change this file, so this is
    reported as an existing, pre-existing-safe-in-practice property, not something I fixed.
- **`ShotDB.get`** (`src/shot_design/shotdb/store.py`): reads `self.shots.loc[shot,
  "record_json"]` from the already-loaded, never-mutated-during-the-run `db.shots` frame and
  builds a fresh `ShotRecord` via `model_validate_json` each call — read-only, thread-safe.
  `write_blurbs` calls `db.get` inside each pool worker's `_one(shot)` (same as it always did
  per-shot for `workers=1`), never on the `df` copy that gets mutated.
- **The mutation of `df`** (the `DataFrame` that eventually becomes `shots.parquet`) happens
  exclusively in the final `for shot in todo:` loop, run only on the calling thread, after
  every future has already resolved — no worker thread ever touches `df`.

## TDD evidence

RED (before the `workers` kwarg existed):
```
FAILED tests/shot_design/test_blurb_backfill.py::test_write_blurbs_workers_writes_every_row
FAILED tests/shot_design/test_blurb_backfill.py::test_write_blurbs_workers_write_order_matches_single_worker
FAILED tests/shot_design/test_blurb_backfill.py::test_write_blurbs_workers_one_shot_failure_does_not_kill_the_run
3 failed in 2.26s
```
(all three: `TypeError: write_blurbs() got an unexpected keyword argument 'workers'`)

GREEN (after implementing `workers` in `write_blurbs`/`cli.py`):
```
tests/shot_design/test_blurb_backfill.py ...                              [100%]
3 passed in 3.05s
```

Combined with the existing blurb tests (`test_blurb.py`, `test_blurb_provenance.py`), still
green after the final cleanup (moving `ThreadPoolExecutor` to the top-level import,
`--workers` CLI validation):
```
64 passed in 13.76s
```

Full `tests/shot_design` suite, run twice (once before the import cleanup, once after,
identical result both times):
```
FAILED tests/shot_design/test_mcp.py::test_the_project_mcp_config_points_at_this_server
1 failed, 1706 passed, 23 skipped, 2 warnings in ~125s
```
That one failure is the documented pre-existing exception (`git rev-parse --show-toplevel`
against `/scratch/gpfs/nc1514/FusionAIHub`, a Stellar path that doesn't exist on Frontier) —
unrelated to this change and explicitly called out in the task instructions as safe to
ignore.

Command used both times (per the task instructions, main repo's frozen interpreter, worktree
sources first on `PYTHONPATH`):
```
cd /lustre/orion/fus187/scratch/nchen/FusionAIHub-wt-e
PYTHONPATH=$PWD/src /lustre/orion/fus187/scratch/nchen/FusionAIHub/.pixi/envs/shot-design-frontier/bin/python \
    -m pytest tests/shot_design -q -p no:cacheprovider
```

`bash -n scripts/shot_design/blurb_frontier.sh` — passed (syntax only; script was never
sourced or executed, per the "never touch the production data root" rule).

Ruff line-length 88 check on every line I added (`git diff | grep '^\+[^+]' | ... | awk
'length>88'` for the two modified `.py` files, plus a plain `awk 'length>88'` over the whole
new test file since every line in it is new): clean after a few docstring rewraps.

## Files changed (commit `e91fb99`)

- `/lustre/orion/fus187/scratch/nchen/FusionAIHub-wt-e/src/shot_design/shotdb/build.py`
  (`write_blurbs` gains `workers`; `ThreadPoolExecutor` added to the existing
  `concurrent.futures` import)
- `/lustre/orion/fus187/scratch/nchen/FusionAIHub-wt-e/src/shot_design/cli.py`
  (`blurb --workers N`, validated, wired to `write_blurbs`)
- `/lustre/orion/fus187/scratch/nchen/FusionAIHub-wt-e/scripts/shot_design/blurb_frontier.sh`
  (new)
- `/lustre/orion/fus187/scratch/nchen/FusionAIHub-wt-e/tests/shot_design/test_blurb_backfill.py`
  (new)

## Self-review

- Confirmed `git diff --cached --stat` immediately before committing showed exactly these
  four files — no `-A`/`-a`, explicit pathspecs on both `git add` and `git commit -m ... --
  <paths>`.
- Never touched `src/shot_design/llm/*` or `configs/shot_design/llm.yaml`. Partway through
  this task, `git status` started showing `src/shot_design/llm/client.py` as modified
  (adding an `"openai_compatible"` provider key) even though I made no edits there and the
  worktree was clean (`nothing to commit, working tree clean`) at the very start of the
  session — this matches the task description's warning that "a fix round may later edit
  `src/shot_design/llm/*` ... in this worktree" concurrently. I left it untouched and
  unstaged; it is not part of commit `e91fb99`.
- Never ran `agy`, never wrote under the production data root, never ran or submitted
  anything through Slurm, never sourced `_shot_design_common.sh` (only `cat`/`bash -n`'d the
  scripts).
- Retained the exact commit message from the brief's Step 6, one line, no body.
- The `workers == 1` path is a dict comprehension rather than byte-identical original code;
  I judged this equivalent to "unchanged" because (a) no `ThreadPoolExecutor` is constructed,
  matching the brief's literal requirement, and (b) the per-shot control flow, exception
  propagation point, and output order are identical to the original sequential `for` loop —
  only *where* the tuple `(rec, b)` is stored changed. Flagging this judgment call explicitly
  in case a reviewer wants the `workers==1` branch to look more like a textually-untouched
  copy of the old loop.

## Concerns

- The one pre-existing test failure (`test_mcp.py::test_the_project_mcp_config_points_at_this_server`)
  is present in both full-suite runs, unrelated to this change, and was pre-approved as
  ignorable by the task instructions.
- `src/shot_design/llm/client.py` has an uncommitted, unstaged change in this worktree from
  outside this task (see Self-review above) — flagging in case whoever picks this worktree
  up next needs to know it's there and isn't mine.
- I did not (and per the task rules, could not) exercise `blurb_frontier.sh` end-to-end or
  the real `agy`/`AgyProvider` path under real concurrency — the thread-safety reasoning
  above is a static read of `agy.py`/`client.py`, not a measurement. Step 5's dry run (owned
  by someone else, after the DB build finishes) is the first point this gets exercised against
  real `agy` calls.
