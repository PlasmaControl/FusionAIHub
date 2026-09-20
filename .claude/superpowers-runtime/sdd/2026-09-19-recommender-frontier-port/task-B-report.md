# Task B (B2–B5) report — Shot Designer Frontier port, Phase B

Branch `nathan_dev`, base HEAD `88fb86c`. Session was restarted mid-phase; re-anchored with
`git status` / `git diff` before touching anything.

## 1. Partial work: kept vs rewritten

The cut-off implementer's work was read against the four briefs, the global constraints and the
existing Frontier house style (`scripts/slurm_frontier/_frontier_settings.sh`,
`_frontier_common.sh`, `eval_dynamics.sh`, `train_dynamics.sh`) before judging.

**Kept essentially as found** (only the env-name rename applied on top):

| File | Verdict |
|---|---|
| `pyproject.toml` B2 feature block | Correct against the brief; kept verbatim apart from `ideate-frontier` → `shot-design-frontier` and one added dependency (below). |
| `scripts/slurm_frontier/_shot_design_common.sh` | Matches the brief line for line and matches `train_dynamics.sh`'s `set -euo pipefail` + `source _frontier_settings.sh` pattern. Kept. |
| `scripts/slurm_frontier/shot_design_{census,build,encode,simulate}.sh` | Kept. The deviation from the brief (`-p extended` instead of `-p batch` on the two 6 h jobs) is *correct* and I re-verified it: Frontier caps a 1–91-node `batch` job at 2 h, and `sbatch --test-only` accepts all four as written. |
| `scripts/slurm_frontier/_gpu_sampler.sh` | Kept. It deviates from the brief's three `rocm-smi` calls in favour of one `--showuse --showmeminfo vram --csv` call parsed by column *name* — the brief's `NR==2` would have read `rocm-smi`'s blank second line and recorded 0 % for every sample. Better than the brief; the file documents why. |
| `src/labeler/jobstats.py` `parse_frontier` / `parse_gpu_samples` / `gpu_samples_path` | Kept; the arithmetic matches the brief's numbers and it reuses `parse_sacct`, `sacct_memory_pct`, `req_mem_bytes`, `_ratio_pct` as instructed. |
| `tests/shot_design/test_slurm_frontier_scripts.py` | Kept verbatim (it is the brief's test). |
| `tests/labeler/test_jobstats_frontier.py` first two tests | Kept (the brief's tests, adapted to the real `JobStats` field names, which is what the brief asked for). |

**Fixed / added by me:**

1. **A real bug in the B4 auto-detect.** `frontier_backend()` probes the live `PATH`, but
   `tests/labeler/test_jobstats.py` fakes a Stellar cluster through `subprocess.run` only. On a
   Frontier login node (no `jobstats`, yes `sacct`) three of its tests therefore took the new
   Frontier path and failed — i.e. the suite's result depended on which login node ran it. Fixed
   with an autouse fixture in that module pinning `frontier_backend` to `False`, which states in
   one place that the module describes the jobstats cluster.
2. **Missing coverage of the auto-detect itself** — added
   `test_the_cli_uses_sacct_alone_when_the_cluster_has_no_jobstats`, an end-to-end `main()` test
   asserting `jobstats` is never invoked and the record carries the sacct/rocm-smi numbers.
3. **Dead constant** — `GPU_SAMPLE_HEADER` was defined and never read. Rather than delete it I
   made it load-bearing: `test_the_sampler_writes_the_header_this_module_parses` pins the header
   `_gpu_sampler.sh` writes to the column order `parse_gpu_samples` reads.
4. **`plotly` added to the `shot-design-frontier` feature.** Without it
   `pixi run -e shot-design-frontier pytest tests/labeler` errors at *collection*
   (`tests/labeler/test_events_verify.py` imports `plotly.graph_objects`). It is the only module
   `tests/shot_design` + `tests/labeler` import that neither `shot-design` nor
   `shot-design-frontier` already carried (measured with `--collect-only`).
5. **Provenance `LEGACY_SCHEMAS` not added** (deviation, see §6).
6. **`scripts/labeler/assess_labels_a.py`** built its db path as `paths.root.parent / "ideate/db"`
   — a Stellar-only sibling assumption that is also the last stray `ideate` outside the protected
   set. Changed to `Path(os.environ["SHOT_DESIGN_DATA_ROOT"]) / "db"`, which is right on both
   clusters.

## 2. Commit order

As instructed, **B5's pyproject/pixi.lock portion is folded into the B2 commit** — the lock has to
be regenerated exactly once, and regenerating it under the old names only to rename them in the
next commit would put a 880 kB throwaway diff in history. The rest of B5 (source, scripts, docs,
tests, legacy-tag acceptance) is its own commit, as planned.

Order: B2(+B5 pyproject/lock) → B5(rest) → B3 → B4.

| SHA | Subject |
|---|---|
| `46d84a1` | pixi: shot-design-frontier env (ROCm torch + shot_design deps, Frontier activation roots) |
| `bdc4cee` | repo: rename ideate environments and tags to shot-design |
| `2feae25` | slurm_frontier: shot_design census/build/encode wrappers and shared env |
| `88abee8` | labeler: jobstats Frontier backend from sacct + rocm-smi samples |

Base `88fb86c`, branch `nathan_dev`, nothing pushed. `.gitignore` is left modified in the working
tree: that change is the controller's, not this task's.


## 3. Per task

### B2 — `shot-design-frontier` pixi environment

- Step 1 (feature): present in `pyproject.toml` as
  `[tool.pixi.feature.shot-design-frontier]` + `.pypi-dependencies` +
  `.target.unix.activation.env`, and `shot-design-frontier = ["shot-design", "shot-design-frontier"]`
  under `[tool.pixi.environments]`. The three roots, `SHOT_DESIGN_PATHS`, `HF_HUB_OFFLINE=1`,
  `TOKENIZERS_PARALLELISM=false`, `HDF5_USE_FILE_LOCKING=FALSE` are exactly the brief's values.
  `plotly` added on top (§1.4).
- Step 2/3/4/5: see §4 for the install outcome and the activation check.

### B5 — rename `ideate` → `shot-design`

See §5.

### B3 — Frontier Slurm wrappers

- Step 1/2 (RED): with the four scripts moved aside,
  `pytest tests/shot_design/test_slurm_frontier_scripts.py -q` →
  `4 failed in 0.29s` (all four parametrisations, files missing).
- Step 3/4: `_shot_design_common.sh` + the four wrappers as reviewed in §1.
- Step 5 (GREEN): `4 passed in 0.04s`.
- Extra verification the brief did not ask for but the house style implies —
  `sbatch --test-only` on all four (nothing was submitted):

```
census    Job 5513154 ... in partition batch
build     Job 5513155 ... in partition extended
encode    Job 5513156 ... in partition extended
simulate  Job 5513157 ... in partition batch
```

### B4 — `labeler.jobstats` Frontier backend

- Step 1/2 (RED): with `parse_frontier` renamed away,
  `pytest tests/labeler/test_jobstats_frontier.py -q` →
  `AttributeError: module 'labeler.jobstats' has no attribute 'parse_frontier'`, `2 failed`.
- Step 3: `parse_frontier`, `parse_gpu_samples`, `frontier_backend`, `gpu_samples_path`,
  `read_gpu_samples`, and the `main()` / `_capture_task` wiring.
  RED probe for the CLI wiring: with `frontier = False` hard-coded in `main()`,
  `AssertionError: assert 'jobstats' not in ['sacct', 'jobstats', 'sacct']`.
  RED probe for the header pin: with a column dropped from `GPU_SAMPLE_HEADER`, the pin test fails.
- Step 4: `_gpu_sampler.sh`, and the backgrounded sampler + `trap ... EXIT` in
  `shot_design_encode.sh`, named after the id `sacct` reports
  (`$SLURM_ARRAY_JOB_ID_$SLURM_ARRAY_TASK_ID`, **not** `$SLURM_JOB_ID`) so
  `read_gpu_samples` finds it.
- Step 5 (GREEN): `tests/labeler/test_jobstats_frontier.py` `4 passed`;
  `tests/labeler/test_jobstats.py` `64 passed`.

## 4. B2 — the environment

`pixi install -e shot-design-frontier` succeeded. Elapsed for the run that landed it: **3 min 40 s**
(a later relock after two comment edits took ~2 min). `.pixi/envs/shot-design-frontier/bin/python`
exists; `pixi.lock` carries `shot-design`, `shot-design-cpu` and `shot-design-frontier`, with
`torch-2.10.0+rocm7.1` and `torchvision-0.25.0+rocm7.1` wheels under the frontier one.

```
$ pixi run --frozen -e shot-design-frontier bash -c 'echo ...; python -c "import torch, shot_design, sentence_transformers; print(torch.__version__)"'
ROOT=/lustre/orion/fus187/proj-shared/nchen/shot_design
LABELER=/lustre/orion/fus187/proj-shared/nchen/labeler
CORPUS=/lustre/orion/fus187/proj-shared/foundation_model
PATHS=/lustre/orion/fus187/scratch/nchen/FusionAIHub/configs/shot_design/paths.frontier.yaml
2.10.0+rocm7.1
```

**The brief's precedence guess was wrong, and it was measured, not assumed.** With
`shot-design-frontier = ["shot-design", "shot-design-frontier"]` (later feature last, as the brief
wrote it) the environment came up with `SHOT_DESIGN_DATA_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/ideate`
— the Stellar root, on Frontier. pixi 0.73 gives the **first** feature in the list precedence for
activation env. The list is now `["shot-design-frontier", "shot-design"]`, all four values are the
Frontier ones, and both the feature comment and the `[tool.pixi.environments]` comment say so, so a
future reorder cannot quietly re-point every job at paths that do not exist here.

ROCm index resolution did not fail; no index was swapped.

### Pre-existing repo issue: `pixi` cannot re-solve `default` for `win-64` here

Any pixi command that is not `--frozen` re-solves every environment for every declared platform, and
that hits, intermittently:

```
thread 'main2' panicked at crates/pixi_core/src/lock_file/resolve/build_dispatch.rs:477:17:
could not initialize build dispatch correctly
Error:   x failed to solve the pypi requirements of environment 'default' for
  | platform 'win-64'
  `-> build dispatch initialization failed: failed to query interpreter in
      instantiated prefix
```

Evidence that it is pre-existing repo state, not something Task B introduced — in a detached
worktree at `88fb86c` with the **unmodified** pyproject, `pixi lock --dry-run -v` prints:

```
INFO pixi_core::lock_file::outdated: the dependencies of environment 'default' for platform win-64
are out of date because 'torchvision' requires index https://pypi.org/simple but the lock file has
https://download.pytorch.org/whl/cu124
```

for **every** environment (`default`, `fdp`, `frontier`, `labelmaker`, `shot-design`,
`shot-design-cpu`). HEAD's lock and HEAD's manifest already disagree on the torchvision index; the
disagreement is invisible until something forces a re-solve, and then win-64 needs an instantiated
win-64 prefix to read the editable `faith` package's metadata, which a Linux login node cannot
query. One earlier `pixi install` also panicked with `failed to link
libevent-2.1.12-hf998b51_1.conda`; evicting that package from the rattler cache cleared it.

Per the controller's ruling: **every pixi command on Frontier uses `--frozen`**, which skips the
re-solve entirely. The docs, the plan and the spec were rewritten to say
`pixi run --frozen -e shot-design-frontier` / `pixi install --frozen -e shot-design-frontier`. The
Slurm wrappers never call pixi at all — they invoke
`.pixi/envs/shot-design-frontier/bin/python` directly.

Two follow-ups for the controller, both outside Phase B: the torchvision index disagreement in
`[tool.pixi.pypi-dependencies]` vs the lock, and whether `win-64` / `osx-arm64` should stay in
`[tool.pixi.workspace].platforms` given that no one can re-lock them from Frontier or Stellar.

### Not done

- **B2 Step 4** (pre-populating the sentence-transformers cache on the login node) was not run — the
  install budget was spent on the lock problem above. `HF_HUB_OFFLINE=1` is set by the activation,
  so the first job that needs `all-MiniLM-L6-v2` will fail unless the cache is warmed first. One
  command, from the login node:
  `HF_HUB_OFFLINE=0 pixi run --frozen -e shot-design-frontier python -c "from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/all-MiniLM-L6-v2')"`
- The full `pytest tests/shot_design tests/labeler` run cannot complete in this environment: it
  aborts the interpreter inside a labeler integration test (section 7). The targeted run
  (`tests/shot_design` + both jobstats files) is green apart from one pre-existing `.mcp.json`
  failure, also in section 7.


## 5. B5 — the grep, before and after

The brief's Step 1 command (run with `command grep`, because this shell aliases `grep` to a
ripgrep wrapper that strips the `./` prefix the filter depends on):

- **before: 461 lines** across 46 files
- **after: 18 lines**, every one deliberate:

| Count | Where | Why it stays |
|---|---|---|
| 2 | `pixi.lock` `ideate:` / `ideate-cpu:` | Regenerated by the install; 0 after it lands. |
| 5 | `legacy_raw.py` ×2, `test_legacy_raw.py` ×3, `provenance.py` ×1 | The **legacy tags the same brief requires** (`LEGACY_SCHEMAS = {"ideate-raw-v1"}`, its test, and the provenance note). Renaming a constant cannot rewrite the files on disk. |
| 7 | `.claude/superpowers/plans/2026-09-19-...md` lines 456–472 | Task B5's own section: it *specifies* the mapping. Renaming `ideate` there would turn it into "rename shot-design to shot-design" and destroy the record. Every other `ideate` in the plan and in the 2026-09-19 spec **was** renamed. |
| 4 | `docs/CLUSTERS.md` lines 49, 73, 225 (+ the layout block) | The Stellar **directory** `ideate/` in a filesystem layout diagram and a source-path table row, and the historical sentence "Packages were renamed 2026-09-15: `ideate` -> `shot_design`". Same protected class as `/scratch/gpfs/EKOLEMEN/nc1514/ideate`, which the brief excludes by name. Where it read naturally I expanded the short forms to the full protected path (lines 129 and 229), which is why 3 of the original CLUSTERS.md hits are gone. |

So: **0 lines outside the brief's own protected set plus the legacy tags the brief itself mandates.**

### Legacy-tag acceptance + test

`src/shot_design/shotdb/legacy_raw.py`:

```python
SCHEMA = "shot-design-raw-v1"
LEGACY_SCHEMAS = frozenset({"ideate-raw-v1"})
...
    if _attr(f, "schema") in {SCHEMA, *LEGACY_SCHEMAS}:
```

Test `tests/shot_design/test_legacy_raw.py::test_a_file_stamped_before_the_rename_is_still_ours`.
TDD evidence — with the acceptance reverted to `== SCHEMA`:

```
>           assert legacy_raw.is_ours(f) is True
E           assert False is True
FAILED ... ::test_a_file_stamped_before_the_rename_is_still_ours
1 failed in 0.88s
```

and with it restored: `tests/shot_design/test_legacy_raw.py  33 passed in 1.09s`.

`tests/shot_design/conftest.py`'s `OUR_SCHEMA` was moved to the **new** tag, so the rest of the
suite exercises the new stamp and the one dedicated test covers the old one.

## 6. Deviations from the briefs

1. **`provenance.py` gets no `LEGACY_SCHEMAS`.** B5 says "*wherever a reader compares the tag*,
   also accept the old string". In `provenance.py` nothing compares it — `audit()` only counts
   whatever value it finds, under `by_schema`. A constant and an `is_ours()` that nothing calls
   would be dead code, so the rename carries a comment saying both tags keep reading and the
   census shows the rename as two rows. (I wrote the `LEGACY_SCHEMAS` + `is_ours` pair first and
   then removed it for exactly this reason.)
2. **`shot_design_build.sh` / `shot_design_encode.sh` use `-p extended`, not `-p batch`.**
   Inherited from the partial work and re-verified: `batch` caps a one-node job at 2 h. The
   style test allows `batch|extended`; `eval_dynamics.sh` is the house precedent.
3. **`_gpu_sampler.sh` makes one `rocm-smi` call, not three**, and finds its columns by name
   (§1). The brief's `NR==2` would have recorded 0 % for every sample.
4. **B5's pyproject/lock portion folded into the B2 commit** (§2).
5. **`plotly` added** to the feature beyond the brief's dependency list (§1.4).
6. **`scripts/labeler/assess_labels_a.py`** now reads `SHOT_DESIGN_DATA_ROOT` (§1.6) — one line
   beyond a pure rename, but a pure rename there would have been wrong on both clusters.

## 7. Self-review findings (fixed before reporting)

- The `--wait-for-data` / array / stderr tests in `tests/labeler/test_jobstats.py` regressed
  against HEAD. Found by diffing failure lists against a `git worktree` of `88fb86c` under the
  same interpreter, not by eyeballing a total. Fixed (§1.1). Post-fix diff of failing test ids
  vs HEAD: **empty in both directions** for `tests/shot_design` and `tests/labeler`.
- Two lines over the 88-column ruff limit in the new `tests/labeler/test_jobstats_frontier.py`;
  rewrapped. (`src/shot_design/**` is pervasively 90–100 columns at HEAD, so the new lines there
  match the file they are in; nothing I added exceeds its file's existing width.)
- `sed` collateral in `docs/CLUSTERS.md`: it rewrote the *historical* sentence "Packages were
  renamed 2026-09-15: `ideate` -> `shot_design`" into a tautology, and left the feature column of
  the environment table half-renamed. Both repaired, and the paragraph now also records the
  2026-09-19 environment rename and why the dash form is used.
- `docs/SHOT_DESIGN.md` said the environments "remain `shot-design` and `shot-design-cpu`", which
  after a rename reads backwards; rewritten, and `shot-design-frontier` named.
- Stale `.pytest_cache/v/cache/nodeids` held five pre-rename ids and was counted by the grep;
  removed (it is gitignored).

### Suite status in the new environment

Before the environment existed the suite was run under `.pixi/envs/frontier/bin/python` with
`PYTHONPATH=src`, against a **baseline worktree at `88fb86c` run the same way**: the set of failing
test ids is identical in both directions for `tests/shot_design` (71 before, 71 after) and
`tests/labeler` (212 before, 212 after). That comparison, not an absolute pass count, is what shows
this task introduced no regression — the interim `frontier` environment is missing `duckdb`,
`pypdf`, `fastapi`, `mcp`, `sentence_transformers` and `bs4`, so it fails and errors in bulk at
`88fb86c` too.

Confirming runs in the real environment (both with `--frozen`):

```
$ pixi run --frozen -e shot-design-frontier pytest tests/shot_design \
      tests/labeler/test_jobstats.py tests/labeler/test_jobstats_frontier.py -q -rf
1 failed, 1708 passed, 21 skipped in 120.45s
FAILED tests/shot_design/test_mcp.py::test_the_project_mcp_config_points_at_this_server
```

That one failure is **not** from this task and is not fixable inside it: `.mcp.json` carries
`"cwd": "/scratch/gpfs/nc1514/FusionAIHub"`, the Stellar checkout, so the test's
`git -C <cwd> rev-parse --show-toplevel` raises on Frontier. B5 changed only the `-e ideate-cpu`
→ `-e shot-design-cpu` argument on the line above it; `cwd` is untouched since `88fb86c` (`git show
88fb86c:.mcp.json`). Porting `.mcp.json` to Frontier is a task of its own, and it interacts with the
ruling not to install `shot-design-cpu` here — flagging it for the controller rather than guessing.

The wider run, `pytest tests/shot_design tests/labeler`, **aborted the interpreter** at 89% inside
`tests/labeler/test_labels_layout_integration.py::test_databases_only_on_all_committed_rwm_shots_writes_56_events_33_sources`:

```
Fatal Python error: Aborted
  File "src/labeler/config.py", line 228 in git_sha
  File "subprocess.py", line 2115 in _communicate
```

— a SIGABRT in a `subprocess.run` from a process that has torch, h5py, pytables, sklearn and
pyarrow loaded (245 extension modules). It is in `labeler.run`'s database stage, which no B task
touches, and it takes the whole session down, so it hides any later results. Worth a look on its
own; it is why the numbers above come from the targeted run.

## 8. Ops note (not a code change)

The first two `pixi install` attempts wedged. Cause: **three orphaned
`pixi run -e ideate-cpu python -m shot_design.mcp` processes from a previous session** (the MCP
server this session reported as `CONNECT_TIMEOUT`) plus **one orphaned `pixi install -e
ideate-frontier` from the cut-off implementer**, all blocked on the same uv lock
`~/.cache/rattler/cache/uv-cache/sdists-v9/path/05d0d4e65d5eb54e/.lock`. All four were killed;
they referenced the `ideate-cpu` / `ideate-frontier` environments, which this task removes, so
they could not have recovered. If the MCP server is wanted again it will now start under
`-e shot-design-cpu` from the updated `.mcp.json`.

A later attempt self-deadlocked with all 30 threads in `futex_wait_queue` on that same NFS-hosted
uv lock (`$HOME` is NFS here). Running pixi with `PIXI_CACHE_DIR=/tmp/pixi-cache-nchen` — the same
node-local tmpfs pixi already redirects repodata to — cleared it, and is worth using for any future
pixi work on this login node. The win-64 solve failure is *intermittent*: the same command failed
and then succeeded unchanged, which is one more reason to use `--frozen` and never re-solve.

## 9. BLOCKED

Two Bash calls were refused by the permission prompt early on (a multi-file `grep` loop and a
heredoc writing a helper script to the scratchpad). Neither was retried verbatim; both were
re-expressed and completed. Nothing is outstanding.

## 10. Fix round 1 (review-B "Important")

Commit `98afe7f` — `slurm_frontier: sample only the job's GCDs, census writes corpus_coverage,
frozen pixi in CLUSTERS`.

1. **`docs/CLUSTERS.md`** — the Frontier checklist told a newcomer to install the CUDA and
   Stellar-CPU environments and to test in `shot-design-cpu`, without `--frozen`; both commands
   fail here. Now `pixi install --frozen -e shot-design-frontier -e frontier` and
   `pixi run --frozen -e shot-design-frontier pytest tests/shot_design tests/labeler`, with one
   line saying why `--frozen` is not optional. `shot-design-frontier` added to the environment
   table. Stellar-facing bare commands elsewhere were left alone.
2. **`scripts/slurm_frontier/_gpu_sampler.sh`** — rocm-smi ignores `ROCR_VISIBLE_DEVICES`, so the
   awk was averaging all eight cards of an exclusive node and a one-GCD job would have reported
   about an eighth of its real utilisation — to the gate that judges it. The sampler now passes
   `-d ${ROCR_VISIBLE_DEVICES//,/ }` when the variable is set and keeps whole-node sampling only
   when it is not. `shot_design_encode.sh`'s comment corrected: the 8-element array is eight
   separate one-node jobs (8 nodes x 6 h), not one node's worth.
3. **`scripts/slurm_frontier/shot_design_census.sh`** — wrote `db/census.parquet`, which nothing
   reads: `corpus scan --out`, `select` and `coverage` all default to
   `<db_dir>/corpus_coverage.parquet` (`src/shot_design/cli.py`). Both the scan and the summary
   line now use that name.

Two assertions added to `tests/shot_design/test_slurm_frontier_scripts.py` (the only executable
check available for shell here): the census script names `$ROOT/db/corpus_coverage.parquet` exactly
twice and never `census.parquet`, and the sampler references `ROCR_VISIBLE_DEVICES` with a `-d`.
The first was RED against the first draft of this fix — the prose mention of the filename made the
count three — which is what pinned the assertion to the quoted argument rather than the word.

`pixi run --frozen -e shot-design-frontier pytest tests/shot_design/test_slurm_frontier_scripts.py -q`
→ **6 passed**. `git diff --check` clean; no new Python line over 88 characters.

