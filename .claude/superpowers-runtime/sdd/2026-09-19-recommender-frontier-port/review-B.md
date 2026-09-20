# Review — Task B (B2, B3, B4, B5), 88fb86c..88abee8

### Spec Compliance

❌ Issues found (3 Important; the mapping itself is complete)

**B2 — `shot-design-frontier` pixi environment: done.** Feature, deps and activation block are the
brief's values (`pyproject.toml` `[tool.pixi.feature.shot-design-frontier]`, its
`.pypi-dependencies` and `.target.unix.activation.env`), plus `plotly` with a stated reason.
`[tool.pixi.environments]` lists `shot-design-frontier = ["shot-design-frontier", "shot-design"]`
— the brief's order reversed, with the precedence finding recorded in the manifest comment.
Verified live: `pixi run --frozen --no-install -e shot-design-frontier` prints
`SHOT_DESIGN_DATA_ROOT=/lustre/orion/fus187/proj-shared/nchen/shot_design`,
`LABELER_ROOT=.../labeler`, `SHOT_DESIGN_CORPUS=/lustre/orion/fus187/proj-shared/foundation_model`,
`SHOT_DESIGN_PATHS=.../configs/shot_design/paths.frontier.yaml`, `HDF5_USE_FILE_LOCKING=FALSE`.
`pixi.lock` is binary in the diff, so I checked it in the tree: environment keys `shot-design:`
(1618), `shot-design-cpu:` (1925), `shot-design-frontier:` (2218), and the frontier environment
resolves `torch-2.10.0+rocm7.1`, `torchvision-0.25.0+rocm7.1`, `triton_rocm-3.6.0` from the
rocm7.1 index (pixi.lock:2423-2425).

**B2 Step 4 (HF cache) — reported "not done"; I verified it is in fact satisfied.**
`~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2` exists, and
`pixi run --frozen -e shot-design-frontier python -c "SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"`
loads offline (`HF_HUB_OFFLINE=1` from activation) and reports 384 dims. No action needed; the
report's open blocker can be closed.

**B3 — wrappers: done.** `_shot_design_common.sh` matches the brief (REPO/ROOT/PY, `RCCL_PLUGIN=0`,
sources `_frontier_settings.sh`, the four exports, `mkdir -p "$ROOT/runs/slurm"`). All four wrappers
carry `-A fus187`, a `batch|extended` partition, the `%j`/`%A_%a` log path under
`$SHOT_DESIGN_DATA_ROOT/runs/slurm`, and source the common file. `shot_design_simulate.sh` is the
header-only stub with `-q debug`. `shot_design_serve_llm.sh` is correctly absent (ruling 3).
`-p extended` on the two 6 h jobs is within ruling 5 and carries its justification in-file
(`shot_design_build.sh:8-11`).

**B4 — jobstats backend: done.** `parse_frontier`, `parse_gpu_samples`, `frontier_backend`,
`gpu_samples_path`, `read_gpu_samples` and the `main()`/`_capture_task` wiring are all present with
the brief's arithmetic, and `tests/labeler/test_jobstats_frontier.py` pins the brief's numbers
against the real `JobStats` field names. The sampler is wired into `shot_design_encode.sh:24-28`
with `trap ... EXIT` and the `$SLURM_ARRAY_JOB_ID_$SLURM_ARRAY_TASK_ID` naming that
`read_gpu_samples` looks for.

**B5 — rename: complete against the exact mapping.** Checked file by file:
pixi features/environments (`pyproject.toml`) ✅; `.mcp.json:5` `-e shot-design-cpu` ✅;
`src/shot_design/mcp/server.py:97` `shot-design://manifest` ✅; `src/shot_design/ui/app.py:29`
`COOKIE = "shot_design_token"` ✅; `src/shot_design/shotdb/legacy_raw.py:44` `SCHEMA =
"shot-design-raw-v1"` + `LEGACY_SCHEMAS` + acceptance at `is_ours` ✅;
`src/shot_design/design/provenance.py:58` ✅; `scripts/shot_design/*.sbatch` env dirs ✅;
`AGENTS.md`, `data/events/README.md` (also corrected `configs/ideate/shot_lists` →
`configs/shot_design/shot_lists`, which is the real path), `docs/*`, the 2026-09-19 plan and spec ✅.
My own `grep -rIn ideate src tests scripts docs configs data AGENTS.md .mcp.json pyproject.toml`
returns 9 lines: 6 are the legacy tag, its test and the two explanatory comments the brief itself
mandates; 3 are `docs/CLUSTERS.md:49,73,225`, all describing the protected Stellar directory or the
historical package rename. Nothing in the do-not-rename set was touched.

Cookie/URI/schema call-site sweep (named risk: a cross-cutting string rename leaving a reader
behind): `COOKIE` has no literal readers outside `ui/app.py:119,121` and the tests that import the
constant; `shot-design://manifest` is read by `tests/shot_design/test_mcp.py:1012,1021,1039`, both
renamed together; `provenance.SCHEMA` is compared only by `tests/shot_design/test_provenance.py:87`
against freshly written sidecars, and `provenance.audit` only counts `body.get("schema")` into
`by_schema` (`provenance.py:437`) — so the report's claim (d) holds and no reader is left broken by
omitting `LEGACY_SCHEMAS` there.

⚠️ Cannot verify from this diff:
- The full `pytest tests/shot_design tests/labeler` run in the new environment never finished
  (report §7). Per-file greens are claimed but the suite-level result is open.
- `sbatch --test-only` acceptance of `-p extended` / `-q debug` for the four wrappers is a report
  claim; I did not submit anything.
- What `rocm-smi` reports from inside a compute-node allocation (see Important #2). The login node
  here shows a single `card0`, which cannot settle it.

### Strengths

- The activation-precedence problem was found by measurement, not assumption, fixed in the
  environment list, and then written down twice where a future editor will hit it
  (`pyproject.toml`, the feature comment and the `[tool.pixi.environments]` comment). Reversing that
  list silently re-points every Frontier job at `/scratch/gpfs/...`; the comment says exactly that.
- `parse_frontier` reuses the existing `fetch_sacct` (`SACCT_FORMAT` at `jobstats.py:81` includes
  `AllocTRES`) rather than the brief's narrower `-o` list. That matters: `_has_gpu`
  (`jobstats.py:750-761`) falls back to `tres_gpus(row.alloc_tres)`, so a GPU job whose sampler never
  ran is still classified as a GPU job and fails "undetermined" instead of silently passing on CPU
  alone. The brief's field list would have opened that hole.
- `_gpu_sampler.sh:18-29` parses `rocm-smi` columns by name. I ran its awk verbatim against real
  `rocm-smi --showuse --showmeminfo vram --csv` output on this node: it emits `0,10,65520`, i.e.
  `gpu_pct,vram_used_mb,vram_total_mb` in exactly the order `parse_gpu_samples` reads
  (`jobstats.py:631-641`). The brief's `NR==2` would indeed have been wrong. Report claim (b) holds.
- The autouse `a_cluster_that_has_jobstats` fixture (`tests/labeler/test_jobstats.py:42-55`) is the
  right call, not a mask: `main()` chose its backend from the live PATH, so those 64 tests would
  have passed on Stellar and failed on Frontier. The Frontier path it pins away is covered
  end-to-end by `test_the_cli_uses_sacct_alone_when_the_cluster_has_no_jobstats`, which asserts
  `jobstats` is never invoked and that the sacct/rocm-smi numbers reach the record. Claim (a) holds.
- `test_a_file_stamped_before_the_rename_is_still_ours` (`tests/shot_design/test_legacy_raw.py:657`)
  tests the behaviour (an old-tag file opens as ours), not the constant, and `conftest.OUR_SCHEMA`
  was moved to the new tag so the rest of the suite exercises the new stamp.
- New files are ruff-clean at 88 columns (`tests/labeler/test_jobstats_frontier.py`,
  `tests/shot_design/test_slurm_frontier_scripts.py`, the new `jobstats.py` region 605-760: zero
  lines over 88).
- Every wrapper carries its sizing/partition reasoning in-file, matching the house style of the
  Stellar `.sbatch` set.

### Issues

#### Critical (Must Fix)

None.

#### Important (Should Fix)

**1. `docs/CLUSTERS.md:251,254` — the Frontier checklist still tells the operator to run bare pixi,
and does not mention the environment this task created.**
```
- [ ] `pixi install -e shot-design-cpu -e labelmaker -e frontier`
- [ ] Populate the HF cache offline; `pixi run -e shot-design-cpu pytest tests/shot_design` green
```
Ruling 2 is that every pixi invocation on Frontier uses `--frozen`, and the implementer's own report
(§4) documents that a bare command re-solves `default`/win-64 here and panics. These two lines are
Frontier instructions and are the ones an operator follows first. The env table at
`docs/CLUSTERS.md:63-70` also still lists only `shot-design`, `shot-design-cpu` and `frontier` —
`shot-design-frontier` is absent from the one table that is supposed to say what the environments
are. Fix: add `--frozen` to both lines, install `shot-design-frontier` instead of
`shot-design-cpu` in the checklist, and add the row to the table. (The bare `pixi run -e ...` in
`AGENTS.md:22-23`, `docs/SHOT_DESIGN.md` and `docs/LABELER.md` are Stellar-facing and out of
ruling 2's scope — left alone correctly.)

**2. `scripts/slurm_frontier/_gpu_sampler.sh:16` + `shot_design_encode.sh:11-13` — the GPU sample
very likely measures the whole node, not the job's one GCD, which would make the encode job
un-gateable.** The awk sums over every row `rocm-smi` prints (`/^card/ { ... n++ }`) and divides by
`n`; the header comment claims this averages "the visible GCDs". `rocm-smi` reads sysfs and does not
consult `ROCR_VISIBLE_DEVICES`, so inside an exclusively-allocated Frontier node it lists all 8
GCDs while `--gres=gpu:1` gives the job one. A job using its GCD at 80 % would then be recorded at
~10 % and fail the 70 % gate every time — the gate this backend exists to serve. I could not settle
it from the login node (it shows a single `card0`), which is why it is stated as a risk rather than
a measured defect, but it is cheap to close: restrict the sampler to the allocated device
(`rocm-smi -d "${ROCR_VISIBLE_DEVICES%%,*}"`, or filter rows by that list) and confirm on one
`-q debug` job before the first 6 h array. The same block's claim that the 8-task array is
"one MI250X node's worth in total" is wrong on Frontier — each array element is its own job and
Frontier allocates whole nodes, so that is 8 nodes × 6 h, not one node; the comment should say so
because it is the figure a reader will size the campaign from.

**3. `scripts/slurm_frontier/shot_design_census.sh:12` writes `db/census.parquet`, which no
consumer looks for (plan-mandated).** Every reader of the census defaults to
`<db_dir>/corpus_coverage.parquet`: `src/shot_design/cli.py:926`, `:1137`, `:1187`, and
`src/shot_design/shotdb/select.py:19` documents that name as the census. The B3 brief specifies
`--out "$ROOT/db/census.parquet"`, so this is faithful to the brief and wrong for the pipeline —
`corpus select` and `build` on Frontier would silently look at a file the census job never wrote,
or fall back to a stale one. Fix: drop the `--out` (the default is already the right path) or pass
`corpus_coverage.parquet`, and keep the `corpus summary` argument in step.

#### Minor (Nice to Have)

**4. `docs/SHOT_DESIGN.md:16-17` contradicts `docs/SHOT_DESIGN.md:6` two paragraphs later.** Line 6
now correctly says the environments are `shot-design` / `shot-design-cpu` /
`shot-design-frontier`; line 16-17 still says "Two pixi environments run it, `shot_design` (CUDA)
and `shot_design-cpu`" — underscore forms pixi cannot accept. The B5 grep could not catch it (no
"ideate" in it), but it is exactly the prose B5 is about.

**5. `src/shot_design/mcp/__init__.py:5` and `docs/SHOT_DESIGN.md:438` name a resource URI that no
longer exists.** Both say `shot_design://manifest`; the resource registered at
`src/shot_design/mcp/server.py:97` is `shot-design://manifest`. A caller following the doc gets a
missing resource. Same one-line fix in both places.

**6. `scripts/labeler/assess_labels_a.py:179` raises a bare `KeyError` when
`SHOT_DESIGN_DATA_ROOT` is unset.** The module docstring tells the reader to run it with the
checkout's `src` on `PYTHONPATH` — i.e. without the pixi activation that sets the variable — and
census mode then dies with `KeyError: 'SHOT_DESIGN_DATA_ROOT'` instead of the `parser.error` style
the rest of `main()` uses. The change itself (dropping `paths.root.parent / "ideate/db"`) is right;
give it a default or an explicit error.

**7. `scripts/slurm_frontier/_shot_design_common.sh:17` — `export
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"` is dead.** `_frontier_settings.sh:18`, sourced five lines
earlier, has already done `export OMP_NUM_THREADS=1` unconditionally, so a caller's value is
clobbered before the `:-` default is ever consulted. Either set it before the `source`, or drop the
line and say the setting comes from `_frontier_settings.sh`.

**8. `_shot_design_common.sh` runs `$PY` from `shot-design-frontier` under the `frontier` env's
`LD_LIBRARY_PATH`, `CONDA_PREFIX` and `PATH`.** `_frontier_settings.sh:11-14` prepends
`.pixi/envs/frontier/{bin,lib}` and points `CONDA_PREFIX` at that env, while `PY` is the *other*
env's interpreter — the classic cross-environment shared-library trap. I checked the concrete risk:
`LD_LIBRARY_PATH=.pixi/envs/frontier/lib .pixi/envs/shot-design-frontier/bin/python -c "import
torch, h5py, duckdb, pyarrow, sentence_transformers, fastapi"` succeeds
(`torch 2.10.0+rocm7.1, h5py 3.16.0, duckdb 1.5.5`), and `shot_design` shells out to no `python`
(no `sys.executable`/`"python"` subprocess anywhere in `src/shot_design`), so nothing is broken
today. It is latent: the next re-solve of either environment can desynchronise the two. Prepending
`$REPO/.pixi/envs/shot-design-frontier/lib` after the source (as Stellar's `encode.sbatch` does for
its own env) removes the coupling.

**9. `tests/labeler/test_jobstats_frontier.py:78-82` pins less than its docstring claims.** It
asserts the sampler *echoes* `GPU_SAMPLE_HEADER`, which catches a header edit but not a reordering
of the awk's `printf "%.0f,%.0f,%.0f", use/n, mem, tot` — the thing that would actually turn VRAM
into a utilisation figure. Pinning the printf order, or the awk's column-name list, would close it.

**10. No migration note for the Stellar checkout.** The rename means `.pixi/envs/ideate` and
`.pixi/envs/ideate-cpu` no longer match `scripts/shot_design/build.sbatch:92`,
`census.sbatch:78`, `encode.sbatch:92` or `encode_cpu.sbatch:117`, all of which now hard-code
`.pixi/envs/shot-design*`. Those jobs fail on Stellar until someone runs the install there.
`docs/CLUSTERS.md:72-76` records that the rename happened but not that it requires a reinstall.

### Assessment

**Task quality:** Needs fixes

**Reasoning:** B2/B4/B5 are solid and in several places better than the briefs — the activation
precedence was measured rather than assumed, and the sacct `AllocTRES` reuse closes a gate hole the
brief would have opened; I verified the environment, the lock, the offline HF load and the sampler's
awk against real `rocm-smi` output myself. The three Important items are all in B3's surface: a
census output path no consumer reads, a GPU sample that probably measures the node instead of the
job's GCD, and a Frontier checklist that still says bare `pixi` — each a small edit, but each one
would surface as a wasted multi-hour allocation.
