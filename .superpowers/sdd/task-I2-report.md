# Task I2 report — ideate library, CLI and tests

Status: **DONE_WITH_CONCERNS** (implementation and tests complete; environment tooling caveats below).
Branch: `recommender`, never switched. Starting target commit: `a9b6362aadea6d8be8895f62e7dcda04d3fd83c0`.
Source: read-only `/scratch/gpfs/nc1514/shot-recommender-system`, verified HEAD `565d548`.
This report is included in the `ideate:` implementation commit; its SHA is printed in the final response.

## Ported

- All 23 non-UI source modules, with package imports, documentation, provenance/schema strings,
  environment variables and CLI invocations renamed to ideate. Every ported Python module,
  including fixtures and package initializers, has the required attribution in its module docstring.
- `schema.py`, `config.py`, `cli.py`, `flags/rules.py`, `llm/client.py` and package initializers.
- `shotdb/features.py`, `text.py`, `ignite.py`, `build.py`, `store.py`;
  `shotdb/rawfile.py` becomes `shotdb/legacy_raw.py`. Its reading behavior remains unchanged
  apart from the required schema/package rename; one obsolete conversion reference was removed
  from a docstring. The optional d3d_fusion_data store remains usable.
- All seven retrieval modules: channels, rank, describe, suggest, scenarios, blurb and actuation.
- Added `src/ideate/__main__.py` so `python -m ideate` calls the original CLI's `main()`.
- All 10 YAML files, including `shot_lists/poc_v1.yaml`, under `configs/ideate/`.
  `ui.yaml` stays because it contains actuation settings used by the kept library; it does not
  install a UI. `labels.yaml` supplies the existing scenario themes. No source CSV evalset exists.
- 18 kept upstream test modules and `conftest.py`; `test_rawfile.py` renamed to
  `test_legacy_raw.py`. Existing I1 `tests/ideate/__init__.py` and `test_package.py` retained.
  Fixture imports are relative so other test packages cannot shadow them.
- `real_data` registered by adding only a markers table to `pyproject.toml`.

## Integration adjustments

- `CONFIG_DIR` derives from `config.py` as `<repo>/configs/ideate`, overridable by
  `IDEATE_CONFIG_DIR`. `IDEATE_PATHS` and `IDEATE_DATA_ROOT` replace their old names.
  An explicitly empty `IDEATE_DATA_ROOT` produces a clear error instead of selecting defaults.
- Default writable root: `/scratch/gpfs/EKOLEMEN/nc1514/ideate`.
  Text remains under `/scratch/gpfs/EKOLEMEN/big_d3d_data/foundation_model_text`, with
  shot summaries under its `shotsummary/` subtree and SQL logs under `sql/`.
  IGNITE weights remain at `/scratch/gpfs/EKOLEMEN/nc1514/shot-recommender/models/IGNITE`.
  MiniLM is `sentence-transformers/all-MiniLM-L6-v2`, verified offline in I1; it is declared
  in `paths.yaml`, exposed through `Paths.sentence_transformers_model`, and used by the loader.
- IGNITE imports `tokamak_foundation_model.ignite.train_dynamics` from the sibling installed
  package. Removed the submodule path constants, path-insertion function and calls.
  Missing processed input produces `FileNotFoundError` with its exact location; batch encoding
  records the failure and continues. Default input is `Paths.foundation_model_processed_dir`.
- `frame_codes()` has an **identical AST** to source `565d548`, including its signature and
  computation. Dataset creation, frame origin, batching, feature encoding and quantization
  remain the upstream logic. No codec or other foundation-model source was changed.
- `retrieval/actuation.py` originally imported `ui.waveforms`. Moved only the required `Trace`,
  decimation, clipping, total and trace-reading helpers into actuation; renamed the trace-specific
  `_empty` to avoid the existing vertex helper. This preserves reference/median seeding while
  removing the UI dependency. Browser group listing, spectrograms and profile endpoints are dropped.
- LLM defaults to `provider: off`, as the plan specifies for iteration 0. The client remains
  functional when explicitly configured. Its missing-endpoint hint now describes `base_url` or
  endpoint JSON rather than the unported Makefile/server scripts.
- CLI catches expected filesystem/validation errors at its boundary. Missing databases, missing
  saved sets and empty build selections return a nonzero status with a clear message.
- Synthetic paths explicitly override `models_dir` into the fixture root. Real-data integration
  builds use temporary roots and disable optional encoding where the test is about scalar/text
  ingestion. Real retrieval tests now build four staged shots under `tmp_path_factory` instead
  of gating on an old shared database and accidentally querying the new empty root. Model/corpus
  reads are read-only; test writes stay under pytest temporary fixtures.
- Added 14 test cases for config discovery/overrides/default paths, empty-root CLI behavior,
  rejected dropped commands, processed-corpus selection and missing-input reporting.
  `test_show_json_round_trips_through_shotrecord` was renamed to
  `test_show_json_round_trips_through_record` to avoid the obsolete package substring in acceptance 4.

## Dropped, and why

- Entire source `ui/`: `__init__.py`, `app.py`, `chat.py`, `jobs.py`, `serve.py`, `waveforms.py`
  and static assets. UI is deferred to a later iteration. Only the library trace helpers noted
  above are retained inside the explicitly kept actuation module.
- Entire source `scripts/`, including fetch/curation/server workflows: outside the port map.
  No fetch or curate package was present under upstream `src/shotrec`; their script workflows
  and the CLI fetch handler/parser/imports were excluded. CLI serve handler/parser removed too.
- IGNITE raw-to-processed conversion: GroupSpec/group maps, resampling/grid/collection helpers,
  ConversionReport, official-file conversion selection, `to_processed_h5`, and process-pool
  `prepare_inputs`. This port consumes processed files directly and must not duplicate preparation.
- No `pcslayout` dependency or `external/` path insertion. Schema already contains `Vertex`;
  descriptive references use PCS terminology and IGNITE config references the sibling source path.
- Whole test modules dropped exactly as requested:
  - `test_curate.py`: dropped curation workflow.
  - `test_fetch_plan.py`: dropped fetch script/planning workflow.
  - `test_ui.py`: deferred browser/API application.
  - `test_serve.py`: deferred UI server/tunnel.
  - `test_serve_llm.py`: unported model-serving script.
  - `test_chat.py`: deferred UI chat.
  - `test_jobs.py`: deferred UI job manager.
  - `test_waveforms.py`: dropped browser diagnostics module; kept actuation tests still exercise
    its extracted trace helpers through reference/median seeds.
- Within kept `test_ignite.py`, 25 conversion-only tests removed:
  - `test_a_group_mid_write_is_skipped_rather_than_half_read`
  - `test_a_shot_with_no_raw_file_says_where_it_looked`
  - `test_a_truncated_official_file_is_not_mistaken_for_a_usable_one`
  - `test_absent_modalities_are_nan_placeholders_not_missing_groups`
  - `test_an_existing_official_file_is_used_rather_than_reconverted`
  - `test_cer_is_48_ignite_channels_taken_from_the_80_column_staged_group`
  - `test_channels_we_cannot_fill_are_nan_rows_inside_a_present_group`
  - `test_conversion_is_not_repeated_unless_forced`
  - `test_conversion_opens_every_raw_file_without_the_hdf5_lock`
  - `test_conversion_writes_the_real_layout`
  - `test_converted_slow_ts_reproduces_the_official_file`
  - `test_ech_channel_order_is_alphabetical_by_pointname`
  - `test_ech_column_lands_on_its_alphabetical_channel`
  - `test_gas_column_lands_in_gas_raw_and_gas_flow_stays_empty`
  - `test_grid_is_uniform_at_the_target_rate_over_the_columns_support`
  - `test_grid_refuses_an_absurd_span_rather_than_allocating_it`
  - `test_grid_spans_the_union_of_the_columns_not_the_stored_axis`
  - `test_group_map_is_the_32_groups_a_real_processed_file_has`
  - `test_our_fetched_file_wins_over_the_staged_copy`
  - `test_our_gas_group_fills_gas_raw_and_never_gas_flow`
  - `test_our_group_map_matches_a_real_processed_file_exactly`
  - `test_our_missing_cer_channels_are_the_official_files_missing_channels`
  - `test_resample_interpolates_over_finite_samples_only`
  - `test_resample_never_extrapolates_past_a_columns_own_support`
  - `test_rmp_is_the_twelve_i_coils_and_i_coil_prepends_the_six_c_coils`
- Within kept `test_cli.py`, four fetch-only tests removed:
  - `test_fetch_passes_a_list_by_name_and_shots_by_number`
  - `test_fetch_plan_matches_the_script_and_touches_no_network`
  - `test_fetch_without_a_selector_says_so`
  - `test_fetch_without_run_prints_the_command_it_would_run`

## Python 3.11 and warning compatibility

**Python 3.11 syntax fixes: none required at this source revision.** Compiled all
29 upstream source Python files and 27 upstream test/fixture Python files with
Python **3.11.16**, without writing bytecode into the source repository: no SyntaxError.
The port's 24 source and 21 test/fixture Python files also compile. Although the upstream project
requires Python 3.12 in its metadata, the supplied revision does not contain a 3.12-only syntax
construct in these files. No invented syntax migration was applied.

Two runtime warning fixes were necessary for `-W error`:

1. `features.stat_std`: scoped `np.errstate(over="ignore", invalid="ignore")` around `np.std`.
   The existing overflow test passes values `1e308` and `-1e308`; unavailable/overflowed statistics
   already map to None, but NumPy's intermediate square warned before `_value` could apply that
   contract. This preserves the None result and leaves warnings outside that operation visible.
2. `_dynamics`: scoped filtering of **only** the `FutureWarning` beginning
   `` `torch.jit.script` is deprecated. `` from module `torch.jit._script` while importing the
   sibling codec package. Torch 2.14 warns on the `@torch.jit.script` decorators in
   x_transformers; Torch 2.6 does not. The filter is restored after import and no torch API,
   decorator or codec computation is replaced. Context7 resolved `/pytorch/pytorch`; fetched docs
   confirm TorchScript deprecation, and installed `torch/jit/_script.py:1491` identifies the exact
   emitted warning. Reference: https://github.com/pytorch/pytorch/blob/main/docs/source/notes/cpu_threading_torchscript_inference.md

Ruff 0.16.5 required import organization and minor lint fixes (pairwise iteration, explicit
subprocess check flags, explicit regex flags/NaN checks, redundant integer casts, concatenation
parentheses and UTC test timestamps). Existing broad catches at per-shot/optional-operation
boundaries retain their source fallback behavior with local, explained `BLE001` annotations.
No E501 reflow or repository lint configuration changes were made.

## Acceptance 1 — full CPU suite

Exact command:
```
pixi run -e ideate-cpu pytest tests/ideate -q -W error
```
Exit status **0**. Final stdout:
```
........................................................................ [ 17%]
........................................................................ [ 35%]
........................................................................ [ 53%]
........................................................................ [ 70%]
........................................................................ [ 88%]
...............................................                          [100%]
407 passed in 38.12s
```
**407 passed; 0 failed; 0 skipped; no pytest warnings.** The offline MiniLM load, real staged
builds, real retrieval slice and real codec checks all ran.

Pixi itself emits pre-existing environment-startup warnings on stderr before pytest; these are
not Python warnings and cannot be repaired within the permitted pyproject markers-only change:
```
 WARN cache for Repodata at /home/nc1514/.cache/rattler/cache/repodata is on a network/parallel filesystem (NFS/SMB/FUSE/BeeGFS/Lustre/GPFS/CephFS), redirected to /tmp/pixi-cache-nc1514/repodata for this run. Set [cache.repodata] in config.toml or PIXI_CACHE_DIR to override, or [cache.netfs-redirect] = "never" to keep the original path.
 WARN cache for PypiMapping at /home/nc1514/.cache/rattler/cache/conda-pypi-mapping is on a network/parallel filesystem (NFS/SMB/FUSE/BeeGFS/Lustre/GPFS/CephFS), redirected to /tmp/pixi-cache-nc1514/conda-pypi-mapping for this run. Set [cache.pypi-mapping] in config.toml or PIXI_CACHE_DIR to override, or [cache.netfs-redirect] = "never" to keep the original path.
 WARN The package `httpx==0.28.1` does not have an extra named `http2`
 WARN The package `httpx==0.28.1` does not have an extra named `http2`
 WARN The package `httpx==0.28.1` does not have an extra named `http2`
```
The authorized direct-interpreter form was also exercised during validation (before the final
parity assertion enhancement):
```
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false IDEATE_DATA_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/ideate .pixi/envs/ideate-cpu/bin/python -m pytest tests/ideate -q -W error --tb=short
407 passed in 45.58s
```

Additional final CUDA-environment verification, torch 2.6.0+cu124:
```
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false IDEATE_DATA_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/ideate /scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/ideate/bin/python -m pytest tests/ideate/test_ignite.py -q -W error
...............                                                          [100%]
15 passed in 13.83s
```
CUDA verification stderr was empty. CPU torch is 2.14.0+cpu.

Parity-test correction: the supplied corpus has no live co2 samples on 190090. The upstream
parity test silently iterated only over returned modalities, so its claimed three-family slice
actually checked two. The final test requests **mhr (spectro), filterscopes (fastts), and
 ts_core_density (slowts)** and asserts every requested modality is returned. All 20 frames of
all three modalities equal the shipped production code tensors exactly on CPU and GPU. The
production frame_codes function itself was not changed. This is a slice check, not a claim that
all 239 frames/all modalities were newly validated in I2.

## Acceptance 2 — CLI and empty data

```
$ pixi run -e ideate-cpu python -m ideate --help
usage: ideate [-h]
              {build,add,model,show,export,coverage,query,llm,blurb,actuation}
              ...

DIII-D shot database: build from local signals, retrieve and inspect shots.

positional arguments:
  {build,add,model,show,export,coverage,query,llm,blurb,actuation}
    build               full rebuild of the database (atomic: writes db.tmp,
                        swaps)
    add                 incremental upsert of one or more shots
    model               IGNITE bundle status; --download copies it from the
                        Hub
    show                print one shot's record
    export              ShotSummary rows as JSON or Parquet
    coverage            present/unavailable/pending per registry field
    query               find similar shots
    llm                 is a language model reachable? prints the endpoint or
                        how to start one
    blurb               write the model's per-shot blurbs into shots.parquet
                        (needs a running model)
    actuation           inspect saved actuator waveform sets

options:
  -h, --help            show this help message and exit
```
Exit **0**. The same pixi startup diagnostics listed above occurred on stderr.

Separate subprocesses used the ideate-cpu interpreter and a freshly empty directory inside
this checkout. Every data-dependent invocation below exited 1 without a traceback; the directory
remained empty. The standalone empty-string environment override also fails clearly:
```
$ IDEATE_DATA_ROOT=/scratch/gpfs/nc1514/FusionAIHub/.tmp/task-I2/empty-root-k7i_bibp python -m ideate show 1
exit=1
no database at /scratch/gpfs/nc1514/FusionAIHub/.tmp/task-I2/empty-root-k7i_bibp/db -- run `ideate build --list poc_v1` first

$ IDEATE_DATA_ROOT=/scratch/gpfs/nc1514/FusionAIHub/.tmp/task-I2/empty-root-k7i_bibp python -m ideate export 1
exit=1
no database at /scratch/gpfs/nc1514/FusionAIHub/.tmp/task-I2/empty-root-k7i_bibp/db -- run `ideate build --list poc_v1` first

$ IDEATE_DATA_ROOT=/scratch/gpfs/nc1514/FusionAIHub/.tmp/task-I2/empty-root-k7i_bibp python -m ideate coverage
exit=1
no database at /scratch/gpfs/nc1514/FusionAIHub/.tmp/task-I2/empty-root-k7i_bibp/db -- run `ideate build --list poc_v1` first

$ IDEATE_DATA_ROOT=/scratch/gpfs/nc1514/FusionAIHub/.tmp/task-I2/empty-root-k7i_bibp python -m ideate query --ref 1
exit=1
no database at /scratch/gpfs/nc1514/FusionAIHub/.tmp/task-I2/empty-root-k7i_bibp/db -- run `ideate build --list poc_v1` first

$ IDEATE_DATA_ROOT=/scratch/gpfs/nc1514/FusionAIHub/.tmp/task-I2/empty-root-k7i_bibp python -m ideate add 1
exit=1
no database at /scratch/gpfs/nc1514/FusionAIHub/.tmp/task-I2/empty-root-k7i_bibp/db -- `ideate build` before `ideate add`

$ IDEATE_DATA_ROOT=/scratch/gpfs/nc1514/FusionAIHub/.tmp/task-I2/empty-root-k7i_bibp python -m ideate build --shots 999999 --workers 1
exit=1
1 of 1 shots have no Ip signal on disk yet and are skipped (fetch them first, or pass --all to
  build them as empty records): 999999
nothing to build

$ IDEATE_DATA_ROOT=/scratch/gpfs/nc1514/FusionAIHub/.tmp/task-I2/empty-root-k7i_bibp python -m ideate actuation show 20260907T000000.000000Z-1
exit=1
no actuation set 20260907T000000.000000Z-1 in /scratch/gpfs/nc1514/FusionAIHub/.tmp/task-I2/empty-root-k7i_bibp/actuations

$ IDEATE_DATA_ROOT='' python -m ideate coverage
exit=1
ideate: IDEATE_DATA_ROOT is empty; set it to a data directory or unset it
```
An empty `actuation list` remains a valid listing with a "no saved actuation sets" message.

## Acceptance 3 — sibling IGNITE import

```
$ pixi run -e ideate-cpu python -c "import ideate.shotdb.ignite as m; print(m.__file__)"
/scratch/gpfs/nc1514/FusionAIHub/src/ideate/shotdb/ignite.py
```
Exit **0**. Also compared `sys.path` before/after this import and inspected `sys.modules`:
```
sys.path unchanged; no pcslayout imported
frame_codes AST identical to source 565d548
```
Source scan finds no `sys.path` mutation, external path insertion or conversion function.

## Acceptance 4 — obsolete-name scan

Removed generated `__pycache__` directories under the two new package/test trees after tests,
so this literal recursive grep reports source lines rather than bytecode matches.
```
$ grep -rn "shotrec\|SHOTREC\|pcslayout\|external/FusionAIHub" src/ideate tests/ideate configs/ideate
src/ideate/cli.py:6:Ported from shot-recommender-system (shotrec) @565d548.
src/ideate/retrieval/rank.py:11:Ported from shot-recommender-system (shotrec) @565d548.
src/ideate/retrieval/actuation.py:12:Ported from shot-recommender-system (shotrec) @565d548.
src/ideate/retrieval/channels.py:18:Ported from shot-recommender-system (shotrec) @565d548.
src/ideate/retrieval/describe.py:25:Ported from shot-recommender-system (shotrec) @565d548.
src/ideate/retrieval/__init__.py:13:Ported from shot-recommender-system (shotrec) @565d548.
src/ideate/retrieval/suggest.py:12:Ported from shot-recommender-system (shotrec) @565d548.
src/ideate/retrieval/scenarios.py:9:Ported from shot-recommender-system (shotrec) @565d548.
src/ideate/retrieval/blurb.py:13:Ported from shot-recommender-system (shotrec) @565d548.
src/ideate/shotdb/build.py:18:Ported from shot-recommender-system (shotrec) @565d548.
src/ideate/shotdb/ignite.py:8:Ported from shot-recommender-system (shotrec) @565d548.
src/ideate/shotdb/legacy_raw.py:16:Ported from shot-recommender-system (shotrec) @565d548.
src/ideate/shotdb/__init__.py:1:"""Ported from shot-recommender-system (shotrec) @565d548."""
src/ideate/shotdb/text.py:12:Ported from shot-recommender-system (shotrec) @565d548.
src/ideate/shotdb/features.py:9:Ported from shot-recommender-system (shotrec) @565d548.
src/ideate/shotdb/store.py:10:Ported from shot-recommender-system (shotrec) @565d548.
src/ideate/__init__.py:3:Ported from shot-recommender-system (shotrec) @565d548.
src/ideate/flags/rules.py:25:Ported from shot-recommender-system (shotrec) @565d548.
src/ideate/flags/__init__.py:7:Ported from shot-recommender-system (shotrec) @565d548.
src/ideate/schema.py:4:Ported from shot-recommender-system (shotrec) @565d548.
src/ideate/llm/client.py:11:Ported from shot-recommender-system (shotrec) @565d548.
src/ideate/llm/__init__.py:3:Ported from shot-recommender-system (shotrec) @565d548.
src/ideate/config.py:7:Ported from shot-recommender-system (shotrec) @565d548.
tests/ideate/test_suggest.py:8:Ported from shot-recommender-system (shotrec) @565d548.
tests/ideate/test_config.py:1:"""Ported from shot-recommender-system (shotrec) @565d548."""
tests/ideate/test_describe.py:9:Ported from shot-recommender-system (shotrec) @565d548.
tests/ideate/test_integration_real.py:15:Ported from shot-recommender-system (shotrec) @565d548.
tests/ideate/test_retrieval.py:14:Ported from shot-recommender-system (shotrec) @565d548.
tests/ideate/test_ignite.py:3:Ported from shot-recommender-system (shotrec) @565d548.
tests/ideate/test_features.py:1:"""Ported from shot-recommender-system (shotrec) @565d548."""
tests/ideate/test_llm_client.py:4:Ported from shot-recommender-system (shotrec) @565d548.
tests/ideate/test_text.py:1:"""Ported from shot-recommender-system (shotrec) @565d548."""
tests/ideate/conftest.py:3:Ported from shot-recommender-system (shotrec) @565d548.
tests/ideate/test_flags.py:10:Ported from shot-recommender-system (shotrec) @565d548.
tests/ideate/test_registry.py:1:"""Ported from shot-recommender-system (shotrec) @565d548."""
tests/ideate/test_scenarios.py:3:Ported from shot-recommender-system (shotrec) @565d548.
tests/ideate/test_build_store.py:11:Ported from shot-recommender-system (shotrec) @565d548.
tests/ideate/test_actuation.py:6:Ported from shot-recommender-system (shotrec) @565d548.
tests/ideate/test_cli.py:9:Ported from shot-recommender-system (shotrec) @565d548.
tests/ideate/test_legacy_raw.py:1:"""Ported from shot-recommender-system (shotrec) @565d548."""
tests/ideate/test_schema.py:1:"""Ported from shot-recommender-system (shotrec) @565d548."""
tests/ideate/test_blurb.py:3:Ported from shot-recommender-system (shotrec) @565d548.
```
Exit **0**, 42 matches, every match an attribution line; none in configs.

## Acceptance 5 — Ruff

The ideate-cpu environment has no ruff executable (`ruff: command not found`, exit 127 on the
first literal attempt). Reused the already-installed executable from labelmaker **by PATH only**;
no package install, environment pin or project configuration was changed:
```
export PATH="$PWD/.pixi/envs/labelmaker/bin:$PATH"
pixi run -e ideate-cpu ruff check src/ideate tests/ideate
All checks passed!
```
Exit **0**, using this repo's `pyproject.toml` (confirmed by `--show-settings`), Ruff 0.16.5.
The same pixi startup warnings listed in acceptance 1 were emitted separately. A direct invocation
of that Ruff executable also reports `All checks passed!`. `git diff --cached --check` is clean.

## Acceptance 6 — protected paths and project metadata

```
$ git diff a9b6362aadea6d8be8895f62e7dcda04d3fd83c0 -- src/labelmaker tests/labelmaker src/tokamak_foundation_model
(no output; exit 0)
$ git diff a9b6362aadea6d8be8895f62e7dcda04d3fd83c0 -- pyproject.toml
diff --git a/pyproject.toml b/pyproject.toml
index d9926e8..37e9a37 100644
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -337,3 +337,6 @@ frontier = ["frontier"]
 labelmaker = ["labelmaker", "fdp"]
 ideate = ["ideate", "cuda"]
 ideate-cpu = ["ideate", "ideate-cpu"]
+
+[tool.pytest.ini_options]
+markers = ["real_data: requires the read-only EKOLEMEN stores and/or local model weights"]
```
`pixi.lock` is unchanged. No changes made to the read-only source repository or the other worktree.
No source/data/model files outside this checkout were written. Tests use pytest temporary paths;
verification transcripts live under the existing untracked `.tmp/task-I2/` in this checkout.
A concurrent edit to `docs/superpowers/plans/2026-09-07-recommender-ledger.md` appeared during
work; it was not made by I2 and is excluded from this commit. Pre-existing `.tmp/` and `figures/`
remain untracked and excluded.

## Files changed

55 committed paths (54 implementation/config/test paths plus this report):
- `configs/ideate/actuators.yaml`
- `configs/ideate/flags.yaml`
- `configs/ideate/ignite_modalities.yaml`
- `configs/ideate/labels.yaml`
- `configs/ideate/llm.yaml`
- `configs/ideate/paths.yaml`
- `configs/ideate/retrieval.yaml`
- `configs/ideate/shot_lists/poc_v1.yaml`
- `configs/ideate/signals.yaml`
- `configs/ideate/ui.yaml`
- `pyproject.toml`
- `src/ideate/__init__.py`
- `src/ideate/__main__.py`
- `src/ideate/cli.py`
- `src/ideate/config.py`
- `src/ideate/flags/__init__.py`
- `src/ideate/flags/rules.py`
- `src/ideate/llm/__init__.py`
- `src/ideate/llm/client.py`
- `src/ideate/retrieval/__init__.py`
- `src/ideate/retrieval/actuation.py`
- `src/ideate/retrieval/blurb.py`
- `src/ideate/retrieval/channels.py`
- `src/ideate/retrieval/describe.py`
- `src/ideate/retrieval/rank.py`
- `src/ideate/retrieval/scenarios.py`
- `src/ideate/retrieval/suggest.py`
- `src/ideate/schema.py`
- `src/ideate/shotdb/__init__.py`
- `src/ideate/shotdb/build.py`
- `src/ideate/shotdb/features.py`
- `src/ideate/shotdb/ignite.py`
- `src/ideate/shotdb/legacy_raw.py`
- `src/ideate/shotdb/store.py`
- `src/ideate/shotdb/text.py`
- `tests/ideate/conftest.py`
- `tests/ideate/test_actuation.py`
- `tests/ideate/test_blurb.py`
- `tests/ideate/test_build_store.py`
- `tests/ideate/test_cli.py`
- `tests/ideate/test_config.py`
- `tests/ideate/test_describe.py`
- `tests/ideate/test_features.py`
- `tests/ideate/test_flags.py`
- `tests/ideate/test_ignite.py`
- `tests/ideate/test_integration_real.py`
- `tests/ideate/test_legacy_raw.py`
- `tests/ideate/test_llm_client.py`
- `tests/ideate/test_registry.py`
- `tests/ideate/test_retrieval.py`
- `tests/ideate/test_scenarios.py`
- `tests/ideate/test_schema.py`
- `tests/ideate/test_suggest.py`
- `tests/ideate/test_text.py`
- `.superpowers/sdd/task-I2-report.md`

## Concerns

1. The pytest process is pristine under `-W error`, but literal pixi invocations still emit the
   environment's existing network-filesystem cache and `httpx[http2]` startup warnings. Thus the
   brief's strict interpretation of completely warning-free combined tool output is not met;
   all application/test checks pass. No warning was hidden globally.
2. Ruff must be supplied on PATH from the already-installed labelmaker env (as above), because
   I1 did not provision it in ideate-cpu. Adding a dependency would violate this task's
   markers-only restriction on pyproject. No change was made under labelmaker source/tests.
3. The two torch versions require the narrowly scoped import compatibility filter described
   above. Also, an ad-hoc direct-interpreter probe importing torch *before* NumPy hit the host's
   old `/lib64/libstdc++.so.6` (`GLIBCXX_3.4.29` missing); importing ideate first succeeds, as do
   the required suite/CLI commands. This is an environment linker-order issue, consistent with
   I1's existing loader notes; no environment or protected package was modified to address it.
4. Ported shot lists/registry metadata still describe the legacy campaigns. A corpus reader,
   new shot selection, event joins, MCP and UI belong to subsequent tasks. The I2 CLI build uses
   the retained legacy reader; IGNITE encoding consumes existing processed corpus files only.

---

# Fix report (review findings on 684880c)

Branch `recommender`. Commit: `ideate: finish config-path rename, restore trace-helper tests`.
Method: red/green for every finding that a test could express — three assertions on the stamped
strings and one on the `__main__` guard were written and observed failing before the fix.

## Important 1 — `configs/` → `configs/ideate/` finished in user-facing strings

Every message, exception text, stamped `source` field, comment and docstring under `src/ideate`
and `tests/ideate` that names a config path now names `configs/ideate/...`. Applied with
`perl -pi -e 's{configs/(?!ideate/)}{configs/ideate/}g'` over the 19 files that mentioned one, so
the bare-directory references (`reads them from configs/`) moved too, not only the four sites the
brief listed by line. `config.CONFIG_DIR` itself was already correct (`parents[2] / "configs" /
"ideate"`) — no functional path was touched, only the strings a reader sees.

Tests written first, all three red before the change:

- `tests/ideate/test_flags.py::test_member_caps_expand_from_the_actuator_registry` — the expanded
  per-member cap rule now asserts `cap["source"] == "configs/ideate/actuators.yaml"`, the same
  path inside `cap["message"]`, and that it survives onto the emitted `Flag.source`.
- `tests/ideate/test_retrieval.py::test_the_run_day_decay_has_one_source_the_yaml` — the missing
  knob's `KeyError` is matched against the full
  `configs/ideate/retrieval\.yaml: retrieval\.dedup_threshold is missing`, not the bare tail.
- `tests/ideate/test_llm_client.py::test_provider_off_raises_without_opening_a_socket` — asserts
  `available() == (False, "configs/ideate/llm.yaml has provider: off")` and matches the same text
  on the `LLMUnavailable` raised by `chat`.

The brief's proof grep, run after the change:

```
$ grep -rn "configs/[a-z_]*\.yaml\|configs/shot_lists" src/ideate tests/ideate
$ echo $?
1
```

No output: no old path remains. The complementary listing of what is there now:

```
$ grep -rho "configs/[A-Za-z_/]*" src/ideate tests/ideate | sort -u
configs/ideate/
configs/ideate/actuators
configs/ideate/flags
configs/ideate/ignite_modalities
configs/ideate/llm
configs/ideate/retrieval
configs/ideate/shot_lists/
configs/ideate/shot_lists/poc_v
configs/ideate/signals
```

The seven extra characters pushed 28 prose lines past the width this codebase actually wraps at
(the peak of the length histogram is 95-99; ruff's `line-length = 88` is not enforced because
`E501` is not in the selected rule set). Those lines were re-wrapped so the file count of lines
over 99 columns is unchanged at 125 — the same set as before the fix. The re-wraps move words
between lines only; no wording changed. `src/ideate/config.py:21` (114 columns) was left alone: it
is pre-existing and already on the reviewer's Minor list.

## Important 2 — trace-helper tests restored

New file `tests/ideate/test_actuation_traces.py` (13 tests), attribution line
`Ported from shot-recommender-system (shotrec) @565d548.`, covering `Trace`, `decimate`, `_clip`,
`_json_values`, `_trace`, `_empty_trace`, `_specs`, `_total` and `read_traces` in
`src/ideate/retrieval/actuation.py`. The upstream fixtures these need (`paths`, `staged_shot_a`,
`staged_shot_b`, `BEAMS`, `write_frame`) were already ported into `tests/ideate/conftest.py`
byte-identically, so the assertions are the upstream ones with `waveforms.` → `actuation.` and
`from conftest import` → `from .conftest import` (this package has an `__init__.py`).

Ported from upstream `tests/test_waveforms.py` (9):

| upstream test | kept as |
|---|---|
| `test_decimate_keeps_every_buckets_min_and_max` | same name |
| `test_decimate_leaves_short_series_alone` | same name |
| `test_decimate_survives_all_nan_buckets` | same name |
| `test_read_traces_registry_total_and_unknown` | same name |
| `test_read_traces_raw_channel_and_window` | same name |
| `test_read_traces_missing_group_reports_status` | same name |
| `test_read_traces_actuator_member_keys` | same name |
| `test_total_is_none_where_no_member_recorded` | same name |
| `test_read_traces_window_outside_the_signal` | same name |

Added to cover the four items the brief named that upstream only exercised indirectly (4):

- `test_clip_keeps_the_closed_window_and_returns_the_inputs_when_unbounded` — `_clip` directly:
  both ends inclusive, one-sided `t0`/`t1`, an empty result, and the same-object return when no
  window is given.
- `test_json_values_turns_every_non_finite_sample_into_none` — NaN and both infinities → `None`.
- `test_an_absent_trace_names_no_source_shot` — `_empty_trace` leaves `source` at `None` for all
  four absent statuses, against a present trace that does carry `source == "staged"`. This is the
  invariant of upstream HEAD 565d548 ("absent traces carry no source shot"), which had no test of
  its own in the trace layer.
- `test_a_single_member_actuator_is_not_a_total` — `gas.GASA` reads through the member branch with
  an empty `members` list, while `gas.total` goes through `_total` and names what it summed.

Dropped from upstream `tests/test_waveforms.py` (9), all of them tests of functions that were not
ported because they render the browser page rather than serve a seed — `waveforms.groups` (the
channel picker), `waveforms.spectrogram` (the MHD spectrogram panel) and `waveforms.profile` (the
profile-slice panel) have no counterpart anywhere in `src/ideate`:

- `test_groups_lists_channels_and_rates`
- `test_groups_marks_the_placeholder`
- `test_spectrogram_decimates_a_1mhz_channel_to_a_250khz_axis`
- `test_spectrogram_leaves_a_500khz_channel_alone`
- `test_spectrogram_reports_absence`
- `test_spectrogram_window_too_short_is_unavailable`
- `test_profile_nearest_slice_sorted_by_rho`
- `test_profile_absent_kind`
- `test_profile_reads_a_transposed_block`

Their helper `_fast_group` / `_profile_group` builders and the `h5py` import went with them; no
HTTP-endpoint test existed in this module to drop.

## Minor items

1. `src/ideate/__main__.py` — `raise SystemExit(main())` is now under `if __name__ == "__main__":`.
   Two tests added to `tests/ideate/test_package.py`, the first red before the fix (importing the
   module under pytest ran the CLI on pytest's own argv and exited 2):
   `test_importing_the_module_entry_point_does_not_run_the_cli` (import and reload are not
   invocations) and `test_python_dash_m_ideate_still_runs_the_cli` (`python -m ideate --help`
   still exits 0 with `usage: ideate`).
2. `src/ideate/cli.py` — the `except SystemExit` comment no longer claims `scripts/fetch_shots.py`
   (never ported) is the caller. The branch is kept, not removed: no `cmd_*` raises SystemExit
   today, but a library one calls can, with either shape, and `int()` on a message shape raised
   ValueError out of `main` — which is the bug the branch exists for and which
   `test_a_string_system_exit_is_printed_not_turned_into_a_valueerror` still covers. That test's
   docstring lost the same stale filename.
3. `tests/ideate/test_cli.py` — the orphan `# ---- fetch` section header and its trailing blank
   lines are gone.

Nothing under `frame_codes()` in `shotdb/ignite.py`, `src/labelmaker`, `tests/labelmaker`,
`src/tokamak_foundation_model` or `pyproject.toml` was touched.

## Verification

```
$ HF_HUB_OFFLINE=1 .pixi/envs/ideate-cpu/bin/python -m pytest tests/ideate -q -W error
422 passed in 69.83s (0:01:09)
```

407 → 422: +13 in `tests/ideate/test_actuation_traces.py`, +2 in `tests/ideate/test_package.py`.
The four assertion-only strengthenings (flags source, rank KeyError, llm client message, and the
`_clip`/`_json_values`/absent-source coverage) added no test count of their own beyond those.

```
$ .pixi/envs/labelmaker/bin/ruff check src/ideate tests/ideate
All checks passed!
```

## Concerns

1. `configs/ideate/` now appears in ~50 doc/comment lines that upstream `shotrec` writes as
   `configs/`. A future re-sync against `shot-recommender-system` will see all of them as
   conflicts. This is inherent to the rename the brief asked for, not avoidable; the mechanical
   `perl` substitution at least makes the delta trivially reproducible.
2. The 28 line re-wraps are diff noise on top of that, for the same reason. They were made
   because the alternative was leaving 28 lines at 100-113 columns in files otherwise wrapped at
   99, and the reviewer had already raised one such line as a Minor.
3. `src/ideate/cli.py`'s `except SystemExit` branch is now documented as defensive rather than as
   guarding a known caller. If the eventual `ideate fetch`/staging command never raises SystemExit
   either, the honest end state is to delete both the branch and its test — deferred rather than
   done here, since removing a tested behaviour is outside a fix pass.
4. `docs/superpowers/plans/2026-09-07-recommender-ledger.md` carried an uncommitted edit (the I2
   review entry) when this fix started. It was left uncommitted and out of this commit, as it
   belongs to the reviewing session.
