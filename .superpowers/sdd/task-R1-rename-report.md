# R1 package rename report

Worktree: `/scratch/gpfs/nc1514/FusionAIHub-R1`; branch: `recommender-R1`.
Starting commit: `abba1f16bc6a6c5dd11131783f5d6c5dbc157e5f`.

## Changes

The first commit uses `git mv` only: 404 files are recorded as `R100` renames,
with no content changes. The moved trees are:

| Before | After |
| --- | --- |
| `src/ideate` | `src/shot_design` |
| `src/labelmaker` | `src/labeler` |
| `tests/ideate` | `tests/shot_design` |
| `tests/labelmaker` | `tests/labeler` |
| `scripts/ideate` | `scripts/shot_design` |
| `scripts/labelmaker` | `scripts/labeler` |
| `configs/ideate` | `configs/shot_design` |
| `docs/IDEATE.md` | `docs/SHOT_DESIGN.md` |
| `docs/LABELMAKER.md` | `docs/LABELER.md` |
| `outputs/labelmaker` | `outputs/labeler` |

Separate reference groups cover imports/module strings and CLI/UI/MCP names;
packaging, pixi tasks and MCP configuration; environment compatibility;
docs/model cards; and scripts. The output-tree scripts receive module-reference
changes only. Recorded results and captured fixtures retain their bytes.

`src/labeler/env.py` and `src/shot_design/env.py` each provide one shared resolver.
Python readers and shell launchers use these helpers. They accept either spelling,
prefer a present new name (including an empty value), and use the old name only
when the new name is absent. A fallback logs one line naming the replacement,
without printing the value; Python warning filters do not turn this log into an
exception. YAML interpolation preserves mapping-key precedence and supports
either namespace. Origin reporting identifies the spelling that supplied the root.

## Pixi outcome: prescribed fallback retained

The requested new-name probe failed before environment activation: this pixi
version rejects underscores in `shot_design`. Accordingly, environment **and
feature** names remain `labelmaker`, `ideate`, and `ideate-cpu`; tasks and Python
packages use the new names. `pixi.lock` is byte-identical to the starting commit.
No install, lock, update, or unfrozen test command was run.

The symlink probe used exactly these worktree links:

```text
.pixi/envs/labeler -> /scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker
.pixi/envs/shot_design-cpu -> /scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/ideate-cpu
.pixi/envs/shot_design -> /scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/ideate
```

Probe command (exit 1):

```bash
PYTHONDONTWRITEBYTECODE=1 XDG_CACHE_HOME=/tmp/r1rename/cache \
PIXI_CACHE_DIR=/tmp/r1rename/pixi-cache \
pixi run --frozen --no-install \
  --manifest-path /scratch/gpfs/nc1514/FusionAIHub-R1/pyproject.toml \
  -e labeler python -c "import sys; print(sys.executable)"
```

Verbatim probe output:

```text
Error:   × Failed to parse environment name 'shot_design', please use only lowercase
  │ letters, numbers and dashes
     ╭─[/scratch/gpfs/nc1514/FusionAIHub-R1/pyproject.toml:344:1]
 343 │ labeler = ["labeler", "fdp"]
 344 │ shot_design = ["shot_design", "cuda"]
     · ───────────
 345 │ shot_design-cpu = ["shot_design", "shot_design-cpu"]
     ╰────
```

Immediately after that probe, `du -s .pixi` printed `3\t.pixi`, and
`find .pixi -type f | wc -l` printed `0`. The symlinks and directories were removed.
The prescribed fallback probe through the main manifest exited 0 and printed:

```text
/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/python
```

**Deviation:** a subsequent attempt to validate the fallback worktree manifest
with `--frozen --no-install` and no environment symlinks exited 127 (`python:
command not found`). Pixi created `.pixi/.gitignore` and `.pixi/.condapackageignore`.
Both files and `.pixi` were removed immediately. Thus the requested zero-regular-file
condition held for the symlink probe and at completion, but was briefly violated by
that additional pixi invocation. All subsequent execution used the main manifest.

## Preserved data and scope

The following remain unchanged: production roots and default path values;
`envs/phase3`; `models/tokeye`; data filenames; dictionary/manifest keys including
`labelmaker_git_sha`, `labelmaker_root`, and `labelmaker_version`; HDF5 attributes;
model-card `labelmaker:` keys; signal-config `labelmaker` fields; stored reader and
resolver values such as `labelmaker:archive`; and the MCP `ideate://manifest` resource identifier; and the `ideate-raw-v1` and
`ideate-frame-codes-provenance-v1` schema identifiers.

Git object hashes confirm byte identity for 18 captured/data fixtures and 74
non-script output/result files. Fixture generator Python references were updated;
the generated fixtures were not regenerated. A structural audit compared all
literal dictionary keys in 94 changed source/script files and found no changes.
The protected production-path and schema occurrences in those files also match.

`docs/superpowers/**`, other `.superpowers/sdd/**` files, `data/**`,
`src/tokamak_foundation_model/**`, `src/faith/**`, `tests/ignite/**`, `tests/e2e/**`,
`CLAUDE.md`, and `AGENTS.md` match the starting commit. The task brief is unchanged.

## Verification

Both full suites exited 0 under `-W error` using the prescribed fallback
environments and main manifest. These are the actual results, not the brief's
production-enabled counts:

| Suite | Passed | Skipped | New fallback cases |
| --- | ---: | ---: | ---: |
| `tests/labeler` | 1,746 | 30 | 118 |
| `tests/shot_design` | 1,351 | 22 | 104 |

Totals are 1,776 and 1,373, respectively: the original total counts plus the new
fallback cases. Relative to the brief's expected 1,655 passed/3 skipped and
1,269 passed, another 27 labeler cases and 22 shot-design cases need production
data or trained weights. They were skipped to honor the binding no-production-read
rule; their production-enabled results are not verified here.

The existing suites contain both marked real-data tests and unmarked tests that
check whether production paths exist at collection. `bwrap` could not create a
namespace (`Creating new namespace failed: No space left on device`). A temporary
libc interposer at `/tmp/r1rename/no_production.so` therefore returned `ENOENT` for
standard open/stat/directory calls under `/scratch/gpfs/EKOLEMEN` and
`/projects/EKOLEMEN`. Its source is `/tmp/r1rename/no_production.c`; all harness,
test, and cache output is under `/tmp/r1rename`. No production files were copied.

Six existing validation unit tests also depended on real `y.npy`/`z.npy` arrays
despite mocking their per-shot behavior. They now use a small synthetic archive
fixture; the absent-shot count test also uses its existing fake adapter. Their
assertions are unchanged. This necessary test-isolation change adds no test cases
and changes no production validation behavior.

Commands, from the worktree (the exports were supplied to each command):

```bash
export LD_PRELOAD=/tmp/r1rename/no_production.so
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TMPDIR=/tmp/r1rename XDG_CACHE_HOME=/tmp/r1rename/cache
export PIXI_CACHE_DIR=/tmp/r1rename/pixi-cache HF_HOME=/tmp/r1rename/huggingface
export TORCH_HOME=/tmp/r1rename/torch MPLCONFIGDIR=/tmp/r1rename/matplotlib
export NUMBA_CACHE_DIR=/tmp/r1rename/numba

pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker python -m pytest tests/labeler -q -W error -p no:cacheprovider -rs --basetemp=/tmp/r1rename/final-labeler
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e ideate-cpu python -m pytest tests/shot_design -q -W error -p no:cacheprovider -rs --basetemp=/tmp/r1rename/final-shot-design

/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/ruff check --no-cache src/labeler src/shot_design scripts/labeler tests/labeler tests/shot_design
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e ideate-cpu python -m shot_design --help
pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker python -m labeler.run --help
```

Ruff: **All checks passed.** Both CLI help commands exited 0. `bash -n` passed
for all affected shell launchers. `git diff --check` passed.

The initial 16 labeler and 4 shot-design compatibility tests were run before
implementing the helpers and failed on the missing behavior. After implementation,
the config/fallback groups passed 126 and 115 tests. After review corrections,
100 UI/MCP tests passed under the same pixi form and `-W error`. A Node VM test
confirmed response caveats survive the preserved `X-Ideate-Caveats` header
(failed before the correction, passed after). A controlled `assess_labels_a.main()`
run with a synthetic 500-shot list confirmed the census receives the preserved
sibling `ideate/db` location, writing only under `/tmp/r1rename`.

The code-review skill's read-only reviewer found those two issues; both were fixed
and the reviewer confirmed no outstanding findings. The full suites preceded
these final UI/header and untested assessment-launcher corrections; the affected
UI/MCP paths and launcher were verified afterward with the checks above.

After pytest's successful labeler summary, the unchanged third-party XRootD
finalizer printed a `FutureWarning` about `torch.distributed.reduce_op`. The suite
still exited 0; no warning filters were added to suppress it. Full logs remain at
`/tmp/r1rename/final-labeler.log` and `/tmp/r1rename/final-shot-design.log`.

The MCP transport test's sanitized subprocess environment initially dropped
`PYTHONDONTWRITEBYTECODE`; the test now passes it explicitly. Generated package
bytecode was removed before the final residual grep. The worktree `.pixi` is absent.

### Final suite summaries and skip reasons

```text
SKIPPED [1] tests/labeler/test_adapter_fidelity.py:29: weights missing: /projects/EKOLEMEN/simple_ae_predictor/models/rt_multi_io/mse_bin_os_w
SKIPPED [1] tests/labeler/test_adapter_fidelity.py:37: weights missing: /projects/EKOLEMEN/simple_ae_predictor/models/rt_multi_io/mse_bin_os_w
SKIPPED [1] tests/labeler/test_adapter_fidelity.py:55: weights missing: /projects/EKOLEMEN/simple_ae_predictor/models/rt_multi_io/mse_bin_os_w
SKIPPED [1] tests/labeler/test_adapter_fidelity.py:104: weights missing: /projects/EKOLEMEN/simple_ae_predictor/models/rt_multi_io/mse_bin_os_w
SKIPPED [1] tests/labeler/test_catalog.py:61: corpus not available: /scratch/gpfs/EKOLEMEN/foundation_model
SKIPPED [1] tests/labeler/test_catalog.py:68: corpus or tm archive missing
SKIPPED [1] tests/labeler/test_dsm_pickle.py:162: upstream survival checkpoint not mounted
SKIPPED [1] tests/labeler/test_events_masks.py:830: corpus file not mounted: /scratch/gpfs/EKOLEMEN/foundation_model/198658_processed.h5
SKIPPED [1] tests/labeler/test_events_unet.py:86: checkpoint not mounted: /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/models/tokeye/big_tf_unet_251210.pt
SKIPPED [1] tests/labeler/test_events_unet.py:94: checkpoint not mounted: /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/models/tokeye/big_tf_unet_251210.pt
SKIPPED [1] tests/labeler/test_events_unet.py:101: checkpoint not mounted: /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/models/tokeye/big_tf_unet_251210.pt
SKIPPED [1] tests/labeler/test_events_unet.py:117: checkpoint not mounted: /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/models/tokeye/big_tf_unet_251210.pt
SKIPPED [1] tests/labeler/test_events_unet.py:127: checkpoint not mounted: /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/models/tokeye/big_tf_unet_251210.pt
SKIPPED [1] tests/labeler/test_events_unet.py:134: checkpoint not mounted: /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/models/tokeye/big_tf_unet_251210.pt
SKIPPED [1] tests/labeler/test_events_unet.py:149: checkpoint not mounted: /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/models/tokeye/big_tf_unet_251210.pt
SKIPPED [1] tests/labeler/test_events_unet.py:171: checkpoint not mounted: /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/models/tokeye/big_tf_unet_251210.pt
SKIPPED [1] tests/labeler/test_keras_h5.py:346: upstream weights not available: /projects/EKOLEMEN/simple_ae_predictor/models/rt_multi_io/mse_bin_os_w
SKIPPED [1] tests/labeler/test_keras_h5.py:364: upstream weights not available: /projects/EKOLEMEN/simple_ae_predictor/models/rt_multi_io/mse_bin_os_w
SKIPPED [1] tests/labeler/test_keras_h5.py:380: upstream weights not available: /projects/EKOLEMEN/simple_ae_predictor/models/rt_multi_io/mse_bin_os_w
SKIPPED [1] tests/labeler/test_l14perf_real_identity.py:45: set L14PERF_REAL_ROOT to an l14perf scratch directory
SKIPPED [1] tests/labeler/test_label_quality.py:509: run `features` on shot 185945 first
SKIPPED [1] tests/labeler/test_reconstruction.py:98: tm archive not available
SKIPPED [1] tests/labeler/test_reconstruction.py:108: run `features` on shot 185945 first
SKIPPED [1] tests/labeler/test_resolve_archive.py:100: archive not available: /projects/EKOLEMEN/profile_predictor/DATA/new_h5_data/example_191450_183224.h5
SKIPPED [1] tests/labeler/test_resolve_corpus.py:82: corpus not available: /scratch/gpfs/EKOLEMEN/foundation_model
SKIPPED [1] tests/labeler/test_resolve_fdp.py:454: live fdp fetch is opt-in: --run-live or LABELER_FDP=1
SKIPPED [1] tests/labeler/test_resolve_fdp.py:476: live fdp fetch is opt-in: --run-live or LABELER_FDP=1
SKIPPED [1] tests/labeler/test_tearing_adapter.py:275: upstream weights not available: /projects/EKOLEMEN/simple_ae_predictor/models/rt_multi_io/mse_bin_os_w
SKIPPED [1] tests/labeler/test_tearing_adapter.py:290: weights not yet copied into the data root
SKIPPED [1] tests/labeler/test_tearing_adapter.py:303: upstream weights not available: /projects/EKOLEMEN/simple_ae_predictor/models/rt_multi_io/mse_bin_os_w
1746 passed, 30 skipped in 271.02s (0:04:31)
```

```text
SKIPPED [1] tests/shot_design/test_describe.py:337: /scratch/gpfs/EKOLEMEN/nc1514/shot-recommender/db not mounted
SKIPPED [1] tests/shot_design/test_flags.py:426: /scratch/gpfs/EKOLEMEN/nc1514/shot-recommender/db not mounted
SKIPPED [1] tests/shot_design/test_flags.py:457: /scratch/gpfs/EKOLEMEN/nc1514/shot-recommender/db not mounted
SKIPPED [1] tests/shot_design/test_ignite.py:185: no IGNITE bundle at /scratch/gpfs/EKOLEMEN/nc1514/shot-recommender/models/IGNITE (shot_design model --download)
SKIPPED [1] tests/shot_design/test_ignite.py:213: no production frame codes for 190090
SKIPPED [1] tests/shot_design/test_ignite.py:238: no official 185601
SKIPPED [1] tests/shot_design/test_integration_real.py:70: real staged/text/QH stores not mounted
SKIPPED [1] tests/shot_design/test_integration_real.py:112: real staged/text/QH stores not mounted
SKIPPED [1] tests/shot_design/test_integration_real.py:122: real staged/text/QH stores not mounted
SKIPPED [1] tests/shot_design/test_integration_real.py:135: real staged/text/QH stores not mounted
SKIPPED [1] tests/shot_design/test_integration_real.py:151: real staged/text/QH stores not mounted
SKIPPED [1] tests/shot_design/test_integration_real.py:160: real staged/text/QH stores not mounted
SKIPPED [1] tests/shot_design/test_logs.py:271: the DIII-D text corpus is not mounted
SKIPPED [1] tests/shot_design/test_retrieval.py:660: /scratch/gpfs/EKOLEMEN/d3d_fusion_data test shots not mounted
SKIPPED [1] tests/shot_design/test_retrieval.py:670: /scratch/gpfs/EKOLEMEN/d3d_fusion_data test shots not mounted
SKIPPED [1] tests/shot_design/test_retrieval.py:680: /scratch/gpfs/EKOLEMEN/d3d_fusion_data test shots not mounted
SKIPPED [1] tests/shot_design/test_retrieval.py:697: /scratch/gpfs/EKOLEMEN/d3d_fusion_data test shots not mounted
SKIPPED [1] tests/shot_design/test_retrieval.py:705: /scratch/gpfs/EKOLEMEN/d3d_fusion_data test shots not mounted
SKIPPED [1] tests/shot_design/test_seed.py:90: no IGNITE bundle at /scratch/gpfs/EKOLEMEN/nc1514/shot-recommender/models/IGNITE
SKIPPED [1] tests/shot_design/test_seed.py:142: no IGNITE bundle at /scratch/gpfs/EKOLEMEN/nc1514/shot-recommender/models/IGNITE
SKIPPED [1] tests/shot_design/test_seed.py:167: no IGNITE bundle at /scratch/gpfs/EKOLEMEN/nc1514/shot-recommender/models/IGNITE
SKIPPED [1] tests/shot_design/test_seed.py:342: no shipped cache at /scratch/gpfs/EKOLEMEN/nc1514/shot-recommender/models/IGNITE/frame_codes/202537.pt
1351 passed, 22 skipped in 78.02s (0:01:18)
```

Temporary no-production interposer source SHA-256: `8af5f98125f4fbffaa8db3614d6196dbaafef1c9e1577233badd20f4dadd3ce0`.

## Controller's post-merge checklist

- Merge the rename commits together; the first commit intentionally contains moves
  without the corresponding import rewrites.
- **Keep the existing pixi environment directories for this result.** The installed
  pixi cannot accept the requested underscore names. Do not execute the following
  future migration commands unless that naming constraint has first been resolved
  and the manifest/lock environment keys have been updated together:

  ```bash
  mv /scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker /scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labeler
  mv /scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/ideate /scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/shot_design
  mv /scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/ideate-cpu /scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/shot_design-cpu
  ```

  The symlink alternative leaves existing environments in place (also conditional;
  aliases alone do not make underscore names valid to pixi):

  ```bash
  ln -s /scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker /scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labeler
  ln -s /scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/ideate /scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/shot_design
  ln -s /scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/ideate-cpu /scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/shot_design-cpu
  ```

  If a future migration moves directories, review absolute interpreter paths and
  keep old aliases as needed for existing conda prefixes, launchers, and shebangs.
- Restart the long-lived UI with `python -m shot_design serve` (pixi environment
  `ideate-cpu` under this fallback), retaining its existing port/token options.
  Restart the MCP client/server process so it reloads `.mcp.json`'s `shot_design`
  server and `python -m shot_design.mcp`. Restart notebooks/workers that still have
  the old Python packages imported. No running service was restarted by this task.
- Update controller memory, active operational notes, shell exports, job submission
  snippets, and launch aliases: `IDEATE_*` becomes `SHOT_DESIGN_*`, and
  `LABELMAKER_*` becomes `LABELER_*`. Include `DATA_ROOT`, `CORPUS`, `CONFIG_DIR`,
  `PATHS`, `HF_ONLINE`, `TEXT_ROOT`, `IGNITE_CKPT`, `PY`, and `ENV` for shot design;
  and `ROOT`, `CORPUS`, `TEXT_ROOT`, `LOGS_JSONL`, `LABEL_TABLES`, `FDP`, `GIT_SHA`,
  `DEMO_OUT`, `DEMO_CONFIG`, and `DEMO_FORCE` for labeler. Keep pixi environment names
  as documented above. Historical plans/ledgers and captured records were not edited.
- **Do not rename the two production data roots:**
  `/scratch/gpfs/EKOLEMEN/nc1514/labelmaker` and
  `/scratch/gpfs/EKOLEMEN/nc1514/ideate`. Their descendants and stored keys remain
  compatible; the phase-3 environment and TokEye weights keep their locations.

## Commit groups

```text
3229a6d rename: move package and companion trees without content changes
b6b09fb rename: update package imports and code references while preserving data contracts
e708aae rename: update packaging and MCP while retaining unsupported pixi environment names
f09b636 rename: resolve new environment variables with tested legacy fallbacks
6fddbb4 rename: update documentation and model cards with preserved data names
96d779a rename: update launchers and output scripts with shared environment fallbacks
```

The final report is its own `rename:` commit. Every commit has the required
`Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` trailer. No commit was
made while a test suite was running. The branch/worktree are retained for the
controller; no merge or push was performed.

## Residual grep: every remaining hit and its reason

Exact command:

```bash
grep -rnE "\b(ideate|labelmaker)\b|IDEATE_|LABELMAKER_" src tests scripts configs docs/*.md pyproject.toml .mcp.json
```

The remaining names include the brief's explicit data/schema/fixture exceptions,
unchanged historical link targets, and the exercised pixi fallback. They cannot
all be removed without breaking those requirements. No old-package import or CLI
module reference remains in the mutable scope.

Each matched line below carries one or more reason codes:

| Code | Reason |
| --- | --- |
| V | Intentional legacy environment fallback, its tests, or migration documentation. |
| E | Retained pixi environment/feature name or interpreter path after the failed probe. |
| D | Unchanged on-disk production, output, or synthetic-fixture data path. |
| S | Preserved serialized schema/card/signal field, source value, or converter provenance identifier. |
| R | Preserved MCP manifest-resource identifier. |
| H | Link to an unchanged historical plan/spec filename. |
| F | Captured fixture or recorded data content, kept byte-identical. |

All **316 matched lines** (including multiple occurrences on a line):

```text
[S] src/labeler/run.py:434:        sha_map = (registry.read_card(slug)["labelmaker"].get("upstream") or {}).get(
[S] src/labeler/run.py:1071:                slug: card["labelmaker"].get("upstream", {})
[D] src/labeler/ae/transform.py:10:``outputs/labelmaker/ae/scripts/pin_transform.py`` runs both side by side on a
[H] src/labeler/events/__init__.py:7:every row. See docs/superpowers/specs/2026-09-07-recommender-labelmaker-v2.md.
[V] src/labeler/env.py:11:    """Prefer LABELER_* even when empty; warn only when LABELMAKER_* supplies it.
[V] src/labeler/env.py:16:    if name.startswith("LABELMAKER_"):
[V] src/labeler/env.py:17:        name = "LABELER_" + name[len("LABELMAKER_"):]
[V] src/labeler/env.py:21:        old = "LABELMAKER_" + name[len("LABELER_"):]
[H] src/labeler/__init__.py:5:docs/superpowers/specs/2026-09-03-labelmaker-design.md.
[S] src/labeler/models/d3d_elm_time_to_event_dsm/README.md:18:labelmaker:
[D] src/labeler/models/d3d_elm_time_to_event_dsm/README.md:26:    path: /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/models/d3d_elm_time_to_event_dsm
[D] src/labeler/models/d3d_elm_time_to_event_dsm/README.md:332:`/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/models/d3d_elm_time_to_event_dsm/`.
[D] src/labeler/models/d3d_elm_time_to_event_dsm/spec.py:86:UPSTREAM = Path("/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/models/d3d_elm_time_to_event_dsm")
[S] src/labeler/models/d3d_tearing_onset_cnn1d/README.md:71:labelmaker:
[S] src/labeler/models/d3d_inpa_image_cnn/README.md:19:labelmaker:
[S] src/labeler/models/d3d_ech_deposition_torbeamnn/README.md:19:labelmaker:
[S] src/labeler/models/d3d_ae_activity_seldnet/README.md:20:labelmaker:
[D] src/labeler/models/d3d_ae_activity_seldnet/README.md:28:    path: /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/models/d3d_ae_activity_seldnet
[D] src/labeler/models/d3d_ae_activity_seldnet/README.md:156:env). Evidence: `outputs/labelmaker/ae/scripts/pin_transform.py` and
[D] src/labeler/models/d3d_ae_activity_seldnet/README.md:157:`outputs/labelmaker/ae/transform_pin.json`.
[D] src/labeler/models/d3d_ae_activity_seldnet/spec.py:63:UPSTREAM = Path("/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/models/d3d_ae_activity_seldnet")
[S] src/labeler/models/d3d_tearing_time_to_event_dsm_continued/README.md:20:labelmaker:
[D] src/labeler/models/d3d_tearing_time_to_event_dsm_continued/README.md:28:    path: /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/models/d3d_tearing_time_to_event_dsm_continued
[D] src/labeler/models/d3d_tearing_time_to_event_dsm_continued/README.md:375:`outputs/labelmaker/presentation_continued/` in the FusionAIHub checkout; the
[D] src/labeler/models/d3d_tearing_time_to_event_dsm_continued/spec.py:45:UPSTREAM = Path("/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/models/"
[S] src/labeler/models/d3d_ech_beam_fate_mlp/README.md:19:labelmaker:
[S] src/labeler/models/registry.py:5:`labelmaker:` block, and a `spec.py` exposing `ADAPTER`. The card is the
[S] src/labeler/models/registry.py:62:    return read_card(slug)["labelmaker"]["status"]
[S] src/labeler/models/registry.py:94:    card = read_card(slug)["labelmaker"]
[S] src/labeler/models/registry.py:135:    card = read_card(slug)["labelmaker"]
[S] src/labeler/models/README.md:7:| `README.md` | HuggingFace-style model card. YAML front matter plus a custom `labelmaker:` block; the front matter is what `registry.py` parses. |
[H] src/labeler/models/README.md:50:and its design is `docs/superpowers/specs/2026-09-05-labelmaker-phase3-design.md`
[S] src/labeler/models/d3d_tearing_time_to_event_dsm/README.md:19:labelmaker:
[D] src/labeler/models/d3d_tearing_time_to_event_dsm/README.md:237:  `outputs/labelmaker/presentation_continued/`.
[H] src/labeler/models/d3d_tearing_time_to_event_dsm/README.md:359:`docs/superpowers/specs/2026-09-05-labelmaker-phase3-design.md` section 2.1.
[D] src/labeler/models/d3d_tearing_time_to_event_dsm/README.md:419:Measured once outside `validate` (2026-09-05, `outputs/labelmaker/dsm_onset_quality.py`
[D] src/labeler/models/d3d_tearing_time_to_event_dsm/README.md:463:per-shot scores are in `outputs/labelmaker/analysis/{187199,186545}/` in the
[D] src/labeler/models/d3d_tearing_time_to_event_dsm/README.md:506:`outputs/labelmaker/presentation/held_out/`.
[S] src/labeler/models/d3d_kinetic_equilibrium_rtcakenn/README.md:19:labelmaker:
[D] src/labeler/config.py:17:DEFAULT_ROOT = Path("/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")
[S] src/shot_design/shotdb/build.py:273:            assumed=assumed and sig is not None and sig.source in ("staged", "labelmaker"),
[S] src/shot_design/shotdb/corpus_signals.py:188:        labeler = [s for s in installed if s.labelmaker is not None and signals[s.name] is None]
[S] src/shot_design/shotdb/corpus_signals.py:351:            feature = spec.labelmaker.feature
[S] src/shot_design/shotdb/corpus_signals.py:362:            y = _reduce(_prepare(np.atleast_2d(arr.y), spec, spec.labelmaker.scale),
[S] src/shot_design/shotdb/corpus_signals.py:363:                        spec.labelmaker.reduce)
[S] src/shot_design/shotdb/corpus_signals.py:371:                spec.labelmaker.units or arr.attrs.get("units") or spec.units,
[S] src/shot_design/shotdb/corpus_signals.py:372:                "labelmaker",
[S] src/shot_design/shotdb/corpus_signals.py:375:                f"labelmaker:{resolver}",
[S] src/shot_design/shotdb/legacy_raw.py:42:SCHEMA = "ideate-raw-v1"
[S] src/shot_design/shotdb/legacy_raw.py:119:    found in: `stamp_file` puts schema="ideate-raw-v1" on the file, and `write_group` puts
[S] src/shot_design/shotdb/reader.py:75:    # <shot>_processed.h5 group and "labelmaker" a canonical feature of <shot>_features.h5.
[S] src/shot_design/shotdb/reader.py:76:    source: Literal["staged", "fetched", "corpus", "labelmaker"]
[S] src/shot_design/labels/join.py:209:    A card records its operating point as `threshold:` on an entry of its `labelmaker.outputs`
[S] src/shot_design/labels/join.py:213:    block = dict(card.get("labelmaker") or {})
[V] src/shot_design/env.py:11:    """Prefer SHOT_DESIGN_* even when empty; warn only when IDEATE_* supplies it.
[V] src/shot_design/env.py:16:    if name.startswith("IDEATE_"):
[V] src/shot_design/env.py:17:        name = "SHOT_DESIGN_" + name[len("IDEATE_"):]
[V] src/shot_design/env.py:21:        old = "IDEATE_" + name[len("SHOT_DESIGN_"):]
[S] src/shot_design/design/provenance.py:54:SCHEMA = "ideate-frame-codes-provenance-v1"
[S] src/shot_design/schema.py:105:    raw_sources: dict[str, Literal["staged", "fetched", "corpus", "labelmaker"]] = Field(
[S] src/shot_design/schema.py:109:    # ("labelmaker:archive", "labelmaker:fdp", "corpus"). Empty on a legacy build, where
[R] src/shot_design/mcp/server.py:1:"""`build_server()`: the four tools of `tools.py` and the `ideate://manifest` resource.
[R] src/shot_design/mcp/server.py:97:        "ideate://manifest",
[E] src/shot_design/mcp/__init__.py:6:what `pixi run -e ideate-cpu shot_design-mcp` and the `.mcp.json` at the repo root invoke.
[V] src/shot_design/config.py:43:                        labeler_getenv if m[1].startswith(("LABELER_", "LABELMAKER_"))
[V] src/shot_design/config.py:121:        name = "SHOT_DESIGN_DATA_ROOT" if "SHOT_DESIGN_DATA_ROOT" in os.environ else "IDEATE_DATA_ROOT"
[V] src/shot_design/config.py:125:        name = "SHOT_DESIGN_PATHS" if "SHOT_DESIGN_PATHS" in os.environ else "IDEATE_PATHS"
[S] src/shot_design/config.py:203:    labelmaker: LabelerAddress | None = None
[D] tests/labeler/test_config.py:9:    assert p.root == Path("/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")
[E] tests/labeler/test_tokeye_sbatch.py:50:    assert "-e labelmaker python" in text
[E] tests/labeler/test_tokeye_sbatch.py:71:    assert "-e labelmaker python" in text
[S] tests/labeler/test_tearing_dsm_adapter.py:129:    assert [f["units"] for f in registry.read_card(dsm.SLUG)["labelmaker"]["outputs"]] == OUTPUT_UNITS
[D] tests/labeler/test_l14perf_real_identity.py:47:    assert root.is_relative_to("/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/l14perf")
[D] tests/labeler/test_l14perf_real_identity.py:49:    production = Path("/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")
[D] tests/labeler/test_ae_model.py:6:    /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/envs/phase3/bin/python \
[E] tests/labeler/conftest.py:4:`pixi run -e labelmaker python -m pytest tests/labeler`, with no server,
[E] tests/labeler/conftest.py:13:    pixi run -e labelmaker fdp run python -m pytest tests/labeler \
[S] tests/labeler/conftest.py:78:        lambda slug: {"labelmaker": {"upstream": {"sha256": {"fake.h5": "00"}}}},
[V] tests/labeler/conftest.py:390:        if name.startswith(("SHOT_DESIGN_", "IDEATE_", "LABELER_", "LABELMAKER_")):
[S] tests/labeler/test_registry.py:6:REQUIRED_TOP = ("pipeline_tag", "tags", "library_name", "labelmaker")
[S] tests/labeler/test_registry.py:15:    assert card["labelmaker"]["status"] == "scaffold"
[S] tests/labeler/test_registry.py:30:        lm = card["labelmaker"]
[S] tests/labeler/test_registry.py:50:        assert card["labelmaker"]["blocked_on"], f"{slug}: no blocked_on entries"
[S] tests/labeler/test_label_quality.py:417:        "labelmaker:\n"
[S] tests/labeler/test_label_quality.py:438:    assert parsed["labelmaker"]["status"] == "implemented"
[S] tests/labeler/test_label_quality.py:480:    approximations_before = registry.parse_card(text_before)["labelmaker"][
[S] tests/labeler/test_label_quality.py:503:    approximations_after = registry.parse_card(copy.read_text())["labelmaker"][
[S] tests/labeler/test_run.py:195:        lambda slug: {"labelmaker": {"upstream": {"sha256": {"absent.h5": "00" * 32}}}},
[S] tests/labeler/test_run.py:217:        lambda slug: {"labelmaker": {"upstream": {"sha256": {"absent.h5": "00" * 32}}}},
[S] tests/labeler/test_run.py:249:        lambda slug: {"labelmaker": {"upstream": {"sha256": {"fake.h5": "00"}}}},
[F] tests/labeler/data/qmin_regimes_recommender_v1.json:2:  "features_root": "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/features",
[F] tests/labeler/data/qmin_regimes_recommender_v1.json:15:  "note": "Written by scripts/labelmaker/qmin_regime_census.py. The `ungated` block is the rule with the Ip flat-top gate REMOVED and is the justification for having the gate at all.",
[F] tests/labeler/data/qmin_regimes_recommender_v1.json:1237:  "shot_file": "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/recommender_v1.txt",
[F] tests/labeler/data/jobstats/2925412_5.jobstats.txt:7:       Job Name: ideate-encode-cpu
[F] tests/labeler/data/jobstats/2925387_0.jobstats.txt:7:       Job Name: ideate-encode-cpu
[F] tests/labeler/data/jobstats/2925387_1.jobstats.txt:7:       Job Name: ideate-encode-cpu
[V] tests/labeler/test_env.py:22:    new, old = f"LABELER_{suffix}", f"LABELMAKER_{suffix}"
[V] tests/labeler/test_env.py:40:    monkeypatch.setenv("LABELMAKER_ROOT", "/unused")
[V] tests/labeler/test_env.py:42:    assert not any("LABELMAKER_ROOT" in r.getMessage() for r in caplog.records)
[V] tests/labeler/test_env.py:53:    new, old = "LABELER_" + suffix, "LABELMAKER_" + suffix
[V] tests/labeler/test_env.py:79:    monkeypatch.setenv("LABELMAKER_ROOT" if legacy else "LABELER_ROOT", "a path with spaces")
[D] tests/labeler/test_ae_transform.py:5:`outputs/labelmaker/ae/scripts/pin_transform.py`, whose measured result (max
[S] tests/labeler/test_tearing_dsm_continued.py:49:    got = registry.read_card(cont.SLUG)["labelmaker"]
[S] tests/labeler/test_tearing_dsm_continued.py:50:    want = registry.read_card(base.SLUG)["labelmaker"]
[D] tests/shot_design/test_config.py:52:    monkeypatch.setenv("SHOT_DESIGN_DATA_ROOT", "/tmp/ideate-test")
[D] tests/shot_design/test_config.py:54:    assert p.raw_dir == Path("/tmp/ideate-test/raw")
[D] tests/shot_design/test_config.py:117:    assert paths.data_root == Path("/scratch/gpfs/EKOLEMEN/nc1514/ideate")
[D] tests/shot_design/test_integration_real.py:51:        mp.setenv("SHOT_DESIGN_DATA_ROOT", str(tmp_path_factory.mktemp("ideate-real")))
[D] tests/shot_design/test_retrieval.py:649:        mp.setenv("SHOT_DESIGN_DATA_ROOT", str(tmp_path_factory.mktemp("ideate-query")))
[S] tests/shot_design/test_labels_join.py:617:    card = {"labelmaker": {"slug": "d3d_x", "outputs": [
[S] tests/shot_design/test_labels_join.py:639:        card = registry.read_card(r.slug)["labelmaker"]
[E] tests/shot_design/test_docs_scratch_db.py:30:    """`pixi run -e ideate*` overrides these, whatever the caller exported -- which is what
[E] tests/shot_design/test_docs_scratch_db.py:33:    env = pinned["tool"]["pixi"]["feature"]["ideate"]["target"]["unix"]["activation"]["env"]
[E] tests/shot_design/test_docs_scratch_db.py:42:    assert ".pixi/envs/ideate-cpu/bin/python -m shot_design" in text
[S] tests/shot_design/conftest.py:22:OUR_SCHEMA = "ideate-raw-v1"  # scripts/fetch_shots.SCHEMA; pinned equal in test_fetch_plan
[D,S] tests/shot_design/conftest.py:658:    root = tmp_path / "labelmaker"
[V] tests/shot_design/conftest.py:796:        if name.startswith(("SHOT_DESIGN_", "IDEATE_", "LABELER_", "LABELMAKER_")):
[S] tests/shot_design/test_corpus_signals.py:176:    assert bt.source == "labelmaker" and bt.resolver == "labelmaker:archive"
[S] tests/shot_design/test_corpus_signals.py:232:        name="ne_probe", labelmaker={"feature": "ne_zipfit", "reduce": reduce}
[S] tests/shot_design/test_corpus_signals.py:361:        labelmaker=config.LabelerAddress(feature="ip"),
[S] tests/shot_design/test_corpus_signals.py:374:        labelmaker=config.LabelerAddress(feature="ip"),
[S] tests/shot_design/test_corpus_signals.py:378:    assert sig is not None and sig.source == "labelmaker"
[S] tests/shot_design/test_build_store.py:342:    assert rec.feature_resolvers["ip"] == "labelmaker:fdp"
[S] tests/shot_design/test_build_store.py:343:    assert rec.feature_resolvers["bt"] == "labelmaker:archive"
[S] tests/shot_design/test_build_store.py:345:    assert rec.raw_sources["dalpha"] == "corpus" and rec.raw_sources["bt"] == "labelmaker"
[S] tests/shot_design/test_build_store.py:403:    assert json.loads(row["feature_resolvers"])["bt"] == "labelmaker:archive"
[D] tests/shot_design/test_select.py:818:    by_dir = {"/data/ideate/frame_codes": 500, "/models/IGNITE/frame_codes": 10}
[D] tests/shot_design/test_select.py:838:    assert "500  /data/ideate/frame_codes" in text and "10  /models/IGNITE/frame_codes" in text
[R] tests/shot_design/test_mcp.py:1012:            manifest = await client.read_resource("ideate://manifest")
[R] tests/shot_design/test_mcp.py:1021:    assert [str(r.uri) for r in resources.resources] == ["ideate://manifest"]
[R] tests/shot_design/test_mcp.py:1039:            return await client.read_resource("ideate://manifest")
[D,S] tests/shot_design/test_ui.py:42:    monkeypatch.setenv("LABELER_ROOT", str(tmp_path / "labelmaker"))
[V] tests/shot_design/test_env.py:15:    monkeypatch.delenv("IDEATE_DATA_ROOT", raising=False)
[V] tests/shot_design/test_env.py:17:    monkeypatch.delenv("IDEATE_PATHS", raising=False)
[V] tests/shot_design/test_env.py:21:        monkeypatch.setenv("IDEATE_DATA_ROOT", str(tmp_path / "old"))
[V] tests/shot_design/test_env.py:25:    notices = [r.getMessage() for r in caplog.records if "IDEATE_DATA_ROOT" in r.getMessage()]
[V] tests/shot_design/test_env.py:33:    monkeypatch.setenv("IDEATE_DATA_ROOT", "/unused")
[V] tests/shot_design/test_env.py:38:@pytest.mark.parametrize("prefix", ["SHOT_DESIGN_", "IDEATE_", "LABELER_", "LABELMAKER_"])
[V] tests/shot_design/test_env.py:41:    old = "LABELMAKER_ROOT" if prefix.startswith("LABEL") else "IDEATE_DATA_ROOT"
[V] tests/shot_design/test_env.py:61:    name = "IDEATE_PATHS" if legacy else "SHOT_DESIGN_PATHS"
[V] tests/shot_design/test_env.py:73:    name = "IDEATE_CONFIG_DIR" if legacy else "SHOT_DESIGN_CONFIG_DIR"
[V] tests/shot_design/test_env.py:80:    assert ("IDEATE_CONFIG_DIR is deprecated" in result.stderr) == legacy
[V] tests/shot_design/test_env.py:91:    new, old = "SHOT_DESIGN_" + suffix, "IDEATE_" + suffix
[V] tests/shot_design/test_env.py:117:    monkeypatch.setenv("IDEATE_DATA_ROOT" if legacy else "SHOT_DESIGN_DATA_ROOT", "a path with spaces")
[D,S] tests/shot_design/test_write_root.py:73:    monkeypatch.setenv("LABELER_ROOT", str(paths.data_root / "labelmaker"))
[D,S] tests/shot_design/test_write_root.py:123:        lm = paths.data_root / "labelmaker"
[D] scripts/labeler/assess_labels_a.py:178:        result = census(paths, paths.root.parent / "ideate/db", pool)
[D] scripts/labeler/features_recommender_v1.sbatch:10:#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out
[D] scripts/labeler/features_recommender_v1.sbatch:11:#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out
[D,E] scripts/labeler/features_recommender_v1.sbatch:30:ROOT="$("/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/python" "$REPO/src/labeler/env.py" LABELER_ROOT "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")"
[E] scripts/labeler/features_recommender_v1.sbatch:31:ENV="$REPO/.pixi/envs/labelmaker"
[D] scripts/labeler/ae_dataset.sbatch:8:#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out
[D] scripts/labeler/ae_dataset.sbatch:9:#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out
[D] scripts/labeler/ae_dataset.sbatch:24:export PYTHONPATH=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/ae/pylibs:$REPO/src
[D] scripts/labeler/ae_dataset.sbatch:35:    --root /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/ae \
[D] scripts/labeler/ae_dataset.sbatch:36:    --out-dir "$REPO/outputs/labelmaker/ae/dataset"
[D] scripts/labeler/elm_write_normalization.py:43:OUT_DIR = Path("/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/models/d3d_elm_time_to_event_dsm")
[D] scripts/labeler/retrain_tearing_dsm.py:96:    "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/models/"
[D] scripts/labeler/ae_dataset.py:1072:    p.add_argument("--root", default="/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/ae")
[D] scripts/labeler/ae_dataset.py:1075:        default="/scratch/gpfs/nc1514/FusionAIHub/outputs/labelmaker/ae/dataset",
[D] scripts/labeler/ae_train.sbatch:11:#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%A_%a.out
[D] scripts/labeler/ae_train.sbatch:12:#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%A_%a.out
[D] scripts/labeler/ae_train.sbatch:37:ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker
[D] scripts/labeler/retrain_tearing_dsm.sbatch:9:#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out
[D] scripts/labeler/retrain_tearing_dsm.sbatch:24:ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker
[D,E] scripts/labeler/make_phase3_env.sh:14:ROOT=$("/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/python" "$REPO/src/labeler/env.py" LABELER_ROOT "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")
[E] scripts/labeler/write_training_shots.py:16:    pixi run -e labelmaker python scripts/labeler/write_training_shots.py
[D,E] scripts/labeler/tokeye_masks.sbatch:39:ROOT="${ROOT:-$("/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/python" "$REPO/src/labeler/env.py" LABELER_ROOT "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")}"
[D] scripts/labeler/tokeye_masks.sbatch:40:RUNTIME_ROOT="${RUNTIME_ROOT:-/scratch/gpfs/EKOLEMEN/nc1514/labelmaker}"
[E] scripts/labeler/tokeye_masks.sbatch:45:CORPUS="$("/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/python" "$REPO/src/labeler/env.py" LABELER_CORPUS "/scratch/gpfs/EKOLEMEN/foundation_model")"
[D] scripts/labeler/label_shot.sh:10:# Output, per shot, under $LABELER_DEMO_OUT (default outputs/labelmaker/analysis in
[E] scripts/labeler/label_shot.sh:20:# corpus needs an fdp token (`pixi run -e labelmaker fdp login`, once).
[D,E] scripts/labeler/label_shot.sh:24:ROOT="$("/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/python" "$REPO/src/labeler/env.py" LABELER_ROOT "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")"
[D,E] scripts/labeler/label_shot.sh:25:OUT="$("/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/python" "$REPO/src/labeler/env.py" LABELER_DEMO_OUT "$REPO/outputs/labelmaker/analysis")"
[E] scripts/labeler/label_shot.sh:26:CONFIG="$("/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/python" "$REPO/src/labeler/env.py" LABELER_DEMO_CONFIG "$REPO/src/labeler/analyze_default.yaml")"
[E] scripts/labeler/label_shot.sh:39:FORCE="$("/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/python" "$REPO/src/labeler/env.py" LABELER_DEMO_FORCE)"
[E] scripts/labeler/label_shot.sh:41:pixi run -q -e labelmaker fdp run python -m labeler.run analyze \
[E] scripts/labeler/label_shot.sh:45:pixi run -q -e labelmaker python - "$ROOT" "$OUT" "$@" <<'PY'
[E] scripts/labeler/fdp_probe.py:4:Run on a login node under ``pixi run -e labelmaker fdp run python ...``.
[E] scripts/labeler/jobstats_check.py:7:        --wrap "pixi run -e labelmaker python scripts/labeler/jobstats_check.py \
[S] scripts/labeler/labels_format.py:96:        "made_by": f"scripts/labelmaker/labels_format.py:{spec.converter}",
[D] scripts/labeler/tokeye_masks_afterok.sbatch:19:#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out
[D,E] scripts/labeler/tokeye_masks_afterok.sbatch:31:ROOT="$("/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/python" "$REPO/src/labeler/env.py" LABELER_ROOT "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")"
[E] scripts/labeler/tokeye_masks_afterok.sbatch:44:    -e labelmaker python "$REPO/scripts/labeler/tokeye_masks.py" \
[E] scripts/labeler/tokeye_masks_afterok.sbatch:47:    -e labelmaker python "$REPO/scripts/labeler/jobstats_check.py" \
[D,E] scripts/labeler/tokeye_text_subset.sh:7:ROOT="$("/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/python" "$REPO/src/labeler/env.py" LABELER_ROOT "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")"
[E] scripts/labeler/tokeye_text_subset.sh:24:    -e labelmaker python "$REPO/scripts/labeler/tokeye_masks.py" \
[D] scripts/labeler/labels_recommender_v1.sbatch:10:#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out
[D] scripts/labeler/labels_recommender_v1.sbatch:11:#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out
[E] scripts/labeler/labels_recommender_v1.sbatch:46:# `pixi run -e labelmaker fdp run python -m labeler.run all --models <the four> ...`, ten shots
[D,E] scripts/labeler/labels_recommender_v1.sbatch:83:ROOT="$("/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/python" "$REPO/src/labeler/env.py" LABELER_ROOT "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")"
[E] scripts/labeler/labels_recommender_v1.sbatch:84:ENV="$REPO/.pixi/envs/labelmaker"
[D,S] scripts/labeler/ae_evaluate.py:214:        "--fig-dir", type=Path, default=REPO / "outputs" / "labelmaker" / "ae" / "training"
[D] scripts/labeler/elm_dsm_train.sbatch:10:#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%A_%a.out
[D] scripts/labeler/elm_dsm_train.sbatch:23:ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker
[D] scripts/labeler/ae_evaluate.sbatch:10:#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out
[D] scripts/labeler/ae_evaluate.sbatch:11:#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/%j.out
[D] scripts/labeler/ae_evaluate.sbatch:19:ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker
[D] scripts/labeler/ae_evaluate.sbatch:31:  --fig-dir "$REPO/outputs/labelmaker/ae/training"
[S] scripts/labeler/fetch_features.py:7:names in its `labelmaker: {feature: ...}` blocks -- `bt betan kappa tritop tribot li qmin gapin
[E] scripts/labeler/fetch_features.py:20:    pixi run -e labelmaker fdp run python scripts/labeler/fetch_features.py \
[D] scripts/labeler/phase3-requirements.txt:6:# outputs/labelmaker/*/scripts/, which never import `labeler` itself.
[E] scripts/labeler/sawtooth_reference_check.py:34:        -e labelmaker python scripts/labeler/sawtooth_reference_check.py \\
[E] scripts/labeler/sawtooth_reference_check.py:58:        -e labelmaker python scripts/labeler/sawtooth_reference_check.py \\
[E] scripts/labeler/pin_unet.py:26:        -e labelmaker python scripts/labeler/pin_unet.py
[E] scripts/labeler/qmin_regime_census.py:21:        -e labelmaker python scripts/labeler/qmin_regime_census.py \\
[D] scripts/labeler/qmin_regime_census.py:22:        --shot-file /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/recommender_v1.txt \\
[D] scripts/labeler/qmin_regime_census.py:55:    "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/recommender_v1.txt"
[D] scripts/labeler/ae_train.py:73:DEFAULT_DATASET = Path("/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/ae/dataset")
[D] scripts/labeler/ae_train.py:75:    "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/models/d3d_ae_activity_seldnet"
[D] scripts/shot_design/encode_cpu.sbatch:11:#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/ideate/runs/slurm/%A_%a.out
[D] scripts/shot_design/encode_cpu.sbatch:12:#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/ideate/runs/slurm/%A_%a.out
[D,E] scripts/shot_design/encode_cpu.sbatch:116:ROOT="$("/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/python" "$REPO/src/shot_design/env.py" SHOT_DESIGN_DATA_ROOT "/scratch/gpfs/EKOLEMEN/nc1514/ideate")"
[E] scripts/shot_design/encode_cpu.sbatch:117:ENV="$("/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/python" "$REPO/src/shot_design/env.py" SHOT_DESIGN_ENV "/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/ideate")"
[D] scripts/shot_design/build.sbatch:10:#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/ideate/runs/slurm/%j.out
[D] scripts/shot_design/build.sbatch:11:#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/ideate/runs/slurm/%j.out
[E] scripts/shot_design/build.sbatch:29:# is the conversion the --mem arithmetic below uses), `.pixi/envs/ideate-cpu/bin/python` called
[E] scripts/shot_design/build.sbatch:93:SHOT_DESIGN_PY="$("/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/python" "$REPO/src/shot_design/env.py" SHOT_DESIGN_PY "/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/ideate-cpu/bin/python")"
[D,E] scripts/shot_design/build.sbatch:94:SHOT_DESIGN_DATA_ROOT="$("/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/python" "$REPO/src/shot_design/env.py" SHOT_DESIGN_DATA_ROOT "/scratch/gpfs/EKOLEMEN/nc1514/ideate")"
[D,E] scripts/shot_design/build.sbatch:105:export LABELER_ROOT="$("/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/python" "$REPO/src/labeler/env.py" LABELER_ROOT "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")"
[D] scripts/shot_design/encode.sbatch:12:#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/ideate/runs/slurm/%A_%a.out
[D] scripts/shot_design/encode.sbatch:13:#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/ideate/runs/slurm/%A_%a.out
[D,E] scripts/shot_design/encode.sbatch:90:ROOT="$("/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/python" "$REPO/src/shot_design/env.py" SHOT_DESIGN_DATA_ROOT "/scratch/gpfs/EKOLEMEN/nc1514/ideate")"
[E] scripts/shot_design/encode.sbatch:91:ENV="$REPO/.pixi/envs/ideate"
[D] scripts/shot_design/census.sbatch:10:#SBATCH --output=/scratch/gpfs/EKOLEMEN/nc1514/ideate/runs/slurm/%j.out
[D] scripts/shot_design/census.sbatch:11:#SBATCH --error=/scratch/gpfs/EKOLEMEN/nc1514/ideate/runs/slurm/%j.out
[E] scripts/shot_design/census.sbatch:24:#     /usr/bin/time -v pixi run -e ideate-cpu python -m shot_design corpus scan \
[D,E] scripts/shot_design/census.sbatch:73:SHOT_DESIGN_DATA_ROOT="$("/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/python" "$REPO/src/shot_design/env.py" SHOT_DESIGN_DATA_ROOT "/scratch/gpfs/EKOLEMEN/nc1514/ideate")"
[E] scripts/shot_design/census.sbatch:78:export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false SHOT_DESIGN_CORPUS="$("/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/python" "$REPO/src/shot_design/env.py" SHOT_DESIGN_CORPUS "/scratch/gpfs/EKOLEMEN/foundation_model")"
[D,E] scripts/shot_design/census.sbatch:79:export LABELER_ROOT="$("/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/labelmaker/bin/python" "$REPO/src/labeler/env.py" LABELER_ROOT "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")"
[E] scripts/shot_design/census.sbatch:80:srun "$REPO/.pixi/envs/ideate-cpu/bin/python" -m shot_design corpus scan \
[D] configs/shot_design/paths.yaml:3:data_root: /scratch/gpfs/EKOLEMEN/nc1514/ideate
[S] configs/shot_design/signals.yaml:14:  ip:        {group: ip, col: ipsip, units: A, tier: raw, abs: true, scale: 1.0e+6, stats: [mean, peak, min, slope], fetch: {kind: ptdata, node: ipsip}, labelmaker: {feature: ip}}
[S] configs/shot_design/signals.yaml:29:  bt:        {group: mag_b0, col: bt, units: T, tier: raw, abs: true, stats: [mean], fetch: {kind: ptdata, node: bt}, labelmaker: {feature: bt}}
[S] configs/shot_design/signals.yaml:61:  betan:     {group: beta, col: betan, tier: derived, stats: [mean, peak], provenance: *efit, fetch: {kind: mds, tree: efit01, node: "\\EFIT01::TOP.RESULTS.AEQDSK:BETAN"}, labelmaker: {feature: betan}}
[S] configs/shot_design/signals.yaml:64:  qmin:      {group: q_psi, col: qmin, tier: derived, stats: [mean, min], provenance: *efit, fetch: {kind: mds, tree: efit01, node: "\\EFIT01::TOP.RESULTS.AEQDSK:QMIN"}, labelmaker: {feature: qmin}}
[S] configs/shot_design/signals.yaml:66:  kappa:     {group: mag_geo_para, col: kappa, tier: derived, stats: [mean], provenance: *efit, fetch: {kind: mds, tree: efit01, node: "\\EFIT01::TOP.RESULTS.AEQDSK:KAPPA"}, labelmaker: {feature: kappa}}
[S] configs/shot_design/signals.yaml:67:  tritop:    {group: mag_geo_para, col: tritop, tier: derived, stats: [mean], provenance: *efit, fetch: {kind: mds, tree: efit01, node: "\\EFIT01::TOP.RESULTS.AEQDSK:TRITOP"}, labelmaker: {feature: tritop}}
[S] configs/shot_design/signals.yaml:68:  tribot:    {group: mag_geo_para, col: tribot, tier: derived, stats: [mean], provenance: *efit, fetch: {kind: mds, tree: efit01, node: "\\EFIT01::TOP.RESULTS.AEQDSK:TRIBOT"}, labelmaker: {feature: tribot}}
[S] configs/shot_design/signals.yaml:69:  aminor:    {group: mag_geo_para, col: aminor, units: m, tier: derived, stats: [mean], provenance: *efit, fetch: {kind: mds, tree: efit01, node: "\\EFIT01::TOP.RESULTS.AEQDSK:AMINOR"}, labelmaker: {feature: aminor}}
[S] configs/shot_design/signals.yaml:70:  r0:        {group: mag_geo_para, col: r0, units: m, tier: derived, stats: [mean], provenance: *efit, fetch: {kind: mds, tree: efit01, node: "\\EFIT01::TOP.RESULTS.AEQDSK:R0"}, labelmaker: {feature: r0}}
[S] configs/shot_design/signals.yaml:71:  volume:    {group: mag_geo_para, col: volume, units: m^3, tier: derived, stats: [mean], provenance: *efit, fetch: {kind: mds, tree: efit01, node: "\\EFIT01::TOP.RESULTS.AEQDSK:VOLUME"}, labelmaker: {feature: volume}}
[S] configs/shot_design/signals.yaml:72:  li:        {group: other_profiles, col: li, tier: derived, stats: [mean], provenance: *efit, fetch: {kind: mds, tree: efit01, node: "\\EFIT01::TOP.RESULTS.AEQDSK:LI"}, labelmaker: {feature: li}}
[S] configs/shot_design/signals.yaml:83:  gapin:     {group: divertor_geo, col: gapin, units: m, tier: derived, stats: [mean], provenance: *efit, fetch: {kind: mds, tree: efit01, node: "\\EFIT01::TOP.RESULTS.AEQDSK:GAPIN"}, labelmaker: {feature: gapin}}
[S] configs/shot_design/signals.yaml:90:  ne0:       {group: e_dens_fit, col: "edensfit0.00", units: "[?]", tier: derived, stats: [mean], provenance: {tool: profile_fit, version: "[?]"}, fetch: null, labelmaker: {feature: ne_zipfit, reduce: core, units: "1e19 m^-3"}}
[S] configs/shot_design/signals.yaml:91:  te0:       {group: e_temp_fit, col: "etempfit0.00", units: "[?] keV", tier: derived, stats: [mean], provenance: {tool: profile_fit, version: "[?]"}, fetch: null, labelmaker: {feature: te_zipfit, reduce: core, units: "keV"}}
[S] configs/shot_design/signals.yaml:105:  pcbcoil:          {units: A, tier: raw, stats: [mean, peak], fetch: null, labelmaker: {feature: pcbcoil}}
[V] docs/LABELER.md:3:Environment settings use `LABELER_*`. The corresponding `LABELMAKER_*` name
[E] docs/LABELER.md:6:`labelmaker` under the environment-rename fallback; Python modules use `labeler`.
[D] docs/LABELER.md:7:Data remains under `/scratch/gpfs/EKOLEMEN/nc1514/labelmaker`, including
[S] docs/LABELER.md:9:and the model-card `labelmaker:` block keep their names.
[H] docs/LABELER.md:15:Design: `docs/superpowers/specs/2026-09-03-labelmaker-design.md`.
[H] docs/LABELER.md:16:Phase 1 build log: `docs/superpowers/plans/2026-09-03-labelmaker-phase1.md`.
[E] docs/LABELER.md:22:pixi install -e labelmaker              # once
[E] docs/LABELER.md:23:pixi run -e labelmaker fdp login        # once per token; only the fdp resolver needs it
[E] docs/LABELER.md:25:pixi run -e labelmaker label 199597     # the one-shot demo: every servable model on one shot
[D] docs/LABELER.md:29:leaves four files in `outputs/labelmaker/analysis/<shot>/` in this repo (override the
[E] docs/LABELER.md:41:pixi run -e labelmaker fdp run python -m labeler.run all \
[E] docs/LABELER.md:101:  --frozen --no-install -e labelmaker fdp run python -m labeler.run features \
[E] docs/LABELER.md:123:pixi run -e labelmaker fdp run python -m labeler.run analyze \
[D] docs/LABELER.md:163:`/scratch/gpfs/EKOLEMEN/nc1514/labelmaker`; `$LABELER_CORPUS` overrides the
[E] docs/LABELER.md:198:pixi run -e labelmaker python -m labeler.run events \
[E] docs/LABELER.md:446:pixi run -e labelmaker python -m labeler.run events --databases-only \
[D] docs/LABELER.md:518:`/scratch/gpfs/EKOLEMEN/nc1514/labelmaker`, while `PHASE3_PYTHON` and `UNET` can
[D] docs/LABELER.md:552:export ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/l14perf/pilot-example
[D] docs/LABELER.md:554:export SHOT_FILE=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/runs/slurm/l12/pilot20.txt
[E] docs/LABELER.md:574:pixi run --frozen --no-install --manifest-path /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker python -m labeler.jobstats \
[E] docs/LABELER.md:693:pixi run -e labelmaker python -m labeler.run events --rules-only \
[E] docs/LABELER.md:715:pixi run -e labelmaker python -m labeler.run events --rules-only \
[S] docs/LABELER.md:782:   `README.md` (HuggingFace card plus the `labelmaker:` block), `spec.py`
[E] docs/LABELER.md:792:5. `pixi run -e labelmaker python -m pytest tests/labeler -q -W error` - the
[E] docs/LABELER.md:794:   `pixi run -e labelmaker ruff check src/labeler tests/labeler`.
[H] docs/LABELER.md:844:  blocks each of them; `docs/superpowers/specs/2026-09-05-labelmaker-phase2-design.md`
[H] docs/LABELER.md:865:- `docs/superpowers/specs/2026-09-05-labelmaker-phase3-design.md` carries the
[V] docs/SHOT_DESIGN.md:3:Environment settings use the `SHOT_DESIGN_*` prefix. Legacy `IDEATE_*` settings
[E] docs/SHOT_DESIGN.md:6:The pixi environments remain `ideate` and `ideate-cpu` because pixi rejects
[D] docs/SHOT_DESIGN.md:8:The production data directory remains `/scratch/gpfs/EKOLEMEN/nc1514/ideate`.
[E] docs/SHOT_DESIGN.md:23:pixi run -e ideate-cpu shot_design <command>          # or: python -m shot_design <command>
[E] docs/SHOT_DESIGN.md:50:`pixi run -e ideate` and `-e ideate-cpu` set `SHOT_DESIGN_DATA_ROOT`, `LABELER_ROOT` and
[E] docs/SHOT_DESIGN.md:61:/scratch/gpfs/nc1514/FusionAIHub/.pixi/envs/ideate-cpu/bin/python -m shot_design build --shots 190000
[E] docs/SHOT_DESIGN.md:68:pixi run -e ideate-cpu env -u SHOT_DESIGN_DATA_ROOT SHOT_DESIGN_PATHS=/tmp/my-paths.yaml \
[E] docs/SHOT_DESIGN.md:402:pixi run -e ideate-cpu shot_design-mcp        # == python -m shot_design.mcp
[D] docs/SHOT_DESIGN.md:525:export HF_HUB_OFFLINE=1 SHOT_DESIGN_DATA_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/ideate \
[D] docs/SHOT_DESIGN.md:526:       LABELER_ROOT=/scratch/gpfs/EKOLEMEN/nc1514/labelmaker \
[E] docs/SHOT_DESIGN.md:529:  -e ideate-cpu python -m shot_design serve --port 8765
[E] pyproject.toml:102:# via pyxrootd) that need the symbol; both the `fdp` and `labelmaker`
[E,S] pyproject.toml:120:[tool.pixi.feature.labelmaker]
[E,S] pyproject.toml:123:[tool.pixi.feature.labelmaker.tasks]
[E] pyproject.toml:124:# `pixi run -e labelmaker label 187199`: every implemented model on one shot ->
[E,S] pyproject.toml:128:[tool.pixi.feature.labelmaker.dependencies]
[E] pyproject.toml:138:# Solving `labelmaker` (labelmaker + fdp + the implicit default feature)
[E,S] pyproject.toml:164:[tool.pixi.feature.labelmaker.pypi-dependencies]
[E] pyproject.toml:173:[tool.pixi.feature.ideate]
[E] pyproject.toml:176:[tool.pixi.feature.ideate.tasks]
[E] pyproject.toml:177:# `pixi run -e ideate shot_design search --text "..."`: the shot analysis and
[E] pyproject.toml:184:[tool.pixi.feature.ideate.dependencies]
[E] pyproject.toml:199:# see the [feature.ideate.pypi-dependencies] block below for why.
[E,S] pyproject.toml:201:# shots.parquet and the per-segment tables. Same range as feature.labelmaker
[E,S] pyproject.toml:206:# directly for the same reason feature.labelmaker declares it: `yaml`
[E,S] pyproject.toml:238:# Same channel-drift pin, and the same reason, as feature.labelmaker: a live
[E] pyproject.toml:247:[tool.pixi.feature.ideate.pypi-dependencies]
[E] pyproject.toml:255:# it to [feature.ideate.dependencies] makes the conda solver choose torch -
[E] pyproject.toml:259:#   x failed to solve the pypi requirements of environment 'ideate'
[E] pyproject.toml:268:# feature.ideate-cpu already decide which wheel each environment gets.
[E] pyproject.toml:284:[tool.pixi.feature.ideate.target.unix.activation.env]
[D] pyproject.toml:285:SHOT_DESIGN_DATA_ROOT = "/scratch/gpfs/EKOLEMEN/nc1514/ideate"
[D] pyproject.toml:286:LABELER_ROOT = "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker"
[E] pyproject.toml:291:[tool.pixi.feature.ideate-cpu]
[E] pyproject.toml:294:[tool.pixi.feature.ideate-cpu.pypi-dependencies]
[E,S] pyproject.toml:295:# A CPU-only twin of the `ideate` environment, exactly as feature.labelmaker
[E] pyproject.toml:299:# touches. Only `design_rollout` needs a GPU, and that is what the `ideate`
[E] pyproject.toml:301:# cpu index on `ideate` itself) so the two environments can differ in the
[E] pyproject.toml:344:labelmaker = ["labelmaker", "fdp"]
[E] pyproject.toml:345:ideate = ["ideate", "cuda"]
[E] pyproject.toml:346:ideate-cpu = ["ideate", "ideate-cpu"]
[E] .mcp.json:5:      "args": ["run", "-e", "ideate-cpu", "python", "-m", "shot_design.mcp"],
```

The required expression intentionally does not match underscore-suffixed stored
keys (`labelmaker_version`, `labelmaker_git_sha`, `labelmaker_sha`,
`labelmaker_root`) or the cookie name `ideate_token`; they are also preserved.
The case-sensitive grep does not match `X-Ideate-Caveats`, which remains the shared
HTTP metadata key in both server and browser. Output-tree README/JSON/binary
records are outside this grep's scope and remain unchanged by hash verification.

## Final state

The worktree `.pixi` directory and generated package bytecode are removed. Only
the intended source, test, configuration, documentation, and report changes are
committed. `git status --porcelain` is empty after the report commit.
