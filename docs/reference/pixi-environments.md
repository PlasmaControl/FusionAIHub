---
title: Pixi environments
sidebar_position: 4
---

# Pixi environments

`pyproject.toml` defines seven environments. Each block there has a one-line
comment. This page keeps the measurements behind those lines.

| Environment | Features | For |
|---|---|---|
| `default` | `cuda` | the model code, cu124 torch |
| `fdp` | `fdp`, `cuda` | direct DIII-D data access (fdp, toksearch) |
| `frontier` | `frontier` | Frontier training, ROCm 7.1 torch |
| `labelmaker` | `labelmaker`, `fdp` | `labeler`, CPU torch |
| `shot-design` | `shot-design`, `cuda` | `shot_design` with a GPU (IGNITE rollouts) |
| `shot-design-cpu` | `shot-design`, `shot-design-cpu` | `shot_design` on a CPU |
| `shot-design-frontier` | `shot-design-frontier`, `shot-design` | `shot_design` and `labeler` on Frontier |

| Task | Environment | Runs |
|---|---|---|
| `label` | `labelmaker` | `scripts/labeler/label_shot.sh`: every labeler model on the given shots |
| `labeler-verify` | `labelmaker` | the [label review page](../labeler/review.md), under `fdp run` |
| `labeler-spectrograms` | `labelmaker` | `python -m labeler.events.review.build`, the review page's row store |
| `labeler-test` | `labelmaker` | `python -m pytest tests/labeler -q -W error` |
| `shot_design` | the `shot-design` three | `python -m shot_design` |
| `shot_design-mcp` | the `shot-design` three | the MCP server, stdio |
| `shot_design-test` | the `shot-design` three | `python -m pytest tests/shot_design -q -W error` |
| `setup-flash-attn`, `verify-flash-attn` | `frontier` | build flash-attn 2 on a compute node, then smoke-test it |

A task runs in the checkout that holds the manifest. To test a worktree, run the
suite in the worktree itself:
`PYTHONPATH=$PWD/src pixi run --frozen --no-install --manifest-path <main checkout>/pyproject.toml -e labelmaker python -m pytest tests/labeler -q -W error`.

## Always `--frozen`

Measured 2026-09-22 with pixi 0.76.1: `pixi run --locked` refuses every
environment, "lock file not up-to-date with the workspace". The reason is
`torchvision`'s index. `[project]` lists it bare, so pypi.org, while the lock
holds it from the pytorch index each feature pins. Without `--frozen`, pixi
re-solves before it runs anything. Re-locking would clear this, but it changes
every environment, so it waits for its own change.

## Roots are defaults

The activation blocks set the roots as `${VAR:-value}`: `SHOT_DESIGN_DATA_ROOT`,
`LABELER_ROOT` and `SHOT_DESIGN_CORPUS`, plus `SHOT_DESIGN_PATHS` on Frontier.
A root you export before `pixi run` is the one the command sees. Unset, the
cluster's production root applies. Measured 2026-09-22 with pixi 0.76.1: `pixi run`
and `pixi shell-hook` both leave the default to the shell (`shell-hook` prints
`export SHOT_DESIGN_DATA_ROOT="${SHOT_DESIGN_DATA_ROOT:-...}"`). So

```bash
SHOT_DESIGN_DATA_ROOT=/tmp/x pixi run --frozen -e shot-design-cpu printenv SHOT_DESIGN_DATA_ROOT
```

prints `/tmp/x`, and with the variable unset it prints the production root.

Until 2026-09-22 the blocks set plain values. Activation runs after your shell,
so a value you exported was replaced without a word. On 2026-09-14 a one-shot
scratch build, run through `pixi run` with `SHOT_DESIGN_DATA_ROOT` exported to a
`/tmp` directory, published itself over the 500-shot production database. Every
`shot_design` command that writes still prints the root it resolved before its
first write ([Shot design](../shot-design/overview.md#scratch-databases-and-the-data-root)).

The roots are absolute paths rather than repo-relative ones: the corpus and the
label store live outside this checkout, on /scratch/gpfs/EKOLEMEN, and the same
environment is used from a login node, a SLURM job and an MCP client launched by
an assistant, none of which share a working directory.

`HF_HUB_OFFLINE`: sentence-transformers reaches into huggingface_hub on every
model load even when the MiniLM checkpoint is already cached, and on a compute
node with no outbound route that call does not fail - it HANGS, for minutes,
inside httpx.connect_tcp (measured in shotrec, see the same note at the top of
its cli.py). The checkpoint is in ~/.cache/huggingface/hub, so offline is the
correct and much faster mode.

`TOKENIZERS_PARALLELISM`: the corpus build forks worker processes after a
tokenizer has been used; without this, every fork prints the HuggingFace
deadlock warning and the fast tokenizer disables its own threads anyway.

## Activation order on Frontier

feature.shot-design's own activation block sets the three roots to their STELLAR
values, so one of the two blocks has to win. MEASURED 2026-09-19 with pixi 0.73:
the FIRST feature listed for the environment wins, not the last -- with
`shot-design-frontier = ["shot-design", "shot-design-frontier"]` the environment
came up with `/scratch/gpfs/EKOLEMEN/nc1514/ideate`, the Stellar root. That is why
[tool.pixi.environments] lists `shot-design-frontier` FIRST; reordering it
silently sends every job at the Stellar paths, which do not exist there.
Measured again 2026-09-22 with pixi 0.76.1 on a two-feature manifest: still the
first, and with defaults the first feature's default is the one used.

SHOT_DESIGN_PATHS selects the Frontier paths file
(configs/shot_design/paths.frontier.yaml); feature.shot-design does not set it,
because on Stellar the default paths.yaml is already the right file.

HDF5_USE_FILE_LOCKING: the corpus lives on Lustre, where h5py's default locking
fails to open a read-only file the way it does on GPFS.

## feature.fdp

MEASURED 2026-09-04 (Task 16b): `import torch` binds the SYSTEM
`/lib64/libstdc++.so.6`, which lacks `GLIBCXX_3.4.29`. Every compiled extension
in this feature that needs that symbol then fails to import after torch does -
`toksearch_d3d` (via `fdp` -> `pyxrootd`) and `scipy` - with a bare
`ImportError`, which `resolve_fdp.available()` swallowed into an
indistinguishable `ToksearchUnavailable` and silently disabled the whole fdp
scaling path. The pixi env ships its OWN `lib/libstdc++.so.6.0.36`, which DOES
have the symbol; it is just not first on the loader's path. `LD_LIBRARY_PATH`
(rather than `LD_PRELOAD`) is sufficient: verified that with it set, `torch`
followed by `fdp`, `toksearch`, `toksearch_d3d` and `scipy` all import, which
means torch's extensions carry `DT_RUNPATH` (which `LD_LIBRARY_PATH` overrides),
not `DT_RPATH` (which it would not). Scoped to this feature - not the shared
workspace - because it is `fdp`'s own packages (toksearch/toksearch_d3d, via
pyxrootd) that need the symbol; both the `fdp` and `labelmaker` environments
include this feature, so both are covered. Prepended, never overwritten: an
inherited `LD_LIBRARY_PATH` (e.g. from a module-loaded library on a SLURM node)
must still take effect first for anything it already covers.

Versions match Nathan's working fetch env (/scratch/gpfs/nc1514/fdp/pixi.toml)
rather than floating. `fdp` itself is REQUIRED, not optional: `toksearch_d3d`
imports it at module scope, so leaving it out resolved an env where
`import toksearch_d3d` raised ModuleNotFoundError while `toksearch` imported fine
- the fdp path was unreachable and nothing said so.

## feature.labelmaker

`labeler-verify` runs the page under `fdp run`, which sets `PTDATA_LIBRARY` and
`default_tree_path` and nothing interactive (measured 2026-09-22 on
stellar-vis2). A shot outside the corpus and the fetch cache can then be fetched
live. Arguments after the task name reach the page: `pixi run -e labelmaker
labeler-verify --port 8812`.

`pyarrow`: labels_index.parquet. pandas 3 can write parquet only through
pyarrow, and the default env does not have it.

`pyyaml`: registry.py parses every model card's YAML front matter. `yaml`
currently reaches this environment only transitively, via hydra-core ->
omegaconf, and hydra-core is declared in [project] for IGNITE's benefit, not
labeler's - anyone tidying IGNITE's dependencies would silently break every card
read in this package. Declare it directly.

`protobuf`: solving `labelmaker` (labelmaker + fdp + the implicit default
feature) against live conda-forge picks up protobuf 7.35.1, published to
conda-forge sometime after pixi.lock's other environments were last solved
(2026-07-31, per `git log -- pixi.lock`). The shared [project.dependencies] pin
`wandb>=0.25.1,<0.26` needs protobuf>4.21.0,!=5.28.0,!=5.29.0,<7, so the two are
unsatisfiable together:

```
Because wandb==0.25.1 depends on one of:
    protobuf>4.21.0,<5.28.0 / >5.28.0,<5.29.0 / >5.29.0,<7
and protobuf==7.35.1, we can conclude that wandb==0.25.1 cannot be used.
help: protobuf==7.35.1 has been pinned by the conda solve.
```

This is channel drift, not a toksearch/torch conflict (the brief's anticipated
failure mode and documented CPU-wheel-drop fallback, tried first, did not change
this error). It reproduced identically with and without the two
pypi-dependencies lines below and with pyarrow removed entirely, isolating the
cause to protobuf's rolling conda-forge version rather than to either of those.
The already-locked `fdp`/`default` environments are unaffected since they stay
pinned to their 2026-07-31 solve (libprotobuf/protobuf 6.31.1) until someone
reruns `pixi update`. Pinning protobuf here, scoped to this feature only, keeps
the new environment on the same already-known-good major version, the same way
`feature.cuda` pins torch's index rather than editing [project].

`fastapi`, `uvicorn`: the review page is a FastAPI app on the loopback, reached
over an SSH forward. Same ranges as feature.shot-design, which runs shot_design's
UI, so an environment holding both cannot want two different majors.

CPU torch: the workspace installs `faith` editable into every environment, which
drags torch in. Labelmaker never uses a GPU (its heaviest model is 12k
parameters evaluated in torch on CPU, see models/runners/keras_h5.py), so take
the CPU wheels and save ~3 GB in an environment that would otherwise duplicate
cu124.

`plotly`, `anywidget`: the verification notebooks draw with plotly
(`labeler.events.verify`), and `tests/labeler/test_events_verify.py` imports it
when the suite is collected. The review page draws its own canvases and needs
neither. `FigureWidget` imports `anywidget` at construction time, so a bare
`import plotly` succeeding says nothing about whether the widget works. That
widget is the notebooks' surface only, so this pin lives and dies with them.

## feature.shot-design

shot_design is ported from the shot-recommender-system repo
(/scratch/gpfs/nc1514/shot-recommender-system), which pins its runtime through a
uv.lock. Every range here was checked on 2026-09-07 against the versions that
lock actually resolves - pydantic 2.13.5, fastapi 0.141.1, sentence-transformers
6.0.1, httpx 0.28.1, pypdf 6.16.2, scikit-learn 1.9.0 - so none of them had to be
widened; a range that excluded the running version would produce a port that
passes here and fails there. `mcp` and `duckdb` have no counterpart in that lock
(the MCP server and the DuckDB-backed store are new in this port).

Everything is a CONDA dependency: conda-forge carries current builds of the
whole set (`mcp` 2.1.1 was published there on 2026-08-26), and the conda solver
is the one that also chooses this environment's python. The single exception is
`sentence-transformers`, which has to come from pypi (below).

- `pyarrow`: shots.parquet and the per-segment tables. Same range as
  feature.labelmaker so the two packages cannot end up wanting different pyarrow
  majors in an environment that includes both; shotrec only asks for
  `pyarrow>=16`.
- `pyyaml`: phenomena.yaml, actuators.yaml and the model cards shot_design reads.
  Declared directly for the same reason feature.labelmaker declares it.
- `pydantic`: the schema layer (PhenomenonHit, DesignReport, CircumstanceReport,
  ...). >=2.12 is also conda-forge `mcp`'s own floor, so this is not a free
  choice.
- `scipy`: signal processing behind the segment finder and the circumstances
  stats. Unpinned to match [project], which also lists a bare `scipy`.
- `h5py`: corpus reads, the DIII-D shot HDF5 files under $SHOT_DESIGN_CORPUS.
- `pypdf`: miniproposal PDFs -> the text that phenomenon search quotes from.
- `beautifulsoup4`: summary.html -> the per-shot text bundles. `shot_design logs
  import` re-runs d3dlogfetching's own composition (shotdb/logs.py) over a synced
  runs/ tree, and that composition parses the session summary with
  BeautifulSoup; the output has to be byte-identical to the 22,950 bundles
  already in the corpus, so the parser is not a free choice. 4.x is what the
  tool pins.
- `fastapi`, `uvicorn`: `shot_design serve`, the browser UI. Kept in the base
  feature rather than a separate one because the MCP server and the UI share the
  same query layer.
- `mcp`: `from mcp.server import MCPServer`. 2.x only: FastMCP, the 1.x entry
  point, is gone, so a solve that fell back to mcp 1.x would import-error rather
  than merely behave differently.
- `protobuf`: the same channel-drift pin, and the same reason, as
  feature.labelmaker.
- `duckdb`: the shot/segment/label store shot_design queries. conda-forge ships
  the Python bindings as `duckdb`.

`sentence-transformers` (MiniLM text embeddings,
sentence-transformers/all-MiniLM-L6-v2, 384-d). The 6.x major is what shotrec
runs; 5.x renamed enough of the encode path that a lower floor would not be the
code that was ported. It is the ONE package of the set taken from pypi rather
than conda-forge. conda-forge does publish it (6.0.1), but its recipe depends on
`pytorch >=1.11.0`, so adding it to the conda dependencies makes the conda
solver choose torch - it picks the newest, 2.12.0 - and pixi then hands that as
a PINNED version to the pypi solve, where it collides head-on with feature.cuda:

```
x failed to solve the pypi requirements of environment 'shot-design'
`-> Because you require torch>=2.5.1,<2.11 and torch==2.12.0, we can
    conclude that your requirements are unsatisfiable.
help: torch==2.12.0 has been pinned by the conda solve.
```

The cu124 range in feature.cuda is not ours to widen (cu124 is what ships sm_70
kernels for the V100S cards these environments run on), and neither environment
wants a second copy of torch. Taking sentence-transformers from pypi keeps torch
entirely on the pypi side, where feature.cuda and feature.shot-design-cpu
already decide which wheel each environment gets.

## feature.shot-design-cpu

A CPU-only twin of the `shot-design` environment, exactly as feature.labelmaker
does it. The workspace installs `faith` editable everywhere, which drags torch
in, and the cu124 wheels are ~3 GB that a laptop-shaped run of shot_design -
phenomenon search, the events tables, the MiniLM encoder - never touches. Only
`design_rollout` needs a GPU, and that is what the `shot-design` environment is
for. Split into its own feature (rather than putting the cpu index on
`shot-design` itself) so the two environments can differ in the one line that
separates them.

## feature.frontier

pip is needed for the `setup-flash-attn` task to install flash-attn from a git
URL with --no-build-isolation. The PyTorch wheels pulled from the rocm7.1 index
don't drag pip in transitively. ninja: aiter (a transitive dep of flash_attn on
ROCm) JIT-compiles a small C++ extension at first `import flash_attn`. It calls
`ninja` from PATH.

The rocm7.1 index ships torch 2.10.0 + torchvision 0.25-0.26 only. torch 2.10
declares triton-rocm as a dep; uv won't auto-discover it through the per-package
`index = ...`, so it is listed explicitly.

Flash-Attention 2 (gfx90a / MI250X) is NOT listed intentionally: the build needs
`module load rocm/7.1.1` + `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`, which
pixi/uv can't set. Install it with the `setup-flash-attn` task; it uses the AMD
Triton backend (not Composable Kernel) per the AMD docs at
rocm.docs.amd.com/.../model-acceleration-libraries.html - Triton skips the
multi-hour CK template/hipcc compile and builds in ~10-15 min.

## feature.shot-design-frontier

The `shot-design` environment on Frontier: feature.shot-design's conda set
(pyarrow, duckdb, fastapi, ...) plus ROCm torch instead of feature.cuda's cu124
wheels. It is a third torch source beside feature.cuda and
feature.shot-design-cpu for the same reason those two are separate: the index is
the one line that differs. Same rocm7.1 pins as feature.frontier, so the two
environments on that machine run the same torch and a checkpoint written by one
loads in the other.

`plotly`: on Stellar the labeler runs in its own `labelmaker` environment, which
pins it; on Frontier there is one environment for both packages, so the pin is
repeated here rather than the suite being split (`tests/labeler` imports plotly
at collection).

## Environment names

Dash form, not `shot_design`: pixi rejects underscores in an environment name
("please use only lowercase letters, numbers and dashes"). The python package
these environments install is still `shot_design`.
