---
title: Frontier (OLCF)
sidebar_position: 2
---

The AMD/ROCm side of FusionAIHub, on OLCF's Frontier. The foundation-model
training side has been on Frontier for a while, on branch `foundation_model`;
the `shot_design`/`labeler` recommender port described here is newer work,
done on `nathan_dev`. See [Stellar](./stellar.md) for the NVIDIA/CUDA side
and [Adding a cluster](./adding-a-cluster.md) for the general path-map /
paths-file pattern this port follows.

## What already exists

| Item | Value |
|---|---|
| Project | `fus187` (`#SBATCH -A fus187`) |
| Checkout | `/lustre/orion/fus187/scratch/$USER/FusionAIHub` |
| Shared data | `/lustre/orion/fus187/proj-shared/foundation_model` (HDF5), `.../foundation_model_meta/preprocessing_stats.pt`, `.../models/` (checkpoints) |
| GPU | MI250X, `gfx90a`, 8 GCDs per node, `--gpu-bind=closest` |
| Partitions used | `batch`, `extended` |
| Modules | `PrgEnv-gnu/8.7.0`, `cpe/26.03`, `rocm/7.1.1`, `craype-accel-amd-gfx90a` (all in `scripts/slurm_frontier/_frontier_settings.sh`) |
| Env | `pixi install --frozen -e frontier`, then `pixi run --frozen -e frontier setup-flash-attn` on a compute node |
| Network | RCCL over Slingshot, `NCCL_SOCKET_IFNAME=hsn0` |

`_frontier_settings.sh` is sourced by every Frontier wrapper and sets
`PATH`/`LD_LIBRARY_PATH` to the pixi env, `PYTORCH_ROCM_ARCH`,
`FLASH_ATTENTION_TRITON_AMD_ENABLE`, the MIOpen cache, and `MASTER_ADDR`.

**Always pass `--frozen` to pixi on Frontier**, and set
`PIXI_CACHE_DIR=/tmp/pixi-cache-nchen` first. A bare `pixi run`/`pixi install`
re-solves every environment for every platform and panics on the
`default`/`win-64` PyPI solve, which needs a Windows interpreter.

## `shot_design` / `labeler` roots and environment variables

The recommender-port roots and the three environment variables that select
them are established and in use (this is the same pattern documented
generically in [Adding a cluster](./adding-a-cluster.md)):

```
SHOT_DESIGN_PATHS     = <repo>/configs/shot_design/paths.frontier.yaml
SHOT_DESIGN_DATA_ROOT = /lustre/orion/fus187/proj-shared/nchen/shot_design
LABELER_ROOT          = /lustre/orion/fus187/proj-shared/nchen/labeler
SHOT_DESIGN_CORPUS    = /lustre/orion/fus187/proj-shared/foundation_model
```

`pixi run --frozen -e shot-design-frontier ...` sets all four (plus
`HF_HUB_OFFLINE=1`, `TOKENIZERS_PARALLELISM=false`,
`HDF5_USE_FILE_LOCKING=FALSE`) through
`[tool.pixi.feature.shot-design-frontier.target.unix.activation.env]` in
`pyproject.toml`. `configs/shot_design/paths.frontier.yaml` carries every key
`paths.yaml` does, pointed at the roots above; `paths.yaml` itself stays the
Stellar file and is unaffected by anything written to the Frontier one.

`shot-design-frontier` (ROCm torch plus the `shot_design` and `labeler`
dependencies) is the **only** environment installed on Frontier for the CLI,
the tests and the GPU jobs; `frontier` stays separate for IGNITE training.
`shot-design`/`shot-design-cpu` are Stellar-only (CUDA/CPU torch) and are not
materialised here. The `fdp`/`labelmaker` envs depend on the `ga-fdp` conda
channel (`toksearch`, MDSplus) and would install but cannot fetch — DIII-D
MDSplus is not reachable from OLCF, so all fetching stays on Stellar. The
Hugging Face cache (`sentence-transformers/all-MiniLM-L6-v2`) is
pre-populated from the login node, since compute nodes have no outbound
network.

## SLURM wrappers

`scripts/slurm_frontier/` holds the shared setup plus one script per
`shot_design` stage:

| Script | Purpose |
|---|---|
| `_frontier_settings.sh` | modules, `PATH`/`LD_LIBRARY_PATH`, ROCm/MIOpen env — sourced by everything |
| `_shot_design_common.sh` | sources `_frontier_settings.sh`; exports `REPO`, `ROOT` (`$SHOT_DESIGN_DATA_ROOT`), `PY` (the `shot-design-frontier` interpreter), the four variables above |
| `shot_design_census.sh` | CPU corpus scan → `db/corpus_coverage.parquet` |
| `shot_design_build.sh` | full database build for a named shot list |
| `shot_design_encode.sh` | 8-task array, one GCD each, IGNITE frame-code encoding |
| `shot_design_genc.sh` | GPU gate comparing fresh encodes against the production frame-code cache |
| `shot_design_simulate.sh` | one design's paired real/proposed rollout (see [Simulation](../shot-design/simulation.md)) |

Every wrapper sources `_shot_design_common.sh` by a path relative to
`$SLURM_SUBMIT_DIR`, so **`sbatch` must be invoked with the repo root as the
current directory** — `sbatch` copies the script into its spool area, so
`dirname "$0"` inside the job is not the checkout.

## LLM: Gemini Flash via `agy`

Frontier has no path to the Stellar Ollama binary (it is an x86_64 build; a
ROCm build for MI250X was never produced), and no outbound network from
compute nodes to pull one. Instead of Ollama/Gemma, the Frontier `llm.yaml`
uses `provider: agy` — the Antigravity CLI backing Gemini — from the login
node, which does have network and an OAuth cache:

| `llm.yaml` key | Frontier value |
|---|---|
| `provider` | `agy` |
| `models.quality` | `gemini-3.8-flash-high` |
| `models.fast` / blurb model | `gemini-3.8-flash-low` |

Ollama/Gemma remains Stellar-only. See
[LLM providers](../shot-design/llm-providers.md) for the client-side
contract (tool-call emulation, error handling) and the blurb backfill.

## IGNITE model pins

The dynamics model pinned for this port is the **v4 generation**: 15
modalities (the 14 of v2 plus `mirnov`), a 1000-entry vocabulary per
modality, 1209 tokens per frame, `t0_start_s: 1.0`, 219 frames per full
shot, and 88 actuators. The dynamics checkpoint is
`ignite_prod_v4/runs/mskfull/dynamics_best.pt` at step 3200 — an early
checkpoint; rollout results are qualitative, not a claimed accuracy bar. See
[IGNITE — v4 generation](../models/ignite.md#12-v4-generation) for the pin/check
commands and the full modality table.

## Database

The Frontier-only shot list `recommender_frontier_v1` targets roughly 5000
shots selected from a full corpus census (`shot_design corpus select`), built
from the Frontier `shotsummary` text bundles (no `sql/` import). Building the
database and backfilling blurbs is ongoing; see
[Database build](../shot-design/database-build.md) for the census → select →
labels → build → blurb → encode sequence and the exact commands.

## Data transfer notes (not in git)

The Stellar-to-Frontier data transfer needed for the recommender workstream —
the `ideate`/`labelmaker` subdirectories and raw label sources — is listed in
[Adding a cluster](./adding-a-cluster.md#data-to-transfer). The DIII-D HDF5
corpus itself (53 TB) is already on Frontier as
`proj-shared/foundation_model`.

## Checklist

- [x] Checkout cloned into `/lustre/orion/fus187/scratch/$USER/FusionAIHub`
- [x] `pixi install --frozen -e shot-design-frontier -e frontier` (Frontier
      installs no other environment)
- [x] `shot_design`/`labeler` roots created under `proj-shared/nchen/`, three
      environment variables wired through the `shot-design-frontier`
      activation env
- [x] Hugging Face cache populated offline
- [x] `shot_design model --pin` / `--check` against the v4 codec + dynamics
      bundle
- [x] `agy` (Gemini Flash) wired in as the Frontier LLM provider; Ollama
      decision resolved (Stellar-only, not ported)
- [x] Frontier SLURM wrappers for census, build, encode, G-ENC gate and
      simulate
- [ ] `recommender_frontier_v1` at its ~5000-shot target, fully built and
      blurbed (in progress)
- [ ] `shot_design describe <shot>` cross-checked against the equivalent
      Stellar record
