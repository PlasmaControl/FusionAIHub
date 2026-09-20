---
title: Stellar (Princeton)
sidebar_position: 1
---

Where FusionAIHub lives on Stellar, what it expects to find there, and what
has to move when a workstream is ported to another cluster. The Frontier
side is documented separately in [Frontier (OLCF)](./frontier.md); the
general path-map / environment-variable pattern used to port between
clusters is in [Adding a cluster](./adding-a-cluster.md).

Related: [Getting started](../getting-started/install.md) (pixi
environments), [`shot_design` overview](../shot-design/overview.md),
[`labeler` overview](../labeler/overview.md).

## Hardware and scheduler

| Item | Value |
|---|---|
| Login nodes | 2x Tesla V100S 32 GB (development only, no jobs) |
| GPU partition `gpu` | 6 nodes x 2x A100-PCIE-40GB (128 CPU, 500 GB) + 1 node x 8x A100 (56 CPU, 1 TB) |
| CPU partitions | `pppl` (126 nodes, 96 CPU, 760 GB), `pu`, `cimes`, `all`, `serial`, `bigmem` (4 TB) |
| QOS used by our scripts | `gpu-stellar` on `gpu`; `pppl-short-stellar` on `pppl` |
| Modules | `module load pixi`; `module load mdsplus` for DIII-D fetches |
| Scheduler | SLURM, no account flag needed |

GPU jobs here are single node, one or two A100s. CPU work (labeler
detectors, database builds) runs on `pppl`.

## Filesystems

| Path | What | Size |
|---|---|---|
| `/scratch/gpfs/nc1514/FusionAIHub` | the git checkout, plus its `.pixi/` | 20 GB of envs |
| `/scratch/gpfs/EKOLEMEN/foundation_model` | DIII-D per-shot HDF5, `<shot>_processed.h5` | 16,911 shots, 53 TB |
| `/scratch/gpfs/EKOLEMEN/big_d3d_data/foundation_model_text` | logbook / text corpus | 27 GB |
| `/projects/EKOLEMEN/foundation_model/preprocessing_stats.pt` | normalization stats used by training | small |
| `/scratch/gpfs/EKOLEMEN/nc1514/labelmaker` | `LABELER_ROOT`: detector features, masks, labels, models, runs | 46 GB |
| `/scratch/gpfs/EKOLEMEN/nc1514/ideate` | `SHOT_DESIGN_DATA_ROOT`: the Shot Designer database and caches | 625 MB |
| `/scratch/gpfs/EKOLEMEN/nc1514/shot-recommender` | Ollama binary, models, home (LLM blurbs) | 128 GB |

Rule that has held since Sep 2026: nothing new under `/scratch/gpfs/nc1514`
except repo source. Data goes under `/scratch/gpfs/EKOLEMEN/nc1514/`.

Layout of the two workstream roots:

```
labelmaker/                     ideate/
  features/    29 GB              db/           94 MB   shots/events parquet + embeddings
  envs/        6.8 GB             frame_codes/  489 MB
  ae/          5.5 GB             llm/          17 MB   endpoint.json, blurb backups
  masks/       3.4 GB             gates/, ignite_inputs/, text_cache/, llm_cache/
  runs/        540 MB             outputs/, actuations/, runs/
  labels/      460 MB
  events/      99 MB
  models/      48 MB   seldnet, dsm, tearing cnn1d, tokeye
  *.txt shot lists (recommender_v1.txt = the 500-shot dev set)
```

## Pixi environments (`pyproject.toml`)

| Env | Features | Use |
|---|---|---|
| `default` | cuda (torch cu124) | foundation-model training on NVIDIA |
| `fdp` | fdp + cuda | direct DIII-D access via `toksearch` (ga-fdp channel, needs MDSplus) |
| `labelmaker` | labelmaker + fdp | `labeler` package, CPU only |
| `shot-design` | shot-design + cuda | `shot_design` package with GPU |
| `shot-design-cpu` | shot-design | `shot_design` on login node / CPU |
| `shot-design-frontier` | shot-design-frontier + shot-design (torch rocm7.1) | `shot_design` and `labeler` on Frontier; the only environment installed there |
| `frontier` | frontier (torch rocm7.1) | training on Frontier MI250X |
| `rocm` | (della-milan) | MI210 variant |

Packages were renamed 2026-09-15: `ideate` -> `shot_design`, `labelmaker` ->
`labeler`; the environments followed on 2026-09-19. An environment name uses the
dash form (`shot-design`, not `shot_design`) because pixi rejects underscores in
one; the package it installs is still `shot_design`. The `labelmaker` environment
keeps its old name, its package having become `labeler`.

## Environment variables

The `shot-design`/`shot-design-cpu` envs pin these on activation, so `pixi run -e shot-design`
always sees production paths, even if you exported something else:

```
SHOT_DESIGN_DATA_ROOT = /scratch/gpfs/EKOLEMEN/nc1514/ideate
LABELER_ROOT          = /scratch/gpfs/EKOLEMEN/nc1514/labelmaker
SHOT_DESIGN_CORPUS    = /scratch/gpfs/EKOLEMEN/foundation_model
```

To point at another root, use the env's interpreter directly, not `pixi run`:
`.pixi/envs/shot-design-cpu/bin/python -m shot_design ...` with your own exports.
Never run a `shot_design` write command through `pixi run` against a scratch
target; it will write to production.

Other variables the code reads: `SHOT_DESIGN_PATHS` (paths file overriding the
root), `SHOT_DESIGN_CONFIG_DIR`, `SHOT_DESIGN_HF_ONLINE` (default offline),
`LABELER_LABEL_TABLES`, `HF_HOME` (leave it alone; see below), and the
`OLLAMA_*` family set by `scripts/shot_design/serve_llm.sh`.

## Tests

Run suites through pixi, never the bare `.pixi` interpreter, and do not
redirect `HF_HOME`:

```bash
pixi run -e shot-design-cpu pytest tests/shot_design
pixi run -e labelmaker pytest tests/labeler
```

## SLURM conventions

- Scripts: `scripts/slurm_stellar/` (training), `scripts/labeler/*.sbatch`,
  `scripts/shot_design/*.sbatch`.
- Logs go to `<data_root>/runs/slurm/%j.out` under EKOLEMEN, not the repo.
- Every job is gated afterwards with `python -m labeler.jobstats <jobid>`:
  CPU, CPU-mem, GPU, GPU-mem utilization all at or above 70 percent, CPU-only
  jobs judged on CPU/CPU-mem. Pilots of 20 shots or fewer are exempt but
  still reported.
- Any production data write (database build, `add`, `labels join`, fdp
  fetch, blurb backfill) needs the owner's explicit go-ahead.
- fdp / MDSplus fetches run on the login node only.

## Services

- **Shot Designer GUI**: `python -m shot_design serve --port 8765`, binds
  127.0.0.1 with a token link. Reach it with
  `ssh -L 8765:localhost:8765 stellar`. Reads the database in
  `SHOT_DESIGN_DATA_ROOT/db`.
- **LLM blurbs**: `sbatch scripts/shot_design/serve_llm.sbatch` starts Ollama
  (Gemma 4 26b) on one A100 for 4 h and writes the endpoint to
  `/scratch/gpfs/EKOLEMEN/nc1514/ideate/llm/endpoint.json`; CPU clients on the login node read that file.
  Binary and weights live under `shot-recommender/`. Ollama/Gemma is
  Stellar-only — Frontier uses Gemini Flash via the `agy` CLI instead (see
  [Frontier](./frontier.md) and
  [LLM providers](../shot-design/llm-providers.md)).
- **MCP server**: `shot_design` exposes `search_shots`, `describe_shot`,
  `get_events`, `phenomenon_locate`; config in
  [`shot_design` MCP server](../shot-design/mcp.md).
