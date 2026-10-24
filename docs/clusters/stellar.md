---
title: Stellar (Princeton)
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
| `/scratch/gpfs/EKOLEMEN/nc1514/labeler` | `LABELER_ROOT`: detector features, masks, labels, models, runs | 46 GB |
| `/scratch/gpfs/EKOLEMEN/nc1514/shot_design` | `SHOT_DESIGN_DATA_ROOT`: the Shot Designer database and caches | 625 MB |
| `/scratch/gpfs/EKOLEMEN/nc1514/shot-recommender` | Ollama binary, models, home (LLM blurbs) | 128 GB |

Rule that has held since Sep 2026: nothing new under `/scratch/gpfs/nc1514`
except repo source. Data goes under `/scratch/gpfs/EKOLEMEN/nc1514/`.

Layout of the two workstream roots:

```
labeler/                        shot_design/
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
| `labeler` | labeler + fdp | `labeler` package, CPU only |
| `shot-design` | shot-design + cuda | `shot_design` package with GPU |
| `shot-design-cpu` | shot-design | `shot_design` on login node / CPU |
| `shot-design-frontier` | shot-design-frontier + shot-design (torch rocm7.1) | `shot_design` and `labeler` on Frontier; the only environment installed there |
| `frontier` | frontier (torch rocm7.1) | training on Frontier MI250X |
| `rocm` | (della-milan) | MI210 variant |

Packages were renamed 2026-09-15: `ideate` -> `shot_design`, `labelmaker` ->
`labeler`; the environments followed (`shot-design` on 2026-09-19, `labeler` on
2026-10-09, when the two data roots were renamed too). An environment name uses
the dash form (`shot-design`, not `shot_design`) because pixi rejects underscores
in one; the package it installs is still `shot_design`.

## Environment variables

On a Stellar host (`stellar-*`, or a Slurm job whose cluster is `stellar`) the three roots come
from the `stellar` row of `labeler.env.CLUSTERS`, so the package commands and the sbatch
launchers need nothing exported
([Environment variables](../reference/environment-variables.md#which-cluster)):

```
SHOT_DESIGN_DATA_ROOT = /scratch/gpfs/EKOLEMEN/nc1514/shot_design
LABELER_ROOT          = /scratch/gpfs/EKOLEMEN/nc1514/labeler
LABELER_LABEL_TABLES  = /scratch/gpfs/EKOLEMEN/nc1514/events
SHOT_DESIGN_CORPUS    = /scratch/gpfs/EKOLEMEN/foundation_model   (the pixi activation default)
```

A root you export first wins: `SHOT_DESIGN_DATA_ROOT=/tmp/scratch-db pixi run --frozen -e
shot-design-cpu python -m shot_design ...` writes under `/tmp/scratch-db`. A write command
prints the root it resolved before its first write; with nothing exported it reads
`cluster stellar default`. The sbatch launchers resolve the same values through
`python src/labeler/env.py NAME` (the cluster's row, unless exported), so a job run on
Stellar and one run on StellarAI each get their own cluster's roots. `python src/labeler/env.py --cluster` prints which row applies.

The row is read by the packages (`labeler.config.Paths.from_env`, `shot_design.config.load_paths`,
the readers that call `labeler.env.resolve`) and by the sbatch launchers, not by every file.
Export `LABELER_ROOT` for the rest: about 50 Python scripts under `scripts/` default it to the
StellarAI `.cache/labeler` path, the model specs' `UPSTREAM` constants name the StellarAI tree,
and `pixi run labeler-test` names a StellarAI scratch folder (absent on Stellar) for pytest's
temporary files. Run the suite there with its temporary files on group storage, not in `/tmp`
(a login node's `/tmp` is small and shared):
`mkdir -p "${LABELER_ROOT:?export LABELER_ROOT}/scratch/pytest" && PYTEST_DEBUG_TEMPROOT="$LABELER_ROOT/scratch/pytest" pixi run -e labeler python -m pytest tests/labeler`.
Run without the export, a script writes under `FusionAIHub/.cache/labeler`, a folder that exists
on Stellar only because it holds the Slurm logs.

Other variables the code reads: `SHOT_DESIGN_PATHS` (paths file overriding the
root), `SHOT_DESIGN_CONFIG_DIR`, `SHOT_DESIGN_HF_ONLINE` (default offline),
`LABELER_CLUSTER` (pins the row: `stellar`, `stellarai`, or empty for none), `LABELER_PIXI_ENV` (the pixi
`labeler` environment the launchers run `env.py` with; Stellar's main checkout by default), `HF_HOME`
(leave it alone; see below), and the
`OLLAMA_*` family set by `scripts/shot_design/serve_llm.sh`.

## IGNITE v4 on Stellar

Stellar runs the pinned v4 bundle at
`/scratch/gpfs/EKOLEMEN/nc1514/shot-recommender/models/IGNITE_v4`.
`model:` in `configs/shot_design/ignite_modalities.yaml` is the only model
configuration: 15 codecs, 1000 codes each, 1209 tokens per 50 ms frame,
frame 0 at 1.0 s. The same bundle is published as `nc1/IGNITE-v4` at
revision `d2f12b82`; install with `shot_design model --download --full`
and verify with `shot_design model --check`.

The UI submits `scripts/shot_design/simulate.sbatch <ident>` on one A100
through `paths.yaml`'s `simulate_submit_cmd`. See
[Simulation on Stellar](../shot-design/simulation.md#on-stellar).

## Tests

Run suites through pixi, never the bare `.pixi` interpreter, and do not
redirect `HF_HOME`. Keep pytest's temporary files on group storage, not in `/tmp`
(a login node's `/tmp` is small and shared); `LABELER_ROOT` must be exported:

```bash
mkdir -p "${LABELER_ROOT:?export LABELER_ROOT}/scratch/pytest"
export PYTEST_DEBUG_TEMPROOT="$LABELER_ROOT/scratch/pytest"
pixi run -e shot-design-cpu pytest tests/shot_design
pixi run -e labeler pytest tests/labeler
```

## SLURM conventions

- Scripts: `scripts/slurm_stellar/` (training), `scripts/labeler/*.sbatch`,
  `scripts/shot_design/*.sbatch`.
- A `#SBATCH --output` line cannot expand a variable, so every launcher names the StellarAI log
  folders, and a job whose log folder is missing fails at launch without a log. The data a job
  writes goes to the resolved root (Stellar's here) whatever folder holds its log.
  - Labeler launchers: `.../nc1514/FusionAIHub/.cache/labeler/runs/slurm`, created once on
    Stellar (2026-10-24). So on Stellar the log is not in `$LABELER_ROOT/runs/slurm`, where
    `labeler.ae.full`'s job-log reader, `jobstats --preserve-dir $LABELER_ROOT/runs/slurm` and the
    overlay messages look: submit with `sbatch -o "$LABELER_ROOT/runs/slurm/%j.out" -e
    "$LABELER_ROOT/runs/slurm/%j.out"` when one of them needs the log, or copy it over.
  - `scripts/shot_design/*.sbatch`: `.../nc1514/FusionAIHub/.cache/shot_design/{runs,experiments}/slurm`,
    which do not exist on Stellar. Create them once (`mkdir -p` both), or submit with
    `sbatch -o "$SHOT_DESIGN_DATA_ROOT/runs/slurm/%j.out" -e "$SHOT_DESIGN_DATA_ROOT/runs/slurm/%j.out"`.
    The UI's Simulate button does the second itself (`paths.yaml`'s `simulate_submit_cmd`).
  - The round-4 one-offs (`sawtooth_*`, `ae_supervision_swap*`, `confinement_bes_ablation`) log under
    `.cache/labeler/round4/...`, which exists only on StellarAI; they are not set up to run here
    without `-o`.
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
  `/scratch/gpfs/EKOLEMEN/nc1514/shot_design/llm/endpoint.json`; CPU clients on the login node read that file.
  Binary and weights live under `shot-recommender/`. Ollama/Gemma is
  Stellar-only — Frontier uses Gemini Flash via the `agy` CLI instead (see
  [Frontier](./frontier.md) and
  [LLM providers](../shot-design/llm-providers.md)).
- **MCP server**: `shot_design` exposes `search_shots`, `describe_shot`,
  `get_events`, `phenomenon_locate`; config in
  [`shot_design` MCP server](../shot-design/mcp.md).
