---
title: Adding a cluster
sidebar_position: 3
---

The pattern used to port `shot_design`/`labeler` from Stellar to Frontier,
generalised so a third cluster can follow it. It has three parts: a path
map (know what has to move), a paths file (one file per cluster, selected
by an environment variable), and three environment variables that a pixi
activation block or a shell wrapper sets.

## The path map

Everything on the left is a hard-coded Stellar path that appears in the
repo (mostly in SLURM scripts, docs and the pixi activation block). Use this
table when porting a script or a config to a new machine — it is also the
worked example for what a new cluster's own map should look like.

| Stellar | Frontier |
|---|---|
| `/scratch/gpfs/nc1514/FusionAIHub` | `/lustre/orion/fus187/scratch/$USER/FusionAIHub` |
| `/scratch/gpfs/EKOLEMEN/foundation_model` | `/lustre/orion/fus187/proj-shared/foundation_model` |
| `/projects/EKOLEMEN/foundation_model/preprocessing_stats.pt` | `/lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt` |
| `/scratch/gpfs/EKOLEMEN/nc1514/labelmaker` | `/lustre/orion/fus187/proj-shared/nchen/labeler` |
| `/scratch/gpfs/EKOLEMEN/nc1514/ideate` | `/lustre/orion/fus187/proj-shared/nchen/shot_design` |
| `/scratch/gpfs/EKOLEMEN/nc1514/shot-recommender/{bin,models,ollama_home}` | not ported — Frontier uses Gemini Flash via `agy` instead of Ollama (see [LLM providers](../shot-design/llm-providers.md)) |
| `module load pixi` | pixi installed in `$HOME` via `curl -fsSL https://pixi.sh/install.sh` |
| `-p gpu --gres=gpu:1 --qos=gpu-stellar` | `-A fus187 -p batch --gres=gpu:1 --gpu-bind=closest` |
| `-p pppl --qos=pppl-short-stellar` (CPU) | `-A fus187 -p batch` (Frontier has no CPU-only partition; a 1-GPU allocation is the CPU job) |

Keep the split (labeler root, shot_design root) so the environment variables
below map one to one — do not merge them into a single shared directory,
even though nothing stops the filesystem from allowing it.

## The three environment variables

Three variables select where `shot_design`/`labeler` read and write, on any
cluster:

- `SHOT_DESIGN_DATA_ROOT` — the shot-design workstream root (database, frame
  codes, caches, outputs, sessions).
- `LABELER_ROOT` — the labeler workstream root (detector features, masks,
  labels, models, runs).
- `SHOT_DESIGN_CORPUS` — the read-only DIII-D per-shot HDF5 corpus.

A fourth, `SHOT_DESIGN_PATHS`, does not set a root directly — it points at
the paths file (below) that resolves every other path the code needs,
including the three roots above and the text-corpus locations. Whichever of
`SHOT_DESIGN_DATA_ROOT` or the paths file's `data_root` you set, the
environment variable always wins, so a paths file can never silently
override an explicit override.

## The paths file pattern

`configs/shot_design/paths.yaml` is the default (Stellar) file; a new
cluster gets its own `configs/shot_design/paths.<cluster>.yaml` with **every
key `paths.yaml` has** (the loader requires the full set), values pointed at
that cluster's roots, and a header comment naming the source file and the
rule: *"`paths.yaml` stays Stellar; this file is selected by
`SHOT_DESIGN_PATHS` — nothing reads it by default, so a Stellar run is
unaffected by anything written here."* `configs/shot_design/paths.frontier.yaml`
is the worked example.

To wire a new cluster's file in:

1. Create the cluster's roots (`shot_design/{db,raw,text_cache,...}`,
   `labeler/{events,labels,models}`), idempotently, with group-writable
   permissions.
2. Write `paths.<cluster>.yaml` from the template above.
3. Add a pixi feature for the cluster (or a plain shell wrapper, if the
   cluster has no pixi) whose activation environment sets
   `SHOT_DESIGN_PATHS`, `SHOT_DESIGN_DATA_ROOT`, `LABELER_ROOT` and
   `SHOT_DESIGN_CORPUS` to that cluster's values — see
   `[tool.pixi.feature.shot-design-frontier.target.unix.activation.env]` in
   `pyproject.toml` for the Frontier version. List that feature **first**
   among the environment's features in `[tool.pixi.environments]`: pixi
   merges activation environments with the first-listed feature's values
   winning, so a shared feature like `shot-design` (which carries the
   Stellar roots) must not sort ahead of the cluster-specific one.
4. Give the cluster's SLURM wrappers a `_shot_design_common.sh`-style shared
   script that sources the site settings and exports the four variables
   again explicitly, rather than relying on pixi activation inside a batch
   job — `_shot_design_common.sh` is the Frontier example, and it is sourced
   relative to `$SLURM_SUBMIT_DIR` (see
   [Frontier](./frontier.md#slurm-wrappers)), which means every wrapper
   must be submitted with the repo root as the current directory.

## Data to transfer

Order by what the workstream needs first; use Globus (or an equivalent
site-to-site tool) for anything over a few GB. This was the Stellar to
Frontier list:

| Source on Stellar | Size | Needed for |
|---|---|---|
| `EKOLEMEN/nc1514/ideate` | 625 MB | Shot Designer database, run the GUI/MCP at once |
| `EKOLEMEN/nc1514/labelmaker/{labels,events,models,*.txt}` | ~600 MB | labels join, detector models, shot lists |
| `EKOLEMEN/nc1514/labelmaker/{masks,ae}` | 9 GB | rerunning detectors |
| `EKOLEMEN/nc1514/labelmaker/features` | 29 GB | feature-based detectors; skip until needed |
| `EKOLEMEN/big_d3d_data/foundation_model_text` | 27 GB | text corpus for the database build |
| `EKOLEMEN/foundation_model` | 53 TB | already on Frontier as `proj-shared/foundation_model`; check shot coverage matches before relying on it |
| Repo-local raw label sources (below) | ~4 GB | re-running the event formatters |

Raw label sources listed in `data/events/events.yaml` are kept out of git
(several exceed GitHub's 100 MB limit). They sit in the Stellar checkout
under `data/events/<event>/raw/` and must be copied by hand if a formatter
is rerun:

| File | Size |
|---|---|
| `alfven_eigenmode/raw/co2_250_detector.pkl` (+ `_adjust.pkl`) | 2.8 GB |
| `neoclassical_tearing_mode/raw/tm_labels.tar`, `tm_labels.h5` | 392 MB + 122 MB |
| `detachment/raw/emission_structure_*.sav` (41 files) | 50 to 120 MB each |
| `edge_localized_mode/raw/*.pkl`, `wide_pedestal_.../raw/*.pkl`, `WPQHphases-*/` | 53 MB pkl, 37 MB of `.dat` |

The formatted outputs (`data/events/*/format/*_format_2026_v1.csv` and
`.meta.json`) are in git, so a database build does not need the raw files.
