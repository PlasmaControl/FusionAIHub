# Cluster setup: Stellar and Frontier

Where FusionAIHub lives on each machine, what it expects to find there, and
what has to move when a workstream is ported from Stellar (Princeton, NVIDIA)
to Frontier (OLCF, AMD). Written 2026-09-19 from the `recommender` branch; the
Stellar side is what is running today, the Frontier side is what already
exists there plus what the recommender port still needs.

Related: `README.md` (Frontier quick start for the foundation-model
training), `docs/SHOT_DESIGN.md` (Shot Designer CLI/GUI/LLM),
`docs/LABELER.md` (event labeling), `data/events/README.md` (label tables).

---

## 1. Stellar (Princeton)

### Hardware and scheduler

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

### Filesystems

| Path | What | Size |
|---|---|---|
| `/scratch/gpfs/nc1514/FusionAIHub` | the git checkout (branch `recommender`), plus its `.pixi/` | 20 GB of envs |
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

### Pixi environments (`pyproject.toml`)

| Env | Features | Use |
|---|---|---|
| `default` | cuda (torch cu124) | foundation-model training on NVIDIA |
| `fdp` | fdp + cuda | direct DIII-D access via `toksearch` (ga-fdp channel, needs MDSplus) |
| `labelmaker` | labelmaker + fdp | `labeler` package, CPU only |
| `ideate` | ideate + cuda | `shot_design` package with GPU |
| `ideate-cpu` | ideate | `shot_design` on login node / CPU |
| `frontier` | frontier (torch rocm7.1) | training on Frontier MI250X |
| `rocm` | (della-milan) | MI210 variant |

Env names keep the old package names because pixi rejects underscores.
Packages were renamed 2026-09-15: `ideate` -> `shot_design`, `labelmaker` ->
`labeler`.

### Environment variables

The `ideate`/`ideate-cpu` envs pin these on activation, so `pixi run -e ideate`
always sees production paths, even if you exported something else:

```
SHOT_DESIGN_DATA_ROOT = /scratch/gpfs/EKOLEMEN/nc1514/ideate
LABELER_ROOT          = /scratch/gpfs/EKOLEMEN/nc1514/labelmaker
SHOT_DESIGN_CORPUS    = /scratch/gpfs/EKOLEMEN/foundation_model
```

To point at another root, use the env's interpreter directly, not `pixi run`:
`.pixi/envs/ideate-cpu/bin/python -m shot_design ...` with your own exports.
Never run a `shot_design` write command through `pixi run` against a scratch
target; it will write to production.

Other variables the code reads: `SHOT_DESIGN_PATHS` (paths file overriding the
root), `SHOT_DESIGN_CONFIG_DIR`, `SHOT_DESIGN_HF_ONLINE` (default offline),
`LABELER_LABEL_TABLES`, `HF_HOME` (leave it alone; see below), and the
`OLLAMA_*` family set by `scripts/shot_design/serve_llm.sh`.

### Tests

Run suites through pixi, never the bare `.pixi` interpreter, and do not
redirect `HF_HOME`:

```bash
pixi run -e ideate-cpu pytest tests/shot_design
pixi run -e labelmaker pytest tests/labeler
```

### SLURM conventions

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

### Services

- **Shot Designer GUI**: `python -m shot_design serve --port 8765`, binds
  127.0.0.1 with a token link. Reach it with
  `ssh -L 8765:localhost:8765 stellar`. Reads the database in
  `SHOT_DESIGN_DATA_ROOT/db`.
- **LLM blurbs**: `sbatch scripts/shot_design/serve_llm.sbatch` starts Ollama
  (Gemma 4 26b) on one A100 for 4 h and writes the endpoint to
  `ideate/llm/endpoint.json`; CPU clients on the login node read that file.
  Binary and weights live under `shot-recommender/`.
- **MCP server**: `shot_design` exposes `search_shots`, `describe_shot`,
  `get_events`, `phenomenon_locate`; config in `docs/SHOT_DESIGN.md`.

---

## 2. Frontier (OLCF)

### What already exists

The foundation-model training side has been on Frontier for a while, on
branch `foundation_model`. Reuse it rather than redoing it.

| Item | Value |
|---|---|
| Project | `fus187` (`#SBATCH -A fus187`) |
| Checkout | `/lustre/orion/fus187/scratch/$USER/FusionAIHub` |
| Shared data | `/lustre/orion/fus187/proj-shared/foundation_model` (HDF5), `.../foundation_model_meta/preprocessing_stats.pt`, `.../models/` (checkpoints) |
| GPU | MI250X, `gfx90a`, 8 GCDs per node, `--gpu-bind=closest` |
| Partitions used | `batch`, `extended` |
| Modules | `PrgEnv-gnu/8.7.0`, `cpe/26.03`, `rocm/7.1.1`, `craype-accel-amd-gfx90a` (all in `scripts/slurm_frontier/_frontier_settings.sh`) |
| Env | `pixi install -e frontier`, then `pixi run -e frontier setup-flash-attn` on a compute node |
| Network | RCCL over Slingshot, `NCCL_SOCKET_IFNAME=hsn0` |

`_frontier_settings.sh` is sourced by every Frontier wrapper and sets
`PATH`/`LD_LIBRARY_PATH` to the pixi env, `PYTORCH_ROCM_ARCH`,
`FLASH_ATTENTION_TRITON_AMD_ENABLE`, the MIOpen cache, and `MASTER_ADDR`.

Note the branch difference: Frontier has been tracking `foundation_model`,
this document describes `recommender`. Pushed 2026-09-19 as
`origin/recommender`.

### Stellar to Frontier path map

Use this when porting a script or a config. Everything on the left is a
hard-coded Stellar path that appears in the repo (mostly in SLURM scripts,
docs and the pixi activation block).

| Stellar | Frontier |
|---|---|
| `/scratch/gpfs/nc1514/FusionAIHub` | `/lustre/orion/fus187/scratch/$USER/FusionAIHub` |
| `/scratch/gpfs/EKOLEMEN/foundation_model` | `/lustre/orion/fus187/proj-shared/foundation_model` |
| `/projects/EKOLEMEN/foundation_model/preprocessing_stats.pt` | `/lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt` |
| `/scratch/gpfs/EKOLEMEN/nc1514/labelmaker` | `/lustre/orion/fus187/proj-shared/labeler` (proposed) |
| `/scratch/gpfs/EKOLEMEN/nc1514/ideate` | `/lustre/orion/fus187/proj-shared/shot_design` (proposed) |
| `/scratch/gpfs/EKOLEMEN/nc1514/shot-recommender/{bin,models,ollama_home}` | `/lustre/orion/fus187/proj-shared/ollama/{bin,models,home}` (proposed) |
| `module load pixi` | pixi installed in `$HOME` via `curl -fsSL https://pixi.sh/install.sh` |
| `-p gpu --gres=gpu:1 --qos=gpu-stellar` | `-A fus187 -p batch --gres=gpu:1 --gpu-bind=closest` |
| `-p pppl --qos=pppl-short-stellar` (CPU) | `-A fus187 -p batch` (Frontier has no CPU-only partition; a 1-GPU allocation is the CPU job) |

The "proposed" rows are the intended layout; nothing has been created there
yet. Keep the split (labeler root, shot_design root, ollama) so the three env
variables map one to one.

### What the recommender port needs

1. **Environments.** `labelmaker`, `ideate` and `ideate-cpu` are `linux-64`
   and should resolve on Frontier. `ideate` pulls CUDA torch; on Frontier use
   `ideate-cpu` for the CLI and the `frontier` env for anything that needs a
   GPU. The `fdp`/`labelmaker` envs depend on the `ga-fdp` conda channel
   (`toksearch`, MDSplus); they will install but cannot fetch, since DIII-D
   MDSplus is not reachable from OLCF. Do all fetching on Stellar.
2. **Activation paths.** The `[tool.pixi.feature.ideate.target.unix.activation.env]`
   block in `pyproject.toml` hard-codes the three Stellar roots. Frontier
   needs either a second feature with its own activation block or the roots
   exported by a wrapper before the interpreter runs. Do not edit the Stellar
   values in place.
3. **Hugging Face models offline.** `shot_design` embeds text with
   `sentence-transformers/all-MiniLM-L6-v2` and defaults to offline mode
   (`SHOT_DESIGN_HF_ONLINE` unset). Frontier compute nodes have no outbound
   network; pre-populate the HF cache from a login node or copy it from
   Stellar.
4. **Ollama.** Copy the binary, the Gemma 4 26b weights and the home dir
   (128 GB total under `shot-recommender/`). The Ollama Linux build is
   x86_64 and needs a ROCm build for MI250X; verify a ROCm-enabled Ollama
   release runs before relying on `serve_llm.sbatch` there.
5. **SLURM scripts.** `scripts/labeler/*.sbatch` and
   `scripts/shot_design/*.sbatch` carry Stellar partitions, QOS and
   `#SBATCH --output` paths. Frontier copies go in `scripts/slurm_frontier/`
   and should source `_frontier_settings.sh`.
6. **`labeler.jobstats`.** It parses Princeton's `jobstats` report plus
   `sacct` (`MaxRSS`, `ReqMem`). Frontier has no `jobstats` command, so the
   GPU-utilization half of the 70 percent gate needs another source there
   (`rocm-smi` sampling inside the job is the obvious one) before the gate
   can be trusted.

### Data to transfer (not in git)

Order by what the recommender workstream needs first. Use Globus between
Princeton and OLCF for anything over a few GB.

| Source on Stellar | Size | Needed for |
|---|---|---|
| `EKOLEMEN/nc1514/ideate` | 625 MB | Shot Designer database, run the GUI/MCP at once |
| `EKOLEMEN/nc1514/labelmaker/{labels,events,models,*.txt}` | ~600 MB | labels join, detector models, shot lists |
| `EKOLEMEN/nc1514/labelmaker/{masks,ae}` | 9 GB | rerunning detectors |
| `EKOLEMEN/nc1514/labelmaker/features` | 29 GB | feature-based detectors; skip until needed |
| `EKOLEMEN/nc1514/shot-recommender` | 128 GB | LLM blurbs |
| `EKOLEMEN/big_d3d_data/foundation_model_text` | 27 GB | text corpus for the database build |
| `EKOLEMEN/foundation_model` | 53 TB | already on Frontier as `proj-shared/foundation_model`; check shot coverage matches (`corpus_coverage.parquet` in `ideate/db` lists the 504 indexed shots) |
| Repo-local raw label sources (below) | ~4 GB | re-running the event formatters |

Raw label sources listed in `data/events/events.yaml` are kept out of git
(several exceed GitHub's 100 MB limit). They sit in the Stellar checkout under
`data/events/<event>/raw/` and must be copied by hand if a formatter is rerun:

| File | Size |
|---|---|
| `alfven_eigenmode/raw/co2_250_detector.pkl` (+ `_adjust.pkl`) | 2.8 GB |
| `neoclassical_tearing_mode/raw/tm_labels.tar`, `tm_labels.h5` | 392 MB + 122 MB |
| `detachment/raw/emission_structure_*.sav` (41 files) | 50 to 120 MB each |
| `edge_localized_mode/raw/*.pkl`, `wide_pedestal_.../raw/*.pkl`, `WPQHphases-*/` | 53 MB pkl, 37 MB of `.dat` |

The formatted outputs (`data/events/*/format/*_format_2026_v1.csv` and
`.meta.json`) are in git, so the database build does not need the raw files.

### Checklist

- [ ] `git clone -b recommender` into `/lustre/orion/fus187/scratch/$USER`
- [ ] `pixi install -e ideate-cpu -e labelmaker -e frontier`
- [ ] Create the three proposed roots under `proj-shared`, transfer `ideate` and the small `labelmaker` subdirs
- [ ] Add a Frontier activation block or wrapper for `SHOT_DESIGN_DATA_ROOT`, `LABELER_ROOT`, `SHOT_DESIGN_CORPUS`
- [ ] Populate the HF cache offline; `pixi run -e ideate-cpu pytest tests/shot_design` green
- [ ] `python -m shot_design describe 190736` returns the same record as on Stellar
- [ ] Port one labeler sbatch to `scripts/slurm_frontier/`, run a 20-shot pilot, read `labeler.jobstats`
- [ ] Decide on Ollama-on-ROCm before copying the 128 GB
